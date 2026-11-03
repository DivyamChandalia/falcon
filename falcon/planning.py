"""Cluster-aware Falcon resource planning."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable, Optional, Tuple

from .models import ComputeRequest, GPURequest, JobRequest, NodeResources, ResourcePlan
from .quantities import (
    QuantityError,
    format_cpu,
    format_memory_gib,
    normalize_memory,
    parse_cpu,
    parse_memory_bytes,
    parse_memory_gib,
    split_pair,
)

DEFAULT_SHARED_MEMORY_PERCENT = Decimal("15")
AUTO_MEMORY_BUFFER_GIB = 1.0
MAX_SCHEDULING_NODES = 20
GPU_MODEL_DISPLAY_ORDER = (
    ("2080ti", "2080Ti"),
    ("a6000", "A6000"),
    ("pro6000", "PRO6000"),
    ("h100", "H100"),
)


def canonical_gpu(product: str) -> str:
    """Normalize common display/product strings to Falcon GPU model keys."""

    normalized = product.lower().replace("_", " ").replace("-", " ")
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    if "h100" in normalized:
        return "h100"
    if "a6000" in normalized:
        return "a6000"
    if "2080" in normalized and ("ti" in normalized or "titanium" in normalized):
        return "2080ti"
    if "pro6000" in compact:
        return "pro6000"
    return compact


def gpu_model_order_key(product: str) -> Tuple[int, str]:
    """Keep known GPU models in Falcon's shared dashboard display order."""

    canonical = canonical_gpu(product)
    known = {
        model: index
        for index, (model, _) in enumerate(GPU_MODEL_DISPLAY_ORDER)
    }
    return known.get(canonical, len(known)), canonical


def plan_cpu_resources(
    cpu: str,
    memory: str,
    *,
    shared_memory: Optional[str] = None,
    shared_memory_percent: Decimal | float | int = DEFAULT_SHARED_MEMORY_PERCENT,
) -> ResourcePlan:
    """Resolve a CPU-only plan without querying cluster GPU inventory."""

    if not cpu:
        raise ValueError("CPU-only jobs require an explicit CPU request")
    if not memory:
        raise ValueError("CPU-only jobs require an explicit memory request")
    cpu_request, _ = split_pair(cpu, parse_cpu, normalize_limit=True)
    memory_request, _ = split_pair(
        memory, parse_memory_bytes, normalize_limit=True
    )
    memory_request = normalize_memory(memory_request)
    _require_positive(cpu_request, parse_cpu, "CPU")
    _require_positive(memory_request, parse_memory_bytes, "memory")
    resolved_shm = calculate_shared_memory(
        memory_request,
        explicit=shared_memory,
        percent=shared_memory_percent,
    )
    return ResourcePlan(
        preset="cpu",
        compute=ComputeRequest(
            cpu=cpu_request,
            cpu_limit=cpu_request,
            memory=memory_request,
            memory_limit=memory_request,
            shared_memory=resolved_shm,
        ),
        gpu=None,
        immediately_schedulable=True,
    )


def plan_resources(
    nodes: Iterable[NodeResources],
    preset: str,
    gpu_type: str,
    gpu_count: int,
    cpu_override: Optional[str] = None,
    memory_override: Optional[str] = None,
    maximize: bool = False,
    *,
    shared_memory: Optional[str] = None,
    shared_memory_percent: Decimal | float | int = DEFAULT_SHARED_MEMORY_PERCENT,
) -> ResourcePlan:
    """Resolve a GPU request from schedulable node capacity and requests.

    A plan records the node used for sizing separately from a node that can
    satisfy the request immediately. Unless the caller later asks to pin the
    request, Kubernetes remains free to place it on any matching node.
    """

    gpu = GPURequest(gpu_type, gpu_count)
    wanted_model = canonical_gpu(gpu.model)
    matching = [
        item
        for item in nodes
        if not item.unschedulable and canonical_gpu(item.gpu_product) == wanted_model
    ]
    if not matching:
        raise ValueError(f"no schedulable {gpu.model} nodes were reported")

    capacity_candidates = [item for item in matching if item.gpu_total >= gpu.count]
    if not capacity_candidates:
        maximum = max((item.gpu_total for item in matching), default=0)
        raise ValueError(
            f"{gpu.model}x{gpu.count} cannot fit on one node; "
            f"largest matching node has {maximum} GPU(s)"
        )
    gpu_candidates = [item for item in matching
                      if item.scheduling_info_available and item.gpu_free >= gpu.count]

    def free_score(item: NodeResources) -> Tuple[float, float, float, int]:
        return (
            item.cpu_free + item.memory_free_gib,
            item.cpu_free,
            item.memory_free_gib,
            item.gpu_free,
        )

    gpu_candidates.sort(key=free_score, reverse=True)
    capacity_candidates.sort(
        key=lambda item: (
            item.gpu_total,
            item.cpu_total,
            item.memory_total_gib,
            item.name,
        ),
        reverse=True,
    )

    requested_cpu = _normalized_override(cpu_override, parse_cpu, "CPU")
    requested_memory = _normalized_override(
        memory_override, parse_memory_bytes, "memory"
    )
    if requested_memory is not None:
        requested_memory = normalize_memory(requested_memory)
    requested_cpu_value = float(parse_cpu(requested_cpu)) if requested_cpu else None
    requested_memory_value = (
        parse_memory_gib(requested_memory) if requested_memory else None
    )

    if maximize:
        sizing_node = max(capacity_candidates, key=_proportional_capacity_score(gpu.count))
        share = gpu.count / sizing_node.gpu_total
        cpu_request = requested_cpu or format_cpu(sizing_node.cpu_total * share * 0.95)
        memory_request = requested_memory or format_memory_gib(
            _buffered_auto_memory(sizing_node.memory_total_gib * share * 0.95)
        )
    else:
        override_feasible = [
            item
            for item in gpu_candidates
            if (requested_cpu_value is None or item.cpu_free >= requested_cpu_value)
            and (
                requested_memory_value is None
                or item.memory_free_gib >= requested_memory_value
            )
        ]
        sizing_node = (
            override_feasible[0]
            if override_feasible
            else (gpu_candidates[0] if gpu_candidates else capacity_candidates[0])
        )
        share = gpu.count / sizing_node.gpu_total
        cpu_request = requested_cpu or format_cpu(
            max(
                0.1,
                min(sizing_node.cpu_total * share, sizing_node.cpu_free)
                if gpu_candidates
                else sizing_node.cpu_total * share,
            )
        )
        automatic_memory = max(
            0.1,
            min(
                sizing_node.memory_total_gib * share,
                sizing_node.memory_free_gib,
            )
            if gpu_candidates
            else sizing_node.memory_total_gib * share,
        )
        memory_request = requested_memory or format_memory_gib(
            _buffered_auto_memory(automatic_memory)
        )

    requested_cpu_value = float(parse_cpu(cpu_request))
    requested_memory_value = parse_memory_gib(memory_request)
    feasible = [
        item
        for item in gpu_candidates
        if item.cpu_free >= requested_cpu_value
        and item.memory_free_gib >= requested_memory_value
    ]
    chosen = feasible[0] if feasible else None
    scheduling = _gpu_scheduling(
        capacity_candidates, gpu, cpu_request, memory_request, sizing_node.name,
        shared_memory=shared_memory, shared_memory_percent=shared_memory_percent,
    )
    scheduling["gpu_limits"] = {
        "max_per_node": max(node.gpu_total for node in matching),
        "max_free_per_node": max(node.gpu_free for node in matching)
        if all(node.scheduling_info_available for node in matching) else None,
        "single_node_required": True,
        "note": "Do not sum GPU counts across nodes. Re-plan after changing GPU count.",
    }
    warning = _scheduling_warning(scheduling) if chosen is None else None

    resolved_shm = calculate_shared_memory(
        memory_request,
        explicit=shared_memory,
        percent=shared_memory_percent,
    )
    return ResourcePlan(
        preset=preset,
        compute=ComputeRequest(
            cpu=cpu_request,
            cpu_limit=cpu_request,
            memory=memory_request,
            memory_limit=memory_request,
            shared_memory=resolved_shm,
        ),
        gpu=gpu,
        node=chosen.name if chosen else None,
        immediately_schedulable=chosen is not None,
        warning=warning,
        sizing_node=sizing_node.name,
        scheduling=scheduling,
    )


def _node_auto_allocation(node: NodeResources, count: int, available: bool) -> dict:
    share = count / node.gpu_total
    cpu = node.cpu_total * share
    memory = node.memory_total_gib * share
    if available:
        cpu = min(cpu, node.cpu_free)
        memory = min(memory, node.memory_free_gib)
    return {
        "cpu": format_cpu(max(0.1, cpu)),
        "memory": format_memory_gib(_buffered_auto_memory(max(0.1, memory))),
        "basis": "currently_available" if available else "node_capacity",
    }


def _gpu_scheduling(nodes, gpu, cpu, memory, sizing_node, *, shared_memory, shared_memory_percent):
    """Expose jointly feasible per-node choices, never combine different nodes."""
    requested = {"gpus": gpu.count, "cpu_cores": float(parse_cpu(cpu)),
                 "memory_gib": parse_memory_gib(memory)}
    available_nodes = [node for node in nodes
                       if node.scheduling_info_available and node.gpu_free >= gpu.count]
    basis = "currently_available" if available_nodes else "node_capacity"
    pool = available_nodes or list(nodes)
    rows = []
    alternatives = []
    for node in nodes:
        capacity = {"gpus": node.gpu_total, "cpu_cores": node.cpu_total,
                    "memory_gib": node.memory_total_gib}
        available = {"gpus": node.gpu_free, "cpu_cores": node.cpu_free,
                     "memory_gib": node.memory_free_gib}
        now_deficit = {key: max(0, round(requested[key] - value, 9))
                       for key, value in available.items()}
        capacity_deficit = {key: max(0, round(requested[key] - value, 9))
                            for key, value in capacity.items()}
        gpu_available = node in available_nodes
        row = {
            "node": node.name, "availability_known": node.scheduling_info_available,
            "gpu_available_now": gpu_available,
            "available": available if node.scheduling_info_available else None,
            "capacity": capacity,
            "shortfall_now": now_deficit if node.scheduling_info_available else None,
            "shortfall_capacity": capacity_deficit,
            "blocking_resources": [key for key, value in now_deficit.items() if value]
            if node.scheduling_info_available else None,
            "fits_now": node.scheduling_info_available and not any(now_deficit.values()),
            "fits_capacity": not any(capacity_deficit.values()),
            # Capacity-only recommendations apply when this node's GPUs are
            # unavailable, regardless of its currently free CPU/RAM.
            "recommended_allocation": _node_auto_allocation(node, gpu.count, gpu_available),
        }
        rows.append(row)
        if node not in pool:
            continue
        budget = available if available_nodes else capacity
        if budget["cpu_cores"] <= 0 or budget["memory_gib"] <= 0:
            continue
        proposed_cpu = cpu if requested["cpu_cores"] <= budget["cpu_cores"] else (
            f"{int(Decimal(str(budget['cpu_cores'])) * 1000)}m"
            if budget["cpu_cores"] < 0.1 else format_cpu(budget["cpu_cores"])
        )
        proposed_memory = memory if requested["memory_gib"] <= budget["memory_gib"] else format_memory_gib(budget["memory_gib"])
        if (parse_cpu(proposed_cpu) <= 0 or float(parse_cpu(proposed_cpu)) > budget["cpu_cores"]
                or parse_memory_gib(proposed_memory) > budget["memory_gib"]):
            continue
        try:
            calculate_shared_memory(proposed_memory, explicit=shared_memory, percent=shared_memory_percent)
        except ValueError:
            continue
        if proposed_cpu == cpu and proposed_memory == memory:
            continue
        alternatives.append({
            "node": node.name, "gpu": {"model": gpu.model, "count": gpu.count},
            "cpu": proposed_cpu, "memory": proposed_memory, "basis": basis,
            "requires_resource_availability": not bool(available_nodes),
        })
    rows.sort(key=lambda row: (not row["fits_now"], not row["gpu_available_now"],
                               not row["fits_capacity"], row["node"]))
    fits_capacity = any(row["fits_capacity"] for row in rows)
    fits_now = any(row["fits_now"] for row in rows)
    known = all(node.scheduling_info_available for node in nodes)
    action = "submit" if fits_now else (
        "revise_request" if not fits_capacity else
        "recheck_inventory" if not known and not available_nodes else
        "choose_alternative_or_wait" if available_nodes else "wait_for_capacity"
    )
    maximum_cpu = max(node.cpu_free if available_nodes else node.cpu_total for node in pool)
    maximum_memory = max(node.memory_free_gib if available_nodes else node.memory_total_gib for node in pool)
    return {
        "requested": {"gpu_model": gpu.model, **requested},
        "gpu_available_now": bool(available_nodes) if known or available_nodes else None,
        "schedulable_now": fits_now, "fits_any_node_capacity": fits_capacity,
        "gpu_available_nodes": sorted(node.name for node in available_nodes),
        "schedulable_nodes": [row["node"] for row in rows if row["fits_now"]],
        "maximum_allocation": {
            "basis": basis, "cpu_cores": maximum_cpu, "memory_gib": maximum_memory,
            "cpu_nodes": sorted(node.name for node in pool
                                if (node.cpu_free if available_nodes else node.cpu_total) == maximum_cpu),
            "memory_nodes": sorted(node.name for node in pool
                                   if (node.memory_free_gib if available_nodes else node.memory_total_gib) == maximum_memory),
            "note": "Independent maxima may be on different nodes; CPU, RAM and GPUs must fit together on one node.",
        },
        "recommended_allocation": {"node": sizing_node, **next(
            row["recommended_allocation"] for row in rows if row["node"] == sizing_node
        )},
        "nodes": rows[:MAX_SCHEDULING_NODES], "matching_node_count": len(rows),
        "nodes_truncated": len(rows) > MAX_SCHEDULING_NODES,
        "alternatives": sorted(alternatives, key=lambda value: value["node"])[:3],
        "next_action": action,
        "guidance": (
            "This request cannot fit any matching node's capacity; waiting alone will not fix it. "
            "Re-plan with sufficient resources on a different GPU/node type or a task-appropriate smaller allocation."
            if not fits_capacity else
            "Choose a smaller alternative only if sufficient for the task, and re-plan before submitting. "
            "If these resources are required and no suitable alternative exists, submit unchanged and "
            "use wait_for_events with scheduling_policy=wait_for_capacity."
        ),
        "note": "Scheduler-request snapshot, not measured utilization, a reservation, or a complete Kubernetes scheduling simulation.",
    }


def _scheduling_warning(scheduling):
    requested = scheduling["requested"]
    summaries = []
    for row in scheduling["nodes"][:3]:
        if row["available"] is None:
            summaries.append(f"{row['node']}: current availability unknown")
            continue
        shortages = [
            f"{label} requested {requested[key]:g}, free {row['available'][key]:g}, "
            f"capacity {row['capacity'][key]:g}, short by {row['shortfall_now'][key]:g}"
            for key, label in (("gpus", "GPUs"), ("cpu_cores", "CPU cores"), ("memory_gib", "RAM GiB"))
            if row["shortfall_now"][key] > 0
        ]
        summaries.append(f"{row['node']}: " + "; ".join(shortages))
    prefix = (
        "Request exceeds every matching node's joint CPU/RAM/GPU capacity; waiting cannot make it fit"
        if not scheduling["fits_any_node_capacity"] else
        "Resources are not currently jointly free; request can remain pending until capacity is available"
    )
    return prefix + ". " + " | ".join(summaries)


def plan_job_request(
    request: JobRequest,
    nodes: Iterable[NodeResources] = (),
    *,
    preset: Optional[str] = None,
    maximize: bool = False,
    shared_memory_percent: Decimal | float | int = DEFAULT_SHARED_MEMORY_PERCENT,
) -> ResourcePlan:
    """Plan either a CPU-only or GPU :class:`JobRequest`."""

    compute = request.compute
    if request.gpu is None:
        if compute is None:
            raise ValueError("CPU-only jobs require explicit CPU and memory resources")
        return plan_cpu_resources(
            compute.cpu_pair,
            compute.memory_pair,
            shared_memory=compute.shared_memory,
            shared_memory_percent=shared_memory_percent,
        )
    return plan_resources(
        nodes,
        preset or canonical_gpu(request.gpu.model),
        request.gpu.model,
        request.gpu.count,
        cpu_override=compute.cpu_pair if compute else None,
        memory_override=compute.memory_pair if compute else None,
        maximize=maximize,
        shared_memory=compute.shared_memory if compute else None,
        shared_memory_percent=shared_memory_percent,
    )


def calculate_shared_memory(
    memory: str,
    *,
    explicit: Optional[str] = None,
    percent: Decimal | float | int = DEFAULT_SHARED_MEMORY_PERCENT,
) -> str:
    """Resolve and validate the memory-backed ``/dev/shm`` volume size."""

    memory_bytes = parse_memory_bytes(memory)
    if explicit is not None:
        shared_bytes = parse_memory_bytes(explicit)
        if shared_bytes <= 0:
            raise ValueError("shared memory size must be positive")
        if shared_bytes > memory_bytes:
            raise ValueError("shared memory size cannot exceed the container memory request")
        return normalize_memory(explicit.strip())

    percentage = Decimal(str(percent))
    if not percentage.is_finite() or percentage <= 0 or percentage > 100:
        raise ValueError("shared-memory percentage must be greater than 0 and at most 100")
    gib = memory_bytes / (Decimal(1024) ** 3)
    size = (gib * percentage / 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    size = max(size, Decimal("0.1"))
    return normalize_memory(f"{size}Gi")


def _normalized_override(
    value: Optional[str],
    parser,
    label: str,
) -> Optional[str]:
    if value is None:
        return None
    request, _ = split_pair(value, parser, normalize_limit=True)
    _require_positive(request, parser, label)
    return request


def _require_positive(value: str, parser, label: str) -> None:
    try:
        parsed = parser(value)
    except QuantityError:
        raise
    if parsed <= 0:
        raise ValueError(f"{label} request must be positive")


def _buffered_auto_memory(memory_gib: float) -> float:
    """Reserve one GiB outside Falcon's automatic container memory request."""

    return max(0.1, memory_gib - AUTO_MEMORY_BUFFER_GIB)


def _proportional_capacity_score(gpu_count: int):
    def score(item: NodeResources) -> Tuple[float, float, float]:
        share = gpu_count / item.gpu_total
        cpu_capacity = item.cpu_total * share
        memory_capacity = item.memory_total_gib * share
        return (
            cpu_capacity + memory_capacity,
            cpu_capacity,
            memory_capacity,
        )

    return score
