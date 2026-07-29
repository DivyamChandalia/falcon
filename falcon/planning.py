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


def canonical_gpu(product: str) -> str:
    """Normalize common display/product strings to Falcon GPU model keys."""

    normalized = product.lower().replace("_", " ").replace("-", " ")
    if "h100" in normalized:
        return "h100"
    if "a6000" in normalized:
        return "a6000"
    if "2080" in normalized and ("ti" in normalized or "titanium" in normalized):
        return "2080ti"
    return re.sub(r"[^a-z0-9]+", "", normalized)


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
    gpu_candidates = [item for item in matching if item.gpu_free >= gpu.count]

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

    warning = None
    if chosen is None:
        if not gpu_candidates:
            detail = (
                f"no node currently has {gpu.count} contiguous free "
                f"{gpu.model} GPU(s)"
            )
        else:
            detail = "the requested CPU or memory is not currently free on a matching node"
        warning = (
            f"{detail}; the job will remain pending and be scheduled once "
            "resources are available"
        )

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
    )


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
