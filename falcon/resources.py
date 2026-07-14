"""Cluster-aware resource planning for Falcon GPU presets."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from jet.utils import _parse_prometheus_metrics


@dataclass
class NodeResources:
    name: str
    cpu_total: float = 0.0
    cpu_used: float = 0.0
    memory_total_gib: float = 0.0
    memory_used_gib: float = 0.0
    gpu_total: int = 0
    gpu_used: int = 0
    gpu_product: str = ""
    unschedulable: bool = False

    @property
    def cpu_free(self) -> float:
        return max(0.0, self.cpu_total - self.cpu_used)

    @property
    def memory_free_gib(self) -> float:
        return max(0.0, self.memory_total_gib - self.memory_used_gib)

    @property
    def gpu_free(self) -> int:
        return max(0, self.gpu_total - self.gpu_used)


@dataclass
class ResourcePlan:
    preset: str
    gpu_type: str
    gpu_count: int
    cpu: str
    memory: str
    node: Optional[str]
    immediately_schedulable: bool
    warning: Optional[str] = None
    sizing_node: Optional[str] = None


def canonical_gpu(product: str) -> str:
    normalized = product.lower().replace("_", " ").replace("-", " ")
    if "h100" in normalized:
        return "h100"
    if "a6000" in normalized:
        return "a6000"
    if "2080" in normalized:
        return "2080ti"
    return re.sub(r"[^a-z0-9]+", "", normalized)


def nodes_from_metrics(text: str) -> List[NodeResources]:
    metrics = _parse_prometheus_metrics(text)
    nodes: Dict[str, NodeResources] = {}
    phases: Dict[str, str] = {}
    requests: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    def node(name: str) -> NodeResources:
        if name not in nodes:
            nodes[name] = NodeResources(name=name)
        return nodes[name]

    for labels, value in metrics.get("kube_node_status_allocatable", []):
        current = node(labels.get("node", ""))
        resource = labels.get("resource")
        if resource == "cpu":
            current.cpu_total = value
        elif resource == "memory":
            current.memory_total_gib = value / (1024 ** 3)
        elif resource == "nvidia_com_gpu":
            current.gpu_total = int(value)

    for labels, value in metrics.get("kube_node_spec_unschedulable", []):
        if value == 1 and labels.get("node"):
            node(labels["node"]).unschedulable = True
    for labels, value in metrics.get("kube_node_spec_taint", []):
        if value == 1 and labels.get("effect") in {"NoSchedule", "NoExecute"}:
            node(labels.get("node", "")).unschedulable = True
    for labels, value in metrics.get("kube_node_labels", []):
        if labels.get("node") and labels.get("label_nvidia_com_gpu_product"):
            node(labels["node"]).gpu_product = labels["label_nvidia_com_gpu_product"].replace("_", " ").replace("-", " ")
    for labels, value in metrics.get("kube_pod_status_phase", []):
        if value == 1:
            phases[f"{labels.get('namespace', '')}/{labels.get('pod', '')}"] = labels.get("phase", "")
    for labels, value in metrics.get("kube_pod_container_resource_requests", []):
        node_name = labels.get("node", "")
        pod_id = f"{labels.get('namespace', '')}/{labels.get('pod', '')}"
        container = labels.get("container", "")
        if not node_name or not container:
            continue
        entry = requests.setdefault((node_name, pod_id, container), {"cpu": 0.0, "memory": 0.0, "gpu": 0.0})
        resource = labels.get("resource")
        if resource == "cpu":
            entry["cpu"] = value
        elif resource == "memory":
            entry["memory"] = value / (1024 ** 3)
        elif resource == "nvidia_com_gpu":
            entry["gpu"] = value

    for (node_name, pod_id, _), request in requests.items():
        if phases.get(pod_id) not in {"Running", "Pending"}:
            continue
        current = node(node_name)
        current.cpu_used += request["cpu"]
        current.memory_used_gib += request["memory"]
        current.gpu_used += int(request["gpu"])
    return list(nodes.values())


def fetch_nodes(url: str, timeout: float = 10.0) -> List[NodeResources]:
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    return nodes_from_metrics(response.text)


def _request_part(value: str) -> str:
    return value.split(":", 1)[0]


def parse_cpu(value: str) -> float:
    raw = _request_part(value).strip()
    return float(raw[:-1]) / 1000 if raw.endswith("m") else float(raw)


def parse_memory_gib(value: str) -> float:
    raw = _request_part(value).strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTE]i)?", raw, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported memory value: {value}")
    amount = float(match.group(1))
    unit = (match.group(2) or "Gi").lower()
    powers = {"ki": -2, "mi": -1, "gi": 0, "ti": 1, "ei": 2}
    return amount * (1024 ** powers[unit])


def _pair(value: str) -> str:
    request = _request_part(value)
    return f"{request}:{request}"


def _format_cpu(value: float) -> str:
    rounded = max(0.1, math.floor((value + 1e-9) * 10) / 10)
    rendered = str(int(rounded)) if rounded.is_integer() else f"{rounded:.1f}"
    return f"{rendered}:{rendered}"


def _format_memory(value: float) -> str:
    rounded = max(0.1, math.floor((value + 1e-9) * 10) / 10)
    rendered = str(int(rounded)) if rounded.is_integer() else f"{rounded:.1f}"
    return f"{rendered}Gi:{rendered}Gi"


def plan_resources(
    nodes: Iterable[NodeResources],
    preset: str,
    gpu_type: str,
    gpu_count: int,
    cpu_override: Optional[str] = None,
    memory_override: Optional[str] = None,
    maximize: bool = False,
) -> ResourcePlan:
    if gpu_count <= 0:
        raise ValueError("GPU count must be positive")
    matching = [
        item for item in nodes
        if not item.unschedulable and canonical_gpu(item.gpu_product) == canonical_gpu(gpu_type)
    ]
    if not matching:
        raise ValueError(f"no schedulable {gpu_type} nodes were reported")
    capacity_candidates = [item for item in matching if item.gpu_total >= gpu_count]
    if not capacity_candidates:
        maximum = max((item.gpu_total for item in matching), default=0)
        raise ValueError(
            f"{gpu_type}x{gpu_count} cannot fit on one node; largest matching node has {maximum} GPU(s)"
        )
    gpu_candidates = [item for item in matching if item.gpu_free >= gpu_count]

    def score(item: NodeResources) -> Tuple[float, float, float, int]:
        # Prefer the eligible node with the most absolute compute available.
        # Ratios favor small nodes that happen to be emptier, which produces a
        # needlessly small CPU/RAM allocation for the same GPU request.
        return (item.cpu_free + item.memory_free_gib, item.cpu_free, item.memory_free_gib, item.gpu_free)

    gpu_candidates.sort(key=score, reverse=True)
    capacity_candidates.sort(
        key=lambda item: (item.gpu_total, item.cpu_total, item.memory_total_gib), reverse=True
    )

    if maximize:
        def maximum_score(item: NodeResources) -> Tuple[float, float, float]:
            share = gpu_count / item.gpu_total
            cpu_capacity = item.cpu_total * share
            memory_capacity = item.memory_total_gib * share
            return (cpu_capacity + memory_capacity, cpu_capacity, memory_capacity)

        sizing_node = max(capacity_candidates, key=maximum_score)
        share = gpu_count / sizing_node.gpu_total
        cpu = _pair(cpu_override) if cpu_override else _format_cpu(sizing_node.cpu_total * share * 0.95)
        memory = (
            _pair(memory_override)
            if memory_override
            else _format_memory(sizing_node.memory_total_gib * share * 0.95)
        )
    else:
        requested_cpu_override = parse_cpu(cpu_override) if cpu_override else None
        requested_memory_override = parse_memory_gib(memory_override) if memory_override else None
        override_feasible = [
            item for item in gpu_candidates
            if (requested_cpu_override is None or item.cpu_free >= requested_cpu_override)
            and (requested_memory_override is None or item.memory_free_gib >= requested_memory_override)
        ]
        sizing_node = (
            override_feasible[0]
            if override_feasible
            else (gpu_candidates[0] if gpu_candidates else capacity_candidates[0])
        )
        share = gpu_count / sizing_node.gpu_total
        cpu = (
            _pair(cpu_override)
            if cpu_override
            else _format_cpu(
                min(sizing_node.cpu_total * share, sizing_node.cpu_free)
                if gpu_candidates else sizing_node.cpu_total * share
            )
        )
        memory = (
            _pair(memory_override)
            if memory_override
            else _format_memory(
                min(sizing_node.memory_total_gib * share, sizing_node.memory_free_gib)
                if gpu_candidates else sizing_node.memory_total_gib * share
            )
        )

    requested_cpu = parse_cpu(cpu)
    requested_memory = parse_memory_gib(memory)
    feasible = [
        item for item in gpu_candidates
        if item.cpu_free >= requested_cpu and item.memory_free_gib >= requested_memory
    ]
    chosen = feasible[0] if feasible else None
    immediate = chosen is not None

    warning = None
    if not immediate:
        if not gpu_candidates:
            detail = f"no node currently has {gpu_count} contiguous free {gpu_type} GPU(s)"
        else:
            detail = "the requested CPU or memory is not currently free on a matching node"
        warning = f"{detail}; the job will remain pending and be scheduled once resources are available"

    return ResourcePlan(
        preset=preset,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        cpu=cpu,
        memory=memory,
        node=chosen.name if chosen else None,
        immediately_schedulable=immediate,
        warning=warning,
        sizing_node=sizing_node.name,
    )
