"""Resource discovery compatibility helpers backed by Falcon-owned models."""

from __future__ import annotations

import math
import re
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, DefaultDict, Dict, List, Optional, Tuple

import httpx

from .cluster import (
    AttemptSummary,
    ClusterSnapshot,
    JobSnapshot,
    NodeSnapshot,
    NodeTaint,
    ResourceVector,
    WorkloadConsumer,
    gpu_memory_bytes_from_labels,
    natural_name_key,
)
from .models import NodeResources, ResourcePlan
from .planning import canonical_gpu, plan_cpu_resources, plan_resources
from .quantities import parse_cpu as _parse_cpu
from .quantities import parse_memory_gib as _parse_memory_gib

DEFAULT_KUBE_STATE_METRICS_URL = "http://localhost:30080/metrics"

_METRIC = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[-+]?Inf)"
    r"(?:\s+\d+)?$"
)

_RESOURCE_METRIC_NAMES = frozenset(
    {
        "kube_node_status_allocatable",
        "kube_node_status_capacity",
        "kube_node_status_condition",
        "kube_node_spec_unschedulable",
        "kube_node_spec_taint",
        "kube_node_labels",
        "kube_pod_status_phase",
        "kube_pod_container_resource_requests",
        "kube_pod_info",
        "kube_pod_owner",
        "kube_replicaset_owner",
        "kube_job_owner",
        "kube_pod_created",
        "kube_job_info",
        "kube_job_status_active",
        "kube_job_status_succeeded",
        "kube_job_status_failed",
        "kube_job_created",
    }
)


def parse_prometheus_metrics(
    text: str,
    *,
    metric_names: Optional[set[str]] = None,
) -> Dict[str, List[Tuple[Dict[str, str], float]]]:
    """Parse the kube-state-metrics subset of Prometheus exposition text.

    This small Falcon-owned parser handles quoted/escaped label values and
    ignores malformed samples.
    """
    metrics: DefaultDict[str, List[Tuple[Dict[str, str], float]]] = defaultdict(list)
    for source in text.splitlines():
        line = source.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC.match(line)
        if not match:
            continue
        if metric_names is not None and match.group("name") not in metric_names:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        labels = _parse_labels(match.group("labels") or "")
        metrics[match.group("name")].append((labels, value))
    return dict(metrics)


def _parse_labels(source: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    index = 0
    length = len(source)
    while index < length:
        while index < length and source[index] in " \t,":
            index += 1
        start = index
        while index < length and source[index] not in "=,":
            index += 1
        key = source[start:index].strip()
        if not key or index >= length or source[index] != "=":
            break
        index += 1
        if index >= length or source[index] != '"':
            break
        index += 1
        value: List[str] = []
        while index < length:
            character = source[index]
            index += 1
            if character == '"':
                break
            if character == "\\" and index < length:
                escaped = source[index]
                index += 1
                value.append({"n": "\n", "t": "\t"}.get(escaped, escaped))
            else:
                value.append(character)
        labels[key] = "".join(value)
        while index < length and source[index] != ",":
            index += 1
        if index < length:
            index += 1
    return labels


def nodes_from_metrics(text: str) -> List[NodeResources]:
    metrics = parse_prometheus_metrics(text, metric_names=_RESOURCE_METRIC_NAMES)
    return _nodes_from_metric_map(metrics)


def _nodes_from_metric_map(
    metrics: Dict[str, List[Tuple[Dict[str, str], float]]],
) -> List[NodeResources]:
    scheduling_metrics = bool(
        metrics.get("kube_node_spec_unschedulable")
        or metrics.get("kube_node_spec_taint")
    )
    phases: Dict[str, str] = {}
    requests: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    # NodeResources is frozen to make planning values safe to share.  Build
    # mutable dictionaries first, then materialize the dataclasses below.
    values: DefaultDict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "cpu_total": 0.0,
            "cpu_used": 0.0,
            "memory_total_gib": 0.0,
            "memory_used_gib": 0.0,
            "gpu_total": 0,
            "gpu_used": 0,
            "gpu_product": "",
            "unschedulable": False,
        }
    )
    for labels, value in metrics.get("kube_node_status_allocatable", []):
        name = labels.get("node", "")
        if not name:
            continue
        resource = labels.get("resource")
        if resource == "cpu":
            values[name]["cpu_total"] = value
        elif resource == "memory":
            values[name]["memory_total_gib"] = value / (1024**3)
        elif resource in {"nvidia_com_gpu", "nvidia.com/gpu"}:
            values[name]["gpu_total"] = int(value)
    for labels, value in metrics.get("kube_node_spec_unschedulable", []):
        if value == 1 and labels.get("node"):
            values[labels["node"]]["unschedulable"] = True
    for labels, value in metrics.get("kube_node_spec_taint", []):
        # A taint is exposed for planning conservatively.  The richer node TUI
        # retains the actual taint because a matching toleration may allow use.
        if (
            value == 1
            and labels.get("node")
            and labels.get("effect") in {"NoSchedule", "NoExecute"}
        ):
            values[labels["node"]]["unschedulable"] = True
    for labels, value in metrics.get("kube_node_labels", []):
        name = labels.get("node", "")
        product = labels.get("label_nvidia_com_gpu_product")
        if name and value == 1 and product:
            values[name]["gpu_product"] = product.replace("_", " ").replace("-", " ")
    for labels, value in metrics.get("kube_pod_status_phase", []):
        if value == 1:
            phases[f"{labels.get('namespace', '')}/{labels.get('pod', '')}"] = labels.get(
                "phase", ""
            )
    for labels, value in metrics.get(
        "kube_pod_container_resource_requests", []
    ):
        node_name = labels.get("node", "")
        pod_id = f"{labels.get('namespace', '')}/{labels.get('pod', '')}"
        container = labels.get("container", "")
        if not node_name or not container:
            continue
        entry = requests.setdefault(
            (node_name, pod_id, container),
            {"cpu": 0.0, "memory": 0.0, "gpu": 0.0},
        )
        resource = labels.get("resource")
        if resource == "cpu":
            entry["cpu"] = value
        elif resource == "memory":
            entry["memory"] = value / (1024**3)
        elif resource in {"nvidia_com_gpu", "nvidia.com/gpu"}:
            entry["gpu"] = value
    for (node_name, pod_id, _container), request in requests.items():
        if phases.get(pod_id) not in {"Running", "Pending"}:
            continue
        values[node_name]["cpu_used"] = float(values[node_name]["cpu_used"]) + request["cpu"]
        values[node_name]["memory_used_gib"] = (
            float(values[node_name]["memory_used_gib"]) + request["memory"]
        )
        values[node_name]["gpu_used"] = int(values[node_name]["gpu_used"]) + int(
            request["gpu"]
        )
    return [
        NodeResources(
            name=name,
            cpu_total=float(data["cpu_total"]),
            cpu_used=float(data["cpu_used"]),
            memory_total_gib=float(data["memory_total_gib"]),
            memory_used_gib=float(data["memory_used_gib"]),
            gpu_total=int(data["gpu_total"]),
            gpu_used=int(data["gpu_used"]),
            gpu_product=str(data["gpu_product"]),
            unschedulable=bool(data["unschedulable"]),
            scheduling_info_available=scheduling_metrics,
        )
        for name, data in sorted(
            values.items(), key=lambda item: natural_name_key(item[0])
        )
    ]


def _iso_timestamp(value: float) -> Optional[str]:
    if value <= 0:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _display_gpu_model(value: str) -> str:
    """Use compact, stable model names in tables while preserving unknowns."""

    normalized = canonical_gpu(value)
    return {
        "2080ti": "2080Ti",
        "a6000": "A6000",
        "h100": "H100",
    }.get(normalized, value or "Unknown")


def cluster_snapshot_from_metrics(
    text: str,
    *,
    collected_at: float = 0.0,
) -> ClusterSnapshot:
    """Build the cluster resource view without cluster-scoped Kubernetes RBAC.

    kube-state-metrics already exposes the scheduler-facing facts used by the
    original ``jet resources`` command.  This adapter retains those exact
    request semantics and additionally uses exported Pod/owner metrics for the
    node inspector.  It never guesses a person from a namespace or Pod name.
    """

    metrics = parse_prometheus_metrics(text, metric_names=_RESOURCE_METRIC_NAMES)
    resource_nodes = {
        node.name: node for node in _nodes_from_metric_map(metrics)
    }
    capacities: DefaultDict[str, Dict[str, float]] = defaultdict(
        lambda: {"cpu": 0.0, "memory": 0.0, "gpu": 0.0}
    )
    for labels, value in metrics.get("kube_node_status_capacity", []):
        node = labels.get("node", "")
        resource = labels.get("resource", "")
        normalized = (
            "gpu"
            if resource in {"nvidia_com_gpu", "nvidia.com/gpu"}
            else resource
        )
        if node and normalized in capacities[node]:
            capacities[node][normalized] = value

    ready: Dict[str, Optional[bool]] = {}
    for labels, value in metrics.get("kube_node_status_condition", []):
        if labels.get("condition") != "Ready" or value != 1:
            continue
        status = labels.get("status", "").lower()
        ready[labels.get("node", "")] = (
            True if status == "true" else False if status == "false" else None
        )

    taints: DefaultDict[str, List[NodeTaint]] = defaultdict(list)
    for labels, value in metrics.get("kube_node_spec_taint", []):
        node = labels.get("node", "")
        if node and value == 1:
            taints[node].append(
                NodeTaint(
                    key=labels.get("key", ""),
                    value=labels.get("value", ""),
                    effect=labels.get("effect", ""),
                )
            )

    node_labels: DefaultDict[str, Dict[str, str]] = defaultdict(dict)
    exported_label_names = {
        "label_kubernetes_io_arch": "kubernetes.io/arch",
        "label_node_kubernetes_io_instance_type": (
            "node.kubernetes.io/instance-type"
        ),
        "label_topology_kubernetes_io_zone": "topology.kubernetes.io/zone",
        "label_gpu_type": "gpu-type",
        "label_nvidia_com_gpu_memory": "nvidia.com/gpu.memory",
        "label_nvidia_com_gpu_product": "nvidia.com/gpu.product",
    }
    for labels, value in metrics.get("kube_node_labels", []):
        node = labels.get("node", "")
        if not node or value != 1:
            continue
        for exported, original in exported_label_names.items():
            if labels.get(exported):
                node_labels[node][original] = labels[exported].replace("_", "-")

    pod_phases: Dict[Tuple[str, str], str] = {}
    for labels, value in metrics.get("kube_pod_status_phase", []):
        if value == 1:
            pod_phases[(labels.get("namespace", ""), labels.get("pod", ""))] = (
                labels.get("phase", "")
            )

    pod_info: Dict[Tuple[str, str], Dict[str, str]] = {}
    for labels, value in metrics.get("kube_pod_info", []):
        if value != 1:
            continue
        key = (labels.get("namespace", ""), labels.get("pod", ""))
        pod_info[key] = {
            "node": labels.get("node", ""),
            "owner_kind": labels.get("created_by_kind", ""),
            "owner_name": labels.get("created_by_name", ""),
        }
    for labels, value in metrics.get("kube_pod_owner", []):
        if value != 1 or labels.get("owner_is_controller") not in {"true", ""}:
            continue
        key = (labels.get("namespace", ""), labels.get("pod", ""))
        info = pod_info.setdefault(key, {})
        info["owner_kind"] = labels.get("owner_kind", "")
        info["owner_name"] = labels.get("owner_name", "")

    replica_owners: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for labels, value in metrics.get("kube_replicaset_owner", []):
        if value == 1 and labels.get("owner_is_controller") in {"true", ""}:
            replica_owners[
                (labels.get("namespace", ""), labels.get("replicaset", ""))
            ] = (
                labels.get("owner_kind", ""),
                labels.get("owner_name", ""),
            )
    job_owners: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for labels, value in metrics.get("kube_job_owner", []):
        if value == 1 and labels.get("owner_name"):
            job_owners[
                (labels.get("namespace", ""), labels.get("job_name", ""))
            ] = (
                labels.get("owner_kind", ""),
                labels.get("owner_name", ""),
            )

    pod_created: Dict[Tuple[str, str], Optional[str]] = {}
    for labels, value in metrics.get("kube_pod_created", []):
        pod_created[(labels.get("namespace", ""), labels.get("pod", ""))] = (
            _iso_timestamp(value)
        )

    pod_requests: DefaultDict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {"cpu": 0.0, "memory": 0.0, "gpu": 0.0}
    )
    for labels, value in metrics.get(
        "kube_pod_container_resource_requests", []
    ):
        key = (labels.get("namespace", ""), labels.get("pod", ""))
        resource = labels.get("resource", "")
        normalized = (
            "gpu"
            if resource in {"nvidia_com_gpu", "nvidia.com/gpu"}
            else resource
        )
        if key[1] and normalized in pod_requests[key]:
            pod_requests[key][normalized] += value
        if labels.get("node"):
            pod_info.setdefault(key, {})["node"] = labels["node"]

    consumers_by_node: DefaultDict[str, List[WorkloadConsumer]] = defaultdict(list)
    pod_vectors: Dict[Tuple[str, str], ResourceVector] = {}
    for key, request in pod_requests.items():
        phase = pod_phases.get(key, "")
        info = pod_info.get(key, {})
        node_name = info.get("node", "")
        if phase not in {"Running", "Pending"} or not node_name:
            continue
        node = resource_nodes.get(node_name)
        vector = ResourceVector(
            cpu_cores=request["cpu"],
            memory_bytes=int(round(request["memory"])),
            gpu_count=int(request["gpu"]),
            gpu_model=(
                _display_gpu_model(node.gpu_product)
                if node and request["gpu"] > 0 and node.gpu_product
                else None
            ),
        )
        if vector.is_empty:
            continue
        pod_vectors[key] = vector
        owner_kind = info.get("owner_kind", "") or "Pod"
        owner_name = info.get("owner_name", "") or key[1]
        if owner_kind == "ReplicaSet":
            owner_kind, owner_name = replica_owners.get(
                (key[0], owner_name),
                (owner_kind, owner_name),
            )
        elif owner_kind == "Job":
            owner_kind, owner_name = job_owners.get(
                (key[0], owner_name),
                (owner_kind, owner_name),
            )
        consumers_by_node[node_name].append(
            WorkloadConsumer(
                node_name=node_name,
                namespace=key[0],
                pod_name=key[1],
                workload_kind=owner_kind,
                workload_name=owner_name,
                status=phase,
                requested=vector,
                owner_identity=None,
                is_falcon_job=False,
                created_at=pod_created.get(key),
            )
        )

    snapshots: List[NodeSnapshot] = []
    for name, node in sorted(
        resource_nodes.items(), key=lambda item: natural_name_key(item[0])
    ):
        capacity = capacities.get(name, {})
        model = _display_gpu_model(node.gpu_product) if node.gpu_product else None
        node_consumers = tuple(
            sorted(
                consumers_by_node.get(name, []),
                key=lambda consumer: (
                    consumer.namespace,
                    consumer.workload_name,
                    consumer.pod_name,
                ),
            )
        )
        requested = ResourceVector()
        for consumer in node_consumers:
            requested = requested.plus(consumer.requested)
        allocatable = ResourceVector(
            cpu_cores=node.cpu_total,
            memory_bytes=int(round(node.memory_total_gib * 1024**3)),
            gpu_count=node.gpu_total,
            gpu_model=model if node.gpu_total else None,
        )
        snapshots.append(
            NodeSnapshot(
                name=name,
                ready=ready.get(name),
                schedulable=not node.unschedulable,
                taints=tuple(taints.get(name, ())),
                labels=dict(node_labels.get(name, {})),
                capacity=ResourceVector(
                    cpu_cores=float(capacity.get("cpu") or node.cpu_total),
                    memory_bytes=int(
                        round(
                            capacity.get("memory")
                            or node.memory_total_gib * 1024**3
                        )
                    ),
                    gpu_count=int(capacity.get("gpu") or node.gpu_total),
                    gpu_model=model if node.gpu_total else None,
                ),
                allocatable=allocatable,
                requested=requested,
                consumers=node_consumers,
                gpu_memory_bytes_per_device=gpu_memory_bytes_from_labels(
                    node_labels.get(name, {})
                ),
            )
        )

    job_names = {
        (labels.get("namespace", ""), labels.get("job_name", ""))
        for labels, value in metrics.get("kube_job_info", [])
        if value == 1
    }
    job_status: DefaultDict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"active": 0, "succeeded": 0, "failed": 0}
    )
    for metric_name, status_name in (
        ("kube_job_status_active", "active"),
        ("kube_job_status_succeeded", "succeeded"),
        ("kube_job_status_failed", "failed"),
    ):
        for labels, value in metrics.get(metric_name, []):
            key = (labels.get("namespace", ""), labels.get("job_name", ""))
            job_status[key][status_name] = int(value)
    job_created: Dict[Tuple[str, str], Optional[str]] = {}
    for labels, value in metrics.get("kube_job_created", []):
        job_created[
            (labels.get("namespace", ""), labels.get("job_name", ""))
        ] = _iso_timestamp(value)

    jobs: List[JobSnapshot] = []
    for key in sorted(job_names):
        status = job_status[key]
        if status["succeeded"] or status["failed"]:
            continue
        related = [
            pod_key
            for pod_key, info in pod_info.items()
            if pod_key[0] == key[0]
            and info.get("owner_kind") == "Job"
            and info.get("owner_name") == key[1]
            and pod_phases.get(pod_key) in {"Running", "Pending"}
        ]
        requested = ResourceVector()
        allocated = ResourceVector()
        active_pods: List[str] = []
        nodes = set()
        for pod_key in related:
            vector = pod_vectors.get(pod_key)
            if vector:
                requested = requested.plus(vector)
                if pod_info.get(pod_key, {}).get("node"):
                    allocated = allocated.plus(vector)
            active_pods.append(pod_key[1])
            if pod_info.get(pod_key, {}).get("node"):
                nodes.add(pod_info[pod_key]["node"])
        jobs.append(
            JobSnapshot(
                name=key[1],
                namespace=key[0],
                uid="",
                status="Running" if status["active"] else "Pending",
                requested=requested,
                allocated=allocated,
                attempts=AttemptSummary(
                    pod_attempts=len(active_pods),
                    active_pod=active_pods[-1] if active_pods else None,
                    active_pods=tuple(active_pods),
                    backoff_limit=None,
                ),
                nodes=tuple(sorted(nodes)),
                created_at=job_created.get(key),
            )
        )

    return ClusterSnapshot(
        nodes=tuple(snapshots),
        jobs=tuple(jobs),
        collected_at=collected_at,
    )


def fetch_cluster_snapshot(
    url: str,
    *,
    timeout: float = 10.0,
    collected_at: float = 0.0,
) -> ClusterSnapshot:
    if not url:
        raise ValueError("kube-state-metrics URL is empty")
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    return cluster_snapshot_from_metrics(
        response.text,
        collected_at=collected_at,
    )


class MetricsClusterCollector:
    """Cached, last-known-good kube-state-metrics cluster collector."""

    def __init__(
        self,
        url: str = DEFAULT_KUBE_STATE_METRICS_URL,
        *,
        inventory_seconds: float = 5.0,
        failure_retry_seconds: float = 2.0,
        timeout: float = 8.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = url
        self.inventory_seconds = inventory_seconds
        self.failure_retry_seconds = min(
            failure_retry_seconds, inventory_seconds
        )
        self.timeout = timeout
        self.clock = clock
        self._lock = threading.Lock()
        self._snapshot: Optional[ClusterSnapshot] = None
        self._next_refresh = 0.0
        self._closed = False

    @property
    def last_snapshot(self) -> Optional[ClusterSnapshot]:
        with self._lock:
            return self._snapshot

    def collect(self, force: bool = False) -> ClusterSnapshot:
        with self._lock:
            if self._closed:
                raise RuntimeError("collector is closed")
            now = float(self.clock())
            if not force and self._snapshot is not None and now < self._next_refresh:
                return self._snapshot
            try:
                snapshot = fetch_cluster_snapshot(
                    self.url,
                    timeout=self.timeout,
                    collected_at=now,
                )
                if not snapshot.nodes:
                    raise ValueError("kube-state-metrics reported no nodes")
            except Exception as exc:
                # Retaining a complete last frame is more useful than flashing
                # zeros during a local metrics proxy or API interruption.
                detail = f"{type(exc).__name__}: {exc}"
                if self._snapshot is None:
                    self._snapshot = ClusterSnapshot.empty(
                        collected_at=now,
                        stale=True,
                        error=detail,
                    )
                else:
                    self._snapshot = self._snapshot.mark_stale(detail)
                self._next_refresh = now + self.failure_retry_seconds
                return self._snapshot
            self._snapshot = snapshot
            self._next_refresh = now + self.inventory_seconds
            return snapshot

    def close(self) -> None:
        with self._lock:
            self._closed = True


def fetch_nodes(url: str, timeout: float = 10.0) -> List[NodeResources]:
    if not url:
        raise ValueError(
            "cluster.kube_state_metrics_url is not configured; "
            "set it or use explicit CPU/memory resources"
        )
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    return nodes_from_metrics(response.text)


def parse_cpu(value: str) -> float:
    return float(_parse_cpu(value.split(":", 1)[0]))


def parse_memory_gib(value: str) -> float:
    return float(_parse_memory_gib(value.split(":", 1)[0]))


__all__ = [
    "DEFAULT_KUBE_STATE_METRICS_URL",
    "MetricsClusterCollector",
    "NodeResources",
    "ResourcePlan",
    "canonical_gpu",
    "cluster_snapshot_from_metrics",
    "fetch_cluster_snapshot",
    "fetch_nodes",
    "nodes_from_metrics",
    "parse_cpu",
    "parse_memory_gib",
    "parse_prometheus_metrics",
    "plan_cpu_resources",
    "plan_resources",
]
