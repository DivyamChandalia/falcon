"""Shared Kubernetes inventory models and resource accounting.

This module deliberately has no dependency on Textual, kubectl, or Falcon's
CLI.  It turns ordinary Kubernetes API dictionaries into durable snapshots
which can be consumed by the human dashboards and machine-readable commands.

The central semantic distinction is:

* ``requested`` comes from a Job's pod template and survives pod deletion.
* ``allocated`` is the sum of non-terminal, node-bound pods right now.
* ``current_usage`` is optional telemetry and is never substituted with zero
  when metrics are unavailable.
"""

from __future__ import annotations

import inspect
import math
import re
import threading
import time
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)


TERMINAL_POD_PHASES = frozenset({"Succeeded", "Failed"})
ACTIVE_POD_PHASES = frozenset({"Pending", "Running", "Unknown", ""})
DEFAULT_OWNER_KEYS = (
    "falcon.dev/user",
    "falcon.dev/owner",
    "falcon.dev/submitted-by",
)
GPU_LABEL_KEYS = (
    "falcon.dev/gpu-type",
    "nvidia.com/gpu.product",
    "cloud.google.com/gke-accelerator",
    "gpu-type",
    "accelerator",
)
GPU_RESOURCE_KEYS = frozenset(
    {
        "nvidia.com/gpu",
        "amd.com/gpu",
        "intel.com/gpu",
    }
)


class InventoryClient(Protocol):
    """Minimal client boundary used by :class:`ClusterCollector`."""

    def list_inventory(self, namespace: Optional[str] = None) -> Any:
        """Return Nodes, Pods, and Jobs as Kubernetes dictionaries."""


@dataclass(frozen=True)
class ResourceVector:
    """A scheduler-facing resource vector.

    Memory is stored as bytes to avoid precision loss.  ``gpu_model`` is
    descriptive metadata; a zero GPU count does not imply that a workload is
    "CPU only" unless its durable Job template also has no GPU request.
    """

    cpu_cores: float = 0.0
    memory_bytes: int = 0
    gpu_count: int = 0
    gpu_model: Optional[str] = None

    @property
    def memory_gib(self) -> float:
        return self.memory_bytes / (1024**3)

    @property
    def is_empty(self) -> bool:
        return (
            math.isclose(self.cpu_cores, 0.0, abs_tol=1e-12)
            and self.memory_bytes == 0
            and self.gpu_count == 0
        )

    def plus(self, other: "ResourceVector") -> "ResourceVector":
        models = {
            model
            for model, count in (
                (self.gpu_model, self.gpu_count),
                (other.gpu_model, other.gpu_count),
            )
            if model and count
        }
        model = next(iter(models)) if len(models) == 1 else ("Mixed" if models else None)
        return ResourceVector(
            cpu_cores=self.cpu_cores + other.cpu_cores,
            memory_bytes=self.memory_bytes + other.memory_bytes,
            gpu_count=self.gpu_count + other.gpu_count,
            gpu_model=model,
        )

    def scaled(self, multiplier: int) -> "ResourceVector":
        count = max(0, int(multiplier))
        return ResourceVector(
            cpu_cores=self.cpu_cores * count,
            memory_bytes=self.memory_bytes * count,
            gpu_count=self.gpu_count * count,
            gpu_model=self.gpu_model if self.gpu_count and count else None,
        )

    def remaining_after(self, requested: "ResourceVector") -> "ResourceVector":
        return ResourceVector(
            cpu_cores=max(0.0, self.cpu_cores - requested.cpu_cores),
            memory_bytes=max(0, self.memory_bytes - requested.memory_bytes),
            gpu_count=max(0, self.gpu_count - requested.gpu_count),
            gpu_model=self.gpu_model if self.gpu_count else None,
        )


@dataclass(frozen=True)
class AttemptSummary:
    """Controller attempts and in-container restarts across every Job pod."""

    container_restarts: int = 0
    pod_attempts: int = 0
    succeeded_attempts: int = 0
    failed_attempts: int = 0
    active_pod: Optional[str] = None
    active_pods: Tuple[str, ...] = ()
    backoff_limit: Optional[int] = None


@dataclass(frozen=True)
class JobSnapshot:
    name: str
    namespace: str
    uid: str
    status: str
    requested: ResourceVector
    allocated: ResourceVector
    attempts: AttemptSummary
    current_usage: Optional[ResourceVector] = None
    nodes: Tuple[str, ...] = ()
    command: Tuple[str, ...] = ()
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    suspended: bool = False

    @property
    def gpu_requested(self) -> int:
        return self.requested.gpu_count

    @property
    def gpu_requested_model(self) -> Optional[str]:
        return self.requested.gpu_model if self.requested.gpu_count else None

    @property
    def gpu_allocated(self) -> int:
        return self.allocated.gpu_count

    @property
    def gpu_allocated_model(self) -> Optional[str]:
        return self.allocated.gpu_model if self.allocated.gpu_count else None

    @property
    def requested_cpu_cores(self) -> float:
        return self.requested.cpu_cores

    @property
    def requested_memory_bytes(self) -> int:
        return self.requested.memory_bytes

    @property
    def allocated_cpu_cores(self) -> float:
        return self.allocated.cpu_cores

    @property
    def allocated_memory_bytes(self) -> int:
        return self.allocated.memory_bytes

    @property
    def container_restarts(self) -> int:
        return self.attempts.container_restarts

    @property
    def pod_attempts(self) -> int:
        return self.attempts.pod_attempts

    @property
    def succeeded_attempts(self) -> int:
        return self.attempts.succeeded_attempts

    @property
    def failed_attempts(self) -> int:
        return self.attempts.failed_attempts

    @property
    def active_pod(self) -> Optional[str]:
        return self.attempts.active_pod

    @property
    def backoff_limit(self) -> Optional[int]:
        return self.attempts.backoff_limit


@dataclass(frozen=True)
class NodeTaint:
    key: str
    value: str = ""
    effect: str = ""

    def __str__(self) -> str:
        assignment = f"{self.key}={self.value}" if self.value else self.key
        return f"{assignment}:{self.effect}" if self.effect else assignment


@dataclass(frozen=True)
class WorkloadConsumer:
    node_name: str
    namespace: str
    pod_name: str
    workload_kind: str
    workload_name: str
    status: str
    requested: ResourceVector
    owner_identity: Optional[str] = None
    is_falcon_job: bool = False
    created_at: Optional[str] = None

    @property
    def display_workload(self) -> str:
        if self.workload_kind and self.workload_name:
            return f"{self.workload_kind}/{self.workload_name}"
        return self.pod_name

    @property
    def owner(self) -> Optional[str]:
        """Compatibility alias for renderers and serializers."""

        return self.owner_identity


@dataclass(frozen=True)
class NodeSnapshot:
    name: str
    ready: Optional[bool]
    schedulable: bool
    taints: Tuple[NodeTaint, ...]
    labels: Mapping[str, str]
    capacity: ResourceVector
    allocatable: ResourceVector
    requested: ResourceVector
    consumers: Tuple[WorkloadConsumer, ...]
    created_at: Optional[str] = None

    @property
    def gpu_model(self) -> Optional[str]:
        return self.allocatable.gpu_model or self.capacity.gpu_model

    @property
    def request_headroom(self) -> ResourceVector:
        """Allocatable capacity not currently claimed by pod requests."""

        return self.allocatable.remaining_after(self.requested)

    @property
    def gpu_free(self) -> int:
        """GPU request headroom, not measured physical idleness."""

        return self.request_headroom.gpu_count

    @property
    def workload_count(self) -> int:
        return len(self.consumers)


@dataclass(frozen=True)
class GPUAvailability:
    model: str
    capacity: int
    allocatable: int
    requested: int

    @property
    def request_headroom(self) -> int:
        return max(0, self.allocatable - self.requested)


@dataclass(frozen=True)
class ClusterSnapshot:
    nodes: Tuple[NodeSnapshot, ...] = ()
    jobs: Tuple[JobSnapshot, ...] = ()
    collected_at: float = 0.0
    stale: bool = False
    error: Optional[str] = None

    @classmethod
    def empty(
        cls,
        *,
        collected_at: float = 0.0,
        stale: bool = False,
        error: Optional[str] = None,
    ) -> "ClusterSnapshot":
        return cls(collected_at=collected_at, stale=stale, error=error)

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    @property
    def schedulable_nodes(self) -> int:
        return sum(node.schedulable and node.ready is True for node in self.nodes)

    @property
    def capacity(self) -> ResourceVector:
        return _sum_vectors(node.capacity for node in self.nodes)

    @property
    def allocatable(self) -> ResourceVector:
        return _sum_vectors(node.allocatable for node in self.nodes)

    @property
    def requested(self) -> ResourceVector:
        return _sum_vectors(node.requested for node in self.nodes)

    @property
    def request_headroom(self) -> ResourceVector:
        return self.allocatable.remaining_after(self.requested)

    @property
    def running_pods(self) -> int:
        return sum(
            consumer.status == "Running"
            for node in self.nodes
            for consumer in node.consumers
        )

    @property
    def pending_pods(self) -> int:
        # Only node-bound pending pods appear as node consumers.  Include
        # unbound active Job pods via the Job attempt list below.
        bound_pending = sum(
            consumer.status == "Pending"
            for node in self.nodes
            for consumer in node.consumers
        )
        active_pods = {
            pod
            for job in self.jobs
            for pod in job.attempts.active_pods
        }
        bound_active = {
            consumer.pod_name
            for node in self.nodes
            for consumer in node.consumers
            if consumer.status in ACTIVE_POD_PHASES
        }
        return bound_pending + len(active_pods - bound_active)

    @property
    def running_jobs(self) -> int:
        return sum(job.status == "Running" for job in self.jobs)

    @property
    def pending_jobs(self) -> int:
        return sum(job.status in {"Pending", "Suspended"} for job in self.jobs)

    @property
    def gpu_availability(self) -> Mapping[str, GPUAvailability]:
        grouped: Dict[str, List[int]] = {}
        for node in self.nodes:
            if not (node.capacity.gpu_count or node.allocatable.gpu_count):
                continue
            model = node.gpu_model or "Unknown"
            values = grouped.setdefault(model, [0, 0, 0])
            values[0] += node.capacity.gpu_count
            values[1] += node.allocatable.gpu_count
            values[2] += node.requested.gpu_count
        return {
            model: GPUAvailability(model, *values)
            for model, values in sorted(grouped.items())
        }

    def mark_stale(self, error: str) -> "ClusterSnapshot":
        return replace(self, stale=True, error=error)


_QUANTITY_RE = re.compile(
    r"^(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+))(?P<suffix>Ki|Mi|Gi|Ti|Pi|Ei|[numkKMGTP E]?)$".replace(
        " ", ""
    )
)


def _parse_decimal_quantity(value: Any) -> Tuple[Decimal, str]:
    raw = str(value if value is not None else "").strip()
    if not raw:
        return Decimal(0), ""
    match = _QUANTITY_RE.fullmatch(raw)
    if match:
        try:
            return Decimal(match.group("number")), match.group("suffix")
        except InvalidOperation as exc:
            raise ValueError(f"invalid Kubernetes quantity: {value!r}") from exc
    # Decimal handles scientific notation, which Kubernetes accepts in the
    # decimal exponent form (for example 1e3).
    try:
        return Decimal(raw), ""
    except InvalidOperation as exc:
        raise ValueError(f"invalid Kubernetes quantity: {value!r}") from exc


def parse_cpu_quantity(value: Any) -> float:
    """Parse a Kubernetes CPU quantity into cores."""

    amount, suffix = _parse_decimal_quantity(value)
    multipliers = {
        "n": Decimal("1e-9"),
        "u": Decimal("1e-6"),
        "m": Decimal("1e-3"),
        "": Decimal(1),
        "k": Decimal("1e3"),
        "K": Decimal("1e3"),
        "M": Decimal("1e6"),
        "G": Decimal("1e9"),
        "T": Decimal("1e12"),
        "P": Decimal("1e15"),
        "E": Decimal("1e18"),
    }
    if suffix not in multipliers:
        raise ValueError(f"invalid CPU quantity: {value!r}")
    result = amount * multipliers[suffix]
    if result < 0:
        raise ValueError(f"CPU quantity cannot be negative: {value!r}")
    return float(result)


def parse_memory_quantity(value: Any) -> int:
    """Parse a Kubernetes memory quantity into bytes."""

    amount, suffix = _parse_decimal_quantity(value)
    decimal = {
        "n": Decimal("1e-9"),
        "u": Decimal("1e-6"),
        "m": Decimal("1e-3"),
        "": Decimal(1),
        "k": Decimal(1000),
        "K": Decimal(1000),
        "M": Decimal(1000) ** 2,
        "G": Decimal(1000) ** 3,
        "T": Decimal(1000) ** 4,
        "P": Decimal(1000) ** 5,
        "E": Decimal(1000) ** 6,
    }
    binary = {
        "Ki": Decimal(1024),
        "Mi": Decimal(1024) ** 2,
        "Gi": Decimal(1024) ** 3,
        "Ti": Decimal(1024) ** 4,
        "Pi": Decimal(1024) ** 5,
        "Ei": Decimal(1024) ** 6,
    }
    if suffix in binary:
        result = amount * binary[suffix]
    elif suffix in decimal:
        result = amount * decimal[suffix]
    else:
        raise ValueError(f"invalid memory quantity: {value!r}")
    if result < 0:
        raise ValueError(f"memory quantity cannot be negative: {value!r}")
    return int(result.to_integral_value(rounding=ROUND_CEILING))


def normalize_gpu_model(value: Any) -> Optional[str]:
    """Return a compact, stable display name without inventing a GPU model."""

    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    normalized = re.sub(r"[_-]+", " ", raw)
    lowered = normalized.lower()
    known = (
        ("h200", "H200"),
        ("h100", "H100"),
        ("a100", "A100"),
        ("a6000", "A6000"),
        ("l40s", "L40S"),
        ("l40", "L40"),
        ("2080", "RTX 2080 Ti"),
    )
    for token, display in known:
        if token in lowered:
            return display
    return re.sub(r"\s+", " ", normalized).strip()


def _gpu_quantity(resources: Mapping[str, Any]) -> int:
    count = 0
    for name, value in resources.items():
        lowered = str(name).lower()
        if name in GPU_RESOURCE_KEYS or lowered.endswith("/gpu") or lowered.endswith(".com/gpu"):
            try:
                numeric = Decimal(str(value))
            except InvalidOperation as exc:
                raise ValueError(f"invalid GPU quantity {value!r} for {name}") from exc
            if numeric < 0 or numeric != numeric.to_integral_value():
                raise ValueError(f"GPU quantity must be a non-negative integer: {value!r}")
            count += int(numeric)
    return count


def _container_vector(container: Mapping[str, Any], gpu_model: Optional[str]) -> ResourceVector:
    resources = container.get("resources") or {}
    requests = resources.get("requests") or {}
    limits = resources.get("limits") or {}
    # Kubernetes copies a container limit into its request when the request is
    # omitted.  Extended GPU resources must have equal request and limit, but
    # max() also keeps malformed fixture data conservative.
    cpu_source = requests.get("cpu", limits.get("cpu", "0"))
    memory_source = requests.get("memory", limits.get("memory", "0"))
    gpu_count = max(_gpu_quantity(requests), _gpu_quantity(limits))
    return ResourceVector(
        cpu_cores=parse_cpu_quantity(cpu_source),
        memory_bytes=parse_memory_quantity(memory_source),
        gpu_count=gpu_count,
        gpu_model=gpu_model if gpu_count else None,
    )


def _max_vectors(vectors: Iterable[ResourceVector]) -> ResourceVector:
    materialized = list(vectors)
    if not materialized:
        return ResourceVector()
    cpu = max(vector.cpu_cores for vector in materialized)
    memory = max(vector.memory_bytes for vector in materialized)
    gpu = max(vector.gpu_count for vector in materialized)
    models = {vector.gpu_model for vector in materialized if vector.gpu_count and vector.gpu_model}
    model = next(iter(models)) if len(models) == 1 else ("Mixed" if models else None)
    return ResourceVector(cpu, memory, gpu, model)


def _sum_vectors(vectors: Iterable[ResourceVector]) -> ResourceVector:
    total = ResourceVector()
    for vector in vectors:
        total = total.plus(vector)
    return total


def _metadata_values(item: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    metadata = item.get("metadata") or {}
    yield metadata.get("labels") or {}
    yield metadata.get("annotations") or {}


def _extract_gpu_model(*items: Mapping[str, Any]) -> Optional[str]:
    for item in items:
        for values in _metadata_values(item):
            for key in GPU_LABEL_KEYS:
                if values.get(key):
                    return normalize_gpu_model(values[key])
        spec = item.get("spec") or {}
        if "template" in spec:
            spec = (spec.get("template") or {}).get("spec") or {}
        selector = spec.get("nodeSelector") or {}
        for key in GPU_LABEL_KEYS:
            if selector.get(key):
                return normalize_gpu_model(selector[key])
        terms = (
            (spec.get("affinity") or {})
            .get("nodeAffinity", {})
            .get("requiredDuringSchedulingIgnoredDuringExecution", {})
            .get("nodeSelectorTerms", [])
        )
        for term in terms:
            for expression in term.get("matchExpressions") or []:
                if expression.get("key") not in GPU_LABEL_KEYS:
                    continue
                values = expression.get("values") or []
                if expression.get("operator") == "In" and len(values) == 1:
                    return normalize_gpu_model(values[0])
    return None


def _pod_resource_vector(pod_spec: Mapping[str, Any], gpu_model: Optional[str]) -> ResourceVector:
    regular = _sum_vectors(
        _container_vector(container, gpu_model)
        for container in (pod_spec.get("containers") or [])
    )
    init = _max_vectors(
        _container_vector(container, gpu_model)
        for container in (pod_spec.get("initContainers") or [])
    )
    effective = _max_vectors((regular, init))
    overhead = pod_spec.get("overhead") or {}
    if overhead:
        effective = effective.plus(
            ResourceVector(
                cpu_cores=parse_cpu_quantity(overhead.get("cpu", "0")),
                memory_bytes=parse_memory_quantity(overhead.get("memory", "0")),
            )
        )
    return effective


def _job_template_vector(job: Mapping[str, Any]) -> ResourceVector:
    spec = job.get("spec") or {}
    template = spec.get("template") or {}
    gpu_model = _extract_gpu_model(job, template)
    per_pod = _pod_resource_vector(template.get("spec") or {}, gpu_model)
    parallelism = int(spec.get("parallelism", 1) or 1)
    return per_pod.scaled(parallelism)


def _pod_name(pod: Mapping[str, Any]) -> str:
    return str((pod.get("metadata") or {}).get("name") or "")


def _pod_namespace(pod: Mapping[str, Any]) -> str:
    return str((pod.get("metadata") or {}).get("namespace") or "default")


def _pod_phase(pod: Mapping[str, Any]) -> str:
    return str((pod.get("status") or {}).get("phase") or "Unknown")


def _timestamp_key(item: Mapping[str, Any]) -> str:
    return str((item.get("metadata") or {}).get("creationTimestamp") or "")


def _pod_restarts(pod: Mapping[str, Any]) -> int:
    status = pod.get("status") or {}
    groups = (
        status.get("initContainerStatuses") or [],
        status.get("containerStatuses") or [],
        status.get("ephemeralContainerStatuses") or [],
    )
    return sum(
        int(container.get("restartCount", 0) or 0)
        for group in groups
        for container in group
    )


def _pod_belongs_to_job(pod: Mapping[str, Any], job: Mapping[str, Any]) -> bool:
    pod_meta = pod.get("metadata") or {}
    job_meta = job.get("metadata") or {}
    if _pod_namespace(pod) != str(job_meta.get("namespace") or "default"):
        return False
    job_name = str(job_meta.get("name") or "")
    job_uid = str(job_meta.get("uid") or "")
    for owner in pod_meta.get("ownerReferences") or []:
        if owner.get("kind") == "Job" and (
            (job_uid and str(owner.get("uid") or "") == job_uid)
            or str(owner.get("name") or "") == job_name
        ):
            return True
    labels = pod_meta.get("labels") or {}
    return labels.get("batch.kubernetes.io/job-name", labels.get("job-name")) == job_name


def _condition_time(job: Mapping[str, Any], condition_type: str) -> Optional[str]:
    for condition in (job.get("status") or {}).get("conditions") or []:
        if condition.get("type") == condition_type and condition.get("status") == "True":
            return condition.get("lastTransitionTime") or condition.get("lastProbeTime")
    return None


def _job_status(job: Mapping[str, Any], pods: Sequence[Mapping[str, Any]]) -> str:
    spec = job.get("spec") or {}
    status = job.get("status") or {}
    if spec.get("suspend"):
        return "Suspended"
    conditions = {
        condition.get("type")
        for condition in status.get("conditions") or []
        if condition.get("status") == "True"
    }
    if "Complete" in conditions:
        return "Succeeded"
    if "Failed" in conditions:
        return "Failed"
    phases = [_pod_phase(pod) for pod in pods]
    if status.get("active") or "Running" in phases:
        return "Running"
    completions = int(spec.get("completions", 1) or 1)
    if int(status.get("succeeded", 0) or 0) >= completions:
        return "Succeeded"
    if phases and all(phase == "Succeeded" for phase in phases):
        return "Succeeded"
    if phases and all(phase in TERMINAL_POD_PHASES for phase in phases) and "Failed" in phases:
        return "Failed"
    return "Pending"


def _command_from_job(job: Mapping[str, Any]) -> Tuple[str, ...]:
    containers = (
        ((job.get("spec") or {}).get("template") or {})
        .get("spec", {})
        .get("containers", [])
    )
    if not containers:
        return ()
    container = containers[0]
    return tuple(str(value) for value in (container.get("command") or [])) + tuple(
        str(value) for value in (container.get("args") or [])
    )


def _metric_vector(
    metrics: Optional[Mapping[Any, Any]],
    pod: Mapping[str, Any],
) -> Optional[ResourceVector]:
    if not metrics:
        return None
    namespace, name = _pod_namespace(pod), _pod_name(pod)
    value = (
        metrics.get((namespace, name))
        or metrics.get(f"{namespace}/{name}")
        or metrics.get(name)
    )
    if value is None:
        return None
    if isinstance(value, ResourceVector):
        return value
    return ResourceVector(
        cpu_cores=parse_cpu_quantity(value.get("cpu", value.get("cpu_cores", "0"))),
        memory_bytes=parse_memory_quantity(
            value.get("memory", value.get("memory_bytes", "0"))
        ),
    )


def build_job_snapshot(
    job: Mapping[str, Any],
    pods: Sequence[Mapping[str, Any]],
    metrics: Optional[Mapping[Any, Any]] = None,
) -> JobSnapshot:
    """Build one durable Job snapshot from its Job and all historical pods."""

    metadata = job.get("metadata") or {}
    related = [pod for pod in pods if _pod_belongs_to_job(pod, job)]
    active = [pod for pod in related if _pod_phase(pod) not in TERMINAL_POD_PHASES]
    active_sorted = sorted(active, key=_timestamp_key)
    active_pod = active_sorted[-1] if active_sorted else None
    allocated_pods = [
        pod
        for pod in active
        if (pod.get("spec") or {}).get("nodeName")
    ]
    allocated = _sum_vectors(
        _pod_resource_vector(
            pod.get("spec") or {},
            _extract_gpu_model(pod, job),
        )
        for pod in allocated_pods
    )
    used_vectors = [
        vector
        for vector in (_metric_vector(metrics, pod) for pod in active)
        if vector is not None
    ]
    phases = [_pod_phase(pod) for pod in related]
    nodes = tuple(
        sorted(
            {
                str((pod.get("spec") or {}).get("nodeName"))
                for pod in allocated_pods
                if (pod.get("spec") or {}).get("nodeName")
            }
        )
    )
    status = job.get("status") or {}
    attempts = AttemptSummary(
        container_restarts=sum(_pod_restarts(pod) for pod in related),
        pod_attempts=len(related),
        succeeded_attempts=sum(phase == "Succeeded" for phase in phases),
        failed_attempts=sum(phase == "Failed" for phase in phases),
        active_pod=_pod_name(active_pod) if active_pod else None,
        active_pods=tuple(_pod_name(pod) for pod in active_sorted),
        backoff_limit=(
            int((job.get("spec") or {}).get("backoffLimit"))
            if (job.get("spec") or {}).get("backoffLimit") is not None
            else 6
        ),
    )
    return JobSnapshot(
        name=str(metadata.get("name") or ""),
        namespace=str(metadata.get("namespace") or "default"),
        uid=str(metadata.get("uid") or ""),
        status=_job_status(job, related),
        requested=_job_template_vector(job),
        allocated=allocated,
        attempts=attempts,
        current_usage=_sum_vectors(used_vectors) if used_vectors else None,
        nodes=nodes,
        command=_command_from_job(job),
        created_at=metadata.get("creationTimestamp"),
        started_at=status.get("startTime"),
        completed_at=_condition_time(job, "Complete") or _condition_time(job, "Failed"),
        suspended=bool((job.get("spec") or {}).get("suspend")),
    )


def _kind(item: Mapping[str, Any]) -> str:
    return str(item.get("kind") or "")


def _items(payload: Any) -> List[Mapping[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, ClusterSnapshot):
        raise TypeError("a ClusterSnapshot is already parsed")
    if isinstance(payload, Mapping):
        if "items" in payload:
            return list(payload.get("items") or [])
        # Some clients expose a compact split inventory.
        if any(key in payload for key in ("nodes", "pods", "jobs")):
            result: List[Mapping[str, Any]] = []
            for key in ("nodes", "pods", "jobs"):
                result.extend(_items(payload.get(key)))
            return result
        return [payload]
    return list(payload)


def build_job_snapshots(
    items: Any,
    metrics: Optional[Mapping[Any, Any]] = None,
) -> Tuple[JobSnapshot, ...]:
    inventory = _items(items)
    jobs = [item for item in inventory if _kind(item) == "Job"]
    pods = [item for item in inventory if _kind(item) == "Pod"]
    return tuple(
        build_job_snapshot(job, pods, metrics)
        for job in sorted(
            jobs,
            key=lambda item: (
                str((item.get("metadata") or {}).get("namespace") or "default"),
                str((item.get("metadata") or {}).get("name") or ""),
            ),
        )
    )


def _owner_reference(metadata: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    owners = metadata.get("ownerReferences") or []
    return next((owner for owner in owners if owner.get("controller")), owners[0] if owners else None)


def _identity(
    pod: Mapping[str, Any],
    owner: Optional[Mapping[str, Any]],
    owner_keys: Sequence[str],
) -> Optional[str]:
    for item in (pod, owner or {}):
        for values in _metadata_values(item):
            for key in owner_keys:
                value = values.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
    return None


def _is_falcon_managed(
    pod: Mapping[str, Any],
    owner: Optional[Mapping[str, Any]],
) -> bool:
    for item in (pod, owner or {}):
        metadata = item.get("metadata") or {}
        labels = metadata.get("labels") or {}
        annotations = metadata.get("annotations") or {}
        if str(labels.get("falcon.dev/managed", "")).lower() == "true":
            return True
        if str(labels.get("app.kubernetes.io/managed-by", "")).lower() == "falcon":
            return True
        if str(annotations.get("falcon.dev/managed", "")).lower() == "true":
            return True
    return False


def _node_ready(node: Mapping[str, Any]) -> Optional[bool]:
    for condition in (node.get("status") or {}).get("conditions") or []:
        if condition.get("type") == "Ready":
            if condition.get("status") == "True":
                return True
            if condition.get("status") == "False":
                return False
            return None
    return None


def _node_vector(
    values: Mapping[str, Any],
    gpu_model: Optional[str],
) -> ResourceVector:
    return ResourceVector(
        cpu_cores=parse_cpu_quantity(values.get("cpu", "0")),
        memory_bytes=parse_memory_quantity(values.get("memory", "0")),
        gpu_count=_gpu_quantity(values),
        gpu_model=gpu_model if _gpu_quantity(values) else None,
    )


def _node_taints(node: Mapping[str, Any]) -> Tuple[NodeTaint, ...]:
    return tuple(
        NodeTaint(
            key=str(taint.get("key") or ""),
            value=str(taint.get("value") or ""),
            effect=str(taint.get("effect") or ""),
        )
        for taint in (node.get("spec") or {}).get("taints") or []
    )


def _workload_owner(
    pod: Mapping[str, Any],
    jobs_by_key: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> Tuple[str, str, Optional[Mapping[str, Any]]]:
    metadata = pod.get("metadata") or {}
    namespace = _pod_namespace(pod)
    reference = _owner_reference(metadata)
    if not reference:
        return "Pod", _pod_name(pod), None
    kind = str(reference.get("kind") or "Pod")
    name = str(reference.get("name") or _pod_name(pod))
    owner = jobs_by_key.get((namespace, name)) if kind == "Job" else None
    # A Job may itself be owned by a CronJob.  Keep Job identity for Falcon
    # operations, while ordinary CronJob consumers are clearer under CronJob.
    if owner and not _is_falcon_managed(pod, owner):
        parent = _owner_reference(owner.get("metadata") or {})
        if parent:
            kind = str(parent.get("kind") or kind)
            name = str(parent.get("name") or name)
    return kind, name, owner


def build_cluster_snapshot(
    items: Any = None,
    metrics: Optional[Mapping[Any, Any]] = None,
    *,
    nodes: Any = None,
    pods: Any = None,
    jobs: Any = None,
    collected_at: float = 0.0,
    stale: bool = False,
    error: Optional[str] = None,
    owner_keys: Sequence[str] = DEFAULT_OWNER_KEYS,
) -> ClusterSnapshot:
    """Parse a mixed or split Kubernetes inventory into one coherent snapshot."""

    inventory = _items(items)
    inventory.extend(_items(nodes))
    inventory.extend(_items(pods))
    inventory.extend(_items(jobs))
    node_items = [item for item in inventory if _kind(item) == "Node"]
    pod_items = [item for item in inventory if _kind(item) == "Pod"]
    job_items = [item for item in inventory if _kind(item) == "Job"]
    jobs_by_key = {
        (
            str((job.get("metadata") or {}).get("namespace") or "default"),
            str((job.get("metadata") or {}).get("name") or ""),
        ): job
        for job in job_items
    }
    consumers_by_node: Dict[str, List[WorkloadConsumer]] = {}
    pod_vectors: Dict[int, ResourceVector] = {}

    for pod in pod_items:
        phase = _pod_phase(pod)
        node_name = str((pod.get("spec") or {}).get("nodeName") or "")
        if not node_name or phase in TERMINAL_POD_PHASES:
            continue
        kind, name, owner = _workload_owner(pod, jobs_by_key)
        vector = _pod_resource_vector(
            pod.get("spec") or {},
            _extract_gpu_model(pod, owner or {}),
        )
        pod_vectors[id(pod)] = vector
        consumer = WorkloadConsumer(
            node_name=node_name,
            namespace=_pod_namespace(pod),
            pod_name=_pod_name(pod),
            workload_kind=kind,
            workload_name=name,
            status=phase,
            requested=vector,
            owner_identity=_identity(pod, owner, owner_keys),
            is_falcon_job=_is_falcon_managed(pod, owner),
            created_at=(pod.get("metadata") or {}).get("creationTimestamp"),
        )
        consumers_by_node.setdefault(node_name, []).append(consumer)

    snapshots: List[NodeSnapshot] = []
    for node in node_items:
        metadata = node.get("metadata") or {}
        name = str(metadata.get("name") or "")
        labels = {
            str(key): str(value)
            for key, value in (metadata.get("labels") or {}).items()
        }
        gpu_model = _extract_gpu_model(node)
        status = node.get("status") or {}
        node_consumers = consumers_by_node.get(name, [])
        # Node labels are the authoritative model for an already placed pod.
        if gpu_model:
            node_consumers = [
                replace(
                    consumer,
                    requested=replace(
                        consumer.requested,
                        gpu_model=gpu_model if consumer.requested.gpu_count else None,
                    ),
                )
                for consumer in node_consumers
            ]
        snapshots.append(
            NodeSnapshot(
                name=name,
                ready=_node_ready(node),
                schedulable=not bool((node.get("spec") or {}).get("unschedulable")),
                taints=_node_taints(node),
                labels=labels,
                capacity=_node_vector(status.get("capacity") or {}, gpu_model),
                allocatable=_node_vector(status.get("allocatable") or {}, gpu_model),
                requested=_sum_vectors(
                    consumer.requested for consumer in node_consumers
                ),
                consumers=tuple(
                    sorted(
                        node_consumers,
                        key=lambda consumer: (
                            not consumer.is_falcon_job,
                            consumer.namespace,
                            consumer.workload_name,
                            consumer.pod_name,
                        ),
                    )
                ),
                created_at=metadata.get("creationTimestamp"),
            )
        )

    job_snapshots = tuple(
        build_job_snapshot(job, pod_items, metrics)
        for job in sorted(
            job_items,
            key=lambda item: (
                str((item.get("metadata") or {}).get("namespace") or "default"),
                str((item.get("metadata") or {}).get("name") or ""),
            ),
        )
    )
    return ClusterSnapshot(
        nodes=tuple(sorted(snapshots, key=lambda node: node.name)),
        jobs=job_snapshots,
        collected_at=float(collected_at),
        stale=stale,
        error=error,
    )


def parse_inventory(*args: Any, **kwargs: Any) -> ClusterSnapshot:
    """Readable alias used by adapters and tests."""

    return build_cluster_snapshot(*args, **kwargs)


def _call_inventory_method(
    method: Callable[..., Any],
    namespace: Optional[str],
) -> Any:
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "namespace" in parameters:
        return method(namespace=namespace)
    return method()


class ClusterCollector:
    """Bounded, last-known-good Kubernetes inventory collector.

    A combined ``list_inventory(namespace=...)`` client method costs one API
    round trip and is preferred.  Small adapters may instead expose
    ``list_nodes()``, ``list_pods(namespace=...)``, and
    ``list_jobs(namespace=...)``.  No query is repeated before the configured
    cadence unless ``force=True``.
    """

    def __init__(
        self,
        client: InventoryClient,
        namespace: Optional[str] = None,
        inventory_seconds: float = 5.0,
        failure_retry_seconds: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        owner_keys: Sequence[str] = DEFAULT_OWNER_KEYS,
    ) -> None:
        if inventory_seconds <= 0:
            raise ValueError("inventory_seconds must be positive")
        if failure_retry_seconds <= 0:
            raise ValueError("failure_retry_seconds must be positive")
        self.client = client
        self.namespace = namespace
        self.inventory_seconds = float(inventory_seconds)
        self.failure_retry_seconds = min(
            float(failure_retry_seconds), self.inventory_seconds
        )
        self.clock = clock
        self.owner_keys = tuple(owner_keys)
        self._lock = threading.Lock()
        self._snapshot: Optional[ClusterSnapshot] = None
        self._next_refresh = 0.0
        self._closed = False

    @property
    def last_snapshot(self) -> Optional[ClusterSnapshot]:
        with self._lock:
            return self._snapshot

    def _fetch(self) -> Any:
        for name in ("list_inventory", "get_inventory", "inventory"):
            method = getattr(self.client, name, None)
            if callable(method):
                return _call_inventory_method(method, self.namespace)
        node_method = getattr(self.client, "list_nodes", None)
        pod_method = getattr(self.client, "list_pods", None)
        job_method = getattr(self.client, "list_jobs", None)
        if not all(callable(method) for method in (node_method, pod_method, job_method)):
            raise TypeError(
                "cluster client must provide list_inventory() or "
                "list_nodes()/list_pods()/list_jobs()"
            )
        return {
            "nodes": _call_inventory_method(node_method, None),
            "pods": _call_inventory_method(pod_method, self.namespace),
            "jobs": _call_inventory_method(job_method, self.namespace),
        }

    def collect(self, force: bool = False) -> ClusterSnapshot:
        with self._lock:
            if self._closed:
                raise RuntimeError("collector is closed")
            now = float(self.clock())
            if not force and self._snapshot is not None and now < self._next_refresh:
                return self._snapshot
            try:
                payload = self._fetch()
                if isinstance(payload, ClusterSnapshot):
                    snapshot = replace(
                        payload,
                        collected_at=now,
                        stale=False,
                        error=None,
                    )
                else:
                    snapshot = build_cluster_snapshot(
                        payload,
                        collected_at=now,
                        owner_keys=self.owner_keys,
                    )
            except Exception as exc:
                # Kubernetes clients surface several implementation-specific
                # exception classes.  Recovery is intentional: retain the
                # complete last display and mark it stale.
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
            if self._closed:
                return
            self._closed = True
            close = getattr(self.client, "close", None)
            if callable(close):
                close()

