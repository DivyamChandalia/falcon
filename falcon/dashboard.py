"""Kubernetes Job metric collection for the Falcon dashboard."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

from .coder import CoderClient, CoderError, resolve_connection
from .config import DEFAULT_DASHBOARD_EMA_ALPHA, save_dashboard_sort, save_hidden_panes
from .resource_service import ResourceServiceCollector
from .resources import canonical_gpu, fetch_nodes
from .theme import metric_color

KUBERNETES_INVENTORY_SECONDS = 5.0
KUBERNETES_USAGE_SECONDS = 5.0
EVENT_REFRESH_SECONDS = 5.0
GPU_AVAILABILITY_SECONDS = 15.0
EMA_WARMUP_SAMPLES = 5
RISK_AVERAGE_SAMPLES = 60


@dataclass
class GpuProcess:
    gpu_uuid: str
    pid: int
    name: str
    memory_used_gib: Optional[float] = None
    gpu_utilization: Optional[float] = None


@dataclass
class GpuDevice:
    index: int
    name: str = "—"
    uuid: str = "—"
    memory_used_gib: Optional[float] = None
    memory_total_gib: Optional[float] = None
    utilization: Optional[float] = None
    temperature_c: Optional[float] = None
    power_w: Optional[float] = None
    ecc_errors: Optional[int] = None
    driver_version: str = "—"
    processes: List[GpuProcess] = field(default_factory=list)


@dataclass
class GpuSample:
    utilization: Optional[float] = None
    memory_used_gib: float = 0.0
    memory_total_gib: float = 0.0
    gpu_count: int = 0
    devices: List[GpuDevice] = field(default_factory=list)


@dataclass
class JobUsage:
    job: str
    status: str
    nodes: str
    gpu_type: str
    gpu_count: int
    pod_count: int
    gpu_util: Optional[float]
    gpu_ema: Optional[float]
    gpu_memory_used_gib: float
    gpu_memory_total_gib: float
    cpu_used: float
    cpu_requested: float
    memory_used_gib: float
    memory_requested_gib: float
    age: str
    at_risk: bool
    uid: str = ""
    active_pod: str = ""
    active_pod_uid: str = ""
    active_pod_state: str = "Unknown"
    command: str = ""
    created_at: str = ""
    started_at: str = ""
    restarts: int = 0
    completions: str = ""
    metrics_updated_at: float = 0.0
    gpu_metrics_available: bool = True
    cpu_metrics_available: bool = True
    gpu_devices: List[GpuDevice] = field(default_factory=list)
    gpu_risk_average: Optional[float] = None
    gpu_risk_threshold: Optional[float] = None
    # Durable request data comes from the Job's pod template and remains
    # available after every Pod has terminated or been garbage-collected.
    gpu_requested_type: str = "-"
    gpu_requested_count: int = 0
    # Allocation describes only currently active Pods.  It is intentionally
    # empty for queued and completed Jobs.
    gpu_allocated_type: str = "-"
    gpu_allocated_count: int = 0
    cpu_allocated: float = 0.0
    memory_allocated_gib: float = 0.0
    container_restarts: int = 0
    pod_attempts: int = 0
    succeeded_attempts: int = 0
    failed_attempts: int = 0
    backoff_limit: Optional[int] = None
    attempt_pods: List[str] = field(default_factory=list)

    @property
    def gpu_memory_percent(self) -> Optional[float]:
        return _percent(self.gpu_memory_used_gib, self.gpu_memory_total_gib) if self.gpu_metrics_available else None

    @property
    def cpu_percent(self) -> Optional[float]:
        return (
            _percent(self.cpu_used, self.cpu_allocated)
            if self.cpu_metrics_available
            else None
        )

    @property
    def memory_percent(self) -> Optional[float]:
        return (
            _percent(self.memory_used_gib, self.memory_allocated_gib)
            if self.cpu_metrics_available
            else None
        )


@dataclass
class JobEvent:
    timestamp: str
    event_type: str
    reason: str
    message: str
    object_name: str = ""
    count: int = 1


def _timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat((value or "").replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _pod_state(item: Dict) -> str:
    metadata, status = item.get("metadata", {}), item.get("status", {})
    if metadata.get("deletionTimestamp"):
        return "Terminating"
    waiting: List[str] = []
    terminated: List[str] = []
    restarts = 0
    for container in status.get("containerStatuses", []):
        restarts += int(container.get("restartCount", 0) or 0)
        state = container.get("state", {})
        if state.get("waiting", {}).get("reason"):
            waiting.append(state["waiting"]["reason"])
        if state.get("terminated", {}).get("reason"):
            terminated.append(state["terminated"]["reason"])
    failure_order = ["CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "CreateContainerError"]
    for reason in failure_order:
        if reason in waiting:
            return reason
    if waiting:
        return waiting[0]
    failure_terminated = [value for value in terminated if value not in {"Completed"}]
    if failure_terminated:
        return failure_terminated[0]
    if status.get("reason"):
        return str(status["reason"])
    return str(status.get("phase") or "Unknown")


def _active_pod(pods: List[Dict]) -> Optional[Dict]:
    if not pods:
        return None
    nonterminal = [pod for pod in pods if pod.get("status", {}).get("phase") not in {"Succeeded", "Failed"}]
    choices = nonterminal or pods
    return max(choices, key=lambda pod: _timestamp(pod.get("metadata", {}).get("creationTimestamp", "")))


def _job_status(job_item: Optional[Dict], pod_states: List[str]) -> str:
    if job_item:
        spec, status = job_item.get("spec", {}), job_item.get("status", {})
        if spec.get("suspend"):
            return "Suspended"
        for condition in status.get("conditions", []):
            if condition.get("status") != "True":
                continue
            if condition.get("type") == "Complete":
                return "Succeeded"
            if condition.get("type") == "Failed":
                return "Failed"
        # ``status.active`` counts Pending as well as Running Pods.  Calling
        # every active Job "Running" hides the most important queued state.
        if status.get("active"):
            return "Running" if "Running" in pod_states else "Pending"
    if "Running" in pod_states:
        return "Running"
    if "Pending" in pod_states:
        return "Pending"
    if "Failed" in pod_states:
        return "Failed"
    if pod_states and all(value == "Succeeded" for value in pod_states):
        return "Succeeded"
    return "Pending" if job_item else (pod_states[0] if pod_states else "Unknown")


def _percent(used: float, requested: float) -> Optional[float]:
    return used / requested * 100 if requested > 0 else None


def parse_cpu_cores(value: str) -> float:
    raw = str(value or "0").strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([num]?)", raw)
    if not match:
        return 0.0
    amount = float(match.group(1))
    return amount * {"": 1.0, "m": 1e-3, "u": 1e-6, "n": 1e-9}[match.group(2)]


def parse_memory_gib(value: str) -> float:
    raw = str(value or "0").strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([numkKMGTPE]i?|)", raw)
    if not match:
        return 0.0
    amount = float(match.group(1))
    unit = match.group(2)
    binary = {"Ki": 1024, "Mi": 1024 ** 2, "Gi": 1024 ** 3,
              "Ti": 1024 ** 4, "Pi": 1024 ** 5, "Ei": 1024 ** 6}
    decimal = {"k": 1000, "K": 1000, "M": 1000 ** 2, "G": 1000 ** 3,
               "T": 1000 ** 4, "P": 1000 ** 5, "E": 1000 ** 6}
    fractional = {"m": 1e-3, "u": 1e-6, "n": 1e-9}
    if unit in binary:
        bytes_value = amount * binary[unit]
    elif unit in decimal:
        bytes_value = amount * decimal[unit]
    elif unit in fractional:
        bytes_value = amount * fractional[unit]
    elif not unit:
        bytes_value = amount
    else:
        return 0.0
    return bytes_value / (1024 ** 3)


def _resource_values(container: Dict) -> Tuple[float, float, int]:
    resources = container.get("resources", {})
    requests = resources.get("requests", {}) or {}
    limits = resources.get("limits", {}) or {}
    gpu_value = requests.get(
        "nvidia.com/gpu", limits.get("nvidia.com/gpu", 0)
    )
    return (
        parse_cpu_cores(requests.get("cpu", "0")),
        parse_memory_gib(requests.get("memory", "0")),
        int(gpu_value or 0),
    )


def _pod_request(spec: Dict) -> Tuple[float, float, int]:
    """Return the scheduler-facing request for one Pod template/spec.

    Application containers run together and are summed.  Init containers run
    serially, so Kubernetes uses the largest init request per resource and the
    larger of that value and the application sum.  Pod overhead is then added.
    """
    regular = [_resource_values(item) for item in spec.get("containers", [])]
    init = [_resource_values(item) for item in spec.get("initContainers", [])]
    regular_sum = tuple(sum(values[index] for values in regular) for index in range(3))
    init_max = tuple(max((values[index] for values in init), default=0) for index in range(3))
    overhead = spec.get("overhead", {}) or {}
    return (
        max(regular_sum[0], init_max[0])
        + parse_cpu_cores(overhead.get("cpu", "0")),
        max(regular_sum[1], init_max[1])
        + parse_memory_gib(overhead.get("memory", "0")),
        int(max(regular_sum[2], init_max[2])),
    )


def _gpu_model(spec: Dict, metadata: Optional[Dict] = None) -> str:
    metadata = metadata or {}
    selectors = spec.get("nodeSelector", {}) or {}
    candidates = [
        selectors.get("gpu-type"),
        selectors.get("nvidia.com/gpu.product"),
        selectors.get("nvidia_com_gpu_product"),
        (metadata.get("labels", {}) or {}).get("falcon.dev/gpu-type"),
        (metadata.get("annotations", {}) or {}).get("falcon.dev/gpu-type"),
    ]
    return next((str(value) for value in candidates if value), "-")


def _short_cpu(value: float) -> str:
    if value < 0.01:
        return f"{value * 1000:.0f}m"
    if value < 10:
        return f"{value:.1f}c"
    return f"{value:.0f}c"


def _short_memory(value: float) -> str:
    if value < 1:
        return f"{value * 1024:.0f}M"
    if value < 10:
        return f"{value:.1f}G"
    return f"{value:.0f}G"


def _metric_color(value: Optional[float]) -> str:
    """Compatibility wrapper for the shared Dashboard/Resources palette."""

    return metric_color(value)


def _job_sort_key(row: JobUsage) -> Tuple[bool, bool, bool, str]:
    """Keep eviction risks first and completed work at the very bottom."""
    succeeded = row.status == "Succeeded"
    return (succeeded, not row.at_risk, row.status != "Running", row.job)


def _kubectl(args: List[str], timeout: int = 15) -> Optional[str]:
    try:
        result = subprocess.run(["kubectl"] + args, capture_output=True, text=True, timeout=timeout)
        return result.stdout if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _age(timestamp: str) -> str:
    try:
        created = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        seconds = max(0, int((datetime.now(timezone.utc) - created).total_seconds()))
        if seconds < 120:
            return f"{seconds}s"
        if seconds < 7200:
            return f"{seconds // 60}m"
        if seconds < 172800:
            return f"{seconds // 3600}h"
        return f"{seconds // 86400}d"
    except (TypeError, ValueError):
        return "-"


def _gpu_metrics(namespace: str, pod: str) -> GpuSample:
    query = (
        "index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,"
        "power.draw,ecc.errors.uncorrected.volatile.total,driver_version"
    )
    output = _kubectl(
        ["exec", "-n", namespace, pod, "--", "nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        timeout=8,
    )
    sample = _parse_gpu_lines(output.splitlines() if output else [])
    processes = _gpu_processes(namespace, pod)
    _apply_gpu_process_utilization(
        processes,
        _gpu_process_utilization(namespace, pod),
    )
    _attach_gpu_processes(sample, processes)
    return sample


def _parse_gpu_process_lines(lines: List[str]) -> List[GpuProcess]:
    processes: List[GpuProcess] = []
    for line in lines:
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        try:
            pid = int(parts[1])
            memory = float(parts[3]) / 1024
        except (TypeError, ValueError):
            continue
        processes.append(
            GpuProcess(
                gpu_uuid=parts[0],
                pid=pid,
                name=parts[2] or "—",
                memory_used_gib=memory,
            )
        )
    return processes


def _gpu_processes(namespace: str, pod: str) -> List[GpuProcess]:
    query = "gpu_uuid,pid,process_name,used_gpu_memory"
    output = _kubectl(
        [
            "exec", "-n", namespace, pod, "--", "nvidia-smi",
            f"--query-compute-apps={query}", "--format=csv,noheader,nounits",
        ],
        timeout=8,
    )
    return _parse_gpu_process_lines(output.splitlines() if output else [])


def _gpu_process_utilization(namespace: str, pod: str) -> Dict[int, float]:
    output = _kubectl(
        [
            "exec", "-n", namespace, pod, "--", "nvidia-smi",
            "pmon", "-c", "1", "-s", "u",
        ],
        timeout=8,
    )
    return _parse_gpu_process_utilization(
        output.splitlines() if output else []
    )


def _parse_gpu_process_utilization(lines: List[str]) -> Dict[int, float]:
    utilization: Dict[int, float] = {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[1])
            gpu_percent = float(parts[3])
        except (TypeError, ValueError):
            continue
        utilization[pid] = gpu_percent
    return utilization


def _apply_gpu_process_utilization(
    processes: List[GpuProcess],
    utilization: Dict[int, float],
) -> None:
    for process in processes:
        process.gpu_utilization = utilization.get(process.pid)


def _attach_gpu_processes(
    sample: GpuSample,
    processes: List[GpuProcess],
) -> None:
    by_uuid: Dict[str, List[GpuProcess]] = {}
    for process in processes:
        by_uuid.setdefault(process.gpu_uuid, []).append(process)
    for device in sample.devices:
        device.processes = sorted(
            by_uuid.get(device.uuid, []),
            key=lambda process: (-float(process.memory_used_gib or 0), process.pid),
        )


def _parse_gpu_lines(lines: List[str]) -> GpuSample:
    utils: List[float] = []
    used = total = 0.0
    devices: List[GpuDevice] = []

    def number(value: str) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:  # Backward-compatible parser for cached/test samples.
            util, memory_used, memory_total = map(number, parts)
            if util is None or memory_used is None or memory_total is None:
                continue
            utils.append(util)
            used += memory_used / 1024
            total += memory_total / 1024
            devices.append(GpuDevice(
                index=len(devices), utilization=util,
                memory_used_gib=memory_used / 1024, memory_total_gib=memory_total / 1024,
            ))
            continue
        if len(parts) != 10:
            continue
        memory_used, memory_total = number(parts[3]), number(parts[4])
        utilization = number(parts[5])
        if utilization is not None:
            utils.append(utilization)
        if memory_used is not None:
            used += memory_used / 1024
        if memory_total is not None:
            total += memory_total / 1024
        ecc = number(parts[8])
        devices.append(GpuDevice(
            index=int(number(parts[0]) or len(devices)), name=parts[1] or "—", uuid=parts[2] or "—",
            memory_used_gib=None if memory_used is None else memory_used / 1024,
            memory_total_gib=None if memory_total is None else memory_total / 1024,
            utilization=utilization, temperature_c=number(parts[6]), power_w=number(parts[7]),
            ecc_errors=None if ecc is None else int(ecc), driver_version=parts[9] or "—",
        ))
    if not utils:
        return GpuSample(devices=devices)
    return GpuSample(sum(utils) / len(utils), used, total, len(devices), devices)


class StreamingGpuSampler:
    """Keep one nvidia-smi stream per running pod instead of execing each frame."""

    def __init__(self, namespace: str, interval_ms: int = 1000):
        self.namespace = namespace
        self.interval_ms = interval_ms
        self._lock = threading.Lock()
        self._processes: Dict[str, subprocess.Popen] = {}
        self._samples: Dict[str, GpuSample] = {}
        self._ready: Dict[str, threading.Event] = {}
        self._process_samples: Dict[str, List[GpuProcess]] = {}
        self._process_samples_at = 0.0
        self._pmon_streams: Dict[str, subprocess.Popen] = {}
        self._process_utilization: Dict[str, Dict[int, float]] = {}

    @staticmethod
    def _terminate(process: Optional[subprocess.Popen]) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=0.2)

    def _stop_pmon(self, pod: str) -> None:
        self._terminate(self._pmon_streams.pop(pod, None))
        with self._lock:
            self._process_utilization.pop(pod, None)

    def _stop(self, pod: str) -> None:
        process = self._processes.pop(pod, None)
        self._ready.pop(pod, None)
        self._terminate(process)
        self._stop_pmon(pod)
        with self._lock:
            self._samples.pop(pod, None)
            self._process_samples.pop(pod, None)

    def _start_pmon(self, pod: str) -> None:
        try:
            process = subprocess.Popen(
                [
                    "kubectl", "exec", "-n", self.namespace, pod, "--",
                    "nvidia-smi", "pmon", "-d", "1", "-s", "u",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            return
        self._pmon_streams[pod] = process

        def read_process_utilization() -> None:
            stream = process.stdout
            if stream is None:
                return
            for line in stream:
                values = _parse_gpu_process_utilization([line])
                if not values:
                    continue
                with self._lock:
                    self._process_utilization.setdefault(pod, {}).update(values)

        threading.Thread(
            target=read_process_utilization,
            name=f"falcon-pmon-{pod}",
            daemon=True,
        ).start()

    def _start(self, pod: str, gpu_count: int) -> None:
        query = (
            "index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,"
            "power.draw,ecc.errors.uncorrected.volatile.total,driver_version"
        )
        try:
            process = subprocess.Popen(
                [
                    "kubectl", "exec", "-n", self.namespace, pod, "--", "nvidia-smi",
                    f"--query-gpu={query}", "--format=csv,noheader,nounits",
                    f"--loop-ms={self.interval_ms}",
                ],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except OSError:
            return
        self._processes[pod] = process
        ready = self._ready[pod] = threading.Event()

        def read_samples() -> None:
            batch: List[str] = []
            stream = process.stdout
            if stream is None:
                return
            try:
                for line in stream:
                    line = line.strip()
                    if len(line.split(",")) not in {3, 10}:
                        continue
                    batch.append(line)
                    if len(batch) < gpu_count:
                        continue
                    sample = _parse_gpu_lines(batch[:gpu_count])
                    batch.clear()
                    if sample.utilization is not None:
                        with self._lock:
                            self._samples[pod] = sample
                        ready.set()
            finally:
                ready.set()

        threading.Thread(target=read_samples, name=f"falcon-gpu-{pod}", daemon=True).start()
        self._start_pmon(pod)

    def samples(self, pods: Dict[str, int]) -> Dict[str, GpuSample]:
        for pod in list(self._processes):
            process = self._processes[pod]
            if pod not in pods or process.poll() is not None:
                self._stop(pod)
        started: List[str] = []
        for pod, gpu_count in pods.items():
            if pod not in self._processes:
                self._start(pod, gpu_count)
                started.append(pod)
            elif (
                pod not in self._pmon_streams
                or self._pmon_streams[pod].poll() is not None
            ):
                self._stop_pmon(pod)
                self._start_pmon(pod)
        deadline = time.monotonic() + 0.5
        for pod in started:
            ready = self._ready.get(pod)
            if ready:
                ready.wait(max(0.0, deadline - time.monotonic()))
        now = time.monotonic()
        if pods and now - self._process_samples_at >= KUBERNETES_USAGE_SECONDS:
            pod_names = list(pods)
            with ThreadPoolExecutor(
                max_workers=min(8, max(1, len(pod_names)))
            ) as pool:
                process_samples = dict(
                    zip(
                        pod_names,
                        pool.map(
                            lambda pod: _gpu_processes(self.namespace, pod),
                            pod_names,
                        ),
                    )
                )
            with self._lock:
                self._process_samples.update(process_samples)
                for pod, pod_processes in process_samples.items():
                    active_pids = {process.pid for process in pod_processes}
                    self._process_utilization[pod] = {
                        pid: value
                        for pid, value in self._process_utilization.get(
                            pod, {}
                        ).items()
                        if pid in active_pids
                    }
            self._process_samples_at = now
        with self._lock:
            samples = {pod: self._samples.get(pod, GpuSample()) for pod in pods}
            processes = {
                pod: list(self._process_samples.get(pod, [])) for pod in pods
            }
            process_utilization = {
                pod: dict(self._process_utilization.get(pod, {}))
                for pod in pods
            }
        for pod, sample in samples.items():
            _apply_gpu_process_utilization(
                processes[pod], process_utilization[pod]
            )
            _attach_gpu_processes(sample, processes[pod])
        return samples

    def close(self) -> None:
        for pod in list(self._processes):
            self._stop(pod)


class UsageCollector:
    def __init__(
        self, namespace: str, thresholds: Dict[str, float], ema_alpha: float,
        job_filter: Optional[str] = None, ema_warmup_samples: int = EMA_WARMUP_SAMPLES,
        streaming_gpu: bool = False, metrics_url: Optional[str] = None,
        risk_average_samples: int = RISK_AVERAGE_SAMPLES,
        collect_availability: bool = True,
        resource_source: Optional[Any] = None,
    ):
        self.namespace = namespace
        self.thresholds = thresholds
        self.ema_alpha = ema_alpha
        self.job_filter = job_filter
        self.ema_warmup_samples = ema_warmup_samples
        self.ema: Dict[str, float] = {}
        self.ema_samples: Dict[str, int] = {}
        self.risk_average_samples = max(1, risk_average_samples)
        self.risk_samples: Dict[str, Deque[float]] = {}
        self._items: Optional[List[Dict]] = None
        self._items_at = 0.0
        self._live: Optional[Dict[str, Tuple[float, float]]] = None
        self._live_at = 0.0
        self._gpu_sampler = StreamingGpuSampler(namespace) if streaming_gpu else None
        self._event_cache: Dict[str, Tuple[float, List[JobEvent]]] = {}
        self.metrics_url = metrics_url
        self.collect_availability = collect_availability
        self.resource_source = resource_source
        self.gpu_availability: Dict[str, Tuple[int, int]] = {}
        self._availability_at = 0.0
        self.last_error = ""
        self.last_successful_refresh = 0.0

    def invalidate(self) -> None:
        self._items_at = 0.0
        self._live_at = 0.0

    def close(self) -> None:
        if self._gpu_sampler:
            self._gpu_sampler.close()
        if self.resource_source is not None:
            self.resource_source.close()

    def _refresh_gpu_availability(self, now: float) -> None:
        if now - self._availability_at < GPU_AVAILABILITY_SECONDS:
            return
        # Rate-limit both successful refreshes and failures. In particular, a
        # temporary local metrics outage must not turn the dashboard's visual
        # refresh loop into a storm of kube-state-metrics and kubectl calls.
        # Existing values remain the last-known-good display until replacement
        # data has been collected.
        self._availability_at = now
        if self.resource_source is not None:
            snapshot = self.resource_source.collect()
            availability = {
                item.model.lower(): (item.request_headroom, item.allocatable)
                for item in snapshot.gpu_availability.values()
            }
            if availability or not snapshot.stale:
                self.gpu_availability = availability
            if snapshot.stale:
                self.last_error = snapshot.error or "resource service snapshot is stale"
            return
        if self.metrics_url:
            try:
                nodes = fetch_nodes(self.metrics_url, timeout=5)
            except Exception:
                nodes = []
            if nodes:
                unschedulable_names = set()
                if any(not node.scheduling_info_available for node in nodes):
                    raw_nodes = _kubectl(["get", "nodes", "-o", "json"], timeout=15)
                    if isinstance(raw_nodes, str):
                        try:
                            for item in json.loads(raw_nodes).get("items", []):
                                spec = item.get("spec", {})
                                taints = spec.get("taints", []) or []
                                if spec.get("unschedulable") or any(
                                    taint.get("effect") in {"NoSchedule", "NoExecute"}
                                    for taint in taints
                                ):
                                    unschedulable_names.add(item.get("metadata", {}).get("name", ""))
                        except (TypeError, json.JSONDecodeError):
                            pass
                availability: Dict[str, Tuple[int, int]] = {}
                for node in nodes:
                    if node.unschedulable or node.name in unschedulable_names or not node.gpu_total:
                        continue
                    gpu_type = canonical_gpu(node.gpu_product)
                    free, total = availability.get(gpu_type, (0, 0))
                    availability[gpu_type] = (free + node.gpu_free, total + node.gpu_total)
                self.gpu_availability = availability
                return
        raw = _kubectl(["get", "nodes,pods", "--all-namespaces", "-o", "json"], timeout=15)
        if raw is None:
            return
        try:
            items = json.loads(raw).get("items", [])
        except json.JSONDecodeError:
            return
        nodes: Dict[str, Dict[str, object]] = {}
        for item in items:
            if item.get("kind") != "Node":
                continue
            metadata, spec, status = item.get("metadata", {}), item.get("spec", {}), item.get("status", {})
            total = int(status.get("allocatable", {}).get("nvidia.com/gpu", 0) or 0)
            if not total or spec.get("unschedulable"):
                continue
            labels = metadata.get("labels", {})
            product = (
                labels.get("nvidia.com/gpu.product")
                or labels.get("nvidia_com_gpu_product")
                or labels.get("gpu-type")
                or ""
            )
            nodes[metadata.get("name", "")] = {"type": canonical_gpu(product), "total": total, "used": 0}
        for item in items:
            if item.get("kind") != "Pod":
                continue
            status, spec = item.get("status", {}), item.get("spec", {})
            node = nodes.get(spec.get("nodeName", ""))
            if node is None or status.get("phase") in {"Succeeded", "Failed"}:
                continue
            used = 0
            for container in spec.get("containers", []):
                resources = container.get("resources", {})
                requests, limits = resources.get("requests", {}), resources.get("limits", {})
                used += int(requests.get("nvidia.com/gpu", limits.get("nvidia.com/gpu", 0)) or 0)
            node["used"] = int(node["used"]) + used
        availability: Dict[str, Tuple[int, int]] = {}
        for values in nodes.values():
            gpu_type = str(values["type"])
            if not gpu_type:
                continue
            free, total = availability.get(gpu_type, (0, 0))
            node_total, node_used = int(values["total"]), int(values["used"])
            availability[gpu_type] = (free + max(0, node_total - node_used), total + node_total)
        self.gpu_availability = availability

    def events(self, row: JobUsage, force: bool = False) -> List[JobEvent]:
        cached = self._event_cache.get(row.uid)
        now = time.monotonic()
        if cached and not force and now - cached[0] < EVENT_REFRESH_SECONDS:
            return cached[1]
        names = {row.job} | {
            name for name in row.attempt_pods if name and name != row.job
        }
        collected: Dict[Tuple[str, str, str, str], JobEvent] = {}
        # One namespace query is considerably cheaper than one subprocess per
        # historical attempt, especially for retried Jobs.
        raw = _kubectl(
            ["get", "events", "-n", self.namespace, "-o", "json"],
            timeout=15,
        )
        try:
            items = json.loads(raw).get("items", []) if raw is not None else []
        except json.JSONDecodeError:
            items = []
            raw = None
        for item in items:
            object_name = str(
                item.get("involvedObject", {}).get("name") or ""
            )
            if object_name not in names:
                continue
            metadata = item.get("metadata", {})
            timestamp = (
                item.get("eventTime")
                or item.get("lastTimestamp")
                or item.get("series", {}).get("lastObservedTime")
                or metadata.get("creationTimestamp", "")
            )
            event = JobEvent(
                timestamp=str(timestamp),
                event_type=str(item.get("type") or "Normal"),
                reason=str(item.get("reason") or "Unknown"),
                message=str(item.get("message") or ""),
                object_name=object_name,
                count=int(
                    item.get("count")
                    or item.get("series", {}).get("count")
                    or 1
                ),
            )
            key = (
                event.timestamp,
                event.reason,
                event.message,
                event.object_name,
            )
            previous = collected.get(key)
            if previous:
                previous.count = max(previous.count, event.count)
            else:
                collected[key] = event
        events = sorted(collected.values(), key=lambda value: _timestamp(value.timestamp))[-200:]
        if raw is not None:
            self._event_cache[row.uid] = (now, events)
            return events
        # Cache the failed attempt time while retaining the last valid data so
        # an API outage cannot trigger a subprocess storm every frame.
        retained = cached[1] if cached else []
        self._event_cache[row.uid] = (now, retained)
        return retained

    def _update_ema(self, job: str, utilization: float) -> float:
        samples = self.ema_samples.get(job, 0)
        previous = self.ema.get(job)
        if previous is None:
            ema = utilization
        elif samples < self.ema_warmup_samples:
            # A cumulative mean gives new agent processes a stable starting
            # value instead of treating their first frame as historical truth.
            ema = (previous * samples + utilization) / (samples + 1)
        else:
            ema = self.ema_alpha * utilization + (1 - self.ema_alpha) * previous
        self.ema[job] = ema
        self.ema_samples[job] = samples + 1
        return ema

    def _update_risk_average(self, job: str, utilization: float) -> float:
        samples = self.risk_samples.setdefault(job, deque(maxlen=self.risk_average_samples))
        samples.append(utilization)
        return sum(samples) / len(samples)

    def _eviction_risk(self, job: str, average: Optional[float], gpu_count: int, threshold: float) -> bool:
        """Flag only a complete rolling arithmetic average, never an individual frame."""
        return (
            average is not None
            and gpu_count > 0
            and len(self.risk_samples.get(job, ())) >= self.risk_average_samples
            and average < threshold
        )

    def collect(self) -> List[JobUsage]:
        now = time.monotonic()
        if self.collect_availability:
            self._refresh_gpu_availability(now)
        if self._items is None or now - self._items_at >= KUBERNETES_INVENTORY_SECONDS:
            # Job objects do not normally carry the controller's ``job-name``
            # Pod label, so a combined label-filtered query loses the durable
            # Job template.  Fetch the named Job and its attempts separately.
            if self.job_filter:
                raw_job = _kubectl(
                    [
                        "get", "job.batch", self.job_filter, "-n",
                        self.namespace, "-o", "json",
                    ]
                )
                raw_pods = _kubectl(
                    [
                        "get", "pods", "-n", self.namespace, "-l",
                        f"job-name={self.job_filter}", "-o", "json",
                    ]
                )
                raw = None
                if raw_job is not None and raw_pods is not None:
                    try:
                        raw = {
                            "items": [json.loads(raw_job)]
                            + list(json.loads(raw_pods).get("items", []))
                        }
                    except (TypeError, json.JSONDecodeError):
                        raw = None
            else:
                combined = _kubectl(
                    [
                        "get", "jobs.batch,pods", "-n", self.namespace,
                        "-o", "json",
                    ]
                )
                try:
                    raw = json.loads(combined) if combined is not None else None
                except json.JSONDecodeError:
                    raw = None
            # Advance the attempt timestamp on failure as well.  Retain the
            # last-known-good inventory and retry at the bounded cadence.
            self._items_at = now
            if raw is not None:
                self._items = raw.get("items", [])
                self.last_error = ""
                self.last_successful_refresh = time.time()
            else:
                self.last_error = "Kubernetes API unavailable or returned invalid JSON"
        if self._items is None:
            return []
        all_items = self._items
        job_items = {
            item.get("metadata", {}).get("name", ""): item
            for item in all_items if item.get("kind") == "Job"
        }
        items = [item for item in all_items if item.get("kind") != "Job"]
        if self._live is None or now - self._live_at >= KUBERNETES_USAGE_SECONDS:
            top_args = ["top", "pods", "-n", self.namespace]
            if self.job_filter:
                top_args.extend(["-l", f"job-name={self.job_filter}"])
            top_args.append("--no-headers")
            top = _kubectl(top_args, timeout=10)
            self._live_at = now
            if top is not None:
                self._live = {
                    parts[0]: (parse_cpu_cores(parts[1]), parse_memory_gib(parts[2]))
                    for line in top.splitlines() if len(parts := line.split()) >= 3
                }
        live = self._live or {}
        live_pods = set(live)
        running_gpu_pods: Dict[str, int] = {}
        for item in items:
            spec = item.get("spec", {})
            gpu_count = sum(
                int(container.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0) or 0)
                for container in spec.get("containers", [])
            )
            if item.get("status", {}).get("phase") == "Running" and gpu_count:
                running_gpu_pods[item.get("metadata", {}).get("name", "")] = gpu_count
        if self._gpu_sampler:
            samples = self._gpu_sampler.samples(running_gpu_pods)
        else:
            pod_names = list(running_gpu_pods)
            with ThreadPoolExecutor(max_workers=min(8, max(1, len(pod_names)))) as pool:
                samples = dict(
                    zip(pod_names, pool.map(lambda pod: _gpu_metrics(self.namespace, pod), pod_names))
                )

        def new_group(job_item: Optional[Dict]) -> Dict:
            template = (
                (job_item or {}).get("spec", {}).get("template", {})
                if job_item
                else {}
            )
            template_spec = template.get("spec", {}) or {}
            cpu_request, memory_request, gpu_request = _pod_request(template_spec)
            containers = template_spec.get("containers", []) or []
            primary = containers[0] if containers else {}
            return {
                "statuses": [],
                "active_nodes": set(),
                "requested_gpu_type": _gpu_model(
                    template_spec, template.get("metadata", {})
                ),
                "requested_gpu_count": gpu_request,
                "allocated_gpu_types": set(),
                "allocated_gpu_count": 0,
                "allocated_cpu": 0.0,
                "allocated_memory": 0.0,
                "pods": 0,
                "gpu_weighted": 0.0,
                "gpu_samples": 0,
                "vram_used": 0.0,
                "vram_total": 0.0,
                "cpu_used": 0.0,
                "cpu_requested": cpu_request,
                "memory_used": 0.0,
                "memory_requested": memory_request,
                "created": [],
                "pod_items": [],
                "job_item": job_item,
                "gpu_devices": [],
                "command": " ".join(
                    str(value)
                    for value in (
                        (primary.get("command") or [])
                        + (primary.get("args") or [])
                    )
                ),
            }

        groups: Dict[str, Dict] = {
            job: new_group(item) for job, item in job_items.items()
        }
        for item in items:
            metadata, spec, status = item.get("metadata", {}), item.get("spec", {}), item.get("status", {})
            is_running = status.get("phase") == "Running"
            pod = metadata.get("name", "")
            labels = metadata.get("labels", {})
            job = labels.get("job-name", pod)
            group = groups.setdefault(job, new_group(job_items.get(job)))
            group["pod_items"].append(item)
            group["statuses"].append(status.get("phase", "Unknown"))
            group["pods"] += 1
            group["created"].append(metadata.get("creationTimestamp", ""))
            if group["job_item"] is None and group["requested_gpu_count"] == 0:
                pod_cpu_request, pod_memory_request, pod_gpu_request = _pod_request(spec)
                group["cpu_requested"] = pod_cpu_request
                group["memory_requested"] = pod_memory_request
                group["requested_gpu_count"] = pod_gpu_request
                group["requested_gpu_type"] = _gpu_model(spec, metadata)
                containers = spec.get("containers", []) or []
                primary = containers[0] if containers else {}
                group["command"] = " ".join(
                    str(value)
                    for value in (
                        (primary.get("command") or [])
                        + (primary.get("args") or [])
                    )
                )
            if is_running:
                if spec.get("nodeName"):
                    group["active_nodes"].add(spec["nodeName"])
                pod_cpu, pod_memory = live.get(pod, (0.0, 0.0))
                group["cpu_used"] += pod_cpu
                group["memory_used"] += pod_memory
                allocated_cpu, allocated_memory, allocated_gpus = _pod_request(spec)
                group["allocated_cpu"] += allocated_cpu
                group["allocated_memory"] += allocated_memory
                group["allocated_gpu_count"] += allocated_gpus
                allocated_type = _gpu_model(spec, metadata)
                if allocated_type != "-":
                    group["allocated_gpu_types"].add(allocated_type)
            sample = samples.get(pod, GpuSample())
            if sample.utilization is not None:
                group["gpu_weighted"] += sample.utilization * sample.gpu_count
                group["gpu_samples"] += sample.gpu_count
            group["vram_used"] += sample.memory_used_gib
            group["vram_total"] += sample.memory_total_gib
            group["gpu_devices"].extend(sample.devices)

        result: List[JobUsage] = []
        for job, group in groups.items():
            utilization = (
                group["gpu_weighted"] / group["gpu_samples"] if group["gpu_samples"] else None
            )
            cpu_metrics_available = any(
                pod.get("metadata", {}).get("name", "") in live_pods
                for pod in group["pod_items"]
                if pod.get("status", {}).get("phase") == "Running"
            )
            ema = None
            risk_average = None
            if utilization is not None:
                ema = self._update_ema(job, utilization)
                risk_average = self._update_risk_average(job, utilization)
            active_attempts = [
                pod for pod in group["pod_items"]
                if pod.get("status", {}).get("phase")
                not in {"Succeeded", "Failed"}
            ]
            relevant = _active_pod(active_attempts)
            relevant_metadata = relevant.get("metadata", {}) if relevant else {}
            relevant_status = relevant.get("status", {}) if relevant else {}
            requested_gpu_type = (
                group["requested_gpu_type"]
                if group["requested_gpu_count"] > 0
                else "-"
            )
            allocated_gpu_types = sorted(group["allocated_gpu_types"])
            allocated_gpu_type = (
                ",".join(allocated_gpu_types)
                if group["allocated_gpu_count"] > 0
                else "-"
            )
            threshold = self.thresholds.get(
                canonical_gpu(requested_gpu_type), 30
            )
            at_risk = self._eviction_risk(
                job,
                risk_average,
                group["allocated_gpu_count"],
                threshold,
            )
            statuses = group["statuses"]
            status = _job_status(group.get("job_item"), statuses)
            job_metadata = (group.get("job_item") or {}).get("metadata", {})
            job_spec = (group.get("job_item") or {}).get("spec", {})
            created = job_metadata.get("creationTimestamp") or min(
                (value for value in group["created"] if value), default=""
            )
            owner = next(
                (value for value in relevant_metadata.get("ownerReferences", []) if value.get("kind") == "Job"),
                {},
            )
            uid = str(job_metadata.get("uid") or owner.get("uid") or job)
            restarts = sum(
                int(container_status.get("restartCount", 0) or 0)
                for pod in group["pod_items"]
                for status_key in (
                    "initContainerStatuses",
                    "containerStatuses",
                )
                for container_status in (
                    pod.get("status", {}).get(status_key, []) or []
                )
            )
            succeeded_attempts = sum(
                value == "Succeeded" for value in statuses
            )
            failed_attempts = sum(value == "Failed" for value in statuses)
            completions = str(job_spec.get("completions", 1)) if group.get("job_item") else "—"
            result.append(JobUsage(
                job=job,
                status=status,
                nodes=",".join(sorted(group["active_nodes"])) or "—",
                # Compatibility fields now mean durable requested resources.
                gpu_type=requested_gpu_type,
                gpu_count=group["requested_gpu_count"],
                pod_count=group["pods"],
                gpu_util=utilization,
                gpu_ema=ema,
                gpu_memory_used_gib=group["vram_used"],
                gpu_memory_total_gib=group["vram_total"],
                cpu_used=group["cpu_used"],
                cpu_requested=group["cpu_requested"],
                memory_used_gib=group["memory_used"],
                memory_requested_gib=group["memory_requested"],
                age=_age(created),
                at_risk=at_risk,
                uid=uid,
                active_pod=str(relevant_metadata.get("name") or ""),
                active_pod_uid=str(relevant_metadata.get("uid") or ""),
                active_pod_state=_pod_state(relevant) if relevant else "No active pod",
                command=group["command"],
                created_at=created,
                started_at=str(relevant_status.get("startTime") or ""),
                restarts=restarts,
                completions=completions,
                metrics_updated_at=(
                    time.time()
                    if utilization is not None or cpu_metrics_available
                    else 0.0
                ),
                gpu_metrics_available=utilization is not None,
                cpu_metrics_available=cpu_metrics_available,
                gpu_devices=group["gpu_devices"],
                gpu_risk_average=risk_average,
                gpu_risk_threshold=threshold,
                gpu_requested_type=requested_gpu_type,
                gpu_requested_count=group["requested_gpu_count"],
                gpu_allocated_type=allocated_gpu_type,
                gpu_allocated_count=group["allocated_gpu_count"],
                cpu_allocated=group["allocated_cpu"],
                memory_allocated_gib=group["allocated_memory"],
                container_restarts=restarts,
                pod_attempts=group["pods"],
                succeeded_attempts=succeeded_attempts,
                failed_attempts=failed_attempts,
                backoff_limit=job_spec.get("backoffLimit"),
                attempt_pods=[
                    str(pod.get("metadata", {}).get("name") or "")
                    for pod in group["pod_items"]
                    if pod.get("metadata", {}).get("name")
                ],
            ))
        return sorted(result, key=_job_sort_key)


# The full-screen implementation lives separately so collection remains
# testable without initializing Textual view state.
from .dashboard_ui import FalconDashboard as FalconDashboard  # noqa: E402


class DemoUsageCollector:
    """Adapt the shared deterministic cluster fixture to the Job dashboard.

    The demo remains on the same durable :class:`JobSnapshot` semantics used
    by ``falcon jobs`` and ``falcon resources``; only synthetic utilization is
    added here because the Job dashboard visualizes telemetry.
    """

    def __init__(self, state: str = "mixed") -> None:
        from .demo import DemoCollector

        self.source = DemoCollector(state)
        self.last_error = (
            "demo Kubernetes API timeout; displaying last valid inventory"
            if state == "stale"
            else ""
        )
        self.last_successful_refresh = time.time()
        self.gpu_availability: Dict[str, Tuple[int, int]] = {}

    def collect(self) -> List[JobUsage]:
        snapshot = self.source.collect()
        self.last_error = snapshot.error or ""
        self.last_successful_refresh = time.time()
        self.gpu_availability = {
            item.model.lower(): (item.request_headroom, item.allocatable)
            for item in snapshot.gpu_availability.values()
        }
        rows: List[JobUsage] = []
        for index, job in enumerate(snapshot.jobs):
            requested = job.requested
            allocated = job.allocated
            running = job.status == "Running"
            has_metrics = running and "missing-metrics" not in job.name
            risk_threshold = (
                75.0
                if canonical_gpu(job.gpu_requested_model or "")
                in {"h100", "pro6000"}
                else 30.0
            )
            utilization = (
                8.0
                if "eviction-risk" in job.name
                else (42.0 + (index * 11) % 51 if has_metrics and allocated.gpu_count else None)
            )
            cpu_used = requested.cpu_cores * (0.25 + (index % 5) * 0.1) if running else 0.0
            memory_used = requested.memory_gib * (0.30 + (index % 4) * 0.1) if running else 0.0
            rows.append(
                JobUsage(
                    job=job.name,
                    status=job.status,
                    nodes=",".join(job.nodes) or "—",
                    gpu_type=job.gpu_requested_model or "-",
                    gpu_count=job.gpu_requested,
                    pod_count=job.attempts.pod_attempts,
                    gpu_util=utilization,
                    gpu_ema=utilization,
                    gpu_memory_used_gib=(
                        allocated.gpu_count * 21.5 if has_metrics else 0.0
                    ),
                    gpu_memory_total_gib=(
                        allocated.gpu_count * 80.0 if has_metrics else 0.0
                    ),
                    cpu_used=cpu_used,
                    cpu_requested=requested.cpu_cores,
                    memory_used_gib=memory_used,
                    memory_requested_gib=requested.memory_gib,
                    age=f"{index + 1}h",
                    at_risk="eviction-risk" in job.name,
                    uid=job.uid,
                    active_pod=job.attempts.active_pod or "",
                    active_pod_state="Running" if running else "No active pod",
                    command=" ".join(job.command),
                    created_at=job.created_at or "",
                    started_at=job.started_at or "",
                    restarts=job.attempts.container_restarts,
                    completions=str(job.attempts.succeeded_attempts),
                    metrics_updated_at=self.last_successful_refresh,
                    gpu_metrics_available=utilization is not None,
                    cpu_metrics_available=running,
                    gpu_risk_average=utilization,
                    gpu_risk_threshold=risk_threshold,
                    gpu_requested_type=job.gpu_requested_model or "-",
                    gpu_requested_count=job.gpu_requested,
                    gpu_allocated_type=job.gpu_allocated_model or "-",
                    gpu_allocated_count=job.gpu_allocated,
                    cpu_allocated=allocated.cpu_cores,
                    memory_allocated_gib=allocated.memory_gib,
                    container_restarts=job.attempts.container_restarts,
                    pod_attempts=job.attempts.pod_attempts,
                    succeeded_attempts=job.attempts.succeeded_attempts,
                    failed_attempts=job.attempts.failed_attempts,
                    backoff_limit=job.attempts.backoff_limit,
                    attempt_pods=list(job.attempts.active_pods),
                )
            )
        return sorted(rows, key=_job_sort_key)

    def events(self, row: JobUsage, force: bool = False) -> List[JobEvent]:
        del force
        values = self.source.events(row.job, 100)
        return [
            JobEvent(
                timestamp=str(item.get("lastTimestamp") or ""),
                event_type=str(item.get("type") or "Normal"),
                reason=str(item.get("reason") or "Unknown"),
                message=str(item.get("message") or ""),
                object_name=str(
                    (item.get("involvedObject") or {}).get("name") or row.job
                ),
                count=int(item.get("count") or 1),
            )
            for item in values
        ]

    def close(self) -> None:
        self.source.close()


def _coder_workspace_action(
    config: Mapping[str, object],
    job_name: str,
    action: str,
) -> None:
    """Run a dashboard action through the owning Coder control plane."""

    if action not in {"delete", "restart"}:
        raise CoderError(f"unsupported dashboard Coder action {action!r}")
    url, token = resolve_connection(config)
    coder_config = config.get("coder", {})
    timeout = float(
        coder_config.get("wait_timeout_seconds", 600)
        if isinstance(coder_config, Mapping)
        else 600
    )
    with CoderClient(url, token) as client:
        user = client.current_user()
        username = str(user.get("username") or user.get("name") or "")
        if not username:
            raise CoderError("Coder did not return the current username")
        workspace = client.workspace_for_job(job_name, username=username)
        name = str(workspace.get("name") or "")
        if not name:
            raise CoderError("Coder workspace response is missing its name")
        if action == "delete":
            client.delete_workspace(workspace, timeout=timeout)
        else:
            client.restart_workspace(
                "me",
                name,
                timeout=timeout,
            )


def run_dashboard(
    config: Dict,
    namespace: Optional[str] = None,
    job: Optional[str] = None,
    config_file: Optional[str] = None,
    demo_state: Optional[str] = None,
    color_mode: Optional[str] = None,
) -> None:
    from .resources_history import stop_legacy_history_collector

    namespace = namespace or config["cluster"]["namespace"]
    thresholds = {
        preset["gpu_type"].lower(): float(preset.get("minimum_utilization", 30))
        for preset in config["presets"].values()
    }
    dashboard = config.get("dashboard", {})
    collector = (
        DemoUsageCollector(demo_state)
        if demo_state
        else UsageCollector(
            namespace, thresholds, float(dashboard.get("ema_alpha", DEFAULT_DASHBOARD_EMA_ALPHA)),
            job_filter=job,
            ema_warmup_samples=EMA_WARMUP_SAMPLES,
            streaming_gpu=True,
            metrics_url=(
                config.get("cluster", {}).get("kube_state_metrics_url")
                if config.get("cluster", {}).get("resource_service_url") is None
                else None
            ),
            resource_source=(
                ResourceServiceCollector(
                    str(config["cluster"]["resource_service_url"]),
                    on_connect=lambda: stop_legacy_history_collector(config),
                )
                if config.get("cluster", {}).get("resource_service_url") is not None
                else None
            ),
            risk_average_samples=RISK_AVERAGE_SAMPLES,
        )
    )

    FalconDashboard(
        collector,
        hidden_panes=dashboard.get("hidden_panes", []),
        sort_field=dashboard.get("sort_field", "Age"),
        sort_direction=dashboard.get("sort_direction", "desc"),
        persist_hidden_panes=lambda panes: save_hidden_panes(panes, config_file),
        persist_sort=lambda field, direction: save_dashboard_sort(field, direction, config_file),
        coder_workspace_action=(
            None
            if demo_state
            else lambda job_name, action: _coder_workspace_action(
                config, job_name, action
            )
        ),
        color_mode=color_mode,
    ).run(mouse=True)
