"""Kubernetes Job metric collection and bounded Falcon dashboard snapshots."""

from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static

from .commands import top as open_job_top
from .config import DEFAULT_DASHBOARD_EMA_ALPHA, save_dashboard_sort, save_hidden_panes
from .resources import canonical_gpu, fetch_nodes


DASHBOARD_REFRESH_SECONDS = 1.0
KUBERNETES_INVENTORY_SECONDS = 5.0
KUBERNETES_USAGE_SECONDS = 5.0
EVENT_REFRESH_SECONDS = 5.0
GPU_AVAILABILITY_SECONDS = 15.0
ABSOLUTE_MINIMUM_WIDTH = 42
ABSOLUTE_MINIMUM_HEIGHT = 14
EMA_WARMUP_SAMPLES = 5
RISK_AVERAGE_SAMPLES = 60
AGENT_DEFAULT_SAMPLES = 5


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

    @property
    def gpu_memory_percent(self) -> Optional[float]:
        return _percent(self.gpu_memory_used_gib, self.gpu_memory_total_gib) if self.gpu_metrics_available else None

    @property
    def cpu_percent(self) -> Optional[float]:
        return _percent(self.cpu_used, self.cpu_requested) if self.cpu_metrics_available else None

    @property
    def memory_percent(self) -> Optional[float]:
        return _percent(self.memory_used_gib, self.memory_requested_gib) if self.cpu_metrics_available else None


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
        if status.get("active"):
            return "Running"
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
    if value is None:
        return "#666666"
    if value >= 80:
        return "#FF5555"
    if value >= 30:
        return "#FFFF55"
    return "#55FF55"


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
    return _parse_gpu_lines(output.splitlines() if output else [])


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

    def _stop(self, pod: str) -> None:
        process = self._processes.pop(pod, None)
        self._ready.pop(pod, None)
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.2)
        with self._lock:
            self._samples.pop(pod, None)

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
        deadline = time.monotonic() + 0.5
        for pod in started:
            ready = self._ready.get(pod)
            if ready:
                ready.wait(max(0.0, deadline - time.monotonic()))
        with self._lock:
            return {pod: self._samples.get(pod, GpuSample()) for pod in pods}

    def close(self) -> None:
        for pod in list(self._processes):
            self._stop(pod)


class UsageCollector:
    def __init__(
        self, namespace: str, thresholds: Dict[str, float], ema_alpha: float,
        job_filter: Optional[str] = None, ema_warmup_samples: int = EMA_WARMUP_SAMPLES,
        streaming_gpu: bool = False, metrics_url: Optional[str] = None,
        risk_average_samples: int = RISK_AVERAGE_SAMPLES,
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

    def _refresh_gpu_availability(self, now: float) -> None:
        if now - self._availability_at < GPU_AVAILABILITY_SECONDS:
            return
        if self.metrics_url:
            try:
                nodes = fetch_nodes(self.metrics_url, timeout=5)
            except Exception:
                nodes = []
            if nodes:
                availability: Dict[str, Tuple[int, int]] = {}
                for node in nodes:
                    if node.unschedulable or not node.gpu_total:
                        continue
                    gpu_type = canonical_gpu(node.gpu_product)
                    free, total = availability.get(gpu_type, (0, 0))
                    availability[gpu_type] = (free + node.gpu_free, total + node.gpu_total)
                self.gpu_availability = availability
                self._availability_at = now
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
        self._availability_at = now

    def events(self, row: JobUsage, force: bool = False) -> List[JobEvent]:
        cached = self._event_cache.get(row.uid)
        now = time.monotonic()
        if cached and not force and now - cached[0] < EVENT_REFRESH_SECONDS:
            return cached[1]
        names = [row.job] + ([row.active_pod] if row.active_pod else [])
        collected: Dict[Tuple[str, str, str, str], JobEvent] = {}
        had_response = False
        for name in names:
            raw = _kubectl([
                "get", "events", "-n", self.namespace,
                "--field-selector", f"involvedObject.name={name}", "-o", "json",
            ])
            if raw is None:
                continue
            had_response = True
            try:
                items = json.loads(raw).get("items", [])
            except json.JSONDecodeError:
                continue
            for item in items:
                metadata = item.get("metadata", {})
                timestamp = (
                    item.get("eventTime") or item.get("lastTimestamp")
                    or item.get("series", {}).get("lastObservedTime")
                    or metadata.get("creationTimestamp", "")
                )
                event = JobEvent(
                    timestamp=str(timestamp),
                    event_type=str(item.get("type") or "Normal"),
                    reason=str(item.get("reason") or "Unknown"),
                    message=str(item.get("message") or ""),
                    object_name=str(item.get("involvedObject", {}).get("name") or name),
                    count=int(item.get("count") or item.get("series", {}).get("count") or 1),
                )
                key = (event.timestamp, event.reason, event.message, event.object_name)
                previous = collected.get(key)
                if previous:
                    previous.count = max(previous.count, event.count)
                else:
                    collected[key] = event
        events = sorted(collected.values(), key=lambda value: _timestamp(value.timestamp))[-200:]
        if had_response:
            self._event_cache[row.uid] = (now, events)
            return events
        return cached[1] if cached else []

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
        self._refresh_gpu_availability(now)
        if self._items is None or now - self._items_at >= KUBERNETES_INVENTORY_SECONDS:
            get_args = ["get", "jobs.batch,pods", "-n", self.namespace]
            if self.job_filter:
                get_args.extend(["-l", f"job-name={self.job_filter}"])
            get_args.extend(["-o", "json"])
            raw = _kubectl(get_args)
            if raw:
                try:
                    self._items = json.loads(raw).get("items", [])
                    self._items_at = now
                    self.last_error = ""
                    self.last_successful_refresh = time.time()
                except json.JSONDecodeError:
                    self.last_error = "invalid Kubernetes API response"
            else:
                self.last_error = "Kubernetes API unavailable"
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
            if top is not None:
                self._live = {
                    parts[0]: (parse_cpu_cores(parts[1]), parse_memory_gib(parts[2]))
                    for line in top.splitlines() if len(parts := line.split()) >= 3
                }
                self._live_at = now
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

        groups: Dict[str, Dict] = {}
        for job, item in job_items.items():
            groups[job] = {
                "statuses": [], "nodes": set(), "gpu_types": set(), "gpu_count": 0, "pods": 0,
                "gpu_weighted": 0.0, "gpu_samples": 0, "vram_used": 0.0, "vram_total": 0.0,
                "cpu_used": 0.0, "cpu_requested": 0.0, "memory_used": 0.0,
                "memory_requested": 0.0, "created": [], "pod_items": [], "job_item": item,
                "gpu_devices": [],
            }
        for item in items:
            metadata, spec, status = item.get("metadata", {}), item.get("spec", {}), item.get("status", {})
            is_running = status.get("phase") == "Running"
            pod = metadata.get("name", "")
            labels = metadata.get("labels", {})
            job = labels.get("job-name", pod)
            group = groups.setdefault(job, {
                "statuses": [], "nodes": set(), "gpu_types": set(), "gpu_count": 0, "pods": 0,
                "gpu_weighted": 0.0, "gpu_samples": 0, "vram_used": 0.0, "vram_total": 0.0,
                "cpu_used": 0.0, "cpu_requested": 0.0, "memory_used": 0.0,
                "memory_requested": 0.0, "created": [], "pod_items": [], "job_item": job_items.get(job),
                "gpu_devices": [],
            })
            group["pod_items"].append(item)
            group["statuses"].append(status.get("phase", "Unknown"))
            if spec.get("nodeName"):
                group["nodes"].add(spec["nodeName"])
            gpu_type = spec.get("nodeSelector", {}).get("gpu-type", labels.get("falcon.dev/gpu-type"))
            if gpu_type:
                group["gpu_types"].add(gpu_type)
            group["pods"] += 1
            group["created"].append(metadata.get("creationTimestamp", ""))
            if is_running:
                pod_cpu, pod_memory = live.get(pod, (0.0, 0.0))
                group["cpu_used"] += pod_cpu
                group["memory_used"] += pod_memory
                for container in spec.get("containers", []):
                    resources = container.get("resources", {})
                    requests = resources.get("requests", {})
                    limits = resources.get("limits", {})
                    group["cpu_requested"] += parse_cpu_cores(requests.get("cpu", "0"))
                    group["memory_requested"] += parse_memory_gib(requests.get("memory", "0"))
                    group["gpu_count"] += int(limits.get("nvidia.com/gpu", 0) or 0)
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
            relevant = _active_pod(group["pod_items"])
            relevant_metadata = relevant.get("metadata", {}) if relevant else {}
            relevant_spec = relevant.get("spec", {}) if relevant else {}
            relevant_status = relevant.get("status", {}) if relevant else {}
            gpu_types = sorted(group["gpu_types"])
            gpu_type = ",".join(gpu_types) if gpu_types else "-"
            threshold = max((self.thresholds.get(value.lower(), 30) for value in gpu_types), default=30)
            at_risk = self._eviction_risk(job, risk_average, group["gpu_count"], threshold)
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
            containers = relevant_spec.get("containers", [])
            primary = containers[0] if containers else {}
            command_parts = [
                str(value) for value in (primary.get("command") or []) + (primary.get("args") or [])
            ]
            restarts = sum(
                int(value.get("restartCount", 0) or 0)
                for value in relevant_status.get("containerStatuses", [])
            )
            completions = str(job_spec.get("completions", 1)) if group.get("job_item") else "—"
            result.append(JobUsage(
                job=job,
                status=status,
                nodes=str(relevant_spec.get("nodeName") or "—"),
                gpu_type=gpu_type,
                gpu_count=group["gpu_count"],
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
                command=" ".join(command_parts),
                created_at=created,
                started_at=str(relevant_status.get("startTime") or ""),
                restarts=restarts,
                completions=completions,
                metrics_updated_at=time.time() if utilization is not None or live else 0.0,
                gpu_metrics_available=utilization is not None,
                cpu_metrics_available=cpu_metrics_available,
                gpu_devices=group["gpu_devices"],
                gpu_risk_average=risk_average,
                gpu_risk_threshold=threshold,
            ))
        return sorted(result, key=_job_sort_key)


CSS = """
Screen { background: #050812; color: #dbe7ff; overflow: hidden; }
Header { background: #101a2e; color: #75fbd3; }
#summary { height: 3; padding: 1 2; background: #0d1526; color: #75fbd3; text-style: bold; }
#jobs { height: 1fr; padding: 0 1; background: #050812; overflow: hidden; }
Footer { background: #101a2e; color: #b7c9ee; }
"""


class FalconDashboard(App):
    TITLE = "Falcon"
    SUB_TITLE = "Job monitor"
    ENABLE_COMMAND_PALETTE = False
    CSS = CSS
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("ctrl+q", "quit", "Quit", show=False, priority=True),
        Binding("r", "update_data", "Refresh"),
        Binding("enter", "open_nvitop", "nvitop"),
        Binding("j", "next_job", "Next"), Binding("k", "previous_job", "Previous"),
        Binding("down", "next_job", "Next", show=False), Binding("up", "previous_job", "Previous", show=False),
    ]

    def __init__(self, collector: UsageCollector, refresh_seconds: float = DASHBOARD_REFRESH_SECONDS):
        super().__init__()
        self.collector = collector
        self.refresh_seconds = refresh_seconds
        self.rows: List[JobUsage] = []
        self.selected = 0
        self._refreshing = False
        self._result_queue: "queue.Queue[List[JobUsage]]" = queue.Queue(maxsize=1)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("◆ collecting job metrics…", id="summary")
        yield Static(id="jobs")
        yield Footer()

    def on_mount(self) -> None:
        self._request_update()
        self.set_interval(self.refresh_seconds, self._request_update)
        self.set_interval(0.2, self._drain_results)

    def on_unmount(self) -> None:
        close = getattr(self.collector, "close", None)
        if close:
            close()

    def on_resize(self, event: events.Resize) -> None:
        self._render()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.prevent_default()
        event.stop()
        self.action_next_job()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.prevent_default()
        event.stop()
        self.action_previous_job()

    def action_next_job(self) -> None:
        if self.rows:
            self.selected = min(len(self.rows) - 1, self.selected + 1)
            self._render()

    def action_previous_job(self) -> None:
        if self.rows:
            self.selected = max(0, self.selected - 1)
            self._render()

    def action_update_data(self) -> None:
        invalidate = getattr(self.collector, "invalidate", None)
        if invalidate:
            invalidate()
        self._request_update()

    def action_open_nvitop(self) -> None:
        if not self.rows:
            return
        if self.rows[self.selected].status == "Succeeded":
            return
        job = self.rows[self.selected].job
        try:
            with self.suspend():
                open_job_top(self.collector.namespace, job)
        except (OSError, ValueError) as exc:
            self.notify(str(exc), title=f"Could not open nvitop for {job}", severity="error")
        self._request_update()

    def _request_update(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True

        def collect() -> None:
            try:
                rows = self.collector.collect()
            except Exception:
                rows = []
            try:
                self._result_queue.put_nowait(rows)
            except queue.Full:
                pass

        threading.Thread(target=collect, name="falcon-dashboard-refresh", daemon=True).start()

    def _drain_results(self) -> None:
        try:
            self.rows = self._result_queue.get_nowait()
        except queue.Empty:
            return
        self._refreshing = False
        self.selected = min(self.selected, max(0, len(self.rows) - 1))
        self._render()

    @staticmethod
    def _bar(percent: Optional[float], width: int, color: str) -> Text:
        if percent is None:
            return Text("·" * width, style="#33415f")
        filled = max(0, min(width, round(width * percent / 100)))
        bar = Text("━" * filled, style=color)
        bar.append("─" * (width - filled), style="#26324c")
        return bar

    def _meter(
        self, label: str, values: str, percent: Optional[float], width: int, color: str, bars: bool
    ) -> Text:
        pct = "  --" if percent is None else f"{percent:4.0f}%"
        label_text = f"{label:<5} "
        values_text = f"{values:<13}"
        prefix_width = len(label_text) + len(values_text)
        line = Text(label_text, style="#b8c8e8")
        line.append(values_text, style=color)
        bar_width = width - prefix_width - len(pct) - 2
        if bars and bar_width >= 8:
            line.append_text(self._bar(percent, bar_width, color))
            line.append("  ")
        else:
            line.append(" ")
        line.append(pct, style=color)
        return line

    def _card(self, row: JobUsage, index: int, width: int, bars: bool) -> Panel:
        inner_width = max(20, width - 4)
        title_width = max(12, inner_width - 8)
        title = row.job if len(row.job) <= title_width else row.job[:title_width - 1] + "…"
        selected = index == self.selected
        marker = "▶ " if selected else ""
        if row.status == "Succeeded":
            details = f"{row.gpu_type}×{row.gpu_count}  {row.nodes}  {row.age}"
            available = max(8, inner_width - 12)
            if len(details) > available:
                details = details[:available - 1] + "…"
            body = Text("✓ Succeeded", style="bold #5cffb1")
            body.append(f"  {details}", style="#78909c")
            return Panel(
                body,
                title=Text(f" {marker}{index + 1} {title} ", style="bold bright_cyan" if selected else "#78909c"),
                subtitle=f" completed · {row.pod_count} pod{'s' if row.pod_count != 1 else ''} ",
                border_style="bright_cyan" if selected else "#315348",
                width=width,
                padding=(0, 1),
            )
        live = "--" if row.gpu_util is None else f"{row.gpu_util:.0f}%"
        ema = "--" if row.gpu_ema is None else f"{row.gpu_ema:.0f}%"
        live_color = _metric_color(row.gpu_util)
        ema_color = "#ff5f6d" if row.at_risk else _metric_color(row.gpu_ema)
        body = Text()
        node_width = max(8, inner_width - 32)
        nodes = row.nodes if len(row.nodes) <= node_width else row.nodes[:node_width - 1] + "…"
        body.append(f"{row.status:<8}", style="green" if row.status == "Running" else "yellow")
        body.append(f" {row.gpu_type}×{row.gpu_count}  {nodes}  {row.age}\n", style="#91a7d0")
        gpu_prefix = f"GPU   {live:>4}  EMA {ema:>4}"
        gpu_line = Text("GPU   ", style="#b8c8e8")
        gpu_line.append(f"{live:>4}", style=live_color)
        gpu_line.append("  EMA ", style="#b8c8e8")
        gpu_line.append(f"{ema:>4}", style=ema_color)
        gpu_bar_width = inner_width - len(gpu_prefix) - 1
        if bars and gpu_bar_width >= 8:
            gpu_line.append(" ")
            gpu_line.append_text(self._bar(row.gpu_util, gpu_bar_width, live_color))
        body.append_text(gpu_line)
        body.append("\n")
        body.append_text(self._meter(
            "VRAM", f"{_short_memory(row.gpu_memory_used_gib)}/{_short_memory(row.gpu_memory_total_gib)}",
            row.gpu_memory_percent, inner_width, _metric_color(row.gpu_memory_percent), bars,
        ))
        body.append("\n")
        body.append_text(self._meter(
            "CPU", f"{_short_cpu(row.cpu_used)}/{_short_cpu(row.cpu_requested)}",
            row.cpu_percent, inner_width, _metric_color(row.cpu_percent), bars,
        ))
        body.append("\n")
        body.append_text(self._meter(
            "RAM", f"{_short_memory(row.memory_used_gib)}/{_short_memory(row.memory_requested_gib)}",
            row.memory_percent, inner_width, _metric_color(row.memory_percent), bars,
        ))
        border = "bright_red" if row.at_risk else ("bright_cyan" if selected else "#3b527a")
        title_style = "bold bright_red" if row.at_risk else ("bold bright_cyan" if selected else "bold white")
        return Panel(
            body,
            title=Text(f" {marker}{index + 1} {title} ", style=title_style),
            subtitle=f" {row.pod_count} pod{'s' if row.pod_count != 1 else ''} ",
            border_style=border,
            style="on #25080e" if row.at_risk else "",
            width=width,
            padding=(0, 1),
        )

    @staticmethod
    def _card_height(row: JobUsage) -> int:
        return 3 if row.status == "Succeeded" else 7

    @classmethod
    def _visible_window(
        cls, rows: List[JobUsage], selected: int, columns: int, available_height: int
    ) -> Tuple[int, int, int]:
        """Pack mixed-height rows and include one clipped row as an overflow cue."""
        total = len(rows)
        row_heights = [
            max(cls._card_height(row) for row in rows[index:index + columns])
            for index in range(0, total, columns)
        ]
        selected_row = selected // columns
        start_row = 0
        while start_row < selected_row and sum(row_heights[start_row:selected_row + 1]) > available_height:
            start_row += 1
        used = 0
        full_row_end = start_row
        while full_row_end < len(row_heights):
            height = row_heights[full_row_end]
            if full_row_end > start_row and used + height > available_height:
                break
            used += height
            full_row_end += 1
        start = start_row * columns
        full_end = min(total, full_row_end * columns)
        render_end = min(total, (full_row_end + 1) * columns)
        return start, full_end, render_end

    def _render(self) -> None:
        width, height = self.size.width, self.size.height
        summary = self.query_one("#summary", Static)
        target = self.query_one("#jobs", Static)
        if width < ABSOLUTE_MINIMUM_WIDTH or height < ABSOLUTE_MINIMUM_HEIGHT:
            summary.update("[bold yellow]Terminal is too small for Falcon[/]")
            target.update(
                f"\n[bold]Minimum: {ABSOLUTE_MINIMUM_WIDTH}×{ABSOLUTE_MINIMUM_HEIGHT}[/]\n"
                f"Current: {width}×{height}"
            )
            return
        risk = sum(row.at_risk for row in self.rows)
        running = sum(row.status == "Running" for row in self.rows)
        summary.update(
            f"◆ {self.collector.namespace}   [green]{running} running[/]   "
            f"[bold red]{risk} eviction risk[/]   {len(self.rows)} jobs   1s sampling"
        )
        if not self.rows:
            target.update("\n[dim]No jobs found in this namespace.[/]")
            return
        columns = 2 if width >= 124 else 1
        gap = 2
        content_width = width - 2
        card_width = max(38, (content_width - gap * (columns - 1)) // columns)
        bars = card_width >= 56
        start, full_end, render_end = self._visible_window(
            self.rows, self.selected, columns, max(1, height - 6)
        )
        visible = self.rows[start:render_end]
        cards = [self._card(row, start + offset, card_width, bars) for offset, row in enumerate(visible)]
        rendered = (
            Columns(cards, width=card_width, padding=(0, gap // 2), equal=True, expand=True)
            if columns > 1
            else Group(*cards)
        )
        target.update(rendered)
        above = start
        below = len(self.rows) - full_end
        cues = []
        if above:
            cues.append(f"↑ {above} above")
        if below:
            cues.append(f"↓ {below} below")
        position = f"Jobs {start + 1}-{full_end} of {len(self.rows)}" if above or below else "Job monitor"
        self.sub_title = f"{position} · {' · '.join(cues + ['wheel/j/k navigate', 'Enter nvitop'])}"


# The full-screen implementation lives separately so the collector remains
# usable by bounded agent snapshots without initializing Textual view state.
from .dashboard_ui import FalconDashboard as FalconDashboard


def format_snapshot(
    rows: List[JobUsage], namespace: str, json_output: bool = False, sample_count: int = 1
) -> str:
    """Render one bounded, ANSI-free dashboard sample for agents and scripts."""
    if json_output:
        jobs = []
        for row in rows:
            item = asdict(row)
            item.update({
                "gpu_memory_percent": row.gpu_memory_percent,
                "cpu_percent": row.cpu_percent,
                "memory_percent": row.memory_percent,
            })
            jobs.append(item)
        return json.dumps({"job_count": len(rows), "sample_count": sample_count, "jobs": jobs}, sort_keys=True)

    risk = sum(row.at_risk for row in rows)
    lines = [f"jobs={len(rows)} samples={sample_count} eviction_risk={risk}"]
    percent = lambda value: "--" if value is None else f"{value:.0f}%"
    for row in rows:
        lines.append(
            f"job={row.job} status={row.status} gpu={row.gpu_type}x{row.gpu_count} "
            f"util={percent(row.gpu_util)} ema={percent(row.gpu_ema)} "
            f"vram={_short_memory(row.gpu_memory_used_gib)}/{_short_memory(row.gpu_memory_total_gib)}"
            f"({percent(row.gpu_memory_percent)}) "
            f"cpu={_short_cpu(row.cpu_used)}/{_short_cpu(row.cpu_requested)}({percent(row.cpu_percent)}) "
            f"ram={_short_memory(row.memory_used_gib)}/{_short_memory(row.memory_requested_gib)}"
            f"({percent(row.memory_percent)}) risk={'yes' if row.at_risk else 'no'}"
        )
    return "\n".join(lines)


def run_dashboard(
    config: Dict, namespace: Optional[str] = None, once: bool = False, json_output: bool = False,
    job: Optional[str] = None, samples: Optional[int] = None, sample_interval: float = 1.0,
    config_file: Optional[str] = None,
) -> None:
    namespace = namespace or config["cluster"]["namespace"]
    thresholds = {
        preset["gpu_type"].lower(): float(preset.get("minimum_utilization", 30))
        for preset in config["presets"].values()
    }
    dashboard = config.get("dashboard", {})
    snapshot = once or json_output or samples is not None or not sys.stdout.isatty()
    sample_count = AGENT_DEFAULT_SAMPLES if samples is None else samples
    if snapshot:
        if not 1 <= sample_count <= 120:
            raise ValueError("dashboard samples must be between 1 and 120")
        if not 0.2 <= sample_interval <= 60:
            raise ValueError("dashboard sample interval must be between 0.2 and 60 seconds")
    collector = UsageCollector(
        namespace, thresholds, float(dashboard.get("ema_alpha", DEFAULT_DASHBOARD_EMA_ALPHA)),
        job_filter=job, ema_warmup_samples=sample_count if snapshot else EMA_WARMUP_SAMPLES,
        streaming_gpu=not snapshot or sample_count > 1,
        metrics_url=config.get("cluster", {}).get("kube_state_metrics_url"),
        risk_average_samples=sample_count if snapshot else RISK_AVERAGE_SAMPLES,
    )
    if snapshot:
        try:
            rows: List[JobUsage] = []
            for index in range(sample_count):
                rows = collector.collect()
                if index + 1 < sample_count:
                    time.sleep(sample_interval)
            print(format_snapshot(rows, namespace, json_output=json_output, sample_count=sample_count))
        finally:
            collector.close()
        return
    FalconDashboard(
        collector,
        hidden_panes=dashboard.get("hidden_panes", []),
        sort_field=dashboard.get("sort_field", "Age"),
        sort_direction=dashboard.get("sort_direction", "desc"),
        persist_hidden_panes=lambda panes: save_hidden_panes(panes, config_file),
        persist_sort=lambda field, direction: save_dashboard_sort(field, direction, config_file),
    ).run(mouse=True)
