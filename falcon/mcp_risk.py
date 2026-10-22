"""MCP-only GPU eviction-risk evaluation.

The interactive dashboard has its own legacy risk display.  MCP monitoring
uses this evaluator so an agent can opt into a startup grace period and receive
per-GPU risk events without changing dashboard behavior.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple

from .mcp_config import (
    DEFAULT_EVICTION_STARTUP_SECONDS,
    MCP_RISK_EMA_TIME_CONSTANT_SECONDS,
    MCP_RISK_MAX_WINDOW_SECONDS,
    MCP_RISK_MIN_WINDOW_SECONDS,
    MIN_EVICTION_STARTUP_SECONDS,
)


@dataclass
class _GpuRiskState:
    first_seen: float
    last_seen: float
    samples: Deque[Tuple[float, Optional[float], Optional[float]]] = field(
        default_factory=deque
    )
    gpu_ema: Optional[float] = None
    vram_ema: Optional[float] = None
    last_gpu_sample_at: Optional[float] = None
    last_vram_sample_at: Optional[float] = None
    qualified: bool = False
    qualified_at: Optional[float] = None


def _value(device: Any, name: str, default: Any = None) -> Any:
    if isinstance(device, Mapping):
        return device.get(name, default)
    return getattr(device, name, default)


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percent(used: Any, total: Any) -> Optional[float]:
    used_value = _number(used)
    total_value = _number(total)
    if used_value is None or total_value is None or total_value <= 0:
        return None
    return used_value / total_value * 100.0


class McpGpuRiskEvaluator:
    """Evaluate rolling GPU/VRAM floors for MCP-observed Jobs."""

    def __init__(
        self,
        startup_time_seconds: float = DEFAULT_EVICTION_STARTUP_SECONDS,
        *,
        clock: Any = time.monotonic,
    ) -> None:
        self.startup_time_seconds = max(
            MIN_EVICTION_STARTUP_SECONDS,
            float(startup_time_seconds),
        )
        self.ema_time_constant_seconds = MCP_RISK_EMA_TIME_CONSTANT_SECONDS
        self.min_window_seconds = MCP_RISK_MIN_WINDOW_SECONDS
        self.max_window_seconds = MCP_RISK_MAX_WINDOW_SECONDS
        self._clock = clock
        self._states: Dict[str, Dict[str, _GpuRiskState]] = {}

    def set_startup_time_seconds(self, value: float) -> None:
        self.startup_time_seconds = max(
            MIN_EVICTION_STARTUP_SECONDS,
            float(value),
        )

    @staticmethod
    def _devices(devices: Any) -> List[Tuple[str, Any]]:
        if isinstance(devices, Mapping):
            values = list(devices.items())
        else:
            values = [(str(index), device) for index, device in enumerate(devices or [])]
        result: List[Tuple[str, Any]] = []
        used: Dict[str, int] = {}
        for fallback, device in values:
            index = _number(_value(device, "index", fallback))
            index_value = int(index) if index is not None else fallback
            uuid = str(_value(device, "uuid", "") or "").strip()
            identity = uuid if uuid not in {"", "-", "—"} else f"index:{index_value}"
            occurrence = used.get(identity, 0)
            used[identity] = occurrence + 1
            if occurrence:
                identity = f"{identity}#{occurrence + 1}"
            result.append((identity, device))
        return sorted(
            result,
            key=lambda item: (
                _number(_value(item[1], "index", 0)) or 0,
                item[0],
            ),
        )

    @staticmethod
    def _window_values(
        samples: Deque[Tuple[float, Optional[float], Optional[float]]],
        now: float,
        window_seconds: float,
        value_index: int,
    ) -> List[Tuple[float, float]]:
        result = []
        for timestamp, gpu_value, vram_value in samples:
            value = gpu_value if value_index == 1 else vram_value
            if timestamp >= now - window_seconds and value is not None:
                result.append((timestamp, value))
        return result

    def _ema(
        self,
        previous: Optional[float],
        value: float,
        previous_at: Optional[float],
        now: float,
    ) -> float:
        if previous is None or previous_at is None:
            return value
        elapsed = max(0.0, now - previous_at)
        alpha = 1.0 - math.exp(-elapsed / self.ema_time_constant_seconds)
        return previous + alpha * (value - previous)

    def evaluate(
        self,
        job: str,
        devices: Iterable[Any] | Mapping[str, Any],
        threshold: float,
        now: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return MCP risk details, or ``None`` if per-GPU data is absent."""

        device_values = self._devices(devices)
        if not device_values:
            return None
        current_time = self._clock() if now is None else float(now)
        states = self._states.setdefault(job, {})
        current_keys = {identity for identity, _device in device_values}
        for identity, state in list(states.items()):
            if (
                identity not in current_keys
                and current_time - state.last_seen > self.max_window_seconds
            ):
                states.pop(identity, None)

        records: List[Dict[str, Any]] = []
        gpu_averages: List[float] = []
        vram_averages: List[float] = []
        gpu_emas: List[float] = []
        vram_emas: List[float] = []
        window_lengths: List[float] = []
        startup_remaining: List[float] = []
        gpu_at_risk = False
        vram_at_risk = False

        for identity, device in device_values:
            gpu_value = _number(_value(device, "utilization"))
            vram_value = _percent(
                _value(device, "memory_used_gib"),
                _value(device, "memory_total_gib"),
            )
            if gpu_value is None and vram_value is None:
                continue
            state = states.get(identity)
            if state is None:
                state = _GpuRiskState(
                    first_seen=current_time,
                    last_seen=current_time,
                )
                states[identity] = state
            state.last_seen = current_time
            state.samples.append((current_time, gpu_value, vram_value))
            while state.samples and state.samples[0][0] < current_time - self.max_window_seconds:
                state.samples.popleft()
            if gpu_value is not None:
                state.gpu_ema = self._ema(
                    state.gpu_ema,
                    gpu_value,
                    state.last_gpu_sample_at,
                    current_time,
                )
                state.last_gpu_sample_at = current_time
            if vram_value is not None:
                state.vram_ema = self._ema(
                    state.vram_ema,
                    vram_value,
                    state.last_vram_sample_at,
                    current_time,
                )
                state.last_vram_sample_at = current_time

            window_seconds = self.min_window_seconds
            if state.qualified and state.qualified_at is not None:
                window_seconds = min(
                    self.min_window_seconds
                    + max(0.0, current_time - state.qualified_at),
                    self.max_window_seconds,
                )
            gpu_window = self._window_values(
                state.samples, current_time, window_seconds, 1
            )
            vram_window = self._window_values(
                state.samples, current_time, window_seconds, 2
            )
            gpu_average = (
                sum(value for _timestamp, value in gpu_window) / len(gpu_window)
                if gpu_window
                else None
            )
            vram_average = (
                sum(value for _timestamp, value in vram_window) / len(vram_window)
                if vram_window
                else None
            )
            if gpu_average is not None:
                gpu_averages.append(gpu_average)
            if vram_average is not None:
                vram_averages.append(vram_average)
            if state.gpu_ema is not None:
                gpu_emas.append(state.gpu_ema)
            if state.vram_ema is not None:
                vram_emas.append(state.vram_ema)
            window_lengths.append(window_seconds)

            elapsed = current_time - state.first_seen
            in_startup = elapsed < self.startup_time_seconds
            if in_startup:
                startup_remaining.append(self.startup_time_seconds - elapsed)
            qualification_start = state.first_seen + self.startup_time_seconds
            # Readiness measures observed history, not the already-clipped
            # average window. With real sampling jitter the oldest sample in
            # a 30s window is almost always less than 30s old, so using that
            # window here could prevent qualification (and risk) forever.
            post_start_gpu = [
                timestamp
                for timestamp, value, _vram in state.samples
                if timestamp >= qualification_start and value is not None
            ]
            post_start_vram = [
                timestamp
                for timestamp, _gpu, value in state.samples
                if timestamp >= qualification_start and value is not None
            ]
            gpu_ready = bool(post_start_gpu) and (
                current_time - min(post_start_gpu) >= self.min_window_seconds
            )
            vram_ready = bool(post_start_vram) and (
                current_time - min(post_start_vram) >= self.min_window_seconds
            )
            qualification_ready = elapsed >= (
                self.startup_time_seconds + self.min_window_seconds
            )
            gpu_low = gpu_average is not None and gpu_average < threshold
            vram_low = vram_average is not None and vram_average < threshold
            device_gpu_risk = False
            device_vram_risk = False
            if not in_startup and not state.qualified:
                if qualification_ready and gpu_ready:
                    device_gpu_risk = gpu_low
                if qualification_ready and vram_ready:
                    device_vram_risk = vram_low
                metrics_ready = (
                    (gpu_value is None or gpu_ready)
                    and (vram_value is None or vram_ready)
                )
                if qualification_ready and metrics_ready and not (
                    device_gpu_risk or device_vram_risk
                ):
                    state.qualified = True
                    state.qualified_at = current_time
            elif state.qualified:
                device_gpu_risk = gpu_low
                device_vram_risk = vram_low

            gpu_at_risk = gpu_at_risk or device_gpu_risk
            vram_at_risk = vram_at_risk or device_vram_risk
            records.append(
                {
                    "id": identity,
                    "index": int(_number(_value(device, "index", 0)) or 0),
                    "name": str(_value(device, "name", "—") or "—"),
                    "uuid": str(_value(device, "uuid", "—") or "—"),
                    "gpu_percent": gpu_value,
                    "vram_percent": vram_value,
                    "gpu_window_average": gpu_average,
                    "vram_window_average": vram_average,
                    "gpu_ema": state.gpu_ema,
                    "vram_ema": state.vram_ema,
                    "window_seconds": window_seconds,
                    "gpu_at_risk": device_gpu_risk,
                    "vram_at_risk": device_vram_risk,
                    "at_risk": device_gpu_risk or device_vram_risk,
                    "qualified": state.qualified,
                }
            )

        if not records:
            return None
        phase = "startup" if startup_remaining else "steady"
        if not startup_remaining and any(
            not states[record["id"]].qualified for record in records
        ):
            phase = "qualification"
        return {
            "available": True,
            "at_risk": gpu_at_risk or vram_at_risk,
            "gpu_at_risk": gpu_at_risk,
            "vram_at_risk": vram_at_risk,
            "gpu_average": min(gpu_averages) if gpu_averages else None,
            "vram_average": min(vram_averages) if vram_averages else None,
            "gpu_ema": min(gpu_emas) if gpu_emas else None,
            "vram_ema": min(vram_emas) if vram_emas else None,
            "window_seconds": min(window_lengths) if window_lengths else None,
            "phase": phase,
            "startup_remaining_seconds": (
                min(startup_remaining) if startup_remaining else 0.0
            ),
            "devices": records,
        }
