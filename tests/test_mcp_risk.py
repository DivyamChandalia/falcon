from __future__ import annotations

import pytest

from falcon.mcp_config import DEFAULT_EVICTION_STARTUP_SECONDS, MCP_RISK_MIN_WINDOW_SECONDS
from falcon.mcp_risk import McpGpuRiskEvaluator


def _device(index: int, utilization: float) -> dict:
    return {
        "index": index,
        "name": "GPU",
        "uuid": f"gpu-{index}",
        "utilization": utilization,
        "memory_used_gib": 0.5,
        "memory_total_gib": 1.0,
    }


def test_mcp_defaults_use_startup_grace_and_thirty_second_minimum_window() -> None:
    evaluator = McpGpuRiskEvaluator()

    assert evaluator.startup_time_seconds == DEFAULT_EVICTION_STARTUP_SECONDS
    assert evaluator.min_window_seconds == MCP_RISK_MIN_WINDOW_SECONDS == 30.0
    assert evaluator.max_window_seconds == 300.0


@pytest.mark.parametrize("cadence", [1.37, 5.21, 17.3])
def test_low_vram_emits_risk_with_non_aligned_sampling(cadence):
    evaluator = McpGpuRiskEvaluator()
    device = _device(0, 99)
    device.update(memory_used_gib=1, memory_total_gib=96)
    first_risk_at = None
    for i in range(200):
        now = i * cadence
        risk = evaluator.evaluate("vram-job", [device], 75, now=now)
        assert not risk["gpu_at_risk"]
        if now < evaluator.startup_time_seconds + evaluator.min_window_seconds:
            assert not risk["at_risk"]
        if risk["vram_at_risk"]:
            first_risk_at = now
            break
    assert first_risk_at is not None, "sampling jitter must not suppress VRAM risk forever"
    assert first_risk_at <= (
        evaluator.startup_time_seconds + evaluator.min_window_seconds + 2 * cadence
    )
    assert risk["devices"][0]["vram_at_risk"]
    assert risk["vram_average"] == pytest.approx(100 / 96)


def test_healthy_irregular_samples_qualify_and_grow_the_window():
    evaluator = McpGpuRiskEvaluator()
    device = _device(0, 99)
    device.update(memory_used_gib=95, memory_total_gib=96)
    for i in range(200):
        risk = evaluator.evaluate("healthy-job", [device], 75, now=i * 1.37)
        assert not risk["at_risk"]
    assert risk["phase"] == "steady"
    assert risk["window_seconds"] > 30


def test_mcp_risk_is_evaluated_per_gpu_after_startup_and_window_grace() -> None:
    evaluator = McpGpuRiskEvaluator()
    evaluator.startup_time_seconds = 2.0
    evaluator.min_window_seconds = 3.0
    evaluator.max_window_seconds = 8.0
    devices = [_device(0, 80.0), _device(1, 5.0)]

    for now in (0.0, 1.0, 2.0, 3.0, 4.0):
        risk = evaluator.evaluate("job-a", devices, 10.0, now=now)
        assert risk is not None
        assert risk["at_risk"] is False
    risk = evaluator.evaluate("job-a", devices, 10.0, now=5.0)

    assert risk is not None
    assert risk["at_risk"] is True
    assert risk["gpu_at_risk"] is True
    assert risk["vram_at_risk"] is False
    assert risk["phase"] == "qualification"
    assert any(item["index"] == 1 and item["at_risk"] for item in risk["devices"])


def test_mcp_qualified_window_grows_and_does_not_debounce_for_five_minutes() -> None:
    evaluator = McpGpuRiskEvaluator()
    evaluator.startup_time_seconds = 2.0
    evaluator.min_window_seconds = 3.0
    evaluator.max_window_seconds = 8.0
    devices = [_device(0, 18.0)]

    for now in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        risk = evaluator.evaluate("job-a", devices, 10.0, now=now)
    assert risk is not None
    assert risk["at_risk"] is False
    assert risk["phase"] == "steady"

    devices[0]["utilization"] = 0.0
    for now in (6.0, 7.0, 8.0, 9.0):
        risk = evaluator.evaluate("job-a", devices, 10.0, now=now)

    assert risk is not None
    assert risk["devices"][0]["window_seconds"] == 7.0
    assert risk["at_risk"] is True
