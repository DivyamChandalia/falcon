from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from falcon.mcp_events import EventStore, Monitor

REF = {"name": "train", "namespace": "mine", "uid": "job-uid"}


@pytest.fixture
def monitor(tmp_path):
    store = EventStore(tmp_path / "events.db")

    async def idle(resource, namespace, resource_version=None):
        await asyncio.Event().wait()
        yield {}

    value = Monitor(SimpleNamespace(executable="kubectl"), "mine", store, watcher=idle)
    yield value, store
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["FailedScheduling", "Unschedulable"])
@pytest.mark.parametrize("message,hidden", [
    ("0/3 nodes are available: 3 Insufficient nvidia.com/gpu.", True),
    ("0/3 nodes are available: 3 Insufficient cpu, 3 Insufficient memory.", True),
    ("0/3 nodes are available: 3 Too many pods.", True),
    ("0/3 nodes are available: 2 node(s) didn't match Pod's node affinity/selector, 1 Insufficient nvidia.com/gpu.", True),
    ("0/3 nodes are available: 3 node(s) didn't match Pod's node affinity/selector.", False),
    ("0/3 nodes are available: 2 Insufficient cpu, 1 pod has unbound immediate PersistentVolumeClaims.", False),
    ("0/3 nodes are available: 2 Insufficient cpu, 1 volume node affinity conflict.", False),
    ("Unknown scheduling failure", False),
    ("no nodes with enough resources were found: 2 node(s) didn't have enough resources: CPU cores.\n4 node(s) didn't have enough resources: memory.\n4 node(s) didn't match Pod's node affinity/selector.\n5 node(s) didn't have enough resources: GPUs.", True),
    ("no nodes with enough resources were found: 4 node(s) didn't match Pod's node affinity/selector.", False),
])
async def test_capacity_policy_filters_only_recognized_shortages(monitor, reason, message, hidden):
    observer, store = monitor
    record = store.append(REF, "warning", {"reason": reason, "message": message})
    try:
        result = await observer.wait([REF], cursor=0, timeout=0, scheduling_policy="wait_for_capacity")
        assert bool(result["events"]) is not hidden
        assert result["timed_out"] is hidden
        assert result["cursor"] >= record["cursor"]
        # The policy filters delivery; diagnostics remain available to other waiters.
        assert store.read([REF["uid"]], event_types=["warning"])
    finally:
        await observer.close()


@pytest.mark.asyncio
async def test_report_blocked_default_and_invalid_policy(monitor):
    observer, store = monitor
    store.append(REF, "warning", {"reason": "FailedScheduling", "message": "Insufficient memory"})
    try:
        result = await observer.wait([REF], cursor=0, timeout=0)
        assert result["events"][0]["type"] == "warning"
        with pytest.raises(ValueError, match="scheduling_policy"):
            await observer.wait([REF], timeout=0, scheduling_policy="later")
    finally:
        await observer.close()


@pytest.mark.asyncio
async def test_capacity_backlog_does_not_hide_actionable_errors_or_skip_cursors(monitor):
    observer, store = monitor
    for _ in range(125):
        store.append(REF, "warning", {"reason": "FailedScheduling", "message": "Insufficient cpu"})
    for event_type, reason in (("warning", "FailedMount"), ("warning", "ErrImagePull"),
                               ("eviction_risk", "Low utilization"), ("failed", "Error")):
        store.append(REF, event_type, {"reason": reason})
    try:
        result = await observer.wait([REF], cursor=0, timeout=0, scheduling_policy="wait_for_capacity")
        assert [e["type"] for e in result["events"]] == ["warning", "warning", "eviction_risk", "failed"]
        assert result["cursor"] == 129
    finally:
        await observer.close()


@pytest.mark.asyncio
async def test_concurrent_policies_wait_until_success_or_report_blocker_independently(monitor):
    observer, store = monitor
    queued = asyncio.create_task(observer.wait([REF], timeout=2, scheduling_policy="wait_for_capacity"))
    immediate = asyncio.create_task(observer.wait([REF], timeout=2))
    try:
        for _ in range(100):
            if len(observer._active_waiters) == 2:
                break
            await asyncio.sleep(.001)
        assert len(observer._active_waiters) == 2
        await observer._handle_kubernetes_event("ADDED", {
            "metadata": {"uid": "warning-uid"}, "type": "Warning", "reason": "FailedScheduling",
            "message": "Insufficient nvidia.com/gpu", "involvedObject": {"kind": "Job", **REF},
        })
        reported = await asyncio.wait_for(immediate, 1)
        assert reported["events"][0]["type"] == "warning"
        await asyncio.sleep(.01)
        assert not queued.done()
        # Scheduled/running is silent. Success still wakes the queued waiter.
        await observer._handle_job("MODIFIED", {
            "metadata": REF, "status": {"conditions": [{"type": "Complete", "status": "True"}]},
        })
        finished = await asyncio.wait_for(queued, 1)
        assert [event["type"] for event in finished["events"]] == ["succeeded"]
        assert finished["cursor"] > reported["cursor"]
        assert len(store.read([REF["uid"]], event_types=["warning"])) == 1
    finally:
        queued.cancel()
        immediate.cancel()
        await asyncio.gather(queued, immediate, return_exceptions=True)
        await observer.close()
