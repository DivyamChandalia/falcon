"""Concurrency and outage regressions for the model-free watch loop."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import anyio
import pytest

from falcon.mcp_events import EventStore, Monitor
from falcon.mcp_risk import McpGpuRiskEvaluator


def job(*, completed=False, gpu=False):
    return {
        "metadata": {"name": "train", "namespace": "mine", "uid": "job-uid"},
        "spec": {"template": {"spec": {"containers": [{
            "name": "main", "resources": {"requests": {"nvidia.com/gpu": "1" if gpu else "0"}},
        }]}}},
        "status": {"conditions": [{"type": "Complete", "status": "True"}]} if completed else {"active": 1},
    }


class Streams:
    def __init__(self):
        self.queues = {name: asyncio.Queue() for name in ("jobs.batch", "pods", "events")}
        self.calls = asyncio.Queue()

    def __call__(self, resource, namespace, resource_version=None):
        self.calls.put_nowait((resource, resource_version))

        async def stream():
            while True:
                yield await self.queues[resource].get()

        return stream()

    async def started(self, resource="jobs.batch"):
        while True:
            name, version = await asyncio.wait_for(self.calls.get(), 2)
            if name == resource:
                return version


async def _until(predicate):
    async def check():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(check(), 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("completed_before_recall", [False, True])
async def test_replacement_wait_recovers_exited_retained_watchers(
    tmp_path, monkeypatch, partial, completed_before_recall,
):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    store = EventStore(tmp_path / "events.db")
    streams = Streams()
    old = job(completed=True)
    current = [old]
    lists = []

    async def inventory(resource, namespace):
        lists.append(resource)
        return {"metadata": {"resourceVersion": "1"},
                "items": current if resource == "jobs.batch" else []}

    monitor = Monitor(SimpleNamespace(executable="kubectl"), "mine", store,
                      watcher=streams, list_provider=inventory)
    waiting = None
    try:
        initial = await monitor.wait([old], cursor=0, timeout=3)
        assert initial["events"][0]["type"] == "succeeded"
        for _ in range(3):
            await asyncio.wait_for(streams.calls.get(), 2)
        await streams.queues["jobs.batch"].put({"type": "DELETED", "object": old})
        await _until(lambda: not monitor._has_observation_targets())
        resources = ["jobs.batch"] if partial else list(streams.queues)
        for resource in resources:
            await streams.queues[resource].put({"type": "BOOKMARK", "object": {"metadata": {"resourceVersion": "2"}}})
        await _until(lambda: sum(task.done() for task in monitor._watch_tasks) == len(resources))
        assert monitor._started, "reproduce the stale started flag"
        replacement = job(completed=completed_before_recall)
        replacement["metadata"]["uid"] = "replacement-uid"
        current[:] = [replacement]
        ref = {"name": "train", "namespace": "mine", "uid": "replacement-uid"}
        waiting = asyncio.create_task(monitor.wait([ref], cursor=initial["cursor"], timeout=3))
        await _until(lambda: len(lists) == 6 and monitor._background_running())
        if not completed_before_recall:
            completed = job(completed=True)
            completed["metadata"]["uid"] = "replacement-uid"
            await streams.queues["jobs.batch"].put({"type": "MODIFIED", "object": completed})
        result = await asyncio.wait_for(waiting, 2)
        assert not result["timed_out"]
        assert result["events"][0]["type"] == "succeeded"
        assert result["events"][0]["job"]["uid"] == "replacement-uid"
        assert len(monitor._watch_tasks) == 3
    finally:
        if waiting is not None:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
        await monitor.close()
        store.close()


@pytest.mark.asyncio
async def test_replacement_registered_while_old_watch_is_closing(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    store = EventStore(tmp_path / "events.db")
    streams = Streams()
    closing = asyncio.Event()
    release = asyncio.Event()
    old = job(completed=True)
    current = [old]
    lists = []

    async def inventory(resource, namespace):
        lists.append(resource)
        return {"metadata": {"resourceVersion": "1"},
                "items": current if resource == "jobs.batch" else []}

    async def watcher(resource, namespace, resource_version):
        source = streams(resource, namespace, resource_version)
        try:
            async for event in source:
                yield event
        finally:
            await source.aclose()
            if resource == "jobs.batch" and not closing.is_set():
                closing.set()
                await release.wait()

    monitor = Monitor(SimpleNamespace(executable="kubectl"), "mine", store,
                      watcher=watcher, list_provider=inventory)
    waiting = None
    try:
        initial = await monitor.wait([old], cursor=0, timeout=3)
        for _ in range(3):
            await asyncio.wait_for(streams.calls.get(), 2)
        await streams.queues["jobs.batch"].put({"type": "DELETED", "object": old})
        await _until(lambda: not monitor._has_observation_targets())
        await streams.queues["jobs.batch"].put({"type": "BOOKMARK", "object": {"metadata": {"resourceVersion": "2"}}})
        await asyncio.wait_for(closing.wait(), 2)
        replacement = job()
        replacement["metadata"]["uid"] = "replacement-uid"
        current[:] = [replacement]
        ref = {"name": "train", "namespace": "mine", "uid": "replacement-uid"}
        waiting = asyncio.create_task(monitor.wait([ref], cursor=initial["cursor"], timeout=3))
        await _until(lambda: len(lists) == 6)
        release.set()
        await streams.started()
        completed = job(completed=True)
        completed["metadata"]["uid"] = "replacement-uid"
        await streams.queues["jobs.batch"].put({"type": "MODIFIED", "object": completed})
        result = await asyncio.wait_for(waiting, 2)
        assert result["events"][0]["type"] == "succeeded"
        assert result["events"][0]["job"]["uid"] == "replacement-uid"
    finally:
        release.set()
        if waiting is not None:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
        await monitor.close()
        store.close()


@pytest.mark.asyncio
async def test_expired_resource_version_relists_and_recovers_without_model_polling(tmp_path):
    store = EventStore(tmp_path / "events.db")
    streams = Streams()
    versions = {name: 0 for name in streams.queues}

    async def inventory(resource, namespace):
        versions[resource] += 1
        return {
            "metadata": {"resourceVersion": str(versions[resource])},
            "items": [job()] if resource == "jobs.batch" else [],
        }

    monitor = Monitor(
        SimpleNamespace(executable="kubectl"), "mine", store,
        watcher=streams, list_provider=inventory,
    )
    waiting = asyncio.create_task(monitor.wait([job()], event_types=["succeeded"], timeout=3))
    try:
        first_version = await streams.started()
        await streams.queues["jobs.batch"].put({"type": "ERROR", "object": {
            "code": 410, "message": "expired resourceVersion",
        }})
        second_version = await streams.started()
        assert int(second_version) > int(first_version)
        assert not waiting.done(), "a collector reconnect is not a model status-poll response"
        await streams.queues["jobs.batch"].put({"type": "MODIFIED", "object": job(completed=True)})
        result = await asyncio.wait_for(waiting, 2)
        assert result["events"][0]["type"] == "succeeded"
        assert any(event["details"].get("gap") for event in store.read(["job-uid"]))
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await monitor.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_first", [False, True])
async def test_waiters_share_watches_but_not_event_consumption(tmp_path, cancel_first):
    store = EventStore(tmp_path / "events.db")
    streams = Streams()
    monitor = Monitor(SimpleNamespace(executable="kubectl"), "mine", store, watcher=streams)
    first = asyncio.create_task(monitor.wait([job()], event_types=["succeeded"], timeout=3))
    second = asyncio.create_task(monitor.wait([job()], event_types=["succeeded"], timeout=3))
    try:
        await streams.started()
        for _ in range(100):
            if len(monitor._active_waiters) == 2:
                break
            await asyncio.sleep(0)
        assert len(monitor._active_waiters) == 2
        assert len(monitor._watch_tasks) == 3
        if cancel_first:
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert len(monitor._watch_tasks) == 3
        await streams.queues["jobs.batch"].put({"type": "MODIFIED", "object": job(completed=True)})
        result = await asyncio.wait_for(second, 2)
        assert result["events"][0]["type"] == "succeeded"
        if not cancel_first:
            other = await asyncio.wait_for(first, 2)
            assert result["events"] == other["events"]
    finally:
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        await monitor.close()
        assert not monitor._watch_tasks
        store.close()


@pytest.mark.asyncio
async def test_background_wait_delivers_vram_risk_after_irregular_samples(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    store = EventStore(tmp_path / "events.db")
    streams = Streams()
    evaluator = McpGpuRiskEvaluator()
    sample_index = 0

    async def metrics(names):
        nonlocal sample_index
        policy = evaluator.evaluate("job-uid", [{
            "index": 0, "uuid": "gpu-0", "utilization": 99,
            "memory_used_gib": 1, "memory_total_gib": 96,
        }], 75, now=sample_index * 1.37)
        sample_index += 1
        return {"job-uid": {
            "available": True, "at_risk": policy["at_risk"],
            "metrics": {"eviction_policy": {"per_gpu": policy["devices"]}},
        }}

    monitor = Monitor(
        SimpleNamespace(executable="kubectl"), "mine", store,
        watcher=streams, metrics_provider=metrics,
    )
    monitor._metrics_interval = 0.001
    waiting = asyncio.create_task(monitor.wait([job(gpu=True)], timeout=3))
    try:
        await streams.started()
        result = await asyncio.wait_for(waiting, 2)
        assert not result["timed_out"]
        assert [event["type"] for event in result["events"]] == ["eviction_risk"]
        device = result["events"][0]["details"]["risk"]["per_gpu"][0]
        assert device["vram_at_risk"]
        assert not device["gpu_at_risk"]
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await monitor.close()
        store.close()


@pytest.mark.asyncio
async def test_missing_metrics_do_not_falsely_clear_an_existing_risk(tmp_path):
    store = EventStore(tmp_path / "events.db")
    streams = Streams()
    samples = asyncio.Queue()

    async def metrics(names):
        return {"train": await samples.get()}

    monitor = Monitor(
        SimpleNamespace(executable="kubectl"), "mine", store,
        watcher=streams, metrics_provider=metrics,
    )
    monitor._metrics_interval = 0.01
    first_wait = asyncio.create_task(
        monitor.wait([job(gpu=True)], event_types=["eviction_risk"], timeout=3)
    )
    try:
        await streams.started()
        await samples.put({"available": True, "at_risk": True})
        first = await asyncio.wait_for(first_wait, 2)
        assert first["events"][0]["type"] == "eviction_risk"

        waiting = asyncio.create_task(
            monitor.wait(
                [job(gpu=True)],
                cursor=first["cursor"],
                event_types=["eviction_risk"],
                timeout=0.3,
            )
        )
        for _ in range(3):
            await samples.put({"available": False, "at_risk": None})
        for _ in range(100):
            if any(event["type"] == "monitor_unavailable" for event in store.read(["job-uid"])):
                break
            await asyncio.sleep(0.01)
        assert not waiting.done(), "unknown telemetry must not be converted to healthy utilization"
        assert not store.read(["job-uid"], event_types=["eviction_risk_cleared"])
        await samples.put({"available": True, "at_risk": False})
        result = await asyncio.wait_for(waiting, 2)
        assert result["events"] == []
        assert result["timed_out"] is True
        assert not store.read(["job-uid"], event_types=["eviction_risk_cleared"])
        assert len(store.read(["job-uid"], event_types=["eviction_risk"])) == 1
    finally:
        first_wait.cancel()
        await asyncio.gather(first_wait, return_exceptions=True)
        if "waiting" in locals():
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
        await monitor.close()
        store.close()


@pytest.mark.asyncio
async def test_mcp_level_cancellation_finishes_wait_cleanup(tmp_path):
    """The SDK cancels an AnyIO scope, which can cancel cleanup awaits too."""
    store = EventStore(tmp_path / "events.db")
    streams = Streams()
    monitor = Monitor(SimpleNamespace(executable="kubectl"), "mine", store, watcher=streams)
    scopes = []
    done = asyncio.Event()

    async def observe():
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await monitor.wait([job()], event_types=["succeeded"], timeout=30)
        done.set()

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(observe)
            await streams.started()
            scopes[0].cancel()
            await asyncio.wait_for(done.wait(), 2)
        assert not monitor._active_waiters
        assert len(monitor._watch_tasks) == 3
        assert all(not task.done() for task in monitor._watch_tasks)
    finally:
        await monitor.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["succeeded", "warning", "eviction_risk"])
async def test_cancelled_wait_replays_event_recorded_before_recall(tmp_path, event_type):
    store = EventStore(tmp_path / "events.db")
    streams = Streams()
    monitor = Monitor(SimpleNamespace(executable="kubectl"), "mine", store, watcher=streams)
    waiting = asyncio.create_task(monitor.wait([job()], cursor=0, timeout=30))
    try:
        await streams.started()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert not monitor._active_waiters
        assert len(monitor._watch_tasks) == 3
        # The parent is answering a user: no MCP request is waiting. The
        # retained watcher must still record even transient, nonterminal events.
        key = next(iter(monitor._tracked))
        await monitor._emit(event_type, {"arrived_during_reply": True}, key=key)
        recorded = store.read(["job-uid"], event_types=[event_type])
        assert len(recorded) == 1
        result = await asyncio.wait_for(monitor.wait([job()], cursor=0, timeout=30), 2)
        assert result["events"][0]["cursor"] == recorded[0]["cursor"]
        assert result["events"][0]["type"] == event_type
        assert result["cursor"] == recorded[0]["cursor"]
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await monitor.close()
        assert not monitor._watch_tasks
        store.close()
