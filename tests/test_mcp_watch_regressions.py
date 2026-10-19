"""Concurrency and outage regressions for the model-free watch loop."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import anyio
import pytest

from falcon.mcp_events import EventStore, Monitor


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
    waiting = asyncio.create_task(monitor.wait(
        [job(gpu=True)], event_types=["eviction_risk_cleared"], timeout=3,
    ))
    try:
        await streams.started()
        await samples.put({"available": True, "at_risk": True})
        await samples.put({"available": False, "at_risk": None})
        for _ in range(100):
            if any(event["type"] == "monitor_unavailable" for event in store.read(["job-uid"])):
                break
            await asyncio.sleep(0.01)
        assert not waiting.done(), "unknown telemetry must not be converted to healthy utilization"
        assert not store.read(["job-uid"], event_types=["eviction_risk_cleared"])
        await samples.put({"available": True, "at_risk": False})
        result = await asyncio.wait_for(waiting, 2)
        assert result["events"][0]["type"] == "eviction_risk_cleared"
        assert len(store.read(["job-uid"], event_types=["eviction_risk"])) == 1
    finally:
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
        assert not monitor._watch_tasks
        assert monitor._metrics_task is None
    finally:
        await monitor.close()
        store.close()
