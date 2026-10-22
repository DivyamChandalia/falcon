from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import falcon.mcp_events as mcp_events
from falcon.mcp_events import (
    ACTIONABLE_EVENT_TYPES,
    MAX_EVENT_DETAILS_BYTES,
    MAX_WAIT_RESPONSE_BYTES,
    MONITOR_HEALTH_EVENT_TYPES,
    MONITOR_HEALTH_FAILURE_THRESHOLD,
    SUPPORTED_EVENT_TYPES,
    EventStore,
    Monitor,
)


class FakeWatcher:
    def __init__(self) -> None:
        self.queues: dict[str, asyncio.Queue[object]] = {
            resource: asyncio.Queue()
            for resource in ("jobs.batch", "pods", "events")
        }
        self.calls: list[tuple[str, str, str | None]] = []
        self.started = asyncio.Event()

    def __call__(
        self,
        resource: str,
        namespace: str,
        resource_version: str | None = None,
    ):
        self.calls.append((resource, namespace, resource_version))
        self.started.set()

        async def stream():
            while True:
                value = await self.queues[resource].get()
                if value is None:
                    return
                yield value

        return stream()

    async def put(self, resource: str, value: object) -> None:
        await self.queues[resource].put(value)


def _client() -> SimpleNamespace:
    return SimpleNamespace(executable="kubectl")


def _job(
    uid: str = "job-uid",
    *,
    name: str = "job",
    status: dict | None = None,
    gpu: bool = False,
) -> dict:
    container: dict = {"name": "main"}
    if gpu:
        container["resources"] = {"requests": {"nvidia.com/gpu": "1"}}
    return {
        "metadata": {"name": name, "namespace": "research", "uid": uid},
        "spec": {
            "template": {"spec": {"containers": [container]}},
        },
        "status": status or {},
    }


def _success_job(uid: str = "job-uid") -> dict:
    return _job(
        uid,
        status={
            "conditions": [
                {
                    "type": "Complete",
                    "status": "True",
                    "reason": "Completed",
                }
            ],
            "succeeded": 1,
        },
    )


@pytest.fixture
def store(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    value = EventStore(tmp_path / "private" / "events.sqlite3")
    try:
        yield value
    finally:
        value.close()


@pytest.mark.asyncio
async def test_presence_only_lasts_for_active_wait_not_retained_watch(store: EventStore):
    from falcon.agent_presence import connected_jobs

    watcher = FakeWatcher()
    monitor = Monitor(_client(), "research", store, watcher=watcher)
    waiting = asyncio.create_task(monitor.wait([_job()], timeout=10))
    try:
        await watcher.started.wait()
        await asyncio.sleep(0)
        assert connected_jobs(None, "research") == {"job-uid"}
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert connected_jobs(None, "research") == set()
        result = await monitor.wait([_job()], timeout=0.01)
        assert result["timed_out"]
        assert monitor._started
        assert connected_jobs(None, "research") == set()
        waiting = asyncio.create_task(monitor.wait([_job()], timeout=10))

        async def reconnected():
            while not connected_jobs(None, "research"):
                await asyncio.sleep(0)

        await asyncio.wait_for(reconnected(), 2)
        assert connected_jobs(None, "research") == {"job-uid"}
        assert not waiting.done()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert connected_jobs(None, "research") == set()
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await monitor.close()


def test_event_store_is_private_and_cursor_stable(tmp_path: Path) -> None:
    caller_root = tmp_path / "caller-root"
    caller_root.mkdir()
    caller_root.chmod(0o755)
    database = caller_root / "private" / "events.sqlite3"
    store = EventStore(database)
    first = store.append(
        {"metadata": {"name": "job", "namespace": "research", "uid": "u1"}},
        "warning",
        {"reason": "FailedScheduling", "token": "must not persist"},
    )
    second = store.append(
        {"name": "job", "namespace": "research", "uid": "u1"},
        "succeeded",
        {"ok": True},
    )

    assert first["cursor"] == 1
    assert second["cursor"] == 2
    assert store.read(["u1"], cursor=0, event_types=["warning"])[0]["details"] == {
        "reason": "FailedScheduling",
        "token": "[redacted]",
    }
    assert stat.S_IMODE(database.stat().st_mode) == 0o600
    assert stat.S_IMODE(caller_root.stat().st_mode) == 0o755
    assert stat.S_IMODE(database.parent.stat().st_mode) == 0o700
    store.close()

    reopened = EventStore(database)
    assert reopened.latest_cursor() == 2
    assert [event["type"] for event in reopened.read(["u1"])] == [
        "warning",
        "succeeded",
    ]
    reopened.close()


def test_event_store_evicts_old_records_without_reusing_cursors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_events, "MAX_EVENT_RECORDS", 3)
    store = EventStore(tmp_path / "events.sqlite3")
    for index in range(5):
        store.append(
            {"name": "job", "namespace": "research", "uid": "u1"},
            "warning",
            {"index": index},
        )

    assert store.latest_cursor() == 5
    events = store.read(["u1"], cursor=0, limit=3)
    assert [event["cursor"] for event in events] == [3, 4, 5]
    store.close()


def test_old_job_warning_cannot_match_same_name_replacement(store: EventStore):
    monitor = Monitor(_client(), "research", store, watcher=FakeWatcher())
    key = monitor._track(_job(uid="new-uid", name="same-name"), 1)
    involved = {"kind": "Job", "name": "same-name", "namespace": "research"}
    assert monitor._event_job_key({"involvedObject": {**involved, "uid": "old-uid"}}) is None
    assert monitor._event_job_key({"involvedObject": {**involved, "uid": "new-uid"}}) == key
    assert monitor._event_job_key({"involvedObject": involved}) == key


def test_event_details_are_bounded_with_explicit_message_truncation(
    store: EventStore,
) -> None:
    event = store.append(
        {"name": "job", "namespace": "research", "uid": "u1"},
        "warning",
        {
            "reason": "FailedScheduling",
            "message": "x" * 100_000,
            "involvedObject": {"kind": "Job", "name": "job"},
        },
    )

    encoded = json.dumps(
        event["details"], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    assert len(encoded) <= MAX_EVENT_DETAILS_BYTES
    assert event["details"]["details_truncated"] is True
    assert event["details"]["message"].endswith("...[truncated]")


@pytest.mark.asyncio
async def test_missing_job_baseline_silently_disconnects_old_uid(
    store: EventStore,
) -> None:
    watcher = FakeWatcher()
    old_job = _job(uid="old-uid", name="same-name")
    replacement = _job(uid="new-uid", name="same-name")

    async def list_provider(resource: str, namespace: str | None = None) -> dict:
        if resource == "jobs.batch":
            return {"metadata": {"resourceVersion": "1"}, "items": [replacement]}
        return {"metadata": {"resourceVersion": "1"}, "items": []}

    monitor = Monitor(
        _client(),
        "research",
        store,
        watcher=watcher,
        list_provider=list_provider,
    )
    result = await monitor.wait([old_job], timeout=0.2)

    assert result["events"] == []
    assert result["disconnected"] is True
    assert not store.read(["old-uid"])
    await monitor.close()


@pytest.mark.asyncio
async def test_missing_job_after_410_relist_silently_disconnects(store: EventStore) -> None:
    watcher = FakeWatcher()
    job = _job(uid="lost-uid")
    job_list_calls = 0

    async def list_provider(resource: str, namespace: str | None = None) -> dict:
        nonlocal job_list_calls
        if resource == "jobs.batch":
            job_list_calls += 1
            items = [job] if job_list_calls == 1 else []
            return {
                "metadata": {"resourceVersion": str(job_list_calls)},
                "items": items,
            }
        return {"metadata": {"resourceVersion": "1"}, "items": []}

    monitor = Monitor(
        _client(),
        "research",
        store,
        watcher=watcher,
        list_provider=list_provider,
    )
    waiting = asyncio.create_task(
        monitor.wait([job], event_types=["succeeded"], timeout=2.0)
    )
    await watcher.started.wait()
    await watcher.put(
        "jobs.batch",
        {
            "type": "ERROR",
            "object": {"code": 410, "message": "resource version expired"},
        },
    )

    result = await waiting
    assert result["events"] == []
    assert result["disconnected"] is True
    assert not store.read(["lost-uid"], event_types=["deleted"])
    assert store.read(["lost-uid"], event_types=["monitor_unavailable"])
    assert job_list_calls >= 2
    await monitor.close()


@pytest.mark.asyncio
async def test_wait_response_byte_budget_paginates_without_losing_events(
    store: EventStore,
) -> None:
    watcher = FakeWatcher()
    job = _job()
    total = 40
    for index in range(total):
        store.append(
            job,
            "warning",
            {"index": index, "message": "m" * 3_000},
        )

    monitor = Monitor(_client(), "research", store, watcher=watcher)
    cursor = 0
    seen: list[int] = []
    while len(seen) < total:
        result = await monitor.wait(
            [job],
            cursor=cursor,
            event_types=["warning"],
            timeout=0,
        )
        assert result["events"]
        encoded = json.dumps(
            result, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        assert len(encoded) <= MAX_WAIT_RESPONSE_BYTES
        assert result["cursor"] == result["events"][-1]["cursor"]
        seen.extend(event["details"]["index"] for event in result["events"])
        cursor = result["cursor"]

    assert seen == list(range(total))
    await monitor.close()


@pytest.mark.asyncio
async def test_terminal_job_is_replayed_for_each_fresh_wait(store: EventStore) -> None:
    watcher = FakeWatcher()
    monitor = Monitor(_client(), "research", store, watcher=watcher)
    job = _success_job()

    first = await monitor.wait([job], timeout=0.2)
    second = await monitor.wait([job], timeout=0.2)

    assert first["timed_out"] is False
    assert second["timed_out"] is False
    assert [event["type"] for event in first["events"]] == ["succeeded"]
    assert [event["type"] for event in second["events"]] == ["succeeded"]
    assert first["events"][0]["job"]["uid"] == "job-uid"
    await monitor.close()


@pytest.mark.asyncio
async def test_empty_selection_does_not_start_watches_or_wait(store: EventStore) -> None:
    watcher = FakeWatcher()
    monitor = Monitor(_client(), "research", store, watcher=watcher)

    result = await monitor.wait([], timeout=7 * 24 * 60 * 60)

    assert result == {"events": [], "cursor": 0, "timed_out": True}
    assert watcher.calls == []
    assert store.latest_cursor() == 0
    await monitor.close()


@pytest.mark.asyncio
async def test_state_edges_are_not_reported_and_invalid_types_fail_fast(
    store: EventStore,
) -> None:
    watcher = FakeWatcher()
    monitor = Monitor(_client(), "research", store, watcher=watcher)

    assert "running" not in SUPPORTED_EVENT_TYPES
    assert "pending" not in SUPPORTED_EVENT_TYPES
    assert set(MONITOR_HEALTH_EVENT_TYPES).isdisjoint(SUPPORTED_EVENT_TYPES)
    assert not set(MONITOR_HEALTH_EVENT_TYPES).intersection(ACTIONABLE_EVENT_TYPES)
    for state_type in ("running", "pending", *MONITOR_HEALTH_EVENT_TYPES):
        with pytest.raises(ValueError, match="unsupported event type"):
            await monitor.wait([_job(status={"active": 1})], event_types=[state_type], timeout=0)

    with pytest.raises(ValueError, match="unsupported event type"):
        await monitor.wait([_job()], event_types=["not-supported"], timeout=0)
    await monitor.close()


@pytest.mark.asyncio
async def test_transient_monitor_health_is_silent_by_default(
    store: EventStore,
) -> None:
    watcher = FakeWatcher()
    monitor = Monitor(_client(), "research", store, watcher=watcher)
    job = _job()
    waiting = asyncio.create_task(monitor.wait([job], timeout=0.2))
    await watcher.started.wait()

    for _ in range(MONITOR_HEALTH_FAILURE_THRESHOLD):
        await monitor._mark_unavailable(
            "jobs.batch", {"operation": "list", "error": "timeout"}
        )

    # Health is recorded, but the default event filter only wakes the agent
    # for events that require a workload decision.
    assert not waiting.done()
    result = await waiting
    assert result["timed_out"] is True
    assert store.read(["job-uid"], event_types=["monitor_unavailable"])

    diagnostics = store.read(["job-uid"], event_types=["monitor_unavailable"])
    assert diagnostics[0]["details"]["consecutive_failures"] == (
        MONITOR_HEALTH_FAILURE_THRESHOLD
    )
    await monitor.close()


@pytest.mark.asyncio
async def test_outage_and_recovery_keep_wait_open_until_job_succeeds(store: EventStore) -> None:
    watcher = FakeWatcher()
    monitor = Monitor(_client(), "research", store, watcher=watcher)
    waiting = asyncio.create_task(
        monitor.wait([_job()], event_types=list(SUPPORTED_EVENT_TYPES), timeout=10)
    )
    try:
        await watcher.started.wait()
        for _ in range(MONITOR_HEALTH_FAILURE_THRESHOLD + 10):
            await monitor._mark_unavailable(
                "jobs.batch", {"operation": "watch", "error": "offline"}
            )
        await asyncio.sleep(0)
        assert not waiting.done()
        await monitor._mark_recovered("jobs.batch")
        await asyncio.sleep(0)
        assert not waiting.done()
        assert {event["type"] for event in store.read(["job-uid"])} == set(
            MONITOR_HEALTH_EVENT_TYPES
        )
        await watcher.put("jobs.batch", {"type": "MODIFIED", "object": _success_job()})
        result = await asyncio.wait_for(waiting, 2)
        assert [event["type"] for event in result["events"]] == ["succeeded"]
        assert not result["timed_out"]
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await monitor.close()


@pytest.mark.asyncio
async def test_rearming_existing_monitor_does_not_repeat_namespace_baseline(
    store: EventStore,
) -> None:
    watcher = FakeWatcher()
    calls = {"jobs.batch": 0, "pods": 0, "events": 0}
    job = _job()

    async def list_provider(resource: str, namespace: str | None = None) -> dict:
        del namespace
        calls[resource] += 1
        items = [job] if resource == "jobs.batch" else []
        return {"metadata": {"resourceVersion": str(calls[resource])}, "items": items}

    monitor = Monitor(
        _client(), "research", store, watcher=watcher, list_provider=list_provider
    )
    first = await monitor.wait([job], event_types=["warning"], timeout=0)
    assert first["timed_out"] is True
    assert calls == {"jobs.batch": 1, "pods": 1, "events": 1}

    second = await monitor.wait(
        [job], cursor=first["cursor"], event_types=["warning"], timeout=0
    )
    assert second["timed_out"] is True
    assert calls == {"jobs.batch": 1, "pods": 1, "events": 1}
    await monitor.close()


@pytest.mark.asyncio
async def test_normal_rearm_keeps_observation_and_risk_state(
    store: EventStore,
) -> None:
    watcher = FakeWatcher()
    risk = True

    async def metrics(names: list[str]) -> dict:
        return {
            name: {
                "available": True,
                "at_risk": risk,
                "metrics": {
                    "eviction_policy": {
                        "phase": "steady",
                        "window_seconds": 300,
                        "per_gpu": [{"index": 1, "gpu_at_risk": True}],
                    },
                },
            }
            for name in names
        }

    monitor = Monitor(
        _client(),
        "research",
        store,
        metrics_provider=metrics,
        watcher=watcher,
    )
    monitor._metrics_interval = 0.01
    job = _job(gpu=True)
    job_reference = {
        "name": job["metadata"]["name"],
        "namespace": job["metadata"]["namespace"],
        "uid": job["metadata"]["uid"],
    }

    first = await monitor.wait([job], event_types=["eviction_risk"], timeout=1.0)
    assert [event["type"] for event in first["events"]] == ["eviction_risk"]
    assert first["events"][0]["details"]["risk"]["per_gpu"][0]["index"] == 1

    second_wait = asyncio.create_task(
        monitor.wait(
            [job_reference],
            cursor=first["cursor"],
            event_types=["eviction_risk"],
            timeout=0.2,
        )
    )
    await asyncio.sleep(0.05)
    assert not second_wait.done()

    risk = False
    second = await second_wait
    assert second["events"] == []
    assert second["timed_out"] is True
    assert not store.read(
        ["job-uid"],
        cursor=first["cursor"],
        event_types=["eviction_risk_cleared"],
    )
    assert store.read(
        ["job-uid"],
        cursor=first["cursor"],
        event_types=["monitor_unavailable"],
    ) == []

    risk = True
    third = await monitor.wait(
        [job_reference],
        cursor=second["cursor"],
        event_types=["eviction_risk"],
        timeout=1.0,
    )
    assert [event["type"] for event in third["events"]] == ["eviction_risk"]
    await monitor.close()


@pytest.mark.asyncio
async def test_deleted_job_silently_disconnects_wait(store: EventStore) -> None:
    watcher = FakeWatcher()
    monitor = Monitor(_client(), "research", store, watcher=watcher)
    job = _job()
    waiting = asyncio.create_task(monitor.wait([job], timeout=5.0))

    await watcher.started.wait()
    await watcher.put("jobs.batch", {"type": "DELETED", "object": job})
    result = await asyncio.wait_for(waiting, 2.0)

    assert result["events"] == []
    assert result["disconnected"] is True
    assert not store.read(["job-uid"])
    await monitor.close()


@pytest.mark.asyncio
async def test_normal_wait_timeout_does_not_restart_watches_or_report_gap(store: EventStore) -> None:
    watcher = FakeWatcher()
    monitor = Monitor(_client(), "research", store, watcher=watcher)
    try:
        first = await monitor.wait([_job()], timeout=.01)
        assert first["timed_out"] is True
        assert len(watcher.calls) == 3
        second = await monitor.wait(
            [_job()], cursor=first["cursor"], timeout=.01,
            event_types=list(SUPPORTED_EVENT_TYPES),
        )
        assert second["events"] == []
        assert second["timed_out"] is True
        assert len(watcher.calls) == 3
        assert monitor._started
        assert not store.read(["job-uid"])
    finally:
        await monitor.close()


@pytest.mark.asyncio
async def test_cancelled_last_waiter_records_watch_gap_internally(
    store: EventStore,
) -> None:
    watcher = FakeWatcher()
    monitor = Monitor(_client(), "research", store, watcher=watcher)
    job = _job()

    waiting = asyncio.create_task(
        monitor.wait([job], event_types=["warning"], timeout=10.0)
    )
    await watcher.started.wait()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    second = await monitor.wait(
        [job],
        cursor=store.latest_cursor(),
        event_types=list(SUPPORTED_EVENT_TYPES),
        timeout=0.2,
    )

    assert second["events"] == []
    assert second["timed_out"] is True
    diagnostics = store.read(["job-uid"], event_types=["monitor_unavailable"])
    assert diagnostics[0]["details"] == {
        "cursor": 0,
        "gap": True,
        "last_observed_cursor": 0,
        "reason": "watch_restarted",
        "source": "monitor",
    }
    await monitor.close()


@pytest.mark.asyncio
async def test_wait_wakes_on_job_watch_and_times_out_without_empty_poll_returns(
    store: EventStore,
) -> None:
    watcher = FakeWatcher()
    monitor = Monitor(_client(), "research", store, watcher=watcher)
    job = _job()

    waiting = asyncio.create_task(monitor.wait([job], timeout=1.0))
    await watcher.started.wait()
    await watcher.put(
        "jobs.batch",
        {
            "type": "MODIFIED",
            "object": {
                **job,
                "metadata": {**job["metadata"], "resourceVersion": "2"},
                "status": {"conditions": [{"type": "Complete", "status": "True"}]},
            },
        },
    )
    result = await waiting
    assert result["timed_out"] is False
    assert result["events"][0]["type"] == "succeeded"

    timed_out = await monitor.wait([_job("other")], timeout=0.02)
    assert timed_out["events"] == []
    assert timed_out["timed_out"] is True
    await monitor.close()


@pytest.mark.asyncio
async def test_wait_cursor_pages_a_matching_backlog_without_skipping(store: EventStore) -> None:
    watcher = FakeWatcher()
    monitor = Monitor(_client(), "research", store, watcher=watcher)
    job = _job()
    for index in range(51):
        store.append(
            job,
            "warning",
            {"reason": "FailedScheduling", "index": index},
        )

    first = await monitor.wait(
        [job], cursor=0, event_types=["warning"], timeout=0.2
    )
    second = await monitor.wait(
        [job], cursor=first["cursor"], event_types=["warning"], timeout=0.2
    )
    assert len(first["events"]) == 50
    assert first["cursor"] == 50
    assert [event["details"]["index"] for event in second["events"]] == [50]
    assert second["cursor"] == 51
    await monitor.close()


@pytest.mark.asyncio
async def test_pod_owner_filter_and_warning_event_diagnostics(store: EventStore) -> None:
    watcher = FakeWatcher()
    monitor = Monitor(_client(), "research", store, watcher=watcher)
    job = _job()
    waiting = asyncio.create_task(monitor.wait([job], timeout=1.0))
    await watcher.started.wait()

    await watcher.put(
        "pods",
        {
            "type": "MODIFIED",
            "object": {
                "metadata": {
                    "name": "attempt",
                    "uid": "pod-uid",
                    "ownerReferences": [{"kind": "Job", "uid": "job-uid"}],
                },
                "status": {"phase": "Failed", "reason": "Evicted"},
            },
        },
    )
    result = await waiting
    assert result["events"][0]["type"] == "evicted"

    # A Pod from a different Job must not leak into this context's selected
    # event stream, while a warning for the selected Job should be retained.
    waiting = asyncio.create_task(
        monitor.wait(
            [job],
            cursor=result["cursor"],
            event_types=["warning"],
            timeout=1.0,
        )
    )
    await watcher.put(
        "pods",
        {
            "type": "MODIFIED",
            "object": {
                "metadata": {
                    "name": "foreign-attempt",
                    "uid": "foreign-pod",
                    "ownerReferences": [{"kind": "Job", "uid": "foreign-job"}],
                },
                "status": {"phase": "Failed", "reason": "Error"},
            },
        },
    )
    await watcher.put(
        "events",
        {
            "type": "ADDED",
            "object": {
                "metadata": {"uid": "event-uid"},
                "type": "Warning",
                "reason": "FailedScheduling",
                "message": "Insufficient nvidia.com/gpu",
                "involvedObject": {"kind": "Job", "uid": "job-uid", "name": "job"},
            },
        },
    )
    result = await waiting
    assert [event["type"] for event in result["events"]] == ["warning"]
    assert "Insufficient" in result["events"][0]["details"]["message"]
    await monitor.close()


def test_command_argv_is_used_for_pinned_context(store: EventStore) -> None:
    class Client:
        def command_argv(self, args):
            return ["kubectl", "--context", "team-context", *args]

    monitor = Monitor(Client(), "research", store, watcher=lambda *_args: [])
    assert monitor._client_argv(["get", "pods"]) == [
        "kubectl",
        "--context",
        "team-context",
        "get",
        "pods",
    ]

