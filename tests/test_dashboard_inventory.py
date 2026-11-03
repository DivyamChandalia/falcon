from __future__ import annotations

import asyncio
import copy
import queue
import threading
import time
from collections import Counter

import pytest

from falcon.dashboard import GpuSample, UsageCollector
from falcon.dashboard_inventory import RESOURCES, DashboardInventory


def job(uid="job-1"):
    return {
        "kind": "Job",
        "metadata": {"uid": uid, "name": "train"},
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "resources": {"requests": {"cpu": "1", "memory": "1Gi"}},
                        }
                    ]
                }
            }
        },
        "status": {"active": 1},
    }


def pod(owner="job-1"):
    return {
        "kind": "Pod",
        "metadata": {
            "uid": "pod-1",
            "name": "train-abc",
            "labels": {"job-name": "train"},
            "ownerReferences": [{"kind": "Job", "name": "train", "uid": owner}],
        },
        "spec": {"containers": [{"name": "main", "resources": {"limits": {"nvidia.com/gpu": 1}}}]},
        "status": {"phase": "Running"},
    }


def eventually(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("inventory update did not arrive")
        time.sleep(0.005)


class Client:
    namespace = "research"

    def __init__(self):
        self.lists = Counter()
        self.items = {"jobs.batch": [job()], "pods": [pod()], "events": []}

    def list_json(self, resource, *, namespace):
        assert namespace == self.namespace
        self.lists[resource] += 1
        return {
            "metadata": {"resourceVersion": str(self.lists[resource])},
            "items": copy.deepcopy(self.items[resource]),
        }


class Transport:
    def __init__(self):
        self.queues = {r: queue.Queue() for r in RESOURCES}
        self.calls = []
        self.closed = False

    async def watch_json(self, resource, version):
        self.calls.append((resource, version))
        while True:
            try:
                raw = self.queues[resource].get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue
            yield raw

    async def close(self):
        self.closed = True


@pytest.fixture
def inventory():
    client, transport = Client(), Transport()
    source = DashboardInventory(client, transport=transport)
    eventually(lambda: len(transport.calls) == 3)
    try:
        yield source, client, transport
    finally:
        source.close()
    assert not source._thread.is_alive()
    assert transport.closed


def test_inventory_changes_reuse_three_streams_and_do_not_relist(inventory):
    source, client, transport = inventory
    revisions = []
    unsubscribe = source.subscribe(lambda: revisions.append(source.snapshot().revision))
    original_revision = source.snapshot().revision
    for version in range(2, 50):
        changed = job()
        changed["metadata"]["resourceVersion"] = str(version)
        changed["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
        transport.queues["jobs.batch"].put({"type": "MODIFIED", "object": changed})
    eventually(lambda: source.snapshot().revision >= original_revision + 48)
    assert client.lists == Counter({r: 1 for r in RESOURCES})
    assert len(transport.calls) == 3
    assert revisions
    unsubscribe()


def test_replacement_does_not_adopt_old_attempt_and_events_are_local(inventory, monkeypatch):
    source, client, transport = inventory
    collector = UsageCollector(
        "research", {}, 0.1, inventory_source=source, collect_availability=False
    )
    monkeypatch.setattr(
        collector, "_kubectl", lambda *a, **k: pytest.fail("unexpected network request")
    )
    assert collector.collect_inventory()[0].pod_count == 1
    transport.queues["jobs.batch"].put({"type": "ADDED", "object": job("replacement")})
    eventually(lambda: source.snapshot().items[0]["metadata"]["uid"] == "replacement")
    rows = collector.collect_inventory()
    assert rows[0].uid == "replacement"
    assert rows[0].pod_count == 0
    event = {
        "metadata": {"uid": "event-1", "name": "event"},
        "type": "Warning",
        "reason": "FailedScheduling",
        "message": "No CPU",
        "count": 3,
        "involvedObject": {"name": "train", "uid": "replacement", "kind": "Job"},
    }
    transport.queues["events"].put({"type": "ADDED", "object": event})
    eventually(lambda: bool(collector.events(rows[0])))
    assert collector.events(rows[0])[0].count == 3
    assert client.lists["events"] == 1


@pytest.mark.parametrize("phase", ["Running", "Pending"])
def test_agent_running_presence_follows_wait_not_ownership(inventory, monkeypatch, tmp_path, phase):
    from falcon.agent_presence import WaitPresence
    from falcon.dashboard_ui import FalconDashboard, _job_status_display

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    source, _, transport = inventory
    collector = UsageCollector(
        "research", {}, 0.1, inventory_source=source, collect_availability=False
    )
    managed = job()
    managed["metadata"]["labels"] = {"falcon.dev/agent-managed": "true"}
    if phase == "Pending":
        pending_pod = pod()
        pending_pod["status"]["phase"] = "Pending"
        transport.queues["pods"].put({"type": "MODIFIED", "object": pending_pod})
        eventually(lambda: any(
            item.get("kind") == "Pod" and item.get("status", {}).get("phase") == "Pending"
            for item in source.snapshot().items
        ))
    revision = source.snapshot().revision
    transport.queues["jobs.batch"].put({"type": "MODIFIED", "object": managed})
    eventually(lambda: source.snapshot().revision > revision)
    assert collector.collect_inventory()[0].agent_state == "disconnected"
    lease = WaitPresence(None, "research", ["job-1"])
    lease.renew()
    try:
        assert collector.collect_inventory()[0].agent_state == "connected"
    finally:
        lease.close()
    row = collector.collect_inventory()[0]
    assert row.agent_state == "disconnected"
    assert row.status == phase
    assert _job_status_display(row, 0)[0] == "●"
    assert not FalconDashboard._agent_is_connected(row)

    reconnected = WaitPresence(None, "research", ["job-1"])
    reconnected.renew()
    try:
        row = collector.collect_inventory()[0]
        assert row.agent_state == "connected"
        assert FalconDashboard._agent_is_connected(row)
        assert _job_status_display(row, 0)[0] == "◴"
        assert _job_status_display(row, 1)[0] == "◷"
    finally:
        reconnected.close()


def test_inventory_only_projection_does_not_sample_risk_or_metrics(inventory):
    source, _, _ = inventory
    collector = UsageCollector(
        "research", {}, 0.1, inventory_source=source, collect_availability=False
    )
    collector._latest_live = {"train-abc": (0.25, 0.5)}
    collector._latest_samples = {"train-abc": GpuSample(utilization=50, gpu_count=1)}
    collector._build_rows(
        source.snapshot().items,
        collector._latest_live,
        collector._latest_samples,
        sample_metrics=True,
    )
    before_risk = copy.deepcopy(collector.risk_samples)
    before_ema = dict(collector.ema_samples)
    before_vram = copy.deepcopy(collector.vram_risk_samples)
    for _ in range(50):
        collector.collect_inventory()
    assert collector.risk_samples == before_risk
    assert collector.ema_samples == before_ema
    assert collector.vram_risk_samples == before_vram
    assert before_ema == {"train": 1}


def test_410_replaces_snapshot_and_explicit_resync_restarts_once(inventory):
    source, client, transport = inventory
    client.items["pods"] = []
    transport.queues["pods"].put({"type": "ERROR", "object": {"code": 410}})
    eventually(lambda: len(source.snapshot().items) == 1)
    assert client.lists["pods"] == 2
    assert client.lists["jobs.batch"] == 1
    source.resync()
    eventually(lambda: client.lists["jobs.batch"] == 2 and client.lists["events"] == 2)
    eventually(lambda: len(transport.calls) == 7)


def test_partial_outage_retains_objects_and_bookmark_recovers_health(inventory):
    source, _, transport = inventory
    before = source.snapshot().items

    async def fail():
        for _ in range(3):
            await source._unavailable("events", {"error": "temporary timeout"})

    asyncio.run_coroutine_threadsafe(fail(), source._loop).result(timeout=3)
    assert source.snapshot().error == "events: temporary timeout"
    assert source.snapshot().items == before
    transport.queues["events"].put(
        {"type": "BOOKMARK", "object": {"metadata": {"resourceVersion": "20"}}}
    )
    eventually(lambda: not source.snapshot().error)
    assert source.snapshot().items == before


def test_raw_lists_without_item_kinds_are_normalized():
    client, transport = Client(), Transport()
    for items in client.items.values():
        for item in items:
            item.pop("kind", None)
    source = DashboardInventory(client, transport=transport)
    try:
        eventually(lambda: len(transport.calls) == 3)
        rows = UsageCollector("research", {}, 0.1, inventory_source=source).collect_inventory()
        assert len(rows) == 1
        assert rows[0].uid == "job-1"
        assert rows[0].pod_count == 1
    finally:
        source.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["Running", "Pending"])
async def test_live_dashboard_reconnects_while_metrics_and_inventory_are_idle(
    inventory, monkeypatch, tmp_path, phase,
):
    from falcon.agent_presence import WaitPresence
    from falcon.dashboard_ui import FalconDashboard, _job_status_display

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.setattr("falcon.dashboard_ui.AGENT_PRESENCE_REFRESH_SECONDS", 0.02)
    source, client, transport = inventory
    managed = job()
    managed["metadata"]["labels"] = {"falcon.dev/agent-managed": "true"}
    if phase == "Pending":
        pending_pod = pod()
        pending_pod["status"]["phase"] = "Pending"
        transport.queues["pods"].put({"type": "MODIFIED", "object": pending_pod})
        eventually(lambda: any(
            item.get("kind") == "Pod" and item.get("status", {}).get("phase") == "Pending"
            for item in source.snapshot().items
        ))
    revision = source.snapshot().revision
    transport.queues["jobs.batch"].put({"type": "MODIFIED", "object": managed})
    eventually(lambda: source.snapshot().revision > revision)
    collector = UsageCollector(
        "research", {}, 0.1, inventory_source=source, collect_availability=False
    )
    release = threading.Event()

    def blocked_collect():
        release.wait(10)
        return collector.collect_inventory()

    collector.collect = blocked_collect
    app = FalconDashboard(collector, refresh_seconds=999)
    lease = WaitPresence(None, "research", ["job-1"])
    try:
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause(0.2)
            assert _job_status_display(app.rows[0], 0)[0] == "●"
            initial_revision = source.snapshot().revision
            baseline_lists = client.lists.copy()

            async def wait_for_state(state):
                async def observed():
                    while app.rows[0].agent_state != state:
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(observed(), 2)

            for _ in range(2):
                lease = WaitPresence(None, "research", ["job-1"])
                lease.renew()
                await wait_for_state("connected")
                assert app._agent_is_connected(app.rows[0])
                assert _job_status_display(app.rows[0], 0)[0] == "◴"
                assert _job_status_display(app.rows[0], 1)[0] == "◷"
                lease.close()
                await wait_for_state("disconnected")
                assert _job_status_display(app.rows[0], 0)[0] == "●"
            assert app._refreshing, "presence must update even with metrics blocked"
            assert source.snapshot().revision == initial_revision
            assert client.lists == baseline_lists
    finally:
        lease.close()
        release.set()
    app._presence_thread.join(timeout=1)
    assert not app._presence_thread.is_alive()


@pytest.mark.asyncio
async def test_textual_inventory_event_updates_while_metrics_worker_is_blocked(inventory):
    from falcon.dashboard_ui import FalconDashboard

    source, _, transport = inventory
    collector = UsageCollector(
        "research", {}, 0.1, inventory_source=source, collect_availability=False
    )
    release = threading.Event()
    started = threading.Event()

    def blocked_collect():
        started.set()
        release.wait(10)
        return collector.collect_inventory()

    collector.collect = blocked_collect
    app = FalconDashboard(collector, refresh_seconds=999)
    try:
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause(0.2)
            assert started.is_set()
            assert app.rows[0].status == "Running"
            app.state.marked_job_uids.add("job-1")
            before = len(app.summary_history)
            changed = job()
            changed["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
            transport.queues["jobs.batch"].put({"type": "MODIFIED", "object": changed})
            await pilot.pause(0.35)
            assert app.rows[0].status == "Succeeded"
            assert app.state.cursor_job_uid == "job-1"
            assert app.state.marked_job_uids == {"job-1"}
            assert len(app.summary_history) == before
    finally:
        release.set()
