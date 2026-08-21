from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from falcon.cluster import ClusterSnapshot
from falcon.config import DEFAULT_CONFIG, validate_config
from falcon.demo import demo_cluster_snapshot
from falcon.resource_service import (
    ResourceHTTPServer,
    ResourceServiceClient,
    ResourceServiceError,
    ResourceState,
    SharedHistoryStore,
    decode_envelope,
    envelope,
    semantic_digest,
    snapshot_from_dict,
    snapshot_to_dict,
)


class ResourceWireTests(unittest.TestCase):
    def test_shared_endpoint_is_the_default_and_null_enables_legacy_mode(self) -> None:
        self.assertEqual(
            DEFAULT_CONFIG["cluster"]["resource_service_url"],
            "http://node1.yoda.hyperverge.org:30081",
        )
        legacy = json.loads(json.dumps(DEFAULT_CONFIG))
        legacy["cluster"]["resource_service_url"] = None
        validate_config(legacy)

    def test_snapshot_schema_round_trip_preserves_planning_model(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        decoded = snapshot_from_dict(snapshot_to_dict(snapshot))
        self.assertEqual(decoded, snapshot)
        revision, decoded_envelope = decode_envelope(
            envelope(snapshot, "revision-1")
        )
        self.assertEqual(revision, "revision-1")
        self.assertEqual(decoded_envelope, snapshot)

    def test_semantic_revision_ignores_collection_timestamp(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        later = replace(snapshot, collected_at=snapshot.collected_at + 100)
        self.assertEqual(semantic_digest(snapshot), semantic_digest(later))

    def test_schema_validation_rejects_other_versions(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema"):
            decode_envelope({"schema": "falcon/resource-service/v2"})


class ResourceStateTests(unittest.TestCase):
    def test_publish_is_change_only_and_subscriber_is_coalesced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = ResourceState(root, SharedHistoryStore(root / "history.sqlite3"))
            snapshot = demo_cluster_snapshot("mixed")
            self.assertTrue(state.publish(snapshot))
            subscriber, initial = state.subscribe()
            self.assertEqual(initial[1], snapshot)
            self.assertFalse(
                state.publish(replace(snapshot, collected_at=snapshot.collected_at + 5))
            )
            stale = snapshot.mark_stale("upstream failed")
            recovered = replace(snapshot, stale=False, error=None)
            self.assertTrue(state.publish(stale))
            self.assertTrue(state.publish(recovered))
            # A size-one queue retains only the newest full replacement.
            self.assertEqual(subscriber.qsize(), 1)
            self.assertEqual(subscriber.get_nowait()[1], recovered)
            state.unsubscribe(subscriber)

    def test_restart_loads_persisted_snapshot_as_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = ResourceState(root, SharedHistoryStore(root / "history.sqlite3"))
            state.publish(demo_cluster_snapshot("mixed"))
            restarted = ResourceState(root, SharedHistoryStore(root / "history.sqlite3"))
            self.assertIsNotNone(restarted.snapshot)
            self.assertTrue(restarted.snapshot.stale)
            self.assertIn("restarted", restarted.snapshot.error)


class SharedHistoryTests(unittest.TestCase):
    def test_records_only_allocation_changes_and_extends_final_step(self) -> None:
        snapshot = replace(demo_cluster_snapshot("mixed"), collected_at=2_000_000_000)
        with tempfile.TemporaryDirectory() as temporary:
            store = SharedHistoryStore(Path(temporary) / "history.sqlite3")
            self.assertTrue(store.record(snapshot))
            self.assertFalse(store.record(replace(snapshot, collected_at=2_000_000_005)))
            empty = ClusterSnapshot(collected_at=2_000_000_010)
            self.assertTrue(store.record(empty))
            points = store.load(now=2_000_000_020)
        self.assertEqual([point.timestamp for point in points], [2_000_000_000, 2_000_000_010, 2_000_000_020])
        self.assertEqual(points[0].total, 5)
        self.assertEqual(points[-1].total, 0)

    def test_retention_keeps_pre_window_baseline(self) -> None:
        snapshot = replace(demo_cluster_snapshot("mixed"), collected_at=2_000_000_000)
        with tempfile.TemporaryDirectory() as temporary:
            store = SharedHistoryStore(Path(temporary) / "history.sqlite3", hours=1)
            store.record(snapshot)
            store.record(ClusterSnapshot(collected_at=2_000_003_700))
            points = store.load(now=2_000_003_800)
        self.assertEqual(points[0].timestamp, 2_000_000_200)
        self.assertEqual(points[0].total, 5)
        self.assertEqual(points[1].total, 0)
        self.assertEqual(points[-1].timestamp, 2_000_003_800)


class ResourceClientTests(unittest.TestCase):
    def test_failure_uses_persisted_snapshot_and_marks_it_stale(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = envelope(snapshot, "one")
        with tempfile.TemporaryDirectory() as temporary:
            client = ResourceServiceClient(
                "http://resource.test", cache_path=Path(temporary) / "cache.json"
            )
            with patch("falcon.resource_service.httpx.get", return_value=response):
                self.assertEqual(client.snapshot(), snapshot)
            with patch(
                "falcon.resource_service.httpx.get",
                side_effect=httpx.ConnectError("offline"),
            ):
                stale = client.snapshot()
        self.assertTrue(stale.stale)
        self.assertEqual(stale.nodes, snapshot.nodes)

    def test_cold_failure_is_clear_and_never_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            client = ResourceServiceClient(
                "http://resource.test", cache_path=Path(temporary) / "missing.json"
            )
            with patch(
                "falcon.resource_service.httpx.get",
                side_effect=httpx.ConnectError("offline"),
            ), self.assertRaisesRegex(ResourceServiceError, "no cached snapshot"):
                client.snapshot()


class ResourceHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.state = ResourceState(root, SharedHistoryStore(root / "history.sqlite3"))
        self.snapshot = demo_cluster_snapshot("mixed")
        self.state.publish(self.snapshot)
        try:
            self.server = ResourceHTTPServer(("127.0.0.1", 0), self.state)
        except PermissionError:
            self.temporary.cleanup()
            self.skipTest("sandbox does not permit loopback sockets")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

    def tearDown(self) -> None:
        if not hasattr(self, "server"):
            return
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def request(self, method: str, path: str, headers=None):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=2)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        body = response.read()
        result = response.status, dict(response.getheaders()), body
        connection.close()
        return result

    def test_snapshot_etag_history_and_read_only_methods(self) -> None:
        status, headers, body = self.request("GET", "/v1/snapshot")
        self.assertEqual(status, 200)
        revision, snapshot = decode_envelope(json.loads(body))
        self.assertEqual(snapshot, self.snapshot)
        self.assertEqual(headers["ETag"], f'"{revision}"')
        self.assertEqual(
            self.request("GET", "/v1/snapshot", {"If-None-Match": headers["ETag"]})[0],
            304,
        )
        self.assertEqual(self.request("POST", "/v1/snapshot")[0], 405)
        self.assertEqual(self.request("GET", "/v1/snapshot?unexpected=1")[0], 400)
        history = json.loads(self.request("GET", "/v1/history?gpu=h100")[2])
        self.assertEqual(history["schema"], "falcon/resource-service/v1")
        self.assertTrue(history["history"])

    def test_sse_is_silent_until_a_semantic_change(self) -> None:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=2)
        connection.request("GET", "/v1/stream")
        response = connection.getresponse()
        self.assertEqual(response.status, 200)

        def event_lines():
            return [response.readline().decode().rstrip("\n") for _ in range(4)]

        initial = event_lines()
        self.assertTrue(initial[0].startswith("id: "))
        self.assertEqual(initial[1], "event: snapshot")
        self.assertTrue(initial[2].startswith("data: "))
        self.assertEqual(initial[3], "")

        received = []
        done = threading.Event()

        def read_change():
            received.extend(event_lines())
            done.set()

        reader = threading.Thread(target=read_change, daemon=True)
        reader.start()
        self.assertFalse(
            self.state.publish(
                replace(self.snapshot, collected_at=self.snapshot.collected_at + 5)
            )
        )
        self.assertFalse(done.wait(0.15))
        self.state.publish(self.snapshot.mark_stale("upstream unavailable"))
        self.assertTrue(done.wait(2))
        self.assertEqual(received[1], "event: snapshot")
        connection.close()


if __name__ == "__main__":
    unittest.main()
