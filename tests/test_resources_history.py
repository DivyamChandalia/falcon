from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from falcon.config import DEFAULT_CONFIG
from falcon.demo import demo_cluster_snapshot
from falcon.resources_charts import GPUHistoryPoint
from falcon.resources_history import ResourceHistoryStore, ensure_history_collector
from falcon.resources_ui import FalconResourcesApp


class ResourceHistoryStoreTests(unittest.TestCase):
    def test_history_survives_store_instances_and_keeps_zero_snapshots(self) -> None:
        nodes = demo_cluster_snapshot("mixed").nodes
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "history.sqlite3"
            store = ResourceHistoryStore(path)
            self.assertTrue(store.record_snapshot(nodes, collected_at=2_000_000_000))
            self.assertFalse(store.record_snapshot(nodes, collected_at=2_000_000_000))
            self.assertTrue(store.record_snapshot((), collected_at=2_000_000_060))
            points = ResourceHistoryStore(path).load(now=2_000_000_061)

        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].values["team-a"], 2)
        self.assertEqual(points[0].total, 5)
        self.assertGreater(points[0].vram_total, 0)
        self.assertEqual(points[1].total, 0)

    def test_node_and_model_filters_apply_to_persisted_history(self) -> None:
        nodes = demo_cluster_snapshot("mixed").nodes
        with tempfile.TemporaryDirectory() as temporary:
            store = ResourceHistoryStore(Path(temporary) / "history.sqlite3")
            store.record_snapshot(nodes, collected_at=2_000_000_000)
            h100 = store.load(gpu_filter="h100", now=2_000_000_001)
            a6000 = store.load(node_filter="node-b", now=2_000_000_001)

        self.assertEqual(h100[0].values, {"team-a": 2.0, "team-b": 1.0})
        self.assertEqual(
            a6000[0].values,
            {"research-with-a-long-namespace": 1.0, "team-c": 1.0},
        )

    def test_retention_prunes_by_age_and_snapshot_limit(self) -> None:
        nodes = demo_cluster_snapshot("mixed").nodes
        with tempfile.TemporaryDirectory() as temporary:
            store = ResourceHistoryStore(
                Path(temporary) / "history.sqlite3", history_hours=1, limit=2
            )
            for timestamp in (2_000_000_000, 2_000_003_700, 2_000_003_701):
                store.record_snapshot(nodes, collected_at=timestamp)
            points = store.load(now=2_000_003_702)

        self.assertEqual(
            [point.timestamp for point in points],
            [2_000_003_700, 2_000_003_701],
        )

    def test_tui_loads_persisted_points_before_mount(self) -> None:
        points = []
        app = FalconResourcesApp(object(), history_loader=lambda: list(points))
        self.assertEqual(app.history, [])
        points.append(GPUHistoryPoint.from_mapping(2_000_000_000, {"team": 2}))
        app._load_persistent_history(notify=False)
        self.assertEqual(app.history, points)


class ResourceHistoryCollectorTests(unittest.TestCase):
    def test_ensure_starts_a_detached_collector(self) -> None:
        config = json.loads(json.dumps(DEFAULT_CONFIG))
        config["cluster"]["resource_service_url"] = None
        with tempfile.TemporaryDirectory() as temporary, patch(
            "falcon.resources_history._state_root", return_value=Path(temporary)
        ), patch("falcon.resources_history.subprocess.Popen") as popen:
            self.assertTrue(ensure_history_collector(config))

        command = popen.call_args.args[0]
        self.assertEqual(command[1:3], ["-m", "falcon.resources_history"])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertTrue(popen.call_args.kwargs["close_fds"])

    def test_disabled_history_does_not_start_collector(self) -> None:
        config = json.loads(json.dumps(DEFAULT_CONFIG))
        config["resources"]["history_enabled"] = False
        with patch("falcon.resources_history.subprocess.Popen") as popen:
            self.assertFalse(ensure_history_collector(config))
        popen.assert_not_called()

    def test_running_collector_is_not_started_twice(self) -> None:
        config = json.loads(json.dumps(DEFAULT_CONFIG))
        with patch(
            "falcon.resources_history._collector_is_running", return_value=True
        ), patch("falcon.resources_history.subprocess.Popen") as popen:
            self.assertFalse(ensure_history_collector(config))
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
