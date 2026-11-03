import unittest
from dataclasses import replace
from datetime import datetime, timezone

from falcon.demo import DEMO_NOW, DemoCollector, demo_cluster_snapshot
from falcon.resources_ui import FalconResourcesApp, _resource_age
from falcon.theme import WHITE


class ResourceAgeTests(unittest.TestCase):
    def test_unit_boundaries_and_week_cap(self):
        for seconds, expected in ((0, "0s"), (1, "1s"), (59, "59s"), (60, "1m"),
                                  (3599, "59m"), (3600, "1h"), (86399, "23h"),
                                  (86400, "1d"), (604799, "6d"), (604800, "1w"),
                                  (99 * 604800, "99w"), (1000 * 604800, "99w")):
            with self.subTest(seconds=seconds):
                created = datetime.fromtimestamp(DEMO_NOW - seconds, timezone.utc).isoformat()
                result = _resource_age(created, DEMO_NOW)
                self.assertEqual(result, expected)
                self.assertLessEqual(len(result), 3)

    def test_missing_invalid_future_and_timezone(self):
        for value in (None, "", "invalid", 123):
            self.assertEqual(_resource_age(value, DEMO_NOW), "-")
        future = datetime.fromtimestamp(DEMO_NOW + 60, timezone.utc).isoformat()
        self.assertEqual(_resource_age(future, DEMO_NOW), "0s")
        created = datetime.fromtimestamp(DEMO_NOW - 60, timezone.utc)
        self.assertEqual(_resource_age(created.replace(tzinfo=None).isoformat(), DEMO_NOW), "1m")
        self.assertEqual(_resource_age(created.isoformat().replace("+00:00", "Z"), DEMO_NOW), "1m")

    def test_job_creation_precedes_pod_creation_with_fallback(self):
        app = FalconResourcesApp(DemoCollector("mixed"))
        snapshot = demo_cluster_snapshot("mixed")
        job = snapshot.jobs[0]
        consumer = snapshot.nodes[0].visible_consumers[0]
        pod_time = datetime.fromtimestamp(DEMO_NOW - 60, timezone.utc).isoformat()
        job_time = datetime.fromtimestamp(DEMO_NOW - 86400, timezone.utc).isoformat()
        consumer = replace(consumer, namespace=job.namespace, workload_name=job.name,
                           workload_kind="Job", created_at=pod_time)
        app.snapshot = replace(snapshot, jobs=(replace(job, created_at=job_time),))
        self.assertEqual(app._consumer_age(consumer), "1d")
        app.snapshot = replace(app.snapshot, jobs=())
        self.assertEqual(app._consumer_age(consumer), "1m")
        self.assertEqual(app._consumer_age(replace(consumer, created_at=None)), "-")


class ResourceAgeLayoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_tables_keep_fixed_age_column_at_supported_sizes(self):
        for size in ((80, 20), (80, 22), (160, 30), (200, 50)):
            with self.subTest(size=size):
                app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999,
                                         initial_view="gpu-allocations")
                async with app.run_test(size=size) as pilot:
                    await pilot.pause(0.3)
                    table = app._allocation_table(width=40 if size[0] == 80 else size[0] // 2 - 4)
                    age = next(column for column in table.columns if column.header == "AGE")
                    self.assertEqual(age.width, 7)
                    self.assertEqual(age.width, table.columns[-2].width)
                    rows = app._allocation_rows()
                    for cell, (_, consumer) in zip(age._cells, rows):
                        self.assertEqual(cell.plain, app._consumer_age(consumer) if consumer else "")
                        if consumer:
                            self.assertEqual(cell.style, WHITE)
                    node = app._selected()
                    for expanded in (False, True):
                        table = app._consumer_table(node, expanded=expanded)
                        age = next(column for column in table.columns if column.header == "AGE")
                        self.assertEqual(age.width, 5)
                        self.assertTrue(all(len(cell.plain) <= 3 for cell in age._cells))
                        self.assertTrue(table.collapse_padding)


if __name__ == "__main__":
    unittest.main()
