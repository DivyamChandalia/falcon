from __future__ import annotations

import unittest

from dataclasses import replace

from falcon.demo import demo_cluster_snapshot
from falcon.resources_telemetry import allocation_snapshot


class AllocationTelemetryTests(unittest.TestCase):
    def test_gpu_and_vram_bases_are_derived_from_the_same_allocations(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        nodes = snapshot.nodes[:2]
        telemetry = allocation_snapshot(nodes, collected_at=123.0)

        self.assertEqual(telemetry.collected_at, 123.0)
        self.assertEqual(telemetry.target_pods, telemetry.sampled_pods)
        self.assertEqual(
            sum(value for _, value in telemetry.effective_gpus_by_namespace),
            sum(node.requested.gpu_count for node in nodes if node.ready and node.schedulable),
        )
        self.assertGreater(sum(value for _, value in telemetry.vram_gib_by_namespace), 0)

    def test_stale_inventory_is_retained_but_not_presented_as_a_new_point(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        stale = replace(snapshot, stale=True, error="metrics endpoint unavailable")
        telemetry = allocation_snapshot(
            stale.nodes,
            collected_at=stale.collected_at,
            stale=stale.stale,
            error=stale.error or "",
        )
        self.assertTrue(telemetry.stale)
        self.assertEqual(telemetry.error, "metrics endpoint unavailable")


if __name__ == "__main__":
    unittest.main()
