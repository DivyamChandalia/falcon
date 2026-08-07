from __future__ import annotations

import unittest
from unittest.mock import patch

from falcon.dashboard import UsageCollector
from falcon.models import NodeResources
from falcon.resources import (
    MetricsClusterCollector,
    cluster_snapshot_from_metrics,
    parse_prometheus_metrics,
)

GIB = 1024**3

METRICS = f"""
kube_node_status_capacity{{node="node-a",resource="cpu",unit="core"}} 10
kube_node_status_capacity{{node="node-a",resource="memory",unit="byte"}} {20 * GIB}
kube_node_status_capacity{{node="node-a",resource="nvidia_com_gpu",unit="integer"}} 6
kube_node_status_allocatable{{node="node-a",resource="cpu",unit="core"}} 8
kube_node_status_allocatable{{node="node-a",resource="memory",unit="byte"}} {16 * GIB}
kube_node_status_allocatable{{node="node-a",resource="nvidia_com_gpu",unit="integer"}} 4
kube_node_status_condition{{condition="Ready",node="node-a",status="true"}} 1
kube_node_spec_unschedulable{{node="node-a"}} 0
kube_node_labels{{node="node-a",label_nvidia_com_gpu_memory="81559",label_nvidia_com_gpu_product="NVIDIA_H100_80GB_HBM3"}} 1

kube_node_status_capacity{{node="node-b",resource="cpu",unit="core"}} 6
kube_node_status_capacity{{node="node-b",resource="memory",unit="byte"}} {10 * GIB}
kube_node_status_capacity{{node="node-b",resource="nvidia_com_gpu",unit="integer"}} 2
kube_node_status_allocatable{{node="node-b",resource="cpu",unit="core"}} 4
kube_node_status_allocatable{{node="node-b",resource="memory",unit="byte"}} {8 * GIB}
kube_node_status_allocatable{{node="node-b",resource="nvidia_com_gpu",unit="integer"}} 2
kube_node_status_condition{{condition="Ready",node="node-b",status="true"}} 1
kube_node_spec_unschedulable{{node="node-b"}} 1
kube_node_labels{{node="node-b",label_nvidia_com_gpu_memory="49140",label_nvidia_com_gpu_product="NVIDIA_RTX_A6000"}} 1
kube_node_spec_taint{{node="node-b",key="maintenance",value="true",effect="NoSchedule"}} 1

kube_pod_info{{namespace="team-a",pod="train-abc",node="node-a",created_by_kind="Job",created_by_name="train"}} 1
kube_pod_owner{{namespace="team-a",pod="train-abc",owner_kind="Job",owner_name="train",owner_is_controller="true"}} 1
kube_pod_status_phase{{namespace="team-a",pod="train-abc",phase="Running"}} 1
kube_pod_created{{namespace="team-a",pod="train-abc"}} 1700000000
kube_pod_container_resource_requests{{namespace="team-a",pod="train-abc",container="main",node="node-a",resource="cpu",unit="core"}} 2
kube_pod_container_resource_requests{{namespace="team-a",pod="train-abc",container="main",node="node-a",resource="memory",unit="byte"}} {4 * GIB}
kube_pod_container_resource_requests{{namespace="team-a",pod="train-abc",container="main",node="node-a",resource="nvidia_com_gpu",unit="integer"}} 2

kube_pod_info{{namespace="services",pod="api-xyz",node="node-a",created_by_kind="ReplicaSet",created_by_name="api-7c9"}} 1
kube_pod_owner{{namespace="services",pod="api-xyz",owner_kind="ReplicaSet",owner_name="api-7c9",owner_is_controller="true"}} 1
kube_replicaset_owner{{namespace="services",replicaset="api-7c9",owner_kind="Deployment",owner_name="api",owner_is_controller="true"}} 1
kube_pod_status_phase{{namespace="services",pod="api-xyz",phase="Pending"}} 1
kube_pod_container_resource_requests{{namespace="services",pod="api-xyz",container="api",node="node-a",resource="cpu",unit="core"}} 1
kube_pod_container_resource_requests{{namespace="services",pod="api-xyz",container="api",node="node-a",resource="memory",unit="byte"}} {GIB}

kube_pod_info{{namespace="kube-system",pod="kube-proxy-node-a",node="node-a",created_by_kind="Node",created_by_name="node-a"}} 1
kube_pod_status_phase{{namespace="kube-system",pod="kube-proxy-node-a",phase="Running"}} 1
kube_pod_container_resource_requests{{namespace="kube-system",pod="kube-proxy-node-a",container="proxy",node="node-a",resource="cpu",unit="core"}} 0.5
kube_pod_container_resource_requests{{namespace="kube-system",pod="kube-proxy-node-a",container="proxy",node="node-a",resource="memory",unit="byte"}} {GIB // 2}

kube_pod_info{{namespace="team-a",pod="old-attempt",node="node-a",created_by_kind="Job",created_by_name="old"}} 1
kube_pod_status_phase{{namespace="team-a",pod="old-attempt",phase="Succeeded"}} 1
kube_pod_container_resource_requests{{namespace="team-a",pod="old-attempt",container="main",node="node-a",resource="cpu",unit="core"}} 4
kube_pod_container_resource_requests{{namespace="team-a",pod="old-attempt",container="main",node="node-a",resource="nvidia_com_gpu",unit="integer"}} 1

kube_pod_info{{namespace="other",pod="busy",node="node-b",created_by_kind="Pod",created_by_name=""}} 1
kube_pod_status_phase{{namespace="other",pod="busy",phase="Running"}} 1
kube_pod_container_resource_requests{{namespace="other",pod="busy",container="main",node="node-b",resource="cpu",unit="core"}} 1
kube_pod_container_resource_requests{{namespace="other",pod="busy",container="main",node="node-b",resource="nvidia_com_gpu",unit="integer"}} 1

kube_job_info{{namespace="team-a",job_name="train"}} 1
kube_job_status_active{{namespace="team-a",job_name="train"}} 1
kube_job_status_succeeded{{namespace="team-a",job_name="train"}} 0
kube_job_status_failed{{namespace="team-a",job_name="train"}} 0
kube_job_created{{namespace="team-a",job_name="train"}} 1699999900
"""


class MetricsResourceTests(unittest.TestCase):
    def test_cluster_snapshot_parses_metrics_once(self) -> None:
        with patch("falcon.resources.parse_prometheus_metrics", wraps=parse_prometheus_metrics) as parser:
            snapshot = cluster_snapshot_from_metrics(METRICS)
        self.assertTrue(snapshot.nodes)
        parser.assert_called_once()

    def test_old_resource_semantics_are_available_without_node_rbac(self) -> None:
        snapshot = cluster_snapshot_from_metrics(METRICS, collected_at=12.0)
        nodes = {node.name: node for node in snapshot.nodes}
        node = nodes["node-a"]

        self.assertEqual(snapshot.total_nodes, 2)
        self.assertEqual(snapshot.schedulable_nodes, 1)
        self.assertEqual(node.capacity.gpu_count, 6)
        self.assertEqual(node.allocatable.gpu_count, 4)
        self.assertEqual(node.requested.gpu_count, 2)
        self.assertEqual(node.requested.cpu_cores, 3.5)
        self.assertEqual(node.requested.memory_bytes, 5 * GIB + GIB // 2)
        self.assertEqual(node.gpu_free, 2)
        self.assertEqual(node.gpu_memory_bytes_per_device, 81559 * 1024**2)

    def test_finished_pods_do_not_consume_current_headroom(self) -> None:
        snapshot = cluster_snapshot_from_metrics(METRICS)
        node = next(item for item in snapshot.nodes if item.name == "node-a")
        self.assertNotIn("old-attempt", {item.pod_name for item in node.consumers})
        self.assertEqual(snapshot.requested.gpu_count, 2)

    def test_unschedulable_nodes_are_excluded_from_cluster_availability(self) -> None:
        snapshot = cluster_snapshot_from_metrics(METRICS)
        self.assertEqual(snapshot.allocatable.cpu_cores, 8)
        self.assertEqual(snapshot.allocatable.gpu_count, 4)
        self.assertEqual(set(snapshot.gpu_availability), {"H100"})
        h100 = snapshot.gpu_availability["H100"]
        self.assertEqual((h100.request_headroom, h100.allocatable), (2, 4))

    def test_node_consumers_retain_reliable_workload_identity_only(self) -> None:
        snapshot = cluster_snapshot_from_metrics(METRICS)
        node = next(item for item in snapshot.nodes if item.name == "node-a")
        consumers = {item.pod_name: item for item in node.consumers}
        self.assertEqual(consumers["train-abc"].workload_name, "train")
        self.assertEqual(consumers["api-xyz"].workload_kind, "Deployment")
        self.assertEqual(consumers["api-xyz"].workload_name, "api")
        self.assertTrue(all(item.owner_identity is None for item in consumers.values()))

    def test_consumer_gpu_models_use_compact_display_names(self) -> None:
        snapshot = cluster_snapshot_from_metrics(METRICS)
        node = next(item for item in snapshot.nodes if item.name == "node-b")
        consumer = next(
            item for item in node.visible_consumers if item.pod_name == "busy"
        )
        self.assertEqual(consumer.requested.gpu_model, "A6000")

    def test_system_pods_are_hidden_from_consumers_and_pod_counts(self) -> None:
        snapshot = cluster_snapshot_from_metrics(METRICS)
        node = next(item for item in snapshot.nodes if item.name == "node-a")
        visible = {item.pod_name for item in node.visible_consumers}
        self.assertNotIn("kube-proxy-node-a", visible)
        self.assertEqual(node.workload_count, 2)
        self.assertEqual(snapshot.running_pods, 2)
        self.assertEqual(snapshot.pending_pods, 1)
        # Infrastructure requests still consume scheduler headroom.
        self.assertEqual(node.requested.cpu_cores, 3.5)

    def test_metrics_jobs_feed_resource_overview_counts(self) -> None:
        snapshot = cluster_snapshot_from_metrics(METRICS)
        self.assertEqual(snapshot.running_jobs, 1)
        self.assertEqual(snapshot.pending_jobs, 0)
        self.assertEqual(snapshot.jobs[0].requested.gpu_count, 2)


class MetricsCollectorTests(unittest.TestCase):
    def test_inventory_is_cached_at_a_bounded_cadence(self) -> None:
        clock = iter((10.0, 11.0))
        collector = MetricsClusterCollector(
            inventory_seconds=5,
            clock=lambda: next(clock),
        )
        snapshot = cluster_snapshot_from_metrics(METRICS, collected_at=10.0)
        with patch(
            "falcon.resources.fetch_cluster_snapshot",
            return_value=snapshot,
        ) as fetch:
            first = collector.collect(force=True)
            second = collector.collect()
        self.assertIs(first, second)
        fetch.assert_called_once()

    def test_transient_failure_keeps_the_last_valid_resource_frame(self) -> None:
        clock = iter((10.0, 20.0))
        collector = MetricsClusterCollector(
            inventory_seconds=5,
            failure_retry_seconds=1,
            clock=lambda: next(clock),
        )
        snapshot = cluster_snapshot_from_metrics(METRICS, collected_at=10.0)
        with patch(
            "falcon.resources.fetch_cluster_snapshot",
            side_effect=(snapshot, OSError("metrics proxy unavailable")),
        ):
            good = collector.collect(force=True)
            stale = collector.collect(force=True)
        self.assertEqual(stale.nodes, good.nodes)
        self.assertTrue(stale.stale)
        self.assertIn("metrics proxy unavailable", stale.error or "")


class DashboardAvailabilityTests(unittest.TestCase):
    def test_dashboard_uses_metrics_headroom_without_listing_nodes(self) -> None:
        nodes = [
            NodeResources(
                "h100-a",
                gpu_total=8,
                gpu_used=7,
                gpu_product="NVIDIA H100 80GB HBM3",
            ),
            NodeResources(
                "h100-cordoned",
                gpu_total=4,
                gpu_used=0,
                gpu_product="H100",
                unschedulable=True,
            ),
            NodeResources(
                "a6000-a",
                gpu_total=4,
                gpu_used=4,
                gpu_product="NVIDIA RTX A6000",
            ),
        ]
        collector = UsageCollector(
            "research",
            {},
            0.1,
            metrics_url="http://localhost:30080/metrics",
        )
        with patch("falcon.dashboard.fetch_nodes", return_value=nodes), patch(
            "falcon.dashboard._kubectl"
        ) as kubectl:
            collector._refresh_gpu_availability(100.0)
        self.assertEqual(
            collector.gpu_availability,
            {"h100": (1, 8), "a6000": (0, 4)},
        )
        kubectl.assert_not_called()

    def test_dashboard_rate_limits_failed_availability_refreshes(self) -> None:
        collector = UsageCollector(
            "research",
            {},
            0.1,
            metrics_url="http://localhost:30080/metrics",
        )
        collector.gpu_availability = {"h100": (1, 8)}
        with patch(
            "falcon.dashboard.fetch_nodes",
            side_effect=OSError("metrics unavailable"),
        ) as fetch, patch("falcon.dashboard._kubectl", return_value=None) as kubectl:
            collector._refresh_gpu_availability(100.0)
            collector._refresh_gpu_availability(101.0)
        self.assertEqual(collector.gpu_availability, {"h100": (1, 8)})
        fetch.assert_called_once()
        kubectl.assert_called_once()


if __name__ == "__main__":
    unittest.main()
