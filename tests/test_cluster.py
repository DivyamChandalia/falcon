from __future__ import annotations

import unittest

from falcon.cluster import (
    ClusterCollector,
    ResourceVector,
    build_job_snapshot,
    gpu_memory_bytes_from_labels,
    is_system_namespace,
    natural_name_key,
    normalize_gpu_model,
    parse_cpu_quantity,
    parse_memory_quantity,
)
from falcon.demo import demo_cluster_snapshot, demo_inventory


class ClusterSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot = demo_cluster_snapshot("mixed")

    def _job(self, name: str):
        return next(job for job in self.snapshot.jobs if job.name == name)

    def test_completed_gpu_request_survives(self) -> None:
        job = self._job("finished-h100-training")
        self.assertEqual(job.status, "Succeeded")
        self.assertEqual((job.gpu_requested_model, job.gpu_requested), ("H100", 2))

    def test_completed_gpu_has_no_current_allocation(self) -> None:
        job = self._job("finished-h100-training")
        self.assertEqual(job.gpu_allocated, 0)
        self.assertIsNone(job.gpu_allocated_model)
        self.assertEqual(job.nodes, ())

    def test_true_cpu_only_job_has_no_gpu(self) -> None:
        job = self._job("finished-cpu-report")
        self.assertEqual(job.requested.gpu_count, 0)
        self.assertEqual(job.allocated.gpu_count, 0)

    def test_queued_gpu_is_requested_but_not_allocated(self) -> None:
        job = self._job("queued-h100-experiment")
        self.assertEqual(job.requested.gpu_count, 2)
        self.assertEqual(job.allocated.gpu_count, 0)
        self.assertEqual(job.status, "Pending")

    def test_running_gpu_is_requested_and_allocated(self) -> None:
        job = self._job("train-h100-two")
        self.assertEqual(job.requested.gpu_count, 2)
        self.assertEqual(job.allocated.gpu_count, 2)
        self.assertEqual(job.nodes, ("node-a-h100",))

    def test_retried_job_aggregates_all_container_restarts(self) -> None:
        job = self._job("retry-eventually-succeeded")
        self.assertEqual(job.attempts.container_restarts, 3)

    def test_retried_job_aggregates_pod_attempts(self) -> None:
        attempts = self._job("retry-eventually-succeeded").attempts
        self.assertEqual(attempts.pod_attempts, 2)
        self.assertEqual(attempts.failed_attempts, 1)
        self.assertEqual(attempts.succeeded_attempts, 1)

    def test_retried_completed_job_has_no_active_pod(self) -> None:
        attempts = self._job("retry-eventually-succeeded").attempts
        self.assertIsNone(attempts.active_pod)
        self.assertEqual(attempts.backoff_limit, 6)

    def test_job_command_includes_command_and_args(self) -> None:
        job = self._job("train-h100-two")
        self.assertEqual(
            job.command,
            ("python", "-u", "train.py", "--model", "large", "--epochs", "100"),
        )

    def test_missing_pods_do_not_erase_template_request(self) -> None:
        inventory = demo_inventory("mixed")
        job = next(
            item for item in inventory
            if item.get("kind") == "Job"
            and item["metadata"]["name"] == "finished-h100-training"
        )
        snapshot = build_job_snapshot(job, [])
        self.assertEqual(snapshot.requested.gpu_count, 2)
        self.assertEqual(snapshot.attempts.pod_attempts, 0)

    def test_init_container_uses_scheduler_max_semantics(self) -> None:
        inventory = demo_inventory("one-job")
        job = next(item for item in inventory if item.get("kind") == "Job")
        pod = next(item for item in inventory if item.get("kind") == "Pod")
        pod["spec"]["initContainers"] = [
            {"name": "init", "resources": {"requests": {"cpu": "20", "memory": "2Gi"}}}
        ]
        snapshot = build_job_snapshot(job, [pod])
        self.assertEqual(snapshot.allocated.cpu_cores, 20)

    def test_quantity_parsers_cover_kubernetes_units(self) -> None:
        self.assertEqual(parse_cpu_quantity("500m"), 0.5)
        self.assertEqual(parse_memory_quantity("512Mi"), 512 * 1024**2)

    def test_gpu_model_normalization_never_invents_missing_data(self) -> None:
        self.assertIsNone(normalize_gpu_model(None))
        self.assertEqual(normalize_gpu_model("NVIDIA_A6000"), "A6000")

    def test_gpu_memory_label_is_per_device_mib(self) -> None:
        self.assertEqual(
            gpu_memory_bytes_from_labels({"nvidia.com/gpu.memory": "49140"}),
            49140 * 1024**2,
        )
        self.assertEqual(
            gpu_memory_bytes_from_labels({"nvidia.com/gpu.memory": "80Gi"}),
            80 * 1024**3,
        )
        self.assertIsNone(
            gpu_memory_bytes_from_labels({"nvidia.com/gpu.memory": "invalid"})
        )

    def test_node_names_use_natural_numeric_order(self) -> None:
        values = ["node10", "nodex1", "node1", "node9", "node3"]
        self.assertEqual(
            sorted(values, key=natural_name_key),
            ["node1", "node3", "node9", "node10", "nodex1"],
        )

    def test_autoscaling_and_gpu_eviction_namespaces_are_system(self) -> None:
        self.assertTrue(is_system_namespace("keda"))
        self.assertTrue(is_system_namespace("gpu-evictor"))


class NodeAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot = demo_cluster_snapshot("mixed")
        self.node = next(node for node in self.snapshot.nodes if node.name == "node-a-h100")

    def test_node_gpu_headroom_reconciles(self) -> None:
        self.assertEqual(self.node.allocatable.gpu_count, 4)
        self.assertEqual(self.node.requested.gpu_count, 3)
        self.assertEqual(self.node.gpu_free, 1)
        self.assertEqual(
            self.node.gpu_memory_bytes_per_device,
            81559 * 1024**2,
        )

    def test_node_lists_falcon_and_other_consumers(self) -> None:
        kinds = {(consumer.namespace, consumer.workload_name) for consumer in self.node.consumers}
        self.assertIn(("team-a", "train-h100-two"), kinds)
        self.assertIn(("team-b", "namespace-b-inference"), kinds)
        self.assertIn(("monitoring", "prometheus-node-exporter"), kinds)

    def test_system_consumers_are_accounted_but_hidden(self) -> None:
        raw = {
            (consumer.namespace, consumer.workload_name)
            for consumer in self.node.consumers
        }
        visible = {
            (consumer.namespace, consumer.workload_name)
            for consumer in self.node.visible_consumers
        }
        self.assertIn(("monitoring", "prometheus-node-exporter"), raw)
        self.assertNotIn(("monitoring", "prometheus-node-exporter"), visible)
        self.assertEqual(self.node.workload_count, len(visible))

    def test_owner_is_shown_only_when_present(self) -> None:
        known = next(
            item for item in self.node.consumers if item.workload_name == "train-h100-two"
        )
        unknown = next(
            item for item in self.node.consumers if item.workload_name == "namespace-b-inference"
        )
        self.assertEqual(known.owner_identity, "alice@example.com")
        self.assertIsNone(unknown.owner_identity)

    def test_unschedulable_node_is_explicit(self) -> None:
        node = next(item for item in self.snapshot.nodes if "cordoned" in item.name)
        self.assertFalse(node.schedulable)
        self.assertTrue(node.taints)

    def test_cluster_gpu_models_are_grouped(self) -> None:
        values = self.snapshot.gpu_availability
        self.assertIn("H100", values)
        self.assertEqual(values["H100"].request_headroom, 1)

    def test_no_jobs_fixture_retains_nodes(self) -> None:
        snapshot = demo_cluster_snapshot("no-jobs")
        self.assertEqual(len(snapshot.nodes), 4)
        self.assertFalse(snapshot.jobs)

    def test_stale_fixture_retains_last_known_good_values(self) -> None:
        snapshot = demo_cluster_snapshot("stale")
        self.assertTrue(snapshot.stale)
        self.assertTrue(snapshot.nodes)
        self.assertIn("timeout", snapshot.error or "")


class CollectorTests(unittest.TestCase):
    def test_inventory_is_cached_by_cadence(self) -> None:
        class Source:
            calls = 0

            def list_inventory(self, namespace=None):
                self.calls += 1
                return demo_inventory("one-job")

        source = Source()
        collector = ClusterCollector(source, inventory_seconds=60)
        first = collector.collect(force=True)
        second = collector.collect()
        self.assertIs(first, second)
        self.assertEqual(source.calls, 1)

    def test_transient_error_preserves_last_good_snapshot(self) -> None:
        class Source:
            calls = 0

            def list_inventory(self, namespace=None):
                self.calls += 1
                if self.calls == 1:
                    return demo_inventory("one-job")
                raise RuntimeError("temporary API failure")

        source = Source()
        clock_values = iter((0.0, 2.0))
        collector = ClusterCollector(
            source,
            inventory_seconds=1,
            failure_retry_seconds=1,
            clock=lambda: next(clock_values),
        )
        good = collector.collect(force=True)
        stale = collector.collect(force=True)
        self.assertEqual(stale.nodes, good.nodes)
        self.assertTrue(stale.stale)
        self.assertIn("temporary API failure", stale.error or "")

    def test_resource_vector_null_usage_is_not_replaced_by_zero(self) -> None:
        job = demo_cluster_snapshot("mixed").jobs[0]
        self.assertIsNone(job.current_usage)
        self.assertNotEqual(job.requested, ResourceVector())


if __name__ == "__main__":
    unittest.main()
