import unittest

from falcon.resources import NodeResources, canonical_gpu, nodes_from_metrics, plan_resources


METRICS = """
kube_node_status_allocatable{node="node5",resource="cpu"} 96
kube_node_status_allocatable{node="node5",resource="memory"} 127237013504
kube_node_status_allocatable{node="node5",resource="nvidia_com_gpu"} 4
kube_node_labels{node="node5",label_nvidia_com_gpu_product="NVIDIA_GeForce_RTX_2080_Ti"} 1
kube_node_spec_unschedulable{node="node5"} 0
kube_pod_status_phase{namespace="work",pod="busy",phase="Running"} 1
kube_pod_container_resource_requests{namespace="work",pod="busy",container="main",node="node5",resource="cpu"} 13.5
kube_pod_container_resource_requests{namespace="work",pod="busy",container="main",node="node5",resource="memory"} 125627793408
kube_pod_container_resource_requests{namespace="work",pod="busy",container="main",node="node5",resource="nvidia_com_gpu"} 2
kube_node_status_allocatable{node="node8",resource="cpu"} 96
kube_node_status_allocatable{node="node8",resource="memory"} 127237013504
kube_node_status_allocatable{node="node8",resource="nvidia_com_gpu"} 4
kube_node_labels{node="node8",label_nvidia_com_gpu_product="NVIDIA_GeForce_RTX_2080_Ti"} 1
kube_node_spec_unschedulable{node="node8"} 0
kube_pod_status_phase{namespace="work",pod="one",phase="Running"} 1
kube_pod_container_resource_requests{namespace="work",pod="one",container="main",node="node8",resource="cpu"} 8.7
kube_pod_container_resource_requests{namespace="work",pod="one",container="main",node="node8",resource="memory"} 34789235097.6
kube_pod_container_resource_requests{namespace="work",pod="one",container="main",node="node8",resource="nvidia_com_gpu"} 1
"""


class ResourcePlanningTests(unittest.TestCase):
    def test_metrics_snapshot_and_best_balanced_node(self):
        nodes = nodes_from_metrics(METRICS)
        plan = plan_resources(nodes, "2080ti", "2080ti", 2)
        self.assertEqual(plan.node, "node8")
        self.assertEqual(plan.cpu, "48:48")
        self.assertEqual(plan.memory, "59.2Gi:59.2Gi")
        self.assertTrue(plan.immediately_schedulable)

    def test_auto_resources_cap_to_free_capacity(self):
        node = NodeResources(
            "only", cpu_total=32, cpu_used=20, memory_total_gib=120,
            memory_used_gib=110, gpu_total=4, gpu_used=2, gpu_product="RTX 2080 Ti",
        )
        plan = plan_resources([node], "2080ti", "2080ti", 2)
        self.assertEqual(plan.cpu, "12:12")
        self.assertEqual(plan.memory, "10Gi:10Gi")

    def test_free_capacity_is_floored_and_never_rounded_above_available(self):
        node = NodeResources(
            "node6", cpu_total=68, cpu_used=38.805,
            memory_total_gib=103.51183700561523, memory_used_gib=74.525390625,
            gpu_total=4, gpu_used=0, gpu_product="RTX 2080 Ti",
        )
        plan = plan_resources([node], "2080ti", "2080ti", 4)
        self.assertEqual(plan.cpu, "29.1:29.1")
        self.assertEqual(plan.memory, "28.9Gi:28.9Gi")

    def test_highest_absolute_free_compute_node_is_used_for_sizing(self):
        smaller = NodeResources(
            "small", cpu_total=64, cpu_used=4, memory_total_gib=64, memory_used_gib=4,
            gpu_total=4, gpu_product="RTX 2080 Ti",
        )
        larger = NodeResources(
            "large", cpu_total=96, cpu_used=20, memory_total_gib=120, memory_used_gib=20,
            gpu_total=4, gpu_product="RTX 2080 Ti",
        )
        plan = plan_resources([smaller, larger], "2080ti", "2080ti", 4)
        self.assertEqual(plan.node, "large")
        self.assertEqual(plan.cpu, "76:76")
        self.assertEqual(plan.memory, "100Gi:100Gi")

    def test_max_requests_five_percent_below_total_capacity_even_when_not_free(self):
        node = NodeResources(
            "node7", cpu_total=96, cpu_used=36, memory_total_gib=120,
            memory_used_gib=50, gpu_total=4, gpu_used=0, gpu_product="RTX 2080 Ti",
        )
        plan = plan_resources([node], "2080ti", "2080ti", 4, maximize=True)
        self.assertEqual(plan.cpu, "91.2:91.2")
        self.assertEqual(plan.memory, "114Gi:114Gi")
        self.assertFalse(plan.immediately_schedulable)
        self.assertEqual(plan.sizing_node, "node7")
        self.assertIn("scheduled once resources are available", plan.warning)

    def test_max_uses_proportional_gpu_share_and_respects_overrides(self):
        node = NodeResources(
            "node7", cpu_total=96, memory_total_gib=120,
            gpu_total=4, gpu_product="RTX 2080 Ti",
        )
        plan = plan_resources(
            [node], "2080ti", "2080ti", 2, cpu_override="40:80", maximize=True,
        )
        self.assertEqual(plan.cpu, "40:40")
        self.assertEqual(plan.memory, "57Gi:57Gi")
        self.assertTrue(plan.immediately_schedulable)

    def test_unschedulable_override_warns(self):
        node = NodeResources(
            "node8", cpu_total=96, cpu_used=8, memory_total_gib=120,
            memory_used_gib=80, gpu_total=4, gpu_used=1, gpu_product="2080 Ti",
        )
        plan = plan_resources([node], "2080ti", "2080ti", 2, memory_override="50Gi:50Gi")
        self.assertFalse(plan.immediately_schedulable)
        self.assertIsNone(plan.node)
        self.assertIn("scheduled once resources are available", plan.warning)

    def test_busy_gpu_node_uses_capacity_instead_of_small_fallback(self):
        node = NodeResources(
            "node8", cpu_total=96, cpu_used=90, memory_total_gib=120,
            memory_used_gib=115, gpu_total=4, gpu_used=4, gpu_product="2080 Ti",
        )
        plan = plan_resources([node], "2080ti", "2080ti", 3)
        self.assertEqual(plan.cpu, "72:72")
        self.assertEqual(plan.memory, "90Gi:90Gi")
        self.assertFalse(plan.immediately_schedulable)

    def test_impossible_gpu_count_is_rejected(self):
        node = NodeResources("node8", cpu_total=96, memory_total_gib=120, gpu_total=4, gpu_product="2080 Ti")
        with self.assertRaisesRegex(ValueError, "largest matching node has 4"):
            plan_resources([node], "2080ti", "2080ti", 5)

    def test_gpu_names_are_canonical(self):
        self.assertEqual(canonical_gpu("NVIDIA H100 80GB HBM3"), "h100")
        self.assertEqual(canonical_gpu("NVIDIA RTX A6000"), "a6000")
        self.assertEqual(canonical_gpu("NVIDIA_GeForce_RTX_2080_Ti"), "2080ti")


if __name__ == "__main__":
    unittest.main()
