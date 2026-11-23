from __future__ import annotations

import json
import unittest

from falcon.resource_service import snapshot_from_dict, snapshot_to_dict
from falcon.resources import cluster_snapshot_from_metrics
from falcon.resources_ui import FalconResourcesApp

GIB = 1024**3


def _node(name: str, product: str, gpu_type: str, gpus: int) -> str:
    return f"""
kube_node_status_capacity{{node="{name}",resource="cpu",unit="core"}} 32
kube_node_status_capacity{{node="{name}",resource="memory",unit="byte"}} {128 * GIB}
kube_node_status_capacity{{node="{name}",resource="nvidia_com_gpu",unit="integer"}} {gpus}
kube_node_status_allocatable{{node="{name}",resource="cpu",unit="core"}} 32
kube_node_status_allocatable{{node="{name}",resource="memory",unit="byte"}} {128 * GIB}
kube_node_status_allocatable{{node="{name}",resource="nvidia_com_gpu",unit="integer"}} {gpus}
kube_node_status_condition{{condition="Ready",node="{name}",status="true"}} 1
kube_node_spec_unschedulable{{node="{name}"}} 0
kube_node_labels{{node="{name}",label_gpu_type="{gpu_type}",label_nvidia_com_gpu_product="{product}"}} 1
"""


def _pod(
    namespace: str,
    pod: str,
    *,
    created: int,
    gpus: int = 0,
    gpu_type: str = "",
    node: str = "",
    phase: str = "Pending",
    job: str = "",
) -> str:
    owner = job or pod
    lines = [
        f'kube_pod_info{{namespace="{namespace}",pod="{pod}",node="{node}",'
        f'created_by_kind="Job",created_by_name="{owner}"}} 1',
        f'kube_pod_status_phase{{namespace="{namespace}",pod="{pod}",phase="{phase}"}} 1',
        f'kube_pod_created{{namespace="{namespace}",pod="{pod}"}} {created}',
        f'kube_pod_container_resource_requests{{namespace="{namespace}",pod="{pod}",'
        f'container="main",node="{node}",resource="cpu",unit="core"}} 4',
        f'kube_pod_container_resource_requests{{namespace="{namespace}",pod="{pod}",'
        f'container="main",node="{node}",resource="memory",unit="byte"}} {8 * GIB}',
    ]
    if gpus:
        lines.append(
            f'kube_pod_container_resource_requests{{namespace="{namespace}",pod="{pod}",'
            f'container="main",node="{node}",resource="nvidia_com_gpu",unit="integer"}} {gpus}'
        )
    if gpu_type:
        lines.append(
            f'kube_pod_labels{{namespace="{namespace}",pod="{pod}",label_gpu_type="{gpu_type}"}} 1'
        )
    return "\n".join(lines) + "\n"


METRICS = (
    _node("node-h", "NVIDIA_H100_80GB_HBM3", "h100", 8)
    + _node("node-a", "NVIDIA_RTX_A6000", "a6000", 2)
    # Queued, deliberately listed out of creation order.
    + _pod("team-b", "late-h100-x1", created=1700000300, gpus=1, gpu_type="h100", job="late-h100")
    + _pod("me-dev", "mine-h100-x1", created=1700000100, gpus=2, gpu_type="h100", job="mine-h100")
    + _pod("team-c", "a6000-wait-x1", created=1700000200, gpus=1, gpu_type="a6000", job="a6000-wait")
    + _pod("team-c", "cpu-only-x1", created=1700000250, job="cpu-only")
    # Excluded: already bound to a node, running, or a system namespace.
    + _pod("team-b", "bound-x1", created=1700000000, gpus=1, gpu_type="h100", node="node-h")
    + _pod("team-b", "running-x1", created=1700000000, gpus=1, gpu_type="h100",
           node="node-h", phase="Running")
    + _pod("kube-system", "sys-x1", created=1700000000)
)


class PendingQueueSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = cluster_snapshot_from_metrics(METRICS)

    def test_queue_holds_unbound_pending_pods_oldest_first_by_model(self):
        queue = self.snapshot.queue
        self.assertEqual(
            [(item.workload_name, item.requested.gpu_model) for item in queue],
            [
                ("mine-h100", "H100"),
                ("a6000-wait", "A6000"),
                ("cpu-only", None),
                ("late-h100", "H100"),
            ],
        )
        self.assertEqual(queue[0].requested.gpu_count, 2)
        self.assertEqual(queue[0].node_name, "")
        self.assertEqual(queue[0].workload_kind, "Job")

    def test_queue_survives_service_roundtrip_and_old_services_report_none(self):
        payload = json.loads(json.dumps(snapshot_to_dict(self.snapshot)))
        self.assertEqual(snapshot_from_dict(payload).queue, self.snapshot.queue)
        payload.pop("queue")
        self.assertIsNone(snapshot_from_dict(payload).queue)


class _Collector:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def collect(self, force=False):
        return self.snapshot

    def last_snapshot(self):
        return self.snapshot

    def close(self):
        pass


def _screen_text(app) -> str:
    return "\n".join(
        "".join(segment.text for segment in strip)
        for strip in app.screen._compositor.render_strips()
    )


class PendingQueueViewTests(unittest.IsolatedAsyncioTestCase):
    async def test_p_swaps_namespace_area_for_queue_grouped_by_gpu(self):
        snapshot = cluster_snapshot_from_metrics(METRICS)
        app = FalconResourcesApp(
            _Collector(snapshot),
            refresh_seconds=999,
            initial_view="nodes",
            own_namespace="me-dev",
            history_loader=lambda: [],
        )
        async with app.run_test(size=(200, 50)) as pilot:
            await pilot.pause(0.5)
            self.assertIn("H100 6/8", _screen_text(app))
            self.assertNotIn("·2q", _screen_text(app))

            def pie_data():
                return (
                    app._namespace_categories(),
                    tuple(app._cluster_namespace_values(basis) for basis in ("cpu", "memory", "gpu", "vram")),
                    app._namespace_colors(),
                )

            pies_before = pie_data()
            self.assertTrue(any(pies_before[1]))
            await pilot.press("p")
            await pilot.pause(0.3)
            # Only the Namespace/Workload table changes; pies keep namespaces.
            self.assertEqual(pie_data(), pies_before)
            rows = app._allocation_rows()
            self.assertEqual(
                [(group.namespace, consumer.workload_name if consumer else None) for group, consumer in rows],
                [
                    ("A6000", None), ("A6000", "a6000-wait"),
                    ("H100", None), ("H100", "mine-h100"), ("H100", "late-h100"),
                    ("CPU", None), ("CPU", "cpu-only"),
                ],
            )
            text = _screen_text(app)
            self.assertIn("PENDING QUEUE BY GPU", text)
            self.assertNotIn("PENDING GPUS BY MODEL", text)
            self.assertIn("GPU BY NAMESPACE", text)
            self.assertIn("#1 ▸mine-h100", text)
            self.assertIn("#2 late-h100", text)
            self.assertIn("p Namespaces", text)

            await pilot.press("p")
            await pilot.pause(0.3)
            self.assertFalse(app.state.queue_view)
            self.assertIn("NAMESPACE / WORKLOAD ALLOCATION", _screen_text(app))
            self.assertNotIn("CPU", {group.namespace for group, _ in app._allocation_rows()})

    async def test_old_service_without_queue_says_unavailable(self):
        snapshot = cluster_snapshot_from_metrics(METRICS)
        from dataclasses import replace

        app = FalconResourcesApp(
            _Collector(replace(snapshot, queue=None)),
            refresh_seconds=999,
            initial_view="gpu-allocations",
            history_loader=lambda: [],
        )
        async with app.run_test(size=(200, 50)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("p")
            await pilot.pause(0.3)
            self.assertIn("Queue unavailable", _screen_text(app))


if __name__ == "__main__":
    unittest.main()
