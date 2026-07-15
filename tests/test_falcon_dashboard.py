import json
import io
import subprocess
import time
import unittest
from collections import deque
from contextlib import nullcontext
from unittest.mock import patch

from rich.console import Console
from textual import events

from falcon.dashboard import (
    ABSOLUTE_MINIMUM_WIDTH,
    DASHBOARD_REFRESH_SECONDS,
    FalconDashboard,
    GpuDevice,
    GpuSample,
    JobEvent,
    JobUsage,
    StreamingGpuSampler,
    UsageCollector,
    _active_pod,
    _job_sort_key,
    _metric_color,
    _parse_gpu_lines,
    _pod_state,
    format_snapshot,
    parse_cpu_cores,
    parse_memory_gib,
    run_dashboard,
)
from falcon.dashboard_ui import CleanupDialog, FilterDialog, KillDialog, MetricPoint, _scaled_history
from falcon.resources import NodeResources


def usage(job="training-job", at_risk=False):
    return JobUsage(
        job=job,
        status="Running",
        nodes="node8",
        gpu_type="2080ti",
        gpu_count=3,
        pod_count=1,
        gpu_util=77.0,
        gpu_ema=72.0,
        gpu_memory_used_gib=31.0,
        gpu_memory_total_gib=33.0,
        cpu_used=37.5,
        cpu_requested=48.0,
        memory_used_gib=38.0,
        memory_requested_gib=50.0,
        age="2h",
        at_risk=at_risk,
        uid=f"uid-{job}",
        active_pod=f"{job}-pod",
        active_pod_state="Running",
        command="python train.py --epochs 100",
        created_at="2026-01-01T00:00:00Z",
    )


def render_text(renderable, width):
    output = io.StringIO()
    console = Console(file=output, record=True, width=width, color_system=None)
    console.print(renderable)
    return console.export_text()


class FakeCollector:
    namespace = "test-dev"
    gpu_availability = {"2080ti": (3, 8), "a6000": (1, 4), "h100": (2, 4)}

    def collect(self):
        return [usage("a-very-long-training-job-name-that-truncates-cleanly")]


class CompletedCollector:
    namespace = "test-dev"

    def collect(self):
        row = usage("finished-job")
        row.status = "Succeeded"
        return [row]


class MetricTests(unittest.TestCase):
    def test_eviction_risk_uses_only_complete_rolling_average(self):
        collector = UsageCollector("test-dev", {}, 0.1, risk_average_samples=3)
        average = collector._update_risk_average("low", 10)
        self.assertFalse(collector._eviction_risk("low", average, 1, 30))
        average = collector._update_risk_average("low", 10)
        self.assertFalse(collector._eviction_risk("low", average, 1, 30))
        average = collector._update_risk_average("low", 10)
        self.assertTrue(collector._eviction_risk("low", average, 1, 30))

        stable = UsageCollector("test-dev", {}, 0.1, risk_average_samples=60)
        for _ in range(60):
            average = stable._update_risk_average("stable", 95)
        average = stable._update_risk_average("stable", 0)
        self.assertAlmostEqual(average, 59 * 95 / 60)
        self.assertFalse(stable._eviction_risk("stable", average, 1, 90))

    def test_gpu_availability_uses_same_metrics_source_as_jet_resources(self):
        nodes = [
            NodeResources(name="node5", gpu_total=4, gpu_used=3, gpu_product="NVIDIA GeForce RTX 2080 Ti"),
            NodeResources(name="node6", gpu_total=4, gpu_used=0, gpu_product="NVIDIA GeForce RTX 2080 Ti"),
            NodeResources(name="node9", gpu_total=4, gpu_used=0, gpu_product="NVIDIA GeForce RTX 2080 Ti", unschedulable=True),
            NodeResources(name="nodex1", gpu_total=8, gpu_used=2, gpu_product="NVIDIA H100 80GB HBM3"),
        ]
        collector = UsageCollector("test-dev", {}, 0.1, metrics_url="http://metrics")
        with patch("falcon.dashboard.fetch_nodes", return_value=nodes) as fetch, patch("falcon.dashboard._kubectl") as kubectl:
            collector._refresh_gpu_availability(100)
        fetch.assert_called_once_with("http://metrics", timeout=5)
        kubectl.assert_not_called()
        self.assertEqual(collector.gpu_availability["2080ti"], (5, 8))
        self.assertEqual(collector.gpu_availability["h100"], (6, 8))

    def test_gpu_availability_is_aggregated_cluster_wide_by_type(self):
        payload = {"items": [
            {
                "kind": "Node", "metadata": {"name": "node1", "labels": {"nvidia.com/gpu.product": "RTX-2080-Ti"}},
                "spec": {}, "status": {"allocatable": {"nvidia.com/gpu": "4"}},
            },
            {
                "kind": "Pod", "metadata": {"name": "other-user"}, "status": {"phase": "Running"},
                "spec": {"nodeName": "node1", "containers": [{"resources": {"limits": {"nvidia.com/gpu": "1"}}}]},
            },
        ]}
        collector = UsageCollector("test-dev", {}, 0.1)
        with patch("falcon.dashboard._kubectl", return_value=json.dumps(payload)):
            collector._refresh_gpu_availability(100)
        self.assertEqual(collector.gpu_availability["2080ti"], (3, 4))

    def test_detailed_gpu_sample_retains_per_device_telemetry(self):
        sample = _parse_gpu_lines([
            "0, NVIDIA H100 80GB HBM3, GPU-abc, 40960, 81920, 91, 67, 612.5, 0, 570.1"
        ])
        self.assertEqual(sample.gpu_count, 1)
        self.assertEqual(sample.devices[0].name, "NVIDIA H100 80GB HBM3")
        self.assertEqual(sample.devices[0].temperature_c, 67)
        self.assertEqual(sample.devices[0].ecc_errors, 0)

    def test_kubernetes_quantities_are_normalized(self):
        self.assertEqual(parse_cpu_cores("1500m"), 1.5)
        self.assertAlmostEqual(parse_cpu_cores("250000u"), 0.25)
        self.assertEqual(parse_memory_gib("4096Mi"), 4.0)
        self.assertEqual(parse_memory_gib("2Gi"), 2.0)
        self.assertAlmostEqual(parse_memory_gib("31782757990400m"), 29.6)
        self.assertAlmostEqual(parse_memory_gib("29600M"), 29600 * 1000 ** 2 / 1024 ** 3)

    def test_active_pod_prefers_newest_nonterminal_and_failure_reason(self):
        completed = {
            "metadata": {"name": "old", "creationTimestamp": "2026-01-01T00:00:02Z"},
            "status": {"phase": "Succeeded"},
        }
        active = {
            "metadata": {"name": "active", "creationTimestamp": "2026-01-01T00:00:01Z"},
            "status": {"phase": "Running", "containerStatuses": [
                {"state": {"running": {}}},
                {"state": {"waiting": {"reason": "CrashLoopBackOff"}}},
            ]},
        }
        self.assertEqual(_active_pod([completed, active])["metadata"]["name"], "active")
        self.assertEqual(_pod_state(active), "CrashLoopBackOff")

    def test_job_uid_and_job_level_status_come_from_job_resource(self):
        job = {
            "kind": "Job",
            "metadata": {"name": "train", "uid": "real-uid", "creationTimestamp": "2026-01-01T00:00:00Z"},
            "status": {"conditions": [{"type": "Complete", "status": "True"}]},
            "spec": {"completions": 1},
        }
        with patch("falcon.dashboard._kubectl", side_effect=lambda args, timeout=15: (
            json.dumps({"items": [job]}) if args[:2] == ["get", "jobs.batch,pods"] else ""
        )):
            row = UsageCollector("test-dev", {}, 0.1).collect()[0]
        self.assertEqual(row.uid, "real-uid")
        self.assertEqual(row.status, "Succeeded")
        self.assertEqual(row.active_pod_state, "No active pod")

    def test_events_are_consolidated_and_cached(self):
        row = usage("train")
        event = {
            "metadata": {"creationTimestamp": "2026-01-01T00:00:00Z"},
            "type": "Warning", "reason": "BackOff", "message": "retrying", "count": 12,
            "involvedObject": {"name": row.job},
        }
        with patch("falcon.dashboard._kubectl", return_value=json.dumps({"items": [event]})) as kubectl:
            collector = UsageCollector("test-dev", {}, 0.1)
            first = collector.events(row)
            second = collector.events(row)
        self.assertEqual(first[0].count, 12)
        self.assertEqual(first, second)
        self.assertEqual(kubectl.call_count, 2)

    def test_pods_are_aggregated_into_one_job_with_request_percentages(self):
        pods = {
            "items": [
                {
                    "metadata": {"name": "pod-a", "labels": {"job-name": "train"}, "creationTimestamp": "2026-01-01T00:00:00Z"},
                    "status": {"phase": "Running"},
                    "spec": {"nodeName": "node8", "nodeSelector": {"gpu-type": "2080ti"}, "containers": [{"resources": {"requests": {"cpu": "2", "memory": "4Gi"}, "limits": {"nvidia.com/gpu": "1"}}}]},
                },
                {
                    "metadata": {"name": "pod-b", "labels": {"job-name": "train"}, "creationTimestamp": "2026-01-01T00:00:01Z"},
                    "status": {"phase": "Running"},
                    "spec": {"nodeName": "node9", "nodeSelector": {"gpu-type": "2080ti"}, "containers": [{"resources": {"requests": {"cpu": "3", "memory": "6Gi"}, "limits": {"nvidia.com/gpu": "1"}}}]},
                },
            ]
        }

        def fake_kubectl(args, timeout=15):
            if args[:2] == ["get", "jobs.batch,pods"]:
                return json.dumps(pods)
            if args[:2] == ["top", "pods"]:
                return "pod-a 500m 1Gi\npod-b 1500m 3Gi\n"
            return None

        samples = {
            "pod-a": GpuSample(50, 2, 10, 1),
            "pod-b": GpuSample(100, 4, 10, 1),
        }
        with patch("falcon.dashboard._kubectl", side_effect=fake_kubectl), patch(
            "falcon.dashboard._gpu_metrics", side_effect=lambda namespace, pod: samples[pod]
        ):
            rows = UsageCollector("test-dev", {"2080ti": 30}, 0.25).collect()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.job, "train")
        self.assertEqual(row.pod_count, 2)
        self.assertEqual(row.gpu_count, 2)
        self.assertEqual(row.gpu_util, 75)
        self.assertEqual(row.gpu_memory_percent, 30)
        self.assertEqual(row.cpu_percent, 40)
        self.assertEqual(row.memory_percent, 40)

    def test_failed_retry_does_not_double_running_resource_requests(self):
        def pod(name, phase):
            return {
                "metadata": {"name": name, "labels": {"job-name": "retry-job"}},
                "status": {"phase": phase},
                "spec": {
                    "nodeName": "node7", "nodeSelector": {"gpu-type": "2080ti"},
                    "containers": [{"resources": {
                        "requests": {"cpu": "80", "memory": "100Gi"},
                        "limits": {"nvidia.com/gpu": "4"},
                    }}],
                },
            }

        def fake_kubectl(args, timeout=15):
            if args[:2] == ["get", "jobs.batch,pods"]:
                return json.dumps({"items": [pod("old-failed", "Failed"), pod("active", "Running")]})
            if args[:2] == ["top", "pods"]:
                return "active 79879m 61444Mi\n"
            return None

        with patch("falcon.dashboard._kubectl", side_effect=fake_kubectl), patch(
            "falcon.dashboard._gpu_metrics", return_value=GpuSample(80, 30, 44, 4)
        ):
            row = UsageCollector("test-dev", {"2080ti": 30}, 0.02).collect()[0]
        self.assertEqual(row.pod_count, 2)
        self.assertEqual(row.gpu_count, 4)
        self.assertEqual(row.cpu_requested, 80)
        self.assertEqual(row.memory_requested_gib, 100)
        self.assertAlmostEqual(row.cpu_percent, 99.84875)
        self.assertAlmostEqual(row.memory_percent, 60.00390625)

    def test_ema_uses_mean_warmup_then_slow_smoothing(self):
        collector = UsageCollector("test-dev", {"2080ti": 30}, 0.02)
        values = [collector._update_ema("train", value) for value in [100, 0, 100, 0, 100]]
        self.assertEqual(values[-1], 60)
        self.assertAlmostEqual(collector._update_ema("train", 0), 58.8)

    def test_agent_ema_averages_the_entire_requested_window(self):
        collector = UsageCollector("test-dev", {}, 0.02, ema_warmup_samples=6)
        values = [collector._update_ema("train", value) for value in [100, 0, 100, 0, 100, 0]]
        self.assertEqual(values[-1], 50)

    def test_individual_job_filter_is_sent_to_kubernetes(self):
        calls = []

        def fake_kubectl(args, timeout=15):
            calls.append(args)
            if args[:2] == ["get", "jobs.batch,pods"]:
                return '{"items": []}'
            return ""

        with patch("falcon.dashboard._kubectl", side_effect=fake_kubectl):
            UsageCollector("test-dev", {}, 0.02, job_filter="train-job").collect()
        inventory_call = next(call for call in calls if call[:2] == ["get", "jobs.batch,pods"])
        self.assertIn("job-name=train-job", inventory_call)
        self.assertIn("job-name=train-job", calls[1])

    def test_inventory_and_cpu_metrics_are_cached_while_gpu_stays_live(self):
        pod = {
            "metadata": {"name": "active", "labels": {"job-name": "train"}},
            "status": {"phase": "Running"},
            "spec": {"containers": [{"resources": {
                "requests": {"cpu": "2", "memory": "4Gi"},
                "limits": {"nvidia.com/gpu": "1"},
            }}]},
        }
        calls = []

        def fake_kubectl(args, timeout=15):
            calls.append(args[:2])
            return json.dumps({"items": [pod]}) if args[:2] == ["get", "jobs.batch,pods"] else "active 1 2Gi\n"

        with patch("falcon.dashboard._kubectl", side_effect=fake_kubectl), patch(
            "falcon.dashboard._gpu_metrics", return_value=GpuSample(70, 2, 10, 1)
        ) as gpu:
            collector = UsageCollector("test-dev", {}, 0.1)
            collector.collect()
            collector.collect()
        self.assertEqual(calls.count(["get", "jobs.batch,pods"]), 1)
        self.assertEqual(calls.count(["top", "pods"]), 1)
        self.assertEqual(gpu.call_count, 2)

    def test_streaming_gpu_sampler_uses_one_long_lived_exec(self):
        class Process:
            stdout = iter(["40, 1024, 11264\n", "80, 2048, 11264\n"])

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

        with patch("falcon.dashboard.subprocess.Popen", return_value=Process()) as popen:
            sampler = StreamingGpuSampler("test-dev")
            sample = sampler.samples({"pod-a": 2})["pod-a"]
            sampler.samples({"pod-a": 2})
            sampler.close()
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(sample.utilization, 60)
        self.assertEqual(sample.memory_used_gib, 3)


class DashboardTests(unittest.IsolatedAsyncioTestCase):
    def test_refresh_target_is_fast_and_fixed(self):
        self.assertEqual(DASHBOARD_REFRESH_SECONDS, 1.0)

    def test_gpu_live_utilization_colors_follow_dashboard_bands(self):
        self.assertEqual(_metric_color(29.9), "#55FF55")
        self.assertEqual(_metric_color(30), "#FFFF55")
        self.assertEqual(_metric_color(60), "#FFFF55")
        self.assertEqual(_metric_color(80), "#FF5555")
        self.assertEqual(_metric_color(80.1), "#FF5555")
        self.assertEqual(_metric_color(100), "#FF5555")

    def test_snapshot_never_exposes_namespace(self):
        rendered = format_snapshot([usage()], "secret-dev")
        structured = format_snapshot([usage()], "secret-dev", json_output=True)
        self.assertNotIn("secret-dev", rendered)
        self.assertNotIn("secret-dev", structured)

    def test_succeeded_jobs_sort_to_the_bottom(self):
        completed = usage("completed")
        completed.status = "Succeeded"
        pending = usage("pending")
        pending.status = "Pending"
        ordered = sorted([completed, pending, usage("running")], key=_job_sort_key)
        self.assertEqual(ordered[-1].status, "Succeeded")

    def test_agent_snapshot_is_compact_and_ansi_free(self):
        rendered = format_snapshot([usage()], "test-dev")
        self.assertEqual(len(rendered.splitlines()), 2)
        self.assertIn("job=training-job", rendered)
        self.assertIn("cpu=38c/48c(78%)", rendered)
        self.assertNotIn("\x1b", rendered)

    def test_json_snapshot_is_structured(self):
        rendered = json.loads(format_snapshot([usage()], "test-dev", json_output=True, sample_count=5))
        self.assertEqual(rendered["job_count"], 1)
        self.assertEqual(rendered["sample_count"], 5)
        self.assertEqual(rendered["jobs"][0]["cpu_percent"], 78.125)

    def test_non_tty_agent_snapshot_collects_five_frames(self):
        config = {"cluster": {"namespace": "test-dev"}, "presets": {}, "dashboard": {"ema_alpha": 0.02}}
        with patch("falcon.dashboard.UsageCollector") as collector_type, patch(
            "falcon.dashboard.sys.stdout.isatty", return_value=False
        ), patch("falcon.dashboard.time.sleep"), patch("builtins.print"):
            collector_type.return_value.collect.return_value = [usage()]
            run_dashboard(config)
        self.assertEqual(collector_type.return_value.collect.call_count, 5)

    async def test_screen_uses_black_background_and_required_panes(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(40, 14)) as pilot:
            await pilot.pause(0.3)
            self.assertIn("#000000", app.CSS)
            self.assertIsNotNone(app.query_one("#jobs-pane"))
            self.assertIsNotNone(app.query_one("#resources-pane"))
            self.assertIsNotNone(app.query_one("#events-pane"))
            app.exit()

    async def test_jobs_table_is_responsive_and_keeps_job_state(self):
        scenarios = (
            ((80, 32), FakeCollector(), ("MARK", "NAME", "STATUS", "ACTIVE POD", "AGE"), ("GPU TYPE",)),
            ((130, 32), FakeCollector(), ("NODE", "GPU TYPE"), ()),
            ((80, 32), CompletedCollector(), ("Succeeded",), ("VRAM",)),
        )
        for size, collector, expected, absent in scenarios:
            with self.subTest(size=size, collector=type(collector).__name__):
                app = FalconDashboard(collector, 60)
                async with app.run_test(size=size) as pilot:
                    await pilot.pause(0.3)
                    rendered = render_text(app.query_one("#jobs-pane").content, size[0] - 2)
                    for label in expected:
                        self.assertIn(label, rendered)
                    for label in absent:
                        self.assertNotIn(label, rendered)
                    app.exit()

    async def test_resource_pane_contains_sparklines_and_all_metrics(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(130, 32)) as pilot:
            await pilot.pause(0.3)
            rendered = render_text(app.query_one("#resources-pane").content, 128)
            self.assertIn("GPU", rendered)
            self.assertIn("VRAM", rendered)
            self.assertIn("CPU", rendered)
            self.assertIn("RAM", rendered)
            self.assertTrue(any(character in rendered for character in "▁▂▃▄▅▆▇█"))
            app.exit()

    async def test_enter_expands_and_escape_restores_focused_pane(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(80, 32)) as pilot:
            await pilot.pause(0.3)
            await pilot.press("enter")
            self.assertEqual(app.state.expanded_pane, "jobs")
            self.assertFalse(app.query_one("#resources-pane").display)
            await pilot.press("escape")
            self.assertIsNone(app.state.expanded_pane)
            self.assertTrue(app.query_one("#resources-pane").display)
            app.exit()

    async def test_minimum_terminal_boundary_preserves_a_usable_layout(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(80, 29)) as pilot:
            await pilot.pause(0.3)
            self.assertTrue(app.query_one("#resize-message").display)
            self.assertFalse(app.query_one("#jobs-pane").display)
            self.assertIn("80×30", render_text(app.query_one("#resize-message").content, 79))
            app.exit()

        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause(0.3)
            app.rows = [usage(f"job-{index}") for index in range(6)]
            app.job_events = []
            app._filter_rows()
            app._render_all()
            self.assertGreaterEqual(app._visible_job_count(), 4)
            self.assertEqual(app.query_one("#events-pane").styles.height.value, 7)
            self.assertEqual(app.query_one("#summary").styles.height.value, 2)
            app.exit()

    async def test_summary_shows_free_over_total_gpu_state(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.3)
            rendered = render_text(app.query_one("#summary").content, 138)
            self.assertIn("RESOURCES AVAILABLE", rendered)
            self.assertIn("2080Ti 3/8", rendered)
            self.assertIn("A6000 1/4", rendered)
            self.assertIn("H100 2/4", rendered)
            self.assertEqual(rendered.count("FREE"), 0)
            self.assertNotIn("GPU 77%", rendered)
            app.exit()

    async def test_eviction_risk_status_is_not_clipped(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause(0.3)
            app.rows = [usage("compact-name", at_risk=True)]
            app._filter_rows()
            app._render_jobs()
            rendered = render_text(app.query_one("#jobs-pane").content, 98)
            self.assertIn("Eviction risk", rendered)
            app.exit()

    async def test_events_viewport_renders_multiple_events(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause(0.3)
            app.job_events = [JobEvent(
                timestamp=f"2026-01-01T00:00:0{index}Z", event_type="Normal",
                reason="Test", message=f"event-{index}",
            ) for index in range(6)]
            app._render_events()
            rendered = render_text(app.query_one("#events-pane").content, 98)
            self.assertGreaterEqual(app._visible_event_count(), 5)
            for index in range(1, 6):
                self.assertIn(f"event-{index}", rendered)
            app.exit()

    async def test_selected_job_is_focusable_expandable_and_uses_compact_gpu_name(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.3)
            app.rows[0].command = "python train.py " + " ".join(f"--option-{index}=value" for index in range(100))
            await pilot.press("tab")
            self.assertEqual(app.state.focused_pane, "selected")
            await pilot.press("enter")
            self.assertEqual(app.state.expanded_pane, "selected")
            rendered = render_text(app.query_one("#selected-pane").content, 118)
            self.assertIn("JOB DETAILS", rendered)
            self.assertIn("GPU allocation", rendered)
            self.assertIn("2080tix3", rendered)
            self.assertIn("COMMAND", rendered)
            self.assertEqual(app.query_one("#selected-pane").border_subtitle, "")
            self.assertEqual(app.state.selected_command_scroll_offset, 0)
            await pilot.press("down")
            self.assertEqual(app.state.selected_command_scroll_offset, 1)
            await pilot.press("end")
            self.assertGreater(app.state.selected_command_scroll_offset, 1)
            await pilot.press("home")
            self.assertEqual(app.state.selected_command_scroll_offset, 0)
            app.exit()

    async def test_k_kill_confirms_and_executes_configured_namespace_delete(self):
        app = FalconDashboard(FakeCollector(), 60)
        completed = subprocess.CompletedProcess([], 0, "", "")
        with patch("falcon.dashboard_ui.subprocess.run", return_value=completed) as run:
            async with app.run_test(size=(100, 32)) as pilot:
                await pilot.pause(0.3)
                await pilot.press("k")
                self.assertIsInstance(app.screen, KillDialog)
                await pilot.press("enter")
                await pilot.pause(0.5)
                self.assertTrue(run.called)
                command = run.call_args.args[0]
                self.assertEqual(command[:3], ["kubectl", "delete", "job"])
                self.assertEqual(command[-2:], ["--namespace", "test-dev"])
                app.exit()

    async def test_three_job_delete_accepts_y_confirmation(self):
        app = FalconDashboard(FakeCollector(), 60)
        completed = subprocess.CompletedProcess([], 0, "", "")
        with patch("falcon.dashboard_ui.subprocess.run", return_value=completed) as run:
            async with app.run_test(size=(100, 32)) as pilot:
                await pilot.pause(0.3)
                app.rows = [usage(f"delete-{index}") for index in range(3)]
                app._filter_rows()
                app.state.marked_job_uids = {row.uid for row in app.rows}
                app._render_all()
                await pilot.press("k")
                self.assertIsInstance(app.screen, KillDialog)
                await pilot.press("enter")
                self.assertEqual(app.screen.stage, 1)
                await pilot.press("y")
                await pilot.pause(0.5)
                self.assertNotIsInstance(app.screen, KillDialog)
                self.assertEqual(run.call_count, 3)
                app.exit()

    async def test_c_cleanup_deletes_only_succeeded_jobs_after_confirmation(self):
        app = FalconDashboard(FakeCollector(), 60)
        completed = subprocess.CompletedProcess([], 0, "", "")
        with patch("falcon.dashboard_ui.subprocess.run", return_value=completed) as run:
            async with app.run_test(size=(100, 32)) as pilot:
                await pilot.pause(0.3)
                succeeded = usage("done")
                succeeded.status = "Succeeded"
                failed = usage("failed")
                failed.status = "Failed"
                running = usage("running")
                app.rows = [succeeded, failed, running]
                app._filter_rows()
                app._render_all()
                await pilot.press("c")
                self.assertIsInstance(app.screen, CleanupDialog)
                self.assertEqual([row.job for row in app.screen.rows], ["done"])
                await pilot.press("enter")
                await pilot.pause(0.5)
                self.assertEqual(run.call_count, 1)
                self.assertIn("done", run.call_args.args[0])
                self.assertNotIn("failed", run.call_args.args[0])
                app.exit()

    def test_history_renderer_preserves_sampling_zoom_and_terminal_capabilities(self):
        values = [10, 20, 30, 40]

        def bottom_bar_count(samples_per_bar):
            rendered = render_text(_scaled_history(values, 10, 2, samples_per_bar), 10).rstrip("\n")
            return sum(character in "▁▂▃▄▅▆▇█" for character in rendered.splitlines()[-1])

        self.assertEqual((bottom_bar_count(1), bottom_bar_count(2), bottom_bar_count(4)), (4, 2, 1))
        cpu = render_text(
            _scaled_history([10] * 5 + [80] * 5, 10, 1, samples_per_bar=1, sample_period=5), 10
        ).rstrip("\n")
        self.assertEqual(len(cpu), 10)
        self.assertEqual(len(set(cpu[:5])), 1)
        self.assertEqual(len(set(cpu[5:])), 1)
        self.assertNotEqual(cpu[0], cpu[-1])

        app = FalconDashboard(FakeCollector(), 60)
        app._sync_available = False
        self.assertEqual((app._resource_history_height("wide"), app._resource_history_height("compact")), (1, 1))
        app._sync_available = True
        self.assertEqual(
            tuple(app._resource_history_height(layout) for layout in ("wide", "compact", "collapsed")),
            (3, 4, 4),
        )

    async def test_expanded_resources_are_responsive_aligned_and_non_wrapping(self):
        widths = {}
        scenarios = (
            ((80, 42), "compact"),
            ((110, 42), "compact"),
            ((124, 42), "compact"),
            ((140, 48), "wide"),
            ((180, 48), "wide"),
            ((120, 32), "collapsed"),
        )
        for size, expected_layout in scenarios:
            with self.subTest(size=size, layout=expected_layout):
                app = FalconDashboard(FakeCollector(), 60)
                async with app.run_test(size=size) as pilot:
                    await pilot.pause(0.3)
                    app.rows[0].gpu_devices = [GpuDevice(
                        index=0, name="NVIDIA RTX 2080 Ti", uuid="GPU-test",
                        memory_used_gib=9.5, memory_total_gib=11.0,
                        utilization=77, temperature_c=68, power_w=220,
                        ecc_errors=0, driver_version="570.1",
                    )]
                    app._record_history()
                    app.action_focus_resources()
                    app.action_expand()
                    layout = app._resource_layout()
                    self.assertEqual(layout, expected_layout)
                    widths[size] = app._resource_history_width(layout)
                    pane = app.query_one("#resources-pane")
                    self.assertEqual(pane.border_subtitle, "")
                    rendered = render_text(pane.content, size[0] - 2)
                    for label in ("SELECTED JOB", "GPU", "VRAM", "CPU", "RAM", "GPU DEVICES"):
                        self.assertIn(label, rendered)
                    self.assertNotIn("collecting", rendered)
                    self.assertNotIn("History", rendered)

                    if layout == "wide":
                        for label in ("Now", "Average", "Peak", "RTX 2080 Ti", "TEMP", "POWER"):
                            self.assertIn(label, rendered)
                    else:
                        self.assertGreaterEqual(rendered.count("Now"), 4)
                        self.assertGreaterEqual(rendered.count("Used"), 4)
                        self.assertGreaterEqual(rendered.count("Average"), 4)
                        self.assertTrue(any(character in rendered for character in "▁▂▃▄▅▆▇█"))
                        lines = rendered.splitlines()
                        top = next(index for index, line in enumerate(lines) if " GPU " in line and " VRAM " in line)
                        bottom = next(index for index in range(top + 1, len(lines)) if lines[index].startswith("└"))
                        self.assertEqual(bottom - top + 1, 6)
                        stat_lines = [next(line for line in lines if label in line) for label in ("Now", "Used", "Average", "Peak")]
                        self.assertEqual(
                            len({line.index(label) for line, label in zip(stat_lines, ("Now", "Used", "Average", "Peak"))}),
                            1,
                        )
                        graph_line = next(
                            line for line in lines
                            if "Peak" in line and "77%" in line and any(block in line for block in "▁▂▃▄▅▆▇█")
                        )
                        right_border = graph_line.index("│", 1)
                        last_block = max(
                            index for index, character in enumerate(graph_line[:right_border])
                            if character in "▁▂▃▄▅▆▇█"
                        )
                        self.assertLessEqual(right_border - last_block, 2)
                    app.exit()

        self.assertLess(widths[(80, 42)], widths[(124, 42)])
        self.assertLess(widths[(140, 48)], widths[(180, 48)])

    async def test_expanded_resource_state_warnings_and_controls(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(110, 42)) as pilot:
            await pilot.pause(0.3)
            row = app.rows[0]
            row.gpu_util = 95
            row.gpu_risk_average = 20
            row.gpu_risk_threshold = 30
            row.at_risk = False
            app._record_history()
            app.action_focus_resources()
            app.action_expand()
            pane = app.query_one("#resources-pane")
            rendered = render_text(pane.content, 108)
            self.assertNotIn("! High", rendered)
            self.assertNotIn("! EVICTION RISK", rendered)

            row.at_risk = True
            app._render_resources()
            rendered = render_text(pane.content, 108)
            self.assertIn("! EVICTION RISK", rendered)
            self.assertIn("60s average 20.0% < 30%", rendered)
            self.assertNotIn("! High", rendered)

            await pilot.press("R")
            await pilot.press("Z")
            self.assertEqual((app.state.resource_range_samples, app.state.resource_zoom), (300, 2))
            footer = render_text(app.query_one("#falcon-footer").content, 108)
            self.assertIn("R Range", footer)
            self.assertIn("Zoom 50%", footer)
            await pilot.press("plus")
            self.assertEqual(app.state.resource_zoom, 1)
            self.assertIn("Zoom 100%", render_text(app.query_one("#falcon-footer").content, 108))
            await pilot.press("minus")
            self.assertEqual(app.state.resource_zoom, 2)

            row.at_risk = False
            row.gpu_metrics_available = False
            row.cpu_metrics_available = False
            app.histories[row.uid] = deque([MetricPoint(time.time(), None, None, None, None)])
            app._render_resources()
            rendered = render_text(pane.content, 108)
            self.assertIn("—", rendered)
            self.assertNotIn("0M / 0M", rendered)
            self.assertNotIn("0m / 0m", rendered)
            app.exit()

    async def test_filter_dialog_arrows_change_and_apply_selected_filter(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause(0.3)
            await pilot.press("f")
            self.assertIsInstance(app.screen, FilterDialog)
            await pilot.press("right")
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(app.state.filters["status"], "Running")
            app.exit()

    async def test_tab_and_shift_tab_cycle_normal_and_expanded_panes(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause(0.3)
            self.assertEqual(app.state.focused_pane, "jobs")
            await pilot.press("tab")
            self.assertEqual(app.state.focused_pane, "selected")
            await pilot.press("tab")
            self.assertEqual(app.state.focused_pane, "resources")
            await pilot.press("tab")
            self.assertEqual(app.state.focused_pane, "events")
            await pilot.press("shift+tab")
            self.assertEqual(app.state.focused_pane, "resources")

            app.action_focus_jobs()
            await pilot.press("enter")
            self.assertEqual(app.state.expanded_pane, "jobs")
            await pilot.press("tab")
            self.assertEqual(app.state.focused_pane, "selected")
            self.assertEqual(app.state.expanded_pane, "selected")
            self.assertTrue(app.query_one("#selected-pane").display)
            self.assertFalse(app.query_one("#jobs-pane").display)
            await pilot.press("tab")
            self.assertEqual(app.state.expanded_pane, "resources")
            await pilot.press("shift+tab")
            self.assertEqual(app.state.expanded_pane, "selected")
            app.exit()

    async def test_expanded_mouse_scroll_does_not_switch_jobs(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.3)
            app.rows = [usage(f"job-{index}") for index in range(12)]
            app.rows[0].command = "python train.py " + " ".join(
                f"--option-{index}=value" for index in range(100)
            )
            app._filter_rows()
            app.state.cursor_job_uid = app.filtered_rows[0].uid
            cursor_uid = app.state.cursor_job_uid
            app.action_expand()
            app.scroll_focused(1, "jobs-pane")
            self.assertEqual(app.state.cursor_job_uid, cursor_uid)
            self.assertEqual(app.state.jobs_scroll_offset, 1)
            await pilot.press("tab")
            self.assertEqual(app.state.expanded_pane, "selected")
            app.scroll_focused(1, "selected-pane")
            self.assertEqual(app.state.cursor_job_uid, cursor_uid)
            self.assertEqual(app.state.selected_command_scroll_offset, 1)
            app.exit()

    async def test_jobs_footer_exposes_sort_shortcut(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.3)
            footer = render_text(app.query_one("#falcon-footer").content, 118)
            self.assertIn("s Sort", footer)
            app.exit()

    async def test_filter_and_kill_dialogs_are_keyboard_accessible(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause(0.3)
            await pilot.press("f")
            self.assertIsInstance(app.screen, FilterDialog)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("f9")
            await pilot.pause()
            self.assertIsInstance(app.screen, KillDialog)
            self.assertEqual(app.screen.rows[0].uid, app.state.cursor_job_uid)
            await pilot.press("escape")
            await pilot.pause()
            app.exit()

    async def test_marks_are_uid_backed_and_independent_of_cursor(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(80, 32)) as pilot:
            await pilot.pause(0.3)
            app.rows = [usage("one"), usage("two")]
            app._filter_rows()
            app.action_toggle_mark()
            marked = app.state.cursor_job_uid
            app.scroll_focused(1, "jobs-pane")
            self.assertEqual(app.selected, 1)
            self.assertIn(marked, app.state.marked_job_uids)
            app.exit()


if __name__ == "__main__":
    unittest.main()
