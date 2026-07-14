import json
import io
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from rich.console import Console
from textual import events

from falcon.dashboard import (
    ABSOLUTE_MINIMUM_WIDTH,
    DASHBOARD_REFRESH_SECONDS,
    FalconDashboard,
    GpuSample,
    JobUsage,
    StreamingGpuSampler,
    UsageCollector,
    _job_sort_key,
    _metric_color,
    format_snapshot,
    parse_cpu_cores,
    parse_memory_gib,
    run_dashboard,
)


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
    )


def render_text(renderable, width):
    output = io.StringIO()
    console = Console(file=output, record=True, width=width, color_system=None)
    console.print(renderable)
    return console.export_text()


class FakeCollector:
    namespace = "test-dev"

    def collect(self):
        return [usage("a-very-long-training-job-name-that-truncates-cleanly")]


class CompletedCollector:
    namespace = "test-dev"

    def collect(self):
        row = usage("finished-job")
        row.status = "Succeeded"
        return [row]


class MetricTests(unittest.TestCase):
    def test_kubernetes_quantities_are_normalized(self):
        self.assertEqual(parse_cpu_cores("1500m"), 1.5)
        self.assertAlmostEqual(parse_cpu_cores("250000u"), 0.25)
        self.assertEqual(parse_memory_gib("4096Mi"), 4.0)
        self.assertEqual(parse_memory_gib("2Gi"), 2.0)
        self.assertAlmostEqual(parse_memory_gib("31782757990400m"), 29.6)
        self.assertAlmostEqual(parse_memory_gib("29600M"), 29600 * 1000 ** 2 / 1024 ** 3)

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
            if args[:2] == ["get", "pods"]:
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
            if args[:2] == ["get", "pods"]:
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
            if args[:2] == ["get", "pods"]:
                return '{"items": []}'
            return ""

        with patch("falcon.dashboard._kubectl", side_effect=fake_kubectl):
            UsageCollector("test-dev", {}, 0.02, job_filter="train-job").collect()
        self.assertIn("job-name=train-job", calls[0])
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
            return json.dumps({"items": [pod]}) if args[:2] == ["get", "pods"] else "active 1 2Gi\n"

        with patch("falcon.dashboard._kubectl", side_effect=fake_kubectl), patch(
            "falcon.dashboard._gpu_metrics", return_value=GpuSample(70, 2, 10, 1)
        ) as gpu:
            collector = UsageCollector("test-dev", {}, 0.1)
            collector.collect()
            collector.collect()
        self.assertEqual(calls.count(["get", "pods"]), 1)
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
        self.assertEqual(_metric_color(29.9), "#5cffb1")
        self.assertEqual(_metric_color(30), "#ffd166")
        self.assertEqual(_metric_color(60), "#ffd166")
        self.assertEqual(_metric_color(80), "#ffd166")
        self.assertEqual(_metric_color(80.1), "#ff5f6d")
        self.assertEqual(_metric_color(100), "#ff5f6d")

    def test_mixed_height_window_packs_completed_jobs_and_keeps_a_peek(self):
        completed = usage("done")
        completed.status = "Succeeded"
        rows = [usage("one"), usage("two"), completed]
        self.assertEqual(FalconDashboard._visible_window(rows, 0, 1, 14), (0, 2, 3))
        self.assertEqual(FalconDashboard._visible_window(rows, 0, 1, 17), (0, 3, 3))
        self.assertEqual(FalconDashboard._visible_window(rows, 2, 1, 14), (1, 3, 3))

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

    async def test_tiny_terminal_gets_a_resize_message(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(40, 14)) as pilot:
            await pilot.pause(0.3)
            rendered = str(app.query_one("#jobs").render())
            self.assertIn(f"Minimum: {ABSOLUTE_MINIMUM_WIDTH}", rendered)
            app.exit()

    async def test_compact_job_box_keeps_numbers_and_hides_bars(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(50, 24)) as pilot:
            await pilot.pause(0.3)
            rendered = render_text(app._card(app.rows[0], 0, 48, False), 48)
            self.assertIn("training-job", rendered)
            self.assertIn("77%", rendered)
            self.assertIn("38c/48c", rendered)
            self.assertNotIn("━", rendered)
            app.exit()

    async def test_succeeded_job_uses_a_compact_card(self):
        app = FalconDashboard(CompletedCollector(), 60)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause(0.3)
            rendered = render_text(app._card(app.rows[0], 0, 78, True), 78)
            self.assertEqual(len(rendered.splitlines()), 3)
            self.assertIn("Succeeded", rendered)
            self.assertNotIn("VRAM", rendered)
            app.exit()

    async def test_wide_job_box_contains_dynamic_bars_and_all_percentages(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(130, 30)) as pilot:
            await pilot.pause(0.3)
            rendered = render_text(app._card(app.rows[0], 0, 63, True), 63)
            self.assertIn("━", rendered)
            self.assertIn("VRAM", rendered)
            self.assertIn("CPU", rendered)
            self.assertIn("RAM", rendered)
            self.assertIn("78%", rendered)
            self.assertIn("76%", rendered)
            meter = app._meter("RAM", "38G/50G", 76, 59, "#ffd166", True)
            self.assertTrue(any(span.style == "#ffd166" and span.start == 6 for span in meter.spans))
            app.exit()

    async def test_enter_opens_nvitop_for_the_selected_job(self):
        app = FalconDashboard(FakeCollector(), 60)
        with patch.object(app, "suspend", return_value=nullcontext()), patch(
            "falcon.dashboard.open_job_top"
        ) as open_job_top:
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause(0.3)
                await pilot.press("enter")
                await pilot.pause()
                open_job_top.assert_called_once_with("test-dev", app.rows[0].job)
                app.exit()

    async def test_enter_does_nothing_for_succeeded_job(self):
        app = FalconDashboard(CompletedCollector(), 60)
        with patch("falcon.dashboard.open_job_top") as open_job_top:
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause(0.3)
                await pilot.press("enter")
                await pilot.pause()
                open_job_top.assert_not_called()
                app.exit()

    async def test_mouse_wheel_is_captured_and_navigates_jobs(self):
        app = FalconDashboard(FakeCollector(), 60)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause(0.3)
            app.rows = [usage("one"), usage("two")]
            event = events.MouseScrollDown(None, 0, 0, 0, 1, 0, False, False, False)
            app.on_mouse_scroll_down(event)
            self.assertEqual(app.selected, 1)
            self.assertTrue(event._stop_propagation)
            self.assertTrue(event._no_default_action)
            app.exit()


if __name__ == "__main__":
    unittest.main()
