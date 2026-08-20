from __future__ import annotations

import hashlib
import json
import os
import re
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rich.color import ColorSystem
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.strip import Strip

# Golden rendering is intentionally truecolor and independent of the parent
# test runner's NO_COLOR setting.
os.environ.pop("NO_COLOR", None)

from falcon.dashboard import (
    DemoUsageCollector,
    GpuDevice,
    GpuProcess,
    JobEvent,
    StreamingGpuSampler,
    _parse_gpu_process_utilization,
)
from falcon.dashboard_ui import (
    DashboardPane,
    DashboardPaneContent,
    FalconDashboard,
    MetricPoint,
    _restart_job_manifest,
)
from falcon.demo import DEMO_NOW, DemoCollector, demo_cluster_snapshot
from falcon.resources_charts import (
    GPUHistoryPoint,
    render_gpu_history,
    render_namespace_pie,
)
from falcon.resources_telemetry import GpuTelemetrySnapshot
from falcon.resources_ui import (
    CONSUMER_SORTS,
    RESOURCE_VIEWS,
    FalconResourcesApp,
    ResourcesPane,
    _gpu_headroom_color,
    _gpu_totals,
    _request_pressure_color,
    _resource_headroom_color,
    _short_cpu,
)
from falcon.theme import (
    CYAN,
    GREEN,
    MUTED,
    PALETTE,
    RED,
    SELECTION,
    YELLOW,
)

DIMENSIONS = (
    (60, 18),
    (79, 21),
    (80, 22),
    (80, 24),
    (80, 30),
    (90, 22),
    (100, 24),
    (100, 30),
    (120, 30),
    (140, 32),
    (160, 40),
    (200, 50),
)


def _golden_digest(svg: str) -> str:
    normalized = re.sub(r"terminal-\d+", "terminal-ID", svg)
    normalized = re.sub(
        r"\d{2}:\d{2}:\d{2}", "12:00:00", normalized
    ).replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _golden_manifest():
    return json.loads(
        (Path(__file__).parent / "snapshots" / "tui_manifest.json").read_text(
            encoding="utf-8"
        )
    )


def _seed_resources_history(app: FalconResourcesApp) -> None:
    namespaces = sorted(
        {
            consumer.namespace
            for node in app._gpu_nodes()
            if node.ready and node.schedulable
            for consumer in node.consumers
            if consumer.requested.gpu_count > 0
        }
    )
    app.history = []
    for index in range(16):
        values = {
                namespace: min(
                    8,
                    (index // (namespace_index + 2) + namespace_index)
                    % 9,
                )
                for namespace_index, namespace in enumerate(namespaces)
            }
        app.history.append(
            GPUHistoryPoint.from_mapping(
                DEMO_NOW - (15 - index) * 5 * 60,
                values,
                {namespace: value * 80 for namespace, value in values.items()},
            )
        )
    app._render_all()


def _rgb_values(value: str) -> tuple[int, int, int]:
    value = value.removeprefix("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


class DashboardInteractionTests(unittest.IsolatedAsyncioTestCase):
    def test_apps_force_direct_truecolor_even_when_tmux_advertises_256(self) -> None:
        """Explicit Falcon hex colours must reach the terminal unchanged."""

        previous_term = os.environ.get("TERM")
        previous_colorterm = os.environ.get("COLORTERM")
        previous_color = os.environ.get("FALCON_COLOR")
        try:
            os.environ["TERM"] = "screen-256color"
            os.environ.pop("COLORTERM", None)
            os.environ.pop("FALCON_COLOR", None)
            dashboard = FalconDashboard(DemoUsageCollector("mixed"))
            resources = FalconResourcesApp(DemoCollector())
            for app in (dashboard, resources):
                self.assertEqual(app.color_mode, "truecolor")
                self.assertEqual(app.console._color_system, ColorSystem.TRUECOLOR)
                rendered = Strip(
                    [
                        Segment(
                            "X",
                            Style(
                                color=PALETTE.accent,
                                bgcolor=PALETTE.success,
                            ),
                        )
                    ],
                    cell_length=1,
                ).render(app.console)
                accent = ";".join(map(str, _rgb_values(PALETTE.accent)))
                success = ";".join(map(str, _rgb_values(PALETTE.success)))
                self.assertIn(
                    f"\x1b[38;2;{accent};48;2;{success}m",
                    rendered,
                )
        finally:
            if previous_term is None:
                os.environ.pop("TERM", None)
            else:
                os.environ["TERM"] = previous_term
            if previous_colorterm is None:
                os.environ.pop("COLORTERM", None)
            else:
                os.environ["COLORTERM"] = previous_colorterm
            if previous_color is None:
                os.environ.pop("FALCON_COLOR", None)
            else:
                os.environ["FALCON_COLOR"] = previous_color

    def test_lower_colour_fallbacks_are_explicit(self) -> None:
        for mode, system in (
            ("256", ColorSystem.EIGHT_BIT),
            ("16", ColorSystem.STANDARD),
        ):
            with self.subTest(mode=mode):
                app = FalconDashboard(DemoUsageCollector("mixed"), color_mode=mode)
                self.assertEqual(app.color_mode, mode)
                self.assertEqual(app.console._color_system, system)
                rendered = Strip(
                    [Segment("X", Style(color=PALETTE.accent))], cell_length=1
                ).render(app.console)
                accent = ";".join(map(str, _rgb_values(PALETTE.accent)))
                self.assertNotIn(f"\x1b[38;2;{accent}m", rendered)

    async def test_dashboard_gpu_summary_includes_shared_model_order(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(160, 32)) as pilot:
            await pilot.pause(0.5)
            app.state.gpu_availability = {
                "h100": (1, 4),
                "pro6000": (1, 2),
                "a6000": (2, 4),
                "2080ti": (6, 8),
            }
            app._render_summary()
            summary = app.query_one("#summary").render().plain
            positions = [
                summary.index(model)
                for model in ("2080Ti", "A6000", "PRO6000", "H100")
            ]
            self.assertEqual(positions, sorted(positions))
            self.assertIn("PRO6000 1/2", summary)

    def test_truecolor_clears_rich_style_quantization_cache(self) -> None:
        fallback = FalconDashboard(DemoUsageCollector("mixed"), color_mode="256")
        fallback.console._render_buffer(
            list(fallback.console.render(Text("X", style=PALETTE.accent)))
        )
        truecolor = FalconDashboard(
            DemoUsageCollector("mixed"), color_mode="truecolor"
        )
        rendered = truecolor.console._render_buffer(
            list(truecolor.console.render(Text("X", style=PALETTE.accent)))
        )
        accent = ";".join(map(str, _rgb_values(PALETTE.accent)))
        self.assertIn(f"\x1b[38;2;{accent}m", rendered)
        self.assertNotIn("\x1b[38;5;", rendered)

    def test_colour_mode_log_reports_framework_and_rgb_triplet(self) -> None:
        with self.assertLogs("falcon.tui.color", level="INFO") as captured:
            FalconDashboard(DemoUsageCollector("mixed"), color_mode="truecolor")
        self.assertIn("colour mode: truecolor", captured.output[0])
        self.assertIn("framework=", captured.output[0])
        self.assertIn(
            f"{PALETTE.accent}=rgb{_rgb_values(PALETTE.accent)}",
            captured.output[0],
        )

    async def test_completed_job_does_not_create_partial_resource_history(self) -> None:
        collector = DemoUsageCollector("mixed")
        app = FalconDashboard(collector, refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            row = next(item for item in app.rows if item.status == "Succeeded")
            app.histories.pop(row.uid, None)
            terminal_with_stale_gpu = replace(
                row,
                gpu_util=65.0,
                gpu_metrics_available=True,
                gpu_memory_used_gib=24.0,
                gpu_memory_total_gib=80.0,
            )
            app.rows = [terminal_with_stale_gpu]
            app._record_history()

            self.assertEqual(list(app.histories[row.uid]), [])
            self.assertIsNone(terminal_with_stale_gpu.cpu_percent)
            self.assertIsNone(terminal_with_stale_gpu.memory_percent)
            self.assertIsNotNone(terminal_with_stale_gpu.gpu_util)
            self.assertIsNotNone(terminal_with_stale_gpu.gpu_memory_percent)

    async def test_completed_job_history_changes_the_rendered_gpu_value(self) -> None:
        collector = DemoUsageCollector("mixed")
        app = FalconDashboard(collector, refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            row = next(
                item
                for item in app.rows
                if item.status == "Running" and item.gpu_metrics_available
            )
            app.histories[row.uid].clear()
            gpu_capacity = row.gpu_allocated_count or row.gpu_count
            cpu_capacity = row.cpu_allocated or row.cpu_requested
            ram_capacity = row.memory_allocated_gib or row.memory_requested_gib

            def usage_sample(percent: float):
                fraction = percent / 100
                return replace(
                    row,
                    gpu_util=percent,
                    gpu_memory_used_gib=row.gpu_memory_total_gib * fraction,
                    cpu_used=cpu_capacity * fraction,
                    memory_used_gib=ram_capacity * fraction,
                )

            samples = (
                usage_sample(10.0),
                usage_sample(80.0),
                replace(
                    usage_sample(80.0),
                    status="Succeeded",
                    gpu_memory_used_gib=0.0,
                    gpu_memory_total_gib=0.0,
                    cpu_used=0.0,
                    cpu_allocated=0.0,
                    memory_used_gib=0.0,
                    memory_allocated_gib=0.0,
                    gpu_allocated_count=0,
                ),
            )
            for sample in samples:
                app.rows = [sample]
                app._record_history()

            completed = samples[-1]
            app.rows = [completed]
            app.filtered_rows = [completed]
            app.state.cursor_job_uid = completed.uid
            app.state.focused_pane = "resources"
            app.state.resource_scroll_offset = 0
            app._render_all()

            history = list(app.histories[row.uid])
            self.assertEqual([point.gpu for point in history], [10.0, 80.0])
            latest_absolutes = [
                metric["absolute"]
                for metric in app._resource_metrics(completed, app._history_slice(row.uid))
            ]
            self.assertEqual(
                latest_absolutes,
                [
                    app._absolute_metric(80.0, gpu_capacity, "GPU"),
                    app._absolute_metric(80.0, row.gpu_memory_total_gib, "GiB"),
                    app._absolute_metric(80.0, cpu_capacity, "vCPU"),
                    app._absolute_metric(80.0, ram_capacity, "GiB"),
                ],
            )
            terminal_metrics = app._resource_metrics(
                completed, app._history_slice(row.uid)
            )
            self.assertTrue(all(metric["terminal"] for metric in terminal_metrics))
            self.assertTrue(
                all(
                    app._resource_metric_color(metric, 80.0) == MUTED
                    for metric in terminal_metrics
                )
            )
            cpu_cell = app._metric_cell(
                "CPU", 80.0, [80.0], "1.0 / 1.0 vCPU", terminal=True
            )
            cpu_offset = cpu_cell.plain.index("80%")
            self.assertEqual(
                next(
                    span.style
                    for span in cpu_cell.spans
                    if span.start <= cpu_offset < span.end
                ),
                MUTED,
            )
            self.assertIn("80%", app.export_screenshot(simplify=True))
            app.action_history_left()
            self.assertEqual(app._history_slice(row.uid)[-1].gpu, 10.0)
            earlier_absolutes = [
                metric["absolute"]
                for metric in app._resource_metrics(completed, app._history_slice(row.uid))
            ]
            self.assertEqual(
                earlier_absolutes,
                [
                    app._absolute_metric(10.0, gpu_capacity, "GPU"),
                    app._absolute_metric(10.0, row.gpu_memory_total_gib, "GiB"),
                    app._absolute_metric(10.0, cpu_capacity, "vCPU"),
                    app._absolute_metric(10.0, ram_capacity, "GiB"),
                ],
            )
            self.assertNotEqual(earlier_absolutes, latest_absolutes)
            earlier = app.export_screenshot(simplify=True)
            self.assertIn("10%", earlier)
            self.assertNotIn("80%", earlier)

    async def test_short_terminal_temporarily_hides_events(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(80, 22)) as pilot:
            await pilot.pause(0.5)
            self.assertIn("events", app._responsive_hidden_panes)
            self.assertNotIn("events", app._visible_panes())
            await pilot.resize_terminal(80, 30)
            await pilot.pause()
            self.assertNotIn("events", app._responsive_hidden_panes)
            self.assertIn("events", app._visible_panes())

    async def test_events_keyboard_scroll_and_follow_lifecycle(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("3")
            self.assertEqual(app.state.focused_pane, "events")
            maximum = max(0, len(app.job_events) - app._visible_event_count())
            self.assertEqual(app.state.events_scroll_offset, maximum)

            await pilot.press("home")
            self.assertEqual(app.state.events_scroll_offset, 0)
            self.assertFalse(app.state.events_auto_follow)
            await pilot.press("down")
            self.assertEqual(app.state.events_scroll_offset, 1)
            await pilot.press("pagedown")
            self.assertGreater(app.state.events_scroll_offset, 1)

            await pilot.press("end")
            self.assertTrue(app.state.events_auto_follow)
            self.assertEqual(app.state.events_scroll_offset, maximum)

    async def test_new_events_do_not_yank_a_manually_scrolled_view(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("3", "home", "pagedown")
            before = app.state.events_scroll_offset
            app.job_events.append(
                JobEvent(
                    "2026-01-15T12:00:00Z",
                    "Warning",
                    "Injected",
                    "a deterministic new event",
                )
            )
            app._render_events()
            self.assertEqual(app.state.events_scroll_offset, before)
            await pilot.press("end")
            self.assertTrue(app.state.events_auto_follow)

    async def test_mouse_click_selects_the_job_row_under_the_pointer(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause(0.5)
            expected = app.filtered_rows[1].uid
            clicked = await pilot.click(".dashboard-pane-content", offset=(5, 3))
            self.assertTrue(clicked)
            self.assertEqual(app.state.cursor_job_uid, expected)

    async def test_clicking_any_dashboard_pane_marks_it_active(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            for pane in ("selected", "resources", "events", "jobs"):
                await pilot.click(
                    f"#{pane}-pane-content", offset=(1, 1)
                )
                self.assertEqual(app.state.focused_pane, pane)
                self.assertIn("focused", app.query_one(f"#{pane}-pane").border_title)

    async def test_dashboard_refresh_keeps_highlight_off_while_app_is_blurred(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            await pilot.click("#resources-pane", offset=(2, 2))
            self.assertEqual(app.state.focused_pane, "resources")

            app.app_focus = False
            await pilot.pause()
            self.assertIsNone(app.focused)
            self.assertNotIn(
                "focused", app.query_one("#resources-pane").border_title
            )

            app._result_queue.put_nowait(
                (
                    app.rows,
                    app.job_events,
                    app.state.cursor_job_uid,
                    None,
                    app.state.last_successful_refresh,
                    app.state.gpu_availability,
                )
            )
            app._drain_results()
            await pilot.pause()
            self.assertIsNone(app.focused)
            self.assertNotIn(
                "focused", app.query_one("#resources-pane").border_title
            )

            app.app_focus = True
            await pilot.pause()
            self.assertIs(app.focused, app.query_one("#resources-pane"))
            self.assertIn("focused", app.query_one("#resources-pane").border_title)

    async def test_dashboard_focus_in_keeps_the_pane_clicked_while_blurred(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            app.app_focus = False
            await pilot.pause()

            resources = app.query_one("#resources-pane", DashboardPane)
            await pilot.click(
                "#resources-pane-content", offset=(1, 1)
            )
            self.assertEqual(app.state.focused_pane, "resources")
            self.assertIs(app.focused, resources)

            # Pilot injects already-forwarded mouse events, so explicitly send
            # the terminal focus-in that follows the activation click.
            app.app_focus = True
            await pilot.pause()
            self.assertTrue(app.app_focus)
            self.assertEqual(app.state.focused_pane, "resources")
            self.assertIs(app.focused, resources)
            self.assertIn("focused", resources.border_title)

    async def test_dashboard_stale_restored_focus_cannot_overwrite_mouse_down(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            jobs = app.query_one("#jobs-pane", DashboardPane)
            resources = app.query_one("#resources-pane", DashboardPane)

            # This is the real terminal-focus race: Textual has queued a Focus
            # for the old pane, but mouse-down has already focused a new pane.
            app.set_focus(resources, scroll_visible=False)
            resources.on_mouse_down(object())
            jobs.on_focus(events.Focus(from_app_focus=True))

            self.assertEqual(app.state.focused_pane, "resources")
            self.assertIs(app.focused, resources)

    async def test_refresh_preserves_the_selected_pane(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("4", "enter")
            selected_uid = app.state.cursor_job_uid
            self.assertEqual(app.state.focused_pane, "selected")
            self.assertEqual(app.state.expanded_pane, "selected")
            app._result_queue.put_nowait(
                (
                    app.rows,
                    app.job_events,
                    selected_uid,
                    None,
                    app.state.last_successful_refresh,
                    app.state.gpu_availability,
                )
            )
            app._drain_results()
            self.assertEqual(app.state.cursor_job_uid, selected_uid)
            self.assertEqual(app.state.focused_pane, "selected")
            self.assertEqual(app.state.expanded_pane, "selected")

    async def test_expanded_selected_job_scrolls_the_entire_inspector(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(80, 22)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("4", "enter")
            await pilot.pause()
            pane = app.query_one("#selected-pane", DashboardPane)
            self.assertGreater(pane.max_scroll_y, 0)
            self.assertEqual(int(pane.scroll_y), 0)
            before = app.export_screenshot(simplify=True)

            await pilot.press("down")
            self.assertGreater(int(pane.scroll_y), 0)
            await pilot.press("end")
            self.assertEqual(int(pane.scroll_y), pane.max_scroll_y)
            after = app.export_screenshot(simplify=True)
            self.assertNotEqual(before, after)
            self.assertIn("COMMAND", after)
            await pilot.press("home")
            self.assertEqual(int(pane.scroll_y), 0)

            class Wheel:
                def prevent_default(self):
                    pass

                def stop(self):
                    pass

            pane.on_mouse_scroll_down(Wheel())
            self.assertGreater(int(pane.scroll_y), 0)

    async def test_expanded_jobs_keeps_job_names_visible_at_minimum_width(self) -> None:
        app = FalconDashboard(DemoUsageCollector("many"), refresh_seconds=999)
        async with app.run_test(size=(80, 22)) as pilot:
            await pilot.pause(0.5)
            expected_name = app.filtered_rows[0].job
            await pilot.press("enter")
            await pilot.pause()
            svg = app.export_screenshot(simplify=True)
            self.assertIn(expected_name, svg)

    async def test_expanded_resources_routes_hovered_history_and_page_scroll(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(80, 22)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("2", "enter")
            await pilot.pause()
            self.assertTrue(app._resource_graph_regions)
            left, top, right, bottom = app._resource_graph_regions[0]
            self.assertLess(left, right)
            self.assertLess(top, bottom)

            class Wheel:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y

                def get_content_offset(self, _pane):
                    return type(
                        "Offset", (), {"x": self.x, "y": self.y}
                    )()

            with patch.object(app, "_scroll_history") as history:
                app.scroll_focused(
                    1,
                    "resources-pane",
                    Wheel((left + right) // 2, (top + bottom) // 2),
                )
                history.assert_called_once_with(1)
            with patch.object(app, "_scroll_expanded_pane") as page:
                app.scroll_focused(1, "resources-pane", Wheel(0, top))
                page.assert_called_once_with("resources", 1)

    async def test_expanded_history_scroll_updates_only_chart_renderables(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("2", "enter")
            row = app._selected_row()
            self.assertIsNotNone(row)
            assert row is not None
            app.histories[row.uid] = [
                MetricPoint(
                    timestamp=float(index),
                    gpu=float(index + 1),
                    vram=float(index + 2),
                    cpu=float(index + 3),
                    ram=float(index + 4),
                    gpu_capacity=1.0,
                    vram_capacity=1.0,
                    cpu_capacity=1.0,
                    ram_capacity=1.0,
                )
                for index in range(4)
            ]
            app._render_all()
            content = app.query_one(
                "#resources-pane .dashboard-pane-content",
                DashboardPaneContent,
            )
            static_content = app.query_one(
                "#resources-pane", DashboardPane
            )._pane_content
            chart_ids = {
                label: id(chart)
                for label, chart in app._expanded_resource_charts.items()
            }
            old_values = {
                label: list(chart.values)
                for label, chart in app._expanded_resource_charts.items()
            }
            app._scroll_history(1)
            self.assertIs(
                app.query_one("#resources-pane", DashboardPane)._pane_content,
                static_content,
            )
            self.assertIsNotNone(content)
            self.assertEqual(
                chart_ids,
                {
                    label: id(chart)
                    for label, chart in app._expanded_resource_charts.items()
                },
            )
            self.assertNotEqual(
                old_values["GPU"],
                app._expanded_resource_charts["GPU"].values,
            )

    async def test_expanded_gpu_devices_show_device_and_process_details(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause(0.5)
            row = next(item for item in app.rows if item.gpu_requested_count)
            detailed = replace(
                row,
                gpu_devices=[
                    GpuDevice(
                        index=0,
                        name="NVIDIA H100 80GB HBM3",
                        uuid="GPU-test-uuid",
                        memory_used_gib=24.0,
                        memory_total_gib=80.0,
                        utilization=65.0,
                        temperature_c=71.0,
                        power_w=312.0,
                        ecc_errors=0,
                        driver_version="570.86",
                        processes=[
                            GpuProcess(
                                gpu_uuid="GPU-test-uuid",
                                pid=4242,
                                name="python train.py",
                                memory_used_gib=23.5,
                                gpu_utilization=37.0,
                            )
                        ],
                    )
                ],
            )
            app.rows = [detailed]
            app.filtered_rows = [detailed]
            app.state.cursor_job_uid = detailed.uid
            app.state.focused_pane = "resources"
            app.state.expanded_pane = "resources"
            app._apply_layout()
            app._render_all()
            await pilot.pause()
            pane = app.query_one("#resources-pane", DashboardPane)
            frames = [app.export_screenshot(simplify=True)]
            while int(pane.scroll_y) < pane.max_scroll_y:
                pane.scroll_relative(
                    y=max(1, pane.size.height // 2),
                    animate=False,
                    force=True,
                    immediate=True,
                )
                frames.append(app.export_screenshot(simplify=True))
            svg = "\n".join(frames)
            for value in (
                "POWER", "ECC", "NVIDIA&#160;H100",
                "4242", "python&#160;train.py", "23.5G", "37.0%",
            ):
                self.assertIn(value, svg)
            self.assertNotIn("PROCESSES", svg)
            self.assertNotIn("Requested", svg)
            self.assertNotIn("Telemetry", svg)

    def test_gpu_process_utilization_parser_ignores_headers_and_missing_data(self) -> None:
        values = _parse_gpu_process_utilization(
            [
                "# gpu pid type sm mem enc dec command",
                "0 4242 C 37 18 - - python",
                "0 5252 C - - - - python",
            ]
        )
        self.assertEqual(values, {4242: 37.0})

    def test_gpu_process_utilization_uses_a_persistent_pmon_stream(self) -> None:
        process = SimpleNamespace(
            stdout=iter(
                [
                    "# gpu pid type sm mem enc dec command\n",
                    "0 4242 C 37 18 - - python\n",
                ]
            )
        )
        sampler = StreamingGpuSampler("team-a")
        with patch(
            "falcon.dashboard.subprocess.Popen", return_value=process
        ) as popen, patch("falcon.dashboard.threading.Thread") as thread:
            sampler._start_pmon("pod-a")
            command = popen.call_args.args[0]
            self.assertEqual(
                command[-6:],
                ["nvidia-smi", "pmon", "-d", "1", "-s", "u"],
            )
            thread.call_args.kwargs["target"]()
        self.assertEqual(sampler._process_utilization["pod-a"], {4242: 37.0})

    async def test_expanded_events_reach_oldest_and_newest(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("3", "enter", "home")
            self.assertEqual(app.state.expanded_pane, "events")
            self.assertEqual(app.state.events_scroll_offset, 0)
            await pilot.press("end")
            expected = max(0, len(app.job_events) - app._visible_event_count())
            self.assertEqual(app.state.events_scroll_offset, expected)

    async def test_expanded_events_omit_redundant_job_object_name(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("3", "enter")
            app.job_events = [
                JobEvent(
                    "2026-01-15T12:00:00Z",
                    "Normal",
                    "Scheduled",
                    "Placed successfully",
                    object_name="redundant-job-name",
                )
            ]
            app.state.events_auto_follow = True
            app._render_events()
            svg = app.export_screenshot(simplify=True)
            self.assertNotIn("redundant-job-name", svg)
            self.assertIn("Placed&#160;successfully", svg)

    async def test_event_search_clamps_scroll_and_remains_reachable(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("3", "home")
            app.event_search = "BackOff"
            app._render_events()
            app.action_end()
            values = app._filtered_events()
            self.assertTrue(values)
            self.assertTrue(all(event.reason == "BackOff" for event in values))
            self.assertLessEqual(
                app.state.events_scroll_offset,
                max(0, len(values) - app._visible_event_count()),
            )

    async def test_mouse_wheel_uses_same_event_scroll_path(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)

        class Wheel:
            def prevent_default(self):
                pass

            def stop(self):
                pass

        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("3", "home")
            pane = app.query_one("#events-pane", DashboardPane)
            pane.on_mouse_scroll_down(Wheel())
            self.assertEqual(app.state.events_scroll_offset, 1)
            pane.on_mouse_scroll_up(Wheel())
            self.assertEqual(app.state.events_scroll_offset, 0)

    async def test_resize_sequence_preserves_selection_focus_and_expansion(self) -> None:
        app = FalconDashboard(DemoUsageCollector("many"), refresh_seconds=999)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("down", "down", "3", "enter")
            uid = app.state.cursor_job_uid
            for width, height in (
                (80, 22), (79, 21), (120, 30), (200, 50)
            ):
                await pilot.resize_terminal(width, height)
                await pilot.pause()
                self.assertEqual(app.state.cursor_job_uid, uid)
            self.assertEqual(app.state.focused_pane, "events")
            self.assertEqual(app.state.expanded_pane, "events")

    async def test_minimum_boundary_and_clean_resize_message(self) -> None:
        app = FalconDashboard(DemoUsageCollector("one-job"), refresh_seconds=999)
        async with app.run_test(size=(79, 21)) as pilot:
            await pilot.pause(0.3)
            self.assertTrue(app.query_one("#resize-message").display)
            await pilot.resize_terminal(80, 22)
            await pilot.pause()
            self.assertFalse(app.query_one("#resize-message").display)
            self.assertTrue(app.query_one("#falcon-footer").display)
            self.assertIn("c Clean", app.query_one("#falcon-footer").render().plain)

    async def test_no_jobs_stale_and_dialog_states_render_without_traceback(self) -> None:
        no_jobs = FalconDashboard(DemoUsageCollector("no-jobs"), refresh_seconds=999)
        async with no_jobs.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.5)
            self.assertFalse(no_jobs.rows)
            self.assertIn("No&#160;Jobs", no_jobs.export_screenshot(simplify=True))
        stale = FalconDashboard(DemoUsageCollector("stale"), refresh_seconds=999)
        async with stale.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.5)
            self.assertTrue(stale._stale)
            stale.action_filters()
            await pilot.pause()
            self.assertEqual(type(stale.screen).__name__, "FilterDialog")
            await pilot.press("escape")

    async def test_marking_and_confirmation_dialogs(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause(0.5)
            selected = app._selected_row()
            self.assertIsNotNone(selected)
            app.action_mark_next()
            self.assertIn(selected.uid, app.state.marked_job_uids)
            self.assertNotEqual(app._selected_row().uid, selected.uid)

            app.action_mark_all()
            self.assertEqual(
                app.state.marked_job_uids,
                {row.uid for row in app.filtered_rows},
            )
            app.action_clear_marks()
            self.assertFalse(app.state.marked_job_uids)

            app.action_kill()
            await pilot.pause()
            self.assertEqual(type(app.screen).__name__, "KillDialog")
            self.assertEqual(app.screen.actions, ["job", "restart"])
            svg = app.export_screenshot(simplify=True)
            self.assertIn("Job&#160;Actions", svg)
            self.assertNotIn("Delete&#160;active&#160;pod", svg)
            await pilot.press("escape")
            self.assertFalse(app.state.kill_dialog["isOpen"])

            app.action_cleanup()
            await pilot.pause()
            self.assertEqual(type(app.screen).__name__, "CleanupDialog")
            self.assertFalse(app.screen.marked)
            self.assertTrue(all(row.status == "Succeeded" for row in app.screen.rows))
            self.assertFalse(app.state.marked_job_uids)
            await pilot.press("escape")

            marked_targets = [
                next(row for row in app.rows if row.status == "Running"),
                next(row for row in app.rows if row.status == "Failed"),
                next(row for row in app.rows if row.status == "Succeeded"),
            ]
            app.state.marked_job_uids.update(row.uid for row in marked_targets)
            app.action_cleanup()
            await pilot.pause()
            self.assertEqual(type(app.screen).__name__, "CleanupDialog")
            self.assertTrue(app.screen.marked)
            self.assertEqual(app.screen.excluded_marked, 2)
            self.assertEqual(
                {row.uid for row in app.screen.rows},
                {row.uid for row in marked_targets if row.status == "Succeeded"},
            )
            dialog = app.export_screenshot(simplify=True)
            self.assertIn("CLEAN&#160;MARKED&#160;SUCCEEDED&#160;JOBS", dialog)
            self.assertIn("2&#160;marked&#160;running&#160;or&#160;failed", dialog)
            await pilot.press("escape")

    async def test_jobs_footer_advertises_mark_kill_and_contextual_clean(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(80, 22)) as pilot:
            await pilot.pause(0.5)
            footer = app.query_one("#falcon-footer").render().plain
            self.assertIn("Space Mark", footer)
            self.assertIn("k Kill", footer)
            self.assertIn("c Clean", footer)
            self.assertIn("s Sort", footer)
            self.assertIn("Tab Next pane", footer)

            app.action_toggle_mark()
            footer = app.query_one("#falcon-footer").render().plain
            self.assertIn("Space Mark (1)", footer)

    async def test_dashboard_footer_uses_consistent_pane_actions(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(160, 32)) as pilot:
            await pilot.pause(0.5)
            for pane in ("jobs", "selected", "resources", "events"):
                app.state.focused_pane = pane
                app.state.expanded_pane = None
                app._render_footer()
                footer = app.query_one("#falcon-footer").render().plain
                self.assertIn("Enter Expand", footer)
                self.assertIn("Tab Next pane", footer)
                self.assertIn("r Refresh", footer)
                self.assertIn("q Quit", footer)

                app.state.expanded_pane = pane
                app._render_footer()
                footer = app.query_one("#falcon-footer").render().plain
                self.assertIn("Esc Restore", footer)
                self.assertIn("Tab Next pane", footer)

    async def test_coder_kill_and_restart_use_coder_instead_of_kubectl(self) -> None:
        actions = []
        app = FalconDashboard(
            DemoUsageCollector("mixed"),
            refresh_seconds=999,
            coder_workspace_action=lambda job, action: actions.append(
                (job, action)
            ),
        )
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause(0.5)
            row = replace(
                app.rows[0],
                job="coder-divyam.c-falcon",
            )
            with patch("falcon.dashboard_ui.subprocess.run") as kubectl:
                app._job_action_confirmed(("restart", [row]))
                await pilot.pause(0.2)
                app._job_action_confirmed(("job", [row]))
                await pilot.pause(0.2)

            self.assertEqual(
                actions,
                [
                    ("coder-divyam.c-falcon", "restart"),
                    ("coder-divyam.c-falcon", "delete"),
                ],
            )
            kubectl.assert_not_called()

    def test_restart_manifest_removes_controller_owned_fields(self) -> None:
        source = {
            "apiVersion": "batch/v1",
            "metadata": {
                "name": "old",
                "uid": "server-owned",
                "labels": {
                    "controller-uid": "server-owned",
                    "falcon.dev/job": "true",
                },
            },
            "spec": {
                "selector": {"matchLabels": {"controller-uid": "server-owned"}},
                "template": {
                    "metadata": {
                        "labels": {
                            "job-name": "old",
                            "falcon.dev/job": "true",
                        }
                    },
                    "spec": {"restartPolicy": "Never", "containers": []},
                },
            },
        }
        manifest = _restart_job_manifest(source, "new", "research")
        self.assertEqual(
            manifest["metadata"],
            {
                "name": "new",
                "namespace": "research",
                "labels": {"falcon.dev/job": "true"},
            },
        )
        self.assertNotIn("selector", manifest["spec"])
        self.assertEqual(
            manifest["spec"]["template"]["metadata"]["labels"],
            {"falcon.dev/job": "true"},
        )


class ResourceInteractionTests(unittest.IsolatedAsyncioTestCase):
    def test_cpu_values_use_decimal_cores(self) -> None:
        self.assertEqual(_short_cpu(0.095), "0.095")
        self.assertEqual(_short_cpu(0.5), "0.5")
        self.assertEqual(_short_cpu(96), "96")

    def test_gpu_headroom_colors_show_last_gpu_and_exhaustion(self) -> None:
        self.assertEqual(_gpu_headroom_color(15, 20), GREEN)
        self.assertEqual(_gpu_headroom_color(2, 4), GREEN)
        self.assertEqual(_gpu_headroom_color(1, 4), YELLOW)
        self.assertEqual(_gpu_headroom_color(0, 4), RED)
        self.assertEqual(_gpu_headroom_color(1, 2), YELLOW)
        self.assertEqual(_gpu_headroom_color(1, 8), RED)
        self.assertEqual(_gpu_headroom_color(0, 2), RED)
        self.assertEqual(_gpu_headroom_color(4, 4), GREEN)
        self.assertEqual(_gpu_headroom_color(0, 0), MUTED)

    def test_cpu_and_memory_colors_use_remaining_headroom(self) -> None:
        self.assertEqual(_resource_headroom_color(100, 100), GREEN)
        self.assertEqual(_resource_headroom_color(70.1, 100), GREEN)
        self.assertEqual(_resource_headroom_color(70, 100), YELLOW)
        self.assertEqual(_resource_headroom_color(20.1, 100), YELLOW)
        self.assertEqual(_resource_headroom_color(20, 100), RED)
        self.assertEqual(_resource_headroom_color(0, 100), RED)
        self.assertEqual(_resource_headroom_color(1, 0), MUTED)

    def test_all_request_metrics_use_dashboard_pressure_colors(self) -> None:
        self.assertEqual(_request_pressure_color(0, 100), GREEN)
        self.assertEqual(_request_pressure_color(29.9, 100), GREEN)
        self.assertEqual(_request_pressure_color(30, 100), YELLOW)
        self.assertEqual(_request_pressure_color(79.9, 100), YELLOW)
        self.assertEqual(_request_pressure_color(80, 100), RED)
        self.assertEqual(_request_pressure_color(1, 0), MUTED)

    async def test_resource_header_and_overview_use_dashboard_style(self) -> None:
        app = FalconResourcesApp(
            DemoCollector("mixed"),
            refresh_seconds=999,
            clock=lambda snapshot: "12:00:00",
        )
        async with app.run_test(size=(80, 22)) as pilot:
            await pilot.pause(0.5)
            header = app.query_one("#resources-header").render()
            overview = app.query_one("#cluster-overview").render()
            self.assertIn("Falcon Resources", header.plain)
            # The clock spinner advances independently of the render loop;
            # assert the stable clock value without depending on its frame.
            self.assertIn("12:00:00", header.plain)
            self.assertNotIn("\n", overview.plain)
            self.assertRegex(overview.plain, r"MEM \d+G/\d+G")
            self.assertLess(overview.plain.index("A6000"), overview.plain.index("H100"))
            for model in ("A6000", "H100"):
                model_style = overview.get_style_at_offset(
                    overview.plain.index(model)
                )
                self.assertTrue(model_style.bold)
                self.assertEqual(model_style.foreground.hex6, YELLOW)

    async def test_resources_gpu_summary_uses_shared_model_order(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(160, 32)) as pilot:
            await pilot.pause(0.5)
            base = next(node for node in app.nodes if node.gpu_model == "H100")

            def model_node(model: str, name: str):
                return replace(
                    base,
                    name=name,
                    capacity=replace(base.capacity, gpu_model=model),
                    allocatable=replace(base.allocatable, gpu_model=model),
                    requested=replace(base.requested, gpu_model=model),
                )

            app.nodes = [
                model_node("H100", "node-h100"),
                model_node("PRO6000", "node-pro6000"),
                model_node("A6000", "node-a6000"),
                model_node("2080Ti", "node-2080ti"),
            ]
            app._render_overview()
            overview = app.query_one("#cluster-overview").render().plain
            positions = [
                overview.index(model)
                for model in ("2080Ti", "A6000", "PRO6000", "H100")
            ]
            self.assertEqual(positions, sorted(positions))

    async def test_selected_node_jobs_have_explicit_resources_and_cycle_sort(self) -> None:
        persisted = []
        app = FalconResourcesApp(
            DemoCollector("mixed"),
            refresh_seconds=999,
            persist_consumer_sort=persisted.append,
        )
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("enter")
            await pilot.pause()
            table = app._consumer_table(app._selected(), expanded=True)
            rendered = "".join(
                segment.text
                for segment in app.console.render(
                    table, app.console.options.update(width=130, height=12)
                )
            )
            positions = [
                rendered.index(label)
                for label in ("NAMESPACE", "JOB", "STATUS", "CPU", "RAM", "GPU")
            ]
            self.assertEqual(positions, sorted(positions))
            self.assertNotIn("NAMESPACE / WORKLOAD", rendered)
            self.assertNotIn("POD", rendered)

            for expected in CONSUMER_SORTS[1:]:
                await pilot.press("s")
                await pilot.pause()
                self.assertEqual(app.state.consumer_sort, expected)
                self.assertEqual(persisted[-1], expected)
            self.assertEqual(app.state.consumer_sort, "gpu")
            values = app._sorted_consumers(app._selected())
            self.assertEqual(
                [consumer.requested.gpu_count for consumer in values],
                sorted(
                    (consumer.requested.gpu_count for consumer in values),
                    reverse=True,
                ),
            )
        fallback = FalconResourcesApp(
            DemoCollector("mixed"), initial_consumer_sort="unsupported"
        )
        self.assertEqual(fallback.state.consumer_sort, "namespace")

    async def test_resources_minimum_height_is_20_rows(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(80, 19)) as pilot:
            await pilot.pause(0.3)
            resize_message = app.query_one("#resize-message")
            self.assertTrue(resize_message.display)
            self.assertIn("80×20", resize_message.render().plain)
            await pilot.resize_terminal(80, 20)
            await pilot.pause()
            self.assertFalse(resize_message.display)
            self.assertTrue(app.query_one("#resources-footer").display)

    async def test_nodes_merge_responsive_gpu_bars_and_highlight_selection(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(80, 22)) as pilot:
            await pilot.pause(0.5)
            compact = app.query_one("#cluster-overview").render().plain
            self.assertRegex(compact, r"A6000 \d+/\d+")
            self.assertRegex(compact, r"H100 \d+/\d+")

            table = vars(app.query_one("#nodes-pane"))["_Static__content"]
            segments = list(
                app.console.render(
                    table,
                    app.console.options.update(width=76, height=12),
                )
            )
            selected = next(segment for segment in segments if "node-a-h100" in segment.text)
            self.assertEqual(selected.style.color.name.upper(), CYAN)
            self.assertEqual(selected.style.bgcolor.name.upper(), SELECTION)
            compact_text = "".join(segment.text for segment in segments)
            # The removed SIMPLE_HEAD edge spacer must give the inventory
            # enough room to render the final node at the supported minimum.
            self.assertIn("node-d-cpu", compact_text)
            headers = ("NODE", "CPUS", "RAM (GB)", "GPUS", "GPU TYPE", "SCHED")
            positions = [compact_text.index(header) for header in headers]
            self.assertEqual(positions, sorted(positions))

            await pilot.resize_terminal(140, 32)
            await pilot.pause()
            wide_overview = app.query_one("#cluster-overview").render().plain
            self.assertIn("GPU AVAILABLE", wide_overview)
            self.assertIn("A6000 2/4", wide_overview)
            self.assertIn("H100 1/4", wide_overview)
            self.assertIn("CPU 91.5/160", wide_overview)
            self.assertIn("MEM 927G/1200G", wide_overview)
            table = vars(app.query_one("#nodes-pane"))["_Static__content"]
            wide_segments = list(
                app.console.render(
                    table, app.console.options.update(width=136, height=20)
                )
            )
            wide_text = "".join(segment.text for segment in wide_segments)
            self.assertIn("█", wide_text)
            self.assertIn("27.5/64", wide_text)
            self.assertIn("335/480", wide_text)
            self.assertIn("1/4", wide_text)
            bar_colors = {
                segment.style.color.name.upper()
                for segment in wide_segments
                if segment.style and segment.style.color and "█" in segment.text
            }
            self.assertTrue({GREEN, YELLOW}.issubset(bar_colors))

            await pilot.press("right")
            await pilot.pause()
            self.assertTrue(app.query_one("#cluster-overview").display)
            self.assertIn(
                "GPU AVAILABLE",
                app.query_one("#cluster-overview").render().plain,
            )
            self.assertEqual(
                app.query_one("#resources-footer").region.y,
                app.size.height - 1,
            )

            await pilot.press("left")
            await pilot.press("enter")
            await pilot.pause()
            inspector = vars(app.query_one("#node-pane"))["_Static__content"]
            inspector_text = "".join(
                segment.text
                for segment in app.console.render(
                    inspector,
                    app.console.options.update(width=136, height=30),
                )
            )
            self.assertLess(inspector_text.index("CPU capacity"), inspector_text.index("RAM capacity"))
            self.assertLess(inspector_text.index("RAM capacity"), inspector_text.index("GPU model"))

    async def test_two_views_wrap_restore_persist_and_support_clicks(self) -> None:
        saved = []
        app = FalconResourcesApp(
            DemoCollector("mixed"),
            refresh_seconds=999,
            initial_view="gpu-overview",
            persist_view=saved.append,
        )
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            self.assertEqual(app.state.view, "nodes")
            self.assertEqual(app.focused.id, "nodes-pane")

            await pilot.press("right")
            self.assertEqual(app.state.view, "gpu-allocations")
            await pilot.press("right")
            self.assertEqual(app.state.view, "nodes")
            await pilot.press("left")
            self.assertEqual(app.state.view, "gpu-allocations")

            start, _, _ = next(
                hitbox for hitbox in app._view_hitboxes if hitbox[2] == "nodes"
            )
            # Pilot offsets include the selector's one-cell left padding.
            clicked = await pilot.click("#resources-views", offset=(start + 1, 0))
            self.assertTrue(clicked)
            self.assertEqual(app.state.view, "nodes")
            self.assertEqual(
                saved,
                ["gpu-allocations", "nodes", "gpu-allocations", "nodes"],
            )

            def fail_to_save(view):
                raise OSError("read-only config")

            app.persist_view = fail_to_save
            await pilot.press("right")
            self.assertEqual(app.state.view, "gpu-allocations")

        fallback = FalconResourcesApp(DemoCollector("mixed"), initial_view="invalid")
        self.assertEqual(fallback.state.view, "nodes")
        self.assertEqual(RESOURCE_VIEWS, ("nodes", "gpu-allocations"))

    async def test_view_switching_preserves_each_scroll_and_node_state(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        source = snapshot.nodes[0]
        base_consumer = next(
            consumer
            for consumer in source.visible_consumers
            if consumer.requested.gpu_count
        )
        source = replace(
            source,
            consumers=tuple(
                replace(
                    base_consumer,
                    pod_name=f"consumer-{index:02d}",
                    workload_name=f"workload-{index:02d}",
                )
                for index in range(20)
            ),
        )
        snapshot = replace(
            snapshot,
            nodes=tuple(
                replace(source, name=f"gpu-node-{index:02d}")
                for index in range(30)
            ),
        )

        class Collector:
            def collect(self, force=False):
                return snapshot

            def close(self):
                pass

        app = FalconResourcesApp(Collector(), refresh_seconds=999)
        async with app.run_test(size=(80, 22)) as pilot:
            await pilot.pause(0.5)
            app.state.selected_node = app.nodes[-1].name
            app.state.selected_consumer = 2
            app.state.consumer_scroll = 1
            app.state.expanded = True
            app.state.focused_panes["nodes"] = "node"
            app.state.active_pane = "node"

            await pilot.press("right")
            app.state.allocation_scroll = 1
            await pilot.resize_terminal(140, 32)
            await pilot.press("right")

            self.assertEqual(app.state.view, "nodes")
            self.assertTrue(app.state.expanded)
            self.assertEqual(app.state.selected_node, app.nodes[-1].name)
            self.assertEqual(app.state.consumer_scroll, 1)
            self.assertEqual(app.state.focused_panes["nodes"], "node")

            await pilot.press("right")
            self.assertEqual(app.state.allocation_scroll, 1)

    async def test_resources_tab_cycles_panes_and_allocation_tab_stays_contained(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            self.assertEqual(app.focused.id, "nodes-pane")

            await pilot.press("tab")
            self.assertEqual(app.focused.id, "node-pane")
            self.assertEqual(app.state.active_pane, "node")
            await pilot.press("tab")
            self.assertEqual(app.focused.id, "nodes-pane")
            self.assertIn("Tab Next pane", app.query_one("#resources-footer").render().plain)

            await pilot.press("right")
            self.assertEqual(app.focused.id, "gpu-allocations-pane")
            self.assertIn("Enter Expand", app.query_one("#resources-footer").render().plain)
            self.assertIn("v COUNT", app.query_one("#resources-footer").render().plain)
            self.assertEqual(app.state.selected_panels["gpu-allocations"], "history")
            await pilot.press("tab")
            self.assertEqual(app.focused.id, "gpu-allocations-pane")
            self.assertEqual(app.state.selected_panels["gpu-allocations"], "pie")
            await pilot.press("tab")
            self.assertEqual(app.state.selected_panels["gpu-allocations"], "pods")
            await pilot.press("tab")
            self.assertEqual(app.state.selected_panels["gpu-allocations"], "history")
            await pilot.press("v")
            self.assertIn("v VRAM", app.query_one("#resources-footer").render().plain)

            await pilot.press("tab")
            self.assertEqual(app.focused.id, "gpu-allocations-pane")
            self.assertEqual(app.state.selected_panels["gpu-allocations"], "pie")
            await pilot.press("shift+tab")
            self.assertEqual(app.state.selected_panels["gpu-allocations"], "history")
            await pilot.press("enter")
            self.assertEqual(app.state.expanded_panels["gpu-allocations"], "history")
            self.assertIn("Enter expand", app.query_one("#gpu-allocations-pane").border_subtitle)
            self.assertNotIn("Enter expand selected", app.query_one("#gpu-allocations-pane").border_subtitle)
            await pilot.press("tab")
            self.assertEqual(app.state.selected_panels["gpu-allocations"], "pie")
            self.assertEqual(app.state.expanded_panels["gpu-allocations"], "pie")
            await pilot.press("shift+tab")
            self.assertEqual(app.state.expanded_panels["gpu-allocations"], "history")

    async def test_gpu_allocation_arrows_use_their_own_scroll_offset(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        source = snapshot.nodes[0]
        nodes = tuple(
            replace(source, name=f"gpu-node-{index:02d}")
            for index in range(30)
        )
        snapshot = replace(snapshot, nodes=nodes)

        class Collector:
            def collect(self, force=False):
                return snapshot

            def close(self):
                pass

        app = FalconResourcesApp(
            Collector(), refresh_seconds=999, initial_view="gpu-allocations"
        )
        async with app.run_test(size=(80, 22)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("down", "down")
            self.assertGreaterEqual(app.state.allocation_scroll, 1)
            self.assertEqual(app.state.node_scroll, 0)

    async def test_gpu_telemetry_collects_before_allocations_view_and_v_toggles_pie(self) -> None:
        app = FalconResourcesApp(
            DemoCollector("mixed"),
            refresh_seconds=999,
        )
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            self.assertEqual(app.state.view, "nodes")
            self.assertEqual(len(app.history), 1)

            await pilot.press("right")
            self.assertEqual(app.state.view, "gpu-allocations")
            self.assertEqual(app.state.namespace_basis, "gpu")
            await pilot.press("v")
            self.assertEqual(app.state.namespace_basis, "vram")
            rendered = app.export_screenshot(simplify=True)
            self.assertIn("ALLOCATION&#160;BY&#160;NAMESPACE", rendered)
            self.assertIn("v&#160;VRAM", rendered)
            self.assertNotIn("VRAM&#160;percentages", rendered)
            self.assertNotIn("GPU&#160;COUNT&#160;%", rendered)
            self.assertEqual(rendered.count("NAMESPACE&#160;LEGEND"), 1)

    async def test_gpu_panel_click_selects_and_enter_expands(self) -> None:
        app = FalconResourcesApp(
            DemoCollector("mixed"),
            refresh_seconds=999,
            initial_view="gpu-allocations",
        )
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            await pilot.click("#gpu-allocations-pane", offset=(5, 2))
            self.assertEqual(app.state.selected_panels["gpu-allocations"], "pie")
            await pilot.click("#gpu-allocations-pane", offset=(95, 2))
            self.assertEqual(app.state.selected_panels["gpu-allocations"], "history")
            self.assertEqual(app.state.expanded_panels["gpu-allocations"], "")
            await pilot.press("enter")
            self.assertEqual(app.state.expanded_panels["gpu-allocations"], "history")
            pane = app.query_one("#gpu-allocations-pane")
            chart_width = max(
                14,
                pane.content_size.width
                - app._allocation_legend_width(pane.content_size.width)
                - 6,
            )
            self.assertLessEqual(
                max(
                    (len(line) for line in app._history_cache.plain.splitlines()),
                    default=0,
                ),
                chart_width,
            )
            self.assertIn("Total&#160;usage", app.export_screenshot(simplify=True))
            await pilot.press("escape")
            await pilot.click("#gpu-allocations-pane", offset=(5, 14))
            self.assertEqual(app.state.selected_panels["gpu-allocations"], "pie")
            self.assertEqual(app.state.expanded_panels["gpu-allocations"], "")
            await pilot.press("enter")
            self.assertEqual(app.state.expanded_panels["gpu-allocations"], "pie")
            await pilot.press("escape")
            await pilot.click("#gpu-allocations-pane", offset=(95, 14))
            self.assertEqual(app.state.selected_panels["gpu-allocations"], "pods")
            self.assertEqual(app.state.expanded_panels["gpu-allocations"], "")
            await pilot.press("enter")
            self.assertEqual(app.state.expanded_panels["gpu-allocations"], "pods")
            await pilot.press("escape")
            self.assertEqual(app.state.expanded_panels["gpu-allocations"], "")

    async def test_hidden_and_unchanged_charts_are_not_rebuilt(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        with patch(
            "falcon.resources_ui.render_gpu_history", wraps=render_gpu_history
        ) as history_renderer, patch(
            "falcon.resources_ui.render_namespace_pie",
            wraps=render_namespace_pie,
        ) as pie_renderer:
            async with app.run_test(size=(140, 32)) as pilot:
                await pilot.pause(0.5)
                app._tick_clock()
                app._results.put_nowait(app.snapshot)
                app._drain_results()
                self.assertEqual(history_renderer.call_count, 0)
                self.assertEqual(pie_renderer.call_count, 0)

                await pilot.press("right")
                self.assertEqual(history_renderer.call_count, 1)
                self.assertEqual(pie_renderer.call_count, 1)

                app._tick_clock()
                app._render_all()
                self.assertEqual(history_renderer.call_count, 1)
                self.assertEqual(pie_renderer.call_count, 1)

    def test_history_deduplicates_stale_points_prunes_and_tracks_disappearing_nodes(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"))
        snapshot = GpuTelemetrySnapshot.from_mappings(
            collected_at=DEMO_NOW,
            effective_gpus_by_node={"node-a-h100": 2.4, "node-b-a6000": 0.7},
            effective_gpus_by_namespace={"team-a": 2.4, "team-b": 0.7},
            vram_gib_by_namespace={"team-a": 41.0, "team-b": 12.0},
            target_pods=2,
            sampled_pods=2,
        )
        app._record_gpu_history(snapshot)
        app._record_gpu_history(snapshot)
        app._record_gpu_history(snapshot.mark_stale("cached failure"))
        self.assertEqual(len(app.history), 1)

        later = GpuTelemetrySnapshot.from_mappings(
            collected_at=snapshot.collected_at + 60,
            effective_gpus_by_node={"node-a-h100": 1.25},
            effective_gpus_by_namespace={"team-a": 1.25},
            vram_gib_by_namespace={"team-a": 32.0},
            target_pods=1,
            sampled_pods=1,
        )
        app._record_gpu_history(later)
        self.assertEqual(len(app.history), 2)
        self.assertEqual(dict(app.history[-1].usage_by_namespace), {"team-a": 1.25})

        newest = replace(snapshot, collected_at=snapshot.collected_at + 25 * 3600)
        app._record_gpu_history(newest)
        self.assertEqual([point.timestamp for point in app.history], [newest.collected_at])

        with patch("falcon.resources_ui.HISTORY_LIMIT", 3):
            for index in range(1, 6):
                point = replace(newest, collected_at=newest.collected_at + index)
                app._record_gpu_history(point)
        self.assertEqual(len(app.history), 3)

        empty = FalconResourcesApp(DemoCollector("mixed"))
        empty._record_gpu_history(
            replace(
                snapshot,
                effective_gpus_by_node=(),
                effective_gpus_by_namespace=(),
                sampled_pods=0,
            )
        )
        self.assertEqual(empty.history, [])

        monotonic = FalconResourcesApp(
            DemoCollector("mixed"), history_clock=lambda: 1_900_000_000.0
        )
        monotonic._record_gpu_history(replace(snapshot, collected_at=1234.0))
        self.assertEqual(monotonic.history[0].timestamp, 1_900_000_000.0)

    def test_filtered_gpu_totals_namespace_and_pods_reconcile_hidden_requests(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        node = snapshot.nodes[0]
        hidden = replace(
            node.consumers[0],
            namespace="kube-system",
            pod_name="system-gpu",
            requested=replace(node.consumers[0].requested, gpu_count=1),
        )
        node = replace(
            node,
            requested=replace(node.requested, gpu_count=4),
            consumers=(*node.consumers, hidden),
        )
        app = FalconResourcesApp(
            DemoCollector("mixed"), node_filter="node-a", gpu_filter="h100"
        )
        app.nodes = [node]

        allocatable, requested, free, pressure = _gpu_totals(app._gpu_nodes())
        self.assertEqual((allocatable, requested, free, pressure), (4, 4, 0, 100.0))
        visible_namespace = next(
            consumer.namespace
            for consumer in node.consumers
            if consumer.requested.gpu_count and not consumer.namespace.startswith("kube-")
        )
        app.gpu_telemetry = GpuTelemetrySnapshot.from_mappings(
            collected_at=DEMO_NOW,
            effective_gpus_by_node={node.name: 1.75},
            effective_gpus_by_namespace={
                visible_namespace: 1.5,
                "kube-system": 0.25,
            },
            vram_gib_by_namespace={
                visible_namespace: 18.0,
                "kube-system": 6.0,
            },
            target_pods=2,
            sampled_pods=2,
        )
        categories = dict(app._namespace_categories())
        self.assertEqual(categories["System/hidden"], 0.25)
        self.assertEqual(sum(categories.values()), 1.75)
        app.state.namespace_basis = "vram"
        vram_categories = dict(app._namespace_categories())
        self.assertEqual(vram_categories[visible_namespace], 18.0)
        self.assertEqual(vram_categories["System/hidden"], 6.0)
        consumers = app._gpu_consumers()
        self.assertEqual(sum(item.requested.gpu_count for item in consumers), 3)
        self.assertEqual(consumers[0].requested.gpu_count, 2)

        excluded = replace(node, schedulable=False)
        self.assertEqual(_gpu_totals([excluded]), (0, 0, 0, 0.0))

        app.gpu_telemetry = GpuTelemetrySnapshot.from_mappings(
            collected_at=DEMO_NOW + 5,
            effective_gpus_by_node={node.name: 36.0},
            effective_gpus_by_namespace={
                f"team-{index}": 8 - index for index in range(8)
            },
            vram_gib_by_namespace={
                f"team-{index}": (8 - index) * 10 for index in range(8)
            },
            target_pods=8,
            sampled_pods=8,
        )
        app.state.namespace_basis = "gpu"
        categories = app._namespace_categories()
        self.assertEqual(len(categories), 7)
        self.assertEqual(categories[-1], ("Other", 3.0))

    async def test_node_expansion_reconciles_consumers_at_narrow_size(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(80, 22)) as pilot:
            await pilot.pause(0.5)
            self.assertEqual(app.state.selected_node, "node-a-h100")
            node = app._selected()
            self.assertIsNotNone(node)
            self.assertEqual((node.gpu_free, node.allocatable.gpu_count), (1, 4))
            compact = app.export_screenshot(simplify=True)
            self.assertIn("VRAM", compact)
            self.assertIn("H100&#160;1/4", compact)
            self.assertIn("free/allocatable", compact)
            await pilot.press("enter")
            self.assertTrue(app.state.expanded)
            self.assertEqual(len(node.visible_consumers), 3)
            await pilot.press("end")
            self.assertEqual(
                app.state.selected_consumer,
                len(node.visible_consumers) - 1,
            )

    async def test_resource_resize_and_stale_last_good_state(self) -> None:
        app = FalconResourcesApp(DemoCollector("stale"), refresh_seconds=999)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.5)
            self.assertTrue(app.snapshot.stale)
            self.assertTrue(app.nodes)
            selected = app.state.selected_node
            for width, height in ((80, 22), (79, 21), (120, 30), (200, 50)):
                await pilot.resize_terminal(width, height)
                await pilot.pause()
            self.assertEqual(app.state.selected_node, selected)

    async def test_resource_resize_after_terminal_reattach_has_no_virtual_scroll(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            # tmux can report an intermediate height while the client is
            # detached/reattached. Resources must reflow against the committed
            # size instead of retaining the previous pane height.
            for size in ((140, 31), (140, 32), (140, 30), (140, 32)):
                await pilot.resize_terminal(*size)
                await pilot.pause(0.1)
                app.app_focus = False
                await pilot.pause()
                app.app_focus = True
                await pilot.pause(0.1)
                self.assertEqual(app.screen.virtual_size, app.size)
                self.assertEqual(app.screen.scroll_y, 0)
                self.assertLessEqual(
                    app.screen.scrollable_content_region.height,
                    app.size.height,
                )

    async def test_resource_keyboard_and_mouse_navigation(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)

        class Wheel:
            def prevent_default(self):
                pass

            def stop(self):
                pass

        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.5)
            first = app.state.selected_node
            pane = app.query_one("#nodes-pane", ResourcesPane)
            pane.on_mouse_scroll_down(Wheel())
            self.assertNotEqual(app.state.selected_node, first)
            pane.on_mouse_scroll_up(Wheel())
            self.assertEqual(app.state.selected_node, first)
            await pilot.press("down")
            await pilot.press("end")
            self.assertEqual(app.state.selected_node, app.nodes[-1].name)
            await pilot.press("home")
            self.assertEqual(app.state.selected_node, app.nodes[0].name)

    async def test_resource_bottom_panel_hides_when_nodes_would_overflow(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        nodes = tuple(
            replace(snapshot.nodes[0], name=f"node-a-h100-{index:02d}")
            for index in range(30)
        )

        class Collector:
            def collect(self, force=False):
                return replace(snapshot, nodes=nodes)

            def close(self):
                pass

        app = FalconResourcesApp(Collector(), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            self.assertTrue(app._detail_auto_hidden)
            self.assertFalse(app.query_one("#node-pane").display)
            for _ in range(4):
                app._results.put_nowait(replace(snapshot, nodes=nodes))
                app._drain_results()
                await pilot.press("down")
                self.assertTrue(app._detail_auto_hidden)
                self.assertFalse(app.query_one("#node-pane").display)
            await pilot.resize_terminal(200, 50)
            await pilot.pause()
            self.assertFalse(app._detail_auto_hidden)
            self.assertTrue(app.query_one("#node-pane").display)

    async def test_resource_boundary_resize_never_hides_one_node_under_detail(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        node = snapshot.nodes[0]
        nodes = tuple(replace(node, name=f"node-{index:02d}") for index in range(10))
        snapshot = replace(snapshot, nodes=nodes)

        class Collector:
            def collect(self, force=False):
                return snapshot

            def close(self):
                pass

        app = FalconResourcesApp(Collector(), refresh_seconds=999)
        async with app.run_test(size=(140, 27)) as pilot:
            await pilot.pause(0.5)
            self.assertFalse(app._detail_auto_hidden)
            # Removing SIMPLE_HEAD's edge spacer gives the inventory one
            # additional row of usable height without hiding any node.
            self.assertEqual(app.query_one("#nodes-pane").region.height, 14)
            self.assertEqual(app.query_one("#node-pane").region.height, 8)
            await pilot.click("#node-pane", offset=(2, 2))
            self.assertEqual(app.state.active_pane, "node")
            await pilot.resize_terminal(140, 24)
            await pilot.pause()
            # At the new boundary the selected-node pane can retain its
            # five-row minimum while all ten inventory rows remain visible.
            self.assertFalse(app._detail_auto_hidden)
            self.assertTrue(app.query_one("#node-pane").display)
            self.assertEqual(app.query_one("#nodes-pane").region.height, 14)
            self.assertEqual(app.query_one("#node-pane").region.height, 5)
            self.assertEqual(app.state.active_pane, "node")
            self.assertIs(app.focused, app.query_one("#node-pane"))

    async def test_resource_node_pane_stays_fixed_while_detail_pane_grows(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            self.assertEqual(
                app.query_one("#nodes-pane").region.height,
                len(app.nodes) + app._NODE_TABLE_OVERHEAD,
            )
            self.assertEqual(
                app.query_one("#node-pane").region.height,
                32 - app._FIXED_LAYOUT_HEIGHT - app.query_one("#nodes-pane").region.height,
            )
            await pilot.resize_terminal(140, 24)
            await pilot.pause()
            self.assertEqual(
                app.query_one("#nodes-pane").region.height,
                len(app.nodes) + app._NODE_TABLE_OVERHEAD,
            )
            self.assertTrue(app.query_one("#node-pane").display)

    async def test_resource_bottom_panel_wheel_scrolls_consumers(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        node = snapshot.nodes[0]
        base = node.visible_consumers[0]
        consumers = tuple(
            replace(base, pod_name=f"consumer-{index:02d}", workload_name=f"workload-{index:02d}")
            for index in range(24)
        )
        snapshot = replace(snapshot, nodes=(replace(node, consumers=consumers), *snapshot.nodes[1:]))

        class Collector:
            def collect(self, force=False):
                return snapshot

            def close(self):
                pass

        class Wheel:
            def prevent_default(self):
                pass

            def stop(self):
                pass

        app = FalconResourcesApp(Collector(), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            selected = app.state.selected_node
            pane = app.query_one("#node-pane", ResourcesPane)
            pane.on_mouse_scroll_down(Wheel())
            self.assertEqual(app.state.selected_node, selected)
            self.assertEqual(app.state.consumer_scroll, 1)

    async def test_resource_refresh_preserves_bottom_panel_scroll(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        node = snapshot.nodes[0]
        base = node.visible_consumers[0]
        consumers = tuple(
            replace(
                base,
                pod_name=f"consumer-{index:02d}",
                workload_name=f"workload-{index:02d}",
            )
            for index in range(20)
        )
        snapshot = replace(
            snapshot,
            nodes=(replace(node, consumers=consumers), *snapshot.nodes[1:]),
        )

        class Collector:
            def collect(self, force=False):
                return snapshot

            def close(self):
                pass

        app = FalconResourcesApp(Collector(), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            for _ in range(4):
                app.scroll_consumers(1)
            self.assertEqual(app.state.consumer_scroll, 4)
            app._results.put_nowait(snapshot)
            app._drain_results()
            await pilot.pause()
            self.assertEqual(app.state.consumer_scroll, 4)

    async def test_resource_refresh_preserves_consumer_anchor_when_order_changes(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        node = snapshot.nodes[0]
        base = node.visible_consumers[0]
        consumers = tuple(
            replace(
                base,
                pod_name=f"consumer-{index:02d}",
                workload_name=f"workload-{index:02d}",
            )
            for index in range(20)
        )
        initial = replace(
            snapshot,
            nodes=(replace(node, consumers=consumers), *snapshot.nodes[1:]),
        )
        reordered_consumers = consumers[:4] + (consumers[5], consumers[4]) + consumers[6:]
        reordered = replace(
            initial,
            nodes=(replace(initial.nodes[0], consumers=reordered_consumers), *initial.nodes[1:]),
        )

        class Collector:
            def collect(self, force=False):
                return initial

            def close(self):
                pass

        app = FalconResourcesApp(Collector(), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            for _ in range(4):
                app.scroll_consumers(1)
            anchor = app._sorted_consumers(app._selected())[app.state.consumer_scroll].pod_name
            app._results.put_nowait(reordered)
            app._drain_results()
            self.assertEqual(
                app._sorted_consumers(app._selected())[app.state.consumer_scroll].pod_name,
                anchor,
            )

    async def test_resource_refresh_preserves_selected_node(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            app.state.selected_node = app.nodes[-1].name
            app._render_all()
            selected = app.state.selected_node
            app._results.put_nowait(app.snapshot)
            app._drain_results()
            self.assertEqual(app.state.selected_node, selected)

    async def test_mouse_selection_and_enter_preserve_the_selected_node(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            # The first data row follows SIMPLE_HEAD's header and separator.
            # Click the second visible row.
            clicked = await pilot.click("#nodes-pane", offset=(5, 4))
            self.assertTrue(clicked)
            selected = app.state.selected_node
            self.assertEqual(selected, "node-b-a6000")
            await pilot.press("enter")
            await pilot.pause()
            self.assertTrue(app.state.expanded)
            self.assertEqual(app.state.selected_node, selected)

    async def test_expanded_consumer_rows_are_mouse_selectable(self) -> None:
        for size in ((140, 32), (80, 22)):
            app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
            async with app.run_test(size=size) as pilot:
                await pilot.pause(0.5)
                await pilot.press("enter")
                await pilot.pause()

                # The first consumer row starts after the capacity panel,
                # consumer panel border, spacer, header, and separator. The
                # click helper's terminal coordinate is one row ahead of the
                # content offset used by the hit-test.
                row_y = app._expanded_facts_height() + 6
                clicked = await pilot.click("#node-pane", offset=(8, row_y))
                await pilot.pause()
                self.assertTrue(clicked)
                self.assertEqual(app.state.selected_consumer, 1)
                self.assertEqual(app.state.active_pane, "node")

    async def test_resource_refresh_preserves_the_active_bottom_pane(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            clicked = await pilot.click("#node-pane", offset=(2, 2))
            self.assertTrue(clicked)
            self.assertEqual(app.state.active_pane, "node")
            self.assertIn("focused", app.query_one("#node-pane").border_title)
            self.assertNotIn("focused", app.query_one("#nodes-pane").border_title)

            app._results.put_nowait(app.snapshot)
            app._drain_results()
            await pilot.pause()
            self.assertEqual(app.state.active_pane, "node")
            self.assertIs(app.focused, app.query_one("#node-pane"))
            self.assertIn("focused", app.query_one("#node-pane").border_title)
            self.assertNotIn("focused", app.query_one("#nodes-pane").border_title)

    async def test_resource_single_click_wins_a_same_frame_refresh(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            detail = app.query_one("#node-pane", ResourcesPane)
            detail.on_click(object())
            self.assertEqual(app.state.active_pane, "node")

            app._results.put_nowait(app.snapshot)
            app._drain_results()
            await pilot.pause()
            self.assertEqual(app.state.active_pane, "node")
            self.assertIs(app.focused, detail)
            self.assertIn("focused", detail.border_title)

    async def test_resource_click_wins_a_focus_change_started_by_refresh(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            detail = app.query_one("#node-pane", ResourcesPane)

            # Put screen focus and view state temporarily out of sync so the
            # layout refresh has to restore the old state-owned pane. That
            # restoration must finish synchronously before the later click.
            app.set_focus(detail, scroll_visible=False)
            app.state.active_pane = "nodes"
            app._apply_layout(recompute_detail=False)
            app.set_focus(detail, scroll_visible=False)
            detail.on_mouse_down(object())
            await pilot.pause()

            self.assertEqual(app.state.active_pane, "node")
            self.assertIs(app.focused, detail)

    async def test_resource_refresh_keeps_highlight_off_while_app_is_blurred(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            await pilot.click("#node-pane", offset=(2, 2))
            self.assertEqual(app.state.active_pane, "node")

            app.app_focus = False
            await pilot.pause()
            self.assertIsNone(app.focused)
            self.assertNotIn("focused", app.query_one("#node-pane").border_title)

            app._results.put_nowait(app.snapshot)
            app._drain_results()
            await pilot.pause()
            self.assertIsNone(app.focused)
            self.assertEqual(app.state.active_pane, "node")
            self.assertNotIn("focused", app.query_one("#node-pane").border_title)

            app.app_focus = True
            await pilot.pause()
            self.assertIs(app.focused, app.query_one("#node-pane"))
            self.assertIn("focused", app.query_one("#node-pane").border_title)

    async def test_resource_focus_in_keeps_the_pane_clicked_while_blurred(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            app.app_focus = False
            await pilot.pause()

            detail = app.query_one("#node-pane", ResourcesPane)
            clicked = await pilot.click("#node-pane", offset=(2, 2))
            self.assertTrue(clicked)
            self.assertEqual(app.state.active_pane, "node")
            self.assertIs(app.focused, detail)

            app.app_focus = True
            await pilot.pause()
            self.assertTrue(app.app_focus)
            self.assertEqual(app.state.active_pane, "node")
            self.assertIs(app.focused, detail)
            self.assertIn("focused", detail.border_title)

    async def test_resource_stale_restored_focus_cannot_overwrite_mouse_down(self) -> None:
        app = FalconResourcesApp(DemoCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            nodes = app.query_one("#nodes-pane", ResourcesPane)
            detail = app.query_one("#node-pane", ResourcesPane)

            app.set_focus(detail, scroll_visible=False)
            detail.on_mouse_down(object())
            nodes.on_focus(events.Focus(from_app_focus=True))

            self.assertEqual(app.state.active_pane, "node")
            self.assertIs(app.focused, detail)

    async def test_expanded_consumers_stay_inside_the_visible_pane(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        node = snapshot.nodes[0]
        base = node.visible_consumers[0]
        consumers = tuple(
            replace(
                base,
                pod_name=f"consumer-{index:02d}",
                workload_name=f"workload-{index:02d}",
            )
            for index in range(30)
        )
        snapshot = replace(
            snapshot,
            nodes=(replace(node, consumers=consumers), *snapshot.nodes[1:]),
        )

        class Collector:
            def collect(self, force=False):
                return snapshot

            def close(self):
                pass

        app = FalconResourcesApp(Collector(), refresh_seconds=999)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.5)
            await pilot.press("enter")
            await pilot.pause()

            visible = app._visible_consumers()
            self.assertEqual(visible, 21)
            self.assertEqual(
                app._consumer_table(app._selected(), expanded=True).row_count,
                visible,
            )
            svg = app.export_screenshot(simplify=True)
            self.assertEqual(
                re.findall(r"workload-(\d\d)", svg),
                [f"{index:02d}" for index in range(visible)],
            )
            self.assertNotIn("OWNER", svg)

            await pilot.press("end")
            await pilot.resize_terminal(80, 22)
            await pilot.pause()
            self.assertEqual(app._visible_consumers(), 4)
            self.assertEqual(app.state.consumer_scroll, 26)
            svg = app.export_screenshot(simplify=True)
            self.assertIn("JOB", svg)
            self.assertIn("workl", svg)
            self.assertNotIn("POD", svg)
            self.assertIn("Quit", svg)


class VisualMatrixTests(unittest.IsolatedAsyncioTestCase):
    async def test_required_dimensions_export_valid_svg_without_overflow(self) -> None:
        manifest = _golden_manifest()
        self.assertEqual(
            manifest["dimensions"],
            [f"{width}x{height}" for width, height in DIMENSIONS],
        )
        for width, height in DIMENSIONS:
            app = FalconDashboard(
                DemoUsageCollector("many"),
                refresh_seconds=999,
                clock=lambda: "12:00:00",
            )
            async with app.run_test(size=(width, height)) as pilot:
                await pilot.pause(0.5)
                app._spinner = 0
                app._render_header()
                name = f"dashboard-many-{width}x{height}"
                svg = app.export_screenshot(
                    title=f"Falcon dashboard · {name}", simplify=True
                )
                self.assertTrue(svg.startswith("<svg"))
                self.assertNotIn("Traceback", svg)
                self.assertRegex(svg, r'width="[0-9.]+".*height="[0-9.]+"')
                if width < 80 or height < 22:
                    self.assertIn("requires&#160;at&#160;least", svg)
                else:
                    self.assertIn("Falcon&#160;Dashboard", svg)
                    self.assertIn("Quit", svg)
                actual = _golden_digest(svg)
                self.assertEqual(actual, manifest["snapshots"][name])

    async def test_named_dashboard_and_resource_states_match_goldens(self) -> None:
        manifest = _golden_manifest()["snapshots"]

        def search_long(app):
            app.state.search_query = "extraordinarily"
            app._filter_rows()
            app._render_all()

        def active_filter(app):
            app.state.filters["status"] = "Running"
            app._filter_rows()
            app._render_all()

        def filter_dialog(app):
            app.action_filters()

        def action_dialog(app):
            app.action_kill()

        dashboard_states = (
            ("dashboard-default", "mixed", (120, 30), ()),
            ("dashboard-selected-job", "mixed", (120, 30), ("4",)),
            ("dashboard-expanded-jobs", "many", (120, 30), ("enter",)),
            ("dashboard-expanded-selected", "mixed", (120, 30), ("4", "enter")),
            (
                "dashboard-expanded-resources",
                "mixed",
                (140, 32),
                ("2", "enter"),
            ),
            ("dashboard-expanded-events", "mixed", (120, 30), ("3", "enter")),
            (
                "dashboard-events-middle",
                "mixed",
                (80, 30),
                ("3", "home", "pagedown"),
            ),
            ("dashboard-search", "mixed", (100, 24), (search_long,)),
            ("dashboard-filters", "mixed", (100, 24), (active_filter,)),
            ("dashboard-filter-dialog", "mixed", (100, 24), (filter_dialog,)),
            ("dashboard-action-dialog", "mixed", (100, 24), (action_dialog,)),
            ("dashboard-stale", "stale", (100, 24), ()),
            ("dashboard-no-jobs", "no-jobs", (100, 24), ()),
            (
                "dashboard-long-content",
                "mixed",
                (80, 22),
                (search_long, "4", "enter"),
            ),
        )
        for name, state, size, actions in dashboard_states:
            app = FalconDashboard(
                DemoUsageCollector(state),
                refresh_seconds=999,
                clock=lambda: "12:00:00",
            )
            async with app.run_test(size=size) as pilot:
                await pilot.pause(0.5)
                for action in actions:
                    if callable(action):
                        action(app)
                        await pilot.pause()
                    else:
                        await pilot.press(*action.split())
                app._spinner = 0
                app._render_header()
                svg = app.export_screenshot(
                    title=f"Falcon dashboard · {name}", simplify=True
                )
            self.assertNotIn("Traceback", svg, name)
            self.assertEqual(_golden_digest(svg), manifest[name], name)

        resource_states = (
            ("resources-80x22", "mixed", (80, 22), ()),
            ("resources-140x32", "mixed", (140, 32), ()),
            ("resources-200x50", "mixed", (200, 50), ()),
            (
                "resources-gpu-allocations-80x22",
                "mixed",
                (80, 22),
                (_seed_resources_history, "right"),
            ),
            (
                "resources-gpu-allocations-140x32",
                "mixed",
                (140, 32),
                (_seed_resources_history, "right"),
            ),
            (
                "resources-gpu-allocations-200x50",
                "mixed",
                (200, 50),
                (_seed_resources_history, "right"),
            ),
            (
                "resources-gpu-allocations-vram-140x32",
                "mixed",
                (140, 32),
                (_seed_resources_history, "right", "v"),
            ),
            ("resources-node-expanded-80x22", "mixed", (80, 22), ("enter",)),
            (
                "resources-node-expanded-140x40",
                "mixed",
                (140, 40),
                ("enter",),
            ),
            ("resources-stale", "stale", (120, 30), ()),
            ("resources-no-jobs", "no-jobs", (120, 30), ()),
            (
                "resources-gpu-allocations-stale",
                "stale",
                (140, 32),
                (_seed_resources_history, "right"),
            ),
            (
                "resources-gpu-allocations-no-gpus",
                "no-gpus",
                (140, 32),
                ("right",),
            ),
        )
        for name, state, size, actions in resource_states:
            app = FalconResourcesApp(
                DemoCollector(state),
                refresh_seconds=999,
                clock=lambda snapshot: "12:00:00",
            )
            async with app.run_test(size=size) as pilot:
                await pilot.pause(0.5)
                for action in actions:
                    if callable(action):
                        action(app)
                        await pilot.pause()
                    else:
                        await pilot.press(*action.split())
                app._spinner = 0
                app._render_header()
                svg = app.export_screenshot(
                    title=f"Falcon resources · {name}", simplify=True
                )
            self.assertNotIn("Traceback", svg, name)
            self.assertEqual(_golden_digest(svg), manifest[name], name)

    async def test_long_content_never_leaks_markup_or_tracebacks(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause(0.5)
            app.state.search_query = "extraordinarily"
            app._filter_rows()
            app._render_all()
            svg = app.export_screenshot(simplify=True)
            self.assertNotIn("Traceback", svg)
            self.assertNotIn("[bold", svg)
            self.assertFalse(re.search(r"overflow-x", svg, re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()
