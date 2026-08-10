from __future__ import annotations

import hashlib
import json
import os
import re
import unittest
from dataclasses import replace
from pathlib import Path

from textual import events

# Golden rendering is intentionally truecolor and independent of the parent
# test runner's NO_COLOR setting.
os.environ.pop("NO_COLOR", None)

from falcon.dashboard import DemoUsageCollector, JobEvent
from falcon.dashboard_ui import (
    DashboardPane,
    FalconDashboard,
    _restart_job_manifest,
)
from falcon.demo import DemoCollector, demo_cluster_snapshot
from falcon.resources_ui import (
    FalconResourcesApp,
    ResourcesPane,
    _gpu_headroom_color,
    _resource_headroom_color,
    _short_cpu,
)
from falcon.theme import GRAY, GREEN, MUTED, RED, YELLOW

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


class DashboardInteractionTests(unittest.IsolatedAsyncioTestCase):
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
            clicked = await pilot.click("#jobs-pane", offset=(5, 5))
            self.assertTrue(clicked)
            self.assertEqual(app.state.cursor_job_uid, expected)

    async def test_clicking_any_dashboard_pane_marks_it_active(self) -> None:
        app = FalconDashboard(DemoUsageCollector("mixed"), refresh_seconds=999)
        async with app.run_test(size=(140, 32)) as pilot:
            await pilot.pause(0.5)
            for pane in ("selected", "resources", "events", "jobs"):
                clicked = await pilot.click(f"#{pane}-pane", offset=(2, 2))
                self.assertTrue(clicked)
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
            clicked = await pilot.click("#resources-pane", offset=(2, 2))
            self.assertTrue(clicked)
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
            svg = app.export_screenshot(simplify=True)
            self.assertIn("Job&#160;Actions", svg)
            await pilot.press("escape")
            self.assertFalse(app.state.kill_dialog["isOpen"])

            app.action_cleanup()
            await pilot.pause()
            self.assertEqual(type(app.screen).__name__, "CleanupDialog")
            await pilot.press("escape")

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
            self.assertIn("12:00:00  ◴", header.plain)
            self.assertNotIn("\n", overview.plain)
            self.assertLess(overview.plain.index("CPU"), overview.plain.index("MEM"))
            self.assertLess(overview.plain.index("MEM"), overview.plain.index("GPU"))
            running_style = overview.get_style_at_offset(overview.plain.index("RUN"))
            self.assertTrue(running_style.bold)
            self.assertEqual(running_style.foreground.hex6, GREEN)
            for label in ("CPU", "MEM", "GPU"):
                label_style = overview.get_style_at_offset(overview.plain.index(label))
                self.assertTrue(label_style.bold)
                self.assertEqual(label_style.foreground.hex6, GRAY)

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
            self.assertEqual(app.query_one("#nodes-pane").region.height, 15)
            self.assertEqual(app.query_one("#node-pane").region.height, 7)
            await pilot.click("#node-pane", offset=(2, 2))
            self.assertEqual(app.state.active_pane, "node")
            await pilot.resize_terminal(140, 24)
            await pilot.pause()
            self.assertTrue(app._detail_auto_hidden)
            self.assertFalse(app.query_one("#node-pane").display)
            self.assertEqual(app.query_one("#nodes-pane").region.height, 19)
            self.assertEqual(app.state.active_pane, "nodes")
            self.assertIs(app.focused, app.query_one("#nodes-pane"))

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
            anchor = app._selected().visible_consumers[app.state.consumer_scroll].pod_name
            app._results.put_nowait(reordered)
            app._drain_results()
            self.assertEqual(
                app._selected().visible_consumers[app.state.consumer_scroll].pod_name,
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
            # The first data row follows SIMPLE_HEAD's blank top line,
            # header, and separator.  Click the second visible row.
            clicked = await pilot.click("#nodes-pane", offset=(5, 5))
            self.assertTrue(clicked)
            selected = app.state.selected_node
            self.assertEqual(selected, "node-b-a6000")
            await pilot.press("enter")
            await pilot.pause()
            self.assertTrue(app.state.expanded)
            self.assertEqual(app.state.selected_node, selected)

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
            self.assertEqual(visible, 22)
            self.assertEqual(
                app._consumer_table(app._selected(), expanded=True).row_count,
                visible,
            )
            svg = app.export_screenshot(simplify=True)
            self.assertEqual(
                re.findall(r"consumer-(\d\d)", svg),
                [f"{index:02d}" for index in range(visible)],
            )
            self.assertNotIn("OWNER", svg)

            await pilot.press("end")
            await pilot.resize_terminal(80, 22)
            await pilot.pause()
            self.assertEqual(app._visible_consumers(), 5)
            self.assertEqual(app.state.consumer_scroll, 25)
            svg = app.export_screenshot(simplify=True)
            self.assertEqual(
                re.findall(r"consumer-(\d\d)", svg),
                ["25", "26", "27", "28", "29"],
            )
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
            ("resources-node-expanded-80x22", "mixed", (80, 22), ("enter",)),
            (
                "resources-node-expanded-140x40",
                "mixed",
                (140, 40),
                ("enter",),
            ),
            ("resources-stale", "stale", (120, 30), ()),
            ("resources-no-jobs", "no-jobs", (120, 30), ()),
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
