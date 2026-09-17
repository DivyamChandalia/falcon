from __future__ import annotations

import io
import time
import unittest
from unittest.mock import patch

from falcon.dashboard import JobUsage, PodAttempt
from falcon.dashboard_logs import DashboardLogManager
from falcon.dashboard_ui import FalconDashboard
from falcon.kubernetes import ProcessResult


class _Process:
    def __init__(self, output: str) -> None:
        self.stdout = io.StringIO(output)
        self._finished = False

    def poll(self):
        return 0 if self._finished else None

    def wait(self, timeout=None):
        self._finished = True
        return 0

    def terminate(self):
        self._finished = True

    def kill(self):
        self._finished = True


class _Client:
    def __init__(self) -> None:
        self.attach_calls = []
        self.log_calls = []

    def attach_stream(self, pod_name, **kwargs):
        self.attach_calls.append((pod_name, kwargs))
        return _Process("\n".join(f"live-{index}" for index in range(80)) + "\n")

    def pod_logs(self, pod_name, **kwargs):
        self.log_calls.append((pod_name, kwargs))
        return ProcessResult(
            ("kubectl", "logs", pod_name),
            0,
            "\n".join(f"terminal-{index}" for index in range(80)) + "\n",
        )


class _Collector:
    def __init__(self, row: JobUsage) -> None:
        self.row = row
        self.last_error = ""
        self.last_successful_refresh = time.time()
        self.gpu_availability = {}

    def collect(self):
        return [self.row]

    def events(self, row):
        return []

    def close(self):
        return None


def _job(*attempts: PodAttempt, status: str = "Running") -> JobUsage:
    active = next(
        (attempt for attempt in reversed(attempts) if attempt.running),
        attempts[-1] if attempts else None,
    )
    return JobUsage(
        job="training-job",
        status=status,
        nodes="node-a",
        gpu_type="h100",
        gpu_count=1,
        pod_count=len(attempts),
        gpu_util=42.0,
        gpu_ema=42.0,
        gpu_memory_used_gib=12.0,
        gpu_memory_total_gib=80.0,
        cpu_used=1.0,
        cpu_requested=4.0,
        memory_used_gib=2.0,
        memory_requested_gib=16.0,
        age="3m",
        at_risk=False,
        uid="job-uid",
        active_pod=active.name if active else "",
        active_pod_uid=active.uid if active else "",
        active_pod_state=active.phase if active else "No active pod",
        command="python -u train.py " + " ".join(f"--arg-{i}" for i in range(30)),
        gpu_requested_type="h100",
        gpu_requested_count=1,
        gpu_allocated_type="h100",
        gpu_allocated_count=1,
        cpu_allocated=4.0,
        memory_allocated_gib=16.0,
        pod_attempts=len(attempts),
        succeeded_attempts=sum(attempt.phase == "Succeeded" for attempt in attempts),
        failed_attempts=sum(attempt.phase == "Failed" for attempt in attempts),
        attempt_pods=[attempt.name for attempt in attempts],
        attempt_details=tuple(attempts),
        image="registry.example/train:latest",
    )


class DashboardInspectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_running_output_is_attached_and_sections_are_independent(self) -> None:
        row = _job(
            PodAttempt("old-attempt", "old-uid", "Succeeded"),
            PodAttempt("current-attempt", "current-uid", "Running"),
        )
        client = _Client()
        manager = DashboardLogManager("team", client=client)
        app = FalconDashboard(
            _Collector(row),
            refresh_seconds=999,
            log_manager=manager,
            launch_config={
                "runtime": {"image": "registry.example/train:latest"},
                "presets": {"h100": {"gpu_type": "h100"}},
            },
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.6)
            self.assertEqual([name for name, _ in client.attach_calls], ["current-attempt"])
            await pilot.press("4", "enter")
            await pilot.pause(0.2)

            command = app.query_one("#selected-command-row")
            command_label = app.query_one("#selected-command-label")
            command_copy = app.query_one("#selected-command-copy")
            details_left = app.query_one("#selected-details-left")
            details_right = app.query_one("#selected-details-right")
            logs = app.query_one("#selected-logs-scroll")
            self.assertEqual(app.query_one("#selected-command-label").content, "Command")
            self.assertFalse(command.can_focus)
            self.assertFalse(command_copy.can_focus)
            self.assertEqual(command_label.region.x, details_left.region.x)
            # The icon's glyph has one cell of Button padding, placing it at
            # the same value-column origin as the RAM and Age values.
            self.assertEqual(command_copy.region.x + 1, details_left.region.x + 19)
            details_right_x = details_right.region.x
            self.assertGreater(logs.max_scroll_y, 0)
            self.assertEqual(int(logs.scroll_y), int(logs.max_scroll_y))
            self.assertTrue(app.state.logs_auto_follow)
            self.assertEqual(
                app.query_one("#selected-command-label").content,
                "Command",
            )
            self.assertIn("LOGS", logs.border_title)
            self.assertIn("SELECTED JOB", app.query_one("#selected-pane").border_title)
            self.assertIn("current-attempt", app.query_one("#selected-pane").border_subtitle)
            self.assertIn("live-79", app.export_screenshot(simplify=True))

            # A terminal reconnect can leave the app's remembered top-level
            # focus on Jobs while the selected inspector is still expanded.
            # The command label is metadata rather than a selectable pane;
            # clicking it must not copy or steal the Logs selection.
            app.state.focused_pane = "jobs"
            with patch.object(app, "copy_to_clipboard") as copy:
                await pilot.click("#selected-command-label")
            self.assertFalse(copy.called)
            self.assertFalse(command.can_focus)
            self.assertEqual(app.state.selected_section, "logs")
            self.assertEqual(details_right.region.x, details_right_x)
            logs_content = app.query_one("#selected-logs-content")
            # The selected log content is taller than its viewport and is
            # scrolled to the tail, so click a visible line near its bottom.
            await pilot.click(
                "#selected-logs-content",
                offset=(10, max(0, logs_content.region.height - 2)),
            )
            self.assertIs(app.screen.focused, logs)
            app.selected_section_focused("selected-logs-scroll")
            self.assertEqual(app.state.focused_pane, "selected")
            self.assertIn("selected", logs.classes)
            self.assertNotIn("selected", command.classes)
            await pilot.press("home")
            await pilot.pause(0.1)
            self.assertEqual(int(logs.scroll_y), 0)
            self.assertFalse(app.state.logs_auto_follow)
            await pilot.press("down")
            self.assertGreater(logs.scroll_y, 0)
            self.assertFalse(app.state.logs_auto_follow)
            self.assertEqual(int(command.scroll_y), 0)

            await pilot.press("c")
            self.assertIn("collapsed", logs.classes)
            self.assertTrue(logs.display)
            self.assertEqual(logs.size.height, 2)
            await pilot.press("c")
            self.assertNotIn("collapsed", logs.classes)

            with patch.object(app, "copy_to_clipboard") as copy:
                await pilot.click("#selected-command-copy")
                await pilot.pause(0.05)
            copied = copy.call_args.args[0]
            self.assertTrue(copied.startswith("falcon h100 -- "))
            self.assertNotIn("--name", copied)
            self.assertNotIn("--gpus", copied)
            self.assertNotIn("--image", copied)
            self.assertEqual(details_right.region.x, details_right_x)

            self.assertIs(app.screen.focused, logs)

            with patch.object(app, "copy_to_clipboard") as copy:
                await pilot.click("#selected-logs-copy")
            copied = copy.call_args.args[0]
            self.assertIn("live-79", copied)

            # The command label remains non-interactive; only its adjacent
            # icon copies the reconstructed invocation.
            self.assertFalse(command.can_focus)

            await pilot.press("left")
            await pilot.pause(0.2)
            deadline = time.monotonic() + 1
            while not client.log_calls and time.monotonic() < deadline:
                await pilot.pause(0.02)
            self.assertEqual([name for name, _ in client.log_calls], ["old-attempt"])

    async def test_succeeded_job_uses_terminal_falcon_logs_capture(self) -> None:
        row = _job(PodAttempt("completed-attempt", "completed-uid", "Succeeded"), status="Succeeded")
        client = _Client()
        app = FalconDashboard(
            _Collector(row),
            refresh_seconds=999,
            log_manager=DashboardLogManager("team", client=client),
        )
        async with app.run_test(size=(120, 32)) as pilot:
            await pilot.pause(0.5)
            self.assertEqual(client.attach_calls, [])
            await pilot.press("4", "enter")
            deadline = time.monotonic() + 1
            while not client.log_calls and time.monotonic() < deadline:
                await pilot.pause(0.02)
            self.assertEqual([name for name, _ in client.log_calls], ["completed-attempt"])
            self.assertEqual(client.log_calls[0][1]["tail"], 200)

    def test_reconstructed_command_uses_shorthand_and_omits_defaults(self) -> None:
        row = _job(PodAttempt("completed-attempt", "completed-uid", "Succeeded"))
        row.gpu_requested_count = 2
        row.command = "python -m train --epochs 3"
        app = FalconDashboard(
            _Collector(row),
            launch_config={
                "runtime": {"image": "registry.example/train:latest"},
                "presets": {"h100": {"gpu_type": "h100"}},
            },
        )
        self.assertEqual(
            app._reconstructed_command(row),
            "falcon h100x2 -- python -m train --epochs 3",
        )

        row.gpu_requested_count = 0
        row.gpu_requested_type = "-"
        row.cpu_requested = 2.0
        row.memory_requested_gib = 4.0
        self.assertEqual(
            app._reconstructed_command(row),
            "falcon --cpu 2 --memory 4Gi -- python -m train --epochs 3",
        )

        row.gpu_requested_type = "mystery-gpu"
        row.gpu_requested_count = 1
        row.image = "registry.example/other:latest"
        row.command = "echo hi"
        self.assertEqual(
            app._reconstructed_command(row),
            "falcon --gpu mystery-gpu --image registry.example/other:latest -- echo hi",
        )


if __name__ == "__main__":
    unittest.main()
