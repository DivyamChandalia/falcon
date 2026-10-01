from __future__ import annotations

import shutil
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from test_dashboard_inspector import _Collector, _job
from textual import events

from falcon.dashboard import PodAttempt
from falcon.dashboard_terminal import (
    DashboardTerminal,
    TerminalSession,
    exec_command,
    key_bytes,
    terminal_shell_rc,
)
from falcon.dashboard_ui import FalconDashboard


class TerminalTests(unittest.TestCase):
    def test_exec_argv_targets_container_and_safe_shell_fallback(self):
        command = exec_command("team", "pod-1", "trainer", "/bin/zsh", "/usr/bin/kubectl")
        self.assertEqual(
            command[:8],
            [
                "/usr/bin/kubectl",
                "exec",
                "--namespace",
                "team",
                "pod-1",
                "--container",
                "trainer",
                "--stdin",
            ],
        )
        self.assertIn("--tty", command)
        self.assertEqual(command[-3], "zsh")
        self.assertIn('"$1" bash zsh sh', command[-5])
        self.assertIn("ZDOTDIR", command[-5])
        self.assertIn('source "$FALCON_USER_RC"', command[-5])
        self.assertEqual(exec_command("team", "pod", "", "unsafe; echo bad")[-3], "bash")

    def test_terminal_keys(self):
        for key, expected in (
            ("ctrl+c", b"\x03"),
            ("enter", b"\r"),
            ("tab", b"\t"),
            ("alt+left", b"\x1b[D"),
            ("alt+right", b"\x1b[C"),
            ("backspace", b"\x7f"),
        ):
            self.assertEqual(key_bytes(key), expected)
        self.assertEqual(key_bytes("x", "λ"), "λ".encode())

    def test_terminal_prompts_omit_hostname_and_keep_working_directory(self):
        self.assertIn("PROMPT='%1/%% '", terminal_shell_rc("zsh"))
        self.assertIn('RPROMPT=""', terminal_shell_rc("zsh"))
        self.assertIn(r"PS1='\W% '", terminal_shell_rc("bash"))

    def test_debug_bootstrap_starts_available_shells(self):
        for shell in ("bash", "zsh"):
            if not shutil.which(shell):
                continue
            with self.subTest(shell=shell):
                command = exec_command(
                    "team", "pod", "main", shell, rc_path="/dev/null", prompt_label="test-debug"
                )
                session = TerminalSession(command[command.index("--") + 1 :], width=80, height=8)
                try:
                    deadline = time.monotonic() + 5
                    prompt = Path.cwd().name + "%"
                    while (
                        prompt not in session.render(cursor=False).plain
                        and time.monotonic() < deadline
                    ):
                        session.poll()
                        time.sleep(0.01)
                    self.assertIn(prompt, session.render(cursor=False).plain)
                    self.assertNotIn("(test-debug)", session.render(cursor=False).plain)
                    expected = "%1/%% " if shell == "zsh" else r"\W% "
                    session.send(b"printf '%s%s%s\\n' PROMPT_ VALUE: \"$PROMPT$PS1\"\r")
                    while (
                        "PROMPT_VALUE:" + expected not in session.render(cursor=False).plain
                        and time.monotonic() < deadline
                    ):
                        session.poll()
                        time.sleep(0.01)
                    self.assertIn("PROMPT_VALUE:" + expected, session.render(cursor=False).plain)
                    session.send(b"cd /tmp\r")
                    while (
                        "tmp%" not in session.render(cursor=False).plain
                        and time.monotonic() < deadline
                    ):
                        session.poll()
                        time.sleep(0.01)
                    self.assertIn("tmp%", session.render(cursor=False).plain)
                    session.send(b"printf 'SHELL_OK:%s\\n' \"$ZSH_VERSION$BASH_VERSION\"\r")
                    while (
                        "SHELL_OK:" not in session.render(cursor=False).plain
                        and time.monotonic() < deadline
                    ):
                        session.poll()
                        time.sleep(0.01)
                    self.assertIn("SHELL_OK:", session.render(cursor=False).plain)
                finally:
                    session.close()

    def test_scrollback_is_bounded_and_cursor_is_visible(self):
        session = TerminalSession(
            [sys.executable, "-c", "import time; time.sleep(10)"], width=20, height=4
        )
        self.addCleanup(session.close)
        session.stream.feed("\r\n".join(f"line{i:03d}" for i in range(300)))
        rendered = session.render(scrollback=True)
        self.assertEqual(len(rendered.plain.splitlines()), 200)
        self.assertIn("line100", rendered.plain)
        self.assertNotIn("line099", rendered.plain)
        self.assertIn("line299", rendered.plain)
        session.screen.cursor.hidden = True
        self.assertTrue(
            any(
                span.style.bgcolor and span.style.bgcolor.name == "#00d7ff"
                for span in session.render().spans
            )
        )
        session.resize(20, 2)
        self.assertIn("line100", session.render(scrollback=True).plain)

    def test_real_pty_input_output_resize_and_reaping(self):
        session = TerminalSession(
            [
                sys.executable,
                "-u",
                "-c",
                "import sys; print('ready', flush=True); text=sys.stdin.readline(); print('received:'+text, flush=True)",
            ],
            width=40,
            height=8,
        )
        self.addCleanup(session.close)
        deadline = time.monotonic() + 3
        while "ready" not in "\n".join(session.screen.display) and time.monotonic() < deadline:
            session.poll()
            time.sleep(0.01)
        self.assertIn("ready", "\n".join(session.screen.display))
        session.resize(50, 10)
        self.assertEqual((session.screen.columns, session.screen.lines), (50, 10))
        session.send(b"hello\n")
        while session.returncode is None and time.monotonic() < deadline:
            session.poll()
            time.sleep(0.01)
        self.assertEqual(session.returncode, 0)
        self.assertIn("received:hello", session.render(cursor=False).plain)
        session.close()
        session.close()
        self.assertTrue(session.closed)

    def test_vt_progress_unicode_colours_and_clear(self):
        session = TerminalSession(
            [sys.executable, "-c", "import time; time.sleep(10)"], width=20, height=4
        )
        self.addCleanup(session.close)
        session.stream.feed("progress 10%\rprogress 20%\x1b[31m λ\x1b[0m")
        rendered = session.render(cursor=False)
        self.assertIn("progress 20% λ", rendered.plain)
        self.assertNotIn("10%", rendered.plain)
        self.assertTrue(rendered.spans)
        session.stream.feed("\x1b[2J\x1b[Hclean")
        self.assertIn("clean", session.render(cursor=False).plain)
        self.assertNotIn("progress", session.render(cursor=False).plain)


class TerminalDashboardTests(unittest.IsolatedAsyncioTestCase):
    async def test_multiple_terminals_input_navigation_and_cleanup(self):
        row = _job(PodAttempt("pod-live", "pod-uid", "Running", container="trainer"))
        collector = _Collector(row)
        collector.namespace = "team"
        sessions = []
        commands = []

        def factory(command, **kwargs):
            commands.append(command)
            session = TerminalSession(["/bin/bash", "--noprofile", "--norc", "-i"], **kwargs)
            sessions.append(session)
            return session

        app = FalconDashboard(collector, refresh_seconds=999, terminal_factory=factory)
        with patch("falcon.config.detect_shell", return_value=("zsh", None)):
            async with app.run_test(size=(160, 40)) as pilot:
                await pilot.pause(0.3)
                app.action_focus_selected()
                await pilot.press("ctrl+t")
                await pilot.pause(0.2)
                self.assertEqual(len(sessions), 1)
                self.assertEqual(commands[0][-3], "zsh")
                self.assertIn("trainer", commands[0])
                self.assertIsInstance(app.focused, DashboardTerminal)
                terminal = app.query_one("#selected-terminal", DashboardTerminal)
                self.assertIn("Ctrl+D Close terminal", app.query_one("#falcon-footer").render().plain)
                await pilot.click("#selected-details-left", offset=(1, 0))
                self.assertIs(app.focused, app.query_one("#selected-pane"))
                footer = app.query_one("#falcon-footer").render().plain
                self.assertIn("k Kill", footer)
                self.assertIn("Tab Next pane", footer)
                self.assertNotIn("Ctrl+D Close terminal", footer)
                with patch.object(sessions[0], "send") as send:
                    await pilot.press("k")
                    self.assertEqual(type(app.screen).__name__, "KillDialog")
                    send.assert_not_called()
                    await pilot.press("escape")
                    await pilot.press("tab")
                    self.assertNotEqual(app.state.focused_pane, "selected")
                    send.assert_not_called()
                app.action_focus_selected()
                self.assertIs(app.focused, app.query_one("#selected-pane"))
                await pilot.click(terminal, offset=(2, 1))
                self.assertIs(app.focused, terminal)
                self.assertIn("Ctrl+D Close terminal", app.query_one("#falcon-footer").render().plain)
                sessions[0].screen.reset()
                sessions[0].stream.feed("alpha bravo charlie\r\nsecond line")
                sessions[0].revision += 1
                await pilot.pause(0.3)
                content = app.query_one("#selected-terminal-screen")
                await pilot.mouse_down(content, offset=(0, 0))
                await pilot._post_mouse_events(
                    [events.MouseMove], widget=content, offset=(8, 0), button=1
                )
                await pilot.mouse_up(content, offset=(8, 0))
                self.assertEqual(terminal.selection_for_copy(), "alpha br")
                with (
                    patch.object(app, "copy_to_clipboard") as copy,
                    patch.object(sessions[0], "send") as send,
                ):
                    await pilot.press("ctrl+c")
                    copy.assert_called_once_with("alpha br")
                    send.assert_not_called()
                app.screen.clear_selection()
                await pilot.click(terminal, offset=(2, 1))
                self.assertIs(app.focused, terminal)
                collapsed = app.state.logs_collapsed
                with patch.object(sessions[0], "send", wraps=sessions[0].send) as send:
                    await pilot.press("c", "tab", "up")
                    self.assertEqual(
                        [call.args[0] for call in send.call_args_list], [b"c", b"\t", b"\x1b[A"]
                    )
                self.assertIs(app.focused, terminal)
                self.assertEqual(app.state.logs_collapsed, collapsed)
                await pilot.press("ctrl+c")
                await pilot.press("p", "r", "i", "n", "t", "f", "space", "o", "k", "enter")
                await pilot.pause(0.2)
                self.assertIn("ok", sessions[0].render(cursor=False).plain)
                self.assertNotEqual(type(app.screen).__name__, "KillDialog")
                await pilot.press("ctrl+t")
                await pilot.pause(0.2)
                self.assertEqual(len(sessions), 2)
                self.assertIs(app._active_terminal(), sessions[1])
                await pilot.press("left")
                self.assertIs(app._active_terminal(), sessions[0])
                await pilot.press("left")
                self.assertIsNone(app._active_terminal())
                self.assertEqual(app.query_one("#selected-logs-copy").tooltip, "Ctrl/Cmd+C")
                await pilot.press("right")
                self.assertIs(app._active_terminal(), sessions[0])
                self.assertEqual(
                    app.query_one("#selected-logs-copy").tooltip, "Copy terminal output"
                )
                await pilot.press("ctrl+c")
                await pilot.pause(0.2)
                self.assertTrue(app.is_running)
                self.assertIsInstance(app.focused, DashboardTerminal)
                app.focused.on_paste(events.Paste("printf pasted\n"))
                await pilot.press("enter")
                await pilot.pause(0.3)
                self.assertIn("pasted", sessions[0].render(cursor=False).plain)
                sessions[0].stream.feed("\r\n".join(f"scroll{i:03d}" for i in range(300)))
                sessions[0].revision += 1
                await pilot.pause(0.3)
                self.assertGreater(terminal.max_scroll_y, 0)
                terminal.scroll_to(y=50, animate=False, force=True, immediate=True)
                await pilot.pause()
                sessions[0].stream.feed("\r\n".join([""] + [f"more{i}" for i in range(10)]))
                sessions[0].revision += 1
                await pilot.pause(0.3)
                self.assertAlmostEqual(terminal.scroll_y, 40)
                with patch.object(app, "copy_to_clipboard") as copy:
                    await pilot.click("#selected-logs-copy")
                    self.assertEqual(copy.call_count, 1)
                    copied = copy.call_args.args[0]
                    self.assertIn("scroll150", copied)
                    self.assertIn("more9", copied)
                    self.assertNotIn("scroll000", copied)
                    self.assertNotIn("\x1b", copied)
                    self.assertTrue(all(line == line.rstrip() for line in copied.splitlines()))
                await pilot.press("c")
                await pilot.pause()
                self.assertAlmostEqual(terminal.scroll_y, terminal.max_scroll_y)
                await pilot.resize_terminal(100, 30)
                await pilot.pause()
                self.assertLessEqual(sessions[0].screen.columns, 100)
                await pilot.resize_terminal(80, 22)
                await pilot.pause()
                terminal = app.query_one("#selected-terminal", DashboardTerminal)
                self.assertGreater(terminal.content_size.height, 0)
                self.assertLessEqual(terminal.region.bottom, app.size.height - 1)
                with patch.object(sessions[0], "send", wraps=sessions[0].send) as send:
                    await pilot.press("ctrl+g")
                    send.assert_called_with(b"\x07")
                self.assertIsInstance(app.focused, DashboardTerminal)
                self.assertEqual(app.state.focused_pane, "selected")
                await pilot.press("ctrl+w")
                self.assertTrue(sessions[0].closed)
                self.assertIsNone(app._active_terminal())
                await pilot.press("right")
                self.assertIs(app._active_terminal(), sessions[1])
                self.assertIsInstance(app.focused, DashboardTerminal)
                await pilot.press("ctrl+d")
                self.assertTrue(sessions[1].closed)
                self.assertIsNone(app._active_terminal())
                self.assertTrue(app.is_running)
        self.assertTrue(all(session.closed for session in sessions))

    async def test_completed_pod_does_not_open_exec(self):
        collector = _Collector(_job(PodAttempt("done", "uid", "Succeeded"), status="Succeeded"))
        collector.namespace = "team"
        with patch("falcon.dashboard_terminal.subprocess.Popen") as spawn:
            app = FalconDashboard(collector, refresh_seconds=999)
            async with app.run_test(size=(160, 40)) as pilot:
                await pilot.pause(0.3)
                app.action_focus_selected()
                await pilot.press("ctrl+t")
                spawn.assert_not_called()
                self.assertIsNone(app._active_terminal())
