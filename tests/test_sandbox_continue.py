from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from falcon.cli import _parser
from falcon.config import DEFAULT_CONFIG
from falcon.sandbox_runtime import AutoContinue, codex_rate_limits, empty_codex_input, quota_reset
from falcon.sandbox_service import launch_session


def limits(primary=100, reset=200, secondary=0, secondary_reset=300):
    return {"rateLimits": {
        "primary": {"usedPercent": primary, "resetsAt": reset},
        "secondary": {"usedPercent": secondary, "resetsAt": secondary_reset},
    }}


class AutoContinueTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = {"agent": "codex", "argv": ["codex"],
                     "workspace": str(self.root), "record": str(self.root / "agent.json")}
        self.watcher = AutoContinue(self.spec)
        self.screen = "You've hit your usage limit. Try again later.\n› "
        self.attached = "0:0:0"
        self.sent = []
        self.running = {"state": "running", "session_id": "test-conversation"}

        def tmux(*args):
            if args[0] == "capture-pane":
                return self.screen
            if args[0] == "display-message":
                return "%0" if args[-1] == "#{pane_id}" else self.attached
            self.sent.append(args)
            return ""

        self.watcher.tmux = tmux
        self.query = patch("falcon.sandbox_runtime.codex_rate_limits", return_value=limits()).start()
        self.addCleanup(patch.stopall)

    def test_opt_in_parser(self):
        parser = _parser(DEFAULT_CONFIG)
        self.assertFalse(parser.parse_args(["sandbox", "--", "codex"]).auto_continue)
        self.assertTrue(parser.parse_args(["sandbox", "--auto-continue", "--", "codex"]).auto_continue)

    def test_other_agents_rejected_before_creating_resources(self):
        args = _parser(DEFAULT_CONFIG).parse_args(["sandbox", "--auto-continue", "--", "claude"])
        with patch("falcon.sandbox_service.discover_agent", return_value=("claude", Path("/tool"))):
            with self.assertRaisesRegex(ValueError, "Codex only"):
                launch_session(args, DEFAULT_CONFIG)

    def test_ordinary_limit_discussion_does_not_arm_watcher(self):
        self.screen = "The error means you've hit your usage limit.\n› "
        self.watcher.tick(self.running, 100)
        self.query.assert_not_called()

    def test_every_exhausted_window_must_reset(self):
        self.assertEqual(quota_reset(limits(secondary=100)), (True, 300))
        self.assertEqual(quota_reset(limits(primary=0)), (False, None))
        for value in ({}, {"rateLimits": {}}, limits(reset=None), limits(primary=float("nan"))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quota_reset(value)
        value = limits(primary=0)
        value["rateLimits"]["rateLimitReachedType"] = "credits"
        with self.assertRaises(ValueError):
            quota_reset(value)

    def test_send_once_after_confirmed_reset_and_survive_restart(self):
        self.watcher.tick(self.running, 100)
        self.watcher.tick(self.running, 209)
        self.assertEqual(self.sent, [])
        self.query.return_value = limits(primary=0)
        self.watcher.tick(self.running, 214)
        self.assertEqual(self.sent, [("send-keys", "-t", "%0", "-l", "continue"),
                                     ("send-keys", "-t", "%0", "Enter")])
        other = AutoContinue(self.spec)
        other.tmux = self.watcher.tmux
        self.query.return_value = limits()
        other.tick(self.running, 400)
        self.assertEqual(len(self.sent), 2)

    def test_attached_copy_mode_dead_pane_and_draft_block_input(self):
        self.watcher.tick(self.running, 100)
        self.query.return_value = limits(primary=0)
        for index, mode in enumerate(("1:0:0", "0:1:0", "0:0:1")):
            self.attached = mode
            self.watcher.tick(self.running, 220 + index * 60)
        self.attached = "0:0:0"
        self.screen += "unfinished command"
        self.watcher.tick(self.running, 500)
        self.assertEqual(self.sent, [])

    def test_exit_or_missing_session_never_queries(self):
        self.watcher.tick({"state": "exited", "session_id": "id"}, 100)
        self.watcher.tick({"state": "running"}, 100)
        self.query.assert_not_called()

    def test_manual_progress_cancels_pending_input(self):
        self.watcher.tick(self.running, 100)
        self.screen = "Working on your request\n› "
        self.watcher.tick(self.running, 220)
        self.assertIsNone(self.watcher.pending)
        self.assertEqual(self.sent, [])

    def test_failed_quota_check_and_still_exhausted_never_send(self):
        self.query.side_effect = ValueError("unavailable")
        self.watcher.tick(self.running, 100)
        self.assertIn("paused", self.watcher.state["status"])
        self.query.side_effect = None
        self.watcher.tick(self.running, 200)
        self.watcher.tick(self.running, 300)
        self.assertEqual(self.sent, [])

    def test_attachment_during_query_prevents_input(self):
        self.watcher.tick(self.running, 100)

        def attach(_):
            self.attached = "1:0:0"
            return limits(primary=0)

        self.query.side_effect = attach
        self.watcher.tick(self.running, 220)
        self.assertEqual(self.sent, [])

    def test_changed_pane_disables_input(self):
        self.watcher.tick(self.running, 100)
        previous = self.watcher.tmux

        def changed(*args):
            return "%1" if args[-1] == "#{pane_id}" else previous(*args)

        self.watcher.tmux = changed
        self.query.return_value = limits(primary=0)
        self.watcher.tick(self.running, 220)
        self.assertEqual(self.sent, [])
        self.assertIn("paused", self.watcher.state["status"])

    def test_next_reset_can_continue_again(self):
        self.watcher.tick(self.running, 100)
        self.query.return_value = limits(primary=0)
        self.watcher.tick(self.running, 220)
        self.screen = "Working\n› "
        self.watcher.tick(self.running, 300)
        self.screen = "■ You've hit your usage limit.\n› \x1b[38;5;8mImplement a feature\x1b[39m"
        self.query.return_value = limits(reset=500)
        self.watcher.tick(self.running, 310)
        self.query.return_value = limits(primary=0)
        self.watcher.tick(self.running, 520)
        self.assertEqual(len(self.sent), 4)

    def test_placeholder_is_not_a_user_draft(self):
        for value in ("› ", "› \x1b[2mFind a bug\x1b[0m",
                      "› \x1b[38;5;8mImplement a feature\x1b[39m"):
            self.assertTrue(empty_codex_input(value), value)
        for value in ("No composer", "› draft", "› \x1b[2mplaceholder\x1b[0m draft"):
            self.assertFalse(empty_codex_input(value), value)

    def test_rpc_handshake_uses_only_read_requests(self):
        patch.stopall()
        executable = self.root / "fake-codex"
        output = self.root / "requests.jsonl"
        executable.write_text(
            "#!/usr/bin/env python3\nimport json,sys\n"
            f"log=open({str(output)!r},'w')\n"
            "for line in sys.stdin:\n"
            " value=json.loads(line); log.write(line); log.flush()\n"
            " if 'id' not in value: continue\n"
            f" result={limits(primary=0)!r} if value['id']==2 else {{}}\n"
            " print(json.dumps({'id':value['id'],'result':result}),flush=True)\n"
        )
        executable.chmod(0o700)
        self.spec["argv"] = [str(executable), "-c", 'sqlite_home="/tmp/local"']
        self.assertEqual(codex_rate_limits(self.spec), limits(primary=0))
        requests = [json.loads(line)["method"] for line in output.read_text().splitlines()]
        self.assertEqual(requests, ["initialize", "initialized", "account/rateLimits/read"])


if __name__ == "__main__":
    unittest.main()
