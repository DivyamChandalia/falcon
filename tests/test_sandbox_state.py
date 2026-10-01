from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from falcon.sandbox_state import (
    CODEX_RUNTIME_HOME,
    MAX_STATE_FILE_BYTES,
    CodexStateSync,
    allowed_file,
    read_file,
    seed_state,
)


class CodexStateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "host"
        self.source.mkdir()
        self.stage = self.root / "copy"
        self.baseline = self.root / "trusted" / "baseline.json"

    def seed(self):
        seed_state(self.source, self.stage, self.baseline)
        return CodexStateSync(self.source, self.stage, self.baseline)

    def test_seed_filters_runtime_and_database_state(self):
        self.stage.mkdir()
        for name in ("config.toml", "auth.json", "history.jsonl", "custom.config.toml", "state_5.sqlite", "state_5.sqlite-wal", "daemon.sock"):
            (self.source / name).write_text("test\n")
        (self.source / "app-server-daemon").mkdir()
        (self.source / "app-server-daemon" / "socket").write_text("host IPC")
        (self.source / "skills").mkdir()
        (self.source / "skills" / "SKILL.md").write_text("skill")
        plugin = self.source / "plugins/cache/tool"
        plugin.parent.mkdir(parents=True)
        plugin.write_text("#!/bin/sh\nexit 0\n")
        plugin.chmod(0o755)
        (plugin.parent / "private.sqlite").write_text("private")
        self.seed()
        for name in ("config.toml", "auth.json", "history.jsonl", "custom.config.toml"):
            self.assertEqual((self.stage / name).read_text(), "test\n")
            self.assertEqual((self.stage / name).stat().st_mode & 0o777, 0o600)
        for name in ("state_5.sqlite", "state_5.sqlite-wal", "daemon.sock"):
            self.assertFalse((self.stage / name).exists())
        self.assertEqual(os.readlink(self.stage / "app-server-daemon"), f"{CODEX_RUNTIME_HOME}/app-server-daemon")
        self.assertEqual((self.stage / "skills" / "SKILL.md").read_text(), "skill")
        self.assertTrue((self.stage / "plugins/cache/tool").stat().st_mode & 0o100)
        self.assertFalse((self.stage / "plugins/cache/private.sqlite").exists())
        self.assertEqual((self.source / "app-server-daemon" / "socket").read_text(), "host IPC")

    def test_sync_settings_and_credentials_both_directions(self):
        (self.source / "config.toml").write_text("old")
        (self.source / "auth.json").write_text("old login")
        sync = self.seed()
        (self.stage / "config.toml").write_text("new")
        (self.source / "auth.json").write_text("new login")
        self.assertEqual(sync.sync(), [])
        self.assertEqual((self.source / "config.toml").read_text(), "new")
        self.assertEqual((self.stage / "auth.json").read_text(), "new login")
        self.assertEqual(sync.sync(), [])

    def test_concurrent_settings_changes_preserve_both_versions(self):
        (self.source / "config.toml").write_text("old")
        sync = self.seed()
        (self.source / "config.toml").write_text("host change")
        (self.stage / "config.toml").write_text("sandbox change")
        self.assertEqual(sync.sync(), ["config.toml"])
        self.assertEqual((self.source / "config.toml").read_text(), "host change")
        self.assertEqual((self.stage / "config.toml").read_text(), "sandbox change")

    def test_new_rollout_is_synced_and_restart_is_idempotent(self):
        sync = self.seed()
        rollout = self.stage / "sessions/2026/09/30/rollout-example.jsonl"
        rollout.parent.mkdir(parents=True)
        rollout.write_text('{"type":"session_meta"}\n')
        self.assertEqual(sync.sync(), [])
        self.assertEqual((self.source / rollout.relative_to(self.stage)).read_bytes(), rollout.read_bytes())
        recovered = CodexStateSync(self.source, self.stage, self.baseline)
        self.assertEqual(recovered.sync(), [])

    def test_concurrent_history_appends_are_merged(self):
        (self.source / "history.jsonl").write_text("base\n")
        sync = self.seed()
        (self.source / "history.jsonl").write_text("base\nhost\n")
        (self.stage / "history.jsonl").write_text("base\nsandbox\n")
        self.assertEqual(sync.sync(), [])
        expected = "base\nhost\nsandbox\n"
        self.assertEqual((self.source / "history.jsonl").read_text(), expected)
        self.assertEqual((self.stage / "history.jsonl").read_text(), expected)
        self.assertEqual(sync.sync(), [])

    def test_journal_recovery_does_not_duplicate_already_appended_lines(self):
        (self.source / "history.jsonl").write_text("base\n")
        sync = self.seed()
        (self.source / "history.jsonl").write_text("base\nhost\nsandbox\n")
        (self.stage / "history.jsonl").write_text("base\nsandbox\n")
        self.assertEqual(sync.sync(), [])
        self.assertEqual((self.source / "history.jsonl").read_text(), "base\nhost\nsandbox\n")

    def test_new_and_incomplete_history_records(self):
        sync = self.seed()
        (self.stage / "history.jsonl").write_text("first\n")
        self.assertEqual(sync.sync(), [])
        (self.stage / "history.jsonl").write_text("first\nincomplete")
        sync.sync()
        self.assertEqual((self.source / "history.jsonl").read_text(), "first\n")
        (self.stage / "history.jsonl").write_text("first\nincomplete\n")
        self.assertEqual(sync.sync(), [])
        self.assertEqual((self.source / "history.jsonl").read_text(), "first\nincomplete\n")

    def test_seed_skips_symlinks_and_sync_cannot_escape(self):
        secret = self.root / "secret"
        secret.write_text("secret")
        (self.source / "auth.json").symlink_to(secret)
        sync = self.seed()
        self.assertFalse((self.stage / "auth.json").exists())
        (self.stage / "auth.json").symlink_to(secret)
        self.assertEqual(sync.sync(), [])
        (self.stage / "sessions").symlink_to(self.root, target_is_directory=True)
        self.assertEqual(sync.sync(), [])
        self.assertEqual(secret.read_text(), "secret")
        with self.assertRaises(OSError):
            read_file(self.stage, "sessions/secret")

    def test_special_files_and_hardlinks_are_not_read(self):
        self.seed()
        os.mkfifo(self.stage / "auth.json")
        with self.assertRaises(ValueError):
            read_file(self.stage, "auth.json")
        secret = self.root / "secret"
        secret.write_text("secret")
        os.link(secret, self.stage / "config.toml")
        with self.assertRaises(ValueError):
            read_file(self.stage, "config.toml")

    def test_deletions_and_unknown_files_never_propagate(self):
        (self.source / "config.toml").write_text("host")
        sync = self.seed()
        (self.stage / "config.toml").unlink()
        (self.stage / "state_5.sqlite").write_text("database")
        (self.stage / "arbitrary.sh").write_text("script")
        sync.sync()
        self.assertEqual((self.source / "config.toml").read_text(), "host")
        self.assertFalse((self.source / "state_5.sqlite").exists())
        self.assertFalse((self.source / "arbitrary.sh").exists())

    def test_path_allowlist(self):
        for path in ("../auth.json", "/auth.json", "sessions/../auth.json", "sessions/anything.db", "packages/code"):
            self.assertFalse(allowed_file(path))
        self.assertNotIn("auth", json.dumps(self.seed().tracked))

    def test_shared_sessions_bypass_copy_limit_and_write_through(self):
        transcript = self.source / "sessions/rollout-large.jsonl"
        transcript.parent.mkdir()
        (self.source / "archived_sessions").mkdir()
        with transcript.open("wb") as stream:
            stream.truncate(MAX_STATE_FILE_BYTES + 1)
        (self.source / "config.toml").write_text("old")
        seed_state(self.source, self.stage, self.baseline, shared_sessions=True)
        self.assertTrue((self.stage / "sessions").is_symlink())
        self.assertTrue((self.stage / "archived_sessions").is_symlink())
        self.assertNotIn("sessions/rollout-large.jsonl", json.loads(self.baseline.read_text()))
        (self.stage / "sessions/rollout-new.jsonl").write_text("new transcript\n")
        self.assertEqual((self.source / "sessions/rollout-new.jsonl").read_text(), "new transcript\n")
        sync = CodexStateSync(self.source, self.stage, self.baseline, shared_sessions=True)
        (self.stage / "config.toml").write_text("new")
        self.assertEqual(sync.sync(), [])
        self.assertEqual((self.source / "config.toml").read_text(), "new")
