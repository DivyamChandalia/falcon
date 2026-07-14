import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from falcon.cli import _looks_like_legacy_submission, resolve_preset, run_legacy
from falcon.completion import candidates, shell_script
from falcon.config import (
    DEFAULT_CONFIG,
    DEFAULT_DASHBOARD_EMA_ALPHA,
    load_config,
    namespace_from_logname,
    run_setup,
    _remove_legacy_falcon_shell,
)
from falcon.launcher import build_jet_command
from falcon.resources import ResourcePlan


class FalconCliTests(unittest.TestCase):
    def test_dynamic_preset_counts_include_odd_counts(self):
        self.assertEqual(resolve_preset("h100", DEFAULT_CONFIG), ("h100", 1))
        self.assertEqual(resolve_preset("h100x2", DEFAULT_CONFIG), ("h100", 2))
        self.assertEqual(resolve_preset("2080tix3", DEFAULT_CONFIG), ("2080ti", 3))
        self.assertIsNone(resolve_preset("2080", DEFAULT_CONFIG))

    def test_legacy_submission_syntax_maps_to_native_preset(self):
        with patch("falcon.cli._launch_request", return_value=0) as launch_request:
            result = run_legacy(
                ["-j", "legacy-job", "-n", "3", "-g", "2080ti", "-a", "--", "python", "train.py"],
                copy.deepcopy(DEFAULT_CONFIG),
            )
        self.assertEqual(result, 0)
        preset, count, args, _ = launch_request.call_args.args
        self.assertEqual((preset, count), ("2080ti", 3))
        self.assertEqual(args.job, "legacy-job")
        self.assertTrue(args.async_mode)
        self.assertEqual(args.command[-2:], ["python", "train.py"])
        self.assertTrue(_looks_like_legacy_submission(["-j", "legacy-job", "-g", "h100"]))

    def test_logname_derives_namespace(self):
        self.assertEqual(namespace_from_logname("divyam.c"), "divyamc-dev")

    def test_setup_writes_only_user_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".falconrc"
            with patch.dict(os.environ, {"HOME": directory}):
                written, rc = run_setup(str(path), non_interactive=True, install_shell=False)
            self.assertEqual(written, path)
            self.assertIsNone(rc)
            self.assertTrue((Path(directory) / ".local" / "bin" / "falcon").exists())
            raw = yaml.safe_load(path.read_text())
            self.assertNotIn("cluster", raw)
            self.assertNotIn("runtime", raw)
            self.assertEqual(raw["resources"]["shared_memory_percent"], 15)
            self.assertNotIn("refresh_seconds", raw["dashboard"])
            config = load_config(str(path))
            self.assertEqual(config["presets"]["h100"]["minimum_utilization"], 90)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_legacy_dashboard_refresh_override_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".falconrc"
            path.write_text("version: 1\ndashboard:\n  refresh_seconds: 99\n  ema_alpha: 0.4\n")
            config = load_config(str(path))
            self.assertNotIn("refresh_seconds", config["dashboard"])
            self.assertEqual(config["dashboard"]["ema_alpha"], 0.4)

    def test_generated_legacy_ema_alpha_migrates_to_smoother_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".falconrc"
            path.write_text("version: 1\ndashboard:\n  ema_alpha: 0.25\n")
            config = load_config(str(path))
            self.assertEqual(config["dashboard"]["ema_alpha"], DEFAULT_DASHBOARD_EMA_ALPHA)
            self.assertEqual(DEFAULT_DASHBOARD_EMA_ALPHA, 0.1)

    def test_setup_installs_detected_shell_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / ".falconrc"
            with patch.dict(os.environ, {"HOME": directory, "FALCON_SHELL": "zsh"}):
                _, rc = run_setup(str(config), non_interactive=True)
            self.assertEqual(rc, Path(directory) / ".zshrc")
            content = rc.read_text()
            self.assertIn(f"{directory}/.local/bin/falcon shell-init zsh", content)
            self.assertIn("# >>> falcon native >>>", content)
            launcher = Path(directory) / ".local" / "bin" / "falcon"
            self.assertIn("-m falcon", launcher.read_text())

    def test_shell_migration_removes_preview_falcon_only(self):
        legacy = "before\nfalcon() {\n echo '[falcon] Exported FALCON_LAST_JOB=x'\n}\n_falcon() { :; }\ncompdef _falcon falcon\nafter\n"
        migrated = _remove_legacy_falcon_shell(legacy)
        self.assertEqual(migrated, "before\nafter\n")

    def test_shell_completion_replaces_old_function_and_is_dynamic(self):
        generated = shell_script("zsh")
        self.assertIn("function falcon", generated)
        self.assertIn("/.local/bin/falcon _complete jobs", generated)
        self.assertIn("words[CURRENT-1]", generated)
        self.assertNotIn("function h100", generated)
        options = candidates("options", DEFAULT_CONFIG, "2080tix3")
        self.assertIn("--shm-percent", options)
        self.assertIn("--max", options)

    def test_jet_command_uses_scheduler_and_calculates_fifteen_percent_shm(self):
        plan = ResourcePlan("h100", "h100", 1, "48:48", "282.6Gi:282.6Gi", "nodex1", True)
        command = build_jet_command(DEFAULT_CONFIG, plan, ["python", "train.py"], name="smoke", dry_run=True)
        self.assertNotIn("kubernetes.io/hostname=nodex1", command)
        self.assertIn("falcon.dev/managed=true", command)
        self.assertIn("CONDA_AUTO_ACTIVATE_BASE=false", command)
        self.assertEqual(command[command.index("--shm-size") + 1], "42.4Gi")
        self.assertIn("--dry-run", command)

    def test_explicit_pin_node_adds_hostname_selector(self):
        plan = ResourcePlan("h100", "h100", 1, "48:48", "282.6Gi:282.6Gi", "nodex1", True)
        command = build_jet_command(DEFAULT_CONFIG, plan, [], name="debug", pin_node=True)
        self.assertIn("kubernetes.io/hostname=nodex1", command)

    def test_preset_can_override_shared_memory_percentage(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["presets"]["h100"]["shared_memory_percent"] = 20
        plan = ResourcePlan("h100", "h100", 1, "48:48", "100Gi:100Gi", "nodex1", True)
        command = build_jet_command(config, plan, [], name="debug")
        self.assertEqual(command[command.index("--shm-size") + 1], "20Gi")


if __name__ == "__main__":
    unittest.main()
