import copy
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from falcon.cli import _looks_like_cpu_submission, _looks_like_legacy_submission, _main_parser, main, resolve_preset, run_legacy
from falcon.completion import candidates, preset_tokens, shell_script
from falcon.commands import clean
from falcon.config import (
    DEFAULT_CONFIG,
    DEFAULT_DASHBOARD_EMA_ALPHA,
    load_config,
    namespace_from_logname,
    run_setup,
    save_dashboard_sort,
    save_hidden_panes,
    _remove_legacy_falcon_shell,
)
from falcon.launcher import build_jet_command
from falcon.resources import NodeResources, ResourcePlan, plan_cpu_resources


class FalconCliTests(unittest.TestCase):
    persisted_identity = (
        "cluster:\n  namespace: test-dev\n"
        "runtime:\n  volumes:\n    - /media/beegfs/users/test/\n    - /media/beegfs/teams/\n"
    )

    def test_clean_deletes_succeeded_jobs_only(self):
        inventory = subprocess.CompletedProcess([], 0, (
            '{"items": ['
            '{"metadata": {"name": "done"}, "status": {"succeeded": 1}},'
            '{"metadata": {"name": "failed"}, "status": {"failed": 1}},'
            '{"metadata": {"name": "running"}, "status": {"active": 1}}'
            ']}'
        ), "")
        deleted = subprocess.CompletedProcess([], 0, "", "")
        with patch("falcon.commands.kubectl", side_effect=[inventory, deleted]) as kubectl:
            self.assertEqual(clean("test-dev"), 0)
        self.assertEqual(kubectl.call_args_list[1].args[0], ["delete", "job.batch", "done", "-n", "test-dev"])

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
        self.assertTrue(_looks_like_cpu_submission(["-c", "2:4", "-m", "12Gi:12Gi", "--", "python", "x.py"]))
        self.assertTrue(_looks_like_cpu_submission(["--cpu=2", "--memory=12Gi", "--", "python", "x.py"]))
        self.assertFalse(_looks_like_cpu_submission(["-c", "2", "-g", "h100", "--", "python", "x.py"]))

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
            self.assertEqual(raw["cluster"]["namespace"], namespace_from_logname())
            self.assertEqual(
                raw["runtime"]["volumes"],
                [f"/media/beegfs/users/{os.environ.get('LOGNAME') or os.environ.get('USER')}/", "/media/beegfs/teams/"],
            )
            self.assertNotIn("image", raw["runtime"])
            self.assertNotIn("scheduler", raw["runtime"])
            self.assertEqual(raw["runtime"]["environment"], {})
            self.assertEqual(raw["resources"]["shared_memory_percent"], 15)
            self.assertIsNone(raw["job"]["backoff_limit"])
            self.assertNotIn("refresh_seconds", raw["dashboard"])
            config = load_config(str(path))
            self.assertEqual(config["presets"]["h100"]["minimum_utilization"], 90)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_runtime_identity_and_mounts_come_from_falconrc(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".falconrc"
            path.write_text(
                "version: 1\n"
                "cluster:\n  namespace: configured-dev\n"
                "runtime:\n  volumes:\n    - /custom/user/\n    - /custom/team/\n"
            )
            with patch.dict(os.environ, {"LOGNAME": "different.user"}):
                config = load_config(str(path))
            self.assertEqual(config["cluster"]["namespace"], "configured-dev")
            self.assertEqual(config["runtime"]["volumes"], ["/custom/user/", "/custom/team/"])

    def test_interactive_setup_prompts_for_namespace_mounts_and_shm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".falconrc"
            answers = [
                "custom-dev", "/data/user, /data/team, /scratch",
                "WANDB_MODE=offline, TOKEN=value=with=equals", "20",
            ]
            with patch.dict(os.environ, {"HOME": directory, "LOGNAME": "setup.user"}), patch(
                "builtins.input", side_effect=answers
            ):
                run_setup(str(path), install_shell=False)
            raw = yaml.safe_load(path.read_text())
            self.assertEqual(raw["cluster"]["namespace"], "custom-dev")
            self.assertEqual(raw["runtime"]["volumes"], ["/data/user", "/data/team", "/scratch"])
            self.assertEqual(
                raw["runtime"]["environment"],
                {"WANDB_MODE": "offline", "TOKEN": "value=with=equals"},
            )
            self.assertEqual(raw["resources"]["shared_memory_percent"], 20)

    def test_runtime_environment_from_falconrc_overrides_internal_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".falconrc"
            path.write_text(
                "version: 1\n" + self.persisted_identity
                + "runtime:\n"
                + "  volumes:\n    - /media/beegfs/users/test/\n    - /media/beegfs/teams/\n"
                + "  environment:\n    CONDA_AUTO_ACTIVATE_BASE: custom\n    EXPERIMENT: demo\n"
            )
            config = load_config(str(path))
            self.assertEqual(config["runtime"]["environment"]["CONDA_AUTO_ACTIVATE_BASE"], "custom")
            self.assertEqual(config["runtime"]["environment"]["EXPERIMENT"], "demo")

    def test_setup_force_can_replace_config_missing_persisted_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".falconrc"
            path.write_text("version: 1\n")
            with patch("falcon.cli.run_setup", return_value=(path, None)) as setup:
                result = main(["--config", str(path), "setup", "--force", "--non-interactive", "--no-shell"])
            self.assertEqual(result, 0)
            setup.assert_called_once_with(
                str(path), force=True, non_interactive=True, install_shell=False
            )

    def test_legacy_dashboard_refresh_override_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".falconrc"
            path.write_text(
                "version: 1\n" + self.persisted_identity
                + "dashboard:\n  refresh_seconds: 99\n  ema_alpha: 0.4\n"
                + "  hidden_panes: [events]\n  sort_field: Name\n  sort_direction: asc\n"
            )
            config = load_config(str(path))
            self.assertNotIn("refresh_seconds", config["dashboard"])
            self.assertEqual(config["dashboard"]["ema_alpha"], 0.4)
            self.assertEqual(config["dashboard"]["hidden_panes"], ["events"])
            self.assertEqual(config["dashboard"]["sort_field"], "Name")
            self.assertEqual(config["dashboard"]["sort_direction"], "asc")
            save_hidden_panes({"resources", "selected"}, str(path))
            save_dashboard_sort("Status", "desc", str(path))
            persisted = yaml.safe_load(path.read_text())
            self.assertEqual(persisted["dashboard"]["hidden_panes"], ["resources", "selected"])
            self.assertEqual(persisted["dashboard"]["sort_field"], "Status")
            self.assertEqual(persisted["dashboard"]["sort_direction"], "asc")
            self.assertEqual(persisted["cluster"]["namespace"], "test-dev")
            self.assertEqual(persisted["runtime"]["volumes"][0], "/media/beegfs/users/test/")

    def test_generated_legacy_ema_alpha_migrates_to_smoother_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".falconrc"
            path.write_text("version: 1\n" + self.persisted_identity + "dashboard:\n  ema_alpha: 0.25\n")
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
        with patch("falcon.completion.preset_tokens", return_value=["h100", "h100x2", "2080ti", "2080tix2"]):
            generated = shell_script("zsh", config=DEFAULT_CONFIG)
        self.assertIn("function falcon", generated)
        self.assertNotIn("_complete", generated)
        self.assertIn("command kubectl get jobs.batch", generated)
        self.assertIn("_falcon_job_cache_time", generated)
        self.assertIn("h100x2", generated)
        self.assertIn("--shm-percent", generated)
        self.assertIn("FALCON_DEBUG_PROMPT", generated)
        self.assertIn("%F{81}", generated)
        self.assertNotIn("function h100", generated)
        options = candidates("options", DEFAULT_CONFIG, "2080tix3")
        self.assertIn("--shm-percent", options)
        self.assertIn("--max", options)

    def test_preset_capacity_completion_is_cached_across_shells(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "falcon.completion._preset_cache_path", return_value=Path(directory) / "presets.json"
        ), patch(
            "falcon.completion.fetch_nodes",
            return_value=[NodeResources("node", gpu_total=4, gpu_product="2080 Ti")],
        ) as fetch:
            first = preset_tokens(DEFAULT_CONFIG)
            second = preset_tokens(DEFAULT_CONFIG)
        self.assertIn("2080tix4", first)
        self.assertEqual(first, second)
        self.assertEqual(fetch.call_count, 1)

    def test_namespace_is_not_a_user_facing_option(self):
        self.assertNotIn("--namespace", candidates("options", DEFAULT_CONFIG, "2080tix3"))
        self.assertNotIn("--namespace", candidates("options", DEFAULT_CONFIG, "dashboard"))
        self.assertEqual(candidates("options", DEFAULT_CONFIG, "logs"), [])
        with self.assertRaises(SystemExit):
            _main_parser(DEFAULT_CONFIG).parse_args(["logs", "--namespace", "other-dev"])

    def test_jet_command_uses_scheduler_and_calculates_fifteen_percent_shm(self):
        plan = ResourcePlan("h100", "h100", 1, "48:48", "282.6Gi:282.6Gi", "nodex1", True)
        command = build_jet_command(DEFAULT_CONFIG, plan, ["python", "train.py"], name="smoke", dry_run=True)
        self.assertNotIn("kubernetes.io/hostname=nodex1", command)
        self.assertIn("falcon.dev/managed=true", command)
        self.assertNotIn("IN_JET_POD=1", command)
        self.assertFalse(any("FALCON_DEBUG_PROMPT=" in value for value in command))
        self.assertIn("CONDA_AUTO_ACTIVATE_BASE=false", command)
        self.assertEqual(command[command.index("--shm-size") + 1], "42.4Gi")
        self.assertIn("--dry-run", command)
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["job"]["backoff_limit"] = 3
        retry_command = build_jet_command(config, plan, ["python", "train.py"], name="retry", dry_run=True)
        self.assertEqual(retry_command[retry_command.index("--backoff-limit") + 1], "3")
        cpu_plan = plan_cpu_resources("2:4", "12Gi:12Gi")
        cpu_command = build_jet_command(DEFAULT_CONFIG, cpu_plan, ["python", "preprocess.py"], name="cpu", dry_run=True)
        self.assertNotIn("--gpu", cpu_command)
        self.assertNotIn("--gpu-type", cpu_command)
        self.assertIn("falcon.dev/managed=true", cpu_command)
        self.assertEqual(cpu_command[cpu_command.index("--cpu") + 1], "2:2")

    def test_explicit_pin_node_adds_hostname_selector(self):
        plan = ResourcePlan("h100", "h100", 1, "48:48", "282.6Gi:282.6Gi", "nodex1", True)
        command = build_jet_command(DEFAULT_CONFIG, plan, [], name="debug", pin_node=True)
        self.assertIn("kubernetes.io/hostname=nodex1", command)
        self.assertIn("FALCON_DEBUG_PROMPT=h100x1", command)

    def test_preset_can_override_shared_memory_percentage(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["presets"]["h100"]["shared_memory_percent"] = 20
        plan = ResourcePlan("h100", "h100", 1, "48:48", "100Gi:100Gi", "nodex1", True)
        command = build_jet_command(config, plan, [], name="debug")
        self.assertEqual(command[command.index("--shm-size") + 1], "20Gi")


if __name__ == "__main__":
    unittest.main()
