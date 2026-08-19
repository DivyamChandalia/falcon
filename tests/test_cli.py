from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from falcon.cli import (
    EXIT_KUBERNETES,
    EXIT_USAGE,
    _parser,
    _resources_command,
    _rewrite_shorthand,
    _skills_setup,
    main,
    resolve_preset,
)
from falcon.completion import COMMAND_ALIASES, candidates, shell_script
from falcon.config import (
    DEFAULT_CONFIG,
    load_config,
    run_setup,
    save_resources_consumer_sort,
    save_resources_view,
    validate_config,
)
from falcon.demo import demo_cluster_snapshot, demo_inventory
from falcon.kubernetes import KubernetesClient, KubernetesError


class CliHarness(unittest.TestCase):
    def invoke(self, *arguments: str):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.dict(
            os.environ,
            {"FALCON_NAMESPACE": "default", "CONDA_PREFIX": "", "VIRTUAL_ENV": ""},
            clear=False,
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = main(list(arguments))
            except SystemExit as exc:
                code = int(exc.code or 0)
        return code, stdout.getvalue(), stderr.getvalue()


class ParserTests(CliHarness):
    def test_tui_color_mode_options_are_explicit(self) -> None:
        parser = _parser(DEFAULT_CONFIG)
        for command in ("dashboard", "resources"):
            for mode in ("truecolor", "256", "16", "auto"):
                with self.subTest(command=command, mode=mode):
                    args = parser.parse_args([command, f"--color={mode}"])
                    self.assertEqual(args.color, mode)

    def test_help(self) -> None:
        code, stdout, _ = self.invoke("--help")
        self.assertEqual(code, 0)
        self.assertIn("Run Kubernetes Jobs like local commands", stdout)
        self.assertNotIn("submit", stdout)
        self.assertIn("metrics", stdout)
        self.assertIn("kill", stdout)
        self.assertNotIn("delete", stdout)

    def test_version(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            main(["--version"])
        self.assertEqual(raised.exception.code, 0)

    def test_preset_shorthand(self) -> None:
        self.assertEqual(resolve_preset("h100x2", DEFAULT_CONFIG), ("h100", 2))
        self.assertEqual(resolve_preset("pro6000", DEFAULT_CONFIG), ("pro6000", 1))
        self.assertEqual(resolve_preset("pro6000x2", DEFAULT_CONFIG), ("pro6000", 2))

    def test_preset_shorthand_respects_per_gpu_count_limits(self) -> None:
        for token, maximum in (
            ("h100x9", 8),
            ("a6000x3", 2),
            ("2080tix5", 4),
            ("pro6000x3", 2),
        ):
            with self.subTest(token=token):
                with self.assertRaisesRegex(ValueError, rf"at most {maximum}"):
                    resolve_preset(token, DEFAULT_CONFIG)

    def test_explicit_gpu_flags_respect_preset_count_limit(self) -> None:
        code, _, stderr = self.invoke(
            "--gpu", "pro6000", "--gpus", "3", "--cpu", "1", "--memory",
            "1Gi", "--dry-run", "--", "true",
        )
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("pro6000 supports at most 2", stderr)

    def test_one_letter_command_aliases_rewrite_to_canonical_commands(self) -> None:
        for alias, canonical in COMMAND_ALIASES.items():
            with self.subTest(alias=alias):
                self.assertEqual(
                    _rewrite_shorthand([alias, "value"], DEFAULT_CONFIG),
                    [canonical, "value"],
                )

    def test_unknown_preset_is_not_rewritten(self) -> None:
        self.assertIsNone(resolve_preset("bogus", DEFAULT_CONFIG))

    def test_removed_submit_has_direct_replacements(self) -> None:
        code, _, stderr = self.invoke("submit", "--cpu", "1", "--memory", "2Gi")
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("was removed", stderr)
        self.assertIn("falcon h100", stderr)
        self.assertIn("falcon -c CPU -m MEMORY", stderr)

    def test_submit_is_absent_from_completion(self) -> None:
        commands = candidates("commands", DEFAULT_CONFIG)
        self.assertEqual(commands[:5], [
            "h100", "a6000", "2080ti", "pro6000", "logs"
        ])
        self.assertNotIn("submit", commands)
        self.assertNotIn("delete", commands)
        self.assertIn("kill", commands)
        self.assertIn("metrics", commands)
        for alias in COMMAND_ALIASES:
            self.assertNotIn(alias, commands)
        self.assertIn("h100", commands)
        self.assertNotIn("h100x2", commands)
        self.assertEqual(
            candidates("counts", DEFAULT_CONFIG, "h100"),
            [f"h100x{count}" for count in range(2, 9)],
        )
        self.assertEqual(
            candidates("counts", DEFAULT_CONFIG, "a6000"),
            ["a6000x2"],
        )
        self.assertEqual(
            candidates("counts", DEFAULT_CONFIG, "2080ti"),
            [f"2080tix{count}" for count in range(2, 5)],
        )
        self.assertEqual(
            candidates("counts", DEFAULT_CONFIG, "pro6000"),
            ["pro6000x2"],
        )
        self.assertIn("--dry-run", candidates("options", DEFAULT_CONFIG, "h100"))
        self.assertNotIn(
            "--gpu", candidates("options", DEFAULT_CONFIG, "h100")
        )
        self.assertIn(
            "--consumer-limit",
            candidates("options", DEFAULT_CONFIG, "r"),
        )
        coder_options = candidates("options", DEFAULT_CONFIG, "coder")
        self.assertIn("2080ti", coder_options)
        self.assertIn("2080tix2", coder_options)
        self.assertIn("-j", coder_options)

    def test_bash_completion_offers_base_preset_then_gpu_counts(self) -> None:
        executable = shutil.which("bash")
        if executable is None:
            self.skipTest("bash is unavailable")
        script = shell_script("bash", config=DEFAULT_CONFIG)

        def complete(words: str, index: int) -> list[str]:
            command = (
                "source /dev/stdin; "
                f"COMP_WORDS=({words}); COMP_CWORD={index}; "
                "_falcon_native; printf '%s\\n' \"${COMPREPLY[@]}\""
            )
            completed = subprocess.run(
                [executable, "--noprofile", "--norc", "-c", command],
                input=script,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            return completed.stdout.splitlines()

        top_level = complete("falcon ''", 1)
        self.assertEqual(
            top_level[:5],
            ["h100", "a6000", "2080ti", "pro6000", "logs"],
        )
        self.assertNotIn("h100x2", top_level)
        self.assertNotIn("--gpu", top_level)
        self.assertEqual(
            top_level[-3:],
            ["--cpu", "--memory", "--gpus"],
        )
        self.assertEqual(
            complete("falcon h100", 1),
            [f"h100x{count}" for count in range(2, 9)],
        )
        self.assertEqual(complete("falcon pro6000", 1), ["pro6000x2"])

    def test_resources_completion_includes_consumer_bound(self) -> None:
        self.assertIn(
            "--consumer-limit",
            candidates("options", DEFAULT_CONFIG, "resources"),
        )
        zsh_completion = shell_script("zsh", config=DEFAULT_CONFIG)
        self.assertIn("2080tix*", zsh_completion)
        self.assertIn('compstate[list]="${compstate[list]} rows"', zsh_completion)

    def test_job_completion_queries_without_a_local_cache(self) -> None:
        for shell in ("bash", "zsh"):
            script = shell_script(shell, config=DEFAULT_CONFIG)
            self.assertNotIn("_falcon_job_cache", script)
            self.assertNotIn("_falcon_refresh_jobs", script)
            self.assertIn("kubectl get jobs.batch", script)
            self.assertIn("app.kubernetes.io/managed-by=coder", script)
            self.assertIn("coder\\.workspace", script)

    def test_bash_coder_completion_includes_only_workspace_names(self) -> None:
        executable = shutil.which("bash")
        if executable is None:
            self.skipTest("bash is unavailable")
        script = shell_script("bash", config=DEFAULT_CONFIG)

        with tempfile.TemporaryDirectory() as temporary:
            kubectl = Path(temporary) / "kubectl"
            kubectl.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' falcon lime-gull-30 '<none>'\n",
                encoding="utf-8",
            )
            kubectl.chmod(0o755)
            command = (
                "source /dev/stdin; "
                "COMP_WORDS=(falcon coder ''); COMP_CWORD=2; "
                "_falcon_native; printf '%s\\n' \"${COMPREPLY[@]}\""
            )
            completed = subprocess.run(
                [executable, "--noprofile", "--norc", "-c", command],
                input=script,
                text=True,
                capture_output=True,
                check=False,
                env={
                    **os.environ,
                    "PATH": f"{temporary}{os.pathsep}{os.environ.get('PATH', '')}",
                },
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        values = completed.stdout.splitlines()
        self.assertEqual(values, ["falcon", "lime-gull-30"])
        self.assertNotIn("2080ti", values)
        self.assertNotIn("--access", values)

    def test_legacy_shell_init_keeps_completion_registered_after_upgrade(self) -> None:
        for shell in ("bash", "zsh"):
            executable = shutil.which(shell)
            if executable is None:
                continue
            with self.subTest(shell=shell):
                code, script, stderr = self.invoke("shell-init", shell)
                self.assertEqual((code, stderr), (0, ""))
                if shell == "zsh":
                    command = (
                        "autoload -Uz compinit && compinit && "
                        "source /dev/stdin && "
                        "[[ ${_comps[falcon]-} == _falcon_native ]]"
                    )
                else:
                    command = (
                        "source /dev/stdin && "
                        "complete -p falcon | grep -q -- "
                        "'complete -F _falcon_native falcon'"
                    )
                completed = subprocess.run(
                    [executable, "-f", "-c", command]
                    if shell == "zsh"
                    else [executable, "--noprofile", "--norc", "-c", command],
                    input=script,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    completed.stderr or completed.stdout,
                )

    def test_legacy_dashboard_json_has_actionable_replacement(self) -> None:
        code, _, stderr = self.invoke("dashboard", "--json")
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("falcon jobs --output json", stderr)

    def test_invalid_environment_variable(self) -> None:
        code, _, stderr = self.invoke(
            "--cpu", "1", "--memory", "2Gi",
            "--environment", "none", "--dry-run", "--env", "INVALID",
        )
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("KEY=VALUE", stderr)

    def test_cpu_job_requires_cpu_and_memory(self) -> None:
        code, _, stderr = self.invoke(
            "--cpu", "1", "--environment", "none", "--dry-run",
        )
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("both --cpu and --memory", stderr)


class SubmissionCliTests(CliHarness):
    def test_submission_prints_resolved_request_before_create(self) -> None:
        from falcon.models import NodeResources, SubmittedJob

        node = NodeResources("gpu-a", 64, 0, 480, 0, 4, 0, "H100")
        client = unittest.mock.Mock()
        submitted = SubmittedJob("request-summary", "default", created=True)
        with patch("falcon.cli._planning_nodes", return_value=[node]), patch(
            "falcon.cli.KubernetesClient", return_value=client
        ), patch("falcon.cli.submit", return_value=submitted), patch(
            "falcon.cli.remember_job"
        ):
            code, stdout, stderr = self.invoke(
                "h100",
                "--namespace", "default",
                "--environment", "none",
                "--name", "request-summary",
                "--",
                "python", "train.py",
            )
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Submitting Job request · request-summary", stdout)
        self.assertIn("namespace=default", stdout)
        self.assertIn("nvidia.com/gpu=1", stdout)
        self.assertIn("cpu=16", stdout)
        self.assertIn("command=python train.py", stdout)

    def test_json_submission_keeps_stdout_structured_and_logs_summary_to_stderr(self) -> None:
        from falcon.models import NodeResources, SubmittedJob

        node = NodeResources("gpu-a", 64, 0, 480, 0, 4, 0, "H100")
        client = unittest.mock.Mock()
        submitted = SubmittedJob("json-summary", "default", created=True)
        with patch("falcon.cli._planning_nodes", return_value=[node]), patch(
            "falcon.cli.KubernetesClient", return_value=client
        ), patch("falcon.cli.submit", return_value=submitted), patch(
            "falcon.cli.remember_job"
        ):
            code, stdout, stderr = self.invoke(
                "h100",
                "--output", "json",
                "--environment", "none",
                "--name", "json-summary",
                "--",
                "python", "train.py",
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["kind"], "SubmittedJob")
        self.assertIn("Submitting Job request · json-summary", stderr)

    def test_cpu_dry_run_yaml(self) -> None:
        code, stdout, stderr = self.invoke(
            "--cpu", "2", "--memory", "4Gi",
            "--environment", "none", "--name", "cpu-test", "--dry-run",
            "--", "python", "work.py",
        )
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("kind: Job", stdout)
        self.assertNotIn("nvidia.com/gpu", stdout)

    def test_cpu_dry_run_json(self) -> None:
        code, stdout, _ = self.invoke(
            "-c", "2", "-m", "4Gi",
            "--environment", "none", "--name", "cpu-json", "--dry-run",
            "--output", "json", "--", "python", "work.py",
        )
        self.assertEqual(code, 0)
        value = json.loads(stdout)
        self.assertEqual(value["kind"], "JobDryRun")
        self.assertEqual(value["data"]["manifest"]["kind"], "Job")

    def test_gpu_dry_run_with_planned_node(self) -> None:
        from falcon.models import NodeResources

        node = NodeResources("gpu-a", 64, 0, 480, 0, 4, 0, "H100")
        with patch("falcon.cli._planning_nodes", return_value=[node]):
            code, stdout, _ = self.invoke(
                "h100x2",
                "--cpu", "8", "--memory", "32Gi",
                "--environment", "none", "--name", "gpu-test",
                "--dry-run", "--output", "json", "--", "python", "train.py",
            )
        self.assertEqual(code, 0)
        manifest = json.loads(stdout)["data"]["manifest"]
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(container["resources"]["requests"]["nvidia.com/gpu"], "2")

    def test_gpu_memory_override_does_not_require_cpu_override(self) -> None:
        from falcon.models import NodeResources

        node = NodeResources("gpu-a", 64, 0, 480, 0, 4, 0, "H100")
        with patch("falcon.cli._planning_nodes", return_value=[node]):
            code, stdout, stderr = self.invoke(
                "h100",
                "-m",
                "80Gi",
                "--environment",
                "none",
                "--name",
                "gpu-memory",
                "--dry-run",
                "--output",
                "json",
                "--",
                "python",
                "train.py",
            )
        resources = json.loads(stdout)["data"]["manifest"]["spec"]["template"][
            "spec"
        ]["containers"][0]["resources"]["requests"]
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(resources["memory"], "80Gi")
        self.assertEqual(resources["cpu"], "16")

    def test_gpu_dry_run_applies_automatic_memory_safety_buffer(self) -> None:
        from falcon.models import NodeResources

        node = NodeResources("gpu-a", 64, 0, 480, 0, 4, 0, "H100")
        with patch("falcon.cli._planning_nodes", return_value=[node]):
            code, stdout, stderr = self.invoke(
                "h100",
                "--environment",
                "none",
                "--name",
                "gpu-buffer",
                "--dry-run",
                "--output",
                "json",
                "--",
                "python",
                "train.py",
            )
        resources = json.loads(stdout)["data"]["manifest"]["spec"]["template"][
            "spec"
        ]["containers"][0]["resources"]["requests"]
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(resources["cpu"], "16")
        self.assertEqual(resources["memory"], "119Gi")

    def test_missing_explicit_environment_is_clear(self) -> None:
        code, _, stderr = self.invoke(
            "--cpu", "1", "--memory", "2Gi",
            "--environment", "/missing/falcon-env", "--dry-run",
        )
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("does not exist", stderr)

    def test_launches_are_detached_unless_follow_is_explicit(self) -> None:
        from falcon.kubernetes import ProcessResult
        from falcon.models import NodeResources, SubmittedJob

        node = NodeResources("gpu-a", 64, 0, 480, 0, 4, 0, "H100")
        client = unittest.mock.Mock()
        client.logs.return_value = ProcessResult(("kubectl", "logs"), 0)
        submitted = SubmittedJob("detached-job", "default", created=True)
        with patch("falcon.cli._planning_nodes", return_value=[node]), patch(
            "falcon.cli.KubernetesClient", return_value=client
        ), patch("falcon.cli.submit", return_value=submitted), patch(
            "falcon.cli.remember_job"
        ):
            code, _, stderr = self.invoke(
                "h100",
                "--environment",
                "none",
                "--name",
                "detached-job",
                "--",
                "python",
                "train.py",
            )
        self.assertEqual((code, stderr), (0, ""))
        client.logs.assert_not_called()
        client.exec_shell.assert_not_called()

    def test_commandless_launch_opens_shell_and_deletes_debug_job(self) -> None:
        from falcon.kubernetes import ProcessResult
        from falcon.models import NodeResources, SubmittedJob

        node = NodeResources("gpu-a", 64, 0, 480, 0, 4, 0, "H100")
        client = unittest.mock.Mock()
        client.exec_shell.return_value = ProcessResult(
            ("kubectl", "exec"), 0
        )
        submitted = SubmittedJob("debug-job", "default", created=True)
        with patch("falcon.cli._planning_nodes", return_value=[node]), patch(
            "falcon.cli.KubernetesClient", return_value=client
        ), patch("falcon.cli.submit", return_value=submitted), patch(
            "falcon.cli.remember_job"
        ), patch(
            "falcon.cli.detect_shell",
            return_value=("zsh", Path("/tmp/.zshrc")),
        ):
            code, stdout, stderr = self.invoke(
                "h100",
                "--environment",
                "none",
                "--name",
                "debug-job",
            )
        self.assertEqual((code, stderr), (0, ""))
        self.assertNotIn("Submitted Job", stdout)
        self.assertIn(
            "Waiting for debug Pod · GPU h100 x1 · CPU 16 · "
            "RAM 119Gi · SHM 18330Mi",
            stdout,
        )
        self.assertIn("Deleting debug Job debug-job", stdout)
        client.exec_shell.assert_called_once_with(
            "debug-job",
            "zsh",
            prompt_label="h100",
            rc_path="/tmp/.zshrc",
        )
        client.delete_jobs.assert_called_once_with(
            ["debug-job"], wait=False
        )
        client.logs.assert_not_called()

    def test_commandless_debug_ctrl_c_still_deletes_job(self) -> None:
        from falcon.models import NodeResources, SubmittedJob

        node = NodeResources("gpu-a", 64, 0, 480, 0, 4, 0, "H100")
        client = unittest.mock.Mock()
        client.exec_shell.side_effect = KeyboardInterrupt
        submitted = SubmittedJob("debug-interrupt", "default", created=True)
        with patch("falcon.cli._planning_nodes", return_value=[node]), patch(
            "falcon.cli.KubernetesClient", return_value=client
        ), patch("falcon.cli.submit", return_value=submitted), patch(
            "falcon.cli.remember_job"
        ):
            code, stdout, stderr = self.invoke(
                "h100",
                "--environment",
                "none",
                "--name",
                "debug-interrupt",
            )
        self.assertEqual((code, stderr), (130, ""))
        self.assertIn("Deleting debug Job debug-interrupt", stdout)
        client.delete_jobs.assert_called_once_with(
            ["debug-interrupt"], wait=False
        )

    def test_commandless_debug_rejects_json_after_dry_run(self) -> None:
        code, _, stderr = self.invoke(
            "h100",
            "--output",
            "json",
        )
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("interactive debug sessions require human output", stderr)

    def test_follow_streams_and_ctrl_c_kills_only_the_new_job(self) -> None:
        from falcon.models import NodeResources, SubmittedJob

        node = NodeResources("gpu-a", 64, 0, 480, 0, 4, 0, "H100")
        client = unittest.mock.Mock()
        client.logs.side_effect = KeyboardInterrupt
        submitted = SubmittedJob("foreground-job", "default", created=True)
        with patch("falcon.cli._planning_nodes", return_value=[node]), patch(
            "falcon.cli.KubernetesClient", return_value=client
        ), patch("falcon.cli.submit", return_value=submitted), patch(
            "falcon.cli.remember_job"
        ):
            code, stdout, stderr = self.invoke(
                "h100",
                "-f",
                "--environment",
                "none",
                "--name",
                "foreground-job",
                "--",
                "python",
                "train.py",
            )
        self.assertEqual(code, 130)
        self.assertIn("Submitted Job foreground-job", stdout)
        self.assertIn("Stopping and killing Job foreground-job", stderr)
        client.logs.assert_called_once_with(
            "foreground-job", tail=100, follow=True
        )
        client.delete_jobs.assert_called_once_with(
            ["foreground-job"], wait=False
        )

    def test_follow_cannot_corrupt_structured_launch_output(self) -> None:
        code, _, stderr = self.invoke(
            "h100",
            "-f",
            "--output",
            "json",
            "--",
            "python",
            "train.py",
        )
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("--follow cannot", stderr)


class FakeClient:
    def __init__(self, namespace: str = "team-a") -> None:
        self.namespace = namespace
        self.inventory = demo_inventory("mixed")

    def list_jobs(self, namespace=None):
        return {"items": [item for item in self.inventory if item.get("kind") == "Job"]}

    def list_pods(self, namespace=None):
        return {"items": [item for item in self.inventory if item.get("kind") == "Pod"]}

    def get_json(self, resource, name, namespace=None):
        return next(
            item for item in self.inventory
            if item.get("kind") == "Job" and item["metadata"]["name"] == name
        )

    def job_pods(self, name):
        return [
            item for item in self.inventory
            if item.get("kind") == "Pod"
            and item["metadata"]["labels"].get("job-name") == name
        ]

    def events(self, names, limit=50):
        from falcon.demo import demo_events

        return demo_events(names[0], limit)


class InspectionCliTests(CliHarness):
    def test_jobs_json_is_bounded_and_versioned(self) -> None:
        with patch("falcon.cli.KubernetesClient", FakeClient):
            code, stdout, stderr = self.invoke("jobs", "--limit", "3", "--output", "json")
        value = json.loads(stdout)
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(value["meta"]["count"], 3)
        self.assertEqual(value["schema_version"], "falcon/v1")

    def test_jobs_human_distinguishes_request_and_allocation(self) -> None:
        with patch("falcon.cli.KubernetesClient", FakeClient):
            code, stdout, _ = self.invoke("jobs", "--limit", "20")
        self.assertEqual(code, 0)
        self.assertIn("GPU REQUESTED", stdout)
        self.assertIn("GPU ALLOCATED", stdout)

    def test_status_filter(self) -> None:
        with patch("falcon.cli.KubernetesClient", FakeClient):
            code, stdout, _ = self.invoke(
                "jobs", "--status", "Failed", "--output", "json",
            )
        jobs = json.loads(stdout)["data"]
        self.assertEqual(code, 0)
        self.assertTrue(jobs)
        self.assertTrue(all(job["status"] == "Failed" for job in jobs))

    def test_get_completed_gpu_retains_request(self) -> None:
        with patch("falcon.cli.KubernetesClient", FakeClient):
            code, stdout, _ = self.invoke(
                "get", "finished-h100-training", "--output", "json",
            )
        job = json.loads(stdout)["data"]["job"]
        self.assertEqual(code, 0)
        self.assertEqual(job["requested"]["gpu_count"], 2)
        self.assertEqual(job["allocated"]["gpu_count"], 0)

    def test_get_retry_summary(self) -> None:
        with patch("falcon.cli.KubernetesClient", FakeClient):
            code, stdout, _ = self.invoke(
                "get", "retry-eventually-succeeded", "--output", "json",
            )
        attempts = json.loads(stdout)["data"]["job"]["attempts"]
        self.assertEqual(code, 0)
        self.assertEqual(attempts["container_restarts"], 3)
        self.assertEqual(attempts["pod_attempts"], 2)

    def test_get_does_not_query_or_embed_events(self) -> None:
        class GetClient(FakeClient):
            def events(self, names, limit=50):
                raise AssertionError("get must not query events")

        with patch("falcon.cli.KubernetesClient", GetClient):
            code, stdout, stderr = self.invoke(
                "get", "train-h100-two", "--output", "json"
            )
        value = json.loads(stdout)
        self.assertEqual((code, stderr), (0, ""))
        self.assertNotIn("events", value["data"])

    def test_events_are_bounded(self) -> None:
        with patch("falcon.cli.KubernetesClient", FakeClient):
            code, stdout, _ = self.invoke(
                "events", "train-h100-two", "--limit", "7", "--output", "json",
            )
        value = json.loads(stdout)
        self.assertEqual(code, 0)
        self.assertEqual(value["meta"]["count"], 7)

    def test_events_follow_prints_new_events_until_interrupted(self) -> None:
        class FollowingEventsClient(FakeClient):
            calls = 0

            def events(self, names, limit=50):
                self.__class__.calls += 1
                first = {
                    "metadata": {"creationTimestamp": "2026-01-01T00:00:00Z"},
                    "involvedObject": {"name": names[0]},
                    "type": "Normal",
                    "reason": "Created",
                    "message": "First event",
                }
                second = {
                    "metadata": {"creationTimestamp": "2026-01-01T00:00:01Z"},
                    "involvedObject": {"name": names[0]},
                    "type": "Warning",
                    "reason": "BackOff",
                    "message": "Second event",
                }
                return (
                    [first]
                    if self.__class__.calls == 1
                    else [first, second]
                )

        FollowingEventsClient.calls = 0
        with patch(
            "falcon.cli.KubernetesClient", FollowingEventsClient
        ), patch(
            "falcon.cli.time.sleep",
            side_effect=[None, KeyboardInterrupt],
        ):
            code, stdout, stderr = self.invoke(
                "events", "train-h100-two", "-f"
            )
        self.assertEqual((code, stderr), (130, ""))
        self.assertIn("First event", stdout)
        self.assertIn("Second event", stdout)

    def test_events_follow_rejects_json_streaming(self) -> None:
        code, _, stderr = self.invoke(
            "events",
            "train-h100-two",
            "-f",
            "--output",
            "json",
        )
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("--follow cannot", stderr)

    def test_logs_json_is_bounded_and_structured(self) -> None:
        from falcon.kubernetes import ProcessResult

        observed = {}

        class LogsClient(FakeClient):
            def logs(self, name, **kwargs):
                observed.update(kwargs)
                return ProcessResult(
                    ("kubectl", "logs"),
                    0,
                    "first\nsecond\n",
                    "",
                )

        with patch("falcon.cli.KubernetesClient", LogsClient), patch(
            "falcon.cli.remember_job"
        ):
            code, stdout, stderr = self.invoke(
                "logs", "train-h100-two", "--tail", "2", "--output", "json",
            )
        value = json.loads(stdout)
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(value["kind"], "JobLogs")
        self.assertEqual(value["data"]["lines"], ["first", "second"])
        self.assertFalse(observed["follow"])

    def test_human_logs_follow_by_default_and_can_be_bounded(self) -> None:
        from falcon.kubernetes import ProcessResult

        calls = []

        class LogsClient(FakeClient):
            def logs(self, name, **kwargs):
                calls.append(kwargs)
                return ProcessResult(
                    ("kubectl", "logs"),
                    0,
                    "" if kwargs["follow"] else "one-shot\n",
                    "",
                )

        with patch("falcon.cli.KubernetesClient", LogsClient), patch(
            "falcon.cli.remember_job"
        ):
            code, stdout, stderr = self.invoke("logs", "train-h100-two")
            bounded_code, bounded_stdout, bounded_stderr = self.invoke(
                "logs", "train-h100-two", "--no-follow"
            )
        self.assertEqual((code, stdout, stderr), (0, "", ""))
        self.assertTrue(calls[0]["follow"])
        self.assertEqual(
            (bounded_code, bounded_stdout, bounded_stderr),
            (0, "one-shot\n", ""),
        )
        self.assertFalse(calls[1]["follow"])

    def test_follow_cannot_emit_unbounded_json(self) -> None:
        code, _, stderr = self.invoke(
            "logs", "train-h100-two", "--follow", "--output", "json",
        )
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("--follow cannot", stderr)

    def test_resources_json_works_without_tty(self) -> None:
        code, stdout, stderr = self.invoke(
            "resources", "--demo", "mixed", "--limit", "2", "--output", "json",
        )
        value = json.loads(stdout)
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(value["meta"]["count"], 2)
        self.assertEqual(value["kind"], "ClusterResources")
        self.assertEqual(
            value["data"]["nodes"][0]["gpu_memory_bytes_per_device"],
            81559 * 1024**2,
        )

    def test_resources_human_output_displays_request_headroom(self) -> None:
        code, stdout, stderr = self.invoke("resources", "--demo", "mixed")
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("CPU free 91.5/160.0", stdout)
        self.assertIn("RAM free 927.0Gi/1200.0Gi", stdout)
        self.assertIn("27.5/64.0", stdout)
        self.assertNotIn("CPU requested", stdout)

    def test_resources_tui_receives_restored_view_and_persistence_callback(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        collector = SimpleNamespace(collect=lambda force=False: snapshot)
        captured = {}

        class App:
            def __init__(self, received_collector, **kwargs):
                self.collector = received_collector
                captured.update(kwargs)

            def run(self, mouse=False):
                captured["mouse"] = mouse

        args = SimpleNamespace(
            limit=100,
            consumer_limit=100,
            demo="mixed",
            output="human",
            node=None,
            gpu=None,
            namespace=None,
        )
        config = json.loads(json.dumps(DEFAULT_CONFIG))
        config["resources"]["last_view"] = "gpu-allocations"
        config["resources"]["consumer_sort"] = "memory"
        with patch("falcon.cli.sys.stdout.isatty", return_value=True), patch(
            "falcon.cli._resource_snapshot", return_value=(collector, snapshot)
        ), patch("falcon.cli.FalconResourcesApp", App), patch(
            "falcon.cli.save_resources_view"
        ) as save, patch("falcon.cli.save_resources_consumer_sort") as save_sort:
            code = _resources_command(args, config, "/tmp/falcon-test-config")
            captured["persist_view"]("gpu-allocations")
            captured["persist_consumer_sort"]("memory")
        self.assertEqual(code, 0)
        self.assertEqual(captured["initial_view"], "gpu-allocations")
        self.assertEqual(captured["initial_consumer_sort"], "memory")
        self.assertTrue(captured["mouse"])
        save.assert_called_once_with(
            "gpu-allocations", "/tmp/falcon-test-config"
        )
        save_sort.assert_called_once_with("memory", "/tmp/falcon-test-config")

    def test_resources_tui_starts_and_loads_persistent_history(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")
        collector = SimpleNamespace(collect=lambda force=False: snapshot)
        captured = {}
        loaded = []

        class Store:
            def load(self, **kwargs):
                loaded.append(kwargs)
                return ["persisted"]

        class App:
            def __init__(self, _collector, **kwargs):
                captured.update(kwargs)

            def run(self, mouse=False):
                captured["history"] = captured["history_loader"]()

        args = SimpleNamespace(
            limit=100,
            consumer_limit=100,
            demo=None,
            output="human",
            node="node-a",
            gpu="h100",
            namespace=None,
        )
        with patch("falcon.cli.sys.stdout.isatty", return_value=True), patch(
            "falcon.cli._resource_snapshot", return_value=(collector, snapshot)
        ), patch("falcon.cli.FalconResourcesApp", App), patch(
            "falcon.cli.history_store", return_value=Store()
        ), patch("falcon.cli.ensure_history_collector") as ensure:
            code = _resources_command(args, DEFAULT_CONFIG, "/tmp/falcon-test-config")

        self.assertEqual(code, 0)
        ensure.assert_called_once_with(
            DEFAULT_CONFIG, config_file="/tmp/falcon-test-config"
        )
        self.assertEqual(
            loaded,
            [{"node_filter": "node-a", "gpu_filter": "h100"}],
        )
        self.assertEqual(captured["history"], ["persisted"])

    def test_resources_prefers_metrics_when_node_api_rbac_is_unavailable(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")

        class MetricsCollector:
            def __init__(self, url):
                self.url = url

            def collect(self, force=False):
                return snapshot

            def close(self):
                pass

        with patch("falcon.cli.load_config", return_value=DEFAULT_CONFIG), patch(
            "falcon.cli.MetricsClusterCollector", MetricsCollector
        ), patch("falcon.cli.ClusterCollector") as kubernetes_collector:
            code, stdout, stderr = self.invoke(
                "resources", "--limit", "2", "--output", "json"
            )
        value = json.loads(stdout)
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(value["kind"], "ClusterResources")
        self.assertEqual(value["meta"]["count"], 2)
        self.assertNotIn("jobs", value["data"]["summary"].get("nodes", {}))
        self.assertIn("gpu_availability", value["data"]["summary"])
        kubernetes_collector.assert_not_called()

    def test_resources_consumer_output_is_bounded(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")

        class MetricsCollector:
            def __init__(self, url):
                pass

            def collect(self, force=False):
                return snapshot

            def close(self):
                pass

        with patch("falcon.cli.load_config", return_value=DEFAULT_CONFIG), patch(
            "falcon.cli.MetricsClusterCollector", MetricsCollector
        ):
            code, stdout, stderr = self.invoke(
                "resources",
                "--node",
                "node-a-h100",
                "--consumer-limit",
                "1",
                "--output",
                "json",
            )
        value = json.loads(stdout)
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(len(value["data"]["nodes"][0]["consumers"]), 1)
        self.assertGreater(value["meta"]["consumers_count"], 1)
        self.assertEqual(value["meta"]["consumers_returned"], 1)

    def test_resources_json_omits_system_consumers_and_counts(self) -> None:
        snapshot = demo_cluster_snapshot("mixed")

        class MetricsCollector:
            def __init__(self, url):
                pass

            def collect(self, force=False):
                return snapshot

            def close(self):
                pass

        with patch("falcon.cli.load_config", return_value=DEFAULT_CONFIG), patch(
            "falcon.cli.MetricsClusterCollector", MetricsCollector
        ):
            code, stdout, stderr = self.invoke(
                "resources",
                "--node",
                "node-a-h100",
                "--output",
                "json",
            )
        value = json.loads(stdout)
        consumers = value["data"]["nodes"][0]["consumers"]
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(value["meta"]["consumers_count"], 3)
        self.assertNotIn("monitoring", {item["namespace"] for item in consumers})

    def test_metrics_reports_allocated_resource_utilization_over_interval(self) -> None:
        def row(
            utilization,
            *,
            vram_used,
            cpu_used,
            memory_used,
        ):
            return SimpleNamespace(
                job="job-a",
                status="Running",
                gpu_allocated_type="H100",
                gpu_allocated_count=2,
                gpu_util=utilization,
                gpu_metrics_available=True,
                gpu_memory_used_gib=vram_used,
                gpu_memory_total_gib=80.0,
                cpu_used=cpu_used,
                cpu_allocated=4.0,
                cpu_metrics_available=True,
                memory_used_gib=memory_used,
                memory_allocated_gib=8.0,
            )

        class Collector:
            last_error = ""

            def __init__(self):
                self.index = 0
                self.closed = False
                self.values = (
                    [row(50.0, vram_used=20.0, cpu_used=2.0, memory_used=4.0)],
                    [row(70.0, vram_used=40.0, cpu_used=3.0, memory_used=6.0)],
                )

            def collect(self):
                value = self.values[self.index]
                self.index += 1
                return value

            def close(self):
                self.closed = True

        collector = Collector()
        with patch("falcon.cli.UsageCollector", return_value=collector), patch(
            "falcon.cli.time.sleep"
        ) as sleeper:
            code, stdout, stderr = self.invoke(
                "metrics",
                "job-a",
                "--interval",
                "1",
            )
        value = json.loads(stdout)
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(value["kind"], "JobMetrics")
        self.assertEqual(value["data"]["allocation"]["gpu"]["count"], 2)
        self.assertEqual(value["data"]["allocation"]["cpu_cores"], 4.0)
        self.assertEqual(value["data"]["allocation"]["memory_bytes"], 8 * 1024**3)
        utilization = value["data"]["utilization"]
        self.assertEqual(utilization["gpu_percent"]["average"], 60.0)
        self.assertEqual(utilization["vram_percent"]["average"], 37.5)
        self.assertEqual(utilization["cpu_percent"]["average"], 62.5)
        self.assertEqual(utilization["memory_percent"]["average"], 62.5)
        self.assertEqual(
            value["data"]["eviction_policy"][
                "minimum_average_gpu_utilization_percent"
            ],
            90.0,
        )
        self.assertFalse(
            value["data"]["eviction_policy"]["observed_average_meets_minimum"]
        )
        self.assertTrue(collector.closed)
        sleeper.assert_called_once_with(1.0)

    def test_metrics_preserves_unavailable_unallocated_metrics_as_null(self) -> None:
        row = SimpleNamespace(
            job="pending-gpu",
            status="Pending",
            gpu_allocated_type="-",
            gpu_allocated_count=0,
            gpu_util=None,
            gpu_metrics_available=False,
            gpu_memory_used_gib=0.0,
            gpu_memory_total_gib=0.0,
            cpu_used=0.0,
            cpu_allocated=0.0,
            cpu_metrics_available=False,
            memory_used_gib=0.0,
            memory_allocated_gib=0.0,
        )

        class Collector:
            last_error = ""

            def collect(self):
                return [row]

            def close(self):
                pass

        with patch("falcon.cli.UsageCollector", return_value=Collector()), patch(
            "falcon.cli.time.sleep"
        ):
            code, stdout, stderr = self.invoke(
                "metrics", "pending-gpu", "--interval", "1", "--output", "json"
            )
        value = json.loads(stdout)
        self.assertEqual((code, stderr), (0, ""))
        self.assertIsNone(value["data"]["allocation"]["gpu"]["model"])
        self.assertIsNone(
            value["data"]["utilization"]["gpu_percent"]["average"]
        )
        self.assertIsNone(
            value["data"]["utilization"]["cpu_percent"]["average"]
        )

    def test_metrics_interval_is_bounded(self) -> None:
        code, _, stderr = self.invoke(
            "metrics", "job", "--interval", "301"
        )
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("--interval must be between 1 and 300", stderr)

    def test_kill_uses_job_deletion_and_delete_has_replacement(self) -> None:
        with patch("falcon.cli.kill", return_value=["job-a"]) as killer:
            code, stdout, stderr = self.invoke("kill", "job-a")
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Killed 1 Job", stdout)
        self.assertEqual(killer.call_args.args[1], ["job-a"])

        removed_code, _, removed_stderr = self.invoke("delete", "job-a")
        self.assertEqual(removed_code, EXIT_USAGE)
        self.assertIn("falcon kill JOB", removed_stderr)

    def test_unavailable_kubernetes_has_distinct_exit(self) -> None:
        class Broken:
            def __init__(self, namespace):
                pass

            def list_jobs(self):
                raise KubernetesError("Unable to connect to the server")

        with patch("falcon.cli.KubernetesClient", Broken):
            code, _, stderr = self.invoke("jobs")
        self.assertEqual(code, EXIT_KUBERNETES)
        self.assertIn("Unable to connect", stderr)


class SetupTests(unittest.TestCase):
    def test_interactive_skill_offer_warns_about_increased_usage(self) -> None:
        args = SimpleNamespace(
            uninstall_skills=None,
            install_skills=None,
            skip_skills=False,
            non_interactive=False,
        )
        stdout = io.StringIO()
        with patch("falcon.cli.detect_agents", return_value=["codex"]), patch(
            "builtins.input", return_value="none"
        ) as prompt, contextlib.redirect_stdout(stdout):
            code = _skills_setup(args)

        self.assertEqual(code, 0)
        self.assertIn("may increase coding-agent and tool usage", stdout.getvalue())
        self.assertIn("CPU/GPU workloads", stdout.getvalue())
        self.assertIn("Install Falcon skill", prompt.call_args.args[0])

    def test_resources_view_default_validation_and_atomic_persistence(self) -> None:
        self.assertEqual(DEFAULT_CONFIG["resources"]["last_view"], "nodes")
        self.assertEqual(DEFAULT_CONFIG["resources"]["consumer_sort"], "namespace")
        self.assertTrue(DEFAULT_CONFIG["resources"]["history_enabled"])
        self.assertEqual(DEFAULT_CONFIG["resources"]["history_hours"], 24)
        self.assertEqual(DEFAULT_CONFIG["presets"]["pro6000"]["max_count"], 2)
        for preset_name, maximum in (
            ("h100", 8),
            ("a6000", 2),
            ("2080ti", 4),
            ("pro6000", 2),
        ):
            self.assertEqual(
                DEFAULT_CONFIG["presets"][preset_name]["max_count"], maximum
            )
        invalid = json.loads(json.dumps(DEFAULT_CONFIG))
        invalid["resources"]["last_view"] = "utilization"
        with self.assertRaisesRegex(ValueError, "resources.last_view"):
            validate_config(invalid)

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / ".falconrc"
            run_setup(str(target), non_interactive=True, install_shell=False)
            original = load_config(str(target))
            saved = save_resources_view("gpu-allocations", str(target))
            saved_sort = save_resources_consumer_sort("memory", str(target))
            reloaded = load_config(str(target))
            self.assertEqual(saved, target)
            self.assertEqual(saved_sort, target)
            self.assertEqual(reloaded["resources"]["last_view"], "gpu-allocations")
            self.assertEqual(reloaded["resources"]["consumer_sort"], "memory")
            self.assertEqual(
                reloaded["resources"]["shared_memory_percent"],
                original["resources"]["shared_memory_percent"],
            )
            self.assertEqual(list(target.parent.glob(".*.tmp")), [])

        with self.assertRaisesRegex(ValueError, "resources.last_view"):
            save_resources_view("not-a-view", "/tmp/unused-falcon-config")
        with self.assertRaisesRegex(ValueError, "resources.last_view"):
            save_resources_view("gpu-overview", "/tmp/unused-falcon-config")
        with self.assertRaisesRegex(ValueError, "resources.consumer_sort"):
            save_resources_consumer_sort("pressure", "/tmp/unused-falcon-config")

        invalid = json.loads(json.dumps(DEFAULT_CONFIG))
        invalid["presets"]["pro6000"]["max_count"] = 3.5
        with self.assertRaisesRegex(ValueError, "presets.pro6000.max_count"):
            validate_config(invalid)

        for key, value in (
            ("history_enabled", "yes"),
            ("history_hours", 0),
            ("history_interval_seconds", 0),
            ("consumer_sort", "pressure"),
        ):
            invalid = json.loads(json.dumps(DEFAULT_CONFIG))
            invalid["resources"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_config(invalid)

    def test_invalid_persisted_resources_view_falls_back_to_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / ".falconrc"
            target.write_text(
                "version: 1\nresources:\n  last_view: retired-preview\n",
                encoding="utf-8",
            )
            config = load_config(str(target))
        self.assertEqual(config["resources"]["last_view"], "nodes")

    def test_retired_gpu_overview_view_migrates_to_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / ".falconrc"
            target.write_text(
                "version: 1\nresources:\n  last_view: gpu-overview\n",
                encoding="utf-8",
            )
            config = load_config(str(target))
        self.assertEqual(config["resources"]["last_view"], "nodes")

    def test_noninteractive_setup_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / ".falconrc"
            first, _ = run_setup(
                str(target), non_interactive=True, install_shell=False
            )
            content = target.read_text(encoding="utf-8")
            second, _ = run_setup(
                str(target), non_interactive=True, install_shell=False
            )
            self.assertEqual((first, second), (target, target))
            self.assertEqual(target.read_text(encoding="utf-8"), content)

    def test_setup_config_loads_without_active_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / ".falconrc"
            with patch.dict(
                os.environ,
                {"CONDA_PREFIX": "", "VIRTUAL_ENV": "", "FALCON_NAMESPACE": "test"},
                clear=False,
            ):
                run_setup(str(target), non_interactive=True, install_shell=False)
                value = load_config(str(target))
            self.assertEqual(value["cluster"]["namespace"], "test")
            self.assertEqual(
                value["cluster"]["kube_state_metrics_url"],
                "http://localhost:30080/metrics",
            )
            self.assertEqual(
                value["runtime"]["image"],
                "registry.gitlab.com/hvlabs/teams/ai/container-images/base:"
                "ubuntu24.04-cuda13.0.2-runtime-withtools-v1.0.0",
            )
            self.assertEqual(
                value["runtime"]["image_pull_secrets"],
                ["hv-gitlab-registry"],
            )

    def test_existing_config_inherits_metrics_endpoint_used_by_old_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / ".falconrc"
            target.write_text(
                "version: 1\ncluster:\n  namespace: research\n",
                encoding="utf-8",
            )
            value = load_config(str(target))
        self.assertEqual(
            value["cluster"]["kube_state_metrics_url"],
            "http://localhost:30080/metrics",
        )

    def test_existing_config_resolves_null_home_for_debug_shells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            target = home / ".falconrc"
            target.write_text(
                "version: 1\n"
                "cluster:\n"
                "  namespace: research\n"
                "runtime:\n"
                "  home: null\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"HOME": str(home)}, clear=False):
                value = load_config(str(target))
        self.assertEqual(value["runtime"]["home"], str(home))

    def test_setup_migrates_obsolete_shell_init_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            target = home / ".falconrc"
            rc_path = home / ".zshrc"
            rc_path.write_text(
                "# user setting\n"
                'eval "$(falcon shell-init zsh)"\n'
                'eval "$(/opt/falcon/bin/falcon shell-init zsh)"\n'
                "export KEEP_ME=yes\n",
                encoding="utf-8",
            )
            environment = {
                "HOME": str(home),
                "FALCON_SHELL": "zsh",
                "FALCON_NAMESPACE": "test",
            }
            with patch.dict(os.environ, environment, clear=False):
                run_setup(str(target), non_interactive=True)
                first = rc_path.read_text(encoding="utf-8")
                run_setup(str(target), non_interactive=True)
                second = rc_path.read_text(encoding="utf-8")

            self.assertNotIn("shell-init", first)
            self.assertIn("export KEEP_ME=yes", first)
            self.assertIn("falcon completion zsh", first)
            self.assertEqual(first.count("# >>> falcon completion >>>"), 1)
            self.assertEqual(second, first)


class KubernetesClientTests(unittest.TestCase):
    def test_commands_are_direct_argv_without_shell(self) -> None:
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, '{"items":[]}', "")

        client = KubernetesClient("research", runner=runner)
        client.list_jobs()
        argv, kwargs = calls[0]
        self.assertEqual(argv[:3], ["kubectl", "get", "jobs.batch"])
        self.assertNotIn("shell", kwargs)

    def test_debug_shell_uses_shell_specific_prompt_and_container_workdir(self) -> None:
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, "", "")

        client = KubernetesClient("research", runner=runner)
        pod = {"metadata": {"name": "debug-pod"}}
        with patch.object(client, "wait_for_pod", return_value=pod):
            result = client.exec_shell(
                "debug-job",
                "zsh",
                prompt_label="2080tix2",
                rc_path="/home/alice/.zshrc",
            )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(calls), 3)
        mkdir_argv, _ = calls[0]
        self.assertEqual(
            mkdir_argv,
            [
                "kubectl", "exec",
                "--namespace", "research", "debug-pod",
                "--", "mkdir", "-p", "/tmp/falcon-zdotdir",
            ],
        )
        tee_argv, tee_kwargs = calls[1]
        self.assertEqual(
            tee_argv,
            [
                "kubectl", "exec", "--stdin",
                "--namespace", "research", "debug-pod",
                "--", "tee", "/tmp/falcon-zdotdir/.zshrc",
            ],
        )
        wrapper = tee_kwargs["input"]
        self.assertIn('source "$FALCON_USER_RC"', wrapper)
        self.assertIn("CONDA_AUTO_ACTIVATE_BASE=false", wrapper)
        self.assertIn('export CONDA_PREFIX="$FALCON_SAVED_CONDA_PREFIX"', wrapper)
        self.assertIn("add-zsh-hook precmd _falcon_prompt_prefix", wrapper)
        self.assertIn('PROMPT="(${FALCON_PROMPT_LABEL})', wrapper)
        argv, kwargs = calls[2]
        self.assertEqual(
            argv,
            [
                "kubectl", "exec", "--stdin", "--tty",
                "--namespace", "research", "debug-pod",
                "--", "env",
                "FALCON_PROMPT_LABEL=2080tix2",
                "FALCON_USER_RC=/home/alice/.zshrc",
                "CONDA_AUTO_ACTIVATE_BASE=false",
                "CONDA_CHANGEPS1=false",
                "ZDOTDIR=/tmp/falcon-zdotdir",
                "zsh", "-i",
            ],
        )
        self.assertFalse(kwargs["capture_output"])

    def test_debug_bash_sources_user_rc_and_restores_environment(self) -> None:
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, "", "")

        client = KubernetesClient("research", runner=runner)
        pod = {"metadata": {"name": "debug-pod"}}
        with patch.object(client, "wait_for_pod", return_value=pod):
            result = client.exec_shell(
                "debug-job",
                "bash",
                prompt_label="a6000",
                rc_path="/home/alice/.bashrc",
            )
        self.assertEqual(result.returncode, 0)
        wrapper = calls[1][1]["input"]
        self.assertIn('source "$FALCON_USER_RC"', wrapper)
        self.assertIn('export VIRTUAL_ENV="$FALCON_SAVED_VIRTUAL_ENV"', wrapper)
        self.assertEqual(
            calls[2][0][-5:],
            [
                "bash",
                "--noprofile",
                "--rcfile",
                "/tmp/falcon-bash/.bashrc",
                "-i",
            ],
        )

    def test_timeout_becomes_kubernetes_error(self) -> None:
        def runner(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

        with self.assertRaisesRegex(KubernetesError, "timed out"):
            KubernetesClient("default", runner=runner).list_jobs()

    def test_invalid_json_is_rejected(self) -> None:
        def runner(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, "not json", "")

        with self.assertRaisesRegex(KubernetesError, "invalid JSON"):
            KubernetesClient("default", runner=runner).list_jobs()


if __name__ == "__main__":
    unittest.main()
