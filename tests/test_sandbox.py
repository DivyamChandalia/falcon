from __future__ import annotations

import copy
import io
import json
import os
import tempfile
import threading
import unittest
import uuid
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from falcon.cli import _parser, _rewrite_shorthand
from falcon.config import DEFAULT_CONFIG
from falcon.kubernetes import KubernetesError
from falcon.models import JobRequest, SubmittedJob
from falcon.planning import plan_cpu_resources
from falcon.sandbox import (
    SandboxMount,
    SandboxPolicy,
    agent_argv,
    agent_state,
    default_session_name,
    discover_agent,
    installation_mounts,
    make_policy,
    relay_service,
    sandbox_manifest,
)
from falcon.sandbox_runtime import Relay, atomic_json, http, proxy_cli, run_agent, session_status
from falcon.sandbox_service import (
    Broker,
    broker_writable_paths,
    environment_mounts,
    launch_session,
    manage_session,
    prepare_session_directory,
    resource_plan,
    runtime_resources,
    stop_exited_session,
)


class FakeClient:
    def __init__(self):
        self.jobs = {}
        self.created = 0
        self.deleted = []

    def get_json(self, resource, name):
        if name not in self.jobs:
            raise KubernetesError("NotFound")
        return copy.deepcopy(self.jobs[name])

    def create_job(self, manifest):
        job = copy.deepcopy(manifest)
        uid = str(uuid.uuid4())
        job["metadata"]["uid"] = uid
        name = job["metadata"]["name"]
        self.jobs[name] = job
        self.created += 1
        return SubmittedJob(name, "test", uid=uid, created=True)

    def list_json(self, resource, labels):
        return {"items": list(self.jobs.values())}

    def delete_jobs(self, names):
        for name in names:
            self.deleted.append(name)
            self.jobs.pop(name, None)


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.home = self.root / "home"
        self.workspace = self.root / "workspace"
        self.data = self.root / "data"
        for path in (self.home, self.workspace, self.data):
            path.mkdir()
        self.addCleanup(patch.stopall)
        patch("pathlib.Path.home", return_value=self.home).start()
        patch.dict(os.environ, {"XDG_STATE_HOME": str(self.root / "host-state"), "KUBECONFIG": str(self.home / ".kube/config"), "CODEX_HOME": str(self.home / ".codex")}).start()
        patch("falcon.sandbox_service.RuntimeEnvironment.from_current", return_value=None).start()
        self.policy = make_policy("agent-test", "test", str(self.workspace), [str(self.data)])
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["runtime"].update(run_as_user=10122, run_as_group=10122)
        self.plan = plan_cpu_resources("2", "4Gi")

    def request(self, **kwargs):
        return JobRequest("child", "test", ("python", "train.py"), compute=self.plan.compute, working_dir=str(self.workspace), **kwargs)

    def test_policy_roundtrip(self):
        self.assertEqual(SandboxPolicy.from_dict(self.policy.to_dict()), self.policy)

    def test_readonly_relative_paths_resolve_from_launch_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(self.workspace)
            policy = make_policy("agent-test", "test", ".", ["../data"])
        finally:
            os.chdir(previous)
        self.assertEqual(policy.workspace, str(self.workspace))
        self.assertEqual(policy.data, (SandboxMount(str(self.data), str(self.data)),))

    def test_cpu_defaults_are_one_core_and_one_gib(self):
        _, plan = resource_plan({}, self.config, "test")
        self.assertEqual(plan.compute.cpu, "1")
        self.assertEqual(plan.compute.memory, "1Gi")

    def test_children_inherit_direct_writable_environment(self):
        broker = self.broker()
        base = self.root / "base"
        base.mkdir()
        broker.policy = replace(broker.policy, python_environment=str(base), python_source=str(base), runtime_tools=(SandboxMount(str(base), str(base), False),))
        self.assertEqual(SandboxPolicy.from_dict(broker.policy.to_dict()), broker.policy)
        for selected in (None, "auto", str(base), "none"):
            with self.subTest(environment=selected):
                result = broker.dispatch(self.message(environment=selected))
                container = broker.client.jobs[result["name"]]["spec"]["template"]["spec"]["containers"][0]
                env = {e["name"]: e["value"] for e in container["env"]}
                self.assertEqual(env["VIRTUAL_ENV"], "" if selected == "none" else str(base))
                if selected != "none":
                    self.assertIn(str(base) + "/bin", env["PATH"])
                base_mount = next(m for m in container["volumeMounts"] if m["mountPath"] == str(base))
                self.assertFalse(base_mount.get("readOnly", False))

    def test_environment_mounts_keep_base_readonly(self):
        from falcon.models import EnvironmentKind, RuntimeEnvironment

        env = self.root / "venv"
        env.mkdir()
        base = self.root / "python/bin"
        base.mkdir(parents=True)
        mounts = environment_mounts(RuntimeEnvironment(EnvironmentKind.VENV, env, base))
        self.assertIn(SandboxMount(str(env), str(env), False), mounts)
        self.assertIn(SandboxMount(str(base.parent), str(base.parent)), mounts)
        self.assertEqual([m.source for m in mounts if not m.read_only], [str(env)])
        policy = replace(self.policy, runtime_tools=mounts)
        self.assertIn(env, broker_writable_paths(policy))
        self.assertNotIn(base.parent, broker_writable_paths(policy))

    def test_direct_environment_launch_does_not_create_overlay(self):
        from falcon.models import EnvironmentKind, RuntimeEnvironment

        env = self.root / "active-env"
        (env / "bin").mkdir(parents=True)
        (env / "conda-meta").mkdir()
        binary = self.root / "release/bin/codex"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"\x7fELF")
        binary.chmod(0o755)
        args = _parser(self.config).parse_args(["sandbox", "--agent-bin", str(binary), "--dry-run", "--output", "json", "--", "codex"])
        with patch("falcon.sandbox_service.RuntimeEnvironment.from_current", return_value=RuntimeEnvironment(EnvironmentKind.CONDA, env)), patch("pathlib.Path.cwd", return_value=self.workspace), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launch_session(args, self.config), 0)
        data = json.loads(output.getvalue())["data"]
        policy = data["policy"]
        self.assertEqual(policy["python_environment"], str(env))
        container = data["manifests"][0]["spec"]["template"]["spec"]["containers"][0]
        self.assertFalse(next(m for m in container["volumeMounts"] if m["mountPath"] == str(env)).get("readOnly", False))
        values = {e["name"]: e["value"] for e in container["env"]}
        self.assertEqual(values["CONDA_PREFIX"], str(env))
        spec = json.loads(data["manifests"][3]["stringData"]["spec.json"])
        self.assertNotIn("python_environment", spec)
        self.assertFalse((self.workspace / ".falcon/sandbox/falcon-codex-workspace/environment").exists())

    def test_environment_cannot_mutate_broker_installation(self):
        import sys

        from falcon.models import RuntimeEnvironment

        binary = self.root / "release/bin/codex"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"\x7fELF")
        binary.chmod(0o755)
        args = _parser(self.config).parse_args(["sandbox", "--agent-bin", str(binary), "--dry-run", "--", "codex"])
        with patch("falcon.sandbox_service.RuntimeEnvironment.from_current", return_value=RuntimeEnvironment.from_path(sys.prefix)), patch("pathlib.Path.cwd", return_value=self.workspace), self.assertRaisesRegex(ValueError, "trusted broker"):
            launch_session(args, self.config)

    def test_default_session_names(self):
        for harness in ("codex", "claude", "opencode"):
            with self.subTest(harness=harness):
                self.assertEqual(default_session_name(harness, Path("/work/jet-k8s")), f"falcon-{harness}-jet-k8s")
        self.assertEqual(default_session_name("codex", Path("/work/My_Project.v2")), "falcon-codex-my-project-v2")
        self.assertEqual(default_session_name("codex", Path("/work/___")), "falcon-codex-workspace")
        long_path = Path("/work/" + "x" * 100)
        name = default_session_name("codex", long_path)
        self.assertLessEqual(len(name), 63)
        self.assertEqual(name, default_session_name("codex", long_path))
        self.assertNotEqual(name, default_session_name("codex", Path(str(long_path) + "y")))
        JobRequest(name, "test")

    def test_readonly_sources_and_workspace(self):
        self.assertEqual(self.policy.workspace, str(self.workspace.resolve()))
        self.assertTrue(self.policy.data[0].read_only)

    def test_rejects_overlapping_mounts(self):
        child = self.workspace / "child"
        child.mkdir()
        for readonly in ([str(child)], [str(self.root)], [str(self.data), str(self.data)]):
            with self.subTest(readonly=readonly), self.assertRaises(ValueError):
                make_policy("agent-test", "test", str(self.workspace), readonly)

    def test_rejects_home_credentials_and_broker_state(self):
        (self.home / ".kube").mkdir()
        for path in (self.home, self.home / ".kube", Path("/etc"), Path("/")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                make_policy("agent-test", "test", str(self.workspace), [str(path)])

    def test_symlink_canonicalization(self):
        alias = self.root / "alias"
        alias.symlink_to(self.data, target_is_directory=True)
        policy = make_policy("agent-test", "test", str(self.workspace), [str(alias)])
        self.assertEqual(policy.data[0].source, str(self.data))

    def test_symlink_to_credentials_is_denied(self):
        (self.home / ".kube").mkdir()
        alias = self.root / "alias"
        alias.symlink_to(self.home / ".kube", target_is_directory=True)
        with self.assertRaises(ValueError):
            make_policy("agent-test", "test", str(self.workspace), [str(alias)])

    def test_manifest_ignores_broad_runtime_mounts_and_security_overrides(self):
        self.config["runtime"].update(volumes=[str(self.root)], mount_home=True, service_account="admin", container_security_context={"privileged": True}, security_context={"runAsUser": 0}, environment={"KUBECONFIG": "/secret"})
        result = sandbox_manifest(self.policy, self.request(), self.plan, self.config)
        pod = result["spec"]["template"]["spec"]
        container = pod["containers"][0]
        self.assertFalse(pod["automountServiceAccountToken"])
        self.assertNotIn("serviceAccountName", pod)
        self.assertNotIn("privileged", container["securityContext"])
        self.assertEqual(container["securityContext"]["capabilities"], {"drop": ["ALL"]})
        self.assertTrue(container["securityContext"]["readOnlyRootFilesystem"])
        hosts = [v["hostPath"]["path"] for v in pod["volumes"] if "hostPath" in v]
        self.assertEqual(hosts, [str(self.workspace), str(self.data)])
        mounts = {m["mountPath"]: m for m in container["volumeMounts"]}
        self.assertNotIn("readOnly", mounts[str(self.workspace)])
        self.assertTrue(mounts[str(self.data)]["readOnly"])

    def test_namespace_and_cwd_escape_denied(self):
        for request in (
            JobRequest("child", "elsewhere", ("true",)),
            JobRequest("child", "test", ("true",), working_dir=str(self.workspace / "../data")),
            JobRequest("child", "test", ("true",), working_dir=str(self.data)),
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                sandbox_manifest(self.policy, request, self.plan, self.config)

    def test_reserved_env_denied(self):
        for key in ("KUBECONFIG", "HOME", "PYTHONPATH", "LD_PRELOAD", "FALCON_SANDBOX_URL"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                sandbox_manifest(self.policy, self.request(env={key: "bad"}), self.plan, self.config)

    def test_parent_has_state_but_children_do_not(self):
        state = self.home / ".codex"
        state.mkdir()
        policy = SandboxPolicy(self.policy.session, "test", str(self.workspace), self.policy.data, state=(SandboxMount(str(state), str(state), False),), private=(str(state / "tmp"),))
        parent = sandbox_manifest(policy, self.request(), self.plan, self.config, parent=True)
        child = sandbox_manifest(policy, self.request(), self.plan, self.config)
        self.assertEqual(parent["spec"]["backoffLimit"], 3)
        self.assertEqual(child["spec"]["backoffLimit"], 0)
        parent_hosts = [v["hostPath"]["path"] for v in parent["spec"]["template"]["spec"]["volumes"] if "hostPath" in v]
        child_hosts = [v["hostPath"]["path"] for v in child["spec"]["template"]["spec"]["volumes"] if "hostPath" in v]
        self.assertIn(str(state), parent_hosts)
        self.assertNotIn(str(state), child_hosts)
        paths = [m["mountPath"] for m in parent["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]]
        self.assertLess(paths.index(str(state)), paths.index(str(state / "tmp")))

    def test_agent_permission_and_recovery_flags(self):
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", agent_argv("codex", "/bin/codex", [], "full"))
        self.assertIn("--no-daemon", agent_argv("codex", "/bin/codex", [], "agent"))
        self.assertNotIn("--dangerously-skip-permissions", agent_argv("claude", "/bin/claude", [], "agent"))
        self.assertIn("--dangerously-skip-permissions", agent_argv("claude", "/bin/claude", [], "full"))
        self.assertEqual(agent_argv("opencode", "/bin/opencode", [], "full", "ses_example"), ["/bin/opencode", "--session", "ses_example"])
        with self.assertRaises(ValueError):
            agent_argv("script", "/bin/script", [], "full", "id")

    def test_missing_agent_is_actionable(self):
        with patch("shutil.which", return_value=None), self.assertRaisesRegex(ValueError, "agent-bin"):
            discover_agent(["claude"])

    def test_native_installer_mounts_release_not_bin_symlinks(self):
        binary = self.root / "release/bin/codex"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"\x7fELF")
        binary.chmod(0o755)
        self.assertEqual(installation_mounts(binary), (SandboxMount(str(binary.parent.parent), str(binary.parent.parent)),))

    def test_agent_state_keeps_daemon_ipc_private(self):
        mounts, private = agent_state("codex")
        self.assertFalse(mounts[0].read_only)
        self.assertIn(str(self.home / ".codex/app-server-daemon"), private)

    def test_service_is_not_public(self):
        service = relay_service(self.policy)
        self.assertNotIn("type", service["spec"])
        self.assertEqual(service["spec"]["selector"], {"job-name": "agent-test"})

    def test_parser_preserves_agent_args(self):
        args = _parser(self.config).parse_args(["sandbox", "--read-only", str(self.data), "--detach", "--", "codex", "--model", "example"])
        self.assertEqual(args.command, ["--", "codex", "--model", "example"])
        self.assertEqual(_rewrite_shorthand(["sandbox", "--help"], self.config), ["sandbox", "--help"])

    def test_dry_run_never_stages_files_or_submits(self):
        binary = self.root / "release/bin/codex"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"\x7fELF")
        binary.chmod(0o755)
        for options, name in (([], "falcon-codex-workspace"), (["--name", "dry-agent"], "dry-agent")):
            args = _parser(self.config).parse_args(["sandbox", *options, "--agent-bin", str(binary), "--dry-run", "--", "codex"])
            with self.subTest(name=name), patch("os.chdir"), patch("pathlib.Path.cwd", return_value=self.workspace), patch("falcon.sandbox_service.KubernetesClient") as client, redirect_stdout(io.StringIO()) as output:
                self.assertEqual(launch_session(args, self.config), 0)
            client.assert_not_called()
            self.assertIn("REDACTED", output.getvalue())
            self.assertIn(name, output.getvalue())
        self.assertFalse((self.root / "host-state/falcon/sandboxes").exists())
        self.assertFalse((self.home / ".codex").exists())

    def test_codex_launch_mounts_copy_not_original_state(self):
        binary = self.root / "release/bin/codex"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"\x7fELF")
        binary.chmod(0o755)
        (self.home / ".codex").mkdir()
        (self.home / ".codex/config.toml").write_text('model = "test"\n')
        args = _parser(self.config).parse_args(["sandbox", "--name", "filtered-test", "--agent-bin", str(binary), "--detach", "--", "codex", "--version"])
        client = FakeClient()
        client._run = lambda *a, **k: None
        with patch("pathlib.Path.cwd", return_value=self.workspace), patch("falcon.sandbox_service.KubernetesClient", return_value=client), patch("falcon.sandbox_service.tool_probe"), patch("falcon.sandbox_service.start_broker"), patch("falcon.sandbox_service.shutil.copytree"), patch("falcon.commands.remember_job"), patch("falcon.sandbox_service.runtime_resources", return_value=({"metadata": {}}, {"metadata": {}})), redirect_stdout(io.StringIO()):
            self.assertEqual(launch_session(args, self.config), 0)
        pod = client.jobs["filtered-test"]["spec"]["template"]["spec"]
        hosts = [v["hostPath"]["path"] for v in pod["volumes"] if "hostPath" in v]
        self.assertNotIn(str(self.home / ".codex"), hosts)
        for component in ("sessions", "archived_sessions"):
            self.assertIn(str(self.home / ".codex" / component), hosts)
            mount = next(m for m in pod["containers"][0]["volumeMounts"] if m["mountPath"] == str(self.home / ".codex" / component))
            self.assertFalse(mount.get("readOnly", False))
        self.assertTrue(any(".falcon-sandbox-state" in p for p in hosts))
        self.assertFalse(any(v["name"].startswith("private-") for v in pod["volumes"]))
        env = {e["name"]: e["value"] for e in pod["containers"][0]["env"]}
        self.assertEqual(env["CODEX_HOME"], "/opt/falcon-agent-state")
        self.assertEqual(env["CODEX_SQLITE_HOME"], "/opt/falcon-codex-runtime")
        self.assertIn("codex-runtime", {v["name"] for v in pod["volumes"]})
        self.assertEqual(pod["containers"][0]["resources"]["requests"]["cpu"], "1")
        self.assertEqual(pod["containers"][0]["resources"]["requests"]["memory"], "1Gi")

    def broker(self, agent="codex"):
        directory = self.root / "broker"
        directory.mkdir(exist_ok=True)
        atomic_json(directory / "session.json", {"policy": self.policy.to_dict(), "config": self.config, "token": "capability", "agent": agent, "legacy_config": str(self.home / ".claude.json")})
        return Broker(directory, FakeClient())

    def registered_session(self):
        directory = self.root / "sessions" / self.policy.session
        directory.mkdir(parents=True)
        atomic_json(directory / "session.json", {"policy": self.policy.to_dict(), "config": self.config, "token": "capability", "parent_uid": "original"})
        atomic_json(directory / "stopped", {"reason": "launch failed"})
        return directory

    def test_stale_registration_is_archived_not_deleted(self):
        directory = self.registered_session()
        with patch("falcon.sandbox_service.KubernetesClient", return_value=FakeClient()):
            prepare_session_directory(directory, "test")
        self.assertFalse(directory.exists())
        archived = list((directory.parent / "archive").iterdir())
        self.assertEqual(len(archived), 1)
        self.assertTrue((archived[0] / "session.json").exists())

    def test_incomplete_pre_job_registration_is_recoverable(self):
        directory = self.root / "partial-launch"
        directory.mkdir()
        with patch("falcon.sandbox_service.KubernetesClient", return_value=FakeClient()):
            prepare_session_directory(directory, "test")
        self.assertFalse(directory.exists())

    def test_existing_job_prevents_relaunch(self):
        directory = self.registered_session()
        client = FakeClient()
        client.jobs[self.policy.session] = {}
        with patch("falcon.sandbox_service.KubernetesClient", return_value=client), self.assertRaisesRegex(ValueError, "already exists"):
            prepare_session_directory(directory, "test")
        self.assertTrue(directory.exists())

    def test_cluster_errors_do_not_release_registration(self):
        directory = self.registered_session()
        with patch("falcon.sandbox_service.KubernetesClient") as client:
            client.return_value.get_json.side_effect = KubernetesError("TLS handshake timeout")
            with self.assertRaisesRegex(KubernetesError, "TLS handshake timeout"):
                prepare_session_directory(directory, "test")
        self.assertTrue(directory.exists())

    def test_stale_session_with_children_requires_stop(self):
        directory = self.registered_session()
        atomic_json(directory / "registry.json", {"request": {"name": "child", "uid": "child-uid"}})
        with patch("falcon.sandbox_service.KubernetesClient", return_value=FakeClient()), self.assertRaisesRegex(ValueError, "child Job records"):
            prepare_session_directory(directory, "test")
        self.assertTrue(directory.exists())

    def test_status_reports_missing_job(self):
        directory = self.registered_session()
        client = FakeClient()
        client.namespace = "test"
        with patch("falcon.sandbox_service.session_path", return_value=directory), patch("falcon.sandbox_service.KubernetesClient", return_value=client), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(manage_session("status", self.policy.session, output="json"), 0)
        status = json.loads(output.getvalue())
        self.assertEqual(status["job"], {"state": "missing"})
        self.assertEqual(status["local_status"]["reason"], "launch failed")

    def test_status_defaults_to_readable_output(self):
        directory = self.registered_session()
        client = FakeClient()
        client.namespace = "test"
        client.jobs[self.policy.session] = {"status": {"active": 1}}
        atomic_json(directory / "broker.json", {"state": "running"})
        with patch("falcon.sandbox_service.session_path", return_value=directory), patch("falcon.sandbox_service.KubernetesClient", return_value=client), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(manage_session("status", self.policy.session), 0)
        rendered = output.getvalue()
        self.assertIn(f"Sandbox:   {self.policy.session}", rendered)
        self.assertIn("Job:       running", rendered)
        self.assertIn("Broker:    running", rendered)
        self.assertIn("Ctrl+B, then D", rendered)
        self.assertFalse(rendered.startswith("{"))

    def test_stop_missing_job_releases_name_on_next_launch(self):
        directory = self.registered_session()
        client = FakeClient()
        with patch("falcon.sandbox_service.session_path", return_value=directory), patch("falcon.sandbox_service.KubernetesClient", return_value=client), redirect_stdout(io.StringIO()):
            self.assertEqual(manage_session("stop", self.policy.session), 0)
            prepare_session_directory(directory, "test")
        self.assertFalse(directory.exists())

    def test_cleanup_error_does_not_mask_launch_failure(self):
        binary = self.root / "tool"
        binary.write_bytes(b"#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        args = _parser(self.config).parse_args(["sandbox", "--name", "cleanup-test", "--agent-bin", str(binary), "--detach", "--", "tool"])
        client = FakeClient()
        client._run = lambda *a, **k: (_ for _ in ()).throw(KubernetesError("original create failure"))
        client.delete_jobs = lambda *a: (_ for _ in ()).throw(KubernetesError("cleanup timeout"))
        with patch("pathlib.Path.cwd", return_value=self.workspace), patch("falcon.sandbox_service.KubernetesClient", return_value=client), patch("falcon.sandbox_service.tool_probe"), patch("falcon.sandbox_service.runtime_resources", return_value=({"metadata": {}}, {"metadata": {}})), patch("falcon.sandbox_service.shutil.copytree"), self.assertRaisesRegex(KubernetesError, "original create failure"):
            launch_session(args, self.config)

    def message(self, operation="launch", **payload):
        if operation == "launch":
            payload = {"cpu": "2", "memory": "4Gi", "command": ["python", "train.py"], **payload}
        return {"protocol": 1, "id": str(uuid.uuid4()), "operation": operation, "payload": payload}

    def test_broker_idempotent_launch(self):
        broker = self.broker()
        message = self.message()
        first = broker.dispatch(message)
        self.assertEqual(first, broker.dispatch(message))
        self.assertEqual(broker.client.created, 1)

    def test_restart_recovers_create_before_ack(self):
        broker = self.broker()
        message = self.message()
        original_save = broker.save
        count = 0

        def save():
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("crash after create")
            original_save()

        with patch.object(broker, "save", side_effect=save), self.assertRaises(OSError):
            broker.dispatch(message)
        recovered = Broker(broker.directory, broker.client)
        result = recovered.dispatch(message)
        self.assertTrue(result["uid"])
        self.assertEqual(broker.client.created, 1)

    def test_same_id_with_changed_payload_is_denied(self):
        broker = self.broker()
        message = self.message()
        broker.dispatch(message)
        message["payload"]["command"] = ["evil"]
        with self.assertRaisesRegex(ValueError, "reused"):
            broker.dispatch(message)

    def test_raw_manifest_and_security_fields_denied(self):
        broker = self.broker()
        for key in ("manifest", "volumes", "namespace", "serviceAccountName", "securityContext", "hostPID"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                broker.dispatch(self.message(**{key: "bad"}))
        self.assertEqual(broker.client.created, 0)

    def test_malformed_launch_values_fail_before_api_calls(self):
        broker = self.broker()
        for key in ("image", "environment", "working_dir", "cpu", "name", "gpu", "pin_node"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                broker.dispatch(self.message(**{key: {"invalid": True}}))
        self.assertEqual(broker.client.created, 0)

    def test_broker_denies_unowned_job_and_replaced_uid(self):
        broker = self.broker()
        with self.assertRaisesRegex(ValueError, "belong"):
            broker.dispatch(self.message("kill", job="someone-else"))
        submitted = broker.dispatch(self.message())
        broker.client.jobs[submitted["name"]]["metadata"]["uid"] = "replacement"
        with self.assertRaisesRegex(ValueError, "UID"):
            broker.dispatch(self.message("kill", job=submitted["name"]))
        self.assertFalse(broker.client.deleted)

    def test_children_have_only_approved_mounts_and_proxy_credentials(self):
        broker = self.broker()
        submitted = broker.dispatch(self.message())
        pod = broker.client.jobs[submitted["name"]]["spec"]["template"]["spec"]
        paths = [v["hostPath"]["path"] for v in pod["volumes"] if "hostPath" in v]
        self.assertEqual(paths, [str(self.workspace), str(self.data)])
        self.assertTrue(any("configMap" in volume for volume in pod["volumes"]))
        self.assertFalse(pod["automountServiceAccountToken"])
        self.assertNotIn("KUBECONFIG", {e["name"] for e in pod["containers"][0]["env"]})

    def test_config_sync_preserves_concurrent_host_changes(self):
        broker = self.broker("claude")
        target = self.home / ".claude.json"
        atomic_json(target, {"model": "host", "theme": "old"})
        result = broker.dispatch(self.message("config-sync", base={"model": "old", "theme": "old"}, value={"model": "agent", "theme": "new"}))
        self.assertEqual(result["value"], {"model": "host", "theme": "new"})
        self.assertEqual(result["conflicts"], ["model"])
        self.assertEqual(json.loads(target.read_text()), result["value"])

    def test_config_sync_cannot_choose_host_path(self):
        broker = self.broker("claude")
        with self.assertRaises(ValueError):
            broker.dispatch(self.message("config-sync", base={}, value={}, path="/etc/passwd"))

    def test_proxy_retains_cpu_only_launch_and_does_not_use_kubectl(self):
        with patch("falcon.sandbox_runtime.proxy_request", return_value={"name": "child", "namespace": "test"}) as request, redirect_stdout(io.StringIO()):
            self.assertEqual(proxy_cli(["-c", "2", "-m", "4Gi", "--", "python", "train.py"]), 0)
        self.assertEqual(request.call_args.args[0], "launch")
        self.assertEqual(request.call_args.args[1]["command"], ["python", "train.py"])

    def test_unknown_generic_command_is_not_replayed(self):
        record = self.root / "agent.json"
        atomic_json(record, {"state": "running", "session_id": None})
        spec = self.root / "spec.json"
        atomic_json(spec, {"record": str(record), "argv": ["script"], "agent": "script"})
        with patch("subprocess.Popen") as spawn:
            self.assertEqual(run_agent(spec), 1)
        spawn.assert_not_called()

    def test_exited_agent_does_not_restart(self):
        record = self.root / "agent.json"
        atomic_json(record, {"state": "exited", "session_id": "id"})
        spec = self.root / "spec.json"
        atomic_json(spec, {"record": str(record), "argv": ["codex"], "agent": "codex"})
        with patch("subprocess.Popen") as spawn, redirect_stdout(io.StringIO()):
            self.assertEqual(run_agent(spec), 0)
        spawn.assert_not_called()

    def test_tmux_control_c_stops_but_detach_keeps_session(self):
        with patch("falcon.sandbox_service.shutil.which", return_value=None):
            _, secret = runtime_resources("test-session", "test", {}, "/usr/bin/python3")
        config = secret["stringData"]["tmux.conf"]
        self.assertIn("bind-key -n C-c kill-session", config)
        self.assertIn("remain-on-exit off", config)
        self.assertNotIn("unbind", config)

    def test_detached_tmux_keeps_agent_running(self):
        record = self.root / "agent.json"
        atomic_json(record, {"state": "running", "session_id": "id"})
        with patch("falcon.sandbox_runtime.subprocess.run") as probe:
            probe.return_value.returncode = 0
            self.assertEqual(session_status(record)["state"], "running")

    def test_closed_tmux_records_exit_for_cleanup(self):
        record = self.root / "agent.json"
        atomic_json(record, {"state": "running", "session_id": "id"})
        with patch("falcon.sandbox_runtime.subprocess.run") as probe:
            probe.return_value.returncode = 1
            status = session_status(record)
        self.assertEqual(status["state"], "exited")
        self.assertEqual(status["session_id"], "id")
        self.assertEqual(json.loads(record.read_text()), status)

    def test_natural_exit_preserves_agent_exit_code(self):
        record = self.root / "agent.json"
        atomic_json(record, {"state": "exited", "exit_code": 0})
        with patch("falcon.sandbox_runtime.subprocess.run") as probe:
            self.assertEqual(session_status(record)["exit_code"], 0)
        probe.assert_not_called()

    def test_exit_cleanup_deletes_parent_not_children(self):
        broker = self.broker()
        broker.session["parent_uid"] = "parent-uid"
        broker.client.jobs[self.policy.session] = {"metadata": {"uid": "parent-uid", "labels": {"falcon.dev/sandbox": self.policy.session}}}
        broker.registry["request"] = {"name": "child", "uid": "child-uid"}
        broker.client.jobs["child"] = {"metadata": {"uid": "child-uid"}}
        stop_exited_session(broker)
        self.assertEqual(broker.client.deleted, [self.policy.session])
        self.assertIn("child", broker.client.jobs)
        self.assertFalse(json.loads((broker.directory / "stopped").read_text())["cleanup_complete"])

    def test_exit_cleanup_refuses_replaced_parent(self):
        broker = self.broker()
        broker.session["parent_uid"] = "parent-uid"
        broker.client.jobs[self.policy.session] = {"metadata": {"uid": "other", "labels": {"falcon.dev/sandbox": self.policy.session}}}
        with self.assertRaisesRegex(ValueError, "ownership changed"):
            stop_exited_session(broker)
        self.assertEqual(broker.client.deleted, [])

    def test_exit_cleanup_missing_parent_is_idempotent(self):
        broker = self.broker()
        stop_exited_session(broker)
        stop_exited_session(broker)
        self.assertTrue(json.loads((broker.directory / "stopped").read_text())["cleanup_complete"])


class RelayTests(unittest.TestCase):
    def setUp(self):
        try:
            self.relay = Relay(("127.0.0.1", 0), "agent", "worker")
        except PermissionError as exc:
            self.skipTest(f"loopback sockets unavailable: {exc}")
        self.thread = threading.Thread(target=self.relay.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.relay.server_port}"
        self.addCleanup(self.relay.server_close)
        self.addCleanup(self.thread.join, 3)
        self.addCleanup(self.relay.shutdown)

    def test_agent_worker_exchange(self):
        message = {"protocol": 1, "id": str(uuid.uuid4()), "operation": "jobs", "payload": {}}
        http(self.base, "agent", "/request", message)
        self.assertEqual(http(self.base, "worker", "/work"), message)
        http(self.base, "worker", "/reply/" + message["id"], {"items": []})
        self.assertEqual(http(self.base, "agent", "/result/" + message["id"]), {"items": []})
        self.assertIsNone(http(self.base, "worker", "/work"))

    def test_wrong_capability_and_agent_worker_route_denied(self):
        for token, path in (("wrong", "/status"), ("agent", "/work"), ("worker", "/status")):
            with self.subTest(token=token), self.assertRaises(HTTPError) as caught:
                http(self.base, token, path)
            self.assertEqual(caught.exception.code, 403)

    def test_malformed_request_id_denied(self):
        with self.assertRaises(HTTPError):
            http(self.base, "agent", "/request", {"id": "../../outside"})


if __name__ == "__main__":
    unittest.main()
