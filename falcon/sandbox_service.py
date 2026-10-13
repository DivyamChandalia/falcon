"""Trusted launcher-machine broker and sandbox session lifecycle."""

from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import hashlib
import io
import json
import os
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.error import URLError

import yaml

from .kubernetes import KubernetesClient, KubernetesError
from .models import GPURequest, JobRequest, RuntimeEnvironment
from .planning import canonical_gpu, plan_cpu_resources, plan_resources
from .sandbox import (
    RELAY_PORT,
    RUNTIME,
    SESSION_LABEL,
    SandboxMount,
    SandboxPolicy,
    agent_argv,
    agent_state,
    checked_path,
    contains,
    default_session_name,
    discover_agent,
    free_port,
    installation_mounts,
    make_policy,
    relay_service,
    sandbox_manifest,
    state_root,
    tool_probe,
)
from .sandbox_runtime import atomic_json, http
from .sandbox_state import (
    CODEX_RUNTIME_HOME,
    CODEX_SANDBOX_HOME,
    SESSION_DIRECTORIES,
    CodexStateSync,
    seed_state,
)


def runtime_name(session: str) -> str:
    return f"{session[:42]}-{hashlib.sha256(session.encode()).hexdigest()[:8]}-runtime"


def runtime_resources(session: str, namespace: str, spec: dict, interpreter: Path) -> tuple[dict, dict]:
    """Project helpers through Kubernetes: host state may not be shared storage."""
    archive = io.BytesIO()
    tmux = shutil.which("tmux")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        if tmux:
            bundle.write(tmux, "tmux")
            result = subprocess.run(["ldd", tmux], capture_output=True, text=True, timeout=10, check=False)
            for line in result.stdout.splitlines():
                words = line.split()
                if len(words) >= 3 and words[1] == "=>" and words[2].startswith("/"):
                    path = Path(words[2])
                    if not path.name.startswith(("libc.", "libm.", "libpthread.", "libdl.", "librt.")):
                        bundle.write(path, path.name)
    data = {
        "runtime.py": Path(__file__).with_name("sandbox_runtime.py").read_text(),
        "falcon": '#!/bin/sh\nexec python3 /opt/falcon-sandbox/runtime.py client "$@"\n',
        "python3": '#!/bin/sh\nif [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python3" ]; then exec "$VIRTUAL_ENV/bin/python3" "$@"; fi\nif [ -x /usr/bin/python3 ]; then exec /usr/bin/python3 "$@"; fi\nif [ -x /usr/local/bin/python3 ]; then exec /usr/local/bin/python3 "$@"; fi\nexec ' + shlex.quote(str(interpreter)) + ' "$@"\n',
        "python": '#!/bin/sh\nexec python3 "$@"\n',
        "tmux": '#!/bin/sh\nif [ -x /usr/bin/tmux ]; then exec /usr/bin/tmux "$@"; fi\nif [ -x /usr/local/bin/tmux ]; then exec /usr/local/bin/tmux "$@"; fi\nexport LD_LIBRARY_PATH=/opt/falcon-cache/native\nexec /opt/falcon-cache/native/tmux "$@"\n',
        "native.b64": base64.b64encode(archive.getvalue()).decode(),
    }
    if len(json.dumps(data).encode()) > 900_000:
        raise ValueError("local tmux dependency bundle exceeds Kubernetes ConfigMap size; supply tmux in the configured image")
    metadata = {"namespace": namespace, "labels": {SESSION_LABEL: session}}
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {**metadata, "name": runtime_name(session)}, "immutable": True, "data": data}
    secret = {"apiVersion": "v1", "kind": "Secret", "metadata": {**metadata, "name": runtime_name(session) + "-spec"}, "immutable": True, "stringData": {"spec.json": json.dumps(spec), "tmux.conf": 'set -g remain-on-exit off\nset -g mouse on\nset -g default-terminal "tmux-256color"\nbind-key -n C-c kill-session\n'}}
    return cm, secret


def session_path(name: str) -> Path:
    JobRequest(name, "default")
    return state_root() / name


def load_session(name: str) -> dict[str, Any]:
    path = session_path(name) / "session.json"
    if not path.is_file():
        raise ValueError(f"sandbox {name} is not registered on this launch machine")
    return json.loads(path.read_text())


def stop_local_broker(directory: Path) -> None:
    state = directory / "broker.json"
    if state.exists():
        pid = json.loads(state.read_text()).get("pid")
        try:
            if str(directory).encode() in Path(f"/proc/{pid}/cmdline").read_bytes():
                os.kill(pid, signal.SIGTERM)
        except (OSError, TypeError):
            pass


def prepare_session_directory(directory: Path, namespace: str) -> None:
    """Release a stale registration only after positively confirming Job absence."""
    if not directory.exists():
        return
    registration = directory / "session.json"
    if not registration.exists() and any((directory / file).exists() for file in ("broker.json", "registry.json")):
        raise ValueError("incomplete sandbox registration has broker/child records; inspect it before recovery")
    session = json.loads(registration.read_text()) if registration.exists() else {"policy": {"namespace": namespace}}
    if session["policy"]["namespace"] != namespace:
        raise ValueError(f"sandbox {directory.name} belongs to another namespace; stop it there first")
    client = KubernetesClient(namespace)
    try:
        client.get_json("job", directory.name)
    except KubernetesError as exc:
        if not exc.not_found:
            raise  # Authentication/timeouts must never be treated as Job absence.
    else:
        raise ValueError(f"sandbox {directory.name} already exists; use sandbox resume/status/stop")
    registry = directory / "registry.json"
    stopped = directory / "stopped"
    cleanup_complete = stopped.exists() and json.loads(stopped.read_text()).get("cleanup_complete")
    if registry.exists() and json.loads(registry.read_text()) and not cleanup_complete:
        raise ValueError(f"sandbox {directory.name} has child Job records; run falcon sandbox stop {directory.name} before retrying")
    atomic_json(stopped, {"reason": "archived for relaunch"})
    stop_local_broker(directory)
    archive = directory.parent / "archive"
    archive.mkdir(mode=0o700, exist_ok=True)
    directory.rename(archive / f"{directory.name}-{uuid.uuid4().hex}")


def resource_plan(payload: dict, config: dict, namespace: str):
    from .cli import _cli_memory, _planning_nodes, resolve_preset

    gpu_name = payload.get("gpu")
    count = payload.get("gpus", 1)
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("GPU count must be a positive integer")
    percent = payload.get("shm_percent")
    if percent is None:
        percent = config.get("resources", {}).get("shared_memory_percent", 15)
    if not gpu_name:
        if count != 1:
            raise ValueError("--gpus requires --gpu")
        return None, plan_cpu_resources(str(payload.get("cpu") or "1"), _cli_memory(str(payload.get("memory") or "1Gi")), shared_memory=payload.get("shm_size"), shared_memory_percent=percent)
    if not isinstance(gpu_name, str):
        raise ValueError("GPU preset must be a string")
    resolved = resolve_preset(gpu_name, config)
    if resolved:
        gpu_name, implicit_count = resolved
        if implicit_count != 1:
            count = implicit_count
    preset = config.get("presets", {}).get(gpu_name, {})
    from .config import gpu_preset_max_count

    if preset and count > gpu_preset_max_count(preset):
        raise ValueError("GPU count exceeds preset maximum")
    gpu = GPURequest(canonical_gpu(str(preset.get("gpu_type", gpu_name))), count)
    nodes = _planning_nodes(config, KubernetesClient(namespace))
    plan = plan_resources(nodes, gpu_name, gpu.model, count,
                          cpu_override=payload.get("cpu"),
                          memory_override=_cli_memory(str(payload["memory"])) if payload.get("memory") else None,
                          maximize=payload.get("maximize", False),
                          shared_memory=payload.get("shm_size"),
                          shared_memory_percent=percent)
    return gpu, plan


def child_spec(policy: SandboxPolicy, payload: dict, config: dict, rid: str, token: str) -> dict:
    allowed = {"gpu", "gpus", "cpu", "memory", "name", "image", "environment", "env", "command", "working_dir", "shm_size", "shm_percent", "maximize", "pin_node"}
    if set(payload) - allowed:
        raise ValueError("unknown/restricted launch fields")
    for key in ("gpu", "cpu", "memory", "name", "image", "environment", "working_dir", "shm_size"):
        if payload.get(key) is not None and not isinstance(payload[key], str):
            raise ValueError(f"{key} must be a string")
    for key in ("maximize", "pin_node"):
        if key in payload and not isinstance(payload[key], bool):
            raise ValueError(f"{key} must be a boolean")
    command = payload.get("command")
    if not isinstance(command, list) or not command or len(command) > 256:
        raise ValueError("child command must be a bounded nonempty argv list")
    gpu, plan = resource_plan(payload, config, policy.namespace)
    environment = None
    selected = payload.get("environment")
    if selected and selected not in {"none", "off", "-", "auto", policy.python_environment}:
        path = Path(selected).expanduser().resolve(strict=True)
        if not any(contains(Path(p), path) for p in [policy.workspace, *(m.source for m in policy.data)]):
            raise ValueError("environment is outside the session mount allowlist")
        environment = RuntimeEnvironment.from_path(path)
    from .cli import _parse_env

    raw_env = payload.get("env", [])
    if not isinstance(raw_env, list) or not all(isinstance(v, str) for v in raw_env):
        raise ValueError("environment must be a list of KEY=VALUE strings")
    env = _parse_env(raw_env)
    if selected in {"none", "off", "-"}:
        env["PATH"] = f"{RUNTIME}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
        env["VIRTUAL_ENV"] = ""
        env["CONDA_PREFIX"] = ""
    elif environment:
        env["PATH"] = f"{RUNTIME}/bin:{environment.path}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
        env["CONDA_PREFIX" if environment.kind.value == "conda" else "VIRTUAL_ENV"] = str(environment.path)
        env.setdefault("VIRTUAL_ENV", "")
    suffix = payload.get("name") or rid[:12]
    if not isinstance(suffix, str):
        raise ValueError("Job name must be a string")
    JobRequest(suffix, policy.namespace)
    name = f"{policy.session[:24]}-{suffix[:38]}"[:63].rstrip("-")
    if name == policy.session:
        raise ValueError("child Job cannot replace its parent")
    request = JobRequest(name, policy.namespace, tuple(command), gpu=gpu, compute=plan.compute,
                         working_dir=payload.get("working_dir") or policy.workspace,
                         image=payload.get("image"), env=env, environment=environment, pin_node=payload.get("pin_node", False))
    manifest = sandbox_manifest(policy, request, plan, config, runtime_configmap=runtime_name(policy.session))
    manifest["metadata"]["annotations"] = {"falcon.dev/sandbox-request": rid}
    manifest["spec"]["template"]["spec"]["containers"][0].setdefault("env", []).extend([
        {"name": "FALCON_SANDBOX_URL", "value": f"http://{policy.session}.{policy.namespace}.svc:{RELAY_PORT}"},
        {"name": "FALCON_SANDBOX_TOKEN", "value": token},
    ])
    return manifest


class Broker:
    """Validate every relay message; never execute caller-supplied host argv."""

    def __init__(self, directory: Path, client: KubernetesClient | None = None):
        self.directory = directory
        self.session = json.loads((directory / "session.json").read_text())
        self.policy = SandboxPolicy.from_dict(self.session["policy"])
        self.client = client or KubernetesClient(self.policy.namespace)
        self.registry_path = directory / "registry.json"
        self.registry = json.loads(self.registry_path.read_text()) if self.registry_path.exists() else {}
        copied = self.session.get("codex_state")
        baseline = directory / "codex-sync.json"
        self.state_sync = CodexStateSync(Path(copied["source"]), Path(copied["stage"]), baseline, shared_sessions=copied.get("shared_sessions", False)) if copied and baseline.exists() else None

    def save(self) -> None:
        atomic_json(self.registry_path, self.registry)

    def owned(self, name: str) -> dict:
        if not isinstance(name, str):
            raise ValueError("Job name must be a string")
        uid = next((v.get("uid") for v in self.registry.values() if v.get("name") == name and v.get("uid")), None)
        if not uid:
            raise ValueError("Job does not belong to this sandbox")
        job = self.client.get_json("job", name)
        metadata = job.get("metadata", {})
        if metadata.get("uid") != uid or metadata.get("labels", {}).get(SESSION_LABEL) != self.policy.session:
            raise ValueError("Job ownership/UID changed")
        return job

    def launch(self, rid: str, payload: dict) -> dict:
        existing = self.registry.get(rid)
        if existing and existing.get("uid"):
            return {"name": existing["name"], "namespace": self.policy.namespace, "uid": existing["uid"]}
        manifest = child_spec(self.policy, payload, self.session["config"], rid, self.session["token"])
        name = manifest["metadata"]["name"]
        if not existing:
            self.registry[rid] = {"name": name, "digest": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}
            self.save()  # Persist intent before the API operation.
        try:
            job = self.client.get_json("job", name)
        except KubernetesError as exc:
            if not exc.not_found:
                raise
            result = self.client.create_job(manifest)
            if not result.uid:
                raise ValueError("Kubernetes did not return the created Job UID")
            uid = result.uid
        else:
            meta = job.get("metadata", {})
            if meta.get("labels", {}).get(SESSION_LABEL) != self.policy.session or meta.get("annotations", {}).get("falcon.dev/sandbox-request") != rid:
                raise ValueError("requested Job name is already in use")
            uid = meta["uid"]
        self.registry[rid]["uid"] = uid
        self.save()
        return {"name": name, "namespace": self.policy.namespace, "uid": uid}

    def dispatch(self, message: dict) -> dict:
        if not isinstance(message, dict) or set(message) != {"protocol", "id", "operation", "payload"} or message["protocol"] != 1:
            raise ValueError("unsupported broker protocol/message")
        rid = str(uuid.UUID(message["id"]))
        payload = message["payload"]
        operation = message["operation"]
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        digest = hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest()
        cache_path = self.directory / "responses" / f"{rid}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if cached["digest"] != digest:
                raise ValueError("request ID reused with different content")
            return cached["reply"]
        if rid in self.registry:
            expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            if operation != "launch" or self.registry[rid]["digest"] != expected:
                raise ValueError("request ID reused with different content")
        if operation == "launch":
            reply = self.launch(rid, payload)
        elif operation == "jobs":
            if payload:
                raise ValueError("unexpected jobs fields")
            jobs = self.client.list_json("jobs", labels={SESSION_LABEL: self.policy.session})
            names = {v["name"]: v.get("uid") for v in self.registry.values()}
            reply = {"items": [j for j in jobs.get("items", []) if names.get(j["metadata"]["name"]) == j["metadata"].get("uid")]}
        elif operation in {"get", "kill", "logs", "metrics"}:
            allowed = {"job"} | ({"tail"} if operation == "logs" else {"interval"} if operation == "metrics" else set())
            if set(payload) - allowed:
                raise ValueError("unexpected/restricted operation fields")
            name = payload.get("job")
            job = self.owned(name)
            if operation == "get":
                reply = job
            elif operation == "kill":
                self.client.delete_jobs([name])
                reply = {"deleted": name}
            elif operation == "logs":
                tail = payload.get("tail", 100)
                if isinstance(tail, bool) or not isinstance(tail, int) or not 0 <= tail <= 2000:
                    raise ValueError("sandbox log tail must be between 0 and 2000")
                logged = self.client.logs(name, tail=tail, follow=False)
                if logged.returncode:
                    raise KubernetesError(logged.stderr.strip() or "child logs are not available yet", returncode=logged.returncode)
                reply = {"text": logged.stdout[-100_000:]}
            else:
                from .cli import _metrics_command

                interval = payload.get("interval", 10)
                if not isinstance(interval, (int, float)) or isinstance(interval, bool) or not 1 <= interval <= 30:
                    raise ValueError("sandbox metrics interval must be 1-30 seconds")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    _metrics_command(argparse.Namespace(job=name, namespace=self.policy.namespace, interval=interval), self.session["config"])
                reply = json.loads(output.getvalue())
        elif operation == "config-sync":
            reply = self.sync_config(payload)
        else:
            raise ValueError("operation is not allowed in a sandbox")
        atomic_json(cache_path, {"digest": digest, "reply": reply})
        responses = sorted(cache_path.parent.glob("*.json"), key=lambda path: path.stat().st_mtime)
        for expired in responses[:-256]:
            expired.unlink()
        return reply

    def sync_config(self, payload: dict) -> dict:
        # Only legacy Claude's home-root file needs mirroring. No caller path
        # is accepted, and no arbitrary file read/write endpoint exists.
        if self.session["agent"] != "claude" or set(payload) != {"base", "value"}:
            raise ValueError("invalid configuration sync request")
        base, value = payload["base"], payload["value"]
        if not isinstance(base, dict) or not isinstance(value, dict):
            raise ValueError("configuration sync requires JSON objects")
        target = Path(self.session["legacy_config"])
        current = json.loads(target.read_text()) if target.exists() else {}
        merged = dict(current)
        conflicts = []
        missing = object()
        for key in base.keys() | value.keys():
            old, new, host = base.get(key, missing), value.get(key, missing), current.get(key, missing)
            if new == old:
                continue
            if host != old and host != new:
                conflicts.append(key)
            elif new is missing:
                merged.pop(key, None)
            else:
                merged[key] = new
        atomic_json(target, merged)
        return {"value": merged, "conflicts": conflicts}


def stop_exited_session(broker: Broker) -> None:
    """Delete only this parent after an agent exit; child Jobs remain owned."""
    if broker.state_sync:
        try:
            broker.state_sync.sync()
        except (OSError, ValueError) as exc:
            print(f"Final Codex state sync failed: {exc}", flush=True)
    try:
        parent = broker.client.get_json("job", broker.policy.session)
        metadata = parent.get("metadata", {})
        if metadata.get("uid") != broker.session.get("parent_uid") or metadata.get("labels", {}).get(SESSION_LABEL) != broker.policy.session:
            raise ValueError("sandbox parent ownership changed")
        broker.client.delete_jobs([broker.policy.session])
    except KubernetesError as exc:
        if not exc.not_found:
            raise
    atomic_json(broker.directory / "stopped", {
        "reason": "agent exited", "cleanup_complete": not bool(broker.registry),
    })


def broker_main(directory: str) -> None:
    broker = Broker(Path(directory))
    atomic_json(broker.directory / "broker.json", {"pid": os.getpid(), "state": "running"})
    running = True

    def stop(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    tunnel = None
    last_parent_check = 0.0
    last_state_sync = 0.0
    try:
        while running and not (broker.directory / "stopped").exists():
            try:
                if time.monotonic() - last_parent_check > 10:
                    parent = broker.client.get_json("job", broker.policy.session)
                    if parent.get("metadata", {}).get("uid") != broker.session.get("parent_uid"):
                        break
                    if any(c.get("type") == "Failed" and c.get("status") == "True" for c in parent.get("status", {}).get("conditions", [])):
                        break
                    last_parent_check = time.monotonic()
                if broker.state_sync and time.monotonic() - last_state_sync > 10:
                    try:
                        conflicts = broker.state_sync.sync()
                        if conflicts:
                            print(f"Codex state sync preserved conflicting/unsafe files: {', '.join(conflicts)}", flush=True)
                    except (OSError, ValueError) as exc:
                        # State-copy failures must not disable child Job requests.
                        print(f"Codex state sync failed: {exc}", flush=True)
                    last_state_sync = time.monotonic()
                if tunnel is None or tunnel.poll() is not None:
                    if tunnel:
                        tunnel.wait()
                    port = free_port()
                    tunnel = subprocess.Popen([
                        broker.client.executable, "port-forward", "--namespace", broker.policy.namespace,
                        "--address", "127.0.0.1", f"service/{broker.policy.session}", f"{port}:{RELAY_PORT}",
                    ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    base = f"http://127.0.0.1:{port}"
                    deadline = time.monotonic() + 20
                    while time.monotonic() < deadline:
                        if tunnel.poll() is not None:
                            raise OSError("Kubernetes port-forward exited before listening")
                        try:
                            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                                break
                        except OSError:
                            time.sleep(0.2)
                    else:
                        raise OSError("Kubernetes port-forward did not become ready")
                status = http(base, broker.session["worker_token"], "/worker-status")
                if status.get("state") == "exited":
                    stop_exited_session(broker)
                    break
                message = http(base, broker.session["worker_token"], "/work")
                if message:
                    try:
                        reply = broker.dispatch(message)
                    except (ValueError, TypeError, KeyError, OSError, KubernetesError) as exc:
                        reply = {"error": str(exc)}
                    # Malformed IDs must never become URL/path components.
                    rid = str(uuid.UUID(message["id"]))
                    http(base, broker.session["worker_token"], "/reply/" + rid, reply)
                time.sleep(0.2)
            except KubernetesError as exc:
                if exc.not_found:
                    break  # Explicit parent deletion must not resurrect it.
                time.sleep(2)
            except (URLError, TimeoutError, OSError) as exc:
                atomic_json(broker.directory / "broker.json", {"pid": os.getpid(), "state": "running", "last_error": str(exc)})
                if tunnel:
                    tunnel.terminate()
                    tunnel.wait(timeout=10)
                    tunnel = None
                time.sleep(1)
            except (ValueError, TypeError, KeyError):
                time.sleep(1)
    finally:
        if broker.state_sync:
            try:
                broker.state_sync.sync()
            except (OSError, ValueError) as exc:
                print(f"Final Codex state sync failed: {exc}", flush=True)
        if tunnel and tunnel.poll() is None:
            tunnel.terminate()
            tunnel.wait(timeout=10)
        atomic_json(broker.directory / "broker.json", {"pid": os.getpid(), "state": "stopped"})


def start_broker(directory: Path) -> None:
    import fcntl

    descriptor = os.open(directory / "launch.lock", os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _start_broker_unlocked(directory)


def _start_broker_unlocked(directory: Path) -> None:
    state = directory / "broker.json"
    if state.exists():
        prior = json.loads(state.read_text())
        try:
            cmdline = Path(f"/proc/{prior['pid']}/cmdline").read_bytes()
            if str(directory).encode() in cmdline and prior.get("state") in {"running", "starting"}:
                return
        except OSError:
            pass
    # Do not let editable import hooks or cwd shadow the frozen broker package.
    policy = SandboxPolicy.from_dict(json.loads((directory / "session.json").read_text())["policy"])
    writable = broker_writable_paths(policy)
    if any(contains(w, Path(sys.executable).resolve()) or contains(w, Path(sys.prefix).resolve()) for w in writable):
        raise ValueError("launch Falcon from a separate Python installation outside the agent's writable environment/mounts")
    paths = [p for p in sys.path if p and not any(contains(w, Path(p).resolve()) for w in writable)]
    paths.insert(0, str(directory / "bundle"))
    code = "import sys,importlib.machinery;sys.path=" + repr(paths) + ";sys.meta_path=[importlib.machinery.BuiltinImporter,importlib.machinery.FrozenImporter,importlib.machinery.PathFinder];from falcon.sandbox_service import broker_main;broker_main(sys.argv[1])"
    logfile = directory / "broker.log"
    descriptor = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "ab") as log:
        process = subprocess.Popen([sys.executable, "-I", "-S", "-c", code, str(directory)], cwd=directory, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    atomic_json(state, {"pid": process.pid, "state": "starting"})


def broker_writable_paths(policy: SandboxPolicy) -> list[Path]:
    return [Path(policy.workspace), *(Path(m.source) for m in (*policy.state, *policy.runtime_tools) if not m.read_only)]


def environment_mounts(environment: RuntimeEnvironment) -> tuple[SandboxMount, ...]:
    """Only the selected environment is writable, not its base installation."""
    selected = checked_path(str(environment.path))
    paths = list(environment.mount_paths)
    if environment.base_path and environment.base_path.name == "bin" and environment.base_path.parent not in {Path("/usr"), Path("/usr/local")}:
        paths.append(checked_path(str(environment.base_path.parent)))
    return tuple(SandboxMount(str(checked_path(str(path))), str(path), Path(path) != selected) for path in dict.fromkeys(paths))


def attach_session(name: str) -> int:
    session = load_session(name)
    directory = session_path(name)
    start_broker(directory)
    client = KubernetesClient(session["policy"]["namespace"])
    pod = client.wait_for_pod(name)
    for _ in range(100):
        probe = client._run(["exec", "--namespace", client.namespace, pod["metadata"]["name"], "--container", "main", "--", "tmux", "-L", "falcon", "has-session", "-t", "agent"], check=False)
        if probe.returncode == 0:
            break
        time.sleep(0.2)
    else:
        raise ValueError("sandbox tools/tmux are not ready; inspect falcon logs for dependency errors")
    return client._run(["exec", "--namespace", client.namespace, pod["metadata"]["name"], "--container", "main", "--stdin", "--tty", "--", "tmux", "-L", "falcon", "attach-session", "-t", "agent"], capture=False, check=False, timeout=0).returncode


def launch_session(args: argparse.Namespace, config: dict) -> int:
    command = list(args.command)
    if command[:1] == ["--"]:
        command.pop(0)
    agent, executable = discover_agent(command, args.agent_bin)
    if getattr(args, "auto_continue", False) and agent != "codex":
        raise ValueError("--auto-continue currently supports Codex only")
    name = args.name or default_session_name(agent, Path.cwd())
    namespace = args.namespace or config["cluster"]["namespace"]
    policy = make_policy(name, namespace, str(Path.cwd()), args.read_only)
    tools = installation_mounts(executable)
    state, private = agent_state(agent)
    codex_state = None
    # Mask tool trees beneath writable user-state roots, not just the selected
    # release: otherwise changing a sibling release/current symlink affects host tools.
    for mount in state:
        for child in ("packages", "local"):
            path = Path(mount.source) / child
            if path.is_dir():
                tools += (SandboxMount(str(path), str(path)),)
    environment = None
    if args.environment not in {"none", "off", "-"}:
        if args.environment and args.environment != "auto":
            environment = RuntimeEnvironment.from_path(args.environment)
        else:
            environment = RuntimeEnvironment.from_current()
    python_environment = None
    if environment:
        python_environment = str(environment.path)
        for mount in [*policy.data, *tools, *state]:
            if contains(environment.path, Path(mount.source)) or contains(Path(mount.source), environment.path):
                raise ValueError("writable Python environment overlaps read-only data or agent tools/state")
        if contains(environment.path, Path(sys.prefix).resolve()) or contains(environment.path, Path(sys.executable).resolve()):
            raise ValueError("the active environment also runs Falcon's trusted broker; launch Falcon from a separate installation, or choose --environment /path/to/another/env")
    if agent == "codex":
        skills = Path.home() / ".agents/skills"
        if skills.is_dir():
            path = checked_path(str(skills))
            tools += (SandboxMount(str(path), str(skills)),)
    # Exposing a state root that contains the workspace, data, or another state
    # tree would accidentally create writable aliases of the restricted mounts.
    for mount in state:
        for restricted in [policy.workspace, *(m.source for m in policy.data)]:
            if contains(Path(mount.source), Path(restricted)) or contains(Path(restricted), Path(mount.source)):
                raise ValueError("agent state overlaps workspace/read-only data")
    if agent == "codex":
        source = Path(state[0].source)
        stage = source.parent / ".falcon-sandbox-state" / f"{name}-{uuid.uuid4().hex}" / "codex-home"
        for mount in [SandboxMount(policy.workspace, policy.workspace), *policy.data, *tools]:
            if contains(Path(mount.source), stage) or contains(stage, Path(mount.source)):
                raise ValueError("filtered Codex state overlaps an exposed mount")
        codex_state = {"source": str(source), "stage": str(stage), "shared_sessions": True}
        session_mounts = []
        for component in SESSION_DIRECTORIES:
            path = source / component
            if path.resolve() != path:
                raise ValueError("Codex session directories must not be symlinks")
            if path.exists():
                checked_path(str(path))
                if not path.is_dir():
                    raise ValueError(f"Codex session path must be a directory: {path}")
            # Mount beside the original tool paths, not over the filtered
            # state's private directories (unsupported on some shared filesystems).
            session_mounts.append(SandboxMount(str(path), str(path), False))
        state = (SandboxMount(str(stage), CODEX_SANDBOX_HOME, False), *session_mounts)
        private = ()
    runtime_tools = ()
    interpreter = Path(sys.executable).resolve()
    prefix = Path(sys.prefix).resolve()
    if prefix not in {Path("/usr"), Path("/usr/local")}:
        checked_path(str(prefix))
        runtime_tools = (SandboxMount(str(prefix), str(prefix)),)
    if environment:
        mounts_by_path = {m.destination: m for m in runtime_tools}
        mounts_by_path.update({m.destination: m for m in environment_mounts(environment)})
        runtime_tools = tuple(mounts_by_path.values())
    policy = replace(policy, tools=tools, state=state, private=private, runtime_tools=runtime_tools, python_environment=python_environment, python_source=str(environment.path) if environment else None)
    directory = session_path(name)
    token, worker_token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    argv = agent_argv(agent, str(executable), command[1:], args.permissions)
    flags = agent_argv(agent, str(executable), [], args.permissions)[1:]
    if codex_state:
        sqlite_override = ["-c", f'sqlite_home="{CODEX_RUNTIME_HOME}"']
        argv += sqlite_override
        flags += sqlite_override
    spec = {
        "agent": agent, "argv": argv, "resume_flags": flags,
        "session": name, "namespace": namespace, "workspace": policy.workspace,
        "record": str(Path(policy.workspace) / ".falcon/sandbox" / name / "agent.json"),
        "permissions": args.permissions, "token": token, "worker_token": worker_token,
        "auto_continue": bool(getattr(args, "auto_continue", False)),
    }
    payload = {"gpu": args.gpu, "gpus": args.gpus, "cpu": args.cpu if args.gpu else args.cpu or "1", "memory": args.memory if args.gpu else args.memory or "1Gi", "shm_size": args.shm_size, "shm_percent": args.shm_percent, "maximize": args.maximize}
    gpu, plan = resource_plan(payload, config, namespace)
    from .cli import _parse_env

    request = JobRequest(name, namespace, ("python3", f"{RUNTIME}/runtime.py", "supervise", "/opt/falcon-session/spec.json"), gpu=gpu, compute=plan.compute, image=args.image, working_dir=policy.workspace, env=_parse_env(args.env), pin_node=args.pin_node)
    manifest = sandbox_manifest(policy, request, plan, config, parent=True, runtime_configmap=runtime_name(name))
    pod = manifest["spec"]["template"]["spec"]
    pod["volumes"].append({"name": "session-spec", "secret": {"secretName": runtime_name(name) + "-spec"}})
    main = pod["containers"][0]
    main["volumeMounts"].append({"name": "session-spec", "mountPath": "/opt/falcon-session", "readOnly": True})
    main["readinessProbe"] = {"tcpSocket": {"port": "relay"}, "periodSeconds": 2}
    python_paths = f"{python_environment}/bin:" if environment else ""
    runtime_env = {"PATH": f"{RUNTIME}/bin:{python_paths}" + ":".join(str(Path(m.source) / "bin") for m in tools if Path(m.source).is_dir()) + ":/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"}
    if agent == "codex":
        runtime_env["CODEX_HOME"] = CODEX_SANDBOX_HOME
        runtime_env["CODEX_SQLITE_HOME"] = CODEX_RUNTIME_HOME
    elif agent == "claude":
        runtime_env["CLAUDE_CONFIG_DIR"] = state[0].source
    elif agent == "opencode":
        for key, mount in zip(("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"), state):
            runtime_env[key] = str(Path(mount.source).parent)
    main["env"] = [e for e in main["env"] if e["name"] not in runtime_env] + [{"name": k, "value": v} for k, v in runtime_env.items()]
    service = relay_service(policy)
    if args.follow:
        raise ValueError("sandbox uses live tmux attachment, not --follow; use --detach to launch without attaching")
    if args.dry_run:
        redacted = {**spec, "token": "REDACTED", "worker_token": "REDACTED"}
        cm, secret = runtime_resources(name, namespace, redacted, interpreter)
        # Capabilities and copied auth/config never appear in dry-run output.
        if args.output == "json":
            from .output import dumps

            print(dumps("SandboxDryRun", {"policy": policy.to_dict(), "manifests": [manifest, service, cm, secret]}))
        else:
            print(yaml.safe_dump_all([manifest, service, cm, secret], sort_keys=False), end="")
        return 0
    if args.output == "json" and not args.detach:
        raise ValueError("--output json requires --detach")
    tool_probe(executable)
    if agent == "claude":
        legacy = Path.home() / ".claude.json"
        if legacy.exists():
            spec["legacy_initial"] = json.loads(legacy.read_text())
    cm, secret = runtime_resources(name, namespace, spec, interpreter)
    prepare_session_directory(directory, namespace)
    directory.mkdir(parents=True, mode=0o700)
    stored = {"policy": policy.to_dict(), "config": copy.deepcopy(config), "agent": agent, "token": token, "worker_token": worker_token, "manifest": manifest, "service": service, "legacy_config": str(Path.home() / ".claude.json")}
    if codex_state:
        stored["codex_state"] = codex_state
    atomic_json(directory / "session.json", stored)
    client = KubernetesClient(namespace)
    try:
        for mount in state:
            Path(mount.source).mkdir(parents=True, exist_ok=True, mode=0o700)
        if codex_state:
            Path(codex_state["source"]).mkdir(parents=True, exist_ok=True, mode=0o700)
            seed_state(Path(codex_state["source"]), Path(codex_state["stage"]), directory / "codex-sync.json", shared_sessions=True)
        shutil.copytree(Path(__file__).parent, directory / "bundle/falcon", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        submitted = client.create_job(manifest)
        if not submitted.uid:
            raise ValueError("Kubernetes did not return the sandbox Job UID")
        stored["parent_uid"] = submitted.uid
        atomic_json(directory / "session.json", stored)
        service["metadata"]["ownerReferences"] = [{"apiVersion": "batch/v1", "kind": "Job", "name": name, "uid": submitted.uid}]
        for resource in (cm, secret, service):
            resource["metadata"]["ownerReferences"] = service["metadata"]["ownerReferences"]
            client._run(["create", "-f", "-"], input_text=yaml.safe_dump(resource))
        start_broker(directory)
    except BaseException:
        # Only this newly-created session is eligible for cleanup.
        atomic_json(directory / "stopped", {"reason": "launch failed"})
        if "submitted" in locals() and submitted.uid:
            try:
                parent = client.get_json("job", name)
                if parent.get("metadata", {}).get("uid") == submitted.uid:
                    client.delete_jobs([name])
            except Exception as cleanup_error:
                print(f"falcon: sandbox cleanup failed: {cleanup_error}; inspect falcon sandbox status {name}", file=sys.stderr)
        raise
    from .commands import remember_job

    remember_job(name)
    print(json.dumps({"name": name, "namespace": namespace}) if args.output == "json" else f"Sandbox {name} · attach: falcon attach {name} · stop: falcon sandbox stop {name}")
    return 0 if args.detach else attach_session(name)


def print_session_status(status: dict[str, Any], workspace: str) -> None:
    job = status["job"]
    conditions = {c["type"] for c in job.get("conditions", []) if c.get("status") == "True"}
    phase = job.get("state") or (
        "failed" if "Failed" in conditions else "completed" if "Complete" in conditions
        else "running" if job.get("active") else "pending"
    )
    local = status["local_status"] or {}
    if local.get("cleanup_complete"):
        phase = "stopped"
    broker = status["broker"] or {}
    agent = status["agent"] or {}
    print(f"Sandbox:   {status['name']}")
    print(f"Namespace: {status['namespace']}")
    print(f"Workspace: {workspace}")
    print(f"Job:       {phase}")
    print(f"Agent:     {agent.get('state', 'not started')}" + (f" (exit {agent['exit_code']})" if "exit_code" in agent else ""))
    if agent.get("auto_continue"):
        continuation = agent["auto_continue"]
        print(f"Continue:  {continuation.get('status', 'watching')}")
        if continuation.get("reset_at"):
            print("Reset:     " + time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(continuation["reset_at"])))
    print(f"Broker:    {broker.get('state', 'not started')}")
    if local.get("reason"):
        print(f"Note:      {local['reason']}")
    children = status["children"]
    print(f"Child jobs: {len(children)}")
    for child in children.values():
        print(f"  {child.get('name', 'unknown')}")
    if phase not in {"stopped", "missing", "failed", "completed"}:
        print(f"\nReconnect: falcon sandbox resume {status['name']}")
        print("Detach:    Ctrl+B, then D")


def manage_session(action: str, name: str, output: str = "human") -> int:
    session = load_session(name)
    directory = session_path(name)
    client = KubernetesClient(session["policy"]["namespace"])
    if action == "status":
        broker_path = directory / "broker.json"
        record = Path(session["policy"]["workspace"]) / ".falcon/sandbox" / name / "agent.json"
        registry = directory / "registry.json"
        try:
            job = client.get_json("job", name).get("status", {})
        except KubernetesError as exc:
            if not exc.not_found:
                raise
            job = {"state": "missing"}
        stopped = directory / "stopped"
        status = {"name": name, "namespace": client.namespace, "broker": json.loads(broker_path.read_text()) if broker_path.exists() else None, "agent": json.loads(record.read_text()) if record.exists() else None, "children": json.loads(registry.read_text()) if registry.exists() else {}, "job": job, "local_status": json.loads(stopped.read_text()) if stopped.exists() else None}
        if output == "json":
            print(json.dumps(status, indent=2))
        else:
            print_session_status(status, session["policy"]["workspace"])
        return 0
    if action == "resume":
        try:
            parent = client.get_json("job", name)
        except KubernetesError as exc:
            if not exc.not_found:
                raise
            raise ValueError("sandbox Job is missing; rerun falcon sandbox with the original command, or stop the session to clean up its children") from exc
        if parent.get("status", {}).get("conditions") and any(c.get("type") == "Failed" and c.get("status") == "True" for c in parent["status"]["conditions"]):
            raise ValueError("sandbox exhausted retries; stop it and launch a new session with the agent's explicit resume command")
        start_broker(directory)
        return attach_session(name)
    if action != "stop":
        raise ValueError("sandbox action must be status, resume, or stop")
    atomic_json(directory / "stopped", {"reason": "user stopped session"})
    stop_local_broker(directory)
    broker = Broker(directory, client)
    if broker.state_sync:
        broker.state_sync.sync()
    for record in broker.registry.values():
        if record.get("uid"):
            try:
                broker.owned(record["name"])
                client.delete_jobs([record["name"]])
            except KubernetesError as exc:
                if not exc.not_found:
                    raise
    try:
        parent = client.get_json("job", name)
        if parent.get("metadata", {}).get("uid") != session.get("parent_uid") or parent.get("metadata", {}).get("labels", {}).get(SESSION_LABEL) != name:
            raise ValueError("sandbox parent ownership changed")
        client.delete_jobs([name])
    except KubernetesError as exc:
        if not exc.not_found:
            raise
    atomic_json(directory / "stopped", {"reason": "user stopped session", "cleanup_complete": True})
    print(f"Stopped sandbox {name}; working files and agent history are preserved.")
    return 0
