"""Sandbox policies and restricted Job construction.

Policy is captured by the launcher and kept outside the agent's mounts. The
relay is deliberately untrusted: it can request operations, never manifests.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import shutil
import socket
import stat
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import build_job_manifest
from .models import JobRequest, ResourcePlan

SESSION_LABEL = "falcon.dev/sandbox"
RUNTIME = "/opt/falcon-sandbox"
RELAY_PORT = 8765
MAX_REQUEST_BYTES = 256 * 1024
SUPPORTED_AGENTS = ("codex", "claude", "opencode")
RESERVED_ENV = {"HOME", "KUBECONFIG", "PYTHONPATH", "PYTHONHOME", "LD_PRELOAD"}


def default_session_name(harness: str, workspace: Path) -> str:
    """Use readable, deterministic names that fit Kubernetes label limits."""
    def slug(value: str, fallback: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or fallback

    name = f"falcon-{slug(harness, 'agent')}-{slug(workspace.name, 'workspace')}"
    if len(name) > 63:
        name = f"{name[:54].rstrip('-')}-{hashlib.sha256(name.encode()).hexdigest()[:8]}"
    return name


@dataclass(frozen=True)
class SandboxMount:
    source: str
    destination: str
    read_only: bool = True

    def volume(self) -> dict[str, Any]:
        return {
            "host_path": self.source,
            "mount_path": self.destination,
            "read_only": self.read_only,
            "type": "Directory" if not Path(self.source).exists() or Path(self.source).is_dir() else "File",
        }


@dataclass(frozen=True)
class SandboxPolicy:
    session: str
    namespace: str
    workspace: str
    data: tuple[SandboxMount, ...] = ()
    tools: tuple[SandboxMount, ...] = ()
    state: tuple[SandboxMount, ...] = ()
    private: tuple[str, ...] = ()
    runtime_tools: tuple[SandboxMount, ...] = ()
    python_environment: str | None = None
    python_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SandboxPolicy:
        return cls(
            session=value["session"], namespace=value["namespace"],
            workspace=value["workspace"],
            **{key: tuple(SandboxMount(**mount) for mount in value.get(key, ()))
               for key in ("data", "tools", "state", "runtime_tools")},
            private=tuple(value.get("private", ())),
            python_environment=value.get("python_environment"),
            python_source=value.get("python_source"),
        )


def contains(parent: Path, child: Path) -> bool:
    return child == parent or parent in child.parents


def state_root() -> Path:
    return Path(os.environ.get("XDG_STATE_HOME", "~/.local/state")).expanduser().resolve() / "falcon" / "sandboxes"


def protected_paths() -> tuple[Path, ...]:
    home = Path.home().resolve()
    credentials = os.environ.get("KUBECONFIG", str(home / ".kube"))
    return (
        state_root(), home / ".kube", home / ".ssh", home / ".aws",
        Path(os.environ.get("CODEX_HOME", str(home / ".codex"))).expanduser().resolve().parent / ".falcon-sandbox-state",
        *(Path(p).expanduser().resolve() for p in credentials.split(os.pathsep) if p),
        Path("/var/run"), Path("/run"), Path("/proc"), Path("/sys"), Path("/etc"),
    )


def checked_path(value: str, *, directory: bool = True) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if directory and not path.is_dir():
        raise ValueError(f"sandbox mount must be a directory: {path}")
    if path == Path("/") or any(
        contains(path, secret) or contains(secret, path) for secret in protected_paths()
    ):
        raise ValueError(f"sandbox path exposes credentials or broker state: {path}")
    return path


def make_policy(session: str, namespace: str, workspace: str, readonly: Sequence[str]) -> SandboxPolicy:
    # Reuse the domain's DNS validation, including the namespace.
    JobRequest(session, namespace)
    root = checked_path(workspace)
    paths = [checked_path(p) for p in readonly]
    for i, path in enumerate(paths):
        for other in [root, *paths[:i]]:
            if contains(path, other) or contains(other, path):
                raise ValueError(f"overlapping sandbox mounts: {path} and {other}")
    return SandboxPolicy(session, namespace, str(root), tuple(
        SandboxMount(str(path), str(path)) for path in paths
    ))


def discover_agent(command: Sequence[str], binary: str | None = None) -> tuple[str, Path]:
    if not command:
        raise ValueError("sandbox requires a command after --")
    agent = Path(command[0]).name
    executable = binary or shutil.which(command[0])
    if not executable:
        raise ValueError(f"{command[0]} is not installed/on PATH; use --agent-bin PATH")
    path = checked_path(executable, directory=False)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"agent executable is not executable: {path}")
    return agent, path


def installation_mounts(executable: Path) -> tuple[SandboxMount, ...]:
    """Mount native installation roots, or an npm package plus Node runtime."""
    with executable.open("rb") as stream:
        prefix = stream.read(256)
    roots = [executable.parent.parent if executable.parent.name == "bin" else executable.parent]
    if prefix.startswith(b"#!") and (b"node" in prefix.split(b"\n")[0]):
        # npm launchers live inside their package; include its package.json,
        # vendored platform binaries, and symlink target rather than ~/.local/bin.
        root = executable.parent
        while root != root.parent and not (root / "package.json").is_file():
            root = root.parent
        if root == root.parent:
            raise ValueError("cannot identify npm agent package; use a native agent executable")
        roots = [root]
        node = shutil.which("node")
        if not node:
            raise ValueError("npm agent requires a local Node runtime")
        node_path = Path(node).resolve(strict=True)
        roots.append(node_path.parent.parent if node_path.parent.name == "bin" and node_path.parent.parent != Path("/usr") else node_path)
    elif prefix.startswith(b"#!"):
        roots = [executable]
    result = []
    for root in roots:
        # Never expose all of /usr or a user's home just to obtain one binary.
        if root in {Path("/usr"), Path.home().resolve(), Path("/")}:
            root = executable
        root = checked_path(str(root), directory=root.is_dir())
        result.append(SandboxMount(str(root), str(root)))
    return tuple(result)


def agent_state(agent: str) -> tuple[tuple[SandboxMount, ...], tuple[str, ...]]:
    home = Path.home()
    if agent == "codex":
        roots = [Path(os.environ.get("CODEX_HOME", str(home / ".codex")))]
        private_names = ("app-server-control", "app-server-daemon", "tmp", ".tmp", "shell_snapshots")
    elif agent == "claude":
        roots = [Path(os.environ.get("CLAUDE_CONFIG_DIR", str(home / ".claude")))]
        private_names = ("sockets", "daemon", "tmp")
    elif agent == "opencode":
        roots = [
            Path(os.environ.get("XDG_CONFIG_HOME", str(home / ".config"))) / "opencode",
            Path(os.environ.get("XDG_DATA_HOME", str(home / ".local/share"))) / "opencode",
            Path(os.environ.get("XDG_STATE_HOME", str(home / ".local/state"))) / "opencode",
        ]
        private_names = ("sockets", "daemon", "tmp")
    else:
        return (), ()
    mounts, private = [], []
    for source in roots:
        # Creation is performed only after dry-run has returned.
        path = source.expanduser().resolve()
        if any(contains(path, p) or contains(p, path) for p in protected_paths()):
            raise ValueError(f"unsafe agent state location: {path}")
        mounts.append(SandboxMount(str(path), str(path), False))
        private.extend(str(path / name) for name in private_names)
        # Do not expose an unexpected host Unix socket through shared state.
        if path.exists() and agent != "codex":
            for base, directories, files in os.walk(path):
                directories[:] = [d for d in directories if d not in private_names]
                for name in files:
                    item = Path(base) / name
                    if stat.S_ISSOCK(item.lstat().st_mode):
                        if item.parent == path:
                            raise ValueError(f"host IPC socket directly in agent state: {item}; use an isolated agent config directory")
                        private.append(str(item.parent))
    return tuple(mounts), tuple(sorted(set(private)))


def agent_argv(agent: str, executable: str, arguments: Sequence[str], permissions: str, session_id: str | None = None) -> list[str]:
    args = list(arguments)
    if session_id:
        if agent == "codex":
            args = ["resume", session_id]
        elif agent == "claude":
            args = ["--resume", session_id]
        elif agent == "opencode":
            args = ["--session", session_id]
        else:
            raise ValueError("arbitrary commands are not automatically replayed")
    if agent == "codex":
        if "--no-daemon" not in args:
            args.append("--no-daemon")
        if permissions == "full" and "--dangerously-bypass-approvals-and-sandbox" not in args:
            args.append("--dangerously-bypass-approvals-and-sandbox")
    if agent == "claude" and permissions == "full" and "--dangerously-skip-permissions" not in args:
        args.append("--dangerously-skip-permissions")
    return [executable, *args]


def sandbox_manifest(policy: SandboxPolicy, request: JobRequest, plan: ResourcePlan, config: Mapping[str, Any], *, parent: bool = False, runtime_configmap: str | None = None) -> dict[str, Any]:
    """Reconstruct mounts/security rather than trusting request/config overrides."""
    root = Path(policy.workspace)
    working_dir = Path(request.working_dir or policy.workspace)
    if not working_dir.is_absolute() or ".." in working_dir.parts or not contains(root, working_dir):
        raise ValueError("child working directory must be inside the original workspace")
    if request.namespace != policy.namespace:
        raise ValueError("sandbox cannot change namespace")
    if not request.command:
        raise ValueError("sandbox child Jobs require a command")
    if any(key in RESERVED_ENV or key.startswith("FALCON_SANDBOX_") for key in request.env):
        raise ValueError("sandbox request overrides protected environment variables")
    safe = copy.deepcopy(dict(config))
    runtime = safe.setdefault("runtime", {})
    allowed = ("image", "image_pull_secrets", "scheduler", "priority_class", "run_as_user", "run_as_group", "supplemental_groups", "fs_group")
    safe["runtime"] = runtime = {k: v for k, v in runtime.items() if k in allowed}
    runtime.update(mount_home=False, mount_working_dir=False, python_environment="none")
    mounts = [SandboxMount(policy.workspace, policy.workspace, False), *policy.data, *policy.runtime_tools]
    if parent:
        mounts += [*policy.state, *policy.tools]
    elif request.environment:
        for path in request.environment.mount_paths:
            if not contains(root, path):
                if not any(contains(Path(m.source), path) for m in policy.data):
                    raise ValueError("child environment must be inside the workspace or an approved read-only mount")
    runtime["volumes"] = [mount.volume() for mount in mounts]
    # Environment mounts are already included in the fixed allowlist. The
    # ordinary renderer would otherwise add them read-write.
    request = replace(request, environment=None, labels={SESSION_LABEL: policy.session}, annotations={})
    manifest = build_job_manifest(request, plan, safe)
    pod = manifest["spec"]["template"]["spec"]
    pod.update(automountServiceAccountToken=False, enableServiceLinks=False)
    pod["securityContext"]["seccompProfile"] = {"type": "RuntimeDefault"}
    container = pod["containers"][0]
    if runtime_configmap:
        pod.setdefault("volumes", []).append({"name": "sandbox-runtime", "configMap": {"name": runtime_configmap, "defaultMode": 0o555, "items": [{"key": key, "path": path} for key, path in (("runtime.py", "runtime.py"), ("falcon", "bin/falcon"), ("python3", "bin/python3"), ("python", "bin/python"), ("tmux", "bin/tmux"), ("native.b64", "native.b64"))]}})
        container.setdefault("volumeMounts", []).append({"name": "sandbox-runtime", "mountPath": RUNTIME, "readOnly": True})
    container["securityContext"]["readOnlyRootFilesystem"] = True
    manifest["spec"]["backoffLimit"] = 3 if parent else 0
    if parent:
        manifest["spec"].pop("activeDeadlineSeconds", None)
        manifest["spec"].pop("ttlSecondsAfterFinished", None)
    ephemeral = [("scratch", "/tmp"), ("home", str(Path.home())), ("cache", "/opt/falcon-cache")]
    if parent:
        ephemeral += [(f"private-{i}", path) for i, path in enumerate(policy.private)]
        if any(m.destination == "/opt/falcon-agent-state" for m in policy.state):
            ephemeral.append(("codex-runtime", "/opt/falcon-codex-runtime"))
    for name, destination in ephemeral:
        pod.setdefault("volumes", []).append({"name": name, "emptyDir": {}})
        container.setdefault("volumeMounts", []).insert(0, {"name": name, "mountPath": destination})
    # A concrete UID is necessary to write emptyDir volumes as non-root.
    uid = int(runtime.get("run_as_user", os.getuid()))
    gid = int(runtime.get("run_as_group", os.getgid()))
    if uid == 0 or gid == 0:
        raise ValueError("sandbox requires a non-root user and group")
    pod["securityContext"].update(runAsUser=uid, runAsGroup=gid, fsGroup=gid)
    env = {"HOME": str(Path.home()), "PATH": f"{RUNTIME}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin", "XDG_CACHE_HOME": "/opt/falcon-cache"}
    if policy.python_environment and "PATH" not in request.env:
        env.update(
            PATH=f"{RUNTIME}/bin:{policy.python_environment}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
            VIRTUAL_ENV=policy.python_environment,
            CONDA_PREFIX=policy.python_environment if (Path(policy.python_environment) / "conda-meta").is_dir() else "",
        )
    elif "VIRTUAL_ENV" in request.env:
        env["VIRTUAL_ENV"] = request.env["VIRTUAL_ENV"]
    if parent:
        container["ports"] = [{"name": "relay", "containerPort": RELAY_PORT}]
    if not parent and "PATH" in request.env:
        env["PATH"] = request.env["PATH"]
    container["env"] = [e for e in container.get("env", []) if e["name"] not in env] + [{"name": k, "value": v} for k, v in env.items()]
    # Ancestors must precede nested mounts. Otherwise a later state/home mount
    # hides the private socket masks and read-only installation submounts.
    container["volumeMounts"].sort(key=lambda mount: len(Path(mount["mountPath"]).parts))
    return manifest


def relay_service(policy: SandboxPolicy) -> dict[str, Any]:
    return {"apiVersion": "v1", "kind": "Service", "metadata": {"name": policy.session, "namespace": policy.namespace, "labels": {SESSION_LABEL: policy.session}}, "spec": {"selector": {"job-name": policy.session}, "ports": [{"port": RELAY_PORT, "targetPort": "relay"}]}}


def free_port() -> int:
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return int(connection.getsockname()[1])


def tool_probe(executable: Path) -> None:
    result = subprocess.run([str(executable), "--version"], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise ValueError(f"agent executable failed its local version check: {executable}")
