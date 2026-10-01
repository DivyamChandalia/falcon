"""Filtered Codex state copies: no shared sockets, databases, or arbitrary writes."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import stat
import uuid
from pathlib import Path, PurePosixPath

from .sandbox_runtime import atomic_json

CODEX_SANDBOX_HOME = "/opt/falcon-agent-state"
CODEX_RUNTIME_HOME = "/opt/falcon-codex-runtime"
MAX_STATE_FILE_BYTES = 128 * 1024 * 1024
MAX_STATE_FILES = 20000
LOCAL_DIRECTORIES = ("app-server-control", "app-server-daemon", "tmp", ".tmp", "shell_snapshots", "logs")
SETTINGS = {"config.toml", "auth.json", "AGENTS.md", "requirements.toml"}
JOURNALS = {"history.jsonl", "session_index.jsonl"}
SESSION_DIRECTORIES = ("sessions", "archived_sessions")


def allowed_file(relative: str, *, seed: bool = False) -> bool:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        return False
    if len(path.parts) == 1:
        return relative in SETTINGS | JOURNALS or relative.endswith(".config.toml")
    if path.parts[0] in {"sessions", "archived_sessions"}:
        return path.name.startswith("rollout-") and path.suffix == ".jsonl"
    return (
        seed and path.parts[0] in {"skills", "prompts", "rules", "agents", "plugins"}
        and not any(part in LOCAL_DIRECTORIES for part in path.parts)
        and ".sqlite" not in path.name
        and path.suffix not in {".db", ".sock", ".wal", ".shm"}
    )


@contextlib.contextmanager
def parent_fd(root: Path, relative: str, *, create: bool = False):
    """Never follow a path component from the agent-controlled copy."""
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("invalid state path")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[:-1]:
            if create:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor, path.name
    finally:
        os.close(descriptor)


def read_file(root: Path, relative: str) -> bytes | None:
    try:
        with parent_fd(root, relative) as (parent, name):
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("state sync rejects special files and hard links")
                value = stream.read(MAX_STATE_FILE_BYTES + 1)
                if len(value) > MAX_STATE_FILE_BYTES:
                    raise ValueError("state file exceeds sync limit")
                return value
    except FileNotFoundError:
        return None


def write_file(root: Path, relative: str, value: bytes, *, executable: bool = False) -> None:
    with parent_fd(root, relative, create=True) as (parent, name):
        temporary = f".falcon-sync-{uuid.uuid4().hex}"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o700 if executable else 0o600, dir_fd=parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(value)
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass


def files(root: Path, *, seed: bool = False, shared_sessions: bool = False) -> dict[str, tuple[int, int, int]]:
    """Bounded inventory; fwalk doesn't descend through directory symlinks."""
    result = {}
    if not root.exists():
        return result
    visited = 0
    for base, directories, names, descriptor in os.fwalk(root, follow_symlinks=False):
        relative = Path(base).relative_to(root)
        if relative == Path("."):
            directories[:] = [d for d in directories if d in (set() if shared_sessions else set(SESSION_DIRECTORIES)) | ({"skills", "prompts", "rules", "agents", "plugins"} if seed else set())]
        directories[:] = [d for d in directories if d not in LOCAL_DIRECTORIES]
        visited += len(names) + len(directories)
        if visited > MAX_STATE_FILES:
            raise ValueError("state inventory exceeds sync limit")
        for name in names:
            key = (relative / name).as_posix()
            if not allowed_file(key, seed=seed):
                continue
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                result[key] = (info.st_mtime_ns, info.st_size, info.st_ino)
    return result


def digest(value: bytes | None) -> str | None:
    return hashlib.sha256(value).hexdigest() if value is not None else None


def seed_state(source: Path, stage: Path, baseline: Path, *, shared_sessions: bool = False) -> None:
    stage.mkdir(parents=True, mode=0o700, exist_ok=True)
    tracked = {}
    for relative in files(source, seed=True, shared_sessions=shared_sessions):
        value = read_file(source, relative)
        if value is None:
            continue
        executable = bool((source / relative).stat(follow_symlinks=False).st_mode & 0o111)
        write_file(stage, relative, value, executable=executable)
        if allowed_file(relative):
            tracked[relative] = {"hash": digest(value), "size": len(value)}
    # All links point to container-local scratch, never to host runtime state.
    for name in LOCAL_DIRECTORIES:
        (stage / name).symlink_to(f"{CODEX_RUNTIME_HOME}/{name}", target_is_directory=True)
    if shared_sessions:
        for name in SESSION_DIRECTORIES:
            (stage / name).symlink_to(source / name, target_is_directory=True)
    atomic_json(baseline, tracked)


class CodexStateSync:
    def __init__(self, source: Path, stage: Path, baseline: Path, *, shared_sessions: bool = False):
        self.source, self.stage, self.baseline = source, stage, baseline
        self.shared_sessions = shared_sessions
        self.tracked = json.loads(baseline.read_text())
        self.fingerprints: dict[str, tuple[object, object]] = {}

    def sync(self) -> list[str]:
        with (self.baseline.parent / "codex-sync.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.tracked = json.loads(self.baseline.read_text())
            return self._sync()

    def _sync(self) -> list[str]:
        host_files = files(self.source, shared_sessions=self.shared_sessions)
        stage_files = files(self.stage, shared_sessions=self.shared_sessions)
        conflicts = []
        changed = False
        for relative in sorted(host_files.keys() | stage_files.keys()):
            fingerprint = (host_files.get(relative), stage_files.get(relative))
            if self.fingerprints.get(relative) == fingerprint:
                continue
            try:
                host = read_file(self.source, relative)
                local = read_file(self.stage, relative)
                base = self.tracked.get(relative, {})
                old = base.get("hash")
                host_hash, local_hash = digest(host), digest(local)
                if relative in JOURNALS and local and not local.endswith(b"\n"):
                    continue
                if host_hash == local_hash:
                    pass
                elif local_hash == old and host is not None:
                    if read_file(self.stage, relative) != local:
                        continue
                    write_file(self.stage, relative, host)
                    local = host
                elif host_hash == old and local is not None and relative not in JOURNALS:
                    if read_file(self.source, relative) != host:
                        continue
                    write_file(self.source, relative, local)
                    host = local
                elif relative in JOURNALS and local is not None and (host is not None or old is None):
                    if host is None:
                        write_file(self.source, relative, local)
                        host = local
                        self.tracked[relative] = {"hash": digest(local), "size": len(local)}
                        changed = True
                        continue
                    # Concurrent append-only histories are additive, unlike config.
                    size = base.get("size", 0)
                    if digest(host[:size]) != old or digest(local[:size]) != old:
                        conflicts.append(relative)
                        continue
                    additions = local[size:]
                    if not additions.endswith(b"\n"):
                        continue  # Never synchronize an incomplete record.
                    # Idempotent even if the broker died after appending but
                    # before persisting its baseline, or the agent wrote again.
                    seen = set(host.splitlines(keepends=True))
                    additions = b"".join(line for line in additions.splitlines(keepends=True) if line not in seen)
                    with parent_fd(self.source, relative) as (parent, name):
                        fd = os.open(name, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
                        with os.fdopen(fd, "ab", buffering=0) as stream:
                            info = os.fstat(stream.fileno())
                            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                                raise ValueError("unsafe history file")
                            stream.write(additions)
                    host = read_file(self.source, relative)
                    if read_file(self.stage, relative) != local:
                        # Retain the old baseline; don't erase a newer agent write.
                        continue
                    write_file(self.stage, relative, host)
                    local = host
                else:
                    # Keep both versions. No deletion propagation or silent loss.
                    conflicts.append(relative)
                    continue
                self.tracked[relative] = {"hash": digest(local), "size": len(local) if local is not None else 0}
                self.fingerprints[relative] = fingerprint
                changed = True
            except (OSError, ValueError):
                conflicts.append(relative)
        if changed:
            atomic_json(self.baseline, self.tracked)
        return conflicts
