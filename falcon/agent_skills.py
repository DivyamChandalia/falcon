"""Install Falcon's packaged agent skill without overwriting user content."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

SKILL_NAME = "falcon"
SKILL_BUNDLE_VERSION = 4
METADATA_FILE = ".falcon-skill.json"
SUPPORTED_AGENTS: Tuple[str, ...] = ("codex", "claude", "opencode")

_RESOURCE_PACKAGE = "falcon.skills"
_BUNDLE_FILES = ("SKILL.md",)
_AGENT_PATHS = {
    "codex": Path(".agents/skills") / SKILL_NAME,
    "claude": Path(".claude/skills") / SKILL_NAME,
    "opencode": Path(".config/opencode/skills") / SKILL_NAME,
}
_AGENT_EXECUTABLES = {
    "codex": "codex",
    "claude": "claude",
    "opencode": "opencode",
}
_AGENT_CONFIG_DIRS = {
    "codex": (Path(".codex"),),
    "claude": (Path(".claude"),),
    "opencode": (Path(".config/opencode"),),
}
_OWNER = "falcon"
_METADATA_FORMAT = 1

AgentSelection = Optional[Union[str, Iterable[str]]]


@dataclass(frozen=True)
class SkillOperation:
    """Outcome of one agent-specific install or uninstall operation."""

    agent: str
    path: Path
    status: str
    detail: str = ""

    @property
    def changed(self) -> bool:
        return self.status in {"installed", "updated", "removed"}


def _home_path(home: Optional[Union[str, os.PathLike[str]]]) -> Path:
    return Path.home() if home is None else Path(home).expanduser()


def _normalize_agents(agents: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(agents, str):
        values = agents.split(",")
    else:
        values = list(agents)
    normalized: List[str] = []
    invalid: List[str] = []
    for value in values:
        name = str(value).strip().lower()
        if not name:
            continue
        if name not in SUPPORTED_AGENTS:
            invalid.append(name)
        elif name not in normalized:
            normalized.append(name)
    if invalid:
        supported = ", ".join(SUPPORTED_AGENTS)
        raise ValueError(
            f"unsupported coding agent(s): {', '.join(invalid)}; expected one of {supported}"
        )
    return normalized


def agent_skill_path(
    agent: str, *, home: Optional[Union[str, os.PathLike[str]]] = None
) -> Path:
    """Return the current official per-user Falcon skill path for an agent."""

    names = _normalize_agents([agent])
    if not names:
        raise ValueError("coding agent name cannot be empty")
    return _home_path(home) / _AGENT_PATHS[names[0]]


def detect_agents(
    *,
    home: Optional[Union[str, os.PathLike[str]]] = None,
    which: Callable[[str], Optional[str]] = shutil.which,
) -> List[str]:
    """Detect supported agents by executable or their user configuration directory."""

    base = _home_path(home)
    detected: List[str] = []
    for agent in SUPPORTED_AGENTS:
        executable = _AGENT_EXECUTABLES[agent]
        configured = any((base / relative).is_dir() for relative in _AGENT_CONFIG_DIRS[agent])
        if which(executable) is not None or configured:
            detected.append(agent)
    return detected


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bundle_digest(hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _bundle() -> Tuple[Dict[str, bytes], Dict[str, str]]:
    root = resources.files(_RESOURCE_PACKAGE).joinpath(SKILL_NAME)
    files: Dict[str, bytes] = {}
    for name in _BUNDLE_FILES:
        source = root.joinpath(name)
        if not source.is_file():
            raise RuntimeError(f"packaged Falcon skill is missing {name}")
        files[name] = source.read_bytes()
    return files, {name: _sha256(content) for name, content in files.items()}


def _metadata(agent: str, hashes: Mapping[str, str]) -> Dict[str, object]:
    return {
        "format": _METADATA_FORMAT,
        "owner": _OWNER,
        "skill": SKILL_NAME,
        "agent": agent,
        "bundle_version": SKILL_BUNDLE_VERSION,
        "source_hash": _bundle_digest(hashes),
        "files": dict(sorted(hashes.items())),
    }


def _atomic_write(path: Path, content: bytes) -> None:
    """Replace one file atomically after fully writing and syncing a sibling temporary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        os.replace(str(temporary), str(path))
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_metadata(path: Path, metadata: Mapping[str, object]) -> None:
    encoded = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(path / METADATA_FILE, encoded)


def _read_metadata(path: Path, agent: str) -> Tuple[Optional[Dict[str, object]], str]:
    marker = path / METADATA_FILE
    if not marker.exists():
        return None, "existing skill is not managed by Falcon"
    if marker.is_symlink() or not marker.is_file():
        return None, "Falcon ownership metadata is not a regular file"
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "Falcon ownership metadata is unreadable"
    if not isinstance(value, dict):
        return None, "Falcon ownership metadata is invalid"
    expected = {
        "format": _METADATA_FORMAT,
        "owner": _OWNER,
        "skill": SKILL_NAME,
        "agent": agent,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        return None, "Falcon ownership metadata does not match this skill"
    installed = value.get("files")
    if not isinstance(installed, dict) or not installed:
        return None, "Falcon ownership metadata has no managed files"
    for name, digest in installed.items():
        candidate = Path(str(name))
        if (
            not isinstance(name, str)
            or candidate.name != name
            or name == METADATA_FILE
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return None, "Falcon ownership metadata contains an invalid file record"
    if (
        not isinstance(value.get("bundle_version"), int)
        or value["bundle_version"] < 0
        or value.get("source_hash") != _bundle_digest(installed)
    ):
        return None, "Falcon ownership metadata has an invalid bundle hash"
    return value, ""


def _managed_files_unchanged(path: Path, metadata: Mapping[str, object]) -> bool:
    installed = metadata["files"]
    assert isinstance(installed, dict)
    for name, expected_hash in installed.items():
        managed = path / str(name)
        if managed.is_symlink() or not managed.is_file():
            return False
        try:
            if _sha256(managed.read_bytes()) != expected_hash:
                return False
        except OSError:
            return False
    return True


def _conflict(agent: str, path: Path, detail: str) -> SkillOperation:
    return SkillOperation(agent, path, "conflict", detail)


def install_skill(
    agent: str, *, home: Optional[Union[str, os.PathLike[str]]] = None
) -> SkillOperation:
    """Install or safely update Falcon's skill for one coding agent."""

    target = agent_skill_path(agent, home=home)
    normalized_agent = _normalize_agents([agent])[0]
    files, hashes = _bundle()
    desired_metadata = _metadata(normalized_agent, hashes)

    if target.is_symlink() or (target.exists() and not target.is_dir()):
        return _conflict(normalized_agent, target, "skill path is not a managed directory")

    if not target.exists():
        target.mkdir(parents=True)
        for name, content in files.items():
            _atomic_write(target / name, content)
        # Write ownership last: an interrupted copy is never mistaken for managed content.
        _write_metadata(target, desired_metadata)
        return SkillOperation(normalized_agent, target, "installed")

    current_metadata, error = _read_metadata(target, normalized_agent)
    if current_metadata is None:
        return _conflict(normalized_agent, target, error)
    if not _managed_files_unchanged(target, current_metadata):
        return _conflict(
            normalized_agent,
            target,
            "a managed Falcon skill file was modified; leaving it unchanged",
        )

    current_hashes = current_metadata["files"]
    assert isinstance(current_hashes, dict)
    for new_name in set(files) - set(current_hashes):
        untracked = target / new_name
        if untracked.exists() or untracked.is_symlink():
            return _conflict(
                normalized_agent,
                target,
                f"packaged update would overwrite unmanaged {new_name}; leaving it unchanged",
            )
    if (
        current_hashes == desired_metadata["files"]
        and current_metadata.get("bundle_version") == SKILL_BUNDLE_VERSION
        and current_metadata.get("source_hash") == desired_metadata["source_hash"]
    ):
        return SkillOperation(normalized_agent, target, "unchanged")

    for name, content in files.items():
        _atomic_write(target / name, content)
    for obsolete_name in set(current_hashes) - set(files):
        (target / str(obsolete_name)).unlink()
    _write_metadata(target, desired_metadata)
    return SkillOperation(normalized_agent, target, "updated")


def install_skills(
    agents: AgentSelection = None,
    *,
    home: Optional[Union[str, os.PathLike[str]]] = None,
    which: Callable[[str], Optional[str]] = shutil.which,
) -> List[SkillOperation]:
    """Install selected agents, or auto-detect them when ``agents`` is omitted."""

    selected = (
        detect_agents(home=home, which=which)
        if agents is None
        else _normalize_agents(agents)
    )
    return [install_skill(agent, home=home) for agent in selected]


def uninstall_skill(
    agent: str, *, home: Optional[Union[str, os.PathLike[str]]] = None
) -> SkillOperation:
    """Remove only an unchanged Falcon-owned skill for one coding agent."""

    target = agent_skill_path(agent, home=home)
    normalized_agent = _normalize_agents([agent])[0]
    if not target.exists() and not target.is_symlink():
        return SkillOperation(normalized_agent, target, "not-installed")
    if target.is_symlink() or not target.is_dir():
        return _conflict(normalized_agent, target, "skill path is not a managed directory")

    current_metadata, error = _read_metadata(target, normalized_agent)
    if current_metadata is None:
        status = "unmanaged" if not (target / METADATA_FILE).exists() else "conflict"
        return SkillOperation(normalized_agent, target, status, error)
    if not _managed_files_unchanged(target, current_metadata):
        return _conflict(
            normalized_agent,
            target,
            "a managed Falcon skill file was modified; refusing to remove it",
        )

    installed = current_metadata["files"]
    assert isinstance(installed, dict)
    for name in installed:
        (target / str(name)).unlink()
    (target / METADATA_FILE).unlink()
    try:
        target.rmdir()
    except OSError:
        # Preserve the directory and any files the user added.
        pass
    return SkillOperation(normalized_agent, target, "removed")


def uninstall_skills(
    agents: AgentSelection = None,
    *,
    home: Optional[Union[str, os.PathLike[str]]] = None,
    which: Callable[[str], Optional[str]] = shutil.which,
) -> List[SkillOperation]:
    """Uninstall selected agents, or auto-detect them when ``agents`` is omitted."""

    selected = (
        detect_agents(home=home, which=which)
        if agents is None
        else _normalize_agents(agents)
    )
    return [uninstall_skill(agent, home=home) for agent in selected]


__all__ = [
    "METADATA_FILE",
    "SKILL_BUNDLE_VERSION",
    "SKILL_NAME",
    "SUPPORTED_AGENTS",
    "SkillOperation",
    "agent_skill_path",
    "detect_agents",
    "install_skill",
    "install_skills",
    "uninstall_skill",
    "uninstall_skills",
]
