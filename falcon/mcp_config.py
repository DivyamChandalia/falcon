"""Render and safely install local Falcon MCP client configuration."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

from packaging.version import InvalidVersion, Version

MCP_SERVER_NAME = "falcon"
MCP_SERVER_COMMAND = ("mcp", "serve")
SUPPORTED_AGENTS = ("codex", "claude", "opencode")
_AGENT_EXECUTABLES = {
    "codex": "codex",
    "claude": "claude",
    "opencode": "opencode",
}

# Claude Code automatically backgrounds a long-running main-conversation MCP
# call starting with this release.  Falcon's direct Claude workflow relies on
# that client behavior; it does not need the newer server-side task API.
CLAUDE_MIN_BACKGROUND_VERSION = Version("2.1.212")
CLAUDE_UPGRADE_COMMAND = "npm install -g @anthropic-ai/claude-code@latest"
_CLAUDE_VERSION_PATTERN = re.compile(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)")

# wait_for_events defaults to 7 days.  Clients get one extra hour so a
# normal client-side timeout does not cut off the server's default wait.
DEFAULT_WAIT_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
CLIENT_TOOL_TIMEOUT_SECONDS = DEFAULT_WAIT_TIMEOUT_SECONDS + 60 * 60
CLIENT_TOOL_TIMEOUT_MILLISECONDS = CLIENT_TOOL_TIMEOUT_SECONDS * 1000

# Eviction monitoring deliberately has a minimum grace period so an agent
# cannot accidentally flag a model while it is still loading or compiling.
# Agents may request a longer grace period on the first wait for a Job.
DEFAULT_EVICTION_STARTUP_SECONDS = 120.0
MIN_EVICTION_STARTUP_SECONDS = 120.0
MAX_EVICTION_STARTUP_SECONDS = 24 * 60 * 60.0
MCP_RISK_EMA_TIME_CONSTANT_SECONDS = 10 * 60.0
MCP_RISK_MIN_WINDOW_SECONDS = 30.0
MCP_RISK_MAX_WINDOW_SECONDS = 5 * 60.0

ConfigPath = Optional[Union[str, os.PathLike[str]]]


@dataclass(frozen=True)
class MCPSetupResult:
    """Result of merging Falcon into one coding-agent configuration."""

    agent: str
    path: Path
    status: str
    backup: Optional[Path] = None


class MCPConfigError(ValueError):
    """Raised when a client configuration cannot be safely updated."""


def claude_code_mcp_warning(
    *,
    environ: Optional[Mapping[str, str]] = None,
    which: Optional[Callable[..., Optional[str]]] = None,
    runner: Optional[Callable[..., subprocess.CompletedProcess[str]]] = None,
) -> Optional[str]:
    """Return a setup warning when Claude cannot background Falcon waits.

    ``None`` means the installed Claude Code version supports Falcon's direct
    background-wait workflow.  The injectable command helpers keep this
    diagnostic deterministic in tests and usable by alternate installers.
    """

    env = os.environ if environ is None else environ
    find_executable = shutil.which if which is None else which
    executable = find_executable("claude", path=env.get("PATH", os.defpath))
    if executable is None:
        return (
            "Claude Code was not found on PATH. Install or upgrade it before "
            "using Falcon's direct background wait workflow: "
            f"{CLAUDE_UPGRADE_COMMAND}"
        )

    run = subprocess.run if runner is None else runner
    try:
        result = run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return (
            f"Falcon could not verify the installed Claude Code version ({exc}). "
            f"Please upgrade Claude Code with: {CLAUDE_UPGRADE_COMMAND}"
        )

    output = "\n".join(
        value.strip()
        for value in (result.stdout or "", result.stderr or "")
        if value and value.strip()
    )
    match = _CLAUDE_VERSION_PATTERN.search(output)
    if match is None:
        return (
            "Falcon could not determine the installed Claude Code version. "
            f"Please upgrade Claude Code with: {CLAUDE_UPGRADE_COMMAND}"
        )
    try:
        version = Version(match.group(1))
    except InvalidVersion:
        return (
            f"Falcon could not parse the installed Claude Code version ({match.group(1)}). "
            f"Please upgrade Claude Code with: {CLAUDE_UPGRADE_COMMAND}"
        )

    if version >= CLAUDE_MIN_BACKGROUND_VERSION:
        return None
    return (
        f"Claude Code {version} is older than {CLAUDE_MIN_BACKGROUND_VERSION}. "
        "Falcon's direct background wait workflow requires Claude Code "
        f"{CLAUDE_MIN_BACKGROUND_VERSION} or newer. Upgrade with: "
        f"{CLAUDE_UPGRADE_COMMAND}."
    )


def _normalize_agent(agent: str) -> str:
    name = str(agent).strip().lower()
    if name not in SUPPORTED_AGENTS:
        supported = ", ".join(SUPPORTED_AGENTS)
        raise ValueError(
            f"unsupported MCP client {agent!r}; expected one of {supported}"
        )
    return name


def _text(value: Union[str, os.PathLike[str]], label: str) -> str:
    try:
        result = os.fspath(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be a path-like string") from exc
    if isinstance(result, bytes):
        result = os.fsdecode(result)
    if not result.strip():
        raise ValueError(f"{label} cannot be empty")
    return result


def _home_path(home: Optional[Union[str, os.PathLike[str]]] = None) -> Path:
    if home is not None:
        return Path(home).expanduser()
    configured = os.environ.get("HOME")
    return Path(configured).expanduser() if configured else Path.home()


def client_config_path(
    agent: str,
    *,
    home: Optional[Union[str, os.PathLike[str]]] = None,
    environ: Optional[dict[str, str]] = None,
) -> Path:
    """Return the user-level configuration path for a supported client."""

    name = _normalize_agent(agent)
    root = _home_path(home)
    # A supplied home is primarily a test/alternate-user override; do not
    # accidentally combine it with the calling user's XDG_CONFIG_HOME.
    env = (os.environ if home is None else {}) if environ is None else environ
    if name == "codex":
        return root / ".codex" / "config.toml"
    if name == "claude":
        return root / ".claude.json"
    config_root = Path(env.get("XDG_CONFIG_HOME", root / ".config")).expanduser()
    return config_root / "opencode" / "opencode.json"


def _client_is_detected(
    agent: str,
    path: Path,
    environ: Optional[dict[str, str]] = None,
) -> bool:
    """Return whether a client appears to be installed or already configured."""

    config_directory_exists = (
        (path.parent / ".claude").exists()
        if agent == "claude"
        else path.parent.exists()
    )
    return (
        path.exists()
        or config_directory_exists
        or shutil.which(
            _AGENT_EXECUTABLES[agent],
            path=(
                os.environ.get("PATH", "")
                if environ is None
                else environ.get("PATH", os.environ.get("PATH", ""))
            ),
        ) is not None
    )


def detected_agents(
    *,
    home: Optional[Union[str, os.PathLike[str]]] = None,
    environ: Optional[dict[str, str]] = None,
) -> list[str]:
    """Return supported coding agents found on this user's machine."""

    return [
        agent
        for agent in SUPPORTED_AGENTS
        if _client_is_detected(
            agent,
            client_config_path(agent, home=home, environ=environ),
            environ,
        )
    ]


def _selected_agents(
    agents: Optional[list[str]],
    *,
    all_agents: bool,
    home: Optional[Union[str, os.PathLike[str]]],
    environ: Optional[dict[str, str]],
) -> list[str]:
    if agents:
        result: list[str] = []
        for agent in agents:
            name = _normalize_agent(agent)
            if name not in result:
                result.append(name)
        return result
    if all_agents:
        return list(SUPPORTED_AGENTS)
    return detected_agents(home=home, environ=environ)


def _server_command(
    executable: str, config_path: ConfigPath
) -> tuple[str, list[str]]:
    command = _text(executable, "executable")
    args: list[str] = []
    if config_path is not None:
        args.extend(("--config", _text(config_path, "config_path")))
    args.extend(MCP_SERVER_COMMAND)
    return command, args


def _json_server_mapping(
    agent: str,
    config_path: ConfigPath,
    executable: str = "falcon",
) -> dict[str, Any]:
    config = configuration(agent, executable=executable, config_path=config_path)
    if agent == "claude":
        return config["mcpServers"][MCP_SERVER_NAME]
    return config["mcp"][MCP_SERVER_NAME]


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw) if raw.strip() else {}
    except (OSError, json.JSONDecodeError) as exc:
        raise MCPConfigError(
            f"cannot safely update {path}: expected a valid JSON file ({exc})"
        ) from exc
    if not isinstance(value, dict):
        raise MCPConfigError(
            f"cannot safely update {path}: the top-level JSON value must be an object"
        )
    return value


def _merge_json_config(
    agent: str,
    path: Path,
    config_path: ConfigPath,
    executable: str = "falcon",
) -> str:
    value = _load_json_mapping(path)
    if agent == "claude":
        servers = value.setdefault("mcpServers", {})
        if not isinstance(servers, dict):
            raise MCPConfigError(f"cannot safely update {path}: mcpServers is not an object")
        servers[MCP_SERVER_NAME] = _json_server_mapping(agent, config_path, executable)
    else:
        mcp = value.setdefault("mcp", {})
        if not isinstance(mcp, dict):
            raise MCPConfigError(f"cannot safely update {path}: mcp is not an object")
        # OpenCode v2 wraps local servers in mcp.servers. Preserve whichever
        # shape the user already has instead of converting their whole file.
        container = mcp.get("servers")
        if container is not None:
            if not isinstance(container, dict):
                raise MCPConfigError(
                    f"cannot safely update {path}: mcp.servers is not an object"
                )
            container[MCP_SERVER_NAME] = _json_server_mapping(
                agent, config_path, executable
            )
        else:
            mcp[MCP_SERVER_NAME] = _json_server_mapping(
                agent, config_path, executable
            )
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


_CODEX_SERVER_HEADER = "[mcp_servers.falcon]"


def _merge_codex_config(
    path: Path,
    config_path: ConfigPath,
    executable: str = "falcon",
) -> str:
    snippet = render_config(
        "codex", executable=executable, config_path=config_path
    ).rstrip() + "\n"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = existing.splitlines(keepends=True)
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == _CODEX_SERVER_HEADER),
        None,
    )
    if start is None:
        prefix = existing
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        if prefix and not prefix.endswith("\n\n"):
            prefix += "\n"
        return prefix + snippet

    end = start + 1
    while end < len(lines) and not lines[end].lstrip().startswith("["):
        end += 1
    return "".join(lines[:start]) + snippet + "".join(lines[end:])


def _backup_path(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return path.with_name(f"{path.name}.falcon-backup-{stamp}")


def _write_config(path: Path, content: str) -> Optional[Path]:
    """Atomically write a config, retaining a timestamped backup if changed."""

    previous = path.read_text(encoding="utf-8") if path.exists() else None
    if previous == content:
        return None

    backup: Optional[Path] = None
    if previous is not None:
        backup = _backup_path(path)
        shutil.copy2(path, backup)

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return backup


def setup_configs(
    agents: Optional[list[str]] = None,
    *,
    all_agents: bool = False,
    executable: str = "falcon",
    config_path: ConfigPath = None,
    home: Optional[Union[str, os.PathLike[str]]] = None,
    environ: Optional[dict[str, str]] = None,
    dry_run: bool = False,
) -> list[MCPSetupResult]:
    """Install Falcon's MCP entry into one or more local agent configs.

    Only the Falcon server entry is changed. Existing files are backed up
    before an update, and ``dry_run`` performs all parsing and merge checks
    without writing anything.
    """

    selected = _selected_agents(
        agents,
        all_agents=all_agents,
        home=home,
        environ=environ,
    )
    results: list[MCPSetupResult] = []
    for agent in selected:
        path = client_config_path(agent, home=home, environ=environ)
        if agent == "codex":
            content = _merge_codex_config(path, config_path, executable)
        else:
            content = _merge_json_config(agent, path, config_path, executable)
        previous = path.read_text(encoding="utf-8") if path.exists() else None
        if previous == content:
            status = "unchanged"
            backup = None
        elif dry_run:
            status = "would configure"
            backup = None
        else:
            backup = _write_config(path, content)
            status = "configured"
        results.append(MCPSetupResult(agent, path, status, backup))
    return results


def configuration(
    agent: str,
    executable: str = "falcon",
    config_path: ConfigPath = None,
) -> Dict[str, Any]:
    """Return a client configuration mapping for Falcon's local MCP server.

    ``config_path`` is passed as Falcon's global ``--config`` option.  It is
    useful when the client should use a Falcon config other than the normal
    ``~/.falconrc``; the path is not opened or changed here.
    """

    name = _normalize_agent(agent)
    command, args = _server_command(executable, config_path)

    if name == "codex":
        return {
            "mcp_servers": {
                MCP_SERVER_NAME: {
                    "command": command,
                    "args": args,
                    "tool_timeout_sec": CLIENT_TOOL_TIMEOUT_SECONDS,
                }
            }
        }
    if name == "claude":
        return {
            "mcpServers": {
                MCP_SERVER_NAME: {
                    "type": "stdio",
                    "command": command,
                    "args": args,
                    "timeout": CLIENT_TOOL_TIMEOUT_MILLISECONDS,
                }
            }
        }

    # Stable OpenCode configuration uses a direct server entry under ``mcp``.
    # OpenCode v2 has a different nesting shape; the generated snippet stays
    # on the stable documented schema and the docs call out that distinction.
    return {
        "mcp": {
            MCP_SERVER_NAME: {
                "type": "local",
                "command": [command, *args],
                "timeout": CLIENT_TOOL_TIMEOUT_MILLISECONDS,
            }
        }
    }


def client_config(
    agent: str,
    executable: str = "falcon",
    config_path: ConfigPath = None,
) -> Dict[str, Any]:
    """Compatibility alias for :func:`configuration`."""

    return configuration(agent, executable=executable, config_path=config_path)


def _toml_string(value: str) -> str:
    """Encode a value as a TOML basic string using JSON-compatible escapes."""

    return json.dumps(value, ensure_ascii=False)


def _render_codex(config: Dict[str, Any]) -> str:
    server = config["mcp_servers"][MCP_SERVER_NAME]
    args = ", ".join(_toml_string(str(value)) for value in server["args"])
    return "\n".join(
        (
            f"[mcp_servers.{MCP_SERVER_NAME}]",
            f"command = {_toml_string(server['command'])}",
            f"args = [{args}]",
            f"tool_timeout_sec = {server['tool_timeout_sec']}",
        )
    )


def render_config(
    agent: str,
    executable: str = "falcon",
    config_path: ConfigPath = None,
) -> str:
    """Render a client configuration snippet without writing any files."""

    name = _normalize_agent(agent)
    config = configuration(name, executable=executable, config_path=config_path)
    if name == "codex":
        return _render_codex(config)
    return json.dumps(config, indent=2, ensure_ascii=False)


def config_command(
    agent: str,
    executable: str = "falcon",
    config_path: ConfigPath = None,
) -> int:
    """Print one configuration snippet and return a CLI success status."""

    print(render_config(agent, executable=executable, config_path=config_path))
    return 0


__all__ = [
    "CLIENT_TOOL_TIMEOUT_MILLISECONDS",
    "CLIENT_TOOL_TIMEOUT_SECONDS",
    "CLAUDE_MIN_BACKGROUND_VERSION",
    "CLAUDE_UPGRADE_COMMAND",
    "DEFAULT_EVICTION_STARTUP_SECONDS",
    "DEFAULT_WAIT_TIMEOUT_SECONDS",
    "MAX_EVICTION_STARTUP_SECONDS",
    "MIN_EVICTION_STARTUP_SECONDS",
    "MCP_RISK_EMA_TIME_CONSTANT_SECONDS",
    "MCP_RISK_MAX_WINDOW_SECONDS",
    "MCP_RISK_MIN_WINDOW_SECONDS",
    "MCP_SERVER_COMMAND",
    "MCP_SERVER_NAME",
    "MCPConfigError",
    "MCPSetupResult",
    "SUPPORTED_AGENTS",
    "client_config",
    "client_config_path",
    "claude_code_mcp_warning",
    "config_command",
    "configuration",
    "detected_agents",
    "render_config",
    "setup_configs",
]
