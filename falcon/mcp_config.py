"""Render local Falcon MCP client configuration snippets.

The MCP server is deliberately configured by the user or coding agent.  These
helpers only build and print snippets; they never read or modify a client
configuration file.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Union

MCP_SERVER_NAME = "falcon"
MCP_SERVER_COMMAND = ("mcp", "serve")
SUPPORTED_AGENTS = ("codex", "claude", "opencode")

# wait_for_events defaults to twelve hours.  Clients get one extra hour so a
# normal client-side timeout does not cut off the server's default wait.
DEFAULT_WAIT_TIMEOUT_SECONDS = 12 * 60 * 60
CLIENT_TOOL_TIMEOUT_SECONDS = 13 * 60 * 60
CLIENT_TOOL_TIMEOUT_MILLISECONDS = CLIENT_TOOL_TIMEOUT_SECONDS * 1000

ConfigPath = Optional[Union[str, os.PathLike[str]]]


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


def _server_command(
    executable: str, config_path: ConfigPath
) -> tuple[str, list[str]]:
    command = _text(executable, "executable")
    args: list[str] = []
    if config_path is not None:
        args.extend(("--config", _text(config_path, "config_path")))
    args.extend(MCP_SERVER_COMMAND)
    return command, args


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
    "DEFAULT_WAIT_TIMEOUT_SECONDS",
    "MCP_SERVER_COMMAND",
    "MCP_SERVER_NAME",
    "SUPPORTED_AGENTS",
    "client_config",
    "config_command",
    "configuration",
    "render_config",
]
