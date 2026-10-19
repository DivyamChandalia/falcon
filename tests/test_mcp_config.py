import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from falcon.mcp_config import (
    CLIENT_TOOL_TIMEOUT_MILLISECONDS,
    CLIENT_TOOL_TIMEOUT_SECONDS,
    client_config,
    config_command,
    configuration,
    render_config,
)


class MCPConfigTests(unittest.TestCase):
    def test_codex_configuration_and_toml(self) -> None:
        expected = {
            "mcp_servers": {
                "falcon": {
                    "command": "falcon",
                    "args": ["mcp", "serve"],
                    "tool_timeout_sec": CLIENT_TOOL_TIMEOUT_SECONDS,
                }
            }
        }
        self.assertEqual(configuration("codex"), expected)
        self.assertEqual(client_config("CODEX"), expected)
        self.assertEqual(
            render_config("codex"),
            "[mcp_servers.falcon]\n"
            'command = "falcon"\n'
            'args = ["mcp", "serve"]\n'
            "tool_timeout_sec = 46800",
        )

    def test_claude_uses_stdio_and_millisecond_timeout(self) -> None:
        config = configuration("claude")
        server = config["mcpServers"]["falcon"]

        self.assertEqual(server["type"], "stdio")
        self.assertEqual(server["command"], "falcon")
        self.assertEqual(server["args"], ["mcp", "serve"])
        self.assertEqual(server["timeout"], CLIENT_TOOL_TIMEOUT_MILLISECONDS)
        self.assertEqual(json.loads(render_config("claude")), config)

    def test_opencode_uses_stable_local_server_shape(self) -> None:
        config = configuration("opencode")
        server = config["mcp"]["falcon"]

        self.assertEqual(server["type"], "local")
        self.assertEqual(server["command"], ["falcon", "mcp", "serve"])
        self.assertEqual(server["timeout"], CLIENT_TOOL_TIMEOUT_MILLISECONDS)
        self.assertEqual(json.loads(render_config("opencode")), config)

    def test_custom_executable_and_falcon_config_path_are_command_arguments(self) -> None:
        config_path = Path("/tmp/falcon-config.yaml")
        expected_args = ["--config", str(config_path), "mcp", "serve"]

        self.assertEqual(
            configuration("codex", executable="/opt/falcon", config_path=config_path)
            ["mcp_servers"]["falcon"],
            {
                "command": "/opt/falcon",
                "args": expected_args,
                "tool_timeout_sec": CLIENT_TOOL_TIMEOUT_SECONDS,
            },
        )
        self.assertEqual(
            configuration("opencode", executable="/opt/falcon", config_path=config_path)
            ["mcp"]["falcon"]["command"],
            ["/opt/falcon", *expected_args],
        )

    def test_config_command_prints_only_the_snippet_and_does_not_write_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "falconrc"
            output = io.StringIO()
            with redirect_stdout(output):
                status = config_command("claude", config_path=target)

            self.assertEqual(status, 0)
            self.assertFalse(target.exists())
            parsed = json.loads(output.getvalue())
            self.assertEqual(
                parsed["mcpServers"]["falcon"]["args"],
                ["--config", str(target), "mcp", "serve"],
            )

    def test_unknown_agent_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported MCP client"):
            configuration("cursor")
        with self.assertRaisesRegex(ValueError, "unsupported MCP client"):
            render_config("cursor")


if __name__ == "__main__":
    unittest.main()
