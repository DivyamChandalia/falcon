import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from falcon.mcp_config import (
    CLAUDE_MIN_BACKGROUND_VERSION,
    CLAUDE_UPGRADE_COMMAND,
    CLIENT_TOOL_TIMEOUT_MILLISECONDS,
    CLIENT_TOOL_TIMEOUT_SECONDS,
    claude_code_mcp_warning,
    client_config,
    client_config_path,
    config_command,
    configuration,
    detected_agents,
    render_config,
    setup_configs,
)


class MCPConfigTests(unittest.TestCase):
    @staticmethod
    def _claude_runner(output: str, returncode: int = 0):
        def runner(*args, **kwargs):
            return subprocess.CompletedProcess(
                args[0], returncode, stdout=output, stderr=""
            )

        return runner

    def test_claude_version_warning_requires_backgrounding_support(self) -> None:
        warning = claude_code_mcp_warning(
            environ={"PATH": "/bin"},
            which=lambda name, path=None: "/bin/claude",
            runner=self._claude_runner("2.1.211 (Claude Code)"),
        )

        self.assertIsNotNone(warning)
        self.assertIn(str(CLAUDE_MIN_BACKGROUND_VERSION), warning)
        self.assertIn(CLAUDE_UPGRADE_COMMAND, warning)

    def test_current_claude_version_has_no_mcp_warning(self) -> None:
        warning = claude_code_mcp_warning(
            environ={"PATH": "/bin"},
            which=lambda name, path=None: "/bin/claude",
            runner=self._claude_runner("2.1.310 (Claude Code)"),
        )

        self.assertIsNone(warning)

    def test_claude_version_warning_covers_missing_or_unreadable_client(self) -> None:
        missing = claude_code_mcp_warning(
            environ={"PATH": ""},
            which=lambda name, path=None: None,
        )
        unreadable = claude_code_mcp_warning(
            environ={"PATH": "/bin"},
            which=lambda name, path=None: "/bin/claude",
            runner=lambda *args, **kwargs: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(args[0], 3)
            ),
        )

        self.assertIn("not found", missing)
        self.assertIn(CLAUDE_UPGRADE_COMMAND, missing)
        self.assertIn("could not verify", unreadable)
        self.assertIn(CLAUDE_UPGRADE_COMMAND, unreadable)

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
            "tool_timeout_sec = 608400",
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

    def test_client_paths_and_detected_agents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            (home / ".codex").mkdir(parents=True)
            (home / ".config" / "opencode").mkdir(parents=True)

            self.assertEqual(
                client_config_path("codex", home=home),
                home / ".codex" / "config.toml",
            )
            self.assertEqual(
                client_config_path("claude", home=home), home / ".claude.json"
            )
            self.assertEqual(
                client_config_path("opencode", home=home),
                home / ".config" / "opencode" / "opencode.json",
            )
            self.assertEqual(
                detected_agents(home=home, environ={"PATH": ""}),
                ["codex", "opencode"],
            )

    def test_setup_merges_all_clients_and_backs_up_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            codex = home / ".codex" / "config.toml"
            claude = home / ".claude.json"
            opencode = home / ".config" / "opencode" / "opencode.json"
            codex.parent.mkdir(parents=True)
            claude.parent.mkdir(parents=True, exist_ok=True)
            opencode.parent.mkdir(parents=True)
            codex.write_text(
                '[mcp_servers.other]\ncommand = "other"\n', encoding="utf-8"
            )
            claude.write_text(
                json.dumps({"theme": "dark", "mcpServers": {"other": {}}}),
                encoding="utf-8",
            )
            opencode.write_text(
                json.dumps({"mcp": {"servers": {"other": {}}}}),
                encoding="utf-8",
            )

            results = setup_configs(all_agents=True, home=home)

            self.assertEqual([result.status for result in results], [
                "configured", "configured", "configured"
            ])
            self.assertIn("[mcp_servers.other]", codex.read_text(encoding="utf-8"))
            self.assertIn("[mcp_servers.falcon]", codex.read_text(encoding="utf-8"))
            self.assertTrue(list(codex.parent.glob("config.toml.falcon-backup-*")))
            claude_value = json.loads(claude.read_text(encoding="utf-8"))
            self.assertEqual(claude_value["theme"], "dark")
            self.assertIn("falcon", claude_value["mcpServers"])
            opencode_value = json.loads(opencode.read_text(encoding="utf-8"))
            self.assertIn("falcon", opencode_value["mcp"]["servers"])

            unchanged = setup_configs(all_agents=True, home=home)
            self.assertEqual(
                [result.status for result in unchanged],
                ["unchanged", "unchanged", "unchanged"],
            )

    def test_setup_dry_run_does_not_write_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            results = setup_configs(
                ["codex", "claude", "opencode"], home=home, dry_run=True
            )

            self.assertEqual(
                [result.status for result in results],
                ["would configure", "would configure", "would configure"],
            )
            self.assertFalse(home.exists())

    def test_setup_preserves_existing_opencode_direct_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "home"
            path = client_config_path("opencode", home=home)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"mcp": {"other": {}}}), encoding="utf-8")

            setup_configs(["opencode"], home=home)

            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("other", value["mcp"])
            self.assertIn("falcon", value["mcp"])
            self.assertNotIn("servers", value["mcp"])


if __name__ == "__main__":
    unittest.main()
