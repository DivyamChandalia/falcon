from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from falcon import __version__
from falcon.cli import EXIT_UPDATE, _maybe_auto_update, main
from falcon.updates import (
    UpdateError,
    install_update,
    latest_changelog,
    latest_version,
    maybe_prompt_for_update,
    newer_version,
    pip_command,
    record_update_check,
    update_check_due,
    uv_tool_environment,
)


class _Response:
    def __init__(self, body: str) -> None:
        self.body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class UpdateTests(unittest.TestCase):
    def test_version_is_semver_and_newer_comparison_is_ordered(self) -> None:
        self.assertRegex(__version__, r"^\d+\.\d+\.\d+$")
        self.assertTrue(newer_version("0.2.0", "0.3.0"))
        self.assertFalse(newer_version("0.3.0", "0.3.0"))
        self.assertFalse(newer_version("0.3.0", "0.2.9"))

    def test_latest_version_reads_repository_manifest(self) -> None:
        result = latest_version(
            url="https://example.invalid/pyproject.toml",
            opener=lambda request, timeout: _Response(
                '__version__ = "0.4.0"\n'
            ),
        )
        self.assertEqual(result, "0.4.0")

    def test_latest_changelog_reads_the_newest_release_entries(self) -> None:
        result = latest_changelog(
            url="https://example.invalid/CHANGELOG.md",
            opener=lambda request, timeout: _Response(
                "# Changelog\n\n"
                "## 0.4.4 — 2026-09-29\n\n"
                "- Added the dashboard update summary.\n"
                "  It remains concise.\n"
                "- Fixed a progress bar flicker.\n\n"
                "## 0.4.3 — 2026-09-22\n"
                "- Older release.\n"
            ),
        )
        self.assertEqual(
            result,
            (
                "0.4.4",
                (
                    "Added the dashboard update summary. It remains concise.",
                    "Fixed a progress bar flicker.",
                ),
            ),
        )

    def test_update_check_timestamp_is_atomic_and_daily(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "falcon" / "update-check"
            self.assertTrue(record_update_check(target, now=100.0))
            self.assertFalse(update_check_due(target, now=100.0 + 60))
            self.assertTrue(update_check_due(target, now=100.0 + 24 * 60 * 60))
            self.assertEqual(target.read_text(encoding="utf-8").strip(), "100.000000")

    def test_daily_prompt_updates_once_and_then_stays_silent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "update-check"
            output = io.StringIO()
            installed = []
            kwargs = {
                "interactive": True,
                "stream": output,
                "input_fn": lambda prompt: "y",
                "state_path": target,
                "now": 100_000.0,
                "fetcher": lambda: "0.4.0",
                "installer": lambda: installed.append(True) or 0,
            }
            self.assertTrue(maybe_prompt_for_update("0.3.0", **kwargs))
            self.assertEqual(installed, [True])
            self.assertIn("A new Falcon version", output.getvalue())

            output.seek(0)
            output.truncate(0)
            self.assertFalse(maybe_prompt_for_update("0.3.0", **kwargs))
            self.assertEqual(output.getvalue(), "")

    def test_daily_prompt_does_not_install_when_declined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            installed = []
            shown = maybe_prompt_for_update(
                "0.3.0",
                interactive=True,
                stream=output,
                input_fn=lambda prompt: "n",
                state_path=Path(directory) / "update-check",
                now=100_000.0,
                fetcher=lambda: "0.4.0",
                installer=lambda: installed.append(True) or 0,
            )
            self.assertTrue(shown)
            self.assertEqual(installed, [])
            self.assertIn("Skipping Falcon update", output.getvalue())

    def test_json_and_noninteractive_update_paths_are_silent(self) -> None:
        output = io.StringIO()
        available = "99.0.0"
        with patch("falcon.cli.latest_version", return_value=available), redirect_stdout(output):
            code = main(["update", "--check"])
        self.assertEqual(code, 0)
        self.assertIn(f"Falcon {available} is available", output.getvalue())

    def test_auto_prompt_skips_json_and_noninteractive_commands(self) -> None:
        args = type("Args", (), {"output": "json"})()
        with patch("falcon.cli.maybe_prompt_for_update") as prompt:
            with patch.object(sys.stdin, "isatty", return_value=True), patch.object(
                sys.stdout, "isatty", return_value=True
            ):
                _maybe_auto_update("jobs", args)
                _maybe_auto_update("jobs", type("Args", (), {"output": "human"})())
            with patch.object(sys.stdin, "isatty", return_value=False), patch.object(
                sys.stdout, "isatty", return_value=False
            ):
                _maybe_auto_update("jobs", type("Args", (), {"output": "human"})())
        prompt.assert_called_once()

    def test_pip_update_returns_pip_status(self) -> None:
        calls = []

        class Result:
            returncode = 0

        status = install_update(
            source="git+https://example.invalid/falcon.git@main",
            runner=lambda command, check: calls.append((command, check)) or Result(),
        )
        self.assertEqual(status, 0)
        self.assertEqual(calls[0][1], False)
        self.assertEqual(calls[0][0][-1], "git+https://example.invalid/falcon.git@main")
        self.assertIn("--upgrade", pip_command())

    def test_uv_tool_update_preserves_owner_and_tool_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tool = Path(directory) / "custom-tools/falcon-k8s"
            tool.mkdir(parents=True)
            (tool / "uv-receipt.toml").write_text('[tool]\nrequirements = []\n')
            with patch("falcon.updates.sys.prefix", str(tool)), patch("falcon.updates.shutil.which", return_value="/usr/local/bin/uv"), patch.dict(os.environ, {"UV_TOOL_DIR": "/wrong/tools"}), patch("falcon.updates.subprocess.run") as runner:
                runner.return_value.returncode = 0
                self.assertEqual(uv_tool_environment(), tool)
                self.assertEqual(install_update(), 0)
            args, kwargs = runner.call_args
            self.assertEqual(args[0], ["/usr/local/bin/uv", "tool", "upgrade", "falcon-k8s"])
            self.assertEqual(kwargs["env"]["UV_TOOL_DIR"], str(tool.parent))
            self.assertFalse(kwargs["check"])
            self.assertNotIn("pip", args[0])
            self.assertNotIn("git+", " ".join(args[0]))

    def test_uv_missing_never_falls_back_to_pip(self) -> None:
        with patch("falcon.updates.uv_tool_environment", return_value=Path("/tools/falcon-k8s")), patch("falcon.updates.shutil.which", return_value=None), patch("falcon.updates.subprocess.run") as runner:
            with self.assertRaisesRegex(UpdateError, "uv is not on PATH"):
                install_update()
        runner.assert_not_called()

    def test_uv_failure_propagates_without_pip_retry(self) -> None:
        with patch("falcon.updates.uv_tool_environment", return_value=Path("/tools/falcon-k8s")), patch("falcon.updates.shutil.which", return_value="uv"), patch("falcon.updates.subprocess.run") as runner:
            runner.return_value.returncode = 12
            self.assertEqual(install_update(), 12)
            self.assertEqual(runner.call_count, 1)
            runner.side_effect = OSError("cannot execute")
            with self.assertRaisesRegex(UpdateError, "could not run uv"):
                install_update()

    def test_uv_created_venv_without_tool_receipt_uses_pip(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch("falcon.updates.sys.prefix", directory), patch("falcon.updates.subprocess.run") as runner:
            (Path(directory) / "pyvenv.cfg").write_text("uv = 0.10.0\n")
            runner.return_value.returncode = 0
            self.assertIsNone(uv_tool_environment())
            self.assertEqual(install_update(), 0)
            self.assertEqual(runner.call_args.args[0][1:4], ["-m", "pip", "install"])

    def test_manual_update_failure_has_distinct_exit_code(self) -> None:
        with patch("falcon.cli.install_update", return_value=17), redirect_stdout(io.StringIO()), patch(
            "sys.stderr", new_callable=io.StringIO
        ):
            code = main(["update"])
        self.assertEqual(code, EXIT_UPDATE)

    def test_manual_update_prints_a_short_changelog(self) -> None:
        output = io.StringIO()
        with patch("falcon.cli.install_update", return_value=0), patch(
            "falcon.cli.latest_changelog",
            return_value=("0.4.4", ("Added a concise update summary.",)),
        ), redirect_stdout(output):
            code = main(["update"])
        self.assertEqual(code, 0)
        self.assertIn("What's new in Falcon 0.4.4:", output.getvalue())
        self.assertIn("Added a concise update summary.", output.getvalue())

    def test_update_can_run_with_an_invalid_existing_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "falconrc"
            config.write_text("version: not-a-supported-version\n", encoding="utf-8")
            output = io.StringIO()
            with patch("falcon.cli.latest_version", return_value="0.3.0"), redirect_stdout(output):
                code = main(["--config", str(config), "update", "--check"])
            self.assertEqual(code, 0)
            self.assertIn("up to date", output.getvalue())


if __name__ == "__main__":
    unittest.main()
