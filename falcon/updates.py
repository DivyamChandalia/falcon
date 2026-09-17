"""Manual and opt-in daily updates for the Falcon CLI.

Falcon is distributed from the project's Git repository rather than a
package index. The updater therefore reads the single version declaration in
``falcon/__init__.py`` and uses the same source for installation.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, TextIO
from urllib.error import URLError
from urllib.request import Request, urlopen

from packaging.version import InvalidVersion, Version

UPDATE_SOURCE = "git+https://github.com/DivyamChandalia/falcon.git@main"
UPDATE_VERSION_URL = (
    "https://raw.githubusercontent.com/DivyamChandalia/falcon/main/falcon/__init__.py"
)
UPDATE_MANIFEST_URL = UPDATE_VERSION_URL
AUTO_UPDATE_INTERVAL_SECONDS = 24 * 60 * 60
_VERSION_DECLARATION = re.compile(
    r"(?m)^\s*(?:__version__|version)\s*=\s*[\"']([^\"']+)[\"']\s*$"
)


class UpdateError(RuntimeError):
    """A remote version check or package installation failed."""


def update_state_path() -> Path:
    """Return the private timestamp file used by the daily check.

    ``FALCON_UPDATE_STATE`` is intentionally supported for tests and for
    installations whose home directory is mounted read-only.  The normal
    location follows XDG state conventions and never touches ``.falconrc``.
    """

    override = os.environ.get("FALCON_UPDATE_STATE")
    if override:
        return Path(override).expanduser()
    root = os.environ.get("XDG_STATE_HOME")
    if root:
        return Path(root).expanduser() / "falcon" / "update-check"
    return Path.home() / ".local" / "state" / "falcon" / "update-check"


def update_check_due(
    path: Optional[Path] = None,
    *,
    now: Optional[float] = None,
    interval: float = AUTO_UPDATE_INTERVAL_SECONDS,
) -> bool:
    """Return whether the daily interactive check is due."""

    target = Path(path) if path is not None else update_state_path()
    try:
        last = float(target.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        last = 0.0
    current = time.time() if now is None else float(now)
    return current - last >= max(0.0, float(interval))


def record_update_check(
    path: Optional[Path] = None, *, now: Optional[float] = None
) -> bool:
    """Atomically record a completed daily check and report write success."""

    target = Path(path) if path is not None else update_state_path()
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    value = time.time() if now is None else float(now)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(f"{value:.6f}\n", encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(target)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        return False
    return True


def _manifest_version(text: str) -> str:
    match = _VERSION_DECLARATION.search(text)
    if not match:
        raise UpdateError("the Falcon update manifest has no project version")
    return match.group(1)


def latest_version(
    *,
    url: str = UPDATE_MANIFEST_URL,
    timeout: float = 3.0,
    opener: Optional[Callable[..., object]] = None,
) -> str:
    """Fetch and parse the version declared by the update source."""

    request = Request(url, headers={"User-Agent": "falcon-k8s-updater"})
    fetch = opener or urlopen
    try:
        response = fetch(request, timeout=timeout)
        with response as opened:
            body = opened.read().decode("utf-8")
    except (OSError, URLError, UnicodeError) as exc:
        raise UpdateError(f"could not check for updates: {exc}") from exc
    return _manifest_version(body)


def newer_version(current: str, latest: str) -> bool:
    """Compare two PEP 440 versions without making malformed input fatal."""

    try:
        return Version(latest) > Version(current)
    except InvalidVersion:
        return latest.strip() != current.strip()


def pip_command(source: str = UPDATE_SOURCE) -> list[str]:
    """Build the interpreter-local command used for a Falcon upgrade."""

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--upgrade",
    ]
    # A virtualenv/Conda environment owns its interpreter site-packages.  The
    # system interpreter should retain the user-level installation documented
    # by Falcon instead of asking for administrator privileges.
    if sys.prefix == sys.base_prefix and not os.environ.get("VIRTUAL_ENV"):
        command.insert(5, "--user")
    command.append(source)
    return command


def install_update(
    source: str = UPDATE_SOURCE,
    *,
    runner: Optional[Callable[..., object]] = None,
) -> int:
    """Install the latest Falcon source and return pip's exit status."""

    run = runner or subprocess.run
    try:
        result = run(pip_command(source), check=False)
    except OSError as exc:
        raise UpdateError(f"could not run pip: {exc}") from exc
    return int(getattr(result, "returncode", 1))


def maybe_prompt_for_update(
    current: str,
    *,
    interactive: bool,
    stream: Optional[TextIO] = None,
    input_fn: Optional[Callable[[str], str]] = None,
    state_path: Optional[Path] = None,
    now: Optional[float] = None,
    fetcher: Callable[[], str] = latest_version,
    installer: Callable[[], int] = install_update,
) -> bool:
    """Run the once-per-day TTY prompt, returning whether it was shown.

    Network failures and an unwritable state directory are deliberately quiet:
    an update check must never stop a normal launch or corrupt JSON output.
    """

    if not interactive or os.environ.get("FALCON_NO_UPDATE_CHECK", "").lower() in {
        "1", "true", "yes", "on",
    }:
        return False
    target = Path(state_path) if state_path is not None else update_state_path()
    if not update_check_due(target, now=now):
        return False
    if not record_update_check(target, now=now):
        return False
    try:
        available = fetcher()
    except UpdateError:
        return False
    if not newer_version(current, available):
        return False

    output = stream or sys.stdout
    print(
        f"A new Falcon version ({available}) is available "
        f"(you have {current}). Update now? [Y/n] ",
        end="",
        file=output,
        flush=True,
    )
    answer_reader = input_fn or input
    try:
        answer = answer_reader("").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(file=output)
        return True
    if answer not in {"", "y", "yes"}:
        print("Skipping Falcon update for 24 hours.", file=output)
        return True
    print("Updating Falcon…", file=output)
    try:
        status = installer()
    except UpdateError as exc:
        print(f"Falcon update failed: {exc}", file=output)
        return True
    if status == 0:
        print("Falcon updated. Start a new shell or rerun the command.", file=output)
    else:
        print(f"Falcon update failed (pip exited with status {status}).", file=output)
    return True


__all__ = [
    "AUTO_UPDATE_INTERVAL_SECONDS",
    "UPDATE_MANIFEST_URL",
    "UPDATE_SOURCE",
    "UPDATE_VERSION_URL",
    "UpdateError",
    "install_update",
    "latest_version",
    "maybe_prompt_for_update",
    "newer_version",
    "pip_command",
    "record_update_check",
    "update_check_due",
    "update_state_path",
]
