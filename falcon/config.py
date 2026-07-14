"""User configuration and internal cluster defaults for Falcon."""

from __future__ import annotations

import copy
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


DEFAULT_DASHBOARD_EMA_ALPHA = 0.1
LEGACY_DASHBOARD_EMA_ALPHAS = {0.02, 0.08, 0.25}


# These describe this Falcon deployment and intentionally stay out of setup and
# ~/.falconrc. Users configure workflow policy, not cluster plumbing.
INFRASTRUCTURE_DEFAULTS: Dict[str, Any] = {
    "cluster": {
        "kube_state_metrics_url": "http://localhost:30080/metrics",
        "gpu_label": "gpu-type",
        "hostname_label": "kubernetes.io/hostname",
    },
    "runtime": {
        "image": "registry.gitlab.com/hvlabs/teams/ai/container-images/base:ubuntu24.04-cuda13.0.2-runtime-withtools-v1.0.0",
        "image_pull_secrets": ["hv-gitlab-registry"],
        "shell": "zsh",
        "scheduler": "kai-scheduler",
        "mount_home": True,
        "environment": {
            "IN_JET_POD": "1",
            "USER": "${USER}",
            "CONDA_AUTO_ACTIVATE_BASE": "false",
        },
    },
}

USER_DEFAULTS: Dict[str, Any] = {
    "version": 1,
    "resources": {"shared_memory_percent": 15},
    "presets": {
        "h100": {"gpu_type": "h100", "minimum_utilization": 90},
        "a6000": {"gpu_type": "a6000", "minimum_utilization": 30},
        "2080ti": {"gpu_type": "2080ti", "minimum_utilization": 30},
    },
    "dashboard": {"ema_alpha": DEFAULT_DASHBOARD_EMA_ALPHA},
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def logname() -> str:
    value = os.environ.get("LOGNAME") or os.environ.get("USER")
    if not value:
        raise ValueError("LOGNAME is not set; Falcon cannot derive namespace and mounts")
    return value


def namespace_from_logname(value: Optional[str] = None) -> str:
    # divyam.c -> divyamc-dev, matching this cluster's namespace convention.
    return f"{(value or logname()).replace('.', '')}-dev"


def effective_defaults() -> Dict[str, Any]:
    config = _merge(INFRASTRUCTURE_DEFAULTS, USER_DEFAULTS)
    identity = logname()
    config["cluster"]["namespace"] = namespace_from_logname(identity)
    config["runtime"]["volumes"] = [
        f"/media/beegfs/users/{identity}/",
        "/media/beegfs/teams/",
    ]
    return config


# Kept as a public value for callers/tests; load_config recomputes identity so
# LOGNAME changes are always respected.
DEFAULT_CONFIG: Dict[str, Any] = effective_defaults()


def config_path(path: Optional[str] = None) -> Path:
    return Path(path or os.environ.get("FALCON_CONFIG", "~/.falconrc")).expanduser()


def load_config(path: Optional[str] = None, require_exists: bool = False) -> Dict[str, Any]:
    target = config_path(path)
    raw: Dict[str, Any] = {}
    if target.exists():
        with target.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Falcon config must be a YAML mapping: {target}")
    elif require_exists:
        raise FileNotFoundError(f"Falcon config not found: {target}. Run 'falcon setup'.")
    # Infrastructure keys from older preview configs are deliberately ignored;
    # they are no longer user configuration. Also discard retired aliases and
    # fixed shm_size fields while retaining custom preset definitions.
    user_raw: Dict[str, Any] = {}
    for key in ("version", "resources"):
        if key in raw:
            user_raw[key] = raw[key]
    if isinstance(raw.get("dashboard"), dict):
        user_raw["dashboard"] = {
            key: value for key, value in raw["dashboard"].items()
            if key == "ema_alpha"
        }
        # Preview/setup versions wrote 0.25, 0.08, and later 0.02 into every config.
        # Treat those known generated values as legacy defaults so existing users
        # receive the smoother dashboard default. Other values remain overrides.
        try:
            if float(user_raw["dashboard"].get("ema_alpha")) in LEGACY_DASHBOARD_EMA_ALPHAS:
                user_raw["dashboard"].pop("ema_alpha", None)
        except (TypeError, ValueError):
            pass
    if isinstance(raw.get("presets"), dict):
        user_raw["presets"] = {
            name: {
                key: value for key, value in preset.items()
                if key in {"gpu_type", "minimum_utilization", "shared_memory_percent"}
            }
            for name, preset in raw["presets"].items()
            if isinstance(preset, dict)
        }
    config = _merge(effective_defaults(), user_raw)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    if config.get("version") != 1:
        raise ValueError("Unsupported .falconrc version (expected 1)")
    percent = float(config.get("resources", {}).get("shared_memory_percent", 15))
    if not 0 < percent <= 100:
        raise ValueError("resources.shared_memory_percent must be between 0 and 100")
    if not config.get("presets"):
        raise ValueError("At least one GPU preset is required")
    ema_alpha = float(config.get("dashboard", {}).get("ema_alpha", DEFAULT_DASHBOARD_EMA_ALPHA))
    if not 0 < ema_alpha <= 1:
        raise ValueError("dashboard.ema_alpha must be greater than 0 and at most 1")
    for name, preset in config["presets"].items():
        if not preset.get("gpu_type"):
            raise ValueError(f"presets.{name}.gpu_type is required")
        override = preset.get("shared_memory_percent")
        if override is not None and not 0 < float(override) <= 100:
            raise ValueError(f"presets.{name}.shared_memory_percent must be between 0 and 100")


def _ask(label: str, default: Any) -> str:
    answer = input(f"{label} [{default}]: ").strip()
    return answer or str(default)


def detect_shell() -> Tuple[str, Path]:
    requested = os.environ.get("FALCON_SHELL")
    candidates = [requested]
    try:
        result = subprocess.run(
            ["ps", "-p", str(os.getppid()), "-o", "comm="], capture_output=True, text=True, timeout=2
        )
        candidates.append(result.stdout.strip().lstrip("-"))
    except (OSError, subprocess.SubprocessError):
        pass
    candidates.append(Path(os.environ.get("SHELL", "bash")).name)
    for candidate in candidates:
        shell = Path(candidate).name if candidate else ""
        if shell in {"zsh", "bash"}:
            return shell, Path.home() / (".zshrc" if shell == "zsh" else ".bashrc")
    return "bash", Path.home() / ".bashrc"


def launcher_path() -> Path:
    return Path.home() / ".local" / "bin" / "falcon"


def install_launcher() -> Path:
    """Install an environment-independent entry path pinned to this Python."""
    target = launcher_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    python = shlex.quote(str(Path(sys.executable).resolve()))
    target.write_text(f'#!/bin/sh\nexec {python} -m falcon "$@"\n', encoding="utf-8")
    target.chmod(0o755)
    return target


def install_shell_integration(launcher: Optional[Path] = None) -> Path:
    shell, rc_path = detect_shell()
    launcher = launcher or launcher_path()
    quoted_launcher = shlex.quote(str(launcher))
    marker_start = "# >>> falcon native >>>"
    marker_end = "# <<< falcon native <<<"
    block = (
        f"{marker_start}\n"
        f'eval "$({quoted_launcher} shell-init {shell})"\n'
        f"{marker_end}\n"
    )
    existing = rc_path.read_text(encoding="utf-8") if rc_path.exists() else ""
    existing = _remove_legacy_falcon_shell(existing)
    if marker_start in existing and marker_end in existing:
        before, rest = existing.split(marker_start, 1)
        _, after = rest.split(marker_end, 1)
        updated = before.rstrip() + "\n\n" + block + after.lstrip("\n")
    else:
        updated = existing.rstrip() + "\n\n" + block
    rc_path.parent.mkdir(parents=True, exist_ok=True)
    rc_path.write_text(updated, encoding="utf-8")
    return rc_path


def _remove_legacy_falcon_shell(content: str) -> str:
    """Remove the preview zsh function/completion now replaced by native Falcon."""
    if "[falcon] Exported FALCON_LAST_JOB" not in content:
        return content
    return re.sub(
        r"(^|\n)falcon\(\) \{\n.*?\ncompdef _falcon falcon\n",
        r"\1",
        content,
        count=1,
        flags=re.DOTALL,
    )


def run_setup(
    path: Optional[str] = None,
    force: bool = False,
    non_interactive: bool = False,
    install_shell: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """Write only user-tunable policy and install completion for the active shell."""
    target = config_path(path)
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists; pass --force to replace it")
    config = copy.deepcopy(USER_DEFAULTS)
    if not non_interactive:
        print(f"Falcon setup for {logname()} ({namespace_from_logname()})")
        config["resources"]["shared_memory_percent"] = float(
            _ask("Shared memory as % of allocated RAM", config["resources"]["shared_memory_percent"])
        )
    validate_config(_merge(effective_defaults(), config))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    try:
        target.chmod(0o600)
    except OSError:
        pass
    launcher = install_launcher()
    rc_path = install_shell_integration(launcher) if install_shell else None
    return target, rc_path


def expanded(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))
