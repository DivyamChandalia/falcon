"""Falcon configuration and idempotent setup.

Configuration is intentionally plain YAML.  Cluster-specific values live in
``~/.falconrc`` rather than in Python code, which keeps the installed package
usable from any shell and on more than one Kubernetes cluster.
"""

from __future__ import annotations

import copy
import getpass
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml


CONFIG_VERSION = 1
DEFAULT_DASHBOARD_EMA_ALPHA = 0.1
LEGACY_DASHBOARD_EMA_ALPHAS = {0.02, 0.08, 0.25}

# These defaults are deliberately portable.  Sites can set a private image,
# scheduler, labels, and mounts during setup or by editing the YAML file.
INFRASTRUCTURE_DEFAULTS: Dict[str, Any] = {
    "cluster": {
        "namespace": "default",
        "kube_state_metrics_url": None,
        "gpu_label": "gpu-type",
        "hostname_label": "kubernetes.io/hostname",
    },
    "runtime": {
        "image": "nvidia/cuda:12.8.1-runtime-ubuntu24.04",
        "image_pull_secrets": [],
        "shell": "/bin/bash",
        "scheduler": None,
        "mount_home": False,
        "volumes": [],
        "environment": {},
        "python_environment": "auto",
    },
}

USER_DEFAULTS: Dict[str, Any] = {
    "version": CONFIG_VERSION,
    "resources": {"shared_memory_percent": 15},
    "job": {"backoff_limit": None, "ttl_seconds_after_finished": None},
    "presets": {
        "h100": {"gpu_type": "h100", "minimum_utilization": 90},
        "a6000": {"gpu_type": "a6000", "minimum_utilization": 30},
        "2080ti": {"gpu_type": "2080ti", "minimum_utilization": 30},
    },
    "dashboard": {
        "ema_alpha": DEFAULT_DASHBOARD_EMA_ALPHA,
        "sort_field": "Age",
        "sort_direction": "desc",
    },
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _identity() -> str:
    """Return a best-effort identity without making ``falcon --help`` fail."""
    value = os.environ.get("LOGNAME") or os.environ.get("USER")
    if value:
        return value
    try:
        value = getpass.getuser()
    except (KeyError, OSError):
        value = ""
    return value or "user"


def logname() -> str:
    return _identity()


def namespace_from_logname(value: Optional[str] = None) -> str:
    """Retain the inexpensive legacy namespace convention as a setup hint."""
    return f"{(value or _identity()).replace('.', '')}-dev"


def _kubectl_namespace() -> Optional[str]:
    try:
        result = subprocess.run(
            [
                "kubectl", "config", "view", "--minify",
                "-o", "jsonpath={..namespace}",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip() if result.returncode == 0 else ""
    return value or None


def effective_defaults() -> Dict[str, Any]:
    config = _merge(INFRASTRUCTURE_DEFAULTS, USER_DEFAULTS)
    config["cluster"]["namespace"] = (
        os.environ.get("FALCON_NAMESPACE")
        or _kubectl_namespace()
        or namespace_from_logname()
    )
    image = os.environ.get("FALCON_IMAGE")
    if image:
        config["runtime"]["image"] = image
    return config


# Kept for callers and tests.  It is now safe in sparse/noninteractive
# environments because effective_defaults never requires login variables.
DEFAULT_CONFIG: Dict[str, Any] = effective_defaults()


def config_path(path: Optional[str] = None) -> Path:
    return Path(path or os.environ.get("FALCON_CONFIG", "~/.falconrc")).expanduser()


def _user_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Keep supported fields while ignoring retired preview-only settings."""
    result: Dict[str, Any] = {}
    for key in ("version", "resources", "job", "presets"):
        if key in raw:
            result[key] = copy.deepcopy(raw[key])
    if isinstance(raw.get("cluster"), dict):
        result["cluster"] = {
            key: value for key, value in raw["cluster"].items()
            if key in {
                "namespace", "kube_state_metrics_url", "gpu_label",
                "hostname_label",
            }
        }
    if isinstance(raw.get("runtime"), dict):
        result["runtime"] = {
            key: copy.deepcopy(value) for key, value in raw["runtime"].items()
            if key in {
                "image", "image_pull_secrets", "shell", "scheduler",
                "mount_home", "volumes", "environment",
                "python_environment",
            }
        }
    if isinstance(raw.get("dashboard"), dict):
        result["dashboard"] = {
            key: copy.deepcopy(value) for key, value in raw["dashboard"].items()
            if key in {
                "ema_alpha", "hidden_panes", "sort_field", "sort_direction",
            }
        }
        try:
            alpha = float(result["dashboard"].get("ema_alpha"))
            if alpha in LEGACY_DASHBOARD_EMA_ALPHAS:
                result["dashboard"].pop("ema_alpha", None)
        except (TypeError, ValueError):
            pass
    return result


def load_config(path: Optional[str] = None, require_exists: bool = False) -> Dict[str, Any]:
    target = config_path(path)
    if not target.exists():
        if require_exists:
            raise FileNotFoundError(
                f"Falcon config not found: {target}. Run 'falcon setup'."
            )
        config = effective_defaults()
        validate_config(config)
        return config
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid Falcon YAML in {target}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Falcon config must be a YAML mapping: {target}")
    config = _merge(_merge(INFRASTRUCTURE_DEFAULTS, USER_DEFAULTS), _user_config(raw))
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    if config.get("version") != CONFIG_VERSION:
        raise ValueError(
            f"Unsupported .falconrc version (expected {CONFIG_VERSION})"
        )
    cluster = config.get("cluster")
    runtime = config.get("runtime")
    if not isinstance(cluster, dict) or not cluster.get("namespace"):
        raise ValueError("cluster.namespace is required")
    if not isinstance(runtime, dict) or not runtime.get("image"):
        raise ValueError("runtime.image is required")
    if not isinstance(runtime.get("volumes", []), list) or not all(
        isinstance(value, str) and value for value in runtime.get("volumes", [])
    ):
        raise ValueError("runtime.volumes must be a list of non-empty paths")
    if not isinstance(runtime.get("image_pull_secrets", []), list):
        raise ValueError("runtime.image_pull_secrets must be a list")
    environment = runtime.get("environment", {})
    if not isinstance(environment, dict) or not all(
        isinstance(key, str)
        and key
        and not isinstance(value, (dict, list))
        for key, value in environment.items()
    ):
        raise ValueError("runtime.environment must be a KEY: VALUE mapping")
    if runtime.get("python_environment") is not None and not isinstance(
        runtime.get("python_environment"), str
    ):
        raise ValueError("runtime.python_environment must be auto, none, or a path")
    percent = float(config.get("resources", {}).get("shared_memory_percent", 15))
    if not 0 < percent <= 100:
        raise ValueError(
            "resources.shared_memory_percent must be between 0 and 100"
        )
    job = config.get("job", {})
    for key in ("backoff_limit", "ttl_seconds_after_finished"):
        value = job.get(key)
        if value is not None and (
            isinstance(value, bool) or int(value) != value or int(value) < 0
        ):
            raise ValueError(f"job.{key} must be an integer >= 0 or null")
    presets = config.get("presets")
    if not isinstance(presets, dict) or not presets:
        raise ValueError("At least one GPU preset is required")
    for name, preset in presets.items():
        if not isinstance(preset, dict) or not preset.get("gpu_type"):
            raise ValueError(f"presets.{name}.gpu_type is required")
        override = preset.get("shared_memory_percent")
        if override is not None and not 0 < float(override) <= 100:
            raise ValueError(
                f"presets.{name}.shared_memory_percent must be between 0 and 100"
            )
    dashboard = config.get("dashboard", {})
    ema_alpha = float(
        dashboard.get("ema_alpha", DEFAULT_DASHBOARD_EMA_ALPHA)
    )
    if not 0 < ema_alpha <= 1:
        raise ValueError("dashboard.ema_alpha must be greater than 0 and at most 1")
    hidden = dashboard.get("hidden_panes", [])
    if not isinstance(hidden, list) or any(
        pane not in {"selected", "resources", "events"} for pane in hidden
    ):
        raise ValueError(
            "dashboard.hidden_panes must contain only selected, resources, or events"
        )
    if dashboard.get("sort_field", "Age") not in {"Age", "Name", "Status"}:
        raise ValueError("dashboard.sort_field must be Age, Name, or Status")
    if dashboard.get("sort_direction", "desc") not in {"asc", "desc"}:
        raise ValueError("dashboard.sort_direction must be asc or desc")


def _save_dashboard_settings(
    settings: Dict[str, Any], path: Optional[str] = None
) -> Path:
    target = config_path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"Falcon config not found: {target}. Run 'falcon setup'."
        )
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Falcon config must be a YAML mapping: {target}")
    dashboard = raw.setdefault("dashboard", {})
    if not isinstance(dashboard, dict):
        raise ValueError("dashboard must be a YAML mapping")
    for key, value in settings.items():
        if value is None:
            dashboard.pop(key, None)
        else:
            dashboard[key] = value
    _atomic_yaml(target, raw)
    return target


def save_hidden_panes(panes: Iterable[str], path: Optional[str] = None) -> Path:
    hidden = sorted(set(panes))
    if any(pane not in {"selected", "resources", "events"} for pane in hidden):
        raise ValueError(
            "dashboard.hidden_panes must contain only selected, resources, or events"
        )
    return _save_dashboard_settings({"hidden_panes": hidden or None}, path)


def save_dashboard_sort(
    field: str, direction: str, path: Optional[str] = None
) -> Path:
    if field not in {"Age", "Name", "Status"}:
        raise ValueError("dashboard.sort_field must be Age, Name, or Status")
    if direction not in {"asc", "desc"}:
        raise ValueError("dashboard.sort_direction must be asc or desc")
    if field == "Status":
        direction = "asc"
    return _save_dashboard_settings(
        {"sort_field": field, "sort_direction": direction}, path
    )


def _ask(label: str, default: Any) -> str:
    answer = input(f"{label} [{default}]: ").strip()
    return answer or str(default)


def _parse_environment(value: str) -> Dict[str, str]:
    environment: Dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, separator, setting = item.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(
                f"Invalid environment variable {item!r}; expected KEY=VALUE"
            )
        environment[key] = setting.strip()
    return environment


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def detect_shell() -> Tuple[str, Path]:
    requested = os.environ.get("FALCON_SHELL")
    candidates = [requested]
    try:
        result = subprocess.run(
            ["ps", "-p", str(os.getppid()), "-o", "comm="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        candidates.append(result.stdout.strip().lstrip("-"))
    except (OSError, subprocess.SubprocessError):
        pass
    candidates.append(Path(os.environ.get("SHELL", "bash")).name)
    for candidate in candidates:
        shell = Path(candidate).name if candidate else ""
        if shell in {"zsh", "bash"}:
            return shell, Path.home() / (
                ".zshrc" if shell == "zsh" else ".bashrc"
            )
    return "bash", Path.home() / ".bashrc"


def install_shell_integration() -> Path:
    """Install completion only; the executable remains the real console script."""
    shell, rc_path = detect_shell()
    marker_start = "# >>> falcon completion >>>"
    marker_end = "# <<< falcon completion <<<"
    block = (
        f"{marker_start}\n"
        f'eval "$(falcon completion {shell})"\n'
        f"{marker_end}\n"
    )
    existing = rc_path.read_text(encoding="utf-8") if rc_path.exists() else ""
    existing = _remove_legacy_falcon_shell(existing)
    pattern = re.compile(
        re.escape(marker_start) + r".*?" + re.escape(marker_end) + r"\n?",
        re.DOTALL,
    )
    if pattern.search(existing):
        updated = pattern.sub(block, existing, count=1)
    else:
        updated = existing.rstrip() + ("\n\n" if existing.strip() else "") + block
    rc_path.parent.mkdir(parents=True, exist_ok=True)
    rc_path.write_text(updated, encoding="utf-8")
    return rc_path


def _remove_legacy_falcon_shell(content: str) -> str:
    if "[falcon] Exported FALCON_LAST_JOB" in content:
        content = re.sub(
            r"(^|\n)falcon\(\) \{\n.*?\ncompdef _falcon falcon\n",
            r"\1",
            content,
            count=1,
            flags=re.DOTALL,
        )
    content = re.sub(
        r"# >>> falcon native >>>.*?# <<< falcon native <<<\n?",
        "",
        content,
        flags=re.DOTALL,
    )
    return content


def _atomic_yaml(target: Path, data: Dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(target)


def run_setup(
    path: Optional[str] = None,
    force: bool = False,
    non_interactive: bool = False,
    install_shell: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """Create or validate config, then idempotently install completion."""
    target = config_path(path)
    if target.exists() and not force:
        load_config(str(target), require_exists=True)
    else:
        config = effective_defaults()
        if not non_interactive:
            print(f"Falcon setup for {_identity()}")
            config["cluster"]["namespace"] = _ask(
                "Kubernetes namespace", config["cluster"]["namespace"]
            )
            config["runtime"]["image"] = _ask(
                "Container image", config["runtime"]["image"]
            )
            scheduler = _ask(
                "Scheduler (blank uses Kubernetes default)",
                config["runtime"].get("scheduler") or "",
            )
            config["runtime"]["scheduler"] = scheduler or None
            config["runtime"]["volumes"] = _parse_csv(
                _ask("Host paths to mount (comma-separated)", "")
            )
            config["runtime"]["mount_home"] = (
                _ask("Mount the current home directory? (y/N)", "N").lower()
                in {"y", "yes"}
            )
            environment = input(
                "Environment variables (comma-separated KEY=VALUE) [none]: "
            ).strip()
            config["runtime"]["environment"] = _parse_environment(environment)
            config["resources"]["shared_memory_percent"] = float(
                _ask(
                    "Shared memory as % of requested RAM",
                    config["resources"]["shared_memory_percent"],
                )
            )
        validate_config(config)
        _atomic_yaml(target, config)
    rc_path = install_shell_integration() if install_shell else None
    return target, rc_path


def expanded(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))
