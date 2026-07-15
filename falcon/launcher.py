"""Translate a Falcon preset into a Jet invocation."""

from __future__ import annotations

import os
import random
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import expanded
from .resources import ResourcePlan, canonical_gpu, parse_memory_gib


def job_name(command: Sequence[str]) -> str:
    parent = re.sub(r"[^a-z0-9]+", "-", Path.cwd().name.lower()).strip("-")[:20] or "falcon"
    useful = next((Path(part).stem for part in command if not part.startswith("-") and part not in {"python", "python3"}), "debug")
    useful = re.sub(r"[^a-z0-9]+", "-", useful.lower()).strip("-")[:20] or "cmd"
    return f"{parent}-{useful}-{random.randint(1000, 9999)}"[:63].rstrip("-")


def jet_executable() -> List[str]:
    # Use the Jet package installed beside Falcon. A random older `jet` script
    # earlier on PATH could otherwise bypass fixes and features in this fork.
    return [sys.executable, "-m", "jet"]


def build_jet_command(
    config: Dict[str, Any],
    plan: ResourcePlan,
    command: Sequence[str],
    name: Optional[str] = None,
    async_mode: bool = False,
    dry_run: bool = False,
    shm_size: Optional[str] = None,
    shm_percent: Optional[float] = None,
    pin_node: bool = False,
    extra_jet_args: Optional[Sequence[str]] = None,
) -> List[str]:
    runtime = config["runtime"]
    cluster = config["cluster"]
    preset = config["presets"][plan.preset]
    percent = float(
        shm_percent
        if shm_percent is not None
        else preset.get("shared_memory_percent", config.get("resources", {}).get("shared_memory_percent", 15))
    )
    if not 0 < percent <= 100:
        raise ValueError("shared-memory percentage must be between 0 and 100")
    calculated_shm = max(0.1, round(parse_memory_gib(plan.memory) * percent / 100, 1))
    calculated_shm_text = f"{int(calculated_shm) if calculated_shm.is_integer() else calculated_shm}Gi"
    launch_type = "job" if command else "debug"
    result = jet_executable() + ["launch", launch_type, name or job_name(command)]
    if cluster.get("namespace"):
        result += ["--namespace", str(cluster["namespace"])]
    result += ["--image", runtime["image"]]
    for secret in runtime.get("image_pull_secrets", []):
        result += ["--image-pull-secrets", secret]
    if runtime.get("shell"):
        result += ["--shell", runtime["shell"]]
    if runtime.get("scheduler"):
        result += ["--scheduler", runtime["scheduler"]]
    if runtime.get("mount_home"):
        result.append("--mount-home")
    for volume in runtime.get("volumes", []):
        result += ["--volume", expanded(str(volume))]
    for key, value in runtime.get("environment", {}).items():
        if key == "FALCON_DEBUG_PROMPT":
            continue
        result += ["--env", f"{key}={expanded(str(value))}"]
    if not command:
        result += ["--env", f"FALCON_DEBUG_PROMPT={canonical_gpu(plan.gpu_type)}x{plan.gpu_count}"]
    python_env = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    if python_env:
        result += ["--pyenv", python_env]
    result += [
        "--working-dir", str(Path.cwd()),
        "--cpu", plan.cpu,
        "--memory", plan.memory,
        "--shm-size", shm_size or calculated_shm_text,
        "--gpu", str(plan.gpu_count),
        "--gpu-type", plan.gpu_type,
        "--job-labels", "falcon.dev/managed=true", f"falcon.dev/gpu-type={plan.gpu_type}",
    ]
    if pin_node and plan.node:
        result += ["--node-selector", f"{cluster.get('hostname_label', 'kubernetes.io/hostname')}={plan.node}"]
    if command:
        result += ["--command", shlex.join(list(command))]
        if not async_mode:
            result.append("--follow")
    if dry_run:
        result.append("--dry-run")
    result += list(extra_jet_args or [])
    return result


def launch(command: List[str], cleanup_name: Optional[str], cleanup: bool, namespace: Optional[str] = None) -> int:
    try:
        try:
            return subprocess.run(command).returncode
        except KeyboardInterrupt:
            return 130
    finally:
        if cleanup and cleanup_name:
            delete = jet_executable() + ["delete", cleanup_name]
            if namespace:
                delete += ["-n", namespace]
            subprocess.run(delete, check=False)
