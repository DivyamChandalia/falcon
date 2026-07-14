"""Operational job commands shared by the CLI and shell completion."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

from .config import logname


def state_path() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "falcon"
    return cache / "last-job"


def remember_job(name: str) -> None:
    target = state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(name + "\n", encoding="utf-8")


def last_job() -> Optional[str]:
    target = state_path()
    if not target.exists():
        return None
    value = target.read_text(encoding="utf-8").strip()
    return value or None


def _target(name: Optional[str]) -> str:
    value = name or last_job()
    if not value:
        raise ValueError("no job supplied and no previous Falcon job was recorded")
    return value


def kubectl(args: List[str], capture: bool = False, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl"] + args,
        capture_output=capture,
        text=True,
        timeout=timeout,
        check=False,
    )


def job_names(namespace: str) -> List[str]:
    result = kubectl(
        ["get", "jobs.batch", "-n", namespace, "-o", "jsonpath={range .items[*]}{.metadata.name}{'\\n'}{end}"],
        capture=True,
        timeout=8,
    )
    return sorted(line for line in result.stdout.splitlines() if line) if result.returncode == 0 else []


def logs(namespace: str, name: Optional[str]) -> int:
    target = _target(name)
    remember_job(target)
    return kubectl(["logs", "-f", f"job.batch/{target}", "-n", namespace]).returncode


def attach(namespace: str, name: Optional[str]) -> int:
    target = _target(name)
    remember_job(target)
    return kubectl(["attach", f"job.batch/{target}", "-n", namespace]).returncode


def delete(namespace: str, names: Iterable[str]) -> int:
    targets = list(names) or [_target(None)]
    return kubectl(["delete", "job.batch", *targets, "-n", namespace]).returncode


def clean(namespace: str) -> int:
    result = kubectl(["get", "jobs.batch", "-n", namespace, "-o", "json"], capture=True, timeout=10)
    if result.returncode != 0:
        return result.returncode
    data = json.loads(result.stdout)
    completed = []
    for item in data.get("items", []):
        status = item.get("status", {})
        if not status.get("active") and (status.get("succeeded") or status.get("failed")):
            completed.append(item.get("metadata", {}).get("name"))
    completed = [name for name in completed if name]
    if not completed:
        print("[falcon] No completed or failed jobs to clean.")
        return 0
    print(f"[falcon] Cleaning {len(completed)} job(s): {' '.join(completed)}")
    return delete(namespace, completed)


def top(namespace: str, name: Optional[str]) -> int:
    target = _target(name)
    remember_job(target)
    result = kubectl(
        [
            "get", "pods", "-n", namespace, "-l", f"job-name={target}",
            "--field-selector=status.phase=Running", "-o", "jsonpath={.items[0].metadata.name}",
        ],
        capture=True,
        timeout=8,
    )
    pod = result.stdout.strip()
    if not pod:
        raise ValueError(f"no running pod found for job {target}")
    python = f"/media/beegfs/users/{logname()}/miniforge/bin/python"
    return kubectl(["exec", "-it", "-n", namespace, pod, "--", python, "-m", "nvitop"]).returncode
