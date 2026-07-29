"""Operational Job commands shared by the CLI and completion."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

from .kubernetes import KubernetesClient, KubernetesError


def state_path() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return cache / "falcon" / "last-job"


def remember_job(name: str) -> None:
    target = state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(name + "\n", encoding="utf-8")
    temporary.replace(target)


def last_job() -> Optional[str]:
    target = state_path()
    if not target.exists():
        return None
    value = target.read_text(encoding="utf-8").strip()
    return value or None


def target_job(name: Optional[str]) -> str:
    value = name or last_job()
    if not value:
        raise ValueError(
            "no Job supplied and no previous Falcon Job was recorded"
        )
    return value


def kubectl(
    args: List[str],
    capture: bool = False,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess[str]:
    """Compatibility wrapper retained for simple external command tests."""
    return subprocess.run(
        ["kubectl", *args],
        capture_output=capture,
        text=True,
        timeout=timeout,
        check=False,
    )


def job_names(namespace: str) -> List[str]:
    try:
        payload = KubernetesClient(namespace).list_jobs()
    except KubernetesError:
        return []
    return sorted(
        str((item.get("metadata") or {}).get("name") or "")
        for item in payload.get("items", [])
        if (item.get("metadata") or {}).get("name")
    )


def logs(
    namespace: str,
    name: Optional[str],
    *,
    tail: int = 100,
    follow: bool = False,
    container: Optional[str] = None,
) -> int:
    target = target_job(name)
    result = KubernetesClient(namespace).logs(
        target,
        tail=tail,
        follow=follow,
        container=container,
    )
    if not follow and result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.returncode == 0:
        remember_job(target)
    return result.returncode


def attach(namespace: str, name: Optional[str]) -> int:
    target = target_job(name)
    result = KubernetesClient(namespace).attach(target)
    if result.returncode == 0:
        remember_job(target)
    return result.returncode


def kill(namespace: str, names: Iterable[str]) -> List[str]:
    targets = list(names) or [target_job(None)]
    KubernetesClient(namespace).delete_jobs(targets)
    return targets


def clean(namespace: str) -> int:
    client = KubernetesClient(namespace)
    payload = client.list_jobs()
    completed = [
        str((item.get("metadata") or {}).get("name") or "")
        for item in payload.get("items", [])
        if not (item.get("status") or {}).get("active")
        and (item.get("status") or {}).get("succeeded")
    ]
    completed = [name for name in completed if name]
    if not completed:
        print("No succeeded Jobs to clean.")
        return 0
    client.delete_jobs(completed)
    print(f"Deleted {len(completed)} succeeded Job(s).")
    return 0


def top(namespace: str, name: Optional[str]) -> int:
    target = target_job(name)
    result = KubernetesClient(namespace).top(target)
    if result.returncode == 0:
        remember_job(target)
    return result.returncode
