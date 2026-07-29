"""Native Falcon submission helpers."""

from __future__ import annotations

import os
import re
import secrets
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .kubernetes import KubernetesClient
from .manifest import build_job_specification
from .models import (
    JobRequest,
    JobSpecification,
    ResourcePlan,
    RuntimeEnvironment,
    SubmittedJob,
)


def job_name(command: Sequence[str]) -> str:
    parent = re.sub(
        r"[^a-z0-9]+", "-", Path.cwd().name.lower()
    ).strip("-")[:20] or "falcon"
    useful = next(
        (
            Path(part).stem
            for part in command
            if not part.startswith("-") and part not in {"python", "python3"}
        ),
        "debug",
    )
    useful = re.sub(r"[^a-z0-9]+", "-", useful.lower()).strip("-")[:20] or "cmd"
    return f"{parent}-{useful}-{secrets.token_hex(2)}"[:63].rstrip("-")


def resolve_environment(
    selection: Optional[str],
    config: Mapping[str, object],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[RuntimeEnvironment]:
    runtime = config.get("runtime", {})
    configured = (
        runtime.get("python_environment", "auto")
        if isinstance(runtime, Mapping)
        else "auto"
    )
    value = selection if selection is not None else configured
    if value is None or str(value).strip().lower() in {"none", "off", "-"}:
        return None
    if str(value).strip().lower() == "auto":
        return RuntimeEnvironment.from_current(os.environ if environ is None else environ)
    return RuntimeEnvironment.from_path(str(value))


def build_specification(
    request: JobRequest,
    plan: ResourcePlan,
    config: Mapping[str, object],
) -> JobSpecification:
    return build_job_specification(request, plan, config)


def submit(
    specification: JobSpecification,
    client: KubernetesClient,
) -> SubmittedJob:
    return client.create_job(specification.manifest)
