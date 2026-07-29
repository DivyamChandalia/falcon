"""Typed domain models shared by Falcon submission and Kubernetes operations.

The classes in this module deliberately contain no Kubernetes client behavior.
They describe a user's request, the resources Falcon resolved for it, the
manifest produced from those values, and the identity returned by the API.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .quantities import parse_cpu, parse_memory_bytes

_DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EnvironmentValidationError(ValueError):
    """Raised when a requested Python environment cannot be mounted safely."""


class EnvironmentKind(str, Enum):
    """Supported Python environment layouts."""

    CONDA = "conda"
    VENV = "venv"


@dataclass(frozen=True)
class GPURequest:
    """A GPU model/count request and its Kubernetes extended resource name."""

    model: str
    count: int = 1
    resource_name: str = "nvidia.com/gpu"

    def __post_init__(self) -> None:
        model = self.model.strip()
        if not model:
            raise ValueError("GPU model must not be empty")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("GPU count must be a positive integer")
        if "/" not in self.resource_name or any(char.isspace() for char in self.resource_name):
            raise ValueError("GPU resource name must be a Kubernetes extended resource name")
        object.__setattr__(self, "model", model)


@dataclass(frozen=True)
class ComputeRequest:
    """CPU, memory, and shared-memory quantities for one Job container.

    CPU and memory limits are optional on an unplanned request. Falcon's
    planners normally resolve them to the same value as the request so cluster
    scheduling and runtime limits remain predictable.
    """

    cpu: str
    memory: str
    cpu_limit: Optional[str] = None
    memory_limit: Optional[str] = None
    shared_memory: Optional[str] = None

    def __post_init__(self) -> None:
        parse_cpu(self.cpu)
        parse_memory_bytes(self.memory)
        if self.cpu_limit is not None:
            if parse_cpu(self.cpu_limit) < parse_cpu(self.cpu):
                raise ValueError("CPU limit must be greater than or equal to its request")
        if self.memory_limit is not None:
            if parse_memory_bytes(self.memory_limit) < parse_memory_bytes(self.memory):
                raise ValueError("memory limit must be greater than or equal to its request")
        if self.shared_memory is not None:
            parse_memory_bytes(self.shared_memory)

    def normalized(self) -> "ComputeRequest":
        """Return a request whose CPU and memory limits equal its requests."""

        return ComputeRequest(
            cpu=self.cpu,
            memory=self.memory,
            cpu_limit=self.cpu,
            memory_limit=self.memory,
            shared_memory=self.shared_memory,
        )

    @property
    def cpu_pair(self) -> str:
        """Return the compact ``request:limit`` representation used by the CLI."""

        return _pair_text(self.cpu, self.cpu_limit)

    @property
    def memory_pair(self) -> str:
        """Return the compact ``request:limit`` representation used by the CLI."""

        return _pair_text(self.memory, self.memory_limit)


@dataclass(frozen=True)
class RuntimeEnvironment:
    """A validated Conda or virtual environment mounted into a Job.

    Use :meth:`from_path` instead of constructing this class directly. For a
    virtual environment, ``base_path`` records the interpreter installation
    named by ``pyvenv.cfg`` so symlinked Python binaries continue to work in
    the container.
    """

    kind: EnvironmentKind
    path: Path
    base_path: Optional[Path] = None

    @classmethod
    def from_path(
        cls,
        path: os.PathLike[str] | str,
        kind: Optional[EnvironmentKind | str] = None,
    ) -> "RuntimeEnvironment":
        """Detect and validate an environment at *path*.

        A Conda environment must contain ``conda-meta`` and ``bin``. A virtual
        environment must contain ``pyvenv.cfg`` and ``bin``. Missing or stale
        paths fail before a Kubernetes Job is created.
        """

        target = Path(path).expanduser()
        try:
            target = target.resolve(strict=True)
        except FileNotFoundError as exc:
            raise EnvironmentValidationError(
                f"Python environment does not exist: {target}"
            ) from exc
        if not target.is_dir():
            raise EnvironmentValidationError(f"Python environment is not a directory: {target}")

        requested_kind: Optional[EnvironmentKind]
        if kind is None:
            requested_kind = None
        elif isinstance(kind, EnvironmentKind):
            requested_kind = kind
        else:
            try:
                requested_kind = EnvironmentKind(str(kind).lower())
            except ValueError as exc:
                raise EnvironmentValidationError(
                    f"unsupported environment kind {kind!r}; expected conda or venv"
                ) from exc

        if requested_kind is None:
            if (target / "conda-meta").is_dir():
                requested_kind = EnvironmentKind.CONDA
            elif (target / "pyvenv.cfg").is_file():
                requested_kind = EnvironmentKind.VENV
            else:
                raise EnvironmentValidationError(
                    f"{target} is not a Conda or virtual environment "
                    "(missing conda-meta or pyvenv.cfg)"
                )

        marker = target / ("conda-meta" if requested_kind is EnvironmentKind.CONDA else "pyvenv.cfg")
        marker_ok = marker.is_dir() if requested_kind is EnvironmentKind.CONDA else marker.is_file()
        if not marker_ok:
            raise EnvironmentValidationError(
                f"{target} is not a {requested_kind.value} environment "
                f"(missing {marker.name})"
            )
        if not (target / "bin").is_dir():
            raise EnvironmentValidationError(
                f"Python environment is incomplete (missing {target / 'bin'})"
            )

        base_path = None
        if requested_kind is EnvironmentKind.VENV:
            home = _read_pyvenv_home(marker)
            if home:
                base = Path(home).expanduser()
                try:
                    base = base.resolve(strict=True)
                except FileNotFoundError as exc:
                    raise EnvironmentValidationError(
                        f"virtual environment base interpreter path no longer exists: {base}"
                    ) from exc
                if not base.is_dir():
                    raise EnvironmentValidationError(
                        f"virtual environment base interpreter is not a directory: {base}"
                    )
                try:
                    base.relative_to(target)
                except ValueError:
                    base_path = base

        return cls(kind=requested_kind, path=target, base_path=base_path)

    @classmethod
    def from_current(
        cls,
        environ: Optional[Mapping[str, str]] = None,
    ) -> Optional["RuntimeEnvironment"]:
        """Resolve ``VIRTUAL_ENV`` or ``CONDA_PREFIX`` from a process environment."""

        values = os.environ if environ is None else environ
        if values.get("VIRTUAL_ENV"):
            return cls.from_path(values["VIRTUAL_ENV"], EnvironmentKind.VENV)
        if values.get("CONDA_PREFIX"):
            return cls.from_path(values["CONDA_PREFIX"], EnvironmentKind.CONDA)
        return None

    @property
    def mount_paths(self) -> Tuple[Path, ...]:
        """Host paths required to use the environment inside a container."""

        return (self.path,) if self.base_path is None else (self.path, self.base_path)


@dataclass(frozen=True)
class JobRequest:
    """A fully parsed user request, independent of CLI and Kubernetes details."""

    name: str
    namespace: str
    command: Tuple[str, ...] = ()
    gpu: Optional[GPURequest] = None
    compute: Optional[ComputeRequest] = None
    environment: Optional[RuntimeEnvironment] = None
    image: Optional[str] = None
    working_dir: Optional[str] = None
    env: Mapping[str, str] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)
    annotations: Mapping[str, str] = field(default_factory=dict)
    pin_node: bool = False

    def __post_init__(self) -> None:
        name = self.name.strip()
        namespace = self.namespace.strip()
        _validate_dns_label(name, "Job name")
        _validate_dns_label(namespace, "namespace")
        command = tuple(self.command)
        if any(not isinstance(part, str) or not part or "\0" in part for part in command):
            raise ValueError("command arguments must be non-empty strings without NUL bytes")
        if self.image is not None and not self.image.strip():
            raise ValueError("container image must not be empty")
        if self.working_dir is not None and not Path(self.working_dir).is_absolute():
            raise ValueError("working directory must be an absolute path")
        env = _string_mapping(self.env, "environment", validate_env_names=True)
        labels = _string_mapping(self.labels, "labels")
        annotations = _string_mapping(self.annotations, "annotations")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "env", env)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "annotations", annotations)


@dataclass(frozen=True)
class NodeResources:
    """Planning-oriented allocatable and currently requested node resources."""

    name: str
    cpu_total: float = 0.0
    cpu_used: float = 0.0
    memory_total_gib: float = 0.0
    memory_used_gib: float = 0.0
    gpu_total: int = 0
    gpu_used: int = 0
    gpu_product: str = ""
    unschedulable: bool = False
    scheduling_info_available: bool = True

    @property
    def cpu_free(self) -> float:
        return max(0.0, self.cpu_total - self.cpu_used)

    @property
    def memory_free_gib(self) -> float:
        return max(0.0, self.memory_total_gib - self.memory_used_gib)

    @property
    def gpu_free(self) -> int:
        return max(0, self.gpu_total - self.gpu_used)


@dataclass(frozen=True)
class ResourcePlan:
    """Resolved resources and scheduling information for a :class:`JobRequest`."""

    preset: str
    compute: ComputeRequest
    gpu: Optional[GPURequest] = None
    node: Optional[str] = None
    immediately_schedulable: bool = True
    warning: Optional[str] = None
    sizing_node: Optional[str] = None

    @property
    def gpu_type(self) -> str:
        """Compatibility/display form of the requested GPU model."""

        return self.gpu.model if self.gpu else ""

    @property
    def gpu_count(self) -> int:
        """Requested GPU count, which is zero only when no GPU was requested."""

        return self.gpu.count if self.gpu else 0

    @property
    def cpu(self) -> str:
        """Compact CPU request/limit pair used by existing output formatters."""

        return self.compute.cpu_pair

    @property
    def memory(self) -> str:
        """Compact memory request/limit pair used by existing output formatters."""

        return self.compute.memory_pair


@dataclass(frozen=True)
class JobSpecification:
    """A request, its resource plan, and the generated Kubernetes manifest."""

    request: JobRequest
    plan: ResourcePlan
    manifest: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return an ordinary mapping suitable for JSON or YAML serialization."""

        return _deep_copy_mapping(self.manifest)


@dataclass(frozen=True)
class SubmittedJob:
    """Stable identity returned after Kubernetes accepts a Job manifest."""

    name: str
    namespace: str
    uid: Optional[str] = None
    resource_version: Optional[str] = None
    created: bool = True


def _pair_text(request: str, limit: Optional[str]) -> str:
    if limit is None:
        return request
    return f"{request}:{limit}"


def _validate_dns_label(value: str, label: str) -> None:
    if not value or len(value) > 63 or not _DNS_LABEL.fullmatch(value):
        raise ValueError(
            f"{label} must be a lowercase Kubernetes DNS label of at most 63 characters"
        )


def _string_mapping(
    values: Mapping[str, Any],
    label: str,
    *,
    validate_env_names: bool = False,
) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        if validate_env_names and not _ENV_NAME.fullmatch(key):
            raise ValueError(f"invalid environment variable name: {key!r}")
        if value is None or isinstance(value, (dict, list, tuple, set)):
            raise ValueError(f"{label} values must be scalar")
        result[key] = str(value)
    return result


def _read_pyvenv_home(path: Path) -> Optional[str]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EnvironmentValidationError(f"could not read {path}: {exc}") from exc
    for line in content.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() == "home":
            return value.strip() or None
    return None


def _deep_copy_mapping(value: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            result[key] = _deep_copy_mapping(item)
        elif isinstance(item, list):
            result[key] = [
                _deep_copy_mapping(entry) if isinstance(entry, Mapping) else entry
                for entry in item
            ]
        else:
            result[key] = item
    return result
