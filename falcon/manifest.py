"""Pure Kubernetes Job manifest generation for Falcon."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from .models import EnvironmentKind, JobRequest, JobSpecification, ResourcePlan
from .planning import canonical_gpu


DEFAULT_CONTAINER_PATH = (
    "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
    "/usr/sbin:/usr/bin:/sbin:/bin"
)


def build_job_specification(
    request: JobRequest,
    plan: ResourcePlan,
    config: Mapping[str, Any],
) -> JobSpecification:
    """Build a typed Job specification from structured Falcon inputs."""

    return JobSpecification(
        request=request,
        plan=plan,
        manifest=build_job_manifest(request, plan, config),
    )


def build_job_manifest(
    request: JobRequest,
    plan: ResourcePlan,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Generate a ``batch/v1`` Job without shell commands or Jet.

    This function performs no API calls and reads no process-global state. Any
    host-specific values (identity, paths, environment expansion) must already
    be represented by ``request`` or ``config``.
    """

    _validate_request_matches_plan(request, plan)
    runtime = _as_mapping(config.get("runtime", {}), "runtime")
    cluster = _as_mapping(config.get("cluster", {}), "cluster")
    job_config = _as_mapping(config.get("job", {}), "job")

    image = request.image or _optional_string(runtime.get("image"))
    if not image:
        raise ValueError("container image is required")

    labels = dict(_string_mapping(runtime.get("labels", {}), "runtime.labels"))
    labels.update(request.labels)
    labels["falcon.dev/managed"] = "true"
    if plan.gpu:
        labels["falcon.dev/gpu-type"] = canonical_gpu(plan.gpu.model)
        labels["falcon.dev/gpu-count"] = str(plan.gpu.count)

    pod_labels = dict(labels)
    pod_labels["job-name"] = request.name
    resources: Dict[str, Dict[str, str]] = {
        "requests": {
            "cpu": plan.compute.cpu,
            "memory": plan.compute.memory,
        }
    }
    limits: Dict[str, str] = {}
    if plan.compute.cpu_limit is not None:
        limits["cpu"] = plan.compute.cpu_limit
    if plan.compute.memory_limit is not None:
        limits["memory"] = plan.compute.memory_limit
    if plan.gpu:
        resources["requests"][plan.gpu.resource_name] = str(plan.gpu.count)
        limits[plan.gpu.resource_name] = str(plan.gpu.count)
    if limits:
        resources["limits"] = limits

    environment = dict(
        _string_mapping(runtime.get("environment", {}), "runtime.environment")
    )
    environment.update(request.env)
    volumes, mounts = _configured_volumes(runtime.get("volumes", ()))

    home_path = _optional_string(runtime.get("home"))
    if runtime.get("mount_home") and not home_path and mounts:
        # Falcon's generated configuration lists the user's home first.
        home_path = mounts[0]["mountPath"]
    if home_path:
        environment.setdefault("HOME", home_path)

    if request.environment is not None:
        _add_environment_mounts(volumes, mounts, request.environment)
        environment["PATH"] = (
            f"{request.environment.path}/bin:"
            f"{environment.get('PATH', DEFAULT_CONTAINER_PATH)}"
        )
        if request.environment.kind is EnvironmentKind.CONDA:
            environment["CONDA_PREFIX"] = str(request.environment.path)
            environment.pop("VIRTUAL_ENV", None)
        else:
            environment["VIRTUAL_ENV"] = str(request.environment.path)
            environment["PYTHONHOME"] = ""
            environment.pop("CONDA_PREFIX", None)

    if plan.compute.shared_memory:
        _add_volume(
            volumes,
            mounts,
            name="shared-memory",
            volume={"emptyDir": {"medium": "Memory", "sizeLimit": plan.compute.shared_memory}},
            mount_path="/dev/shm",
            read_only=False,
        )

    node_selector: Dict[str, str] = {}
    if plan.gpu:
        gpu_label = _optional_string(cluster.get("gpu_label")) or "gpu-type"
        node_selector[gpu_label] = plan.gpu.model
    configured_selectors = _string_mapping(
        runtime.get("node_selector", {}), "runtime.node_selector"
    )
    node_selector.update(configured_selectors)
    if request.pin_node:
        pinned_node = plan.node or plan.sizing_node
        if not pinned_node:
            raise ValueError("node pinning was requested but the resource plan has no node")
        hostname_label = (
            _optional_string(cluster.get("hostname_label"))
            or "kubernetes.io/hostname"
        )
        node_selector[hostname_label] = pinned_node

    pod_security, container_security = _security_context(runtime)
    command = list(request.command)
    if not command:
        debug_command = runtime.get("debug_command", ("sleep", "infinity"))
        if (
            isinstance(debug_command, (str, bytes))
            or not isinstance(debug_command, Iterable)
        ):
            raise ValueError("runtime.debug_command must be an argv list")
        command = [str(part) for part in debug_command]
        if not command or any(not part for part in command):
            raise ValueError("runtime.debug_command must contain non-empty arguments")

    container: Dict[str, Any] = {
        "name": "main",
        "image": image,
        "command": command,
        "resources": resources,
        "securityContext": container_security,
    }
    if request.working_dir:
        container["workingDir"] = request.working_dir
    image_pull_policy = _optional_string(runtime.get("image_pull_policy"))
    if image_pull_policy:
        if image_pull_policy not in {"Always", "IfNotPresent", "Never"}:
            raise ValueError("runtime.image_pull_policy is invalid")
        container["imagePullPolicy"] = image_pull_policy
    if environment:
        container["env"] = [
            {"name": key, "value": value}
            for key, value in sorted(environment.items())
        ]
    if mounts:
        container["volumeMounts"] = mounts

    pod_spec: Dict[str, Any] = {
        "restartPolicy": "Never",
        "securityContext": pod_security,
        "containers": [container],
    }
    scheduler = _optional_string(runtime.get("scheduler"))
    if scheduler:
        pod_spec["schedulerName"] = scheduler
    priority = _optional_string(runtime.get("priority_class"))
    if priority:
        pod_spec["priorityClassName"] = priority
    service_account = _optional_string(runtime.get("service_account"))
    if service_account:
        pod_spec["serviceAccountName"] = service_account
    secrets = runtime.get("image_pull_secrets", ())
    if secrets:
        if isinstance(secrets, (str, bytes)) or not isinstance(secrets, Iterable):
            raise ValueError("runtime.image_pull_secrets must be a list")
        names = []
        for secret in secrets:
            if not isinstance(secret, str) or not secret.strip():
                raise ValueError("image pull secret names must be non-empty strings")
            if secret not in names:
                names.append(secret)
        pod_spec["imagePullSecrets"] = [{"name": name} for name in names]
    if node_selector:
        pod_spec["nodeSelector"] = node_selector
    if volumes:
        pod_spec["volumes"] = volumes

    spec: Dict[str, Any] = {
        "template": {
            "metadata": {"labels": pod_labels},
            "spec": pod_spec,
        }
    }
    _optional_nonnegative_integer(job_config, "backoff_limit", spec, "backoffLimit")
    _optional_nonnegative_integer(
        job_config,
        "ttl_seconds_after_finished",
        spec,
        "ttlSecondsAfterFinished",
    )
    _optional_positive_integer(
        job_config,
        "active_deadline_seconds",
        spec,
        "activeDeadlineSeconds",
    )

    metadata: Dict[str, Any] = {
        "name": request.name,
        "namespace": request.namespace,
        "labels": labels,
    }
    if request.annotations:
        metadata["annotations"] = dict(request.annotations)
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": metadata,
        "spec": spec,
    }


def _validate_request_matches_plan(request: JobRequest, plan: ResourcePlan) -> None:
    if (request.gpu is None) != (plan.gpu is None):
        raise ValueError("Job request and resource plan disagree about GPU resources")
    if request.gpu and plan.gpu:
        if (
            canonical_gpu(request.gpu.model) != canonical_gpu(plan.gpu.model)
            or request.gpu.count != plan.gpu.count
            or request.gpu.resource_name != plan.gpu.resource_name
        ):
            raise ValueError("Job request and resource plan contain different GPU requests")


def _configured_volumes(values: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if values is None:
        return [], []
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError("runtime.volumes must be a list")
    volumes: List[Dict[str, Any]] = []
    mounts: List[Dict[str, Any]] = []
    for index, value in enumerate(values):
        if isinstance(value, str):
            host, separator, mounted = value.partition(":")
            host_path = host.strip()
            mount_path = mounted.strip() if separator else host_path
            name = f"host-{index}"
            host_type = "Directory"
            read_only = False
        elif isinstance(value, Mapping):
            host_path = _optional_string(value.get("host_path") or value.get("path"))
            mount_path = _optional_string(value.get("mount_path")) or host_path
            name = _optional_string(value.get("name")) or f"host-{index}"
            host_type = _optional_string(value.get("type")) or "Directory"
            read_only = bool(value.get("read_only", False))
        else:
            raise ValueError("runtime volume entries must be paths or mappings")
        if not host_path or not mount_path:
            raise ValueError("runtime volumes require host and mount paths")
        if not Path(host_path).is_absolute() or not Path(mount_path).is_absolute():
            raise ValueError("runtime volume paths must be absolute")
        if host_type not in {"Directory", "DirectoryOrCreate", "File", "FileOrCreate"}:
            raise ValueError(f"unsupported hostPath type: {host_type}")
        _add_volume(
            volumes,
            mounts,
            name=name,
            volume={"hostPath": {"path": host_path, "type": host_type}},
            mount_path=mount_path,
            read_only=read_only,
        )
    return volumes, mounts


def _add_environment_mounts(volumes, mounts, environment) -> None:
    _add_volume(
        volumes,
        mounts,
        name="python-environment",
        volume={
            "hostPath": {
                "path": str(environment.path),
                "type": "Directory",
            }
        },
        mount_path=str(environment.path),
        read_only=False,
    )
    if environment.base_path is not None:
        _add_volume(
            volumes,
            mounts,
            name="python-environment-base",
            volume={
                "hostPath": {
                    "path": str(environment.base_path),
                    "type": "Directory",
                }
            },
            mount_path=str(environment.base_path),
            read_only=True,
        )


def _add_volume(
    volumes: List[Dict[str, Any]],
    mounts: List[Dict[str, Any]],
    *,
    name: str,
    volume: Mapping[str, Any],
    mount_path: str,
    read_only: bool,
) -> None:
    clean_name = _volume_name(name)
    for existing_mount in mounts:
        if existing_mount["mountPath"] == mount_path:
            # An explicitly configured parent/exact mount remains authoritative.
            return
    existing_names = {item["name"] for item in volumes}
    candidate = clean_name
    suffix = 2
    while candidate in existing_names:
        candidate = f"{clean_name[: max(1, 61 - len(str(suffix)))]}-{suffix}"
        suffix += 1
    volume_item = {"name": candidate}
    volume_item.update(volume)
    mount_item: Dict[str, Any] = {"name": candidate, "mountPath": mount_path}
    if read_only:
        mount_item["readOnly"] = True
    volumes.append(volume_item)
    mounts.append(mount_item)


def _volume_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    normalized = re.sub(r"-+", "-", normalized)[:63].rstrip("-")
    return normalized or "volume"


def _security_context(runtime: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    configured_pod = dict(
        _as_mapping(runtime.get("security_context", {}), "runtime.security_context")
    )
    configured_container = dict(
        _as_mapping(
            runtime.get("container_security_context", {}),
            "runtime.container_security_context",
        )
    )
    pod: Dict[str, Any] = {"runAsNonRoot": True}
    container: Dict[str, Any] = {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
    }
    identity_keys = {
        "run_as_user": "runAsUser",
        "run_as_group": "runAsGroup",
        "supplemental_groups": "supplementalGroups",
        "fs_group": "fsGroup",
    }
    for config_key, manifest_key in identity_keys.items():
        if config_key in runtime:
            pod[manifest_key] = runtime[config_key]
    pod.update(configured_pod)
    container.update(configured_container)
    return pod, container


def _optional_nonnegative_integer(
    source: Mapping[str, Any],
    source_key: str,
    target: MutableMapping[str, Any],
    target_key: str,
) -> None:
    if source.get(source_key) is None:
        return
    value = source[source_key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"job.{source_key} must be an integer greater than or equal to zero")
    target[target_key] = value


def _optional_positive_integer(
    source: Mapping[str, Any],
    source_key: str,
    target: MutableMapping[str, Any],
    target_key: str,
) -> None:
    if source.get(source_key) is None:
        return
    value = source[source_key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"job.{source_key} must be a positive integer")
    target[target_key] = value


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _string_mapping(value: Any, label: str) -> Dict[str, str]:
    source = _as_mapping(value, label)
    result: Dict[str, str] = {}
    for key, item in source.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        if item is None or isinstance(item, (dict, list, tuple, set)):
            raise ValueError(f"{label} values must be scalar")
        result[key] = str(item)
    return result


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"expected a string, got {type(value).__name__}")
    return value.strip() or None
