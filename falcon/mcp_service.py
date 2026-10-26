"""Domain service used by Falcon's MCP server.

The MCP transport deliberately knows very little about Kubernetes.  This
module is the boundary where structured MCP requests become Falcon domain
models, plans, bounded Kubernetes operations, and durable job references.

There are two details worth keeping local to this service:

* MCP-created Jobs are always agent-managed, have no Kubernetes retry loop,
  and carry an idempotency marker.
* The idempotency database stores only operation metadata and a digest.  It
  never stores a request, a manifest, or environment values.

The event implementation is imported lazily.  ``mcp_events`` is maintained by
the event-monitoring part of the server and is intentionally not required for
read-only job operations or for importing this module during startup.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .cluster import build_cluster_snapshot, build_job_snapshot
from .coder import generate_workspace_name
from .config import gpu_preset_max_count
from .dashboard import UsageCollector
from .kubernetes import KubernetesClient, KubernetesError, ProcessResult
from .launcher import resolve_environment
from .manifest import build_job_manifest
from .mcp_config import (
    DEFAULT_EVICTION_STARTUP_SECONDS,
    DEFAULT_WAIT_TIMEOUT_SECONDS,
    MAX_EVICTION_STARTUP_SECONDS,
    MIN_EVICTION_STARTUP_SECONDS,
)
from .mcp_risk import McpGpuRiskEvaluator
from .models import GPURequest, JobRequest, ResourcePlan
from .planning import canonical_gpu, plan_cpu_resources, plan_resources
from .resource_service import ResourceServiceClient, snapshot_nodes
from .resources import fetch_nodes
from .restart_history import HISTORY_ANNOTATION, RestartHistory, history_path

AGENT_LABEL = "falcon.dev/agent-managed"
IDEMPOTENCY_LABEL = "falcon.dev/idempotency-hash"
PREVIOUS_JOB_UID_ANNOTATION = "falcon.dev/previous-job-uid"
_RESTART_DELETE_TIMEOUT_SECONDS = 15.0

_DEFAULT_LOG_LIMIT = 3
_DEFAULT_LOG_TAIL = 50
_MAX_LIST_LIMIT = 500
_MAX_LOG_LIMIT = 50
_MAX_LOG_TAIL = 100_000
_MAX_LOG_PODS = 10
_MAX_LOG_RESPONSE_BYTES = 32 * 1024
_MAX_LOG_CHARS_PER_POD = 8_192
_MAX_LOG_BYTES_PER_POD = 8_192
_MAX_FAILURE_LOG_CHARS = 16 * 1024
_MEMORY_NUMBER = re.compile(
    r"^[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)$"
)
_GPU_COUNT_SUFFIX = re.compile(r"^(?P<model>.+?)[xX](?P<count>[0-9]+)$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JOB_CONTROLLER_LABELS = frozenset(
    {
        "job-name",
        "controller-uid",
        "batch.kubernetes.io/job-name",
        "batch.kubernetes.io/controller-uid",
    }
)

_REQUEST_FIELDS = frozenset(
    {
        "gpu",
        "gpus",
        "cpu",
        "memory",
        "command",
        "name",
        "namespace",
        "environment",
        "image",
        "working_dir",
        "env",
        "pin_node",
        "maximize",
        "shared_memory",
        "shared_memory_percent",
    }
)
_RETRY_FIELDS = frozenset(
    {
        "backoff_limit",
        "backoffLimit",
        "retry",
        "retries",
        "retry_policy",
        "restart_policy",
        "restartPolicy",
    }
)
_RESTART_FIELDS = frozenset(
    {
        "gpu",
        "gpus",
        "cpu",
        "memory",
        "command",
        "image",
        "working_dir",
        "env",
        "shared_memory",
        "shared_memory_percent",
    }
)


class MCPServiceError(ValueError):
    """A safe, client-facing MCP domain error."""


class IdempotencyConflict(MCPServiceError):
    """An idempotency key was reused for a different request."""


class StaleJobReference(MCPServiceError):
    """A supplied Job reference no longer identifies the live Job."""


class CoderControlledJob(MCPServiceError):
    """A destructive operation was attempted on a Coder-owned Job."""


@dataclasses.dataclass(frozen=True)
class _Operation:
    kind: str
    key: str
    request_digest: str
    planned_name: str
    namespace: str
    source_uid: str = ""
    created_uid: str = ""


class _OperationStore:
    """Small concurrency-safe SQLite store for create/restart reconciliation."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.is_symlink():
            raise MCPServiceError(
                f"refusing to use symlinked MCP state database: {self.path}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                file_stat = os.fstat(descriptor)
                if not stat.S_ISREG(file_stat.st_mode):
                    raise MCPServiceError(
                        f"MCP state database is not a regular file: {self.path}"
                    )
                os.fchmod(descriptor, 0o600)
                if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
                    raise MCPServiceError(
                        f"could not set private permissions on MCP state database: {self.path}"
                    )
            finally:
                os.close(descriptor)
        except MCPServiceError:
            raise
        except OSError as exc:
            raise MCPServiceError(
                f"could not create private MCP state database {self.path}: {exc}"
            ) from exc
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA busy_timeout = 30000")
        # EventStore opens this same database and intentionally uses DELETE
        # journaling.  Keep both connections on the same journal mode; the
        # busy timeout and short IMMEDIATE transactions provide the needed
        # cross-thread/process serialization.
        self._connection.execute("PRAGMA journal_mode = DELETE")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS falcon_mcp_operations (
                kind TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                planned_name TEXT NOT NULL,
                namespace TEXT NOT NULL,
                source_uid TEXT NOT NULL DEFAULT '',
                created_uid TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                PRIMARY KEY (kind, idempotency_key)
            )
            """
        )
        self._connection.commit()

    def get(self, kind: str, key: str) -> Optional[_Operation]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT kind, idempotency_key, request_digest, planned_name,
                       namespace, source_uid, created_uid
                  FROM falcon_mcp_operations
                 WHERE kind = ? AND idempotency_key = ?
                """,
                (kind, key),
            ).fetchone()
        if row is None:
            return None
        return _Operation(
            kind=str(row[0]),
            key=str(row[1]),
            request_digest=str(row[2]),
            planned_name=str(row[3]),
            namespace=str(row[4]),
            source_uid=str(row[5] or ""),
            created_uid=str(row[6] or ""),
        )

    def reserve(
        self,
        *,
        kind: str,
        key: str,
        request_digest: str,
        planned_name: str,
        namespace: str,
        source_uid: str = "",
    ) -> _Operation:
        """Insert an operation before a Kubernetes create, atomically."""

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    """
                    SELECT kind, idempotency_key, request_digest, planned_name,
                           namespace, source_uid, created_uid
                      FROM falcon_mcp_operations
                     WHERE kind = ? AND idempotency_key = ?
                    """,
                    (kind, key),
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        """
                        INSERT INTO falcon_mcp_operations
                            (kind, idempotency_key, request_digest, planned_name,
                             namespace, source_uid, created_uid, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, '', ?)
                        """,
                        (
                            kind,
                            key,
                            request_digest,
                            planned_name,
                            namespace,
                            source_uid,
                            time.time(),
                        ),
                    )
                    self._connection.commit()
                    return _Operation(
                        kind=kind,
                        key=key,
                        request_digest=request_digest,
                        planned_name=planned_name,
                        namespace=namespace,
                        source_uid=source_uid,
                    )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
        return _Operation(
            kind=str(row[0]),
            key=str(row[1]),
            request_digest=str(row[2]),
            planned_name=str(row[3]),
            namespace=str(row[4]),
            source_uid=str(row[5] or ""),
            created_uid=str(row[6] or ""),
        )

    def mark_created(self, kind: str, key: str, uid: str) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE falcon_mcp_operations
                   SET created_uid = ?
                 WHERE kind = ? AND idempotency_key = ?
                """,
                (uid, kind, key),
            )
            self._connection.commit()

    def restart_key(self, ref: Mapping[str, str]) -> Optional[str]:
        """Recover legacy explicit keys as well as Falcon-generated restarts."""
        with self._lock:
            row = self._connection.execute(
                """SELECT idempotency_key FROM falcon_mcp_operations
                   WHERE kind = 'restart' AND namespace = ? AND source_uid = ?
                   ORDER BY (created_uid != '') DESC, created_at ASC LIMIT 1""",
                (ref["namespace"], ref["uid"]),
            ).fetchone()
        return str(row[0]) if row else None

    def restarted_source(self, ref: Mapping[str, str]) -> bool:
        """Authorize historical log reads only for a recorded replacement."""
        with self._lock:
            return self._connection.execute(
                """
                SELECT 1 FROM falcon_mcp_operations
                 WHERE kind = 'restart' AND namespace = ? AND planned_name = ?
                   AND source_uid = ? AND created_uid != '' LIMIT 1
                """,
                (ref["namespace"], ref["name"], ref["uid"]),
            ).fetchone() is not None

    def close(self) -> None:
        with self._lock:
            try:
                self._connection.close()
            except sqlite3.ProgrammingError:
                pass


class JobService:
    """Synchronous Falcon job domain operations plus asynchronous event waits."""

    def __init__(
        self,
        config: Mapping[str, Any],
        client: Any = None,
        state_dir: Optional[os.PathLike[str] | str] = None,
    ) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("Falcon config must be a mapping")
        self.config = copy.deepcopy(dict(config))
        cluster = self.config.get("cluster") or {}
        if not isinstance(cluster, Mapping) or not str(cluster.get("namespace") or "").strip():
            raise ValueError("config.cluster.namespace is required")
        self.namespace = str(cluster["namespace"]).strip()

        self._owns_client = client is None
        if client is None:
            self.client, self.context = self._make_client(self.namespace)
        else:
            self.client = client
            self.context = getattr(client, "context", None)

        self._state_dir, self._db_path = self._state_paths(
            state_dir,
            context=self.context,
            namespace=self.namespace,
        )
        self._operations = _OperationStore(self._db_path)
        self._restart_history = RestartHistory(history_path(self.context, self.namespace, self._state_dir))
        self._metrics_lock = threading.RLock()
        self._metrics_collectors: dict[tuple[str, str, str], Any] = {}
        self._metrics_snapshots: dict[tuple[str, str, str], tuple[float, Any]] = {}
        self._risk_evaluators: dict[tuple[str, str, str], McpGpuRiskEvaluator] = {}
        self._risk_snapshots: dict[tuple[str, str, str], Optional[dict[str, Any]]] = {}
        self._event_lock = threading.RLock()
        self._event_store: Any = None
        self._monitor: Any = None
        self._monitor_refs: dict[str, dict[str, str]] = {}
        # Startup grace is an observation preference, not Job state. Keep it
        # in memory alongside the monitor so a later wait can adjust it
        # without changing the Kubernetes manifest.
        self._monitor_startup_times: dict[str, float] = {}
        self._closed = False
        # mcp_events.Monitor consumes this callback.  Keeping it as an
        # attribute also gives the server a narrow seam for a custom monitor.
        self._metrics_provider = self._monitor_metrics

    @staticmethod
    def _state_paths(
        state_dir: Optional[os.PathLike[str] | str],
        *,
        context: Optional[str],
        namespace: str,
    ) -> tuple[Path, Path]:
        if state_dir is not None:
            root = Path(state_dir).expanduser()
        else:
            root = Path(
                os.environ.get("XDG_STATE_HOME", "~/.local/state")
            ).expanduser() / "falcon" / "mcp"
        # A caller-provided root is a test/configuration root, not a shared
        # database.  Keep the context/namespace partition in both modes so a
        # reused idempotency key can never cross a kubeconfig context.
        context_key = hashlib.sha256(
            (str(context or "default") + "\0" + namespace).encode("utf-8")
        ).hexdigest()[:24]
        private = root / context_key
        if private.is_symlink():
            raise MCPServiceError(
                f"refusing symlinked MCP state directory: {private}"
            )
        try:
            private.mkdir(parents=True, exist_ok=True, mode=0o700)
            if private.is_symlink() or not private.is_dir():
                raise MCPServiceError(
                    f"MCP state path is not a private directory: {private}"
                )
            private.chmod(0o700)
            if stat.S_IMODE(private.stat().st_mode) != 0o700:
                raise MCPServiceError(
                    f"could not set private permissions on MCP state directory: {private}"
                )
        except MCPServiceError:
            raise
        except OSError as exc:
            raise MCPServiceError(
                f"could not create private MCP state directory {private}: {exc}"
            ) from exc
        return private, private / "mcp.sqlite3"

    @staticmethod
    def _make_client(namespace: str) -> tuple[Any, Optional[str]]:
        """Resolve the kubeconfig context before constructing the real client."""

        probe = KubernetesClient(namespace)
        context: Optional[str] = None
        try:
            result = probe._run(["config", "current-context"])
            context = str(getattr(result, "stdout", "") or "").strip() or None
        finally:
            close = getattr(probe, "close", None)
            if callable(close):
                close()
        return KubernetesClient(namespace, context=context), context

    # ------------------------------------------------------------------
    # Request parsing and planning
    # ------------------------------------------------------------------

    def plan_job(self, request: dict) -> dict:
        if not isinstance(request, Mapping):
            raise MCPServiceError("job request must be an object")
        resource_only = "command" not in request
        preview = {**request, **({"command": ["true"]} if resource_only else {})}
        job_request, plan, manifest = self._prepare_request(preview, force_agent=True)
        result = self._plan_response(job_request, plan, manifest)
        if resource_only:
            # Never imply the internal preview placeholder is the workload to
            # submit. No command is executed and submit_job still requires one.
            result["summary"].pop("command", None)
            result["resource_only"] = True
        return result

    def _prepare_request(
        self,
        request: Mapping[str, Any],
        *,
        force_agent: bool,
        generated_name_token: Optional[str] = None,
    ) -> tuple[JobRequest, ResourcePlan, dict[str, Any]]:
        job_request, plan = self._parse_request(
            request,
            generated_name_token=generated_name_token,
        )
        build_config = copy.deepcopy(self.config)
        if force_agent:
            self._force_config_no_retry(build_config)
        manifest = build_job_manifest(job_request, plan, build_config)
        if force_agent:
            self._force_agent_manifest(manifest)
        return job_request, plan, manifest

    def _parse_request(
        self,
        request: Mapping[str, Any],
        *,
        generated_name_token: Optional[str] = None,
    ) -> tuple[JobRequest, ResourcePlan]:
        if not isinstance(request, Mapping):
            raise MCPServiceError("job request must be an object")
        supplied = set(request)
        retry_fields = sorted(supplied & _RETRY_FIELDS)
        if retry_fields:
            raise MCPServiceError(
                "MCP Jobs do not accept retry settings; Falcon sets "
                "backoffLimit=0 and restartPolicy=Never"
            )
        unknown = sorted(supplied - _REQUEST_FIELDS)
        if unknown:
            raise MCPServiceError(
                "unsupported MCP submission field(s): " + ", ".join(unknown)
            )
        supplied_namespace = request.get("namespace")
        if supplied_namespace is not None and str(supplied_namespace) != self.namespace:
            raise MCPServiceError(
                f"MCP Jobs use configured namespace {self.namespace!r}; "
                f"namespace {supplied_namespace!r} is not allowed"
            )

        name = request.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise MCPServiceError("name must be a non-empty string when supplied")
        command = request.get("command")
        if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
            raise MCPServiceError("command must be a non-empty argv array")
        command_values = tuple(command)
        if not command_values or any(
            not isinstance(part, str) or not part or "\0" in part
            for part in command_values
        ):
            raise MCPServiceError(
                "command must be a non-empty argv array of non-empty strings"
            )
        if name is None:
            name = _generated_job_name(
                request,
                command_values,
                uniqueness=generated_name_token,
            )
        else:
            name = name.strip()

        gpu_token = request.get("gpu")
        gpu_count = request.get("gpus", 1)
        if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count <= 0:
            raise MCPServiceError("gpus must be a positive integer")
        gpu_model: Optional[str] = None
        preset_name: Optional[str] = None
        preset: Mapping[str, Any] = {}
        if gpu_token is not None:
            if isinstance(gpu_token, Mapping):
                gpu_model = str(gpu_token.get("model") or "").strip()
                if "count" in gpu_token:
                    embedded_count = gpu_token["count"]
                    if (
                        isinstance(embedded_count, bool)
                        or not isinstance(embedded_count, int)
                        or embedded_count <= 0
                    ):
                        raise MCPServiceError("gpu.count must be a positive integer")
                    if "gpus" in request and gpu_count != embedded_count:
                        raise MCPServiceError("gpu.count and gpus disagree")
                    gpu_count = embedded_count
            else:
                if not isinstance(gpu_token, str) or not gpu_token.strip():
                    raise MCPServiceError("gpu must be a non-empty preset or model")
                gpu_model = gpu_token.strip()
                match = _GPU_COUNT_SUFFIX.fullmatch(gpu_model)
                if match and "gpus" not in request:
                    gpu_model = match.group("model").strip()
                    gpu_count = int(match.group("count"))
            if not gpu_model:
                raise MCPServiceError("gpu must identify a GPU model or preset")
            preset_key = next(
                (
                    key
                    for key in (str(gpu_token), gpu_model)
                    if key in (self.config.get("presets") or {})
                ),
                None,
            )
            if preset_key is not None:
                preset_name = preset_key
                candidate = (self.config.get("presets") or {}).get(preset_key)
                if isinstance(candidate, Mapping):
                    preset = candidate
                    gpu_model = str(candidate.get("gpu_type") or gpu_model)
                    maximum = gpu_preset_max_count(candidate)
                    if gpu_count > maximum:
                        raise MCPServiceError(
                            f"GPU preset {preset_key} supports at most {maximum} GPU(s); "
                            f"requested {gpu_count}"
                        )
            gpu_model = canonical_gpu(gpu_model)
        elif gpu_count != 1:
            raise MCPServiceError("gpus requires gpu")

        cpu = self._optional_string(request.get("cpu"), "cpu")
        memory = self._optional_memory(request.get("memory"), "memory")
        shared_memory = self._optional_memory(
            request.get("shared_memory"), "shared_memory"
        )
        shared_percent = request.get(
            "shared_memory_percent",
            preset.get(
                "shared_memory_percent",
                (self.config.get("resources") or {}).get("shared_memory_percent", 15),
            ),
        )
        if isinstance(shared_percent, bool):
            raise MCPServiceError("shared_memory_percent must be numeric")
        try:
            shared_percent_value = float(shared_percent)
        except (TypeError, ValueError) as exc:
            raise MCPServiceError("shared_memory_percent must be numeric") from exc
        if not 0 < shared_percent_value <= 100:
            raise MCPServiceError("shared_memory_percent must be greater than 0 and at most 100")

        if gpu_model is None:
            if not cpu or not memory:
                raise MCPServiceError(
                    "CPU-only Jobs require both cpu and memory"
                )
            plan = plan_cpu_resources(
                cpu,
                memory,
                shared_memory=shared_memory,
                shared_memory_percent=shared_percent_value,
            )
            gpu = None
        else:
            nodes = self._planning_nodes()
            gpu = GPURequest(gpu_model, gpu_count)
            plan = plan_resources(
                nodes,
                preset_name or gpu_model,
                gpu.model,
                gpu.count,
                cpu_override=cpu,
                memory_override=memory,
                maximize=self._boolean(request.get("maximize", False), "maximize"),
                shared_memory=shared_memory,
                shared_memory_percent=shared_percent_value,
            )

        environment_value = request.get("environment")
        if environment_value is not None and not isinstance(environment_value, str):
            raise MCPServiceError("environment must be a path, auto, none, or null")
        environment = resolve_environment(environment_value, self.config)
        image = request.get("image")
        if image is not None and (not isinstance(image, str) or not image.strip()):
            raise MCPServiceError("image must be a non-empty string")
        working_dir = request.get("working_dir")
        if working_dir is not None and not isinstance(working_dir, str):
            raise MCPServiceError("working_dir must be an absolute path")
        env = request.get("env", {})
        if not isinstance(env, Mapping):
            raise MCPServiceError("env must be a mapping of names to scalar values")
        pin_node = self._boolean(request.get("pin_node", False), "pin_node")
        job_request = JobRequest(
            name=name,
            namespace=self.namespace,
            command=command_values,
            gpu=gpu,
            compute=plan.compute,
            environment=environment,
            image=image,
            working_dir=working_dir,
            env=env,
            pin_node=pin_node,
        )
        return job_request, plan

    def _planning_nodes(self) -> list[Any]:
        """Use the same public planning source order as the CLI."""

        cluster = self.config.get("cluster") or {}
        service_url = cluster.get("resource_service_url")
        if service_url is not None:
            source = ResourceServiceClient(str(service_url))
            try:
                return list(snapshot_nodes(source.snapshot()))
            finally:
                close = getattr(source, "close", None)
                if callable(close):
                    close()

        metrics_url = cluster.get("kube_state_metrics_url")
        if metrics_url:
            try:
                return list(fetch_nodes(str(metrics_url), timeout=8))
            except Exception:
                # Fall through to the Kubernetes inventory path.  That path
                # produces the clearest error when both sources are absent.
                pass
        snapshot = build_cluster_snapshot(
            nodes=self.client.list_nodes(),
            pods=self.client.list_pods(""),
        )
        return [
            _node_resources(node)
            for node in snapshot.nodes
        ]

    @staticmethod
    def _optional_string(value: Any, field: str) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise MCPServiceError(f"{field} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _optional_memory(value: Any, field: str) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise MCPServiceError(f"{field} must be a non-empty quantity")
        value = value.strip()

        def add_unit(part: str) -> str:
            part = part.strip()
            return f"{part}Gi" if _MEMORY_NUMBER.fullmatch(part) else part

        request, separator, limit = value.partition(":")
        return (
            f"{add_unit(request)}:{add_unit(limit)}"
            if separator
            else add_unit(request)
        )

    @staticmethod
    def _boolean(value: Any, field: str) -> bool:
        if not isinstance(value, bool):
            raise MCPServiceError(f"{field} must be boolean")
        return value

    @staticmethod
    def _force_config_no_retry(config: dict[str, Any]) -> None:
        job = config.setdefault("job", {})
        if not isinstance(job, dict):
            raise MCPServiceError("config.job must be a mapping")
        job["backoff_limit"] = 0

    @staticmethod
    def _force_agent_manifest(
        manifest: dict[str, Any],
        *,
        idempotency_tag: Optional[str] = None,
        previous_uid: Optional[str] = None,
    ) -> None:
        metadata = manifest.setdefault("metadata", {})
        labels = metadata.setdefault("labels", {})
        labels[AGENT_LABEL] = "true"
        if idempotency_tag:
            labels[IDEMPOTENCY_LABEL] = idempotency_tag
        template = (manifest.setdefault("spec", {})).setdefault("template", {})
        template_metadata = template.setdefault("metadata", {})
        template_labels = template_metadata.setdefault("labels", {})
        template_labels[AGENT_LABEL] = "true"
        if idempotency_tag:
            template_labels[IDEMPOTENCY_LABEL] = idempotency_tag
        template_spec = template.setdefault("spec", {})
        template_spec["restartPolicy"] = "Never"
        manifest["spec"]["backoffLimit"] = 0
        if previous_uid:
            annotations = metadata.setdefault("annotations", {})
            annotations[PREVIOUS_JOB_UID_ANNOTATION] = previous_uid

    def _plan_response(
        self,
        request: JobRequest,
        plan: ResourcePlan,
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        container = _main_container(manifest)
        summary: dict[str, Any] = {
            "name": request.name,
            "namespace": request.namespace,
            "command": list(request.command),
            "image": container.get("image") or request.image,
            "working_dir": request.working_dir,
            "gpu": (
                {"model": plan.gpu.model, "count": plan.gpu.count}
                if plan.gpu
                else None
            ),
            "cpu": plan.compute.cpu_pair,
            "memory": plan.compute.memory_pair,
            "shared_memory": plan.compute.shared_memory,
            "pin_node": request.pin_node,
            "node": plan.node,
            "sizing_node": plan.sizing_node,
            "immediately_schedulable": plan.immediately_schedulable,
            "warning": plan.warning,
            "agent_managed": True,
            "retry": {"backoff_limit": 0, "restart_policy": "Never"},
        }
        result: dict[str, Any] = {
            "job": {"name": request.name, "namespace": request.namespace, "uid": None},
            "name": request.name,
            "namespace": request.namespace,
            "uid": None,
            "summary": summary,
            "resource_plan": {
                "preset": plan.preset,
                "cpu": plan.compute.cpu_pair,
                "memory": plan.compute.memory_pair,
                "shared_memory": plan.compute.shared_memory,
                "gpu": summary["gpu"],
                "node": plan.node,
                "sizing_node": plan.sizing_node,
                "immediately_schedulable": plan.immediately_schedulable,
                "warning": plan.warning,
                **({"scheduling": plan.scheduling} if plan.scheduling is not None else {}),
            },
        }
        # The manifest is deliberately not returned.  MCP callers receive a
        # useful summary without accidentally echoing mounted paths or env.
        return result

    # ------------------------------------------------------------------
    # Submission and idempotency
    # ------------------------------------------------------------------

    def submit_job(self, request: dict, idempotency_key: str) -> dict:
        key = self._idempotency_key(idempotency_key)
        request_digest = self._request_digest(request)
        job_request, plan, manifest = self._prepare_request(
            request,
            force_agent=True,
            generated_name_token=key,
        )
        tag = self._idempotency_tag(key)
        self._force_agent_manifest(manifest, idempotency_tag=tag)
        operation = self._operations.reserve(
            kind="submit",
            key=key,
            request_digest=request_digest,
            planned_name=job_request.name,
            namespace=self.namespace,
        )
        if operation.request_digest != request_digest:
            raise IdempotencyConflict(
                "idempotency key was already used for a different request"
            )
        target_name = operation.planned_name
        manifest = _rename_manifest(manifest, target_name, self.namespace)
        existing = self._get_job_optional(target_name, self.namespace)
        if existing is not None:
            self._verify_idempotent_job(existing, tag, operation)
            identity = _identity_from_job(existing, self.namespace)
            self._operations.mark_created("submit", key, identity["uid"])
            return self._submission_response(
                identity,
                plan,
                created=False,
                reused=True,
            )
        if operation.created_uid:
            # The operation created this exact UID, but the Job is gone now.
            # Reusing the key must never resurrect a same-name replacement.
            return self._deleted_replay_response(
                {
                    "name": target_name,
                    "namespace": operation.namespace,
                    "uid": operation.created_uid,
                },
                plan,
            )
        if plan.scheduling is not None and not plan.scheduling["fits_any_node_capacity"]:
            raise MCPServiceError(
                f"GPU request cannot fit any matching node. {plan.warning}. "
                "Use plan_job's per-node maximum/recommended allocations to revise it; "
                "waiting for availability cannot solve a capacity mismatch."
            )
        try:
            created = self.client.create_job(manifest)
        except Exception:
            # A network timeout may happen after the API server accepted the
            # create. Reconcile the planned name before surfacing the error;
            # importantly, never invent a fresh name for the same key.
            existing = self._get_job_optional(target_name, self.namespace)
            if existing is None:
                raise
            self._verify_idempotent_job(existing, tag, operation)
            identity = _identity_from_job(existing, self.namespace)
            self._operations.mark_created("submit", key, identity["uid"])
            return self._submission_response(
                identity,
                plan,
                created=False,
                reused=True,
            )
        identity = _identity_from_created(created, self.namespace, target_name)
        if not identity["uid"]:
            live = self._get_job_optional(target_name, self.namespace)
            if live is not None:
                identity = _identity_from_job(live, self.namespace)
        if not identity["uid"]:
            raise MCPServiceError(
                "Kubernetes did not return a Job UID; refusing to persist an "
                "unbound idempotency result"
            )
        self._operations.mark_created("submit", key, identity["uid"])
        return self._submission_response(identity, plan, created=True, reused=False)

    @staticmethod
    def _idempotency_key(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise MCPServiceError("idempotency_key must be a non-empty string")
        raw = value.strip()
        if len(raw) > 256:
            raise MCPServiceError("idempotency_key must not exceed 256 characters")
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in raw):
            raise MCPServiceError("idempotency_key contains a control character")
        # Never persist or use the caller's opaque key as a database key.  A
        # coding agent may put a secret or a user identifier in it.
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _request_digest(request: Mapping[str, Any]) -> str:
        try:
            payload = json.dumps(
                request,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                default=str,
            )
        except (TypeError, ValueError) as exc:
            raise MCPServiceError("job request is not JSON-serializable") from exc
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _idempotency_tag(key: str) -> str:
        # Kubernetes label values are limited to 63 characters.
        return key[:63]

    def _verify_idempotent_job(
        self,
        job: Mapping[str, Any],
        tag: str,
        operation: _Operation,
    ) -> None:
        metadata = job.get("metadata") or {}
        labels = metadata.get("labels") or {}
        existing_tag = str(labels.get(IDEMPOTENCY_LABEL) or "")
        if existing_tag != tag:
            if existing_tag:
                raise IdempotencyConflict(
                    f"Job {operation.planned_name!r} is already used by another "
                    "Falcon submission; reuse the original idempotency key if "
                    "this is a retry, or choose another job name"
                )
            raise IdempotencyConflict(
                f"Job {operation.planned_name!r} already exists without the "
                "matching Falcon idempotency marker"
            )
        if operation.created_uid and str(metadata.get("uid") or "") != operation.created_uid:
            raise IdempotencyConflict(
                f"Job {operation.planned_name!r} was replaced after the original "
                "idempotent create; use a new idempotency key"
            )
        if str(metadata.get("namespace") or operation.namespace) != operation.namespace:
            raise StaleJobReference("idempotent Job namespace does not match the operation")
        if not str(metadata.get("uid") or ""):
            raise IdempotencyConflict(
                f"Job {operation.planned_name!r} has no UID; refusing an "
                "unbound idempotent result"
            )

    def _submission_response(
        self,
        identity: Mapping[str, Any],
        plan: ResourcePlan,
        *,
        created: bool,
        reused: bool,
    ) -> dict[str, Any]:
        ref = {
            "name": str(identity.get("name") or ""),
            "namespace": str(identity.get("namespace") or self.namespace),
            "uid": str(identity.get("uid") or ""),
        }
        result = {
            "job": ref,
            "name": ref["name"],
            "namespace": ref["namespace"],
            "uid": ref["uid"],
            "created": bool(created),
            "reused": bool(reused),
            "agent_managed": True,
            "retry": {"backoff_limit": 0, "restart_policy": "Never"},
            "resource_plan": {
                "preset": plan.preset,
                "cpu": plan.compute.cpu_pair,
                "memory": plan.compute.memory_pair,
                "shared_memory": plan.compute.shared_memory,
                "gpu": (
                    {"model": plan.gpu.model, "count": plan.gpu.count}
                    if plan.gpu
                    else None
                ),
                "node": plan.node,
                "sizing_node": plan.sizing_node,
                "warning": plan.warning,
            },
        }
        return result

    def _deleted_replay_response(
        self,
        identity: Mapping[str, Any],
        plan: Optional[ResourcePlan],
        *,
        previous: Optional[Mapping[str, str]] = None,
    ) -> dict[str, Any]:
        """Replay a completed create without resurrecting a deleted name."""

        ref = {
            "name": str(identity.get("name") or ""),
            "namespace": str(identity.get("namespace") or self.namespace),
            "uid": str(identity.get("uid") or ""),
        }
        result: dict[str, Any] = {
            "job": ref,
            "name": ref["name"],
            "namespace": ref["namespace"],
            "uid": ref["uid"],
            "created": False,
            "reused": True,
            "deleted": True,
            "message": (
                "the Job created by this idempotency operation is no longer present; "
                "use a new idempotency key to create a replacement"
            ),
            "agent_managed": True,
            "retry": {"backoff_limit": 0, "restart_policy": "Never"},
        }
        if previous is not None:
            result["previous_job"] = dict(previous)
        if plan is not None:
            result["resource_plan"] = {
                "preset": plan.preset,
                "cpu": plan.compute.cpu_pair,
                "memory": plan.compute.memory_pair,
                "shared_memory": plan.compute.shared_memory,
                "gpu": (
                    {"model": plan.gpu.model, "count": plan.gpu.count}
                    if plan.gpu
                    else None
                ),
                "node": plan.node,
                "sizing_node": plan.sizing_node,
                "warning": plan.warning,
            }
        return result

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_jobs(self, limit: int = 50, status: Optional[str] = None) -> dict:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIST_LIMIT:
            raise MCPServiceError(
                f"limit must be an integer between 1 and {_MAX_LIST_LIMIT}"
            )
        if status is not None and (not isinstance(status, str) or not status.strip()):
            raise MCPServiceError("status must be a non-empty string or null")
        payload = self.client.list_jobs(self.namespace)
        items = list(payload.get("items") or []) if isinstance(payload, Mapping) else []
        jobs: list[tuple[Any, dict[str, Any]]] = []
        wanted = status.casefold() if status else None
        for item in items:
            if not isinstance(item, Mapping):
                continue
            summary = self._job_summary(item)
            if wanted is not None and str(summary["status"]).casefold() != wanted:
                continue
            jobs.append(
                (summary.get("created_at"), _compact_list_summary(item, summary))
            )
        jobs.sort(key=lambda value: str(value[0] or ""), reverse=True)
        compact_jobs = [value for _created_at, value in jobs[:limit]]
        return {
            "jobs": compact_jobs,
            "count": len(compact_jobs),
            "limit": limit,
            "namespace": self.namespace,
        }

    def inspect_job(self, job: dict) -> dict:
        ref, live = self._validated_live_job(job)
        pods = self._job_pods(ref["name"], ref["namespace"], uid=ref["uid"])
        summary = self._job_summary(live, pods)
        result: dict[str, Any] = {
            "job": ref,
            "name": ref["name"],
            "namespace": ref["namespace"],
            "uid": ref["uid"],
            "status": summary["status"],
            "summary": summary,
            "retry": summary["retry"],
            "pods": [_pod_summary(pod) for pod in pods],
        }
        events = self._events_for_job(
            ref["name"],
            ref["namespace"],
            job_uid=ref["uid"],
            pods=pods,
        )
        if events is not None:
            result["events"] = events
        return result

    def _job_summary(
        self,
        job: Mapping[str, Any],
        pods: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> dict[str, Any]:
        metadata = job.get("metadata") or {}
        spec = job.get("spec") or {}
        template = spec.get("template") or {}
        template_spec = template.get("spec") or {}
        containers = template_spec.get("containers") or []
        container = containers[0] if containers and isinstance(containers[0], Mapping) else {}
        if pods is None:
            pods = []
        snapshot = build_job_snapshot(job, list(pods))
        requests = ((container.get("resources") or {}).get("requests") or {})
        limits = ((container.get("resources") or {}).get("limits") or {})
        labels = metadata.get("labels") or {}
        backoff = spec.get("backoffLimit", 6)
        try:
            backoff = int(backoff)
        except (TypeError, ValueError):
            pass
        return {
            "name": str(metadata.get("name") or ""),
            "namespace": str(metadata.get("namespace") or self.namespace),
            "uid": str(metadata.get("uid") or ""),
            "status": snapshot.status,
            "created_at": metadata.get("creationTimestamp"),
            "started_at": (job.get("status") or {}).get("startTime"),
            "completed_at": (job.get("status") or {}).get("completionTime"),
            "active": (job.get("status") or {}).get("active", 0),
            "succeeded": (job.get("status") or {}).get("succeeded", 0),
            "failed": (job.get("status") or {}).get("failed", 0),
            "command": list(snapshot.command),
            "image": str(container.get("image") or ""),
            "resources": {"requests": dict(requests), "limits": dict(limits)},
            "gpu": {
                "model": snapshot.requested.gpu_model,
                "count": snapshot.requested.gpu_count,
            },
            "agent_managed": str(labels.get(AGENT_LABEL) or "").lower() == "true",
            "labels": dict(labels),
            "retry": {
                "backoff_limit": backoff,
                "restart_policy": str(template_spec.get("restartPolicy") or "Never"),
            },
            "attempts": {
                "pods": snapshot.attempts.pod_attempts,
                "failed": snapshot.attempts.failed_attempts,
                "succeeded": snapshot.attempts.succeeded_attempts,
                "container_restarts": snapshot.attempts.container_restarts,
                "active_pod": snapshot.attempts.active_pod,
            },
        }

    @staticmethod
    def _job_terminal_condition(job: Mapping[str, Any]) -> Optional[str]:
        status = job.get("status") or {}
        conditions = status.get("conditions") or []
        for condition in conditions:
            if not isinstance(condition, Mapping):
                continue
            if str(condition.get("status") or "").lower() != "true":
                continue
            condition_type = str(condition.get("type") or "")
            if condition_type == "Complete":
                return "Succeeded"
            if condition_type == "Failed":
                return "Failed"
        return None

    def _events_for_job(
        self,
        name: str,
        namespace: str,
        *,
        job_uid: str,
        pods: Sequence[Mapping[str, Any]],
    ) -> Optional[list[dict[str, Any]]]:
        events = getattr(self.client, "events", None)
        if not callable(events):
            return None
        pod_names = [
            str((pod.get("metadata") or {}).get("name") or "")
            for pod in pods
            if (pod.get("metadata") or {}).get("name")
        ]
        pod_uids = {
            str((pod.get("metadata") or {}).get("uid") or "")
            for pod in pods
            if (pod.get("metadata") or {}).get("uid")
        }
        allowed_uids = {job_uid, *pod_uids}
        values = events(
            [name, *pod_names],
            namespace=namespace,
            limit=50,
        )
        compacted = []
        for value in values or []:
            if not isinstance(value, Mapping):
                continue
            involved = value.get("involvedObject") or value.get("regarding")
            if not isinstance(involved, Mapping):
                continue
            involved_uid = str(involved.get("uid") or "")
            if involved_uid not in allowed_uids:
                continue
            compacted.append(_compact_inspect_event(value, involved))
        return compacted

    def _validated_live_job(self, job: Mapping[str, Any]) -> tuple[dict[str, str], Mapping[str, Any]]:
        ref = self._job_ref(job)
        live = self._get_job(ref["name"], ref["namespace"])
        metadata = live.get("metadata") or {}
        live_uid = str(metadata.get("uid") or "")
        if not live_uid:
            raise StaleJobReference("Kubernetes Job has no UID; refusing an unbound reference")
        if live_uid != ref["uid"]:
            raise StaleJobReference(
                f"Job {ref['name']!r} UID changed; refresh the Job reference"
            )
        live_namespace = str(metadata.get("namespace") or ref["namespace"])
        if live_namespace != ref["namespace"]:
            raise StaleJobReference("Job namespace changed; refresh the Job reference")
        return ref, live

    def _job_ref(self, job: Mapping[str, Any]) -> dict[str, str]:
        if not isinstance(job, Mapping):
            raise MCPServiceError("job reference must be an object")
        source: Mapping[str, Any] = job
        if "metadata" in job and "name" not in job:
            metadata = job.get("metadata")
            if isinstance(metadata, Mapping):
                source = metadata
        name = source.get("name")
        namespace = source.get("namespace", self.namespace)
        uid = source.get("uid")
        if not isinstance(name, str) or not name.strip():
            raise MCPServiceError("job reference name must not be empty")
        if not isinstance(namespace, str) or not namespace.strip():
            raise MCPServiceError("job reference namespace must not be empty")
        if namespace != self.namespace:
            raise MCPServiceError(
                f"job reference namespace {namespace!r} does not match configured "
                f"namespace {self.namespace!r}"
            )
        if uid is None:
            live = self._get_job(name.strip(), self.namespace)
            metadata = live.get("metadata") or {}
            if (metadata.get("name") != name.strip()
                    or str(metadata.get("namespace") or self.namespace) != self.namespace):
                raise StaleJobReference("Kubernetes returned a different Job identity")
            uid = metadata.get("uid")
        if not isinstance(uid, str) or not uid.strip():
            raise MCPServiceError("job reference UID must not be empty")
        return {"name": name.strip(), "namespace": namespace.strip(), "uid": uid.strip()}

    def _get_job(self, name: str, namespace: str) -> Mapping[str, Any]:
        value = self.client.get_json("job.batch", name, namespace=namespace)
        if not isinstance(value, Mapping):
            raise MCPServiceError("Kubernetes returned a malformed Job")
        return value

    def _get_job_optional(self, name: str, namespace: str) -> Optional[Mapping[str, Any]]:
        try:
            return self._get_job(name, namespace)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise

    def _job_pods(
        self,
        name: str,
        namespace: str,
        *,
        uid: Optional[str] = None,
    ) -> list[Mapping[str, Any]]:
        values = self.client.job_pods(name, namespace=namespace)
        return [
            value
            for value in (values or [])
            if isinstance(value, Mapping) and _pod_matches_job(value, name, uid)
        ]

    # ------------------------------------------------------------------
    # Logs and metrics
    # ------------------------------------------------------------------

    def job_logs(
        self,
        job: dict,
        pods: Optional[list[str]] = None,
        failed_only: bool = False,
        limit: int = _DEFAULT_LOG_LIMIT,
        tail: int = _DEFAULT_LOG_TAIL,
        cursors: Optional[dict[str, str]] = None,
        container: Optional[str] = None,
    ) -> dict:
        ref = self._job_ref(job)
        archived = self._restart_history.logs(ref["uid"])
        if archived is not None and archived["name"] != ref["name"]:
            raise StaleJobReference("Archived Job UID does not match its name")
        if archived is None and not self._operations.restarted_source(ref):
            self._validated_live_job(ref)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LOG_LIMIT:
            raise MCPServiceError(f"limit must be between 1 and {_MAX_LOG_LIMIT}")
        if isinstance(tail, bool) or not isinstance(tail, int) or not 0 <= tail <= _MAX_LOG_TAIL:
            raise MCPServiceError(f"tail must be between 0 and {_MAX_LOG_TAIL}")
        if container is not None and (not isinstance(container, str) or not container.strip()):
            raise MCPServiceError("container must be a non-empty string or null")
        cursor_map = self._validate_cursors(cursors)
        if archived is not None:
            entries = archived["logs"]
            if pods is not None:
                if not isinstance(pods, list) or any(not isinstance(p, str) or not p.strip() for p in pods):
                    raise MCPServiceError("pods must be an array of non-empty Pod names")
                if set(pods) - {entry["pod"] for entry in entries}:
                    raise MCPServiceError("Pod is not an archived attempt of this Job")
                entries = [entry for entry in entries if entry["pod"] in pods]
            if failed_only:
                entries = [entry for entry in entries if entry["phase"] == "Failed"]
            if container:
                entries = [entry for entry in entries if entry["container"] == container]
            result = []
            next_cursors = {}
            for entry in entries[-limit:]:
                key = f"{entry['pod_uid']}/{entry['container']}"
                raw = cursor_map.get(key)
                decoded = _decode_log_cursor(raw) if raw else None
                if decoded is not None and (decoded["pod_uid"] != entry["pod_uid"] or decoded["container"] != entry["container"]):
                    raise MCPServiceError("cursor belongs to a different Pod or container")
                stdout, next_cursor, state = _incremental_log_text(
                    entry["stdout"], pod_uid=entry["pod_uid"], container=entry["container"],
                    cursor=decoded, max_chars=_MAX_LOG_CHARS_PER_POD,
                    max_bytes=_MAX_LOG_BYTES_PER_POD, tail=tail,
                )
                if next_cursor or raw:
                    next_cursors[key] = next_cursor or raw
                stderr, stderr_truncated = _bound_text(entry["stderr"], 4096)
                result.append({**entry, "stdout": stdout, "stderr": stderr,
                               "stderr_truncated": stderr_truncated, "stdout_truncated": bool(state.get("truncated")),
                               "cursor_key": key, "cursor": raw, "next_cursor": next_cursor or raw, **state})
            return _bound_job_logs_response({"job": ref, "logs": result, "cursors": next_cursors,
                                            "count": len(result), "failed_only": failed_only,
                                            "archived": True, "summary": archived["summary"],
                                            "truncated": False}, _MAX_LOG_RESPONSE_BYTES)
        all_pods = self._job_pods(ref["name"], ref["namespace"], uid=ref["uid"])
        by_name = {
            str((value.get("metadata") or {}).get("name") or ""): value
            for value in all_pods
        }
        if pods is not None:
            if not isinstance(pods, list):
                raise MCPServiceError("pods must be an array of Pod names or null")
            pod_names = []
            seen_pods: set[str] = set()
            for pod_name in pods:
                if not isinstance(pod_name, str) or not pod_name.strip():
                    raise MCPServiceError("Pod names must not be empty")
                clean = pod_name.strip()
                if clean not in by_name:
                    raise MCPServiceError(
                        f"Pod {clean!r} is not an attempt of Job {ref['name']!r}"
                    )
                if clean in seen_pods:
                    continue
                seen_pods.add(clean)
                pod_names.append(clean)
            if len(pod_names) > _MAX_LOG_PODS:
                raise MCPServiceError(
                    f"pods must contain at most {_MAX_LOG_PODS} distinct Pod names"
                )
            pod_names = pod_names[:limit]
        else:
            selected = [
                value
                for value in all_pods
                if not failed_only or _pod_phase(value) == "Failed"
            ]
            selected.sort(
                key=lambda value: str(
                    (value.get("metadata") or {}).get("creationTimestamp") or ""
                )
            )
            if not failed_only:
                running = [value for value in selected if _pod_phase(value) == "Running"]
                selected = running[-1:] if running else selected[-1:]
            else:
                selected = selected[-limit:]
            pod_names = [
                str((value.get("metadata") or {}).get("name") or "")
                for value in selected
                if (value.get("metadata") or {}).get("name")
            ]
        entries = []
        next_cursors: dict[str, str] = {}
        for pod_name in pod_names:
            pod = by_name[pod_name]
            phase = _pod_phase(pod)
            if failed_only and phase != "Failed":
                continue
            metadata = pod.get("metadata") or {}
            pod_uid = str(metadata.get("uid") or pod_name)
            container_name = container or "main"
            cursor_key = f"{pod_uid}/{container_name}"
            raw_cursor = cursor_map.get(cursor_key)
            decoded_cursor = (
                _decode_log_cursor(raw_cursor)
                if raw_cursor is not None
                else None
            )
            if decoded_cursor is not None and (
                decoded_cursor["pod_uid"] != pod_uid
                or decoded_cursor["container"] != container_name
            ):
                raise MCPServiceError(
                    f"cursor {cursor_key!r} belongs to a different Pod or container"
                )
            result = self._pod_logs(
                pod_name,
                ref["namespace"],
                tail=tail,
                container=container_name,
                timestamps=True,
                since_time=(
                    decoded_cursor["timestamp"] if decoded_cursor is not None else None
                ),
            )
            stdout, stderr, returncode = _process_output(result)
            visible, next_cursor, log_state = _incremental_log_text(
                stdout,
                pod_uid=pod_uid,
                container=container_name,
                cursor=decoded_cursor,
                max_chars=_MAX_LOG_CHARS_PER_POD,
                max_bytes=_MAX_LOG_BYTES_PER_POD,
                tail=tail,
            )
            if next_cursor is not None:
                next_cursors[cursor_key] = next_cursor
            elif raw_cursor is not None:
                # A log response with no new lines still needs to carry the
                # caller's opaque cursor, otherwise the next poll replays the
                # entire log stream.
                next_cursors[cursor_key] = raw_cursor
            bounded_stderr, stderr_truncated = _bound_text(stderr, 4_096)
            container_details = _pod_container_details(pod, container_name)
            entries.append(
                {
                    "pod": pod_name,
                    "pod_uid": pod_uid,
                    "phase": phase,
                    "reason": container_details["reason"],
                    "container_exit_code": container_details["exit_code"],
                    "container_reason": container_details["container_reason"],
                    "created_at": metadata.get("creationTimestamp"),
                    "container": container_name,
                    "cursor_key": cursor_key,
                    "stdout": visible,
                    "stderr": bounded_stderr,
                    # This is the kubectl process status, not the workload's
                    # terminated container exit code above.
                    "returncode": returncode,
                    "kubectl_returncode": returncode,
                    "cursor": raw_cursor,
                    "next_cursor": next_cursor or raw_cursor,
                    **log_state,
                    "stdout_truncated": bool(log_state.get("truncated")),
                    "stderr_truncated": stderr_truncated,
                }
            )
        response = {
            "job": ref,
            "logs": entries,
            "cursors": next_cursors,
            "count": len(entries),
            "failed_only": bool(failed_only),
            "truncated": False,
        }
        return _bound_job_logs_response(response, _MAX_LOG_RESPONSE_BYTES)

    @staticmethod
    def _validate_cursors(cursors: Optional[dict[str, str]]) -> dict[str, str]:
        if cursors is None:
            return {}
        if not isinstance(cursors, Mapping):
            raise MCPServiceError("cursors must be an object or null")
        result: dict[str, str] = {}
        for key, value in cursors.items():
            if not isinstance(key, str) or not key.strip():
                raise MCPServiceError("cursor keys must be non-empty Pod UID/container keys")
            if not isinstance(value, str) or not value.strip():
                raise MCPServiceError("cursor values must be opaque non-empty strings")
            result[key] = value.strip()
        return result

    def _pod_logs(
        self,
        pod_name: str,
        namespace: str,
        *,
        tail: int,
        container: Optional[str],
        timestamps: bool,
        since_time: Optional[str],
    ) -> Any:
        return self.client.pod_logs(
            pod_name,
            namespace=namespace,
            tail=tail,
            container=container,
            timestamps=timestamps,
            since_time=since_time,
        )

    def job_metrics(self, job: dict) -> dict:
        ref, _live = self._validated_live_job(job)
        try:
            key = (ref["namespace"], ref["name"], ref["uid"])
            with self._metrics_lock:
                if self._closed:
                    raise MCPServiceError("JobService is closed")
                collector = self._collector_for(ref)
                cached = self._metrics_snapshots.get(key)
                now = time.monotonic()
                if cached is not None and now - cached[0] < 1.0:
                    rows = cached[1]
                    risk_policy = self._risk_snapshots.get(key)
                else:
                    # UsageCollector owns mutable EMA/streaming sampler state,
                    # while the MCP evaluator owns eviction-risk state. Keep
                    # both under this lock so a monitor callback cannot race a
                    # metrics tool call or advance risk without a new sample.
                    rows = collector.collect()
                    row = _select_metric_row(rows, ref["name"])
                    risk_policy = self._evaluate_mcp_risk(ref, row)
                    self._metrics_snapshots[key] = (time.monotonic(), rows)
                    self._risk_snapshots[key] = risk_policy
            row = _select_metric_row(rows, ref["name"])
            if row is None:
                return {
                    "job": ref,
                    "available": False,
                    "metrics": None,
                    "at_risk": None,
                    "error": str(getattr(collector, "last_error", "") or "") or None,
                }
            metrics = _metric_mapping(row, risk_policy=risk_policy)
            return {
                "job": ref,
                "available": True,
                "metrics": metrics,
                "at_risk": metrics.get("at_risk"),
                "error": str(getattr(collector, "last_error", "") or "") or None,
            }
        except Exception as exc:
            # Metrics are observational. A telemetry outage must not turn a
            # valid Job reference into a destructive or retrying operation.
            return {
                "job": ref,
                "available": False,
                "metrics": None,
                "at_risk": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _evaluate_mcp_risk(
        self,
        ref: Mapping[str, str],
        row: Any,
    ) -> Optional[dict[str, Any]]:
        """Evaluate MCP-only risk from the row's physical GPU samples."""

        if row is None:
            return None

        def value(name: str, default: Any = None) -> Any:
            if isinstance(row, Mapping):
                return row.get(name, default)
            return getattr(row, name, default)

        devices = value("gpu_devices", [])
        if not devices:
            # Older/custom collectors may expose only aggregate metrics. Keep
            # their compatibility output, but do not claim per-GPU policy
            # without physical-device samples.
            return None
        key = (ref["namespace"], ref["name"], ref["uid"])
        evaluator = self._risk_evaluators.get(key)
        startup_time = self._monitor_startup_times.get(
            ref["uid"], DEFAULT_EVICTION_STARTUP_SECONDS
        )
        if evaluator is None:
            evaluator = McpGpuRiskEvaluator(startup_time)
            self._risk_evaluators[key] = evaluator
        else:
            evaluator.set_startup_time_seconds(startup_time)
        gpu_type = (
            value("gpu_requested_type")
            or value("gpu_allocated_type")
            or value("gpu_type")
            or ""
        )
        thresholds = {
            canonical_gpu(str(item.get("gpu_type"))): float(
                item.get("minimum_utilization", 30)
            )
            for item in (self.config.get("presets") or {}).values()
            if isinstance(item, Mapping) and item.get("gpu_type")
        }
        threshold = thresholds.get(canonical_gpu(str(gpu_type)), 30.0)
        return evaluator.evaluate(ref["uid"], devices, threshold)

    def _collector_for(self, ref: Mapping[str, str]) -> Any:
        if self._closed:
            raise MCPServiceError("JobService is closed")
        key = (ref["namespace"], ref["name"], ref["uid"])
        with self._metrics_lock:
            for old_key in list(self._metrics_collectors):
                if old_key[:2] == key[:2] and old_key != key:
                    old = self._metrics_collectors.pop(old_key)
                    self._metrics_snapshots.pop(old_key, None)
                    self._risk_evaluators.pop(old_key, None)
                    self._risk_snapshots.pop(old_key, None)
                    close = getattr(old, "close", None)
                    if callable(close):
                        close()
            collector = self._metrics_collectors.get(key)
            if collector is not None:
                return collector
            thresholds = {
                canonical_gpu(str(value.get("gpu_type"))): float(
                    value.get("minimum_utilization", 30)
                )
                for value in (self.config.get("presets") or {}).values()
                if isinstance(value, Mapping) and value.get("gpu_type")
            }
            dashboard = self.config.get("dashboard") or {}
            kwargs = {
                "job_filter": ref["name"],
                "streaming_gpu": True,
                "collect_availability": False,
                "context": self.context,
            }
            collector = UsageCollector(
                ref["namespace"],
                thresholds,
                float(dashboard.get("ema_alpha", 0.1)),
                **kwargs,
            )
            self._metrics_collectors[key] = collector
            return collector

    def _monitor_metrics(self, names: Any) -> Any:
        """Metrics callback shape consumed by :class:`mcp_events.Monitor`."""

        if isinstance(names, str):
            requested = [names]
            scalar = True
        else:
            requested = list(names or [])
            scalar = False
        values: dict[str, dict[str, Any]] = {}
        for identifier in requested:
            if not isinstance(identifier, str) or not identifier:
                continue
            try:
                ref = self._monitor_refs.get(identifier)
                if ref is None:
                    live = self._get_job(identifier, self.namespace)
                    metadata = live.get("metadata") or {}
                    ref = {
                        "name": identifier,
                        "namespace": self.namespace,
                        "uid": str(metadata.get("uid") or ""),
                    }
                if not ref["uid"]:
                    raise MCPServiceError("Job has no UID")
                values[identifier] = self.job_metrics(ref)
            except Exception as exc:
                values[identifier] = {
                    "available": False,
                    "metrics": None,
                    "at_risk": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return values[requested[0]] if scalar and requested else values

    # ------------------------------------------------------------------
    # Cancellation and safe restart
    # ------------------------------------------------------------------

    def cancel_job(self, job: dict) -> dict:
        ref, live = self._validated_live_job(job)
        self._reject_coder_job(ref, live)
        self.client.delete_job_uid(
            ref["name"],
            ref["uid"],
            namespace=ref["namespace"],
        )
        return {
            "job": ref,
            "name": ref["name"],
            "namespace": ref["namespace"],
            "uid": ref["uid"],
            "cancelled": True,
        }

    def restart_job(
        self,
        job: dict,
        idempotency_key: Optional[str] = None,
        overrides: Optional[dict] = None,
        summary: Optional[dict] = None,
    ) -> dict:
        """Replace one terminal Job under its original name, with a new UID.

        ``overrides["env"]`` is a patch by variable name: entries not named
        in the patch, including ``valueFrom`` references such as secrets, are
        retained exactly; named entries are replaced with plain scalar values.
        Changing the Python environment itself is intentionally unsupported.
        """
        # Validate the reference shape and operation identity before looking
        # up the source Job. A successful restart must remain replayable even
        # after a subsequent restart has reused the same name again.
        ref = self._job_ref(job)
        if summary is not None:
            if (not isinstance(summary, dict) or set(summary) != {"issue", "fix"}
                    or any(not isinstance(v, str) or not v.strip() or len(v) > 1000
                           for v in summary.values())):
                raise MCPServiceError("summary requires non-empty issue and fix strings, each at most 1000 characters")
            summary = {k: v.strip() for k, v in summary.items()}
        normalized_overrides = self._restart_overrides(overrides)
        if idempotency_key is None:
            # One source UID can be replaced only once. Repeated requests
            # resume/replay it, including after a transport/process failure.
            key = self._operations.restart_key(ref) or self._idempotency_key(
                "restart-" + hashlib.sha256(
                    json.dumps(ref, sort_keys=True).encode("utf-8")
                ).hexdigest()
            )
        else:
            key = self._idempotency_key(idempotency_key)
        digest = self._restart_digest(ref, normalized_overrides, summary)
        operation = self._operations.get("restart", key)
        if operation is not None and operation.request_digest != digest:
            raise IdempotencyConflict(
                "a different restart is already recorded; retry with the "
                "original Job reference, overrides and summary"
            )
        planned_name = (
            operation.planned_name
            if operation is not None
            else ref["name"]
        )
        tag = self._idempotency_tag(key)
        if operation is not None:
            if operation.created_uid:
                return self._completed_restart(ref, operation)
            existing = self._get_job_optional(planned_name, ref["namespace"])
            if existing is not None and (
                planned_name != ref["name"]
                or str((existing.get("metadata") or {}).get("uid") or "") != ref["uid"]
            ):
                self._verify_idempotent_job(existing, tag, operation)
                identity = _identity_from_job(existing, ref["namespace"])
                self._operations.mark_created("restart", key, identity["uid"])
                self._clear_restart_checkpoint(key)
                return self._restart_response(ref, identity, created=False, reused=True)

        live = self._get_job_optional(ref["name"], ref["namespace"])
        if live is not None:
            self._validate_restart_source(ref, live)
            manifest = _restart_manifest(
                live,
                namespace=ref["namespace"],
                name=planned_name,
                previous_uid=ref["uid"],
                overrides=normalized_overrides,
                config=self.config,
            )
            self._force_agent_manifest(
                manifest, idempotency_tag=tag, previous_uid=ref["uid"],
            )
            root_uid = str((live.get("metadata", {}).get("annotations") or {}).get(HISTORY_ANNOTATION) or ref["uid"])
            manifest["metadata"].setdefault("annotations", {})[HISTORY_ANNOTATION] = root_uid
            operation = self._operations.reserve(
                kind="restart", key=key, request_digest=digest,
                planned_name=planned_name, namespace=ref["namespace"],
                source_uid=ref["uid"],
            )
            if operation.request_digest != digest:
                raise IdempotencyConflict(
                    "a different restart is already recorded; retry with the "
                    "original Job reference and overrides"
                )
            if operation.created_uid:
                return self._completed_restart(ref, operation)
            manifest = self._restart_checkpoint(operation, manifest)
            self._archive_restart(ref, root_uid, summary)
        elif operation is not None:
            # The source was deleted but create (or the MCP process) failed.
            # Recover the exact prepared manifest, not a fresh submission.
            manifest = self._restart_checkpoint(operation)
        else:
            raise StaleJobReference(f"Job {ref['name']!r} no longer exists")

        if operation.planned_name == ref["name"]:
            existing = self._remove_restart_source(ref)
        else:
            # Retain reconciliation for operations recorded by older Falcon
            # versions which generated a different replacement name.
            existing = self._get_job_optional(operation.planned_name, ref["namespace"])
        if existing is not None:
            self._verify_idempotent_job(existing, tag, operation)
            identity = _identity_from_job(existing, ref["namespace"])
            self._operations.mark_created("restart", key, identity["uid"])
            self._clear_restart_checkpoint(key)
            return self._restart_response(ref, identity, created=False, reused=True)
        try:
            created = self.client.create_job(manifest)
        except Exception:
            existing = self._get_job_optional(operation.planned_name, ref["namespace"])
            if existing is None:
                raise
            self._verify_idempotent_job(existing, tag, operation)
            identity = _identity_from_job(existing, ref["namespace"])
            self._operations.mark_created("restart", key, identity["uid"])
            self._clear_restart_checkpoint(key)
            return self._restart_response(ref, identity, created=False, reused=True)
        identity = _identity_from_created(created, ref["namespace"], operation.planned_name)
        if not identity["uid"]:
            live_new = self._get_job_optional(operation.planned_name, ref["namespace"])
            if live_new is not None:
                self._verify_idempotent_job(live_new, tag, operation)
                identity = _identity_from_job(live_new, ref["namespace"])
        if not identity["uid"]:
            raise MCPServiceError(
                "Kubernetes did not return a restarted Job UID; refusing to "
                "persist an unbound idempotency result"
            )
        self._operations.mark_created("restart", key, identity["uid"])
        self._clear_restart_checkpoint(key)
        return self._restart_response(ref, identity, created=True, reused=False)

    def _archive_restart(self, ref, root_uid, summary):
        if self._restart_history.contains(ref["uid"]):
            return
        entries = []
        for pod in self._job_pods(ref["name"], ref["namespace"], uid=ref["uid"]):
            metadata = pod.get("metadata") or {}
            spec = pod.get("spec") or {}
            for container in (spec.get("initContainers") or []) + (spec.get("containers") or []):
                name = str(container.get("name") or "main")
                statuses = (pod.get("status", {}).get("initContainerStatuses") or []) + (pod.get("status", {}).get("containerStatuses") or [])
                waiting = next((status.get("state", {}).get("waiting") for status in statuses
                                if status.get("name") == name and not status.get("restartCount")), None)
                if waiting is not None:
                    stdout, stderr, code = "", f"Container did not start: {waiting.get('reason', 'waiting')}", 0
                else:
                    result = self._pod_logs(metadata["name"], ref["namespace"], tail=-1,
                                            container=name, timestamps=True, since_time=None)
                    stdout, stderr, code = _process_output(result)
                evicted = (pod.get("status") or {}).get("reason") == "Evicted"
                if code and not evicted and _pod_phase(pod) in {"Failed", "Succeeded", "Running"}:
                    raise MCPServiceError(f"Could not archive {metadata['name']}/{name}; restart stopped before deletion: {stderr}")
                entries.append({
                    "pod": metadata["name"], "pod_uid": metadata.get("uid") or metadata["name"],
                    "container": name, "phase": _pod_phase(pod),
                    "created_at": metadata.get("creationTimestamp") or "",
                    "stdout": stdout, "stderr": stderr, "returncode": code,
                    "archived": True, "unavailable": bool(code),
                })
        self._restart_history.save(ref, root_uid, summary, entries)

    def _completed_restart(self, ref: dict[str, str], operation: _Operation) -> dict:
        self._clear_restart_checkpoint(operation.key)
        return self._restart_response(
            ref,
            {"name": operation.planned_name, "namespace": operation.namespace,
             "uid": operation.created_uid},
            created=False, reused=True,
        )

    def _validate_restart_source(self, ref: dict[str, str], live: Mapping[str, Any]) -> None:
        if _identity_from_job(live, ref["namespace"]) != ref:
            raise StaleJobReference(
                f"Job {ref['name']!r} UID changed; refresh the Job reference"
            )
        self._reject_coder_job(ref, live)
        if self._job_terminal_condition(live) is None:
            raise MCPServiceError(
                f"Job {ref['name']!r} has no Kubernetes terminal Complete=True or "
                "Failed=True condition; wait for the Job controller before restarting"
            )
        if any(
            _pod_phase(pod) not in {"Failed", "Succeeded"}
            for pod in self._job_pods(ref["name"], ref["namespace"], uid=ref["uid"])
        ):
            raise MCPServiceError("wait for all source Job Pods to terminate before restarting")

    def _remove_restart_source(self, ref: dict[str, str]) -> Optional[Mapping[str, Any]]:
        live = self._get_job_optional(ref["name"], ref["namespace"])
        if live is None or str((live.get("metadata") or {}).get("uid") or "") != ref["uid"]:
            return live
        self._validate_restart_source(ref, live)
        # Delete attempts explicitly as well as cascading the Job: this also
        # recovers interrupted legacy Orphan deletes without orphaning Pods.
        for pod in self._job_pods(ref["name"], ref["namespace"], uid=ref["uid"]):
            metadata = pod.get("metadata") or {}
            try:
                self.client.delete_pod_uid(
                    str(metadata.get("name") or ""), str(metadata.get("uid") or ""),
                    namespace=ref["namespace"],
                )
            except Exception as exc:
                if not _is_not_found(exc):
                    raise
        live = self._get_job_optional(ref["name"], ref["namespace"])
        if live is None or str((live.get("metadata") or {}).get("uid") or "") != ref["uid"]:
            return live
        if (live.get("metadata") or {}).get("deletionTimestamp"):
            try:
                self.client.finish_orphaned_job_deletion(live, namespace=ref["namespace"])
            except Exception:
                current = self._get_job_optional(ref["name"], ref["namespace"])
                if current is None or str((current.get("metadata") or {}).get("uid") or "") != ref["uid"]:
                    return current
                raise
            live = self._get_job_optional(ref["name"], ref["namespace"])
            if live is None or str((live.get("metadata") or {}).get("uid") or "") != ref["uid"]:
                return live
        try:
            self.client.delete_job_uid(
                ref["name"], ref["uid"], namespace=ref["namespace"],
                propagation_policy="Background",
            )
        except Exception:
            # Another retry may have completed the same UID-preconditioned
            # deletion. Never fall back to a name-only delete.
            current = self._get_job_optional(ref["name"], ref["namespace"])
            if current is not None and str((current.get("metadata") or {}).get("uid") or "") == ref["uid"]:
                raise
            return current
        deadline = time.monotonic() + _RESTART_DELETE_TIMEOUT_SECONDS
        while True:
            current = self._get_job_optional(ref["name"], ref["namespace"])
            if current is None or str((current.get("metadata") or {}).get("uid") or "") != ref["uid"]:
                return current
            if time.monotonic() >= deadline:
                raise MCPServiceError(
                    "source Job deletion is still pending; retry restart_job with "
                    "the same Job reference and arguments to finish the restart"
                )
            time.sleep(0.1)

    def _restart_checkpoint(self, operation: _Operation, manifest: Optional[dict] = None) -> dict:
        """Durably bridge delete/create without persisting manifests in SQLite.

        A private, temporary checkpoint survives interrupted restarts and is
        removed once the replacement UID has been durably recorded.
        """
        path = self._state_dir / f"pending-restart-{operation.key}.json"
        if manifest is not None:
            descriptor, temporary = tempfile.mkstemp(prefix=".restart-", dir=self._state_dir)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump({"digest": operation.request_digest, "manifest": manifest}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    # Publish once: concurrent retries must share the first
                    # prepared manifest rather than overwrite each other.
                    os.link(temporary, path)
                except FileExistsError:
                    pass
                directory = os.open(self._state_dir, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                os.unlink(temporary)
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
                    raise MCPServiceError("restart checkpoint must be a private regular file")
                payload = json.load(stream)
            value = payload["manifest"]
            metadata = value["metadata"]
            if (
                payload["digest"] != operation.request_digest
                or metadata["name"] != operation.planned_name
                or metadata["namespace"] != operation.namespace
                or metadata["annotations"][PREVIOUS_JOB_UID_ANNOTATION] != operation.source_uid
                or metadata["labels"][IDEMPOTENCY_LABEL] != self._idempotency_tag(operation.key)
            ):
                raise MCPServiceError("restart checkpoint does not match the operation")
            return value
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise MCPServiceError(
                "cannot recover restart checkpoint; source Job was not modified "
                "by this attempt"
            ) from exc

    def _clear_restart_checkpoint(self, key: str) -> None:
        (self._state_dir / f"pending-restart-{key}.json").unlink(missing_ok=True)

    @staticmethod
    def _restart_digest(ref: Mapping[str, str], overrides: Mapping[str, Any], summary=None) -> str:
        payload = json.dumps(
            {"source_uid": ref["uid"], "overrides": overrides, **({"summary": summary} if summary is not None else {})},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _restart_overrides(self, overrides: Optional[dict]) -> dict[str, Any]:
        """Validate typed relaunch fields and normalize an env-name patch.

        The ``env`` mapping is deliberately a patch rather than a complete
        replacement.  That lets a caller change one value without discarding
        inherited ``valueFrom`` secret/config-map references or runtime
        variables such as ``PATH`` and ``HOME``.
        """
        if overrides is None:
            return {}
        if not isinstance(overrides, Mapping):
            raise MCPServiceError("restart overrides must be an object or null")
        unsupported = sorted(set(overrides) - _RESTART_FIELDS)
        if "environment" in overrides:
            raise MCPServiceError(
                "changing the Python environment during restart is unsupported; "
                "submit a new Job so its environment mounts can be planned safely"
            )
        retry_fields = sorted(set(overrides) & _RETRY_FIELDS)
        if retry_fields:
            raise MCPServiceError(
                "restart overrides cannot change retry settings; "
                "Falcon enforces backoffLimit=0 and restartPolicy=Never"
            )
        if unsupported:
            raise MCPServiceError(
                "unsupported restart override(s): " + ", ".join(unsupported)
            )
        result = dict(overrides)
        if "command" in result:
            command = result["command"]
            if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
                raise MCPServiceError("restart command must be a non-empty argv array")
            if not command or any(
                not isinstance(value, str) or not value or "\0" in value
                for value in command
            ):
                raise MCPServiceError("restart command must be a non-empty argv array")
            result["command"] = list(command)
        for field in ("cpu", "memory", "image", "working_dir"):
            if field in result:
                if not isinstance(result[field], str) or not result[field].strip():
                    raise MCPServiceError(f"restart {field} must be a non-empty string")
                result[field] = result[field].strip()
        if "memory" in result:
            result["memory"] = self._optional_memory(result["memory"], "memory")
        if "env" in result:
            if not isinstance(result["env"], Mapping):
                raise MCPServiceError("restart env must be a mapping")
            env_patch: dict[str, Any] = {}
            for key, value in result["env"].items():
                if not isinstance(key, str) or not _ENV_NAME.fullmatch(key):
                    raise MCPServiceError(
                        f"restart env contains an invalid variable name: {key!r}"
                    )
                if value is None or isinstance(value, (Mapping, list, tuple, set)):
                    raise MCPServiceError(
                        f"restart env value for {key!r} must be scalar"
                    )
                env_patch[key] = value
            result["env"] = env_patch
        if "gpu" in result and not isinstance(result["gpu"], (str, Mapping)):
            raise MCPServiceError("restart gpu must be a preset or model")
        if "gpus" in result:
            count = result["gpus"]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise MCPServiceError("restart gpus must be a positive integer")
        if "shared_memory" in result:
            result["shared_memory"] = self._optional_memory(
                result["shared_memory"], "shared_memory"
            )
        if "shared_memory_percent" in result:
            value = result["shared_memory_percent"]
            if isinstance(value, bool):
                raise MCPServiceError("restart shared_memory_percent must be numeric")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise MCPServiceError("restart shared_memory_percent must be numeric") from exc
            if not 0 < value <= 100:
                raise MCPServiceError(
                    "restart shared_memory_percent must be greater than 0 and at most 100"
                )
            result["shared_memory_percent"] = value
        return result

    @staticmethod
    def _restart_response(
        previous: Mapping[str, str],
        identity: Mapping[str, Any],
        *,
        created: bool,
        reused: bool,
    ) -> dict[str, Any]:
        ref = {
            "name": str(identity.get("name") or ""),
            "namespace": str(identity.get("namespace") or previous["namespace"]),
            "uid": str(identity.get("uid") or ""),
        }
        return {
            "job": ref,
            "name": ref["name"],
            "namespace": ref["namespace"],
            "uid": ref["uid"],
            "previous_job": dict(previous),
            "created": bool(created),
            "reused": bool(reused),
            "agent_managed": True,
            "link_annotation": PREVIOUS_JOB_UID_ANNOTATION,
            "retry": {"backoff_limit": 0, "restart_policy": "Never"},
        }

    @staticmethod
    def _reject_coder_job(ref: Mapping[str, str], job: Mapping[str, Any]) -> None:
        metadata = job.get("metadata") or {}
        labels = metadata.get("labels") or {}
        name = ref["name"].lower()
        coder_label = any(
            str(key).lower().startswith("coder")
            or str(value).lower() == "coder"
            for key, value in labels.items()
        )
        if name.startswith("coder-") or coder_label:
            raise CoderControlledJob(
                f"Job {ref['name']!r} is Coder-controlled; use Coder's lifecycle "
                "operations instead of bypassing its controller"
            )

    # ------------------------------------------------------------------
    # Events and lifecycle
    # ------------------------------------------------------------------

    async def wait_for_events(
        self,
        jobs: list[dict],
        cursor: Optional[int] = None,
        event_types: Optional[list[str]] = None,
        timeout: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
        startup_time_seconds: Optional[float] = None,
        scheduling_policy: str = "report_blocked",
    ) -> dict:
        if not isinstance(jobs, list):
            raise MCPServiceError("jobs must be an array of Job references")
        if scheduling_policy not in ("report_blocked", "wait_for_capacity"):
            raise MCPServiceError("scheduling_policy must be report_blocked or wait_for_capacity")
        # Resolving name-only references performs Kubernetes reads. Keep them
        # off the event loop, then bind this wait to the resolved UIDs.
        refs = await asyncio.to_thread(lambda: [self._job_ref(value) for value in jobs])
        if cursor is not None and (
            isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0
        ):
            raise MCPServiceError("cursor must be a non-negative integer or null")
        if event_types is not None:
            if not isinstance(event_types, list) or any(
                not isinstance(value, str) or not value.strip() for value in event_types
            ):
                raise MCPServiceError("event_types must contain non-empty strings")
            event_types = [value.strip() for value in event_types]
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout < 0
        ):
            raise MCPServiceError("timeout must be a finite non-negative number")
        if startup_time_seconds is not None:
            if (
                isinstance(startup_time_seconds, bool)
                or not isinstance(startup_time_seconds, (int, float))
                or not math.isfinite(float(startup_time_seconds))
                or not MIN_EVICTION_STARTUP_SECONDS
                <= float(startup_time_seconds)
                <= MAX_EVICTION_STARTUP_SECONDS
            ):
                raise MCPServiceError(
                    "startup_time_seconds must be between "
                    f"{int(MIN_EVICTION_STARTUP_SECONDS)} and "
                    f"{int(MAX_EVICTION_STARTUP_SECONDS)} seconds"
                )
            startup_time_seconds = float(startup_time_seconds)
        try:
            from .mcp_events import EventStore, Monitor
        except ImportError as exc:
            raise MCPServiceError("Falcon event monitor is not installed") from exc

        with self._event_lock:
            if self._closed:
                raise MCPServiceError("JobService is closed")
            if self._event_store is None:
                self._event_store = EventStore(self._db_path)
            if self._monitor is None:
                self._monitor = Monitor(
                    self.client,
                    self.namespace,
                    self._event_store,
                    metrics_provider=self._metrics_provider,
                )
            for ref in refs:
                self._monitor_refs[ref["uid"]] = dict(ref)
                self._monitor_refs[ref["name"]] = dict(ref)
                if startup_time_seconds is not None:
                    self._monitor_startup_times[ref["uid"]] = startup_time_seconds
            monitor = self._monitor
        result = await monitor.wait(
            refs,
            cursor=cursor,
            event_types=event_types,
            timeout=float(timeout),
            scheduling_policy=scheduling_policy,
        )
        output = await self._enrich_failure_events(result, refs)
        output["jobs"] = refs
        return output

    async def _enrich_failure_events(
        self,
        result: Any,
        refs: Sequence[Mapping[str, str]],
    ) -> dict[str, Any]:
        if isinstance(result, Mapping):
            output = copy.deepcopy(dict(result))
        elif isinstance(result, list):
            output = {"events": copy.deepcopy(result)}
        else:
            output = {"events": [] if result is None else [result]}
        events = output.get("events")
        if not isinstance(events, list):
            return output
        refs_by_name = {ref["name"]: dict(ref) for ref in refs}
        delivered_bytes = 0
        seen_attempts: set[tuple[str, str]] = set()
        for event in events:
            if not isinstance(event, Mapping) or not _failure_event(event):
                continue
            name = _event_job_name(event)
            details = event.get("details")
            pod_name = details.get("pod") if isinstance(details, Mapping) else None
            if not name or name not in refs_by_name:
                continue
            ref = refs_by_name[name]
            if not isinstance(pod_name, str) or not pod_name.strip():
                # A Job failure can race the Pod watch.  Resolve only Pods
                # owned by this exact Job UID; a same-name replacement cannot
                # pass _job_pods' owner UID filter.
                try:
                    owned_pods = await asyncio.to_thread(
                        self._job_pods,
                        ref["name"],
                        ref["namespace"],
                        uid=ref["uid"],
                    )
                except Exception:
                    owned_pods = []
                failed_pods = [
                    pod for pod in owned_pods if _pod_phase(pod) == "Failed"
                ]
                failed_pods.sort(
                    key=lambda pod: str(
                        (pod.get("metadata") or {}).get("creationTimestamp") or ""
                    )
                )
                if failed_pods:
                    pod_name = str(
                        (failed_pods[-1].get("metadata") or {}).get("name") or ""
                    )
            if not isinstance(pod_name, str) or not pod_name.strip():
                continue
            attempt = (name, pod_name.strip())
            if attempt in seen_attempts:
                continue
            seen_attempts.add(attempt)
            if delivered_bytes >= 16 * 1024:
                break
            try:
                logs = await asyncio.to_thread(
                    self.job_logs,
                    ref,
                    pods=[pod_name.strip()],
                    failed_only=False,
                    limit=1,
                    tail=_DEFAULT_LOG_TAIL,
                )
            except Exception as exc:
                logs = {
                    "job": refs_by_name[name],
                    "logs": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if isinstance(event, dict):
                _bound_failure_log_result(logs, 16 * 1024 - delivered_bytes)
                event["failure_logs"] = logs
                delivered_bytes += _log_result_size(logs)
        return output

    async def close(self) -> None:
        with self._event_lock:
            if self._closed:
                return
            self._closed = True
            monitor = self._monitor
            self._monitor = None
            event_store = self._event_store
            self._event_store = None
            self._monitor_refs.clear()
            self._monitor_startup_times.clear()
        # Monitor owns watch subprocesses/tasks.  It must be stopped before
        # any collector or SQLite resource it can still call is closed.
        if monitor is not None:
            await monitor.close()
        with self._metrics_lock:
            collectors = list(self._metrics_collectors.values())
            self._metrics_collectors.clear()
            self._metrics_snapshots.clear()
            self._risk_evaluators.clear()
            self._risk_snapshots.clear()
            # Keep the lock while closing: no in-flight collector can still
            # be executing, and a concurrent call cannot create a replacement
            # collector after the map is cleared.
            for collector in collectors:
                close = getattr(collector, "close", None)
                if callable(close):
                    close()
        if event_store is not None:
            close = getattr(event_store, "close", None)
            if callable(close):
                close()
        self._operations.close()
        close_client = getattr(self.client, "close", None)
        if callable(close_client):
            close_client()
        self._closed = True


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


def _node_resources(node: Any) -> Any:
    from .models import NodeResources

    return NodeResources(
        name=node.name,
        cpu_total=node.allocatable.cpu_cores,
        cpu_used=node.requested.cpu_cores,
        memory_total_gib=node.allocatable.memory_gib,
        memory_used_gib=node.requested.memory_gib,
        gpu_total=node.allocatable.gpu_count,
        gpu_used=node.requested.gpu_count,
        gpu_product=node.gpu_model or "",
        unschedulable=not node.schedulable or node.ready is False,
        scheduling_info_available=True,
    )


def _main_container(manifest: Mapping[str, Any]) -> dict[str, Any]:
    containers = (
        (((manifest.get("spec") or {}).get("template") or {}).get("spec") or {})
        .get("containers")
        or []
    )
    if not containers or not isinstance(containers[0], Mapping):
        raise MCPServiceError("Job manifest has no main container")
    return dict(containers[0])


def _rename_manifest(
    manifest: Mapping[str, Any],
    name: str,
    namespace: str,
    *,
    add_job_label: bool = True,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(manifest))
    metadata = result.setdefault("metadata", {})
    metadata["name"] = name
    metadata["namespace"] = namespace
    template = ((result.setdefault("spec", {})).setdefault("template", {}))
    template_metadata = template.setdefault("metadata", {})
    if add_job_label:
        labels = template_metadata.setdefault("labels", {})
        labels["job-name"] = name
    return result


def _identity_from_job(job: Mapping[str, Any], namespace: str) -> dict[str, str]:
    metadata = job.get("metadata") or {}
    return {
        "name": str(metadata.get("name") or ""),
        "namespace": str(metadata.get("namespace") or namespace),
        "uid": str(metadata.get("uid") or ""),
    }


def _identity_from_created(value: Any, namespace: str, fallback_name: str) -> dict[str, str]:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        metadata = value.get("metadata") if isinstance(value.get("metadata"), Mapping) else value
        return {
            "name": str(metadata.get("name") or fallback_name),
            "namespace": str(metadata.get("namespace") or namespace),
            "uid": str(metadata.get("uid") or ""),
        }
    return {
        "name": str(getattr(value, "name", "") or fallback_name),
        "namespace": str(getattr(value, "namespace", "") or namespace),
        "uid": str(getattr(value, "uid", "") or ""),
    }


def _generated_job_name(
    request: Mapping[str, Any],
    command: Sequence[str],
    *,
    uniqueness: Optional[str] = None,
) -> str:
    """Create a readable random DNS name when the MCP caller omits ``name``.

    ``uniqueness`` is retained in the signature because submit idempotency is
    persisted by :class:`JobService`; the first generated name is recorded in
    the operation store and reused on later attempts with the same key.
    """

    del request, command, uniqueness
    return f"falcon-{generate_workspace_name()}"


def _strip_controller_metadata(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return copy.deepcopy(value)
    result = {str(key): copy.deepcopy(item) for key, item in value.items()}
    for key in (
        "uid",
        "resourceVersion",
        "generation",
        "creationTimestamp",
        "deletionTimestamp",
        "deletionGracePeriodSeconds",
        "managedFields",
        "ownerReferences",
        "selfLink",
    ):
        result.pop(key, None)
    return result


def _strip_job_controller_labels(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove labels that bind metadata to the source Job controller."""

    result = copy.deepcopy(dict(value))
    labels = result.get("labels")
    if isinstance(labels, Mapping):
        result["labels"] = {
            str(key): copy.deepcopy(label)
            for key, label in labels.items()
            if str(key) not in _JOB_CONTROLLER_LABELS
        }
    return result


def _restart_manifest(
    source: Mapping[str, Any],
    *,
    namespace: str,
    name: str,
    previous_uid: str,
    overrides: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Clone a live Job spec without copying controller-owned metadata."""

    metadata = _strip_controller_metadata(source.get("metadata") or {})
    metadata = _strip_job_controller_labels(metadata)
    metadata["name"] = name
    metadata["namespace"] = namespace
    spec = copy.deepcopy(source.get("spec") or {})
    _validate_restart_shape(spec)
    # A selector/manualSelector copied from a live Job is tied to its old
    # controller UID and can make the API reject the replacement or adopt old
    # Pods.  Let the new Job controller create its own selector and labels.
    spec.pop("selector", None)
    spec.pop("manualSelector", None)
    # These policies can cause the Kubernetes controller to retry or replace
    # Pods independently of backoffLimit.  An agent relaunch owns the retry
    # decision, so inherited policy/index fields must not survive the clone.
    for field in (
        "podFailurePolicy",
        "backoffLimitPerIndex",
        "maxFailedIndexes",
    ):
        spec.pop(field, None)
    template = copy.deepcopy(spec.get("template") or {})
    template_metadata = _strip_controller_metadata(template.get("metadata") or {})
    template_metadata = _strip_job_controller_labels(template_metadata)
    template_spec = copy.deepcopy(template.get("spec") or {})
    template["metadata"] = template_metadata
    template["spec"] = template_spec
    for field in ("containers", "initContainers", "ephemeralContainers"):
        values = template_spec.get(field) or []
        for value in values:
            if isinstance(value, dict):
                value.pop("restartPolicy", None)
                value.pop("restartPolicyRules", None)
    spec["template"] = template
    result: dict[str, Any] = {
        "apiVersion": str(source.get("apiVersion") or "batch/v1"),
        "kind": "Job",
        "metadata": metadata,
        "spec": spec,
    }
    containers = template_spec.get("containers") or []
    if not containers or not isinstance(containers[0], Mapping):
        raise MCPServiceError("source Job has no restartable main container")
    container = containers[0]
    _apply_restart_overrides(
        container,
        template_spec,
        spec,
        overrides,
        config=config,
    )
    # A replacement Job must not retain controller status or retry behavior.
    spec.pop("status", None)
    spec["backoffLimit"] = 0
    template_spec["restartPolicy"] = "Never"
    metadata.setdefault("annotations", {})[PREVIOUS_JOB_UID_ANNOTATION] = previous_uid
    return result


def _validate_restart_shape(spec: Mapping[str, Any]) -> None:
    """Reject Job shapes whose relaunch would silently multiply work."""

    completion_mode = str(spec.get("completionMode") or "NonIndexed")
    if completion_mode == "Indexed":
        raise MCPServiceError(
            "restarting Indexed Jobs is unsupported; submit a new non-indexed Job"
        )
    for field in ("completions", "parallelism"):
        value = spec.get(field)
        try:
            count = int(value) if value is not None else 1
        except (TypeError, ValueError) as exc:
            raise MCPServiceError(
                f"restarting Jobs with invalid {field} is unsupported"
            ) from exc
        if count > 1:
            raise MCPServiceError(
                "restarting multi-completion Jobs is unsupported; submit a new "
                "single-completion Job"
            )


def _apply_restart_overrides(
    container: dict[str, Any],
    template_spec: dict[str, Any],
    spec: dict[str, Any],
    overrides: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> None:
    if "command" in overrides:
        container["command"] = list(overrides["command"])
        container.pop("args", None)
    if "image" in overrides:
        container["image"] = overrides["image"]
    if "working_dir" in overrides:
        container["workingDir"] = overrides["working_dir"]
    if "env" in overrides:
        container["env"] = _merge_env_patch(
            container.get("env") or [],
            overrides["env"],
        )

    resources = container.setdefault("resources", {})
    requests = resources.setdefault("requests", {})
    limits = resources.setdefault("limits", {})
    if "cpu" in overrides:
        requests["cpu"] = _request_part(str(overrides["cpu"]))
        limits["cpu"] = _limit_part(str(overrides["cpu"]), requests["cpu"])
    if "memory" in overrides:
        requests["memory"] = _request_part(str(overrides["memory"]))
        limits["memory"] = _limit_part(str(overrides["memory"]), requests["memory"])

    if "gpu" in overrides or "gpus" in overrides:
        existing_gpu_key = next(
            (key for key in list(requests) + list(limits) if str(key).endswith("/gpu")),
            None,
        )
        existing_gpu_count = 0
        if existing_gpu_key is not None:
            try:
                existing_gpu_count = int(
                    requests.get(existing_gpu_key, limits.get(existing_gpu_key, 0)) or 0
                )
            except (TypeError, ValueError):
                existing_gpu_count = 0
        cluster = config.get("cluster") or {}
        selector_key = str(cluster.get("gpu_label") or "gpu-type")
        existing_gpu_model = str(
            (template_spec.get("nodeSelector") or {}).get(selector_key) or ""
        )
        gpu_model, gpu_count = _restart_gpu_values(
            overrides,
            config,
            default_model=existing_gpu_model,
            default_count=existing_gpu_count,
        )
        gpu_key = next(
            (key for key in list(requests) + list(limits) if str(key).endswith("/gpu")),
            "nvidia.com/gpu",
        )
        requests[gpu_key] = str(gpu_count)
        limits[gpu_key] = str(gpu_count)
        selector = template_spec.setdefault("nodeSelector", {})
        selector[selector_key] = gpu_model

    if "shared_memory" in overrides or "shared_memory_percent" in overrides:
        memory = requests.get("memory")
        if not memory:
            raise MCPServiceError("shared memory override requires a memory request")
        size = overrides.get("shared_memory")
        if size is None:
            from .planning import calculate_shared_memory

            size = calculate_shared_memory(
                str(memory),
                percent=overrides.get("shared_memory_percent", 15),
            )
        found = False
        for volume in template_spec.get("volumes") or []:
            if volume.get("name") == "shared-memory":
                empty_dir = volume.setdefault("emptyDir", {})
                empty_dir["medium"] = "Memory"
                empty_dir["sizeLimit"] = size
                found = True
        if not found:
            template_spec.setdefault("volumes", []).append(
                {"name": "shared-memory", "emptyDir": {"medium": "Memory", "sizeLimit": size}}
            )
        mounts = container.setdefault("volumeMounts", [])
        if not any(item.get("mountPath") == "/dev/shm" for item in mounts):
            mounts.append({"name": "shared-memory", "mountPath": "/dev/shm"})


def _merge_env_patch(
    existing: Any,
    patch: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Preserve inherited env entries while replacing only patched names."""

    merged: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for entry in existing if isinstance(existing, list) else []:
        if not isinstance(entry, Mapping):
            continue
        copied = copy.deepcopy(dict(entry))
        merged.append(copied)
        name = copied.get("name")
        if isinstance(name, str) and name:
            positions[name] = len(merged) - 1

    for name, value in patch.items():
        position = positions.get(name)
        replacement = {"name": name, "value": str(value)}
        if position is None:
            positions[name] = len(merged)
            merged.append(replacement)
        else:
            merged[position] = replacement
    return merged


def _restart_gpu_values(
    overrides: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    default_model: str = "",
    default_count: int = 1,
) -> tuple[str, int]:
    token = overrides.get("gpu")
    count = int(overrides.get("gpus", default_count if token is None else 1))
    if isinstance(token, Mapping):
        model = str(token.get("model") or "")
        if "count" in token:
            count = int(token["count"])
    elif token is not None:
        model = str(token)
    else:
        model = default_model
    match = _GPU_COUNT_SUFFIX.fullmatch(model)
    if match and "gpus" not in overrides:
        model = match.group("model")
        count = int(match.group("count"))
    presets = config.get("presets") or {}
    preset = presets.get(model)
    if isinstance(preset, Mapping):
        maximum = gpu_preset_max_count(preset)
        if count > maximum:
            raise MCPServiceError(
                f"GPU preset {model} supports at most {maximum} GPU(s); requested {count}"
            )
        model = str(preset.get("gpu_type") or model)
    if not model:
        raise MCPServiceError(
            "restart GPU count/model override cannot be resolved from the source Job"
        )
    return canonical_gpu(model), count


def _request_part(value: str) -> str:
    return value.split(":", 1)[0].strip()


def _limit_part(value: str, request: str) -> str:
    parts = value.split(":", 1)
    return parts[1].strip() if len(parts) == 2 and parts[1].strip() else request


def _pod_matches_job(
    pod: Mapping[str, Any],
    name: str,
    uid: Optional[str] = None,
) -> bool:
    metadata = pod.get("metadata") or {}
    owners = metadata.get("ownerReferences") or []
    job_owners = [
        owner
        for owner in owners
        if str(owner.get("kind") or "") == "Job"
    ]
    if uid is not None:
        # Once a Job UID is known, a same-name label is not sufficient: a
        # replacement Job can inherit that label while old attempt Pods are
        # still visible for a short time.
        if job_owners:
            return any(
                str(owner.get("name") or "") == name
                and str(owner.get("uid") or "") == uid
                for owner in job_owners
            )
        # Orphan deletion preserves the old controller UID labels while
        # removing owner references. Require both identity labels, never
        # just the Job name, so attempts cannot leak into a replacement.
        labels = metadata.get("labels") or {}
        return (
            labels.get("job-name", labels.get("batch.kubernetes.io/job-name")) == name
            and labels.get(
                "controller-uid", labels.get("batch.kubernetes.io/controller-uid")
            ) == uid
        )
    labels = metadata.get("labels") or {}
    if labels.get("job-name") == name or labels.get("batch.kubernetes.io/job-name") == name:
        return True
    return any(
        str(owner.get("kind") or "") == "Job" and str(owner.get("name") or "") == name
        for owner in job_owners
    )


def _pod_phase(pod: Mapping[str, Any]) -> str:
    return str((pod.get("status") or {}).get("phase") or "Unknown")


def _pod_container_details(
    pod: Mapping[str, Any],
    container_name: Optional[str],
) -> dict[str, Any]:
    status = pod.get("status") or {}
    reason = status.get("reason")
    exit_code: Optional[int] = None
    container_reason: Optional[str] = None
    for field in ("containerStatuses", "initContainerStatuses"):
        values = status.get(field) or []
        for value in values:
            if not isinstance(value, Mapping):
                continue
            if container_name and value.get("name") not in {None, container_name}:
                continue
            state = value.get("state") or {}
            terminated = state.get("terminated") or {}
            waiting = state.get("waiting") or {}
            if isinstance(terminated, Mapping):
                if terminated.get("exitCode") is not None:
                    try:
                        exit_code = int(terminated["exitCode"])
                    except (TypeError, ValueError):
                        exit_code = None
                container_reason = terminated.get("reason") or container_reason
            if isinstance(waiting, Mapping):
                container_reason = waiting.get("reason") or container_reason
            if reason is None:
                reason = container_reason
            if exit_code is not None or container_reason:
                break
        if exit_code is not None or container_reason:
            break
    return {
        "reason": str(reason) if reason is not None else None,
        "exit_code": exit_code,
        "container_reason": (
            str(container_reason) if container_reason is not None else None
        ),
    }


def _compact_pod_conditions(value: Any) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    for condition in value if isinstance(value, list) else []:
        if not isinstance(condition, Mapping):
            continue
        compact: dict[str, Any] = {}
        for source_key, output_key, limit in (
            ("type", "type", (128, 256)),
            ("status", "status", (64, 128)),
            ("reason", "reason", (256, 512)),
            ("message", "message", (1_024, 4_096)),
            ("lastTransitionTime", "last_transition_time", (128, 256)),
        ):
            raw = condition.get(source_key)
            if raw is None:
                continue
            bounded, truncated = _bound_text(raw, limit[0], limit[1])
            compact[output_key] = bounded
            if source_key == "message" and truncated:
                compact["message_truncated"] = True
        if compact:
            conditions.append(compact)
        if len(conditions) >= 16:
            break
    return conditions


def _compact_inspect_event(
    value: Mapping[str, Any],
    involved: Mapping[str, Any],
) -> dict[str, Any]:
    raw_message = value.get("note") or value.get("message") or ""
    message, message_truncated = _bound_text(raw_message, 2_048, 4_096)
    compact: dict[str, Any] = {
        "type": str(value.get("type") or "")[:64],
        "reason": str(value.get("reason") or "")[:256],
        "message": message,
        "involved_object": {
            "kind": str(involved.get("kind") or "")[:64],
            "name": str(involved.get("name") or "")[:256],
            "namespace": str(involved.get("namespace") or "")[:256],
            "uid": str(involved.get("uid") or "")[:256],
        },
    }
    timestamp = (
        value.get("eventTime")
        or value.get("lastTimestamp")
        or ((value.get("series") or {}).get("lastObservedTime") if isinstance(value.get("series"), Mapping) else None)
        or (value.get("metadata") or {}).get("creationTimestamp")
    )
    if timestamp is not None:
        compact["timestamp"] = str(timestamp)[:128]
    count = value.get("count")
    if isinstance(count, int) and not isinstance(count, bool):
        compact["count"] = count
    if message_truncated:
        compact["message_truncated"] = True
    return compact


def _pod_summary(pod: Mapping[str, Any]) -> dict[str, Any]:
    metadata = pod.get("metadata") or {}
    status = pod.get("status") or {}
    container_details = _pod_container_details(pod, None)
    return {
        "name": str(metadata.get("name") or ""),
        "uid": str(metadata.get("uid") or ""),
        "phase": str(status.get("phase") or "Unknown"),
        "created_at": metadata.get("creationTimestamp"),
        "node": str((pod.get("spec") or {}).get("nodeName") or "") or None,
        "reason": container_details["reason"],
        "container_reason": container_details["container_reason"],
        "exit_code": container_details["exit_code"],
        "conditions": _compact_pod_conditions(status.get("conditions")),
    }


def _age(timestamp: Any) -> Optional[str]:
    if not isinstance(timestamp, str) or not timestamp:
        return None
    parsed = _parse_log_timestamp(timestamp)
    if parsed is None:
        return None
    seconds = max(0, int(time.time() - parsed.timestamp()))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def _compact_list_summary(
    job: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the bounded list schema; detailed fields belong to inspect."""

    metadata = job.get("metadata") or {}
    spec = job.get("spec") or {}
    template_spec = (spec.get("template") or {}).get("spec") or {}
    containers = template_spec.get("containers") or []
    container = containers[0] if containers and isinstance(containers[0], Mapping) else {}
    resources = container.get("resources") or {}
    requests = resources.get("requests") or {}
    gpu = summary.get("gpu")
    if isinstance(gpu, Mapping) and int(gpu.get("count", 0) or 0) > 0:
        request_gpu: Optional[dict[str, Any]] = {
            "model": gpu.get("model"),
            "count": int(gpu.get("count", 0) or 0),
        }
    else:
        request_gpu = None
    ref = {
        "name": str(metadata.get("name") or ""),
        "namespace": str(metadata.get("namespace") or ""),
        "uid": str(metadata.get("uid") or ""),
    }
    return {
        "job": ref,
        "name": ref["name"],
        "namespace": ref["namespace"],
        "uid": ref["uid"],
        "status": summary.get("status"),
        "age": _age(metadata.get("creationTimestamp")),
        "request": {
            "cpu": requests.get("cpu"),
            "memory": requests.get("memory"),
            "gpu": request_gpu,
            "image": container.get("image"),
        },
        "agent_managed": bool(summary.get("agent_managed")),
        "retry": summary.get("retry"),
    }


def _process_output(value: Any) -> tuple[str, str, int]:
    if isinstance(value, ProcessResult):
        return value.stdout or "", value.stderr or "", int(value.returncode)
    if isinstance(value, Mapping):
        return (
            str(value.get("stdout") or ""),
            str(value.get("stderr") or ""),
            int(value.get("returncode", 0) or 0),
        )
    return (
        str(getattr(value, "stdout", "") or ""),
        str(getattr(value, "stderr", "") or ""),
        int(getattr(value, "returncode", 0) or 0),
    )


_LOG_TIMESTAMP = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))(?P<separator>\s+)(?P<message>.*)$"
)


def _parse_log_timestamp(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _log_timestamp_key(value: str) -> Optional[tuple[datetime, int]]:
    """Return a UTC second plus nanosecond fraction without float loss."""

    parsed = _parse_log_timestamp(value)
    if parsed is None:
        return None
    fraction_match = re.search(r"\.(\d+)(?:Z|[+-]\d{2}:\d{2})$", value)
    fraction = 0
    if fraction_match:
        # Kubernetes emits RFC3339Nano (at most nine fractional digits).
        # Preserve all representable precision even though datetime itself
        # stores only microseconds.
        fraction = int((fraction_match.group(1) + "000000000")[:9])
    return parsed.replace(microsecond=0), fraction


def _encode_log_cursor(
    *,
    pod_uid: str,
    container: str,
    timestamp: str,
    occurrence_count: int,
) -> str:
    payload = json.dumps(
        {
            "version": 1,
            "pod_uid": pod_uid,
            "container": container,
            "timestamp": timestamp,
            "occurrence_count": occurrence_count,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_log_cursor(value: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        raise MCPServiceError("log cursor must be an opaque non-empty string")
    try:
        padding = "=" * (-len(value) % 4)
        decoded = base64.urlsafe_b64decode((value + padding).encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise MCPServiceError("log cursor is invalid or corrupt") from exc
    if not isinstance(payload, Mapping) or payload.get("version") != 1:
        raise MCPServiceError("log cursor version is unsupported")
    pod_uid = payload.get("pod_uid")
    container = payload.get("container")
    timestamp = payload.get("timestamp")
    occurrence_count = payload.get("occurrence_count")
    if (
        not isinstance(pod_uid, str)
        or not pod_uid
        or not isinstance(container, str)
        or not container
        or not isinstance(timestamp, str)
        or _log_timestamp_key(timestamp) is None
        or isinstance(occurrence_count, bool)
        or not isinstance(occurrence_count, int)
        or occurrence_count < 1
    ):
        raise MCPServiceError("log cursor is malformed")
    return {
        "pod_uid": pod_uid,
        "container": container,
        "timestamp": timestamp,
        "timestamp_value": _parse_log_timestamp(timestamp),
        "timestamp_key": _log_timestamp_key(timestamp),
        "occurrence_count": occurrence_count,
    }


def _bounded_prefix(value: str, max_chars: int, max_bytes: int) -> tuple[str, bool]:
    if max_chars <= 0 or max_bytes <= 0:
        return "", bool(value)
    candidate = value[:max_chars]
    while len(candidate.encode("utf-8")) > max_bytes:
        candidate = candidate[:-1]
    return candidate, candidate != value


def _bound_text(
    value: Any,
    max_chars: int,
    max_bytes: Optional[int] = None,
) -> tuple[str, bool]:
    text = str(value or "")
    byte_limit = max_chars if max_bytes is None else max_bytes
    return _bounded_prefix(text, max_chars, byte_limit)


def _incremental_log_text(
    text: str,
    *,
    pod_uid: str,
    container: str,
    cursor: Optional[Mapping[str, Any]],
    max_chars: int,
    max_bytes: int,
    tail: int,
) -> tuple[str, Optional[str], dict[str, Any]]:
    """Filter timestamped lines after a cursor and return bounded latest text.

    Kubernetes includes the timestamp on every line when ``--timestamps`` is
    used.  A cursor records the timestamp and the occurrence number at that
    timestamp, so repeated timestamps do not cause a line to be replayed.  A
    full ``tail`` window is explicitly reported as a possible gap because the
    Kubernetes log endpoint does not expose how many older lines it discarded.
    """

    raw_lines = str(text or "").splitlines()
    cursor_value = cursor.get("timestamp_key") if cursor else None
    cursor_count = int(cursor.get("occurrence_count", 0)) if cursor else 0
    timestamp_counts: dict[tuple[datetime, int], int] = {}
    candidates: list[dict[str, Any]] = []
    unparsed = False
    for raw_line in raw_lines:
        match = _LOG_TIMESTAMP.match(raw_line)
        timestamp = match.group("timestamp") if match else None
        timestamp_key = _log_timestamp_key(timestamp) if timestamp else None
        if timestamp_key is None:
            unparsed = True
            candidates.append({"raw": raw_line, "token": None})
            continue
        occurrence = timestamp_counts.get(timestamp_key, 0) + 1
        timestamp_counts[timestamp_key] = occurrence
        if cursor_value is not None:
            if timestamp_key < cursor_value:
                continue
            if timestamp_key == cursor_value and occurrence <= cursor_count:
                continue
        candidates.append(
            {
                "raw": raw_line,
                "token": _encode_log_cursor(
                    pod_uid=pod_uid,
                    container=container,
                    timestamp=timestamp or "",
                    occurrence_count=occurrence,
                ),
            }
        )

    full_count = len(candidates)
    tail_limited = bool(tail > 0 and full_count > tail)
    tail_window_full = bool(tail > 0 and full_count >= tail)
    if tail > 0:
        candidates = candidates[-tail:]
    elif tail == 0:
        candidates = []

    # Build from the newest line backwards so a byte/character cap retains the
    # most useful part of a rolling log window.
    selected: list[dict[str, Any]] = []
    used_chars = 0
    used_bytes = 0
    line_truncated = False
    for record in reversed(candidates):
        raw_line = str(record["raw"])
        separator_chars = 1 if selected else 0
        separator_bytes = 1 if selected else 0
        allowed_chars = max_chars - used_chars - separator_chars
        allowed_bytes = max_bytes - used_bytes - separator_bytes
        bounded, was_truncated = _bounded_prefix(
            raw_line,
            max(0, allowed_chars),
            max(0, allowed_bytes),
        )
        if not bounded and raw_line:
            line_truncated = True
            break
        selected.append({**record, "raw": bounded})
        used_chars += len(bounded) + separator_chars
        used_bytes += len(bounded.encode("utf-8")) + separator_bytes
        if was_truncated:
            line_truncated = True
            break
        if used_chars >= max_chars or used_bytes >= max_bytes:
            line_truncated = len(selected) < len(candidates)
            break
    selected.reverse()
    visible = "\n".join(str(record["raw"]) for record in selected)
    last_token: Optional[str] = None
    for record in reversed(selected):
        if record.get("token"):
            last_token = str(record["token"])
            break
    previous_token = (
        _encode_log_cursor(
            pod_uid=pod_uid,
            container=container,
            timestamp=str(cursor["timestamp"]),
            occurrence_count=int(cursor["occurrence_count"]),
        )
        if cursor
        else None
    )
    cursor_advanced = last_token is not None and last_token != previous_token
    gap_possible = bool(unparsed or tail_window_full or tail_limited or line_truncated)
    truncated = bool(tail_limited or line_truncated)
    state = {
        "truncated": truncated,
        "log_truncated": truncated,
        "gap_possible": gap_possible,
        "tail_window_full": tail_window_full,
        "cursor_advanced": cursor_advanced,
        "lines_available": full_count,
        "lines_delivered": len(selected),
        "bytes_delivered": len(visible.encode("utf-8")),
        "timestamped": not unparsed,
    }
    return visible, last_token, state


def _log_result_size(value: Any) -> int:
    try:
        return len(
            json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
            .encode("utf-8")
        )
    except (TypeError, ValueError):
        return len(str(value).encode("utf-8"))


def _hold_log_entry_cursor(value: dict[str, Any], entry: Mapping[str, Any]) -> None:
    """Keep a cursor at the last fully delivered response boundary."""

    key = entry.get("cursor_key")
    raw_cursor = entry.get("cursor")
    if isinstance(key, str) and key:
        cursors = value.setdefault("cursors", {})
        if isinstance(cursors, dict):
            if isinstance(raw_cursor, str) and raw_cursor:
                cursors[key] = raw_cursor
            else:
                cursors.pop(key, None)
    if isinstance(entry, dict):
        entry["next_cursor"] = raw_cursor
        entry["cursor_advanced"] = False
        entry["gap_possible"] = True
        entry["truncated"] = True
        entry["log_truncated"] = True


def _bound_job_logs_response(
    value: dict[str, Any],
    budget: int,
) -> dict[str, Any]:
    """Bound the complete serialized log response, not just each Pod."""

    if _log_result_size(value) <= budget:
        return value
    value["truncated"] = True
    value["response_truncated"] = True
    entries = value.get("logs")
    if not isinstance(entries, list):
        entries = []
        value["logs"] = entries

    while _log_result_size(value) > budget:
        text_fields = [
            (entry, field)
            for entry in entries
            if isinstance(entry, dict)
            for field in ("stdout", "stderr")
            if isinstance(entry.get(field), str) and entry.get(field)
        ]
        if text_fields:
            entry, field = max(
                text_fields,
                key=lambda item: len(item[0][item[1]].encode("utf-8")),
            )
            current = str(entry[field])
            bounded, _ = _bounded_prefix(
                current,
                max(0, len(current) // 2),
                max(0, len(current.encode("utf-8")) // 2),
            )
            entry[field] = bounded
            entry[f"{field}_truncated"] = True
            if field == "stdout":
                _hold_log_entry_cursor(value, entry)
            continue

        optional_fields = (
            "stderr_truncated",
            "container_reason",
            "reason",
            "created_at",
            "phase",
            "cursor",
            "next_cursor",
            "lines_available",
            "lines_delivered",
            "bytes_delivered",
            "tail_window_full",
            "timestamped",
            "cursor_advanced",
        )
        removed_field = False
        for entry in reversed(entries):
            if not isinstance(entry, dict):
                continue
            field = next((name for name in optional_fields if name in entry), None)
            if field is not None:
                entry.pop(field, None)
                removed_field = True
                break
        if removed_field:
            continue

        if entries:
            removed = entries.pop()
            if isinstance(removed, Mapping):
                _hold_log_entry_cursor(value, removed)
            continue

        cursors = value.get("cursors")
        if isinstance(cursors, dict) and cursors:
            cursors.pop(next(reversed(cursors)))
            continue
        # The remaining fields are the bounded Job reference and status
        # metadata.  This final fallback keeps the explicit contract even if
        # an adapter supplied unexpectedly large opaque cursor strings.
        value.pop("failed_only", None)
        value["logs"] = []
        value["count"] = 0
        value["cursors"] = {}
        break

    value["count"] = len(entries)
    return value


def _bound_failure_log_result(value: Any, budget: int) -> None:
    """Bound an enrichment payload in place without retaining raw log data."""

    if not isinstance(value, dict):
        return
    budget = max(0, int(budget))
    truncated = False
    error = value.get("error")
    if error is not None:
        bounded_error, error_truncated = _bound_text(str(error), 1_024, 1_024)
        value["error"] = bounded_error
        truncated = truncated or error_truncated
    logs = value.get("logs")
    if isinstance(logs, list):
        for entry in logs:
            if not isinstance(entry, dict):
                continue
            stdout, stdout_truncated = _bound_text(
                entry.get("stdout"), _MAX_LOG_CHARS_PER_POD, _MAX_LOG_BYTES_PER_POD
            )
            stderr, stderr_truncated = _bound_text(entry.get("stderr"), 4_096, 4_096)
            entry["stdout"] = stdout
            entry["stderr"] = stderr
            if stdout_truncated:
                entry["stdout_truncated"] = True
            if stderr_truncated:
                entry["stderr_truncated"] = True
            truncated = truncated or stdout_truncated or stderr_truncated

    # In normal use a failure enrichment contains one log entry and is already
    # below the budget.  The shrink loop handles unusual adapters that return
    # several entries or oversized diagnostic fields.
    while _log_result_size(value) > budget:
        candidates = [
            entry
            for entry in value.get("logs", [])
            if isinstance(entry, dict)
        ]
        text_fields = [
            (entry, field)
            for entry in candidates
            for field in ("stdout", "stderr")
            if isinstance(entry.get(field), str) and entry.get(field)
        ]
        if text_fields:
            entry, field = max(text_fields, key=lambda item: len(item[0][item[1]]))
            current = entry[field]
            entry[field], _ = _bounded_prefix(
                current,
                max(0, len(current) // 2),
                max(0, len(current.encode("utf-8")) // 2),
            )
            truncated = True
            continue
        if candidates:
            value["logs"] = candidates[:1]
            truncated = True
            continue
        for field in ("cursors", "count", "failed_only", "job"):
            if field in value:
                value.pop(field, None)
                truncated = True
                break
        else:
            break
    if truncated:
        value["failure_logs_truncated"] = True


def _select_metric_row(rows: Any, name: str) -> Any:
    if isinstance(rows, Mapping):
        if name in rows:
            return rows[name]
        values = rows.get("items")
        if isinstance(values, list):
            rows = values
        else:
            return rows if rows.get("job") == name else None
    for row in rows or []:
        row_name = getattr(row, "job", None)
        if row_name is None and isinstance(row, Mapping):
            row_name = row.get("job") or row.get("name")
        if str(row_name or "") == name:
            return row
    return None


def _metric_mapping(
    row: Any,
    *,
    risk_policy: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    def value(name: str, default: Any = None) -> Any:
        if isinstance(row, Mapping):
            return row.get(name, default)
        return getattr(row, name, default)

    gpu_available = value("gpu_metrics_available")
    if gpu_available is None:
        gpu_available = value("gpu_util") is not None
    cpu_available = value("cpu_metrics_available")
    if cpu_available is None:
        cpu_available = value("cpu_used") is not None
    gpu_requested = int(value("gpu_requested_count", value("gpu_count", 0)) or 0)
    gpu_allocated = int(value("gpu_allocated_count", 0) or 0)
    has_gpu = gpu_requested > 0 or gpu_allocated > 0
    updated_at = value("metrics_updated_at")
    if updated_at in (None, 0, 0.0):
        updated_at = None
    at_risk = value("at_risk")
    if at_risk is None:
        at_risk = bool(value("gpu_at_risk") or value("vram_at_risk"))
    if has_gpu and not bool(gpu_available):
        at_risk = None
    if risk_policy is not None:
        at_risk = risk_policy.get("at_risk")
    risk_average = (
        risk_policy.get("gpu_average")
        if risk_policy is not None
        else value("gpu_risk_average")
    )
    vram_risk_average = (
        risk_policy.get("vram_average")
        if risk_policy is not None
        else value("vram_risk_average")
    )
    risk_ema = (
        risk_policy.get("gpu_ema")
        if risk_policy is not None
        else value("gpu_risk_ema")
    )
    vram_risk_ema = (
        risk_policy.get("vram_ema")
        if risk_policy is not None
        else value("vram_risk_ema")
    )
    risk_phase = (
        risk_policy.get("phase")
        if risk_policy is not None
        else value("risk_phase")
    )
    startup_remaining = (
        risk_policy.get("startup_remaining_seconds")
        if risk_policy is not None
        else value("risk_startup_remaining_seconds")
    )
    risk_window = (
        risk_policy.get("window_seconds")
        if risk_policy is not None
        else value("risk_window_seconds")
    )
    per_gpu = (
        risk_policy.get("devices", [])
        if risk_policy is not None
        else value("gpu_risk_devices", [])
    )
    gpu_at_risk = (
        risk_policy.get("gpu_at_risk")
        if risk_policy is not None
        else value("gpu_at_risk")
    )
    vram_at_risk = (
        risk_policy.get("vram_at_risk")
        if risk_policy is not None
        else value("vram_at_risk")
    )

    gpu = {
        "requested_type": value("gpu_requested_type", value("gpu_type")),
        "requested_count": gpu_requested,
        "allocated_type": value("gpu_allocated_type"),
        "allocated_count": gpu_allocated,
        "utilization_percent": value("gpu_util") if gpu_available else None,
        "ema_percent": value("gpu_ema") if gpu_available else None,
        "memory_used_gib": value("gpu_memory_used_gib") if gpu_available else None,
        "memory_total_gib": value("gpu_memory_total_gib") if gpu_available else None,
        "memory_percent": (
            _safe_percent(value("gpu_memory_used_gib"), value("gpu_memory_total_gib"))
            if gpu_available
            else None
        ),
        "risk_average_percent": risk_average if gpu_available else None,
        "risk_threshold_percent": value("gpu_risk_threshold") if gpu_available else None,
        "risk_ema_percent": risk_ema if gpu_available else None,
        "risk_phase": risk_phase,
        "startup_remaining_seconds": startup_remaining,
        "risk_window_seconds": risk_window,
        "available": bool(gpu_available),
    }
    cpu = {
        "requested_cores": value("cpu_requested"),
        "allocated_cores": value("cpu_allocated"),
        "used_cores": value("cpu_used") if cpu_available else None,
        "percent": _safe_percent(value("cpu_used"), value("cpu_allocated"))
        if cpu_available
        else None,
        "available": bool(cpu_available),
    }
    memory = {
        "requested_gib": value("memory_requested_gib"),
        "allocated_gib": value("memory_allocated_gib"),
        "used_gib": value("memory_used_gib") if cpu_available else None,
        "percent": _safe_percent(value("memory_used_gib"), value("memory_allocated_gib"))
        if cpu_available
        else None,
        "available": bool(cpu_available),
    }
    freshness = {
        "updated_at": updated_at,
        "age_seconds": (
            max(0.0, time.time() - float(updated_at))
            if updated_at is not None
            else None
        ),
    }
    return {
        "job": value("job"),
        "status": value("status"),
        "nodes": value("nodes"),
        "pod_count": value("pod_count"),
        "gpu": gpu,
        "cpu": cpu,
        "memory": memory,
        "gpu_metrics_available": bool(gpu_available),
        "cpu_metrics_available": bool(cpu_available),
        "available": bool(gpu_available or cpu_available),
        "freshness": freshness,
        "at_risk": at_risk,
        "gpu_at_risk": gpu_at_risk if gpu_available else None,
        "vram_at_risk": vram_at_risk if gpu_available else None,
        "eviction_policy": {
            "phase": risk_phase,
            "startup_remaining_seconds": startup_remaining,
            "window_seconds": risk_window,
            "gpu_ema_percent": risk_ema if gpu_available else None,
            "gpu_average_percent": risk_average if gpu_available else None,
            "vram_average_percent": vram_risk_average if gpu_available else None,
            "vram_ema_percent": vram_risk_ema if gpu_available else None,
            "per_gpu": per_gpu if gpu_available else [],
        },
    }


def _safe_percent(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        denominator_value = float(denominator)
        if denominator_value <= 0:
            return None
        return float(numerator) / denominator_value * 100.0
    except (TypeError, ValueError):
        return None


def _is_not_found(exc: BaseException) -> bool:
    if isinstance(exc, KubernetesError):
        return exc.not_found
    text = str(exc).lower()
    return "not found" in text or "notfound" in text


def _failure_event(event: Mapping[str, Any]) -> bool:
    values = [
        event.get("type"),
        event.get("event_type"),
        event.get("reason"),
        event.get("message"),
    ]
    text = " ".join(str(value or "") for value in values).lower()
    return any(
        marker in text
        for marker in (
            "failed",
            "failure",
            "error",
            "backofflimit",
            "backoff limit",
            "evict",
        )
    )


def _event_job_name(event: Mapping[str, Any]) -> str:
    for key in ("job", "job_name", "object_name", "name"):
        value = event.get(key)
        if isinstance(value, Mapping):
            value = value.get("name")
        if value:
            return str(value)
    involved = event.get("involvedObject")
    if isinstance(involved, Mapping):
        return str(involved.get("name") or "")
    return ""


__all__ = [
    "AGENT_LABEL",
    "CoderControlledJob",
    "IDEMPOTENCY_LABEL",
    "IdempotencyConflict",
    "MCPServiceError",
    "PREVIOUS_JOB_UID_ANNOTATION",
    "StaleJobReference",
    "JobService",
]
