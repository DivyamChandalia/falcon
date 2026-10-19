"""Durable Kubernetes events and long-lived wait support for Falcon MCP.

The MCP service deliberately keeps this module separate from the command and
submission paths.  A monitor observes Jobs, Pods, and Kubernetes Events; it
does not create, retry, or recover workloads.  The small amount of subprocess
code here is an async equivalent of ``kubectl get --watch`` and is kept local
so the existing synchronous Kubernetes adapter does not need a public API
change.
"""

from __future__ import annotations

import asyncio
import codecs
import inspect
import json
import os
import re
import sqlite3
import stat
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from datetime import datetime, timezone
from json import JSONDecodeError, JSONDecoder
from pathlib import Path
from typing import Any

import anyio

from .kubernetes import KubernetesClient

MAX_EVENT_RECORDS = 10_000
MAX_EVENT_DETAILS_BYTES = 4 * 1024
MAX_WAIT_RESPONSE_BYTES = 32 * 1024
DEFAULT_WAIT_TIMEOUT = 12 * 60 * 60.0
METRICS_INTERVAL = 1.0

# These are the events that can change what an MCP caller should do next.  The
# store keeps all event types, while wait() uses this set when no filter is
# supplied.
ACTIONABLE_EVENT_TYPES = (
    "succeeded",
    "failed",
    "deleted",
    "pod_failed",
    "evicted",
    "warning",
    "eviction_risk",
    "eviction_risk_cleared",
    "monitor_unavailable",
    "monitor_recovered",
)
SUPPORTED_EVENT_TYPES = ACTIONABLE_EVENT_TYPES + ("running", "pending")

_TERMINAL_TYPES = frozenset(("succeeded", "failed"))
_SECRET_KEY = re.compile(
    r"(?:pass(?:word)?|token|secret|authorization|api[_-]?key|private[_-]?key)",
    re.IGNORECASE,
)
_TRUNCATION_SUFFIX = "...[truncated]"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _string(value: Any) -> str:
    return str(value) if value is not None else ""


def _metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = value.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _job_identity(job: Mapping[str, Any]) -> dict[str, str]:
    """Extract the intentionally small Job identity persisted in the store."""

    metadata = _metadata(job)
    name = job.get("name") or metadata.get("name")
    namespace = job.get("namespace") or metadata.get("namespace")
    uid = job.get("uid") or metadata.get("uid")
    return {
        "name": _string(name),
        "namespace": _string(namespace),
        "uid": _string(uid),
    }


def _json_safe(value: Any, *, key: str = "") -> Any:
    """Return JSON data while removing values that must not enter private state."""

    if key and _SECRET_KEY.search(key):
        return "[redacted]"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            _string(k): _json_safe(v, key=_string(k))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
    except (TypeError, ValueError, OverflowError):
        return _string(value)
    return value


def _json_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= len(_TRUNCATION_SUFFIX):
        return _TRUNCATION_SUFFIX[:limit]
    return value[: limit - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX


def _shrink_detail_value(
    value: Any,
    *,
    string_limit: int = 512,
    message_limit: int = 3_072,
) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, string_limit)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = _string(key)
            limit = (
                message_limit
                if key_text.lower() in {"message", "note"}
                else string_limit
            )
            result[key_text] = _shrink_detail_value(
                item,
                string_limit=limit,
                message_limit=message_limit,
            )
        return result
    if isinstance(value, list):
        items = [
            _shrink_detail_value(
                item,
                string_limit=string_limit,
                message_limit=message_limit,
            )
            for item in value[:64]
        ]
        if len(value) > 64:
            items.append(_TRUNCATION_SUFFIX)
        return items
    return value


def _fit_message(candidate: dict[str, Any]) -> None:
    message_key = next(
        (
            key
            for key in ("message", "note")
            if isinstance(candidate.get(key), str)
        ),
        None,
    )
    if message_key is None:
        return
    original = candidate[message_key]
    low, high = 0, len(original)
    best: str | None = None
    while low <= high:
        middle = (low + high) // 2
        candidate[message_key] = _truncate_text(original, middle)
        if _json_size(candidate) <= MAX_EVENT_DETAILS_BYTES:
            best = candidate[message_key]
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        candidate.pop(message_key, None)
    else:
        candidate[message_key] = best


def _bounded_details(details: Mapping[str, Any]) -> dict[str, Any]:
    """Keep durable diagnostics small while preserving useful metadata."""

    candidate = _shrink_detail_value(dict(details))
    if not isinstance(candidate, dict):
        candidate = {"value": candidate}
    if _json_size(candidate) <= MAX_EVENT_DETAILS_BYTES:
        if _TRUNCATION_SUFFIX in json.dumps(candidate, ensure_ascii=False):
            marked = {**candidate, "details_truncated": True}
            if _json_size(marked) <= MAX_EVENT_DETAILS_BYTES:
                candidate = marked
        return candidate

    candidate["details_truncated"] = True
    priority = {
        "message",
        "note",
        "reason",
        "severity",
        "metadata",
        "involvedobject",
        "involved_object",
        "source",
        "details_truncated",
    }
    while _json_size(candidate) > MAX_EVENT_DETAILS_BYTES:
        removable = [key for key in candidate if key.lower() not in priority]
        if removable:
            largest = max(removable, key=lambda key: _json_size(candidate[key]))
            candidate.pop(largest, None)
            continue
        before = _json_size(candidate)
        _fit_message(candidate)
        if _json_size(candidate) < before:
            continue
        compacted = {
            key: _shrink_detail_value(
                value,
                string_limit=64,
                message_limit=256,
            )
            for key, value in candidate.items()
        }
        if _json_size(compacted) < before:
            candidate = compacted
            continue
        break

    if _json_size(candidate) <= MAX_EVENT_DETAILS_BYTES:
        return candidate

    fallback: dict[str, Any] = {"details_truncated": True}
    for key in ("reason", "severity", "source"):
        if key in candidate:
            fallback[key] = candidate[key]
    for key in ("metadata", "involvedObject", "involved_object"):
        if key in candidate:
            fallback[key] = _shrink_detail_value(
                candidate[key],
                string_limit=64,
                message_limit=128,
            )
    message = next(
        (
            candidate[key]
            for key in ("message", "note")
            if isinstance(candidate.get(key), str)
        ),
        None,
    )
    if message is not None:
        fallback["message"] = message
    while _json_size(fallback) > MAX_EVENT_DETAILS_BYTES:
        for key in ("metadata", "involvedObject", "involved_object", "source"):
            if key in fallback:
                fallback.pop(key)
                break
        else:
            _fit_message(fallback)
            break
    return fallback


class EventStore:
    """A bounded, cursor-addressable SQLite event log.

    One store belongs to one MCP context.  Cursors are SQLite rowids and are
    never reused when old records are evicted, so a caller can safely retain a
    cursor across reconnects and process restarts.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._memory = str(path) == ":memory:"
        self._lock = threading.RLock()
        self._closed = False

        if self._memory:
            database = ":memory:"
        else:
            self._prepare_parent(self.path.parent)
            if os.path.lexists(self.path):
                path_stat = os.lstat(self.path)
                if stat.S_ISLNK(path_stat.st_mode):
                    raise ValueError("EventStore database path must not be a symlink")
                if not stat.S_ISREG(path_stat.st_mode):
                    raise ValueError("EventStore database path must be a regular file")
            database = str(self.path)

        self._connection = sqlite3.connect(
            database,
            check_same_thread=False,
            isolation_level=None,
            timeout=30.0,
        )
        if not self._memory:
            # SQLite follows the process umask, but private MCP state should
            # remain private even when the process was launched with a loose
            # umask.  DELETE journaling avoids extra database sidecar files.
            path_stat = os.lstat(self.path)
            if stat.S_ISLNK(path_stat.st_mode):
                self._connection.close()
                raise ValueError("EventStore database path must not be a symlink")
            os.chmod(self.path, 0o600)
        with self._lock:
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.execute("PRAGMA journal_mode = DELETE")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    cursor INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    job_name TEXT NOT NULL,
                    job_namespace TEXT NOT NULL,
                    job_uid TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS events_job_cursor "
                "ON events(job_uid, cursor)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS events_type_cursor "
                "ON events(event_type, cursor)"
            )

    @staticmethod
    def _prepare_parent(parent: Path) -> None:
        """Create missing parents privately without changing existing roots.

        ``EventStore`` is also useful with a caller-provided state root such as
        ``/tmp``.  Existing directories are only checked for symlinks; their
        mode is deliberately left untouched.  Every directory created by this
        method is chmod'ed immediately so a permissive process umask cannot
        expose a fresh context folder.
        """

        if str(parent) == "":
            return
        absolute = Path(os.path.abspath(str(parent)))
        current = Path(absolute.anchor or ".")
        for component in absolute.parts[1:] if absolute.anchor else absolute.parts:
            candidate = current / component
            try:
                candidate_stat = os.lstat(candidate)
            except FileNotFoundError:
                try:
                    candidate.mkdir(mode=0o700)
                except FileExistsError:
                    candidate_stat = os.lstat(candidate)
                else:
                    os.chmod(candidate, 0o700)
                    candidate_stat = os.lstat(candidate)
            if stat.S_ISLNK(candidate_stat.st_mode):
                raise ValueError("EventStore state directory must not contain symlinks")
            if not stat.S_ISDIR(candidate_stat.st_mode):
                raise NotADirectoryError(str(candidate))
            current = candidate

    def _ensure_open(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("EventStore is closed")
        return self._connection

    @staticmethod
    def _event(
        cursor: int,
        event_type: str,
        job: Mapping[str, str],
        timestamp: str,
        details: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "cursor": int(cursor),
            "type": event_type,
            "job": {
                "name": _string(job.get("name")),
                "namespace": _string(job.get("namespace")),
                "uid": _string(job.get("uid")),
            },
            "timestamp": timestamp,
            "details": dict(details),
        }

    def append(
        self,
        job: dict,
        event_type: str,
        details: dict,
    ) -> dict:
        """Append one event and evict the oldest records over the bound."""

        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(job, Mapping):
            raise TypeError("job must be a mapping")
        if not isinstance(details, Mapping):
            raise TypeError("details must be a mapping")
        identity = _job_identity(job)
        safe_details = _json_safe(details)
        if not isinstance(safe_details, Mapping):
            safe_details = {"value": safe_details}
        bounded_details = _bounded_details(safe_details)
        details_json = json.dumps(
            bounded_details,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        timestamp = _utc_timestamp()

        with self._lock:
            connection = self._ensure_open()
            connection.execute("BEGIN IMMEDIATE")
            try:
                result = connection.execute(
                    """
                    INSERT INTO events(
                        event_type, job_name, job_namespace, job_uid,
                        timestamp, details
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_type.strip(),
                        identity["name"],
                        identity["namespace"],
                        identity["uid"],
                        timestamp,
                        details_json,
                    ),
                )
                cursor = int(result.lastrowid)
                count = int(
                    connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                )
                if count > MAX_EVENT_RECORDS:
                    connection.execute(
                        """
                        DELETE FROM events
                        WHERE cursor IN (
                            SELECT cursor FROM events
                            ORDER BY cursor ASC
                            LIMIT ?
                        )
                        """,
                        (count - MAX_EVENT_RECORDS,),
                    )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

        return self._event(
            cursor,
            event_type.strip(),
            identity,
            timestamp,
            bounded_details,
        )

    def read(
        self,
        job_uids: list[str],
        cursor: int = 0,
        event_types: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Read matching events strictly after ``cursor`` in cursor order."""

        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("cursor must be a non-negative integer")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be positive")
        if limit > MAX_EVENT_RECORDS:
            raise ValueError(f"limit must not exceed {MAX_EVENT_RECORDS}")
        uids = [_string(uid) for uid in job_uids if _string(uid)]
        if not uids:
            return []
        types = None if event_types is None else [
            _string(event_type) for event_type in event_types if _string(event_type)
        ]
        if types == []:
            return []

        placeholders = ",".join("?" for _ in uids)
        query = (
            "SELECT cursor, event_type, job_name, job_namespace, job_uid, "
            "timestamp, details FROM events "
            f"WHERE cursor > ? AND job_uid IN ({placeholders})"
        )
        parameters: list[Any] = [cursor, *uids]
        if types is not None:
            type_placeholders = ",".join("?" for _ in types)
            query += f" AND event_type IN ({type_placeholders})"
            parameters.extend(types)
        query += " ORDER BY cursor ASC LIMIT ?"
        parameters.append(limit)

        with self._lock:
            connection = self._ensure_open()
            rows = connection.execute(query, parameters).fetchall()

        result: list[dict] = []
        for row in rows:
            try:
                details = json.loads(row[6])
            except (TypeError, JSONDecodeError):
                details = {}
            if not isinstance(details, Mapping):
                details = {"value": details}
            result.append(
                self._event(
                    int(row[0]),
                    _string(row[1]),
                    {
                        "name": _string(row[2]),
                        "namespace": _string(row[3]),
                        "uid": _string(row[4]),
                    },
                    _string(row[5]),
                    details,
                )
            )
        return result

    def latest_cursor(self) -> int:
        with self._lock:
            connection = self._ensure_open()
            row = connection.execute(
                "SELECT COALESCE(MAX(cursor), 0) FROM events"
            ).fetchone()
        return int(row[0])

    def oldest_cursor(self) -> int:
        """Return the oldest retained cursor, or zero when the store is empty."""

        with self._lock:
            connection = self._ensure_open()
            row = connection.execute(
                "SELECT COALESCE(MIN(cursor), 0) FROM events"
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._connection.close()


class _WatchError(RuntimeError):
    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class _WatchGone(_WatchError):
    pass


def _watch_code(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def _read_limited(stream: Any, limit: int = 4096) -> bytes:
    """Drain a subprocess stream while keeping diagnostics bounded."""

    captured = bytearray()
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return bytes(captured)
        if len(captured) < limit:
            captured.extend(chunk[: limit - len(captured)])


def _resource_version(value: Mapping[str, Any]) -> str:
    return _string(_metadata(value).get("resourceVersion"))


def _items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        values = payload.get("items") or []
    elif isinstance(payload, list):
        values = payload
    else:
        values = []
    return [dict(value) for value in values if isinstance(value, Mapping)]


def _terminal_type(job: Mapping[str, Any]) -> str | None:
    status = job.get("status")
    if not isinstance(status, Mapping):
        return None
    conditions = status.get("conditions") or []
    if isinstance(conditions, Iterable) and not isinstance(conditions, (str, bytes)):
        for condition in conditions:
            if not isinstance(condition, Mapping):
                continue
            if _string(condition.get("status")).lower() != "true":
                continue
            condition_type = _string(condition.get("type")).lower()
            if condition_type == "complete":
                return "succeeded"
            if condition_type == "failed":
                return "failed"
    if _watch_code(status.get("succeeded")) and _watch_code(status.get("succeeded")) > 0:
        return "succeeded"
    if _watch_code(status.get("failed")) and _watch_code(status.get("failed")) > 0:
        return "failed"
    return None


def _condition_details(job: Mapping[str, Any], event_type: str) -> dict[str, Any]:
    status = job.get("status") if isinstance(job.get("status"), Mapping) else {}
    conditions = status.get("conditions") or []
    selected: dict[str, Any] = {}
    if isinstance(conditions, Iterable) and not isinstance(conditions, (str, bytes)):
        wanted = "Complete" if event_type == "succeeded" else "Failed"
        for condition in conditions:
            if isinstance(condition, Mapping) and _string(condition.get("type")) == wanted:
                for key in ("reason", "message", "lastTransitionTime"):
                    if condition.get(key) is not None:
                        selected[key] = condition[key]
                break
    for key in (
        "active",
        "succeeded",
        "failed",
        "completionTime",
        "startTime",
        "uncountedTerminatedPods",
    ):
        if status.get(key) is not None:
            selected[key] = status[key]
    return _json_safe(selected)


def _job_state_type(job: Mapping[str, Any]) -> str | None:
    if _terminal_type(job) is not None:
        return None
    status = job.get("status") if isinstance(job.get("status"), Mapping) else {}
    active = _watch_code(status.get("active"))
    return "running" if active is not None and active > 0 else "pending"


def _state_details(job: Mapping[str, Any], event_type: str) -> dict[str, Any]:
    status = job.get("status") if isinstance(job.get("status"), Mapping) else {}
    selected: dict[str, Any] = {"state": event_type}
    for key in (
        "active",
        "succeeded",
        "failed",
        "completionTime",
        "startTime",
    ):
        if status.get(key) is not None:
            selected[key] = status[key]
    return _json_safe(selected)


def _owner_job_uid(pod: Mapping[str, Any]) -> str:
    owners = _metadata(pod).get("ownerReferences") or []
    if not isinstance(owners, Iterable) or isinstance(owners, (str, bytes)):
        return ""
    for owner in owners:
        if not isinstance(owner, Mapping):
            continue
        if _string(owner.get("kind")).lower() == "job" and owner.get("uid"):
            return _string(owner.get("uid"))
    return ""


def _container_failures(pod: Mapping[str, Any]) -> list[dict[str, Any]]:
    status = pod.get("status") if isinstance(pod.get("status"), Mapping) else {}
    failures: list[dict[str, Any]] = []
    for field in ("initContainerStatuses", "containerStatuses"):
        values = status.get(field) or []
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            state = value.get("state") if isinstance(value.get("state"), Mapping) else {}
            terminated = state.get("terminated") if isinstance(state, Mapping) else {}
            waiting = state.get("waiting") if isinstance(state, Mapping) else {}
            if not isinstance(terminated, Mapping):
                terminated = {}
            if not isinstance(waiting, Mapping):
                waiting = {}
            reason = terminated.get("reason") or waiting.get("reason")
            message = terminated.get("message") or waiting.get("message")
            if reason or message:
                failures.append(
                    {
                        "name": _string(value.get("name")),
                        "reason": _string(reason),
                        "message": _string(message),
                        "exitCode": terminated.get("exitCode"),
                    }
                )
    return failures


def _pod_details(pod: Mapping[str, Any]) -> dict[str, Any]:
    status = pod.get("status") if isinstance(pod.get("status"), Mapping) else {}
    metadata = _metadata(pod)
    details: dict[str, Any] = {
        "pod": _string(metadata.get("name")),
        "phase": _string(status.get("phase")),
    }
    for key in ("reason", "message", "nodeName"):
        if status.get(key) is not None:
            details[key] = status[key]
    failures = _container_failures(pod)
    if failures:
        details["containers"] = failures
    return _json_safe(details)


def _gpu_requested(job: Mapping[str, Any]) -> bool:
    spec = job.get("spec") if isinstance(job.get("spec"), Mapping) else {}
    template = spec.get("template") if isinstance(spec.get("template"), Mapping) else {}
    pod_spec = template.get("spec") if isinstance(template.get("spec"), Mapping) else {}
    containers: list[Any] = []
    for field in ("containers", "initContainers"):
        values = pod_spec.get(field) or []
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            containers.extend(values)
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        resources = container.get("resources")
        if not isinstance(resources, Mapping):
            continue
        for bucket_name in ("requests", "limits"):
            bucket = resources.get(bucket_name)
            if not isinstance(bucket, Mapping):
                continue
            for resource_name, amount in bucket.items():
                resource_name_text = _string(resource_name).lower()
                if "gpu" not in resource_name_text:
                    continue
                try:
                    if float(amount) > 0:
                        return True
                except (TypeError, ValueError):
                    if _string(amount).strip() not in {"", "0", "0.0"}:
                        return True
    return False


class Monitor:
    """Observe a selected set of Jobs without polling normal status.

    ``watcher`` is intentionally a keyword-only seam for tests and for
    embedders that already have a Kubernetes watch implementation.  It may be
    an async-generator factory accepting ``(resource, namespace,
    resource_version)`` or an object with ``watch_json``.  With no watcher,
    this class uses the local executable configured on ``KubernetesClient``.
    """

    def __init__(
        self,
        client: KubernetesClient,
        namespace: str,
        store: EventStore,
        metrics_provider: Callable[[list[str]], dict] | None = None,
        *,
        watcher: Any = None,
        watch_source: Any = None,
        list_provider: Any = None,
    ) -> None:
        if not namespace:
            raise ValueError("Monitor namespace must not be empty")
        if watcher is not None and watch_source is not None:
            raise ValueError("pass watcher or watch_source, not both")
        self.client = client
        self.namespace = namespace
        self.store = store
        self.metrics_provider = metrics_provider
        self._watcher = watcher if watcher is not None else watch_source
        self._list_provider = list_provider

        self._condition = asyncio.Condition()
        self._lifecycle_lock = asyncio.Lock()
        self._tracked: dict[str, dict[str, Any]] = {}
        self._uid_to_key: dict[str, str] = {}
        self._pod_to_job: dict[str, str] = {}
        self._active_waiters: dict[int, set[str]] = {}
        self._waiter_number = 0
        self._watch_tasks: set[asyncio.Task[Any]] = set()
        self._metrics_task: asyncio.Task[Any] | None = None
        self._processes: set[asyncio.subprocess.Process] = set()
        self._resource_versions: dict[str, str] = {}
        self._health_failures: set[str] = set()
        self._seen_pod_events: set[tuple[Any, ...]] = set()
        self._seen_kubernetes_events: set[tuple[Any, ...]] = set()
        self._reported_cursor_gaps: set[tuple[int, int]] = set()
        self._risk_state: dict[str, bool] = {}
        # A newly-created monitor has no continuity with a cursor from an
        # earlier process.  Durable history is the evidence that an earlier
        # observation epoch existed; an empty store has no missed cursor to
        # report.  A cursor-less registration establishes a fresh epoch.
        initial_cursor = self.store.latest_cursor()
        self._restart_gap_pending = initial_cursor > 0
        self._last_observed_cursor = initial_cursor
        self._closed = False
        self._started = False
        self._metrics_interval = METRICS_INTERVAL

    @staticmethod
    def _job_key(job: Mapping[str, Any]) -> str:
        identity = _job_identity(job)
        if identity["uid"]:
            return f"uid:{identity['uid']}"
        return f"name:{identity['namespace']}/{identity['name']}"

    def _track(self, job: Mapping[str, Any], waiter: int) -> str:
        identity = _job_identity(job)
        key = self._job_key(job)
        entry = self._tracked.get(key)
        if entry is None:
            entry = {
                "identity": identity,
                "source": dict(job),
                "refs": set(),
                "emitted": set(),
                "terminal": set(),
                "state": None,
                "retained": False,
            }
            self._tracked[key] = entry
        else:
            entry["identity"] = identity
            entry["source"] = dict(job)
        entry["refs"].add(waiter)
        if identity["uid"]:
            self._uid_to_key[identity["uid"]] = key
        return key

    def _entry_for_job_object(self, obj: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
        metadata = _metadata(obj)
        uid = _string(metadata.get("uid") or obj.get("uid"))
        key = self._uid_to_key.get(uid) if uid else None
        if key is None:
            name = _string(metadata.get("name") or obj.get("name"))
            namespace = _string(
                metadata.get("namespace") or obj.get("namespace") or self.namespace
            )
            for candidate, entry in self._tracked.items():
                identity = entry["identity"]
                if (
                    not identity["uid"]
                    and identity["name"] == name
                    and identity["namespace"] == namespace
                ):
                    key = candidate
                    break
        if key is None or key not in self._tracked:
            return None
        return key, self._tracked[key]

    def _entry_for_uid(self, uid: str) -> tuple[str, dict[str, Any]] | None:
        key = self._uid_to_key.get(uid)
        if key is None:
            return None
        entry = self._tracked.get(key)
        return (key, entry) if entry is not None else None

    @staticmethod
    def _adapt_call(
        function: Callable[..., Any],
        positional: list[Any],
        keyword_values: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        """Adapt seams used by small fake clients without masking call errors."""

        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            return positional, keyword_values
        parameters = list(signature.parameters.values())
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        positional_parameters = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if accepts_kwargs:
            return positional, keyword_values
        names = {parameter.name for parameter in parameters}
        kwargs = {key: value for key, value in keyword_values.items() if key in names}
        if kwargs:
            # The first positional argument is the resource for watch/list
            # seams; keyword-only namespace/RV parameters remain keywords.
            return positional[:1], kwargs
        return positional[: len(positional_parameters)], {}

    async def _invoke_async_source(
        self,
        source: Callable[..., Any],
        resource: str,
        resource_version: str | None,
    ) -> Any:
        positional, keyword = self._adapt_call(
            source,
            [resource, self.namespace, resource_version],
            {
                "namespace": self.namespace,
                "resource_version": resource_version,
                "resourceVersion": resource_version,
                "rv": resource_version,
            },
        )
        result = source(*positional, **keyword)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _invoke_list_source(self, function: Callable[..., Any], resource: str) -> Any:
        positional, keyword = self._adapt_call(
            function,
            [resource, self.namespace],
            {"namespace": self.namespace},
        )
        if inspect.iscoroutinefunction(function):
            result = function(*positional, **keyword)
        else:
            result = await asyncio.to_thread(function, *positional, **keyword)
        if inspect.isawaitable(result):
            result = await result
        return result

    def _has_list_support(self) -> bool:
        if self._list_provider is not None:
            return True
        return callable(getattr(self.client, "list_json", None)) or any(
            callable(getattr(self.client, name, None))
            for name in ("list_jobs", "list_pods", "events")
        )

    async def _list_resource(self, resource: str) -> Any:
        if self._list_provider is not None:
            function = self._list_provider
            return await self._invoke_list_source(function, resource)

        list_json = getattr(self.client, "list_json", None)
        if callable(list_json):
            return await self._invoke_list_source(list_json, resource)

        method_name = {
            "jobs.batch": "list_jobs",
            "pods": "list_pods",
        }.get(resource)
        if method_name:
            function = getattr(self.client, method_name, None)
            if callable(function):
                return await self._invoke_list_source(function, resource)
        if resource == "events":
            function = getattr(self.client, "events", None)
            if callable(function):
                values = await self._invoke_list_source(function, resource)
                return {"items": values if isinstance(values, list) else []}
        raise RuntimeError(f"Kubernetes client cannot list {resource}")

    async def _safe_list_resource(self, resource: str) -> Any:
        try:
            payload = await self._list_resource(resource)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as monitor event
            await self._mark_unavailable(
                resource,
                {
                    "operation": "list",
                    "error": _string(exc),
                },
            )
            return None
        await self._mark_recovered(resource)
        return payload

    async def _refresh_baseline(self, keys: set[str]) -> None:
        if not self._has_list_support():
            return
        resources = ("jobs.batch", "pods", "events")
        payloads = await asyncio.gather(
            *(self._safe_list_resource(resource) for resource in resources)
        )
        for resource, payload in zip(resources, payloads):
            if payload is None:
                continue
            if isinstance(payload, Mapping):
                version = _string((payload.get("metadata") or {}).get("resourceVersion"))
                if version:
                    self._resource_versions[resource] = version
            values = _items(payload)
            if resource == "jobs.batch":
                await self._reconcile_job_list(values, set(self._tracked) | keys)
            elif resource == "pods":
                for pod in values:
                    await self._handle_pod("ADDED", pod)
            else:
                for event in values:
                    await self._handle_kubernetes_event("ADDED", event)

    async def _reconcile_job_list(
        self,
        jobs: list[Mapping[str, Any]],
        target_keys: set[str] | None = None,
    ) -> None:
        """Reconcile selected Jobs against a successful namespace snapshot."""

        keys = set(self._tracked) if target_keys is None else target_keys
        seen_uids = {
            identity["uid"]
            for identity in (_job_identity(job) for job in jobs)
            if identity["uid"]
        }
        for job in jobs:
            found = self._entry_for_job_object(job)
            if found is None or found[0] not in keys:
                continue
            found[1]["source"] = dict(job)
            await self._handle_job("ADDED", job)

        for key in keys:
            entry = self._tracked.get(key)
            if entry is None:
                continue
            uid = entry["identity"]["uid"]
            if uid and uid not in seen_uids:
                await self._emit_job_once(
                    key,
                    "deleted",
                    {
                        "name": entry["identity"]["name"],
                        "reason": "Job absent from namespace list",
                    },
                )

    async def _reconcile_initial_jobs(
        self,
        jobs: list[Mapping[str, Any]],
    ) -> None:
        for job in jobs:
            found = self._entry_for_job_object(job)
            if found is None:
                continue
            key, entry = found
            entry["source"] = dict(job)
            uid = entry["identity"]["uid"]
            if uid:
                existing = self.store.read(
                    [uid],
                    cursor=0,
                    event_types=list(_TERMINAL_TYPES),
                    limit=MAX_EVENT_RECORDS,
                )
                entry["terminal"] = {
                    event["type"] for event in existing if event["type"] in _TERMINAL_TYPES
                }
                # Existing terminal records are already emitted.  Seed both
                # sets so a fresh waiter receives terminal replay without
                # appending a second durable status event.
                entry["emitted"].update(entry["terminal"])
            if "deleted" in entry["emitted"]:
                continue
            terminal = _terminal_type(job)
            if terminal:
                await self._emit_job_once(
                    key,
                    terminal,
                    _condition_details(job, terminal),
                )
            else:
                state_type = _job_state_type(job)
                if state_type and state_type != entry.get("state"):
                    entry["state"] = state_type
                    await self._emit(
                        state_type,
                        _state_details(job, state_type),
                        key=key,
                    )

    async def _start_background(self) -> None:
        if self._started or self._closed:
            return
        self._started = True
        for resource in ("jobs.batch", "pods", "events"):
            task = asyncio.create_task(
                self._watch_loop(resource),
                name=f"falcon-mcp-watch-{resource}",
            )
            self._watch_tasks.add(task)
        if self.metrics_provider is not None:
            self._metrics_task = asyncio.create_task(
                self._metrics_loop(),
                name="falcon-mcp-metrics",
            )

    async def _stop_background(self) -> None:
        if self._started and not self._closed:
            self._restart_gap_pending = True
            self._last_observed_cursor = self.store.latest_cursor()
        tasks = list(self._watch_tasks)
        if self._metrics_task is not None:
            tasks.append(self._metrics_task)
        if not tasks:
            self._started = False
            return
        for task in tasks:
            if not task.done():
                task.cancel()
        await self._terminate_processes()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._watch_tasks.clear()
        self._metrics_task = None
        self._started = False

    async def _register(
        self,
        jobs: list[Mapping[str, Any]],
        *,
        report_restart_gap: bool,
    ) -> tuple[int, set[str], bool]:
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Monitor is closed")
            self._waiter_number += 1
            waiter = self._waiter_number
            keys = {self._track(job, waiter) for job in jobs}
            self._active_waiters[waiter] = keys
            restart_gap = report_restart_gap and self._restart_gap_pending
            if restart_gap:
                self._restart_gap_pending = False
            elif not report_restart_gap:
                # A cursor-less wait intentionally starts a new current
                # observation epoch; it does not claim continuity from an
                # earlier monitor process or an explicitly stopped monitor.
                self._restart_gap_pending = False
            try:
                await self._refresh_baseline(keys)
                await self._reconcile_initial_jobs(jobs)
                await self._start_background()
            except BaseException:
                self._active_waiters.pop(waiter, None)
                for key in keys:
                    entry = self._tracked.get(key)
                    if entry is None:
                        continue
                    entry["refs"].discard(waiter)
                    if entry["refs"] or entry.get("retained"):
                        continue
                    identity = entry["identity"]
                    if identity["uid"]:
                        self._uid_to_key.pop(identity["uid"], None)
                    self._tracked.pop(key, None)
                    self._risk_state.pop(key, None)
                if not self._has_observation_targets() and not self._closed:
                    await self._stop_background()
                raise
            return waiter, keys, restart_gap

    async def _unregister(self, waiter: int, *, retain: bool) -> None:
        async with self._lifecycle_lock:
            keys = self._active_waiters.pop(waiter, set())
            for key in keys:
                entry = self._tracked.get(key)
                if entry is None:
                    continue
                entry["refs"].discard(waiter)
                if retain:
                    entry["retained"] = True
                    continue
                if entry["refs"] or entry.get("retained"):
                    continue
                identity = entry["identity"]
                if identity["uid"]:
                    self._uid_to_key.pop(identity["uid"], None)
                self._tracked.pop(key, None)
                self._risk_state.pop(key, None)
                for pod_uid, pod_key in list(self._pod_to_job.items()):
                    if pod_key == key:
                        self._pod_to_job.pop(pod_uid, None)
            if not self._has_observation_targets() and not self._closed:
                await self._stop_background()

    async def _notify(self) -> None:
        async with self._condition:
            self._condition.notify_all()

    def _targets(self, key: str | None) -> list[tuple[str, dict[str, Any]]]:
        if key is not None:
            entry = self._tracked.get(key)
            return [(key, entry)] if entry is not None else []
        return list(self._tracked.items())

    def _has_observation_targets(self) -> bool:
        return any(
            bool(entry["refs"]) or bool(entry.get("retained"))
            for entry in self._tracked.values()
        )

    async def _emit(
        self,
        event_type: str,
        details: Mapping[str, Any],
        *,
        key: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._closed:
            return []
        events: list[dict[str, Any]] = []
        for target_key, entry in self._targets(key):
            event = self.store.append(
                dict(entry["identity"]),
                event_type,
                dict(details),
            )
            events.append(event)
        if events:
            await self._notify()
        return events

    async def _emit_job_once(
        self,
        key: str,
        event_type: str,
        details: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        entry = self._tracked.get(key)
        if entry is None:
            return []
        emitted = entry["emitted"]
        if event_type in emitted:
            return []
        emitted.add(event_type)
        if event_type in _TERMINAL_TYPES:
            entry["terminal"].add(event_type)
        try:
            return await self._emit(event_type, details, key=key)
        except BaseException:
            emitted.discard(event_type)
            if event_type in _TERMINAL_TYPES:
                entry["terminal"].discard(event_type)
            raise

    async def _mark_unavailable(self, source: str, details: Mapping[str, Any]) -> None:
        if source in self._health_failures or self._closed:
            return
        was_healthy = not self._health_failures
        self._health_failures.add(source)
        if was_healthy:
            await self._emit(
                "monitor_unavailable",
                {"source": source, **dict(details)},
            )

    async def _mark_recovered(self, source: str) -> None:
        if source not in self._health_failures or self._closed:
            return
        self._health_failures.remove(source)
        if not self._health_failures:
            await self._emit(
                "monitor_recovered",
                {"source": source},
            )

    async def _handle_job(self, action: str, job: Mapping[str, Any]) -> None:
        found = self._entry_for_job_object(job)
        if found is None:
            return
        key, entry = found
        if action == "DELETED":
            await self._emit_job_once(
                key,
                "deleted",
                {
                    "name": entry["identity"]["name"],
                    "reason": "Job object deleted",
                },
            )
            return
        entry["source"] = dict(job)
        terminal = _terminal_type(job)
        if terminal:
            await self._emit_job_once(
                key,
                terminal,
                _condition_details(job, terminal),
            )
            return
        state_type = _job_state_type(job)
        if state_type and state_type != entry.get("state"):
            entry["state"] = state_type
            await self._emit(state_type, _state_details(job, state_type), key=key)

    async def _handle_pod(self, action: str, pod: Mapping[str, Any]) -> None:
        owner_uid = _owner_job_uid(pod)
        found = self._entry_for_uid(owner_uid)
        if found is None:
            return
        key, _entry = found
        metadata = _metadata(pod)
        pod_uid = _string(metadata.get("uid"))
        if pod_uid:
            self._pod_to_job[pod_uid] = key
        if action == "DELETED":
            return
        status = pod.get("status") if isinstance(pod.get("status"), Mapping) else {}
        phase = _string(status.get("phase"))
        reason = _string(status.get("reason"))
        if phase != "Failed" and "evict" not in reason.lower():
            return
        event_type = "evicted" if "evict" in reason.lower() else "pod_failed"
        fingerprint = (
            event_type,
            pod_uid or _string(metadata.get("name")),
            phase,
            reason,
            _string(status.get("message")),
        )
        if fingerprint in self._seen_pod_events:
            return
        self._seen_pod_events.add(fingerprint)
        await self._emit(event_type, _pod_details(pod), key=key)

    def _event_job_key(self, event: Mapping[str, Any]) -> str | None:
        involved = event.get("involvedObject") or event.get("regarding")
        if not isinstance(involved, Mapping):
            return None
        uid = _string(involved.get("uid"))
        direct = self._entry_for_uid(uid) if uid else None
        if direct is not None:
            return direct[0]
        kind = _string(involved.get("kind")).lower()
        if kind == "pod" and uid in self._pod_to_job:
            return self._pod_to_job[uid]
        if kind == "job":
            name = _string(involved.get("name"))
            for key, entry in self._tracked.items():
                identity = entry["identity"]
                if identity["name"] == name and (
                    not involved.get("namespace")
                    or identity["namespace"] == _string(involved.get("namespace"))
                ):
                    return key
        return None

    async def _handle_kubernetes_event(
        self,
        action: str,
        event: Mapping[str, Any],
    ) -> None:
        if action == "DELETED":
            return
        key = self._event_job_key(event)
        if key is None:
            return
        event_kind = _string(event.get("type"))
        reason = _string(event.get("reason"))
        message = _string(event.get("note") or event.get("message"))
        reason_lower = reason.lower()
        message_lower = message.lower()
        meaningful_reasons = (
            "failedscheduling",
            "failedmount",
            "failedattachvolume",
            "errimagepull",
            "imagepullbackoff",
            "backoff",
            "failedcreatepodsandbox",
            "invalidimage",
        )
        meaningful = (
            event_kind.lower() == "warning"
            or reason_lower in meaningful_reasons
            or "imagepull" in reason_lower
            or "scheduling" in reason_lower
            or "image pull" in message_lower
            or "failed to schedule" in message_lower
        )
        if not meaningful or not (reason or message):
            return
        metadata = _metadata(event)
        fingerprint = (
            _string(metadata.get("uid")),
            _string(metadata.get("resourceVersion")),
            reason,
            message,
            _string(event.get("lastTimestamp") or event.get("eventTime")),
            _string(event.get("count")),
        )
        if fingerprint in self._seen_kubernetes_events:
            return
        self._seen_kubernetes_events.add(fingerprint)
        details: dict[str, Any] = {
            "reason": reason,
            "message": message,
            "severity": event_kind or "Warning",
        }
        for field in (
            "count",
            "firstTimestamp",
            "lastTimestamp",
            "eventTime",
            "source",
            "reportingComponent",
        ):
            if event.get(field) is not None:
                details[field] = event[field]
        involved = event.get("involvedObject") or event.get("regarding")
        if isinstance(involved, Mapping):
            details["involvedObject"] = {
                key_name: involved[key_name]
                for key_name in ("kind", "name", "namespace", "uid")
                if involved.get(key_name) is not None
            }
        await self._emit("warning", _json_safe(details), key=key)

    @staticmethod
    def _watch_payload(raw: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
        if isinstance(raw.get("object"), Mapping):
            return _string(raw.get("type") or "MODIFIED").upper(), raw["object"]
        return _string(raw.get("type") or "MODIFIED").upper(), raw

    async def _watch_json(
        self,
        resource: str,
        resource_version: str | None = None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        source = self._watcher
        if source is None:
            client_source = getattr(self.client, "watch_json", None)
            if callable(client_source):
                source = client_source
        if source is not None:
            if not callable(source) and callable(getattr(source, "watch_json", None)):
                source = source.watch_json
            if isinstance(source, Mapping):
                source = source.get(resource)
            if source is None:
                raise _WatchError(f"watch source for {resource} is missing")
            if callable(source):
                result = await self._invoke_async_source(
                    source, resource, resource_version
                )
            else:
                result = source
                if inspect.isawaitable(result):
                    result = await result
            if hasattr(result, "__aiter__"):
                try:
                    async for item in result:
                        if isinstance(item, Mapping):
                            yield item
                    return
                finally:
                    close = getattr(result, "aclose", None)
                    if callable(close):
                        await close()
            if isinstance(result, Iterable) and not isinstance(result, (str, bytes, Mapping)):
                for item in result:
                    if isinstance(item, Mapping):
                        yield item
                return
            raise _WatchError(f"watch source for {resource} is not iterable")

        result = self._subprocess_watch_json(resource, resource_version)
        try:
            async for item in result:
                yield item
        finally:
            close = getattr(result, "aclose", None)
            if callable(close):
                await close()

    def _client_argv(self, args: list[str]) -> list[str]:
        command_argv = getattr(self.client, "command_argv", None)
        if callable(command_argv):
            value = command_argv(args)
            if value:
                return [str(item) for item in value]
        argv_builder = getattr(self.client, "_argv", None)
        if callable(argv_builder):
            try:
                value = argv_builder(args)
            except TypeError:
                value = argv_builder(*args)
            if value:
                return [str(item) for item in value]
        executable = _string(getattr(self.client, "executable", "kubectl")) or "kubectl"
        argv = [executable, *args]
        context = getattr(self.client, "context", None) or getattr(
            self.client, "_context", None
        )
        if context:
            argv.extend(["--context", _string(context)])
        return argv

    @staticmethod
    def _decoder_objects(
        decoder: JSONDecoder,
        buffer: str,
        *,
        final: bool = False,
    ) -> tuple[list[Mapping[str, Any]], str]:
        objects: list[Mapping[str, Any]] = []
        while True:
            stripped = buffer.lstrip()
            if not stripped:
                return objects, ""
            try:
                value, end = decoder.raw_decode(stripped)
            except JSONDecodeError:
                if final:
                    raise _WatchError("kubectl watch ended with incomplete JSON")
                return objects, stripped
            buffer = stripped[end:]
            if isinstance(value, Mapping):
                objects.append(value)

    async def _subprocess_watch_json(
        self,
        resource: str,
        resource_version: str | None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        args = [
            "get",
            resource,
            "--namespace",
            self.namespace,
            "--watch",
            "--output-watch-events",
            "--output",
            "json",
        ]
        if resource_version:
            args.extend(["--resource-version", resource_version])
        argv = self._client_argv(args)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise _WatchError(f"could not start {argv[0]!r}: executable not found") from exc
        except OSError as exc:
            raise _WatchError(f"could not start Kubernetes watch: {exc}") from exc

        self._processes.add(process)
        stderr_task = (
            asyncio.create_task(_read_limited(process.stderr))
            if process.stderr
            else None
        )
        decoder = JSONDecoder()
        utf8 = codecs.getincrementaldecoder("utf-8")()
        buffer = ""
        try:
            assert process.stdout is not None
            while True:
                chunk = await process.stdout.read(4096)
                if not chunk:
                    break
                buffer += utf8.decode(chunk, final=False)
                objects, buffer = self._decoder_objects(decoder, buffer)
                for value in objects:
                    yield value
            buffer += utf8.decode(b"", final=True)
            objects, buffer = self._decoder_objects(decoder, buffer, final=True)
            for value in objects:
                yield value
            returncode = await process.wait()
            stderr = ""
            if stderr_task is not None:
                stderr = (await stderr_task).decode("utf-8", errors="replace")
            if returncode:
                code = 410 if "410" in stderr or "too old resource version" in stderr.lower() else returncode
                if code == 410:
                    raise _WatchGone(stderr.strip() or "watch resource version is gone", code=410)
                raise _WatchError(
                    stderr.strip() or f"kubectl watch exited {returncode}",
                    code=code,
                )
            raise _WatchError("kubectl watch stream ended")
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
            if stderr_task is not None and not stderr_task.done():
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
            self._processes.discard(process)

    async def _terminate_processes(self) -> None:
        processes = list(self._processes)
        for process in processes:
            if process.returncode is not None:
                continue
            try:
                process.terminate()
            except ProcessLookupError:
                continue
        if processes:
            await asyncio.gather(
                *(
                    asyncio.wait_for(process.wait(), timeout=2.0)
                    for process in processes
                    if process.returncode is None
                ),
                return_exceptions=True,
            )
        for process in processes:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

    async def _load_for_watch(self, resource: str) -> None:
        if not self._has_list_support():
            return
        payload = await self._safe_list_resource(resource)
        if payload is None:
            return
        version = _string((payload.get("metadata") or {}).get("resourceVersion")) if isinstance(payload, Mapping) else ""
        if version:
            self._resource_versions[resource] = version
        values = _items(payload)
        if resource == "jobs.batch":
            await self._reconcile_job_list(values)
        elif resource == "pods":
            for value in values:
                await self._handle_pod("ADDED", value)
        else:
            for value in values:
                await self._handle_kubernetes_event("ADDED", value)

    async def _watch_loop(self, resource: str) -> None:
        delay = 0.25
        while not self._closed:
            if not self._has_observation_targets():
                return
            source: Any = None
            try:
                if not self._resource_versions.get(resource):
                    await self._load_for_watch(resource)
                source = self._watch_json(resource, self._resource_versions.get(resource))
                if inspect.isawaitable(source):
                    source = await source
                yielded = False
                async for raw in source:
                    if self._closed:
                        return
                    if not isinstance(raw, Mapping):
                        continue
                    yielded = True
                    action, obj = self._watch_payload(raw)
                    if action == "ERROR":
                        code = _watch_code(obj.get("code") or obj.get("status"))
                        if code == 410:
                            raise _WatchGone(
                                _string(obj.get("message") or obj.get("reason") or "watch gap"),
                                code=410,
                            )
                        raise _WatchError(
                            _string(obj.get("message") or obj.get("reason") or "watch error"),
                            code=code,
                        )
                    await self._mark_recovered(resource)
                    version = _resource_version(obj)
                    if version:
                        self._resource_versions[resource] = version
                    if action == "BOOKMARK":
                        continue
                    if resource == "jobs.batch":
                        await self._handle_job(action, obj)
                    elif resource == "pods":
                        await self._handle_pod(action, obj)
                    else:
                        await self._handle_kubernetes_event(action, obj)
                if self._closed:
                    return
                raise _WatchError(
                    f"{resource} watch stream ended"
                    + (" after an event" if yielded else "")
                )
            except asyncio.CancelledError:
                raise
            except _WatchGone as exc:
                self._resource_versions.pop(resource, None)
                await self._mark_unavailable(
                    resource,
                    {"operation": "watch", "gap": True, "error": _string(exc)},
                )
                # Re-list immediately to obtain a current resourceVersion.
                if self._closed or not self._has_observation_targets():
                    return
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 30.0)
                continue
            except Exception as exc:  # noqa: BLE001 - watcher must stay alive
                await self._mark_unavailable(
                    resource,
                    {"operation": "watch", "error": _string(exc)},
                )
                if self._closed or not self._has_observation_targets():
                    return
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 30.0)
            else:
                delay = 0.25
            finally:
                if source is not None and hasattr(source, "aclose"):
                    try:
                        await source.aclose()
                    except Exception:  # noqa: BLE001 - cleanup must not mask watch error
                        pass

    async def _metrics_loop(self) -> None:
        while not self._closed and self._has_observation_targets():
            await asyncio.sleep(self._metrics_interval)
            if self._closed or not self._has_observation_targets():
                return
            active: list[tuple[str, str, str]] = []
            for key, entry in self._tracked.items():
                source = entry.get("source") or {}
                if "deleted" in entry["emitted"]:
                    continue
                if _terminal_type(source):
                    continue
                if not _gpu_requested(source):
                    continue
                uid = entry["identity"]["uid"]
                name = entry["identity"]["name"]
                if uid and name:
                    # Identify samples by UID; retain name fallback for legacy providers.
                    active.append((key, uid, name))
            if not active or self.metrics_provider is None:
                continue
            identifiers = [uid for _key, uid, _name in active]
            try:
                if inspect.iscoroutinefunction(self.metrics_provider):
                    result = self.metrics_provider(identifiers)
                else:
                    result = await asyncio.to_thread(self.metrics_provider, identifiers)
                if inspect.isawaitable(result):
                    result = await result
                if not isinstance(result, Mapping):
                    raise TypeError("metrics provider must return a mapping")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced as one edge event
                await self._mark_unavailable(
                    "metrics",
                    {"operation": "metrics", "error": _string(exc)},
                )
                continue

            metrics_unavailable = False
            for key, uid, name in active:
                entry = self._tracked.get(key)
                if entry is None:
                    continue
                value = result.get(uid)
                if value is None:
                    value = result.get(name)
                if value is None:
                    metrics_unavailable = True
                    await self._mark_unavailable(
                        "metrics",
                        {
                            "operation": "metrics",
                            "job": name,
                            "error": "risk sample unavailable",
                        },
                    )
                    continue
                if isinstance(value, Mapping):
                    if value.get("available") is False or value.get("at_risk") is None:
                        metrics_unavailable = True
                        await self._mark_unavailable(
                            "metrics",
                            {
                                "operation": "metrics",
                                "job": name,
                                "error": value.get("error") or "risk unavailable",
                            },
                        )
                        continue
                    at_risk = bool(value.get("at_risk"))
                elif isinstance(value, bool):
                    at_risk = value
                else:
                    metrics_unavailable = True
                    await self._mark_unavailable(
                        "metrics",
                        {
                            "operation": "metrics",
                            "job": name,
                            "error": "invalid risk sample",
                        },
                    )
                    continue
                previous = self._risk_state.get(key)
                self._risk_state[key] = at_risk
                if previous is None:
                    if at_risk:
                        await self._emit(
                            "eviction_risk",
                            {"at_risk": True, "source": "metrics"},
                            key=key,
                        )
                elif previous != at_risk:
                    await self._emit(
                        "eviction_risk" if at_risk else "eviction_risk_cleared",
                        {"at_risk": at_risk, "source": "metrics"},
                        key=key,
                    )
            if not metrics_unavailable:
                await self._mark_recovered("metrics")

    def _allowed_types(self, event_types: list[str] | None) -> list[str]:
        if event_types is None:
            return list(ACTIONABLE_EVENT_TYPES)
        values = [_string(value) for value in event_types if _string(value)]
        unsupported = sorted(set(values).difference(SUPPORTED_EVENT_TYPES))
        if unsupported:
            raise ValueError(
                "unsupported event type(s): " + ", ".join(unsupported)
            )
        return values

    def _terminal_replays(
        self,
        keys: set[str],
        start_cursor: int,
        allowed: list[str],
    ) -> list[dict[str, Any]]:
        if not keys or not _TERMINAL_TYPES.intersection(allowed):
            return []
        uids = [
            self._tracked[key]["identity"]["uid"]
            for key in keys
            if key in self._tracked and self._tracked[key]["identity"]["uid"]
        ]
        if not uids:
            return []
        records = self.store.read(
            uids,
            cursor=0,
            event_types=[event_type for event_type in allowed if event_type in _TERMINAL_TYPES],
            limit=MAX_EVENT_RECORDS,
        )
        latest: dict[str, dict[str, Any]] = {}
        for record in records:
            if record["cursor"] > start_cursor:
                continue
            uid = record["job"]["uid"]
            if uid in uids:
                latest[uid] = record
        return sorted(latest.values(), key=lambda event: event["cursor"])

    @staticmethod
    def _merge_events(
        first: list[dict[str, Any]],
        second: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[Any, ...], dict[str, Any]] = {}
        for event in [*first, *second]:
            identity = (
                event.get("cursor"),
                event.get("type"),
                (event.get("job") or {}).get("uid"),
            )
            merged[identity] = event
        return sorted(merged.values(), key=lambda event: int(event.get("cursor", 0)))[:50]

    @staticmethod
    def _fit_wait_events(
        events: list[dict[str, Any]],
        start_cursor: int,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for event in events:
            candidate = [*selected, event]
            payload = {
                "events": candidate,
                "cursor": max(start_cursor, int(event["cursor"])),
                "timed_out": False,
            }
            if selected and _json_size(payload) > MAX_WAIT_RESPONSE_BYTES:
                break
            selected.append(event)
        return selected

    async def _report_cursor_gap(self, cursor: int, keys: set[str]) -> None:
        oldest = self.store.oldest_cursor()
        if not oldest or cursor + 1 >= oldest:
            return
        signature = (cursor, oldest)
        if signature in self._reported_cursor_gaps:
            return
        self._reported_cursor_gaps.add(signature)
        # A gap is itself a context event.  Store it against every selected
        # Job so the normal job-UID read path can surface it to the caller.
        for key in keys:
            await self._emit(
                "monitor_unavailable",
                {
                    "source": "event_store",
                    "gap": True,
                    "cursor": cursor,
                    "oldest_cursor": oldest,
                },
                key=key,
            )

    async def _report_restart_gap(self, cursor: int, keys: set[str]) -> None:
        if not keys:
            return
        for key in keys:
            await self._emit(
                "monitor_unavailable",
                {
                    "source": "monitor",
                    "gap": True,
                    "reason": "watch_restarted",
                    "cursor": cursor,
                    "last_observed_cursor": self._last_observed_cursor,
                },
                key=key,
            )

    async def wait(
        self,
        jobs: list[dict],
        cursor: int | None = None,
        event_types: list[str] | None = None,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
    ) -> dict:
        """Wait for selected actionable events, or explicitly time out.

        A wait with ``cursor=None`` starts at the current store watermark.  A
        currently terminal Job is nevertheless replayed once for that waiter,
        which makes a fresh MCP call useful even if the Job completed before
        the call registered its watches.
        """

        if not isinstance(jobs, list):
            raise TypeError("jobs must be a list")
        if cursor is not None and (
            isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0
        ):
            raise ValueError("cursor must be None or a non-negative integer")
        if timeout < 0:
            raise ValueError("timeout must not be negative")
        normalized_jobs = [
            job for job in jobs if isinstance(job, Mapping)
        ]
        allowed = self._allowed_types(event_types)
        start_cursor = self.store.latest_cursor() if cursor is None else cursor
        if not normalized_jobs or not allowed:
            return {
                "events": [],
                "cursor": max(start_cursor, self.store.latest_cursor()),
                "timed_out": True,
            }
        waiter: int | None = None
        normal_return = False
        try:
            waiter, keys, restart_gap = await self._register(
                normalized_jobs,
                report_restart_gap=cursor is not None,
            )
            deadline = asyncio.get_running_loop().time() + float(timeout)
            if cursor is not None:
                if restart_gap:
                    await self._report_restart_gap(start_cursor, keys)
                await self._report_cursor_gap(start_cursor, keys)
            replay = self._terminal_replays(keys, start_cursor, allowed) if cursor is None else []
            job_uids = [
                self._tracked[key]["identity"]["uid"]
                for key in keys
                if key in self._tracked and self._tracked[key]["identity"]["uid"]
            ]
            while True:
                events = self.store.read(
                    job_uids,
                    cursor=start_cursor,
                    event_types=allowed,
                    limit=50,
                )
                events = self._merge_events(replay, events)
                events = self._fit_wait_events(events, start_cursor)
                replay = []
                if events:
                    normal_return = True
                    return {
                        "events": events,
                        # Do not advance over a matching backlog larger than
                        # the read limit.  The caller must be able to pass
                        # this cursor back and receive the next page.
                        "cursor": max(start_cursor, int(events[-1]["cursor"])),
                        "timed_out": False,
                    }
                remaining = deadline - asyncio.get_running_loop().time()
                latest = max(start_cursor, self.store.latest_cursor())
                if self._closed or remaining <= 0:
                    normal_return = True
                    return {"events": [], "cursor": latest, "timed_out": True}
                async with self._condition:
                    # Recheck while holding the condition to close the
                    # append-vs-wait race without polling.
                    events = self.store.read(
                        job_uids,
                        cursor=start_cursor,
                        event_types=allowed,
                        limit=50,
                    )
                    events = self._fit_wait_events(events, start_cursor)
                    if events:
                        normal_return = True
                        return {
                            "events": events,
                            "cursor": max(start_cursor, int(events[-1]["cursor"])),
                            "timed_out": False,
                        }
                    try:
                        await asyncio.wait_for(self._condition.wait(), remaining)
                    except asyncio.TimeoutError:
                        normal_return = True
                        return {
                            "events": [],
                            "cursor": max(start_cursor, self.store.latest_cursor()),
                            "timed_out": True,
                        }
        finally:
            if waiter is not None:
                with anyio.CancelScope(shield=True):
                    await self._unregister(
                        waiter,
                        retain=normal_return and not self._closed,
                    )

    async def close(self) -> None:
        """Stop all watches and await every owned task."""

        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            async with self._condition:
                self._condition.notify_all()
            tasks = list(self._watch_tasks)
            if self._metrics_task is not None:
                tasks.append(self._metrics_task)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await self._terminate_processes()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._watch_tasks.clear()
            self._metrics_task = None
            self._started = False


__all__ = [
    "ACTIONABLE_EVENT_TYPES",
    "DEFAULT_WAIT_TIMEOUT",
    "EventStore",
    "MAX_EVENT_DETAILS_BYTES",
    "MAX_EVENT_RECORDS",
    "MAX_WAIT_RESPONSE_BYTES",
    "METRICS_INTERVAL",
    "Monitor",
    "SUPPORTED_EVENT_TYPES",
]
