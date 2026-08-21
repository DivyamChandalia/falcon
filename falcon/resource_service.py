"""Shared, read-only Falcon cluster resource service and client.

The wire format deliberately mirrors :mod:`falcon.cluster` instead of
introducing a second planning model.  The server is the only component that
scrapes kube-state-metrics and writes allocation history; clients either use
this service exclusively or opt out by configuring ``resource_service_url``
as null.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
import queue
import signal
import socket
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlsplit

import httpx

from .cluster import (
    AttemptSummary,
    ClusterSnapshot,
    JobSnapshot,
    NodeSnapshot,
    NodeTaint,
    ResourceVector,
    WorkloadConsumer,
)
from .resources import fetch_cluster_snapshot
from .resources_charts import GPUHistoryPoint

SCHEMA = "falcon/resource-service/v1"
DEFAULT_URL = "http://node1.yoda.hyperverge.org:30081"
DEFAULT_STATE_DIR = Path("/var/lib/falcon-resource-service")
DEFAULT_INTERVAL = 5.0
HISTORY_HOURS = 24.0
HISTORY_LIMIT = 20_000


class ResourceServiceError(RuntimeError):
    """The configured authoritative resource service is unavailable."""


def snapshot_to_dict(snapshot: ClusterSnapshot) -> dict[str, Any]:
    """Serialize a snapshot using only JSON-compatible schema-v1 values."""

    return dataclasses.asdict(snapshot)


def _vector(value: Mapping[str, Any]) -> ResourceVector:
    return ResourceVector(
        cpu_cores=float(value.get("cpu_cores", 0.0)),
        memory_bytes=int(value.get("memory_bytes", 0)),
        gpu_count=int(value.get("gpu_count", 0)),
        gpu_model=value.get("gpu_model"),
    )


def snapshot_from_dict(value: Mapping[str, Any]) -> ClusterSnapshot:
    """Validate and deserialize a schema-v1 snapshot payload."""

    if not isinstance(value, Mapping):
        raise ValueError("snapshot must be an object")
    nodes = []
    for raw in value.get("nodes", ()):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str):
            raise ValueError("snapshot node is malformed")
        consumers = tuple(
            WorkloadConsumer(
                node_name=str(item.get("node_name", "")),
                namespace=str(item.get("namespace", "")),
                pod_name=str(item.get("pod_name", "")),
                workload_kind=str(item.get("workload_kind", "")),
                workload_name=str(item.get("workload_name", "")),
                status=str(item.get("status", "")),
                requested=_vector(item.get("requested", {})),
                owner_identity=item.get("owner_identity"),
                is_falcon_job=bool(item.get("is_falcon_job", False)),
                created_at=item.get("created_at"),
            )
            for item in raw.get("consumers", ())
            if isinstance(item, Mapping)
        )
        nodes.append(
            NodeSnapshot(
                name=raw["name"],
                ready=raw.get("ready"),
                schedulable=bool(raw.get("schedulable", False)),
                taints=tuple(
                    NodeTaint(
                        key=str(item.get("key", "")),
                        value=str(item.get("value", "")),
                        effect=str(item.get("effect", "")),
                    )
                    for item in raw.get("taints", ())
                    if isinstance(item, Mapping)
                ),
                labels=dict(raw.get("labels", {})),
                capacity=_vector(raw.get("capacity", {})),
                allocatable=_vector(raw.get("allocatable", {})),
                requested=_vector(raw.get("requested", {})),
                consumers=consumers,
                gpu_memory_bytes_per_device=raw.get("gpu_memory_bytes_per_device"),
                created_at=raw.get("created_at"),
            )
        )
    jobs = []
    for raw in value.get("jobs", ()):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str):
            raise ValueError("snapshot job is malformed")
        attempts = raw.get("attempts", {})
        if not isinstance(attempts, Mapping):
            raise ValueError("snapshot job attempts are malformed")
        usage = raw.get("current_usage")
        jobs.append(
            JobSnapshot(
                name=raw["name"], namespace=str(raw.get("namespace", "")),
                uid=str(raw.get("uid", "")), status=str(raw.get("status", "")),
                requested=_vector(raw.get("requested", {})),
                allocated=_vector(raw.get("allocated", {})),
                attempts=AttemptSummary(
                    container_restarts=int(attempts.get("container_restarts", 0)),
                    pod_attempts=int(attempts.get("pod_attempts", 0)),
                    succeeded_attempts=int(attempts.get("succeeded_attempts", 0)),
                    failed_attempts=int(attempts.get("failed_attempts", 0)),
                    active_pod=attempts.get("active_pod"),
                    active_pods=tuple(attempts.get("active_pods", ())),
                    backoff_limit=attempts.get("backoff_limit"),
                ),
                current_usage=_vector(usage) if isinstance(usage, Mapping) else None,
                nodes=tuple(raw.get("nodes", ())), command=tuple(raw.get("command", ())),
                created_at=raw.get("created_at"), started_at=raw.get("started_at"),
                completed_at=raw.get("completed_at"), suspended=bool(raw.get("suspended", False)),
            )
        )
    return ClusterSnapshot(
        nodes=tuple(nodes), jobs=tuple(jobs),
        collected_at=float(value.get("collected_at", 0.0)),
        stale=bool(value.get("stale", False)), error=value.get("error"),
    )


def envelope(snapshot: ClusterSnapshot, revision: str) -> dict[str, Any]:
    return {"schema": SCHEMA, "revision": revision, "snapshot": snapshot_to_dict(snapshot)}


def decode_envelope(value: Any) -> tuple[str, ClusterSnapshot]:
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA}")
    revision = value.get("revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("resource snapshot revision is missing")
    return revision, snapshot_from_dict(value.get("snapshot", {}))


def semantic_digest(snapshot: ClusterSnapshot) -> str:
    """Hash resource meaning, excluding scrape timestamps."""

    value = snapshot_to_dict(snapshot)
    value.pop("collected_at", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


class SharedHistoryStore:
    """Change-only, bounded allocation history owned by the service."""

    def __init__(self, path: Path, *, hours: float = HISTORY_HOURS, limit: int = HISTORY_LIMIT):
        self.path = Path(path)
        self.seconds = float(hours) * 3600
        self.limit = int(limit)
        if self.seconds <= 0 or self.limit <= 0:
            raise ValueError("history retention and limit must be positive")
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS transitions (
              timestamp REAL NOT NULL, node TEXT NOT NULL, gpu_model TEXT NOT NULL,
              namespace TEXT NOT NULL, gpu_count REAL NOT NULL, vram_gib REAL NOT NULL,
              PRIMARY KEY(timestamp, node, gpu_model, namespace)
            );
            CREATE INDEX IF NOT EXISTS transitions_time ON transitions(timestamp);
        """)
        return connection

    @staticmethod
    def _allocations(nodes: Sequence[NodeSnapshot]) -> dict[tuple[str, str, str], tuple[float, float]]:
        result: dict[tuple[str, str, str], tuple[float, float]] = {}
        for node in nodes:
            if node.ready is not True or not node.schedulable:
                continue
            counts: defaultdict[str, float] = defaultdict(float)
            for consumer in node.consumers:
                counts[consumer.namespace] += max(0, int(consumer.requested.gpu_count))
            per_device = (node.gpu_memory_bytes_per_device or 0) / (1024 ** 3)
            for namespace, count in counts.items():
                if count:
                    result[(node.name, node.gpu_model or "", namespace)] = (count, count * per_device)
        return result

    def record(self, snapshot: ClusterSnapshot) -> bool:
        if snapshot.stale or snapshot.collected_at <= 0:
            return False
        now = float(snapshot.collected_at)
        current = self._allocations(snapshot.nodes)
        with self._lock, self._connect() as connection:
            latest: dict[tuple[str, str, str], tuple[float, float]] = {}
            for node, model, namespace, count, vram in connection.execute("""
                SELECT t.node,t.gpu_model,t.namespace,t.gpu_count,t.vram_gib
                FROM transitions t JOIN (
                  SELECT node,gpu_model,namespace,MAX(timestamp) timestamp
                  FROM transitions GROUP BY node,gpu_model,namespace
                ) x USING(node,gpu_model,namespace,timestamp)
            """):
                latest[(node, model, namespace)] = (float(count), float(vram))
            changed = False
            for key in sorted(set(latest) | set(current)):
                value = current.get(key, (0.0, 0.0))
                if value != latest.get(key, (0.0, 0.0)):
                    connection.execute(
                        "INSERT INTO transitions VALUES (?,?,?,?,?,?)",
                        (now, *key, *value),
                    )
                    changed = True
            self._prune(connection, now - self.seconds)
            return changed

    def _prune(self, connection: sqlite3.Connection, cutoff: float) -> None:
        # Preserve the final pre-window value for every series as the window baseline.
        connection.execute("""
            DELETE FROM transitions WHERE timestamp < ? AND rowid NOT IN (
              SELECT rowid FROM transitions old WHERE timestamp=(
                SELECT MAX(timestamp) FROM transitions newer
                WHERE newer.node=old.node AND newer.gpu_model=old.gpu_model
                  AND newer.namespace=old.namespace AND newer.timestamp < ?
              )
            )
        """, (cutoff, cutoff))
        connection.execute("""
            DELETE FROM transitions WHERE rowid NOT IN (
              SELECT rowid FROM transitions ORDER BY timestamp DESC LIMIT ?
            )
        """, (self.limit,))

    def load(self, *, node: str = "", gpu: str = "", now: Optional[float] = None) -> list[GPUHistoryPoint]:
        if not self.path.exists():
            return []
        end = float(now if now is not None else time.time())
        cutoff = end - self.seconds
        with self._lock, self._connect() as connection:
            rows = list(connection.execute("""
                SELECT timestamp,node,gpu_model,namespace,gpu_count,vram_gib
                FROM transitions
                WHERE instr(lower(node),lower(?))>0 AND instr(lower(gpu_model),lower(?))>0
                ORDER BY timestamp,node,gpu_model,namespace
            """, (node, gpu)))
        state: dict[tuple[str, str, str], tuple[float, float]] = {}
        by_time: defaultdict[float, list[tuple[str, str, str, float, float]]] = defaultdict(list)
        for timestamp, node_name, model, namespace, count, vram in rows:
            key = (str(node_name), str(model), str(namespace))
            if timestamp < cutoff:
                state[key] = (float(count), float(vram))
            else:
                by_time[float(timestamp)].append((*key, float(count), float(vram)))

        def point(timestamp: float) -> GPUHistoryPoint:
            counts: defaultdict[str, float] = defaultdict(float)
            vrams: defaultdict[str, float] = defaultdict(float)
            for (_node, _model, namespace), (count, vram) in state.items():
                counts[namespace] += count
                vrams[namespace] += vram
            return GPUHistoryPoint.from_mapping(timestamp, counts, vrams)

        points: list[GPUHistoryPoint] = []
        if state:
            points.append(point(cutoff))
        for timestamp, changes in sorted(by_time.items()):
            for node_name, model, namespace, count, vram in changes:
                state[(node_name, model, namespace)] = (count, vram)
            points.append(point(timestamp))
        if points and points[-1].timestamp < end:
            points.append(point(end))
        return points


class ResourceState:
    """Authoritative snapshot, revision, subscribers, and persistence."""

    def __init__(self, state_dir: Path, history: SharedHistoryStore):
        self.state_dir = Path(state_dir)
        self.snapshot_path = self.state_dir / "snapshot.json"
        self.history = history
        self.condition = threading.Condition()
        self.snapshot: Optional[ClusterSnapshot] = None
        self.revision = ""
        self.digest = ""
        self.subscribers: set[queue.Queue[tuple[str, ClusterSnapshot]]] = set()
        self.upstream_requests = 0
        self.last_upstream_at = 0.0
        self.last_success_at = 0.0
        self.last_upstream_error: Optional[str] = None
        self.upstream_in_flight = False
        self.started_at = time.time()
        self._load()

    def _load(self) -> None:
        try:
            revision, snapshot = decode_envelope(json.loads(self.snapshot_path.read_text()))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return
        self.snapshot = snapshot.mark_stale("resource service restarted; awaiting upstream")
        self.digest = semantic_digest(self.snapshot)
        self.revision = self.digest[:24]

    def publish(self, snapshot: ClusterSnapshot) -> bool:
        digest = semantic_digest(snapshot)
        with self.condition:
            if digest == self.digest:
                # Keep one-shot/restart state at the newest scrape timestamp
                # without treating that timestamp as an SSE-visible change.
                self.snapshot = snapshot
                self._persist(envelope(snapshot, self.revision))
                return False
            self.snapshot = snapshot
            self.digest = digest
            self.revision = digest[:24]
            self._persist(envelope(snapshot, self.revision))
            if not snapshot.stale:
                self.history.record(snapshot)
            for subscriber in tuple(self.subscribers):
                try:
                    subscriber.put_nowait((self.revision, snapshot))
                except queue.Full:
                    try:
                        subscriber.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        subscriber.put_nowait((self.revision, snapshot))
                    except queue.Full:
                        pass
            self.condition.notify_all()
            return True

    def _persist(self, payload: Mapping[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.snapshot_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        os.replace(temporary, self.snapshot_path)

    def subscribe(
        self,
    ) -> tuple[
        queue.Queue[tuple[str, ClusterSnapshot]],
        tuple[str, ClusterSnapshot],
    ]:
        target: queue.Queue[tuple[str, ClusterSnapshot]] = queue.Queue(maxsize=1)
        with self.condition:
            if self.snapshot is None:
                raise ResourceServiceError("no upstream snapshot is available")
            self.subscribers.add(target)
            current = (self.revision, self.snapshot)
        return target, current

    def unsubscribe(self, target: queue.Queue[tuple[str, ClusterSnapshot]]) -> None:
        with self.condition:
            self.subscribers.discard(target)

    def wait_for_snapshot(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.snapshot is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
            return True


class ResourcePoller(threading.Thread):
    def __init__(self, state: ResourceState, upstream: str, *, interval: float = DEFAULT_INTERVAL,
                 timeout: float = 4.0, clock: Callable[[], float] = time.time):
        super().__init__(name="falcon-resource-poller", daemon=True)
        self.state, self.upstream = state, upstream
        self.interval, self.timeout, self.clock = float(interval), float(timeout), clock
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.is_set():
            started = time.monotonic()
            now = self.clock()
            self.state.upstream_requests += 1
            self.state.last_upstream_at = now
            self.state.upstream_in_flight = True
            try:
                snapshot = fetch_cluster_snapshot(self.upstream, timeout=self.timeout, collected_at=now)
                if not snapshot.nodes:
                    raise ValueError("kube-state-metrics reported no nodes")
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                self.state.last_upstream_error = detail
                previous = self.state.snapshot
                snapshot = (
                    previous.mark_stale(detail) if previous is not None
                    else ClusterSnapshot.empty(collected_at=now, stale=True, error=detail)
                )
            else:
                self.state.last_success_at = now
                self.state.last_upstream_error = None
            finally:
                self.state.upstream_in_flight = False
            self.state.publish(snapshot)
            self.stop_event.wait(max(0.0, self.interval - (time.monotonic() - started)))

    def stop(self) -> None:
        self.stop_event.set()


class _ResourceHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "falcon-resource-service/1"

    @property
    def state(self) -> ResourceState:
        return self.server.resource_state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("resource-service: " + (format % args) + "\n")

    def _json(self, status: int, value: Any, *, etag: str = "") -> None:
        body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        if etag:
            self.send_header("ETag", f'"{etag}"')
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status: int, detail: str) -> None:
        self._json(status, {"schema": SCHEMA, "error": detail})

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if parsed.path != "/v1/history" and (parsed.query or parsed.fragment):
            self._error(HTTPStatus.BAD_REQUEST, "this endpoint does not accept query parameters")
            return
        if parsed.path == "/healthz":
            state = self.state
            now = time.time()
            self._json(200, {
                "schema": SCHEMA, "ok": state.snapshot is not None and not state.snapshot.stale,
                "fresh": bool(state.snapshot and not state.snapshot.stale),
                "revision": state.revision, "subscribers": len(state.subscribers),
                "history_points": len(state.history.load(now=now)),
                "upstream_requests": state.upstream_requests,
                "last_upstream_at": state.last_upstream_at,
                "last_success_at": state.last_success_at,
                "snapshot_age_seconds": (
                    max(0.0, now - state.snapshot.collected_at)
                    if state.snapshot and state.snapshot.collected_at >= 946_684_800
                    else None
                ),
                "upstream_in_flight": state.upstream_in_flight,
                "upstream_error": state.last_upstream_error,
            }, etag=state.revision)
            return
        if parsed.path == "/v1/snapshot":
            if not self.state.wait_for_snapshot():
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, "no upstream snapshot is available")
                return
            if self.headers.get("If-None-Match", "").strip('"') == self.state.revision:
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header("ETag", f'"{self.state.revision}"')
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._json(200, envelope(self.state.snapshot, self.state.revision), etag=self.state.revision)
            return
        if parsed.path == "/v1/history":
            parameters = parse_qs(parsed.query, keep_blank_values=True)
            if any(key not in {"node", "gpu"} or len(values) != 1 for key, values in parameters.items()):
                self._error(HTTPStatus.BAD_REQUEST, "history accepts one node and one gpu filter")
                return
            points = self.state.history.load(
                node=parameters.get("node", [""])[0], gpu=parameters.get("gpu", [""])[0]
            )
            self._json(200, {"schema": SCHEMA, "history": [
                {"timestamp": p.timestamp, "values": p.values, "vram_values": p.vram_values}
                for p in points
            ]})
            return
        if parsed.path == "/v1/stream":
            self._stream()
            return
        self._error(HTTPStatus.NOT_FOUND, "not found")

    def _stream(self) -> None:
        if not self.state.wait_for_snapshot():
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "no upstream snapshot is available")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        subscriber, current = self.state.subscribe()
        try:
            while True:
                revision, snapshot = current
                data = json.dumps(envelope(snapshot, revision), sort_keys=True, separators=(",", ":"))
                self.wfile.write(f"id: {revision}\nevent: snapshot\ndata: {data}\n\n".encode())
                self.wfile.flush()
                current = subscriber.get()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.state.unsubscribe(subscriber)

    def do_HEAD(self) -> None:  # noqa: N802
        if urlsplit(self.path).path == "/v1/stream":
            self._readonly()
            return
        self.do_GET()

    def _readonly(self) -> None:
        self.close_connection = True
        body = json.dumps({"schema": SCHEMA, "error": "resource service is read-only"}, separators=(",", ":")).encode()
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    do_POST = do_PUT = do_PATCH = do_DELETE = _readonly  # type: ignore[assignment]


class ResourceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], state: ResourceState, *, inherited_socket: Optional[socket.socket] = None):
        super().__init__(address, _ResourceHandler, bind_and_activate=inherited_socket is None)
        self.resource_state = state
        if inherited_socket is not None:
            self.socket.close()
            self.socket = inherited_socket
            self.server_address = inherited_socket.getsockname()
            self.server_name = socket.getfqdn(self.server_address[0])
            self.server_port = self.server_address[1]

    def get_request(self):
        connection, address = super().get_request()
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        return connection, address


def _systemd_socket() -> Optional[socket.socket]:
    if int(os.environ.get("LISTEN_PID", "0") or 0) != os.getpid():
        return None
    if int(os.environ.get("LISTEN_FDS", "0") or 0) != 1:
        return None
    return socket.socket(fileno=3)


def serve(*, host: str = "192.168.1.117", port: int = 30081,
          upstream: str = "http://127.0.0.1:30080/metrics",
          state_dir: Path = DEFAULT_STATE_DIR, interval: float = DEFAULT_INTERVAL) -> int:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (state_dir / "publisher.lock").open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ResourceServiceError("another resource publisher owns the state directory") from exc
    state = ResourceState(state_dir, SharedHistoryStore(state_dir / "history.sqlite3"))
    inherited = _systemd_socket()
    server = ResourceHTTPServer((host, port), state, inherited_socket=inherited)
    poller = ResourcePoller(state, upstream, interval=interval)
    stop = threading.Event()

    def shutdown(_signum, _frame):
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    poller.start()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        poller.stop()
        poller.join(timeout=max(1.0, interval + 1))
        server.server_close()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
    return 0


def _client_cache_path(url: str) -> Path:
    root = Path(os.environ.get("XDG_STATE_HOME", "~/.local/state")).expanduser() / "falcon"
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    return root / f"resource-service-{key}.json"


class ResourceServiceClient:
    """One-shot snapshot/history client with a persistent last-known snapshot."""

    def __init__(self, url: str = DEFAULT_URL, *, timeout: float = 8.0, cache_path: Optional[Path] = None):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.cache_path = Path(cache_path) if cache_path else _client_cache_path(self.url)

    def _save(self, value: Any) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
        os.replace(temporary, self.cache_path)

    def _cached(self, detail: str) -> ClusterSnapshot:
        try:
            _revision, snapshot = decode_envelope(json.loads(self.cache_path.read_text()))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ResourceServiceError(
                f"resource service {self.url} is unavailable and no cached snapshot exists: {detail}"
            ) from exc
        return snapshot.mark_stale(detail)

    def snapshot(self, *, allow_cached: bool = True) -> ClusterSnapshot:
        try:
            response = httpx.get(f"{self.url}/v1/snapshot", timeout=self.timeout)
            response.raise_for_status()
            value = response.json()
            _revision, snapshot = decode_envelope(value)
            self._save(value)
            return snapshot
        except Exception as exc:
            detail = f"resource service {self.url}: {type(exc).__name__}: {exc}"
            if allow_cached:
                return self._cached(detail)
            raise ResourceServiceError(detail) from exc

    def collect(self, force: bool = False) -> ClusterSnapshot:
        del force
        return self.snapshot()

    def close(self) -> None:
        return None

    def history(self, *, node: str = "", gpu: str = "") -> list[GPUHistoryPoint]:
        response = httpx.get(f"{self.url}/v1/history", params={"node": node, "gpu": gpu}, timeout=self.timeout)
        response.raise_for_status()
        value = response.json()
        if value.get("schema") != SCHEMA or not isinstance(value.get("history"), list):
            raise ResourceServiceError("resource service returned malformed history")
        return [GPUHistoryPoint.from_mapping(
            float(item["timestamp"]), item.get("values", {}), item.get("vram_values", {})
        ) for item in value["history"]]


class ResourceServiceCollector:
    """One SSE connection, coalesced by the server, exposed as ``collect()``."""

    def __init__(
        self,
        url: str = DEFAULT_URL,
        *,
        timeout: float = 8.0,
        cache_path: Optional[Path] = None,
        on_connect: Optional[Callable[[], None]] = None,
    ):
        self.client = ResourceServiceClient(url, timeout=timeout, cache_path=cache_path)
        self.url, self.timeout = self.client.url, timeout
        self._condition = threading.Condition()
        self._snapshot: Optional[ClusterSnapshot] = None
        self._closed = threading.Event()
        self._on_connect = on_connect
        self._connected = False
        self._revision = ""
        self._thread = threading.Thread(target=self._stream, name="falcon-resource-stream", daemon=True)
        self._thread.start()

    @property
    def last_snapshot(self) -> Optional[ClusterSnapshot]:
        with self._condition:
            return self._snapshot

    def _set(self, snapshot: ClusterSnapshot) -> None:
        with self._condition:
            self._snapshot = snapshot
            self._condition.notify_all()

    def _stream(self) -> None:
        delay = 0.25
        while not self._closed.is_set():
            try:
                headers = (
                    {"Last-Event-ID": self._revision} if self._revision else None
                )
                with httpx.stream(
                    "GET",
                    f"{self.url}/v1/stream",
                    headers=headers,
                    timeout=httpx.Timeout(self.timeout, read=None),
                ) as response:
                    response.raise_for_status()
                    data: list[str] = []
                    for line in response.iter_lines():
                        if self._closed.is_set():
                            return
                        if line.startswith("data:"):
                            data.append(line[5:].lstrip())
                        elif line == "" and data:
                            value = json.loads("\n".join(data))
                            self._revision, snapshot = decode_envelope(value)
                            self.client._save(value)
                            self._set(snapshot)
                            if not self._connected and self._on_connect is not None:
                                self._connected = True
                                self._on_connect()
                            data = []
                            delay = 0.25
            except Exception as exc:
                detail = f"resource service {self.url}: {type(exc).__name__}: {exc}"
                with self._condition:
                    if self._snapshot is not None:
                        self._snapshot = self._snapshot.mark_stale(detail)
                        self._condition.notify_all()
                    else:
                        try:
                            self._snapshot = self.client._cached(detail)
                            self._condition.notify_all()
                        except ResourceServiceError:
                            pass
                self._closed.wait(delay)
                delay = min(5.0, delay * 2)

    def collect(self, force: bool = False) -> ClusterSnapshot:
        del force
        deadline = time.monotonic() + self.timeout
        with self._condition:
            while self._snapshot is None and not self._closed.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._snapshot is None:
                raise ResourceServiceError(
                    f"resource service {self.url} did not provide an initial snapshot"
                )
            return self._snapshot

    def close(self) -> None:
        self._closed.set()


def snapshot_nodes(snapshot: ClusterSnapshot):
    """Convert the shared snapshot to the existing launch-planning model."""
    from .models import NodeResources

    return [NodeResources(
        name=node.name, cpu_total=node.allocatable.cpu_cores,
        cpu_used=node.requested.cpu_cores, memory_total_gib=node.allocatable.memory_gib,
        memory_used_gib=node.requested.memory_gib, gpu_total=node.allocatable.gpu_count,
        gpu_used=node.requested.gpu_count, gpu_product=node.gpu_model or "",
        unschedulable=not node.schedulable or node.ready is False,
        scheduling_info_available=True,
    ) for node in snapshot.nodes]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Falcon shared resource service")
    parser.add_argument("--host", default="192.168.1.117")
    parser.add_argument("--port", type=int, default=30081)
    parser.add_argument("--upstream", default="http://127.0.0.1:30080/metrics")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    args = parser.parse_args(argv)
    try:
        return serve(host=args.host, port=args.port, upstream=args.upstream,
                     state_dir=args.state_dir, interval=args.interval)
    except (OSError, ResourceServiceError) as exc:
        print(f"falcon-resource-server: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
