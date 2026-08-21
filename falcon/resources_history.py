"""Persistent background history for scheduler-facing GPU allocations."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from threading import Event
from typing import Any, Mapping, Optional, Sequence

from .cluster import ClusterCollector, NodeSnapshot
from .config import config_path, load_config
from .kubernetes import KubernetesClient
from .resources import MetricsClusterCollector
from .resources_charts import GPUHistoryPoint

HISTORY_LIMIT = 20_000
DEFAULT_HISTORY_HOURS = 24.0
DEFAULT_HISTORY_INTERVAL_SECONDS = 5.0


def _state_root() -> Path:
    configured = os.environ.get("XDG_STATE_HOME")
    return (
        Path(configured).expanduser()
        if configured
        else Path("~/.local/state").expanduser()
    ) / "falcon"


def _source_key(config: Mapping[str, Any]) -> str:
    cluster = config.get("cluster", {})
    identity = {
        "namespace": cluster.get("namespace") if isinstance(cluster, Mapping) else "",
        "metrics_url": (
            cluster.get("kube_state_metrics_url")
            if isinstance(cluster, Mapping)
            else ""
        ),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def history_path(config: Mapping[str, Any]) -> Path:
    return _state_root() / f"resources-history-{_source_key(config)}.sqlite3"


def _pid_path(config: Mapping[str, Any]) -> Path:
    return _state_root() / f"resources-history-{_source_key(config)}.pid"


class ResourceHistoryStore:
    """SQLite-backed allocation observations safe for concurrent readers."""

    def __init__(
        self,
        path: Path,
        *,
        history_hours: float = DEFAULT_HISTORY_HOURS,
        limit: int = HISTORY_LIMIT,
    ) -> None:
        self.path = Path(path)
        self.history_seconds = float(history_hours) * 3600.0
        self.limit = int(limit)
        if self.history_seconds <= 0 or self.limit <= 0:
            raise ValueError("history retention and limit must be positive")

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                source_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS allocations (
                source_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                node TEXT NOT NULL,
                gpu_model TEXT NOT NULL,
                namespace TEXT NOT NULL,
                gpu_count REAL NOT NULL,
                vram_gib REAL NOT NULL,
                PRIMARY KEY (source_id, node, namespace),
                FOREIGN KEY (source_id) REFERENCES snapshots(source_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS allocations_timestamp
                ON allocations(timestamp);
            """
        )
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _rows(
        nodes: Sequence[NodeSnapshot],
    ) -> list[tuple[str, str, str, float, float]]:
        rows: list[tuple[str, str, str, float, float]] = []
        for node in nodes:
            if node.ready is not True or not node.schedulable:
                continue
            per_device_gib = (
                node.gpu_memory_bytes_per_device / (1024**3)
                if node.gpu_memory_bytes_per_device is not None
                else 0.0
            )
            namespaces: defaultdict[str, float] = defaultdict(float)
            for consumer in node.consumers:
                count = max(0, int(consumer.requested.gpu_count))
                if count:
                    namespaces[consumer.namespace] += count
            for namespace, count in sorted(namespaces.items()):
                rows.append(
                    (
                        node.name,
                        node.gpu_model or "",
                        namespace,
                        count,
                        count * per_device_gib if per_device_gib > 0 else 0.0,
                    )
                )
        return rows

    def record_snapshot(
        self,
        nodes: Sequence[NodeSnapshot],
        *,
        collected_at: float,
    ) -> bool:
        rows = self._rows(nodes)
        if collected_at <= 0:
            return False
        timestamp = float(collected_at)
        if timestamp < 946_684_800:  # monotonic adapters are not wall time
            timestamp = time.time()
        signature = hashlib.sha256(
            json.dumps(rows, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        source_id = f"{float(collected_at):.9f}:{signature}"
        cutoff = timestamp - self.history_seconds
        with self._connect() as connection:
            inserted = connection.execute(
                "INSERT OR IGNORE INTO snapshots(source_id, timestamp) VALUES (?, ?)",
                (source_id, timestamp),
            ).rowcount
            if not inserted:
                return False
            connection.executemany(
                """
                INSERT INTO allocations(
                    source_id, timestamp, node, gpu_model, namespace,
                    gpu_count, vram_gib
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (source_id, timestamp, node, model, namespace, count, vram)
                    for node, model, namespace, count, vram in rows
                ],
            )
            connection.execute(
                "DELETE FROM snapshots WHERE timestamp < ?",
                (cutoff,),
            )
            connection.execute(
                """
                DELETE FROM snapshots
                WHERE source_id NOT IN (
                    SELECT source_id FROM snapshots
                    ORDER BY timestamp DESC LIMIT ?
                )
                """,
                (self.limit,),
            )
        return True

    def load(
        self,
        *,
        node_filter: str = "",
        gpu_filter: str = "",
        now: Optional[float] = None,
    ) -> list[GPUHistoryPoint]:
        if not self.path.exists():
            return []
        cutoff = float(now if now is not None else time.time()) - self.history_seconds
        query = """
            SELECT s.timestamp, a.namespace,
                   SUM(a.gpu_count), SUM(a.vram_gib)
            FROM snapshots AS s
            LEFT JOIN allocations AS a
              ON a.source_id = s.source_id
             AND instr(lower(a.node), lower(?)) > 0
             AND instr(lower(a.gpu_model), lower(?)) > 0
            WHERE s.timestamp >= ?
            GROUP BY s.timestamp, a.namespace
            ORDER BY s.timestamp, a.namespace
        """
        grouped: dict[float, tuple[dict[str, float], dict[str, float]]] = {}
        with self._connect() as connection:
            for timestamp, namespace, count, vram in connection.execute(
                query,
                (node_filter, gpu_filter, cutoff),
            ):
                counts, vrams = grouped.setdefault(float(timestamp), ({}, {}))
                if namespace is not None:
                    counts[str(namespace)] = float(count or 0.0)
                    vrams[str(namespace)] = float(vram or 0.0)
        return [
            GPUHistoryPoint.from_mapping(timestamp, counts, vrams)
            for timestamp, (counts, vrams) in grouped.items()
        ]


def _history_settings(config: Mapping[str, Any]) -> tuple[float, float]:
    resources = config.get("resources", {})
    if not isinstance(resources, Mapping):
        resources = {}
    return (
        float(resources.get("history_hours", DEFAULT_HISTORY_HOURS)),
        float(
            resources.get(
                "history_interval_seconds",
                DEFAULT_HISTORY_INTERVAL_SECONDS,
            )
        ),
    )


def history_store(config: Mapping[str, Any]) -> ResourceHistoryStore:
    hours, _ = _history_settings(config)
    return ResourceHistoryStore(history_path(config), history_hours=hours)


def _collector(config: Mapping[str, Any], interval: float):
    cluster = config.get("cluster", {})
    metrics_url = (
        cluster.get("kube_state_metrics_url")
        if isinstance(cluster, Mapping)
        else None
    )
    if metrics_url:
        return MetricsClusterCollector(
            str(metrics_url),
            inventory_seconds=interval,
            clock=time.time,
        )
    namespace = (
        str(cluster.get("namespace") or "")
        if isinstance(cluster, Mapping)
        else ""
    )
    return ClusterCollector(
        KubernetesClient(namespace),
        namespace="",
        inventory_seconds=interval,
        clock=time.time,
    )


def _collector_is_running(path: Path) -> bool:
    """Use the daemon's file lock rather than trusting a potentially stale PID."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as pid_file:
        try:
            fcntl.flock(pid_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(pid_file.fileno(), fcntl.LOCK_UN)
    return False


def ensure_history_collector(
    config: Mapping[str, Any],
    *,
    config_file: Optional[str] = None,
) -> bool:
    cluster = config.get("cluster", {})
    if isinstance(cluster, Mapping) and cluster.get("resource_service_url") is not None:
        return False
    resources = config.get("resources", {})
    if isinstance(resources, Mapping) and not resources.get("history_enabled", True):
        return False
    pid_path = _pid_path(config)
    if _collector_is_running(pid_path):
        return False
    command = [sys.executable, "-m", "falcon.resources_history"]
    target = config_path(config_file)
    if target.exists():
        command.extend(["--config", str(target)])
    subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return True


def stop_legacy_history_collector(config: Mapping[str, Any]) -> bool:
    """Stop only this user's positively identified legacy publisher.

    PID reuse is handled by checking ownership and the exact module name in
    ``/proc`` before signalling.  Personal SQLite databases are untouched.
    """

    path = _pid_path(config)
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        proc = Path("/proc") / str(pid)
        if proc.stat().st_uid != os.getuid():
            return False
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", "replace"
        )
        if "-m falcon.resources_history" not in command:
            return False
        os.kill(pid, signal.SIGTERM)
        return True
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, OSError):
        return False


def run_collector(config_file: Optional[str] = None) -> int:
    config = load_config(config_file)
    resources = config.get("resources", {})
    if isinstance(resources, Mapping) and not resources.get("history_enabled", True):
        return 0
    _, interval = _history_settings(config)
    store = history_store(config)
    pid_path = _pid_path(config)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    stop = Event()
    with pid_path.open("a+", encoding="utf-8") as pid_file:
        try:
            fcntl.flock(pid_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        pid_file.seek(0)
        pid_file.truncate()
        pid_file.write(str(os.getpid()))
        pid_file.flush()

        def request_stop(_signum, _frame) -> None:
            stop.set()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        collector = _collector(config, interval)
        try:
            while not stop.is_set():
                snapshot = collector.collect(force=True)
                if not snapshot.stale:
                    store.record_snapshot(
                        snapshot.nodes,
                        collected_at=snapshot.collected_at,
                    )
                stop.wait(interval)
        finally:
            close = getattr(collector, "close", None)
            if callable(close):
                close()
            pid_file.seek(0)
            pid_file.truncate()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    return run_collector(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
