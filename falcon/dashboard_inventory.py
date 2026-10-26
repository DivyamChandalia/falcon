"""Thread-safe Dashboard inventory backed by namespace Kubernetes watches."""

from __future__ import annotations

import asyncio
import copy
import functools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from .kubernetes_watch import WatchTransport, watch_resource

RESOURCES = ("jobs.batch", "pods", "events")
MAX_CACHED_EVENTS = 10_000


@dataclass(frozen=True)
class InventorySnapshot:
    items: list[dict]
    events: list[dict]
    revision: int
    error: str
    observed_at: float


class DashboardInventory:
    """Own three watch tasks on one loop; publish copies to collector threads."""

    def __init__(self, client: Any, *, transport=None):
        self.client = client
        self.namespace = client.namespace
        self.transport = transport or WatchTransport(client, self.namespace)
        self._lock = threading.Lock()
        self._objects: dict[str, dict[str, dict]] = {r: {} for r in RESOURCES}
        self._versions: dict[str, str] = {}
        self._initialized: set[str] = set()
        self._failures: dict[str, int] = {}
        self._errors: dict[str, str] = {}
        self._revision = 0
        self._observed_at = 0.0
        self._callbacks: set[Callable[[], None]] = set()
        self._closed = threading.Event()
        self._loop = None
        self._stop = None
        self._tasks: dict[str, asyncio.Task] = {}
        self._resync_task = None
        self._list_executor = ThreadPoolExecutor(
            max_workers=len(RESOURCES),
            thread_name_prefix="falcon-inventory-list",
        )
        self._thread = threading.Thread(target=self._run, name="falcon-inventory", daemon=True)
        self._thread.start()

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        with self._lock:
            self._callbacks.add(callback)

        def unsubscribe():
            with self._lock:
                self._callbacks.discard(callback)

        return unsubscribe

    def _notify(self):
        with self._lock:
            self._revision += 1
            callbacks = tuple(self._callbacks)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                # A closing UI must not interrupt Kubernetes observation.
                pass

    def snapshot(self, *, include_events=True) -> InventorySnapshot:
        with self._lock:
            return InventorySnapshot(
                copy.deepcopy(
                    list(self._objects["jobs.batch"].values())
                    + list(self._objects["pods"].values())
                ),
                copy.deepcopy(list(self._objects["events"].values())) if include_events else [],
                self._revision,
                "; ".join(f"{r}: {e}" for r, e in self._errors.items()),
                self._observed_at,
            )

    def events_for(self, uids: set[str], names: set[str]) -> list[dict]:
        with self._lock:
            selected = []
            for obj in self._objects["events"].values():
                target = obj.get("involvedObject") or obj.get("regarding") or {}
                uid = target.get("uid")
                matches = uid in uids if uid else target.get("name") in names
                if matches:
                    selected.append(obj)
            selected.sort(
                key=lambda obj: str(
                    obj.get("eventTime")
                    or obj.get("lastTimestamp")
                    or (obj.get("series") or {}).get("lastObservedTime")
                    or (obj.get("metadata") or {}).get("creationTimestamp")
                    or ""
                )
            )
            return copy.deepcopy(selected[-200:])

    @staticmethod
    def _key(obj):
        metadata = obj.get("metadata") or {}
        return str(metadata.get("uid") or metadata.get("name") or "")

    async def _load(self, resource):
        if hasattr(self.transport, "list_json"):
            payload = await self.transport.list_json(resource)
        else:
            payload = await asyncio.get_running_loop().run_in_executor(
                self._list_executor,
                functools.partial(
                    self.client.list_json, resource, namespace=self.namespace
                ),
            )
        values = payload.get("items")
        version = (payload.get("metadata") or {}).get("resourceVersion")
        if not isinstance(values, list) or not version:
            raise ValueError(f"invalid {resource} snapshot or missing resourceVersion")
        kind = {"jobs.batch": "Job", "pods": "Pod", "events": "Event"}[resource]
        objects = {
            self._key(obj): {"kind": kind, **obj}
            for obj in values
            if isinstance(obj, dict) and self._key(obj)
        }
        if resource == "events":
            objects = dict(list(objects.items())[-MAX_CACHED_EVENTS:])
        with self._lock:
            self._objects[resource] = objects
            self._initialized.add(resource)
            self._versions[resource] = str(version)
            self._observed_at = time.time()
        await self._recovered(resource)
        self._notify()

    async def _handle(self, resource, action, obj):
        key = self._key(obj)
        if not key or action not in {"ADDED", "MODIFIED", "DELETED"}:
            return
        with self._lock:
            objects = self._objects[resource]
            if action == "DELETED":
                changed = objects.pop(key, None) is not None
            else:
                changed = objects.get(key) != obj
                if changed:
                    # Same-name replacements must not inherit old attempts.
                    name = (obj.get("metadata") or {}).get("name")
                    for old_key, old in list(objects.items()):
                        if (
                            name
                            and old_key != key
                            and (old.get("metadata") or {}).get("name") == name
                        ):
                            objects.pop(old_key)
                    objects[key] = dict(obj)
                    if resource == "events" and len(objects) > MAX_CACHED_EVENTS:
                        objects.pop(next(iter(objects)))
            self._observed_at = time.time()
        if changed:
            self._notify()

    async def _unavailable(self, resource, details):
        with self._lock:
            count = self._failures.get(resource, 0) + 1
            self._failures[resource] = count
            changed = count == 3 or resource not in self._initialized
            if changed:
                self._errors[resource] = str(details.get("error") or "watch unavailable")
        if changed:
            self._notify()

    async def _recovered(self, resource):
        with self._lock:
            self._failures.pop(resource, None)
            changed = self._errors.pop(resource, None) is not None
        if changed:
            self._notify()

    async def _watch(self, resource):
        await watch_resource(
            resource,
            versions=self._versions,
            load=self._load,
            stream=self.transport.watch_json,
            handle=self._handle,
            unavailable=self._unavailable,
            recovered=self._recovered,
            active=lambda: not self._closed.is_set(),
            poll_fallback=True,
        )

    def _run(self):
        asyncio.run(self._serve())

    async def _serve(self):
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        if self._closed.is_set():
            return
        self._tasks = {r: asyncio.create_task(self._watch(r)) for r in RESOURCES}
        try:
            await self._stop.wait()
        finally:
            if self._resync_task is not None:
                self._resync_task.cancel()
                await asyncio.gather(self._resync_task, return_exceptions=True)
            for task in self._tasks.values():
                task.cancel()
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
            await self.transport.close()
            self._list_executor.shutdown(wait=True, cancel_futures=True)

    def resync(self):
        async def restart():
            tasks = list(self._tasks.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if not self._closed.is_set():
                self._versions.clear()
                self._tasks = {r: asyncio.create_task(self._watch(r)) for r in RESOURCES}

        def schedule():
            if self._closed.is_set():
                return
            if self._resync_task is None or self._resync_task.done():
                self._resync_task = asyncio.create_task(restart())

        if self._loop and not self._closed.is_set():
            try:
                self._loop.call_soon_threadsafe(schedule)
            except RuntimeError:
                pass

    def close(self):
        self._closed.set()
        if self._loop and self._stop:
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:
                pass
        self._thread.join(timeout=3)
