"""Shared asynchronous kubectl watch transport for Dashboard and MCP."""

from __future__ import annotations

import asyncio
import codecs
import inspect
import json
import random
from collections.abc import AsyncIterator, Mapping
from json import JSONDecodeError, JSONDecoder
from typing import Any
from urllib.parse import urlencode

POLL_FALLBACK_SECONDS = 5.0
WATCH_PERMISSION_RETRY_SECONDS = 60.0


async def watch_resource(
    resource,
    *,
    versions,
    load,
    stream,
    handle,
    unavailable,
    recovered,
    active,
    poll_fallback=False,
):
    """Resume one resource stream; callers own snapshots and domain events.

    ``load`` must raise on failed lists so watch-only never skips its baseline.
    All callbacks execute on the owning event loop, in resource order.
    """
    delay = 0.25
    poll_until = 0.0
    loop = asyncio.get_running_loop()
    while active():
        source = None
        started = loop.time()
        try:
            if poll_until > started:
                await load(resource)
                await asyncio.sleep(min(POLL_FALLBACK_SECONDS, max(0, poll_until - loop.time())))
                continue
            if not versions.get(resource):
                await load(resource)
            source = stream(resource, versions.get(resource))
            if inspect.isawaitable(source):
                source = await source
            async for raw in source:
                if not active():
                    return
                if not isinstance(raw, Mapping):
                    continue
                obj = raw.get("object", raw)
                if not isinstance(obj, Mapping):
                    continue
                action = str(raw.get("type") or "MODIFIED").upper()
                if action == "ERROR" or (
                    obj.get("kind") == "Status" and obj.get("status") == "Failure"
                ):
                    code = obj.get("code")
                    error = WatchGone if str(code) == "410" else WatchError
                    raise error(
                        str(obj.get("message") or "watch error"),
                        code=int(code) if str(code).isdigit() else None,
                    )
                delay = 0.25
                await recovered(resource)
                version = (obj.get("metadata") or {}).get("resourceVersion")
                if version:
                    versions[resource] = str(version)
                if action != "BOOKMARK":
                    await handle(resource, action, obj)
            raise WatchError(f"{resource} watch stream ended")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            details = {"operation": "watch", "error": str(exc)}
            if isinstance(exc, WatchGone):
                versions.pop(resource, None)
                details["gap"] = True
            if poll_fallback and getattr(exc, "code", None) == 403:
                poll_until = loop.time() + WATCH_PERMISSION_RETRY_SECONDS
            await unavailable(resource, details)
            if loop.time() - started >= 30:
                delay = 0.25
            if active():
                await asyncio.sleep(delay * random.uniform(0.8, 1.2))
                delay = min(delay * 2, 30.0)
        finally:
            if source is not None and hasattr(source, "aclose"):
                try:
                    await source.aclose()
                except Exception:
                    pass


class WatchError(RuntimeError):
    def __init__(self, message: str, *, code: int | None = None):
        super().__init__(message)
        self.code = code


class WatchGone(WatchError):
    pass


def _string(value: Any) -> str:
    return str(value) if value is not None else ""


async def _read_limited(stream: Any, limit: int = 4096) -> bytes:
    captured = bytearray()
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return bytes(captured)
        if len(captured) < limit:
            captured.extend(chunk[: limit - len(captured)])


class WatchTransport:
    def __init__(self, client: Any, namespace: str):
        self.client = client
        self.namespace = namespace
        self._processes: set[asyncio.subprocess.Process] = set()

    def resource_path(self, resource: str) -> str:
        # kubectl's normal JSON printer can rebuild a generic List with an
        # empty resourceVersion. Raw reads retain the server's list watermark.
        paths = {
            "jobs.batch": f"/apis/batch/v1/namespaces/{self.namespace}/jobs",
            "pods": f"/api/v1/namespaces/{self.namespace}/pods",
            "events": f"/api/v1/namespaces/{self.namespace}/events",
        }
        return paths[resource]

    async def list_json(self, resource: str) -> dict:
        """Cancellable initial/resync list, using the same context as watches."""
        argv = self.command_argv(["get", "--raw", self.resource_path(resource)])
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._processes.add(process)
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=getattr(self.client, "timeout", 15),
            )
            if process.returncode:
                raise WatchError(
                    stderr.decode(errors="replace").strip() or "Kubernetes list failed"
                )
            payload = json.loads(stdout)
            if not isinstance(payload, dict):
                raise WatchError("Kubernetes list returned a non-object")
            return payload
        finally:
            await self._stop_process(process)
            self._processes.discard(process)

    @staticmethod
    async def _stop_process(process):
        if process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()

    def command_argv(self, args: list[str]) -> list[str]:
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
        context = getattr(self.client, "context", None) or getattr(self.client, "_context", None)
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
                    raise WatchError("kubectl watch ended with incomplete JSON")
                return objects, stripped
            buffer = stripped[end:]
            if isinstance(value, Mapping):
                objects.append(value)

    async def watch_json(
        self,
        resource: str,
        resource_version: str | None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        if not resource_version:
            raise WatchError("watch requires a list resourceVersion")
        query = urlencode(
            {
                "watch": "true",
                "resourceVersion": resource_version,
                "allowWatchBookmarks": "true",
            }
        )
        # get --resource-version is absent from some supported kubectl builds.
        # Raw watch preserves the list watermark and streams only changes.
        args = ["get", "--raw", f"{self.resource_path(resource)}?{query}", "--request-timeout=0"]
        argv = self.command_argv(args)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise WatchError(f"could not start {argv[0]!r}: executable not found") from exc
        except OSError as exc:
            raise WatchError(f"could not start Kubernetes watch: {exc}") from exc

        self._processes.add(process)
        stderr_task = asyncio.create_task(_read_limited(process.stderr)) if process.stderr else None
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
                code = returncode
                if "410" in stderr or "too old resource version" in stderr.lower():
                    code = 410
                elif "forbidden" in stderr.lower():
                    code = 403
                if code == 410:
                    raise WatchGone(stderr.strip() or "watch resource version is gone", code=410)
                raise WatchError(
                    stderr.strip() or f"kubectl watch exited {returncode}",
                    code=code,
                )
            raise WatchError("kubectl watch stream ended")
        finally:
            await self._stop_process(process)
            if stderr_task is not None and not stderr_task.done():
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
            self._processes.discard(process)

    async def close(self) -> None:
        await asyncio.gather(
            *(self._stop_process(process) for process in list(self._processes)),
            return_exceptions=True,
        )
