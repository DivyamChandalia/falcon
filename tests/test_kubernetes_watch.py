from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

import falcon.kubernetes_watch as watching
from falcon.kubernetes_watch import WatchError, WatchTransport, watch_resource


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resource", "path"),
    [
        ("pods", "/api/v1/namespaces/mine/pods"),
        ("jobs.batch", "/apis/batch/v1/namespaces/mine/jobs"),
        ("events", "/api/v1/namespaces/mine/events"),
    ],
)
async def test_watch_only_decodes_fragmented_utf8_and_closes_idle_process(
    monkeypatch, resource, path,
):
    original = asyncio.create_subprocess_exec
    processes, commands = [], []
    obj = {"type": "ADDED", "object": {"metadata": {"name": "π", "resourceVersion": "2"}}}
    script = (
        "import os,time; data=" + repr(json.dumps(obj, ensure_ascii=False).encode()) + "; "
        "[(os.write(1, data[i:i+3]), time.sleep(.001)) for i in range(0,len(data),3)]; "
        "time.sleep(60)"
    )

    async def spawn(*args, **kwargs):
        assert "--resource-version" not in args
        commands.append(args)
        process = await original(sys.executable, "-c", script, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    client = SimpleNamespace(executable="kubectl", context="pinned")
    transport = WatchTransport(client, "mine")
    stream = transport.watch_json(resource, "1")
    try:
        assert await asyncio.wait_for(anext(stream), 3) == obj
        position = commands[0].index("--raw")
        url = urlsplit(commands[0][position + 1])
        assert url.path == path
        assert parse_qs(url.query) == {
            "watch": ["true"],
            "resourceVersion": ["1"],
            "allowWatchBookmarks": ["true"],
        }
        assert "--request-timeout=0" in commands[0]
        assert "pinned" in commands[0]
    finally:
        await stream.aclose()
    assert processes[0].returncode is not None
    assert not transport._processes


@pytest.mark.asyncio
async def test_forbidden_watch_falls_back_to_lists_then_retries(monkeypatch):
    monkeypatch.setattr(watching, "POLL_FALLBACK_SECONDS", 0.005)
    monkeypatch.setattr(watching, "WATCH_PERMISSION_RETRY_SECONDS", 0.35)
    versions = {}
    calls = {"load": 0, "watch": 0, "changes": 0}
    active = True

    async def load(resource):
        calls["load"] += 1
        versions[resource] = str(calls["load"])

    async def stream(resource, version):
        calls["watch"] += 1
        if calls["watch"] == 1:
            raise WatchError("forbidden", code=403)
        yield {"type": "MODIFIED", "object": {"metadata": {"resourceVersion": "100"}}}

    async def handle(resource, action, obj):
        nonlocal active
        calls["changes"] += 1
        active = False

    async def health(*args):
        pass

    await asyncio.wait_for(
        watch_resource(
            "pods",
            versions=versions,
            load=load,
            stream=stream,
            handle=handle,
            unavailable=health,
            recovered=health,
            active=lambda: active,
            poll_fallback=True,
        ),
        3,
    )
    assert calls["watch"] == 2
    assert calls["load"] > 1
    assert calls["changes"] == 1
    assert versions["pods"] == "100"


@pytest.mark.asyncio
async def test_cancelled_initial_list_reaps_child(monkeypatch):
    original = asyncio.create_subprocess_exec
    started = asyncio.Event()
    children = []

    async def spawn(*args, **kwargs):
        child = await original(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
        children.append(child)
        started.set()
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    transport = WatchTransport(SimpleNamespace(executable="kubectl", timeout=15), "mine")
    task = asyncio.create_task(transport.list_json("pods"))
    await asyncio.wait_for(started.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert children[0].returncode is not None
    assert not transport._processes
