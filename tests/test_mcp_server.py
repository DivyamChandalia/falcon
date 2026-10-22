"""Exercise the actual MCP protocol without a cluster or model invocation."""

from __future__ import annotations

import asyncio
import contextlib
import io
import sys
from datetime import timedelta
from unittest.mock import patch

import pytest

pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from falcon.config import DEFAULT_CONFIG
from falcon.mcp_server import create_server

REF = {"name": "train", "namespace": "default", "uid": "uid-1"}


class FakeService:
    def __init__(self, config, state_dir=None):
        self.calls = []
        self.closed = False
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()

    def plan_job(self, request):
        self.calls.append(("plan", request))
        return {"resources": {"cpu": request.get("cpu")}, "backoff_limit": 0}

    def submit_job(self, request, idempotency_key):
        self.calls.append(("submit", request, idempotency_key))
        return {"job": REF}

    def list_jobs(self, **kwargs):
        return {"jobs": [REF]}

    def cancel_job(self, job):
        self.calls.append(("cancel", job))
        return {"job": job, "cancelled": True}

    async def wait_for_events(self, jobs, **kwargs):
        self.calls.append(("wait", jobs, kwargs))
        self.waiting.set()
        await self.release.wait()
        return {"events": [{"type": "failed", "job": jobs[0]}], "cursor": 1}

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_catalog_is_lazy_and_annotations_preserve_host_approvals():
    with patch("falcon.mcp_server.JobService") as service:
        server = create_server(DEFAULT_CONFIG)
        async with create_connected_server_and_client_session(server) as session:
            catalog = await session.list_tools()
            tools = {tool.name: tool for tool in catalog.tools}
            assert set(tools) == {
                "plan_job", "submit_job", "list_jobs", "inspect_job", "job_logs",
                "job_metrics", "wait_for_events", "cancel_job", "restart_job",
            }
            assert tools["wait_for_events"].annotations.readOnlyHint
            assert not tools["submit_job"].annotations.readOnlyHint
            assert tools["cancel_job"].annotations.destructiveHint
            assert tools["restart_job"].annotations.destructiveHint
            assert tools["submit_job"].annotations.idempotentHint
            schema = tools["submit_job"].inputSchema
            assert "idempotency_key" in schema["required"]
            service.assert_not_called()


@pytest.mark.asyncio
async def test_protocol_submission_and_schema_validation():
    service = FakeService(DEFAULT_CONFIG)
    server = create_server(DEFAULT_CONFIG, service_factory=lambda *a, **k: service)
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("submit_job", {
            "request": {"command": ["python", "train.py"], "cpu": "1", "memory": "1Gi"},
            "idempotency_key": "run-1",
        })
        assert not result.isError
        assert result.structuredContent["job"] == REF
        invalid = await session.call_tool("submit_job", {
            "request": {"command": ["true"], "backoffLimit": 6}, "idempotency_key": "bad",
        })
        assert invalid.isError
        wrong_namespace = await session.call_tool("cancel_job", {"job": {"name": "train"}})
        assert wrong_namespace.isError
        assert len(service.calls) == 1
    assert service.closed


@pytest.mark.asyncio
async def test_wait_does_not_block_other_tools_or_produce_empty_updates():
    service = FakeService(DEFAULT_CONFIG)
    server = create_server(DEFAULT_CONFIG, service_factory=lambda *a, **k: service)
    async with create_connected_server_and_client_session(server) as session:
        waiting = asyncio.create_task(session.call_tool("wait_for_events", {"jobs": [REF]}))
        await asyncio.wait_for(service.waiting.wait(), 2)
        result = await asyncio.wait_for(session.call_tool("list_jobs", {}), 2)
        assert not result.isError
        assert not waiting.done()
        service.release.set()
        event = await asyncio.wait_for(waiting, 2)
        assert event.structuredContent["events"][0]["type"] == "failed"
        assert service.calls[0][2]["timeout"] == 43200


@pytest.mark.asyncio
async def test_real_stdio_catalog_has_no_stdout_noise_or_cluster_access(tmp_path):
    state = tmp_path / "state"
    params = StdioServerParameters(command=sys.executable, args=[
        "-m", "falcon", "--config", str(tmp_path / "missing-config"),
        "mcp", "serve", "--state-dir", str(state),
    ])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=15)) as session:
            await session.initialize()
            catalog = await session.list_tools()
            assert len(catalog.tools) == 9
    assert not state.exists(), "catalog discovery must not open state or query Kubernetes"


def test_cli_mcp_help_without_loading_sdk_or_starting_a_server():
    from falcon.cli import _parser, _rewrite_shorthand

    assert _rewrite_shorthand(["mcp", "serve"], DEFAULT_CONFIG) == ["mcp", "serve"]
    args = _parser(DEFAULT_CONFIG).parse_args(["mcp", "config", "--agent", "codex"])
    assert args.mcp_command == "config"


def test_cli_missing_optional_dependency_has_actionable_message():
    import builtins

    from falcon.cli import main

    original = builtins.__import__

    def without_sdk(name, *args, **kwargs):
        if name == "mcp_server":
            raise ImportError("mcp not installed")
        return original(name, *args, **kwargs)

    out, err = io.StringIO(), io.StringIO()
    with patch("builtins.__import__", side_effect=without_sdk), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["mcp", "serve"])
    assert code == 2
    assert "falcon-k8s[mcp]" in err.getvalue()
    assert out.getvalue() == ""
