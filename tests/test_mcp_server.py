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

REF = {"name": "train", "uid": "uid-1"}


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
        return {"job": {**REF, "namespace": "default"}, "namespace": "default"}

    def list_jobs(self, **kwargs):
        return {"jobs": [{**REF, "namespace": "default"}]}

    def cancel_job(self, job):
        self.calls.append(("cancel", job))
        return {"job": job, "cancelled": True}

    def restart_job(self, job, overrides=None, summary=None):
        self.calls.append(("restart", job, overrides))
        if summary is not None:
            self.calls.append(("summary", summary))
        return {"job": {**job, "uid": "replacement-uid"}}

    async def wait_for_events(self, jobs, **kwargs):
        self.calls.append(("wait", jobs, kwargs))
        self.waiting.set()
        await self.release.wait()
        return {"events": [{"type": "failed", "job": {**jobs[0], "namespace": "default"}}], "cursor": 1}

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
            assert "idempotency_key" not in tools["restart_job"].inputSchema["properties"]
            reference = tools["inspect_job"].inputSchema["$defs"]["JobReference"]
            assert "namespace" not in reference["properties"]
            assert reference["required"] == ["name"]
            assert tools["submit_job"].annotations.idempotentHint
            schema = tools["submit_job"].inputSchema
            assert "idempotency_key" in schema["required"]
            planning_schema = tools["plan_job"].inputSchema
            planning_ref = planning_schema["properties"]["request"]["$ref"].rsplit("/", 1)[-1]
            assert "command" not in planning_schema["$defs"][planning_ref].get("required", [])
            policy = tools["wait_for_events"].inputSchema["properties"]["scheduling_policy"]
            assert "timeout" not in tools["wait_for_events"].inputSchema["properties"]
            assert "seven" not in tools["wait_for_events"].description.lower()
            assert "7d" not in tools["wait_for_events"].description.lower()
            assert "functions.wait" in tools["wait_for_events"].description
            assert policy["default"] == "report_blocked"
            assert policy["enum"] == ["report_blocked", "wait_for_capacity"]
            event_filter = tools["wait_for_events"].inputSchema["properties"]["event_types"]
            choices = next(item for item in event_filter["anyOf"] if item["type"] == "array")
            assert set(choices["items"]["enum"]) == {
                "succeeded", "failed", "pod_failed", "evicted", "eviction_risk", "warning",
            }
            service.assert_not_called()


@pytest.mark.asyncio
async def test_protocol_resource_only_preview_does_not_relax_submission_command():
    service = FakeService(DEFAULT_CONFIG)
    server = create_server(DEFAULT_CONFIG, service_factory=lambda *a, **k: service)
    async with create_connected_server_and_client_session(server) as session:
        planned = await session.call_tool("plan_job", {"request": {"gpu": "2080ti", "gpus": 1}})
        assert not planned.isError
        assert service.calls[0][0] == "plan"
        assert "command" not in service.calls[0][1]
        submitted = await session.call_tool("submit_job", {
            "request": {"gpu": "2080ti", "gpus": 1}, "idempotency_key": "needs-command",
        })
        assert submitted.isError
        assert len(service.calls) == 1


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
        assert "namespace" not in result.structuredContent
        invalid = await session.call_tool("submit_job", {
            "request": {"command": ["true"], "backoffLimit": 6}, "idempotency_key": "bad",
        })
        assert invalid.isError
        wrong_namespace = await session.call_tool("cancel_job", {"job": {"name": "train", "namespace": "default"}})
        assert wrong_namespace.isError
        assert len(service.calls) == 1
        assert not service.waiting.is_set(), "submission must not start an event wait"
    assert service.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["inspect_job", "job_logs", "job_metrics", "cancel_job", "restart_job"])
async def test_protocol_accepts_name_only_and_hides_namespace_recursively(method):
    service = FakeService(DEFAULT_CONFIG)
    seen = []

    def operation(job, **kwargs):
        seen.append(job)
        return {"namespace": "default", "job": {**REF, "namespace": "default"},
                "nested": [{"namespace": "default", "previous_job": {**REF, "namespace": "default"}}],
                "stdout": '{"namespace": "workload output"}'}

    setattr(service, method, operation)
    server = create_server(DEFAULT_CONFIG, service_factory=lambda *a, **k: service)
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(method, {"job": {"name": "train"}})
        assert not result.isError
        assert seen == [{"name": "train"}]
        assert result.structuredContent == {
            "job": REF, "nested": [{"previous_job": REF}],
            "stdout": '{"namespace": "workload output"}',
        }
        invalid = await session.call_tool(method, {"job": {"name": "train", "namespace": "default"}})
        assert invalid.isError
        assert len(seen) == 1


@pytest.mark.asyncio
async def test_name_only_wait_continuation_uses_resolved_uid_without_namespace():
    service = FakeService(DEFAULT_CONFIG)

    async def wait(jobs, **kwargs):
        return {"events": [], "jobs": [{**REF, "namespace": "default"}],
                "cursor": 12, "timed_out": True}

    service.wait_for_events = wait
    server = create_server(DEFAULT_CONFIG, service_factory=lambda *a, **k: service)
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("wait_for_events", {"jobs": [{"name": "train"}]})
        assert not result.isError
        assert result.structuredContent["jobs"] == [REF]
        assert result.structuredContent["continuation"]["arguments"]["jobs"] == [REF]


@pytest.mark.asyncio
@pytest.mark.parametrize("field", [
    "backoffLimit", "backoff_limit", "restartPolicy", "restart_policy",
    "retry", "retries", "retry_policy", "podFailurePolicy", "restartPolicyRules",
])
async def test_agent_cannot_supply_retry_settings_to_submit_or_restart(field):
    service = FakeService(DEFAULT_CONFIG)
    server = create_server(DEFAULT_CONFIG, service_factory=lambda *a, **k: service)
    async with create_connected_server_and_client_session(server) as session:
        submitted = await session.call_tool("submit_job", {
            "request": {"command": ["true"], field: 5}, "idempotency_key": "blocked-retry",
        })
        restarted = await session.call_tool("restart_job", {
            "job": REF, "overrides": {field: 5},
        })
        assert submitted.isError
        assert restarted.isError
        assert service.calls == [], "retry settings must be rejected before any mutation"


@pytest.mark.asyncio
async def test_synchronous_wait_stays_pending_until_event_and_allows_other_tools():
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
        assert service.calls[0][2]["timeout"] == 604800
        assert service.calls[0][2]["scheduling_policy"] == "report_blocked"


@pytest.mark.asyncio
async def test_protocol_restart_manages_identity_without_agent_supplied_key():
    service = FakeService(DEFAULT_CONFIG)
    server = create_server(DEFAULT_CONFIG, service_factory=lambda *a, **k: service)
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("restart_job", {
            "job": REF, "overrides": {"command": ["true"]},
        })
        assert not result.isError
        assert result.structuredContent["job"] == {**REF, "uid": "replacement-uid"}
        assert service.calls == [("restart", REF, {"command": ["true"]})]


@pytest.mark.asyncio
async def test_protocol_restart_summary_is_optional_bounded_and_passed_to_service():
    service = FakeService(DEFAULT_CONFIG)
    server = create_server(DEFAULT_CONFIG, service_factory=lambda *a, **k: service)
    summary = {"issue": "Missing module", "fix": "Installed the dependency"}
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("restart_job", {"job": REF, "summary": summary})
        assert not result.isError
        assert ("summary", summary) in service.calls
        count = len(service.calls)
        for bad in [{"issue": "x"}, {"issue": "x", "fix": "x" * 1001}, {"issue": "x", "fix": "y", "extra": "z"}]:
            invalid = await session.call_tool("restart_job", {"job": REF, "summary": bad})
            assert invalid.isError
        assert len(service.calls) == count


@pytest.mark.asyncio
async def test_host_can_detach_and_reattach_to_the_same_pending_mcp_wait():
    service = FakeService(DEFAULT_CONFIG)
    server = create_server(DEFAULT_CONFIG, service_factory=lambda *a, **k: service)
    async with create_connected_server_and_client_session(server) as session:
        pending = asyncio.create_task(session.call_tool("wait_for_events", {
            "jobs": [REF], "cursor": 0, "startup_time_seconds": 600,
            "scheduling_policy": "wait_for_capacity",
        }))
        await asyncio.wait_for(service.waiting.wait(), 2)
        # A yielding/reattaching host must retain the request task, rather
        # than cancelling the MCP call or making another wait_for_events call.
        for _ in range(2):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(pending), 0.01)
            assert not pending.done()
            question = await session.call_tool("plan_job", {"request": {"command": ["true"], "cpu": "1"}})
            assert not question.isError
        assert len([call for call in service.calls if call[0] == "wait"]) == 1
        service.release.set()
        result = await asyncio.wait_for(pending, 2)
        assert result.structuredContent["events"][0]["job"] == REF
        assert result.structuredContent["cursor"] == 1
        wait = next(call for call in service.calls if call[0] == "wait")
        assert wait[2]["startup_time_seconds"] == 600
        assert wait[2]["scheduling_policy"] == "wait_for_capacity"


@pytest.mark.asyncio
async def test_event_result_survives_while_parent_handles_another_request():
    service = FakeService(DEFAULT_CONFIG)
    server = create_server(DEFAULT_CONFIG, service_factory=lambda *a, **k: service)
    async with create_connected_server_and_client_session(server) as session:
        pending = asyncio.create_task(session.call_tool("wait_for_events", {"jobs": [REF]}))
        finished = asyncio.Event()
        pending.add_done_callback(lambda _: finished.set())
        await asyncio.wait_for(service.waiting.wait(), 2)
        service.release.set()
        # The event arrives before the parent retrieves the result. Finishing
        # another request must not cancel, consume, or require a second wait.
        await asyncio.wait_for(finished.wait(), 2)
        question = await session.call_tool("plan_job", {"request": {"command": ["true"], "cpu": "1"}})
        assert not question.isError
        assert pending.done()
        result = await pending
        assert result.structuredContent["events"][0]["job"] == REF
        assert result.structuredContent["cursor"] == 1
        assert len([call for call in service.calls if call[0] == "wait"]) == 1


@pytest.mark.asyncio
async def test_expired_wait_offers_cursor_safe_continuation_with_original_settings():
    service = FakeService(DEFAULT_CONFIG)

    async def wait(jobs, **kwargs):
        service.calls.append(("wait", jobs, kwargs))
        if len(service.calls) == 1:
            return {"events": [], "cursor": 71, "timed_out": True}
        return {"events": [{"type": "succeeded", "job": jobs[0]}], "cursor": 72}

    service.wait_for_events = wait
    server = create_server(DEFAULT_CONFIG, service_factory=lambda *a, **k: service)
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("wait_for_events", {
            "jobs": [REF], "cursor": 12, "event_types": ["succeeded", "failed"],
            "scheduling_policy": "wait_for_capacity", "startup_time_seconds": 600,
        })
        assert not result.isError
        continuation = result.structuredContent["continuation"]
        assert continuation["arguments"] == {
            "jobs": [REF], "cursor": 71, "event_types": ["succeeded", "failed"],
            "scheduling_policy": "wait_for_capacity", "startup_time_seconds": 600,
        }
        assert service.calls[0][2]["timeout"] == 604800
        resumed = await session.call_tool(continuation["tool"], continuation["arguments"])
        assert resumed.structuredContent["events"][0]["type"] == "succeeded"
        assert "continuation" not in resumed.structuredContent
        assert service.calls[1][2]["cursor"] == 71
        assert service.calls[1][2]["timeout"] == 604800


@pytest.mark.asyncio
async def test_deleted_job_does_not_offer_wait_continuation():
    service = FakeService(DEFAULT_CONFIG)

    async def wait(jobs, **kwargs):
        return {"events": [], "cursor": 71, "timed_out": False, "disconnected": True}

    service.wait_for_events = wait
    server = create_server(DEFAULT_CONFIG, service_factory=lambda *a, **k: service)
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("wait_for_events", {"jobs": [REF]})
        assert not result.isError
        assert result.structuredContent["disconnected"]
        assert "continuation" not in result.structuredContent


@pytest.mark.asyncio
async def test_agent_cannot_shorten_the_fixed_wait():
    service = FakeService(DEFAULT_CONFIG)
    service.release.set()
    server = create_server(DEFAULT_CONFIG, service_factory=lambda *a, **k: service)
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("wait_for_events", {
            "jobs": [REF], "timeout": 0,
        })
        # Hosts may reject unknown arguments or ignore them, but an old caller
        # must never be able to shorten the wait through a removed parameter.
        if result.isError:
            assert not service.calls
        else:
            assert service.calls[0][2]["timeout"] == 604800


@pytest.mark.asyncio
async def test_protocol_wait_accepts_capacity_policy_and_rejects_unknown_value():
    service = FakeService(DEFAULT_CONFIG)
    service.release.set()
    server = create_server(DEFAULT_CONFIG, service_factory=lambda *a, **k: service)
    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("wait_for_events", {
            "jobs": [REF], "scheduling_policy": "wait_for_capacity",
        })
        assert not result.isError
        assert service.calls[0][2]["scheduling_policy"] == "wait_for_capacity"
        invalid = await session.call_tool("wait_for_events", {
            "jobs": [REF], "scheduling_policy": "later",
        })
        assert invalid.isError
        assert len(service.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["monitor_unavailable", "monitor_recovered"])
async def test_protocol_rejects_internal_monitor_events(event_type):
    with patch("falcon.mcp_server.JobService") as service:
        server = create_server(DEFAULT_CONFIG)
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool("wait_for_events", {
                "jobs": [REF], "event_types": [event_type],
            })
            assert result.isError
            service.assert_not_called()


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


def test_cli_missing_mcp_dependency_has_actionable_message():
    import builtins

    from falcon.cli import main

    original = builtins.__import__

    def without_sdk(name, *args, **kwargs):
        if name in {"mcp_server", "falcon.mcp_server"}:
            raise ImportError("mcp not installed")
        return original(name, *args, **kwargs)

    out, err = io.StringIO(), io.StringIO()
    with patch("builtins.__import__", side_effect=without_sdk), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["mcp", "serve"])
    assert code == 2
    assert "pip install --force-reinstall falcon-k8s" in err.getvalue()
    assert "optional" not in err.getvalue().lower()
    assert out.getvalue() == ""
