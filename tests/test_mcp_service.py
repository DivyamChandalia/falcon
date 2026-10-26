from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from falcon.config import DEFAULT_CONFIG
from falcon.kubernetes import KubernetesError, ProcessResult
from falcon.mcp_service import (
    AGENT_LABEL,
    IDEMPOTENCY_LABEL,
    PREVIOUS_JOB_UID_ANNOTATION,
    IdempotencyConflict,
    JobService,
    MCPServiceError,
    StaleJobReference,
)
from falcon.models import NodeResources


def _config() -> dict:
    value = copy.deepcopy(DEFAULT_CONFIG)
    value["cluster"].update(
        {
            "namespace": "research",
            "resource_service_url": None,
            "kube_state_metrics_url": None,
            "gpu_label": "gpu-type",
        }
    )
    value["runtime"]["python_environment"] = "none"
    value["runtime"]["environment"] = {}
    value["job"]["backoff_limit"] = 7
    return value


class FakeClient:
    context = "research-context"

    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.pods: list[dict] = []
        self.log_text: dict[str, str] = {}
        self.log_calls: list[dict] = []
        self.event_items: list[dict] = []
        self.event_queries: list[list[str]] = []
        self.created_manifests: list[dict] = []
        self.deleted: list[tuple[str, str]] = []
        self.delete_policies: list[str] = []
        self.closed = False
        self._uid_number = 0

    def get_json(self, resource: str, name: str, *, namespace: str | None = None):
        del namespace
        if resource not in {"job.batch", "jobs.batch"} or name not in self.jobs:
            raise KubernetesError("NotFound")
        return copy.deepcopy(self.jobs[name])

    def list_jobs(self, namespace: str | None = None):
        del namespace
        return {"items": copy.deepcopy(list(self.jobs.values()))}

    def events(self, names, *, namespace: str | None = None, limit: int = 50):
        del namespace, limit
        self.event_queries.append(list(names))
        return copy.deepcopy(self.event_items)

    def create_job(self, manifest: dict):
        if manifest["metadata"]["name"] in self.jobs:
            raise KubernetesError("AlreadyExists")
        self._uid_number += 1
        value = copy.deepcopy(manifest)
        value.setdefault("metadata", {})["uid"] = f"job-uid-{self._uid_number}"
        value["metadata"].setdefault("creationTimestamp", "2026-01-01T00:00:00Z")
        self.created_manifests.append(copy.deepcopy(value))
        self.jobs[value["metadata"]["name"]] = value
        return value

    def job_pods(self, name: str, *, namespace: str | None = None):
        del namespace
        return [
            copy.deepcopy(pod)
            for pod in self.pods
            if (pod.get("metadata") or {}).get("labels", {}).get("job-name") == name
        ]

    def pod_logs(
        self,
        pod_name: str,
        *,
        namespace: str | None = None,
        tail: int = -1,
        container: str | None = None,
        timestamps: bool = False,
        since_time: str | None = None,
    ):
        self.log_calls.append(
            {
                "pod": pod_name,
                "namespace": namespace,
                "tail": tail,
                "container": container,
                "timestamps": timestamps,
                "since_time": since_time,
            }
        )
        return ProcessResult(
            ("kubectl", "logs", pod_name),
            0,
            stdout=self.log_text.get(pod_name, ""),
        )

    def delete_pod_uid(self, name, uid, *, namespace=None):
        self.pods[:] = [pod for pod in self.pods if not (
            pod["metadata"]["name"] == name and pod["metadata"]["uid"] == uid
        )]

    def finish_orphaned_job_deletion(self, job, *, namespace=None):
        current = self.jobs.get(job["metadata"]["name"])
        if current and current["metadata"].get("deletionTimestamp"):
            finalizers = current["metadata"].get("finalizers", [])
            if "orphan" in finalizers:
                finalizers.remove("orphan")
                if not finalizers:
                    del self.jobs[job["metadata"]["name"]]

    def delete_job_uid(
        self, name: str, uid: str, *, namespace: str | None = None,
        propagation_policy: str = "Background",
    ):
        del namespace
        current = self.jobs.get(name)
        if current is None or (current.get("metadata") or {}).get("uid") != uid:
            raise KubernetesError("NotFound")
        del self.jobs[name]
        self.deleted.append((name, uid))
        self.delete_policies.append(propagation_policy)
        if propagation_policy == "Orphan":
            for pod in self.pods:
                metadata = pod["metadata"]
                metadata["ownerReferences"] = [
                    owner for owner in metadata.get("ownerReferences", [])
                    if owner.get("uid") != uid
                ]
        return SimpleNamespace(returncode=0)

    def close(self) -> None:
        self.closed = True


def _service(tmp_path: Path, client: FakeClient | None = None) -> JobService:
    return JobService(_config(), client=client or FakeClient(), state_dir=tmp_path)


def _request(**overrides) -> dict:
    value = {
        "name": "train",
        "command": ["python", "train.py"],
        "cpu": "1",
        "memory": "1Gi",
    }
    value.update(overrides)
    return value


def _pod(
    job: dict,
    name: str,
    uid: str,
    *,
    phase: str = "Running",
    created: str = "2026-01-01T00:00:00Z",
    exit_code: int | None = None,
    reason: str | None = None,
) -> dict:
    status: dict = {"phase": phase}
    if reason:
        status["reason"] = reason
    if exit_code is not None:
        status["containerStatuses"] = [
            {
                "name": "main",
                "state": {"terminated": {"exitCode": exit_code, "reason": reason}},
            }
        ]
    return {
        "kind": "Pod",
        "metadata": {
            "name": name,
            "uid": uid,
            "creationTimestamp": created,
            "labels": {
                "job-name": job["metadata"]["name"],
                "batch.kubernetes.io/controller-uid": job["metadata"]["uid"],
            },
            "ownerReferences": [
                {
                    "kind": "Job",
                    "name": job["metadata"]["name"],
                    "uid": job["metadata"]["uid"],
                }
            ],
        },
        "spec": {"containers": [{"name": "main"}]},
        "status": status,
    }


def test_resource_only_gpu_plan_exposes_recommendations_without_creating_job(tmp_path, monkeypatch):
    client = FakeClient()
    service = _service(tmp_path, client)
    monkeypatch.setattr(service, "_planning_nodes", lambda: [
        NodeResources("node-gpu", 64, 4, 256, 32, 4, 3, "2080ti"),
    ])
    request = {"gpu": "2080ti", "gpus": 1}
    try:
        result = service.plan_job(request)
        assert result["resource_only"]
        assert "command" not in result["summary"]
        assert request == {"gpu": "2080ti", "gpus": 1}
        info = result["resource_plan"]["scheduling"]
        assert info["schedulable_nodes"] == ["node-gpu"]
        assert info["maximum_allocation"]["cpu_cores"] == 60
        assert info["maximum_allocation"]["memory_gib"] == 224
        assert info["recommended_allocation"]["cpu"] == "16"
        assert info["recommended_allocation"]["memory"] == "63Gi"
        assert client.created_manifests == []
        with pytest.raises(MCPServiceError, match="command"):
            service.submit_job(request, "no-command")
    finally:
        asyncio.run(service.close())


def test_impossible_gpu_cpu_ram_request_can_be_previewed_but_not_submitted(tmp_path, monkeypatch):
    client = FakeClient()
    service = _service(tmp_path, client)
    monkeypatch.setattr(service, "_planning_nodes", lambda: [
        NodeResources("small", 8, 0, 32, 0, 2, 0, "2080ti"),
    ])
    request = _request(gpu="2080ti", gpus=1, cpu="16", memory="64Gi")
    try:
        result = service.plan_job(request)
        assert not result["resource_plan"]["scheduling"]["fits_any_node_capacity"]
        with pytest.raises(MCPServiceError, match="cannot fit any matching node"):
            service.submit_job(request, "impossible")
        assert client.created_manifests == []
    finally:
        asyncio.run(service.close())


def test_busy_gpu_submission_is_allowed_and_planned_from_capacity(tmp_path, monkeypatch):
    client = FakeClient()
    service = _service(tmp_path, client)
    monkeypatch.setattr(service, "_planning_nodes", lambda: [
        NodeResources("busy", 64, 63, 256, 255, 4, 4, "2080ti"),
    ])
    try:
        preview = service.plan_job({"gpu": "2080ti", "gpus": 1})
        info = preview["resource_plan"]["scheduling"]
        assert info["recommended_allocation"]["basis"] == "node_capacity"
        result = service.submit_job(_request(gpu="2080ti", gpus=1, cpu="16", memory="63Gi"), "queued")
        assert result["created"]
        assert len(client.created_manifests) == 1
    finally:
        asyncio.run(service.close())


def test_plan_is_deep_copied_and_always_non_retrying(tmp_path: Path) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    planned = service.plan_job(_request())

    assert planned["job"] == {"name": "train", "namespace": "research", "uid": None}
    assert planned["summary"]["retry"] == {
        "backoff_limit": 0,
        "restart_policy": "Never",
    }
    assert "manifest" not in planned
    assert service.config["job"]["backoff_limit"] == 7
    asyncio.run(service.close())


@pytest.mark.parametrize("field", [
    "backoffLimit", "backoff_limit", "restartPolicy", "restart_policy",
    "retry", "retries", "retry_policy", "podFailurePolicy", "restartPolicyRules",
])
def test_service_rejects_agent_retry_configuration_before_mutation(tmp_path, field):
    client = FakeClient()
    service = _service(tmp_path, client)
    try:
        with pytest.raises(MCPServiceError):
            service.submit_job(_request(**{field: 5}), "blocked-submit")
        assert client.created_manifests == []
        source = _terminal_job(service, client)
        with pytest.raises(MCPServiceError):
            service.restart_job(source, overrides={field: 5})
        assert len(client.created_manifests) == 1
        assert client.deleted == []
    finally:
        asyncio.run(service.close())


def test_submit_idempotency_reconciles_and_does_not_resurrect_deleted_job(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    first = service.submit_job(_request(), "opaque-secret-key")
    manifest = client.created_manifests[0]
    assert manifest["metadata"]["labels"][AGENT_LABEL] == "true"
    assert IDEMPOTENCY_LABEL in manifest["metadata"]["labels"]
    assert manifest["spec"]["backoffLimit"] == 0
    assert manifest["spec"]["template"]["spec"]["restartPolicy"] == "Never"

    reused = service.submit_job(_request(), "opaque-secret-key")
    assert reused["uid"] == first["uid"]
    assert reused["reused"] is True
    assert len(client.created_manifests) == 1

    with pytest.raises(IdempotencyConflict):
        service.submit_job(_request(cpu="2"), "opaque-secret-key")

    del client.jobs["train"]
    deleted = service.submit_job(_request(), "opaque-secret-key")
    assert deleted["deleted"] is True
    assert deleted["uid"] == first["uid"]
    assert len(client.created_manifests) == 1

    raw_db = service._db_path.read_bytes()
    assert b"opaque-secret-key" not in raw_db
    asyncio.run(service.close())


def test_generated_submit_names_include_key_but_retries_reuse_one_job(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    request = _request()
    request.pop("name")

    first = service.submit_job(request, "independent-key-a")
    second = service.submit_job(request, "independent-key-b")
    assert first["name"] != second["name"]
    assert len(client.created_manifests) == 2

    retry = service.submit_job(request, "independent-key-a")
    assert retry["uid"] == first["uid"]
    assert retry["reused"] is True
    assert len(client.created_manifests) == 2
    asyncio.run(service.close())


def test_list_jobs_is_compact_and_reports_actual_retry_settings(tmp_path: Path) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    service.submit_job(
        _request(command=["python", "secret.py", "--secret", "value"], env={"TOKEN": "secret"}),
        "list-key",
    )
    listed = service.list_jobs()["jobs"][0]

    assert listed["job"]["uid"]
    assert set(listed) == {
        "job", "name", "namespace", "uid", "status", "age", "request",
        "agent_managed", "retry",
    }
    assert "command" not in listed
    assert "labels" not in listed
    assert "resources" not in listed
    assert listed["retry"]["backoff_limit"] == 0
    asyncio.run(service.close())


def test_inspect_includes_owned_pending_pod_diagnostics_and_compact_events(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    ref = service.submit_job(_request(), "inspect-diagnostics-key")["job"]
    job = client.jobs[ref["name"]]
    pending = _pod(job, "pending", "pod-pending", phase="Pending")
    pending["status"]["containerStatuses"] = [{
        "name": "main",
        "state": {
            "waiting": {
                "reason": "ImagePullBackOff",
                "message": "Back-off pulling image",
            }
        },
    }]
    pending["status"]["conditions"] = [{
        "type": "PodScheduled",
        "status": "False",
        "reason": "Unschedulable",
        "message": "0/3 nodes are available: insufficient memory",
    }]
    client.pods[:] = [pending]
    client.event_items[:] = [
        {
            "metadata": {"uid": "event-job", "resourceVersion": "1"},
            "involvedObject": {
                "kind": "Job",
                "name": ref["name"],
                "namespace": ref["namespace"],
                "uid": ref["uid"],
            },
            "type": "Warning",
            "reason": "FailedScheduling",
            "message": "0/3 nodes are available: insufficient memory",
        },
        {
            "metadata": {"uid": "event-pod", "resourceVersion": "2"},
            "involvedObject": {
                "kind": "Pod",
                "name": "pending",
                "namespace": ref["namespace"],
                "uid": "pod-pending",
            },
            "type": "Warning",
            "reason": "ImagePullBackOff",
            "message": "x" * 5_000,
        },
        {
            "metadata": {"uid": "event-foreign"},
            "involvedObject": {
                "kind": "Pod",
                "name": "pending",
                "namespace": ref["namespace"],
                "uid": "different-pod-uid",
            },
            "type": "Warning",
            "reason": "ForeignEvent",
            "message": "must be filtered",
        },
    ]

    inspected = service.inspect_job(ref)
    assert client.event_queries == [[ref["name"], "pending"]]
    pod = inspected["pods"][0]
    assert pod["reason"] == "ImagePullBackOff"
    assert pod["container_reason"] == "ImagePullBackOff"
    assert pod["exit_code"] is None
    assert pod["conditions"][0]["reason"] == "Unschedulable"
    assert pod["conditions"][0]["message"] == (
        "0/3 nodes are available: insufficient memory"
    )

    events = inspected["events"]
    assert [event["reason"] for event in events] == [
        "FailedScheduling",
        "ImagePullBackOff",
    ]
    assert all("metadata" not in event for event in events)
    assert events[1]["message_truncated"] is True
    assert len(events[1]["message"]) <= 2_048
    asyncio.run(service.close())


def test_logs_use_exact_owner_uid_incremental_cursors_and_workload_exit_code(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    ref = service.submit_job(_request(), "logs-key")["job"]
    job = client.jobs[ref["name"]]
    running = _pod(job, "running", "pod-running", created="2026-01-01T00:00:01Z")
    failed = _pod(
        job,
        "failed",
        "pod-failed",
        phase="Failed",
        created="2026-01-01T00:00:02Z",
        exit_code=17,
        reason="Error",
    )
    foreign = copy.deepcopy(running)
    foreign["metadata"]["name"] = "foreign"
    foreign["metadata"]["uid"] = "pod-foreign"
    foreign["metadata"]["ownerReferences"][0]["uid"] = "different-job-uid"
    client.pods[:] = [running, failed, foreign]
    client.log_text["running"] = (
        "2026-01-01T00:00:01Z one\n"
        "2026-01-01T00:00:02Z two\n"
        "2026-01-01T00:00:03Z three\n"
    )
    client.log_text["failed"] = "2026-01-01T00:00:04Z failed\n"

    current = service.job_logs(ref, tail=2)
    assert len(current["logs"]) == 1
    assert "three" in current["logs"][0]["stdout"]
    assert current["logs"][0]["gap_possible"] is True
    cursor = current["cursors"]["pod-running/main"]

    client.log_text["running"] = (
        "2026-01-01T00:00:03Z three\n"
        "2026-01-01T00:00:04Z four\n"
        "2026-01-01T00:00:05Z five\n"
    )
    incremental = service.job_logs(ref, tail=2, cursors={"pod-running/main": cursor})
    assert "three" not in incremental["logs"][0]["stdout"]
    assert "four" in incremental["logs"][0]["stdout"]
    assert client.log_calls[-1]["timestamps"] is True
    assert client.log_calls[-1]["since_time"] == "2026-01-01T00:00:03Z"

    failed_result = service.job_logs(ref, pods=["failed"])
    entry = failed_result["logs"][0]
    assert entry["returncode"] == 0
    assert entry["kubectl_returncode"] == 0
    assert entry["container_exit_code"] == 17
    assert entry["reason"] == "Error"
    assert entry["created_at"] == "2026-01-01T00:00:02Z"
    with pytest.raises(MCPServiceError, match="not an attempt"):
        service.job_logs(ref, pods=["foreign"])
    asyncio.run(service.close())


def test_job_logs_deduplicate_limit_and_carry_cursor_when_no_new_lines(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    ref = service.submit_job(_request(), "bounded-log-key")["job"]
    job = client.jobs[ref["name"]]
    first_pod = _pod(job, "first", "pod-first", created="2026-01-01T00:00:01Z")
    second_pod = _pod(job, "second", "pod-second", created="2026-01-01T00:00:02Z")
    client.pods[:] = [first_pod, second_pod]
    client.log_text["first"] = "2026-01-01T00:00:01Z first\n"
    client.log_text["second"] = "2026-01-01T00:00:02Z second\n"

    initial = service.job_logs(ref, pods=["first", "first", "second"], limit=2)
    assert [entry["pod"] for entry in initial["logs"]] == ["first", "second"]
    cursor = initial["cursors"]["pod-first/main"]

    # The server returns only the already-seen boundary line.  The service
    # must retain the opaque cursor rather than making the next poll replay it.
    unchanged = service.job_logs(
        ref,
        pods=["first"],
        cursors={"pod-first/main": cursor},
    )
    assert unchanged["cursors"]["pod-first/main"] == cursor
    assert unchanged["logs"][0]["next_cursor"] == cursor
    assert unchanged["logs"][0]["stdout"] == ""

    many_pods = [
        _pod(
            job,
            f"pod-{index}",
            f"pod-uid-{index}",
            created=f"2026-01-01T00:00:{index + 10:02d}Z",
        )
        for index in range(11)
    ]
    client.pods[:] = many_pods
    with pytest.raises(MCPServiceError, match="at most 10"):
        service.job_logs(ref, pods=[pod["metadata"]["name"] for pod in many_pods])

    client.pods[:] = many_pods[:10]
    for index in range(10):
        name = f"pod-{index}"
        client.log_text[name] = (
            f"2026-01-01T00:00:{index + 10:02d}Z " + "x" * 9_000 + "\n"
        )
    bounded = service.job_logs(
        ref,
        pods=[pod["metadata"]["name"] for pod in many_pods[:10]],
        limit=10,
        tail=1,
    )
    assert len(json.dumps(bounded, ensure_ascii=False).encode("utf-8")) <= 32 * 1024
    assert bounded["truncated"] is True
    assert any(entry.get("truncated") for entry in bounded["logs"])
    asyncio.run(service.close())


def test_log_cursors_preserve_rfc3339_nanoseconds(tmp_path: Path) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    ref = service.submit_job(_request(), "nanosecond-log-key")["job"]
    job = client.jobs[ref["name"]]
    pod = _pod(job, "running", "pod-nanos", created="2026-01-01T00:00:01Z")
    client.pods[:] = [pod]
    client.log_text["running"] = (
        "2026-01-01T00:00:00.123456001Z first\n"
        "2026-01-01T00:00:00.123456999Z old\n"
    )
    first = service.job_logs(ref, tail=2)
    cursor = first["cursors"]["pod-nanos/main"]

    # Kubernetes can return the old boundary line again with --since-time;
    # the new line has the same nanosecond timestamp and must be counted as a
    # second occurrence, not merged with a different timestamp in datetime's
    # six-digit microsecond representation.
    client.log_text["running"] = (
        "2026-01-01T00:00:00.123456999Z old\n"
        "2026-01-01T00:00:00.123456999Z new\n"
    )
    second = service.job_logs(
        ref,
        tail=2,
        cursors={"pod-nanos/main": cursor},
    )
    assert "new" in second["logs"][0]["stdout"]
    assert "old" not in second["logs"][0]["stdout"]
    asyncio.run(service.close())


def test_restart_rejects_failed_pod_before_job_terminal_condition(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = {
        "metadata": {
            "name": "retrying-job",
            "namespace": "research",
            "uid": "retrying-job-uid",
        },
        "spec": {"backoffLimit": 6},
        "status": {"failed": 1, "active": 0},
    }
    client.jobs["retrying-job"] = copy.deepcopy(source)
    client.pods[:] = [_pod(source, "failed-attempt", "failed-attempt-uid", phase="Failed")]

    with pytest.raises(MCPServiceError, match="no Kubernetes terminal"):
        service.restart_job(
            {"name": "retrying-job", "namespace": "research", "uid": "retrying-job-uid"},
            "retrying-job-key",
        )
    assert client.created_manifests == []
    asyncio.run(service.close())


def test_restart_strips_controller_retry_policies_and_links_source(tmp_path: Path) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "failed-job",
            "namespace": "research",
            "uid": "source-uid",
            "labels": {
                "falcon.dev/managed": "true",
                "job-name": "failed-job",
                "controller-uid": "old-controller",
            },
        },
        "spec": {
            "backoffLimit": 6,
            "selector": {"matchLabels": {"controller-uid": "old-controller"}},
            "manualSelector": True,
            "podFailurePolicy": {"rules": [{"action": "Ignore"}]},
            "backoffLimitPerIndex": 2,
            "maxFailedIndexes": 1,
            "template": {
                "metadata": {
                    "labels": {
                        "job-name": "failed-job",
                        "batch.kubernetes.io/controller-uid": "old-controller",
                        "keep-me": "yes",
                    }
                },
                "spec": {
                    "restartPolicy": "OnFailure",
                    "nodeSelector": {"gpu-type": "h100"},
                    "containers": [{
                        "name": "main",
                        "image": "image:old",
                        "command": ["python", "train.py"],
                        "restartPolicy": "Always",
                        "restartPolicyRules": [{"action": "Restart"}],
                        "env": [
                            {"name": "PATH", "value": "/usr/local/bin"},
                            {
                                "name": "SECRET_TOKEN",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "runtime-secret",
                                        "key": "token",
                                    }
                                },
                            },
                            {"name": "CHANGE_ME", "value": "old"},
                        ],
                        "resources": {
                            "requests": {"cpu": "1", "memory": "1Gi", "nvidia.com/gpu": "1"},
                            "limits": {"cpu": "1", "memory": "1Gi", "nvidia.com/gpu": "1"},
                        },
                    }],
                },
            },
        },
        "status": {"conditions": [{"type": "Failed", "status": "True"}]},
    }
    client.jobs[source["metadata"]["name"]] = copy.deepcopy(source)
    result = service.restart_job(
        {"name": "failed-job", "namespace": "research", "uid": "source-uid"},
        "restart-key",
        overrides={"gpus": 2, "env": {"CHANGE_ME": "new"}},
    )
    restarted = client.jobs[result["name"]]
    assert result["name"] == "failed-job"
    assert result["uid"] != "source-uid"
    assert client.deleted == [("failed-job", "source-uid")]
    assert client.delete_policies == ["Background"]
    spec = restarted["spec"]
    template_spec = spec["template"]["spec"]
    container = template_spec["containers"][0]
    assert spec["backoffLimit"] == 0
    assert "selector" not in spec
    assert "manualSelector" not in spec
    assert "podFailurePolicy" not in spec
    assert "backoffLimitPerIndex" not in spec
    assert "maxFailedIndexes" not in spec
    assert template_spec["restartPolicy"] == "Never"
    assert "restartPolicy" not in container
    assert "restartPolicyRules" not in container
    assert restarted["metadata"]["annotations"][PREVIOUS_JOB_UID_ANNOTATION] == "source-uid"
    assert restarted["metadata"]["labels"][AGENT_LABEL] == "true"
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "2"
    assert "job-name" not in restarted["metadata"]["labels"]
    assert "controller-uid" not in restarted["metadata"]["labels"]
    template_labels = spec["template"]["metadata"]["labels"]
    assert "job-name" not in template_labels
    assert "batch.kubernetes.io/controller-uid" not in template_labels
    assert template_labels["keep-me"] == "yes"
    env = {entry["name"]: entry for entry in container["env"]}
    assert env["PATH"]["value"] == "/usr/local/bin"
    assert env["SECRET_TOKEN"]["valueFrom"]["secretKeyRef"]["name"] == "runtime-secret"
    assert env["CHANGE_ME"] == {"name": "CHANGE_ME", "value": "new"}

    reused = service.restart_job(
        {"name": "failed-job", "namespace": "research", "uid": "source-uid"},
        "restart-key",
        overrides={"gpus": 2, "env": {"CHANGE_ME": "new"}},
    )
    assert reused["uid"] == result["uid"]
    assert reused["reused"] is True
    with pytest.raises(StaleJobReference):
        service.restart_job(
            {"name": "failed-job", "namespace": "research", "uid": "source-uid"},
            "restart-key-2",
            overrides={"gpus": 2, "env": {"CHANGE_ME": "new"}},
        )

    # TTL cleanup must not cause a retry to resurrect the replacement.
    del client.jobs["failed-job"]
    replay_after_ttl = service.restart_job(
        {"name": "failed-job", "namespace": "research", "uid": "source-uid"},
        "restart-key",
        overrides={"gpus": 2, "env": {"CHANGE_ME": "new"}},
    )
    assert replay_after_ttl["uid"] == result["uid"]
    assert replay_after_ttl["reused"] is True
    asyncio.run(service.close())


def test_restart_rejects_indexed_or_environment_changes(tmp_path: Path) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = {
        "metadata": {"name": "indexed", "namespace": "research", "uid": "uid-indexed"},
        "spec": {
            "completionMode": "Indexed",
            "template": {"spec": {"containers": [{"name": "main"}]}},
        },
        "status": {"conditions": [{"type": "Failed", "status": "True"}]},
    }
    client.jobs["indexed"] = source
    ref = {"name": "indexed", "namespace": "research", "uid": "uid-indexed"}
    with pytest.raises(MCPServiceError, match="Indexed"):
        service.restart_job(ref, "indexed-key")
    with pytest.raises(MCPServiceError, match="environment"):
        service._restart_overrides({"environment": "new"})
    assert client.deleted == []
    asyncio.run(service.close())


def _terminal_job(service: JobService, client: FakeClient) -> dict:
    ref = service.submit_job(_request(), "source-submit")["job"]
    client.jobs[ref["name"]]["status"] = {
        "conditions": [{"type": "Failed", "status": "True"}],
    }
    return ref


def test_restart_deletes_old_attempts_without_mixing_new_logs(tmp_path: Path) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    old_pod = _pod(client.jobs[source["name"]], "train-old", "pod-old", phase="Failed")
    client.pods.append(old_pod)
    client.log_text["train-old"] = "2026-01-01T00:00:01Z old failure\n"

    restarted = service.restart_job(source, "restart-logs")["job"]
    assert old_pod not in client.pods
    client.pods.append(_pod(client.jobs[source["name"]], "train-new", "pod-new"))
    client.log_text["train-new"] = "2026-01-01T00:00:02Z new attempt\n"

    old_logs = service.job_logs(source, failed_only=True)
    new_logs = service.job_logs(restarted)
    assert old_logs["archived"]
    assert [entry["pod"] for entry in old_logs["logs"]] == ["train-old"]
    assert "old failure" in old_logs["logs"][0]["stdout"]
    assert [entry["pod"] for entry in new_logs["logs"]] == ["train-new"]
    with pytest.raises(MCPServiceError, match="not an attempt"):
        service.job_logs(restarted, pods=["train-old"])
    with pytest.raises(StaleJobReference):
        service.cancel_job(source)
    with pytest.raises(StaleJobReference):
        service.job_logs({**source, "uid": "unrecorded-uid"})
    asyncio.run(service.close())


def test_name_only_references_follow_latest_run_and_explicit_uid_reads_archive(tmp_path: Path) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    original = _terminal_job(service, client)
    client.pods.append(_pod(client.jobs["train"], "old", "old-pod", phase="Failed"))
    client.log_text["old"] = "2026-01-01T00:00:01Z old failure\n"
    assert service.inspect_job({"name": "train"})["job"] == original
    replacement = service.restart_job({"name": "train"})["job"]
    assert replacement["uid"] != original["uid"]
    assert service.inspect_job({"name": "train"})["job"] == replacement
    client.pods.append(_pod(client.jobs["train"], "new", "new-pod"))
    client.log_text["new"] = "2026-01-01T00:00:02Z latest output\n"
    assert service.job_logs({"name": "train"})["logs"][0]["pod"] == "new"
    old = service.job_logs({"name": "train", "uid": original["uid"]})
    assert old["archived"]
    assert "old failure" in old["logs"][0]["stdout"]
    with pytest.raises(StaleJobReference):
        service.cancel_job({"name": "train", "uid": original["uid"]})
    cancelled = service.cancel_job({"name": "train"})
    assert cancelled["job"] == replacement
    assert client.deleted[-1] == ("train", replacement["uid"])
    asyncio.run(service.close())


def test_resolved_name_only_cancellation_rejects_a_racing_replacement(tmp_path: Path, monkeypatch) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    service.submit_job(_request(), "source")
    get = client.get_json
    calls = []

    def racing_get(resource, name, *, namespace=None):
        calls.append(namespace)
        if len(calls) == 2:
            client.jobs[name]["metadata"]["uid"] = "replacement"
        return get(resource, name, namespace=namespace)

    monkeypatch.setattr(client, "get_json", racing_get)
    with pytest.raises(StaleJobReference, match="UID changed"):
        service.cancel_job({"name": "train"})
    assert calls == ["research", "research"]
    assert not client.deleted
    asyncio.run(service.close())


@pytest.mark.asyncio
async def test_name_only_wait_binds_resolved_uid_and_returns_it(tmp_path: Path, monkeypatch) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = service.submit_job(_request(), "source")["job"]

    class Monitor:
        def __init__(self, *args, **kwargs):
            pass

        async def wait(self, jobs, **kwargs):
            assert jobs == [source]
            client.jobs["train"]["metadata"]["uid"] = "replacement"
            return {"events": [], "cursor": 12, "timed_out": True}

        async def close(self):
            pass

    monkeypatch.setattr("falcon.mcp_events.Monitor", Monitor)
    result = await service.wait_for_events([{"name": "train"}], timeout=0)
    assert result["jobs"] == [source]
    assert service._job_ref({"name": "train"})["uid"] == "replacement"
    await service.close()


def test_restart_archives_full_logs_summary_and_lineage_across_restarts(tmp_path: Path) -> None:
    from falcon.restart_history import HISTORY_ANNOTATION

    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    client.pods.append(_pod(client.jobs[source["name"]], "old", "old-uid", phase="Failed"))
    raw = "".join(f"2026-01-01T00:00:01Z failure {i}\n" for i in range(300))
    client.log_text["old"] = raw
    summary = {"issue": "Host memory exhausted", "fix": "Stream batches instead of caching them"}
    first = service.restart_job(source, summary=summary)["job"]
    assert service._restart_history.logs(source["uid"])["logs"][0]["stdout"] == raw
    assert client.log_calls[0]["tail"] == -1
    assert service.job_logs(source, tail=2)["summary"] == summary
    assert service.job_logs(source, tail=2)["logs"][0]["stdout"].splitlines()[-1].endswith("failure 299")
    assert client.jobs[first["name"]]["metadata"]["annotations"][HISTORY_ANNOTATION] == source["uid"]
    count = len(client.log_calls)
    assert service.restart_job(source, summary=summary)["job"] == first
    assert len(client.log_calls) == count
    with pytest.raises(IdempotencyConflict):
        service.restart_job(source, summary={"issue": "different", "fix": "different"})

    client.jobs[first["name"]]["status"] = {"conditions": [{"type": "Failed", "status": "True"}]}
    client.pods.append(_pod(client.jobs[first["name"]], "second", "second-uid", phase="Failed"))
    client.log_text["second"] = "2026-01-01T00:00:02Z second failure\n"
    second = service.restart_job(first)["job"]
    assert second["uid"] != first["uid"]
    attempts = service._restart_history.attempts(source["uid"], source["name"])
    assert [entry["pod"] for entry in attempts] == ["old", "second"]
    assert [entry["summary"] for entry in attempts] == [summary, None]
    asyncio.run(service.close())
    fresh = _service(tmp_path, client)
    assert fresh.job_logs(source)["summary"] == summary
    assert "second failure" in fresh.job_logs(first)["logs"][0]["stdout"]
    asyncio.run(fresh.close())


def test_restart_archive_failure_keeps_source_job_and_pods(tmp_path: Path) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    client.pods.append(_pod(client.jobs[source["name"]], "old", "old-uid", phase="Failed"))
    original = client.pod_logs
    client.pod_logs = lambda *a, **kw: ProcessResult(("kubectl",), 1, stderr="temporarily unavailable")
    with pytest.raises(MCPServiceError, match="before deletion"):
        service.restart_job(source, summary={"issue": "bug", "fix": "patch"})
    assert client.deleted == []
    assert client.pods
    client.pod_logs = original
    assert service.restart_job(source, summary={"issue": "bug", "fix": "patch"})["created"]
    asyncio.run(service.close())


def test_restart_archive_survives_delete_create_interruption(tmp_path: Path, monkeypatch) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    client.pods.append(_pod(client.jobs[source["name"]], "old", "old-uid", phase="Failed"))
    client.log_text["old"] = "2026-01-01T00:00:01Z original traceback\n"
    summary = {"issue": "Crash", "fix": "Fixed input parsing"}
    create = client.create_job

    def unavailable(manifest):
        raise KubernetesError("service unavailable")

    monkeypatch.setattr(client, "create_job", unavailable)
    with pytest.raises(KubernetesError):
        service.restart_job(source, summary=summary)
    assert source["name"] not in client.jobs
    assert not client.pods
    assert "original traceback" in service.job_logs(source)["logs"][0]["stdout"]
    monkeypatch.setattr(client, "create_job", create)
    replacement = service.restart_job(source, summary=summary)["job"]
    assert replacement["uid"] != source["uid"]
    assert service.job_logs(source)["summary"] == summary
    assert len(client.log_calls) == 1
    asyncio.run(service.close())


def test_restart_records_unavailable_evicted_logs_without_blocking_replacement(tmp_path: Path) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    pod = _pod(client.jobs[source["name"]], "evicted", "old-uid", phase="Failed")
    pod["status"]["reason"] = "Evicted"
    client.pods.append(pod)
    client.pod_logs = lambda *a, **kw: ProcessResult(("kubectl",), 1, stderr="container logs no longer available")
    result = service.restart_job(source)
    assert result["uid"] != source["uid"]
    entry = service.job_logs(source)["logs"][0]
    assert entry["unavailable"]
    assert "no longer available" in entry["stderr"]
    asyncio.run(service.close())


def test_restart_archives_init_failure_without_fetching_never_started_container(tmp_path: Path) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    pod = _pod(client.jobs[source["name"]], "old", "old-uid", phase="Failed")
    pod["spec"]["initContainers"] = [{"name": "init"}]
    pod["status"]["containerStatuses"] = [{"name": "main", "state": {"waiting": {"reason": "PodInitializing"}}}]
    pod["status"]["initContainerStatuses"] = [{"name": "init", "state": {"terminated": {"exitCode": 1}}}]
    client.pods.append(pod)
    client.log_text["old"] = "2026-01-01T00:00:01Z init failure\n"
    service.restart_job(source)
    entries = service.job_logs(source, failed_only=True)["logs"]
    assert [entry["container"] for entry in entries] == ["init", "main"]
    assert "init failure" in entries[0]["stdout"]
    assert entries[1]["stdout"] == ""
    assert "did not start" in entries[1]["stderr"]
    assert [call["container"] for call in client.log_calls] == ["init"]
    asyncio.run(service.close())


@pytest.mark.parametrize("summary", [{}, {"issue": "failure"}, {"issue": "", "fix": "fixed"}, {"issue": "x" * 1001, "fix": "fixed"}])
def test_invalid_restart_summary_rejected_before_deletion(tmp_path: Path, summary) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    with pytest.raises(MCPServiceError, match="summary"):
        service.restart_job(source, summary=summary)
    assert not client.deleted
    asyncio.run(service.close())


def test_restart_replay_cannot_delete_a_later_same_name_attempt(tmp_path: Path) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    first = service.restart_job(source, "first-restart")["job"]
    client.jobs[first["name"]]["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
    second = service.restart_job(first, "second-restart")["job"]
    assert second["name"] == first["name"] == source["name"]
    assert len({source["uid"], first["uid"], second["uid"]}) == 3
    replay = service.restart_job(source, "first-restart")
    assert replay["job"] == first
    assert replay["reused"] is True
    assert client.jobs[second["name"]]["metadata"]["uid"] == second["uid"]
    assert len(client.created_manifests) == 3
    assert len(client.deleted) == 2
    assert list(service._state_dir.glob("pending-restart-*.json")) == []
    asyncio.run(service.close())


def test_restart_implicit_identity_survives_reconnect_and_new_attempt(tmp_path: Path) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    first = service.restart_job(source)["job"]
    asyncio.run(service.close())
    service = _service(tmp_path, client)
    assert service.restart_job(source)["job"] == first
    assert len(client.created_manifests) == 2
    with pytest.raises(IdempotencyConflict):
        service.restart_job(source, overrides={"cpu": "2"})
    client.jobs[first["name"]]["status"] = {"conditions": [{"type": "Failed", "status": "True"}]}
    second = service.restart_job(first)["job"]
    assert len({source["uid"], first["uid"], second["uid"]}) == 3
    assert service.restart_job(source)["job"] == first
    assert client.jobs[second["name"]]["metadata"]["uid"] == second["uid"]
    asyncio.run(service.close())


def test_implicit_restart_recovers_legacy_orphan_pending_operation(tmp_path: Path, monkeypatch) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    pod = _pod(client.jobs[source["name"]], "old-attempt", "old-pod", phase="Failed")
    client.pods.append(pod)
    original_delete = client.delete_job_uid

    def pending(*args, **kwargs):
        client.jobs[source["name"]]["metadata"].update(
            deletionTimestamp="2026-10-22T00:00:00Z", finalizers=["orphan"],
        )

    monkeypatch.setattr(client, "delete_job_uid", pending)
    monkeypatch.setattr("falcon.mcp_service._RESTART_DELETE_TIMEOUT_SECONDS", 0)
    with pytest.raises(MCPServiceError, match="same Job reference"):
        service.restart_job(source, "legacy-pending")
    monkeypatch.setattr(client, "delete_job_uid", original_delete)
    result = service.restart_job(source)
    assert result["uid"] != source["uid"]
    assert result["name"] == source["name"]
    assert pod not in client.pods
    assert len(client.created_manifests) == 2
    assert len(client.pods) == 0
    asyncio.run(service.close())


def test_restart_recovers_stuck_orphan_and_deletes_old_pods(tmp_path: Path, monkeypatch) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    pod = _pod(client.jobs[source["name"]], "old-attempt", "old-pod", phase="Failed")
    client.pods.append(pod)
    original_delete = client.delete_job_uid

    def pending(*args, **kwargs):
        client.jobs[source["name"]]["metadata"].update(
            deletionTimestamp="2026-10-22T00:00:00Z", finalizers=["orphan"],
        )

    monkeypatch.setattr(client, "delete_job_uid", pending)
    monkeypatch.setattr("falcon.mcp_service._RESTART_DELETE_TIMEOUT_SECONDS", 0)
    with pytest.raises(MCPServiceError, match="same Job reference"):
        service.restart_job(source, "pending-orphan")
    monkeypatch.setattr(client, "delete_job_uid", original_delete)
    result = service.restart_job(source)
    assert result["uid"] != source["uid"]
    assert result["name"] == source["name"]
    assert pod not in client.pods
    assert len(client.created_manifests) == 2
    assert len(client.pods) == 0
    assert service.restart_job(source)["job"] == result["job"]
    asyncio.run(service.close())


def test_restart_leaves_unrelated_finalizers_and_does_not_create_replacement(tmp_path: Path, monkeypatch) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    client.jobs[source["name"]]["metadata"].update(
        deletionTimestamp="2026-10-22T00:00:00Z", finalizers=["unrelated", "orphan"],
    )
    monkeypatch.setattr("falcon.mcp_service._RESTART_DELETE_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(client, "delete_job_uid", lambda *a, **k: None)
    with pytest.raises(MCPServiceError, match="deletion is still pending"):
        service.restart_job(source)
    assert client.jobs[source["name"]]["metadata"]["finalizers"] == ["unrelated"]
    assert len(client.created_manifests) == 1
    asyncio.run(service.close())


def test_restart_recovers_after_delete_and_process_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    original_create = client.create_job

    def fail_create(manifest):
        raise KubernetesError("connection interrupted")

    monkeypatch.setattr(client, "create_job", fail_create)
    overrides = {"env": {"SECRET_VALUE": "private-restart-value"}, "cpu": "2"}
    with pytest.raises(KubernetesError, match="interrupted"):
        service.restart_job(source, "recover-restart", overrides=overrides)
    assert source["name"] not in client.jobs
    checkpoint, = service._state_dir.glob("pending-restart-*.json")
    assert checkpoint.stat().st_mode & 0o777 == 0o600
    assert "private-restart-value" not in service._db_path.read_bytes().decode("latin1")
    asyncio.run(service.close())

    monkeypatch.setattr(client, "create_job", original_create)
    service = _service(tmp_path, client)
    result = service.restart_job(source, "recover-restart", overrides=overrides)
    assert result["name"] == source["name"]
    assert result["uid"] != source["uid"]
    assert len(client.deleted) == 1
    env = client.jobs[source["name"]]["spec"]["template"]["spec"]["containers"][0]["env"]
    assert {"name": "SECRET_VALUE", "value": "private-restart-value"} in env
    assert not checkpoint.exists()
    assert not list(service._state_dir.glob(".restart-*"))
    asyncio.run(service.close())


def test_restart_reconciles_create_with_lost_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    original_create = client.create_job

    def lost_response(manifest):
        original_create(manifest)
        raise KubernetesError("response lost")

    monkeypatch.setattr(client, "create_job", lost_response)
    result = service.restart_job(source, "lost-reply")
    assert result["reused"] is True
    assert result["name"] == source["name"]
    assert result["uid"] != source["uid"]
    assert not list(service._state_dir.glob("pending-restart-*.json"))
    asyncio.run(service.close())


def test_restart_waits_for_deletion_and_retries_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    original_delete = client.delete_job_uid

    def pending_delete(*args, **kwargs):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(client, "delete_job_uid", pending_delete)
    monkeypatch.setattr("falcon.mcp_service._RESTART_DELETE_TIMEOUT_SECONDS", 0)
    with pytest.raises(MCPServiceError, match="same Job reference"):
        service.restart_job(source, "pending-delete")
    assert len(client.created_manifests) == 1
    monkeypatch.setattr(client, "delete_job_uid", original_delete)
    result = service.restart_job(source, "pending-delete")
    assert result["name"] == source["name"]
    assert result["uid"] != source["uid"]
    asyncio.run(service.close())


def test_restart_refuses_active_pods_even_with_terminal_job(tmp_path: Path) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    client.pods.append(_pod(client.jobs[source["name"]], "terminating", "pod-uid"))
    with pytest.raises(MCPServiceError, match="Pods to terminate"):
        service.restart_job(source, "still-terminating")
    assert client.deleted == []
    assert len(client.created_manifests) == 1
    asyncio.run(service.close())


@pytest.mark.parametrize("invalid", ["symlink", "malformed", "mismatch"])
def test_restart_checks_recovery_file_before_deleting_source(
    tmp_path: Path, invalid: str,
) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    key = service._idempotency_key("invalid-checkpoint")
    checkpoint = service._state_dir / f"pending-restart-{key}.json"
    if invalid == "symlink":
        checkpoint.symlink_to(tmp_path / "do-not-follow")
    else:
        checkpoint.write_text("invalid json" if invalid == "malformed" else "{}")
        checkpoint.chmod(0o600)
    with pytest.raises(MCPServiceError, match="recover restart checkpoint"):
        service.restart_job(source, "invalid-checkpoint")
    assert client.deleted == []
    assert client.jobs[source["name"]]["metadata"]["uid"] == source["uid"]
    asyncio.run(service.close())


def test_restart_reconciles_legacy_different_name_operation(tmp_path: Path) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)
    key = service._idempotency_key("legacy-restart")
    service._operations.reserve(
        kind="restart", key=key, request_digest=service._restart_digest(source, {}),
        planned_name="train-restart-legacy", namespace=source["namespace"],
        source_uid=source["uid"],
    )
    restarted = service.restart_job(source, "legacy-restart")
    assert restarted["name"] == "train-restart-legacy"
    assert restarted["uid"] != source["uid"]
    assert client.deleted == []
    asyncio.run(service.close())


def test_restart_does_not_delete_a_racing_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    source = _terminal_job(service, client)

    def replace_during_delete(*args, **kwargs):
        client.jobs[source["name"]]["metadata"]["uid"] = "unrelated-replacement"
        raise KubernetesError("UID precondition failed")

    monkeypatch.setattr(client, "delete_job_uid", replace_during_delete)
    with pytest.raises(IdempotencyConflict):
        service.restart_job(source, "racing-restart")
    assert client.jobs[source["name"]]["metadata"]["uid"] == "unrelated-replacement"
    assert client.deleted == []
    assert len(client.created_manifests) == 1
    asyncio.run(service.close())


def test_state_database_is_context_partitioned_and_symlinks_are_rejected(
    tmp_path: Path,
) -> None:
    first_client = FakeClient()
    first = _service(tmp_path, first_client)
    second_client = FakeClient()
    second_client.context = "other-context"
    second = _service(tmp_path, second_client)
    assert first._db_path != second._db_path
    assert first._db_path.stat().st_mode & 0o777 == 0o600
    assert first._state_dir.stat().st_mode & 0o777 == 0o700
    asyncio.run(first.close())
    asyncio.run(second.close())

    symlink_root = tmp_path / "symlink-root"
    symlink_root.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    leaf = symlink_root / hashlib.sha256(b"ctx\0research").hexdigest()[:24]
    leaf.symlink_to(target, target_is_directory=True)
    with pytest.raises(MCPServiceError, match="symlinked MCP state directory"):
        JobService._state_paths(symlink_root, context="ctx", namespace="research")


def test_metric_collector_is_context_pinned_and_sampled_once_per_second(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    monkeypatch.setattr(
        service,
        "_planning_nodes",
        lambda: [NodeResources("node-a", cpu_total=8, memory_total_gib=32, gpu_total=4, gpu_product="h100")],
    )
    service.submit_job(_request(gpu="h100"), "metric-key")
    calls = {"collect": 0, "close": 0}

    class Collector:
        def __init__(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            self.last_error = ""

        def collect(self):
            calls["collect"] += 1
            return [{
                "job": "train",
                "gpu_requested_count": 1,
                "gpu_metrics_available": True,
                "gpu_util": 42.0,
                "cpu_metrics_available": True,
                "cpu_used": 0.2,
                "cpu_allocated": 1.0,
                "metrics_updated_at": 1_700_000_000.0,
                "at_risk": False,
            }]

        def close(self):
            calls["close"] += 1

    monkeypatch.setattr("falcon.mcp_service.UsageCollector", Collector)
    ref = {"name": "train", "namespace": "research", "uid": "job-uid-1"}
    first = service.job_metrics(ref)
    second = service.job_metrics(ref)
    assert first["available"] is True
    assert second["metrics"]["gpu"]["utilization_percent"] == 42.0
    assert calls["collect"] == 1
    assert calls["kwargs"] == {
        "job_filter": "train",
        "streaming_gpu": True,
        "collect_availability": False,
        "context": "research-context",
    }
    asyncio.run(service.close())
    assert calls["close"] == 1


@pytest.mark.asyncio
async def test_wait_rejects_closed_service_before_creating_event_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)

    class ShouldNotCreate:
        def __init__(self, path):
            del path
            raise AssertionError("EventStore must not be created after close")

    monkeypatch.setattr("falcon.mcp_events.EventStore", ShouldNotCreate)
    await service.close()
    with pytest.raises(MCPServiceError, match="closed"):
        await service.wait_for_events([], timeout=0)


@pytest.mark.asyncio
async def test_wait_accepts_longer_gpu_startup_grace_and_rejects_shorter_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    ref = {"name": "train", "namespace": "research", "uid": "job-uid-1"}

    class Monitor:
        def __init__(self, client, namespace, store, metrics_provider=None):
            del client, namespace, store, metrics_provider

        async def wait(self, jobs, *, cursor=None, event_types=None, timeout=0,
                       scheduling_policy="report_blocked"):
            assert scheduling_policy == "wait_for_capacity"
            del jobs, cursor, event_types, timeout
            return {"events": [], "cursor": 0, "timed_out": True}

        async def close(self):
            return None

    monkeypatch.setattr("falcon.mcp_events.Monitor", Monitor)
    await service.wait_for_events(
        [ref], timeout=0, startup_time_seconds=300,
        scheduling_policy="wait_for_capacity",
    )
    assert service._monitor_startup_times["job-uid-1"] == 300.0
    with pytest.raises(MCPServiceError, match="between 120 and"):
        await service.wait_for_events(
            [ref], timeout=0, startup_time_seconds=119
        )
    await service.close()


@pytest.mark.asyncio
async def test_wait_reuses_one_monitor_and_closes_before_event_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    service = _service(tmp_path, client)
    service.submit_job(_request(), "wait-key")
    ref = {"name": "train", "namespace": "research", "uid": "job-uid-1"}
    job = client.jobs["train"]
    failed = _pod(
        job,
        "failed-attempt",
        "pod-failed",
        phase="Failed",
        created="2026-01-01T00:00:03Z",
        exit_code=9,
        reason="Error",
    )
    client.pods.append(failed)
    client.log_text["failed-attempt"] = "2026-01-01T00:00:03Z workload failed\n"
    order: list[str] = []

    class Store:
        def __init__(self, path):
            del path

        def close(self):
            order.append("store")

    class Monitor:
        instances = 0

        def __init__(self, client, namespace, store, metrics_provider=None):
            del client, namespace, store, metrics_provider
            Monitor.instances += 1

        async def wait(self, jobs, *, cursor=None, event_types=None, timeout=0,
                       scheduling_policy="report_blocked"):
            assert scheduling_policy == "report_blocked"
            assert jobs == [ref]
            assert timeout == 0
            return {
                "events": [{
                    "type": "failed",
                    "job": ref,
                    "details": {"reason": "Job condition failed"},
                }],
                "cursor": cursor or 0,
                "timed_out": False,
            }

        async def close(self):
            order.append("monitor")

    monkeypatch.setattr("falcon.mcp_events.EventStore", Store)
    monkeypatch.setattr("falcon.mcp_events.Monitor", Monitor)
    first = await service.wait_for_events([ref], timeout=0)
    second = await service.wait_for_events([ref], timeout=0)
    assert first["events"][0]["failure_logs"]["logs"][0]["pod"] == "failed-attempt"
    assert "failure_logs" not in first
    assert second["events"][0]["failure_logs"]["logs"][0]["pod"] == "failed-attempt"
    assert Monitor.instances == 1
    await service.close()
    assert order == ["monitor", "store"]

