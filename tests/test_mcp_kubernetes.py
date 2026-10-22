from __future__ import annotations

import json
import subprocess

import pytest

from falcon.kubernetes import KubernetesClient


def test_context_pinned_for_reads_and_uid_precondition_delete():
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, '{"items": []}', "")

    client = KubernetesClient("mine", context="original-cluster", runner=run)
    client.list_jobs()
    client.delete_job_uid("train", "original-uid")
    assert all(argv[:3] == ["kubectl", "--context", "original-cluster"] for argv, _ in calls)
    delete, options = calls[1]
    assert delete[3:] == [
        "delete", "--raw", "/apis/batch/v1/namespaces/mine/jobs/train", "--filename", "-",
    ]
    assert json.loads(options["input"])["preconditions"] == {"uid": "original-uid"}
    assert json.loads(options["input"])["propagationPolicy"] == "Background"


def test_default_context_keeps_existing_cli_argv():
    assert KubernetesClient().command_argv(["get", "pods"]) == ["kubectl", "get", "pods"]


def test_uid_delete_can_preserve_old_pods_for_restart():
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    client = KubernetesClient("mine", runner=run)
    client.delete_job_uid("train", "old-uid", propagation_policy="Orphan")
    options = json.loads(calls[0][1]["input"])
    assert options["preconditions"] == {"uid": "old-uid"}
    assert options["propagationPolicy"] == "Orphan"
    with pytest.raises(ValueError, match="propagation"):
        client.delete_job_uid("train", "old-uid", propagation_policy="invalid")
    assert len(calls) == 1


def test_pod_logs_can_resume_from_timestamp_without_changing_default_reads():
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    client = KubernetesClient("mine", context="cluster-a", runner=run)
    client.pod_logs("attempt", tail=50, timestamps=True, since_time="2026-10-19T00:00:00Z")
    assert calls[0][-3:] == ["--timestamps=true", "--since-time", "2026-10-19T00:00:00Z"]
    client.pod_logs("attempt", tail=50)
    assert "--timestamps=true" not in calls[1]


def test_pod_deletion_is_context_and_uid_guarded():
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    client = KubernetesClient("mine", context="cluster-a", runner=run)
    client.delete_pod_uid("old-pod", "pod-uid")
    argv, kwargs = calls[0]
    assert argv[:3] == ["kubectl", "--context", "cluster-a"]
    assert "/api/v1/namespaces/mine/pods/old-pod" in argv
    assert json.loads(kwargs["input"])["preconditions"] == {"uid": "pod-uid"}
    with pytest.raises(ValueError, match="name and UID"):
        client.delete_pod_uid("old-pod", "")


def test_orphan_recovery_removes_only_orphan_and_requires_deletion():
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    client = KubernetesClient("mine", runner=run)
    job = {"metadata": {"name": "train", "uid": "old-job", "resourceVersion": "10",
                        "deletionTimestamp": "2026-10-22T00:00:00Z",
                        "finalizers": ["other-controller", "orphan"]}}
    client.finish_orphaned_job_deletion(job)
    assert json.loads(calls[0][-1]) == [
        {"op": "test", "path": "/metadata/uid", "value": "old-job"},
        {"op": "test", "path": "/metadata/resourceVersion", "value": "10"},
        {"op": "test", "path": "/metadata/finalizers", "value": ["other-controller", "orphan"]},
        {"op": "remove", "path": "/metadata/finalizers/1"},
    ]
    del job["metadata"]["deletionTimestamp"]
    client.finish_orphaned_job_deletion(job)
    assert len(calls) == 1


@pytest.mark.parametrize("name,uid", [("", "uid"), ("name", "")])
def test_uid_delete_rejects_missing_identity(name, uid):
    with pytest.raises(ValueError, match="name and UID"):
        KubernetesClient().delete_job_uid(name, uid)
