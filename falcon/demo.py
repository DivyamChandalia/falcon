"""Deterministic Kubernetes fixtures for demos, tests, and visual review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .cluster import ClusterSnapshot, build_cluster_snapshot

DEMO_NOW = 1_768_478_400.0
DEMO_NOW_RFC3339 = "2026-01-15T12:00:00Z"


def _metadata(
    name: str,
    namespace: Optional[str] = None,
    *,
    uid: Optional[str] = None,
    labels: Optional[Mapping[str, str]] = None,
    annotations: Optional[Mapping[str, str]] = None,
    created: str = "2026-01-15T08:00:00Z",
) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "name": name,
        "uid": uid or f"uid-{namespace or 'cluster'}-{name}",
        "creationTimestamp": created,
    }
    if namespace:
        value["namespace"] = namespace
    if labels:
        value["labels"] = dict(labels)
    if annotations:
        value["annotations"] = dict(annotations)
    return value


def _node(
    name: str,
    *,
    cpu: str,
    memory: str,
    gpu: int = 0,
    gpu_model: Optional[str] = None,
    gpu_memory_mib: Optional[int] = None,
    ready: Optional[bool] = True,
    unschedulable: bool = False,
    taints: Optional[Sequence[Mapping[str, str]]] = None,
) -> Dict[str, Any]:
    labels = {
        "kubernetes.io/hostname": name,
        "kubernetes.io/arch": "amd64",
        "node.kubernetes.io/instance-type": "falcon-demo",
    }
    resources: Dict[str, str] = {"cpu": cpu, "memory": memory, "pods": "110"}
    if gpu:
        resources["nvidia.com/gpu"] = str(gpu)
        labels["nvidia.com/gpu.product"] = gpu_model or "Unknown"
        if gpu_memory_mib:
            labels["nvidia.com/gpu.memory"] = str(gpu_memory_mib)
    conditions = []
    if ready is not None:
        conditions.append(
            {
                "type": "Ready",
                "status": "True" if ready else "False",
                "reason": "KubeletReady" if ready else "KubeletNotReady",
            }
        )
    return {
        "apiVersion": "v1",
        "kind": "Node",
        "metadata": _metadata(
            name,
            labels=labels,
            created="2025-11-01T00:00:00Z",
        ),
        "spec": {
            "unschedulable": unschedulable,
            "taints": list(taints or []),
        },
        "status": {
            "capacity": dict(resources),
            "allocatable": dict(resources),
            "conditions": conditions,
        },
    }


def _container(
    *,
    cpu: str,
    memory: str,
    gpu: int = 0,
    command: Optional[Sequence[str]] = None,
    args: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    requests: Dict[str, str] = {"cpu": cpu, "memory": memory}
    limits: Dict[str, str] = {"cpu": cpu, "memory": memory}
    if gpu:
        requests["nvidia.com/gpu"] = str(gpu)
        limits["nvidia.com/gpu"] = str(gpu)
    value: Dict[str, Any] = {
        "name": "main",
        "image": "ghcr.io/falcon/demo:latest",
        "resources": {"requests": requests, "limits": limits},
    }
    if command:
        value["command"] = list(command)
    if args:
        value["args"] = list(args)
    return value


def _job(
    name: str,
    namespace: str,
    *,
    cpu: str,
    memory: str,
    gpu: int = 0,
    gpu_model: Optional[str] = None,
    status: str = "Pending",
    command: Optional[Sequence[str]] = None,
    args: Optional[Sequence[str]] = None,
    owner: Optional[str] = None,
    managed: bool = True,
    demo_state: Optional[str] = None,
    backoff_limit: int = 6,
    created: str = "2026-01-15T08:00:00Z",
) -> Dict[str, Any]:
    labels = {}
    if managed:
        labels["falcon.dev/managed"] = "true"
        labels["app.kubernetes.io/managed-by"] = "Falcon"
    if gpu_model:
        labels["falcon.dev/gpu-type"] = gpu_model
    if owner:
        labels["falcon.dev/user"] = owner
    if demo_state:
        labels["falcon.dev/demo-state"] = demo_state
    template_labels = {
        "job-name": name,
        "batch.kubernetes.io/job-name": name,
        **labels,
    }
    pod_spec: Dict[str, Any] = {
        "restartPolicy": "Never",
        "containers": [
            _container(
                cpu=cpu,
                memory=memory,
                gpu=gpu,
                command=command,
                args=args,
            )
        ],
    }
    if gpu_model:
        pod_spec["nodeSelector"] = {"gpu-type": gpu_model}
    job_status: Dict[str, Any] = {"startTime": "2026-01-15T08:00:10Z"}
    if status == "Running":
        job_status["active"] = 1
    elif status == "Succeeded":
        job_status.update(
            {
                "succeeded": 1,
                "completionTime": "2026-01-15T09:00:00Z",
                "conditions": [
                    {
                        "type": "Complete",
                        "status": "True",
                        "lastTransitionTime": "2026-01-15T09:00:00Z",
                    }
                ],
            }
        )
    elif status == "Failed":
        job_status.update(
            {
                "failed": 1,
                "conditions": [
                    {
                        "type": "Failed",
                        "status": "True",
                        "lastTransitionTime": "2026-01-15T09:00:00Z",
                    }
                ],
            }
        )
    elif status == "Suspended":
        job_status = {}
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": _metadata(
            name,
            namespace,
            labels=labels,
            created=created,
        ),
        "spec": {
            "parallelism": 1,
            "completions": 1,
            "backoffLimit": backoff_limit,
            "suspend": status == "Suspended",
            "template": {
                "metadata": {"labels": template_labels},
                "spec": pod_spec,
            },
        },
        "status": job_status,
    }


def _pod(
    job: Mapping[str, Any],
    suffix: str,
    *,
    node: Optional[str],
    phase: str,
    restart_count: int = 0,
    created: str = "2026-01-15T08:00:05Z",
    owner_override: Optional[str] = None,
) -> Dict[str, Any]:
    job_meta = job["metadata"]
    template = job["spec"]["template"]
    name = f"{job_meta['name']}-{suffix}"
    metadata = _metadata(
        name,
        job_meta["namespace"],
        labels=template["metadata"]["labels"],
        created=created,
    )
    metadata["ownerReferences"] = [
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "name": job_meta["name"],
            "uid": job_meta["uid"],
            "controller": True,
        }
    ]
    if owner_override:
        metadata.setdefault("annotations", {})["falcon.dev/user"] = owner_override
    spec = dict(template["spec"])
    if node:
        spec["nodeName"] = node
    container_state: Dict[str, Any]
    if phase == "Running":
        container_state = {"running": {"startedAt": "2026-01-15T08:00:10Z"}}
    elif phase == "Succeeded":
        container_state = {
            "terminated": {
                "reason": "Completed",
                "exitCode": 0,
                "finishedAt": "2026-01-15T09:00:00Z",
            }
        }
    elif phase == "Failed":
        container_state = {
            "terminated": {
                "reason": "Error",
                "exitCode": 1,
                "finishedAt": "2026-01-15T08:30:00Z",
            }
        }
    else:
        container_state = {"waiting": {"reason": "ContainerCreating"}}
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": metadata,
        "spec": spec,
        "status": {
            "phase": phase,
            "containerStatuses": [
                {
                    "name": "main",
                    "restartCount": restart_count,
                    "state": container_state,
                }
            ],
        },
    }


def _ordinary_pod(
    name: str,
    namespace: str,
    *,
    node: str,
    cpu: str,
    memory: str,
    owner_kind: str,
    owner_name: str,
    gpu: int = 0,
) -> Dict[str, Any]:
    metadata = _metadata(
        name,
        namespace,
        labels={"app.kubernetes.io/name": owner_name},
        created="2026-01-15T07:00:00Z",
    )
    metadata["ownerReferences"] = [
        {
            "apiVersion": "apps/v1",
            "kind": owner_kind,
            "name": owner_name,
            "uid": f"uid-{namespace}-{owner_name}",
            "controller": True,
        }
    ]
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": metadata,
        "spec": {
            "nodeName": node,
            "containers": [_container(cpu=cpu, memory=memory, gpu=gpu)],
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {"name": "main", "restartCount": 0, "state": {"running": {}}}
            ],
        },
    }


def demo_nodes() -> List[Dict[str, Any]]:
    return [
        _node(
            name="node-a-h100",
            cpu="64",
            memory="480Gi",
            gpu=4,
            gpu_model="NVIDIA-H100-80GB-HBM3",
            gpu_memory_mib=81559,
        ),
        _node(
            name="node-b-a6000",
            cpu="96",
            memory="720Gi",
            gpu=4,
            gpu_model="NVIDIA RTX A6000",
            gpu_memory_mib=49140,
        ),
        _node(
            name="node-c-2080ti-cordoned",
            cpu="48",
            memory="240Gi",
            gpu=8,
            gpu_model="NVIDIA_GeForce_RTX_2080_Ti",
            gpu_memory_mib=11264,
            unschedulable=True,
            taints=[
                {"key": "maintenance", "value": "planned", "effect": "NoSchedule"}
            ],
        ),
        _node(name="node-d-cpu-not-ready", cpu="32", memory="120Gi", ready=False),
    ]


def _mixed_workloads() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    jobs: List[Dict[str, Any]] = []
    pods: List[Dict[str, Any]] = []

    running_gpu = _job(
        "train-h100-two",
        "team-a",
        cpu="16",
        memory="64Gi",
        gpu=2,
        gpu_model="h100",
        status="Running",
        command=["python", "-u", "train.py"],
        args=["--model", "large", "--epochs", "100"],
        owner="alice@example.com",
    )
    jobs.append(running_gpu)
    pods.append(_pod(running_gpu, "abc12", node="node-a-h100", phase="Running"))

    shared_gpu = _job(
        "namespace-b-inference",
        "team-b",
        cpu="8",
        memory="32Gi",
        gpu=1,
        gpu_model="H100",
        status="Running",
        command=["python", "serve.py"],
        managed=False,
    )
    jobs.append(shared_gpu)
    pods.append(_pod(shared_gpu, "def34", node="node-a-h100", phase="Running"))

    cpu_job = _job(
        "preprocess-cpu",
        "data",
        cpu="12",
        memory="48Gi",
        status="Running",
        command=["python", "preprocess.py"],
        owner="data-owner",
    )
    jobs.append(cpu_job)
    pods.append(_pod(cpu_job, "cpu01", node="node-a-h100", phase="Running"))

    pending = _job(
        "queued-h100-experiment",
        "team-a",
        cpu="24",
        memory="160Gi",
        gpu=2,
        gpu_model="H100",
        status="Pending",
        command=["python", "train.py"],
    )
    jobs.append(pending)
    pods.append(_pod(pending, "pending", node=None, phase="Pending"))

    succeeded_gpu = _job(
        "finished-h100-training",
        "team-a",
        cpu="20",
        memory="96Gi",
        gpu=2,
        gpu_model="H100",
        status="Succeeded",
        command=["python", "train.py"],
        owner="alice@example.com",
    )
    jobs.append(succeeded_gpu)
    pods.append(
        _pod(
            succeeded_gpu,
            "done1",
            node="node-a-h100",
            phase="Succeeded",
            created="2026-01-15T06:00:00Z",
        )
    )

    succeeded_cpu = _job(
        "finished-cpu-report",
        "data",
        cpu="4",
        memory="8Gi",
        status="Succeeded",
        command=["python", "report.py"],
    )
    jobs.append(succeeded_cpu)
    pods.append(
        _pod(
            succeeded_cpu,
            "done2",
            node="node-d-cpu-not-ready",
            phase="Succeeded",
        )
    )

    failed = _job(
        "failed-checkpoint-conversion",
        "team-b",
        cpu="6",
        memory="24Gi",
        status="Failed",
        command=["python", "convert.py"],
    )
    jobs.append(failed)
    pods.append(
        _pod(
            failed,
            "fail1",
            node="node-b-a6000",
            phase="Failed",
            restart_count=1,
        )
    )

    retried = _job(
        "retry-eventually-succeeded",
        "team-a",
        cpu="10",
        memory="40Gi",
        gpu=1,
        gpu_model="A6000",
        status="Succeeded",
        command=["python", "unstable_train.py"],
        owner="bob@example.com",
    )
    jobs.append(retried)
    pods.extend(
        [
            _pod(
                retried,
                "attempt-a",
                node="node-b-a6000",
                phase="Failed",
                restart_count=2,
                created="2026-01-15T07:00:00Z",
            ),
            _pod(
                retried,
                "attempt-b",
                node="node-b-a6000",
                phase="Succeeded",
                restart_count=1,
                created="2026-01-15T07:30:00Z",
            ),
        ]
    )

    long_job = _job(
        "training-with-an-extraordinarily-long-name-that-must-never-break-the-terminal-layout",
        "research-with-a-long-namespace",
        cpu="18",
        memory="72Gi",
        gpu=1,
        gpu_model="A6000",
        status="Running",
        command=["python", "-u", "experiments/very_long_training_entrypoint.py"],
        args=[
            "--checkpoint-directory",
            "/media/beegfs/teams/research/checkpoints/a/path/that/is/intentionally/very/long",
            "--experiment-description",
            "deterministic visual regression content",
        ],
        demo_state="long-content",
    )
    jobs.append(long_job)
    pods.append(_pod(long_job, "long1", node="node-b-a6000", phase="Running"))

    eviction_risk = _job(
        "idle-a6000-eviction-risk",
        "team-c",
        cpu="8",
        memory="32Gi",
        gpu=1,
        gpu_model="A6000",
        status="Running",
        command=["python", "waiting_for_data.py"],
        demo_state="eviction-risk",
        owner="carol@example.com",
    )
    jobs.append(eviction_risk)
    pods.append(_pod(eviction_risk, "risk1", node="node-b-a6000", phase="Running"))

    jobs.append(
        _job(
            "suspended-debug-run",
            "team-c",
            cpu="2",
            memory="4Gi",
            status="Suspended",
            command=["sleep", "infinity"],
        )
    )
    pods.extend(
        [
            _ordinary_pod(
                "prometheus-node-exporter-very-long-generated-pod-name-7d9c8",
                "monitoring",
                node="node-a-h100",
                cpu="500m",
                memory="1Gi",
                owner_kind="DaemonSet",
                owner_name="prometheus-node-exporter",
            ),
            _ordinary_pod(
                "shared-database-0",
                "platform",
                node="node-b-a6000",
                cpu="6",
                memory="24Gi",
                owner_kind="StatefulSet",
                owner_name="shared-database",
            ),
            _ordinary_pod(
                "cordoned-node-agent-2fh8d",
                "kube-system",
                node="node-c-2080ti-cordoned",
                cpu="250m",
                memory="256Mi",
                owner_kind="DaemonSet",
                owner_name="node-agent",
            ),
        ]
    )
    return jobs, pods


def demo_inventory(state: str = "mixed") -> List[Dict[str, Any]]:
    """Return deterministic mixed Kubernetes objects.

    Supported states are ``no-jobs``, ``no-gpus``, ``one-job``, ``mixed``,
    ``many``, and ``stale``. ``stale`` uses the same last-known-good objects
    as ``mixed``; the collector/snapshot carries the stale marker separately.
    """

    normalized = state.lower().replace("_", "-")
    if normalized not in {
        "no-jobs",
        "no-gpus",
        "one-job",
        "mixed",
        "many",
        "stale",
    }:
        raise ValueError(f"unknown demo state: {state}")
    if normalized == "no-gpus":
        return [
            _node("node-cpu-1", cpu="64", memory="256Gi"),
            _node("node-cpu-2", cpu="64", memory="256Gi"),
        ]
    nodes = demo_nodes()
    jobs, pods = _mixed_workloads()
    if normalized == "no-jobs":
        return nodes
    if normalized == "one-job":
        return nodes + [jobs[0], pods[0]]
    if normalized == "many":
        for index in range(1, 31):
            status = "Succeeded" if index % 4 else "Failed"
            job = _job(
                f"historical-batch-{index:02d}",
                "batch",
                cpu="2",
                memory="4Gi",
                status=status,
                command=["python", "batch.py"],
                args=["--shard", str(index)],
                created=f"2026-01-{max(1, 15 - index // 3):02d}T06:00:00Z",
            )
            jobs.append(job)
            pods.append(
                _pod(
                    job,
                    f"history-{index:02d}",
                    node="node-d-cpu-not-ready",
                    phase=status,
                )
            )
    return nodes + jobs + pods


def demo_events(job_name: str = "train-h100-two", count: int = 100) -> List[Dict[str, Any]]:
    """Create bounded, ordered Events including long and warning messages."""

    reasons = ("Scheduled", "Pulled", "Created", "Started", "BackOff", "Unhealthy")
    events: List[Dict[str, Any]] = []
    for index in range(max(0, count)):
        reason = reasons[index % len(reasons)]
        warning = reason in {"BackOff", "Unhealthy"}
        message = (
            f"Deterministic event {index + 1:03d} for {job_name}; "
            "this deliberately long message verifies that narrow panes truncate "
            "cleanly while expanded event views remain readable."
        )
        events.append(
            {
                "apiVersion": "v1",
                "kind": "Event",
                "metadata": _metadata(
                    f"{job_name}.{index:04d}",
                    "team-a",
                    created=f"2026-01-15T10:{index // 60:02d}:{index % 60:02d}Z",
                ),
                "type": "Warning" if warning else "Normal",
                "reason": reason,
                "message": message,
                "count": 1 + (index % 3 if warning else 0),
                "involvedObject": {
                    "kind": "Job",
                    "namespace": "team-a",
                    "name": job_name,
                },
                "lastTimestamp": f"2026-01-15T10:{index // 60:02d}:{index % 60:02d}Z",
            }
        )
    return events


@dataclass(frozen=True)
class DemoFixture:
    inventory: Tuple[Mapping[str, Any], ...]
    events: Tuple[Mapping[str, Any], ...]
    metrics_available: bool
    stale: bool


def demo_fixture(state: str = "mixed") -> DemoFixture:
    normalized = state.lower().replace("_", "-")
    return DemoFixture(
        inventory=tuple(demo_inventory(normalized)),
        events=tuple(demo_events()),
        metrics_available=normalized != "stale",
        stale=normalized == "stale",
    )


def demo_cluster_snapshot(state: str = "mixed") -> ClusterSnapshot:
    fixture = demo_fixture(state)
    return build_cluster_snapshot(
        fixture.inventory,
        collected_at=DEMO_NOW,
        stale=fixture.stale,
        error="demo Kubernetes API timeout; displaying last valid inventory"
        if fixture.stale
        else None,
    )


class DemoCollector:
    """Collector-compatible deterministic source used by both TUIs."""

    namespace = "all namespaces"

    def __init__(self, state: str = "mixed") -> None:
        self.state = state
        self.calls = 0
        self.closed = False

    def set_state(self, state: str) -> None:
        # Validate early and retain a cheap string rather than mutable fixture
        # objects, so every collect call remains independent.
        demo_inventory(state)
        self.state = state

    def collect(self, force: bool = False) -> ClusterSnapshot:
        del force
        if self.closed:
            raise RuntimeError("demo collector is closed")
        self.calls += 1
        return demo_cluster_snapshot(self.state)

    def events(self, job_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        return demo_events(job_name, min(max(0, limit), 100))

    def close(self) -> None:
        self.closed = True
