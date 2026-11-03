from __future__ import annotations

import asyncio
import copy
import json
import subprocess

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from test_mcp_service import FakeClient, _pod, _request, _service

from falcon.config import DEFAULT_CONFIG
from falcon.kubernetes import KubernetesClient, KubernetesError
from falcon.mcp_server import create_server
from falcon.mcp_service import IdempotencyConflict, MCPServiceError, StaleJobReference

PORTS = [{"name": "http", "port": 8000, "target_port": 8000}]


class Client(FakeClient):
    def __init__(self):
        super().__init__()
        self.services = {}
        self.slices = []
        self.service_creates = []

    def get_json(self, resource, name, *, namespace=None):
        if resource == "service":
            if name not in self.services:
                raise KubernetesError("NotFound")
            return copy.deepcopy(self.services[name])
        return super().get_json(resource, name, namespace=namespace)

    def create_service(self, manifest):
        self.service_creates.append(copy.deepcopy(manifest))
        value = copy.deepcopy(manifest)
        value["metadata"]["uid"] = f"service-{len(self.service_creates)}"
        value["spec"]["clusterIP"] = "10.43.1.2"
        self.services[value["metadata"]["name"]] = value
        return copy.deepcopy(value)

    def list_json(self, resource, *, namespace=None, labels=None):
        assert namespace == "research"
        if resource == "services":
            values = list(self.services.values())
            return {"items": [copy.deepcopy(value) for value in values if all(
                value["metadata"].get("labels", {}).get(k) == v for k, v in (labels or {}).items())]}
        assert resource == "endpointslices.discovery.k8s.io"
        return {"items": copy.deepcopy(self.slices)}

    def delete_service_uid(self, name, uid, *, namespace=None):
        if name not in self.services:
            raise KubernetesError("NotFound")
        if self.services[name]["metadata"]["uid"] != uid:
            raise KubernetesError("Conflict")
        del self.services[name]


@pytest.fixture
def fixture(tmp_path):
    client = Client()
    service = _service(tmp_path, client)
    job = service.submit_job(_request(), "job-key")["job"]
    yield service, client, job
    asyncio.run(service.close())


def test_create_is_clusterip_uid_bound_and_does_not_change_job(fixture):
    service, client, job = fixture
    original = copy.deepcopy(client.jobs)
    result = service.create_service(job, PORTS, "endpoint")
    assert result["type"] == "ClusterIP"
    assert result["application_health"] == "not_checked"
    assert result["dns_name"] == f"{job['name']}.research.svc"
    assert result["ports"][0]["address"].endswith(":8000")
    manifest = client.service_creates[0]
    assert manifest["spec"]["selector"] == {"batch.kubernetes.io/controller-uid": job["uid"]}
    assert manifest["metadata"]["ownerReferences"][0]["uid"] == job["uid"]
    assert client.jobs == original
    assert service.create_service(job, PORTS, "endpoint")["reused"]
    assert len(client.service_creates) == 1
    assert len(service.list_services()["services"]) == 1


def test_lost_create_response_reconciles_without_duplicate(fixture, monkeypatch):
    service, client, job = fixture
    create = client.create_service

    def accepted(manifest):
        create(manifest)
        raise KubernetesError("TLS handshake timeout")

    monkeypatch.setattr(client, "create_service", accepted)
    result = service.create_service(job, PORTS, "endpoint")
    assert result["reused"]
    assert len(client.service_creates) == 1


def test_conflicting_keys_ports_and_replacements_are_rejected(fixture):
    service, client, job = fixture
    result = service.create_service(job, PORTS, "endpoint")
    with pytest.raises(IdempotencyConflict):
        service.create_service(job, PORTS, "different-key")
    with pytest.raises(IdempotencyConflict):
        service.create_service(job, [{**PORTS[0], "port": 8001}], "endpoint")
    client.services[job["name"]]["metadata"]["uid"] = "replacement"
    with pytest.raises(StaleJobReference):
        service.delete_service(result["service"])
    with pytest.raises(StaleJobReference):
        service.create_service(job, PORTS, "endpoint")


def test_delete_does_not_delete_job_or_resurrect_endpoint(fixture):
    service, client, job = fixture
    result = service.create_service(job, PORTS, "endpoint")
    assert service.delete_service(result["service"])["deleted"]
    assert service.delete_service(result["service"])["deleted"]
    assert job["name"] in client.jobs
    assert service.create_service(job, PORTS, "endpoint")["deleted"]
    assert len(client.service_creates) == 1


def test_endpoint_orphaned_during_restart_remains_inspectable_and_deletable(fixture):
    service, client, job = fixture
    result = service.create_service(job, PORTS, "endpoint")
    # restart_job may orphan dependents before replacing the source Job.
    client.services[job["name"]]["metadata"].pop("ownerReferences")
    client.jobs[job["name"]]["metadata"]["uid"] = "replacement-job"
    assert service.inspect_service(result["service"])["job"]["uid"] == job["uid"]
    assert service.delete_service(result["service"])["deleted"]
    assert client.jobs[job["name"]]["metadata"]["uid"] == "replacement-job"


def test_services_cannot_be_bound_to_stale_or_non_falcon_jobs(fixture):
    service, client, job = fixture
    with pytest.raises(StaleJobReference):
        service.create_service({**job, "uid": "different-run"}, PORTS, "endpoint")
    client.jobs[job["name"]]["metadata"]["labels"].pop("falcon.dev/managed")
    with pytest.raises(MCPServiceError, match="Only Falcon-managed"):
        service.create_service(job, PORTS, "endpoint")
    assert client.service_creates == []


def test_manually_modified_public_service_is_rejected(fixture):
    service, client, job = fixture
    result = service.create_service(job, PORTS, "endpoint")
    client.services[job["name"]]["spec"]["type"] = "LoadBalancer"
    with pytest.raises(MCPServiceError, match="only internal ClusterIP"):
        service.inspect_service(result["service"])


def test_endpoint_inspection_filters_foreign_pods_and_service_uids(fixture):
    service, client, job = fixture
    result = service.create_service(job, PORTS, "endpoint")
    client.pods = [_pod(client.jobs[job["name"]], "worker", "pod-uid")]
    base = {"metadata": {"ownerReferences": [{"kind": "Service", "uid": result["service"]["uid"]}]},
            "endpoints": [{"targetRef": {"kind": "Pod", "uid": "pod-uid", "name": "worker"},
                           "addresses": ["10.42.1.1"], "conditions": {"ready": True}},
                          {"targetRef": {"kind": "Pod", "uid": "foreign"}, "addresses": ["10.42.1.2"]}]}
    wrong = copy.deepcopy(base)
    wrong["metadata"]["ownerReferences"][0]["uid"] = "unrelated-service"
    client.slices = [base, wrong]
    inspected = service.inspect_service(result["service"])
    assert inspected["backend_count"] == 1
    assert inspected["backends"][0]["kubernetes_ready"] is True
    assert inspected["application_health"] == "not_checked"
    client.slices = []
    assert service.inspect_service(result["service"])["backend_count"] == 0


@pytest.mark.parametrize("port", [
    {"name": "http", "port": 0, "target_port": 80},
    {"name": "http", "port": True, "target_port": 80},
    {"name": "123", "port": 80, "target_port": 80},
    {"name": "http", "port": 80, "target_port": 80, "nodePort": 30080},
])
def test_invalid_or_public_ports_never_create(fixture, port):
    service, client, job = fixture
    with pytest.raises(MCPServiceError):
        service.create_service(job, [port], "endpoint")
    assert client.service_creates == []


@pytest.mark.asyncio
async def test_protocol_rejects_public_exposure_and_raw_selectors(tmp_path):
    client = Client()
    service = _service(tmp_path, client)
    job = service.submit_job(_request(), "job")["job"]
    server = create_server(DEFAULT_CONFIG, service_factory=lambda *a, **k: service)
    async with create_connected_server_and_client_session(server) as session:
        for field, value in [("type", "NodePort"), ("type", "LoadBalancer"), ("externalIPs", ["1.2.3.4"]),
                             ("externalName", "example.com"), ("selector", {}), ("namespace", "other")]:
            result = await session.call_tool("create_service", {
                "job": {"name": job["name"]}, "ports": PORTS, "idempotency_key": "endpoint", field: value,
            })
            assert result.isError
        assert client.service_creates == []
        result = await session.call_tool("create_service", {
            "job": {"name": job["name"]}, "ports": PORTS, "idempotency_key": "endpoint",
        })
        assert not result.isError
        assert "namespace" not in result.structuredContent
        ref = result.structuredContent["service"]
        inspected = await session.call_tool("inspect_service", {"service": ref})
        assert not inspected.isError
        assert inspected.structuredContent["application_health"] == "not_checked"
        listed = await session.call_tool("list_services", {})
        assert len(listed.structuredContent["services"]) == 1
        deleted = await session.call_tool("delete_service", {"service": ref})
        assert deleted.structuredContent["deleted"]


def test_service_delete_has_api_uid_precondition():
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "{}", "")

    client = KubernetesClient("mine", context="cluster", runner=run)
    client.delete_service_uid("endpoint", "old-uid")
    argv, kwargs = calls[0]
    assert "/api/v1/namespaces/mine/services/endpoint" in argv
    assert json.loads(kwargs["input"])["preconditions"] == {"uid": "old-uid"}
