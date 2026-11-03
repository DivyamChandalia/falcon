"""ClusterIP endpoints for exact Falcon Job runs; no application health checks."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .kubernetes import KubernetesError
from .mcp_service import IdempotencyConflict, MCPServiceError, StaleJobReference

SERVICE_LABEL = "falcon.dev/job-service"
JOB_UID_ANNOTATION = "falcon.dev/service-job-uid"
JOB_NAME_ANNOTATION = "falcon.dev/service-job-name"
KEY_LABEL = "falcon.dev/idempotency-hash"
_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")


class ServiceManager:
    def __init__(self, jobs):
        self.jobs = jobs
        self.client = jobs.client
        self.namespace = jobs.namespace

    @staticmethod
    def _name(name):
        if not isinstance(name, str) or len(name) > 63 or _NAME.fullmatch(name) is None:
            raise MCPServiceError("Service name must be a lowercase DNS label of at most 63 characters")
        return name

    def _get(self, name):
        try:
            return self.client.get_json("service", name, namespace=self.namespace)
        except KubernetesError as exc:
            if exc.not_found:
                return None
            raise

    def _managed(self, value):
        metadata = value.get("metadata") or {}
        spec = value.get("spec") or {}
        if metadata.get("namespace", self.namespace) != self.namespace:
            raise StaleJobReference("Service namespace changed")
        if not metadata.get("uid") or (metadata.get("labels") or {}).get(SERVICE_LABEL) != "true":
            raise MCPServiceError("Service is not managed by Falcon's MCP Service tools")
        if (spec.get("type", "ClusterIP") != "ClusterIP" or spec.get("externalIPs")
                or spec.get("externalName") or spec.get("clusterIP") == "None"
                or any(port.get("nodePort") for port in spec.get("ports", []))):
            raise MCPServiceError("Falcon supports only internal ClusterIP Services")
        job_uid = (metadata.get("annotations") or {}).get(JOB_UID_ANNOTATION)
        job_name = (metadata.get("annotations") or {}).get(JOB_NAME_ANNOTATION)
        if not job_uid or not job_name or spec.get("selector") != {"batch.kubernetes.io/controller-uid": job_uid}:
            raise MCPServiceError("Service selector does not match its exact Falcon Job UID")
        owners = metadata.get("ownerReferences") or []
        if owners and not any(owner.get("kind") == "Job" and owner.get("uid") == job_uid
                              and owner.get("name") == job_name for owner in owners):
            raise MCPServiceError("Service has a different Job owner")
        return value

    def resolve(self, service):
        if not isinstance(service, Mapping) or set(service) - {"name", "uid"}:
            raise MCPServiceError("Service references accept only name and optional uid")
        name = self._name(service.get("name"))
        value = self._get(name)
        if value is None:
            raise KubernetesError(f"Service {name!r} not found")
        self._managed(value)
        uid = value["metadata"]["uid"]
        if service.get("uid") is not None and service["uid"] != uid:
            raise StaleJobReference("Service UID changed; refresh the Service reference")
        return value

    def describe(self, value):
        metadata, spec = value["metadata"], value["spec"]
        # Kubernetes' conventional cluster-local DNS name; short DNS names
        # also work within the namespace and avoid assuming a cluster domain.
        dns = f"{metadata['name']}.{self.namespace}.svc"
        return {
            "service": {"name": metadata["name"], "uid": metadata["uid"]},
            "type": "ClusterIP", "cluster_ip": spec.get("clusterIP"), "dns_name": dns,
            "job": {"name": metadata["annotations"][JOB_NAME_ANNOTATION], "uid": metadata["annotations"][JOB_UID_ANNOTATION]},
            "ports": [{"name": port["name"], "port": port["port"],
                       "target_port": port["targetPort"], "protocol": port.get("protocol", "TCP"),
                       "address": f"{dns}:{port['port']}"} for port in spec.get("ports", [])],
            "application_health": "not_checked",
            "health_check_hint": "Check the application's endpoint separately from inside the cluster, for example with curl.",
        }

    def create(self, job, ports, idempotency_key, name=None):
        ref, live = self.jobs._validated_live_job(job)
        self.jobs._reject_coder_job(ref, live)
        if (live.get("metadata", {}).get("labels") or {}).get("falcon.dev/managed") != "true":
            raise MCPServiceError("Only Falcon-managed Jobs can be exposed")
        name = self._name(name if name is not None else ref["name"])
        if not isinstance(ports, list) or not 1 <= len(ports) <= 16:
            raise MCPServiceError("Supply between 1 and 16 Service ports")
        canonical = []
        for port in ports:
            if not isinstance(port, Mapping) or set(port) - {"name", "port", "target_port", "protocol"}:
                raise MCPServiceError("Ports accept only name, port, target_port, and protocol")
            label = port.get("name")
            if not isinstance(label, str) or len(label) > 15 or _NAME.fullmatch(label) is None or not any(c.isalpha() for c in label):
                raise MCPServiceError("Port name must be a DNS label of at most 15 characters containing a letter")
            for field in ("port", "target_port"):
                number = port.get(field)
                if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= 65535:
                    raise MCPServiceError(f"{field} must be an integer between 1 and 65535")
            protocol = port.get("protocol", "TCP")
            if protocol not in {"TCP", "UDP"}:
                raise MCPServiceError("protocol must be TCP or UDP")
            canonical.append({"name": label, "port": port["port"], "targetPort": port["target_port"], "protocol": protocol})
        if len({p["name"] for p in canonical}) != len(canonical) or len({(p["port"], p["protocol"]) for p in canonical}) != len(canonical):
            raise MCPServiceError("Service port names and port/protocol pairs must be unique")
        canonical.sort(key=lambda p: p["name"])
        key = self.jobs._idempotency_key(idempotency_key)
        digest = self.jobs._request_digest({"job": ref, "name": name, "ports": canonical})
        operation = self.jobs._operations.reserve(
            kind="service", key=key, request_digest=digest, planned_name=name,
            namespace=self.namespace, source_uid=ref["uid"],
        )
        if operation.request_digest != digest:
            raise IdempotencyConflict("Service idempotency key was already used for a different request")
        tag = self.jobs._idempotency_tag(key)
        existing = self._get(name)
        if existing is not None:
            self._managed(existing)
            metadata = existing["metadata"]
            if (metadata.get("labels") or {}).get(KEY_LABEL) != tag or (metadata.get("annotations") or {}).get(JOB_UID_ANNOTATION) != ref["uid"]:
                raise IdempotencyConflict("Service name is already in use by a different request or Job run")
            if existing["spec"].get("ports") != canonical:
                # Kubernetes adds defaulted fields such as appProtocol only
                # when requested; compare just the fields Falcon owns.
                actual = [{key: p.get(key, "TCP" if key == "protocol" else None) for key in ("name", "port", "targetPort", "protocol")}
                          for p in existing["spec"].get("ports", [])]
                if sorted(actual, key=lambda p: p["name"]) != canonical:
                    raise IdempotencyConflict("Existing Service ports differ from the original request")
            if operation.created_uid and operation.created_uid != metadata["uid"]:
                raise StaleJobReference("Service was replaced; refusing an idempotency replay against a different UID")
            self.jobs._operations.mark_created("service", key, metadata["uid"])
            return {**self.describe(existing), "created": False, "reused": True}
        if operation.created_uid:
            return {"service": {"name": name, "uid": operation.created_uid}, "deleted": True, "created": False, "reused": True}
        manifest = {
            "apiVersion": "v1", "kind": "Service",
            "metadata": {"name": name, "namespace": self.namespace,
                         "labels": {SERVICE_LABEL: "true", "falcon.dev/managed": "true", KEY_LABEL: tag},
                         "annotations": {JOB_UID_ANNOTATION: ref["uid"], JOB_NAME_ANNOTATION: ref["name"]},
                         "ownerReferences": [{"apiVersion": "batch/v1", "kind": "Job", "name": ref["name"], "uid": ref["uid"]}]},
            "spec": {"type": "ClusterIP", "selector": {"batch.kubernetes.io/controller-uid": ref["uid"]}, "ports": canonical},
        }
        try:
            created = self.client.create_service(manifest)
        except Exception:
            # Creation may have succeeded before the response was lost.
            if self._get(name) is None:
                raise
            return self.create(ref, ports, idempotency_key, name)
        self._managed(created)
        self.jobs._operations.mark_created("service", key, created["metadata"]["uid"])
        return {**self.describe(created), "created": True, "reused": False}

    def list(self, limit=50):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise MCPServiceError("Service limit must be between 1 and 100")
        values = self.client.list_json("services", namespace=self.namespace, labels={SERVICE_LABEL: "true"}).get("items", [])
        values.sort(key=lambda item: item.get("metadata", {}).get("name", ""))
        return {"services": [self.describe(self._managed(value)) for value in values[:limit]], "truncated": len(values) > limit}

    def inspect(self, service):
        value = self.resolve(service)
        result = self.describe(value)
        slices = self.client.list_json(
            "endpointslices.discovery.k8s.io", namespace=self.namespace,
            labels={"kubernetes.io/service-name": value["metadata"]["name"]},
        )
        pods = {pod.get("metadata", {}).get("uid") for pod in self.jobs._job_pods(
            result["job"]["name"], self.namespace, uid=result["job"]["uid"])}
        backends = []
        for item in slices.get("items", []):
            if not any(owner.get("kind") == "Service" and owner.get("uid") == value["metadata"]["uid"]
                       for owner in item.get("metadata", {}).get("ownerReferences", [])):
                continue
            for endpoint in item.get("endpoints", []):
                target = endpoint.get("targetRef") or {}
                if target.get("kind") != "Pod" or target.get("uid") not in pods:
                    continue
                backends.append({"pod": target.get("name"), "pod_uid": target["uid"],
                                 "addresses": endpoint.get("addresses", []),
                                 "kubernetes_ready": (endpoint.get("conditions") or {}).get("ready")})
        result["backends"] = backends
        result["backend_count"] = len(backends)
        return result

    def delete(self, service):
        # Supplying a UID also makes repeated deletion safe after the object
        # disappears. A same-name replacement must never be deleted.
        name = self._name(service.get("name"))
        try:
            value = self.resolve(service)
        except KubernetesError as exc:
            if exc.not_found and service.get("uid"):
                return {"service": {"name": name, "uid": service["uid"]}, "deleted": True}
            raise
        ref = {"name": name, "uid": value["metadata"]["uid"]}
        try:
            self.client.delete_service_uid(name, ref["uid"], namespace=self.namespace)
        except KubernetesError as exc:
            if not exc.not_found:
                raise
        return {"service": ref, "deleted": True}
