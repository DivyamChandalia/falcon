from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from falcon.config import DEFAULT_CONFIG, validate_config
from falcon.manifest import build_job_manifest
from falcon.models import (
    ComputeRequest,
    EnvironmentKind,
    GPURequest,
    JobRequest,
    NodeResources,
    ResourcePlan,
    RuntimeEnvironment,
)
from falcon.output import dumps, render_table
from falcon.planning import canonical_gpu, plan_cpu_resources, plan_resources
from falcon.quantities import (
    QuantityError,
    format_cpu,
    format_memory_gib,
    normalize_memory,
    parse_cpu,
    parse_memory_bytes,
    split_pair,
)


class QuantityTests(unittest.TestCase):
    def test_cpu_units(self) -> None:
        self.assertEqual(float(parse_cpu("250m")), 0.25)

    def test_cpu_pair_reads_request(self) -> None:
        self.assertEqual(float(parse_cpu("2:4")), 2)

    def test_sub_millicore_is_rejected(self) -> None:
        with self.assertRaises(QuantityError):
            parse_cpu("0.0001")

    def test_binary_memory(self) -> None:
        self.assertEqual(parse_memory_bytes("2Gi"), 2 * 1024**3)

    def test_fractional_binary_memory_is_valid(self) -> None:
        self.assertGreater(parse_memory_bytes("0.1Gi"), 100_000_000)

    def test_invalid_quantity_is_rejected(self) -> None:
        with self.assertRaises(QuantityError):
            parse_memory_bytes("-1Gi")

    def test_pair_limit_must_cover_request(self) -> None:
        with self.assertRaises(QuantityError):
            split_pair("4:2", parse_cpu)

    def test_cpu_format_never_rounds_up(self) -> None:
        self.assertEqual(format_cpu(3.99), "3.9")

    def test_memory_format_never_rounds_up(self) -> None:
        self.assertEqual(format_memory_gib(12.39), "12687Mi")

    def test_fractional_binary_memory_normalizes_to_integral_bytes(self) -> None:
        value = normalize_memory("24.6Gi")
        self.assertEqual(value, "25191Mi")
        self.assertEqual(
            parse_memory_bytes(value),
            parse_memory_bytes(value).to_integral_value(),
        )


class PlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes = [
            NodeResources(
                "h100-a",
                cpu_total=64,
                cpu_used=8,
                memory_total_gib=480,
                memory_used_gib=40,
                gpu_total=4,
                gpu_used=1,
                gpu_product="NVIDIA H100 80GB",
            ),
            NodeResources(
                "h100-b",
                cpu_total=96,
                cpu_used=80,
                memory_total_gib=720,
                memory_used_gib=650,
                gpu_total=8,
                gpu_used=7,
                gpu_product="H100",
            ),
        ]

    def test_gpu_normalization(self) -> None:
        self.assertEqual(canonical_gpu("NVIDIA_GeForce_RTX_2080_Ti"), "2080ti")

    def test_cpu_only_plan_has_no_gpu(self) -> None:
        plan = plan_cpu_resources("8", "32Gi")
        self.assertIsNone(plan.gpu)
        self.assertEqual(plan.compute.cpu, "8")

    def test_gpu_plan_uses_best_free_node(self) -> None:
        plan = plan_resources(self.nodes, "h100", "h100", 2)
        self.assertEqual(plan.node, "h100-a")
        self.assertTrue(plan.immediately_schedulable)

    def test_gpu_plan_proportional_resources(self) -> None:
        plan = plan_resources(self.nodes, "h100", "h100", 2)
        self.assertEqual(plan.compute.cpu, "32")
        self.assertEqual(plan.compute.memory, "239Gi")

    def test_gpu_auto_memory_keeps_one_gib_safety_buffer(self) -> None:
        plan = plan_resources(self.nodes, "h100", "h100", 1, maximize=True)
        self.assertEqual(plan.compute.memory, "113Gi")

    def test_gpu_override_is_preserved(self) -> None:
        plan = plan_resources(
            self.nodes, "h100", "h100", 1,
            cpu_override="12", memory_override="64Gi",
        )
        self.assertEqual((plan.compute.cpu, plan.compute.memory), ("12", "64Gi"))

    def test_memory_can_be_overridden_without_overriding_cpu(self) -> None:
        plan = plan_resources(
            self.nodes,
            "h100",
            "h100",
            1,
            memory_override="80Gi",
        )
        self.assertEqual(plan.compute.cpu, "16")
        self.assertEqual(plan.compute.memory, "80Gi")

    def test_busy_cluster_returns_pending_warning(self) -> None:
        plan = plan_resources(
            self.nodes, "h100", "h100", 4,
            cpu_override="60", memory_override="400Gi",
        )
        self.assertFalse(plan.immediately_schedulable)
        self.assertIn("pending", plan.warning or "")

    def test_impossible_single_node_gpu_request_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            plan_resources(self.nodes, "h100", "h100", 9)

    def test_unschedulable_nodes_are_not_planned(self) -> None:
        node = NodeResources(
            "cordoned", 64, 0, 480, 0, 8, 0, "H100", True
        )
        with self.assertRaisesRegex(ValueError, "no schedulable"):
            plan_resources([node], "h100", "h100", 1)


class ManifestTests(unittest.TestCase):
    def _request(self, gpu: GPURequest | None = None) -> JobRequest:
        return JobRequest(
            name="experiment",
            namespace="research",
            command=("python", "train.py", "--epochs", "3"),
            gpu=gpu,
            working_dir="/workspace/project",
            env={"RUN_ID": "abc"},
        )

    def test_cpu_job_has_no_gpu_selector_or_resource(self) -> None:
        request = self._request()
        plan = plan_cpu_resources("4", "16Gi")
        manifest = build_job_manifest(request, plan, DEFAULT_CONFIG)
        pod = manifest["spec"]["template"]["spec"]
        resources = pod["containers"][0]["resources"]
        environment = {
            item["name"]: item["value"]
            for item in pod["containers"][0].get("env", [])
        }
        self.assertNotIn("nodeSelector", pod)
        self.assertNotIn("nvidia.com/gpu", resources["requests"])
        self.assertEqual(environment["USER"], DEFAULT_CONFIG["runtime"]["environment"]["USER"])
        self.assertEqual(environment["CONDA_AUTO_ACTIVATE_BASE"], "false")

    def test_explicit_environment_overrides_identity_defaults(self) -> None:
        request = JobRequest(
            name="identity-job",
            namespace="research",
            command=("python", "-c", "pass"),
            env={"USER": "pipeline", "CONDA_AUTO_ACTIVATE_BASE": "true"},
        )
        manifest = build_job_manifest(
            request, plan_cpu_resources("1", "1Gi"), DEFAULT_CONFIG
        )
        environment = {
            item["name"]: item["value"]
            for item in manifest["spec"]["template"]["spec"]["containers"][0]["env"]
        }
        self.assertEqual(environment["USER"], "pipeline")
        self.assertEqual(environment["CONDA_AUTO_ACTIVATE_BASE"], "true")

    def test_gpu_job_has_request_limit_and_selector(self) -> None:
        gpu = GPURequest("h100", 2)
        request = self._request(gpu)
        plan = ResourcePlan(
            "h100",
            ComputeRequest("8", "32Gi", "8", "32Gi", "4.8Gi"),
            gpu,
        )
        manifest = build_job_manifest(request, plan, DEFAULT_CONFIG)
        pod = manifest["spec"]["template"]["spec"]
        resources = pod["containers"][0]["resources"]
        self.assertEqual(resources["requests"]["nvidia.com/gpu"], "2")
        self.assertEqual(pod["nodeSelector"], {"gpu-type": "h100"})
        self.assertEqual(pod["schedulerName"], "kai-scheduler")
        self.assertIsInstance(pod["securityContext"]["runAsUser"], int)
        self.assertTrue(pod["securityContext"]["runAsNonRoot"])
        container_security = pod["containers"][0]["securityContext"]
        self.assertEqual(
            container_security["runAsUser"],
            pod["securityContext"]["runAsUser"],
        )
        self.assertTrue(container_security["runAsNonRoot"])

    def test_command_is_argv_not_shell_text(self) -> None:
        manifest = build_job_manifest(
            self._request(), plan_cpu_resources("2", "4Gi"), DEFAULT_CONFIG
        )
        self.assertEqual(
            manifest["spec"]["template"]["spec"]["containers"][0]["command"],
            ["python", "train.py", "--epochs", "3"],
        )

    def test_commandless_manifest_is_bounded_debug_session(self) -> None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["runtime"]["home"] = "/home/alice"
        config["runtime"]["mount_home"] = False
        request = JobRequest(
            name="debug",
            namespace="research",
            command=(),
        )
        manifest = build_job_manifest(
            request, plan_cpu_resources("2", "4Gi"), config
        )
        spec = manifest["spec"]
        pod = spec["template"]["spec"]
        container = pod["containers"][0]
        self.assertEqual(spec["backoffLimit"], 0)
        self.assertEqual(spec["activeDeadlineSeconds"], 6 * 60 * 60)
        self.assertEqual(
            container["command"],
            ["sleep", "infinity"],
        )
        self.assertIn(
            {
                "name": "falcon-home",
                "hostPath": {
                    "path": "/home/alice",
                    "type": "Directory",
                },
            },
            pod["volumes"],
        )
        self.assertIn(
            {
                "name": "falcon-home",
                "mountPath": "/home/alice",
            },
            container["volumeMounts"],
        )

    def test_environment_mount_and_path_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "env"
            (root / "conda-meta").mkdir(parents=True)
            (root / "bin").mkdir()
            environment = RuntimeEnvironment.from_path(root)
            request = JobRequest(
                name="env-job",
                namespace="research",
                command=("python", "-V"),
                environment=environment,
            )
            manifest = build_job_manifest(
                request, plan_cpu_resources("1", "2Gi"), DEFAULT_CONFIG
            )
            container = manifest["spec"]["template"]["spec"]["containers"][0]
            values = {item["name"]: item["value"] for item in container["env"]}
            self.assertEqual(values["CONDA_PREFIX"], str(root.resolve()))
            self.assertTrue(values["PATH"].startswith(str(root.resolve()) + "/bin:"))

    def test_stale_environment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            RuntimeEnvironment.from_path("/definitely/missing/falcon-env")

    def test_virtual_environment_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "venv"
            root.mkdir()
            (root / "bin").mkdir()
            (root / "pyvenv.cfg").write_text("", encoding="utf-8")
            value = RuntimeEnvironment.from_path(root)
            self.assertEqual(value.kind, EnvironmentKind.VENV)

    def test_environment_kind_enum_is_accepted_without_stringifying_its_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "conda"
            root.mkdir()
            (root / "bin").mkdir()
            (root / "conda-meta").mkdir()
            value = RuntimeEnvironment.from_path(
                root, EnvironmentKind.CONDA
            )
            self.assertEqual(value.kind, EnvironmentKind.CONDA)

    def test_job_name_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "DNS label"):
            JobRequest("Not_Valid", "default")

    def test_config_validation_rejects_empty_namespace(self) -> None:
        value = json.loads(json.dumps(DEFAULT_CONFIG))
        value["cluster"]["namespace"] = ""
        with self.assertRaisesRegex(ValueError, "namespace"):
            validate_config(value)


class OutputTests(unittest.TestCase):
    def test_json_is_one_ansi_free_versioned_object(self) -> None:
        value = dumps("Example", {"available": None, "count": 0})
        parsed = json.loads(value)
        self.assertEqual(parsed["schema_version"], "falcon/v1")
        self.assertNotIn("\x1b", value)
        self.assertIsNone(parsed["data"]["available"])

    def test_json_dataclasses_are_serialized(self) -> None:
        parsed = json.loads(dumps("GPU", GPURequest("h100", 2)))
        self.assertEqual(parsed["data"]["count"], 2)

    def test_plain_table_truncates_without_wrapping(self) -> None:
        value = render_table(("NAME",), (("a" * 100,),), maximum_widths=(12,))
        self.assertEqual(len(value.splitlines()), 3)
        self.assertIn("…", value)


if __name__ == "__main__":
    unittest.main()
