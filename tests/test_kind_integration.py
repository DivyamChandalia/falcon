from __future__ import annotations

import asyncio
import copy
import os
import shutil
import subprocess
import tempfile
import unittest
import uuid


@unittest.skipUnless(
    os.environ.get("FALCON_KIND_INTEGRATION") == "1",
    "set FALCON_KIND_INTEGRATION=1 to run against the current disposable kind cluster",
)
class KindIntegrationTests(unittest.TestCase):
    """Opt-in lifecycle coverage; never mutates a cluster unless requested."""

    namespace = ""

    @classmethod
    def setUpClass(cls) -> None:
        if not shutil.which("kubectl") or not shutil.which("kind"):
            raise unittest.SkipTest("kubectl and kind are required")
        context = subprocess.run(
            ["kubectl", "config", "current-context"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        if not context.startswith("kind-"):
            raise unittest.SkipTest("current context is not a kind cluster")
        cls.context = context
        cls.namespace = f"falcon-test-{uuid.uuid4().hex[:8]}"
        subprocess.run(
            ["kubectl", "create", "namespace", cls.namespace],
            check=True, timeout=20,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.namespace:
            subprocess.run(
                ["kubectl", "delete", "namespace", cls.namespace, "--wait=true"],
                check=False, timeout=60,
            )

    def test_cpu_job_lifecycle_and_cleanup(self) -> None:
        manifest = (
            "apiVersion: batch/v1\nkind: Job\nmetadata:\n"
            "  name: falcon-integration\nspec:\n  template:\n    spec:\n"
            "      restartPolicy: Never\n      containers:\n"
            "      - name: main\n        image: busybox:1.36\n"
            "        command: [sh, -c, 'echo falcon']\n"
            "        resources:\n          requests: {cpu: 10m, memory: 8Mi}\n"
        )
        subprocess.run(
            ["kubectl", "apply", "-n", self.namespace, "-f", "-"],
            input=manifest, text=True, check=True, timeout=30,
        )
        subprocess.run(
            [
                "kubectl", "wait", "-n", self.namespace,
                "--for=condition=complete", "job/falcon-integration", "--timeout=120s",
            ],
            check=True, timeout=130,
        )
        value = subprocess.run(
            [
                "kubectl", "get", "job", "falcon-integration",
                "-n", self.namespace, "-o", "jsonpath={.spec.template.spec.containers[0].resources.requests.cpu}",
            ],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout
        self.assertEqual(value, "10m")

    def test_mcp_failure_handoff_and_explicit_relaunch(self) -> None:
        """Validate API-populated selectors, zero retries, and UID deletion."""
        from falcon.config import DEFAULT_CONFIG
        from falcon.kubernetes import KubernetesClient
        from falcon.mcp_service import JobService

        config = copy.deepcopy(DEFAULT_CONFIG)
        config["cluster"]["namespace"] = self.namespace
        config["job"]["backoff_limit"] = 6
        config["runtime"].update({
            "image": "busybox:1.36", "volumes": [], "mount_home": False,
            "mount_working_dir": False, "python_environment": "none",
            "home": "/tmp", "scheduler": "default-scheduler",
            "image_pull_secrets": [],
        })

        async def lifecycle(state):
            service = JobService(
                config, state_dir=state,
                client=KubernetesClient(self.namespace, context=self.context),
            )
            try:
                submitted = service.submit_job({
                    "command": ["sh", "-c", "echo mcp-failure; exit 7"],
                    "cpu": "10m", "memory": "16Mi", "environment": "none",
                    "working_dir": "/tmp",
                }, "kind-failure")
                ref = submitted["job"]
                failed = await service.wait_for_events([ref], event_types=["failed"], timeout=120)
                self.assertFalse(failed["timed_out"])
                details = service.inspect_job(ref)
                self.assertEqual(details["retry"]["backoff_limit"], 0)
                self.assertEqual(len(details["pods"]), 1)
                logs = service.job_logs(ref, failed_only=True)
                self.assertIn("mcp-failure", logs["logs"][0]["stdout"])
                self.assertEqual(logs["logs"][0]["container_exit_code"], 7)

                recovered = service.restart_job(ref, "kind-recover", overrides={
                    "command": ["sh", "-c", "echo mcp-recovered"],
                })
                new_ref = recovered["job"]
                self.assertNotEqual(ref["uid"], new_ref["uid"])
                done = await service.wait_for_events([new_ref], event_types=["succeeded"], timeout=120)
                self.assertFalse(done["timed_out"])
                service.cancel_job(new_ref)
            finally:
                await service.close()

        with tempfile.TemporaryDirectory() as state:
            asyncio.run(lifecycle(state))


if __name__ == "__main__":
    unittest.main()
