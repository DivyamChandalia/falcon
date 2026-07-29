from __future__ import annotations

import os
import shutil
import subprocess
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


if __name__ == "__main__":
    unittest.main()
