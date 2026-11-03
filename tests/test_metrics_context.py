from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from falcon.dashboard import (
    GpuSample,
    StreamingGpuSampler,
    UsageCollector,
    _gpu_metrics,
    _kubectl,
)


class MetricsContextTests(unittest.TestCase):
    def test_kubectl_context_prefixes_execution_without_changing_default_argv(self) -> None:
        result = SimpleNamespace(returncode=0, stdout="ok")
        with patch("falcon.dashboard.subprocess.run", return_value=result) as run:
            self.assertEqual(_kubectl(["get", "pods"]), "ok")
        run.assert_called_once_with(
            ["kubectl", "get", "pods"],
            capture_output=True,
            text=True,
            timeout=15,
        )

        with patch("falcon.dashboard.subprocess.run", return_value=result) as run:
            self.assertEqual(
                _kubectl(["get", "pods"], context="pinnedcontext"), "ok"
            )
        run.assert_called_once_with(
            ["kubectl", "--context", "pinnedcontext", "get", "pods"],
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_usage_collector_prefixes_context_and_omits_default_context_kwarg(self) -> None:
        with patch("falcon.dashboard._kubectl", return_value=None) as kubectl:
            UsageCollector("team-a", {}, 0.1)._kubectl(["get", "pods"])
        kubectl.assert_called_once_with(["get", "pods"])

        collector = UsageCollector("team-a", {}, 0.1, context="pinnedcontext")
        with patch("falcon.dashboard._kubectl", return_value=None) as kubectl:
            collector._kubectl(["get", "pods"], timeout=7)
        kubectl.assert_called_once_with(
            ["--context", "pinnedcontext", "get", "pods"], timeout=7
        )

    def test_gpu_metrics_prefixes_all_nonstreaming_queries(self) -> None:
        with patch(
            "falcon.dashboard._kubectl",
            side_effect=(
                "10,20,40\n",
                "gpu-uuid,4242,python,512\n",
                "# gpu pid type sm mem enc dec command\n"
                "0 4242 C 37 18 - - python\n",
            ),
        ) as kubectl:
            _gpu_metrics("team-a", "pod-a", context="pinnedcontext")

        self.assertEqual(kubectl.call_count, 3)
        for call in kubectl.call_args_list:
            self.assertEqual(call.args[0][:3], ["--context", "pinnedcontext", "exec"])

    def test_streaming_gpu_sampler_pins_both_streams(self) -> None:
        process = SimpleNamespace(stdout=None, poll=lambda: None)
        sampler = StreamingGpuSampler("team-a", context="pinnedcontext")
        with patch(
            "falcon.dashboard.subprocess.Popen", return_value=process
        ) as popen, patch("falcon.dashboard.threading.Thread"):
            sampler._start("pod-a", 1)

        commands = [call.args[0] for call in popen.call_args_list]
        self.assertEqual(len(commands), 2)
        self.assertTrue(
            all(command[:4] == ["kubectl", "--context", "pinnedcontext", "exec"]
                for command in commands)
        )

    def test_streaming_process_snapshot_passes_context_to_gpu_query(self) -> None:
        process = SimpleNamespace(poll=lambda: None)
        sampler = StreamingGpuSampler("team-a", context="pinnedcontext")
        sampler._processes["pod-a"] = process
        sampler._pmon_streams["pod-a"] = process
        with patch(
            "falcon.dashboard._gpu_processes", return_value=[]
        ) as gpu_processes:
            sampler.samples({"pod-a": 1})
        gpu_processes.assert_called_once_with(
            "team-a", "pod-a", context="pinnedcontext"
        )

    def test_usage_collector_passes_context_to_nonstreaming_gpu_pool(self) -> None:
        collector = UsageCollector(
            "team-a",
            {},
            0.1,
            collect_availability=False,
            context="pinnedcontext",
        )
        collector._items = [
            {
                "kind": "Pod",
                "metadata": {"name": "pod-a", "labels": {"job-name": "job-a"}},
                "spec": {
                    "containers": [
                        {"resources": {"limits": {"nvidia.com/gpu": 1}}}
                    ]
                },
                "status": {"phase": "Running"},
            }
        ]
        collector._items_at = 1e100
        collector._live = {"pod-a": (0.0, 0.0)}
        collector._live_at = 1e100
        with patch(
            "falcon.dashboard._gpu_metrics",
            return_value=GpuSample(utilization=10.0, gpu_count=1),
        ) as gpu_metrics:
            collector.collect()
        gpu_metrics.assert_called_once_with(
            "team-a", "pod-a", context="pinnedcontext"
        )


if __name__ == "__main__":
    unittest.main()
