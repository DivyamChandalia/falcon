from __future__ import annotations

import io
import subprocess
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from falcon.commands import capture_logs
from falcon.dashboard import PodAttempt
from falcon.dashboard_logs import DashboardLogManager
from falcon.kubernetes import KubernetesClient, ProcessResult


class _ChunkStream:
    def __init__(self, chunks) -> None:
        self.chunks = iter(chunks)

    def read(self, _size=-1):
        return next(self.chunks, b"")


class _Process:
    def __init__(self, output="", returncode: int = 0) -> None:
        self.stdout = (
            _ChunkStream(output)
            if isinstance(output, (list, tuple))
            else io.BytesIO(output)
            if isinstance(output, bytes)
            else io.StringIO(output)
        )
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return self.returncode if self.terminated else None

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


class _Client:
    def __init__(
        self,
        output="attach line\n",
        terminal_output: str = "terminal line\n",
    ) -> None:
        self.output = output
        self.terminal_output = terminal_output
        self.attach_calls = []
        self.log_calls = []

    def attach_stream(self, pod_name, **kwargs):
        self.attach_calls.append((pod_name, kwargs))
        return _Process(self.output)

    def pod_logs(self, pod_name, **kwargs):
        self.log_calls.append((pod_name, kwargs))
        return ProcessResult(
            ("kubectl", "logs", pod_name),
            0,
            self.terminal_output,
        )


def _row(*attempts: PodAttempt):
    return SimpleNamespace(
        uid="job-uid",
        job="training",
        active_pod=attempts[-1].name if attempts else "",
        active_pod_uid=attempts[-1].uid if attempts else "",
        active_pod_state="Running" if attempts and attempts[-1].running else "Succeeded",
        attempt_details=tuple(attempts),
    )


class DashboardLogManagerTests(unittest.TestCase):
    def test_reconcile_attaches_every_running_attempt_but_not_terminal_ones(self):
        client = _Client()
        manager = DashboardLogManager("team", client=client)
        row = _row(
            PodAttempt("old", "old-uid", "Succeeded"),
            PodAttempt("active-a", "a-uid", "Running"),
            PodAttempt("active-b", "b-uid", "Running"),
        )
        manager.reconcile([row])
        self.assertEqual(
            [call[0] for call in client.attach_calls], ["active-a", "active-b"]
        )
        self.assertEqual(manager.snapshot(row, row.attempt_details[0]).status, "waiting")
        manager.close()

    def test_terminal_attempt_uses_captured_falcon_logs_path(self):
        client = _Client()
        manager = DashboardLogManager("team", client=client)
        row = _row(PodAttempt("completed", "completed-uid", "Succeeded"))
        manager.ensure_terminal_logs(row, row.attempt_details[0])
        deadline = time.monotonic() + 2
        while not client.log_calls and time.monotonic() < deadline:
            time.sleep(0.01)
        snapshot = manager.snapshot(row, row.attempt_details[0])
        self.assertEqual(client.log_calls[0][0], "completed")
        self.assertEqual(client.log_calls[0][1]["tail"], 200)
        self.assertEqual(snapshot.lines, ("terminal line",))
        manager.close()

    def test_lines_are_capped_and_expire_after_retention(self):
        now = [100.0]
        client = _Client("\n".join(f"line-{i}" for i in range(5)) + "\n")
        manager = DashboardLogManager(
            "team", client=client, max_lines=3, retention_seconds=10,
            clock=lambda: now[0],
        )
        row = _row(PodAttempt("active", "active-uid", "Running"))
        manager.reconcile([row])
        deadline = time.monotonic() + 2
        while not manager.snapshot(row, row.attempt_details[0]).lines and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(
            manager.snapshot(row, row.attempt_details[0]).lines,
            ("line-2", "line-3", "line-4"),
        )
        now[0] = 111.0
        self.assertEqual(manager.snapshot(row, row.attempt_details[0]).lines, ())
        manager.close()

    def test_live_tqdm_carriage_returns_replace_one_wrapped_log_row(self):
        client = _Client(
            [
                b"\r\x1b[32m  0%|          | 0/10",
                b"\x1b[0m\r 50%|#####     | 5/10",
                b"\r100%|##########| 10/10\nfinished\n",
            ]
        )
        manager = DashboardLogManager("team", client=client)
        row = _row(PodAttempt("active", "active-uid", "Running"))
        manager.reconcile([row])
        deadline = time.monotonic() + 2
        while (
            manager.snapshot(row, row.attempt_details[0]).status != "exited"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        snapshot = manager.snapshot(row, row.attempt_details[0])
        self.assertEqual(
            snapshot.lines,
            ("100%|##########| 10/10", "finished"),
        )
        self.assertNotIn("\x1b", "".join(snapshot.lines))
        manager.close()

    def test_terminal_tqdm_history_keeps_only_final_progress_state(self):
        client = _Client(
            terminal_output="0%|          | 0/10\r50%|#####     | 5/10\r"
            "100%|##########| 10/10\nfinished\n"
        )
        manager = DashboardLogManager("team", client=client)
        row = _row(PodAttempt("completed", "completed-uid", "Succeeded"))
        manager.ensure_terminal_logs(row, row.attempt_details[0])
        deadline = time.monotonic() + 2
        while not client.log_calls and time.monotonic() < deadline:
            time.sleep(0.01)
        snapshot = manager.snapshot(row, row.attempt_details[0])
        self.assertEqual(
            snapshot.lines,
            ("100%|##########| 10/10", "finished"),
        )
        manager.close()

    def test_newline_normalized_tqdm_updates_still_replace_previous_bar(self):
        client = _Client(
            "  0%|          | 0/10 [00:00<?, ?it/s]\n"
            " 50%|#####     | 5/10 [00:01<00:01, 5.00it/s]\n"
            "100%|##########| 10/10 [00:02<00:00, 5.00it/s]\n"
            "finished\n"
        )
        manager = DashboardLogManager("team", client=client)
        row = _row(PodAttempt("active", "active-uid", "Running"))
        manager.reconcile([row])
        deadline = time.monotonic() + 2
        while (
            manager.snapshot(row, row.attempt_details[0]).status != "exited"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertEqual(
            manager.snapshot(row, row.attempt_details[0]).lines,
            (
                "100%|##########| 10/10 [00:02<00:00, 5.00it/s]",
                "finished",
            ),
        )
        manager.close()

    def test_reconcile_stops_stream_when_attempt_becomes_terminal(self):
        client = _Client()
        manager = DashboardLogManager("team", client=client)
        running = _row(PodAttempt("active", "active-uid", "Running"))
        manager.reconcile([running])
        process = _Process("")
        manager._states[("job-uid", "active-uid")].process = process
        terminal = _row(PodAttempt("active", "active-uid", "Succeeded"))
        manager.reconcile([terminal])
        self.assertTrue(process.terminated)
        manager.close()


class KubernetesAttachStreamTests(unittest.TestCase):
    def test_pod_logs_uses_the_bounded_read_only_command(self):
        completed = ProcessResult(("kubectl", "logs"), 0, "line\n")

        class Runner:
            def __call__(self, argv, **kwargs):
                self.argv = argv
                return type(
                    "Completed",
                    (),
                    {
                        "returncode": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    },
                )()

        runner = Runner()
        client = KubernetesClient("team", runner=runner)
        result = capture_logs(
            "team",
            "job",
            pod_name="pod-a",
            tail=200,
            follow=False,
            container="main",
            client=client,
        )
        self.assertEqual(result.stdout, "line\n")
        self.assertEqual(
            runner.argv,
            [
                "kubectl", "logs", "pod-a", "--namespace", "team",
                "--tail", "200", "--container", "main",
            ],
        )

    def test_attach_stream_disables_stdin_and_tty(self):
        with patch("falcon.kubernetes.subprocess.Popen", return_value=object()) as popen:
            client = KubernetesClient("team", executable="kubectl")
            result = client.attach_stream("pod-a", container="main")
        self.assertIsNotNone(result)
        argv = popen.call_args.args[0]
        self.assertEqual(
            argv,
            [
                "kubectl", "attach", "pod-a", "--namespace", "team",
                "--stdin=false", "--tty=false", "--container", "main",
            ],
        )
        self.assertIs(popen.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertFalse(popen.call_args.kwargs["text"])
        self.assertEqual(popen.call_args.kwargs["bufsize"], 0)


if __name__ == "__main__":
    unittest.main()
