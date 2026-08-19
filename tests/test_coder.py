from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from unittest.mock import MagicMock, patch

import httpx

from falcon.cli import EXIT_CODER, main
from falcon.coder import (
    CoderClient,
    CoderError,
    build_access_links,
    build_parameter_values,
    generate_workspace_name,
    resolve_connection,
    save_connection,
    select_access_links,
    workspace_job_name,
)
from falcon.config import DEFAULT_CONFIG, validate_config
from falcon.models import NodeResources
from falcon.planning import plan_cpu_resources, plan_resources


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def ready_workspace(name: str = "lime-gull-30") -> dict:
    return {
        "name": name,
        "owner_name": "divyam.c",
        "latest_build": {
            "status": "running",
            "resources": [
                {
                    "name": f"coder-divyam.c-{name}",
                    "type": "batch/v1::Job",
                    "agents": [
                        {
                            "name": "main",
                            "status": "connected",
                            "display_apps": ["vscode", "web_terminal"],
                            "apps": [
                                {
                                    "display_name": "JupyterLab",
                                    "slug": "jupyterlab",
                                    "external": False,
                                    "url": "http://localhost:8888",
                                    "subdomain": True,
                                    "subdomain_name": (
                                        f"jupyterlab--{name}--divyamc"
                                    ),
                                    "health": "healthy",
                                },
                                {
                                    "display_name": "Antigravity",
                                    "slug": "antigravity",
                                    "external": True,
                                    "url": (
                                        "antigravity://coder/open?"
                                        "token=$SESSION_TOKEN"
                                    ),
                                },
                                {
                                    "display_name": "Antigravity 2.0",
                                    "slug": "antigravity-2-0",
                                    "external": True,
                                    "url": (
                                        "antigravity://coder/v2/open?"
                                        "token=$SESSION_TOKEN"
                                    ),
                                },
                                {
                                    "display_name": "Cursor",
                                    "slug": "cursor",
                                    "external": True,
                                    "url": "cursor://coder/open?token=$SESSION_TOKEN",
                                },
                            ],
                        }
                    ],
                }
            ],
        },
    }


class CoderNameTests(unittest.TestCase):
    def test_name_matches_coder_color_animal_number_shape(self) -> None:
        with patch(
            "falcon.coder.get_random_name", return_value="lime-gull"
        ), patch("falcon.coder.secrets.randbelow", return_value=30):
            self.assertEqual(generate_workspace_name(), "lime-gull-30")

    def test_connection_reuses_coder_cli_files_without_storing_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "url").write_text("https://coder.example.test\n")
            (root / "session").write_text("secret-session\n")
            with patch.dict(
                os.environ,
                {"CODER_CONFIG_DIR": directory},
                clear=True,
            ):
                self.assertEqual(
                    resolve_connection({"coder": {"url": "https://fallback"}}),
                    ("https://coder.example.test", "secret-session"),
                )

    def test_save_connection_uses_coder_files_with_owner_only_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"CODER_CONFIG_DIR": f"{directory}/coder"},
            clear=True,
        ):
            session = save_connection(
                "https://coder.example.test/", "  secret-session  "
            )
            root = session.parent
            self.assertEqual(
                (root / "url").read_text(encoding="utf-8"),
                "https://coder.example.test",
            )
            self.assertEqual(
                session.read_text(encoding="utf-8"), "secret-session"
            )
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(session.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE((root / "url").stat().st_mode), 0o600)
            self.assertEqual(list(root.glob(".*.tmp")), [])
            self.assertEqual(
                resolve_connection({"coder": {"url": "https://fallback"}}),
                ("https://coder.example.test", "secret-session"),
            )


class CoderClientTests(unittest.TestCase):
    def test_client_uses_v2_workspace_api_and_session_header(self) -> None:
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            self.assertEqual(
                request.headers["Coder-Session-Token"], "session-token"
            )
            if request.url.path == "/api/v2/users/me":
                return httpx.Response(200, json={"username": "divyam.c"})
            if request.url.path == "/api/v2/templates":
                return httpx.Response(200, json=[{"id": "template-1"}])
            if request.url.path.endswith("/workspaces"):
                body = json.loads(request.content)
                self.assertEqual(body["name"], "lime-gull-30")
                self.assertEqual(body["template_id"], "template-1")
                return httpx.Response(201, json=ready_workspace())
            self.fail(f"unexpected request {request.method} {request.url}")

        client = CoderClient(
            "https://coder.example.test",
            "session-token",
            transport=httpx.MockTransport(handler),
        )
        try:
            self.assertEqual(client.current_user()["username"], "divyam.c")
            self.assertEqual(client.templates()[0]["id"], "template-1")
            created = client.create_workspace(
                "me",
                template_id="template-1",
                name="lime-gull-30",
                parameters=({"name": "cpu", "value": "4"},),
            )
            self.assertEqual(created["name"], "lime-gull-30")
        finally:
            client.close()
        self.assertEqual(len(requests), 3)

    def test_workspace_job_resolution_does_not_assume_coder_username_slug(self) -> None:
        workspace = ready_workspace("falcon")
        workspace["id"] = "workspace-id"
        workspace["owner_name"] = "divyamc"
        workspace["latest_build"]["resources"][0]["name"] = "workspace"

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/v2/workspaces")
            self.assertEqual(request.url.params["q"], "owner:me")
            return httpx.Response(
                200,
                json={"count": 1, "workspaces": [workspace]},
            )

        client = CoderClient(
            "https://coder.example.test",
            "session-token",
            transport=httpx.MockTransport(handler),
        )
        try:
            resolved = client.workspace_for_job(
                "coder-divyam.c-falcon",
                username="divyamc",
            )
        finally:
            client.close()

        self.assertEqual(resolved["id"], "workspace-id")

    def test_wait_uses_coder_workspace_running_status(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertIn("/workspace/lime-gull-30", request.url.path)
            return httpx.Response(200, json=ready_workspace())

        client = CoderClient(
            "https://coder.example.test",
            "session-token",
            transport=httpx.MockTransport(handler),
        )
        try:
            workspace = client.wait_until_ready(
                "divyam.c", "lime-gull-30", timeout=1, interval=0
            )
        finally:
            client.close()
        self.assertEqual(workspace["latest_build"]["status"], "running")

    def test_wait_fails_immediately_for_an_agent_that_is_off(self) -> None:
        workspace = ready_workspace()
        agent = workspace["latest_build"]["resources"][0]["agents"][0]
        agent["status"] = "disconnected"
        agent["lifecycle_state"] = "off"

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=workspace)

        client = CoderClient(
            "https://coder.example.test",
            "session-token",
            transport=httpx.MockTransport(handler),
        )
        try:
            with self.assertRaisesRegex(CoderError, "disconnected, lifecycle off"):
                client.wait_until_ready(
                    "divyam.c", "falcon", timeout=600, interval=0
                )
        finally:
            client.close()

    def test_delete_workspace_submits_native_coder_delete_build(self) -> None:
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            self.assertEqual(
                request.url.path,
                "/api/v2/workspaces/workspace-id/builds",
            )
            self.assertEqual(json.loads(request.content), {"transition": "delete"})
            return httpx.Response(
                201,
                json={"id": "delete-build", "transition": "delete"},
            )

        client = CoderClient(
            "https://coder.example.test",
            "session-token",
            transport=httpx.MockTransport(handler),
        )
        try:
            build = client.delete_workspace({"id": "workspace-id"})
        finally:
            client.close()

        self.assertEqual(build["transition"], "delete")
        self.assertEqual(len(requests), 1)

    def test_delete_workspace_requires_coder_workspace_id(self) -> None:
        client = CoderClient(
            "https://coder.example.test",
            "session-token",
            transport=httpx.MockTransport(
                lambda _request: self.fail("no request should be sent")
            ),
        )
        try:
            with self.assertRaisesRegex(CoderError, "missing its ID"):
                client.delete_workspace({"name": "falcon"})
        finally:
            client.close()

    def test_restart_workspace_uses_stop_then_start_and_waits_until_ready(self) -> None:
        running = ready_workspace("falcon")
        running["id"] = "workspace-id"
        stopped = copy.deepcopy(running)
        stopped["latest_build"]["status"] = "stopped"
        stopped["latest_build"]["resources"][0]["agents"][0][
            "status"
        ] = "disconnected"
        workspaces = [running, stopped, running]
        transitions = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json=workspaces.pop(0))
            transitions.append(json.loads(request.content)["transition"])
            return httpx.Response(
                201,
                json={"id": f"{transitions[-1]}-build"},
            )

        client = CoderClient(
            "https://coder.example.test",
            "session-token",
            transport=httpx.MockTransport(handler),
        )
        try:
            workspace = client.restart_workspace(
                "divyam.c",
                "falcon",
                timeout=1,
                interval=0,
            )
        finally:
            client.close()

        self.assertEqual(transitions, ["stop", "start"])
        self.assertEqual(workspace["latest_build"]["status"], "running")


class CoderParameterTests(unittest.TestCase):
    def test_ides_is_the_default_template(self) -> None:
        self.assertEqual(DEFAULT_CONFIG["coder"]["template"], "IDEs")

    def test_cpu_memory_pairs_map_to_request_and_limit_parameters(self) -> None:
        plan = plan_cpu_resources("4:4", "8Gi:8Gi")
        parameters = [
            {"name": "cpu", "type": "number"},
            {"name": "cpu_limit", "type": "number"},
            {"name": "memory", "type": "number"},
            {"name": "memory_limit", "type": "string"},
            {"name": "image", "type": "string"},
        ]
        values = build_parameter_values(
            parameters,
            plan,
            overrides={"image": "ubuntu24"},
        )
        self.assertEqual(
            {value["name"]: value["value"] for value in values},
            {
                "cpu": "4",
                "cpu_limit": "4",
                "memory": "8",
                "memory_limit": "8Gi",
                "image": "ubuntu24",
            },
        )

    def test_configured_parameter_names_are_validated(self) -> None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["coder"]["parameters"]["cpu"] = ""
        with self.assertRaisesRegex(ValueError, "coder.parameters"):
            validate_config(config)

    def test_gpu_plan_maps_model_and_count_to_template_parameters(self) -> None:
        plan = plan_resources(
            [
                NodeResources(
                    "gpu-node",
                    cpu_total=16,
                    memory_total_gib=64,
                    gpu_total=4,
                    gpu_product="2080ti",
                )
            ],
            "2080ti",
            "2080ti",
            1,
        )
        parameters = [
            {"name": "cpu", "type": "number"},
            {"name": "memory", "type": "number"},
            {
                "name": "gpu_type",
                "type": "string",
                "options": [
                    {"value": "none"},
                    {"value": "2080ti"},
                    {"value": "a6000"},
                ],
            },
            {
                "name": "gpu_count",
                "type": "string",
                "options": [{"value": "0"}, {"value": "1"}, {"value": "2"}],
            },
        ]
        values = build_parameter_values(parameters, plan)
        self.assertEqual(
            {value["name"]: value["value"] for value in values},
            {
                "cpu": "4",
                "memory": "15",
                "gpu_type": "2080ti",
                "gpu_count": "1",
            },
        )

    def test_numeric_memory_parameter_floors_fractional_gib(self) -> None:
        plan = plan_cpu_resources("24", "29313Mi")
        values = build_parameter_values(
            [
                {"name": "cpu", "type": "number", "validation_min": 1},
                {
                    "name": "memory",
                    "type": "number",
                    "validation_min": 1,
                    "validation_max": 2000,
                },
            ],
            plan,
        )
        self.assertEqual(
            {value["name"]: value["value"] for value in values},
            {"cpu": "24", "memory": "28"},
        )

    def test_numeric_resource_below_one_has_a_local_error(self) -> None:
        plan = plan_cpu_resources("500m", "1Gi")
        with self.assertRaisesRegex(CoderError, "only accepts whole numbers"):
            build_parameter_values(
                [
                    {"name": "cpu", "type": "number", "validation_min": 1},
                    {"name": "memory", "type": "number", "validation_min": 1},
                ],
                plan,
            )


class CoderLinkTests(unittest.TestCase):
    def test_links_use_reported_apps_and_one_shared_order(self) -> None:
        workspace = ready_workspace()
        folder = "/media/beegfs/users/divyam.c/jet-k8s"
        workspace_url, links = build_access_links(
            workspace,
            "https://coder.example.test",
            folder=folder,
        )
        self.assertEqual(
            workspace_url,
            "https://coder.example.test/@divyam.c/lime-gull-30",
        )
        self.assertEqual(
            [link.label for link in links],
            [
                "VS Code",
                "Antigravity",
                "Antigravity 2.0 IDE",
                "Cursor",
                "JupyterLab",
                "Terminal",
            ],
        )
        self.assertEqual(
            workspace_job_name(workspace),
            "coder-divyam.c-lime-gull-30",
        )
        self.assertIn(
            "lime-gull-30.main/terminal",
            links[-1].target,
        )
        antigravity_2 = select_access_links(links, "antigravity 2.0")
        self.assertEqual(
            [link.label for link in antigravity_2],
            ["Antigravity 2.0 IDE"],
        )
        antigravity_ide = select_access_links(links, "antigravity 2.0 ide")
        self.assertEqual(
            [link.label for link in antigravity_ide], ["Antigravity 2.0 IDE"]
        )
        self.assertTrue(antigravity_ide[0].target.startswith("antigravity-ide://"))
        self.assertEqual(
            parse_qs(urlsplit(antigravity_ide[0].target).query)["token"],
            ["$SESSION_TOKEN"],
        )
        for link in links[:-2]:
            self.assertEqual(parse_qs(urlsplit(link.target).query)["folder"], [folder])
            self.assertNotIn("openRecent", parse_qs(urlsplit(link.target).query))
        jupyter = select_access_links(links, "notebook")
        self.assertEqual([link.label for link in jupyter], ["JupyterLab"])
        self.assertEqual(
            jupyter[0].target,
            "https://jupyterlab--lime-gull-30--divyamc.coder.example.test/",
        )
        vscode = links[0].with_token("fresh_app_key")
        self.assertFalse(vscode.requires_token)
        self.assertIn("token=fresh_app_key", vscode.target)

    def test_links_without_a_folder_keep_vscode_recent_behavior(self) -> None:
        _, links = build_access_links(
            ready_workspace(), "https://coder.example.test"
        )
        vscode_query = parse_qs(urlsplit(links[0].target).query)
        self.assertEqual(vscode_query["openRecent"], ["true"])
        self.assertNotIn("folder", vscode_query)

    def test_editor_folder_must_be_absolute(self) -> None:
        with self.assertRaisesRegex(CoderError, "absolute path"):
            build_access_links(
                ready_workspace(),
                "https://coder.example.test",
                folder="relative/project",
            )

    def test_generic_antigravity_ide_app_prints_only_ide_link(self) -> None:
        workspace = ready_workspace()
        agent = workspace["latest_build"]["resources"][0]["agents"][0]
        agent["apps"] = [
            {
                "display_name": "Antigravity IDE",
                "slug": "antigravity",
                "external": True,
                "url": (
                    "antigravity://coder.coder-remote/open?"
                    "owner=divyam.c&workspace=falcon&folder=%2Ftmp&"
                    "url=https%3A%2F%2Fcoder.example.test&token=$SESSION_TOKEN"
                ),
            },
            {
                "display_name": "Cursor Desktop",
                "slug": "cursor",
                "external": True,
                "url": "cursor://coder.coder-remote/open?token=$SESSION_TOKEN",
            },
        ]
        _, links = build_access_links(
            workspace,
            "https://coder.example.test",
            folder="/media/beegfs/users/divyam.c/jet-k8s",
        )
        antigravity = [
            link for link in links if link.label.startswith("Antigravity 2.0")
        ]
        self.assertEqual(
            [link.label for link in antigravity],
            ["Antigravity 2.0 IDE"],
        )
        self.assertEqual(
            [urlsplit(link.target).scheme for link in antigravity],
            ["antigravity-ide"],
        )

    def test_job_name_falls_back_when_coder_reports_terraform_resource_name(self) -> None:
        workspace = ready_workspace("falcon")
        workspace["latest_build"]["resources"][0]["name"] = "workspace"
        workspace["latest_build"]["resources"][0]["type"] = "kubernetes_job_v1"
        self.assertEqual(
            workspace_job_name(workspace), "coder-divyam.c-falcon"
        )


class CoderCliTests(unittest.TestCase):
    def test_kill_routes_coder_job_through_workspace_delete(self) -> None:
        client = MagicMock()
        client.__enter__.return_value = client
        client.current_user.return_value = {"username": "divyamc"}
        workspace = {"id": "workspace-id", "name": "falcon"}
        client.workspaces.return_value = (workspace,)
        client.workspace_for_job.return_value = workspace

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("falcon.cli.load_config", return_value=DEFAULT_CONFIG), patch(
            "falcon.cli.CoderClient", return_value=client
        ), patch(
            "falcon.cli.resolve_connection",
            return_value=("https://coder.example.test", "session"),
        ), patch("falcon.cli.kill") as kubernetes_kill, contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(stderr):
            code = main(["kill", "coder-divyam.c-falcon"])

        self.assertEqual((code, stderr.getvalue()), (0, ""))
        client.workspace_for_job.assert_called_once_with(
            "coder-divyam.c-falcon",
            username="divyamc",
            workspaces=(workspace,),
        )
        client.delete_workspace.assert_called_once_with(workspace)
        kubernetes_kill.assert_not_called()
        self.assertIn("Deleting 1 Coder workspace", stdout.getvalue())
        self.assertIn("falcon (coder-divyam.c-falcon)", stdout.getvalue())

    def test_kill_partitions_coder_and_ordinary_jobs(self) -> None:
        client = MagicMock()
        client.__enter__.return_value = client
        client.current_user.return_value = {"username": "divyam.c"}
        workspace = {"id": "workspace-id", "name": "falcon"}
        client.workspaces.return_value = (workspace,)
        client.workspace_for_job.return_value = workspace

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("falcon.cli.load_config", return_value=DEFAULT_CONFIG), patch(
            "falcon.cli.CoderClient", return_value=client
        ), patch(
            "falcon.cli.resolve_connection",
            return_value=("https://coder.example.test", "session"),
        ), patch(
            "falcon.cli.kill", return_value=["training"]
        ) as kubernetes_kill, contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(stderr):
            code = main([
                "kill", "coder-divyam.c-falcon", "training",
            ])

        self.assertEqual((code, stderr.getvalue()), (0, ""))
        client.delete_workspace.assert_called_once()
        self.assertEqual(kubernetes_kill.call_args.args[1], ["training"])
        self.assertIn("Deleting 1 Coder workspace", stdout.getvalue())
        self.assertIn("Killed 1 Job(s): training", stdout.getvalue())

    def test_command_creates_workspace_and_prints_terminal_link(self) -> None:
        calls = {}

        class FakeCoderClient:
            def __init__(self, url, token):
                calls["connection"] = (url, token)

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def current_user(self):
                return {"username": "divyam.c"}

            def templates(self):
                return ({
                    "id": "template-1",
                    "name": "IDEs",
                    "active_version_id": "version-1",
                },)

            def rich_parameters(self, _):
                return (
                    {"name": "cpu", "type": "number"},
                    {"name": "memory", "type": "string"},
                )

            def create_workspace(self, user, **kwargs):
                calls["create"] = (user, kwargs)
                return {"name": kwargs["name"]}

            def wait_until_ready(self, user, name, **kwargs):
                calls["wait"] = (user, name, kwargs)
                return ready_workspace(name)

            def create_app_token(self):
                raise AssertionError("terminal-only access needs no editor token")

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.dict(
            os.environ,
            {"FALCON_NAMESPACE": "default"},
            clear=False,
        ), patch("falcon.cli.CoderClient", FakeCoderClient), patch(
            "falcon.cli.resolve_connection",
            return_value=("https://coder.example.test", "session"),
        ), patch(
            "falcon.cli.generate_workspace_name", return_value="lime-gull-30"
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main([
                "coder", "-c", "4:4", "-m", "8Gi:8Gi",
                "--access", "terminal",
            ])
        self.assertEqual((code, stderr.getvalue()), (0, ""))
        self.assertEqual(calls["connection"], ("https://coder.example.test", "session"))
        self.assertEqual(calls["create"][0], "me")
        parameter_values = {
            item["name"]: item["value"]
            for item in calls["create"][1]["parameters"]
        }
        self.assertEqual(parameter_values, {"cpu": "4", "memory": "8Gi"})
        self.assertIn("Workspace ready: https://coder.example.test/", stdout.getvalue())
        self.assertIn("Kubernetes Job: coder-divyam.c-lime-gull-30", stdout.getvalue())
        self.assertIn("Terminal", stdout.getvalue())

    def test_gpu_preset_and_job_alias_create_named_coder_workspace(self) -> None:
        calls = {}

        class FakeCoderClient:
            def __init__(self, _url, _token):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def current_user(self):
                return {"username": "divyam.c"}

            def templates(self):
                return ({
                    "id": "template-1",
                    "name": "IDEs",
                    "active_version_id": "version-1",
                },)

            def rich_parameters(self, _):
                return (
                    {"name": "cpu", "type": "number"},
                    {"name": "memory", "type": "number"},
                    {
                        "name": "gpu_type",
                        "type": "string",
                        "options": [{"value": "none"}, {"value": "2080ti"}],
                    },
                    {
                        "name": "gpu_count",
                        "type": "string",
                        "options": [{"value": "0"}, {"value": "1"}],
                    },
                )

            def create_workspace(self, user, **kwargs):
                calls["create"] = (user, kwargs)
                return {"name": kwargs["name"]}

            def wait_until_ready(self, _user, name, **_kwargs):
                return ready_workspace(name)

            def create_app_token(self):
                raise AssertionError("terminal-only access needs no editor token")

        node = NodeResources(
            "2080ti-node",
            cpu_total=16,
            memory_total_gib=64,
            gpu_total=4,
            gpu_product="NVIDIA GeForce RTX 2080 Ti",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("falcon.cli.load_config", return_value=DEFAULT_CONFIG), patch(
            "falcon.cli._planning_nodes", return_value=[node]
        ), patch("falcon.cli.CoderClient", FakeCoderClient), patch(
            "falcon.cli.resolve_connection",
            return_value=("https://coder.example.test", "session"),
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main([
                "coder", "2080ti", "-j", "falcon", "--access", "terminal",
            ])

        self.assertEqual((code, stderr.getvalue()), (0, ""))
        self.assertEqual(calls["create"][0], "me")
        self.assertEqual(calls["create"][1]["name"], "falcon")
        self.assertEqual(
            {
                item["name"]: item["value"]
                for item in calls["create"][1]["parameters"]
            },
            {
                "cpu": "4",
                "memory": "15",
                "gpu_type": "2080ti",
                "gpu_count": "1",
            },
        )
        self.assertIn("GPU 2080tix1", stdout.getvalue())
        self.assertIn("Kubernetes Job: coder-divyam.c-falcon", stdout.getvalue())

    def test_named_existing_workspace_reprints_links(self) -> None:
        client = MagicMock()
        client.__enter__.return_value = client
        client.current_user.return_value = {"username": "divyam.c"}
        client.templates.return_value = ({
            "id": "template-1",
            "name": "IDEs",
            "active_version_id": "version-1",
        },)
        client.rich_parameters.return_value = (
            {"name": "cpu", "type": "number"},
            {"name": "memory", "type": "number"},
        )
        client.create_workspace.side_effect = CoderError(
            "workspace already exists", status_code=409
        )
        client.workspace.return_value = ready_workspace("falcon")
        client.wait_until_ready.return_value = ready_workspace("falcon")

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("falcon.cli.load_config", return_value=DEFAULT_CONFIG), patch(
            "falcon.cli.CoderClient", return_value=client
        ), patch(
            "falcon.cli.resolve_connection",
            return_value=("https://coder.example.test", "session"),
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main([
                "coder", "-c", "4", "-m", "8Gi", "-j", "falcon",
                "--access", "terminal",
            ])

        self.assertEqual((code, stderr.getvalue()), (0, ""))
        client.workspace.assert_called_once_with("divyam.c", "falcon")
        self.assertIn("workspace falcon already exists", stdout.getvalue())
        self.assertIn("Kubernetes Job: coder-divyam.c-falcon", stdout.getvalue())

    def test_positional_workspace_or_job_reprints_links_without_creation(self) -> None:
        for reference in ("falcon", "coder-divyam.c-falcon"):
            with self.subTest(reference=reference):
                client = MagicMock()
                client.__enter__.return_value = client
                client.current_user.return_value = {"username": "divyamc"}
                client.workspace.return_value = ready_workspace("falcon")
                client.workspace_for_job.return_value = ready_workspace("falcon")
                client.wait_until_ready.return_value = ready_workspace("falcon")

                stdout = io.StringIO()
                stderr = io.StringIO()
                with patch(
                    "falcon.cli.load_config", return_value=DEFAULT_CONFIG
                ), patch(
                    "falcon.cli.CoderClient", return_value=client
                ), patch(
                    "falcon.cli.resolve_connection",
                    return_value=("https://coder.example.test", "session"),
                ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                    stderr
                ):
                    code = main(["coder", reference])

                self.assertEqual((code, stderr.getvalue()), (0, ""))
                if reference.startswith("coder-"):
                    client.workspace_for_job.assert_called_once_with(
                        reference,
                        username="divyamc",
                    )
                    client.workspace.assert_not_called()
                else:
                    client.workspace.assert_called_once_with("divyamc", "falcon")
                    client.workspace_for_job.assert_not_called()
                client.templates.assert_not_called()
                client.rich_parameters.assert_not_called()
                client.create_workspace.assert_not_called()
                self.assertIn(
                    "Connecting to existing Coder workspace falcon",
                    stdout.getvalue(),
                )
                self.assertIn("Antigravity 2.0", stdout.getvalue())
                self.assertIn("Antigravity 2.0 IDE", stdout.getvalue())
                self.assertIn(
                    "Kubernetes Job: coder-divyam.c-falcon", stdout.getvalue()
                )

    def test_missing_auth_uses_distinct_coder_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"HOME": directory, "CODER_CONFIG_DIR": directory},
            clear=True,
        ):
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(["coder", "-c", "4", "-m", "8Gi"])
        self.assertEqual(code, EXIT_CODER)
        self.assertIn("Coder authentication is required", stderr.getvalue())

    def test_missing_auth_prompts_validates_and_saves_in_a_terminal(self) -> None:
        connections = []

        class FakeCoderClient:
            def __init__(self, url, token):
                connections.append((url, token))

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def current_user(self):
                return {"username": "divyam.c"}

            def templates(self):
                return ({
                    "id": "template-1",
                    "name": "IDEs",
                    "active_version_id": "version-1",
                },)

            def rich_parameters(self, _):
                return (
                    {"name": "cpu", "type": "number"},
                    {"name": "memory", "type": "string"},
                )

            def create_workspace(self, _user, **kwargs):
                return {"name": kwargs["name"]}

            def wait_until_ready(self, _user, name, **_kwargs):
                return ready_workspace(name)

            def create_app_token(self):
                raise AssertionError("terminal-only access needs no editor token")

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "HOME": directory,
                "CODER_CONFIG_DIR": f"{directory}/coderv2",
                "FALCON_NAMESPACE": "default",
            },
            clear=True,
        ), patch("falcon.cli.CoderClient", FakeCoderClient), patch(
            "falcon.cli.getpass.getpass", return_value="pasted-session"
        ), patch(
            "falcon.cli.generate_workspace_name", return_value="lime-gull-30"
        ), patch(
            "falcon.cli.sys.stdin", TtyStringIO()
        ):
            stdout = TtyStringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main([
                    "coder", "-c", "4", "-m", "8Gi",
                    "--access", "terminal",
                ])

            root = Path(directory) / "coderv2"
            self.assertEqual((code, stderr.getvalue()), (0, ""))
            self.assertEqual(
                connections,
                [
                    ("https://coder.yoda.hyperverge.org", "pasted-session"),
                    ("https://coder.yoda.hyperverge.org", "pasted-session"),
                ],
            )
            self.assertIn("/cli-auth", stdout.getvalue())
            self.assertIn("Coder login saved for divyam.c", stdout.getvalue())
            self.assertEqual(
                (root / "session").read_text(encoding="utf-8"),
                "pasted-session",
            )
            self.assertEqual(
                (root / "url").read_text(encoding="utf-8"),
                "https://coder.yoda.hyperverge.org",
            )

    def test_expired_session_reopens_login_page_and_retries(self) -> None:
        connections = []

        class ExpiredThenReadyCoderClient:
            def __init__(self, url, token):
                connections.append((url, token))
                self.token = token

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def current_user(self):
                if self.token == "expired-session":
                    raise CoderError(
                        "You are signed out or your session has expired",
                        status_code=401,
                    )
                return {"username": "divyam.c"}

            def workspace_for_job(self, _job, **_kwargs):
                return ready_workspace("lime-gull-30")

            def wait_until_ready(self, _user, name, **_kwargs):
                return ready_workspace(name)

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "HOME": directory,
                "CODER_CONFIG_DIR": f"{directory}/coderv2",
            },
            clear=True,
        ), patch(
            "falcon.cli.resolve_connection",
            return_value=("https://coder.example.test", "expired-session"),
        ), patch(
            "falcon.cli.CoderClient", ExpiredThenReadyCoderClient
        ), patch(
            "falcon.cli.getpass.getpass", return_value="replacement-session"
        ), patch("falcon.cli.sys.stdin", TtyStringIO()):
            stdout = TtyStringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main([
                    "coder",
                    "coder-divyam.c-lime-gull-30",
                    "--access",
                    "terminal",
                ])

            root = Path(directory) / "coderv2"
            self.assertEqual((code, stderr.getvalue()), (0, ""))
            self.assertEqual(
                connections,
                [
                    ("https://coder.example.test", "expired-session"),
                    ("https://coder.example.test", "replacement-session"),
                    ("https://coder.example.test", "replacement-session"),
                ],
            )
            self.assertIn("/cli-auth", stdout.getvalue())
            self.assertIn("Coder login saved for divyam.c", stdout.getvalue())
            self.assertIn("Workspace ready:", stdout.getvalue())
            self.assertEqual(
                (root / "session").read_text(encoding="utf-8"),
                "replacement-session",
            )

    def test_rejected_interactive_token_is_not_saved(self) -> None:
        class RejectingCoderClient:
            def __init__(self, _url, _token):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def current_user(self):
                raise CoderError("Unauthorized", status_code=401)

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "HOME": directory,
                "CODER_CONFIG_DIR": f"{directory}/coderv2",
            },
            clear=True,
        ), patch("falcon.cli.CoderClient", RejectingCoderClient), patch(
            "falcon.cli.getpass.getpass", return_value="bad-session"
        ), patch(
            "falcon.cli.sys.stdin", TtyStringIO()
        ):
            stdout = TtyStringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(["coder", "-c", "4", "-m", "8Gi"])

            root = Path(directory) / "coderv2"
            self.assertEqual(code, EXIT_CODER)
            self.assertIn("Coder rejected that session token", stderr.getvalue())
            self.assertFalse((root / "session").exists())


if __name__ == "__main__":
    unittest.main()
