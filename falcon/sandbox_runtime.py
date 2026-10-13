"""Dependency-free in-Pod relay, supervisor, and restricted Falcon client.

This file is copied into a read-only runtime bundle. Nothing here possesses
cluster credentials; all requests remain subject to the external broker.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import io
import json
import math
import os
import re
import selectors
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener, urlopen

MAX_BODY = 256 * 1024
PORT = 8765


def codex_rate_limits(spec: dict) -> dict:
    """Read quota only; never start a turn, redeem credits, or change login."""
    command = [spec["argv"][0], "app-server", "--listen", "stdio://"]
    # Keep provider/config overrides (including the container-local SQLite
    # path), without passing TUI-only permission or resume flags to app-server.
    arguments = spec["argv"][1:]
    for index, argument in enumerate(arguments):
        if argument in {"-c", "--config"} and index + 1 < len(arguments):
            command.extend([argument, arguments[index + 1]])
    environment = dict(os.environ)
    environment.pop("FALCON_SANDBOX_WORKER_TOKEN", None)
    process = subprocess.Popen(command, cwd=spec["workspace"], env=environment,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    deadline = time.monotonic() + 15
    buffer = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)

        def send(value: dict) -> None:
            process.stdin.write((json.dumps(value) + "\n").encode())
            process.stdin.flush()

        def reply(identifier: int) -> dict:
            while time.monotonic() < deadline:
                while b"\n" in buffer:
                    line, _, remaining = buffer.partition(b"\n")
                    buffer[:] = remaining
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("Invalid Codex quota response")
                    if value.get("id") == identifier:
                        if "error" in value or not isinstance(value.get("result"), dict):
                            raise ValueError("Codex quota query failed")
                        return value["result"]
                if not selector.select(max(0, deadline - time.monotonic())):
                    break
                chunk = os.read(process.stdout.fileno(), 8192)
                if not chunk:
                    break
                buffer.extend(chunk)
                if len(buffer) > MAX_BODY:
                    raise ValueError("Codex quota response exceeds limit")
            raise ValueError("Codex quota query timed out")

        try:
            send({"id": 1, "method": "initialize", "params": {
                "clientInfo": {"name": "falcon-sandbox", "version": "1"}}})
            reply(1)
            send({"method": "initialized"})
            send({"id": 2, "method": "account/rateLimits/read"})
            return reply(2)
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
            process.stdin.close()
            process.stdout.close()


def quota_reset(limits: dict) -> tuple[bool, float | None]:
    """Fail closed on unknown snapshots; wait for every exhausted window."""
    snapshot = limits.get("rateLimits")
    if not isinstance(snapshot, dict):
        raise ValueError("Codex did not provide quota windows")
    exhausted = []
    windows = 0
    for name in ("primary", "secondary"):
        window = snapshot.get(name)
        if window is None:
            continue
        if not isinstance(window, dict):
            raise ValueError("Unknown Codex quota window")
        used = window.get("usedPercent")
        reset = window.get("resetsAt")
        if isinstance(used, bool) or not isinstance(used, (int, float)) or not math.isfinite(used) or used < 0:
            raise ValueError("Unknown Codex quota usage")
        windows += 1
        if used >= 100:
            if isinstance(reset, bool) or not isinstance(reset, (int, float)) or not math.isfinite(reset) or reset <= 0:
                raise ValueError("Codex did not provide a reset time")
            exhausted.append(float(reset))
    if not windows:
        raise ValueError("Codex did not provide quota windows")
    # A server-classified credit/billing/unknown limit must not be interpreted
    # as available quota merely because percentage windows are below 100%.
    if not exhausted and snapshot.get("rateLimitReachedType"):
        raise ValueError("Codex reports a non-window limit")
    return bool(exhausted), max(exhausted) if exhausted else None


SGR = re.compile(r"\x1b\[([0-9;]*)m")


def empty_codex_input(screen: str) -> bool:
    """Allow empty input or styled placeholder, never ordinary draft text."""
    lines = [line for line in screen.splitlines()
             if SGR.sub("", line).strip().startswith("›")]
    if not lines:
        return False
    line = lines[-1].split("›", 1)[1]
    muted = False
    position = 0
    for match in SGR.finditer(line):
        if line[position:match.start()].strip() and not muted:
            return False
        codes = [int(code or "0") for code in match.group(1).split(";")]
        if 0 in codes or 22 in codes or 39 in codes:
            muted = False
        if 2 in codes or 90 in codes or codes == [38, 5, 8]:
            muted = True
        elif 38 in codes or any(30 <= code <= 37 for code in codes):
            muted = False
        position = match.end()
    return not line[position:].strip() or muted


class AutoContinue:
    """One continuation per confirmed reset, scoped to the original tmux pane."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.path = Path(spec["record"]).with_name("auto-continue.json")
        try:
            self.state = json.loads(self.path.read_text())
            if not isinstance(self.state, dict):
                self.state = {}
        except (OSError, ValueError):
            self.state = {}
        self.next_check = 0.0
        self.pending = self.state.get("reset_at")
        if self.pending is not None and (isinstance(self.pending, bool) or not isinstance(self.pending, (int, float)) or not math.isfinite(self.pending) or self.pending <= 0):
            self.pending = None
            self.state.pop("reset_at", None)
        self.pane = None

    def tmux(self, *arguments: str) -> str:
        result = subprocess.run(["tmux", "-L", "falcon", *arguments],
                                capture_output=True, text=True, check=True, timeout=5)
        if len(result.stdout) > MAX_BODY:
            raise ValueError("tmux response exceeds limit")
        return result.stdout.strip()

    def tick(self, status: dict, now: float) -> dict:
        if status.get("state") != "running" or not status.get("session_id"):
            return {"status": "waiting for running Codex session"}
        if now < self.next_check:
            return dict(self.state)
        self.next_check = now + 5
        try:
            pane = self.tmux("display-message", "-p", "-t", "agent:0.0", "#{pane_id}")
            if not pane.startswith("%") or not pane[1:].isdigit():
                raise ValueError("Cannot identify agent pane")
            if self.pane is not None and pane != self.pane:
                raise ValueError("Agent pane changed; automatic input disabled")
            self.pane = pane
            screen = self.tmux("capture-pane", "-p", "-e", "-t", pane)
            text = SGR.sub("", screen).replace("’", "'")
            notice = re.search(r"(?im)^\s*[■●•]?\s*you've\s+hit\s+your\s+(?:usage\s+)?limit\b", text) is not None
            if not notice:
                self.pending = None
                self.state.pop("reset_at", None)
                self.state["status"] = "watching for usage limit"
            elif self.pending is None:
                exhausted, reset = quota_reset(codex_rate_limits(self.spec))
                self.next_check = now + 60
                if exhausted and reset != self.state.get("last_sent_reset"):
                    self.pending = reset
                    self.state.update(status="waiting for quota reset", reset_at=reset)
                else:
                    self.state["status"] = "no new confirmed exhausted quota window"
            elif now >= self.pending + 10:
                # Never type into an attached session or tmux copy mode.
                attached = self.tmux("display-message", "-p", "-t", pane,
                                     "#{session_attached}:#{pane_in_mode}:#{pane_dead}")
                if attached != "0:0:0":
                    self.state["status"] = "reset due; detach to allow continuation"
                else:
                    # A detached user can still leave an unfinished draft.
                    # Unknown composer content is safer to leave untouched.
                    if not empty_codex_input(screen):
                        self.state["status"] = "reset due; clear the input draft before detaching"
                    else:
                        exhausted, _ = quota_reset(codex_rate_limits(self.spec))
                        self.next_check = now + 60
                        if exhausted:
                            self.state["status"] = "waiting for confirmed available quota"
                        else:
                            # Recheck after the network query: attachment or
                            # manual progress during it invalidates this turn.
                            if self.tmux("display-message", "-p", "-t", pane,
                                         "#{session_attached}:#{pane_in_mode}:#{pane_dead}") != "0:0:0" or self.tmux("capture-pane", "-p", "-e", "-t", pane) != screen:
                                self.state["status"] = "input changed during quota check; waiting"
                                atomic_json(self.path, self.state)
                                return dict(self.state)
                            # Record before sending: a crash between writing
                            # and input delivery must not duplicate a prompt.
                            self.state.update(status="continue sent", last_sent_reset=self.pending)
                            self.state.pop("reset_at", None)
                            atomic_json(self.path, self.state)
                            self.pending = None
                            self.tmux("send-keys", "-t", pane, "-l", "continue")
                            self.tmux("send-keys", "-t", pane, "Enter")
            atomic_json(self.path, self.state)
        except (OSError, ValueError, subprocess.SubprocessError):
            self.state["status"] = "automatic continuation paused: could not safely verify quota/pane"
            self.next_check = now + 60
        return dict(self.state)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_: object, **__: object) -> None:
        return None


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream)
    temporary.replace(path)


def http(base: str, token: str, route: str, value: object = None, *, timeout: float = 20) -> object:
    payload = None if value is None else json.dumps(value).encode()
    if payload and len(payload) > MAX_BODY:
        raise ValueError("sandbox request is too large")
    request = Request(base + route, data=payload, headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    # The relay is local/cluster-private. Never forward session capabilities
    # through a launcher's or image's generic HTTP proxy settings.
    with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=timeout) as response:
        body = response.read(MAX_BODY + 1)
    if len(body) > MAX_BODY:
        raise ValueError("sandbox response is too large")
    return json.loads(body)


class Relay(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], token: str, worker_token: str):
        super().__init__(address, RelayHandler)
        self.token = token
        self.worker_token = worker_token
        self.lock = threading.Lock()
        self.requests: dict[str, dict] = {}
        self.replies: dict[str, dict] = {}
        self.status: dict = {"state": "starting"}
        self.heartbeat = 0.0


class RelayHandler(BaseHTTPRequestHandler):
    server: Relay

    def log_message(self, *_: object) -> None:
        pass  # Never log tokens, commands, config, or request bodies.

    def answer(self, code: int, value: object) -> None:
        data = json.dumps(value).encode()
        if len(data) > MAX_BODY:
            code, data = 413, b'{"error":"response too large"}'
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self, worker: bool = False) -> bool:
        expected = self.server.worker_token if worker else self.server.token
        actual = self.headers.get("Authorization", "")
        if not hmac.compare_digest(actual, "Bearer " + expected):
            self.answer(403, {"error": "invalid sandbox capability"})
            return False
        return True

    def do_GET(self) -> None:
        worker = self.path in {"/work", "/worker-status"}
        if not self.authorized(worker):
            return
        with self.server.lock:
            if self.path == "/work":
                self.server.heartbeat = time.monotonic()
                self.answer(200, next(iter(self.server.requests.values()), None))
            elif self.path in {"/status", "/worker-status"}:
                self.answer(200, {**self.server.status, "broker_connected": time.monotonic() - self.server.heartbeat < 10})
            elif self.path.startswith("/result/"):
                self.answer(200, self.server.replies.get(self.path.removeprefix("/result/")))
            else:
                self.answer(404, {"error": "unknown relay route"})

    def do_POST(self) -> None:
        worker = self.path.startswith("/reply/")
        if not self.authorized(worker):
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise ValueError("invalid request size")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("request must be an object")
            with self.server.lock:
                if self.path == "/request":
                    rid = value.get("id", "")
                    uuid.UUID(rid)
                    if len(self.server.requests) >= 128:
                        raise ValueError("sandbox queue is full")
                    prior = self.server.requests.get(rid)
                    if prior is not None and prior != value:
                        raise ValueError("request ID reused for a different operation")
                    if rid not in self.server.replies:
                        self.server.requests[rid] = value
                    self.answer(200, {"id": rid})
                elif self.path.startswith("/reply/"):
                    rid = self.path.removeprefix("/reply/")
                    self.server.requests.pop(rid, None)
                    self.server.replies[rid] = value
                    if len(self.server.replies) > 256:
                        self.server.replies.pop(next(iter(self.server.replies)))
                    self.answer(200, {"ok": True})
                else:
                    self.answer(404, {"error": "unknown relay route"})
        except (ValueError, TypeError, UnicodeError) as exc:
            self.answer(400, {"error": str(exc)})


def proxy_request(operation: str, payload: dict, request_id: str | None = None) -> dict:
    base = os.environ["FALCON_SANDBOX_URL"]
    token = os.environ["FALCON_SANDBOX_TOKEN"]
    rid = request_id or str(uuid.uuid4())
    value = {"protocol": 1, "id": rid, "operation": operation, "payload": payload}
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            # Repost after a relay/Pod restart. Broker deduplication is durable.
            http(base, token, "/request", value)
            result = http(base, token, "/result/" + rid)
            if isinstance(result, dict):
                if "error" in result:
                    raise ValueError(result["error"])
                return result
        except HTTPError as exc:
            raise ValueError(f"sandbox relay rejected request (HTTP {exc.code})") from exc
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(0.25)
    raise ValueError(f"broker unavailable; existing Jobs are unaffected. Retry with --request-id {rid}")


def proxy_cli(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print("Sandbox Falcon: [GPU_PRESET | -c CPU -m MEMORY] -- COMMAND; jobs, get, logs, metrics, kill. CPU defaults: 1 core, 1Gi. All Jobs inherit this session's mount restrictions and Python environment.")
        return 0
    operation = argv[0] if argv[0] in {"jobs", "get", "logs", "metrics", "kill"} else "launch"
    parser = argparse.ArgumentParser(prog="falcon")
    parser.add_argument("--request-id")
    parser.add_argument("--output", choices=("human", "json"), default="human")
    if operation == "launch":
        preset = None
        if not argv[0].startswith("-"):
            preset, argv = argv[0], argv[1:]
        parser.add_argument("--gpu", default=preset)
        parser.add_argument("--gpus", "-n", type=int, default=1)
        parser.add_argument("--cpu", "-c")
        parser.add_argument("--memory", "-m")
        parser.add_argument("--name", "-j")
        parser.add_argument("--image")
        parser.add_argument("--environment")
        parser.add_argument("--shm-size")
        parser.add_argument("--shm-percent", type=float)
        parser.add_argument("--max", dest="maximize", action="store_true")
        parser.add_argument("--pin-node", action="store_true")
        parser.add_argument("--env", action="append", default=[])
        parser.add_argument("command", nargs=argparse.REMAINDER)
    else:
        argv = argv[1:]
        if operation != "jobs":
            parser.add_argument("job")
        if operation == "logs":
            parser.add_argument("--tail", type=int, default=100)
            parser.add_argument("--no-follow", action="store_true")
            parser.add_argument("--follow", action="store_true")
        if operation == "metrics":
            parser.add_argument("--interval", type=float, default=10)
    args = vars(parser.parse_args(argv))
    rid = args.pop("request_id")
    output = args.pop("output")
    args.pop("no_follow", None)
    follow = args.pop("follow", False)
    if operation == "launch":
        if args["command"][:1] == ["--"]:
            args["command"] = args["command"][1:]
        args["working_dir"] = str(Path.cwd())
    previous = ""
    while True:
        result = proxy_request(operation, args, rid)
        if output == "json" or operation == "metrics":
            print(json.dumps(result))
        elif "text" in result:
            text = result["text"]
            print(text[len(previous):] if text.startswith(previous) else text, end="", flush=True)
            previous = text
        elif operation == "launch":
            print(f"Submitted Job {result['name']} in namespace {result['namespace']}.")
        else:
            print(json.dumps(result, indent=2))
        if not follow:
            return 0
        rid = None
        time.sleep(2)


def exact_codex_session(pid: int) -> str | None:
    """Attribute the rollout to this process, not shared history's --last."""
    try:
        for descriptor in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = descriptor.resolve(strict=True)
                if target.name.startswith("rollout-") and target.suffix == ".jsonl":
                    with target.open() as stream:
                        record = json.loads(stream.readline(65536))
                    if record.get("type") == "session_meta":
                        return str(uuid.UUID(record["payload"]["id"]))
            except (OSError, ValueError, KeyError):
                continue
    except OSError:
        pass
    try:
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        for child in children:
            session = exact_codex_session(int(child))
            if session:
                return session
    except OSError:
        pass
    return None


def opencode_session(executable: str, cwd: str) -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = subprocess.Popen([executable, "serve", "--hostname", "127.0.0.1", "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if server.poll() is not None:
                raise ValueError("OpenCode session server failed; verify the installed version")
            try:
                request = Request(f"http://127.0.0.1:{port}/session", data=b"{}", headers={"Content-Type": "application/json", "x-opencode-directory": cwd})
                with urlopen(request, timeout=2) as response:
                    value = json.load(response)
                session = value["id"]
                if not isinstance(session, str) or not session.startswith("ses_"):
                    raise ValueError("OpenCode did not return a session ID")
                return session
            except (URLError, TimeoutError):
                time.sleep(0.1)
        raise ValueError("OpenCode session server did not become ready")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def run_agent(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    record = Path(spec["record"])
    previous = json.loads(record.read_text()) if record.exists() else {}
    argv = list(spec["argv"])
    agent = spec["agent"]
    session_id = previous.get("session_id")
    if previous:
        if previous.get("state") == "exited":
            print("Agent exited; the sandbox will be stopped.")
            return 0
        if not session_id or agent not in {"codex", "claude", "opencode"}:
            print("Cannot safely resume: no exact saved conversation ID. Reconnect and restart manually.", file=sys.stderr)
            return 1
        argv = [argv[0], *( ["resume", session_id] if agent == "codex" else ["--resume", session_id] if agent == "claude" else ["--session", session_id]), *spec["resume_flags"]]
    elif agent == "claude" and not any(a in argv for a in ("--version", "--help", "-h")):
        if "--session-id" in argv:
            session_id = argv[argv.index("--session-id") + 1]
        elif not any(a in argv for a in ("-r", "--resume", "-c", "--continue")):
            session_id = str(uuid.uuid4())
            argv += ["--session-id", session_id]
    elif agent == "opencode" and not any(a in argv for a in ("--version", "--help", "-h")):
        if "--session" in argv:
            session_id = argv[argv.index("--session") + 1]
        elif not any(a in argv for a in ("-c", "--continue", "-s")):
            session_id = opencode_session(argv[0], spec["workspace"])
            argv += ["--session", session_id]
    environment = dict(os.environ)
    environment.pop("FALCON_SANDBOX_WORKER_TOKEN", None)
    environment["FALCON_SANDBOX_URL"] = f"http://{spec['session']}.{spec['namespace']}.svc:{PORT}"
    environment["FALCON_SANDBOX_TOKEN"] = spec["token"]
    if agent == "opencode" and spec["permissions"] == "full":
        settings = json.loads(environment.get("OPENCODE_CONFIG_CONTENT", "{}"))
        if not isinstance(settings, dict):
            raise ValueError("OPENCODE_CONFIG_CONTENT must be a JSON object")
        settings["permission"] = "allow"
        environment["OPENCODE_CONFIG_CONTENT"] = json.dumps(settings)
    state = {"state": "running", "session_id": session_id, "attempt": previous.get("attempt", 0) + 1}
    atomic_json(record, state)
    process = subprocess.Popen(argv, cwd=spec["workspace"], env=environment)
    while process.poll() is None:
        if agent == "codex" and not session_id:
            session_id = exact_codex_session(process.pid)
            if session_id:
                state["session_id"] = session_id
                atomic_json(record, state)
        time.sleep(0.25)
    atomic_json(record, {**state, "state": "exited", "exit_code": process.returncode})
    return int(process.returncode)


def session_status(record: Path) -> dict:
    """Detect tmux termination even when the agent wrapper receives SIGHUP."""
    status = json.loads(record.read_text()) if record.exists() else {}
    if status.get("state") != "exited":
        probe = subprocess.run(
            ["tmux", "-L", "falcon", "has-session", "-t", "agent"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        if probe.returncode:
            status = {**status, "state": "exited", "reason": "terminal session closed"}
            atomic_json(record, status)
    return status


def prepare_python_environment(spec: dict) -> None:
    selected = spec.get("python_environment")
    if not selected:
        return
    source, target = Path(selected["source"]), Path(selected["path"])
    workspace = Path(spec["workspace"]).resolve()
    if not target.resolve().is_relative_to(workspace):
        raise ValueError("writable Python environment escapes workspace")
    marker = target / ".falcon-environment.json"
    if marker.exists():
        if json.loads(marker.read_text()).get("source") != str(source):
            raise ValueError("sandbox environment already uses another base; use a new sandbox name")
        if not (target / "bin/python").is_file():
            raise ValueError("sandbox Python environment is incomplete")
        return
    python = source / "bin/python"
    print(f"Preparing writable Python environment from {source}", flush=True)
    subprocess.run([str(python), "-I", "-m", "venv", "--system-site-packages", str(target)], check=True, timeout=120)
    query = "import json,sysconfig; print(json.dumps(sorted({sysconfig.get_path('purelib'),sysconfig.get_path('platlib')})))"
    inherited = json.loads(subprocess.check_output([str(python), "-I", "-c", query], text=True, timeout=30))
    own = json.loads(subprocess.check_output([str(target / "bin/python"), "-I", "-c", query], text=True, timeout=30))
    # A source virtualenv's packages are not inherited by --system-site-packages
    # alone, which only adds its interpreter's base installation.
    if any("\n" in path or "\r" in path for path in inherited):
        raise ValueError("unsupported newline in inherited package path")
    (Path(own[0]) / "falcon-inherited.pth").write_text("\n".join(inherited) + "\n")
    atomic_json(marker, {"source": str(source)})


def supervise(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    archive = Path(__file__).parent / "native.b64"
    if archive.exists():
        directory = Path("/opt/falcon-cache/native")
        directory.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(archive.read_bytes()))) as bundle:
            for item in bundle.infolist():
                if Path(item.filename).name != item.filename:
                    raise ValueError("invalid native runtime archive")
                target = directory / item.filename
                target.write_bytes(bundle.read(item))
                target.chmod(0o555)
    for command in ("python3", "tmux"):
        subprocess.run([command, "--version" if command == "python3" else "-V"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([spec["argv"][0], "--version"], check=True, stdout=subprocess.DEVNULL)
    os.chdir(spec["workspace"])
    record = Path(spec["record"])
    record.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    (record.parent.parent / ".gitignore").write_text("*\n")
    prepare_python_environment(spec)
    os.environ["FALCON_SANDBOX_URL"] = f"http://127.0.0.1:{PORT}"
    os.environ["FALCON_SANDBOX_TOKEN"] = spec["token"]
    if spec["agent"] == "codex" and os.environ.get("CODEX_HOME") == "/opt/falcon-agent-state":
        for name in ("app-server-control", "app-server-daemon", "tmp", ".tmp", "shell_snapshots", "logs"):
            (Path("/opt/falcon-codex-runtime") / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    if spec["agent"] == "claude":
        mirror = record.parent / "claude-config.json"
        base_file = record.parent / "claude-config-base.json"
        initial = json.loads(mirror.read_text()) if mirror.exists() else spec.get("legacy_initial", {})
        atomic_json(Path.home() / ".claude.json", initial)
        if not base_file.exists():
            atomic_json(base_file, spec.get("legacy_initial", {}))

        def sync() -> None:
            while True:
                try:
                    value = json.loads((Path.home() / ".claude.json").read_text())
                    base = json.loads(base_file.read_text())
                    atomic_json(mirror, value)
                    result = proxy_request("config-sync", {"base": base, "value": value})
                    if json.loads((Path.home() / ".claude.json").read_text()) == value:
                        atomic_json(Path.home() / ".claude.json", result["value"])
                        atomic_json(base_file, result["value"])
                    if result["conflicts"]:
                        print("Claude config sync conflict; concurrent host changes were preserved.", file=sys.stderr)
                except (ValueError, OSError, KeyError):
                    pass
                time.sleep(2)

        threading.Thread(target=sync, daemon=True).start()
    relay = Relay(("0.0.0.0", PORT), spec["token"], spec["worker_token"])
    threading.Thread(target=relay.serve_forever, daemon=True).start()
    executable = shlex.join([sys.executable, str(Path(__file__)), "agent", str(spec_path)])
    subprocess.run(["tmux", "-L", "falcon", "-f", str(spec_path.parent / "tmux.conf"), "new-session", "-d", "-s", "agent", executable], check=True)
    continuation = AutoContinue(spec) if spec.get("auto_continue") and spec["agent"] == "codex" else None
    try:
        while True:
            try:
                status = session_status(record)
                if continuation is not None:
                    status["auto_continue"] = continuation.tick(status, time.time())
                with relay.lock:
                    relay.status = status
            except (OSError, ValueError):
                pass
            time.sleep(1)
    finally:
        relay.shutdown()


def main() -> int:
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "client":
            spec_file = Path("/opt/falcon-session/spec.json")
            if "FALCON_SANDBOX_TOKEN" not in os.environ and spec_file.exists():
                spec = json.loads(spec_file.read_text())
                os.environ["FALCON_SANDBOX_TOKEN"] = spec["token"]
                os.environ["FALCON_SANDBOX_URL"] = f"http://{spec['session']}.{spec['namespace']}.svc:{PORT}"
            return proxy_cli(sys.argv[2:])
        if len(sys.argv) == 3 and sys.argv[1] == "agent":
            return run_agent(Path(sys.argv[2]))
        if len(sys.argv) == 3 and sys.argv[1] == "supervise":
            supervise(Path(sys.argv[2]))
            return 0
        raise ValueError("invalid sandbox runtime invocation")
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(f"falcon sandbox: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
