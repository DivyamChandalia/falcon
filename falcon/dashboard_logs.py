"""Non-interactive Job output capture for the Falcon dashboard.

Running Pods are observed with ``kubectl attach`` while terminal attempts use
the same captured log path as ``falcon logs --no-follow``.  The manager keeps
the process and buffer lifetime separate from the Textual event loop so a
noisy or unavailable Pod can never block rendering.
"""

from __future__ import annotations

import codecs
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple

from .commands import capture_logs
from .kubernetes import KubernetesClient

MAX_LOG_LINES = 200
LOG_RETENTION_SECONDS = 24 * 60 * 60
ATTACH_RETRY_SECONDS = 5.0

_ANSI_ESCAPE_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[@-_])"
)
_TQDM_PERCENT_RE = re.compile(r"(?<!\d)(?:100|[1-9]?\d)%\|")
_TQDM_ITER_RE = re.compile(r"(?<!\w)\d+(?:\.\d+)?it\s*\[")


def _clean_terminal_text(value: str) -> str:
    """Remove terminal controls while preserving the text they decorated."""

    value = _ANSI_ESCAPE_RE.sub("", value)
    output: List[str] = []
    for character in value:
        if character == "\b":
            if output:
                output.pop()
        elif character >= " " or character == "\t":
            output.append(character)
    return "".join(output)


def _tqdm_progress_key(value: str) -> str:
    """Return a stable identity for a tqdm-shaped line, or an empty string."""

    match = _TQDM_PERCENT_RE.search(value)
    if match is not None and "|" in value[match.end() :]:
        return f"percent:{value[:match.start()].strip()}"
    match = _TQDM_ITER_RE.search(value)
    if match is not None and ("it/s" in value or "s/it" in value):
        return f"iter:{value[:match.start()].strip()}"
    return ""


class _TerminalLineParser:
    """Turn a byte stream into committed lines and replaceable CR previews."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._current = ""
        self._after_carriage_return = False

    def feed(self, chunk: bytes | str) -> List[Tuple[str, str]]:
        text = (
            self._decoder.decode(chunk)
            if isinstance(chunk, bytes)
            else str(chunk)
        )
        events: List[Tuple[str, str]] = []
        for character in text:
            if character == "\r":
                if self._current:
                    events.append(("preview", _clean_terminal_text(self._current)))
                self._current = ""
                self._after_carriage_return = True
            elif character == "\n":
                if self._after_carriage_return and not self._current:
                    events.append(("commit-existing", ""))
                else:
                    events.append(("commit", _clean_terminal_text(self._current)))
                self._current = ""
                self._after_carriage_return = False
            else:
                self._current += character
                self._after_carriage_return = False
        if self._current:
            events.append(("preview", _clean_terminal_text(self._current)))
        return events

    def finish(self) -> List[Tuple[str, str]]:
        events = self.feed(self._decoder.decode(b"", final=True))
        if self._current:
            events.append(("commit-existing", ""))
            self._current = ""
        return events


@dataclass(frozen=True)
class LogSnapshot:
    """Immutable view of one Pod's captured output."""

    pod_name: str
    status: str = "unavailable"
    lines: Tuple[str, ...] = ()
    error: str = ""
    # Changes whenever the selected Pod's visible state or retained output
    # changes.  The dashboard uses this cheap token to avoid re-rendering 200
    # lines of Rich content on every log polling tick.
    revision: int = 0


@dataclass
class _Line:
    captured_at: float
    value: str
    progress_key: str = ""


@dataclass
class _PodLogState:
    job_uid: str
    job_name: str
    pod_uid: str
    pod_name: str
    phase: str
    container: str
    lines: Deque[_Line] = field(
        default_factory=lambda: deque(maxlen=MAX_LOG_LINES)
    )
    status: str = "waiting"
    error: str = ""
    process: Optional[subprocess.Popen] = None
    connecting: bool = False
    loading_terminal: bool = False
    terminal_loaded: bool = False
    retry_at: float = 0.0
    revision: int = 0
    transient_line: bool = False


class DashboardLogManager:
    """Own attach processes and bounded buffers for dashboard Job attempts."""

    def __init__(
        self,
        namespace: str,
        *,
        max_lines: int = MAX_LOG_LINES,
        retention_seconds: float = LOG_RETENTION_SECONDS,
        retry_seconds: float = ATTACH_RETRY_SECONDS,
        clock: Callable[[], float] = time.time,
        client: Optional[KubernetesClient] = None,
        on_change: Optional[Callable[[], None]] = None,
    ) -> None:
        if not namespace:
            raise ValueError("dashboard log namespace must not be empty")
        if max_lines <= 0:
            raise ValueError("dashboard log line limit must be positive")
        if retention_seconds <= 0:
            raise ValueError("dashboard log retention must be positive")
        self.namespace = namespace
        self.max_lines = int(max_lines)
        self.retention_seconds = float(retention_seconds)
        self.retry_seconds = max(0.1, float(retry_seconds))
        self._clock = clock
        self._client = client or KubernetesClient(namespace)
        self._on_change = on_change
        self._lock = threading.RLock()
        self._states: Dict[Tuple[str, str], _PodLogState] = {}
        self._closed = False

    @staticmethod
    def _job_key(row: object) -> str:
        return str(getattr(row, "uid", "") or getattr(row, "job", ""))

    @staticmethod
    def _attempts(row: object) -> Iterable[object]:
        attempts = tuple(getattr(row, "attempt_details", ()) or ())
        if attempts:
            return attempts
        pod_name = str(getattr(row, "active_pod", "") or "")
        if not pod_name:
            return ()
        # Older/demo collectors do not expose attempt metadata.  Retain a
        # small compatibility target so log rendering still has a useful
        # empty state when a real manager is injected in a test.
        return (
            type(
                "DashboardAttempt",
                (),
                {
                    "name": pod_name,
                    "uid": str(getattr(row, "active_pod_uid", "") or pod_name),
                    "phase": "Running"
                    if str(getattr(row, "active_pod_state", "")).lower()
                    == "running"
                    else "Unknown",
                    "container": "main",
                },
            )(),
        )

    def _key(self, row: object, attempt: object) -> Tuple[str, str]:
        job_uid = self._job_key(row)
        pod_name = str(getattr(attempt, "name", "") or "")
        pod_uid = str(getattr(attempt, "uid", "") or pod_name)
        return job_uid, pod_uid

    def _notify(self) -> None:
        callback = self._on_change
        if callback:
            try:
                callback()
            except Exception:
                # A UI callback must never take down a reader thread.
                pass

    @staticmethod
    def _error_text(error: BaseException) -> str:
        return next(iter(str(error).splitlines()), type(error).__name__)

    def _prune_locked(self, now: Optional[float] = None) -> None:
        current = self._clock() if now is None else now
        cutoff = current - self.retention_seconds
        for state in self._states.values():
            changed = False
            while state.lines and state.lines[0].captured_at < cutoff:
                state.lines.popleft()
                changed = True
            while len(state.lines) > self.max_lines:
                state.lines.popleft()
                changed = True
            if changed:
                state.revision += 1

    @staticmethod
    def _terminate(process: Optional[subprocess.Popen]) -> None:
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.5)
        except (OSError, subprocess.SubprocessError):
            pass

    def _new_state(self, row: object, attempt: object) -> _PodLogState:
        key = self._key(row, attempt)
        state = _PodLogState(
            job_uid=key[0],
            job_name=str(getattr(row, "job", "") or ""),
            pod_uid=key[1],
            pod_name=str(getattr(attempt, "name", "") or ""),
            phase=str(getattr(attempt, "phase", "Unknown") or "Unknown"),
            container=str(getattr(attempt, "container", "main") or "main"),
        )
        state.lines = deque(maxlen=self.max_lines)
        self._states[key] = state
        return state

    def reconcile(self, rows: Iterable[object]) -> None:
        """Start attach streams for all running attempts and stop stale ones."""

        if self._closed:
            return
        rows = list(rows)
        now = self._clock()
        running: Dict[Tuple[str, str], Tuple[object, object]] = {}
        known: set[Tuple[str, str]] = set()
        with self._lock:
            self._prune_locked(now)
            for row in rows:
                for attempt in self._attempts(row):
                    key = self._key(row, attempt)
                    if not key[0] or not key[1] or not getattr(attempt, "name", ""):
                        continue
                    known.add(key)
                    state = self._states.get(key) or self._new_state(row, attempt)
                    previous = (
                        state.job_name,
                        state.pod_name,
                        state.phase,
                        state.container,
                    )
                    state.job_name = str(getattr(row, "job", "") or state.job_name)
                    state.pod_name = str(getattr(attempt, "name", "") or state.pod_name)
                    state.phase = str(getattr(attempt, "phase", "Unknown") or "Unknown")
                    state.container = str(
                        getattr(attempt, "container", "main") or "main"
                    )
                    if previous != (
                        state.job_name,
                        state.pod_name,
                        state.phase,
                        state.container,
                    ):
                        state.revision += 1
                    if state.phase.lower() == "running":
                        running[key] = (row, attempt)
                    elif state.process is not None:
                        process = state.process
                        state.process = None
                        state.status = "exited"
                        state.revision += 1
                        self._terminate(process)
            # Rows disappearing from the collector should not leave attach
            # processes alive.  Their buffers are intentionally discarded too
            # because there is no longer a selectable Job to display them.
            for key, state in list(self._states.items()):
                if key not in known:
                    self._terminate(state.process)
                    del self._states[key]

            starts: List[Tuple[Tuple[str, str], object, object]] = []
            for key, (row, attempt) in running.items():
                state = self._states[key]
                if (
                    state.process is None
                    and not state.connecting
                    and now >= state.retry_at
                    and not self._closed
                ):
                    state.status = "connecting"
                    state.connecting = True
                    state.error = ""
                    state.retry_at = now + self.retry_seconds
                    state.revision += 1
                    starts.append((key, row, attempt))
        for key, row, attempt in starts:
            self._start_attach(key, row, attempt)
        self._notify()

    def _start_attach(self, key: Tuple[str, str], row: object, attempt: object) -> None:
        try:
            process = self._client.attach_stream(
                str(getattr(attempt, "name", "")),
                container=str(getattr(attempt, "container", "main") or "main"),
            )
        except Exception as exc:
            with self._lock:
                state = self._states.get(key)
                if state:
                    state.connecting = False
                    state.status = "error"
                    state.error = self._error_text(exc)
                    state.retry_at = self._clock() + self.retry_seconds
                    state.revision += 1
            self._notify()
            return
        with self._lock:
            state = self._states.get(key)
            if (
                state is None
                or self._closed
                or state.phase.lower() != "running"
            ):
                if state is not None:
                    state.connecting = False
                self._terminate(process)
                return
            state.connecting = False
            state.process = process
            state.status = "streaming"
            state.error = ""
            state.revision += 1
        threading.Thread(
            target=self._read_attach,
            args=(key, process),
            name=f"falcon-dashboard-attach-{getattr(attempt, 'name', 'pod')}",
            daemon=True,
        ).start()
        self._notify()

    def _read_attach(
        self,
        key: Tuple[str, str],
        process: subprocess.Popen,
    ) -> None:
        stream = getattr(process, "stdout", None)
        parser = _TerminalLineParser()
        try:
            if stream is not None:
                while True:
                    raw = stream.read(4096)
                    if not raw:
                        break
                    events = parser.feed(raw)
                    with self._lock:
                        state = self._states.get(key)
                        if state is None:
                            continue
                        changed = self._apply_terminal_events(state, events)
                        self._prune_locked()
                    if changed:
                        self._notify()
                with self._lock:
                    state = self._states.get(key)
                    changed = (
                        self._apply_terminal_events(state, parser.finish())
                        if state is not None
                        else False
                    )
                if changed:
                    self._notify()
            returncode = process.wait()
        except Exception as exc:
            returncode = 1
            error = self._error_text(exc)
        else:
            error = ""
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return
            state.process = None
            state.retry_at = self._clock() + self.retry_seconds
            if returncode:
                state.status = "error"
                state.error = error or f"attach exited with status {returncode}"
            else:
                state.status = "exited"
                state.error = ""
            state.revision += 1
        self._notify()

    def _apply_terminal_events(
        self,
        state: _PodLogState,
        events: Iterable[Tuple[str, str]],
    ) -> bool:
        """Apply terminal line events, replacing the active tqdm row on CR."""

        changed = False
        for action, value in events:
            if action == "preview":
                # tqdm pads shorter redraws with spaces to erase the previous
                # bar. They should not create wrapped blank rows in the TUI.
                value = value.rstrip(" ")
                if not value:
                    continue
                progress_key = _tqdm_progress_key(value)
                line = _Line(self._clock(), value, progress_key)
                if state.transient_line and state.lines:
                    if state.lines[-1].value != value:
                        state.lines[-1] = line
                        changed = True
                else:
                    if (
                        progress_key
                        and state.lines
                        and state.lines[-1].progress_key == progress_key
                    ):
                        state.lines[-1] = line
                    else:
                        state.lines.append(line)
                    changed = True
                state.transient_line = True
            elif action == "commit":
                progress_key = _tqdm_progress_key(value)
                line = _Line(self._clock(), value, progress_key)
                if state.transient_line and state.lines:
                    if state.lines[-1].value != value:
                        state.lines[-1] = line
                        changed = True
                else:
                    if (
                        progress_key
                        and state.lines
                        and state.lines[-1].progress_key == progress_key
                    ):
                        state.lines[-1] = line
                    else:
                        state.lines.append(line)
                    changed = True
                state.transient_line = False
            elif action == "commit-existing":
                state.transient_line = False
        if changed:
            state.revision += 1
        return changed

    def ensure_terminal_logs(self, row: object, attempt: object) -> None:
        """Load one terminal attempt once, asynchronously, on first display."""

        if self._closed:
            return
        key = self._key(row, attempt)
        with self._lock:
            state = self._states.get(key) or self._new_state(row, attempt)
            phase = str(getattr(attempt, "phase", "Unknown") or "Unknown")
            if state.phase != phase:
                state.phase = phase
                state.revision += 1
            if state.terminal_loaded or state.loading_terminal:
                return
            state.loading_terminal = True
            state.status = "loading"
            state.error = ""
            state.revision += 1
        threading.Thread(
            target=self._load_terminal,
            args=(key, row, attempt),
            name=f"falcon-dashboard-logs-{getattr(attempt, 'name', 'pod')}",
            daemon=True,
        ).start()
        self._notify()

    def _load_terminal(
        self,
        key: Tuple[str, str],
        row: object,
        attempt: object,
    ) -> None:
        try:
            result = capture_logs(
                self.namespace,
                str(getattr(row, "job", "") or ""),
                pod_name=str(getattr(attempt, "name", "") or ""),
                # Ask kubectl for no more than the in-memory viewport can
                # retain.  The default is 200 lines, preventing a completed
                # Pod with a huge log from ever being copied into the TUI.
                tail=self.max_lines,
                follow=False,
                container=str(getattr(attempt, "container", "main") or "main"),
                client=self._client,
            )
            if result.returncode:
                raise RuntimeError(
                    result.stderr.strip()
                    or result.stdout.strip()
                    or f"falcon logs exited with status {result.returncode}"
                )
            parser = _TerminalLineParser()
            events = parser.feed(result.stdout)
            events.extend(parser.finish())
            values: Deque[Tuple[str, str]] = deque(maxlen=self.max_lines)
            transient = False
            for action, value in events:
                if action == "preview":
                    value = value.rstrip(" ")
                    if not value:
                        continue
                    progress_key = _tqdm_progress_key(value)
                    item = (value, progress_key)
                    if transient and values:
                        values[-1] = item
                    elif (
                        progress_key
                        and values
                        and values[-1][1] == progress_key
                    ):
                        values[-1] = item
                    else:
                        values.append(item)
                    transient = True
                elif action == "commit":
                    progress_key = _tqdm_progress_key(value)
                    item = (value, progress_key)
                    if transient and values:
                        values[-1] = item
                    elif (
                        progress_key
                        and values
                        and values[-1][1] == progress_key
                    ):
                        values[-1] = item
                    else:
                        values.append(item)
                    transient = False
                elif action == "commit-existing":
                    transient = False
            error = ""
        except Exception as exc:
            values = []
            error = self._error_text(exc)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return
            state.loading_terminal = False
            state.terminal_loaded = True
            state.lines.clear()
            state.transient_line = False
            now = self._clock()
            state.lines.extend(
                _Line(now, value, progress_key)
                for value, progress_key in values
            )
            state.status = "loaded" if not error else "error"
            state.error = error
            state.revision += 1
            self._prune_locked(now)
        self._notify()

    def snapshot(self, row: object, attempt: object) -> LogSnapshot:
        key = self._key(row, attempt)
        with self._lock:
            self._prune_locked()
            state = self._states.get(key)
            if state is None:
                return LogSnapshot(
                    pod_name=str(getattr(attempt, "name", "") or ""),
                    status="unavailable",
                )
            return LogSnapshot(
                pod_name=state.pod_name,
                status=state.status,
                lines=tuple(line.value for line in state.lines),
                error=state.error,
                revision=state.revision,
            )

    def snapshot_revision(self, row: object, attempt: object) -> int:
        """Return a selected Pod revision without copying its retained lines."""

        key = self._key(row, attempt)
        with self._lock:
            self._prune_locked()
            state = self._states.get(key)
            return state.revision if state is not None else 0

    def prune(self) -> None:
        """Discard expired output even while the selected pane is collapsed."""

        if self._closed:
            return
        with self._lock:
            self._prune_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            states = list(self._states.values())
            self._states.clear()
        for state in states:
            self._terminate(state.process)
