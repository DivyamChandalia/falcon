"""Keyboard-first nvitop/htop-inspired Textual interface for Falcon Jobs."""

from __future__ import annotations

import subprocess
import threading
import textwrap
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean
from typing import Deque, Dict, List, Optional, Set, Tuple

from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from .dashboard import (
    KUBERNETES_USAGE_SECONDS, JobEvent, JobUsage, _metric_color, _percent,
    _short_cpu, _short_memory, _timestamp,
)


CYAN = "#00FFFF"
CYAN_2 = "#4DDDDD"
GREEN = "#55FF55"
YELLOW = "#FFFF55"
RED = "#FF5555"
WHITE = "#F2F2F2"
GRAY = "#AAAAAA"
MUTED = "#666666"
BORDER = "#555555"
MINIMUM_WIDTH = 80
MINIMUM_HEIGHT = 30


@dataclass
class ViewState:
    cursor_job_uid: str = ""
    marked_job_uids: Set[str] = field(default_factory=set)
    focused_pane: str = "jobs"
    expanded_pane: Optional[str] = None
    jobs_scroll_offset: int = 0
    resource_scroll_offset: int = 0
    events_scroll_offset: int = 0
    events_auto_follow: bool = True
    selected_command_scroll_offset: int = 0
    search_query: str = ""
    filters: Dict[str, str] = field(default_factory=lambda: {
        "status": "All", "pod": "All", "node": "All", "gpu": "All", "marked": "All",
    })
    sort_field: str = "Age"
    sort_direction: str = "desc"
    resource_zoom: int = 1
    resource_range_samples: int = 60
    kill_dialog: Dict = field(default_factory=lambda: {
        "isOpen": False, "targets": [], "action": "job", "confirmationStage": 0,
    })
    last_successful_refresh: float = 0.0
    loading_states: Dict[str, bool] = field(default_factory=dict)
    gpu_availability: Dict[str, Tuple[int, int]] = field(default_factory=dict)


@dataclass
class MetricPoint:
    timestamp: float
    gpu: Optional[float]
    vram: Optional[float]
    cpu: Optional[float]
    ram: Optional[float]


def _truncate(value: str, width: int) -> str:
    if width <= 0:
        return ""
    return value if len(value) <= width else value[: max(1, width - 1)] + "…"


def _spark(values: List[Optional[float]], width: int = 12) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    usable = [value for value in values[-width:] if value is not None]
    if not usable:
        return "—"
    rendered = []
    for value in values[-width:]:
        if value is None:
            rendered.append("·")
        else:
            rendered.append(blocks[min(7, max(0, round(value / 100 * 7)))])
    return "".join(rendered)


def _scaled_history(
    values: List[Optional[float]], width: int = 12, height: int = 3,
    samples_per_bar: int = 1, sample_period: int = 1,
) -> Text:
    """Render bottom-aligned history with an exact metric-sample scale."""
    partials = "▁▂▃▄▅▆▇"
    if not values or not any(value is not None for value in values):
        return Text("No metrics", style=MUTED, no_wrap=True, overflow="crop")
    width = max(1, width)
    height = max(1, height)
    samples_per_bar = max(1, samples_per_bar)
    sample_period = max(1, sample_period)
    source = values[-width * samples_per_bar:]
    # CPU/RAM values are cached between kubectl-top refreshes. Collapse each
    # native polling interval to one value, then expand it across the seconds
    # for which that sample is valid. At 100%, a five-second CPU sample is
    # therefore a five-column plateau rather than five pretend samples.
    timeline: List[Optional[float]] = []
    for start in range(0, len(source), sample_period):
        chunk = source[start:start + sample_period]
        valid = [value for value in chunk if value is not None]
        native = mean(valid) if valid else None
        timeline.extend([native] * len(chunk))
    grouped: List[Optional[float]] = []
    for start in range(0, len(timeline), samples_per_bar):
        valid = [value for value in timeline[start:start + samples_per_bar] if value is not None]
        grouped.append(mean(valid) if valid else None)
    empty = object()
    sampled: List[object] = list(grouped[-width:])
    # Keep newest history anchored to the right edge during warm-up while
    # preserving exactly one bar per sample/group.
    sampled = [empty] * (width - len(sampled)) + sampled
    # Width calculations are based on the terminal, while Rich ultimately
    # owns the exact nested table-cell width. Never let a one-column rounding
    # difference turn the final GPU blocks into an extra visual row.
    text = Text(no_wrap=True, overflow="crop")
    for level in reversed(range(height)):
        for value in sampled:
            if value is empty:
                text.append(" ")
                continue
            if value is None:
                text.append("·" if level == 0 else " ", style=MUTED)
                continue
            numeric = float(value)
            scaled = max(0.125, min(100.0, numeric) / 100 * height)
            full, fraction = int(scaled), scaled - int(scaled)
            if level < full:
                text.append("█", style=_metric_color(numeric))
            elif level == full and fraction > 0:
                index = min(6, max(0, round(fraction * 7) - 1))
                text.append(partials[index], style=_metric_color(numeric))
            else:
                text.append(" ")
        if level:
            text.append("\n")
    return text


def _status_style(status: str) -> Tuple[str, str]:
    lowered = status.lower()
    if lowered in {"running", "succeeded"}:
        return ("●" if lowered == "running" else "✓", GREEN)
    if lowered in {"failed", "unknown"}:
        return ("✕", RED)
    return ("●", YELLOW)


def _gpu_display(gpu_type: str, count: int) -> str:
    if count <= 0:
        return "CPU only"
    normalized = gpu_type.strip().replace(" ", "").lower()
    return f"{normalized}x{count}"


def _event_style(event: JobEvent) -> str:
    failure = {"failed", "backoff", "unhealthy", "evicted", "oomkilled", "failedscheduling"}
    if event.reason.lower() in failure:
        return RED
    return YELLOW if event.event_type.lower() == "warning" else GREEN


class DashboardPane(Static):
    can_focus = True

    def on_focus(self) -> None:
        pane = self.id.replace("-pane", "") if self.id else "jobs"
        callback = getattr(self.app, "pane_focused", None)
        if callback:
            callback(pane)

    def on_click(self, event: events.Click) -> None:
        self.focus()
        callback = getattr(self.app, "pane_clicked", None)
        if callback:
            callback(self.id or "", event)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.prevent_default()
        event.stop()
        callback = getattr(self.app, "scroll_focused", None)
        if callback:
            callback(1, self.id)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.prevent_default()
        event.stop()
        callback = getattr(self.app, "scroll_focused", None)
        if callback:
            callback(-1, self.id)


class KillDialog(ModalScreen[Optional[Tuple[str, List[JobUsage]]]]):
    CSS = f"""
    KillDialog {{ align: center middle; background: #000000; }}
    #kill-box {{ width: 76; height: auto; max-height: 28; border: solid {RED}; background: #000000; padding: 1 2; }}
    #kill-text {{ height: auto; background: #000000; color: {WHITE}; }}
    #kill-count {{ display: none; height: 1; background: #000000; color: {WHITE}; border: none; }}
    """
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("enter", "advance", "Confirm", priority=True),
        Binding("left", "toggle_action", "Action", priority=True),
        Binding("right", "toggle_action", "Action", priority=True),
        Binding("y", "advance", "Confirm", show=False, priority=True),
        Binding("shift+y", "advance", "Confirm", show=False, priority=True),
    ]

    def __init__(self, rows: List[JobUsage]):
        super().__init__()
        self.rows = rows
        self.action = "job"
        self.stage = 0

    def compose(self) -> ComposeResult:
        with Container(id="kill-box"):
            yield Static(id="kill-text")
            yield Input(
                placeholder=f"Type {len(self.rows)} to confirm deletion",
                id="kill-count", disabled=True,
            )

    def on_mount(self) -> None:
        self._render_dialog()

    def _render_dialog(self) -> None:
        names = "\n".join(f"  {row.job}" for row in self.rows[:12])
        if len(self.rows) > 12:
            names += f"\n  … and {len(self.rows) - 12} more"
        job_marker = "●" if self.action == "job" else "○"
        pod_marker = "●" if self.action == "pod" else "○"
        prompt = "Enter Confirm    Esc Cancel"
        if self.stage == 1:
            prompt = f"Delete {len(self.rows)} Jobs? Press y or Enter to confirm."
        elif self.stage == 2:
            prompt = f"Type {len(self.rows)} to confirm deletion:"
        text = Text()
        text.append(f"Kill Jobs\n\nDelete {len(self.rows)} Job{'s' if len(self.rows) != 1 else ''}?\n\n", style=f"bold {RED}")
        text.append(names + "\n\n", style=WHITE)
        text.append("Action\n", style=GRAY)
        text.append(f"  {job_marker} Delete Job and managed pods\n", style=CYAN if self.action == "job" else WHITE)
        text.append(f"  {pod_marker} Delete active pod only\n", style=CYAN if self.action == "pod" else WHITE)
        if self.action == "pod":
            text.append("\nThe Job controller may create another pod.\n", style=YELLOW)
        text.append("\n" + prompt, style=GRAY)
        self.query_one("#kill-text", Static).update(text)

    def action_toggle_action(self) -> None:
        self.action = "pod" if self.action == "job" else "job"
        self._render_dialog()

    def action_advance(self) -> None:
        count = len(self.rows)
        if self.stage == 1:
            self.dismiss((self.action, self.rows))
            return
        if count == 1:
            self.dismiss((self.action, self.rows))
        elif count >= 10:
            self.stage = 2
            control = self.query_one("#kill-count", Input)
            control.disabled = False
            control.display = True
            control.focus()
            self._render_dialog()
        else:
            self.stage = 1
            self._render_dialog()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.stage == 2 and event.value.strip() == str(len(self.rows)):
            self.dismiss((self.action, self.rows))
        elif self.stage == 2:
            self.notify("Confirmation count does not match", severity="error")

    def action_cancel(self) -> None:
        self.dismiss(None)


class CleanupDialog(ModalScreen[bool]):
    CSS = f"""
    CleanupDialog {{ align: center middle; background: #000000; }}
    #cleanup-box {{ width: 76; height: auto; max-height: 28; border: solid {YELLOW}; background: #000000; padding: 1 2; }}
    #cleanup-text {{ height: auto; background: #000000; color: {WHITE}; }}
    """
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("enter", "confirm", "Clean", priority=True),
        Binding("y", "confirm", "Clean", show=False, priority=True),
        Binding("shift+y", "confirm", "Clean", show=False, priority=True),
    ]

    def __init__(self, rows: List[JobUsage]):
        super().__init__()
        self.rows = rows

    def compose(self) -> ComposeResult:
        names = "\n".join(f"  {row.job}" for row in self.rows[:16])
        if len(self.rows) > 16:
            names += f"\n  … and {len(self.rows) - 16} more"
        text = Text("CLEAN SUCCEEDED JOBS\n\n", style=f"bold {YELLOW}")
        text.append(f"Delete {len(self.rows)} succeeded Job{'s' if len(self.rows) != 1 else ''}?\n\n", style=WHITE)
        text.append(names, style=GRAY)
        text.append("\n\nRunning and failed Jobs will not be touched.\n", style=GREEN)
        text.append("\nEnter/y Clean    Esc Cancel", style=GRAY)
        with Container(id="cleanup-box"):
            yield Static(text, id="cleanup-text")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class FilterDialog(ModalScreen[Optional[Dict[str, str]]]):
    CSS = f"""
    FilterDialog {{ align: center middle; background: #000000; }}
    #filter-box {{ width: 76; height: 17; border: solid {CYAN}; background: #000000; padding: 1 2; }}
    #filter-text {{ background: #000000; color: {WHITE}; }}
    """
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("enter", "apply", "Apply", priority=True),
        Binding("up", "up", "Previous", priority=True),
        Binding("down", "down", "Next", priority=True),
        Binding("left", "previous_value", "Previous value", priority=True),
        Binding("right", "next_value", "Next value", priority=True),
        Binding("space", "next_value", "Next value", show=False, priority=True),
    ]

    def __init__(self, current: Dict[str, str], rows: List[JobUsage]):
        super().__init__()
        self.values = dict(current)
        self.fields = ["status", "pod", "node", "gpu", "marked"]
        self.index = 0
        self.options = {
            "status": ["All"] + sorted({row.status for row in rows}),
            "pod": ["All"] + sorted({row.active_pod_state for row in rows}),
            "node": ["All"] + sorted({row.nodes for row in rows}),
            "gpu": ["All"] + sorted({row.gpu_type for row in rows}),
            "marked": ["All", "Marked"],
        }

    def compose(self) -> ComposeResult:
        with Container(id="filter-box"):
            yield Static(id="filter-text")

    def on_mount(self) -> None:
        self._render_dialog()

    def _render_dialog(self) -> None:
        text = Text("FILTER JOBS\n", style=f"bold {CYAN}")
        text.append("Choose a row, then change its value.\n\n", style=GRAY)
        for index, field in enumerate(self.fields):
            selected = index == self.index
            text.append("> " if selected else "  ", style=CYAN if selected else GRAY)
            text.append(f"{field.title():<10}  [ {self.values[field]} ]\n", style=CYAN if selected else WHITE)
        text.append("\n↑/↓ Select filter   ←/→ or Space Change   Enter Apply   Esc Cancel", style=GRAY)
        self.query_one("#filter-text", Static).update(text)

    def action_up(self) -> None:
        self.index = (self.index - 1) % len(self.fields); self._render_dialog()

    def action_down(self) -> None:
        self.index = (self.index + 1) % len(self.fields); self._render_dialog()

    def _cycle(self, amount: int) -> None:
        field = self.fields[self.index]
        options = self.options[field]
        current = self.values.get(field, "All")
        index = options.index(current) if current in options else 0
        self.values[field] = options[(index + amount) % len(options)]
        self._render_dialog()

    def action_previous_value(self) -> None: self._cycle(-1)
    def action_next_value(self) -> None: self._cycle(1)
    def action_apply(self) -> None: self.dismiss(self.values)
    def action_cancel(self) -> None: self.dismiss(None)


CSS = f"""
Screen {{ background: #000000; color: {WHITE}; overflow: hidden; }}
Static, Input, Container {{ background: #000000; }}
#falcon-header {{ height: 1; color: {WHITE}; padding: 0 1; }}
#summary {{ height: 2; color: {WHITE}; border-bottom: solid {BORDER}; }}
#controls {{ height: 1; color: {GRAY}; padding: 0 1; }}
#search-input {{ height: 1; display: none; border: none; padding: 0 1; color: {WHITE}; }}
DashboardPane {{ border: solid {BORDER}; background: #000000; color: {WHITE}; padding: 0 1; }}
DashboardPane:focus {{ border: solid {CYAN}; }}
#jobs-pane {{ height: 1fr; min-height: 7; }}
#selected-pane {{ height: 3; min-height: 3; }}
#resources-pane {{ height: 6; min-height: 5; }}
#events-pane {{ height: 7; min-height: 7; }}
#resize-message {{ display: none; height: 1fr; content-align: center middle; color: {YELLOW}; }}
#falcon-footer {{ height: 1; color: {GRAY}; padding: 0 1; }}
"""


class FalconDashboard(App):
    TITLE = "Falcon Dashboard"
    ENABLE_COMMAND_PALETTE = False
    CSS = CSS
    BINDINGS = [
        Binding("q", "quit", "Quit"), Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("tab", "next_pane", "Next pane", priority=True),
        Binding("shift+tab", "previous_pane", "Previous pane", priority=True),
        Binding("1", "focus_jobs", "Jobs", show=False), Binding("2", "focus_resources", "Resources", show=False),
        Binding("3", "focus_events", "Events", show=False), Binding("4", "focus_selected", "Selected Job", show=False),
        Binding("enter", "expand", "Expand"), Binding("z", "toggle_expand", "Expand", show=False),
        Binding("escape", "escape", "Restore", show=False), Binding("r", "update_data", "Refresh"),
        Binding("up", "up", "Up", show=False), Binding("down", "down", "Down", show=False),
        Binding("left", "left", "Left", show=False), Binding("right", "right", "Right", show=False),
        Binding("j", "down", "Down", show=False), Binding("k", "kill_or_up", "Kill / Up", show=False),
        Binding("pageup", "page_up", "Page up", show=False), Binding("pagedown", "page_down", "Page down", show=False),
        Binding("home", "home", "Oldest", show=False), Binding("end", "end", "Latest", show=False),
        Binding("h", "history_left", "Earlier", show=False), Binding("l", "history_right", "Later", show=False),
        Binding("space", "toggle_mark", "Mark", show=False), Binding("shift+space", "mark_next", "Mark next", show=False),
        Binding("a", "mark_all", "Mark all", show=False), Binding("A", "clear_marks", "Clear marks", show=False),
        Binding("m", "marked_only", "Marked only", show=False), Binding("f9", "kill", "Kill", show=False),
        Binding("c", "cleanup", "Clean succeeded", show=False),
        Binding("/", "search", "Search", show=False), Binding("f", "filters", "Filters", show=False),
        Binding("s", "cycle_sort", "Sort", show=False), Binding("?", "help", "Help", show=False),
        Binding("R", "cycle_resource_range", "Range", show=False),
        Binding("Z", "cycle_resource_zoom", "Zoom", show=False),
        Binding("plus", "resource_zoom_in", "Zoom in", show=False),
        Binding("minus", "resource_zoom_out", "Zoom out", show=False),
    ]

    def __init__(self, collector, refresh_seconds: float = 1.0):
        super().__init__()
        self.collector = collector
        self.refresh_seconds = refresh_seconds
        self.state = ViewState()
        self.rows: List[JobUsage] = []
        self.filtered_rows: List[JobUsage] = []
        self.job_events: List[JobEvent] = []
        self.event_search = ""
        self.histories: Dict[str, Deque[MetricPoint]] = {}
        self.summary_history: Deque[MetricPoint] = deque(maxlen=120)
        self._refreshing = False
        self._stale = False
        self._spinner = 0
        self._result_queue = __import__("queue").Queue(maxsize=1)

    @property
    def selected(self) -> int:
        for index, row in enumerate(self.filtered_rows):
            if row.uid == self.state.cursor_job_uid:
                return index
        return 0

    @selected.setter
    def selected(self, value: int) -> None:
        if self.filtered_rows:
            value = max(0, min(len(self.filtered_rows) - 1, value))
            self.state.cursor_job_uid = self.filtered_rows[value].uid

    def compose(self) -> ComposeResult:
        yield Static(id="falcon-header")
        yield Static(id="summary")
        yield Static(id="controls")
        yield Input(placeholder="Search jobs…", id="search-input")
        yield DashboardPane(id="jobs-pane")
        yield DashboardPane(id="selected-pane")
        yield DashboardPane(id="resources-pane")
        yield DashboardPane(id="events-pane")
        yield Static(id="resize-message")
        yield Static(id="falcon-footer")

    def on_mount(self) -> None:
        self.query_one("#jobs-pane", DashboardPane).border_title = " JOBS "
        self.query_one("#selected-pane", DashboardPane).border_title = " SELECTED JOB "
        self.query_one("#resources-pane", DashboardPane).border_title = " RESOURCE USAGE "
        self.query_one("#events-pane", DashboardPane).border_title = " EVENTS — last 20 "
        self.query_one("#jobs-pane", DashboardPane).focus()
        self._apply_layout()
        self._request_update()
        self.set_interval(self.refresh_seconds, self._request_update)
        self.set_interval(0.2, self._drain_results)
        self.set_interval(1.0, self._tick_clock)
        self._render_all()

    def on_unmount(self) -> None:
        close = getattr(self.collector, "close", None)
        if close:
            close()

    def on_resize(self, event: events.Resize) -> None:
        self._apply_layout()
        self._render_all()

    def _tick_clock(self) -> None:
        self._spinner = (self._spinner + 1) % 4
        try:
            self._render_header()
        except Exception:
            # A timer may race with Textual tearing down the default screen.
            return

    def pane_focused(self, pane: str) -> None:
        self.state.focused_pane = pane
        self._render_footer()
        self._set_titles()

    def pane_clicked(self, pane_id: str, event: events.Click) -> None:
        if pane_id == "jobs-pane" and self.filtered_rows:
            index = self.state.jobs_scroll_offset + max(0, event.y - 2)
            if index < len(self.filtered_rows):
                self.selected = index
                if getattr(event, "ctrl", False):
                    self.action_toggle_mark()
                self._selection_changed()

    def scroll_focused(self, amount: int, pane_id: Optional[str] = None) -> None:
        pane = (pane_id or self.state.focused_pane).replace("-pane", "")
        if pane == "jobs":
            if self.state.expanded_pane == "jobs":
                self._scroll_jobs_view(amount)
            else:
                self._move_cursor(amount)
        elif pane == "selected":
            if self.state.expanded_pane == "selected":
                self._scroll_selected_command(amount)
            else:
                self._move_cursor(amount)
        elif pane == "events":
            self._scroll_events(amount)
        else:
            self._scroll_history(amount)

    def _selected_row(self) -> Optional[JobUsage]:
        return next((row for row in self.rows if row.uid == self.state.cursor_job_uid), None)

    def _filter_rows(self) -> None:
        query = self.state.search_query.lower().strip()
        result = []
        for row in self.rows:
            values = [row.job, row.status, row.active_pod_state, row.nodes, row.gpu_type]
            if query and not any(query in str(value).lower() for value in values):
                continue
            filters = self.state.filters
            if filters["status"] != "All" and row.status != filters["status"]:
                continue
            if filters["pod"] != "All" and row.active_pod_state != filters["pod"]:
                continue
            if filters["node"] != "All" and row.nodes != filters["node"]:
                continue
            if filters["gpu"] != "All" and row.gpu_type != filters["gpu"]:
                continue
            if filters["marked"] == "Marked" and row.uid not in self.state.marked_job_uids:
                continue
            result.append(row)
        reverse = self.state.sort_direction == "desc"
        key = {
            "Name": lambda row: row.job.lower(),
            "Status": lambda row: row.status,
            "Age": lambda row: -_timestamp(row.created_at),
        }.get(self.state.sort_field, lambda row: -_timestamp(row.created_at))
        self.filtered_rows = sorted(result, key=key, reverse=reverse)
        if self.state.cursor_job_uid not in {row.uid for row in self.filtered_rows}:
            self.state.cursor_job_uid = self.filtered_rows[0].uid if self.filtered_rows else ""
        self._ensure_cursor_visible()

    def _record_history(self) -> None:
        for row in self.rows:
            values = (
                row.gpu_util if row.gpu_metrics_available else None,
                row.gpu_memory_percent,
                row.cpu_percent,
                row.memory_percent,
            )
            if not any(value is not None for value in values):
                continue
            history = self.histories.setdefault(row.uid, deque(maxlen=600))
            history.append(MetricPoint(time.time(), *values))
        gpu_weight = sum(row.gpu_count for row in self.rows if row.gpu_util is not None)
        gpu = (
            sum((row.gpu_util or 0) * row.gpu_count for row in self.rows if row.gpu_util is not None) / gpu_weight
            if gpu_weight else None
        )
        cpu = _percent(sum(row.cpu_used for row in self.rows), sum(row.cpu_requested for row in self.rows))
        ram = _percent(sum(row.memory_used_gib for row in self.rows), sum(row.memory_requested_gib for row in self.rows))
        self.summary_history.append(MetricPoint(time.time(), gpu, None, cpu, ram))

    def _request_update(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        self.state.loading_states["refresh"] = True
        selected_uid = self.state.cursor_job_uid

        def collect() -> None:
            try:
                rows = self.collector.collect()
                selected = next((row for row in rows if row.uid == selected_uid), rows[0] if rows else None)
                events_method = getattr(self.collector, "events", None)
                job_events = events_method(selected) if events_method and selected else []
                payload = (
                    rows, job_events, selected.uid if selected else "",
                    getattr(self.collector, "last_error", "") or None,
                    getattr(self.collector, "last_successful_refresh", 0.0) or time.time(),
                    dict(getattr(self.collector, "gpu_availability", {})),
                )
            except Exception as exc:
                payload = (None, None, "", str(exc), 0.0, None)
            try:
                self._result_queue.put_nowait(payload)
            except __import__("queue").Full:
                pass

        threading.Thread(target=collect, name="falcon-dashboard-refresh", daemon=True).start()

    def _drain_results(self) -> None:
        try:
            rows, job_events, event_uid, error, refreshed_at, gpu_availability = self._result_queue.get_nowait()
        except __import__("queue").Empty:
            return
        self._refreshing = False
        self.state.loading_states["refresh"] = False
        if rows is not None:
            current_uids = {row.uid for row in rows}
            self.state.marked_job_uids.intersection_update(current_uids)
            self.rows = rows
            self.job_events = job_events or []
            self.state.last_successful_refresh = refreshed_at
            if gpu_availability is not None:
                self.state.gpu_availability = gpu_availability
            self._stale = bool(error)
            self._record_history()
            self._filter_rows()
            if error:
                self.notify(f"API error: {error} · retrying…", severity="warning")
            if event_uid != self.state.cursor_job_uid:
                self._request_update()
        else:
            self._stale = True
            if error:
                self.notify(f"API error: {error} · retrying…", severity="error")
        self._render_all()

    def _set_titles(self) -> None:
        for pane in ("jobs", "selected", "resources", "events"):
            widget = self.query_one(f"#{pane}-pane", DashboardPane)
            focused = pane == self.state.focused_pane
            base = {
                "jobs": "JOBS", "selected": "SELECTED JOB",
                "resources": "RESOURCE USAGE", "events": "EVENTS — last 20",
            }[pane]
            widget.border_title = f" {base}{' · focused' if focused else ''} "

    def _render_header(self) -> None:
        target = self.query_one("#falcon-header", Static)
        clock = datetime.now().strftime("%H:%M:%S")
        glyph = "◴◷◶◵"[self._spinner]
        status = f"[bold {RED}]STALE[/]" if self._stale else f"[{CYAN}]{glyph}[/]"
        width = max(30, self.size.width - 2)
        left = f"[bold {CYAN}]Falcon Dashboard[/]"
        right = f"[{GRAY}]{clock}[/]  {status}"
        gap = max(1, width - len("Falcon Dashboard") - len(clock) - 4)
        target.update(left + " " * gap + right)

    def _render_summary(self) -> None:
        running = sum(row.status == "Running" for row in self.rows)
        risk = sum(row.at_risk for row in self.rows)
        succeeded = sum(row.status == "Succeeded" for row in self.rows)
        failed = sum(row.status == "Failed" for row in self.rows)
        nodes = len({row.nodes for row in self.rows if row.nodes not in {"—", "-"}})
        left = Text()
        for value, label, color in (
            (running, "RUNNING", GREEN), (risk, "RISK", YELLOW),
            (succeeded, "SUCCESS", GREEN), (failed, "FAILED", RED),
            (len(self.rows), "JOBS", WHITE), (nodes, "NODES", WHITE),
        ):
            if left:
                left.append("   ")
            left.append(f"{value} {label}", style=f"bold {color}")

        right = Text("RESOURCES AVAILABLE  ", style=f"bold {GRAY}")
        for gpu_type, label in (("2080ti", "2080Ti"), ("a6000", "A6000"), ("h100", "H100")):
            if not right.plain.endswith("  "):
                right.append("   ")
            free_total = self.state.gpu_availability.get(gpu_type)
            availability = "—/—" if free_total is None else f"{free_total[0]}/{free_total[1]}"
            used_percent = (
                None if free_total is None or free_total[1] <= 0
                else (free_total[1] - free_total[0]) / free_total[1] * 100
            )
            right.append(
                f"{label} {availability}",
                style=f"bold {_metric_color(used_percent) if used_percent is not None else MUTED}",
            )
        gap = max(2, self.size.width - len(left.plain) - len(right.plain) - 4)
        left.append(" " * gap)
        left.append_text(right)
        self.query_one("#summary", Static).update(left)

    def _render_controls(self) -> None:
        filters = self.state.filters
        def control(label: str, value: str) -> str:
            color = CYAN if value != "All" else GRAY
            suffix = " ×" if value != "All" else " ▾"
            return f"[{color}]{label}: {value}{suffix}[/]"
        query = self.state.search_query or "Search jobs… (/)"
        text = (
            f"[{CYAN}][ {query} ][/]  {control('Status', filters['status'])}  "
            f"{control('Pod', filters['pod'])}  {control('Node', filters['node'])}  "
            f"{control('GPU', filters['gpu'])}  [{GRAY}]Sort: {self.state.sort_field} "
            f"{'↓' if self.state.sort_direction == 'desc' else '↑'}[/]"
        )
        count = f"{len(self.filtered_rows)} jobs"
        gap = max(1, self.size.width - len(Text.from_markup(text).plain) - len(count) - 2)
        self.query_one("#controls", Static).update(text + " " * gap + f"[{GRAY}]{count}[/]")

    def _visible_job_count(self) -> int:
        widget = self.query_one("#jobs-pane", DashboardPane)
        return max(1, widget.size.height - 3)

    def _ensure_cursor_visible(self) -> None:
        count = self._visible_job_count() if self.is_mounted else 4
        index = self.selected
        if index < self.state.jobs_scroll_offset:
            self.state.jobs_scroll_offset = index
        elif index >= self.state.jobs_scroll_offset + count:
            self.state.jobs_scroll_offset = index - count + 1
        maximum = max(0, len(self.filtered_rows) - count)
        self.state.jobs_scroll_offset = max(0, min(self.state.jobs_scroll_offset, maximum))

    def _scroll_jobs_view(self, amount: int) -> None:
        maximum = max(0, len(self.filtered_rows) - self._visible_job_count())
        self.state.jobs_scroll_offset = max(0, min(maximum, self.state.jobs_scroll_offset + amount))
        self._render_jobs()

    def _render_jobs(self) -> None:
        target = self.query_one("#jobs-pane", DashboardPane)
        if not self.rows:
            target.update(Align.center("No Jobs found.\nPress f to change filters or r to refresh.", vertical="middle"))
            return
        if not self.filtered_rows:
            target.update(Align.center(
                f"No Jobs match “{self.state.search_query}”.\nPress Esc to clear search.", vertical="middle"
            ))
            return
        width = self.size.width
        expanded = self.state.expanded_pane == "jobs"
        table = Table(box=None, expand=True, padding=(0, 1), show_header=True, header_style=f"bold {CYAN_2}")
        table.add_column("MARK", width=5, no_wrap=True)
        table.add_column("NAME", ratio=3, no_wrap=True, overflow="ellipsis")
        table.add_column("STATUS", width=16, no_wrap=True)
        table.add_column("ACTIVE POD", width=18, no_wrap=True)
        if width >= 90:
            table.add_column("NODE", width=12, no_wrap=True)
        if width >= 115:
            table.add_column("GPU TYPE", width=13, no_wrap=True)
        if expanded:
            table.add_column("RESTARTS", width=8, justify="right")
            table.add_column("COMPLETIONS", width=11, justify="right")
        table.add_column("AGE", width=7, justify="right")
        count = self._visible_job_count()
        start = self.state.jobs_scroll_offset
        jobs_focused = self.state.focused_pane == "jobs"
        for index, row in enumerate(self.filtered_rows[start:start + count], start=start):
            selected = row.uid == self.state.cursor_job_uid
            marked = row.uid in self.state.marked_job_uids
            marker = ">" if selected else " "
            mark = "[x]" if marked else "[ ]"
            icon, status_color = _status_style("Pending" if row.at_risk else row.status)
            status_text = "Eviction risk" if row.at_risk else row.status
            selection_active = selected and jobs_focused
            cells: List[Text] = [Text(f"{marker}{mark}", style=CYAN if selection_active or marked else GRAY)]
            name_style = f"bold {CYAN}" if selection_active else (f"bold {WHITE}" if selected else WHITE)
            cells.append(Text(row.job, style=name_style, no_wrap=True, overflow="ellipsis"))
            cells.append(Text(f"{icon} {status_text}", style=status_color))
            pod_color = RED if row.active_pod_state in {"CrashLoopBackOff", "OOMKilled", "ImagePullBackOff", "Evicted"} else WHITE
            cells.append(Text(row.active_pod_state, style=pod_color, no_wrap=True, overflow="ellipsis"))
            if width >= 90:
                cells.append(Text(row.nodes, style=GRAY))
            if width >= 115:
                cells.append(Text(_gpu_display(row.gpu_type, row.gpu_count), style=WHITE))
            if expanded:
                cells.extend([Text(str(row.restarts), style=GRAY), Text(row.completions, style=GRAY)])
            cells.append(Text(row.age, style=GRAY))
            table.add_row(*cells)
        position = f" {min(start + 1, len(self.filtered_rows))}-{min(start + count, len(self.filtered_rows))}/{len(self.filtered_rows)} "
        target.border_subtitle = position if len(self.filtered_rows) > count else ""
        target.update(table)

    def _wrapped_command(self, row: JobUsage) -> List[str]:
        width = max(20, self.size.width - 10)
        lines: List[str] = []
        for source_line in (row.command or "—").splitlines() or ["—"]:
            lines.extend(textwrap.wrap(source_line, width=width, replace_whitespace=False) or [""])
        return lines

    def _command_view_height(self) -> int:
        return max(3, self.size.height - 25)

    def _scroll_selected_command(self, amount: int) -> None:
        row = self._selected_row()
        if not row:
            return
        maximum = max(0, len(self._wrapped_command(row)) - self._command_view_height())
        self.state.selected_command_scroll_offset = max(
            0, min(maximum, self.state.selected_command_scroll_offset + amount)
        )
        self._render_selected()

    def _render_selected(self) -> None:
        row = self._selected_row()
        target = self.query_one("#selected-pane", DashboardPane)
        if not row:
            target.update(Text("No Job selected", style=MUTED))
            return
        if self.state.expanded_pane == "selected":
            status_icon, status_color = _status_style(row.status)
            overview = Table.grid(expand=True, padding=(0, 2))
            overview.add_column(style=GRAY, width=18)
            overview.add_column(style=WHITE, ratio=1)
            details = [
                ("Job", row.job), ("Status", f"{status_icon} {row.status}"),
                ("Active pod state", row.active_pod_state), ("Active pod", row.active_pod or "—"),
                ("Node", row.nodes), ("Age", row.age),
                ("Created", row.created_at or "—"), ("Started", row.started_at or "—"),
                ("GPU allocation", _gpu_display(row.gpu_type, row.gpu_count)),
                ("CPU request", f"{_short_cpu(row.cpu_requested)} vCPU"),
                ("RAM request", _short_memory(row.memory_requested_gib)),
                ("Restarts", str(row.restarts)), ("Completions", row.completions or "—"),
                ("GPU EMA", "—" if row.gpu_ema is None else f"{row.gpu_ema:.1f}%"),
                ("GPU 60s average", "—" if row.gpu_risk_average is None else f"{row.gpu_risk_average:.1f}%"),
                ("Eviction risk", "YES" if row.at_risk else "No"),
            ]
            for label, value in details:
                value_style = status_color if label == "Status" else (RED if label == "Eviction risk" and row.at_risk else WHITE)
                overview.add_row(label, Text(value, style=value_style))
            command_lines = self._wrapped_command(row)
            command_height = self._command_view_height()
            maximum = max(0, len(command_lines) - command_height)
            self.state.selected_command_scroll_offset = min(self.state.selected_command_scroll_offset, maximum)
            command_start = self.state.selected_command_scroll_offset
            command_end = min(len(command_lines), command_start + command_height)
            command_position = (
                f" lines {command_start + 1}-{command_end}/{len(command_lines)} "
                if len(command_lines) > command_height else ""
            )
            command = Panel(
                Text("\n".join(command_lines[command_start:command_end]), style=WHITE),
                title=Text(" COMMAND ", style=f"bold {CYAN}"), subtitle=command_position,
                border_style=BORDER, box=box.SQUARE,
            )
            content = Table.grid(expand=True)
            content.add_column(ratio=1)
            content.add_row(Panel(
                overview, title=Text(" JOB DETAILS ", style=f"bold {CYAN}"),
                border_style=BORDER, box=box.SQUARE,
            ))
            content.add_row(command)
            target.border_subtitle = ""
            target.update(content)
            return
        command = _truncate(row.command or "—", max(8, self.size.width // 3))
        marked = len(self.state.marked_job_uids)
        text = Text(row.job, style=f"bold {WHITE}")
        text.append(f"   {row.status}   Active pod: {row.active_pod_state}   {row.nodes}   {row.age}", style=GRAY)
        if command and self.size.width >= 90:
            text.append(f"   {command}", style=WHITE)
        if marked:
            text.append(f"   Marked: {marked}", style=CYAN)
        target.border_subtitle = " Enter expand " if self.state.focused_pane == "selected" else ""
        target.update(text)

    def _history_slice(self, uid: str) -> List[MetricPoint]:
        history = list(self.histories.get(uid, []))
        width = min(
            600,
            max(self.state.resource_range_samples, self.size.width) * self.state.resource_zoom,
        )
        end = len(history) - self.state.resource_scroll_offset
        return history[max(0, end - width):max(0, end)]

    def _metric_cell(self, label: str, current: Optional[float], values: List[Optional[float]], detail: str) -> Text:
        color = _metric_color(current)
        text = Text(label + "\n", style=f"bold {WHITE}")
        text.append("—" if current is None else f"{current:.0f}%", style=color)
        text.append("\n" + _spark(values, 24) + "\n", style=color)
        text.append(detail, style=GRAY)
        return text

    @staticmethod
    def _device_value(value: Optional[float], suffix: str = "", precision: int = 0) -> str:
        return "—" if value is None else f"{value:.{precision}f}{suffix}"

    def _resource_layout(self) -> str:
        """Choose a layout from the real terminal dimensions, not sample count."""
        if self.size.height < 38:
            return "collapsed"
        if self.size.width >= 125 and self.size.height >= 48:
            return "wide"
        return "compact"

    @staticmethod
    def _absolute_metric(value: Optional[float], capacity: float, unit: str) -> str:
        if value is None or capacity <= 0:
            return "—"
        used = capacity * value / 100
        if unit == "GPU":
            return f"{used:.2f} / {capacity:.2f} GPU"
        if unit == "vCPU":
            return f"{_short_cpu(used)} / {_short_cpu(capacity)} vCPU"
        return f"{_short_memory(used)} / {_short_memory(capacity)}"

    def _resource_metrics(self, row: JobUsage, points: List[MetricPoint]) -> List[Dict]:
        last = points[-1]
        return [
            {
                "label": "GPU", "values": [point.gpu for point in points],
                "current": last.gpu, "capacity": float(row.gpu_count), "unit": "GPU",
                "sample_period": 1,
                "absolute": "—" if row.gpu_count == 0 or last.gpu is None else
                    f"{(last.gpu or 0) * row.gpu_count / 100:.2f} / {row.gpu_count:.2f} GPU",
            },
            {
                "label": "VRAM", "values": [point.vram for point in points],
                "current": last.vram, "capacity": row.gpu_memory_total_gib, "unit": "GiB",
                "sample_period": 1,
                "absolute": "—" if last.vram is None else
                    f"{_short_memory(row.gpu_memory_used_gib)} / {_short_memory(row.gpu_memory_total_gib)}",
            },
            {
                "label": "CPU", "values": [point.cpu for point in points],
                "current": last.cpu, "capacity": row.cpu_requested, "unit": "vCPU",
                "sample_period": int(KUBERNETES_USAGE_SECONDS),
                "absolute": "—" if last.cpu is None else
                    f"{_short_cpu(row.cpu_used)} / {_short_cpu(row.cpu_requested)} vCPU",
            },
            {
                "label": "RAM", "values": [point.ram for point in points],
                "current": last.ram, "capacity": row.memory_requested_gib, "unit": "GiB",
                "sample_period": int(KUBERNETES_USAGE_SECONDS),
                "absolute": "—" if last.ram is None else
                    f"{_short_memory(row.memory_used_gib)} / {_short_memory(row.memory_requested_gib)}",
            },
        ]

    def _resource_history_width(self, layout: str, stats_width: int = 0) -> int:
        """Use all chart space made available by terminal resizing."""
        if stats_width:
            if layout == "wide":
                return max(24, self.size.width - stats_width - 8)
            card_width = max(20, (self.size.width - 4) // 2)
            return max(8, card_width - stats_width - 5)
        if layout == "wide":
            return max(24, self.size.width - 48)
        return max(8, self.size.width // 5 - 5)

    def _resource_history_height(self, layout: str) -> int:
        # Multi-row ANSI updates visibly tear on terminals that don't support
        # synchronized output. Preserve the richer graph only when Textual has
        # confirmed that the complete frame can be presented atomically.
        if not getattr(self, "_sync_available", False):
            return 1
        return 3 if layout == "wide" else 4

    def _wide_metric_panel(self, metric: Dict) -> Panel:
        values = metric["values"]
        valid = [value for value in values if value is not None]
        current = metric["current"]
        average = mean(valid) if valid else None
        peak = max(valid) if valid else None
        summary = Text()
        summary.append("Now      ", style=GRAY)
        summary.append("—" if current is None else f"{current:.0f}%", style=f"bold {_metric_color(current)}")
        summary.append(f"  {metric['absolute']}\n", style=WHITE)
        summary.append("Average  ", style=GRAY)
        summary.append("—" if average is None else f"{average:.0f}%", style=_metric_color(average))
        summary.append("  " + self._absolute_metric(average, metric["capacity"], metric["unit"]) + "\n", style=WHITE)
        summary.append("Peak     ", style=GRAY)
        summary.append("—" if peak is None else f"{peak:.0f}%", style=_metric_color(peak))
        summary.append("  " + self._absolute_metric(peak, metric["capacity"], metric["unit"]), style=WHITE)
        stats_width = max(len(line) for line in summary.plain.splitlines())
        history = _scaled_history(
            values, self._resource_history_width("wide", stats_width), self._resource_history_height("wide"),
            self.state.resource_zoom, metric["sample_period"],
        )
        body = Table.grid(expand=True, padding=0)
        body.add_column(width=stats_width, justify="left")
        body.add_column(width=1)
        body.add_column(ratio=1, justify="right", vertical="bottom")
        body.add_row(summary, " ", history)
        return Panel(
            body, title=Text(f" {metric['label']} ", style=f"bold {CYAN}"),
            border_style=CYAN_2, box=box.SQUARE, padding=(0, 1),
        )

    def _compact_metric_panel(self, metric: Dict, collapsed: bool = False) -> Panel:
        valid = [value for value in metric["values"] if value is not None]
        current = metric["current"]
        average = mean(valid) if valid else None
        peak = max(valid) if valid else None
        stats = Text()
        stats.append("Now      ", style=GRAY)
        stats.append("—" if current is None else f"{current:.0f}%", style=f"bold {_metric_color(current)}")
        stats.append("\nUsed     ", style=GRAY)
        stats.append(metric["absolute"], style=WHITE)
        stats.append("\nAverage  ", style=GRAY)
        stats.append("—" if average is None else f"{average:.0f}%", style=_metric_color(average))
        stats.append("\nPeak     ", style=GRAY)
        stats.append("—" if peak is None else f"{peak:.0f}%", style=_metric_color(peak))
        layout = "collapsed" if collapsed else "compact"
        stats_width = max(len(line) for line in stats.plain.splitlines())
        history = _scaled_history(
            metric["values"], self._resource_history_width(layout, stats_width),
            self._resource_history_height(layout), self.state.resource_zoom, metric["sample_period"],
        )
        body = Table.grid(expand=True, padding=0)
        body.add_column(width=stats_width, justify="left")
        body.add_column(width=1)
        body.add_column(ratio=2, justify="right", vertical="bottom")
        body.add_row(stats, " ", history)
        return Panel(
            body, title=Text(f" {metric['label']} ", style=f"bold {CYAN}"),
            border_style=CYAN_2, box=box.SQUARE, padding=(0, 1 if not collapsed else 0),
        )

    def _compact_metrics_grid(self, metrics: List[Dict], collapsed: bool = False) -> Table:
        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(
            self._compact_metric_panel(metrics[0], collapsed),
            self._compact_metric_panel(metrics[1], collapsed),
        )
        grid.add_row(
            self._compact_metric_panel(metrics[2], collapsed),
            self._compact_metric_panel(metrics[3], collapsed),
        )
        return grid

    def _selected_resource_strip(self, row: JobUsage, layout: str) -> Panel:
        icon, status_color = _status_style(row.status)
        command = row.command or "—"
        risk = ""
        if row.at_risk:
            average = "—" if row.gpu_risk_average is None else f"{row.gpu_risk_average:.1f}%"
            threshold = "—" if row.gpu_risk_threshold is None else f"{row.gpu_risk_threshold:.0f}%"
            risk = f"! EVICTION RISK · 60s average {average} < {threshold}"
        text = Text()
        if layout == "wide":
            text.append("Selected job: ", style=GRAY)
            text.append(_truncate(row.job, max(20, self.size.width // 4)), style=f"bold {CYAN_2}")
            text.append(f"   {icon} {row.status}", style=status_color)
            text.append(f"   Active Pod {row.active_pod_state}   ", style=GRAY)
            text.append(row.nodes, style=CYAN_2)
            text.append(f"   {row.age}   ", style=GRAY)
            text.append(_truncate(command, max(12, self.size.width - 106)), style=WHITE)
        elif layout == "compact":
            text.append("Job: ", style=GRAY)
            text.append(_truncate(row.job, max(20, self.size.width - 8)), style=f"bold {CYAN_2}")
            text.append(f"\n{icon} {row.status} | Pod {row.active_pod_state} | ", style=status_color)
            text.append(f"{row.nodes} | {row.age} | ", style=GRAY)
            text.append(_truncate(command, max(12, self.size.width // 2)), style=WHITE)
        else:
            text.append("Job: ", style=GRAY)
            text.append(_truncate(row.job, max(12, self.size.width // 3)), style=f"bold {CYAN_2}")
            text.append(f" | {row.status} | Pod {row.active_pod_state} | {row.nodes} | {row.age} | ", style=GRAY)
            text.append(_truncate(command, max(8, self.size.width // 4)), style=WHITE)
        if risk:
            text.append("\n" + risk, style=f"bold {RED}")
        return Panel(
            text, title=Text(" SELECTED JOB ", style=f"bold {CYAN}"),
            border_style=CYAN_2, box=box.SQUARE, padding=(0, 1),
        )

    def _gpu_devices_panel(self, row: JobUsage, layout: str) -> Panel:
        full = layout == "wide"
        devices = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1), header_style=f"bold {CYAN}")
        devices.add_column("GPU", width=4)
        if full:
            devices.add_column("MODEL / UUID", ratio=3, overflow="ellipsis", no_wrap=True)
        devices.add_column("VRAM", width=15, justify="right")
        devices.add_column("UTIL", width=7, justify="right")
        devices.add_column("TEMP", width=7, justify="right")
        if full:
            devices.add_column("POWER", width=9, justify="right")
            devices.add_column("ECC", width=7, justify="right")
            devices.add_column("DRIVER", width=10)
        limit = 2 if layout == "collapsed" else 4
        for device in row.gpu_devices[:limit]:
            memory = (
                "—" if device.memory_used_gib is None or device.memory_total_gib is None
                else f"{device.memory_used_gib:.1f}/{device.memory_total_gib:.1f}G"
            )
            cells = [Text(str(device.index), style=WHITE)]
            if full:
                identity = device.name if self.size.width < 150 else f"{device.name} · {device.uuid}"
                cells.append(Text(identity, style=WHITE, no_wrap=True, overflow="ellipsis"))
            cells.extend([
                Text(memory, style=WHITE),
                Text(self._device_value(device.utilization, "%"), style=_metric_color(device.utilization)),
                Text(self._device_value(device.temperature_c, "°C"), style=WHITE),
            ])
            if full:
                cells.extend([
                    Text(self._device_value(device.power_w, "W"), style=WHITE),
                    Text("—" if device.ecc_errors is None else str(device.ecc_errors), style=WHITE),
                    Text(device.driver_version, style=WHITE),
                ])
            devices.add_row(*cells)
        if not row.gpu_devices:
            span = 8 if full else 4
            devices.add_row(*(["—", "GPU device metrics unavailable"] + [""] * (span - 2)))
        elif len(row.gpu_devices) > limit:
            span = 8 if full else 4
            devices.add_row(*(["…", f"{len(row.gpu_devices) - limit} more devices"] + [""] * (span - 2)))
        return Panel(
            devices, title=Text(f" GPU DEVICES ({len(row.gpu_devices)}) ", style=f"bold {CYAN}"),
            border_style=CYAN_2, box=box.SQUARE, padding=(0, 0),
        )

    def _render_expanded_resources(self, target: DashboardPane, row: JobUsage, points: List[MetricPoint]) -> None:
        layout = self._resource_layout()
        metrics = self._resource_metrics(row, points)
        content = Table.grid(expand=True)
        content.add_column(ratio=1)
        content.add_row(self._selected_resource_strip(row, layout))
        if layout == "wide":
            for metric in metrics:
                content.add_row(self._wide_metric_panel(metric))
        elif layout == "compact":
            content.add_row(self._compact_metrics_grid(metrics))
        else:
            content.add_row(self._compact_metrics_grid(metrics, collapsed=True))
        content.add_row(self._gpu_devices_panel(row, layout))
        target.border_subtitle = ""
        target.update(content)

    def _render_resources(self) -> None:
        target = self.query_one("#resources-pane", DashboardPane)
        row = self._selected_row()
        target.border_title = f" RESOURCE USAGE{(' — ' + _truncate(row.job, 42)) if row else ''} "
        if not row:
            target.update(Align.center("Resource metrics unavailable for this Job.", vertical="middle"))
            return
        points = self._history_slice(row.uid)
        if not points:
            target.update(Align.center("Waiting for samples", vertical="middle"))
            return
        last = points[-1]
        if row.status in {"Succeeded", "Failed"}:
            elapsed = max(0, int(time.time() - last.timestamp))
            age = f"{elapsed}s" if elapsed < 120 else (f"{elapsed // 60}m" if elapsed < 7200 else f"{elapsed // 3600}h")
            target.border_subtitle = f" Final sample · {age} ago "
        elif self._stale and self.state.last_successful_refresh:
            elapsed = max(0, int(time.time() - self.state.last_successful_refresh))
            target.border_subtitle = f" Metrics stale · last updated {elapsed}s ago "
        else:
            target.border_subtitle = ""
        if self.state.expanded_pane == "resources":
            self._render_expanded_resources(target, row, points)
            return
        cells = [
            self._metric_cell("GPU", last.gpu, [p.gpu for p in points],
                              "—" if row.gpu_count == 0 else f"{(last.gpu or 0) * row.gpu_count / 100:.2f} / {row.gpu_count:.2f} GPU"),
            self._metric_cell("VRAM", last.vram, [p.vram for p in points],
                              f"{_short_memory(row.gpu_memory_used_gib)} / {_short_memory(row.gpu_memory_total_gib)}"),
            self._metric_cell("CPU", last.cpu, [p.cpu for p in points],
                              f"{_short_cpu(row.cpu_used)} / {_short_cpu(row.cpu_requested)} vCPU"),
            self._metric_cell("RAM", last.ram, [p.ram for p in points],
                              f"{_short_memory(row.memory_used_gib)} / {_short_memory(row.memory_requested_gib)}"),
        ]
        table = Table(box=None, expand=True, padding=(0, 1), show_header=False)
        columns = 2 if self.size.width < 90 else 4
        for _ in range(columns):
            table.add_column(ratio=1)
        if columns == 4:
            table.add_row(*cells)
        else:
            table.add_row(*cells[:2])
            table.add_row(*cells[2:])
        target.update(table)

    def _filtered_events(self) -> List[JobEvent]:
        query = self.event_search.lower().strip()
        if not query:
            return self.job_events
        return [event for event in self.job_events if query in " ".join(
            [event.event_type, event.reason, event.message, event.object_name]
        ).lower()]

    def _visible_event_count(self) -> int:
        # Textual's Widget.size is already the content box (the border is not
        # included), and the compact event table has no header. Every content
        # line can therefore hold an event.
        return max(1, self.query_one("#events-pane").size.height)

    def _render_events(self) -> None:
        target = self.query_one("#events-pane", DashboardPane)
        events_list = self._filtered_events()
        if not events_list:
            target.update(Align.center("No events found for the selected Job.", vertical="middle"))
            return
        visible = self._visible_event_count()
        if self.state.events_auto_follow:
            self.state.events_scroll_offset = max(0, len(events_list) - visible)
        start = max(0, min(self.state.events_scroll_offset, max(0, len(events_list) - visible)))
        table = Table(box=None, expand=True, padding=(0, 1), show_header=False)
        table.add_column("TIME", width=9, style=GRAY)
        table.add_column("TYPE", width=9)
        table.add_column("REASON", width=20)
        if self.state.expanded_pane == "events":
            table.add_column("OBJECT", width=24, style=GRAY)
            table.add_column("COUNT", width=7, justify="right")
        table.add_column("MESSAGE", ratio=4, overflow="fold" if self.state.expanded_pane == "events" else "ellipsis")
        for event in events_list[start:start + visible]:
            stamp = datetime.fromtimestamp(_timestamp(event.timestamp)).strftime("%H:%M:%S") if _timestamp(event.timestamp) else "—"
            color = _event_style(event)
            reason = f"{event.reason} ×{event.count}" if event.count > 1 else event.reason
            cells = [Text(stamp, style=GRAY), Text(event.event_type, style=color), Text(reason, style=color)]
            if self.state.expanded_pane == "events":
                cells.extend([Text(event.object_name, style=GRAY), Text(str(event.count), style=GRAY)])
            cells.append(Text(event.message, style=WHITE, no_wrap=self.state.expanded_pane != "events", overflow="ellipsis"))
            table.add_row(*cells)
        newer = len(events_list) - (start + visible)
        target.border_subtitle = f" {newer} newer events ↓ " if newer > 0 and not self.state.events_auto_follow else ""
        target.update(table)

    def _render_footer(self) -> None:
        if self.size.width < MINIMUM_WIDTH or self.size.height < MINIMUM_HEIGHT:
            self.query_one("#falcon-footer", Static).update(Text("q Quit   r Retry after resizing", style=GRAY))
            return
        marked = len(self.state.marked_job_uids)
        if self.state.focused_pane == "jobs":
            value = "↑/↓ Navigate   s Sort   Space Mark   f Filters   k/F9 Kill   c Clean   Enter Expand   Tab Next pane   / Search   r Refresh   q Quit"
            if marked:
                value += f"      {marked} marked"
        elif self.state.focused_pane == "selected":
            value = (
                "↑/↓ Command   PgUp/PgDn Page   Home/End   Tab Next pane   Esc Restore   r Refresh   q Quit"
                if self.state.expanded_pane == "selected"
                else "↑/↓ Change Job   Enter Expand   Tab Next pane   r Refresh   q Quit"
            )
        elif self.state.focused_pane == "resources":
            zoom = round(100 / self.state.resource_zoom)
            if self.state.expanded_pane == "resources":
                value = f"←/→ History   R Range   +/- Zoom {zoom}%   Z Cycle   r Refresh   Tab Next pane   Esc Restore   q Quit"
            else:
                value = f"←/→ History   Home/End Range   +/- Zoom {zoom}%   Enter Expand   Tab Next pane   r Refresh   q Quit"
        else:
            value = "↑/↓ Scroll   PgUp/PgDn Page   Home Oldest   End Newest   / Search   Enter Expand   Tab Next pane   q Quit"
        self.query_one("#falcon-footer", Static).update(Text(value, style=GRAY))

    def _render_all(self) -> None:
        if not self.is_mounted:
            return
        self._apply_layout()
        self._set_titles()
        self._render_header()
        self._render_summary()
        self._render_controls()
        self._render_jobs()
        self._render_selected()
        self._render_resources()
        self._render_events()
        self._render_footer()

    def _apply_layout(self) -> None:
        if not self.is_mounted:
            return
        pane_ids = ["jobs-pane", "selected-pane", "resources-pane", "events-pane", "summary", "controls"]
        resize = self.query_one("#resize-message", Static)
        if self.size.width < MINIMUM_WIDTH or self.size.height < MINIMUM_HEIGHT:
            for pane_id in pane_ids:
                self.query_one(f"#{pane_id}").display = False
            resize.display = True
            resize.update(
                f"Falcon Dashboard requires at least {MINIMUM_WIDTH}×{MINIMUM_HEIGHT}.\n"
                f"Current terminal: {self.size.width}×{self.size.height}.\n\nResize the terminal to inspect and manage Jobs."
            )
            return
        resize.display = False
        if self.state.expanded_pane:
            active = self.state.expanded_pane + "-pane"
            for pane_id in pane_ids:
                self.query_one(f"#{pane_id}").display = pane_id == active
            self.query_one(f"#{active}").styles.height = "1fr"
            return
        for pane_id in pane_ids:
            self.query_one(f"#{pane_id}").display = True
        self.query_one("#jobs-pane").styles.height = "1fr"
        self.query_one("#summary").styles.height = 2
        self.query_one("#selected-pane").styles.height = 3
        self.query_one("#resources-pane").styles.height = 6
        self.query_one("#events-pane").styles.height = 7

    def _focus(self, pane: str) -> None:
        self.state.focused_pane = pane
        self.query_one(f"#{pane}-pane", DashboardPane).focus()
        self._set_titles()
        self._render_footer()

    def action_next_pane(self) -> None:
        panes = ["jobs", "selected", "resources", "events"]
        self._cycle_pane(1, panes)

    def action_previous_pane(self) -> None:
        panes = ["jobs", "selected", "resources", "events"]
        self._cycle_pane(-1, panes)

    def _cycle_pane(self, amount: int, panes: Optional[List[str]] = None) -> None:
        panes = panes or ["jobs", "selected", "resources", "events"]
        pane = panes[(panes.index(self.state.focused_pane) + amount) % len(panes)]
        if self.state.expanded_pane:
            self.state.focused_pane = pane
            self.state.expanded_pane = pane
            self._apply_layout()
            self._focus(pane)
            self._render_all()
        else:
            self._focus(pane)

    def action_focus_jobs(self) -> None: self._focus("jobs")
    def action_focus_selected(self) -> None: self._focus("selected")
    def action_focus_resources(self) -> None: self._focus("resources")
    def action_focus_events(self) -> None: self._focus("events")

    def action_expand(self) -> None:
        self.state.expanded_pane = self.state.focused_pane
        self._apply_layout()
        self._render_all()

    def action_toggle_expand(self) -> None:
        self.state.expanded_pane = None if self.state.expanded_pane else self.state.focused_pane
        self._apply_layout()
        self._render_all()

    def action_escape(self) -> None:
        # Application bindings remain active while a ModalScreen is mounted in
        # the Textual version supported by Falcon.  Handle Escape here as well
        # as on each dialog so it can never leak through to the dashboard.
        if isinstance(self.screen, (FilterDialog, KillDialog, CleanupDialog)):
            self.screen.dismiss(None)
            return
        search = self.query_one("#search-input", Input)
        if search.display:
            search.display = False
            self.query_one("#controls").display = True
            self._focus(self.state.focused_pane)
        elif self.state.expanded_pane:
            self.state.expanded_pane = None
            self._apply_layout()
            self._render_all()
        elif self.state.search_query:
            self.state.search_query = ""
            self._filter_rows()
            self._render_all()
        else:
            self._focus("jobs")

    def _move_cursor(self, amount: int) -> None:
        if self.filtered_rows:
            self.selected = self.selected + amount
            self._ensure_cursor_visible()
            self._selection_changed()

    def _selection_changed(self) -> None:
        self.state.events_scroll_offset = 0
        self.state.events_auto_follow = True
        self.state.resource_scroll_offset = 0
        self.state.selected_command_scroll_offset = 0
        self._request_update()
        self._render_all()

    def action_up(self) -> None:
        if self.state.focused_pane == "selected" and self.state.expanded_pane == "selected":
            self._scroll_selected_command(-1)
            return
        if self.state.focused_pane in {"jobs", "selected"}: self._move_cursor(-1)
        elif self.state.focused_pane == "events": self._scroll_events(-1)
        else: self._scroll_history(1)

    def action_down(self) -> None:
        if self.state.focused_pane == "selected" and self.state.expanded_pane == "selected":
            self._scroll_selected_command(1)
            return
        if self.state.focused_pane in {"jobs", "selected"}: self._move_cursor(1)
        elif self.state.focused_pane == "events": self._scroll_events(1)
        else: self._scroll_history(-1)

    def action_kill_or_up(self) -> None:
        if self.state.focused_pane == "jobs": self.action_kill()
        else: self.action_up()

    def action_left(self) -> None:
        if self.state.focused_pane == "resources":
            self._scroll_history(1)

    def action_right(self) -> None:
        if self.state.focused_pane == "resources":
            self._scroll_history(-1)

    def _scroll_events(self, amount: int) -> None:
        events_list = self._filtered_events()
        visible = self._visible_event_count()
        self.state.events_auto_follow = False
        self.state.events_scroll_offset = max(0, min(
            max(0, len(events_list) - visible), self.state.events_scroll_offset + amount
        ))
        self._render_events()

    def _scroll_history(self, amount: int) -> None:
        row = self._selected_row()
        maximum = max(0, len(self.histories.get(row.uid, [])) - 1) if row else 0
        self.state.resource_scroll_offset = max(0, min(maximum, self.state.resource_scroll_offset + amount))
        self._render_resources()

    def action_page_up(self) -> None:
        if self.state.focused_pane == "selected" and self.state.expanded_pane == "selected":
            self._scroll_selected_command(-self._command_view_height())
        elif self.state.focused_pane == "resources":
            self.state.resource_zoom = min(16, self.state.resource_zoom * 2)
            self._render_resources()
            self._render_footer()
        else:
            self._scroll_events(-self._visible_event_count())

    def action_page_down(self) -> None:
        if self.state.focused_pane == "selected" and self.state.expanded_pane == "selected":
            self._scroll_selected_command(self._command_view_height())
        elif self.state.focused_pane == "resources":
            self.state.resource_zoom = max(1, self.state.resource_zoom // 2)
            self._render_resources()
            self._render_footer()
        else:
            self._scroll_events(self._visible_event_count())

    def action_home(self) -> None:
        if self.state.focused_pane == "selected" and self.state.expanded_pane == "selected":
            self.state.selected_command_scroll_offset = 0
            self._render_selected()
        elif self.state.focused_pane == "events":
            self.state.events_auto_follow = False
            self.state.events_scroll_offset = 0
            self._render_events()
        elif self.state.focused_pane == "resources":
            row = self._selected_row()
            self.state.resource_scroll_offset = max(0, len(self.histories.get(row.uid, [])) - 1) if row else 0
            self._render_resources()

    def action_end(self) -> None:
        if self.state.focused_pane == "selected" and self.state.expanded_pane == "selected":
            row = self._selected_row()
            if row:
                self.state.selected_command_scroll_offset = max(
                    0, len(self._wrapped_command(row)) - self._command_view_height()
                )
            self._render_selected()
        elif self.state.focused_pane == "events":
            self.state.events_auto_follow = True
            self._render_events()
        elif self.state.focused_pane == "resources":
            self.state.resource_scroll_offset = 0
            self._render_resources()

    def action_history_left(self) -> None:
        if self.state.focused_pane == "resources": self._scroll_history(1)
    def action_history_right(self) -> None:
        if self.state.focused_pane == "resources": self._scroll_history(-1)

    def action_cycle_resource_range(self) -> None:
        if self.state.focused_pane != "resources":
            return
        ranges = [60, 300, 600]
        current = self.state.resource_range_samples
        self.state.resource_range_samples = ranges[(ranges.index(current) + 1) % len(ranges)] if current in ranges else 60
        self.state.resource_scroll_offset = 0
        self._render_resources()

    def action_cycle_resource_zoom(self) -> None:
        if self.state.focused_pane != "resources":
            return
        zooms = [1, 2, 4, 8, 16]
        current = self.state.resource_zoom
        self.state.resource_zoom = zooms[(zooms.index(current) + 1) % len(zooms)] if current in zooms else 1
        self._render_resources()
        self._render_footer()

    def action_resource_zoom_in(self) -> None:
        if self.state.focused_pane != "resources":
            return
        self.state.resource_zoom = max(1, self.state.resource_zoom // 2)
        self._render_resources()
        self._render_footer()

    def action_resource_zoom_out(self) -> None:
        if self.state.focused_pane != "resources":
            return
        self.state.resource_zoom = min(16, self.state.resource_zoom * 2)
        self._render_resources()
        self._render_footer()

    def action_toggle_mark(self) -> None:
        row = self._selected_row()
        if not row: return
        if row.uid in self.state.marked_job_uids: self.state.marked_job_uids.remove(row.uid)
        else: self.state.marked_job_uids.add(row.uid)
        self._filter_rows()
        self._render_all()

    def action_mark_next(self) -> None:
        self.action_toggle_mark()
        self._move_cursor(1)

    def action_mark_all(self) -> None:
        self.state.marked_job_uids.update(row.uid for row in self.filtered_rows)
        self._render_all()

    def action_clear_marks(self) -> None:
        self.state.marked_job_uids.clear()
        self._filter_rows()
        self._render_all()

    def action_marked_only(self) -> None:
        self.state.filters["marked"] = "All" if self.state.filters["marked"] == "Marked" else "Marked"
        self._filter_rows()
        self._render_all()

    def action_search(self) -> None:
        control = self.query_one("#search-input", Input)
        control.placeholder = "Search events…" if self.state.focused_pane == "events" else "Search jobs…"
        control.value = self.event_search if self.state.focused_pane == "events" else self.state.search_query
        self.query_one("#controls").display = False
        control.display = True
        control.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "search-input": return
        if "events" in event.input.placeholder.lower(): self.event_search = event.value.strip()
        else: self.state.search_query = event.value.strip()
        event.input.display = False
        self.query_one("#controls").display = True
        self._filter_rows()
        self._focus(self.state.focused_pane)
        self._render_all()

    def action_cycle_status(self) -> None:
        values = ["All", "Running", "Pending", "Succeeded", "Failed", "Suspended"]
        current = self.state.filters["status"]
        self.state.filters["status"] = values[(values.index(current) + 1) % len(values)]
        self._filter_rows()
        self._render_all()

    def action_filters(self) -> None:
        self.push_screen(FilterDialog(self.state.filters, self.rows), self._filters_applied)

    def _filters_applied(self, filters: Optional[Dict[str, str]]) -> None:
        if filters is None:
            return
        self.state.filters = filters
        self._filter_rows()
        self._render_all()

    def action_cycle_sort(self) -> None:
        values = ["Age", "Name", "Status"]
        current = self.state.sort_field
        self.state.sort_field = values[(values.index(current) + 1) % len(values)]
        self._filter_rows()
        self._render_all()

    def action_update_data(self) -> None:
        invalidate = getattr(self.collector, "invalidate", None)
        if invalidate: invalidate()
        self._request_update()

    def action_help(self) -> None:
        self.notify("Tab panes · 1/2/3 focus · Space mark · k/F9 delete · c clean succeeded · / search · z expand · r refresh · q quit", timeout=8)

    def action_cleanup(self) -> None:
        targets = [row for row in self.rows if row.status == "Succeeded"]
        if not targets:
            self.notify("No succeeded Jobs to clean")
            return
        self.push_screen(CleanupDialog(targets), lambda confirmed: self._cleanup_confirmed(confirmed, targets))

    def _cleanup_confirmed(self, confirmed: bool, targets: List[JobUsage]) -> None:
        if confirmed:
            self._delete_confirmed(("job", targets))

    def action_kill(self) -> None:
        selected = self._selected_row()
        targets = [row for row in self.rows if row.uid in self.state.marked_job_uids]
        if not targets and selected: targets = [selected]
        if not targets: return
        self.state.kill_dialog.update({"isOpen": True, "targets": [row.uid for row in targets]})
        self.push_screen(KillDialog(targets), self._delete_confirmed)

    def _delete_confirmed(self, result: Optional[Tuple[str, List[JobUsage]]]) -> None:
        self.state.kill_dialog["isOpen"] = False
        if not result: return
        action, rows = result

        def delete_targets() -> None:
            succeeded = 0
            for row in rows:
                if action == "pod":
                    if not row.active_pod: continue
                    command = [
                        "kubectl", "delete", "pod", row.active_pod, "--wait=false",
                        "--namespace", self.collector.namespace,
                    ]
                else:
                    command = [
                        "kubectl", "delete", "job", row.job, "--wait=false",
                        "--namespace", self.collector.namespace,
                    ]
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
                    succeeded += int(result.returncode == 0)
                except (OSError, subprocess.SubprocessError):
                    pass
            def finish() -> None:
                if succeeded == len(rows): self.notify(f"Deleted {succeeded} Job{'s' if succeeded != 1 else ''}")
                else: self.notify(f"Deleted {succeeded} of {len(rows)} Jobs · {len(rows) - succeeded} failed", severity="error")
                invalidate = getattr(self.collector, "invalidate", None)
                if invalidate: invalidate()
                self._request_update()
            self.call_from_thread(finish)

        threading.Thread(target=delete_targets, name="falcon-delete", daemon=True).start()
