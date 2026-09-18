"""Keyboard-first nvitop/htop-inspired Textual interface for Falcon Jobs."""

from __future__ import annotations

import copy
import json
import shlex
import subprocess
import textwrap
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Set, Tuple

from rich import box
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.geometry import NULL_OFFSET, Region, Size
from textual.layout import ArrangeResult, Layout, WidgetPlacement
from textual.layouts.vertical import VerticalLayout
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from .dashboard import (
    KUBERNETES_USAGE_SECONDS,
    JobEvent,
    JobUsage,
    PodAttempt,
    _metric_color,
    _percent,
    _short_cpu,
    _short_memory,
    _timestamp,
)
from .dashboard_logs import DashboardLogManager, LogSnapshot
from .planning import GPU_MODEL_DISPLAY_ORDER, canonical_gpu
from .theme import (
    BACKGROUND,
    BORDER,
    CYAN,
    CYAN_2,
    GRAY,
    GREEN,
    MINIMUM_HEIGHT,
    MINIMUM_WIDTH,
    MUTED,
    PALETTE,
    RED,
    WHITE,
    YELLOW,
    configure_color,
)


def _restart_job_manifest(job: Dict, new_name: str, namespace: str) -> Dict:
    """Create a portable manifest for rerunning an existing Kubernetes Job."""
    metadata = job.get("metadata", {})
    labels = copy.deepcopy(metadata.get("labels") or {})
    annotations = copy.deepcopy(metadata.get("annotations") or {})
    controller_labels = {
        "controller-uid", "batch.kubernetes.io/controller-uid",
        "job-name", "batch.kubernetes.io/job-name",
    }
    for key in controller_labels:
        labels.pop(key, None)
    annotations.pop("kubectl.kubernetes.io/last-applied-configuration", None)

    spec = copy.deepcopy(job.get("spec") or {})
    spec.pop("selector", None)
    spec.pop("manualSelector", None)
    template_metadata = spec.setdefault("template", {}).setdefault("metadata", {})
    template_labels = copy.deepcopy(template_metadata.get("labels") or {})
    template_annotations = copy.deepcopy(template_metadata.get("annotations") or {})
    for key in controller_labels:
        template_labels.pop(key, None)
    template_metadata.clear()
    if template_labels:
        template_metadata["labels"] = template_labels
    if template_annotations:
        template_metadata["annotations"] = template_annotations

    new_metadata = {"name": new_name, "namespace": namespace}
    if labels:
        new_metadata["labels"] = labels
    if annotations:
        new_metadata["annotations"] = annotations
    return {
        "apiVersion": job.get("apiVersion", "batch/v1"),
        "kind": "Job",
        "metadata": new_metadata,
        "spec": spec,
    }


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
    hidden_panes: Set[str] = field(default_factory=set)
    selected_section: str = "logs"
    selected_attempt_index: int = -1
    logs_collapsed: bool = False
    # Keep the live log viewport pinned to the newest output until the user
    # deliberately scrolls up.  This is separate from the selected section so
    # refreshing the selected Job preserves tail-follow behaviour.
    logs_auto_follow: bool = True


@dataclass
class MetricPoint:
    timestamp: float
    gpu: Optional[float]
    vram: Optional[float]
    cpu: Optional[float]
    ram: Optional[float]
    # Retain the denominator used for each percentage. Live Job fields drop
    # to zero when a Pod exits, but history must continue to show the absolute
    # value that belonged to the selected sample.
    gpu_capacity: float = 0.0
    vram_capacity: float = 0.0
    cpu_capacity: float = 0.0
    ram_capacity: float = 0.0


def _truncate(value: str, width: int) -> str:
    if width <= 0:
        return ""
    return value if len(value) <= width else value[: max(1, width - 1)] + "…"


MAX_DISPLAY_LOG_LINE_CHARS = 4096

WIDE_LAYOUT_MIN_WIDTH = 160
WIDE_LAYOUT_MIN_HEIGHT = 30
RESOURCE_PANE_HEIGHT = 6

# Dashboard Jobs columns are intentionally fixed. These widths cover the
# longest labels Falcon renders for each field while leaving the flexible NAME
# column to absorb the remaining pane width.
# Rich applies the shared one-cell horizontal padding outside the declared
# width. Reserve those cells so the four-character header and the ``>[ ]``
# selected/mark indicator are both rendered in full.
JOBS_MARK_WIDTH = 6
JOBS_STATUS_WIDTH = 15  # ``● Eviction risk``
JOBS_ACTIVE_POD_WIDTH = 13  # ``No active pod``
JOBS_NODE_WIDTH = 12
JOBS_GPU_WIDTH = 9  # ``pro6000x2``
JOBS_RESTARTS_WIDTH = 8
JOBS_COMPLETIONS_WIDTH = 11
JOBS_AGE_WIDTH = 4


def _display_log_line(value: str) -> str:
    """Keep pathological single-line output from dominating Rich layout."""

    return _truncate(value, MAX_DISPLAY_LOG_LINE_CHARS)


class DashboardBodyLayout(Layout):
    """Arrange Dashboard panes as a stack or two independent columns."""

    name = "falcon-dashboard-body"

    def __init__(self) -> None:
        self._vertical = VerticalLayout()

    def arrange(
        self,
        parent,
        children,
        size: Size,
        greedy: bool = True,
    ) -> ArrangeResult:
        if not getattr(parent.app, "_wide_layout", False):
            return self._vertical.arrange(parent, children, size, greedy)

        parent.pre_layout(self)
        by_id = {child.id: child for child in children if child.id}
        if not by_id or size.width <= 0 or size.height <= 0:
            return []

        left_ids = [pane_id for pane_id in ("jobs-pane", "events-pane") if pane_id in by_id]
        right_ids = [pane_id for pane_id in ("selected-pane", "resources-pane") if pane_id in by_id]
        if left_ids and right_ids:
            left_width = size.width // 2
            right_x = left_width
            right_width = size.width - left_width
        else:
            left_width = size.width if left_ids else 0
            right_x = 0 if not left_ids else left_width
            right_width = size.width if right_ids else 0

        placements: list[WidgetPlacement] = []

        def place(pane_id: str, x: int, y: int, width: int, height: int) -> None:
            if width <= 0 or height <= 0:
                return
            widget = by_id[pane_id]
            styles = widget.styles
            placements.append(
                WidgetPlacement(
                    Region(x, y, width, height),
                    NULL_OFFSET,
                    styles.margin,
                    widget,
                    len(placements),
                    False,
                    styles.overlay == "screen",
                    styles.position == "absolute",
                )
            )

        if left_ids:
            if len(left_ids) == 2:
                jobs_height = (size.height + 1) // 2
                place("jobs-pane", 0, 0, left_width, jobs_height)
                place(
                    "events-pane",
                    0,
                    jobs_height,
                    left_width,
                    size.height - jobs_height,
                )
            else:
                place(left_ids[0], 0, 0, left_width, size.height)

        if right_ids:
            if len(right_ids) == 2:
                resource_height = min(RESOURCE_PANE_HEIGHT, size.height)
                selected_height = size.height - resource_height
                place("selected-pane", right_x, 0, right_width, selected_height)
                place(
                    "resources-pane",
                    right_x,
                    selected_height,
                    right_width,
                    resource_height,
                )
            else:
                place(right_ids[0], right_x, 0, right_width, size.height)

        return placements


class DashboardBody(Container):
    """Container whose layout switches between stack and split modes."""

    def __init__(self, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self._default_layout = DashboardBodyLayout()

    @property
    def layout(self) -> Layout:
        # Container's default CSS requests a vertical layout. This body owns
        # the responsive layout, so always return the dedicated dispatcher.
        return self._default_layout


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
    color: Optional[str] = None,
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
                text.append("█", style=color or _metric_color(numeric))
            elif level == full and fraction > 0:
                index = min(6, max(0, round(fraction * 7) - 1))
                text.append(partials[index], style=color or _metric_color(numeric))
            else:
                text.append(" ")
        if level:
            text.append("\n")
    return text


class _ResourceHistoryChart:
    """Mutable chart cell used by the expanded resource inspector.

    The surrounding metric panel is deliberately kept stable while the user
    browses history. Only this renderable's values change on each horizontal
    history movement.
    """

    def __init__(
        self,
        values: List[Optional[float]],
        width: int,
        height: int,
        zoom: int,
        sample_period: int,
        color: Optional[str],
    ) -> None:
        self.values = values
        self.width = width
        self.height = height
        self.zoom = zoom
        self.sample_period = sample_period
        self.color = color

    def update(
        self,
        values: List[Optional[float]],
        zoom: int,
        color: Optional[str],
    ) -> None:
        self.values = values
        self.zoom = zoom
        self.color = color

    def __rich_console__(self, console, options):
        yield _scaled_history(
            self.values,
            self.width,
            self.height,
            self.zoom,
            self.sample_period,
            self.color,
        )


def _status_style(status: str) -> Tuple[str, str]:
    lowered = status.lower()
    if lowered == "running":
        return ("●", PALETTE.accent_soft)
    if lowered == "succeeded":
        return ("✓", PALETTE.success)
    if lowered in {"failed", "unknown"}:
        return ("✕", PALETTE.danger)
    if lowered in {"queued", "pending"}:
        return ("●", PALETTE.text)
    return ("●", PALETTE.warning)


def _job_status_display(row: JobUsage) -> Tuple[str, str, str]:
    """Return icon, label, and color for the Jobs-table status cell."""
    if row.at_risk:
        return "●", "Eviction risk", PALETTE.warning
    icon, color = _status_style(row.status)
    return icon, row.status, color


def _gpu_display(gpu_type: str, count: int) -> str:
    if count <= 0:
        return "-"
    normalized = gpu_type.strip().replace(" ", "").lower()
    return f"{normalized}x{count}"


def _event_style(event: JobEvent) -> str:
    failure = {"failed", "backoff", "unhealthy", "evicted", "oomkilled", "failedscheduling"}
    if event.reason.lower() in failure:
        return RED
    return YELLOW if event.event_type.lower() == "warning" else GREEN


class DashboardPaneContent(Static):
    def __init__(self, content="", *args, **kwargs):
        self._natural_content = content
        self._natural_height_cache: Optional[Tuple[int, int]] = None
        super().__init__(content, *args, **kwargs)

    def update(self, content="", *, layout: bool = True) -> None:
        self._natural_content = content
        if layout:
            self._natural_height_cache = None
        super().update(content, layout=layout)

    def get_content_height(self, container, viewport, width: int) -> int:
        cached = self._natural_height_cache
        if cached is not None and cached[0] == width:
            return cached[1]
        options = self.app.console.options.update(
            width=max(1, width), height=None
        )
        options.max_height = None
        height = len(
            self.app.console.render_lines(
                self._natural_content,
                options,
                pad=True,
            )
        )
        self._natural_height_cache = (width, height)
        return height


class SelectedJobScroll(VerticalScroll):
    """A nested viewport that owns its mouse wheel and keyboard focus."""

    can_focus = True
    # Let FalconDashboard's context-aware bindings (Home/End, page movement,
    # and the Pod arrows) handle keys instead of VerticalScroll's generic
    # bindings, which would bypass the selected-section state.
    BINDINGS = []

    def _activate_section(self) -> None:
        app = self.app
        callback = getattr(app, "selected_section_focused", None)
        if callback:
            callback(self.id or "")

    def _notify_scroll_position(self) -> None:
        """Tell the dashboard whether this viewport is currently at its end."""

        callback = getattr(self.app, "selected_section_scrolled", None)
        if callback:
            callback(self.id or "", self.scroll_y, self.max_scroll_y)

    def on_focus(self, event: events.Focus) -> None:
        if self.screen.focused is self:
            self._activate_section()

    def on_key(self, event: events.Key) -> None:
        """Route navigation through the dashboard's selected-section logic."""

        actions = {
            "up": "action_up", "k": "action_up",
            "down": "action_down", "j": "action_down",
            "left": "action_left", "right": "action_right",
            "pageup": "action_page_up", "pagedown": "action_page_down",
            "home": "action_home", "end": "action_end",
            "c": "action_toggle_selected_or_cleanup",
            "ctrl+c": "action_copy_or_quit",
            "command+c": "action_copy_or_quit",
        }
        action_name = actions.get(event.key)
        if action_name is None:
            return
        event.prevent_default()
        event.stop()
        callback = getattr(self.app, action_name, None)
        if callback:
            callback()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._activate_section()
        self.app.set_focus(self, scroll_visible=False)

    def on_click(self, event: events.Click) -> None:
        # DashboardPane also listens for clicks to focus the outer pane. Stop
        # this event here so clicking Logs leaves the nested pane selected (and
        # its focused border visible).
        self._activate_section()
        self.app.set_focus(self, scroll_visible=False)
        event.stop()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.prevent_default()
        event.stop()
        if getattr(getattr(self.app, "state", None), "focused_pane", None) != "selected":
            self._activate_section()
        self.scroll_relative(
            y=1, animate=False, force=True, immediate=True
        )
        self._notify_scroll_position()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.prevent_default()
        event.stop()
        if getattr(getattr(self.app, "state", None), "focused_pane", None) != "selected":
            self._activate_section()
        self.scroll_relative(
            y=-1, animate=False, force=True, immediate=True
        )
        self._notify_scroll_position()


class SelectedCommandRow(Horizontal):
    """A non-selectable command label with a value-column copy button."""

    # The command itself is metadata, not a second scrollable inspector pane.
    # Keep only the copy button interactive so clicking the label cannot steal
    # focus from the Logs viewport.
    can_focus = False
    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield Static("Command", id="selected-command-label", markup=False)
        copy_button = Button(
            "⧉",
            id="selected-command-copy",
            tooltip="Copy command",
            compact=True,
            flat=True,
        )
        # The icon is an action, not another selectable inspector pane. Keep
        # focus on Logs when it is clicked so the details layout cannot enter
        # a focus-within reflow while a dashboard refresh is rendering.
        copy_button.can_focus = False
        yield copy_button


class SelectedJobInspector(Container):
    """Compact metadata with a command affordance and a log-only lower half."""

    can_focus = True

    def compose(self) -> ComposeResult:
        with Horizontal(id="selected-details"):
            with Vertical(classes="selected-detail-column"):
                yield Static(id="selected-details-left", markup=False)
                yield SelectedCommandRow(id="selected-command-row")
            with Vertical(classes="selected-detail-column"):
                yield Static(id="selected-details-right", markup=False)
        with Horizontal(classes="selected-section-actions"):
            yield Button(
                "⧉",
                id="selected-logs-copy",
                tooltip="Ctrl/Cmd+C",
                compact=True,
                flat=True,
            )
        with SelectedJobScroll(id="selected-logs-scroll"):
            yield DashboardPaneContent(
                "No output yet", id="selected-logs-content", markup=False
            )

    def clear(self, message: str = "No Job selected") -> None:
        self.query_one("#selected-details-left", Static).update(
            Text(message, style=MUTED)
        )
        self.query_one("#selected-details-right", Static).update(
            ""
        )
        self.query_one("#selected-logs-content", DashboardPaneContent).update(
            "No output yet"
        )
        self.query_one("#selected-logs-scroll", SelectedJobScroll).remove_class(
            "collapsed"
        )
        self.query_one("#selected-logs-scroll", SelectedJobScroll).remove_class(
            "selected"
        )
        self.query_one("#selected-logs-scroll", SelectedJobScroll).border_title = " LOGS "

    def update_view(
        self,
        detail_columns,
        logs: LogSnapshot,
        *,
        logs_collapsed: bool,
        attempt_label: str,
    ) -> None:
        left_details, right_details = detail_columns
        self.query_one("#selected-details-left", Static).update(left_details)
        self.query_one("#selected-details-right", Static).update(right_details)
        log_text = (
            "\n".join(_display_log_line(line) for line in logs.lines)
            if logs.lines
            else "No output yet"
        )
        if logs.error:
            log_text += f"\n\n[{logs.error}]"
        self.query_one("#selected-logs-content", DashboardPaneContent).update(
            log_text
        )
        logs_scroll = self.query_one("#selected-logs-scroll", SelectedJobScroll)
        logs_scroll.border_title = f" LOGS · {attempt_label} "
        logs_scroll.set_class(logs_collapsed, "collapsed")

        # ``update`` invalidates the content height, so Textual may not know
        # the new maximum until the next layout pass.  Scroll immediately for
        # an already-laid-out viewport and once more after refresh to ensure a
        # newly appended line can never leave a following subscriber above
        # the tail.
        app = self.app
        if getattr(getattr(app, "state", None), "logs_auto_follow", True):
            logs_scroll.scroll_end(
                animate=False, force=True, immediate=True
            )

            def follow_tail() -> None:
                if not self.is_mounted:
                    return
                if not getattr(getattr(app, "state", None), "logs_auto_follow", True):
                    return
                try:
                    current = self.query_one(
                        "#selected-logs-scroll", SelectedJobScroll
                    )
                except NoMatches:
                    return
                current.scroll_end(
                    animate=False, force=True, immediate=True
                )

            app.call_after_refresh(follow_tail)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        callback = getattr(self.app, "selected_button_pressed", None)
        if callback:
            callback(event.button.id or "")
        event.stop()


class DashboardPane(VerticalScroll):
    can_focus = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pane_content = ""

    def compose(self) -> ComposeResult:
        if self.id == "selected-pane":
            yield DashboardPaneContent(
                self._pane_content,
                id="selected-pane-content",
                classes="dashboard-pane-content",
                markup=False,
            )
            yield SelectedJobInspector(id="selected-inspector")
            return
        yield DashboardPaneContent(
            self._pane_content,
            id=f"{self.id}-content" if self.id else None,
            classes="dashboard-pane-content",
            markup=False,
        )

    def update(self, content="", *, layout: bool = True) -> None:
        self._pane_content = content
        if self.is_mounted:
            self.query_one(".dashboard-pane-content", DashboardPaneContent).update(
                content, layout=layout
            )

    def _activate(self) -> None:
        pane = self.id.replace("-pane", "") if self.id else "jobs"
        callback = getattr(self.app, "pane_focused", None)
        if callback:
            callback(pane)

    def on_focus(self, event: events.Focus) -> None:
        # Focus messages are queued per widget. When a terminal regains focus,
        # Textual may have already queued a Focus for the previously active
        # pane before the mouse-down focuses the pane under the pointer. Do not
        # let that now-stale message overwrite the pane chosen by the click.
        if self.screen.focused is not self:
            return
        self._activate()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        # Mouse-down is the first event terminals send when a click brings the
        # app back into focus. Record it before any queued focus-restoration
        # messages can run.
        self._activate()

    def on_click(self, event: events.Click) -> None:
        self._activate()
        self.app.set_focus(self, scroll_visible=False)
        callback = getattr(self.app, "pane_clicked", None)
        if callback:
            callback(self.id or "", event)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.prevent_default()
        event.stop()
        callback = getattr(self.app, "scroll_focused", None)
        if callback:
            callback(1, self.id, event)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.prevent_default()
        event.stop()
        callback = getattr(self.app, "scroll_focused", None)
        if callback:
            callback(-1, self.id, event)


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
        Binding("up", "previous_action", "Previous action", priority=True),
        Binding("down", "next_action", "Next action", priority=True),
        Binding("y", "advance", "Confirm", show=False, priority=True),
        Binding("shift+y", "advance", "Confirm", show=False, priority=True),
    ]

    def __init__(self, rows: List[JobUsage]):
        super().__init__()
        self.rows = rows
        self.actions = ["job", "restart"]
        self.action = "job"
        self.stage = 0

    def compose(self) -> ComposeResult:
        with Container(id="kill-box"):
            yield Static(id="kill-text")
            yield Input(
                placeholder=f"Type {len(self.rows)} to confirm action",
                id="kill-count", disabled=True,
            )

    def on_mount(self) -> None:
        self._render_dialog()

    def _render_dialog(self) -> None:
        coder_rows = [row for row in self.rows if row.job.startswith("coder-")]
        names = "\n".join(f"  {row.job}" for row in self.rows[:12])
        if len(self.rows) > 12:
            names += f"\n  … and {len(self.rows) - 12} more"
        verb = "Restart" if self.action == "restart" else "Delete"
        prompt = "Enter Confirm    Esc Cancel"
        if self.stage == 1:
            prompt = f"{verb} {len(self.rows)} Jobs? Press y or Enter to confirm."
        elif self.stage == 2:
            prompt = f"Type {len(self.rows)} to confirm {verb.lower()}:"
        text = Text()
        text.append(f"Job Actions\n\nChoose an action for {len(self.rows)} Job{'s' if len(self.rows) != 1 else ''}:\n\n", style=f"bold {RED}")
        text.append(names + "\n\n", style=WHITE)
        text.append("Action\n", style=GRAY)
        action_labels = {
            "job": "Delete Job and managed pods",
            "restart": "Restart Job with the same name",
        }
        for action in self.actions:
            marker = "●" if self.action == action else "○"
            text.append(
                f"  {marker} {action_labels[action]}\n",
                style=CYAN if self.action == action else WHITE,
            )
        if self.action == "restart":
            if coder_rows:
                text.append(
                    "\nCoder-owned Jobs use Coder's native workspace restart.\n",
                    style=YELLOW,
                )
            else:
                text.append("\nThe current Job will be deleted and recreated with the same name.\n", style=YELLOW)
        elif coder_rows:
            text.append(
                "\nCoder-owned Jobs are deleted as workspaces through Coder.\n",
                style=YELLOW,
            )
        text.append("\n" + prompt, style=GRAY)
        self.query_one("#kill-text", Static).update(text)

    def action_toggle_action(self) -> None:
        self.action_next_action()

    def action_next_action(self) -> None:
        index = (self.actions.index(self.action) + 1) % len(self.actions)
        self.action = self.actions[index]
        self._render_dialog()

    def action_previous_action(self) -> None:
        index = (self.actions.index(self.action) - 1) % len(self.actions)
        self.action = self.actions[index]
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

    def __init__(
        self,
        rows: List[JobUsage],
        *,
        marked: bool = False,
        excluded_marked: int = 0,
    ):
        super().__init__()
        self.rows = rows
        self.marked = marked
        self.excluded_marked = excluded_marked

    def compose(self) -> ComposeResult:
        names = "\n".join(f"  {row.job}" for row in self.rows[:16])
        if len(self.rows) > 16:
            names += f"\n  … and {len(self.rows) - 16} more"
        title = "CLEAN MARKED SUCCEEDED JOBS" if self.marked else "CLEAN SUCCEEDED JOBS"
        scope = "marked succeeded" if self.marked else "succeeded"
        text = Text(f"{title}\n\n", style=f"bold {YELLOW}")
        text.append(
            f"Delete {len(self.rows)} {scope} Job{'s' if len(self.rows) != 1 else ''}?\n\n",
            style=WHITE,
        )
        text.append(names, style=GRAY)
        if self.marked and self.excluded_marked:
            text.append(
                f"\n\n{self.excluded_marked} marked running or failed "
                f"Job{'s' if self.excluded_marked != 1 else ''} will not be touched.\n",
                style=GREEN,
            )
        else:
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
        for index, field_name in enumerate(self.fields):
            selected = index == self.index
            text.append("> " if selected else "  ", style=CYAN if selected else GRAY)
            text.append(
                f"{field_name.title():<10}  [ {self.values[field_name]} ]\n",
                style=CYAN if selected else WHITE,
            )
        text.append("\n↑/↓ Select filter   ←/→ or Space Change   Enter Apply   Esc Cancel", style=GRAY)
        self.query_one("#filter-text", Static).update(text)

    def action_up(self) -> None:
        self.index = (self.index - 1) % len(self.fields)
        self._render_dialog()

    def action_down(self) -> None:
        self.index = (self.index + 1) % len(self.fields)
        self._render_dialog()

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


class PaneVisibilityDialog(ModalScreen[Optional[Set[str]]]):
    CSS = f"""
    PaneVisibilityDialog {{ align: center middle; background: #000000; }}
    #panes-box {{ width: 62; height: 14; border: solid {CYAN}; background: #000000; padding: 1 2; }}
    #panes-text {{ background: #000000; color: {WHITE}; }}
    """
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("enter", "apply", "Apply", priority=True),
        Binding("up", "up", "Previous", priority=True),
        Binding("down", "down", "Next", priority=True),
        Binding("space", "toggle", "Toggle", priority=True),
    ]

    def __init__(self, hidden: Set[str]):
        super().__init__()
        self.hidden = set(hidden)
        self.panes = ["selected", "resources", "events"]
        self.index = 0

    def compose(self) -> ComposeResult:
        with Container(id="panes-box"):
            yield Static(id="panes-text")

    def on_mount(self) -> None:
        self._render_dialog()

    def _render_dialog(self) -> None:
        labels = {"selected": "Selected Job", "resources": "Resource Usage", "events": "Events"}
        text = Text("VISIBLE PANES\n", style=f"bold {CYAN}")
        text.append("Jobs is always visible. Toggle optional panes below.\n\n", style=GRAY)
        for index, pane in enumerate(self.panes):
            selected = index == self.index
            marker = "[ ]" if pane in self.hidden else "[x]"
            text.append("> " if selected else "  ", style=CYAN if selected else GRAY)
            text.append(f"{marker} {labels[pane]}\n", style=CYAN if selected else WHITE)
        text.append("\n↑/↓ Select   Space Toggle   Enter Apply   Esc Cancel", style=GRAY)
        self.query_one("#panes-text", Static).update(text)

    def action_up(self) -> None:
        self.index = (self.index - 1) % len(self.panes)
        self._render_dialog()

    def action_down(self) -> None:
        self.index = (self.index + 1) % len(self.panes)
        self._render_dialog()

    def action_toggle(self) -> None:
        pane = self.panes[self.index]
        if pane in self.hidden:
            self.hidden.remove(pane)
        else:
            self.hidden.add(pane)
        self._render_dialog()

    def action_apply(self) -> None: self.dismiss(self.hidden)
    def action_cancel(self) -> None: self.dismiss(None)


CSS = f"""
Screen {{ background: {BACKGROUND}; color: {WHITE}; overflow: hidden; }}
Static, Input, Container {{ background: {BACKGROUND}; }}
#falcon-header {{ height: 1; color: {WHITE}; padding: 0 1; }}
#summary {{ height: 2; color: {WHITE}; border-bottom: solid {BORDER}; padding: 0 1; }}
#controls {{ height: 1; color: {GRAY}; padding: 0 1; }}
#search-input {{ height: 1; display: none; border: none; padding: 0 1; color: {WHITE}; }}
#dashboard-body {{ width: 1fr; height: 1fr; }}
DashboardPane {{ border: solid {BORDER}; background: {BACKGROUND}; color: {WHITE}; padding: 0 1; scrollbar-visibility: hidden; }}
DashboardPane:focus {{ border: solid {CYAN}; }}
# Keep the Selected Job frame highlighted while its nested Logs viewport owns
# focus; otherwise clicking inside the inspector makes the parent frame look
# inactive even though it is still the active pane.
#selected-pane:focus-within {{ border: solid {CYAN}; }}
#selected-pane.selected-active {{ border: solid {CYAN}; }}
.dashboard-pane-content {{ width: 1fr; height: auto; }}
#jobs-pane {{ height: 1fr; min-height: 7; }}
#selected-pane {{ height: 3; min-height: 3; }}
#resources-pane {{ height: 6; min-height: 6; }}
#selected-pane, #resources-pane {{ overflow-y: auto; scrollbar-size-vertical: 1; }}
#selected-inspector {{ display: none; width: 1fr; height: 1fr; }}
#selected-details {{ width: 1fr; min-width: 0; height: auto; padding: 0 1; }}
.selected-detail-column {{ width: 50%; max-width: 50%; min-width: 0; height: auto; }}
#selected-details-left, #selected-details-right {{ width: 1fr; min-width: 0; height: auto; }}
.selected-section-actions {{ width: 1fr; height: 1; min-height: 1; padding: 0 1; align: right middle; }}
.selected-section-actions Button {{ width: 3; min-width: 3; height: 1; min-height: 1; padding: 0 1; border: none; color: {CYAN}; background: {BACKGROUND}; }}
.selected-section-actions Button:hover, .selected-section-actions Button:focus {{ color: {WHITE}; background: {BORDER}; }}
#selected-command-row {{ width: 1fr; min-width: 0; height: 1; min-height: 1; padding: 0; align: left middle; }}
#selected-command-label {{ width: 18; min-width: 18; height: 1; min-height: 1; color: {GRAY}; }}
#selected-command-row Button {{ width: 3; min-width: 3; height: 1; min-height: 1; padding: 0 1; border: none; color: {CYAN}; background: {BACKGROUND}; }}
#selected-command-row Button:hover, #selected-command-row Button:focus {{ color: {WHITE}; background: {BORDER}; }}
SelectedJobScroll {{ width: 1fr; border: solid {BORDER}; padding: 0 1; scrollbar-size-vertical: 1; }}
SelectedJobScroll:focus {{ border: solid {CYAN}; }}
SelectedJobScroll.selected {{ border: solid {CYAN}; }}
#selected-logs-scroll {{ width: 1fr; height: 1fr; min-height: 3; border: solid {BORDER}; padding: 0 1; scrollbar-size-vertical: 1; }}
#selected-logs-scroll.collapsed {{ height: 4; min-height: 4; max-height: 4; }}
#events-pane {{ height: 7; min-height: 3; }}
#resize-message {{ display: none; height: 1fr; content-align: center middle; color: {YELLOW}; }}
#falcon-footer {{ height: 1; color: {GRAY}; padding: 0 1; }}
"""


class FalconDashboard(App):
    TITLE = "Falcon Dashboard"
    ENABLE_COMMAND_PALETTE = False
    CSS = CSS
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "copy_or_quit", "Copy or quit", show=False, priority=True),
        Binding("command+c", "copy_or_quit", "Copy or quit", show=False, priority=True),
        Binding("cmd+c", "copy_or_quit", "Copy or quit", show=False, priority=True),
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
        Binding("c", "toggle_selected_or_cleanup", "Collapse / expand", show=False),
        Binding("/", "search", "Search", show=False), Binding("f", "filters", "Filters", show=False),
        Binding("v", "panes", "Visible panes", show=False),
        Binding("s", "cycle_sort", "Sort", show=False), Binding("?", "help", "Help", show=False),
        Binding("R", "cycle_resource_range", "Range", show=False),
        Binding("Z", "cycle_resource_zoom", "Zoom", show=False),
        Binding("plus", "resource_zoom_in", "Zoom in", show=False),
        Binding("minus", "resource_zoom_out", "Zoom out", show=False),
    ]

    def __init__(
        self, collector, refresh_seconds: float = 1.0,
        hidden_panes: Optional[List[str]] = None,
        sort_field: str = "Age", sort_direction: str = "desc",
        persist_hidden_panes: Optional[Callable[[Set[str]], None]] = None,
        persist_sort: Optional[Callable[[str, str], None]] = None,
        coder_workspace_action: Optional[Callable[[str, str], None]] = None,
        clock: Optional[Callable[[], str]] = None,
        color_mode: Optional[str] = None,
        log_manager: Optional[DashboardLogManager] = None,
        launch_config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self.color_mode = configure_color(self.console, color_mode)
        self.collector = collector
        self.refresh_seconds = refresh_seconds
        self.state = ViewState()
        self.state.hidden_panes = set(hidden_panes or []) & {"selected", "resources", "events"}
        self.state.sort_field = sort_field if sort_field in {"Age", "Name", "Status"} else "Age"
        self.state.sort_direction = sort_direction if sort_direction in {"asc", "desc"} else "desc"
        if self.state.sort_field == "Status":
            self.state.sort_direction = "asc"
        self._persist_hidden_panes = persist_hidden_panes
        self._persist_sort = persist_sort
        self._coder_workspace_action = coder_workspace_action
        self._clock = clock or (lambda: datetime.now().strftime("%H:%M:%S"))
        self.log_manager = log_manager
        self.launch_config: Mapping[str, Any] = launch_config or {}
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
        self._last_terminal_size: Tuple[int, int] = (-1, -1)
        self._responsive_hidden_panes: Set[str] = set()
        self._wide_layout = False
        self._resource_graph_regions: List[Tuple[int, int, int, int]] = []
        self._expanded_resource_charts: Dict[str, _ResourceHistoryChart] = {}
        self._expanded_resource_uid: Optional[str] = None
        self._resource_history_refresh_pending = False
        self._resource_pane_scroll_pending = 0
        self._resource_pane_scroll_callback_pending = False
        self._building_expanded_resource = False
        self._selected_log_view_key: Optional[Tuple[str, str, int]] = None

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
        with DashboardBody(id="dashboard-body"):
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
        self.query_one("#events-pane", DashboardPane).border_title = " EVENTS "
        self.set_focus(
            self.query_one("#jobs-pane", DashboardPane),
            scroll_visible=False,
        )
        self._apply_layout()
        self._request_update()
        self.set_interval(self.refresh_seconds, self._request_update)
        self.set_interval(0.2, self._drain_results)
        if self.log_manager is not None:
            self.set_interval(0.2, self._render_log_updates)
        # Textual delivers terminal Resize to the active Screen rather than
        # reliably bubbling it to App in every supported version.  A cheap
        # size watcher makes live resize deterministic across that range.
        self.set_interval(0.1, self._check_terminal_size)
        self.set_interval(1.0, self._tick_clock)
        self._render_all()

    def on_unmount(self) -> None:
        if self.log_manager is not None:
            self.log_manager.close()
        close = getattr(self.collector, "close", None)
        if close:
            close()

    def on_resize(self, event: events.Resize) -> None:
        # Some terminal backends dispatch Resize before Textual commits the
        # App's new size. Defer layout until the next refresh so minimum-size
        # and responsive-pane decisions use the final dimensions.
        self._last_terminal_size = (-1, -1)
        self.call_after_refresh(self._apply_resized_layout)

    def _apply_resized_layout(self) -> None:
        if not self.is_mounted:
            return
        self._last_terminal_size = (self.size.width, self.size.height)
        try:
            self._apply_layout()
            self._render_all()
        except NoMatches:
            # Resize callbacks may outlive the default screen during teardown.
            return

    def _check_terminal_size(self) -> None:
        current = (self.size.width, self.size.height)
        if current == self._last_terminal_size:
            return
        self._last_terminal_size = current
        try:
            self._apply_layout()
            self._render_all()
        except NoMatches:
            # The compatibility timer may tick after the screen unmounts.
            return

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

    def watch_app_focus(self, focused: bool) -> None:
        """Keep visual focus in sync with terminal-window focus."""

        if not self.is_mounted:
            return
        try:
            self._set_titles()
            self._render_jobs()
        except NoMatches:
            return

    def pane_clicked(self, pane_id: str, event: events.Click) -> None:
        if pane_id != "jobs-pane" or not self.filtered_rows or self.state.expanded_pane:
            return
        pane = self.query_one("#jobs-pane", DashboardPane)
        content = pane.query_one(".dashboard-pane-content", DashboardPaneContent)
        offset = event.get_content_offset(content)
        if offset is None:
            return
        # With ``show_edge=False`` Rich renders the header and its separator
        # directly at the top of the table. Resolve coordinates against the
        # inner content widget so the pane border doesn't introduce an offset.
        row = offset.y - 2
        if row < 0:
            return
        index = self.state.jobs_scroll_offset + row
        if index < len(self.filtered_rows):
            self.selected = index
            if getattr(event, "ctrl", False):
                self.action_toggle_mark()
            self._selection_changed()

    def scroll_focused(
        self,
        amount: int,
        pane_id: Optional[str] = None,
        event: Optional[events.MouseScrollDown | events.MouseScrollUp] = None,
    ) -> None:
        pane = (pane_id or self.state.focused_pane).replace("-pane", "")
        if pane == "jobs":
            if self.state.expanded_pane == "jobs":
                self._scroll_jobs_view(amount)
            else:
                self._move_cursor(amount)
        elif pane == "selected":
            if self.state.expanded_pane == "selected":
                if self.log_manager is None:
                    self._scroll_expanded_pane("selected", amount)
                else:
                    self._scroll_selected_section(amount)
            elif self._selected_inspector_active() and self._selected_logs_focused():
                self._scroll_selected_section(amount)
            else:
                self._move_cursor(amount)
        elif pane == "events":
            self._scroll_events(amount)
        elif self.state.expanded_pane == "resources":
            resource_pane = self.query_one("#resources-pane", DashboardPane)
            content = resource_pane.query_one(
                ".dashboard-pane-content", DashboardPaneContent
            )
            offset = event.get_content_offset(content) if event else None
            over_graph = offset is not None and any(
                left <= offset.x < right and top <= offset.y < bottom
                for left, top, right, bottom in self._resource_graph_regions
            )
            if over_graph:
                self._scroll_history(amount)
            else:
                self._scroll_expanded_pane("resources", amount)
        else:
            self._scroll_history(amount)

    def _scroll_expanded_pane(self, pane: str, amount: int) -> None:
        if pane == "resources" and self.state.expanded_pane == "resources":
            # Mouse wheels can deliver several events before Textual has had
            # a chance to paint the previous scroll. Apply their combined
            # delta in one compositor update instead of forcing a full Rich
            # inspector repaint for every tick.
            self._resource_pane_scroll_pending += amount
            if self._resource_pane_scroll_callback_pending:
                return
            self._resource_pane_scroll_callback_pending = True

            def flush() -> None:
                self._resource_pane_scroll_callback_pending = False
                amount_to_scroll = self._resource_pane_scroll_pending
                self._resource_pane_scroll_pending = 0
                if (
                    not amount_to_scroll
                    or not self.is_mounted
                    or self.state.expanded_pane != "resources"
                ):
                    return
                target = self.query_one("#resources-pane", DashboardPane)
                target.scroll_relative(
                    y=amount_to_scroll,
                    animate=False,
                    force=True,
                    immediate=True,
                )

            self.call_after_refresh(flush)
            return
        target = self.query_one(f"#{pane}-pane", DashboardPane)
        target.scroll_relative(
            y=amount,
            animate=False,
            force=True,
            immediate=True,
        )

    def _page_expanded_pane(self, pane: str, direction: int) -> None:
        target = self.query_one(f"#{pane}-pane", DashboardPane)
        amount = max(1, target.size.height - 1) * direction
        self._scroll_expanded_pane(pane, amount)

    def _jump_expanded_pane(self, pane: str, end: bool) -> None:
        target = self.query_one(f"#{pane}-pane", DashboardPane)
        if end:
            target.scroll_end(animate=False, force=True, immediate=True)
        else:
            target.scroll_home(animate=False, force=True, immediate=True)

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
        if self.state.sort_field == "Status":
            status_order = {
                "running": 0,
                "pending": 1, "queued": 1, "suspended": 1,
                "failed": 2, "unknown": 2,
                "succeeded": 3,
            }
            def status_key(row: JobUsage) -> Tuple[int, float]:
                return (
                    status_order.get(row.status.lower(), 2),
                    -_timestamp(row.created_at),
                )

            key = status_key
            reverse = False
        else:
            reverse = self.state.sort_direction == "desc"
            key = {
                "Name": lambda row: row.job.lower(),
                "Age": lambda row: _timestamp(row.created_at),
            }.get(self.state.sort_field, lambda row: _timestamp(row.created_at))
        self.filtered_rows = sorted(result, key=key, reverse=reverse)
        if self.state.cursor_job_uid not in {row.uid for row in self.filtered_rows}:
            self.state.cursor_job_uid = self.filtered_rows[0].uid if self.filtered_rows else ""
        self._ensure_cursor_visible()

    def _record_history(self) -> None:
        for row in self.rows:
            history = self.histories.setdefault(row.uid, deque(maxlen=600))
            # A terminal Pod has no current allocation to sample.  If this
            # dashboard observed the Job while it was active, keep its last
            # captured live series as the final sample; otherwise do not let
            # a backend's partial/stale terminal values create a GPU-only or
            # CPU-only history.  GPU utilization, VRAM, CPU, and RAM therefore
            # have identical completed-job semantics.
            if row.status in {"Succeeded", "Failed"}:
                continue
            values = (
                row.gpu_util if row.gpu_metrics_available else None,
                row.gpu_memory_percent,
                row.cpu_percent,
                row.memory_percent,
            )
            if not any(value is not None for value in values):
                continue
            history.append(
                MetricPoint(
                    time.time(),
                    *values,
                    gpu_capacity=float(
                        row.gpu_allocated_count or row.gpu_count
                    ),
                    vram_capacity=row.gpu_memory_total_gib,
                    cpu_capacity=row.cpu_allocated or row.cpu_requested,
                    ram_capacity=(
                        row.memory_allocated_gib or row.memory_requested_gib
                    ),
                )
            )
        gpu_weight = sum(
            row.gpu_allocated_count
            for row in self.rows
            if row.gpu_util is not None
        )
        gpu = (
            sum(
                (row.gpu_util or 0) * row.gpu_allocated_count
                for row in self.rows
                if row.gpu_util is not None
            )
            / gpu_weight
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
                job_events = (
                    events_method(selected)
                    if events_method and selected and "events" not in self.state.hidden_panes
                    else []
                )
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
            if self.log_manager is not None:
                self.log_manager.reconcile(self.rows)
            if error:
                self.notify(f"API error: {error} · retrying…", severity="warning")
            if event_uid != self.state.cursor_job_uid:
                self._request_update()
        else:
            self._stale = True
            if error:
                self.notify(f"API error: {error} · retrying…", severity="error")
        try:
            self._render_all()
        except NoMatches:
            # The collector thread may deliver its final result while the
            # Textual screen is being torn down.
            return

    def _set_titles(self) -> None:
        for pane in ("jobs", "selected", "resources", "events"):
            widget = self.query_one(f"#{pane}-pane", DashboardPane)
            focused = self.app_focus and pane == self.state.focused_pane
            base = {
                "jobs": "JOBS", "selected": "SELECTED JOB",
                "resources": "RESOURCE USAGE", "events": "EVENTS",
            }[pane]
            widget.border_title = f" {base}{' · focused' if focused else ''} "

    def _render_header(self) -> None:
        target = self.query_one("#falcon-header", Static)
        clock = self._clock()
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
        width = self.size.width
        state_items = [
            (running, "RUNNING", GREEN), (risk, "RISK", YELLOW),
            (succeeded, "SUCCESS", GREEN), (failed, "FAILED", RED),
            (len(self.rows), "JOBS", WHITE), (nodes, "NODES", WHITE),
        ]
        if width < 100:
            state_items = [state_items[1], state_items[3]]
        elif width < 130:
            state_items = state_items[:4]

        left = Text()
        for value, label, color in state_items:
            if left:
                left.append("   ")
            left.append(f"{value} {label}", style=f"bold {color}")

        right = Text("RESOURCES AVAILABLE  " if width >= 130 else "", style=f"bold {GRAY}")
        for gpu_type, label in GPU_MODEL_DISPLAY_ORDER:
            if right and not right.plain.endswith("  "):
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
        gap = max(2, width - len(left.plain) - len(right.plain) - 4)
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
        try:
            widget = self.query_one("#jobs-pane", DashboardPane)
        except NoMatches:
            # A refresh worker can finish while Textual is tearing down the
            # test/app screen. Keep filtering state updates safe during that
            # short lifecycle window.
            return 4
        # With ``show_edge=False`` SIMPLE_HEAD contributes the header and its
        # separator. Its trailing spacer may be clipped, so retain every Job
        # data row.
        return max(1, widget.content_size.height - 2)

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
        self._render_jobs(layout=False)

    def _render_jobs(self, *, layout: bool = True) -> None:
        target = self.query_one("#jobs-pane", DashboardPane)
        if not self.rows:
            target.update(
                Align.center(
                    "No Jobs found.\nPress f to change filters or r to refresh.",
                    vertical="middle",
                ),
                layout=layout,
            )
            return
        if not self.filtered_rows:
            target.update(
                Align.center(
                    f"No Jobs match “{self.state.search_query}”.\nPress Esc to clear search.",
                    vertical="middle",
                ),
                layout=layout,
            )
            return
        width = max(1, target.content_size.width or self.size.width - 4)
        expanded = self.state.expanded_pane == "jobs"
        gpu_header = "GPUs"
        # Keep these fixed rather than resizing on every refresh or changing
        # widths as Jobs scroll. GPU request values remain visible within the
        # supported Falcon GPU model/count range.
        mark_width = JOBS_MARK_WIDTH
        status_width = JOBS_STATUS_WIDTH
        active_pod_width = JOBS_ACTIVE_POD_WIDTH
        node_width = JOBS_NODE_WIDTH
        gpu_width = JOBS_GPU_WIDTH
        restart_width = JOBS_RESTARTS_WIDTH
        completion_width = JOBS_COMPLETIONS_WIDTH
        age_width = JOBS_AGE_WIDTH
        # GPU requests remain useful even in the half-width Jobs pane. They
        # take precedence over the active Pod and timestamp when the minimum
        # supported Dashboard width cannot hold every identity column.
        show_gpu = width >= 72
        show_active_pod = width >= 72 and not (expanded and width < 95)
        if show_gpu and width < 80:
            show_active_pod = False
        show_node = width >= 95
        show_age = width >= 80
        show_restarts = expanded and width >= 95
        show_completions = expanded and width >= 115
        table = Table(
            box=box.SIMPLE_HEAD,
            expand=True,
            # Use one shared horizontal cell space between every adjacent
            # column, rather than two spaces from both cells' padding.
            padding=(0, 1),
            collapse_padding=True,
            show_edge=False,
            show_header=True,
            header_style=f"bold {CYAN_2}",
        )
        table.add_column("MARK", width=mark_width, no_wrap=True)
        table.add_column(
            "NAME", ratio=3, min_width=16, no_wrap=True, overflow="ellipsis"
        )
        table.add_column("STATUS", width=status_width, no_wrap=True)
        if show_active_pod:
            table.add_column("ACTIVE POD", width=active_pod_width, no_wrap=True)
        if show_node:
            table.add_column("NODE", width=node_width, no_wrap=True)
        if show_gpu:
            table.add_column(
                gpu_header,
                width=gpu_width,
                no_wrap=False,
                overflow="fold",
            )
        if show_restarts:
            table.add_column("RESTARTS", width=restart_width, justify="right")
        if show_completions:
            table.add_column("COMPLETIONS", width=completion_width, justify="right")
        if show_age:
            table.add_column("AGE", width=age_width, justify="right")
        count = self._visible_job_count()
        start = self.state.jobs_scroll_offset
        jobs_focused = self.app_focus and self.state.focused_pane == "jobs"
        for index, row in enumerate(self.filtered_rows[start:start + count], start=start):
            selected = row.uid == self.state.cursor_job_uid
            marked = row.uid in self.state.marked_job_uids
            marker = ">" if selected else " "
            mark = "[x]" if marked else "[ ]"
            icon, status_text, status_color = _job_status_display(row)
            selection_active = selected and jobs_focused
            cells: List[Text] = [Text(f"{marker}{mark}", style=CYAN if selection_active or marked else GRAY)]
            name_style = f"bold {WHITE}" if selected else WHITE
            cells.append(Text(row.job, style=name_style, no_wrap=True, overflow="ellipsis"))
            cells.append(Text(f"{icon} {status_text}", style=status_color))
            if show_active_pod:
                cells.append(Text(row.active_pod_state, style=WHITE, no_wrap=True, overflow="ellipsis"))
            if show_node:
                cells.append(Text(row.nodes, style=WHITE))
            if show_gpu:
                cells.append(Text(_gpu_display(row.gpu_type, row.gpu_count), style=WHITE))
            if show_restarts:
                cells.append(Text(str(row.restarts), style=WHITE))
            if show_completions:
                cells.append(Text(row.completions, style=WHITE))
            if show_age:
                cells.append(Text(row.age, style=WHITE))
            table.add_row(*cells)
        position = f" {min(start + 1, len(self.filtered_rows))}-{min(start + count, len(self.filtered_rows))}/{len(self.filtered_rows)} "
        target.border_subtitle = position if len(self.filtered_rows) > count else ""
        target.update(table, layout=layout)

    def _wrapped_command(self, row: JobUsage) -> List[str]:
        width = max(20, self.size.width - 10)
        lines: List[str] = []
        for source_line in (row.command or "—").splitlines() or ["—"]:
            lines.extend(textwrap.wrap(source_line, width=width, replace_whitespace=False) or [""])
        return lines

    def _selected_attempts(self, row: JobUsage) -> List[PodAttempt]:
        attempts = list(row.attempt_details)
        if attempts:
            return attempts
        if row.active_pod:
            return [
                PodAttempt(
                    name=row.active_pod,
                    uid=row.active_pod_uid or row.active_pod,
                    phase=(
                        "Running"
                        if row.active_pod_state.lower() == "running"
                        else "Unknown"
                    ),
                )
            ]
        return []

    def _selected_attempt(self, row: JobUsage) -> Optional[PodAttempt]:
        attempts = self._selected_attempts(row)
        if not attempts:
            self.state.selected_attempt_index = -1
            return None
        index = self.state.selected_attempt_index
        if index < 0:
            active = [
                attempt_index
                for attempt_index, attempt in enumerate(attempts)
                if not attempt.terminal
            ]
            index = active[-1] if active else len(attempts) - 1
        index = max(0, min(len(attempts) - 1, index))
        self.state.selected_attempt_index = index
        return attempts[index]

    @staticmethod
    def _command_quantity(value: float, suffix: str = "") -> str:
        if float(value).is_integer():
            return f"{int(value)}{suffix}"
        return f"{value:g}{suffix}"

    def _reconstructed_command(self, row: JobUsage) -> str:
        """Build the shortest valid Falcon invocation for a Job.

        The Job inventory contains resolved manifest values, while Falcon's
        CLI intentionally supplies most of those values from configuration or
        planner defaults.  Prefer the public GPU-preset shorthand, omit the
        generated name and default image, and only retain arguments that a
        caller must provide to express this workload.
        """

        command: List[str] = ["falcon"]
        if row.gpu_requested_count > 0 and row.gpu_requested_type not in {"", "-"}:
            model = canonical_gpu(row.gpu_requested_type) or row.gpu_requested_type
            preset = self._gpu_preset_name(model, row.gpu_requested_count)
            if preset:
                token = preset
                if row.gpu_requested_count != 1:
                    token = f"{token}x{row.gpu_requested_count}"
                command.append(token)
            else:
                # An inventory can outlive a changed preset list or count
                # limit.  The explicit form remains valid for custom models
                # and avoids emitting a shorthand that the current config
                # would reject.
                command.extend(["--gpu", row.gpu_requested_type])
                if row.gpu_requested_count != 1:
                    command.extend(["--gpus", str(row.gpu_requested_count)])
        else:
            # CPU-only launches have no planner shorthand; both values are
            # required by the CLI even when their limits default to requests.
            command.extend(
                [
                    "--cpu",
                    self._command_quantity(row.cpu_requested),
                    "--memory",
                    self._command_quantity(row.memory_requested_gib, "Gi"),
                ]
            )

        default_image = self._configured_default_image()
        if row.image and row.image != default_image:
            command.extend(["--image", row.image])
        try:
            command_argv = shlex.split(row.command) if row.command else []
        except ValueError:
            command_argv = [row.command] if row.command else []
        if command_argv:
            command.extend(["--", *command_argv])
        return shlex.join(command)

    def _configured_default_image(self) -> Optional[str]:
        runtime = self.launch_config.get("runtime", {})
        if isinstance(runtime, Mapping):
            image = runtime.get("image")
            return str(image) if image else None
        return None

    def _gpu_preset_name(self, model: str, count: int) -> Optional[str]:
        presets = self.launch_config.get("presets", {})
        if not isinstance(presets, Mapping):
            return None
        wanted = canonical_gpu(model)
        for name, preset in presets.items():
            if not isinstance(preset, Mapping):
                continue
            configured = preset.get("gpu_type", name)
            if canonical_gpu(str(configured)) == wanted:
                try:
                    maximum = int(preset.get("max_count", 8))
                except (TypeError, ValueError):
                    maximum = 8
                if count <= maximum:
                    return str(name)
                return None
        return None

    def selected_section_focused(self, section_id: str) -> None:
        # A nested inspector viewport is still part of the Selected Job pane.
        # Record both levels of focus here.  In particular, after a terminal
        # reconnect Textual may restore the last top-level pane (usually
        # Jobs) without changing the widget under the mouse; relying only on
        # ``screen.focused`` would then leave arrows and the footer routed to
        # the wrong pane even though Logs visibly received the click.
        self.state.focused_pane = "selected"
        if section_id == "selected-logs-scroll":
            self.state.selected_section = "logs"
        self._set_titles()
        self._set_selected_subtitle()
        self._render_footer()

    def selected_section_scrolled(
        self, section_id: str, scroll_y: float, max_scroll_y: float
    ) -> None:
        """Track whether a manually scrolled Logs viewport should follow."""

        if section_id != "selected-logs-scroll":
            return
        self.state.logs_auto_follow = float(scroll_y) >= float(max_scroll_y) - 0.01

    def _selected_attempt_label(self, row: Optional[JobUsage]) -> str:
        if row is None:
            return "No Pod attempt"
        attempts = self._selected_attempts(row)
        attempt = self._selected_attempt(row)
        if attempt is None:
            return "No Pod attempt"
        index = attempts.index(attempt) if attempt in attempts else -1
        return (
            f"Pod {index + 1}/{len(attempts)} · {attempt.name} · {attempt.phase}"
            if index >= 0
            else f"Pod · {attempt.name} · {attempt.phase}"
        )

    def _set_selected_subtitle(self) -> None:
        """Show the active inspector section and selected Pod in its border."""

        try:
            target = self.query_one("#selected-pane", DashboardPane)
        except NoMatches:
            return
        inspector_visible = self._selected_inspector_visible()
        selected_active = (
            inspector_visible
            and self.app_focus
            and self.state.focused_pane == "selected"
        )
        target.set_class(selected_active, "selected-active")
        if not inspector_visible:
            target.border_subtitle = ""
            return
        # Keep the selected section visibly marked even when the terminal
        # itself is unfocused (Textual may clear its :focus pseudo-class while
        # a tmux pane is being reattached).
        for section, selector in (("logs", "#selected-logs-scroll"),):
            try:
                widget = self.query_one(selector)
                widget.set_class(
                    self.state.selected_section == section, "selected"
                )
            except NoMatches:
                pass
        row = self._selected_row()
        section = self.state.selected_section.upper()
        target.border_subtitle = f" {section} · {self._selected_attempt_label(row)} "

    def _focus_selected_section(self) -> None:
        """Give the active Logs viewport the real keyboard focus."""

        if not self._selected_inspector_visible() or self.log_manager is None:
            return
        self.state.focused_pane = "selected"
        target_id = "#selected-logs-scroll"
        try:
            target = self.query_one(target_id)
        except NoMatches:
            return
        self.set_focus(target, scroll_visible=False)
        self._set_selected_subtitle()

    def _selected_inspector_active(self) -> bool:
        return (
            self._selected_inspector_visible()
            and (self.log_manager is not None or self._wide_layout)
        )

    def _selected_inspector_visible(self) -> bool:
        """Return whether Selected Job is showing its detailed inspector."""

        if self._selected_row() is None:
            return False
        if self.state.expanded_pane == "selected":
            return True
        return self._wide_layout and "selected" in self._visible_panes()

    def _selected_logs_focused(self) -> bool:
        """Return whether the nested log viewport currently owns focus."""

        try:
            return self.screen.focused is self.query_one(
                "#selected-logs-scroll", SelectedJobScroll
            )
        except NoMatches:
            return False

    def _copy_selected_content(self, section: Optional[str] = None) -> None:
        """Copy the focused Logs pane or the command copy affordance."""

        row = self._selected_row()
        if row is None:
            self.notify("No Job selected", severity="warning")
            return
        requested_section = section
        if section == "logs":
            self.state.selected_section = "logs"
        section = requested_section or self.state.selected_section
        if section == "command":
            value = self._reconstructed_command(row)
            label = "command"
        else:
            attempt = self._selected_attempt(row)
            if self.log_manager is None or attempt is None:
                self.notify("Logs unavailable", severity="warning")
                return
            self._ensure_selected_terminal_logs(row)
            snapshot = self.log_manager.snapshot(row, attempt)
            value = "\n".join(snapshot.lines)
            if snapshot.error:
                value += f"\n\n[{snapshot.error}]" if value else f"[{snapshot.error}]"
            if not value:
                self.notify("No log output yet", severity="warning")
                return
            label = "logs"
        try:
            self.copy_to_clipboard(value)
        except Exception as exc:
            self.notify(
                f"Could not copy {label}: {exc}",
                severity="warning",
            )
        else:
            self.notify(f"Copied {label}")
        self._set_selected_subtitle()

    def selected_button_pressed(self, button_id: str) -> None:
        if button_id == "selected-command-copy":
            self._copy_selected_content("command")
            # Command is a metadata action, not a selectable inspector pane.
            # Return focus to Logs after the click so the copy action cannot
            # leave the outer inspector in a transient focus-within layout.
            self.call_after_refresh(self._focus_selected_section)
            return
        elif button_id == "selected-logs-copy":
            self._copy_selected_content("logs")
        else:
            return
        self.call_after_refresh(self._focus_selected_section)

    def action_copy_or_quit(self) -> None:
        if self._selected_inspector_active():
            self._copy_selected_content()
            return
        self.action_quit()

    def action_toggle_selected_or_cleanup(self) -> None:
        # The Selected Job inspector remains visible beside Jobs in the wide
        # layout, but visibility is not keyboard focus. Only the nested Logs
        # viewport owns ``c`` for collapsing its height; when Jobs (or the
        # outer Selected Job pane) has focus, ``c`` must retain its global
        # cleanup action.
        if self._selected_logs_focused():
            if self.state.selected_section != "logs":
                return
            self.state.logs_collapsed = not self.state.logs_collapsed
            # Re-opening a log viewport starts at its newest output.
            self.state.logs_auto_follow = True
            self._render_selected()
            self.call_after_refresh(self._focus_selected_section)
            return
        self.action_cleanup()

    def _ensure_selected_terminal_logs(self, row: JobUsage) -> None:
        if self.log_manager is None:
            return
        attempt = self._selected_attempt(row)
        if attempt is not None and attempt.terminal:
            self.log_manager.ensure_terminal_logs(row, attempt)

    def _ensure_selected_terminal_logs_for_current(self) -> None:
        row = self._selected_row()
        if row is not None:
            self._ensure_selected_terminal_logs(row)

    def _render_log_updates(self) -> None:
        if self.log_manager is not None and self.is_mounted:
            prune = getattr(self.log_manager, "prune", None)
            if callable(prune):
                prune()
        if (
            self.log_manager is not None
            and self.is_mounted
            and self._selected_inspector_visible()
        ):
            try:
                row = self._selected_row()
                attempt = self._selected_attempt(row) if row is not None else None
                revision_for = getattr(
                    self.log_manager, "snapshot_revision", None
                )
                if row is not None and attempt is not None and callable(revision_for):
                    view_key = (
                        row.uid,
                        attempt.uid or attempt.name,
                        int(revision_for(row, attempt)),
                    )
                    if view_key == self._selected_log_view_key:
                        return
                else:
                    view_key = None
                self._render_selected()
                if view_key is not None:
                    self._selected_log_view_key = view_key
            except NoMatches:
                return

    def _scroll_selected_section(self, amount: int) -> None:
        target_id = "#selected-logs-scroll"
        try:
            target = self.query_one(target_id, SelectedJobScroll)
        except NoMatches:
            return
        target.scroll_relative(
            y=amount,
            animate=False,
            force=True,
            immediate=True,
        )
        if self.state.selected_section == "logs":
            self.state.logs_auto_follow = (
                float(target.scroll_y) >= float(target.max_scroll_y) - 0.01
            )

    def _page_selected_section(self, direction: int) -> None:
        target_id = "#selected-logs-scroll"
        try:
            target = self.query_one(target_id, SelectedJobScroll)
        except NoMatches:
            return
        self._scroll_selected_section(max(1, target.size.height - 1) * direction)

    def _jump_selected_section(self, end: bool) -> None:
        target_id = "#selected-logs-scroll"
        try:
            target = self.query_one(target_id, SelectedJobScroll)
        except NoMatches:
            return
        if end:
            target.scroll_end(animate=False, force=True, immediate=True)
        else:
            target.scroll_home(animate=False, force=True, immediate=True)
        if self.state.selected_section == "logs":
            self.state.logs_auto_follow = end

    def _move_selected_attempt(self, amount: int) -> None:
        row = self._selected_row()
        if row is None:
            return
        attempts = self._selected_attempts(row)
        if not attempts:
            return
        current = self._selected_attempt(row)
        index = attempts.index(current) if current in attempts else len(attempts) - 1
        next_index = max(0, min(len(attempts) - 1, index + amount))
        if next_index == index:
            return
        self.state.selected_attempt_index = next_index
        self.state.logs_auto_follow = True
        self._ensure_selected_terminal_logs(row)
        try:
            self.query_one("#selected-logs-scroll", SelectedJobScroll).scroll_home(
                animate=False, force=True, immediate=True
            )
        except NoMatches:
            pass
        self._set_selected_subtitle()
        self._render_selected()

    def _render_selected(self) -> None:
        row = self._selected_row()
        target = self.query_one("#selected-pane", DashboardPane)
        inspector_visible = self._selected_inspector_visible()
        target.set_class(inspector_visible, "selected-active")
        compact_content = target.query_one(
            "#selected-pane-content", DashboardPaneContent
        )
        inspector = target.query_one("#selected-inspector", SelectedJobInspector)
        if not row:
            self._selected_log_view_key = None
            compact_content.update(Text("No Job selected", style=MUTED))
            inspector.clear()
            compact_content.display = True
            inspector.display = False
            return
        if inspector_visible:
            compact_content.display = False
            inspector.display = True
            status_icon, status_color = _status_style(row.status)
            attempt = self._selected_attempt(row)
            attempts = self._selected_attempts(row)
            attempt_index = attempts.index(attempt) if attempt in attempts else -1
            selected_pod = attempt.name if attempt is not None else (row.active_pod or "—")
            pod_phase = attempt.phase if attempt is not None else (row.active_pod_state or "—")
            pod_uid = _truncate(attempt.uid, 16) if attempt is not None and attempt.uid else "—"
            pod_container = attempt.container if attempt is not None and attempt.container else "—"
            details = [
                ("Job", row.job), ("Status", f"{status_icon} {row.status}"),
                ("Active pod state", row.active_pod_state), ("Active pod", row.active_pod or "—"),
                ("Node", row.nodes), ("Age", row.age),
                (
                    "GPU requested",
                    _gpu_display(
                        row.gpu_requested_type,
                        row.gpu_requested_count,
                    ),
                ),
                (
                    "GPU allocated now",
                    _gpu_display(
                        row.gpu_allocated_type,
                        row.gpu_allocated_count,
                    ),
                ),
                ("CPU request", f"{_short_cpu(row.cpu_requested)} vCPU"),
                ("RAM request", _short_memory(row.memory_requested_gib)),
            ]
            right_details = [
                ("Container restarts", str(row.container_restarts)),
                ("Pod attempts", str(row.pod_attempts)),
                ("Succeeded attempts", str(row.succeeded_attempts)),
                ("Failed attempts", str(row.failed_attempts)),
                (
                    "Backoff limit",
                    "—" if row.backoff_limit is None else str(row.backoff_limit),
                ),
                ("Completions", row.completions or "—"),
                ("GPU EMA", "—" if row.gpu_ema is None else f"{row.gpu_ema:.1f}%"),
                ("GPU 60s average", "—" if row.gpu_risk_average is None else f"{row.gpu_risk_average:.1f}%"),
                ("VRAM 60s average", "—" if row.vram_risk_average is None else f"{row.vram_risk_average:.1f}%"),
                ("Eviction risk", "YES" if row.at_risk else "No"),
            ]
            # Real dashboards have a selected attempt and log manager, so
            # expose the Pod identity alongside the Job-level metrics.  The
            # deterministic demo collector keeps its legacy single-page
            # layout for visual fixtures and does not pretend its synthetic
            # Pod metadata came from Kubernetes.
            if self.log_manager is not None:
                details[4:4] = [
                    ("Selected pod", selected_pod),
                    ("Pod phase", pod_phase),
                ]
                # Keep both detail columns at the same fixed row budget.  A
                # separate UID and container row made the right column one
                # line taller than the left column; its final Eviction-risk
                # line then overlapped the left Command row at the exact same
                # screen coordinate, causing Rich/Textual to composite the
                # line with a changing horizontal offset on clicks.
                pod_identity = pod_uid
                if pod_container not in {"", "—"}:
                    pod_identity = f"{pod_uid} · {_truncate(pod_container, 12)}"
                right_details[0:0] = [
                    ("Pod identity", pod_identity),
                    (
                        "Pod selection",
                        f"{attempt_index + 1}/{len(attempts)}"
                        if attempt_index >= 0
                        else "—",
                    ),
                ]
            def detail_table(values: List[Tuple[str, str]]) -> Table:
                table = Table.grid(expand=True, padding=(0, 1))
                # Keep each metadata item to one terminal row.  In addition
                # to avoiding wrapped pod names, this guarantees that the
                # Command row cannot overlap the right column's last row.
                table.add_column(
                    style=GRAY, width=18, no_wrap=True, overflow="ellipsis"
                )
                table.add_column(
                    style=WHITE, ratio=1, no_wrap=True, overflow="ellipsis"
                )
                for label, value in values:
                    value_style = (
                        status_color
                        if label == "Status"
                        else RED
                        if label == "Eviction risk" and row.at_risk
                        else WHITE
                    )
                    table.add_row(label, Text(value, style=value_style))
                return table

            detail_columns = (detail_table(details), detail_table(right_details))
            # Demo collectors do not own a Kubernetes log manager. Keep their
            # explicit full-screen rendering as a single Rich page so the
            # deterministic visual/scroll fixtures remain useful without
            # starting fake subprocesses; wide mode still uses the nested
            # inspector below to exercise the responsive layout.
            if self.log_manager is None and not self._wide_layout:
                compact_content.display = True
                inspector.display = False
                legacy_overview = Table.grid(expand=True, padding=(0, 2))
                legacy_overview.add_column(style=GRAY, width=18)
                legacy_overview.add_column(style=WHITE, ratio=1)
                legacy_values = (
                    details[:6]
                    + [("Created", row.created_at or "—"), ("Started", row.started_at or "—")]
                    + details[6:]
                    + right_details
                )
                for label, value in legacy_values:
                    value_style = (
                        status_color
                        if label == "Status"
                        else RED
                        if label == "Eviction risk" and row.at_risk
                        else WHITE
                    )
                    legacy_overview.add_row(label, Text(value, style=value_style))
                target.update(
                    Group(
                        Panel(
                            legacy_overview,
                            title=Text(" JOB DETAILS ", style=f"bold {CYAN}"),
                            border_style=BORDER,
                            box=box.SQUARE,
                        ),
                        Panel(
                            Text("\n".join(self._wrapped_command(row)), style=WHITE),
                            title=Text(" COMMAND ", style=f"bold {CYAN}"),
                            border_style=BORDER,
                            box=box.SQUARE,
                        ),
                    )
                )
                target.border_subtitle = ""
                return
            if attempt is not None:
                self._ensure_selected_terminal_logs(row)
            if self.log_manager is not None and attempt is not None:
                logs = self.log_manager.snapshot(row, attempt)
                self._selected_log_view_key = (
                    row.uid,
                    attempt.uid or attempt.name,
                    int(getattr(logs, "revision", 0)),
                )
            else:
                logs = LogSnapshot(
                    pod_name=attempt.name if attempt else "",
                    status="unavailable",
                    error="Logs unavailable in demo mode" if self.log_manager is None else "",
                )
                self._selected_log_view_key = None
            attempt_label = (
                f"Pod {attempt_index + 1}/{len(attempts)} · {attempt.name} · {attempt.phase}"
                if attempt is not None
                else "No Pod attempt"
            )
            inspector.update_view(
                detail_columns,
                logs,
                logs_collapsed=self.state.logs_collapsed,
                attempt_label=attempt_label,
            )
            self._set_selected_subtitle()
            return
        compact_content.display = True
        inspector.display = False
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

    def _metric_cell(
        self,
        label: str,
        current: Optional[float],
        values: List[Optional[float]],
        detail: str,
        *,
        terminal: bool = False,
    ) -> Text:
        color = MUTED if terminal else _metric_color(current)
        text = Text(label + "\n", style=f"bold {WHITE}")
        text.append("—" if current is None else f"{current:.0f}%", style=color)
        text.append("\n" + _spark(values, 24) + "\n", style=color)
        text.append(detail, style=GRAY)
        return text

    @staticmethod
    def _resource_metric_color(metric: Dict, value: Optional[float]) -> str:
        return MUTED if metric.get("terminal") else _metric_color(value)

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
        terminal = row.status in {"Succeeded", "Failed"}
        gpu_capacity = last.gpu_capacity or float(row.gpu_count)
        vram_capacity = last.vram_capacity or row.gpu_memory_total_gib
        cpu_capacity = last.cpu_capacity or row.cpu_allocated or row.cpu_requested
        ram_capacity = (
            last.ram_capacity
            or row.memory_allocated_gib
            or row.memory_requested_gib
        )
        return [
            {
                "label": "GPU", "values": [point.gpu for point in points],
                "current": last.gpu, "capacity": gpu_capacity, "unit": "GPU",
                "terminal": terminal,
                "sample_period": 1,
                "absolute": self._absolute_metric(
                    last.gpu, gpu_capacity, "GPU"
                ),
            },
            {
                "label": "VRAM", "values": [point.vram for point in points],
                "current": last.vram, "capacity": vram_capacity, "unit": "GiB",
                "terminal": terminal,
                "sample_period": 1,
                "absolute": self._absolute_metric(
                    last.vram, vram_capacity, "GiB"
                ),
            },
            {
                "label": "CPU", "values": [point.cpu for point in points],
                "current": last.cpu, "capacity": cpu_capacity, "unit": "vCPU",
                "terminal": terminal,
                "sample_period": int(KUBERNETES_USAGE_SECONDS),
                "absolute": self._absolute_metric(
                    last.cpu, cpu_capacity, "vCPU"
                ),
            },
            {
                "label": "RAM", "values": [point.ram for point in points],
                "current": last.ram, "capacity": ram_capacity, "unit": "GiB",
                "terminal": terminal,
                "sample_period": int(KUBERNETES_USAGE_SECONDS),
                "absolute": self._absolute_metric(
                    last.ram, ram_capacity, "GiB"
                ),
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
        color = self._resource_metric_color(metric, current)
        average = mean(valid) if valid else None
        peak = max(valid) if valid else None
        summary = Text()
        summary.append("Now      ", style=GRAY)
        summary.append("—" if current is None else f"{current:.0f}%", style=f"bold {color}")
        summary.append(f"  {metric['absolute']}\n", style=WHITE)
        summary.append("Average  ", style=GRAY)
        summary.append("—" if average is None else f"{average:.0f}%", style=self._resource_metric_color(metric, average))
        summary.append("  " + self._absolute_metric(average, metric["capacity"], metric["unit"]) + "\n", style=WHITE)
        summary.append("Peak     ", style=GRAY)
        summary.append("—" if peak is None else f"{peak:.0f}%", style=self._resource_metric_color(metric, peak))
        summary.append("  " + self._absolute_metric(peak, metric["capacity"], metric["unit"]), style=WHITE)
        stats_width = max(len(line) for line in summary.plain.splitlines())
        history_width = self._resource_history_width("wide", stats_width)
        history_height = self._resource_history_height("wide")
        history_color = (
            self._resource_metric_color(metric, current)
            if metric.get("terminal")
            else None
        )
        history = _ResourceHistoryChart(
            values,
            history_width,
            history_height,
            self.state.resource_zoom,
            metric["sample_period"],
            history_color,
        )
        if self._building_expanded_resource:
            self._expanded_resource_charts[metric["label"]] = history
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
        stats.append("—" if current is None else f"{current:.0f}%", style=f"bold {self._resource_metric_color(metric, current)}")
        stats.append("\nUsed     ", style=GRAY)
        stats.append(metric["absolute"], style=WHITE)
        stats.append("\nAverage  ", style=GRAY)
        stats.append("—" if average is None else f"{average:.0f}%", style=self._resource_metric_color(metric, average))
        stats.append("\nPeak     ", style=GRAY)
        stats.append("—" if peak is None else f"{peak:.0f}%", style=self._resource_metric_color(metric, peak))
        layout = "collapsed" if collapsed else "compact"
        stats_width = max(len(line) for line in stats.plain.splitlines())
        history_width = self._resource_history_width(layout, stats_width)
        history_height = self._resource_history_height(layout)
        history_color = (
            self._resource_metric_color(metric, current)
            if metric.get("terminal")
            else None
        )
        history = _ResourceHistoryChart(
            metric["values"],
            history_width,
            history_height,
            self.state.resource_zoom,
            metric["sample_period"],
            history_color,
        )
        if self._building_expanded_resource:
            self._expanded_resource_charts[metric["label"]] = history
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
            threshold = "—" if row.gpu_risk_threshold is None else f"{row.gpu_risk_threshold:.0f}%"
            reasons = []
            if row.gpu_at_risk or not row.vram_at_risk:
                average = "—" if row.gpu_risk_average is None else f"{row.gpu_risk_average:.1f}%"
                reasons.append(f"GPU {average} < {threshold}")
            if row.vram_at_risk:
                average = "—" if row.vram_risk_average is None else f"{row.vram_risk_average:.1f}%"
                reasons.append(f"VRAM {average} < {threshold}")
            risk = "! EVICTION RISK · " + " · ".join(reasons)
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
        del layout
        show_power = self.size.width >= 95
        show_ecc = self.size.width >= 110
        show_driver = self.size.width >= 135
        show_uuid = self.size.width >= 175
        devices = Table(
            box=box.SIMPLE_HEAD,
            expand=True,
            padding=(0, 1),
            header_style=f"bold {CYAN}",
        )
        devices.add_column("GPU", width=4)
        devices.add_column("MODEL", ratio=2, overflow="ellipsis", no_wrap=True)
        if show_uuid:
            devices.add_column("UUID", ratio=2, overflow="ellipsis", no_wrap=True)
        devices.add_column("VRAM", width=13, justify="right")
        devices.add_column("UTIL", width=6, justify="right")
        devices.add_column("TEMP", width=6, justify="right")
        if show_power:
            devices.add_column("POWER", width=8, justify="right")
        if show_ecc:
            devices.add_column("ECC", width=5, justify="right")
        if show_driver:
            devices.add_column("DRIVER", width=8, overflow="ellipsis", no_wrap=True)
        for device in row.gpu_devices:
            memory = (
                "—" if device.memory_used_gib is None or device.memory_total_gib is None
                else f"{device.memory_used_gib:.1f}/{device.memory_total_gib:.1f}G"
            )
            cells = [
                Text(str(device.index), style=WHITE),
                Text(device.name, style=WHITE, no_wrap=True, overflow="ellipsis"),
            ]
            if show_uuid:
                cells.append(
                    Text(device.uuid, style=GRAY, no_wrap=True, overflow="ellipsis")
                )
            cells.extend([
                Text(memory, style=WHITE),
                Text(
                    self._device_value(device.utilization, "%"),
                    style=(MUTED if row.status in {"Succeeded", "Failed"} else _metric_color(device.utilization)),
                ),
                Text(self._device_value(device.temperature_c, "°C"), style=WHITE),
            ])
            if show_power:
                cells.append(Text(self._device_value(device.power_w, "W"), style=WHITE))
            if show_ecc:
                cells.append(
                    Text(
                        "—" if device.ecc_errors is None else str(device.ecc_errors),
                        style=WHITE,
                    )
                )
            if show_driver:
                cells.append(Text(device.driver_version, style=WHITE))
            devices.add_row(*cells)
            for process in device.processes:
                process_cells = [
                    Text("", style=GRAY),
                    Text(
                        f"  └─ {process.pid}  {process.name}",
                        style=GRAY,
                        no_wrap=True,
                        overflow="ellipsis",
                    ),
                ]
                if show_uuid:
                    process_cells.append(Text("", style=GRAY))
                process_cells.extend(
                    [
                        Text(
                            "—"
                            if process.memory_used_gib is None
                            else f"{process.memory_used_gib:.1f}G",
                            style=CYAN_2,
                        ),
                        Text(
                            "—"
                            if process.gpu_utilization is None
                            else f"{process.gpu_utilization:.1f}%",
                            style=CYAN_2,
                        ),
                        Text("", style=GRAY),
                    ]
                )
                if show_power:
                    process_cells.append(Text("", style=GRAY))
                if show_ecc:
                    process_cells.append(Text("", style=GRAY))
                if show_driver:
                    process_cells.append(Text("", style=GRAY))
                devices.add_row(*process_cells)

        column_count = 5 + sum((show_uuid, show_power, show_ecc, show_driver))
        if not row.gpu_devices:
            devices.add_row(
                *(["—", "GPU device metrics unavailable"] + [""] * (column_count - 2))
            )

        return Panel(
            devices,
            title=Text(f" GPU DEVICES ({len(row.gpu_devices)}) ", style=f"bold {CYAN}"),
            border_style=CYAN_2, box=box.SQUARE, padding=(0, 0),
        )

    def _renderable_height(self, renderable, width: int) -> int:
        options = self.console.options.update(width=max(1, width), height=None)
        options.max_height = None
        return len(self.console.render_lines(renderable, options, pad=True))

    def _render_expanded_resources(self, target: DashboardPane, row: JobUsage, points: List[MetricPoint]) -> None:
        screen_layout = self._resource_layout()
        # The expanded Resource Usage inspector has enough room to make a
        # four-card overview useful at every supported size.  The old
        # ``wide`` layout stacked all four cards vertically, which made a
        # taller terminal spend most of its viewport on one metric at a time.
        # Keep the wide selected-job strip, but use the compact card density
        # whenever the screen would otherwise choose that vertical layout.
        layout = "compact" if screen_layout == "wide" else screen_layout
        metrics = self._resource_metrics(row, points)
        selected = self._selected_resource_strip(row, screen_layout)
        self._expanded_resource_charts = {}
        self._expanded_resource_uid = row.uid
        self._building_expanded_resource = True
        width = max(1, target.size.width)
        metric_start = self._renderable_height(selected, width)
        graph_height = self._resource_history_height(layout)
        graph_regions: List[Tuple[int, int, int, int]] = []
        if layout == "compact":
            panels = [self._compact_metric_panel(metric) for metric in metrics]
            metric_content = Table.grid(expand=True, padding=(0, 1))
            metric_content.add_column(ratio=1)
            metric_content.add_column(ratio=1)
            metric_content.add_row(panels[0], panels[1])
            metric_content.add_row(panels[2], panels[3])
            card_width = max(1, (width - 2) // 2)
            row_top = metric_start
            for row_index in range(2):
                row_panels = panels[row_index * 2:row_index * 2 + 2]
                row_height = max(
                    self._renderable_height(panel, card_width)
                    for panel in row_panels
                )
                for column, panel in enumerate(row_panels):
                    stats_width = int(panel.renderable.columns[0].width or 0)
                    history_width = self._resource_history_width(
                        "compact", stats_width
                    )
                    card_left = column * (card_width + 2)
                    graph_regions.append(
                        (
                            card_left + max(2, card_width - history_width - 2),
                            max(row_top + 1, row_top + row_height - 1 - graph_height),
                            min(width, card_left + card_width - 1),
                            row_top + row_height - 1,
                        )
                    )
                row_top += row_height
        else:
            panels = [
                self._compact_metric_panel(metric, collapsed=True)
                for metric in metrics
            ]
            metric_content = Table.grid(expand=True, padding=(0, 1))
            metric_content.add_column(ratio=1)
            metric_content.add_column(ratio=1)
            metric_content.add_row(panels[0], panels[1])
            metric_content.add_row(panels[2], panels[3])
            card_width = max(1, (width - 2) // 2)
            row_top = metric_start
            for row_index in range(2):
                row_panels = panels[row_index * 2:row_index * 2 + 2]
                row_height = max(
                    self._renderable_height(panel, card_width)
                    for panel in row_panels
                )
                for column, panel in enumerate(row_panels):
                    stats_width = int(panel.renderable.columns[0].width or 0)
                    history_width = self._resource_history_width(
                        "collapsed", stats_width
                    )
                    card_left = column * (card_width + 2)
                    graph_regions.append(
                        (
                            card_left + max(2, card_width - history_width - 2),
                            max(row_top + 1, row_top + row_height - 1 - graph_height),
                            min(width, card_left + card_width - 1),
                            row_top + row_height - 1,
                        )
                    )
                row_top += row_height
        self._building_expanded_resource = False
        self._resource_graph_regions = graph_regions
        content = Group(
            selected,
            metric_content,
            self._gpu_devices_panel(row, screen_layout),
        )
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
        metrics = self._resource_metrics(row, points)
        cells = [
            self._metric_cell(
                metric["label"],
                metric["current"],
                metric["values"],
                metric["absolute"],
                terminal=metric.get("terminal", False),
            )
            for metric in metrics
        ]
        table = Table(box=None, expand=True, padding=(0, 1), show_header=False)
        pane_width = target.content_size.width or target.size.width or self.size.width
        columns = 2 if pane_width < 90 else 4
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
        # One event must always consume exactly one terminal row.  Folding
        # messages made logical offsets diverge from the physical viewport,
        # which was the root cause of the old unreachable-last-event bug.
        table.add_column("MESSAGE", ratio=4, no_wrap=True, overflow="ellipsis")
        for event in events_list[start:start + visible]:
            stamp = datetime.fromtimestamp(_timestamp(event.timestamp)).strftime("%H:%M:%S") if _timestamp(event.timestamp) else "—"
            color = _event_style(event)
            reason = f"{event.reason} ×{event.count}" if event.count > 1 else event.reason
            cells = [Text(stamp, style=GRAY), Text(event.event_type, style=color), Text(reason, style=color)]
            cells.append(
                Text(
                    event.message,
                    style=WHITE,
                    no_wrap=True,
                    overflow="ellipsis",
                )
            )
            table.add_row(*cells)
        newer = len(events_list) - (start + visible)
        target.border_subtitle = f" {newer} newer events ↓ " if newer > 0 and not self.state.events_auto_follow else ""
        target.update(table)

    def _render_footer(self) -> None:
        if self.size.width < MINIMUM_WIDTH or self.size.height < MINIMUM_HEIGHT:
            self.query_one("#falcon-footer", Static).update(Text("q Quit   r Retry after resizing", style=GRAY))
            return
        marked = len(self.state.marked_job_uids)
        pane_action = (
            "Esc Restore"
            if self.state.expanded_pane == self.state.focused_pane
            else "Enter Expand"
        )
        if self.state.focused_pane == "jobs":
            mark_label = f"Space Mark ({marked})" if marked else "Space Mark"
            clean_label = "c Clean marked" if marked else "c Clean succeeded"
            if self.size.width < 140:
                value = (
                    f"s Sort  {mark_label}  k Kill  c Clean  {pane_action}  "
                    "Tab Next pane  q Quit"
                )
            elif self.size.width < 160:
                value = (
                    f"↑/↓ Jobs  s Sort  {mark_label}  k/F9 Kill  {clean_label}  "
                    f"f Filter  / Search  {pane_action}  Tab Next pane  r Refresh  q Quit"
                )
            else:
                value = (
                    f"↑/↓ Navigate   s Sort   {mark_label}   f Filters   v Panes   "
                    f"k/F9 Kill   {clean_label}   {pane_action}   Tab Next pane   "
                    "/ Search   r Refresh   q Quit"
                )
        elif self.state.focused_pane == "selected":
            selected_action = (
                "Esc Restore"
                if self.state.expanded_pane == "selected"
                else "Enter Expand"
            )
            value = (
                f"↑/↓ Scroll   ←/→ Pods   PgUp/PgDn Page   Home/End   {selected_action}   Tab Next pane   r Refresh   q Quit"
                if self._selected_inspector_visible()
                else f"↑/↓ Change Job   v Panes   {pane_action}   Tab Next pane   r Refresh   q Quit"
            )
        elif self.state.focused_pane == "resources":
            zoom = round(100 / self.state.resource_zoom)
            if self.state.expanded_pane == "resources":
                value = f"↑/↓ Scroll   ←/→ History   Hover charts: history   R Range   +/- Zoom {zoom}%   Esc Restore   Tab Next pane   q Quit"
            else:
                value = f"←/→ History   Home/End Range   +/- Zoom {zoom}%   v Panes   {pane_action}   Tab Next pane   r Refresh   q Quit"
        else:
            value = f"↑/↓ Scroll   PgUp/PgDn Page   Home Oldest   End Newest   / Search   v Panes   {pane_action}   Tab Next pane   r Refresh   q Quit"
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
        pane_ids = [
            "jobs-pane",
            "selected-pane",
            "resources-pane",
            "events-pane",
            "summary",
            "controls",
        ]
        body = self.query_one("#dashboard-body")
        resize = self.query_one("#resize-message", Static)
        if self.size.width < MINIMUM_WIDTH or self.size.height < MINIMUM_HEIGHT:
            self._wide_layout = False
            body.display = False
            for pane_id in pane_ids:
                self.query_one(f"#{pane_id}").display = False
            resize.display = True
            resize.update(
                f"Falcon Dashboard requires at least {MINIMUM_WIDTH}×{MINIMUM_HEIGHT}.\n"
                f"Current terminal: {self.size.width}×{self.size.height}.\n\nResize the terminal to inspect and manage Jobs."
            )
            return
        resize.display = False
        # Keep the resource-usage pane at a readable fixed height. At the
        # smallest supported heights, Events is the least disruptive pane to
        # hide temporarily; persisted pane preferences remain unchanged.
        self._responsive_hidden_panes = (
            {"events"}
            if self.size.height < 28 and "events" not in self.state.hidden_panes
            else set()
        )
        visible_panes = self._visible_panes()
        if (
            self.state.expanded_pane
            and self.state.expanded_pane in self._responsive_hidden_panes
        ):
            # An already-expanded pane remains usable while resizing; the
            # responsive hide applies to the compact multi-pane layout.
            visible_panes = [*visible_panes, self.state.expanded_pane]
        if self.state.focused_pane not in visible_panes:
            # Keep focus state across a transient responsive hide so restoring
            # the terminal does not unexpectedly move the user to Jobs.
            if self.state.focused_pane not in self._responsive_hidden_panes:
                self.state.focused_pane = "jobs"
        if (
            self.state.expanded_pane
            and self.state.expanded_pane not in visible_panes
            and self.state.expanded_pane not in self._responsive_hidden_panes
        ):
            self.state.expanded_pane = None
        if self.state.expanded_pane:
            self._wide_layout = False
            body.display = True
            active = self.state.expanded_pane + "-pane"
            for pane_id in pane_ids:
                if pane_id in {"summary", "controls"}:
                    self.query_one(f"#{pane_id}").display = False
                else:
                    self.query_one(f"#{pane_id}").display = pane_id == active
            expanded = self.query_one(f"#{active}")
            expanded.styles.height = "1fr"
            return
        self._wide_layout = (
            self.size.width >= WIDE_LAYOUT_MIN_WIDTH
            and self.size.height >= WIDE_LAYOUT_MIN_HEIGHT
        )
        body.display = True
        for pane_id in pane_ids:
            pane = pane_id.removesuffix("-pane")
            widget = self.query_one(f"#{pane_id}")
            widget.display = pane_id in {"summary", "controls"} or pane in visible_panes
        if self._wide_layout:
            # DashboardBodyLayout owns the exact regions in split mode. Give
            # each visible pane a flexible height so its content does not
            # fight the placement calculated for that region.
            for pane in visible_panes:
                self.query_one(f"#{pane}-pane").styles.height = "1fr"
        else:
            self.query_one("#jobs-pane").styles.height = "1fr"
            self.query_one("#summary").styles.height = 2
            self.query_one("#selected-pane").styles.height = 3
            self.query_one("#resources-pane").styles.height = RESOURCE_PANE_HEIGHT
            self.query_one("#events-pane").styles.height = 7

    def _visible_panes(self) -> List[str]:
        hidden = self.state.hidden_panes | self._responsive_hidden_panes
        return [
            pane
            for pane in ("jobs", "selected", "resources", "events")
            if pane not in hidden
        ]

    def _focus(self, pane: str) -> None:
        if pane not in self._visible_panes():
            self.notify(f"{pane.title()} pane is hidden · press v to configure panes")
            return
        self.state.focused_pane = pane
        self.set_focus(
            self.query_one(f"#{pane}-pane", DashboardPane),
            scroll_visible=False,
        )
        self._set_titles()
        self._render_footer()

    def action_next_pane(self) -> None:
        self._cycle_pane(1)

    def action_previous_pane(self) -> None:
        self._cycle_pane(-1)

    def _cycle_pane(self, amount: int, panes: Optional[List[str]] = None) -> None:
        panes = panes or self._visible_panes()
        if self.state.focused_pane not in panes:
            self.state.focused_pane = "jobs"
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
        if self.state.expanded_pane == "selected":
            self.state.selected_section = "logs"
            self.state.logs_auto_follow = True
            self._ensure_selected_terminal_logs_for_current()
        self._apply_layout()
        self._render_all()
        self.call_after_refresh(self._render_all)
        if self.state.expanded_pane == "selected" and self.log_manager is not None:
            self.call_after_refresh(self._focus_selected_section)

    def action_toggle_expand(self) -> None:
        self.state.expanded_pane = None if self.state.expanded_pane else self.state.focused_pane
        if self.state.expanded_pane == "selected":
            self.state.selected_section = "logs"
            self.state.logs_auto_follow = True
            self._ensure_selected_terminal_logs_for_current()
        self._apply_layout()
        self._render_all()
        self.call_after_refresh(self._render_all)
        if self.state.expanded_pane == "selected" and self.log_manager is not None:
            self.call_after_refresh(self._focus_selected_section)

    def action_escape(self) -> None:
        # Application bindings remain active while a ModalScreen is mounted in
        # the Textual version supported by Falcon.  Handle Escape here as well
        # as on each dialog so it can never leak through to the dashboard.
        if isinstance(self.screen, (FilterDialog, KillDialog, CleanupDialog, PaneVisibilityDialog)):
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
        self.state.selected_attempt_index = -1
        self.state.selected_section = "logs"
        self.state.logs_collapsed = False
        self.state.logs_auto_follow = True
        for pane in ("selected", "resources"):
            self.query_one(f"#{pane}-pane", DashboardPane).scroll_home(
                animate=False,
                force=True,
                immediate=True,
            )
        self._request_update()
        self._render_all()

    def action_up(self) -> None:
        if self.state.focused_pane == "selected" and self.state.expanded_pane == "selected":
            if self.log_manager is None:
                self._scroll_expanded_pane("selected", -1)
            else:
                self._scroll_selected_section(-1)
            return
        if (
            self.state.focused_pane == "selected"
            and self._selected_inspector_active()
            and self._selected_logs_focused()
        ):
            self._scroll_selected_section(-1)
            return
        if self.state.focused_pane in {"jobs", "selected"}:
            self._move_cursor(-1)
        elif self.state.focused_pane == "events":
            self._scroll_events(-1)
        elif self.state.expanded_pane == "resources":
            self._scroll_expanded_pane("resources", -1)
        else:
            self._scroll_history(1)

    def action_down(self) -> None:
        if self.state.focused_pane == "selected" and self.state.expanded_pane == "selected":
            if self.log_manager is None:
                self._scroll_expanded_pane("selected", 1)
            else:
                self._scroll_selected_section(1)
            return
        if (
            self.state.focused_pane == "selected"
            and self._selected_inspector_active()
            and self._selected_logs_focused()
        ):
            self._scroll_selected_section(1)
            return
        if self.state.focused_pane in {"jobs", "selected"}:
            self._move_cursor(1)
        elif self.state.focused_pane == "events":
            self._scroll_events(1)
        elif self.state.expanded_pane == "resources":
            self._scroll_expanded_pane("resources", 1)
        else:
            self._scroll_history(-1)

    def action_kill_or_up(self) -> None:
        if self.state.focused_pane == "jobs":
            self.action_kill()
        else:
            self.action_up()

    def action_left(self) -> None:
        if (
            self.log_manager is not None
            and self.state.focused_pane == "selected"
            and (
                self.state.expanded_pane == "selected"
                or (self._selected_inspector_active() and self._selected_logs_focused())
            )
        ):
            self._move_selected_attempt(-1)
        elif self.state.focused_pane == "resources":
            self._scroll_history(1)

    def action_right(self) -> None:
        if (
            self.log_manager is not None
            and self.state.focused_pane == "selected"
            and (
                self.state.expanded_pane == "selected"
                or (self._selected_inspector_active() and self._selected_logs_focused())
            )
        ):
            self._move_selected_attempt(1)
        elif self.state.focused_pane == "resources":
            self._scroll_history(-1)

    def _scroll_events(self, amount: int) -> None:
        events_list = self._filtered_events()
        visible = self._visible_event_count()
        maximum = max(0, len(events_list) - visible)
        if self.state.events_auto_follow:
            self.state.events_scroll_offset = maximum
        self.state.events_scroll_offset = max(
            0,
            min(maximum, self.state.events_scroll_offset + amount),
        )
        # Reaching the newest page is the natural way to resume follow mode.
        # Scrolling anywhere above it freezes the viewport when events arrive.
        self.state.events_auto_follow = self.state.events_scroll_offset >= maximum
        self._render_events()

    def _schedule_resource_history_refresh(self) -> None:
        """Repaint expanded history once after a burst of wheel events.

        Updating the four chart renderables is cheap, but invalidating the
        Rich/Textual content tree on every wheel tick makes a fast mouse wheel
        repaint the entire expanded inspector repeatedly.  Keep the latest
        values in the mutable charts immediately, then let Textual coalesce
        the actual repaint at the next refresh boundary.
        """
        if self._resource_history_refresh_pending:
            return
        self._resource_history_refresh_pending = True

        def refresh() -> None:
            self._resource_history_refresh_pending = False
            if (
                not self.is_mounted
                or self.state.expanded_pane != "resources"
            ):
                return
            try:
                self.query_one(
                    "#resources-pane .dashboard-pane-content",
                    DashboardPaneContent,
                ).refresh(layout=False)
            except NoMatches:
                return

        self.call_after_refresh(refresh)

    def _scroll_history(self, amount: int) -> None:
        row = self._selected_row()
        maximum = max(0, len(self.histories.get(row.uid, [])) - 1) if row else 0
        self.state.resource_scroll_offset = max(0, min(maximum, self.state.resource_scroll_offset + amount))
        if (
            row
            and self.state.expanded_pane == "resources"
            and self._expanded_resource_uid == row.uid
            and self._expanded_resource_charts
        ):
            points = self._history_slice(row.uid)
            metrics = self._resource_metrics(row, points) if points else []
            for metric in metrics:
                chart = self._expanded_resource_charts.get(metric["label"])
                if chart is not None:
                    chart.update(
                        metric["values"],
                        self.state.resource_zoom,
                        (
                            self._resource_metric_color(metric, metric["current"])
                            if metric.get("terminal")
                            else None
                        ),
                )
            self._schedule_resource_history_refresh()
            return
        self._render_resources()

    def action_page_up(self) -> None:
        if self.state.focused_pane == "selected" and self.state.expanded_pane == "selected":
            if self.log_manager is None:
                self._page_expanded_pane("selected", -1)
            else:
                self._page_selected_section(-1)
        elif (
            self.state.focused_pane == "selected"
            and self._selected_inspector_active()
            and self._selected_logs_focused()
        ):
            self._page_selected_section(-1)
        elif self.state.focused_pane == "resources" and self.state.expanded_pane == "resources":
            self._page_expanded_pane("resources", -1)
        elif self.state.focused_pane == "resources":
            self.state.resource_zoom = min(16, self.state.resource_zoom * 2)
            self._render_resources()
            self._render_footer()
        else:
            self._scroll_events(-self._visible_event_count())

    def action_page_down(self) -> None:
        if self.state.focused_pane == "selected" and self.state.expanded_pane == "selected":
            if self.log_manager is None:
                self._page_expanded_pane("selected", 1)
            else:
                self._page_selected_section(1)
        elif (
            self.state.focused_pane == "selected"
            and self._selected_inspector_active()
            and self._selected_logs_focused()
        ):
            self._page_selected_section(1)
        elif self.state.focused_pane == "resources" and self.state.expanded_pane == "resources":
            self._page_expanded_pane("resources", 1)
        elif self.state.focused_pane == "resources":
            self.state.resource_zoom = max(1, self.state.resource_zoom // 2)
            self._render_resources()
            self._render_footer()
        else:
            self._scroll_events(self._visible_event_count())

    def action_home(self) -> None:
        if self.state.focused_pane == "selected" and self.state.expanded_pane == "selected":
            if self.log_manager is None:
                self._jump_expanded_pane("selected", False)
            else:
                self._jump_selected_section(False)
        elif (
            self.state.focused_pane == "selected"
            and self._selected_inspector_active()
            and self._selected_logs_focused()
        ):
            self._jump_selected_section(False)
        elif self.state.focused_pane == "events":
            self.state.events_auto_follow = False
            self.state.events_scroll_offset = 0
            self._render_events()
        elif self.state.focused_pane == "resources":
            if self.state.expanded_pane == "resources":
                self._jump_expanded_pane("resources", False)
            else:
                row = self._selected_row()
                self.state.resource_scroll_offset = max(0, len(self.histories.get(row.uid, [])) - 1) if row else 0
                self._render_resources()

    def action_end(self) -> None:
        if self.state.focused_pane == "selected" and self.state.expanded_pane == "selected":
            if self.log_manager is None:
                self._jump_expanded_pane("selected", True)
            else:
                self._jump_selected_section(True)
        elif (
            self.state.focused_pane == "selected"
            and self._selected_inspector_active()
            and self._selected_logs_focused()
        ):
            self._jump_selected_section(True)
        elif self.state.focused_pane == "events":
            self.state.events_auto_follow = True
            self._render_events()
        elif self.state.focused_pane == "resources":
            if self.state.expanded_pane == "resources":
                self._jump_expanded_pane("resources", True)
            else:
                self.state.resource_scroll_offset = 0
                self._render_resources()

    def action_history_left(self) -> None:
        if self.state.focused_pane == "resources":
            self._scroll_history(1)

    def action_history_right(self) -> None:
        if self.state.focused_pane == "resources":
            self._scroll_history(-1)

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
        if not row:
            return
        if row.uid in self.state.marked_job_uids:
            self.state.marked_job_uids.remove(row.uid)
        else:
            self.state.marked_job_uids.add(row.uid)
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
        if event.input.id != "search-input":
            return
        if "events" in event.input.placeholder.lower():
            self.event_search = event.value.strip()
        else:
            self.state.search_query = event.value.strip()
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
        self.state.sort_direction = {
            "Age": "desc", "Name": "asc", "Status": "asc",
        }[self.state.sort_field]
        self._filter_rows()
        self._render_all()
        if self._persist_sort:
            try:
                self._persist_sort(self.state.sort_field, self.state.sort_direction)
            except (OSError, ValueError) as exc:
                self.notify(f"Could not save sort selection: {exc}", severity="error")

    def action_update_data(self) -> None:
        invalidate = getattr(self.collector, "invalidate", None)
        if invalidate:
            invalidate()
        self._request_update()

    def action_help(self) -> None:
        self.notify("Tab panes · v show/hide panes · 1/2/3 focus · Space mark · k/F9 kill marked · c clean succeeded Jobs within marked set · / search · z expand · r refresh · q quit", timeout=8)

    def action_panes(self) -> None:
        self.push_screen(PaneVisibilityDialog(self.state.hidden_panes), self._panes_selected)

    def _panes_selected(self, hidden: Optional[Set[str]]) -> None:
        if hidden is None:
            return
        self.state.hidden_panes = set(hidden)
        if self.state.focused_pane in hidden:
            self.state.focused_pane = "jobs"
        if self.state.expanded_pane in hidden:
            self.state.expanded_pane = None
        self._apply_layout()
        self._focus(self.state.focused_pane)
        self._render_all()
        self._request_update()
        if self._persist_hidden_panes:
            try:
                self._persist_hidden_panes(self.state.hidden_panes)
            except (OSError, ValueError) as exc:
                self.notify(f"Could not save pane visibility: {exc}", severity="error")

    def action_cleanup(self) -> None:
        marked_rows = [row for row in self.rows if row.uid in self.state.marked_job_uids]
        candidates = marked_rows or self.rows
        targets = [row for row in candidates if row.status == "Succeeded"]
        if not targets:
            message = "No marked succeeded Jobs to clean" if marked_rows else "No succeeded Jobs to clean"
            self.notify(message)
            return
        self.push_screen(
            CleanupDialog(
                targets,
                marked=bool(marked_rows),
                excluded_marked=len(marked_rows) - len(targets),
            ),
            lambda confirmed: self._cleanup_confirmed(confirmed, targets),
        )

    def _cleanup_confirmed(self, confirmed: bool, targets: List[JobUsage]) -> None:
        if confirmed:
            self._job_action_confirmed(("job", targets))

    def action_kill(self) -> None:
        selected = self._selected_row()
        targets = [row for row in self.rows if row.uid in self.state.marked_job_uids]
        if not targets and selected:
            targets = [selected]
        if not targets:
            return
        self.state.kill_dialog.update({"isOpen": True, "targets": [row.uid for row in targets]})
        self.push_screen(KillDialog(targets), self._job_action_confirmed)

    def _job_action_confirmed(self, result: Optional[Tuple[str, List[JobUsage]]]) -> None:
        self.state.kill_dialog["isOpen"] = False
        if not result:
            return
        action, rows = result

        def apply_action() -> None:
            succeeded = 0
            coder_succeeded = 0
            failures: List[str] = []
            for row in rows:
                if row.job.startswith("coder-") and action in {"job", "restart"}:
                    if self._coder_workspace_action is None:
                        failures.append(
                            f"{row.job}: Coder workspace actions are unavailable"
                        )
                        continue
                    try:
                        self._coder_workspace_action(
                            row.job,
                            "delete" if action == "job" else "restart",
                        )
                        succeeded += 1
                        coder_succeeded += 1
                    except Exception as exc:
                        detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                        failures.append(f"{row.job}: {detail}")
                    continue
                if action == "restart":
                    try:
                        fetched = subprocess.run(
                            ["kubectl", "get", "job", row.job, "--namespace", self.collector.namespace, "-o", "json"],
                            capture_output=True, text=True, timeout=20,
                        )
                        if fetched.returncode != 0:
                            failures.append(f"{row.job}: could not read Job manifest")
                            continue
                        manifest = _restart_job_manifest(json.loads(fetched.stdout), row.job, self.collector.namespace)
                        deleted = subprocess.run(
                            [
                                "kubectl", "delete", "job", row.job,
                                "--namespace", self.collector.namespace,
                                "--wait=true", "--timeout=30s",
                            ],
                            capture_output=True, text=True, timeout=40,
                        )
                        if deleted.returncode != 0:
                            failures.append(f"{row.job}: Kubernetes deletion failed")
                            continue
                        created = subprocess.run(
                            ["kubectl", "create", "-f", "-", "--namespace", self.collector.namespace],
                            input=json.dumps(manifest), capture_output=True, text=True, timeout=20,
                        )
                        if created.returncode == 0:
                            succeeded += 1
                        else:
                            failures.append(f"{row.job}: Kubernetes creation failed")
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        failures.append(f"{row.job}: {exc}")
                    continue
                command = [
                    "kubectl", "delete", "job", row.job, "--wait=false",
                    "--namespace", self.collector.namespace,
                ]
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
                    succeeded += int(result.returncode == 0)
                    if result.returncode != 0:
                        failures.append(f"{row.job}: Kubernetes deletion failed")
                except (OSError, subprocess.SubprocessError) as exc:
                    failures.append(f"{row.job}: {exc}")
            def finish() -> None:
                if action == "restart":
                    if succeeded == len(rows):
                        detail = f" {rows[0].job}" if succeeded == 1 else ""
                        self.notify(f"Restarted {succeeded} Job{'s' if succeeded != 1 else ''}{detail}")
                    else:
                        detail = f" · {failures[0]}" if failures else ""
                        self.notify(f"Restarted {succeeded} of {len(rows)} Jobs · {len(rows) - succeeded} failed{detail}", severity="error")
                elif succeeded == len(rows):
                    if coder_succeeded == len(rows):
                        self.notify(
                            f"Deleting {succeeded} Coder workspace"
                            f"{'s' if succeeded != 1 else ''} through Coder"
                        )
                    else:
                        self.notify(f"Deleted {succeeded} Job{'s' if succeeded != 1 else ''}")
                else:
                    detail = f" · {failures[0]}" if failures else ""
                    self.notify(f"Deleted {succeeded} of {len(rows)} Jobs · {len(rows) - succeeded} failed{detail}", severity="error")
                invalidate = getattr(self.collector, "invalidate", None)
                if invalidate:
                    invalidate()
                self._request_update()
            self.call_from_thread(finish)

        threading.Thread(target=apply_action, name="falcon-job-action", daemon=True).start()
