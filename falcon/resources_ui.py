"""Realtime cluster and node resource dashboard."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Callable, List, Mapping, Optional, Sequence

from rich import box
from rich.align import Align
from rich.cells import cell_len
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.containers import Container
from textual.css.query import NoMatches
from textual.errors import NoWidget
from textual.geometry import NULL_OFFSET, Region, Size
from textual.layout import ArrangeResult, Layout, WidgetPlacement
from textual.layouts.vertical import VerticalLayout
from textual.widgets import Static

from .cluster import (
    ClusterSnapshot,
    NodeSnapshot,
    WorkloadConsumer,
    is_system_consumer,
    is_system_namespace,
    natural_name_key,
)
from .dashboard_ui import (
    WIDE_LAYOUT_MIN_HEIGHT,
    WIDE_LAYOUT_MIN_WIDTH,
)
from .planning import gpu_model_order_key
from .resources_charts import (
    HISTORY_LIMIT,
    HISTORY_SECONDS,
    GPUHistoryPoint,
    allocation_colors,
    render_allocation_legend,
    render_gpu_history,
    render_namespace_pie,
)
from .resources_telemetry import GpuTelemetrySnapshot, allocation_snapshot
from .theme import (
    BACKGROUND,
    BORDER,
    CYAN,
    CYAN_2,
    GRAY,
    GREEN,
    MINIMUM_WIDTH,
    MUTED,
    PALETTE,
    RED,
    SELECTION,
    WHITE,
    YELLOW,
    configure_color,
    metric_color,
)

RESOURCE_VIEWS = ("nodes", "gpu-allocations")
RESOURCE_MINIMUM_HEIGHT = 20
CONSUMER_SORTS = ("namespace", "cpu", "memory", "gpu")
CONSUMER_SORT_LABELS = {
    "namespace": "Namespace",
    "cpu": "CPU",
    "memory": "Memory",
    "gpu": "GPU",
}
RESOURCE_VIEW_LABELS = {
    "nodes": "Nodes",
    "gpu-allocations": "GPU Allocations",
}
ALLOCATION_MODES = ("gpu", "memory", "cpu")

ALLOCATION_LEGEND_WIDTH = 24
# Panel borders consume two rows and the pie renderer needs five content rows
# before it can draw a filled footprint rather than a numeric fallback.
ALLOCATION_MIN_PIE_HEIGHT = 7
ALLOCATION_MIN_HISTORY_HEIGHT = 7
ALLOCATION_MIN_JOBS_HEIGHT = 5


@dataclass(frozen=True)
class AllocationGeometry:
    """Cell regions shared by allocation rendering and mouse hitboxes."""

    width: int
    height: int
    legend: Region
    pie: Region
    history: Region
    jobs: Region
    left_width: int
    right_width: int
    remainder_height: int = 0

    @property
    def signature(self) -> tuple[tuple[int, int, int, int], ...]:
        def value(region: Region) -> tuple[int, int, int, int]:
            return (region.x, region.y, region.width, region.height)

        return (
            value(self.legend),
            value(self.pie),
            value(self.history),
            value(self.jobs),
        )


class ResourcesBodyLayout(Layout):
    """Arrange Resources panes as pages or as the wide two-side view."""

    name = "falcon-resources-body"

    def __init__(self) -> None:
        self._vertical = VerticalLayout()

    def arrange(
        self,
        parent,
        children,
        size: Size,
        greedy: bool = True,
    ) -> ArrangeResult:
        if not getattr(parent.app, "_wide_resources_layout", False):
            return self._vertical.arrange(parent, children, size, greedy)

        parent.pre_layout(self)
        by_id = {child.id: child for child in children if child.id}
        if not by_id or size.width <= 0 or size.height <= 0:
            return []

        visible = {
            child_id: child
            for child_id, child in by_id.items()
            if child.display
        }
        if not visible:
            return []

        split = "gpu-allocations-pane" in visible and any(
            child_id in visible for child_id in ("nodes-pane", "node-pane")
        )
        left_width = size.width // 2 if split else size.width
        right_x = left_width if split else 0
        right_width = size.width - left_width if split else size.width
        placements: list[WidgetPlacement] = []

        def place(child_id: str, x: int, y: int, width: int, height: int) -> None:
            if child_id not in visible or width <= 0 or height <= 0:
                return
            widget = visible[child_id]
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

        if "gpu-allocations-pane" in visible:
            place(
                "gpu-allocations-pane",
                right_x,
                0,
                right_width,
                size.height,
            )

        left_children = [
            child_id
            for child_id in ("nodes-pane", "node-pane")
            if child_id in visible
        ]
        if left_children:
            if len(left_children) == 1:
                place(
                    left_children[0],
                    0,
                    0,
                    left_width,
                    size.height,
                )
            else:
                nodes = visible["nodes-pane"]
                if nodes.styles.height.is_cells:
                    nodes_height = int(nodes.styles.height.value)
                else:
                    nodes_height = size.height // 2
                nodes_height = max(1, min(size.height - 1, nodes_height))
                place("nodes-pane", 0, 0, left_width, nodes_height)
                place(
                    "node-pane",
                    0,
                    nodes_height,
                    left_width,
                    size.height - nodes_height,
                )

        return placements


class ResourcesBody(Container):
    """Container whose layout changes at the shared wide-view breakpoint."""

    def __init__(self, *children, **kwargs) -> None:
        super().__init__(*children, **kwargs)
        self._resources_layout = ResourcesBodyLayout()

    @property
    def layout(self) -> Layout:
        return self._resources_layout


def _valid_view(value: object) -> str:
    normalized = str(value or "")
    if normalized == "gpu-overview":
        # The retired overview is now represented responsively in Nodes.
        return "nodes"
    return normalized if normalized in RESOURCE_VIEWS else "nodes"


def _valid_consumer_sort(value: object) -> str:
    normalized = str(value or "")
    return normalized if normalized in CONSUMER_SORTS else "namespace"


def _eligible(node: NodeSnapshot) -> bool:
    return node.ready is True and node.schedulable


def _gpu_totals(nodes: Sequence[NodeSnapshot]) -> tuple[int, int, int, float]:
    eligible = [node for node in nodes if _eligible(node)]
    allocatable = sum(node.allocatable.gpu_count for node in eligible)
    requested = sum(node.requested.gpu_count for node in eligible)
    free = max(0, allocatable - requested)
    pressure = requested / allocatable * 100 if allocatable else 0.0
    return allocatable, requested, free, pressure


def _short_cpu(value: float) -> str:
    """Format CPU as decimal cores consistently across the resource view."""

    value = max(0.0, float(value))
    if value < 1:
        return f"{value:.3f}".rstrip("0").rstrip(".") or "0"
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _short_memory(value: int) -> str:
    gib = value / (1024**3)
    return f"{gib:.1f}G" if gib < 10 else f"{gib:.0f}G"


def _truncate(value: object, width: int) -> str:
    text = "-" if value is None else str(value)
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def _schedulable(node: NodeSnapshot) -> tuple[str, str]:
    if node.ready is False:
        return "Not ready", PALETTE.danger
    if node.ready is None:
        return "Unknown", PALETTE.warning
    if not node.schedulable:
        return "Cordoned", PALETTE.danger
    return "Yes", PALETTE.success


def _gpu_headroom_color(headroom: int, allocatable: int) -> str:
    if allocatable <= 0:
        return PALETTE.muted
    remaining = max(0, headroom)
    if remaining == 0:
        return PALETTE.danger
    # A single remaining GPU is not equally risky on every node: 1/2 and
    # 1/4 are cautionary, while 1/8 is already critical availability.
    if remaining * 4 < allocatable:
        return PALETTE.danger
    if remaining == 1 or remaining * 2 < allocatable:
        return PALETTE.warning
    return PALETTE.success


def _resource_headroom_color(free: float, allocatable: float) -> str:
    """Color remaining scheduler headroom using the CPU/memory pressure scale."""

    if allocatable <= 0:
        return PALETTE.muted
    percent_free = min(allocatable, max(0.0, free)) / allocatable * 100
    # These are the availability equivalents of the established request
    # pressure bands: >=80% requested is <=20% free, while <30% requested is
    # >70% free.
    if percent_free <= 20:
        return PALETTE.danger
    if percent_free <= 70:
        return PALETTE.warning
    return PALETTE.success


def _request_pressure_color(requested: float, allocatable: float) -> str:
    """Use Dashboard's green/yellow/red thresholds for every request bar."""

    if allocatable <= 0:
        return PALETTE.muted
    return metric_color(max(0.0, requested) / allocatable * 100)


@dataclass
class ResourcesViewState:
    view: str = "nodes"
    selected_node: str = ""
    node_scroll: int = 0
    expanded: bool = False
    selected_consumer: int = 0
    consumer_scroll: int = 0
    consumer_sort: str = "namespace"
    active_pane: str = "nodes"
    allocation_scroll: int = 0
    namespace_basis: str = "gpu"
    # ``m`` cycles the primary allocation metric. ``namespace_basis`` is only
    # the GPU sub-mode selected by ``v``; keep ``memory_basis`` for callers
    # that inspect the older state shape and to preserve the last CPU/memory
    # selection across a GPU pass.
    allocation_mode: str = "gpu"
    memory_basis: str = "memory"
    history_log_scale: bool = False
    expanded_panels: dict[str, str] = field(
        default_factory=lambda: {
            "gpu-allocations": "",
        }
    )
    selected_panels: dict[str, str] = field(
        default_factory=lambda: {
            "gpu-allocations": "history",
        }
    )
    focused_panes: dict[str, str] = field(
        default_factory=lambda: {
            "nodes": "nodes",
            "gpu-allocations": "gpu-allocations",
        }
    )


class ResourcesChrome(Static):
    """Non-scrollable Resources chrome that consumes terminal wheel input."""

    # Resources is an interactive monitor, not a text editor.  Letting
    # Textual start its built-in drag-to-select state on every Static widget
    # means a terminal that loses the matching MouseUp can leave the screen
    # in a captured-pointer state.  Subsequent presses then look like a dead
    # TUI (and some terminals show their text cursor at the last rendered
    # position).  Keep selection available to the shell after the app exits,
    # but never start it inside this screen.
    ALLOW_SELECT = False

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.prevent_default()
        event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.prevent_default()
        event.stop()


class ResourcesPane(Static):
    can_focus = True
    ALLOW_SELECT = False

    def _activate(self) -> None:
        pane = self.id.replace("-pane", "") if self.id else "nodes"
        callback = getattr(self.app, "pane_focused", None)
        if callback:
            callback(pane)

    def on_focus(self, event: events.Focus) -> None:
        # Textual posts focus messages to individual widget queues. A restore
        # event for the pane used before terminal blur can therefore arrive
        # after mouse-down has already focused another pane. Ignore it once it
        # is no longer the screen's real focus.
        if self.screen.focused is not self:
            return
        self._activate()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        # Activate on the first forwarded mouse event, rather than waiting for
        # mouse-up to synthesize a Click after terminal focus-in.
        self._activate()
        # Do not depend on Screen's implicit focus-on-click path.  In
        # browser/tmux terminals a press can be forwarded after an
        # AppFocus transition, or its release can be dropped.  Focusing here
        # makes the press authoritative and keeps refresh/layout callbacks
        # from restoring the previous pane.
        self.app.set_focus(self, scroll_visible=False)
        # Some tmux/iTerm combinations deliver the press immediately but
        # delay or omit the corresponding release while the pane is gaining
        # focus. Apply the same selection action on mouse-down so a click is
        # never lost. A normal Click repeats this idempotently.
        offset_for = getattr(event, "get_content_offset", None)
        offset = offset_for(self) if callable(offset_for) else None
        if offset is None:
            return
        if self.id == "gpu-allocations-pane":
            callback = getattr(self.app, "gpu_panel_selected", None)
            if callback:
                callback(self.id.replace("-pane", ""), offset)
        elif self.id == "nodes-pane":
            callback = getattr(self.app, "node_clicked", None)
            if callback:
                callback(offset)
        elif self.id == "node-pane":
            callback = getattr(self.app, "consumer_clicked", None)
            if callback:
                callback(offset)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.prevent_default()
        event.stop()
        if self.id == "node-pane":
            self.app.scroll_consumers(1)
        else:
            self.app.action_down()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.prevent_default()
        event.stop()
        if self.id == "node-pane":
            self.app.scroll_consumers(-1)
        else:
            self.app.action_up()

    def on_click(self, event: events.Click) -> None:
        self._activate()
        self.app.set_focus(self, scroll_visible=False)
        if self.id == "gpu-allocations-pane":
            callback = getattr(self.app, "gpu_panel_selected", None)
            if callback:
                callback(
                    self.id.replace("-pane", ""),
                    event.get_content_offset(self),
                )
            return
        if self.id == "node-pane":
            callback = getattr(self.app, "consumer_clicked", None)
            offset_for = getattr(event, "get_content_offset", None)
            if callback and callable(offset_for):
                callback(offset_for(self))
            return
        if self.id != "nodes-pane":
            return
        callback = getattr(self.app, "node_clicked", None)
        if callback:
            callback(event.get_content_offset(self))


class ResourcesViewSelector(ResourcesChrome):
    """Clickable view labels; keyboard navigation remains available globally."""

    def on_mouse_down(self, event: events.MouseDown) -> None:
        callback = getattr(self.app, "view_clicked", None)
        offset_for = getattr(event, "get_content_offset", None)
        offset = offset_for(self) if callable(offset_for) else None
        if callback and offset is not None:
            callback(offset.x)

    def on_click(self, event: events.Click) -> None:
        callback = getattr(self.app, "view_clicked", None)
        offset_for = getattr(event, "get_content_offset", None)
        offset = offset_for(self) if callable(offset_for) else None
        if callback and offset is not None:
            callback(offset.x)


CSS = f"""
Screen {{ background: {BACKGROUND}; color: {WHITE}; overflow: hidden; }}
Static {{ background: {BACKGROUND}; }}
#resources-header {{ height: 1; padding: 0 1; color: {WHITE}; }}
# The view selector is an overlay on the header row. Its horizontal offset is
# calculated from the terminal width in ``_render_views`` so the labels remain
# centered without consuming a second layout row.
#resources-views {{
    position: absolute;
    offset: 0 0;
    width: auto;
    height: 1;
    padding: 0 1;
    color: {GRAY};
    background: transparent;
}}
#cluster-overview {{ height: 2; border-bottom: solid {BORDER}; padding: 0 1; }}
#resource-controls {{ height: 1; padding: 0 1; color: {GRAY}; }}
#resources-body {{ width: 1fr; height: 1fr; }}
ResourcesPane {{
    border: solid {BORDER};
    padding: 0 1;
    color: {WHITE};
    overflow: hidden;
}}
ResourcesPane:focus {{ border: solid {CYAN}; }}
# The node table has no Rich edge spacer; keep its minimum at the smallest
# useful one-row table (border + header + separator + one data row).
#nodes-pane {{ height: 1fr; min-height: 5; }}
#node-pane {{ height: 9; min-height: 5; }}
#gpu-allocations-pane {{ height: 1fr; min-height: 8; }}
#resize-message {{ display: none; height: 1fr; content-align: center middle; color: {YELLOW}; }}
# Keep the footer in normal document flow. Docking it leaves a stale one-row
# virtual overflow after tmux briefly reports a shorter terminal on reattach,
# which lets the Resources screen scroll even though every pane is clipped.
# This mirrors Dashboard's footer layout.
#resources-footer {{ height: 1; padding: 0 1; color: {GRAY}; }}
"""


class FalconResourcesApp(App[None]):
    """Keyboard-first realtime view of schedulable request headroom."""

    # The fixed sections surrounding the two resource panes are one combined
    # header/view row, two overview rows, one controls row, and one footer. The node
    # table needs its header, separator, and two pane borders. The trailing
    # Rich spacer can be clipped without hiding a data row.
    _FIXED_LAYOUT_HEIGHT = 5
    _NODE_TABLE_OVERHEAD = 4
    _DETAIL_MIN_HEIGHT = 5
    _HISTORY_LOAD_INTERVAL = 2.0

    TITLE = "Falcon Resources"
    ENABLE_COMMAND_PALETTE = False
    CSS = CSS
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("up", "up", "Up", show=False),
        Binding("down", "down", "Down", show=False),
        Binding("k", "up", "Up", show=False),
        Binding("j", "down", "Down", show=False),
        Binding("pageup", "page_up", "Page up", show=False),
        Binding("pagedown", "page_down", "Page down", show=False),
        Binding("home", "home", "First", show=False),
        Binding("end", "end", "Last", show=False),
        Binding("left", "previous_view", "Previous view", show=False),
        Binding("right", "next_view", "Next view", show=False),
        Binding("tab", "next_pane", "Next pane", priority=True),
        Binding("shift+tab", "previous_pane", "Previous pane", priority=True),
        Binding("enter", "expand", "Expand"),
        Binding("escape", "collapse", "Back", show=False),
        Binding("s", "cycle_consumer_sort", "Sort consumers", show=False),
        Binding("v", "toggle_namespace_basis", "GPU/VRAM", show=False),
        Binding("m", "toggle_allocation_basis", "GPU/Memory/CPU", show=False),
        Binding("l", "toggle_history_scale", "History log scale", show=False),
        Binding("r", "refresh_data", "Refresh"),
    ]

    def __init__(
        self,
        collector,
        *,
        refresh_seconds: float = 1.0,
        node_filter: Optional[str] = None,
        gpu_filter: Optional[str] = None,
        clock: Optional[Callable[[ClusterSnapshot], str]] = None,
        history_clock: Optional[Callable[[], float]] = None,
        history_loader: Optional[Callable[[], Sequence[GPUHistoryPoint]]] = None,
        history_hours: float = 24.0,
        history_warning: str = "",
        initial_view: str = "nodes",
        persist_view: Optional[Callable[[str], object]] = None,
        initial_consumer_sort: str = "namespace",
        persist_consumer_sort: Optional[Callable[[str], object]] = None,
        telemetry_collector=None,
        telemetry_refresh_seconds: float = 5.0,
        color_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.color_mode = configure_color(self.console, color_mode)
        self.collector = collector
        self.refresh_seconds = refresh_seconds
        self.node_filter = (node_filter or "").lower()
        self.gpu_filter = (gpu_filter or "").lower()
        self.clock = clock or self._snapshot_clock
        self.history_clock = history_clock or time.time
        self.history_loader = history_loader
        self.history_hours = float(history_hours)
        self.history_warning = history_warning
        view = _valid_view(initial_view)
        self.state = ResourcesViewState(
            view=view,
            consumer_sort=_valid_consumer_sort(initial_consumer_sort),
            active_pane={
                "nodes": "nodes",
                "gpu-allocations": "gpu-allocations",
            }[view],
        )
        self.persist_view = persist_view
        self.persist_consumer_sort = persist_consumer_sort
        # Kept as a soft compatibility argument for callers of the preview
        # API. Resource allocation data comes from ``collector`` itself.
        del telemetry_collector, telemetry_refresh_seconds
        self.snapshot = ClusterSnapshot.empty()
        self.nodes: List[NodeSnapshot] = []
        self.history: list[GPUHistoryPoint] = []
        self._history_revision = 0
        self._transient_history_point: Optional[GPUHistoryPoint] = None
        self._history_load_error_notified = False
        self._last_history_load_at = 0.0
        self._history_loading = False
        self._history_results: (
            "queue.Queue[tuple[list[GPUHistoryPoint], Optional[Exception]]]"
        ) = queue.Queue(maxsize=1)
        self._load_persistent_history(notify=False)
        self.gpu_telemetry = GpuTelemetrySnapshot()
        self._refreshing = False
        self._results: "queue.Queue[ClusterSnapshot]" = queue.Queue(maxsize=1)
        self._last_terminal_size = (-1, -1)
        self._wide_resources_layout = False
        self._spinner = 0
        self._detail_auto_hidden = False
        self._layout_node_names: tuple[str, ...] = ()
        self._last_allocation_history_key: object = None
        self._view_hitboxes: list[tuple[int, int, str]] = []
        self._history_cache_key: object = None
        self._history_cache: Optional[Text] = None
        self._pie_cache_key: object = None
        self._pie_cache: Optional[Text] = None
        self._allocation_render_key: object = None
        self._allocation_geometry: Optional[AllocationGeometry] = None
        self._allocation_regions: dict[str, Region] = {}
        self._nodes_revision = 0
        self._gpu_consumers_cache_revision = -1
        self._gpu_consumers_cache_sort = ""
        self._gpu_consumers_cache: tuple[WorkloadConsumer, ...] = ()

    def compose(self) -> ComposeResult:
        yield ResourcesChrome(id="resources-header")
        yield ResourcesViewSelector(id="resources-views")
        yield ResourcesChrome(id="cluster-overview")
        yield ResourcesChrome(id="resource-controls")
        with ResourcesBody(id="resources-body"):
            yield ResourcesPane(id="nodes-pane")
            yield ResourcesPane(id="node-pane")
            yield ResourcesPane(id="gpu-allocations-pane")
        yield ResourcesChrome(id="resize-message")
        yield ResourcesChrome(id="resources-footer")

    def on_mount(self) -> None:
        self._restore_terminal_modes()
        self._set_titles()
        self._apply_layout(recompute_detail=True)
        self._request_update(force=True)
        self.set_interval(self.refresh_seconds, self._request_update)
        # Refreshes arrive at a one-to-five-second cadence in normal use. A
        # 5 Hz result poll is responsive enough for the TUI while avoiding a
        # needless 10 Hz wakeup/render loop. Resize events are handled by
        # ``on_resize``; this slower watcher remains only as a compatibility
        # fallback for Textual versions that do not bubble them reliably.
        self.set_interval(0.2, self._drain_results)
        # Resize events do the immediate work; this slower watcher remains a
        # compatibility fallback for terminals/Textual versions that fail to
        # bubble them.
        self.set_interval(0.25, self._check_terminal_size)
        self.set_interval(1.0, self._tick_clock)
        self._render_all()
        if self.history_warning:
            self.notify(self.history_warning, severity="warning", timeout=5)

    async def on_event(self, event: events.Event) -> None:
        """Keep wheel events outside panes inside the Resources screen.

        Textual forwards a wheel event to the widget under the pointer. Most
        of the screen is non-scrollable chrome, but the top edge can resolve
        directly to the Screen (or to the absolute view selector) instead of
        a pane. If that event is allowed to continue to the terminal, tmux
        may scroll its history rather than the Resources UI. Pane handlers
        still receive their own events and implement navigation below.
        """

        if (
            not event.is_forwarded
            and isinstance(event, (events.MouseScrollUp, events.MouseScrollDown))
        ):
            try:
                target, _ = self.get_widget_at(event.x, event.y)
            except NoWidget:
                target = None
            if not isinstance(target, ResourcesPane):
                event.prevent_default()
                event.stop()
                return
        await super().on_event(event)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        """Keep unhandled wheel events inside the Resources application."""

        event.prevent_default()
        event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        """Keep unhandled wheel events inside the Resources application."""

        event.prevent_default()
        event.stop()

    @property
    def _history_window_label(self) -> str:
        hours = self.history_hours
        rendered = str(int(hours)) if hours.is_integer() else f"{hours:g}"
        return f"persistent {rendered}h"

    def _history_render_signature(self) -> tuple[object, ...]:
        """Identify chart data without copying a 20,000-point history list."""

        if not self.history:
            return (self._history_revision, id(self.history), 0, None, None)
        return (
            self._history_revision,
            id(self.history),
            len(self.history),
            self.history[0],
            self.history[-1],
        )

    def on_unmount(self) -> None:
        close = getattr(self.collector, "close", None)
        if callable(close):
            close()

    def on_resize(self, event: events.Resize) -> None:
        # Textual dispatches Resize before the App's ``size`` property has
        # been committed on some terminal backends (notably after tmux
        # reattach). Applying our height calculations against that stale size
        # leaves a one-row virtual overflow until the polling fallback runs.
        # Defer the layout pass until the next refresh, when ``self.size`` and
        # the widget regions agree.
        self._restore_terminal_modes()
        self._last_terminal_size = (-1, -1)
        self.call_after_refresh(self._apply_resized_layout)

    def _restore_terminal_modes(self) -> None:
        """Re-assert modes that tmux may reset while reattaching a client.

        Textual enables these once when the driver starts.  A tmux server or
        terminal emulator can restore the pty with the cursor visible and
        mouse reporting disabled after a detach/reattach, while the Python
        process continues running.  Reasserting the modes is safe for normal
        resizes and keeps the cursor from appearing at the compositor's
        default origin (top-left).
        """

        driver = getattr(self, "_driver", None)
        write = getattr(driver, "write", None)
        if not callable(write) or getattr(driver, "is_headless", False):
            return
        try:
            # Hide the terminal cursor and continue receiving focus events.
            write("\x1b[?25l\x1b[?1004h")
            if getattr(driver, "_mouse", True):
                enable_mouse = getattr(driver, "_enable_mouse_support", None)
                if callable(enable_mouse):
                    enable_mouse()
                else:
                    # Keep this compatible with Textual drivers that don't
                    # expose LinuxDriver's private helper.
                    write("\x1b[?1000h\x1b[?1003h\x1b[?1015h\x1b[?1006h")
            flush = getattr(driver, "flush", None)
            if callable(flush):
                flush()
        except (AssertionError, OSError, RuntimeError):
            # During shutdown the writer thread may already be closed.  Mode
            # recovery is best-effort and must never terminate the TUI.
            return

    def _apply_resized_layout(self) -> None:
        if not self.is_mounted:
            return
        self._last_terminal_size = (self.size.width, self.size.height)
        try:
            self.screen.scroll_to(y=0, animate=False)
            self._apply_layout(recompute_detail=True)
            # Setting pane heights invalidates their content geometry. Wait
            # for Textual to commit those heights before calculating visible
            # rows; doing it in this same callback reuses the previous size
            # and can leave selection or colour-bar rows clipped after resize.
            self.call_after_refresh(self._finish_resized_layout)
        except NoMatches:
            # Resize events can arrive as Textual is tearing down the screen.
            return

    def _finish_resized_layout(self) -> None:
        if not self.is_mounted:
            return
        try:
            self.screen.scroll_to(y=0, animate=False)
            self._ensure_visible()
            self._render_all()
        except NoMatches:
            return

    def _check_terminal_size(self) -> None:
        current = (self.size.width, self.size.height)
        if current == self._last_terminal_size:
            return
        self._restore_terminal_modes()
        self._last_terminal_size = current
        try:
            self.screen.scroll_to(y=0, animate=False)
            self._apply_layout(recompute_detail=True)
            self.call_after_refresh(self._finish_resized_layout)
        except NoMatches:
            # The polling timer may tick after the default screen unmounts.
            return

    def _tick_clock(self) -> None:
        self._spinner = (self._spinner + 1) % 4
        try:
            self._render_header()
        except Exception:
            # A timer may race with Textual tearing down the default screen.
            return

    def pane_focused(self, pane: str) -> None:
        if self._wide_resources_layout:
            if pane == "gpu-allocations":
                self.state.view = "gpu-allocations"
            elif pane in {"nodes", "node"}:
                self.state.view = "nodes"
            else:
                return
            self.state.active_pane = pane
            self.state.focused_panes[self.state.view] = pane
            self._set_titles()
            self._render_footer()
            return
        valid = {
            "nodes": {"nodes", "node"},
            "gpu-allocations": {"gpu-allocations"},
        }
        if pane not in valid[self.state.view]:
            return
        self.state.active_pane = pane
        self.state.focused_panes[self.state.view] = pane
        self._set_titles()

    def watch_app_focus(self, focused: bool) -> None:
        """Remove pane emphasis while the terminal window is inactive."""

        if not self.is_mounted:
            return
        try:
            self._restore_terminal_modes()
            if focused:
                # A terminal reattach can restore the Screen with the
                # previous scroll offset even though Resources has no root
                # scrollable content. Always return the application viewport
                # to its origin before restoring pane emphasis.
                self.screen.scroll_to(y=0, animate=False)
            self._set_titles()
            if self.state.view == "nodes":
                self._render_nodes()
        except NoMatches:
            return

    def view_clicked(self, x: int) -> None:
        for start, end, view in self._view_hitboxes:
            if start <= x < end:
                self._switch_view(view)
                return

    def gpu_panel_selected(self, view: str, offset) -> None:
        """Select the Rich sub-pane under a mouse click.

        Mouse interaction follows the rest of the Resources screen: clicking
        changes focus/selection only. ``Enter`` performs the expansion.
        """

        if view not in self.state.selected_panels or offset is None:
            return
        x, y = max(0, int(offset.x)), max(0, int(offset.y))
        panel = ""
        if view == "gpu-allocations":
            expanded = self.state.expanded_panels.get(view, "")
            geometry = self._allocation_geometry_for(
                expanded=expanded
            )
            pie_region = self._allocation_pie_hit_region(geometry, expanded)
            self._allocation_geometry = geometry
            self._allocation_regions = {
                name: region
                for name, region in (
                    ("legend", geometry.legend),
                    ("pie", pie_region),
                    ("history", geometry.history),
                    ("jobs", geometry.jobs),
                )
                if region.width > 0 and region.height > 0
            }

            def contains(region: Region) -> bool:
                return (
                    region.width > 0
                    and region.height > 0
                    and region.x <= x < region.x + region.width
                    and region.y <= y < region.y + region.height
                )

            if contains(geometry.legend) or contains(pie_region):
                # The legend is part of the pie stack and selects Pie for
                # expansion; its own fixed width remains a real hitbox.
                panel = "pie"
            elif contains(geometry.history):
                panel = "history"
            elif contains(geometry.jobs):
                panel = "pods"
        if panel:
            self.state.selected_panels[view] = panel
            if self._wide_resources_layout:
                self.state.view = "gpu-allocations"
                self.state.active_pane = "gpu-allocations"
                self.state.focused_panes["gpu-allocations"] = "gpu-allocations"
        self._render_all()

    def _switch_view(self, view: str) -> None:
        view = _valid_view(view)
        combined = (
            self._is_wide_resources()
            and not self.state.expanded
            and not self.state.expanded_panels.get("gpu-allocations", "")
        )
        previous_view = self.state.view
        if view == previous_view and not combined:
            return
        if previous_view in self.state.focused_panes:
            self.state.focused_panes[previous_view] = self.state.active_pane
        self.state.view = view
        if view == "gpu-allocations":
            self.state.active_pane = "gpu-allocations"
        else:
            self.state.active_pane = self.state.focused_panes[view]
        if self._allocation_is_visible() or view == "gpu-allocations":
            # The detached collector keeps writing while Nodes is visible.
            # Refresh the persisted window whenever the allocation pane is
            # visible, including when Nodes is the active side in wide mode.
            self._load_persistent_history(force=True)
        self._apply_layout(recompute_detail=False)
        # Page geometry is committed on the next refresh. Render only shared
        # chrome now so charts are not built once at stale hidden dimensions
        # and immediately rebuilt at their real size.
        self._set_titles()
        self._render_header()
        self._render_views()
        self._render_footer()
        self.call_after_refresh(self._finish_view_switch)
        if self.persist_view is not None and view != previous_view:
            try:
                self.persist_view(view)
            except Exception as exc:
                self.notify(
                    f"Could not save Resources view: {exc}",
                    severity="warning",
                    timeout=4,
                )

    def _finish_view_switch(self) -> None:
        self._ensure_visible()
        self._render_all()

    def action_previous_view(self) -> None:
        if self._is_wide_resources():
            return
        index = RESOURCE_VIEWS.index(self.state.view)
        self._switch_view(RESOURCE_VIEWS[(index - 1) % len(RESOURCE_VIEWS)])

    def action_next_view(self) -> None:
        if self._is_wide_resources():
            return
        index = RESOURCE_VIEWS.index(self.state.view)
        self._switch_view(RESOURCE_VIEWS[(index + 1) % len(RESOURCE_VIEWS)])

    def _focusable_panes(self) -> tuple[str, ...]:
        """Return visible top-level panes for Tab navigation."""

        if (
            self.size.width < MINIMUM_WIDTH
            or self.size.height < RESOURCE_MINIMUM_HEIGHT
        ):
            return ()
        if self._wide_resources_layout:
            panes: list[str] = []
            if self.query_one("#gpu-allocations-pane", ResourcesPane).display:
                panes.extend(("history", "pie", "pods"))
            if self.query_one("#nodes-pane", ResourcesPane).display:
                panes.append("nodes")
            if self.query_one("#node-pane", ResourcesPane).display:
                panes.append("node")
            return tuple(panes)
        if self.state.view == "gpu-allocations":
            # Allocation is rendered as one Textual pane containing three
            # selectable Rich panels.  Keep focus on the outer widget (so
            # Tab cannot leak into the terminal), while cycling the inner
            # selection used for the cyan focus border and Enter expansion.
            return ("history", "pie", "pods")
        if self.state.expanded:
            return ("node",)
        try:
            detail = self.query_one("#node-pane", ResourcesPane)
        except NoMatches:
            return ("nodes",)
        return ("nodes", "node") if detail.display else ("nodes",)

    def _cycle_pane(self, amount: int) -> None:
        panes = self._focusable_panes()
        if not panes:
            return
        if self._wide_resources_layout:
            current = (
                self.state.selected_panels.get("gpu-allocations", "history")
                if self.state.active_pane == "gpu-allocations"
                else self.state.active_pane
            )
            if current not in panes:
                current = panes[0]
            target_name = panes[(panes.index(current) + amount) % len(panes)]
            if target_name in {"history", "pie", "pods"}:
                self.state.selected_panels["gpu-allocations"] = target_name
                self.state.active_pane = "gpu-allocations"
                self.state.view = "gpu-allocations"
                target = self.query_one("#gpu-allocations-pane", ResourcesPane)
            else:
                self.state.active_pane = target_name
                self.state.view = "nodes"
                self.state.focused_panes["nodes"] = target_name
                target = self.query_one(f"#{target_name}-pane", ResourcesPane)
            self.state.focused_panes[self.state.view] = self.state.active_pane
            self.set_focus(target, scroll_visible=False)
            self._set_titles()
            self._render_all()
            return
        if self.state.view == "gpu-allocations":
            current = self.state.selected_panels.get("gpu-allocations", "history")
            if current not in panes:
                current = panes[0]
            panel = panes[(panes.index(current) + amount) % len(panes)]
            self.state.selected_panels["gpu-allocations"] = panel
            if self.state.expanded_panels["gpu-allocations"]:
                # Match Dashboard behavior: Tab remains useful while a pane
                # is expanded by moving the expanded content with focus.
                self.state.expanded_panels["gpu-allocations"] = panel
            self.state.active_pane = "gpu-allocations"
            self.state.focused_panes["gpu-allocations"] = "gpu-allocations"
            self.set_focus(
                self.query_one("#gpu-allocations-pane", ResourcesPane),
                scroll_visible=False,
            )
            self._render_all()
            return
        current = self.state.active_pane
        if current not in panes:
            current = panes[0]
        pane = panes[(panes.index(current) + amount) % len(panes)]
        self.state.active_pane = pane
        self.state.focused_panes[self.state.view] = pane
        target = self.query_one(f"#{pane}-pane", ResourcesPane)
        self.set_focus(target, scroll_visible=False)
        self._render_all()

    def action_next_pane(self) -> None:
        self._cycle_pane(1)

    def action_previous_pane(self) -> None:
        self._cycle_pane(-1)

    @staticmethod
    def _snapshot_clock(snapshot: ClusterSnapshot) -> str:
        if not snapshot.collected_at:
            return "--:--:--"
        timestamp = float(snapshot.collected_at)
        # MetricsClusterCollector uses a monotonic clock for cache cadence.
        # Keep the machine-readable snapshot untouched, but do not render a
        # monotonic value as a date in 1970 in the human TUI header.
        if timestamp < 946_684_800:  # 2000-01-01 UTC
            timestamp = time.time()
        return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

    def _request_update(self, force: bool = False) -> None:
        if self._refreshing:
            return
        self._refreshing = True

        def collect() -> None:
            try:
                try:
                    value = self.collector.collect(force=force)
                except TypeError:
                    value = self.collector.collect()
                if not isinstance(value, ClusterSnapshot):
                    raise TypeError("resource collector did not return ClusterSnapshot")
            except Exception as exc:
                value = self.snapshot.mark_stale(f"{type(exc).__name__}: {exc}")
            try:
                self._results.put_nowait(value)
            except queue.Full:
                pass

        threading.Thread(
            target=collect,
            name="falcon-resources-refresh",
            daemon=True,
        ).start()

    def _drain_results(self) -> None:
        self._drain_history_results()
        try:
            snapshot = self._results.get_nowait()
        except queue.Empty:
            return
        self._refreshing = False
        # ClusterCollector and MetricsClusterCollector return the exact same
        # object while their inventory cache is warm. Do not rebuild every
        # Rich table, overview, and history cache for that unchanged frame.
        if snapshot is self.snapshot:
            return
        previous_node_name = self.state.selected_node
        self.snapshot = snapshot
        nodes = sorted(snapshot.nodes, key=lambda node: natural_name_key(node.name))
        if self.node_filter:
            nodes = [node for node in nodes if self.node_filter in node.name.lower()]
        if self.gpu_filter:
            nodes = [
                node for node in nodes
                if self.gpu_filter in (node.gpu_model or "").lower()
            ]
        previous_names = self._layout_node_names
        nodes_changed = nodes != self.nodes
        previous_selected_key = None
        previous_anchor_key = None
        if nodes_changed:
            previous_node = self._selected()
            previous_consumers = self._sorted_consumers(previous_node)
            previous_selected_key = (
                self._consumer_identity(previous_consumers[self.state.selected_consumer])
                if previous_consumers
                and 0 <= self.state.selected_consumer < len(previous_consumers)
                else None
            )
            previous_anchor_key = (
                self._consumer_identity(previous_consumers[self.state.consumer_scroll])
                if previous_consumers
                and 0 <= self.state.consumer_scroll < len(previous_consumers)
                else None
            )
        self.nodes = nodes
        if nodes_changed:
            self._nodes_revision += 1
        # The resource collector already runs in the background against the
        # The collector refreshes the configured local metrics endpoint in the
        # background. Derive both selectable allocation bases from every fresh
        # snapshot, even while this page is hidden.
        if nodes_changed:
            self.gpu_telemetry = allocation_snapshot(
                self.nodes,
                collected_at=snapshot.collected_at,
                stale=snapshot.stale,
                error=snapshot.error or "",
            )
        else:
            # A fresh collector timestamp still represents a distinct history
            # sample, but the allocation totals are unchanged. Avoid walking
            # every consumer again just to update stale/error metadata.
            self.gpu_telemetry = replace(
                self.gpu_telemetry,
                collected_at=snapshot.collected_at,
                stale=snapshot.stale,
                error=snapshot.error or "",
            )
        if self._allocation_is_visible():
            self._load_persistent_history()
        self._record_gpu_history(self.gpu_telemetry)
        current_names = tuple(node.name for node in nodes)
        names = {node.name for node in nodes}
        if self.state.selected_node not in names:
            self.state.selected_node = nodes[0].name if nodes else ""
            self.state.selected_consumer = 0
            self.state.consumer_scroll = 0
        elif self.state.selected_node == previous_node_name:
            refreshed_node = self._selected()
            refreshed_consumers = self._sorted_consumers(refreshed_node)
            if previous_selected_key is not None:
                for index, consumer in enumerate(refreshed_consumers):
                    if self._consumer_identity(consumer) == previous_selected_key:
                        self.state.selected_consumer = index
                        break
            if previous_anchor_key is not None:
                for index, consumer in enumerate(refreshed_consumers):
                    if self._consumer_identity(consumer) == previous_anchor_key:
                        self.state.consumer_scroll = index
                        break
        try:
            self._apply_layout(
                recompute_detail=current_names != previous_names,
            )
            self._ensure_visible()
            self._render_all()
        except NoMatches:
            return

    def _gpu_nodes(self) -> list[NodeSnapshot]:
        return [
            node
            for node in self.nodes
            if (
                node.capacity.gpu_count
                or node.allocatable.gpu_count
                or node.requested.gpu_count
                or node.gpu_model
            )
        ]

    def _record_gpu_history(self, telemetry: GpuTelemetrySnapshot) -> None:
        """Append one point per fresh allocation snapshot and prune the window."""

        if telemetry.stale:
            return
        identity = (
            telemetry.collected_at,
            telemetry.effective_gpus_by_namespace,
            telemetry.vram_gib_by_namespace,
            telemetry.cpu_cores_by_namespace,
            telemetry.memory_gib_by_namespace,
        )
        if identity == self._last_allocation_history_key:
            return
        self._last_allocation_history_key = identity
        values = dict(telemetry.effective_gpus_by_namespace)
        vram_values = dict(telemetry.vram_gib_by_namespace)
        cpu_values = dict(telemetry.cpu_cores_by_namespace)
        memory_values = dict(telemetry.memory_gib_by_namespace)
        if not (values or cpu_values or memory_values):
            return
        timestamp = float(telemetry.collected_at)
        if timestamp <= 0:
            return
        # Test and adapter collectors may use monotonic cache timestamps.
        if timestamp < 946_684_800:  # 2000-01-01 UTC
            timestamp = float(self.history_clock())
        point = GPUHistoryPoint.from_mapping(
            timestamp,
            values,
            vram_values,
            cpu_values,
            memory_values,
        )
        self._transient_history_point = point
        history_changed = False
        if point not in self.history:
            self.history.append(point)
            history_changed = True
        self.history.sort(key=lambda point: point.timestamp)
        newest = self.history[-1].timestamp
        cutoff = newest - HISTORY_SECONDS
        retained = [point for point in self.history if point.timestamp >= cutoff]
        if len(retained) != len(self.history):
            history_changed = True
        self.history = retained
        if len(self.history) > HISTORY_LIMIT:
            self.history = self.history[-HISTORY_LIMIT:]
            history_changed = True
        if history_changed:
            self._history_revision += 1

    def _load_persistent_history(
        self,
        *,
        notify: bool = True,
        force: bool = False,
    ) -> None:
        """Replace the chart window with the background collector's history."""

        if self.history_loader is None:
            return
        if getattr(self, "_is_mounted", False) and not force:
            now = time.monotonic()
            if now - self._last_history_load_at < self._HISTORY_LOAD_INTERVAL:
                return
        self._last_history_load_at = time.monotonic()
        if getattr(self, "_is_mounted", False):
            if self._history_loading:
                return
            self._history_loading = True

            def load() -> None:
                try:
                    loaded = list(self.history_loader())
                    result = (loaded, None)
                except Exception as exc:
                    result = ([], exc)
                try:
                    self._history_results.put_nowait(result)
                except queue.Full:
                    pass

            threading.Thread(
                target=load,
                name="falcon-resources-history-read",
                daemon=True,
            ).start()
            return
        try:
            loaded = list(self.history_loader())
            self._apply_loaded_history(loaded)
        except Exception as exc:
            if notify and not self._history_load_error_notified and self.is_mounted:
                self.notify(
                    f"Could not load Resources history: {exc}",
                    severity="warning",
                    timeout=4,
                )
                self._history_load_error_notified = True

    def _apply_loaded_history(self, loaded: Sequence[GPUHistoryPoint]) -> None:
        transient = self._transient_history_point
        values = list(loaded)
        if transient is not None and transient not in values:
            values.append(transient)
            values.sort(key=lambda point: point.timestamp)
        if values != self.history:
            self.history = values
            self._history_revision += 1
        self._history_load_error_notified = False

    def _drain_history_results(self) -> None:
        if not self._history_loading:
            return
        try:
            loaded, error = self._history_results.get_nowait()
        except queue.Empty:
            return
        self._history_loading = False
        if error is not None:
            if not self._history_load_error_notified and self.is_mounted:
                self.notify(
                    f"Could not load Resources history: {error}",
                    severity="warning",
                    timeout=4,
                )
                self._history_load_error_notified = True
            return
        self._apply_loaded_history(loaded)
        if self._allocation_is_visible():
            self._render_gpu_allocations()

    def _selected(self) -> Optional[NodeSnapshot]:
        return next(
            (node for node in self.nodes if node.name == self.state.selected_node),
            None,
        )

    @staticmethod
    def _consumer_identity(consumer) -> tuple[str, ...]:
        """Return a stable workload identity across telemetry refreshes."""

        return (
            consumer.namespace,
            consumer.workload_kind,
            consumer.workload_name,
            consumer.pod_name,
            consumer.node_name,
        )

    def _consumer_sort_key(self, consumer):
        """Return the configured stable key for any visible workload."""

        namespace = natural_name_key(consumer.namespace)
        workload = natural_name_key(
            consumer.workload_name or consumer.pod_name
        )
        pod = natural_name_key(consumer.pod_name)
        node = natural_name_key(consumer.node_name)
        sort = _valid_consumer_sort(self.state.consumer_sort)
        if sort == "cpu":
            return (
                -float(consumer.requested.cpu_cores),
                namespace,
                workload,
                pod,
                node,
            )
        if sort == "memory":
            return (
                -int(consumer.requested.memory_bytes),
                namespace,
                workload,
                pod,
                node,
            )
        if sort == "gpu":
            return (
                -int(consumer.requested.gpu_count),
                namespace,
                workload,
                pod,
                node,
            )
        return (namespace, workload, pod, node)

    def _sorted_consumers(
        self, node: Optional[NodeSnapshot]
    ) -> tuple[WorkloadConsumer, ...]:
        if node is None:
            return ()
        return tuple(sorted(node.visible_consumers, key=self._consumer_sort_key))

    def _selected_index(self) -> int:
        for index, node in enumerate(self.nodes):
            if node.name == self.state.selected_node:
                return index
        return 0

    def _visible_nodes(self) -> int:
        # With ``show_edge=False`` SIMPLE_HEAD contributes the header and its
        # separator before the data rows. The trailing spacer is allowed to
        # clip so the last node still fits in the fixed inventory height.
        pane = self.query_one("#nodes-pane")
        # Use the configured cell height when available. During the first
        # refresh Textual may not have committed the new region yet, while
        # the inline numeric height is already authoritative.
        if pane.styles.height.is_cells:
            return max(1, int(pane.styles.height.value) - self._NODE_TABLE_OVERHEAD)
        return max(1, pane.content_size.height - 2)

    def _node_inventory_height(self) -> int:
        """Return the outer height needed to show every node row."""

        return max(self._NODE_TABLE_OVERHEAD, len(self.nodes) + self._NODE_TABLE_OVERHEAD)

    def _visible_consumers(self) -> int:
        pane = self.query_one("#node-pane")
        if not self.state.expanded:
            # The compact pane reserves one row for the node summary.
            return max(1, pane.content_size.height - 1)
        # The expanded view contains a fixed capacity panel followed by the
        # consumer Panel border, one spacer row, the table header, and its
        # separator. Keep this calculation identical to _render_node so the
        # selection/scroll viewport cannot extend beneath the visible panel.
        return max(
            1,
            pane.content_size.height - self._expanded_facts_height() - 5,
        )

    def _expanded_facts_height(self) -> int:
        """Rendered height of the capacity Panel, including its border."""

        try:
            width = self.query_one("#node-pane", ResourcesPane).content_size.width
        except NoMatches:
            width = self.size.width
        return 8 if width < 100 else 9

    def _node_pane_width(self) -> int:
        try:
            return max(
                1,
                self.query_one("#node-pane", ResourcesPane).content_size.width,
            )
        except NoMatches:
            return max(1, self.size.width)

    def _ensure_visible(self) -> None:
        if not self.is_mounted:
            return
        if self.state.view == "gpu-allocations":
            visible = self._allocation_visible_rows()
            self.state.allocation_scroll = min(
                max(0, len(self._gpu_consumers()) - visible),
                max(0, self.state.allocation_scroll),
            )
            return
        index = self._selected_index()
        count = self._visible_nodes()
        if index < self.state.node_scroll:
            self.state.node_scroll = index
        elif index >= self.state.node_scroll + count:
            self.state.node_scroll = index - count + 1
        self.state.node_scroll = min(
            max(0, len(self.nodes) - count),
            max(0, self.state.node_scroll),
        )
        consumers = self._sorted_consumers(self._selected())
        self.state.selected_consumer = min(
            max(0, self.state.selected_consumer),
            max(0, len(consumers) - 1),
        )
        visible = self._visible_consumers()
        if self.state.expanded or (
            self._wide_resources_layout and self.state.active_pane == "node"
        ):
            if self.state.selected_consumer < self.state.consumer_scroll:
                self.state.consumer_scroll = self.state.selected_consumer
            elif self.state.selected_consumer >= self.state.consumer_scroll + visible:
                self.state.consumer_scroll = self.state.selected_consumer - visible + 1
        self.state.consumer_scroll = min(
            max(0, len(consumers) - visible),
            max(0, self.state.consumer_scroll),
        )

    def _is_wide_resources(self) -> bool:
        return (
            self.size.width >= WIDE_LAYOUT_MIN_WIDTH
            and self.size.height >= WIDE_LAYOUT_MIN_HEIGHT
        )

    def _allocation_is_visible(self) -> bool:
        if (
            self.size.width < MINIMUM_WIDTH
            or self.size.height < RESOURCE_MINIMUM_HEIGHT
        ):
            return False
        return self._wide_resources_layout or self.state.view == "gpu-allocations"

    @staticmethod
    def _allocation_stack_heights(height: int) -> tuple[int, int]:
        """Split the independent history/jobs stack without underflowing it.

        History benefits from the larger share because its chart remains useful
        as it gains vertical resolution, while the jobs table is intentionally
        a compact list. At the minimum pane height the pie takes precedence;
        that constraint can force the two right-hand panels to be equal.
        """

        height = max(1, int(height))
        target_history = max(
            ALLOCATION_MIN_HISTORY_HEIGHT,
            (height * 55 + 99) // 100,
        )
        # The legend shares this height and the pie must retain a drawable
        # footprint, so cap History before it can consume the whole left side.
        history = min(target_history, max(1, height - ALLOCATION_MIN_PIE_HEIGHT))
        return history, max(1, height - history)

    def _allocation_legend_height(
        self,
        categories: Sequence[tuple[str, float]],
        *,
        basis: Optional[str] = None,
        max_height: Optional[int] = None,
    ) -> int:
        """Return the natural outer height of the fixed-width legend."""

        unit = self._allocation_unit(basis or self._allocation_basis())
        colors = self._allocation_colors(categories)
        # The panel's two border rows and two horizontal padding cells are not
        # part of the renderer's text width. A generous row cap is enough to
        # measure all currently supported categories without adding filler.
        lines = render_allocation_legend(
            categories,
            width=max(1, ALLOCATION_LEGEND_WIDTH - 4),
            height=max(1, len(categories) + 2),
            unit=unit,
            colors=colors,
            columns=1,
            include_total=True,
        )
        natural = max(3, len(lines.plain.splitlines()) + 2)
        if max_height is not None:
            return max(1, min(natural, int(max_height)))
        return natural

    def _allocation_geometry_for(
        self,
        *,
        categories: Optional[Sequence[tuple[str, float]]] = None,
        expanded: str = "",
    ) -> AllocationGeometry:
        """Compute all allocation panel regions in pane-content coordinates."""

        pane = self.query_one("#gpu-allocations-pane")
        width = max(1, int(pane.content_size.width))
        height = max(1, int(pane.content_size.height))
        categories = tuple(categories if categories is not None else self._namespace_categories())

        empty = Region(0, 0, 0, 0)
        if expanded == "pods":
            return AllocationGeometry(
                width,
                height,
                empty,
                empty,
                empty,
                Region(0, 0, width, height),
                width,
                0,
            )

        if expanded == "history":
            history = Region(
                min(ALLOCATION_LEGEND_WIDTH, width),
                0,
                max(1, width - ALLOCATION_LEGEND_WIDTH),
                height,
            )
            return AllocationGeometry(
                width,
                height,
                Region(0, 0, min(ALLOCATION_LEGEND_WIDTH, width), height),
                empty,
                history,
                empty,
                width,
                0,
                0,
            )

        if expanded == "pie":
            available_width = max(1, width - ALLOCATION_LEGEND_WIDTH)
            desired_width = max(1, 2 * height)
            pie_width = min(available_width, desired_width)
            pie_height = height
            if pie_width < desired_width:
                pie_height = max(ALLOCATION_MIN_PIE_HEIGHT, pie_width // 2)
            pie = Region(
                min(ALLOCATION_LEGEND_WIDTH, width),
                0,
                max(1, pie_width),
                max(1, pie_height),
            )
            return AllocationGeometry(
                width,
                height,
                Region(
                    0,
                    0,
                    min(ALLOCATION_LEGEND_WIDTH, width),
                    pie_height,
                ),
                pie,
                empty,
                empty,
                width,
                0,
                max(0, height - pie_height),
            )

        history_height, jobs_height = self._allocation_stack_heights(height)
        # Keep the two top panels aligned. The legend renderer still emits
        # only its Total/category rows; the remaining panel area is ordinary
        # breathing room rather than blank legend rows.
        legend_height = history_height
        available_left = max(1, width - 44)
        pie_height = max(1, height - legend_height)
        desired_pie_width = max(1, 2 * pie_height)
        pie_width = min(desired_pie_width, available_left)
        remainder = 0
        if pie_width < desired_pie_width:
            # Width is the limiting dimension. Recompute the height from that
            # width instead of stretching a visually circular pie vertically.
            pie_height = max(ALLOCATION_MIN_PIE_HEIGHT, pie_width // 2)
            remainder = max(0, height - legend_height - pie_height)
        left_width = max(ALLOCATION_LEGEND_WIDTH, pie_width)
        right_width = max(1, width - left_width)
        history_x = min(ALLOCATION_LEGEND_WIDTH, width)
        history_width = max(1, width - history_x)
        return AllocationGeometry(
            width,
            height,
            Region(0, 0, min(ALLOCATION_LEGEND_WIDTH, width), legend_height),
            Region(0, legend_height, pie_width, pie_height),
            # History has its own horizontal span. It starts at the fixed
            # legend edge even when the aspect-correct pie below is wider
            # than the legend, so it is never constrained to the Jobs width.
            Region(history_x, 0, history_width, history_height),
            Region(left_width, history_height, right_width, jobs_height),
            left_width,
            right_width,
            remainder,
        )

    @staticmethod
    def _allocation_pie_hit_region(
        geometry: AllocationGeometry,
        expanded: str,
    ) -> Region:
        """Include the aligned outer pie panel in its mouse hitbox."""

        if expanded:
            return geometry.pie
        return Region(
            geometry.pie.x,
            geometry.pie.y,
            max(geometry.pie.width, geometry.left_width),
            geometry.pie.height,
        )

    def _allocation_layout(self) -> tuple[int, int]:
        """Compatibility tuple for callers that need history/jobs heights."""

        geometry = self._allocation_geometry_for(
            expanded=self.state.expanded_panels.get("gpu-allocations", "")
        )
        return geometry.history.height or geometry.height, geometry.jobs.height or geometry.height

    def _allocation_visible_rows(self) -> int:
        expanded = self.state.expanded_panels["gpu-allocations"]
        geometry = self._allocation_geometry_for(expanded=expanded)
        # The surrounding Panel consumes two border rows and the table uses
        # one header row plus its separator. Its internal top edge is disabled,
        # so the remaining row can display one more GPU-requesting Job.
        return max(1, geometry.jobs.height - 4)

    def _apply_layout(self, *, recompute_detail: bool = False) -> None:
        if not self.is_mounted:
            return
        small = (
            self.size.width < MINIMUM_WIDTH
            or self.size.height < RESOURCE_MINIMUM_HEIGHT
        )
        resize = self.query_one("#resize-message")
        body = self.query_one("#resources-body", ResourcesBody)
        ids = (
            "cluster-overview",
            "resource-controls",
            "nodes-pane",
            "node-pane",
            "gpu-allocations-pane",
        )
        if small:
            self._wide_resources_layout = False
            body.display = False
            for identifier in ids:
                self.query_one(f"#{identifier}").display = False
            self.query_one("#resources-views").display = False
            resize.display = True
            resize.update(
                f"Falcon Resources requires at least "
                f"{MINIMUM_WIDTH}×{RESOURCE_MINIMUM_HEIGHT}.\n"
                f"Current terminal: {self.size.width}×{self.size.height}.\n\n"
                "Resize to inspect cluster resources."
            )
            return
        resize.display = False
        for identifier in ids:
            self.query_one(f"#{identifier}").display = False

        expanded_allocation = self.state.expanded_panels.get(
            "gpu-allocations", ""
        )
        # Once a pane is expanded, Esc is the unambiguous way back to the
        # responsive Resources layout. The Nodes/GPU Allocations selector is
        # unnecessary chrome while the body is dedicated to one pane.
        self.query_one("#resources-views").display = not (
            self.state.expanded or expanded_allocation
        )
        wide = self._is_wide_resources() and not self.state.expanded and not expanded_allocation
        self._wide_resources_layout = wide
        body.display = True

        nodes_pane = self.query_one("#nodes-pane", ResourcesPane)
        detail = self.query_one("#node-pane", ResourcesPane)
        allocation = self.query_one("#gpu-allocations-pane", ResourcesPane)
        overview = self.query_one("#cluster-overview")
        controls = self.query_one("#resource-controls")

        if wide:
            # All chrome remains full width while the body container owns the
            # equal-width allocation/node split.
            body.styles.height = max(1, self.size.height - self._FIXED_LAYOUT_HEIGHT)
            overview.display = True
            controls.display = True
            allocation.display = True
            nodes_pane.display = True
            detail.border_title = " SELECTED NODE "
            self._layout_node_names = tuple(node.name for node in self.nodes)

            available = max(0, self.size.height - self._FIXED_LAYOUT_HEIGHT)
            required = self._node_inventory_height()
            if recompute_detail:
                self._detail_auto_hidden = (
                    available - required < self._DETAIL_MIN_HEIGHT
                )
            if self._detail_auto_hidden:
                detail.display = False
                nodes_height = available
            else:
                detail.display = True
                nodes_height = min(required, max(1, available - self._DETAIL_MIN_HEIGHT))
            nodes_pane.styles.height = max(1, nodes_height)
            detail.styles.height = max(
                self._DETAIL_MIN_HEIGHT,
                available - nodes_height,
            )
            if not detail.display and self.state.active_pane == "node":
                self.state.active_pane = "nodes"
                self.state.focused_panes["nodes"] = "nodes"
            focus_id = {
                "gpu-allocations": allocation,
                "node": detail,
                "nodes": nodes_pane,
            }.get(self.state.active_pane, nodes_pane)
            if not focus_id.display:
                focus_id = nodes_pane if nodes_pane.display else allocation
            if self.app_focus and self.focused is not focus_id:
                self.set_focus(focus_id, scroll_visible=False)
            if self._allocation_is_visible():
                self._load_persistent_history()
            return

        # Expansion from the wide layout keeps the surrounding Resources
        # chrome full width. Only the body switches from the two-side layout
        # to one full-width active pane.
        if self._is_wide_resources() and (self.state.expanded or expanded_allocation):
            overview.display = True
            controls.display = True
            body.styles.height = max(1, self.size.height - self._FIXED_LAYOUT_HEIGHT)
            if expanded_allocation:
                allocation.display = True
                allocation.styles.height = "1fr"
                if self.app_focus and self.focused is not allocation:
                    self.set_focus(allocation, scroll_visible=False)
            else:
                detail.display = True
                detail.styles.height = "1fr"
                detail.border_title = " NODE INSPECTOR "
                if self.app_focus and self.focused is not detail:
                    self.set_focus(detail, scroll_visible=False)
            return

        # Below the shared breakpoint, retain the original separate pages.
        if self.state.view != "nodes":
            overview.display = True
            allocation.display = True
            body.display = True
            # Header (1), overview (2), and footer (1) are the only rows
            # outside this pane. The body container occupies that remainder.
            body.styles.height = max(1, self.size.height - 4)
            allocation.styles.height = max(1, self.size.height - 4)
            self.state.active_pane = self.state.focused_panes[self.state.view]
            if self.app_focus and self.focused is not allocation:
                self.set_focus(allocation, scroll_visible=False)
            return

        if self.state.expanded:
            self._wide_resources_layout = False
            self._detail_auto_hidden = False
            self._layout_node_names = tuple(node.name for node in self.nodes)
            # Preserve the compact inspector's historical one-row safety
            # margin below the full-height body; its consumer viewport and
            # mouse offsets rely on that stable height.
            body.styles.height = max(1, self.size.height - 3)
            overview.display = False
            controls.display = False
            detail.display = True
            detail.styles.height = "1fr"
            detail.border_title = " NODE INSPECTOR "
            if self.app_focus and self.focused is not detail:
                self.set_focus(detail, scroll_visible=False)
            return

        overview.display = True
        controls.display = True
        nodes_pane.display = True
        body.styles.height = max(1, self.size.height - self._FIXED_LAYOUT_HEIGHT)
        detail.border_title = " SELECTED NODE "

        # Give the node inventory a stable height that includes every row,
        # then let the selected-node pane consume the remainder.
        if recompute_detail:
            available = max(0, self.size.height - self._FIXED_LAYOUT_HEIGHT)
            required = self._node_inventory_height()
            self._detail_auto_hidden = (
                available - required < self._DETAIL_MIN_HEIGHT
            )
            self._layout_node_names = tuple(node.name for node in self.nodes)
        available = max(0, self.size.height - self._FIXED_LAYOUT_HEIGHT)
        required = self._node_inventory_height()
        if self._detail_auto_hidden:
            nodes_height = available
            detail.display = False
        else:
            nodes_height = required
            detail.display = True
        nodes_pane.styles.height = max(1, nodes_height)
        detail.styles.height = max(
            self._DETAIL_MIN_HEIGHT,
            available - nodes_height,
        )
        if self.state.active_pane == "node" and detail.display:
            focus_target = detail
        else:
            self.state.active_pane = "nodes"
            focus_target = nodes_pane
        if self.app_focus and self.focused is not focus_target:
            self.set_focus(focus_target, scroll_visible=False)

    def _render_header(self) -> None:
        width = max(30, self.size.width - 2)
        clock = self.clock(self.snapshot)
        glyph = "◴◷◶◵"[self._spinner]
        status = (
            f"[bold {RED}]STALE[/]"
            if self.snapshot.stale
            else f"[{CYAN}]{glyph}[/]"
        )
        left = f"[bold {CYAN}]Falcon Resources[/]"
        right = f"[{GRAY}]{clock}[/]  {status}"
        gap = max(1, width - len("Falcon Resources") - len(clock) - 4)
        self.query_one("#resources-header", Static).update(
            left + " " * gap + right
        )

    def _render_views(self) -> None:
        target = self.query_one("#resources-views", ResourcesViewSelector)
        visible = (
            self.size.width >= MINIMUM_WIDTH
            and self.size.height >= RESOURCE_MINIMUM_HEIGHT
            and not self.state.expanded
            and not self.state.expanded_panels.get("gpu-allocations", "")
        )
        target.display = visible
        if not visible:
            self._view_hitboxes = []
            return
        available = max(1, self.size.width - 2)
        labels = [RESOURCE_VIEW_LABELS[view] for view in RESOURCE_VIEWS]
        total_width = sum(len(label) + 2 for label in labels) + 3 * (len(labels) - 1)
        offset = max(0, (available - total_width) // 2)
        target.styles.offset = (offset, 0)
        line = Text()
        hitboxes: list[tuple[int, int, str]] = []
        for index, view in enumerate(RESOURCE_VIEWS):
            if index:
                line.append(" · ", style=BORDER)
            start = len(line.plain)
            label = f" {RESOURCE_VIEW_LABELS[view]} "
            line.append(
                label,
                style=(f"bold {BACKGROUND} on {CYAN}" if view == self.state.view else GRAY),
            )
            hitboxes.append((start, start + len(label), view))
        self._view_hitboxes = hitboxes
        target.update(line)

    def _set_titles(self) -> None:
        titles = {
            "nodes": ("nodes-pane", "NODES"),
            "node": ("node-pane", "SELECTED NODE"),
            "gpu-allocations": ("gpu-allocations-pane", "GPU ALLOCATIONS"),
        }
        for pane, (identifier, base) in titles.items():
            self.query_one(f"#{identifier}", ResourcesPane).border_title = (
                f" {base}{' · focused' if self.app_focus and pane == self.state.active_pane else ''} "
            )

    def _render_overview(self) -> None:
        snapshot = ClusterSnapshot(
            nodes=tuple(self.nodes),
            jobs=self.snapshot.jobs,
            collected_at=self.snapshot.collected_at,
            stale=self.snapshot.stale,
            error=self.snapshot.error,
        )
        headroom = snapshot.request_headroom
        cpu_color = _resource_headroom_color(
            headroom.cpu_cores,
            snapshot.allocatable.cpu_cores,
        )
        memory_color = _resource_headroom_color(
            headroom.memory_bytes,
            snapshot.allocatable.memory_bytes,
        )
        right = Text(
            "GPU AVAILABLE  " if self.size.width >= 130 else "",
            style=f"bold {GRAY}",
        )
        availability = sorted(
            snapshot.gpu_availability.values(),
            key=lambda item: gpu_model_order_key(item.model),
        )
        for item in availability:
            if right:
                right.append("  ")
            free = item.request_headroom
            right.append(
                f"{item.model} {free}/{item.allocatable}",
                style=(
                    f"bold {_resource_headroom_color(free, item.allocatable)}"
                ),
            )
        if not availability:
            right.append("GPU —", style=f"bold {MUTED}")

        def render_left(metrics: set[str]) -> Text:
            result = Text(no_wrap=True, overflow="ellipsis")
            result.append(
                f"{snapshot.schedulable_nodes}/{snapshot.total_nodes} NODES  ",
                style=f"bold {GREEN if snapshot.schedulable_nodes else YELLOW}",
            )
            if "running" in metrics:
                result.append(
                    f"{snapshot.running_jobs} RUNNING  ",
                    style=f"bold {GREEN}",
                )
            if "cpu" in metrics:
                result.append("CPU ", style=f"bold {GRAY}")
                result.append(
                    f"{_short_cpu(headroom.cpu_cores)}/"
                    f"{_short_cpu(snapshot.allocatable.cpu_cores)}  ",
                    style=f"bold {cpu_color}",
                )
            if "memory" in metrics:
                result.append("MEM ", style=f"bold {GRAY}")
                result.append(
                    f"{_short_memory(headroom.memory_bytes)}/"
                    f"{_short_memory(snapshot.allocatable.memory_bytes)}",
                    style=f"bold {memory_color}",
                )
            return result

        # GPU availability is rendered at the right edge, so protect it from
        # being cropped by the summary metrics on the left. Remove the longer
        # metrics first as the terminal narrows; keep the existing compact
        # 80-column summary whenever the GPU list still fits beside it.
        metrics = {"memory"}
        if self.size.width >= 100:
            metrics.update(("running", "cpu"))
        left = render_left(metrics)
        content_width = max(1, self.size.width - 4)
        for metric in ("cpu", "running", "memory"):
            if len(left.plain) + len(right.plain) + 2 <= content_width:
                break
            if metric in metrics:
                metrics.remove(metric)
                left = render_left(metrics)
        gap = max(2, content_width - len(left.plain) - len(right.plain))
        left.append(" " * gap)
        left.append_text(right)
        self.query_one("#cluster-overview", Static).update(left)

    def _render_controls(self) -> None:
        filters = []
        if self.node_filter:
            filters.append(f"node={self.node_filter}")
        if self.gpu_filter:
            filters.append(f"gpu={self.gpu_filter}")
        suffix = f"  Filters: {', '.join(filters)}" if filters else ""
        self.query_one("#resource-controls", Static).update(
            Text(
                "Scheduler headroom (free/allocatable) · Enter inspect node"
                f"{suffix}",
                style=GRAY,
            )
        )

    @staticmethod
    def _usage_bar(
        *,
        free: float,
        allocatable: float,
        width: int,
        formatter: Callable[[float], str],
        eligible: bool = True,
    ) -> Text:
        """Render one free/allocatable bar using headroom thresholds."""

        width = max(1, int(width))
        allocatable = max(0.0, float(allocatable))
        free = max(0.0, min(allocatable, float(free)))
        if allocatable <= 0:
            return Text("-", style=MUTED, justify="right")
        label = f"{formatter(free)}/{formatter(allocatable)}"
        color = _resource_headroom_color(free, allocatable)
        if not eligible:
            color = PALETTE.danger
        bar_width = width - len(label) - 1
        if bar_width < 3:
            return Text(label, style=f"bold {color}", justify="right")
        filled = min(bar_width, round(free / allocatable * bar_width))
        result = Text(no_wrap=True)
        result.append("█" * filled, style=color)
        result.append("·" * (bar_width - filled), style=BORDER)
        result.append(" ")
        result.append(label, style=f"bold {color}")
        return result

    def _gpu_consumers(self) -> tuple[WorkloadConsumer, ...]:
        sort = _valid_consumer_sort(self.state.consumer_sort)
        if (
            self._gpu_consumers_cache_revision == self._nodes_revision
            and self._gpu_consumers_cache_sort == sort
        ):
            return self._gpu_consumers_cache
        consumers = [
            consumer
            for node in self._gpu_nodes()
            if _eligible(node)
            for consumer in node.consumers
            if consumer.requested.gpu_count > 0 and not is_system_consumer(consumer)
        ]
        consumers.sort(key=self._consumer_sort_key)
        self._gpu_consumers_cache = tuple(consumers)
        self._gpu_consumers_cache_revision = self._nodes_revision
        self._gpu_consumers_cache_sort = sort
        return self._gpu_consumers_cache

    def _namespace_categories(self) -> tuple[tuple[str, float], ...]:
        source = self._allocation_values(self._allocation_basis())
        visible: defaultdict[str, float] = defaultdict(float)
        hidden = 0.0
        for namespace, value in source:
            if is_system_namespace(namespace):
                hidden += value
            else:
                visible[namespace] += value
        ordered = sorted(visible.items(), key=lambda item: (-item[1], item[0].casefold()))
        system = ("System/hidden", hidden) if hidden else None
        capacity = self._allocation_category_capacity()
        if capacity is None:
            # Keep the pre-mount/unit-test behavior deterministic. Once the
            # pane has a committed height, use every row that the legend can
            # actually display before introducing an aggregate category.
            categories = list(ordered[:6])
            other = sum(value for _, value in ordered[6:])
            if other:
                categories.append(("Other", other))
            if system is not None:
                categories.append(system)
            return tuple(categories)

        all_categories = list(ordered)
        if system is not None:
            all_categories.append(system)
        if len(all_categories) <= capacity:
            return tuple(all_categories)
        if capacity <= 0:
            return ()

        reserve_system = system is not None and capacity > 1
        visible_slots = capacity - (1 if reserve_system else 0)
        omitted = len(ordered) > visible_slots
        if omitted:
            top_count = max(0, visible_slots - 1)  # reserve one row for Other
            categories = list(ordered[:top_count])
            other = sum(value for _, value in ordered[top_count:])
            if other:
                categories.append(("Other", other))
        else:
            categories = list(ordered[:visible_slots])
        if system is not None and len(categories) < capacity:
            categories.append(system)
        return tuple(categories)

    def _allocation_legend_panel_height(self) -> Optional[int]:
        """Return the height available for the rendered namespace legend."""

        if not self.is_mounted:
            return None
        try:
            pane = self.query_one("#gpu-allocations-pane")
        except (NoMatches, ScreenStackError):
            return None
        width = max(1, int(pane.content_size.width))
        height = max(1, int(pane.content_size.height))
        expanded = self.state.expanded_panels.get("gpu-allocations", "")
        if expanded == "history":
            return height
        if expanded == "pie":
            available_width = max(1, width - ALLOCATION_LEGEND_WIDTH)
            desired_width = max(1, 2 * height)
            pie_height = height
            if available_width < desired_width:
                pie_height = max(ALLOCATION_MIN_PIE_HEIGHT, available_width // 2)
            return max(1, pie_height)
        if expanded == "pods":
            return 0
        return self._allocation_stack_heights(height)[0]

    def _allocation_category_capacity(self) -> Optional[int]:
        panel_height = self._allocation_legend_panel_height()
        if panel_height is None:
            return None
        # Two border rows plus the Total row are structural, not namespace
        # rows. The remaining rows can be used for actual categories.
        return max(0, panel_height - 3)

    @staticmethod
    def _allocation_unit(basis: str) -> str:
        if basis in {"vram", "memory"}:
            return "G"
        if basis == "cpu":
            return "c"
        return ""

    @staticmethod
    def _allocation_label(basis: str) -> str:
        return {
            "gpu": "GPU",
            "vram": "VRAM",
            "memory": "MEMORY",
            "cpu": "CPU",
        }.get(basis, "GPU")

    def _allocation_basis(self) -> str:
        """Return the chart metric selected by ``m`` and, for GPU, ``v``."""

        if self.state.allocation_mode in {"memory", "cpu"}:
            return self.state.allocation_mode
        return "vram" if self.state.namespace_basis == "vram" else "gpu"

    def _allocation_mode_label(self) -> str:
        """Return the primary metric name shown beside the ``m`` shortcut."""

        return self._allocation_label(
            self.state.allocation_mode
            if self.state.allocation_mode in ALLOCATION_MODES
            else "gpu"
        )

    def _history_scale_label(self) -> str:
        """Return the current Allocation History y-axis scale label."""

        return "LOG" if self.state.history_log_scale else "LINEAR"

    def _allocation_values(self, basis: str) -> tuple[tuple[str, float], ...]:
        telemetry = self.gpu_telemetry
        if basis == "vram":
            return telemetry.vram_gib_by_namespace
        if basis == "cpu":
            return telemetry.cpu_cores_by_namespace
        if basis == "memory":
            return telemetry.memory_gib_by_namespace
        return telemetry.effective_gpus_by_namespace

    def _allocation_empty_label(self, basis: str) -> str:
        values = self._allocation_values(basis)
        if basis in {"cpu", "memory"}:
            label = self._allocation_label(basis)
            if self.gpu_telemetry.stale:
                return f"{label} allocation unavailable"
            if not values:
                return f"No {label.lower()} allocation"
            return f"No {label.lower()} allocation"
        if not self.gpu_telemetry.target_pods:
            return "No running GPU Jobs"
        if self.gpu_telemetry.stale:
            return (
                "VRAM allocation unavailable"
                if basis == "vram"
                else "GPU allocation unavailable"
            )
        if not self.gpu_telemetry.sampled_pods:
            return "No GPU allocations"
        if basis == "vram" and not values:
            return "VRAM allocation unavailable"
        return (
            "No VRAM allocation"
            if basis == "vram"
            else "No GPU allocation"
        )

    def _allocation_pie_subtitle(self, basis: str) -> str:
        if basis not in {"cpu", "memory"}:
            return (
                f" {self.gpu_telemetry.target_pods} Pods accounted "
                if self.gpu_telemetry.target_pods
                else " No GPU allocations "
            )
        total = sum(value for _, value in self._allocation_values(basis))
        if math.isclose(total, round(total), abs_tol=0.05):
            rendered = str(int(round(total)))
        elif total < 10:
            rendered = f"{total:.1f}"
        else:
            rendered = f"{total:.0f}"
        return f" {rendered}{self._allocation_unit(basis)} requested "

    def _gpu_pod_table(
        self,
        consumers: Optional[Sequence[WorkloadConsumer]] = None,
        *,
        width: Optional[int] = None,
    ) -> Table:
        if width is None:
            pane = self.query_one("#gpu-allocations-pane")
            width = pane.content_size.width
        width = max(1, int(width))
        consumers = tuple(consumers) if consumers is not None else self._gpu_consumers()
        narrow = width < 100
        gpu_header = "GPU/#" if narrow else "GPU MODEL / COUNT"

        def gpu_model_for(consumer: WorkloadConsumer) -> Optional[str]:
            model = consumer.requested.gpu_model
            if model:
                return model
            node = next(
                (candidate for candidate in self.nodes if candidate.name == consumer.node_name),
                None,
            )
            return node.gpu_model if node else None

        def gpu_request_for(consumer: WorkloadConsumer) -> str:
            model = gpu_model_for(consumer) or "GPU"
            separator = "" if narrow else " "
            return f"{model}{separator}×{consumer.requested.gpu_count}"

        # The request column is a priority column: reserve enough room for
        # every request label before Rich distributes any remaining width to
        # namespace, node, or Job text.  This matters for custom/product
        # names as well as the short normalized names normally reported by
        # Kubernetes.  A request count must never disappear behind an
        # ellipsis just because a workload name is long.
        gpu_requests = {
            id(consumer): gpu_request_for(consumer) for consumer in consumers
        }
        gpu_width = max(
            1,
            cell_len(gpu_header),
            *(cell_len(request) for request in gpu_requests.values()),
        )
        default_gpu_width = 7 if narrow else 17
        gpu_width = max(default_gpu_width, gpu_width)
        node_width = 6 if narrow else 10

        # At the supported minimum Resources width this keeps all normal
        # columns.  As a custom GPU label grows, remove the least important
        # identity columns in order, while retaining the complete request
        # label.  The thresholds include Rich's cell padding and table edge
        # space; they intentionally leave a little breathing room around the
        # fixed GPU column.
        show_job = width >= gpu_width + node_width + 12
        show_node = width >= gpu_width + 12
        show_namespace = width >= gpu_width + 6
        table = Table(
            box=box.SIMPLE_HEAD,
            expand=True,
            padding=(0, 1),
            collapse_padding=True,
            # The surrounding Panel owns the border. Rich's whitespace-only
            # SIMPLE_HEAD top edge would otherwise add an empty row before
            # the GPU Jobs headings.
            show_edge=False,
            header_style=f"bold {CYAN_2}",
        )
        if show_namespace:
            table.add_column(
                "NS" if narrow else "NAMESPACE",
                ratio=1,
                no_wrap=True,
                overflow="ellipsis",
            )
        if show_node:
            table.add_column(
                "NODE",
                width=node_width,
                no_wrap=True,
                overflow="ellipsis",
            )
        # GPU allocations are grouped by their owning workload.  A Pod name
        # is generated per attempt and is therefore noisy (and changes when a
        # Job is retried); show the stable Job/workload name instead.  It is
        # the first column sacrificed when the GPU request needs more room.
        if show_job:
            table.add_column("JOB", ratio=2, no_wrap=True, overflow="ellipsis")
        table.add_column(
            gpu_header,
            width=gpu_width,
            justify="right",
            # This only comes into play when a label is physically wider than
            # the terminal itself.  Folding preserves the complete model and
            # count; ellipsis would violate the GPU-column priority.
            no_wrap=False,
            overflow="fold",
        )
        start = self.state.allocation_scroll
        visible = self._allocation_visible_rows()
        for consumer in consumers[start : start + visible]:
            row: list[Text] = []
            if show_namespace:
                row.append(Text(consumer.namespace, style=WHITE))
            if show_node:
                row.append(Text(consumer.node_name, style=GRAY))
            if show_job:
                row.append(
                    Text(consumer.workload_name or consumer.pod_name, style=WHITE)
                )
            row.append(Text(gpu_requests[id(consumer)], style=CYAN))
            table.add_row(*row)
        if not consumers:
            table.add_row(Text("No active GPU-requesting Jobs", style=MUTED))
        return table

    @staticmethod
    def _allocation_legend_width(width: int) -> int:
        """Return the fixed outer width of the namespace legend."""

        del width
        return ALLOCATION_LEGEND_WIDTH

    @staticmethod
    def _allocation_pie_title(width: int, basis: str = "gpu") -> str:
        """Keep the pie title legible when aspect-correct sizing gets tight."""

        if basis in {"gpu", "vram"}:
            label = "GPU" if basis == "gpu" else "VRAM"
            return f" {label} BY NAMESPACE " if width >= 29 else f" {label} SHARE "
        label = {
            "memory": "MEMORY",
            "cpu": "CPU",
        }.get(basis, "ALLOCATION")
        return (
            f" {label} BY NAMESPACE "
            if width >= 29
            else f" {label} SHARE "
        )

    @staticmethod
    def _allocation_history_title(basis: str) -> str:
        if basis in {"gpu", "vram"}:
            return " ALLOCATION HISTORY "
        return f" {FalconResourcesApp._allocation_label(basis)} ALLOCATION HISTORY "

    @staticmethod
    def _allocation_colors(categories: Sequence[tuple[str, float]]) -> dict[str, str]:
        return allocation_colors(categories)

    def _allocation_chart_with_legend(
        self,
        chart: Text,
        categories: Sequence[tuple[str, float]],
        *,
        basis: str,
        width: int,
        height: int,
        colors: Mapping[str, str],
    ) -> Table:
        legend_width = self._allocation_legend_width(width)
        legend = render_allocation_legend(
            categories,
            # The legend itself is bordered and its grid cell is padded on
            # both sides, so its text cell is four columns narrower than the
            # nominal legend column.
            width=max(1, legend_width - 4),
            height=max(1, height - 2),
            unit=self._allocation_unit(basis),
            colors=colors,
            include_total=True,
        )
        legend_panel = Panel(
            legend,
            title=Text(" NAMESPACE LEGEND ", style=f"bold {PALETTE.accent}"),
            box=box.SQUARE,
            border_style=BORDER,
            height=height,
        )
        content = Table.grid(expand=True, padding=(0, 1))
        content.add_column(width=legend_width)
        content.add_column(ratio=1)
        content.add_row(legend_panel, chart)
        return content

    def _render_gpu_allocations(self) -> None:
        target = self.query_one("#gpu-allocations-pane", ResourcesPane)
        expanded = self.state.expanded_panels["gpu-allocations"]
        basis = self._allocation_basis()
        categories = self._namespace_categories()
        colors = self._allocation_colors(categories)
        consumers = self._gpu_consumers()
        telemetry = self.gpu_telemetry
        geometry = self._allocation_geometry_for(
            categories=categories,
            expanded=expanded,
        )
        pie_hit_region = self._allocation_pie_hit_region(geometry, expanded)
        self._allocation_geometry = geometry
        self._allocation_regions = {
            name: region
            for name, region in (
                ("legend", geometry.legend),
                ("pie", pie_hit_region),
                ("history", geometry.history),
                ("jobs", geometry.jobs),
            )
            if region.width > 0 and region.height > 0
        }
        render_key = (
            expanded,
            basis,
            self.state.selected_panels.get("gpu-allocations", ""),
            self.state.allocation_scroll,
            self.app_focus,
            self.color_mode,
            geometry.signature,
            _valid_consumer_sort(self.state.consumer_sort),
            self.state.history_log_scale,
            self.history_hours,
            self._history_revision,
            telemetry.effective_gpus_by_namespace,
            telemetry.vram_gib_by_namespace,
            telemetry.cpu_cores_by_namespace,
            telemetry.memory_gib_by_namespace,
            telemetry.target_pods,
            telemetry.sampled_pods,
            telemetry.stale,
            self._nodes_revision,
            categories,
            tuple(colors.items()),
        )
        if render_key == self._allocation_render_key:
            return
        self._allocation_render_key = render_key
        def panel_width(region: Region) -> int:
            return max(1, region.width)

        def inner_width(region: Region) -> int:
            return max(1, region.width - 4)

        def inner_height(region: Region) -> int:
            return max(1, region.height - 2)

        def legend_panel(region: Region) -> Panel:
            legend = render_allocation_legend(
                categories,
                width=max(1, panel_width(region) - 4),
                height=max(1, inner_height(region)),
                unit=self._allocation_unit(basis),
                colors=colors,
                columns=1,
                include_total=True,
            )
            return Panel(
                legend,
                title=Text(" NAMESPACE LEGEND ", style=f"bold {PALETTE.accent}"),
                box=box.SQUARE,
                border_style=BORDER,
                width=panel_width(region),
                height=max(1, region.height),
                padding=(0, 1),
            )

        selected = self.state.selected_panels.get("gpu-allocations")
        target.border_subtitle = (
            " Enter expand · Esc restore " if expanded else ""
        )
        if expanded == "pods":
            visible_requests = sum(
                consumer.requested.gpu_count for consumer in consumers
            )
            total_requested = _gpu_totals(self._gpu_nodes())[1]
            hidden = max(0, total_requested - visible_requests)
            target.update(
                Panel(
                    self._gpu_pod_table(
                        consumers,
                        width=max(1, geometry.jobs.width - 4),
                    ),
                    title=Text(
                        " GPU-REQUESTING JOBS ",
                        style=f"bold {PALETTE.accent}",
                    ),
                    subtitle=(
                        Text(
                            f" {visible_requests} visible + {hidden} system/hidden ",
                            style=GRAY,
                        )
                        if hidden
                        else Text(f" {visible_requests} requested ", style=GRAY)
                    ),
                    box=box.SQUARE,
                    border_style=BORDER,
                    width=geometry.width,
                    height=geometry.height,
                )
            )
            return

        if expanded == "history":
            history = geometry.history
            history_key = (
                "history",
                self._history_render_signature(),
                categories,
                basis,
                self.state.history_log_scale,
                (history.x, history.y, history.width, history.height),
            )
            if history_key != self._history_cache_key or self._history_cache is None:
                self._history_cache_key = history_key
                self._history_cache = render_gpu_history(
                    self.history,
                    width=inner_width(history),
                    height=inner_height(history),
                    basis=basis,
                    categories=categories,
                    colors=colors,
                    show_legend=False,
                    log_scale=self.state.history_log_scale,
                )
            history_panel = Panel(
                self._history_cache,
                title=Text(
                    self._allocation_history_title(basis),
                    style=f"bold {PALETTE.accent}",
                ),
                subtitle=Text(f" {self._history_window_label} ", style=GRAY),
                box=box.SQUARE,
                border_style=BORDER,
                width=panel_width(history),
                height=history.height,
                padding=(0, 1),
            )
            content = Table.grid(expand=False, padding=(0, 0))
            content.add_column(width=ALLOCATION_LEGEND_WIDTH)
            content.add_column(width=panel_width(history))
            content.add_row(legend_panel(geometry.legend), history_panel)
            target.update(content)
            return

        if expanded == "pie":
            pie = geometry.pie
            pie_key = (
                "pie",
                categories,
                basis,
                self._allocation_empty_label(basis),
                (pie.x, pie.y, pie.width, pie.height),
            )
            if pie_key != self._pie_cache_key or self._pie_cache is None:
                self._pie_cache_key = pie_key
                self._pie_cache = render_namespace_pie(
                    categories,
                    width=inner_width(pie),
                    height=inner_height(pie),
                    unit=self._allocation_unit(basis),
                    empty_label=self._allocation_empty_label(basis),
                    colors=colors,
                    show_legend=False,
                )
            pie_panel = Panel(
                Align.center(self._pie_cache, vertical="middle"),
                title=Text(
                    self._allocation_pie_title(panel_width(pie), basis),
                    style=f"bold {PALETTE.accent}",
                ),
                subtitle=Text(
                    self._allocation_pie_subtitle(basis),
                    style=RED if self.gpu_telemetry.stale else GRAY,
                ),
                box=box.SQUARE,
                border_style=BORDER,
                width=panel_width(pie),
                height=pie.height,
                padding=(0, 1),
            )
            content = Table.grid(expand=False, padding=(0, 0))
            content.add_column(width=ALLOCATION_LEGEND_WIDTH)
            content.add_column(width=panel_width(pie))
            content.add_row(legend_panel(geometry.legend), pie_panel)
            target.update(content)
            return

        history = geometry.history
        pie = geometry.pie
        jobs = geometry.jobs
        history_key = (
            "history",
            self._history_render_signature(),
            categories,
            basis,
            self.state.history_log_scale,
            (history.x, history.y, history.width, history.height),
        )
        if history_key != self._history_cache_key or self._history_cache is None:
            self._history_cache_key = history_key
            self._history_cache = render_gpu_history(
                self.history,
                width=inner_width(history),
                height=inner_height(history),
                basis=basis,
                categories=categories,
                colors=colors,
                show_legend=False,
                log_scale=self.state.history_log_scale,
            )
        history_panel = Panel(
            self._history_cache,
                title=Text(
                    self._allocation_history_title(basis),
                    style=f"bold {PALETTE.accent}",
                ),
            subtitle=Text(f" {self._history_window_label} ", style=GRAY),
            box=box.SQUARE,
            border_style=CYAN if selected == "history" else BORDER,
            width=panel_width(history),
            height=history.height,
            padding=(0, 1),
        )

        pie_key = (
            "pie",
            categories,
            basis,
            self._allocation_empty_label(basis),
            (pie.x, pie.y, pie.width, pie.height),
        )
        if pie_key != self._pie_cache_key or self._pie_cache is None:
            self._pie_cache_key = pie_key
            self._pie_cache = render_namespace_pie(
                categories,
                width=inner_width(pie),
                height=inner_height(pie),
                unit=self._allocation_unit(basis),
                empty_label=self._allocation_empty_label(basis),
                colors=colors,
                show_legend=False,
            )
        pie_panel = Panel(
            Align.center(self._pie_cache, vertical="middle"),
            title=Text(
                self._allocation_pie_title(geometry.left_width, basis),
                style=f"bold {PALETTE.accent}",
            ),
            subtitle=Text(
                self._allocation_pie_subtitle(basis),
                style=RED if self.gpu_telemetry.stale else GRAY,
            ),
            box=box.SQUARE,
            border_style=CYAN if selected == "pie" else BORDER,
            # The chart footprint stays aspect-correct in ``pie``. Expand
            # only its outer panel to the fixed legend column so the left
            # stack meets the Jobs panel without a one-column void.
            width=max(panel_width(pie), geometry.left_width),
            height=pie.height,
            padding=(0, 1),
        )
        visible_requests = sum(
            consumer.requested.gpu_count for consumer in consumers
        )
        total_requested = _gpu_totals(self._gpu_nodes())[1]
        hidden = max(0, total_requested - visible_requests)
        sort_label = CONSUMER_SORT_LABELS[
            _valid_consumer_sort(self.state.consumer_sort)
        ]
        pods = Panel(
            self._gpu_pod_table(
                consumers,
                width=max(1, jobs.width - 4),
            ),
            title=Text(" GPU-REQUESTING JOBS ", style=f"bold {PALETTE.accent}"),
            subtitle=(
                Text(
                    f" {visible_requests} visible + {hidden} system/hidden · "
                    f"sort {sort_label} · s cycle ",
                    style=GRAY,
                )
                if hidden
                else Text(
                    f" {visible_requests} requested · sort {sort_label} · s cycle ",
                    style=GRAY,
                )
            ),
            box=box.SQUARE,
            border_style=CYAN if selected == "pods" else BORDER,
            width=panel_width(jobs),
            height=jobs.height,
            padding=(0, 1),
        )

        # The top and bottom rows intentionally use different column grids:
        # History extends from the fixed Namespace Legend edge to the pane's
        # right edge, while Jobs starts after the aspect-correct pie footprint.
        # A single two-column grid would couple both panels to the wider of
        # those left-hand regions and recreate the empty gap beside the legend.
        top = Table.grid(expand=False, padding=(0, 0))
        top.add_column(width=geometry.legend.width)
        top.add_column(width=panel_width(history))
        top.add_row(legend_panel(geometry.legend), history_panel)
        bottom = Table.grid(expand=False, padding=(0, 0))
        bottom.add_column(width=geometry.left_width)
        bottom.add_column(width=panel_width(jobs))
        bottom.add_row(pie_panel, pods)
        content = Table.grid(expand=False, padding=(0, 0))
        content.add_column(width=geometry.width)
        content.add_row(top)
        content.add_row(bottom)
        start = self.state.allocation_scroll
        visible = self._allocation_visible_rows()
        end = min(len(consumers), start + visible)
        target.border_subtitle = (
            f" Jobs {start + 1}-{end}/{len(consumers)} "
            if len(consumers) > visible
            else ""
        )
        target.update(content)

    def _render_nodes(self) -> None:
        target = self.query_one("#nodes-pane", ResourcesPane)
        if not self.nodes:
            message = (
                "No nodes match the active filter."
                if self.snapshot.nodes
                else "No node inventory available."
            )
            target.update(Align.center(message, vertical="middle"))
            return
        # In combined mode this pane is only half the terminal. Responsive
        # decisions must therefore use the pane's committed content width,
        # not the terminal width that contains it.
        width = max(1, target.content_size.width)
        content_width = max(30, width)
        # Six columns have two cells of Rich padding each. Reserve readable
        # identity/status columns, then divide every remaining cell between
        # the three pressure bars.
        node_width = 14 if width < 100 else 18
        gpu_type_width = min(
            16,
            max(8, *(len(node.gpu_model or "-") for node in self.nodes)),
        )
        sched_width = min(
            10,
            max(5, *(len(_schedulable(node)[0]) for node in self.nodes)),
        )
        metric_space = max(
            18,
            content_width - node_width - gpu_type_width - sched_width - 12,
        )
        cpu_width = max(6, metric_space // 3)
        ram_width = max(7, (metric_space - cpu_width) // 2)
        gpu_width = max(5, metric_space - cpu_width - ram_width)
        table = Table(
            box=box.SIMPLE_HEAD,
            expand=True,
            padding=(0, 1),
            collapse_padding=True,
            show_edge=False,
            header_style=f"bold {CYAN_2}",
        )
        table.add_column(
            "NODE",
            width=node_width,
            no_wrap=True,
            overflow="ellipsis",
        )
        table.add_column(
            "CPUS",
            width=cpu_width,
            justify="right",
            no_wrap=True,
        )
        table.add_column(
            "RAM (GB)",
            width=ram_width,
            justify="right",
            no_wrap=True,
        )
        table.add_column(
            "GPUS",
            width=gpu_width,
            justify="right",
            no_wrap=True,
        )
        table.add_column(
            "GPU TYPE",
            width=gpu_type_width,
            no_wrap=True,
            overflow="ellipsis",
        )
        table.add_column("SCHED", width=sched_width, no_wrap=True)
        count = self._visible_nodes()
        start = self.state.node_scroll
        for node in self.nodes[start : start + count]:
            selected = node.name == self.state.selected_node
            selection_active = (
                selected
                and self.app_focus
                and self.state.active_pane == "nodes"
            )
            sched, sched_color = _schedulable(node)
            headroom = node.request_headroom
            cells = [
                Text(
                    f"{'>' if selected else ' '} {node.name}",
                    style=(
                        f"bold {CYAN}"
                        if selection_active
                        else (f"bold {WHITE}" if selected else WHITE)
                    ),
                    no_wrap=True,
                    overflow="ellipsis",
                ),
            ]
            cells.extend(
                [
                    self._usage_bar(
                        free=headroom.cpu_cores,
                        allocatable=node.allocatable.cpu_cores,
                        width=cpu_width,
                        formatter=_short_cpu,
                        eligible=_eligible(node),
                    ),
                    self._usage_bar(
                        free=headroom.memory_bytes / (1024**3),
                        allocatable=node.allocatable.memory_bytes / (1024**3),
                        width=ram_width,
                        formatter=lambda value: f"{value:.0f}",
                        eligible=_eligible(node),
                    ),
                    self._usage_bar(
                        free=headroom.gpu_count,
                        allocatable=node.allocatable.gpu_count,
                        width=gpu_width,
                        formatter=lambda value: str(round(value)),
                        eligible=_eligible(node),
                    ),
                    Text(node.gpu_model or "-", style=WHITE),
                    Text(sched, style=sched_color),
                ]
            )
            table.add_row(
                *cells,
                style=(
                    f"bold on {SELECTION}"
                    if selection_active
                    else (f"on {SELECTION}" if selected else None)
                ),
            )
        end = min(len(self.nodes), start + count)
        target.border_subtitle = (
            f" {start + 1}-{end}/{len(self.nodes)} "
            if len(self.nodes) > count
            else ""
        )
        target.update(table)

    def _consumer_table(
        self,
        node: NodeSnapshot,
        *,
        expanded: bool,
    ) -> Table:
        table = Table(
            box=box.SIMPLE_HEAD if expanded else None,
            expand=True,
            padding=(0, 1),
            show_header=expanded,
            header_style=f"bold {CYAN_2}",
        )
        # Keep namespace compact and give the job name the room it usually
        # needs.  Using the same proportions at every size also prevents the
        # namespace column from opening a conspicuous blank gap before JOB in
        # the minimum-width layout.
        namespace_ratio, job_ratio = 2, 3
        table.add_column(
            "NAMESPACE",
            ratio=namespace_ratio,
            no_wrap=True,
            overflow="ellipsis",
        )
        table.add_column(
            "JOB",
            ratio=job_ratio,
            no_wrap=True,
            overflow="ellipsis",
        )
        table.add_column("STATUS", width=9)
        table.add_column("CPU", width=8, justify="right")
        table.add_column("RAM", width=9, justify="right")
        table.add_column("GPU", width=10, justify="right")
        visible = self._visible_consumers()
        start = self.state.consumer_scroll
        visible_consumers = self._sorted_consumers(node)
        consumers = visible_consumers[start : start + visible]
        for absolute, consumer in enumerate(consumers, start=start):
            selected = expanded and absolute == self.state.selected_consumer
            selection_active = (
                selected
                and self.app_focus
                and self.state.active_pane == "node"
            )
            job_name = consumer.workload_name or consumer.pod_name
            cells = [
                Text(
                    f"{'>' if selected else ' '} {consumer.namespace}",
                    style=f"bold {CYAN}" if selected else WHITE,
                    no_wrap=True,
                    overflow="ellipsis",
                ),
                Text(
                    job_name,
                    style=GRAY,
                    no_wrap=True,
                    overflow="ellipsis",
                ),
            ]
            color = GREEN if consumer.status == "Running" else YELLOW
            cells.append(Text(consumer.status, style=color))
            vector = consumer.requested
            cells.extend(
                [
                    Text(_short_cpu(vector.cpu_cores), style=WHITE),
                    Text(_short_memory(vector.memory_bytes), style=WHITE),
                    Text(
                        (
                            f"{vector.gpu_model or ''}x{vector.gpu_count}"
                            if vector.gpu_count
                            else "-"
                        ),
                        style=WHITE,
                    ),
                ]
            )
            table.add_row(
                *cells,
                style=(
                    f"bold on {SELECTION}"
                    if selection_active
                    else (f"on {SELECTION}" if selected else None)
                ),
            )
        if not visible_consumers:
            table.add_row(Text("No active workloads", style=MUTED))
        return table

    def _render_node(self) -> None:
        target = self.query_one("#node-pane", ResourcesPane)
        node = self._selected()
        if node is None:
            target.update(Align.center("Select a node to inspect its consumers.", vertical="middle"))
            return
        # Resource inspection is scheduler-facing: show allocatable headroom
        # (free) over the allocatable total consistently in both the compact
        # selected-node strip and the expanded inspector. ``requested`` is
        # retained for allocation/history views, but is not displayed here as
        # physical usage.
        headroom = node.request_headroom
        width = self._node_pane_width()
        if not self.state.expanded:
            sched, sched_color = _schedulable(node)
            line = Text(node.name, style=f"bold {CYAN}")
            if width >= 100 or sched != "Yes":
                line.append(f"   {sched}", style=sched_color)
            line.append("   CPU ", style=GRAY)
            line.append(
                f"{_short_cpu(headroom.cpu_cores)}/"
                f"{_short_cpu(node.allocatable.cpu_cores)}",
                style=_resource_headroom_color(
                    headroom.cpu_cores,
                    node.allocatable.cpu_cores,
                ),
            )
            line.append("   RAM ", style=GRAY)
            line.append(
                f"{_short_memory(headroom.memory_bytes)}/"
                f"{_short_memory(node.allocatable.memory_bytes)}",
                style=_resource_headroom_color(
                    headroom.memory_bytes,
                    node.allocatable.memory_bytes,
                ),
            )
            line.append(
                f"   GPU {node.gpu_model or '-'} ",
                style=WHITE,
            )
            line.append(
                f"{headroom.gpu_count}/{node.allocatable.gpu_count}",
                style=_resource_headroom_color(
                    headroom.gpu_count,
                    node.allocatable.gpu_count,
                ),
            )
            if node.gpu_memory_bytes_per_device is not None:
                line.append(
                    f"   VRAM {_short_memory(node.gpu_memory_bytes_per_device)}"
                    + ("/GPU" if width >= 100 else ""),
                    style=GRAY,
                )
            content = Table.grid(expand=True)
            content.add_column()
            content.add_row(line)
            content.add_row(self._consumer_table(node, expanded=False))
            consumers = self._sorted_consumers(node)
            if consumers:
                visible = self._visible_consumers()
                start = self.state.consumer_scroll + 1
                end = min(len(consumers), start + visible - 1)
                target.border_subtitle = (
                    f" consumers {start}-{end}/{len(consumers)} · "
                    f"sort {CONSUMER_SORT_LABELS[self.state.consumer_sort]} · "
                    "s cycle "
                )
            else:
                target.border_subtitle = ""
            target.update(content)
            return
        sched, sched_color = _schedulable(node)
        taints = ", ".join(map(str, node.taints)) or "-"
        label_keys = (
            "kubernetes.io/arch",
            "node.kubernetes.io/instance-type",
            "topology.kubernetes.io/zone",
        )
        labels = ", ".join(
            f"{key}={node.labels[key]}" for key in label_keys if key in node.labels
        ) or "-"
        facts = Table.grid(expand=True, padding=(0, 2))
        facts.add_column(style=GRAY, width=18)
        facts.add_column(style=WHITE, ratio=1)
        if width < 100:
            node_value = Text(_truncate(node.name, 35), style=WHITE)
            node_value.append(" · ", style=GRAY)
            node_value.append(sched, style=sched_color)
            gpu_value = Text(f"{node.gpu_model or '-'}  ", style=WHITE)
            gpu_value.append(
                f"{headroom.gpu_count}/{node.allocatable.gpu_count}",
                style=_resource_headroom_color(
                    headroom.gpu_count,
                    node.allocatable.gpu_count,
                ),
            )
            gpu_value.append(" · ", style=GRAY)
            gpu_value.append(
                (
                    f"{_short_memory(node.gpu_memory_bytes_per_device)}/GPU"
                    if node.gpu_memory_bytes_per_device is not None
                    else "- VRAM"
                ),
                style=WHITE,
            )
            facts.add_row(
                "CPU",
                Text(
                    f"{_short_cpu(headroom.cpu_cores)} / "
                    f"{_short_cpu(node.allocatable.cpu_cores)}",
                    style=_resource_headroom_color(
                        headroom.cpu_cores,
                        node.allocatable.cpu_cores,
                    ),
                ),
            )
            facts.add_row(
                "RAM",
                Text(
                    f"{_short_memory(headroom.memory_bytes)} / "
                    f"{_short_memory(node.allocatable.memory_bytes)}",
                    style=_resource_headroom_color(
                        headroom.memory_bytes,
                        node.allocatable.memory_bytes,
                    ),
                ),
            )
            facts.add_row(
                "GPU",
                gpu_value,
            )
            facts.add_row("Node", node_value)
            facts.add_row("Taints", _truncate(taints, 48))
            facts.add_row("Labels", _truncate(labels, 48))
        else:
            facts.add_column(style=GRAY, width=18)
            facts.add_column(style=WHITE, ratio=1)
            facts.add_row(
                "CPU capacity",
                _short_cpu(node.capacity.cpu_cores),
                "CPU free / alloc",
                Text(
                    f"{_short_cpu(headroom.cpu_cores)} / "
                    f"{_short_cpu(node.allocatable.cpu_cores)}",
                    style=_resource_headroom_color(
                        headroom.cpu_cores,
                        node.allocatable.cpu_cores,
                    ),
                ),
            )
            facts.add_row(
                "RAM capacity",
                _short_memory(node.capacity.memory_bytes),
                "RAM free / alloc",
                Text(
                    f"{_short_memory(headroom.memory_bytes)} / "
                    f"{_short_memory(node.allocatable.memory_bytes)}",
                    style=_resource_headroom_color(
                        headroom.memory_bytes,
                        node.allocatable.memory_bytes,
                    ),
                ),
            )
            facts.add_row(
                "GPU model",
                node.gpu_model or "-",
                "VRAM / GPU",
                (
                    _short_memory(node.gpu_memory_bytes_per_device)
                    if node.gpu_memory_bytes_per_device is not None
                    else "-"
                ),
            )
            facts.add_row(
                "GPU capacity",
                str(node.allocatable.gpu_count),
                "GPU free / alloc",
                Text(
                    f"{headroom.gpu_count}/{node.allocatable.gpu_count}",
                    style=_resource_headroom_color(
                        headroom.gpu_count,
                        node.allocatable.gpu_count,
                    ),
                ),
            )
            facts.add_row(
                "Node",
                node.name,
                "Schedulable",
                Text(sched, style=sched_color),
            )
            facts.add_row(
                "Visible workloads",
                str(node.workload_count),
                "System Pods",
                "Hidden",
            )
            facts.add_row(
                "Taints",
                _truncate(taints, 42),
                "Labels",
                _truncate(labels, 42),
            )
        content = Table.grid(expand=True)
        content.add_column()
        facts_height = self._expanded_facts_height()
        consumer_height = max(
            6,
            target.content_size.height - facts_height,
        )
        content.add_row(
            Panel(
                facts,
                title=Text(" NODE CAPACITY & HEADROOM ", style=f"bold {CYAN}"),
                border_style=BORDER,
                box=box.SQUARE,
            )
        )
        content.add_row(
            Panel(
                self._consumer_table(node, expanded=True),
                title=Text(
                    " RESOURCE CONSUMERS · who/what is using this node ",
                    style=f"bold {CYAN}",
                ),
                border_style=CYAN_2,
                box=box.SQUARE,
                height=consumer_height,
            )
        )
        consumers = self._sorted_consumers(node)
        target.border_subtitle = (
            f" consumer {self.state.selected_consumer + 1}/"
            f"{len(consumers)} · "
            f"sort {CONSUMER_SORT_LABELS[self.state.consumer_sort]} · s cycle "
            if consumers
            else ""
        )
        target.update(content)

    def _render_footer(self) -> None:
        if (
            self.size.width < MINIMUM_WIDTH
            or self.size.height < RESOURCE_MINIMUM_HEIGHT
        ):
            value = "q Quit   r Retry"
        elif self._wide_resources_layout:
            basis_label = "VRAM" if self.state.namespace_basis == "vram" else "COUNT"
            mode_label = self._allocation_mode_label()
            scale_label = self._history_scale_label()
            gpu_control = f"v {basis_label}   " if self.state.allocation_mode == "gpu" else ""
            value = (
                "↑/↓ Scroll active pane   Enter Expand   Tab Next pane   "
                f"s Sort {CONSUMER_SORT_LABELS[_valid_consumer_sort(self.state.consumer_sort)]}   "
                f"{gpu_control}m {mode_label}   l {scale_label}   r Refresh   q Quit"
            )
        elif self.state.view == "gpu-allocations":
            expanded_panel = self.state.expanded_panels["gpu-allocations"]
            basis_label = "VRAM" if self.state.namespace_basis == "vram" else "COUNT"
            mode_label = self._allocation_mode_label()
            scale_label = self._history_scale_label()
            gpu_control = f"v {basis_label}  " if self.state.allocation_mode == "gpu" else ""
            prefix = "Esc restore panels   " if expanded_panel else ""
            value = prefix + (
                "←/→ Views  ↑/↓ Scroll Jobs  Enter Expand  Tab  "
                f"s Sort {CONSUMER_SORT_LABELS[_valid_consumer_sort(self.state.consumer_sort)]}  "
                f"{gpu_control}m {mode_label}  l {scale_label}  r Refresh  q Quit"
                if self.size.width < 100
                else (
                    "←/→ Views   ↑/↓ Scroll Jobs   Enter Expand   "
                    "Tab Next pane   "
                    f"s Sort {CONSUMER_SORT_LABELS[_valid_consumer_sort(self.state.consumer_sort)]}   "
                    f"{gpu_control}m {mode_label}   l {scale_label}   r Refresh   q Quit"
                )
            )
        elif self.state.expanded:
            if self.size.width < 100:
                value = (
                    "↑/↓ Consumers   "
                    f"s Sort {CONSUMER_SORT_LABELS[self.state.consumer_sort]}   "
                    "Tab   Esc Nodes   q Quit"
                )
            else:
                value = (
                    "←/→ Views   ↑/↓ Consumers   PgUp/PgDn Page   "
                    f"s Sort {CONSUMER_SORT_LABELS[self.state.consumer_sort]}   "
                    "Tab Next pane   Esc Nodes   r Refresh   q Quit"
                )
        elif self.size.width < 100:
            value = (
                "←/→ Views  ↑/↓ Nodes  Enter Expand  Tab  "
                f"s Sort {CONSUMER_SORT_LABELS[self.state.consumer_sort]}   "
                "r Refresh   q Quit"
            )
        else:
            value = (
                "←/→ Views   ↑/↓ Navigate nodes   Enter Expand   "
                "Tab Next pane   "
                f"s Sort {CONSUMER_SORT_LABELS[self.state.consumer_sort]}   "
                "r Refresh   q Quit"
            )
        self.query_one("#resources-footer", Static).update(Text(value, style=GRAY))

    def _render_all(self) -> None:
        if not self.is_mounted:
            return
        self._set_titles()
        self._render_header()
        self._render_views()
        if (
            self.size.width >= MINIMUM_WIDTH
            and self.size.height >= RESOURCE_MINIMUM_HEIGHT
        ):
            if self._wide_resources_layout:
                self._render_overview()
                self._render_controls()
                self._render_nodes()
                self._render_node()
                self._render_gpu_allocations()
            elif self._is_wide_resources() and (
                self.state.expanded
                or self.state.expanded_panels.get("gpu-allocations", "")
            ):
                self._render_overview()
                self._render_controls()
                if self.state.expanded:
                    self._render_node()
                else:
                    self._render_gpu_allocations()
            elif self.state.view == "nodes":
                self._render_overview()
                self._render_controls()
                self._render_nodes()
                self._render_node()
            else:
                self._render_overview()
                self._render_gpu_allocations()
        self._render_footer()

    def node_clicked(self, offset) -> None:
        if self.state.expanded or not self.nodes or offset is None or offset.y <= 0:
            return
        # With ``show_edge=False`` SIMPLE_HEAD renders the header and header
        # separator directly above the first data row. Ignore those non-row
        # lines; ``offset`` is relative to the pane content (not its border).
        row = offset.y - 2
        if row < 0:
            return
        index = self.state.node_scroll + row
        if index < len(self.nodes):
            self.state.selected_node = self.nodes[index].name
            self.state.selected_consumer = 0
            self.state.consumer_scroll = 0
            self._render_all()

    def consumer_clicked(self, offset) -> None:
        """Select a workload row in the selected-node pane."""

        if (
            self.state.view != "nodes"
            or (not self.state.expanded and not self._wide_resources_layout)
            or offset is None
        ):
            return
        try:
            y = int(offset.y)
        except (AttributeError, TypeError, ValueError):
            return
        consumers = self._sorted_consumers(self._selected())
        if not consumers:
            return

        if self.state.expanded:
            # The inspector is a Rich grid with a capacity panel followed by
            # a bordered consumer panel. That panel contributes its top
            # border, spacer, table header, and header separator before the
            # first data row.
            row = y - self._expanded_facts_height() - 4
        else:
            # The combined Selected Node pane starts with one summary row and
            # then a compact, headerless consumer table.
            row = y - 1
        if row < 0:
            return
        index = self.state.consumer_scroll + row
        if index < 0 or index >= len(consumers):
            return
        self.state.selected_consumer = index
        self._ensure_visible()
        self._render_all()

    def _move(self, amount: int) -> None:
        if self._wide_resources_layout and self.state.active_pane == "node":
            consumers = self._sorted_consumers(self._selected())
            if consumers:
                self.state.selected_consumer = min(
                    len(consumers) - 1,
                    max(0, self.state.selected_consumer + amount),
                )
            self._ensure_visible()
            self._render_node()
            return
        if self._wide_resources_layout and self.state.active_pane == "gpu-allocations":
            maximum = max(0, len(self._gpu_consumers()) - self._allocation_visible_rows())
            self.state.allocation_scroll = min(
                maximum,
                max(0, self.state.allocation_scroll + amount),
            )
            self._render_gpu_allocations()
            return
        if self.state.view == "gpu-allocations":
            maximum = max(0, len(self._gpu_consumers()) - self._allocation_visible_rows())
            self.state.allocation_scroll = min(
                maximum,
                max(0, self.state.allocation_scroll + amount),
            )
            self._render_gpu_allocations()
            return
        if self.state.expanded:
            node = self._selected()
            consumers = self._sorted_consumers(node)
            if consumers:
                self.state.selected_consumer = min(
                    len(consumers) - 1,
                    max(0, self.state.selected_consumer + amount),
                )
        elif self.nodes:
            index = min(
                len(self.nodes) - 1,
                max(0, self._selected_index() + amount),
            )
            if self.nodes[index].name != self.state.selected_node:
                self.state.selected_node = self.nodes[index].name
                self.state.selected_consumer = 0
                self.state.consumer_scroll = 0
        self._ensure_visible()
        self._render_all()

    def action_up(self) -> None:
        self._move(-1)

    def action_down(self) -> None:
        self._move(1)

    def scroll_consumers(self, amount: int) -> None:
        """Scroll the workload list under the mouse without changing nodes."""

        node = self._selected()
        consumers = self._sorted_consumers(node)
        if not consumers:
            return
        if self.state.expanded or (
            self._wide_resources_layout and self.state.active_pane == "node"
        ):
            self._move(amount)
            return
        visible = self._visible_consumers()
        maximum = max(0, len(consumers) - visible)
        self.state.consumer_scroll = max(
            0,
            min(maximum, self.state.consumer_scroll + amount),
        )
        self._render_node()

    def action_page_up(self) -> None:
        if self._wide_resources_layout and self.state.active_pane == "node":
            self._move(-self._visible_consumers())
            return
        if self._wide_resources_layout and self.state.active_pane == "gpu-allocations":
            self._move(-self._allocation_visible_rows())
            return
        if self.state.view == "gpu-allocations":
            self._move(-self._allocation_visible_rows())
            return
        self._move(
            -(self._visible_consumers() if self.state.expanded else self._visible_nodes())
        )

    def action_page_down(self) -> None:
        if self._wide_resources_layout and self.state.active_pane == "node":
            self._move(self._visible_consumers())
            return
        if self._wide_resources_layout and self.state.active_pane == "gpu-allocations":
            self._move(self._allocation_visible_rows())
            return
        if self.state.view == "gpu-allocations":
            self._move(self._allocation_visible_rows())
            return
        self._move(
            self._visible_consumers() if self.state.expanded else self._visible_nodes()
        )

    def action_home(self) -> None:
        if self._wide_resources_layout and self.state.active_pane == "gpu-allocations":
            self.state.allocation_scroll = 0
        elif self._wide_resources_layout and self.state.active_pane == "node":
            self.state.selected_consumer = 0
        elif self.state.view == "gpu-allocations":
            self.state.allocation_scroll = 0
        elif self.state.expanded:
            self.state.selected_consumer = 0
        elif self.nodes:
            self.state.selected_node = self.nodes[0].name
        self._ensure_visible()
        self._render_all()

    def action_end(self) -> None:
        if self._wide_resources_layout and self.state.active_pane == "gpu-allocations":
            self.state.allocation_scroll = max(
                0, len(self._gpu_consumers()) - self._allocation_visible_rows()
            )
        elif self._wide_resources_layout and self.state.active_pane == "node":
            node = self._selected()
            consumers = self._sorted_consumers(node)
            self.state.selected_consumer = (
                max(0, len(consumers) - 1) if consumers else 0
            )
        elif self.state.view == "gpu-allocations":
            self.state.allocation_scroll = max(
                0, len(self._gpu_consumers()) - self._allocation_visible_rows()
            )
        elif self.state.expanded:
            node = self._selected()
            consumers = self._sorted_consumers(node)
            self.state.selected_consumer = (
                max(0, len(consumers) - 1) if consumers else 0
            )
        elif self.nodes:
            self.state.selected_node = self.nodes[-1].name
        self._ensure_visible()
        self._render_all()

    def action_expand(self) -> None:
        active_view = self.state.view
        if self._wide_resources_layout:
            active_view = (
                "gpu-allocations"
                if self.state.active_pane == "gpu-allocations"
                else "nodes"
            )
        if active_view in self.state.expanded_panels:
            selected = self.state.selected_panels[active_view]
            if selected:
                self.state.expanded_panels[active_view] = selected
                self._apply_layout(recompute_detail=False)
                self._render_all()
            return
        if active_view != "nodes" or self._selected() is None:
            return
        self.state.expanded = True
        self.state.active_pane = "node"
        self.state.focused_panes["nodes"] = "node"
        self._apply_layout(recompute_detail=True)
        self.call_after_refresh(self._render_all)

    def action_collapse(self) -> None:
        active_view = self.state.view
        if self._wide_resources_layout:
            active_view = (
                "gpu-allocations"
                if self.state.active_pane == "gpu-allocations"
                else "nodes"
            )
        if active_view in self.state.expanded_panels:
            if self.state.expanded_panels[active_view]:
                self.state.expanded_panels[active_view] = ""
                self._apply_layout(recompute_detail=False)
                self._render_all()
                return
            return
        if self.state.view != "nodes" or not self.state.expanded:
            return
        self.state.expanded = False
        self.state.active_pane = "nodes"
        self.state.focused_panes["nodes"] = "nodes"
        self._apply_layout(recompute_detail=True)
        self.call_after_refresh(self._render_all)

    def action_refresh_data(self) -> None:
        self._request_update(force=True)

    def action_cycle_consumer_sort(self) -> None:
        if self.state.view not in RESOURCE_VIEWS:
            return
        node = self._selected()
        before = self._sorted_consumers(node)
        before_gpu = self._gpu_consumers()
        selected_key = (
            self._consumer_identity(before[self.state.selected_consumer])
            if before and 0 <= self.state.selected_consumer < len(before)
            else None
        )
        anchor_key = (
            self._consumer_identity(before[self.state.consumer_scroll])
            if before and 0 <= self.state.consumer_scroll < len(before)
            else None
        )
        allocation_anchor_key = (
            self._consumer_identity(before_gpu[self.state.allocation_scroll])
            if before_gpu
            and 0 <= self.state.allocation_scroll < len(before_gpu)
            else None
        )
        current = _valid_consumer_sort(self.state.consumer_sort)
        self.state.consumer_sort = CONSUMER_SORTS[
            (CONSUMER_SORTS.index(current) + 1) % len(CONSUMER_SORTS)
        ]
        after = self._sorted_consumers(node)
        after_gpu = self._gpu_consumers()
        if selected_key is not None:
            for index, consumer in enumerate(after):
                if self._consumer_identity(consumer) == selected_key:
                    self.state.selected_consumer = index
                    break
        if anchor_key is not None:
            for index, consumer in enumerate(after):
                if self._consumer_identity(consumer) == anchor_key:
                    self.state.consumer_scroll = index
                    break
        if allocation_anchor_key is not None:
            for index, consumer in enumerate(after_gpu):
                if self._consumer_identity(consumer) == allocation_anchor_key:
                    self.state.allocation_scroll = index
                    break
        self._ensure_visible()
        self._render_all()
        if self.persist_consumer_sort is not None:
            try:
                self.persist_consumer_sort(self.state.consumer_sort)
            except Exception as exc:
                self.notify(
                    f"Could not save consumer sort: {exc}",
                    severity="warning",
                    timeout=4,
                )

    def action_toggle_namespace_basis(self) -> None:
        if (
            self.state.view != "gpu-allocations"
            or self.state.allocation_mode != "gpu"
        ):
            return
        self.state.namespace_basis = (
            "vram" if self.state.namespace_basis == "gpu" else "gpu"
        )
        self._render_gpu_allocations()
        self._render_footer()

    def action_toggle_allocation_basis(self) -> None:
        """Cycle GPU, requested memory, and requested CPU allocation modes."""

        if self.state.view != "gpu-allocations":
            return
        current = (
            self.state.allocation_mode
            if self.state.allocation_mode in ALLOCATION_MODES
            else "gpu"
        )
        self.state.allocation_mode = ALLOCATION_MODES[
            (ALLOCATION_MODES.index(current) + 1) % len(ALLOCATION_MODES)
        ]
        if self.state.allocation_mode in {"memory", "cpu"}:
            self.state.memory_basis = self.state.allocation_mode
        self._render_gpu_allocations()
        self._render_footer()

    def action_toggle_history_scale(self) -> None:
        """Toggle the Allocation History y-axis between linear and log scale."""

        if not self._allocation_is_visible():
            return
        self.state.history_log_scale = not self.state.history_log_scale
        self._render_gpu_allocations()
        self._render_footer()


ResourcesDashboard = FalconResourcesApp
