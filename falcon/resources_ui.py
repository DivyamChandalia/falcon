"""Realtime cluster and node resource dashboard."""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Callable, List, Mapping, Optional, Sequence

from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.errors import NoWidget
from textual.widgets import Static

from .cluster import (
    ClusterSnapshot,
    NodeSnapshot,
    WorkloadConsumer,
    is_system_consumer,
    is_system_namespace,
    natural_name_key,
)
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

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.prevent_default()
        event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.prevent_default()
        event.stop()


class ResourcesPane(Static):
    can_focus = True

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
        if callback and callable(offset_for):
            callback(offset_for(self).x)

    def on_click(self, event: events.Click) -> None:
        callback = getattr(self.app, "view_clicked", None)
        if callback:
            callback(event.get_content_offset(self).x)


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
ResourcesPane {{
    border: solid {BORDER};
    padding: 0 1;
    color: {WHITE};
    overflow: hidden;
}}
ResourcesPane:focus {{ border: solid {CYAN}; }}
#nodes-pane {{ height: 1fr; min-height: 6; }}
#node-pane {{ height: 9; min-height: 5; }}
#gpu-allocations-pane {{ height: 1fr; min-height: 8; }}
#resize-message {{ display: none; height: 1fr; content-align: center middle; color: {YELLOW}; }}
#resources-footer {{ dock: bottom; height: 1; padding: 0 1; color: {GRAY}; }}
"""


class FalconResourcesApp(App[None]):
    """Keyboard-first realtime view of schedulable request headroom."""

    # The fixed sections surrounding the two resource panes are one combined
    # header/view row, two overview rows, one controls row, and one footer. The node
    # table needs its top spacer, header, separator, and two pane borders. The
    # trailing Rich spacer can be clipped without hiding a data row.
    _FIXED_LAYOUT_HEIGHT = 5
    _NODE_TABLE_OVERHEAD = 5
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
        self._nodes_revision = 0
        self._gpu_consumers_cache_revision = -1
        self._gpu_consumers_cache: tuple[WorkloadConsumer, ...] = ()

    def compose(self) -> ComposeResult:
        yield ResourcesChrome(id="resources-header")
        yield ResourcesViewSelector(id="resources-views")
        yield ResourcesChrome(id="cluster-overview")
        yield ResourcesChrome(id="resource-controls")
        yield ResourcesPane(id="nodes-pane")
        yield ResourcesPane(id="node-pane")
        yield ResourcesPane(id="gpu-allocations-pane")
        yield ResourcesChrome(id="resize-message")
        yield ResourcesChrome(id="resources-footer")

    def on_mount(self) -> None:
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
        self._last_terminal_size = (self.size.width, self.size.height)
        try:
            self._apply_layout(recompute_detail=True)
            self._ensure_visible()
            self._render_all()
        except NoMatches:
            # Resize events can arrive as Textual is tearing down the screen.
            return

    def _check_terminal_size(self) -> None:
        current = (self.size.width, self.size.height)
        if current == self._last_terminal_size:
            return
        self._last_terminal_size = current
        try:
            self._apply_layout(recompute_detail=True)
            self._ensure_visible()
            self._render_all()
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
            history_height, _ = self._allocation_layout()
            pane = self.query_one("#gpu-allocations-pane")
            namespace_width = max(
                22,
                round(
                    pane.content_size.width
                    * (0.42 if self.size.width >= 120 else 0.38)
                ),
            )
            if y < history_height:
                # The shared legend belongs to the pie pane in the top-left
                # cell; clicking it selects the pie for Enter expansion.
                panel = "pie" if x < namespace_width else "history"
            else:
                panel = "pie" if x < namespace_width else "pods"
        self.state.selected_panels[view] = panel
        self._render_all()

    def _switch_view(self, view: str) -> None:
        view = _valid_view(view)
        if view == self.state.view:
            return
        self.state.focused_panes[self.state.view] = self.state.active_pane
        self.state.view = view
        self.state.active_pane = self.state.focused_panes[view]
        if view == "gpu-allocations":
            # The detached collector keeps writing while Nodes is visible.
            # Refresh the persisted window once when the user opens the
            # allocation view instead of querying SQLite on every inventory
            # refresh while the page is hidden.
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
        if self.persist_view is not None:
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
        index = RESOURCE_VIEWS.index(self.state.view)
        self._switch_view(RESOURCE_VIEWS[(index - 1) % len(RESOURCE_VIEWS)])

    def action_next_view(self) -> None:
        index = RESOURCE_VIEWS.index(self.state.view)
        self._switch_view(RESOURCE_VIEWS[(index + 1) % len(RESOURCE_VIEWS)])

    def _focusable_panes(self) -> tuple[str, ...]:
        """Return visible top-level panes for Tab navigation."""

        if (
            self.size.width < MINIMUM_WIDTH
            or self.size.height < RESOURCE_MINIMUM_HEIGHT
        ):
            return ()
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
        if self.state.view == "gpu-allocations":
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

        if telemetry.stale or not telemetry.sampled_pods:
            return
        identity = (
            telemetry.collected_at,
            telemetry.effective_gpus_by_namespace,
            telemetry.vram_gib_by_namespace,
        )
        if identity == self._last_allocation_history_key:
            return
        self._last_allocation_history_key = identity
        values = dict(telemetry.effective_gpus_by_namespace)
        vram_values = dict(telemetry.vram_gib_by_namespace)
        if not values:
            return
        timestamp = float(telemetry.collected_at)
        if timestamp <= 0:
            return
        # Test and adapter collectors may use monotonic cache timestamps.
        if timestamp < 946_684_800:  # 2000-01-01 UTC
            timestamp = float(self.history_clock())
        point = GPUHistoryPoint.from_mapping(timestamp, values, vram_values)
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
        if self.state.view == "gpu-allocations":
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
        )

    def _consumer_sort_key(self, consumer):
        """Return the configured stable key for a selected-node workload."""

        namespace = natural_name_key(consumer.namespace)
        workload = natural_name_key(
            consumer.workload_name or consumer.pod_name
        )
        pod = natural_name_key(consumer.pod_name)
        sort = _valid_consumer_sort(self.state.consumer_sort)
        if sort == "cpu":
            return (
                -float(consumer.requested.cpu_cores),
                namespace,
                workload,
                pod,
            )
        if sort == "memory":
            return (
                -int(consumer.requested.memory_bytes),
                namespace,
                workload,
                pod,
            )
        if sort == "gpu":
            return (
                -int(consumer.requested.gpu_count),
                namespace,
                workload,
                pod,
            )
        return (namespace, workload, pod)

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
        # ``box.SIMPLE_HEAD`` contributes a blank top row, the header, and its
        # separator before the data rows. The trailing spacer is allowed to
        # clip so the last node still fits in the fixed inventory height.
        pane = self.query_one("#nodes-pane")
        # Use the configured cell height when available. During the first
        # refresh Textual may not have committed the new region yet, while
        # the inline numeric height is already authoritative.
        if pane.styles.height.is_cells:
            return max(1, int(pane.styles.height.value) - self._NODE_TABLE_OVERHEAD)
        return max(1, pane.content_size.height - 3)

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

        return 8 if self.size.width < 100 else 9

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
        if self.state.expanded:
            if self.state.selected_consumer < self.state.consumer_scroll:
                self.state.consumer_scroll = self.state.selected_consumer
            elif self.state.selected_consumer >= self.state.consumer_scroll + visible:
                self.state.consumer_scroll = self.state.selected_consumer - visible + 1
        self.state.consumer_scroll = min(
            max(0, len(consumers) - visible),
            max(0, self.state.consumer_scroll),
        )

    def _allocation_layout(self) -> tuple[int, int]:
        pane = self.query_one("#gpu-allocations-pane")
        content_height = max(10, pane.content_size.height)
        history_height = max(7, min(16, round(content_height * 0.46)))
        if self.size.width < 100:
            # The namespace legend can contain six visible teams plus Other
            # and System/hidden. Reserve all eight rows at the 80×20 minimum.
            history_height = min(
                history_height,
                max(7, content_height - 10),
            )
        return history_height, max(5, content_height - history_height)

    def _allocation_visible_rows(self) -> int:
        if self.state.expanded_panels["gpu-allocations"] == "pods":
            pane = self.query_one("#gpu-allocations-pane")
            return max(1, pane.content_size.height - 5)
        _, bottom_height = self._allocation_layout()
        return max(1, bottom_height - 5)

    def _apply_layout(self, *, recompute_detail: bool = False) -> None:
        if not self.is_mounted:
            return
        small = (
            self.size.width < MINIMUM_WIDTH
            or self.size.height < RESOURCE_MINIMUM_HEIGHT
        )
        resize = self.query_one("#resize-message")
        ids = (
            "cluster-overview",
            "resource-controls",
            "nodes-pane",
            "node-pane",
            "gpu-allocations-pane",
        )
        if small:
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
        self.query_one("#resources-views").display = True
        for identifier in ids:
            self.query_one(f"#{identifier}").display = False
        if self.state.view != "nodes":
            self.query_one("#cluster-overview").display = True
            identifier = f"{self.state.view}-pane"
            target = self.query_one(f"#{identifier}", ResourcesPane)
            target.display = True
            target.styles.height = "1fr"
            self.state.active_pane = self.state.focused_panes[self.state.view]
            if self.app_focus and self.focused is not target:
                self.set_focus(target, scroll_visible=False)
            return
        if self.state.expanded:
            self._detail_auto_hidden = False
            self._layout_node_names = tuple(node.name for node in self.nodes)
            for identifier in ("cluster-overview", "resource-controls", "nodes-pane"):
                self.query_one(f"#{identifier}").display = False
            detail = self.query_one("#node-pane")
            detail.display = True
            detail.styles.height = "1fr"
            detail.border_title = " NODE INSPECTOR "
            if self.app_focus and self.focused is not detail:
                self.set_focus(detail, scroll_visible=False)
        else:
            for identifier in ("cluster-overview", "resource-controls", "nodes-pane"):
                self.query_one(f"#{identifier}").display = True
            nodes_pane = self.query_one("#nodes-pane", ResourcesPane)
            detail = self.query_one("#node-pane", ResourcesPane)
            detail.border_title = " SELECTED NODE "

            # Give the node inventory a stable height that includes every
            # row, then let the selected-node pane consume the remainder.
            # This makes the detail pane shrink continuously and disappear
            # only when it reaches its minimum useful height.  Recompute only
            # after a resize or actual inventory change; telemetry redraws
            # must not toggle the layout.
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
            # Telemetry refreshes run this layout path every second. Preserve
            # the pane the user clicked instead of unconditionally returning
            # focus to the node inventory. If the detail pane has to be
            # hidden, the node inventory is the only valid focus target.
            if self.state.active_pane == "node" and detail.display:
                focus_target = detail
            else:
                self.state.active_pane = "nodes"
                focus_target = nodes_pane
            if self.app_focus and self.focused is not focus_target:
                # Never leave a refresh-owned focus change queued behind a
                # later mouse event. The click must be the last operation and
                # therefore the winner.
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
        left = Text(no_wrap=True, overflow="ellipsis")
        left.append(
            f"{snapshot.schedulable_nodes}/{snapshot.total_nodes} NODES  ",
            style=f"bold {GREEN if snapshot.schedulable_nodes else YELLOW}",
        )
        if self.size.width >= 100:
            left.append(
                f"{snapshot.running_jobs} RUNNING  ",
                style=f"bold {GREEN}",
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
        if self.size.width >= 100:
            left.append("CPU ", style=f"bold {GRAY}")
            left.append(
                f"{_short_cpu(headroom.cpu_cores)}/"
                f"{_short_cpu(snapshot.allocatable.cpu_cores)}  ",
                style=f"bold {cpu_color}",
            )
        # Keep scheduler memory headroom visible even at the 80-column
        # minimum. The compact Nodes layout still has room for a short
        # ``free/allocatable`` value, and hiding it makes the top summary
        # inconsistent with the per-node RAM column.
        left.append("MEM ", style=f"bold {GRAY}")
        left.append(
            f"{_short_memory(headroom.memory_bytes)}/"
            f"{_short_memory(snapshot.allocatable.memory_bytes)}",
            style=f"bold {memory_color}",
        )

        right = Text(
            "GPU AVAILABLE  " if self.size.width >= 130 else "",
            style=f"bold {GRAY}",
        )
        availability = list(snapshot.gpu_availability.values())
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
        gap = max(2, self.size.width - len(left.plain) - len(right.plain) - 4)
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
        if self._gpu_consumers_cache_revision == self._nodes_revision:
            return self._gpu_consumers_cache
        consumers = [
            consumer
            for node in self._gpu_nodes()
            if _eligible(node)
            for consumer in node.consumers
            if consumer.requested.gpu_count > 0 and not is_system_consumer(consumer)
        ]
        consumers.sort(
            key=lambda consumer: (
                -consumer.requested.gpu_count,
                consumer.namespace.casefold(),
                natural_name_key(consumer.node_name),
                consumer.pod_name.casefold(),
            )
        )
        self._gpu_consumers_cache = tuple(consumers)
        self._gpu_consumers_cache_revision = self._nodes_revision
        return self._gpu_consumers_cache

    def _namespace_categories(self) -> tuple[tuple[str, float], ...]:
        source = (
            self.gpu_telemetry.vram_gib_by_namespace
            if self.state.namespace_basis == "vram"
            else self.gpu_telemetry.effective_gpus_by_namespace
        )
        visible: defaultdict[str, float] = defaultdict(float)
        hidden = 0.0
        for namespace, value in source:
            if is_system_namespace(namespace):
                hidden += value
            else:
                visible[namespace] += value
        ordered = sorted(visible.items(), key=lambda item: (-item[1], item[0].casefold()))
        categories = list(ordered[:6])
        other = sum(value for _, value in ordered[6:])
        if other:
            categories.append(("Other", other))
        if hidden:
            categories.append(("System/hidden", hidden))
        return tuple(categories)

    def _allocation_empty_label(self, basis: str) -> str:
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
        if basis == "vram" and not self.gpu_telemetry.vram_gib_by_namespace:
            return "VRAM allocation unavailable"
        return (
            "No VRAM allocation"
            if basis == "vram"
            else "No GPU allocation"
        )

    def _gpu_pod_table(
        self,
        consumers: Optional[Sequence[WorkloadConsumer]] = None,
    ) -> Table:
        width = self.size.width
        table = Table(
            box=box.SIMPLE_HEAD,
            expand=True,
            padding=(0, 1),
            collapse_padding=True,
            header_style=f"bold {CYAN_2}",
        )
        narrow = width < 100
        table.add_column(
            "NS" if narrow else "NAMESPACE",
            ratio=1,
            no_wrap=True,
            overflow="ellipsis",
        )
        table.add_column(
            "NODE",
            width=6 if narrow else 10,
            no_wrap=True,
            overflow="ellipsis",
        )
        # GPU allocations are grouped by their owning workload.  A Pod name
        # is generated per attempt and is therefore noisy (and changes when a
        # Job is retried); show the stable Job/workload name instead.
        table.add_column("JOB", ratio=2, no_wrap=True, overflow="ellipsis")
        table.add_column(
            "GPU/#" if narrow else "GPU MODEL / COUNT",
            width=7 if narrow else 17,
            justify="right",
            no_wrap=True,
        )
        consumers = tuple(consumers) if consumers is not None else self._gpu_consumers()
        start = self.state.allocation_scroll
        visible = self._allocation_visible_rows()
        for consumer in consumers[start : start + visible]:
            vector = consumer.requested
            model = vector.gpu_model
            if not model:
                node = next(
                    (candidate for candidate in self.nodes if candidate.name == consumer.node_name),
                    None,
                )
                model = node.gpu_model if node else None
            table.add_row(
                Text(consumer.namespace, style=WHITE),
                Text(consumer.node_name, style=GRAY),
                Text(consumer.workload_name or consumer.pod_name, style=WHITE),
                Text(
                    (
                        f"{model or 'GPU'}×{vector.gpu_count}"
                        if narrow
                        else f"{model or 'GPU'} ×{vector.gpu_count}"
                    ),
                    style=CYAN,
                ),
            )
        if not consumers:
            table.add_row(Text("No active GPU-requesting Jobs", style=MUTED))
        return table

    @staticmethod
    def _allocation_legend_width(width: int) -> int:
        """Reserve one readable shared legend column for both charts."""

        # Keep enough inner width for the aggregate's "Total usage" label at
        # the supported 80-column minimum after the panel and grid padding.
        return min(40, max(24, round(max(1, width) * 0.28)))

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
            unit="G" if basis == "vram" else "",
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
        width = max(20, target.content_size.width)
        expanded = self.state.expanded_panels["gpu-allocations"]
        basis = self.state.namespace_basis
        categories = self._namespace_categories()
        colors = self._allocation_colors(categories)
        consumers = self._gpu_consumers()
        telemetry = self.gpu_telemetry
        render_key = (
            expanded,
            basis,
            self.state.selected_panels.get("gpu-allocations", ""),
            self.state.allocation_scroll,
            self.app_focus,
            self.color_mode,
            width,
            target.content_size.height,
            self.history_hours,
            self._history_revision,
            telemetry.effective_gpus_by_namespace,
            telemetry.vram_gib_by_namespace,
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
        legend_width = self._allocation_legend_width(width)
        # The chart sits inside a bordered Panel and a padded grid cell.
        # Reserve both columns so Rich cannot wrap a line at the cell edge.
        # The chart is inside an outer Panel (two border cells) and a
        # two-column grid with one-cell padding on both sides of each cell.
        # Reserve all six cells so Rich never wraps the final step segment
        # onto the following terminal row in the expanded layout.
        chart_width = max(14, width - legend_width - 6)
        if expanded:
            height = max(8, target.content_size.height)
            target.border_subtitle = " Enter expand · Esc restore "
            if expanded == "history":
                history_key = (
                    "expanded",
                    self._history_render_signature(),
                    categories,
                    basis,
                    width,
                    height,
                )
                if (
                    history_key != self._history_cache_key
                    or self._history_cache is None
                ):
                    self._history_cache_key = history_key
                    self._history_cache = render_gpu_history(
                        self.history,
                        width=chart_width,
                        height=max(5, height - 2),
                        basis=basis,
                        categories=categories,
                        colors=colors,
                        show_legend=False,
                    )
                content = self._allocation_chart_with_legend(
                    self._history_cache,
                    categories,
                    basis=basis,
                    width=width,
                    height=height,
                    colors=colors,
                )
                target.update(
                    Panel(
                        content,
                        title=Text(" ALLOCATION HISTORY ", style=f"bold {PALETTE.accent}"),
                        subtitle=Text(
                            f" {self._history_window_label} ",
                            style=GRAY,
                        ),
                        box=box.SQUARE,
                        border_style=BORDER,
                        height=height,
                    )
                )
                return
            if expanded == "pods":
                target.update(
                    Panel(
                        self._gpu_pod_table(consumers),
                        title=Text(
                            " GPU-REQUESTING JOBS ",
                            style=f"bold {PALETTE.accent}",
                        ),
                        box=box.SQUARE,
                        border_style=BORDER,
                        height=height,
                    )
                )
                return
            unit = "G" if basis == "vram" else ""
            pie_key = ("expanded", categories, basis, chart_width, height)
            if pie_key != self._pie_cache_key or self._pie_cache is None:
                self._pie_cache_key = pie_key
                self._pie_cache = render_namespace_pie(
                    categories,
                    width=chart_width,
                    height=max(5, height - 2),
                    unit=unit,
                    empty_label=self._allocation_empty_label(basis),
                    colors=colors,
                    show_legend=False,
                )
            content = self._allocation_chart_with_legend(
                Align.center(self._pie_cache, vertical="middle"),
                categories,
                basis=basis,
                width=width,
                height=height,
                colors=colors,
            )
            target.update(
                Panel(
                    content,
                    title=Text(" ALLOCATION BY NAMESPACE ", style=f"bold {PALETTE.accent}"),
                    box=box.SQUARE,
                    border_style=BORDER,
                    height=height,
                )
            )
            return
        history_height, bottom_height = self._allocation_layout()
        # The shared namespace legend sits above the pie in the left column;
        # Allocation History and GPU-requesting Jobs occupy the right column.
        namespace_width = max(
            22,
            round(width * (0.42 if width >= 120 else 0.38)),
        )
        chart_width = max(14, width - namespace_width - 4)
        history_key = (
            self._history_render_signature(),
            categories,
            basis,
            chart_width,
            history_height,
        )
        if history_key != self._history_cache_key or self._history_cache is None:
            self._history_cache_key = history_key
            self._history_cache = render_gpu_history(
                self.history,
                width=chart_width,
                height=max(5, history_height - 2),
                basis=basis,
                categories=categories,
                colors=colors,
                show_legend=False,
            )
        history_panel = Panel(
            self._history_cache,
            title=Text(" ALLOCATION HISTORY ", style=f"bold {PALETTE.accent}"),
            subtitle=Text(
                f" {self._history_window_label} ",
                style=GRAY,
            ),
            box=box.SQUARE,
            border_style=(
                CYAN
                if self.state.selected_panels.get("gpu-allocations") == "history"
                else BORDER
            ),
            height=history_height,
        )
        unit = "G" if basis == "vram" else ""
        empty_label = self._allocation_empty_label(basis)
        pie_key = (
            categories,
            basis,
            empty_label,
            namespace_width,
            bottom_height,
        )
        if pie_key != self._pie_cache_key or self._pie_cache is None:
            self._pie_cache_key = pie_key
            self._pie_cache = render_namespace_pie(
                categories,
                width=max(16, namespace_width - 4),
                height=max(3, bottom_height - 2),
                unit=unit,
                empty_label=empty_label,
                colors=colors,
                show_legend=False,
            )
        legend = render_allocation_legend(
            categories,
            width=max(16, namespace_width - 4),
            height=max(1, history_height - 2),
            unit=unit,
            colors=colors,
            columns=1,
            include_total=True,
        )
        legend_panel = Panel(
            legend,
            title=Text(" NAMESPACE LEGEND ", style=f"bold {PALETTE.accent}"),
            box=box.SQUARE,
            border_style=BORDER,
            height=history_height,
        )
        pie_panel = Panel(
            Align.center(self._pie_cache, vertical="middle"),
            title=Text(
                " ALLOCATION BY NAMESPACE ",
                style=f"bold {PALETTE.accent}",
            ),
            subtitle=Text(
                (
                    f" {self.gpu_telemetry.target_pods} Pods accounted "
                    if self.gpu_telemetry.target_pods
                    else " No GPU allocations "
                ),
                style=RED if self.gpu_telemetry.stale else GRAY,
            ),
            box=box.SQUARE,
            border_style=(
                CYAN
                if self.state.selected_panels.get("gpu-allocations") == "pie"
                else BORDER
            ),
            height=bottom_height,
        )
        visible_requests = sum(
            consumer.requested.gpu_count for consumer in consumers
        )
        total_requested = _gpu_totals(self._gpu_nodes())[1]
        hidden = max(0, total_requested - visible_requests)
        pods = Panel(
            self._gpu_pod_table(),
            title=Text(" GPU-REQUESTING JOBS ", style=f"bold {PALETTE.accent}"),
            subtitle=(
                Text(f" {visible_requests} visible + {hidden} system/hidden ", style=GRAY)
                if hidden
                else Text(f" {visible_requests} requested ", style=GRAY)
            ),
            box=box.SQUARE,
            border_style=(
                CYAN
                if self.state.selected_panels.get("gpu-allocations") == "pods"
                else BORDER
            ),
            height=bottom_height,
        )
        content = Table.grid(expand=True)
        content.add_column(width=namespace_width)
        content.add_column(ratio=1)
        content.add_row(legend_panel, history_panel)
        content.add_row(pie_panel, pods)
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
        width = self.size.width
        content_width = max(60, target.content_size.width)
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
        if not self.state.expanded:
            sched, sched_color = _schedulable(node)
            line = Text(node.name, style=f"bold {CYAN}")
            if self.size.width >= 100 or sched != "Yes":
                line.append(f"   {sched}", style=sched_color)
            line.append("   CPU ", style=GRAY)
            line.append(
                f"{_short_cpu(node.requested.cpu_cores)}/"
                f"{_short_cpu(node.allocatable.cpu_cores)}",
                style=_request_pressure_color(
                    node.requested.cpu_cores,
                    node.allocatable.cpu_cores,
                ),
            )
            line.append("   RAM ", style=GRAY)
            line.append(
                f"{_short_memory(node.requested.memory_bytes)}/"
                f"{_short_memory(node.allocatable.memory_bytes)}",
                style=_request_pressure_color(
                    node.requested.memory_bytes,
                    node.allocatable.memory_bytes,
                ),
            )
            line.append(
                f"   GPU {node.gpu_model or '-'} ",
                style=WHITE,
            )
            line.append(
                f"{node.requested.gpu_count}/{node.allocatable.gpu_count}",
                style=_request_pressure_color(
                    node.requested.gpu_count,
                    node.allocatable.gpu_count,
                ),
            )
            if node.gpu_memory_bytes_per_device is not None:
                line.append(
                    f"   VRAM {_short_memory(node.gpu_memory_bytes_per_device)}"
                    + ("/GPU" if self.size.width >= 100 else ""),
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
        if self.size.width < 100:
            node_value = Text(_truncate(node.name, 35), style=WHITE)
            node_value.append(" · ", style=GRAY)
            node_value.append(sched, style=sched_color)
            gpu_value = Text(f"{node.gpu_model or '-'}  ", style=WHITE)
            gpu_value.append(
                f"{node.requested.gpu_count}/{node.allocatable.gpu_count}",
                style=_request_pressure_color(
                    node.requested.gpu_count,
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
                    f"{_short_cpu(node.requested.cpu_cores)} / "
                    f"{_short_cpu(node.allocatable.cpu_cores)}",
                    style=_request_pressure_color(
                        node.requested.cpu_cores,
                        node.allocatable.cpu_cores,
                    ),
                ),
            )
            facts.add_row(
                "RAM",
                Text(
                    f"{_short_memory(node.requested.memory_bytes)} / "
                    f"{_short_memory(node.allocatable.memory_bytes)}",
                    style=_request_pressure_color(
                        node.requested.memory_bytes,
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
                "CPU used / alloc",
                Text(
                    f"{_short_cpu(node.requested.cpu_cores)} / "
                    f"{_short_cpu(node.allocatable.cpu_cores)}",
                    style=_request_pressure_color(
                        node.requested.cpu_cores,
                        node.allocatable.cpu_cores,
                    ),
                ),
            )
            facts.add_row(
                "RAM capacity",
                _short_memory(node.capacity.memory_bytes),
                "RAM used / alloc",
                Text(
                    f"{_short_memory(node.requested.memory_bytes)} / "
                    f"{_short_memory(node.allocatable.memory_bytes)}",
                    style=_request_pressure_color(
                        node.requested.memory_bytes,
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
                "GPU used / alloc",
                Text(
                    f"{node.requested.gpu_count}/{node.allocatable.gpu_count}",
                    style=_request_pressure_color(
                        node.requested.gpu_count,
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
        elif self.state.view == "gpu-allocations":
            expanded_panel = self.state.expanded_panels["gpu-allocations"]
            basis_label = "VRAM" if self.state.namespace_basis == "vram" else "COUNT"
            prefix = "Esc restore panels   " if expanded_panel else ""
            value = prefix + (
                "←/→ Views  ↑/↓ Scroll Jobs  Enter Expand  Tab  "
                f"v {basis_label}  r Refresh  q Quit"
                if self.size.width < 100
                else (
                    "←/→ Views   ↑/↓ Scroll Jobs   Enter Expand   "
                    "Tab Next pane   "
                    f"v {basis_label}   r Refresh   q Quit"
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
            if self.state.view == "nodes":
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
        # Rich's SIMPLE_HEAD box renders a blank top line, header, and header
        # separator before the first data row.  Ignore those non-row lines;
        # ``offset`` is relative to the pane content (not its border).
        row = offset.y - 3
        if row < 0:
            return
        index = self.state.node_scroll + row
        if index < len(self.nodes):
            self.state.selected_node = self.nodes[index].name
            self.state.selected_consumer = 0
            self.state.consumer_scroll = 0
            self._render_all()

    def consumer_clicked(self, offset) -> None:
        """Select a workload row in the expanded node inspector."""

        if (
            self.state.view != "nodes"
            or not self.state.expanded
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

        # The inspector is a Rich grid with a capacity panel followed by a
        # bordered consumer panel. That panel contributes its top border,
        # spacer, table header, and header separator before the first data
        # row. ``offset`` is relative to the pane content, so subtract those
        # stable rows and then add the current scroll anchor.
        row = y - self._expanded_facts_height() - 4
        if row < 0:
            return
        index = self.state.consumer_scroll + row
        if index < 0 or index >= len(consumers):
            return
        self.state.selected_consumer = index
        self._ensure_visible()
        self._render_all()

    def _move(self, amount: int) -> None:
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
        if self.state.expanded:
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
        if self.state.view == "gpu-allocations":
            self._move(-self._allocation_visible_rows())
            return
        self._move(
            -(self._visible_consumers() if self.state.expanded else self._visible_nodes())
        )

    def action_page_down(self) -> None:
        if self.state.view == "gpu-allocations":
            self._move(self._allocation_visible_rows())
            return
        self._move(
            self._visible_consumers() if self.state.expanded else self._visible_nodes()
        )

    def action_home(self) -> None:
        if self.state.view == "gpu-allocations":
            self.state.allocation_scroll = 0
        elif self.state.expanded:
            self.state.selected_consumer = 0
        elif self.nodes:
            self.state.selected_node = self.nodes[0].name
        self._ensure_visible()
        self._render_all()

    def action_end(self) -> None:
        if self.state.view == "gpu-allocations":
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
        if self.state.view in self.state.expanded_panels:
            selected = self.state.selected_panels[self.state.view]
            if selected:
                self.state.expanded_panels[self.state.view] = selected
                self._render_all()
            return
        if self.state.view != "nodes" or self._selected() is None:
            return
        self.state.expanded = True
        self.state.active_pane = "node"
        self.state.focused_panes["nodes"] = "node"
        self._apply_layout(recompute_detail=True)
        self.call_after_refresh(self._render_all)

    def action_collapse(self) -> None:
        if self.state.view in self.state.expanded_panels:
            if self.state.expanded_panels[self.state.view]:
                self.state.expanded_panels[self.state.view] = ""
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
        if self.state.view != "nodes":
            return
        node = self._selected()
        before = self._sorted_consumers(node)
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
        current = _valid_consumer_sort(self.state.consumer_sort)
        self.state.consumer_sort = CONSUMER_SORTS[
            (CONSUMER_SORTS.index(current) + 1) % len(CONSUMER_SORTS)
        ]
        after = self._sorted_consumers(node)
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
        if self.state.view != "gpu-allocations":
            return
        self.state.namespace_basis = (
            "vram" if self.state.namespace_basis == "gpu" else "gpu"
        )
        self._render_gpu_allocations()
        self._render_footer()


ResourcesDashboard = FalconResourcesApp
