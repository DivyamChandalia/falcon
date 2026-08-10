"""Realtime cluster and node resource dashboard."""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
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
    MINIMUM_HEIGHT,
    MINIMUM_WIDTH,
    MUTED,
    PALETTE,
    RED,
    WHITE,
    YELLOW,
)

RESOURCE_VIEWS = ("nodes", "gpu-overview", "gpu-allocations")
RESOURCE_VIEW_LABELS = {
    "nodes": "Nodes",
    "gpu-overview": "GPU Overview",
    "gpu-allocations": "GPU Allocations",
}


def _valid_view(value: object) -> str:
    normalized = str(value or "")
    return normalized if normalized in RESOURCE_VIEWS else "nodes"


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


@dataclass
class ResourcesViewState:
    view: str = "nodes"
    selected_node: str = ""
    node_scroll: int = 0
    expanded: bool = False
    selected_consumer: int = 0
    consumer_scroll: int = 0
    active_pane: str = "nodes"
    overview_scroll: int = 0
    allocation_scroll: int = 0
    namespace_basis: str = "gpu"
    expanded_panels: dict[str, str] = field(
        default_factory=lambda: {
            "gpu-overview": "",
            "gpu-allocations": "",
        }
    )
    selected_panels: dict[str, str] = field(
        default_factory=lambda: {
            "gpu-overview": "summary",
            "gpu-allocations": "history",
        }
    )
    focused_panes: dict[str, str] = field(
        default_factory=lambda: {
            "nodes": "nodes",
            "gpu-overview": "gpu-overview",
            "gpu-allocations": "gpu-allocations",
        }
    )


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
        if self.id == "gpu-overview-pane" or self.id == "gpu-allocations-pane":
            callback = getattr(self.app, "gpu_panel_selected", None)
            if callback:
                callback(
                    self.id.replace("-pane", ""),
                    event.get_content_offset(self),
                )
            return
        if self.id != "nodes-pane":
            return
        callback = getattr(self.app, "node_clicked", None)
        if callback:
            callback(event.get_content_offset(self))


class ResourcesViewSelector(Static):
    """Clickable view labels; keyboard navigation remains available globally."""

    def on_click(self, event: events.Click) -> None:
        callback = getattr(self.app, "view_clicked", None)
        if callback:
            callback(event.get_content_offset(self).x)


CSS = f"""
Screen {{ background: {BACKGROUND}; color: {WHITE}; overflow: hidden; }}
Static {{ background: {BACKGROUND}; }}
#resources-header {{ height: 1; padding: 0 1; color: {WHITE}; }}
#resources-views {{ height: 1; padding: 0 1; color: {GRAY}; }}
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
#gpu-overview-pane, #gpu-allocations-pane {{ height: 1fr; min-height: 8; }}
#resize-message {{ display: none; height: 1fr; content-align: center middle; color: {YELLOW}; }}
#resources-footer {{ height: 1; padding: 0 1; color: {GRAY}; }}
"""


class FalconResourcesApp(App[None]):
    """Keyboard-first realtime view of schedulable request headroom."""

    # The fixed sections surrounding the two resource panes are one header,
    # one view selector, two overview rows, one controls row, and one footer. The node
    # table needs its top spacer, header, separator, and two pane borders. The
    # trailing Rich spacer can be clipped without hiding a data row.
    _FIXED_LAYOUT_HEIGHT = 6
    _NODE_TABLE_OVERHEAD = 5
    _DETAIL_MIN_HEIGHT = 5

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
        Binding("enter", "expand", "Inspect"),
        Binding("escape", "collapse", "Back", show=False),
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
        initial_view: str = "nodes",
        persist_view: Optional[Callable[[str], object]] = None,
        telemetry_collector=None,
        telemetry_refresh_seconds: float = 5.0,
    ) -> None:
        super().__init__()
        self.collector = collector
        self.refresh_seconds = refresh_seconds
        self.node_filter = (node_filter or "").lower()
        self.gpu_filter = (gpu_filter or "").lower()
        self.clock = clock or self._snapshot_clock
        self.history_clock = history_clock or time.time
        view = _valid_view(initial_view)
        self.state = ResourcesViewState(
            view=view,
            active_pane={
                "nodes": "nodes",
                "gpu-overview": "gpu-overview",
                "gpu-allocations": "gpu-allocations",
            }[view],
        )
        self.persist_view = persist_view
        # Kept as a soft compatibility argument for callers of the preview
        # API. Resource allocation data comes from ``collector`` itself.
        del telemetry_collector, telemetry_refresh_seconds
        self.snapshot = ClusterSnapshot.empty()
        self.nodes: List[NodeSnapshot] = []
        self.history: list[GPUHistoryPoint] = []
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

    def compose(self) -> ComposeResult:
        yield Static(id="resources-header")
        yield ResourcesViewSelector(id="resources-views")
        yield Static(id="cluster-overview")
        yield Static(id="resource-controls")
        yield ResourcesPane(id="nodes-pane")
        yield ResourcesPane(id="node-pane")
        yield ResourcesPane(id="gpu-overview-pane")
        yield ResourcesPane(id="gpu-allocations-pane")
        yield Static(id="resize-message")
        yield Static(id="resources-footer")

    def on_mount(self) -> None:
        self._set_titles()
        self._apply_layout(recompute_detail=True)
        self._request_update(force=True)
        self.set_interval(self.refresh_seconds, self._request_update)
        self.set_interval(0.1, self._drain_results)
        self.set_interval(0.1, self._check_terminal_size)
        self.set_interval(1.0, self._tick_clock)
        self._render_all()

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
            "gpu-overview": {"gpu-overview"},
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
        if view == "gpu-overview":
            pane = self.query_one("#gpu-overview-pane")
            summary_height = 3 if self.size.width < 120 else 5
            if y < summary_height:
                panel = "summary"
            elif x < max(1, pane.content_size.width // 2):
                panel = "free"
            else:
                panel = "pressure"
        elif view == "gpu-allocations":
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
        try:
            snapshot = self._results.get_nowait()
        except queue.Empty:
            return
        self._refreshing = False
        previous_node_name = self.state.selected_node
        previous_node = self._selected()
        previous_consumers = previous_node.visible_consumers if previous_node else ()
        previous_selected_key = (
            self._consumer_identity(previous_consumers[self.state.selected_consumer])
            if previous_consumers and 0 <= self.state.selected_consumer < len(previous_consumers)
            else None
        )
        previous_anchor_key = (
            self._consumer_identity(previous_consumers[self.state.consumer_scroll])
            if previous_consumers and 0 <= self.state.consumer_scroll < len(previous_consumers)
            else None
        )
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
        self.nodes = nodes
        # The resource collector already runs in the background against the
        # The collector refreshes the configured local metrics endpoint in the
        # background. Derive both selectable allocation bases from every fresh
        # snapshot, even while this page is hidden.
        self.gpu_telemetry = allocation_snapshot(
            self.nodes,
            collected_at=snapshot.collected_at,
            stale=snapshot.stale,
            error=snapshot.error or "",
        )
        self._record_gpu_history(self.gpu_telemetry)
        current_names = tuple(node.name for node in nodes)
        names = {node.name for node in nodes}
        if self.state.selected_node not in names:
            self.state.selected_node = nodes[0].name if nodes else ""
            self.state.selected_consumer = 0
            self.state.consumer_scroll = 0
        elif self.state.selected_node == previous_node_name:
            refreshed_node = self._selected()
            refreshed_consumers = refreshed_node.visible_consumers if refreshed_node else ()
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
        self.history.append(
            GPUHistoryPoint.from_mapping(timestamp, values, vram_values)
        )
        self.history.sort(key=lambda point: point.timestamp)
        newest = self.history[-1].timestamp
        cutoff = newest - HISTORY_SECONDS
        self.history = [point for point in self.history if point.timestamp >= cutoff]
        if len(self.history) > HISTORY_LIMIT:
            self.history = self.history[-HISTORY_LIMIT:]

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
        if self.state.view == "gpu-overview":
            visible = self._overview_visible_rows()
            self.state.overview_scroll = min(
                max(0, len(self._gpu_nodes()) - visible),
                max(0, self.state.overview_scroll),
            )
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
        node = self._selected()
        consumers = node.visible_consumers if node else ()
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

    def _overview_visible_rows(self) -> int:
        pane = self.query_one("#gpu-overview-pane")
        summary_height = 3 if self.size.width < 120 else 5
        return max(1, pane.content_size.height - summary_height - 2)

    def _allocation_layout(self) -> tuple[int, int]:
        pane = self.query_one("#gpu-allocations-pane")
        content_height = max(10, pane.content_size.height)
        history_height = max(7, min(16, round(content_height * 0.46)))
        if self.size.width < 100:
            # The namespace legend can contain six visible teams plus Other
            # and System/hidden. Reserve all eight rows at the 80×22 minimum.
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
        small = self.size.width < MINIMUM_WIDTH or self.size.height < MINIMUM_HEIGHT
        resize = self.query_one("#resize-message")
        ids = (
            "cluster-overview",
            "resource-controls",
            "nodes-pane",
            "node-pane",
            "gpu-overview-pane",
            "gpu-allocations-pane",
        )
        if small:
            for identifier in ids:
                self.query_one(f"#{identifier}").display = False
            self.query_one("#resources-views").display = False
            resize.display = True
            resize.update(
                f"Falcon Resources requires at least "
                f"{MINIMUM_WIDTH}×{MINIMUM_HEIGHT}.\n"
                f"Current terminal: {self.size.width}×{self.size.height}.\n\n"
                "Resize to inspect cluster resources."
            )
            return
        resize.display = False
        self.query_one("#resources-views").display = True
        for identifier in ids:
            self.query_one(f"#{identifier}").display = False
        if self.state.view != "nodes":
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
        line = Text(" " * offset)
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
            "gpu-overview": ("gpu-overview-pane", "GPU OVERVIEW"),
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
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append(
            f"{snapshot.schedulable_nodes}/{snapshot.total_nodes} NODES  ",
            style=f"bold {GREEN if snapshot.schedulable_nodes else YELLOW}",
        )
        text.append(
            f"{snapshot.running_jobs} {'RUNNING' if self.size.width >= 100 else 'RUN'}  ",
            style=f"bold {GREEN}",
        )
        cpu_color = _resource_headroom_color(
            headroom.cpu_cores,
            snapshot.allocatable.cpu_cores,
        )
        memory_color = _resource_headroom_color(
            headroom.memory_bytes,
            snapshot.allocatable.memory_bytes,
        )
        text.append("CPU ", style=f"bold {GRAY}")
        text.append(
            f"{_short_cpu(headroom.cpu_cores)}/"
            f"{_short_cpu(snapshot.allocatable.cpu_cores)}  ",
            style=f"bold {cpu_color}",
        )
        text.append("MEM ", style=f"bold {GRAY}")
        text.append(
            f"{_short_memory(headroom.memory_bytes)}/"
            f"{_short_memory(snapshot.allocatable.memory_bytes)}",
            style=f"bold {memory_color}",
        )
        gpu = Text("GPU  ", style=f"bold {GRAY}")
        if snapshot.gpu_availability:
            for index, availability in enumerate(snapshot.gpu_availability.values()):
                if index:
                    gpu.append("   ")
                gpu.append(
                    f"{availability.model} "
                    f"{availability.request_headroom}/{availability.allocatable}",
                    style="bold "
                    + _gpu_headroom_color(
                        availability.request_headroom,
                        availability.allocatable,
                    ),
                )
        else:
            gpu.append("-", style=f"bold {MUTED}")
        available_width = max(1, self.size.width - 2)
        if len(text.plain) + len(gpu.plain) + 2 <= available_width:
            text.append(" " * max(2, available_width - len(text.plain) - len(gpu.plain)))
            text.append_text(gpu)
        else:
            # Keep the overview a single row even when several long GPU model
            # names cannot fit. Rich's no-wrap ellipsis then clips only the
            # overflowing tail instead of moving CPU/memory to another row.
            text.append("  ")
            text.append_text(gpu)
        self.query_one("#cluster-overview", Static).update(text)

    def _render_controls(self) -> None:
        filters = []
        if self.node_filter:
            filters.append(f"node={self.node_filter}")
        if self.gpu_filter:
            filters.append(f"gpu={self.gpu_filter}")
        suffix = f"  Filters: {', '.join(filters)}" if filters else ""
        self.query_one("#resource-controls", Static).update(
            Text(
                "Request headroom (free/allocatable) · Enter inspect node"
                f"{suffix}",
                style=GRAY,
            )
        )

    @staticmethod
    def _gpu_bar(
        node: NodeSnapshot,
        *,
        width: int,
        pressure: bool,
    ) -> Text:
        allocatable = max(0, node.allocatable.gpu_count)
        requested = max(0, node.requested.gpu_count)
        free = max(0, allocatable - requested)
        eligible = _eligible(node)
        value = requested if pressure else free
        ratio = (
            min(1.0, value / allocatable)
            if eligible and allocatable
            else 0.0
        )
        suffix = (
            f"{requested}/{allocatable}"
            if pressure
            else f"{free}/{allocatable}"
        )
        if not eligible:
            suffix = "excluded"
        label_width = max(8, min(24, width // 3))
        bar_width = max(3, width - label_width - len(suffix) - 4)
        label = f"{node.name} · {node.gpu_model or 'GPU'}"
        label = _truncate(label, label_width).ljust(label_width)
        filled = min(bar_width, round(ratio * bar_width))
        color = _gpu_headroom_color(free, allocatable) if eligible else RED
        line = Text()
        line.append(label, style=WHITE if eligible else MUTED)
        line.append("  ")
        line.append("█" * filled, style=color)
        line.append("·" * (bar_width - filled), style=BORDER)
        line.append(f"  {suffix}", style=color if eligible else RED)
        return line

    def _gpu_summary(self, *, width: int, height: int):
        allocatable, requested, free, pressure = _gpu_totals(self._gpu_nodes())
        color = _gpu_headroom_color(free, allocatable)
        selected = self.state.selected_panels.get("gpu-overview") == "summary"
        summary_border = CYAN if selected else BORDER
        if self.size.width < 120:
            line = Text(justify="center")
            values = (
                ("ALLOCATABLE", str(allocatable), CYAN),
                ("REQUESTED", str(requested), color),
                ("FREE", str(free), color),
                ("PRESSURE", f"{pressure:.0f}%", color),
            )
            for index, (label, value, value_color) in enumerate(values):
                if index:
                    line.append("   ", style=BORDER)
                line.append(f"{label} ", style=GRAY)
                line.append(value, style=f"bold {value_color}")
            return Panel(
                line,
                box=box.SQUARE,
                border_style=summary_border,
                height=height,
            )

        cards = Table.grid(expand=True, padding=(0, 1))
        for _ in range(4):
            cards.add_column(ratio=1)
        card_values = (
            ("ALLOCATABLE GPUS", str(allocatable), CYAN),
            ("REQUESTED NOW", str(requested), color),
            ("FREE GPUS", str(free), color),
            ("REQUEST PRESSURE", f"{pressure:.1f}%", color),
        )
        cards.add_row(
            *(
                Panel(
                    Align.center(Text(value, style=f"bold {value_color}"), vertical="middle"),
                    title=Text(f" {label} ", style=f"bold {GRAY}"),
                    box=box.SQUARE,
                    border_style=summary_border,
                    height=height,
                )
                for label, value, value_color in card_values
            )
        )
        return cards

    def _render_gpu_overview(self) -> None:
        target = self.query_one("#gpu-overview-pane", ResourcesPane)
        width = max(20, target.content_size.width)
        height = max(8, target.content_size.height)
        expanded = self.state.expanded_panels["gpu-overview"]
        summary_height = 3 if self.size.width < 120 else 5
        if expanded:
            if expanded == "summary":
                target.border_subtitle = " Enter expand selected · Esc restore "
                target.update(
                    Panel(
                        self._gpu_summary(width=width, height=max(5, height - 2)),
                        title=Text(" GPU SUMMARY ", style=f"bold {PALETTE.accent}"),
                        box=box.SQUARE,
                        border_style=BORDER,
                        height=height,
                    )
                )
                return
            nodes = self._gpu_nodes()
            visible = max(1, height - 3)
            start = self.state.overview_scroll
            shown = nodes[start : start + visible]
            lines = Text()
            if shown:
                for index, node in enumerate(shown):
                    lines.append_text(
                        self._gpu_bar(
                            node,
                            width=max(12, width - 6),
                            pressure=expanded == "pressure",
                        )
                    )
                    if index != len(shown) - 1:
                        lines.append("\n")
            else:
                lines.append("No GPU nodes match the active filters.", style=MUTED)
            title = (
                " FREE GPUS PER NODE "
                if expanded == "free"
                else " GPU REQUEST PRESSURE PER NODE "
            )
            target.border_subtitle = " Enter expand selected · Esc restore "
            target.update(
                Panel(
                    lines,
                    title=Text(title, style=f"bold {PALETTE.accent}"),
                    box=box.SQUARE,
                    border_style=BORDER,
                    height=height,
                )
            )
            return
        bars_height = max(5, height - summary_height)
        visible = self._overview_visible_rows()
        nodes = self._gpu_nodes()
        start = self.state.overview_scroll
        shown = nodes[start : start + visible]
        column_width = max(12, (width - 3) // 2)
        free_lines = Text()
        pressure_lines = Text()
        if shown:
            for index, node in enumerate(shown):
                free_lines.append_text(
                    self._gpu_bar(node, width=column_width - 4, pressure=False)
                )
                pressure_lines.append_text(
                    self._gpu_bar(node, width=column_width - 4, pressure=True)
                )
                if index != len(shown) - 1:
                    free_lines.append("\n")
                    pressure_lines.append("\n")
        else:
            message = "No GPU nodes match the active filters."
            free_lines.append(message, style=MUTED)
            pressure_lines.append(message, style=MUTED)
        bars = Table.grid(expand=True, padding=(0, 1))
        bars.add_column(ratio=1)
        bars.add_column(ratio=1)
        free_border = (
            CYAN
            if self.state.selected_panels.get("gpu-overview") == "free"
            else BORDER
        )
        pressure_border = (
            CYAN
            if self.state.selected_panels.get("gpu-overview") == "pressure"
            else BORDER
        )
        bars.add_row(
            Panel(
                free_lines,
                title=Text(" FREE GPUS PER NODE ", style=f"bold {PALETTE.accent}"),
                box=box.SQUARE,
                border_style=free_border,
                height=bars_height,
            ),
            Panel(
                pressure_lines,
                title=Text(" GPU REQUEST PRESSURE PER NODE ", style=f"bold {PALETTE.accent}"),
                box=box.SQUARE,
                border_style=pressure_border,
                height=bars_height,
            ),
        )
        content = Table.grid(expand=True)
        content.add_column()
        content.add_row(self._gpu_summary(width=width, height=summary_height))
        content.add_row(bars)
        end = min(len(nodes), start + visible)
        excluded = sum(not _eligible(node) for node in nodes)
        subtitle = ""
        if len(nodes) > visible:
            subtitle += f" nodes {start + 1}-{end}/{len(nodes)} "
        if excluded:
            subtitle += f" · {excluded} excluded from totals "
        target.border_subtitle = subtitle
        target.update(content)

    def _gpu_consumers(self) -> list[WorkloadConsumer]:
        consumers = [
            consumer
            for node in self._gpu_nodes()
            if _eligible(node)
            for consumer in node.consumers
            if consumer.requested.gpu_count > 0 and not is_system_consumer(consumer)
        ]
        return sorted(
            consumers,
            key=lambda consumer: (
                -consumer.requested.gpu_count,
                consumer.namespace.casefold(),
                natural_name_key(consumer.node_name),
                consumer.pod_name.casefold(),
            ),
        )

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
            return "No running GPU Pods"
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

    def _gpu_pod_table(self) -> Table:
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
        table.add_column("POD", ratio=2, no_wrap=True, overflow="ellipsis")
        table.add_column(
            "GPU/#" if narrow else "GPU MODEL / COUNT",
            width=7 if narrow else 17,
            justify="right",
            no_wrap=True,
        )
        consumers = self._gpu_consumers()
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
                Text(consumer.pod_name, style=WHITE),
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
            table.add_row(Text("No active GPU-requesting Pods", style=MUTED))
        return table

    @staticmethod
    def _allocation_legend_width(width: int) -> int:
        """Reserve one readable shared legend column for both charts."""

        return min(40, max(20, round(max(1, width) * 0.28)))

    @staticmethod
    def _allocation_colors(categories: Sequence[tuple[str, float]]) -> dict[str, str]:
        return allocation_colors(name for name, _ in categories)

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
            width=legend_width,
            height=max(1, height - 2),
            unit="G" if basis == "vram" else "",
            colors=colors,
        )
        legend_panel = Panel(
            legend,
            title=Text(" NAMESPACE LEGEND ", style=f"bold {PALETTE.accent}"),
            subtitle=Text(
                " VRAM % " if basis == "vram" else " GPU COUNT % ",
                style=GRAY,
            ),
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
        legend_width = self._allocation_legend_width(width)
        chart_width = max(18, width - legend_width - 2)
        if expanded:
            height = max(8, target.content_size.height)
            target.border_subtitle = " Enter expand selected · Esc restore "
            if expanded == "history":
                history_key = (
                    "expanded",
                    tuple(self.history),
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
                            " VRAM percentages · since launch "
                            if basis == "vram"
                            else " GPU count percentages · since launch ",
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
                        self._gpu_pod_table(),
                        title=Text(
                            " GPU-REQUESTING PODS ",
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
                self._pie_cache,
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
                    subtitle=Text(
                        " VRAM percentages " if basis == "vram" else " GPU count percentages ",
                        style=GRAY,
                    ),
                    box=box.SQUARE,
                    border_style=BORDER,
                    height=height,
                )
            )
            return
        history_height, bottom_height = self._allocation_layout()
        # The shared namespace legend sits above the pie in the left column;
        # Allocation History and GPU-requesting Pods occupy the right column.
        namespace_width = max(
            22,
            round(width * (0.42 if width >= 120 else 0.38)),
        )
        chart_width = max(18, width - namespace_width - 2)
        history_key = (
            tuple(self.history),
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
                " VRAM percentages · since launch (up to 24h) "
                if basis == "vram"
                else " GPU count percentages · since launch (up to 24h) ",
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
        )
        legend_panel = Panel(
            legend,
            title=Text(" NAMESPACE LEGEND ", style=f"bold {PALETTE.accent}"),
            subtitle=Text(
                " VRAM % " if basis == "vram" else " GPU COUNT % ",
                style=GRAY,
            ),
            box=box.SQUARE,
            border_style=BORDER,
            height=history_height,
        )
        pie_panel = Panel(
            self._pie_cache,
            title=Text(
                " ALLOCATION BY NAMESPACE ",
                style=f"bold {PALETTE.accent}",
            ),
            subtitle=Text(
                (
                    (
                        " VRAM percentages · "
                        if basis == "vram"
                        else " GPU count percentages · "
                    )
                    + (
                        f"{self.gpu_telemetry.target_pods} Pods accounted "
                        if self.gpu_telemetry.target_pods
                        else "No GPU allocations "
                    )
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
        consumers = self._gpu_consumers()
        visible_requests = sum(
            consumer.requested.gpu_count for consumer in consumers
        )
        total_requested = _gpu_totals(self._gpu_nodes())[1]
        hidden = max(0, total_requested - visible_requests)
        pods = Panel(
            self._gpu_pod_table(),
            title=Text(" GPU-REQUESTING PODS ", style=f"bold {PALETTE.accent}"),
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
            f" Pods {start + 1}-{end}/{len(consumers)} "
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
        table = Table(
            box=box.SIMPLE_HEAD,
            expand=True,
            padding=(0, 1),
            collapse_padding=True,
            header_style=f"bold {CYAN_2}",
        )
        table.add_column(
            "NODE",
            width=15 if width < 100 else 18,
            no_wrap=True,
            overflow="ellipsis",
        )
        table.add_column("SCHED" if width < 100 else "SCHEDULABLE", width=9 if width < 100 else 11)
        table.add_column("", ratio=1)
        table.add_column("GPU", width=12 if width < 100 else 13, justify="right", no_wrap=True)
        table.add_column("VRAM", width=5 if width < 100 else 7, justify="right", no_wrap=True)
        table.add_column("CPU", width=9 if width < 100 else 11, justify="right", no_wrap=True)
        table.add_column("MEM", width=10 if width < 100 else 11, justify="right", no_wrap=True)
        table.add_column("PODS", width=4 if width < 100 else 5, justify="right")
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
                Text(sched, style=sched_color),
                Text(""),
            ]
            cells.extend(
                [
                    Text(
                        (
                            f"{node.gpu_model} "
                            f"{node.gpu_free}/{node.allocatable.gpu_count}"
                            if node.allocatable.gpu_count
                            else "-"
                        ),
                        style=_gpu_headroom_color(
                            node.gpu_free,
                            node.allocatable.gpu_count,
                        ),
                    ),
                    Text(
                        (
                            _short_memory(node.gpu_memory_bytes_per_device)
                            if node.gpu_memory_bytes_per_device is not None
                            else "-"
                        ),
                        style=WHITE,
                    ),
                    Text(
                        f"{_short_cpu(node.request_headroom.cpu_cores)}/"
                        f"{_short_cpu(node.allocatable.cpu_cores)}",
                        style=_resource_headroom_color(
                            node.request_headroom.cpu_cores,
                            node.allocatable.cpu_cores,
                        ),
                    ),
                    Text(
                        f"{_short_memory(node.request_headroom.memory_bytes)}/"
                        f"{_short_memory(node.allocatable.memory_bytes)}",
                        style=_resource_headroom_color(
                            node.request_headroom.memory_bytes,
                            node.allocatable.memory_bytes,
                        ),
                    ),
                ]
            )
            cells.append(Text(str(node.workload_count), style=WHITE))
            table.add_row(*cells)
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
        width = self.size.width
        table.add_column("NAMESPACE / WORKLOAD", ratio=3, no_wrap=True, overflow="ellipsis")
        if width >= 100 or expanded:
            table.add_column("POD", ratio=2, no_wrap=True, overflow="ellipsis")
        table.add_column("STATUS", width=9)
        table.add_column("GPU", width=8, justify="right")
        table.add_column("CPU", width=8, justify="right")
        if width >= 115:
            table.add_column("RAM", width=9, justify="right")
        visible = self._visible_consumers()
        start = self.state.consumer_scroll
        visible_consumers = node.visible_consumers
        consumers = visible_consumers[start : start + visible]
        for absolute, consumer in enumerate(consumers, start=start):
            selected = expanded and absolute == self.state.selected_consumer
            identity = f"{consumer.namespace}/{consumer.display_workload}"
            cells = [
                Text(
                    f"{'>' if selected else ' '} {identity}",
                    style=f"bold {CYAN}" if selected else WHITE,
                    no_wrap=True,
                    overflow="ellipsis",
                )
            ]
            if width >= 100 or expanded:
                cells.append(
                    Text(
                        consumer.pod_name,
                        style=GRAY,
                        no_wrap=True,
                        overflow="ellipsis",
                    )
                )
            color = GREEN if consumer.status == "Running" else YELLOW
            cells.append(Text(consumer.status, style=color))
            vector = consumer.requested
            cells.extend(
                [
                    Text(
                        (
                            f"{vector.gpu_model or ''}x{vector.gpu_count}"
                            if vector.gpu_count
                            else "-"
                        ),
                        style=WHITE,
                    ),
                    Text(_short_cpu(vector.cpu_cores), style=WHITE),
                ]
            )
            if width >= 115:
                cells.append(Text(_short_memory(vector.memory_bytes), style=WHITE))
            table.add_row(*cells)
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
            line.append(
                f"   GPU {node.gpu_model or '-'} ",
                style=WHITE,
            )
            line.append(
                f"{node.gpu_free}/{node.allocatable.gpu_count}",
                style=_gpu_headroom_color(
                    node.gpu_free,
                    node.allocatable.gpu_count,
                ),
            )
            if node.gpu_memory_bytes_per_device is not None:
                line.append(
                    f"   VRAM {_short_memory(node.gpu_memory_bytes_per_device)}"
                    + ("/GPU" if self.size.width >= 100 else ""),
                    style=GRAY,
                )
            line.append("   CPU ", style=GRAY)
            line.append(
                f"{_short_cpu(node.request_headroom.cpu_cores)}/"
                f"{_short_cpu(node.allocatable.cpu_cores)}",
                style=_resource_headroom_color(
                    node.request_headroom.cpu_cores,
                    node.allocatable.cpu_cores,
                ),
            )
            line.append("   MEM ", style=GRAY)
            line.append(
                f"{_short_memory(node.request_headroom.memory_bytes)}/"
                f"{_short_memory(node.allocatable.memory_bytes)}",
                style=_resource_headroom_color(
                    node.request_headroom.memory_bytes,
                    node.allocatable.memory_bytes,
                ),
            )
            content = Table.grid(expand=True)
            content.add_column()
            content.add_row(line)
            content.add_row(self._consumer_table(node, expanded=False))
            if node.visible_consumers:
                visible = self._visible_consumers()
                start = self.state.consumer_scroll + 1
                end = min(len(node.visible_consumers), start + visible - 1)
                target.border_subtitle = (
                    f" consumers {start}-{end}/{len(node.visible_consumers)} · "
                    "Enter inspect "
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
            facts.add_row("Node", node_value)
            gpu_value = Text(
                f"{node.gpu_model or '-'}  "
                f"{node.gpu_free}/{node.allocatable.gpu_count}",
                style=WHITE,
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
                "GPU",
                gpu_value,
            )
            facts.add_row(
                "CPU",
                Text(
                    f"{_short_cpu(node.request_headroom.cpu_cores)} / "
                    f"{_short_cpu(node.allocatable.cpu_cores)}",
                    style=_resource_headroom_color(
                        node.request_headroom.cpu_cores,
                        node.allocatable.cpu_cores,
                    ),
                ),
            )
            facts.add_row(
                "MEM",
                Text(
                    f"{_short_memory(node.request_headroom.memory_bytes)} / "
                    f"{_short_memory(node.allocatable.memory_bytes)}",
                    style=_resource_headroom_color(
                        node.request_headroom.memory_bytes,
                        node.allocatable.memory_bytes,
                    ),
                ),
            )
            facts.add_row("Taints", _truncate(taints, 48))
            facts.add_row("Labels", _truncate(labels, 48))
        else:
            facts.add_column(style=GRAY, width=18)
            facts.add_column(style=WHITE, ratio=1)
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
                f"{node.gpu_free}/{node.allocatable.gpu_count}",
            )
            facts.add_row(
                "CPU capacity",
                _short_cpu(node.capacity.cpu_cores),
                "CPU free / alloc",
                Text(
                    f"{_short_cpu(node.request_headroom.cpu_cores)} / "
                    f"{_short_cpu(node.allocatable.cpu_cores)}",
                    style=_resource_headroom_color(
                        node.request_headroom.cpu_cores,
                        node.allocatable.cpu_cores,
                    ),
                ),
            )
            facts.add_row(
                "RAM capacity",
                _short_memory(node.capacity.memory_bytes),
                "RAM free / alloc",
                Text(
                    f"{_short_memory(node.request_headroom.memory_bytes)} / "
                    f"{_short_memory(node.allocatable.memory_bytes)}",
                    style=_resource_headroom_color(
                        node.request_headroom.memory_bytes,
                        node.allocatable.memory_bytes,
                    ),
                ),
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
        target.border_subtitle = (
            f" consumer {self.state.selected_consumer + 1}/"
            f"{len(node.visible_consumers)} "
            if node.visible_consumers
            else ""
        )
        target.update(content)

    def _render_footer(self) -> None:
        if self.size.width < MINIMUM_WIDTH or self.size.height < MINIMUM_HEIGHT:
            value = "q Quit   r Retry"
        elif self.state.view == "gpu-overview" and self.state.expanded_panels["gpu-overview"]:
            value = "Enter expand selected   Esc restore   ←/→ Views   ↑/↓ Scroll   r Refresh   q Quit"
        elif self.state.view == "gpu-overview":
            value = "←/→ Views   ↑/↓ Scroll nodes   Enter Expand selected   r Refresh   q Quit"
        elif self.state.view == "gpu-allocations":
            expanded_panel = self.state.expanded_panels["gpu-allocations"]
            value = (
                "Esc restore panels   " if expanded_panel else ""
            ) + "←/→ Views   ↑/↓ Scroll Pods   Enter Expand selected   v GPU/VRAM allocation pie   r Refresh   q Quit"
        elif self.state.expanded:
            value = "←/→ Views   ↑/↓ Consumers   PgUp/PgDn Page   Esc Nodes   r Refresh   q Quit"
        elif self.size.width < 100:
            value = "←/→ Views   ↑/↓ Nodes   Enter Inspect   r Refresh   q Quit"
        else:
            value = "←/→ Views   ↑/↓ Navigate nodes   Enter Inspect consumers   r Refresh   q Quit"
        self.query_one("#resources-footer", Static).update(Text(value, style=GRAY))

    def _render_all(self) -> None:
        if not self.is_mounted:
            return
        self._set_titles()
        self._render_header()
        self._render_views()
        if self.size.width >= MINIMUM_WIDTH and self.size.height >= MINIMUM_HEIGHT:
            if self.state.view == "nodes":
                self._render_overview()
                self._render_controls()
                self._render_nodes()
                self._render_node()
            elif self.state.view == "gpu-overview":
                self._render_gpu_overview()
            else:
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

    def _move(self, amount: int) -> None:
        if self.state.view == "gpu-overview":
            maximum = max(0, len(self._gpu_nodes()) - self._overview_visible_rows())
            self.state.overview_scroll = min(
                maximum,
                max(0, self.state.overview_scroll + amount),
            )
            self._render_gpu_overview()
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
            if node and node.visible_consumers:
                self.state.selected_consumer = min(
                    len(node.visible_consumers) - 1,
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
        if node is None or not node.visible_consumers:
            return
        if self.state.expanded:
            self._move(amount)
            return
        visible = self._visible_consumers()
        maximum = max(0, len(node.visible_consumers) - visible)
        self.state.consumer_scroll = max(
            0,
            min(maximum, self.state.consumer_scroll + amount),
        )
        self._render_node()

    def action_page_up(self) -> None:
        if self.state.view == "gpu-overview":
            self._move(-self._overview_visible_rows())
            return
        if self.state.view == "gpu-allocations":
            self._move(-self._allocation_visible_rows())
            return
        self._move(
            -(self._visible_consumers() if self.state.expanded else self._visible_nodes())
        )

    def action_page_down(self) -> None:
        if self.state.view == "gpu-overview":
            self._move(self._overview_visible_rows())
            return
        if self.state.view == "gpu-allocations":
            self._move(self._allocation_visible_rows())
            return
        self._move(
            self._visible_consumers() if self.state.expanded else self._visible_nodes()
        )

    def action_home(self) -> None:
        if self.state.view == "gpu-overview":
            self.state.overview_scroll = 0
        elif self.state.view == "gpu-allocations":
            self.state.allocation_scroll = 0
        elif self.state.expanded:
            self.state.selected_consumer = 0
        elif self.nodes:
            self.state.selected_node = self.nodes[0].name
        self._ensure_visible()
        self._render_all()

    def action_end(self) -> None:
        if self.state.view == "gpu-overview":
            self.state.overview_scroll = max(
                0, len(self._gpu_nodes()) - self._overview_visible_rows()
            )
        elif self.state.view == "gpu-allocations":
            self.state.allocation_scroll = max(
                0, len(self._gpu_consumers()) - self._allocation_visible_rows()
            )
        elif self.state.expanded:
            node = self._selected()
            self.state.selected_consumer = (
                max(0, len(node.visible_consumers) - 1) if node else 0
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

    def action_toggle_namespace_basis(self) -> None:
        if self.state.view != "gpu-allocations":
            return
        self.state.namespace_basis = (
            "vram" if self.state.namespace_basis == "gpu" else "gpu"
        )
        self._render_gpu_allocations()


ResourcesDashboard = FalconResourcesApp
