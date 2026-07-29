"""Realtime cluster and node resource dashboard."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Static

from .cluster import ClusterSnapshot, NodeSnapshot, natural_name_key
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
    RED,
    WHITE,
    YELLOW,
)


def _short_cpu(value: float) -> str:
    if value < 1:
        return f"{value * 1000:.0f}m"
    return f"{value:.1f}".rstrip("0").rstrip("-").rstrip(".")


def _short_memory(value: int) -> str:
    gib = value / (1024**3)
    return f"{gib:.1f}G" if gib < 10 else f"{gib:.0f}G"


def _truncate(value: object, width: int) -> str:
    text = "-" if value is None else str(value)
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def _schedulable(node: NodeSnapshot) -> tuple[str, str]:
    if node.ready is False:
        return "Not ready", RED
    if node.ready is None:
        return "Unknown", YELLOW
    if not node.schedulable:
        return "Cordoned", RED
    return "Yes", GREEN


def _gpu_headroom_color(headroom: int, allocatable: int) -> str:
    if allocatable <= 0:
        return MUTED
    remaining = max(0, headroom)
    if remaining == 0:
        return RED
    if remaining == 1:
        return YELLOW
    return GREEN


def _resource_headroom_color(free: float, allocatable: float) -> str:
    """Color remaining scheduler headroom using the CPU/memory pressure scale."""

    if allocatable <= 0:
        return MUTED
    percent_free = min(allocatable, max(0.0, free)) / allocatable * 100
    # These are the availability equivalents of the established request
    # pressure bands: >=80% requested is <=20% free, while <30% requested is
    # >70% free.
    if percent_free <= 20:
        return RED
    if percent_free <= 70:
        return YELLOW
    return GREEN


@dataclass
class ResourcesViewState:
    selected_node: str = ""
    node_scroll: int = 0
    expanded: bool = False
    selected_consumer: int = 0
    consumer_scroll: int = 0


class ResourcesPane(Static):
    can_focus = True

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.prevent_default()
        event.stop()
        self.app.action_down()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.prevent_default()
        event.stop()
        self.app.action_up()

    def on_click(self, event: events.Click) -> None:
        if self.id != "nodes-pane":
            return
        callback = getattr(self.app, "node_clicked", None)
        if callback:
            callback(event.get_content_offset(self))


CSS = f"""
Screen {{ background: {BACKGROUND}; color: {WHITE}; overflow: hidden; }}
Static {{ background: {BACKGROUND}; }}
#resources-header {{ height: 1; padding: 0 1; color: {WHITE}; }}
#cluster-overview {{ height: 3; border-bottom: solid {BORDER}; padding: 0 1; }}
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
#resize-message {{ display: none; height: 1fr; content-align: center middle; color: {YELLOW}; }}
#resources-footer {{ height: 1; padding: 0 1; color: {GRAY}; }}
"""


class FalconResourcesApp(App[None]):
    """Keyboard-first realtime view of schedulable request headroom."""

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
        Binding("enter", "expand", "Inspect"),
        Binding("escape", "collapse", "Back", show=False),
        Binding("r", "refresh_data", "Refresh"),
    ]

    def __init__(
        self,
        collector,
        *,
        refresh_seconds: float = 2.0,
        node_filter: Optional[str] = None,
        gpu_filter: Optional[str] = None,
        clock: Optional[Callable[[ClusterSnapshot], str]] = None,
    ) -> None:
        super().__init__()
        self.collector = collector
        self.refresh_seconds = refresh_seconds
        self.node_filter = (node_filter or "").lower()
        self.gpu_filter = (gpu_filter or "").lower()
        self.clock = clock or self._snapshot_clock
        self.state = ResourcesViewState()
        self.snapshot = ClusterSnapshot.empty()
        self.nodes: List[NodeSnapshot] = []
        self._refreshing = False
        self._results: "queue.Queue[ClusterSnapshot]" = queue.Queue(maxsize=1)
        self._last_terminal_size = (-1, -1)

    def compose(self) -> ComposeResult:
        yield Static(id="resources-header")
        yield Static(id="cluster-overview")
        yield Static(id="resource-controls")
        yield ResourcesPane(id="nodes-pane")
        yield ResourcesPane(id="node-pane")
        yield Static(id="resize-message")
        yield Static(id="resources-footer")

    def on_mount(self) -> None:
        self.query_one("#nodes-pane", ResourcesPane).border_title = " NODES "
        self.query_one("#node-pane", ResourcesPane).border_title = " SELECTED NODE "
        self.query_one("#nodes-pane", ResourcesPane).focus()
        self._apply_layout()
        self._request_update(force=True)
        self.set_interval(self.refresh_seconds, self._request_update)
        self.set_interval(0.1, self._drain_results)
        self.set_interval(0.1, self._check_terminal_size)
        self._render_all()

    def on_unmount(self) -> None:
        close = getattr(self.collector, "close", None)
        if callable(close):
            close()

    def on_resize(self, event: events.Resize) -> None:
        self._last_terminal_size = (self.size.width, self.size.height)
        self._apply_layout()
        self._ensure_visible()
        self._render_all()

    def _check_terminal_size(self) -> None:
        current = (self.size.width, self.size.height)
        if current == self._last_terminal_size:
            return
        self._last_terminal_size = current
        self._apply_layout()
        self._ensure_visible()
        self._render_all()

    @staticmethod
    def _snapshot_clock(snapshot: ClusterSnapshot) -> str:
        if not snapshot.collected_at:
            return "--:--:--"
        return datetime.fromtimestamp(snapshot.collected_at).strftime("%H:%M:%S")

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
        self.snapshot = snapshot
        nodes = sorted(snapshot.nodes, key=lambda node: natural_name_key(node.name))
        if self.node_filter:
            nodes = [node for node in nodes if self.node_filter in node.name.lower()]
        if self.gpu_filter:
            nodes = [
                node for node in nodes
                if self.gpu_filter in (node.gpu_model or "").lower()
            ]
        self.nodes = nodes
        names = {node.name for node in nodes}
        if self.state.selected_node not in names:
            self.state.selected_node = nodes[0].name if nodes else ""
            self.state.selected_consumer = 0
            self.state.consumer_scroll = 0
        self._ensure_visible()
        self._render_all()

    def _selected(self) -> Optional[NodeSnapshot]:
        return next(
            (node for node in self.nodes if node.name == self.state.selected_node),
            None,
        )

    def _selected_index(self) -> int:
        for index, node in enumerate(self.nodes):
            if node.name == self.state.selected_node:
                return index
        return 0

    def _visible_nodes(self) -> int:
        return max(1, self.query_one("#nodes-pane").content_size.height - 1)

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
        if self.state.selected_consumer < self.state.consumer_scroll:
            self.state.consumer_scroll = self.state.selected_consumer
        elif self.state.selected_consumer >= self.state.consumer_scroll + visible:
            self.state.consumer_scroll = self.state.selected_consumer - visible + 1
        self.state.consumer_scroll = min(
            max(0, len(consumers) - visible),
            max(0, self.state.consumer_scroll),
        )

    def _apply_layout(self) -> None:
        if not self.is_mounted:
            return
        small = self.size.width < MINIMUM_WIDTH or self.size.height < MINIMUM_HEIGHT
        resize = self.query_one("#resize-message")
        ids = (
            "cluster-overview", "resource-controls", "nodes-pane", "node-pane"
        )
        if small:
            for identifier in ids:
                self.query_one(f"#{identifier}").display = False
            resize.display = True
            resize.update(
                f"Falcon Resources requires at least "
                f"{MINIMUM_WIDTH}×{MINIMUM_HEIGHT}.\n"
                f"Current terminal: {self.size.width}×{self.size.height}.\n\n"
                "Resize to inspect nodes and resource consumers."
            )
            return
        resize.display = False
        if self.state.expanded:
            for identifier in ("cluster-overview", "resource-controls", "nodes-pane"):
                self.query_one(f"#{identifier}").display = False
            detail = self.query_one("#node-pane")
            detail.display = True
            detail.styles.height = "1fr"
            detail.border_title = " NODE INSPECTOR "
            detail.focus()
        else:
            for identifier in ids:
                self.query_one(f"#{identifier}").display = True
            self.query_one("#nodes-pane").styles.height = "1fr"
            self.query_one("#node-pane").styles.height = 7 if self.size.height < 28 else 9
            self.query_one("#node-pane").border_title = " SELECTED NODE "
            self.query_one("#nodes-pane").focus()

    def _render_header(self) -> None:
        width = max(30, self.size.width - 2)
        left = Text("Falcon Resources", style=f"bold {CYAN}")
        status = "STALE" if self.snapshot.stale else "LIVE"
        status_color = RED if self.snapshot.stale else GREEN
        right = Text(
            f"{self.clock(self.snapshot)}  {status}",
            style=f"bold {status_color}",
        )
        left.append(" " * max(1, width - len(left.plain) - len(right.plain)))
        left.append_text(right)
        self.query_one("#resources-header", Static).update(left)

    def _render_overview(self) -> None:
        snapshot = self.snapshot
        headroom = snapshot.request_headroom
        text = Text()
        if self.size.width < 100:
            text.append(
                f"{snapshot.schedulable_nodes}/{snapshot.total_nodes} NODES  ",
                style=f"bold {GREEN if snapshot.schedulable_nodes else YELLOW}",
            )
            text.append(
                f"{snapshot.running_jobs} RUN  {snapshot.pending_jobs} PEND  ",
                style=WHITE,
            )
            text.append(
                f"CPU {headroom.cpu_cores:.1f}/"
                f"{snapshot.allocatable.cpu_cores:.1f}\n",
                style=_resource_headroom_color(
                    headroom.cpu_cores,
                    snapshot.allocatable.cpu_cores,
                ),
            )
            text.append("GPU ", style=GRAY)
            if snapshot.gpu_availability:
                for index, availability in enumerate(
                    snapshot.gpu_availability.values()
                ):
                    if index:
                        text.append(" ")
                    text.append(
                        f"{availability.model} "
                        f"{availability.request_headroom}/"
                        f"{availability.allocatable}",
                        style="bold "
                        + _gpu_headroom_color(
                            availability.request_headroom,
                            availability.allocatable,
                        ),
                    )
            else:
                text.append("-", style=MUTED)
            text.append(
                f"  MEM {_short_memory(headroom.memory_bytes)}/"
                f"{_short_memory(snapshot.allocatable.memory_bytes)}",
                style=_resource_headroom_color(
                    headroom.memory_bytes,
                    snapshot.allocatable.memory_bytes,
                ),
            )
            self.query_one("#cluster-overview", Static).update(text)
            return
        text.append(
            f"{snapshot.schedulable_nodes}/{snapshot.total_nodes} NODES  ",
            style=f"bold {GREEN if snapshot.schedulable_nodes else YELLOW}",
        )
        text.append(
            f"{snapshot.running_jobs} RUNNING  {snapshot.pending_jobs} PENDING  ",
            style=WHITE,
        )
        text.append(
            f"CPU {headroom.cpu_cores:.1f}/"
            f"{snapshot.allocatable.cpu_cores:.1f}  ",
            style=_resource_headroom_color(
                headroom.cpu_cores,
                snapshot.allocatable.cpu_cores,
            ),
        )
        text.append(
            f"MEM {_short_memory(headroom.memory_bytes)}/"
            f"{_short_memory(snapshot.allocatable.memory_bytes)}",
            style=_resource_headroom_color(
                headroom.memory_bytes,
                snapshot.allocatable.memory_bytes,
            ),
        )
        text.append("\n")
        if snapshot.gpu_availability:
            text.append("GPU  ", style=GRAY)
            for index, availability in enumerate(snapshot.gpu_availability.values()):
                if index:
                    text.append("   ")
                text.append(
                    f"{availability.model} "
                    f"{availability.request_headroom}/{availability.allocatable}",
                    style="bold "
                    + _gpu_headroom_color(
                        availability.request_headroom,
                        availability.allocatable,
                    ),
                )
        else:
            text.append("No GPU allocatable resources reported", style=MUTED)
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
            box=None,
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
            sched, sched_color = _schedulable(node)
            cells = [
                Text(
                    f"{'>' if selected else ' '} {node.name}",
                    style=f"bold {CYAN}" if selected else WHITE,
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
            target.border_subtitle = (
                " Enter inspect " if node.visible_consumers else ""
            )
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
        elif self.state.expanded:
            value = "↑/↓ Consumers   PgUp/PgDn Page   Home/End   Esc Nodes   r Refresh   q Quit"
        elif self.size.width < 100:
            value = "↑/↓ Nodes   Enter Inspect   r Refresh   q Quit"
        else:
            value = "↑/↓ Navigate nodes   PgUp/PgDn Page   Enter Inspect consumers   r Refresh   q Quit"
        self.query_one("#resources-footer", Static).update(Text(value, style=GRAY))

    def _render_all(self) -> None:
        if not self.is_mounted:
            return
        self._render_header()
        self._render_overview()
        self._render_controls()
        self._render_nodes()
        self._render_node()
        self._render_footer()

    def node_clicked(self, offset) -> None:
        if self.state.expanded or not self.nodes or offset is None or offset.y <= 0:
            return
        index = self.state.node_scroll + offset.y - 1
        if index < len(self.nodes):
            self.state.selected_node = self.nodes[index].name
            self.state.selected_consumer = 0
            self.state.consumer_scroll = 0
            self._render_all()

    def _move(self, amount: int) -> None:
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

    def action_page_up(self) -> None:
        self._move(
            -(self._visible_consumers() if self.state.expanded else self._visible_nodes())
        )

    def action_page_down(self) -> None:
        self._move(
            self._visible_consumers() if self.state.expanded else self._visible_nodes()
        )

    def action_home(self) -> None:
        if self.state.expanded:
            self.state.selected_consumer = 0
        elif self.nodes:
            self.state.selected_node = self.nodes[0].name
        self._ensure_visible()
        self._render_all()

    def action_end(self) -> None:
        if self.state.expanded:
            node = self._selected()
            self.state.selected_consumer = (
                max(0, len(node.visible_consumers) - 1) if node else 0
            )
        elif self.nodes:
            self.state.selected_node = self.nodes[-1].name
        self._ensure_visible()
        self._render_all()

    def action_expand(self) -> None:
        if self._selected() is None:
            return
        self.state.expanded = True
        self._apply_layout()
        self.call_after_refresh(self._render_all)

    def action_collapse(self) -> None:
        if not self.state.expanded:
            return
        self.state.expanded = False
        self._apply_layout()
        self.call_after_refresh(self._render_all)

    def action_refresh_data(self) -> None:
        self._request_update(force=True)


ResourcesDashboard = FalconResourcesApp
