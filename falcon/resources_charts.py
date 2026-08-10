"""Dependency-free Rich renderers used by the Resources GPU pages."""

from __future__ import annotations

import math
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from rich.text import Text

from .cluster import is_system_namespace, natural_name_key
from .theme import CYAN, CYAN_2, GRAY, GREEN, MUTED, RED, WHITE, YELLOW

HISTORY_SECONDS = 24 * 60 * 60
HISTORY_LIMIT = 20_000
TOTAL_COLOR = "#ff8700"
CHART_COLORS = (
    CYAN_2,
    GREEN,
    YELLOW,
    "#af87ff",
    "#5f87ff",
    "#ff5f87",
    "#5faf5f",
    "#d7af00",
    "#87d7ff",
    "#ff875f",
    "#87af5f",
    "#d787ff",
    "#5fd7d7",
    "#ffaf5f",
    "#8787ff",
    "#d7d75f",
    "#5fd787",
    "#ff87d7",
    "#87afff",
    "#d75f5f",
    "#afff87",
    "#af87d7",
    "#5fafd7",
    "#ffafaf",
)


@dataclass(frozen=True)
class GPUHistoryPoint:
    """One namespace GPU-allocation observation."""

    timestamp: float
    usage_by_namespace: tuple[tuple[str, float], ...]
    vram_by_namespace: tuple[tuple[str, float], ...] = ()

    @classmethod
    def from_mapping(
        cls,
        timestamp: float,
        values: Mapping[str, float],
        vram_values: Mapping[str, float] | None = None,
    ) -> "GPUHistoryPoint":
        def ordered(source: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
            return tuple(
                sorted(
                    (
                        (str(name), max(0.0, float(value)))
                        for name, value in source.items()
                    ),
                    key=lambda item: natural_name_key(item[0]),
                )
            )

        return cls(
            float(timestamp),
            ordered(values),
            ordered(vram_values or {}),
        )

    @property
    def values(self) -> dict[str, float]:
        return dict(self.usage_by_namespace)

    @property
    def total(self) -> float:
        return sum(value for _, value in self.usage_by_namespace)

    @property
    def vram_values(self) -> dict[str, float]:
        return dict(self.vram_by_namespace)

    @property
    def vram_total(self) -> float:
        return sum(value for _, value in self.vram_by_namespace)

    def values_for(self, basis: str) -> dict[str, float]:
        return self.vram_values if basis == "vram" else self.values

    def total_for(self, basis: str) -> float:
        return self.vram_total if basis == "vram" else self.total

    @property
    def usage_by_node(self) -> tuple[tuple[str, float], ...]:
        """Compatibility alias for callers using the original node chart."""

        return self.usage_by_namespace


def downsample_history(
    points: Sequence[GPUHistoryPoint], columns: int
) -> list[GPUHistoryPoint]:
    """Keep step endpoints while bounding chart work to the visible width."""

    if not points:
        return []
    if columns <= 1:
        return [points[-1]]
    if len(points) <= columns:
        return list(points)
    if columns == 2:
        return [points[0], points[-1]]
    result = [points[0]]
    interior = points[1:-1]
    buckets = columns - 2
    for bucket in range(buckets):
        start = math.floor(bucket * len(interior) / buckets)
        end = math.floor((bucket + 1) * len(interior) / buckets)
        if end > start:
            result.append(interior[end - 1])
    if points[-1] is not result[-1]:
        result.append(points[-1])
    return result


def _series_colors(names: Iterable[str]) -> dict[str, str]:
    ordered = sorted(set(names), key=natural_name_key)
    colors: dict[str, str] = {}
    used: set[int] = set()
    for name in ordered:
        start = zlib.crc32(name.encode("utf-8")) % len(CHART_COLORS)
        index = start
        if len(used) < len(CHART_COLORS):
            while index in used:
                index = (index + 1) % len(CHART_COLORS)
            used.add(index)
        colors[name] = CHART_COLORS[index]
    return colors


def allocation_colors(names: Iterable[str]) -> dict[str, str]:
    """Return stable, collision-free colors for shared allocation legends."""

    return _series_colors(names)


def _number(value: float, *, unit: str = "") -> str:
    if math.isclose(value, round(value), abs_tol=0.05):
        label = str(int(round(value)))
    elif value < 10:
        label = f"{value:.1f}"
    else:
        label = f"{value:.0f}"
    return f"{label}{unit}"


def _legend_lines(
    values: Sequence[tuple[str, float]],
    *,
    total: float,
    width: int,
    colors: Mapping[str, str],
    unit: str = "",
    max_rows: int | None = None,
    include_total: bool = True,
) -> list[Text]:
    """Build one compact percentage legend shared by both allocation charts."""

    if total <= 0:
        return []
    entries = [
        *(([("Total", total, TOTAL_COLOR)] if include_total else [])),
        *[(name, value, colors[name]) for name, value in values],
    ]
    lines: list[Text] = []
    for name, value, color in entries:
        percent = value / total * 100
        suffix = f" {_number(value, unit=unit)} {percent:.0f}%"
        label_width = max(1, width - len(suffix) - 2)
        label = name if len(name) <= label_width else name[: max(1, label_width - 1)] + "…"
        line = Text()
        line.append("█ ", style=f"bold {color}")
        line.append(label.ljust(label_width), style=WHITE)
        line.append(suffix, style=GRAY)
        lines.append(line)
        if max_rows is not None and len(lines) >= max_rows:
            break
    return lines


def _pie_dimensions(
    values: Sequence[tuple[str, float]],
    *,
    width: int,
    height: int,
    unit: str = "",
) -> tuple[int, int]:
    """Return the left legend width and right pie width for a pane."""

    longest_label = max(
        len(name) + len(_number(value, unit=unit)) + 7
        for name, value in values
    )
    legend_width = min(max(14, longest_label), max(14, width // 2))
    pie_width = max(9, width - legend_width - 2)
    # A terminal cell is roughly twice as tall as it is wide. Limiting the
    # horizontal diameter by the available height keeps the pie circular,
    # while removing the old fixed 25-column cap lets it grow with the pane.
    pie_width = min(pie_width, max(9, 2 * height - 1))
    legend_width = min(legend_width, max(1, width - pie_width - 2))
    return legend_width, pie_width


def render_allocation_legend(
    categories: Sequence[tuple[str, float]],
    *,
    width: int,
    height: int,
    unit: str = "",
    colors: Mapping[str, str] | None = None,
) -> Text:
    """Render the one namespace percentage legend shared by both charts."""

    width = max(1, int(width))
    height = max(1, int(height))
    values = [
        (str(name), max(0.0, float(value)))
        for name, value in categories
        if value > 0
    ]
    total = sum(value for _, value in values)
    if not total:
        return Text("No GPU allocation"[:width], style=MUTED)
    palette = dict(colors or allocation_colors(name for name, _ in values))
    if "System/hidden" in palette:
        palette["System/hidden"] = MUTED
    lines = _legend_lines(
        values,
        total=total,
        width=width,
        colors=palette,
        unit=unit,
        max_rows=height,
        include_total=False,
    )
    output = Text()
    for index, line in enumerate(lines):
        output.append_text(line)
        if index != len(lines) - 1:
            output.append("\n")
    return output


def render_gpu_history(
    points: Sequence[GPUHistoryPoint],
    *,
    width: int,
    height: int,
    basis: str = "gpu",
    categories: Sequence[tuple[str, float]] | None = None,
    colors: Mapping[str, str] | None = None,
    show_legend: bool = True,
) -> Text:
    """Render a width-aware namespace allocation step chart.

    The legend is deliberately placed to the left of the graph. This keeps
    namespace names out of the chart's x-axis and gives the same percentage
    treatment as the namespace pie.
    """

    width = max(1, int(width))
    height = max(1, int(height))
    basis = "vram" if basis == "vram" else "gpu"
    selected = [point.values_for(basis) for point in points]
    category_names = {
        str(name) for name, value in (categories or ()) if value > 0
    }
    if categories is not None:
        def grouped(values: Mapping[str, float]) -> dict[str, float]:
            result: defaultdict[str, float] = defaultdict(float)
            for name, value in values.items():
                if name in category_names:
                    result[name] += value
                elif (
                    is_system_namespace(name)
                    and "System/hidden" in category_names
                ):
                    result["System/hidden"] += value
                elif "Other" in category_names:
                    result["Other"] += value
            return dict(result)

        selected = [grouped(values) for values in selected]
    if basis == "vram" and not any(selected):
        return Text("VRAM allocation history unavailable"[:width], style=MUTED, justify="center")
    if len(points) < 2 or width < 18 or height < 5:
        count = len(points)
        message = (
            "Collecting history"
            if not count
            else f"Collecting history · {count} sample since launch"
        )
        return Text(message[:width], style=MUTED, justify="center")

    if categories is None:
        names = sorted(
            {name for values in selected for name in values},
            key=natural_name_key,
        )
    else:
        names = [name for name, _ in categories if name in category_names]
    if not names:
        return Text("Collecting history"[:width], style=MUTED, justify="center")
    colors = dict(colors or allocation_colors(names))
    if "System/hidden" in colors:
        colors["System/hidden"] = MUTED
    latest_values = selected[-1]
    latest_total = sum(latest_values.values())
    unit = "G" if basis == "vram" else ""
    longest_label = max(
        len(name) + len(_number(latest_values.get(name, 0), unit=unit)) + 7
        for name in names
    )
    legend_width = (
        min(max(14, longest_label), max(14, width // 2))
        if show_legend
        else 0
    )
    graph_width = (
        max(8, width - legend_width - 2)
        if show_legend
        else max(8, width)
    )
    label_width = min(
        5,
        max(2, len(_number(max(point.total_for(basis) for point in points), unit=unit))),
    )
    chart_width = max(2, graph_width - label_width - 1)
    sampled = downsample_history(points, chart_width)
    chart_height = max(2, height - 2)
    maximum = max(
        1,
        *(point.total_for(basis) for point in sampled),
        *(value for point in sampled for value in point.values_for(basis).values()),
    )
    cells: list[list[tuple[str, str, int] | None]] = [
        [None for _ in range(chart_width)] for _ in range(chart_height)
    ]

    def row_for(value: float) -> int:
        return chart_height - 1 - round(max(0, value) / maximum * (chart_height - 1))

    def put(x: int, y: int, glyph: str, color: str, priority: int) -> None:
        if not (0 <= x < chart_width and 0 <= y < chart_height):
            return
        existing = cells[y][x]
        if existing is None or priority >= existing[2]:
            cells[y][x] = (glyph, color, priority)

    point_values = [point.values_for(basis) for point in sampled]
    series: list[tuple[str, list[float], str, int]] = [
        (name, [values.get(name, 0) for values in point_values], colors[name], 1)
        for name in names
    ]
    series.append(
        ("Total", [point.total_for(basis) for point in sampled], TOTAL_COLOR, 2)
    )
    denominator = max(1, len(sampled) - 1)
    xs = [round(index / denominator * (chart_width - 1)) for index in range(len(sampled))]
    for _, values, color, priority in series:
        previous_x = xs[0]
        previous_y = row_for(values[0])
        put(previous_x, previous_y, "━" if priority == 2 else "─", color, priority)
        for x, value in zip(xs[1:], values[1:]):
            y = row_for(value)
            horizontal = "━" if priority == 2 else "─"
            vertical = "┃" if priority == 2 else "│"
            for column in range(previous_x, x + 1):
                put(column, previous_y, horizontal, color, priority)
            if y != previous_y:
                for row in range(min(y, previous_y), max(y, previous_y) + 1):
                    put(x, row, vertical, color, priority)
            put(x, y, horizontal, color, priority)
            previous_x, previous_y = x, y

    graph = Text()
    for row, line in enumerate(cells):
        axis_value = maximum if row == 0 else (0 if row == chart_height - 1 else None)
        axis = (
            " " * label_width
            if axis_value is None
            else _number(float(axis_value), unit=unit).rjust(label_width)
        )
        graph.append(axis, style=GRAY)
        graph.append("┤" if row in {0, chart_height - 1} else "│", style=GRAY)
        for cell in line:
            if cell is None:
                graph.append(" ")
            else:
                glyph, color, priority = cell
                graph.append(glyph, style=("bold " if priority == 2 else "") + color)
        graph.append("\n")

    first = datetime.fromtimestamp(sampled[0].timestamp).strftime("%H:%M")
    last = datetime.fromtimestamp(sampled[-1].timestamp).strftime("%H:%M")
    timestamp_capacity = max(1, chart_width - label_width - 1)
    if len(first) + len(last) + 1 > timestamp_capacity:
        first = ""
        last = last[-timestamp_capacity:]
    timestamp_gap = max(0, timestamp_capacity - len(first) - len(last))
    graph.append(" " * (label_width + 1), style=GRAY)
    graph.append(first, style=GRAY)
    graph.append(" " * timestamp_gap)
    graph.append(last, style=GRAY)

    legend = (
        _legend_lines(
            tuple((name, latest_values.get(name, 0.0)) for name in names),
            total=latest_total,
            width=legend_width,
            colors={**colors, "Total": TOTAL_COLOR},
            unit=unit,
            max_rows=height,
        )
        if show_legend
        else []
    )
    # Retain the graph styling while placing each row beside the legend.
    styled_graph_lines: list[Text] = []
    for line in graph.split("\n"):
        styled_graph_lines.append(line)
    rows = max(len(styled_graph_lines), len(legend))
    output = Text()
    for index in range(rows):
        if show_legend and index < len(legend):
            output.append_text(legend[index])
            if len(legend[index].plain) < legend_width:
                output.append(" " * (legend_width - len(legend[index].plain)))
        elif show_legend:
            output.append(" " * legend_width)
        if show_legend:
            output.append("  ")
        if index < len(styled_graph_lines):
            output.append_text(styled_graph_lines[index])
        if index != rows - 1:
            output.append("\n")
    return output


def render_namespace_pie(
    categories: Sequence[tuple[str, float]],
    *,
    width: int,
    height: int,
    unit: str = "",
    empty_label: str = "No GPU allocation",
    colors: Mapping[str, str] | None = None,
    show_legend: bool = True,
) -> Text:
    """Render an aspect-correct, fully filled namespace allocation pie."""

    width = max(1, int(width))
    height = max(1, int(height))
    values = [
        (str(name), max(0.0, float(value)))
        for name, value in categories
        if value > 0
    ]
    total = sum(value for _, value in values)
    if not total:
        return Text(empty_label[:width], style=MUTED, justify="center")
    if width < 16 or height < 5:
        return Text(_number(total, unit=unit)[:width], style=CYAN, justify="center")

    if show_legend:
        legend_width, pie_width = _pie_dimensions(
            values,
            width=width,
            height=height,
            unit=unit,
        )
    else:
        legend_width = 0
        pie_width = min(max(9, width), max(9, 2 * height - 1))
    center_x = (pie_width - 1) / 2
    center_y = (height - 1) / 2
    radius = max(2.0, min(pie_width / 2 - 0.5, height - 1.0))
    cumulative: list[tuple[float, str]] = []
    progress = 0.0
    palette = dict(colors or allocation_colors(name for name, _ in values))
    if "System/hidden" in palette:
        palette["System/hidden"] = MUTED
    slice_colors: dict[str, str] = {}
    visible_color_index = 0
    for name, value in values:
        progress += value / total
        cumulative.append((progress, name))
        if name == "System/hidden":
            slice_colors[name] = MUTED
        else:
            slice_colors[name] = palette.get(
                name,
                CHART_COLORS[visible_color_index % len(CHART_COLORS)],
            )
            visible_color_index += 1

    canvas: list[list[tuple[str, str]]] = [
        [(" ", WHITE) for _ in range(pie_width)] for _ in range(height)
    ]
    for y in range(height):
        for x in range(pie_width):
            dx = x - center_x
            dy = (y - center_y) * 2.0
            distance = math.hypot(dx, dy)
            if distance > radius:
                continue
            angle = (math.atan2(dx, -dy) + math.tau) % math.tau
            fraction = angle / math.tau
            name = cumulative[-1][1]
            for boundary, candidate in cumulative:
                if fraction <= boundary:
                    name = candidate
                    break
            canvas[y][x] = ("█", slice_colors[name])

    legend_lines = (
        _legend_lines(
            values,
            total=total,
            width=legend_width,
            colors={**slice_colors, "Total": TOTAL_COLOR},
            unit=unit,
            max_rows=height,
            include_total=False,
        )
        if show_legend
        else []
    )

    output = Text()
    for y in range(height):
        if show_legend and y < len(legend_lines):
            output.append_text(legend_lines[y])
            if len(legend_lines[y].plain) < legend_width:
                output.append(" " * (legend_width - len(legend_lines[y].plain)))
        elif show_legend:
            output.append(" " * legend_width)
        if show_legend:
            output.append("  ")
        for glyph, color in canvas[y]:
            output.append(glyph, style=f"bold {color}" if glyph != " " else WHITE)
        if y != height - 1:
            output.append("\n")
    return output
