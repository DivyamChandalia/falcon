"""Dependency-free Rich renderers used by the Resources GPU pages."""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from rich.text import Text

from .cluster import natural_name_key
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

    @classmethod
    def from_mapping(
        cls,
        timestamp: float,
        values: Mapping[str, float],
    ) -> "GPUHistoryPoint":
        return cls(
            float(timestamp),
            tuple(
                sorted(
                    (
                        (str(name), max(0.0, float(value)))
                        for name, value in values.items()
                    ),
                    key=lambda item: natural_name_key(item[0]),
                )
            ),
        )

    @property
    def values(self) -> dict[str, float]:
        return dict(self.usage_by_namespace)

    @property
    def total(self) -> float:
        return sum(value for _, value in self.usage_by_namespace)

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


def _number(value: float, *, unit: str = "") -> str:
    if math.isclose(value, round(value), abs_tol=0.05):
        label = str(int(round(value)))
    elif value < 10:
        label = f"{value:.1f}"
    else:
        label = f"{value:.0f}"
    return f"{label}{unit}"


def render_gpu_history(
    points: Sequence[GPUHistoryPoint],
    *,
    width: int,
    height: int,
) -> Text:
    """Render a width-aware namespace allocation step chart."""

    width = max(1, int(width))
    height = max(1, int(height))
    if len(points) < 2 or width < 18 or height < 5:
        count = len(points)
        message = (
            "Collecting history"
            if not count
            else f"Collecting history · {count} sample since launch"
        )
        return Text(message[:width], style=MUTED, justify="center")

    names = sorted(
        {name for point in points for name, _ in point.usage_by_namespace},
        key=natural_name_key,
    )
    colors = _series_colors(names)
    label_width = min(
        5,
        max(2, len(_number(max(point.total for point in points)))),
    )
    chart_width = max(2, width - label_width - 1)
    sampled = downsample_history(points, chart_width)
    chart_height = max(2, height - 2)
    maximum = max(
        1,
        *(point.total for point in sampled),
        *(value for point in sampled for _, value in point.usage_by_namespace),
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

    point_values = [point.values for point in sampled]
    series: list[tuple[str, list[float], str, int]] = [
        (name, [values.get(name, 0) for values in point_values], colors[name], 1)
        for name in names
    ]
    series.append(
        ("Total", [point.total for point in sampled], TOTAL_COLOR, 2)
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

    output = Text()
    for row, line in enumerate(cells):
        axis_value = maximum if row == 0 else (0 if row == chart_height - 1 else None)
        axis = (
            " " * label_width
            if axis_value is None
            else _number(float(axis_value)).rjust(label_width)
        )
        output.append(axis, style=GRAY)
        output.append("┤" if row in {0, chart_height - 1} else "│", style=GRAY)
        for cell in line:
            if cell is None:
                output.append(" ")
            else:
                glyph, color, priority = cell
                output.append(glyph, style=("bold " if priority == 2 else "") + color)
        output.append("\n")

    first = datetime.fromtimestamp(sampled[0].timestamp).strftime("%H:%M")
    last = datetime.fromtimestamp(sampled[-1].timestamp).strftime("%H:%M")
    timestamp_gap = max(1, chart_width - len(first) - len(last))
    output.append(" " * (label_width + 1), style=GRAY)
    output.append(first, style=GRAY)
    output.append(" " * timestamp_gap)
    output.append(last, style=GRAY)

    legend = Text()
    legend.append("Total", style=f"bold {TOTAL_COLOR}")
    for name in names:
        item = f"  {name}"
        if len(legend.plain) + len(item) > width:
            legend.append("  …", style=MUTED)
            break
        legend.append(item, style=colors[name])
    return Text.assemble(legend, "\n", output)


def render_namespace_pie(
    categories: Sequence[tuple[str, float]],
    *,
    width: int,
    height: int,
    unit: str = "",
    empty_label: str = "No GPU allocation",
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

    pie_width = min(max(9, width // 2), max(9, 2 * height - 1), 25)
    legend_width = max(1, width - pie_width - 2)
    center_x = (pie_width - 1) / 2
    center_y = (height - 1) / 2
    radius = max(2.0, min(pie_width / 2 - 0.5, height - 1.0))
    cumulative: list[tuple[float, str]] = []
    progress = 0.0
    colors: dict[str, str] = {}
    visible_color_index = 0
    for name, value in values:
        progress += value / total
        cumulative.append((progress, name))
        if name == "System/hidden":
            colors[name] = MUTED
        else:
            # The Resources view renders at most eight visible categories.
            # Assign sequentially so two slices cannot collide in one pie.
            colors[name] = CHART_COLORS[visible_color_index % len(CHART_COLORS)]
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
            canvas[y][x] = ("█", colors[name])

    legend_lines: list[Text] = []
    for name, value in values[:height]:
        percent = value / total * 100
        count = f"{_number(value, unit=unit)} {percent:.0f}%"
        available = max(1, legend_width - len(count) - 2)
        label = name if len(name) <= available else name[: max(1, available - 1)] + "…"
        line = Text()
        line.append("█ ", style=colors[name])
        line.append(label.ljust(available), style=WHITE)
        line.append(count, style=GRAY)
        legend_lines.append(line)

    output = Text()
    for y, row in enumerate(canvas):
        for glyph, color in row:
            output.append(glyph, style=f"bold {color}" if glyph != " " else WHITE)
        output.append("  ")
        if y < len(legend_lines):
            output.append_text(legend_lines[y])
        if y != height - 1:
            output.append("\n")
    return output
