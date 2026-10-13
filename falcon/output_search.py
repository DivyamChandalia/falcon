"""Literal, case-insensitive search in bounded Dashboard output."""

import re

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.strip import Strip

MATCH_STYLE = "black on #e69f00"
ACTIVE_MATCH_STYLE = "black on #5de6e6"


def matches(lines, query):
    if not query:
        return []
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return [(row, match.start(), match.end())
            for row, line in enumerate(lines) for match in pattern.finditer(line)]


def highlight_text(content: Text, query: str, active=None) -> Text:
    if not query:
        return content
    result = content.copy()
    result.highlight_regex("(?i)" + re.escape(query), MATCH_STYLE)
    if active:
        row, start, end = active
        lines = result.plain.split("\n")
        if row < len(lines):
            offset = sum(len(line) + 1 for line in lines[:row])
            result.stylize(ACTIVE_MATCH_STYLE, offset + start, offset + end)
    return result


def highlight_strip(strip: Strip, original: str, query: str, row: int,
                    scroll_x: int, active=None) -> Strip:
    for _, start, end in matches([original], query):
        left = max(0, cell_len(original[:start]) - scroll_x)
        right = min(strip.cell_length, cell_len(original[:end]) - scroll_x)
        if right > left:
            style = ACTIVE_MATCH_STYLE if active == (row, start, end) else MATCH_STYLE
            selected = strip.crop(left, right)
            highlighted = Strip(Segment.apply_style(selected._segments,
                                                     post_style=Style.parse(style)),
                                selected.cell_length)
            strip = Strip.join((strip.crop(0, left), highlighted, strip.crop(right)))
    return strip
