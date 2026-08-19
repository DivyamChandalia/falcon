"""Shared terminal visual system and true-colour palette."""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Literal, cast

from rich.color import ColorSystem
from rich.console import Console

ColorMode = Literal["truecolor", "256", "16", "auto"]
COLOR_MODES = ("truecolor", "256", "16", "auto")
_COLOR_LOGGER = logging.getLogger("falcon.tui.color")

# Keep semantic colours shared by Dashboard and Resources.  These muted
# values are intentionally terminal-true-colour rather than the old neon
# defaults, so the two screens have the same visual language.
CYAN = "#8AFAFF"
CYAN_2 = CYAN
GREEN = "#4FD874"
YELLOW = "#FFF87A"
RED = "#E35C5C"
WHITE = "#FFFFFF"
GRAY = "#AAAAAA"
MUTED = "#666666"
BORDER = "#555555"
SELECTION = "#202020"
BACKGROUND = "#000000"
MINIMUM_WIDTH = 80
MINIMUM_HEIGHT = 22


# Series colours intentionally reuse the Dashboard's existing true-colour
# tokens. Resources must not introduce a second set of visually similar hues.
SERIES_COLORS = (
    CYAN,
    CYAN_2,
    GREEN,
    YELLOW,
    RED,
    WHITE,
    GRAY,
    MUTED,
)

# Color-blind-friendly true-colour palette for namespace allocation slices.
PIE_COLORS = (
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
)


@dataclass(frozen=True)
class FalconPalette:
    """Semantic and categorical colours shared by Dashboard and Resources."""

    accent: str = CYAN
    accent_soft: str = CYAN_2
    success: str = GREEN
    warning: str = YELLOW
    danger: str = RED
    text: str = WHITE
    secondary: str = GRAY
    muted: str = MUTED
    border: str = BORDER
    selection: str = SELECTION
    background: str = BACKGROUND
    series: tuple[str, ...] = SERIES_COLORS
    pie: tuple[str, ...] = PIE_COLORS
    total: str = CYAN


PALETTE = FalconPalette()


def _color_mode_from_system(system: ColorSystem | None) -> str:
    if system == ColorSystem.TRUECOLOR:
        return "truecolor"
    if system == ColorSystem.EIGHT_BIT:
        return "256"
    return "16"


def _normalize_color_mode(value: object) -> ColorMode:
    normalized = str(value or "").strip().lower()
    aliases = {
        "24bit": "truecolor",
        "24-bit": "truecolor",
        "true": "truecolor",
        "256color": "256",
        "8bit": "256",
        "8-bit": "256",
        "standard": "16",
        "ansi": "16",
    }
    normalized = aliases.get(normalized, normalized)
    return cast(ColorMode, normalized) if normalized in COLOR_MODES else "truecolor"


def _rgb_triplet(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.removeprefix("#")
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
    )


def _clear_rich_colour_caches() -> None:
    """Prevent Rich styles parsed in one mode from leaking into another."""

    # Rich caches parsed Style objects and stores the generated SGR string on
    # each object. If a process creates a 256-colour app and then a truecolour
    # app, reusing that object would otherwise keep the old ``38;5`` sequence.
    from rich.style import Style
    from textual.strip import Strip

    for name in ("parse", "normalize", "_add", "clear_meta_and_links"):
        cache_clear = getattr(getattr(Style, name, None), "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
    render_ansi = getattr(Strip, "render_ansi", None)
    cache_clear = getattr(render_ansi, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


def configure_color(console: Console, requested: object = None) -> str:
    """Select Falcon's terminal colour mode and configure Rich's renderer.

    Falcon defaults to truecolor because ``TERM=screen`` and an empty
    ``COLORTERM`` are not capability probes: tmux routinely reports both while
    forwarding 24-bit SGR sequences correctly. ``auto`` therefore remains
    truecolor for every non-dumb terminal. The lower-colour modes are explicit
    fallbacks for terminals known by the user to lack RGB support.
    """

    requested_value = requested
    if requested_value is None:
        requested_value = os.environ.get("FALCON_COLOR") or "truecolor"
    requested_mode = _normalize_color_mode(requested_value)
    framework_mode = _color_mode_from_system(console._color_system)
    if requested_mode == "auto":
        # Do not infer a downgrade from screen/screen-256color. A truly dumb
        # terminal is the only automatic fallback; users can explicitly pick
        # ``256`` or ``16`` for other unsupported terminals.
        term = os.environ.get("TERM", "").strip().lower()
        selected_mode = "16" if term == "dumb" else "truecolor"
    else:
        selected_mode = requested_mode
    systems = {
        "truecolor": ColorSystem.TRUECOLOR,
        "256": ColorSystem.EIGHT_BIT,
        "16": ColorSystem.STANDARD,
    }
    console._color_system = systems[selected_mode]
    _clear_rich_colour_caches()

    rgb = _rgb_triplet(PALETTE.accent)
    message = (
        f"Falcon TUI colour mode: {selected_mode} "
        f"(requested={requested_mode}, framework={framework_mode}); "
        f"{PALETTE.accent}=rgb{rgb}"
    )
    _COLOR_LOGGER.info(message)
    if os.environ.get("FALCON_COLOR_DEBUG", "").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        print(message, file=sys.stderr)
    return selected_mode


def force_truecolor(console: Console) -> None:
    """Backward-compatible shorthand for selecting direct 24-bit RGB.

    Textual's default ``auto`` mode follows ``TERM``.  tmux commonly exposes
    ``screen-256color`` even when the outer terminal (for example iTerm2)
    supports truecolor, which makes Rich downgrade explicit hex colours to
    the xterm-256 palette.  Falcon only uses explicit RGB colours, so make the
    final renderer use ``38;2``/``48;2`` sequences regardless of ``TERM``.
    """

    configure_color(console, "truecolor")


def metric_color(value: float | None) -> str:
    """Return the Dashboard's shared green/yellow/red pressure colour."""

    if value is None:
        return PALETTE.muted
    if value >= 80:
        return PALETTE.danger
    if value >= 30:
        return PALETTE.warning
    return PALETTE.success
