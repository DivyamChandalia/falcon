"""Shared terminal visual system and true-colour palette."""

from dataclasses import dataclass

from rich.color import ColorSystem
from rich.console import Console

# Keep semantic colours shared by Dashboard and Resources.  These muted
# values are intentionally terminal-true-colour rather than the old neon
# defaults, so the two screens have the same visual language.
CYAN = "#56B4E9"
CYAN_2 = CYAN
GREEN = "#009E73"
YELLOW = "#F0E442"
RED = "#D55E00"
WHITE = "#E6E6E6"
GRAY = "#AAAAAA"
MUTED = "#666666"
BORDER = "#555555"
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
    background: str = BACKGROUND
    series: tuple[str, ...] = SERIES_COLORS
    pie: tuple[str, ...] = PIE_COLORS
    total: str = CYAN


PALETTE = FalconPalette()


def force_truecolor(console: Console) -> None:
    """Pin Falcon's Rich/Textual output to direct 24-bit ANSI RGB.

    Textual's default ``auto`` mode follows ``TERM``.  tmux commonly exposes
    ``screen-256color`` even when the outer terminal (for example iTerm2)
    supports truecolor, which makes Rich downgrade explicit hex colours to
    the xterm-256 palette.  Falcon only uses explicit RGB colours, so make the
    final renderer use ``38;2``/``48;2`` sequences regardless of ``TERM``.
    """

    # Rich has no public setter for the output colour system.  This is the
    # Console instance Textual passes to every CompositorUpdate/Strip render;
    # setting it before the first frame prevents any lower-colour cache from
    # being populated.
    console._color_system = ColorSystem.TRUECOLOR


def metric_color(value: float | None) -> str:
    """Return the Dashboard's shared green/yellow/red pressure colour."""

    if value is None:
        return PALETTE.muted
    if value >= 80:
        return PALETTE.danger
    if value >= 30:
        return PALETTE.warning
    return PALETTE.success
