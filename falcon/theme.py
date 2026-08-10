"""Shared terminal visual system and true-colour palette."""

from dataclasses import dataclass

# Keep semantic colours shared by Dashboard and Resources.  These muted
# values are intentionally terminal-true-colour rather than the old neon
# defaults, so the two screens have the same visual language.
CYAN = "#3DAFC2"
CYAN_2 = CYAN
GREEN = "#4EBD62"
YELLOW = "#CCB83D"
RED = "#CC4F55"
WHITE = "#D0D0D0"
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


def metric_color(value: float | None) -> str:
    """Return the Dashboard's shared green/yellow/red pressure colour."""

    if value is None:
        return PALETTE.muted
    if value >= 80:
        return PALETTE.danger
    if value >= 30:
        return PALETTE.warning
    return PALETTE.success
