"""Shared terminal visual system and true-colour palette."""

from dataclasses import dataclass

CYAN = "#00FFFF"
CYAN_2 = "#4DDDDD"
GREEN = "#55FF55"
YELLOW = "#FFFF55"
RED = "#FF5555"
WHITE = "#F2F2F2"
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
