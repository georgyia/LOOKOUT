"""Angle-to-screen mapping.

Turns a gaze angle into a point on a viewer's screen, or reports ``off_screen``.
The mapping is a pure, linear model parameterized by the angles that hit the
screen edges. v1 ships a documented prior for typical laptop geometry; the same
structure lets #18 fit per-viewer parameters later.

Convention (from ``models.py``): +yaw -> larger x, +pitch -> up (smaller y).
With the camera above the screen, looking straight at the camera (pitch 0) lands
near the top edge and the screen extends downward into negative pitch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import GazeDirection, GazePoint, GazeTarget

__all__ = ["ScreenMappingParams", "project", "map_direction"]


@dataclass(frozen=True)
class ScreenMappingParams:
    """Angles (radians) that map to the screen edges.

    Defaults encode the v1 prior: about +/-16 deg horizontally and 0 to -19 deg
    vertically for a camera mounted above the screen centre.
    """

    yaw_at_left: float = math.radians(-16.0)
    yaw_at_right: float = math.radians(16.0)
    pitch_at_top: float = math.radians(0.0)
    pitch_at_bottom: float = math.radians(-19.0)
    off_screen_margin: float = 0.05

    def __post_init__(self) -> None:
        if self.yaw_at_right <= self.yaw_at_left:
            raise ValueError("yaw_at_right must exceed yaw_at_left")
        if self.pitch_at_top <= self.pitch_at_bottom:
            raise ValueError("pitch_at_top must exceed pitch_at_bottom")


def project(yaw: float, pitch: float, params: ScreenMappingParams) -> tuple[float, float] | None:
    """Map an angle to normalized ``(x, y)``, or ``None`` if off screen.

    A point within ``off_screen_margin`` beyond an edge is clamped onto the
    screen; further out is off screen.
    """

    x = (yaw - params.yaw_at_left) / (params.yaw_at_right - params.yaw_at_left)
    y = (params.pitch_at_top - pitch) / (params.pitch_at_top - params.pitch_at_bottom)
    margin = params.off_screen_margin
    if not (-margin <= x <= 1.0 + margin and -margin <= y <= 1.0 + margin):
        return None
    return (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))


def map_direction(
    direction: GazeDirection,
    params: ScreenMappingParams | None = None,
) -> GazePoint | GazeTarget:
    """Map a :class:`GazeDirection` to a :class:`GazePoint` or ``OFF_SCREEN``."""

    params = params or ScreenMappingParams()
    projected = project(direction.yaw, direction.pitch, params)
    if projected is None:
        return GazeTarget.OFF_SCREEN
    x, y = projected
    return GazePoint(direction.timestamp, direction.person_id, x, y, direction.confidence)
