"""Core data contracts for LOOKOUT.

The pipeline separates observation from interpretation:

- A *raw observation* is a :class:`GazeDirection`: an angle in the camera frame,
  independent of any screen or layout.
- An *interpretation* maps that angle to a :class:`GazePoint` on a viewer's
  screen and then to an :class:`Attribution` against that viewer's
  :class:`Layout`.

Angle sign conventions (radians), fixed here and relied on everywhere:

- ``yaw``:   positive means the gaze is directed toward the viewer's right,
             i.e. toward a larger normalized screen ``x``.
- ``pitch``: positive means the gaze is directed upward,
             i.e. toward a smaller normalized screen ``y`` (image coordinates
             increase downward).

All coordinates are normalized to ``[0, 1]``. All timestamps are seconds from
the start of the recording and are non-negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "GazeTarget",
    "RegionKind",
    "LayoutSource",
    "GazePoint",
    "HeadPose",
    "GazeDirection",
    "Region",
    "Layout",
    "Attribution",
    "GazeEvent",
]


class GazeTarget(StrEnum):
    """Explicit non-participant results. A target is never invented."""

    UNKNOWN = "unknown"
    OFF_SCREEN = "off_screen"
    NOT_VISIBLE = "not_visible"
    LOW_CONFIDENCE = "low_confidence"


class RegionKind(StrEnum):
    """What a screen region represents."""

    PARTICIPANT = "participant"
    SHARED_CONTENT = "shared_content"
    UI = "ui"
    UNKNOWN = "unknown"


class LayoutSource(StrEnum):
    """Where a viewer's layout came from, and how much to trust it.

    ``recording`` is the layout actually visible in the recorded video.
    ``assumed_shared`` applies that layout to every viewer (the v1 assumption:
    it is only correct for the participant whose screen was recorded).
    ``manifest`` is a user-supplied per-viewer layout; ``inferred`` is one
    reconstructed from gaze behavior.
    """

    RECORDING = "recording"
    ASSUMED_SHARED = "assumed_shared"
    MANIFEST = "manifest"
    INFERRED = "inferred"


def _check_unit(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _check_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class GazePoint:
    """A gaze estimate expressed as a point on a viewer's screen.

    Derived from a :class:`GazeDirection` by an explicit, uncertain mapping;
    it is not a raw observation.
    """

    timestamp: float
    person_id: str
    x: float
    y: float
    confidence: float

    def __post_init__(self) -> None:
        _check_unit(self.x, "x")
        _check_unit(self.y, "y")
        _check_unit(self.confidence, "confidence")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")


@dataclass(frozen=True)
class HeadPose:
    """Head orientation in the camera frame, in radians."""

    yaw: float
    pitch: float
    roll: float

    def __post_init__(self) -> None:
        _check_finite(self.yaw, "head yaw")
        _check_finite(self.pitch, "head pitch")
        _check_finite(self.roll, "head roll")


@dataclass(frozen=True)
class GazeDirection:
    """A raw gaze observation: an angle in the camera frame with confidence.

    This is the unit of the raw gaze stream. It never refers to a screen or a
    layout, so it can be stored once and reinterpreted whenever the mapping or
    the layout changes.
    """

    timestamp: float
    person_id: str
    yaw: float
    pitch: float
    head_pose: HeadPose
    confidence: float

    def __post_init__(self) -> None:
        _check_finite(self.yaw, "yaw")
        _check_finite(self.pitch, "pitch")
        _check_unit(self.confidence, "confidence")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")


@dataclass(frozen=True)
class Region:
    """A normalized rectangle on a viewer's screen and what it represents.

    A region carries no time information of its own; validity is expressed by
    the :class:`Layout` that owns it.
    """

    kind: RegionKind
    x: float
    y: float
    width: float
    height: float
    participant_id: str | None = None

    def __post_init__(self) -> None:
        _check_unit(self.x, "x")
        _check_unit(self.y, "y")
        if not 0.0 < self.width <= 1.0:
            raise ValueError("width must be in (0, 1]")
        if not 0.0 < self.height <= 1.0:
            raise ValueError("height must be in (0, 1]")
        if self.x + self.width > 1.0 + 1e-9:
            raise ValueError("region exceeds right edge")
        if self.y + self.height > 1.0 + 1e-9:
            raise ValueError("region exceeds bottom edge")
        if self.kind is RegionKind.PARTICIPANT and self.participant_id is None:
            raise ValueError("participant region requires a participant_id")
        if self.kind is not RegionKind.PARTICIPANT and self.participant_id is not None:
            raise ValueError("only participant regions may carry a participant_id")

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def edge_distance(self, x: float, y: float) -> float:
        """Distance from ``(x, y)`` to the nearest edge.

        Positive inside the region, zero on an edge, negative outside. Used by
        attribution to turn "how central is the hit" into a confidence.
        """

        return min(
            x - self.x,
            (self.x + self.width) - x,
            y - self.y,
            (self.y + self.height) - y,
        )


@dataclass(frozen=True)
class Layout:
    """A viewer's screen layout over a time interval.

    The layout is always viewer-relative: it describes what ``viewer_id`` saw,
    not what the camera recorded. ``source`` records how much to trust it.
    """

    viewer_id: str
    regions: tuple[Region, ...]
    start_time: float
    source: LayoutSource
    end_time: float | None = None

    def __post_init__(self) -> None:
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if self.end_time is not None and self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")

    def active_at(self, timestamp: float) -> bool:
        if timestamp < self.start_time:
            return False
        return self.end_time is None or timestamp < self.end_time

    def participant_regions(self) -> tuple[Region, ...]:
        return tuple(r for r in self.regions if r.kind is RegionKind.PARTICIPANT)

    def regions_containing(self, x: float, y: float) -> tuple[Region, ...]:
        return tuple(r for r in self.regions if r.contains(x, y))


@dataclass(frozen=True)
class Attribution:
    """The result of attributing one gaze point to a target.

    ``target`` is either a participant id or a :class:`GazeTarget` value.
    ``layout_source`` is propagated so consumers know how much the assumed
    layout influenced the result.
    """

    target: str
    confidence: float
    reason: str
    layout_source: LayoutSource

    def __post_init__(self) -> None:
        _check_unit(self.confidence, "confidence")
        if not self.target:
            raise ValueError("target must be non-empty")


@dataclass(frozen=True)
class GazeEvent:
    """A sustained attribution of a viewer's gaze to a single target."""

    viewer_id: str
    target: str
    start_time: float
    end_time: float
    confidence: float
    layout_source: LayoutSource

    def __post_init__(self) -> None:
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if self.end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")
        _check_unit(self.confidence, "confidence")

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
