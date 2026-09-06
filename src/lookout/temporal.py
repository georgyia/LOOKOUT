"""Temporal normalization and fixation segmentation.

Raw per-viewer gaze points are noisy and unevenly sampled. This module smooths
them and groups them into fixations, so attribution acts on stable dwells rather
than per-frame jitter.

Fixations use the dispersion-threshold (I-DT) method: a run of points whose
spatial spread stays within a threshold for at least a minimum duration is one
fixation. A time gap larger than ``max_gap`` breaks a run.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .models import GazePoint

__all__ = ["Fixation", "median_smooth", "detect_fixations"]


@dataclass(frozen=True)
class Fixation:
    """A stable dwell: the centroid of a run of gaze points."""

    person_id: str
    x: float
    y: float
    start_time: float
    end_time: float
    confidence: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def median_smooth(points: list[GazePoint], window: int = 3) -> list[GazePoint]:
    """Rolling-median smooth of ``x`` and ``y`` over an odd window.

    Timestamps, ids, and confidences are preserved; only coordinates change.
    """

    if window <= 1:
        return list(points)
    if window % 2 == 0:
        raise ValueError("window must be odd")

    ordered = sorted(points, key=lambda p: p.timestamp)
    half = window // 2
    smoothed: list[GazePoint] = []
    for i, point in enumerate(ordered):
        lo = max(0, i - half)
        hi = min(len(ordered), i + half + 1)
        neighborhood = ordered[lo:hi]
        smoothed.append(
            GazePoint(
                timestamp=point.timestamp,
                person_id=point.person_id,
                x=statistics.median(p.x for p in neighborhood),
                y=statistics.median(p.y for p in neighborhood),
                confidence=point.confidence,
            )
        )
    return smoothed


def _dispersion(points: list[GazePoint]) -> float:
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def _make_fixation(points: list[GazePoint]) -> Fixation:
    return Fixation(
        person_id=points[0].person_id,
        x=statistics.fmean(p.x for p in points),
        y=statistics.fmean(p.y for p in points),
        start_time=points[0].timestamp,
        end_time=points[-1].timestamp,
        confidence=statistics.fmean(p.confidence for p in points),
    )


def detect_fixations(
    points: list[GazePoint],
    dispersion_threshold: float = 0.12,
    min_duration: float = 0.15,
    max_gap: float = 0.3,
) -> list[Fixation]:
    """Segment a single viewer's gaze points into fixations (I-DT)."""

    ordered = sorted(points, key=lambda p: p.timestamp)
    n = len(ordered)
    fixations: list[Fixation] = []
    i = 0
    while i < n:
        # Grow a window to at least min_duration, breaking on large time gaps.
        j = i
        while j + 1 < n and ordered[j].timestamp - ordered[i].timestamp < min_duration:
            if ordered[j + 1].timestamp - ordered[j].timestamp > max_gap:
                break
            j += 1
        if ordered[j].timestamp - ordered[i].timestamp < min_duration:
            i += 1
            continue
        if _dispersion(ordered[i : j + 1]) <= dispersion_threshold:
            while (
                j + 1 < n
                and ordered[j + 1].timestamp - ordered[j].timestamp <= max_gap
                and _dispersion(ordered[i : j + 2]) <= dispersion_threshold
            ):
                j += 1
            fixations.append(_make_fixation(ordered[i : j + 1]))
            i = j + 1
        else:
            i += 1
    return fixations
