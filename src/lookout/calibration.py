"""Implicit per-viewer calibration via the conversation prior.

Listeners look at the current speaker far more often than at anyone else, so a
speaker segment is a weak label for where a listener's gaze should land: the
speaker's tile centre. Most such labels are correct, but many are not, so the
per-viewer mapping is fit robustly (RANSAC) and rejects the wrong ones.

The mapping is linear per axis (``x = m*yaw + b``, ``y = m*pitch + b``), so
calibration reduces to two robust line fits that are converted back into
:class:`~lookout.screen_mapping.ScreenMappingParams`.
"""

from __future__ import annotations

import numpy as np

from .models import GazeDirection, Layout, RegionKind
from .screen_mapping import ScreenMappingParams
from .speaker import SpeakerSegment

__all__ = ["fit_linear_ransac", "calibrate_mapping", "weak_labels_from_speaker"]

_EPS = 1e-9

# One weak label: (yaw, pitch, target_x, target_y).
WeakLabel = tuple[float, float, float, float]


def fit_linear_ransac(
    xs: list[float],
    ys: list[float],
    threshold: float,
    iterations: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """Robustly fit ``y = slope * x + intercept``, rejecting outliers.

    Returns the least-squares fit over the largest inlier set found.
    """

    if len(xs) != len(ys):
        raise ValueError("xs and ys must have equal length")
    if len(xs) < 2:
        raise ValueError("need at least two points")

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.ptp(x)) < _EPS:
        raise ValueError("x has no variation")
    rng = np.random.default_rng(seed)
    n = len(x)

    best_inliers: np.ndarray | None = None
    for _ in range(iterations):
        i, j = rng.choice(n, size=2, replace=False)
        if abs(x[i] - x[j]) < _EPS:
            continue
        slope = (y[j] - y[i]) / (x[j] - x[i])
        intercept = y[i] - slope * x[i]
        residuals = np.abs(y - (slope * x + intercept))
        inliers = residuals <= threshold
        if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers

    if best_inliers is None or int(best_inliers.sum()) < 2:
        mask = np.ones(n, dtype=bool)
    else:
        mask = best_inliers

    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return float(slope), float(intercept)


def calibrate_mapping(
    labels: list[WeakLabel],
    threshold: float = 0.05,
    iterations: int = 200,
    seed: int = 0,
) -> ScreenMappingParams:
    """Fit per-viewer :class:`ScreenMappingParams` from weak labels.

    Raises ``ValueError`` if the geometry is degenerate (e.g. x does not
    increase with yaw), so the caller can fall back to the prior.
    """

    if len(labels) < 2:
        raise ValueError("need at least two labels to calibrate")

    yaws = [label[0] for label in labels]
    pitches = [label[1] for label in labels]
    xs = [label[2] for label in labels]
    ys = [label[3] for label in labels]

    mx, bx = fit_linear_ransac(yaws, xs, threshold, iterations, seed)
    if mx <= _EPS:
        raise ValueError("x must increase with yaw")

    my, by = fit_linear_ransac(pitches, ys, threshold, iterations, seed)
    if my >= -_EPS:
        raise ValueError("y must decrease as pitch increases")

    return ScreenMappingParams(
        yaw_at_left=-bx / mx,
        yaw_at_right=(1.0 - bx) / mx,
        pitch_at_top=-by / my,
        pitch_at_bottom=(1.0 - by) / my,
    )


def weak_labels_from_speaker(
    directions: list[GazeDirection],
    segments: list[SpeakerSegment],
    layout: Layout,
) -> list[WeakLabel]:
    """Build weak labels for one viewer from speaker segments.

    For each of the viewer's gaze directions, if someone else is speaking and
    that speaker has a tile in the viewer's layout, the tile centre is the weak
    target. Self-speaking frames and unknown speakers are skipped.
    """

    centers = {
        region.participant_id: region.center
        for region in layout.regions
        if region.kind is RegionKind.PARTICIPANT
    }
    labels: list[WeakLabel] = []
    for direction in directions:
        speaker = _speaker_at(segments, direction.timestamp)
        if speaker is None or speaker == direction.person_id:
            continue
        center = centers.get(speaker)
        if center is None:
            continue
        labels.append((direction.yaw, direction.pitch, center[0], center[1]))
    return labels


def _speaker_at(segments: list[SpeakerSegment], timestamp: float) -> str | None:
    for segment in segments:
        if segment.start_time <= timestamp < segment.end_time:
            return segment.participant_id
    return None
