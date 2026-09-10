"""A scripted run with known answers.

Ground truth for gaze normally requires recording a person following a cue
script, which means nothing about scoring can run in CI and nobody can reproduce
the evaluation path without their own recording.

This generates a run whose answers are known by construction: gaze directions
are produced by *inverting* the angle-to-screen mapping onto the centre of an
intended tile, so a perfect pipeline must recover the tile it started from.

What this validates: mapping, smoothing, fixation detection, attribution, event
aggregation and scoring. What it does not validate: face detection, landmarks,
or any gaze model — those still need the cue protocol on real recordings (see
docs/research/07-eval-protocol.md). A report generated from this fixture names
its adapter `synthetic` so the distinction is never lost.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from lookout.evaluate import TruthInterval
from lookout.models import (
    GazeDirection,
    GazeTarget,
    HeadPose,
    Layout,
    LayoutSource,
    Region,
    RegionKind,
)
from lookout.screen_mapping import ScreenMappingParams

AWAY = "away"


@dataclass(frozen=True)
class Cue:
    """Where one viewer was told to look over ``[start, end)``.

    ``target`` is a participant id, ``off_screen`` for a deliberate look away
    from the screen, or ``away`` for a window with no observation at all, so the
    silent case is exercised rather than assumed.
    """

    viewer_id: str
    target: str
    start_time: float
    end_time: float


def grid_layout(
    rows: int,
    cols: int,
    viewer_id: str = "recording",
    source: LayoutSource = LayoutSource.RECORDING,
) -> Layout:
    """An equal ``rows x cols`` gallery, slots numbered in reading order."""

    regions = tuple(
        Region(
            RegionKind.PARTICIPANT,
            c / cols,
            r / rows,
            1.0 / cols,
            1.0 / rows,
            f"slot_{r * cols + c}",
        )
        for r in range(rows)
        for c in range(cols)
    )
    return Layout(viewer_id, regions, 0.0, source)


def angles_for(x: float, y: float, params: ScreenMappingParams) -> tuple[float, float]:
    """The angle that ``screen_mapping.project`` maps to ``(x, y)``.

    The exact inverse of the projection, so a noiseless fixture round-trips.
    """

    yaw = params.yaw_at_left + x * (params.yaw_at_right - params.yaw_at_left)
    pitch = params.pitch_at_top - y * (params.pitch_at_top - params.pitch_at_bottom)
    return yaw, pitch


def _target_point(layout: Layout, target: str) -> tuple[float, float] | None:
    if target == GazeTarget.OFF_SCREEN.value:
        return (1.9, 0.5)  # comfortably past the off-screen margin
    for region in layout.participant_regions():
        if region.participant_id == target:
            return region.center
    raise ValueError(f"no region for target {target!r}")


def scripted_directions(
    layout: Layout,
    cues: list[Cue],
    fps: float = 5.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
    params: ScreenMappingParams | None = None,
) -> list[GazeDirection]:
    """Observations a perfect estimator would produce for ``cues``.

    ``noise_sigma`` adds gaussian angular error in radians, which is how the
    fixture is used to check that scoring degrades rather than collapses.
    """

    params = params or ScreenMappingParams()
    rng = random.Random(seed)
    step = 1.0 / fps
    directions: list[GazeDirection] = []

    for cue in cues:
        if cue.target == AWAY:
            continue  # no observation at all: the prediction should be silent
        point = _target_point(layout, cue.target)
        assert point is not None
        timestamp = cue.start_time
        while timestamp < cue.end_time - 1e-9:
            yaw, pitch = angles_for(point[0], point[1], params)
            if noise_sigma:
                yaw += rng.gauss(0.0, noise_sigma)
                pitch += rng.gauss(0.0, noise_sigma)
            directions.append(
                GazeDirection(
                    timestamp=round(timestamp, 6),
                    person_id=cue.viewer_id,
                    yaw=yaw,
                    pitch=pitch,
                    head_pose=HeadPose(0.0, 0.0, 0.0),
                    confidence=0.9,
                )
            )
            timestamp += step

    return directions


def truth_intervals(cues: list[Cue], grid: str) -> list[TruthInterval]:
    """The cue schedule as ground truth. The schedule *is* the answer."""

    return [
        TruthInterval(
            viewer_id=cue.viewer_id,
            start_time=cue.start_time,
            end_time=cue.end_time,
            target=GazeTarget.UNKNOWN.value if cue.target == AWAY else cue.target,
            grid=grid,
        )
        for cue in cues
    ]


def tour(viewer: str, targets: list[str], dwell: float = 2.0, start: float = 0.0) -> list[Cue]:
    """A viewer looking at each target in turn for ``dwell`` seconds."""

    return [
        Cue(viewer, target, start + i * dwell, start + (i + 1) * dwell)
        for i, target in enumerate(targets)
    ]
