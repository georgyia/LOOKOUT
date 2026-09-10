"""Distribution diagnostics: does this result look like signal?

A run can complete cleanly, reconcile its funnel, and still be meaningless. The
recorded three-minute gallery clip is the case to beat: 88.7% of attributed
duration pointed at one tile, which was the signature of a gaze proxy collapsing
every frontal face toward screen centre. Nothing in the output distinguished
that from a room genuinely watching one speaker; a person had to open the frames
to find out.

These are the shape checks that catch it. They cannot prove a result is right —
only measurement against ground truth does that — but they can say when it is
too degenerate to be worth reading, which is the more common failure.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .coverage import Degradation
from .models import GazeDirection, GazeEvent, GazePoint, GazeTarget

__all__ = [
    "HISTOGRAM_BINS",
    "Diagnostics",
    "DiagnosticThresholds",
    "screen_histogram",
    "normalized_entropy",
    "build_diagnostics",
    "distribution_warnings",
]

HISTOGRAM_BINS = 16

_NON_PARTICIPANT = {target.value for target in GazeTarget}


def normalized_entropy(shares: dict[str, float], support: int | None = None) -> float:
    """Shannon entropy over ``shares``, scaled to ``[0, 1]``.

    1.0 means duration was spread evenly across every available target; 0.0
    means it all went to one.

    ``support`` is how many targets there were to choose from, not how many were
    actually seen. Scaling by the targets seen would call a nine-tile call that
    collapsed onto two of them evenly spread, which is the opposite of what a
    reader needs — collapsing the field is itself the symptom.
    """

    positive = [value for value in shares.values() if value > 0.0]
    if not positive:
        return 0.0
    candidates = max(support or 0, len(positive))
    if candidates < 2:
        return 0.0
    total = sum(positive)
    entropy = -sum((v / total) * math.log(v / total) for v in positive)
    return entropy / math.log(candidates)


def screen_histogram(
    points: list[GazePoint], bins: int = HISTOGRAM_BINS
) -> tuple[tuple[int, ...], ...]:
    """A coarse 2-D histogram of where gaze landed, row-major from the top."""

    grid = [[0] * bins for _ in range(bins)]
    for point in points:
        column = min(bins - 1, int(point.x * bins))
        row = min(bins - 1, int(point.y * bins))
        grid[row][column] += 1
    return tuple(tuple(row) for row in grid)


def _centre_mass(histogram: tuple[tuple[int, ...], ...]) -> float:
    """Share of gaze landing in the central ninth of the screen."""

    total = sum(sum(row) for row in histogram)
    if not total:
        return 0.0
    bins = len(histogram)
    lo, hi = bins // 3, 2 * bins // 3
    inner = sum(sum(row[lo:hi]) for row in histogram[lo:hi])
    return inner / total


@dataclass(frozen=True)
class Diagnostics:
    """The shape of a run's results, independent of whether they are correct."""

    target_duration: dict[str, float] = field(default_factory=dict)
    reason_duration: dict[str, float] = field(default_factory=dict)
    top_target: str | None = None
    top_target_share: float = 0.0
    normalized_entropy: float = 0.0
    self_view_share: float = 0.0
    unresolved_share: float = 0.0
    centre_mass: float = 0.0
    limiting_factors: dict[str, int] = field(default_factory=dict)
    confidence_histogram: tuple[int, ...] = ()
    screen_histogram: tuple[tuple[int, ...], ...] = ()


def build_diagnostics(
    events: list[GazeEvent],
    points: list[GazePoint] | None = None,
    directions: list[GazeDirection] | None = None,
    participants: int | None = None,
) -> Diagnostics:
    """Summarize the shape of a run's results.

    ``participants`` is how many targets were available to look at, which sets
    the scale entropy is measured against.
    """

    target_duration: dict[str, float] = defaultdict(float)
    reason_duration: dict[str, float] = defaultdict(float)
    self_view = 0.0
    unresolved = 0.0

    for event in events:
        target_duration[event.target] += event.duration
        reason_duration[event.reason or "unrecorded"] += event.duration
        if event.target == event.viewer_id:
            self_view += event.duration
        if event.target in _NON_PARTICIPANT:
            unresolved += event.duration

    total = sum(target_duration.values())
    # Concentration is a claim about who was looked at, so the non-participant
    # outcomes are excluded: a run that is mostly "unknown" is not concentrated,
    # it is empty, and the funnel already says so.
    participant_shares = {
        target: duration
        for target, duration in target_duration.items()
        if target not in _NON_PARTICIPANT
    }
    participant_total = sum(participant_shares.values())
    top_target = (
        max(participant_shares, key=lambda t: participant_shares[t]) if participant_shares else None
    )

    histogram = screen_histogram(points or [])
    confidence_bins = [0] * 10
    for direction in directions or []:
        confidence_bins[min(9, int(direction.confidence * 10))] += 1

    limiting: dict[str, int] = defaultdict(int)
    for direction in directions or []:
        if direction.quality is not None:
            limiting[direction.quality.limiting] += 1

    return Diagnostics(
        target_duration={t: round(d, 3) for t, d in sorted(target_duration.items())},
        reason_duration={r: round(d, 3) for r, d in sorted(reason_duration.items())},
        top_target=top_target,
        top_target_share=(
            round(participant_shares[top_target] / participant_total, 4)
            if top_target and participant_total
            else 0.0
        ),
        normalized_entropy=round(normalized_entropy(participant_shares, participants), 4),
        self_view_share=round(self_view / total, 4) if total else 0.0,
        unresolved_share=round(unresolved / total, 4) if total else 0.0,
        centre_mass=round(_centre_mass(histogram), 4),
        limiting_factors=dict(sorted(limiting.items())),
        confidence_histogram=tuple(confidence_bins),
        screen_histogram=histogram,
    )


@dataclass(frozen=True)
class DiagnosticThresholds:
    """When a distribution is degenerate enough to be worth saying out loud.

    These are priors. They should be revisited once the evaluation harness can
    say which of them actually predict a bad result.
    """

    max_target_share: float = 0.6
    min_normalized_entropy: float = 0.4
    max_centre_mass: float = 0.5
    max_self_view_share: float = 0.25


def distribution_warnings(
    diagnostics: Diagnostics,
    layout_sources: tuple[str, ...] = (),
    thresholds: DiagnosticThresholds | None = None,
) -> tuple[Degradation, ...]:
    """Flag distributions that look like an artifact rather than a finding."""

    thresholds = thresholds or DiagnosticThresholds()
    found: list[Degradation] = []
    assumed = "assumed_shared" in layout_sources

    if diagnostics.top_target and diagnostics.top_target_share > thresholds.max_target_share:
        found.append(
            Degradation(
                stage="attribution",
                code="concentrated_targets",
                detail=(
                    f"{diagnostics.top_target_share:.1%} of attributed duration points at "
                    f"{diagnostics.top_target}."
                ),
                impact=(
                    "A single dominant target is what a degenerate estimator produces as "
                    "readily as a room watching one speaker."
                    + (
                        " Under an assumed shared layout the dominant tile is the same "
                        "screen position for every viewer, which makes the coincidence "
                        "more likely than the finding."
                        if assumed
                        else ""
                    )
                ),
            )
        )

    if (
        diagnostics.top_target
        and diagnostics.normalized_entropy < thresholds.min_normalized_entropy
    ):
        found.append(
            Degradation(
                stage="attribution",
                code="low_target_entropy",
                detail=(
                    f"Attributed duration is spread across targets with normalized entropy "
                    f"{diagnostics.normalized_entropy:.2f}."
                ),
                impact="The result carries little information about who looked at whom.",
            )
        )

    if diagnostics.centre_mass > thresholds.max_centre_mass:
        found.append(
            Degradation(
                stage="mapping",
                code="centre_clustered_gaze",
                detail=(
                    f"{diagnostics.centre_mass:.1%} of mapped points fall in the central "
                    f"ninth of the screen."
                ),
                impact=(
                    "Either the estimator has no usable eye signal and is reporting head "
                    "orientation, or the angle-to-screen prior is too narrow for this "
                    "geometry. Both collapse distinct targets onto the centre tile."
                ),
            )
        )

    if diagnostics.self_view_share > thresholds.max_self_view_share:
        found.append(
            Degradation(
                stage="attribution",
                code="high_self_view",
                detail=(
                    f"{diagnostics.self_view_share:.1%} of attributed duration is a viewer "
                    f"looking at their own tile."
                ),
                impact=(
                    "Self-view is plausible in a gallery call but rarely dominant; a high "
                    "share suggests the mapping is centred on the viewer's own position."
                ),
            )
        )

    return tuple(found)
