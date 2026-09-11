"""Evaluation harness.

Compares predicted :class:`~lookout.models.GazeEvent`s against a ground-truth
timeline produced by the cue protocol (see docs/research/07-eval-protocol.md).

A hit rate on its own is unreadable. On a 3x3 grid uniform guessing scores about
11%, and an estimator that always answers "the centre tile" can score far higher
than that on a real call while containing no gaze signal at all — which is
precisely the failure mode observed on the recorded gallery clip. Every result
here is therefore reported against baselines, with a confidence interval, and
with the model's own degenerate-constant score computed explicitly.

A silent or unknown prediction is never a hit. Preferring ``unknown`` to a guess
is not penalized as a wrong answer, but it is not rewarded as a right one
either.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .models import GazeEvent, GazeTarget

__all__ = [
    "TruthInterval",
    "ReliabilityBin",
    "EvaluationResult",
    "load_truth",
    "predicted_target_at",
    "predicted_event_at",
    "evaluate",
]

_UNKNOWN_TARGETS = {
    GazeTarget.UNKNOWN.value,
    GazeTarget.LOW_CONFIDENCE.value,
    GazeTarget.NOT_VISIBLE.value,
}

_BOOTSTRAP_SAMPLES = 1000
_BOOTSTRAP_SEED = 20260101
_RELIABILITY_BINS = 5


@dataclass(frozen=True)
class TruthInterval:
    """A known gaze target for one viewer over ``[start, end)``.

    ``grid`` optionally records the layout size (e.g. "2x2") so hit-rate can be
    broken down by difficulty.
    """

    viewer_id: str
    start_time: float
    end_time: float
    target: str
    grid: str | None = None

    @property
    def midpoint(self) -> float:
        return (self.start_time + self.end_time) / 2.0

    def samples(self, step: float | None) -> list[float]:
        """Times at which to check the prediction.

        Midpoint sampling weights a 30-second window the same as a one-second
        one, and cannot tell a prediction that held for the window from one that
        happened to be right in the middle of it. ``step`` samples densely
        instead.
        """

        if step is None or step <= 0:
            return [self.midpoint]
        times: list[float] = []
        t = self.start_time + step / 2.0
        while t < self.end_time:
            times.append(t)
            t += step
        return times or [self.midpoint]


@dataclass(frozen=True)
class ReliabilityBin:
    """How often predictions in one confidence band were actually right."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    hit_rate: float


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    hits: int
    unknown: int
    off_screen_total: int
    off_screen_hits: int
    per_grid: dict[str, tuple[int, int]] = field(default_factory=dict)
    silent: int = 0
    explicit_unknown: int = 0
    low_confidence: int = 0
    not_visible: int = 0
    wrong: int = 0
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    baselines: dict[str, float] = field(default_factory=dict)
    hit_rate_ci: tuple[float, float] = (0.0, 0.0)
    reliability: tuple[ReliabilityBin, ...] = ()
    expected_calibration_error: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def unknown_rate(self) -> float:
        return self.unknown / self.total if self.total else 0.0

    @property
    def off_screen_recall(self) -> float:
        return self.off_screen_hits / self.off_screen_total if self.off_screen_total else 0.0

    @property
    def best_baseline(self) -> float:
        return max(self.baselines.values(), default=0.0)

    @property
    def beats_baseline(self) -> bool:
        """Whether the estimator carries evidence of gaze signal at all."""

        return self.hit_rate > self.best_baseline

    def grid_hit_rate(self, grid: str) -> float:
        hits, total = self.per_grid.get(grid, (0, 0))
        return hits / total if total else 0.0


def load_truth(path: str | Path) -> list[TruthInterval]:
    intervals: list[TruthInterval] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            intervals.append(
                TruthInterval(
                    viewer_id=str(row["viewer_id"]),
                    start_time=float(row["start_time"]),
                    end_time=float(row["end_time"]),
                    target=str(row["target"]),
                    grid=str(row["grid"]) if row.get("grid") is not None else None,
                )
            )
    return intervals


def predicted_event_at(
    events: list[GazeEvent], viewer: str, timestamp: float
) -> GazeEvent | None:
    """The predicted event for ``viewer`` at ``timestamp``, if any."""

    for event in events:
        if event.viewer_id != viewer:
            continue
        if event.start_time <= timestamp < event.end_time:
            return event
    return None


def predicted_target_at(events: list[GazeEvent], viewer: str, timestamp: float) -> str | None:
    """The predicted target for ``viewer`` at ``timestamp``, or ``None`` if the
    prediction is silent there."""

    event = predicted_event_at(events, viewer, timestamp)
    return event.target if event else None


def _baselines(
    truth_targets: list[str],
    predictions: list[str | None],
    candidates: int,
) -> dict[str, float]:
    """What trivial strategies would score on the same questions.

    ``model_modal_target`` is the sharpest of these: it replaces every prediction
    with whichever target the model named most often. An estimator scoring no
    better than that has no per-interval discrimination, however high its hit
    rate looks.
    """

    total = len(truth_targets)
    if not total:
        return {}

    counts = Counter(truth_targets)
    majority = counts.most_common(1)[0][1] / total

    named = [p for p in predictions if p is not None and p not in _UNKNOWN_TARGETS]
    modal_score = 0.0
    if named:
        modal_target = Counter(named).most_common(1)[0][0]
        modal_score = sum(1 for t in truth_targets if t == modal_target) / total

    return {
        "always_unknown": 0.0,
        "uniform_random": (1.0 / candidates) if candidates else 0.0,
        "majority_truth": majority,
        "model_modal_target": modal_score,
    }


def _bootstrap_ci(outcomes: list[bool]) -> tuple[float, float]:
    """A 95% interval on the hit rate, so a rate over 12 cues is not read like
    one over 1200."""

    if not outcomes:
        return (0.0, 0.0)
    rng = random.Random(_BOOTSTRAP_SEED)
    n = len(outcomes)
    rates = sorted(
        sum(outcomes[rng.randrange(n)] for _ in range(n)) / n for _ in range(_BOOTSTRAP_SAMPLES)
    )
    lo = rates[int(0.025 * _BOOTSTRAP_SAMPLES)]
    hi = rates[min(_BOOTSTRAP_SAMPLES - 1, int(0.975 * _BOOTSTRAP_SAMPLES))]
    return (round(lo, 4), round(hi, 4))


def _reliability(
    scored: list[tuple[float, bool]],
) -> tuple[tuple[ReliabilityBin, ...], float]:
    """Whether a stated confidence means what it says.

    Confidence is reported on every event and used to gate attribution, but has
    never been checked against how often such predictions were right.
    """

    if not scored:
        return ((), 0.0)

    bins: list[ReliabilityBin] = []
    error = 0.0
    width = 1.0 / _RELIABILITY_BINS
    for index in range(_RELIABILITY_BINS):
        lower, upper = index * width, (index + 1) * width
        members = [
            (c, hit)
            for c, hit in scored
            if (lower <= c < upper) or (index == _RELIABILITY_BINS - 1 and c == 1.0)
        ]
        if not members:
            continue
        mean_confidence = sum(c for c, _ in members) / len(members)
        hit_rate = sum(1 for _, hit in members if hit) / len(members)
        bins.append(
            ReliabilityBin(
                lower=round(lower, 3),
                upper=round(upper, 3),
                count=len(members),
                mean_confidence=round(mean_confidence, 4),
                hit_rate=round(hit_rate, 4),
            )
        )
        error += (len(members) / len(scored)) * abs(mean_confidence - hit_rate)

    return (tuple(bins), round(error, 4))


def evaluate(
    events: list[GazeEvent],
    truth: list[TruthInterval],
    *,
    sample_step: float | None = None,
    candidates: int | None = None,
) -> EvaluationResult:
    """Score predicted events against a ground-truth timeline.

    Each truth interval is checked at its midpoint, or densely when
    ``sample_step`` is given. A hit is a prediction equal to the truth target. A
    silent or unknown/low-confidence prediction counts as unknown, never as a
    hit, honoring the conservative default.

    ``candidates`` is how many targets there were to choose from, which sets the
    uniform-random baseline. It defaults to the distinct on-screen targets the
    ground truth mentions.
    """

    total = hits = unknown = wrong = 0
    silent = explicit_unknown = low_confidence = not_visible = 0
    off_total = off_hits = 0
    per_grid: dict[str, list[int]] = {}
    confusion: dict[str, dict[str, int]] = {}
    truth_targets: list[str] = []
    predictions: list[str | None] = []
    outcomes: list[bool] = []
    scored_confidence: list[tuple[float, bool]] = []

    for interval in truth:
        samples = interval.samples(sample_step)
        is_off = interval.target == GazeTarget.OFF_SCREEN.value

        for timestamp in samples:
            total += 1
            truth_targets.append(interval.target)
            event = predicted_event_at(events, interval.viewer_id, timestamp)
            predicted = event.target if event else None
            predictions.append(predicted)

            if is_off:
                off_total += 1

            if predicted is None:
                silent += 1
                unknown += 1
                hit = False
            elif predicted in _UNKNOWN_TARGETS:
                unknown += 1
                if predicted == GazeTarget.LOW_CONFIDENCE.value:
                    low_confidence += 1
                elif predicted == GazeTarget.NOT_VISIBLE.value:
                    not_visible += 1
                else:
                    explicit_unknown += 1
                hit = False
            else:
                hit = predicted == interval.target
                if not hit:
                    wrong += 1

            outcomes.append(hit)
            if event is not None and predicted not in _UNKNOWN_TARGETS:
                scored_confidence.append((event.confidence, hit))

            if hit:
                hits += 1
                if is_off:
                    off_hits += 1

            row = confusion.setdefault(interval.target, {})
            key = predicted if predicted is not None else "silent"
            row[key] = row.get(key, 0) + 1

            if interval.grid is not None:
                bucket = per_grid.setdefault(interval.grid, [0, 0])
                bucket[0] += int(hit)
                bucket[1] += 1

    if candidates is None:
        on_screen = {
            interval.target
            for interval in truth
            if interval.target not in _UNKNOWN_TARGETS
            and interval.target != GazeTarget.OFF_SCREEN.value
        }
        candidates = len(on_screen)

    reliability, calibration_error = _reliability(scored_confidence)

    return EvaluationResult(
        total=total,
        hits=hits,
        unknown=unknown,
        off_screen_total=off_total,
        off_screen_hits=off_hits,
        per_grid={grid: (h, t) for grid, (h, t) in per_grid.items()},
        silent=silent,
        explicit_unknown=explicit_unknown,
        low_confidence=low_confidence,
        not_visible=not_visible,
        wrong=wrong,
        confusion=confusion,
        baselines=_baselines(truth_targets, predictions, candidates),
        hit_rate_ci=_bootstrap_ci(outcomes),
        reliability=reliability,
        expected_calibration_error=calibration_error,
    )
