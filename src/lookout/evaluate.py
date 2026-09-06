"""Evaluation harness.

Compares predicted :class:`~lookout.models.GazeEvent`s against a ground-truth
timeline produced by the cue protocol (see docs/research/07-eval-protocol.md).
Reports target hit-rate (overall and per grid size), the rate at which the
prediction was unknown/low-confidence, and off-screen recall. These are the
numbers that settle the geometric-vs-appearance and mapping-prior decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import GazeEvent, GazeTarget

__all__ = ["TruthInterval", "EvaluationResult", "load_truth", "predicted_target_at", "evaluate"]

_UNKNOWN_TARGETS = {
    GazeTarget.UNKNOWN.value,
    GazeTarget.LOW_CONFIDENCE.value,
    GazeTarget.NOT_VISIBLE.value,
}


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


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    hits: int
    unknown: int
    off_screen_total: int
    off_screen_hits: int
    per_grid: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def unknown_rate(self) -> float:
        return self.unknown / self.total if self.total else 0.0

    @property
    def off_screen_recall(self) -> float:
        return self.off_screen_hits / self.off_screen_total if self.off_screen_total else 0.0

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


def predicted_target_at(events: list[GazeEvent], viewer: str, timestamp: float) -> str | None:
    """The predicted target for ``viewer`` at ``timestamp``, or ``None`` if the
    prediction is silent there."""

    for event in events:
        if event.viewer_id != viewer:
            continue
        if event.start_time <= timestamp < event.end_time:
            return event.target
    return None


def evaluate(events: list[GazeEvent], truth: list[TruthInterval]) -> EvaluationResult:
    """Score predicted events against a ground-truth timeline.

    Each truth interval is checked at its midpoint. A hit is a prediction equal
    to the truth target. A silent or unknown/low-confidence prediction counts as
    unknown, never as a hit, honoring the conservative default.
    """

    total = 0
    hits = 0
    unknown = 0
    off_total = 0
    off_hits = 0
    per_grid: dict[str, list[int]] = {}

    for interval in truth:
        total += 1
        predicted = predicted_target_at(events, interval.viewer_id, interval.midpoint)
        is_off = interval.target == GazeTarget.OFF_SCREEN.value
        if is_off:
            off_total += 1

        if predicted is None or predicted in _UNKNOWN_TARGETS:
            unknown += 1
            hit = False
        else:
            hit = predicted == interval.target

        if hit:
            hits += 1
            if is_off:
                off_hits += 1

        if interval.grid is not None:
            bucket = per_grid.setdefault(interval.grid, [0, 0])
            bucket[0] += int(hit)
            bucket[1] += 1

    return EvaluationResult(
        total=total,
        hits=hits,
        unknown=unknown,
        off_screen_total=off_total,
        off_screen_hits=off_hits,
        per_grid={grid: (h, t) for grid, (h, t) in per_grid.items()},
    )
