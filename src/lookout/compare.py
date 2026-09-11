"""Comparing two runs.

An experiment is two runs and a question — geometric gaze against appearance,
one mapping prior against another, calibrated against assumed — and until now it
produced two directories and a manual diff.

The order here is the argument, as it is in the report. Whether two runs are
comparable at all comes first: different adapters or a different recording mean
they answer different questions, and a difference between them is not a finding.
Coverage comes next, because a result that moved when the pipeline saw more of
the recording is a different claim from one that moved when attribution changed
its mind. Results come last.

Nothing here calls a difference significant. Where both runs were scored, an
interval that overlaps the other run's is reported as overlapping, because a
difference inside the noise is the most common way a comparison misleads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .runrecord import RunRecord

__all__ = ["Difference", "Comparison", "compare"]


@dataclass(frozen=True)
class Difference:
    """One field that differs, with both values."""

    field: str
    left: Any
    right: Any

    def __str__(self) -> str:
        return f"{self.field}: {self.left} -> {self.right}"


@dataclass(frozen=True)
class Comparison:
    """What differs between two runs, grouped by what it means."""

    comparable: bool
    blockers: tuple[Difference, ...] = ()
    configuration: tuple[Difference, ...] = ()
    coverage: tuple[Difference, ...] = ()
    results: tuple[Difference, ...] = ()
    evaluation: tuple[Difference, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def identical(self) -> bool:
        return not (
            self.blockers
            or self.configuration
            or self.coverage
            or self.results
            or self.evaluation
        )


def _rounded(value: float, places: int = 4) -> float:
    return round(value, places)


def _compare_provenance(left: RunRecord, right: RunRecord) -> tuple[Difference, ...]:
    """Reasons the two runs may not be answering the same question."""

    blockers: list[Difference] = []

    left_video = left.provenance.video
    right_video = right.provenance.video
    if left_video and right_video and left_video.sha256 != right_video.sha256:
        blockers.append(Difference("video", left_video.path, right_video.path))

    def adapters(record: RunRecord) -> dict[str, str]:
        return {a.role: a.implementation for a in record.provenance.adapters}

    left_adapters, right_adapters = adapters(left), adapters(right)
    for role in sorted(set(left_adapters) | set(right_adapters)):
        if left_adapters.get(role) != right_adapters.get(role):
            blockers.append(
                Difference(
                    f"adapter.{role}",
                    left_adapters.get(role, "none"),
                    right_adapters.get(role, "none"),
                )
            )
    return tuple(blockers)


def _flatten(values: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(values, dict):
        flat: dict[str, Any] = {}
        for key, value in values.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(value, path))
        return flat
    return {prefix: values}


def _compare_configuration(left: RunRecord, right: RunRecord) -> tuple[Difference, ...]:
    """Named per field. "The hashes differ" is true and useless."""

    left_flat = _flatten(left.config)
    right_flat = _flatten(right.config)
    return tuple(
        Difference(f"config.{key}", left_flat.get(key), right_flat.get(key))
        for key in sorted(set(left_flat) | set(right_flat))
        if left_flat.get(key) != right_flat.get(key)
    )


_COVERAGE_FIELDS = (
    "frames",
    "participants_detected",
    "face_attempts",
    "face_hits",
    "directions",
    "points",
    "off_screen",
    "fixations",
    "attributions",
    "events",
    "not_visible",
)


def _compare_coverage(left: RunRecord, right: RunRecord) -> tuple[Difference, ...]:
    if not left.coverage or not right.coverage:
        return ()
    found = [
        Difference(f"coverage.{name}", getattr(left.coverage, name), getattr(right.coverage, name))
        for name in _COVERAGE_FIELDS
        if getattr(left.coverage, name) != getattr(right.coverage, name)
    ]
    for name in ("face_hit_rate", "off_screen_rate"):
        a, b = _rounded(getattr(left.coverage, name)), _rounded(getattr(right.coverage, name))
        if a != b:
            found.append(Difference(f"coverage.{name}", a, b))
    return tuple(found)


def _compare_results(left: RunRecord, right: RunRecord) -> tuple[Difference, ...]:
    found: list[Difference] = []
    if left.diagnostics and right.diagnostics:
        for name in (
            "top_target",
            "top_target_share",
            "normalized_entropy",
            "self_view_share",
            "unresolved_share",
            "centre_mass",
        ):
            a, b = getattr(left.diagnostics, name), getattr(right.diagnostics, name)
            if a != b:
                found.append(Difference(f"distribution.{name}", a, b))

    left_totals = _durations(left)
    right_totals = _durations(right)
    for target in sorted(set(left_totals) | set(right_totals)):
        a = _rounded(left_totals.get(target, 0.0), 1)
        b = _rounded(right_totals.get(target, 0.0), 1)
        if a != b:
            found.append(Difference(f"duration.{target}", a, b))
    return tuple(found)


def _durations(record: RunRecord) -> dict[str, float]:
    viewers = record.results.get("viewers") if record.results else None
    if not isinstance(viewers, dict):
        return {}
    totals: dict[str, float] = {}
    for stats in viewers.values():
        by_target = stats.get("duration_by_target", {}) if isinstance(stats, dict) else {}
        for target, duration in by_target.items():
            totals[target] = totals.get(target, 0.0) + float(duration)
    return totals


def _interval(record: RunRecord) -> tuple[float, float] | None:
    if not record.evaluation:
        return None
    raw = record.evaluation.get("hit_rate_ci95")
    if isinstance(raw, list) and len(raw) == 2:
        return (float(raw[0]), float(raw[1]))
    return None


def _compare_evaluation(
    left: RunRecord, right: RunRecord
) -> tuple[tuple[Difference, ...], tuple[str, ...]]:
    if not left.evaluation or not right.evaluation:
        if bool(left.evaluation) != bool(right.evaluation):
            return (
                (Difference("evaluation", bool(left.evaluation), bool(right.evaluation)),),
                ("Only one run was scored, so accuracy cannot be compared.",),
            )
        return ((), ())

    found: list[Difference] = []
    notes: list[str] = []
    for key in ("hit_rate", "unknown_rate", "off_screen_recall", "beats_baseline"):
        a, b = left.evaluation.get(key), right.evaluation.get(key)
        if a != b:
            found.append(Difference(f"evaluation.{key}", a, b))

    left_ci, right_ci = _interval(left), _interval(right)
    if left_ci and right_ci:
        overlap = left_ci[0] <= right_ci[1] and right_ci[0] <= left_ci[1]
        if overlap:
            notes.append(
                f"Hit-rate intervals overlap ({left_ci[0]:.1%}-{left_ci[1]:.1%} against "
                f"{right_ci[0]:.1%}-{right_ci[1]:.1%}): the difference is inconclusive."
            )
        else:
            notes.append(
                f"Hit-rate intervals are disjoint ({left_ci[0]:.1%}-{left_ci[1]:.1%} against "
                f"{right_ci[0]:.1%}-{right_ci[1]:.1%})."
            )
    return tuple(found), tuple(notes)


def compare(left: RunRecord, right: RunRecord) -> Comparison:
    """Report what differs between two runs, grouped by what it means."""

    blockers = _compare_provenance(left, right)
    evaluation, notes = _compare_evaluation(left, right)

    extra: list[str] = list(notes)
    if blockers:
        extra.insert(
            0,
            "These runs are not directly comparable: they differ in what was "
            "analyzed or in what produced the observations.",
        )

    coverage = _compare_coverage(left, right)
    if coverage and any(d.field == "coverage.frames" for d in coverage):
        extra.append(
            "The runs cover different amounts of recording, so a difference in "
            "durations does not by itself mean attribution changed."
        )

    return Comparison(
        comparable=not blockers,
        blockers=blockers,
        configuration=_compare_configuration(left, right),
        coverage=coverage,
        results=_compare_results(left, right),
        evaluation=evaluation,
        notes=tuple(extra),
    )
