"""Run coverage and degradations.

Coverage is the funnel a run actually achieved: how many frames were sampled,
how many faces were found in them, how many directions survived mapping, and how
many of those became events. A hit rate reported without it is unreadable — a
tidy-looking result over 8% of the frames is not the same claim as the same
result over 90% of them.

Degradations are what the run had to settle for. They are derived from the
coverage rather than logged in passing, so they are a pure, testable judgement
about a run's state instead of a side effect scattered through the stages.

This module sits below the pipeline and the run record and imports neither.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "ParticipantCoverage",
    "Coverage",
    "Degradation",
    "DegradationThresholds",
    "detect_degradations",
]


def _check_counts(values: dict[str, int]) -> None:
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class ParticipantCoverage:
    """The same funnel, for one participant.

    Aggregate coverage hides the case that matters most: one tile contributing
    almost nothing while the total still looks healthy.
    """

    participant_id: str
    face_attempts: int = 0
    face_hits: int = 0
    directions: int = 0
    points: int = 0
    off_screen: int = 0
    fixations: int = 0
    attributions: int = 0
    events: int = 0

    def __post_init__(self) -> None:
        if not self.participant_id:
            raise ValueError("participant_id must be non-empty")
        _check_counts(
            {
                "face_attempts": self.face_attempts,
                "face_hits": self.face_hits,
                "directions": self.directions,
                "points": self.points,
                "off_screen": self.off_screen,
                "fixations": self.fixations,
                "attributions": self.attributions,
                "events": self.events,
            }
        )
        if self.face_hits > self.face_attempts:
            raise ValueError("face_hits cannot exceed face_attempts")

    @property
    def face_hit_rate(self) -> float:
        return self.face_hits / self.face_attempts if self.face_attempts else 0.0


@dataclass(frozen=True)
class Coverage:
    """How much of the recording survived each stage.

    The funnel must reconcile: every direction is either mapped onto the screen
    or rejected as off screen, and a face cannot be found more often than it was
    looked for. Both are enforced here so a miscounted stage fails loudly rather
    than producing a plausible report.
    """

    frames: int = 0
    layouts: int = 0
    layout_changes: int = 0
    participants_detected: int = 0
    face_attempts: int = 0
    face_hits: int = 0
    directions: int = 0
    points: int = 0
    off_screen: int = 0
    fixations: int = 0
    attributions: int = 0
    events: int = 0
    layout_sources: tuple[str, ...] = ()
    per_participant: tuple[ParticipantCoverage, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_counts(
            {
                "frames": self.frames,
                "layouts": self.layouts,
                "layout_changes": self.layout_changes,
                "participants_detected": self.participants_detected,
                "face_attempts": self.face_attempts,
                "face_hits": self.face_hits,
                "directions": self.directions,
                "points": self.points,
                "off_screen": self.off_screen,
                "fixations": self.fixations,
                "attributions": self.attributions,
                "events": self.events,
            }
        )
        if self.face_hits > self.face_attempts:
            raise ValueError("face_hits cannot exceed face_attempts")
        if self.points + self.off_screen != self.directions:
            raise ValueError("every direction must be mapped on screen or rejected as off screen")

    @property
    def face_hit_rate(self) -> float:
        return self.face_hits / self.face_attempts if self.face_attempts else 0.0

    @property
    def off_screen_rate(self) -> float:
        return self.off_screen / self.directions if self.directions else 0.0

    def with_observation(
        self,
        *,
        frames: int,
        layouts: int,
        layout_changes: int,
        participants_detected: int,
        faces_per_participant: dict[str, tuple[int, int]],
    ) -> Coverage:
        """Merge the observation stage's counts into attribution's coverage.

        Observation and attribution run in separate passes — attribution can be
        re-run from the raw store with no video at all — so their counts are
        collected separately and joined here. The join is over the union of
        participants, not attribution's alone: a tile where no face was ever
        found produces no directions, and that is exactly the participant the
        reader needs to see.
        """

        attributed = {entry.participant_id: entry for entry in self.per_participant}
        empty = ParticipantCoverage(participant_id="_")
        merged = tuple(
            ParticipantCoverage(
                participant_id=participant,
                face_attempts=faces_per_participant.get(participant, (0, 0))[0],
                face_hits=faces_per_participant.get(participant, (0, 0))[1],
                directions=attributed.get(participant, empty).directions,
                points=attributed.get(participant, empty).points,
                off_screen=attributed.get(participant, empty).off_screen,
                fixations=attributed.get(participant, empty).fixations,
                attributions=attributed.get(participant, empty).attributions,
                events=attributed.get(participant, empty).events,
            )
            for participant in sorted(set(attributed) | set(faces_per_participant))
        )
        return Coverage(
            frames=frames,
            layouts=layouts,
            layout_changes=layout_changes,
            participants_detected=participants_detected,
            face_attempts=sum(a for a, _ in faces_per_participant.values()),
            face_hits=sum(h for _, h in faces_per_participant.values()),
            directions=self.directions,
            points=self.points,
            off_screen=self.off_screen,
            fixations=self.fixations,
            attributions=self.attributions,
            events=self.events,
            layout_sources=self.layout_sources,
            per_participant=merged,
        )


@dataclass(frozen=True)
class Degradation:
    """Something the run had to settle for, and what it costs the reader."""

    stage: str
    code: str
    detail: str
    impact: str


@dataclass(frozen=True)
class DegradationThresholds:
    """When a funnel stage is thin enough to be worth saying out loud."""

    min_face_hit_rate: float = 0.5
    max_off_screen_rate: float = 0.5


def detect_degradations(
    coverage: Coverage,
    thresholds: DegradationThresholds | None = None,
) -> tuple[Degradation, ...]:
    """Derive the degradations implied by a run's coverage."""

    thresholds = thresholds or DegradationThresholds()
    found: list[Degradation] = []

    if coverage.layouts == 0:
        found.append(
            Degradation(
                stage="layout",
                code="no_layout",
                detail="No stable layout was reconstructed from the recording.",
                impact="No gaze could be attributed to a participant.",
            )
        )
    elif coverage.participants_detected == 0:
        found.append(
            Degradation(
                stage="layout",
                code="no_participant_regions",
                detail="A layout was found but it contained no participant regions.",
                impact="Every attribution is unknown; there was nothing to look at.",
            )
        )

    if "assumed_shared" in coverage.layout_sources:
        found.append(
            Degradation(
                stage="layout",
                code="assumed_shared_layout",
                detail=(
                    "The recorded layout was applied to every viewer, "
                    "because per-viewer layouts were not supplied."
                ),
                impact=(
                    "Targets are only credible for the participant whose screen was "
                    "recorded; every other viewer saw a different arrangement."
                ),
            )
        )

    if coverage.face_attempts and coverage.face_hit_rate < thresholds.min_face_hit_rate:
        found.append(
            Degradation(
                stage="observation",
                code="low_face_hit_rate",
                detail=(
                    f"A face was found in {coverage.face_hits} of "
                    f"{coverage.face_attempts} tile crops "
                    f"({coverage.face_hit_rate:.1%})."
                ),
                impact="Coverage is sparse; durations understate time spent looking.",
            )
        )

    if coverage.directions and coverage.off_screen_rate > thresholds.max_off_screen_rate:
        found.append(
            Degradation(
                stage="mapping",
                code="high_off_screen_rate",
                detail=(
                    f"{coverage.off_screen} of {coverage.directions} directions "
                    f"({coverage.off_screen_rate:.1%}) fell outside the screen."
                ),
                impact=(
                    "Either the participants were looking away, or the angle-to-screen "
                    "prior does not fit this geometry."
                ),
            )
        )

    if coverage.directions and coverage.events == 0:
        found.append(
            Degradation(
                stage="events",
                code="no_events",
                detail="Directions were recorded but none survived into an event.",
                impact="There is nothing to report beyond the raw observations.",
            )
        )

    return tuple(found)
