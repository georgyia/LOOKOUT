import pytest

from lookout.coverage import (
    Coverage,
    DegradationThresholds,
    ParticipantCoverage,
    detect_degradations,
)


def _codes(coverage: Coverage, thresholds: DegradationThresholds | None = None) -> set[str]:
    return {d.code for d in detect_degradations(coverage, thresholds)}


def test_the_funnel_must_reconcile() -> None:
    """A miscounted stage should fail loudly rather than read plausibly."""

    with pytest.raises(ValueError, match="off screen"):
        Coverage(directions=10, points=4, off_screen=3)

    with pytest.raises(ValueError, match="face_hits cannot exceed"):
        Coverage(face_attempts=2, face_hits=3)

    with pytest.raises(ValueError, match="non-negative"):
        Coverage(frames=-1)

    Coverage(directions=10, points=7, off_screen=3)  # reconciles


def test_rates_are_zero_rather_than_undefined_when_nothing_was_attempted() -> None:
    empty = Coverage()
    assert empty.face_hit_rate == 0.0
    assert empty.off_screen_rate == 0.0


def test_observation_counts_join_over_the_union_of_participants() -> None:
    """Attribution never sees a participant whose face was never found."""

    attributed = Coverage(
        directions=6,
        points=6,
        off_screen=0,
        events=2,
        per_participant=(ParticipantCoverage("slot_0", directions=6, points=6, events=2),),
    )
    merged = attributed.with_observation(
        frames=3,
        layouts=1,
        layout_changes=0,
        participants_detected=2,
        faces_per_participant={"slot_0": (3, 3), "slot_1": (3, 0)},
    )

    seen = {entry.participant_id: entry for entry in merged.per_participant}
    assert set(seen) == {"slot_0", "slot_1"}
    assert seen["slot_1"].face_attempts == 3
    assert seen["slot_1"].face_hits == 0
    assert seen["slot_1"].directions == 0
    assert merged.face_attempts == 6
    assert merged.face_hits == 3
    assert merged.face_hit_rate == 0.5


def test_a_healthy_run_reports_no_degradations() -> None:
    healthy = Coverage(
        frames=10,
        layouts=1,
        participants_detected=4,
        face_attempts=40,
        face_hits=38,
        directions=38,
        points=36,
        off_screen=2,
        events=8,
        layout_sources=("manifest",),
    )
    assert detect_degradations(healthy) == ()


def test_the_shared_layout_assumption_is_always_declared() -> None:
    """It is the v1 assumption, so it is never surprising and never omitted."""

    coverage = Coverage(
        frames=10,
        layouts=1,
        participants_detected=4,
        face_attempts=40,
        face_hits=40,
        directions=40,
        points=40,
        events=8,
        layout_sources=("assumed_shared",),
    )
    (degradation,) = detect_degradations(coverage)
    assert degradation.code == "assumed_shared_layout"
    assert "every viewer" in degradation.detail
    assert "recorded" in degradation.impact


def test_thin_stages_are_reported_with_their_consequence() -> None:
    coverage = Coverage(
        frames=10,
        layouts=1,
        participants_detected=4,
        face_attempts=40,
        face_hits=8,
        directions=8,
        points=1,
        off_screen=7,
        events=1,
        layout_sources=("manifest",),
    )
    found = {d.code: d for d in detect_degradations(coverage)}
    assert set(found) == {"low_face_hit_rate", "high_off_screen_rate"}
    assert "8 of 40" in found["low_face_hit_rate"].detail
    assert "20.0%" in found["low_face_hit_rate"].detail
    assert found["high_off_screen_rate"].stage == "mapping"


def test_thresholds_are_configurable() -> None:
    coverage = Coverage(
        frames=10,
        layouts=1,
        participants_detected=4,
        face_attempts=10,
        face_hits=6,
        directions=6,
        points=6,
        events=2,
        layout_sources=("manifest",),
    )
    assert _codes(coverage) == set()
    assert _codes(coverage, DegradationThresholds(min_face_hit_rate=0.9)) == {"low_face_hit_rate"}


def test_a_layout_without_participants_is_distinguished_from_no_layout() -> None:
    assert _codes(Coverage(layouts=0)) == {"no_layout"}
    assert _codes(Coverage(layouts=1, participants_detected=0)) == {"no_participant_regions"}


def test_observations_that_produce_nothing_are_reported() -> None:
    coverage = Coverage(
        frames=10,
        layouts=1,
        participants_detected=2,
        face_attempts=10,
        face_hits=10,
        directions=10,
        points=10,
        events=0,
        layout_sources=("manifest",),
    )
    assert "no_events" in _codes(coverage)
