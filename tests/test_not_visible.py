"""Absence of a result is not absence of a person.

`GazeTarget.NOT_VISIBLE` was defined with the first data contracts and emitted by
no stage, so a participant the pipeline could not see was reported exactly like
one who was never in the call.
"""

from pathlib import Path

from lookout import artifacts, store
from lookout.evaluate import TruthInterval, evaluate
from lookout.models import (
    GazeDirection,
    GazeTarget,
    HeadPose,
    Layout,
    LayoutSource,
    Region,
    RegionKind,
)
from lookout.pipeline import EVENTS, GAZE_RAW, LAYOUT, AnalysisConfig, attribute
from tests.synthetic import angles_for, grid_layout, scripted_directions, tour

CONFIG = AnalysisConfig(min_fixation=0.3, max_gap=0.4, event_min_duration=0.4)
NOT_VISIBLE = GazeTarget.NOT_VISIBLE.value


def _run(tmp_path: Path, layout: Layout, directions: list[GazeDirection]) -> list:
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    store.write_gaze(out / GAZE_RAW, directions)
    artifacts.write_layouts(out / LAYOUT, [layout])
    attribute(out, CONFIG)
    return artifacts.read_events(out / EVENTS)


def test_a_participant_who_was_never_resolved_is_reported_not_omitted(
    tmp_path: Path,
) -> None:
    layout = grid_layout(2, 2)
    cues = tour("slot_0", ["slot_1", "slot_2"], dwell=3.0)
    events = _run(tmp_path, layout, scripted_directions(layout, cues, fps=5.0))

    by_viewer = {e.viewer_id: e for e in events}
    assert set(by_viewer) == {"slot_0", "slot_1", "slot_2", "slot_3"}
    for silent in ("slot_1", "slot_2", "slot_3"):
        assert by_viewer[silent].target == NOT_VISIBLE
        assert by_viewer[silent].reason == "no usable observation"


def test_an_observed_participant_is_not_also_reported_unseen(tmp_path: Path) -> None:
    layout = grid_layout(2, 2)
    cues = tour("slot_0", ["slot_1", "slot_2"], dwell=3.0)
    events = _run(tmp_path, layout, scripted_directions(layout, cues, fps=5.0))

    observed = [e for e in events if e.viewer_id == "slot_0"]
    assert observed
    assert all(e.target != NOT_VISIBLE for e in observed)


def test_not_visible_carries_the_layout_provenance(tmp_path: Path) -> None:
    """A recording layout applied to every viewer is assumed_shared, and the
    result must say so rather than claiming the recording's own provenance."""

    layout = grid_layout(2, 2)
    cues = tour("slot_0", ["slot_1"], dwell=3.0)
    events = _run(tmp_path, layout, scripted_directions(layout, cues, fps=5.0))

    unseen = [e for e in events if e.target == NOT_VISIBLE]
    assert unseen
    assert all(e.layout_source is LayoutSource.ASSUMED_SHARED for e in unseen)


def test_a_participant_absent_from_the_layout_produces_nothing(tmp_path: Path) -> None:
    """Not visible means present but unobservable. Someone who is not in the
    layout is not present, and inventing a result for them would be worse."""

    layout = Layout(
        "recording",
        (Region(RegionKind.PARTICIPANT, 0.0, 0.0, 0.5, 1.0, "alice"),),
        0.0,
        LayoutSource.RECORDING,
    )
    yaw, pitch = angles_for(0.25, 0.5, CONFIG.mapping)
    directions = [
        GazeDirection(i * 0.2, "alice", yaw, pitch, HeadPose(0, 0, 0), 0.9) for i in range(20)
    ]
    events = _run(tmp_path, layout, directions)
    assert {e.viewer_id for e in events} == {"alice"}


def test_coverage_counts_what_could_not_be_seen(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    layout = grid_layout(2, 2)
    cues = tour("slot_0", ["slot_1", "slot_2"], dwell=3.0)
    store.write_gaze(out / GAZE_RAW, scripted_directions(layout, cues, fps=5.0))
    artifacts.write_layouts(out / LAYOUT, [layout])

    coverage, _ = attribute(out, CONFIG)

    assert coverage.not_visible == 3
    per_participant = {e.participant_id: e for e in coverage.per_participant}
    assert per_participant["slot_0"].not_visible == 0
    assert per_participant["slot_3"].not_visible == 1
    assert per_participant["slot_0"].observed is False  # no face counts here
    assert set(per_participant) == {"slot_0", "slot_1", "slot_2", "slot_3"}


def test_scoring_treats_not_visible_as_declining_never_as_wrong() -> None:
    """Evaluation already refused to count it as a hit; it must also not be
    counted as a wrong answer, and it is now reported separately."""

    truth = [TruthInterval("slot_1", 0.0, 2.0, "slot_0", "2x2")]
    from lookout.models import GazeEvent

    events = [
        GazeEvent("slot_1", NOT_VISIBLE, 0.0, 2.0, 1.0, LayoutSource.ASSUMED_SHARED)
    ]
    result = evaluate(events, truth)

    assert result.hits == 0
    assert result.wrong == 0
    assert result.unknown == 1
    assert result.not_visible == 1
    assert result.explicit_unknown == 0
    assert result.silent == 0


def test_not_visible_is_distinguished_from_silence_and_low_confidence() -> None:
    from lookout.models import GazeEvent

    truth = [
        TruthInterval("v", 0.0, 2.0, "slot_0", "2x2"),
        TruthInterval("v", 2.0, 4.0, "slot_0", "2x2"),
        TruthInterval("v", 4.0, 6.0, "slot_0", "2x2"),
        TruthInterval("v", 6.0, 8.0, "slot_0", "2x2"),
    ]
    source = LayoutSource.ASSUMED_SHARED
    events = [
        GazeEvent("v", NOT_VISIBLE, 0.0, 2.0, 1.0, source),
        GazeEvent("v", GazeTarget.UNKNOWN.value, 2.0, 4.0, 0.5, source),
        GazeEvent("v", GazeTarget.LOW_CONFIDENCE.value, 4.0, 6.0, 0.1, source),
        # 6.0-8.0 left silent
    ]
    result = evaluate(events, truth)

    assert (result.not_visible, result.explicit_unknown, result.low_confidence,
            result.silent) == (1, 1, 1, 1)
    assert result.unknown == 4
