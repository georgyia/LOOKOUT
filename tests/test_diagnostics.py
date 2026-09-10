"""A run can complete cleanly and still be meaningless.

These pin the shape checks that say so. They cannot prove a result is right —
only ground truth does that — but they catch the more common failure, which is a
result too degenerate to be worth reading.
"""

import math

from lookout.diagnostics import (
    DiagnosticThresholds,
    build_diagnostics,
    distribution_warnings,
    normalized_entropy,
    screen_histogram,
)
from lookout.models import GazeEvent, GazePoint, LayoutSource

SOURCE = LayoutSource.ASSUMED_SHARED
ASSUMED = ("assumed_shared",)


def _event(viewer: str, target: str, start: float, end: float, reason: str = "center hit"):
    return GazeEvent(viewer, target, start, end, 0.8, SOURCE, reason=reason)


def _codes(events, points=None, sources=ASSUMED, thresholds=None, participants=None) -> set[str]:
    diagnostics = build_diagnostics(events, points or [], participants=participants)
    return {d.code for d in distribution_warnings(diagnostics, sources, thresholds)}


def test_normalized_entropy_spans_zero_to_one() -> None:
    assert normalized_entropy({"a": 1.0}) == 0.0
    assert normalized_entropy({"a": 1.0, "b": 1.0}) == 1.0
    assert normalized_entropy({"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}) == 1.0
    assert 0.0 < normalized_entropy({"a": 9.0, "b": 1.0}) < 1.0


def test_entropy_is_scaled_by_available_targets_not_observed_ones() -> None:
    """Scaling by the targets seen would call a nine-tile call that collapsed
    onto two of them evenly spread — the opposite of what a reader needs, since
    collapsing the field is itself the symptom."""

    balanced_pair = {"a": 5.0, "b": 5.0}
    assert normalized_entropy(balanced_pair, support=2) == 1.0
    assert normalized_entropy(balanced_pair, support=9) < 0.35

    full = {str(i): 5.0 for i in range(9)}
    assert normalized_entropy(full, support=9) == 1.0


def test_the_recorded_gallery_artifact_is_flagged() -> None:
    """The case this exists for: on the real 3-minute clip 88.7% of attributed
    duration pointed at one tile, and nothing said so."""

    events = [
        _event("slot_0", "slot_4", 0.0, 140.0),
        _event("slot_1", "slot_4", 0.0, 142.0),
        _event("slot_2", "slot_4", 0.0, 152.0),
        _event("slot_3", "slot_4", 0.0, 130.0),
        _event("slot_5", "slot_4", 0.0, 178.0),
        _event("slot_7", "slot_3", 0.0, 75.0),
    ]
    codes = _codes(events, participants=9)  # a 3x3 gallery
    assert "concentrated_targets" in codes
    assert "low_target_entropy" in codes

    diagnostics = build_diagnostics(events, [], participants=9)
    assert diagnostics.top_target == "slot_4"
    assert diagnostics.top_target_share > 0.88


def test_a_well_spread_result_is_not_flagged() -> None:
    events = [
        _event("slot_0", "slot_1", 0.0, 30.0),
        _event("slot_1", "slot_2", 0.0, 28.0),
        _event("slot_2", "slot_3", 0.0, 31.0),
        _event("slot_3", "slot_0", 0.0, 29.0),
    ]
    assert _codes(events, participants=4) == set()


def test_the_shared_layout_sharpens_the_concentration_warning() -> None:
    """Under a shared layout the dominant tile is the same screen position for
    everyone, which makes the coincidence likelier than the finding."""

    events = [_event(f"slot_{i}", "slot_4", 0.0, 100.0) for i in range(4)]

    (assumed,) = [
        d
        for d in distribution_warnings(build_diagnostics(events, []), ASSUMED)
        if d.code == "concentrated_targets"
    ]
    (known,) = [
        d
        for d in distribution_warnings(build_diagnostics(events, []), ("manifest",))
        if d.code == "concentrated_targets"
    ]
    assert "same screen position" in assumed.impact
    assert "same screen position" not in known.impact


def test_unresolved_outcomes_do_not_count_as_concentration() -> None:
    """A run that is mostly unknown is not concentrated, it is empty, and the
    funnel already reports that."""

    events = [
        _event("slot_0", "unknown", 0.0, 200.0),
        _event("slot_1", "slot_2", 0.0, 10.0),
        _event("slot_2", "slot_3", 0.0, 10.0),
    ]
    diagnostics = build_diagnostics(events, [])
    assert diagnostics.unresolved_share > 0.9
    assert diagnostics.top_target in {"slot_2", "slot_3"}
    assert "concentrated_targets" not in _codes(events)


def test_centre_clustered_gaze_is_flagged() -> None:
    centred = [GazePoint(i * 0.1, "slot_0", 0.5, 0.5, 0.9) for i in range(50)]
    spread = [
        GazePoint(i * 0.1, "slot_0", (i % 10) / 10.0, ((i // 10) % 10) / 10.0, 0.9)
        for i in range(50)
    ]
    assert "centre_clustered_gaze" in _codes([], centred)
    assert "centre_clustered_gaze" not in _codes([], spread)


def test_dominant_self_view_is_flagged() -> None:
    events = [
        _event("slot_0", "slot_0", 0.0, 100.0, reason="self"),
        _event("slot_1", "slot_2", 0.0, 20.0),
    ]
    assert "high_self_view" in _codes(events)


def test_thresholds_are_configurable() -> None:
    events = [
        _event("slot_0", "slot_1", 0.0, 55.0),
        _event("slot_1", "slot_2", 0.0, 45.0),
    ]
    assert "concentrated_targets" not in _codes(events)
    strict = DiagnosticThresholds(max_target_share=0.5)
    assert "concentrated_targets" in _codes(events, thresholds=strict)


def test_reasons_are_summarized_by_duration() -> None:
    events = [
        _event("slot_0", "slot_1", 0.0, 30.0, reason="center hit"),
        _event("slot_1", "low_confidence", 0.0, 10.0, reason="near border: slot_2"),
    ]
    diagnostics = build_diagnostics(events, [])
    assert diagnostics.reason_duration == {"center hit": 30.0, "near border: slot_2": 10.0}


def test_screen_histogram_bins_the_full_unit_square() -> None:
    corners = [
        GazePoint(0.0, "a", 0.0, 0.0, 0.5),
        GazePoint(1.0, "a", 1.0, 1.0, 0.5),
    ]
    histogram = screen_histogram(corners, bins=4)
    assert histogram[0][0] == 1
    assert histogram[3][3] == 1  # x=1.0 and y=1.0 clamp into the last bin
    assert sum(sum(row) for row in histogram) == 2


def test_empty_input_is_not_a_division_by_zero() -> None:
    diagnostics = build_diagnostics([], [])
    assert diagnostics.top_target is None
    assert diagnostics.top_target_share == 0.0
    assert diagnostics.normalized_entropy == 0.0
    assert diagnostics.centre_mass == 0.0
    assert not math.isnan(diagnostics.self_view_share)
    assert distribution_warnings(diagnostics, ASSUMED) == ()
