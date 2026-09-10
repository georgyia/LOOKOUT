import json
from pathlib import Path

import pytest

from lookout.evaluate import TruthInterval, evaluate, load_truth, predicted_target_at
from lookout.models import GazeEvent, LayoutSource

SOURCE = LayoutSource.ASSUMED_SHARED


def _event(target: str, start: float, end: float) -> GazeEvent:
    return GazeEvent("v", target, start, end, 0.9, SOURCE)


def _truth(target: str, start: float, end: float, grid: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "viewer_id": "v",
        "start_time": start,
        "end_time": end,
        "target": target,
    }
    if grid is not None:
        row["grid"] = grid
    return row


def test_predicted_target_at_reads_active_event() -> None:
    events = [_event("alice", 0.0, 2.0)]
    assert predicted_target_at(events, "v", 1.0) == "alice"
    assert predicted_target_at(events, "v", 2.0) is None


def test_evaluate_scores_hits_unknown_and_off_screen(tmp_path: Path) -> None:
    events = [
        _event("alice", 0.0, 2.0),
        _event("unknown", 2.0, 4.0),
        _event("off_screen", 5.0, 6.0),
    ]
    truth_path = tmp_path / "truth.jsonl"
    rows = [
        _truth("alice", 0.5, 1.5, "2x2"),  # hit
        _truth("carol", 2.0, 3.0, "3x3"),  # predicted unknown -> not a hit
        _truth("alice", 4.0, 5.0, "2x2"),  # no event -> unknown
        _truth("off_screen", 5.0, 6.0),  # hit, off-screen
    ]
    truth_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    result = evaluate(events, load_truth(truth_path))

    assert result.total == 4
    assert result.hit_rate == pytest.approx(0.5)
    assert result.unknown_rate == pytest.approx(0.5)
    assert result.off_screen_recall == pytest.approx(1.0)
    assert result.grid_hit_rate("2x2") == pytest.approx(0.5)
    assert result.grid_hit_rate("3x3") == pytest.approx(0.0)


def test_silent_prediction_never_counts_as_a_hit() -> None:
    result = evaluate([], [TruthInterval("v", 0.0, 1.0, "alice")])
    assert result.hits == 0
    assert result.unknown == 1


# --------------------------------------------------------------- evaluation v2


def _intervals(pairs: list[tuple[str, float, float]], grid: str = "3x3") -> list[TruthInterval]:
    return [TruthInterval("bob", start, end, target, grid) for target, start, end in pairs]


def _predict(pairs: list[tuple[str, float, float]], confidence: float = 0.8) -> list[GazeEvent]:
    return [
        GazeEvent("bob", target, start, end, confidence, LayoutSource.ASSUMED_SHARED)
        for target, start, end in pairs
    ]


def test_a_hit_rate_is_reported_against_what_guessing_would_score() -> None:
    """On a 3x3 grid uniform guessing scores about 11%; a hit rate quoted
    without that is unreadable."""

    truth = _intervals([(f"slot_{i}", i * 2.0, i * 2.0 + 2.0) for i in range(9)])
    events = _predict([(f"slot_{i}", i * 2.0, i * 2.0 + 2.0) for i in range(9)])
    result = evaluate(events, truth)

    assert result.hit_rate == 1.0
    assert result.baselines["uniform_random"] == pytest.approx(1 / 9)
    assert result.baselines["always_unknown"] == 0.0
    assert result.beats_baseline


def test_an_estimator_that_always_says_one_target_is_caught() -> None:
    """The failure mode observed on the real gallery clip: a constant answer can
    post a respectable hit rate while carrying no per-interval discrimination."""

    truth = _intervals(
        [("slot_4", 0.0, 2.0), ("slot_4", 2.0, 4.0), ("slot_4", 4.0, 6.0), ("slot_1", 6.0, 8.0)]
    )
    events = _predict([("slot_4", 0.0, 8.0)])
    result = evaluate(events, truth)

    assert result.hit_rate == 0.75  # looks respectable
    assert result.baselines["model_modal_target"] == 0.75  # and is exactly the constant
    assert not result.beats_baseline


def test_the_confidence_interval_narrows_as_evidence_accumulates() -> None:
    """A rate over 12 cues must not read like one over 1200."""

    def width(n: int) -> float:
        truth = _intervals([("slot_0", i * 2.0, i * 2.0 + 2.0) for i in range(n)])
        events = _predict([("slot_0", 0.0, n * 2.0)])
        lo, hi = evaluate(events, truth).hit_rate_ci
        return hi - lo

    mixed_small = width(8)
    mixed_large = width(200)
    assert mixed_large <= mixed_small


def test_declining_to_answer_is_separated_from_being_silent() -> None:
    """Silence, an explicit unknown and a border rejection are different
    failures and were previously collapsed together."""

    truth = _intervals([("slot_0", 0.0, 2.0), ("slot_1", 2.0, 4.0), ("slot_2", 4.0, 6.0)])
    events = _predict([("unknown", 2.0, 4.0), ("low_confidence", 4.0, 6.0)])
    result = evaluate(events, truth)

    assert result.silent == 1
    assert result.explicit_unknown == 1
    assert result.low_confidence == 1
    assert result.unknown == 3
    assert result.wrong == 0  # declining is never counted as a wrong answer


def test_a_wrong_answer_is_distinguished_from_no_answer() -> None:
    truth = _intervals([("slot_0", 0.0, 2.0), ("slot_1", 2.0, 4.0)])
    events = _predict([("slot_5", 0.0, 2.0)])
    result = evaluate(events, truth)

    assert result.wrong == 1
    assert result.silent == 1
    assert result.hits == 0


def test_the_confusion_matrix_shows_where_the_errors_go() -> None:
    truth = _intervals([("slot_0", 0.0, 2.0), ("slot_0", 2.0, 4.0), ("slot_1", 4.0, 6.0)])
    events = _predict([("slot_1", 0.0, 4.0), ("slot_1", 4.0, 6.0)])
    result = evaluate(events, truth)

    assert result.confusion["slot_0"] == {"slot_1": 2}
    assert result.confusion["slot_1"] == {"slot_1": 1}


def test_confidence_is_checked_against_how_often_it_was_right() -> None:
    """Confidence gates attribution but had never been checked against outcomes."""

    truth = _intervals([("slot_0", 0.0, 2.0), ("slot_1", 2.0, 4.0)])
    events = [
        GazeEvent("bob", "slot_0", 0.0, 2.0, 0.9, LayoutSource.ASSUMED_SHARED),
        GazeEvent("bob", "slot_9", 2.0, 4.0, 0.9, LayoutSource.ASSUMED_SHARED),
    ]
    result = evaluate(events, truth)

    (band,) = [b for b in result.reliability if b.count == 2]
    assert band.mean_confidence == 0.9
    assert band.hit_rate == 0.5
    # Claiming 0.9 while being right half the time is a calibration error of 0.4.
    assert result.expected_calibration_error == pytest.approx(0.4)


def test_a_perfectly_calibrated_run_has_no_calibration_error() -> None:
    truth = _intervals([("slot_0", 0.0, 2.0), ("slot_1", 2.0, 4.0)])
    events = [
        GazeEvent("bob", "slot_0", 0.0, 2.0, 1.0, LayoutSource.ASSUMED_SHARED),
        GazeEvent("bob", "slot_1", 2.0, 4.0, 1.0, LayoutSource.ASSUMED_SHARED),
    ]
    assert evaluate(events, truth).expected_calibration_error == 0.0


def test_dense_sampling_weights_a_long_window_more_than_a_short_one() -> None:
    """At the midpoint a 30-second window counts the same as a one-second one,
    and a prediction that held throughout is indistinguishable from one that
    happened to be right in the middle."""

    truth = _intervals([("slot_0", 0.0, 30.0), ("slot_1", 30.0, 31.0)])
    events = _predict([("slot_0", 0.0, 30.0), ("slot_1", 30.0, 31.0)])

    midpoint = evaluate(events, truth)
    dense = evaluate(events, truth, sample_step=1.0)

    assert midpoint.total == 2
    assert dense.total > 25
    assert dense.hit_rate == midpoint.hit_rate == 1.0


def test_dense_sampling_exposes_a_prediction_that_only_held_briefly() -> None:
    truth = _intervals([("slot_0", 0.0, 10.0)])
    events = _predict([("slot_0", 4.5, 5.5)])  # right at the midpoint, nowhere else

    assert evaluate(events, truth).hit_rate == 1.0
    assert evaluate(events, truth, sample_step=1.0).hit_rate < 0.2


def test_dense_and_midpoint_agree_on_uniform_windows() -> None:
    truth = _intervals([(f"slot_{i}", i * 2.0, i * 2.0 + 2.0) for i in range(6)])
    events = _predict([(f"slot_{i}", i * 2.0, i * 2.0 + 2.0) for i in range(6)])
    assert evaluate(events, truth).hit_rate == evaluate(
        events, truth, sample_step=0.5
    ).hit_rate


def test_the_candidate_count_can_be_stated_rather_than_inferred() -> None:
    """Ground truth naming two tiles does not mean the call had two tiles."""

    truth = _intervals([("slot_0", 0.0, 2.0), ("slot_1", 2.0, 4.0)])
    events = _predict([("slot_0", 0.0, 2.0), ("slot_1", 2.0, 4.0)])

    assert evaluate(events, truth).baselines["uniform_random"] == 0.5
    assert evaluate(events, truth, candidates=9).baselines["uniform_random"] == pytest.approx(
        1 / 9
    )


def test_empty_truth_does_not_divide_by_zero() -> None:
    result = evaluate([], [])
    assert result.total == 0
    assert result.hit_rate == 0.0
    assert result.baselines == {}
    assert result.hit_rate_ci == (0.0, 0.0)
    assert result.reliability == ()
