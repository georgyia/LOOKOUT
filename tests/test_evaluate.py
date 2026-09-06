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
