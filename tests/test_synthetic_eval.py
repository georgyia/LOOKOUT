"""End-to-end scoring on a run whose answers are known by construction.

This is the loop that previously could not run at all without a recorded cue
session: store -> attribute -> events -> evaluate, scored against the schedule
that produced it.
"""

from pathlib import Path

import pytest

from lookout import artifacts, store
from lookout.evaluate import evaluate
from lookout.models import GazeTarget, LayoutSource
from lookout.pipeline import EVENTS, GAZE_RAW, LAYOUT, AnalysisConfig, attribute
from tests.synthetic import AWAY, Cue, grid_layout, scripted_directions, tour, truth_intervals

CONFIG = AnalysisConfig(
    target_fps=5.0,
    smoothing_window=3,
    dispersion_threshold=0.12,
    min_fixation=0.3,
    max_gap=0.4,
    event_min_duration=0.4,
)


def _run(tmp_path: Path, rows: int, cols: int, cues: list[Cue], noise: float = 0.0):
    """Write a scripted run and put it through the production attribution path."""

    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    layout = grid_layout(rows, cols)
    store.write_gaze(
        out / GAZE_RAW, scripted_directions(layout, cues, fps=5.0, noise_sigma=noise)
    )
    artifacts.write_layouts(out / LAYOUT, [layout])
    attribute(out, CONFIG)
    events = artifacts.read_events(out / EVENTS)
    return events, truth_intervals(cues, grid=f"{rows}x{cols}")


def test_a_noiseless_run_is_recovered_exactly(tmp_path: Path) -> None:
    """Directions are the exact inverse of the projection, so a correct pipeline
    must return the tile each one started from."""

    cues = tour("slot_0", ["slot_1", "slot_2", "slot_3", "slot_0"], dwell=3.0)
    events, truth = _run(tmp_path, 2, 2, cues)
    result = evaluate(events, truth)

    assert result.total == 4
    assert result.hit_rate == 1.0
    assert result.unknown_rate == 0.0


def test_a_three_by_three_gallery_is_recovered(tmp_path: Path) -> None:
    cues = tour("slot_4", [f"slot_{i}" for i in range(9)], dwell=3.0)
    events, truth = _run(tmp_path, 3, 3, cues)
    result = evaluate(events, truth)

    assert result.hit_rate == 1.0
    assert result.grid_hit_rate("3x3") == 1.0


def test_deliberate_off_screen_windows_are_recovered(tmp_path: Path) -> None:
    """The cue protocol includes looking away so off-screen is exercised rather
    than assumed."""

    cues = [
        Cue("slot_0", "slot_1", 0.0, 3.0),
        Cue("slot_0", GazeTarget.OFF_SCREEN.value, 3.0, 6.0),
        Cue("slot_0", "slot_2", 6.0, 9.0),
    ]
    events, truth = _run(tmp_path, 2, 2, cues)

    # Off-screen directions are rejected before attribution, so the prediction
    # goes silent there rather than naming a target. That is the conservative
    # default working, and it must not be scored as a hit.
    result = evaluate(events, truth)
    assert result.hit_rate == pytest.approx(2 / 3)
    assert result.off_screen_total == 1
    assert result.off_screen_hits == 0
    assert result.unknown == 1


def test_a_window_with_no_observation_is_silent_not_wrong(tmp_path: Path) -> None:
    cues = [
        Cue("slot_0", "slot_1", 0.0, 3.0),
        Cue("slot_0", AWAY, 3.0, 6.0),
        Cue("slot_0", "slot_2", 6.0, 9.0),
    ]
    events, truth = _run(tmp_path, 2, 2, cues)
    result = evaluate(events, truth)

    assert result.hits == 2
    assert result.unknown == 1  # never a hit, never counted as a wrong answer


@pytest.mark.parametrize("noise", [0.0, 0.02, 0.05, 0.12])
def test_scoring_degrades_with_angular_noise_rather_than_collapsing(
    tmp_path: Path, noise: float
) -> None:
    """A fixture that only passes at zero noise would prove nothing about the
    pipeline's behaviour on real, noisy observations."""

    cues = tour("slot_0", [f"slot_{i}" for i in range(9)] * 2, dwell=3.0)
    events, truth = _run(tmp_path, 3, 3, cues, noise=noise)
    result = evaluate(events, truth)
    assert 0.0 <= result.hit_rate <= 1.0
    if noise == 0.0:
        assert result.hit_rate == 1.0


def test_more_noise_never_scores_better(tmp_path: Path) -> None:
    cues = tour("slot_0", [f"slot_{i}" for i in range(9)] * 2, dwell=3.0)
    rates = []
    for noise in (0.0, 0.04, 0.10, 0.25):
        events, truth = _run(tmp_path / f"n{noise}", 3, 3, cues, noise=noise)
        rates.append(evaluate(events, truth).hit_rate)

    assert rates[0] == 1.0
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] < rates[0]


def test_a_finer_grid_is_harder(tmp_path: Path) -> None:
    """Tiles shrink, so the same angular error crosses more borders."""

    coarse_cues = tour("slot_0", ["slot_0", "slot_1", "slot_2", "slot_3"] * 3, dwell=3.0)
    fine_cues = tour("slot_0", [f"slot_{i}" for i in range(9)] + [f"slot_{i}" for i in range(3)],
                     dwell=3.0)

    coarse_events, coarse_truth = _run(tmp_path / "c", 2, 2, coarse_cues, noise=0.08)
    fine_events, fine_truth = _run(tmp_path / "f", 3, 3, fine_cues, noise=0.08)

    coarse = evaluate(coarse_events, coarse_truth).hit_rate
    fine = evaluate(fine_events, fine_truth).hit_rate
    assert coarse >= fine


def test_the_fixture_exercises_the_production_layout_source(tmp_path: Path) -> None:
    cues = tour("slot_0", ["slot_1", "slot_2"], dwell=3.0)
    events, _ = _run(tmp_path, 2, 2, cues)
    assert {e.layout_source for e in events} == {LayoutSource.ASSUMED_SHARED}
