import pytest

from lookout.models import GazePoint
from lookout.temporal import detect_fixations, median_smooth


def _points(coords: list[tuple[float, float, float]], person: str = "bob") -> list[GazePoint]:
    return [GazePoint(t, person, x, y, 0.9) for t, x, y in coords]


def test_median_smooth_removes_single_sample_spike() -> None:
    pts = _points([(0.0, 0.5, 0.5), (0.1, 0.9, 0.5), (0.2, 0.5, 0.5)])
    smoothed = median_smooth(pts, window=3)
    assert smoothed[1].x == pytest.approx(0.5)  # spike replaced by median
    assert [p.timestamp for p in smoothed] == [0.0, 0.1, 0.2]


def test_median_smooth_requires_odd_window() -> None:
    with pytest.raises(ValueError):
        median_smooth(_points([(0.0, 0.5, 0.5)]), window=2)


def test_detect_single_fixation() -> None:
    coords = [(round(0.1 * k, 1), 0.5 + 0.005 * (k % 2), 0.5) for k in range(6)]
    fixations = detect_fixations(_points(coords))
    assert len(fixations) == 1
    fix = fixations[0]
    assert fix.x == pytest.approx(0.5, abs=0.01)
    assert fix.start_time == pytest.approx(0.0)
    assert fix.end_time == pytest.approx(0.5)


def test_detect_two_fixations_with_a_saccade_between() -> None:
    first = [(round(0.1 * k, 1), 0.2, 0.2) for k in range(4)]  # 0.0..0.3
    # a saccade sample far away and short, then a second dwell
    saccade = [(0.4, 0.6, 0.6)]
    second = [(round(0.1 * k, 1), 0.8, 0.8) for k in range(5, 9)]  # 0.5..0.8
    fixations = detect_fixations(_points(first + saccade + second))
    assert len(fixations) == 2
    assert fixations[0].x == pytest.approx(0.2, abs=0.01)
    assert fixations[1].x == pytest.approx(0.8, abs=0.01)


def test_gap_breaks_a_fixation() -> None:
    # Two dwells at the same location; a ~1 s gap between them exceeds max_gap.
    first = [(0.0, 0.5, 0.5), (0.1, 0.5, 0.5), (0.2, 0.5, 0.5)]
    second = [(1.2, 0.5, 0.5), (1.3, 0.5, 0.5), (1.4, 0.5, 0.5)]
    fixations = detect_fixations(_points(first + second), min_duration=0.15, max_gap=0.3)
    assert len(fixations) == 2
