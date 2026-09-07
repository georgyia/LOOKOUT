import pytest

from lookout.attribution import resolve_gaze
from lookout.models import GazePoint, ScreenRegion


def test_gaze_resolves_to_participant() -> None:
    gaze = GazePoint(10.0, "bob", 0.82, 0.17, 0.9)
    regions = [ScreenRegion("alice", 0.75, 0.0, 0.25, 0.25, 0.0)]

    assert resolve_gaze(gaze, regions) == "alice"


def test_gaze_remains_unknown_when_no_region_matches() -> None:
    gaze = GazePoint(10.0, "bob", 0.2, 0.8, 0.9)
    regions = [ScreenRegion("alice", 0.75, 0.0, 0.25, 0.25, 0.0)]

    assert resolve_gaze(gaze, regions) == "unknown"


def test_region_can_change_over_time() -> None:
    gaze = GazePoint(20.0, "bob", 0.8, 0.1, 0.9)
    regions = [
        ScreenRegion("alice", 0.75, 0.0, 0.25, 0.25, 0.0, 15.0),
        ScreenRegion("charlie", 0.75, 0.0, 0.25, 0.25, 15.0),
    ]

    assert resolve_gaze(gaze, regions) == "charlie"


@pytest.mark.parametrize(
    "x,y",
    [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)],
)
def test_gaze_coordinates_are_normalized(x: float, y: float) -> None:
    with pytest.raises(ValueError):
        GazePoint(0.0, "bob", x, y, 0.5)
