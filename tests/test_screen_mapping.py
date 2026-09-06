import math

import pytest

from lookout.models import GazeDirection, GazePoint, GazeTarget, HeadPose
from lookout.screen_mapping import ScreenMappingParams, map_direction, project

PARAMS = ScreenMappingParams()
POSE = HeadPose(0.0, 0.0, 0.0)


def test_center_and_edges_map_as_expected() -> None:
    # Yaw 0 is horizontal centre; the vertical centre is midway between the
    # top and bottom pitches.
    mid_pitch = (PARAMS.pitch_at_top + PARAMS.pitch_at_bottom) / 2
    x, y = project(0.0, mid_pitch, PARAMS)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.5)

    left = project(PARAMS.yaw_at_left, PARAMS.pitch_at_top, PARAMS)
    assert left == pytest.approx((0.0, 0.0))
    right = project(PARAMS.yaw_at_right, PARAMS.pitch_at_bottom, PARAMS)
    assert right == pytest.approx((1.0, 1.0))


def test_positive_yaw_moves_right_positive_pitch_moves_up() -> None:
    center_x, _ = project(0.0, PARAMS.pitch_at_bottom / 2, PARAMS)
    right_x, _ = project(math.radians(8), PARAMS.pitch_at_bottom / 2, PARAMS)
    assert right_x > center_x

    _, low_y = project(0.0, math.radians(-10), PARAMS)
    _, high_y = project(0.0, math.radians(-2), PARAMS)
    assert high_y < low_y  # looking up -> smaller y


def test_off_screen_beyond_margin() -> None:
    assert project(math.radians(40), 0.0, PARAMS) is None
    assert project(0.0, math.radians(30), PARAMS) is None


def test_within_margin_is_clamped() -> None:
    result = project(PARAMS.yaw_at_right + math.radians(0.5), PARAMS.pitch_at_top, PARAMS)
    assert result is not None
    assert result[0] == pytest.approx(1.0)


def test_map_direction_returns_point_or_off_screen() -> None:
    on = GazeDirection(1.0, "bob", 0.0, math.radians(-9), POSE, 0.9)
    result = map_direction(on)
    assert isinstance(result, GazePoint)
    assert result.confidence == pytest.approx(0.9)

    off = GazeDirection(1.0, "bob", math.radians(40), 0.0, POSE, 0.9)
    assert map_direction(off) is GazeTarget.OFF_SCREEN
