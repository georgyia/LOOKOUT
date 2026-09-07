import math

import pytest

from lookout.gaze_geometric import GeometricGazeParams, estimate_gaze, eye_offsets
from lookout.models import HeadPose
from tests.conftest import make_observation

PARAMS = GeometricGazeParams()


def test_centered_iris_yields_head_angle_only() -> None:
    obs = make_observation(head_pose=HeadPose(0.2, -0.1, 0.0))
    gaze = estimate_gaze(obs, timestamp=1.0, person_id="bob")
    assert gaze.yaw == pytest.approx(0.2, abs=1e-6)
    assert gaze.pitch == pytest.approx(-0.1, abs=1e-6)


def test_horizontal_iris_offset_adds_signed_yaw() -> None:
    h, v, _ = eye_offsets(make_observation(h_shift=0.5))
    assert h == pytest.approx(0.5)
    assert v == pytest.approx(0.0)

    gaze = estimate_gaze(make_observation(h_shift=0.5), timestamp=0.0, person_id="bob")
    assert gaze.yaw == pytest.approx(0.5 * PARAMS.max_eye_yaw, abs=1e-6)

    left = estimate_gaze(make_observation(h_shift=-0.5), timestamp=0.0, person_id="bob")
    assert left.yaw < 0 < gaze.yaw


def test_vertical_iris_offset_adds_positive_pitch_when_looking_up() -> None:
    gaze = estimate_gaze(make_observation(v_shift=0.5), timestamp=0.0, person_id="bob")
    assert gaze.pitch == pytest.approx(0.5 * PARAMS.max_eye_pitch, abs=1e-6)


def test_confidence_is_full_for_a_clear_frontal_face() -> None:
    gaze = estimate_gaze(make_observation(), timestamp=0.0, person_id="bob")
    assert gaze.confidence == pytest.approx(1.0)


def test_confidence_drops_for_near_closed_eyes() -> None:
    open_face = estimate_gaze(make_observation(half_h=0.05), timestamp=0.0, person_id="bob")
    closed_face = estimate_gaze(make_observation(half_h=0.004), timestamp=0.0, person_id="bob")
    assert closed_face.confidence < open_face.confidence
    assert closed_face.confidence == pytest.approx(0.0, abs=1e-6)


def test_confidence_drops_for_a_small_face() -> None:
    big = estimate_gaze(make_observation(), timestamp=0.0, person_id="bob")
    small = estimate_gaze(
        make_observation(left_center=(0.47, 0.5), right_center=(0.53, 0.5)),
        timestamp=0.0,
        person_id="bob",
    )
    assert small.confidence < big.confidence
    assert 0.0 <= small.confidence <= 1.0


def test_confidence_drops_for_extreme_head_pose() -> None:
    frontal = estimate_gaze(make_observation(), timestamp=0.0, person_id="bob")
    turned = estimate_gaze(
        make_observation(head_pose=HeadPose(math.radians(60), 0.0, 0.0)),
        timestamp=0.0,
        person_id="bob",
    )
    assert turned.confidence < frontal.confidence
