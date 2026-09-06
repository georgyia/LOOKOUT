import numpy as np
import pytest

from lookout.gaze_appearance import GazeEstimator, appearance_gaze
from lookout.models import HeadPose


class _FakeEstimator:
    def __init__(self, yaw: float, pitch: float) -> None:
        self._yaw = yaw
        self._pitch = pitch

    def estimate(self, crop: np.ndarray) -> tuple[float, float]:
        return self._yaw, self._pitch


def test_appearance_gaze_wraps_estimator_output() -> None:
    estimator: GazeEstimator = _FakeEstimator(0.15, -0.05)
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    gaze = appearance_gaze(
        estimator,
        crop,
        timestamp=2.0,
        person_id="alice",
        head_pose=HeadPose(0.1, 0.0, 0.0),
        confidence=0.7,
    )
    assert gaze.yaw == pytest.approx(0.15)
    assert gaze.pitch == pytest.approx(-0.05)
    assert gaze.head_pose == HeadPose(0.1, 0.0, 0.0)
    assert gaze.confidence == pytest.approx(0.7)
    assert gaze.person_id == "alice"


def test_appearance_gaze_defaults_head_pose() -> None:
    gaze = appearance_gaze(
        _FakeEstimator(0.0, 0.0),
        np.zeros((8, 8, 3), dtype=np.uint8),
        timestamp=0.0,
        person_id="bob",
    )
    assert gaze.head_pose == HeadPose(0.0, 0.0, 0.0)
