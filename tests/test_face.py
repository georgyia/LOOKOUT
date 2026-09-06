import math

import numpy as np
import pytest

from lookout.face import FaceObservation, FaceObserver, head_pose_from_matrix, iris_center
from lookout.models import HeadPose


def _ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _matrix(rotation: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rotation
    return m


def test_head_pose_identity_is_zero() -> None:
    pose = head_pose_from_matrix(np.eye(4))
    assert pose.yaw == pytest.approx(0.0, abs=1e-9)
    assert pose.pitch == pytest.approx(0.0, abs=1e-9)
    assert pose.roll == pytest.approx(0.0, abs=1e-9)


def test_head_pose_recovers_yaw_and_pitch() -> None:
    yaw_pose = head_pose_from_matrix(_matrix(_ry(0.3)))
    assert yaw_pose.yaw == pytest.approx(0.3, abs=1e-6)
    assert yaw_pose.pitch == pytest.approx(0.0, abs=1e-6)

    pitch_pose = head_pose_from_matrix(_matrix(_rx(0.2)))
    assert pitch_pose.pitch == pytest.approx(0.2, abs=1e-6)
    assert pitch_pose.yaw == pytest.approx(0.0, abs=1e-6)


def test_iris_center_is_mean() -> None:
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[468:473, :2] = np.array(
        [[0.1, 0.2], [0.2, 0.2], [0.15, 0.1], [0.15, 0.3], [0.15, 0.2]]
    )
    assert iris_center(landmarks, (468, 469, 470, 471, 472)) == pytest.approx((0.15, 0.2))


def test_observation_iris_properties() -> None:
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[list(range(468, 473)), :2] = (0.3, 0.5)
    landmarks[list(range(473, 478)), :2] = (0.7, 0.5)
    obs = FaceObservation(landmarks, HeadPose(0, 0, 0), {}, 1.0)
    assert obs.right_iris == pytest.approx((0.3, 0.5))
    assert obs.left_iris == pytest.approx((0.7, 0.5))


class _AbsentFaceObserver:
    """Observer that never finds a face; used for the not-visible contract."""

    def observe(self, image: np.ndarray) -> FaceObservation | None:
        return None


def test_absent_face_returns_none() -> None:
    observer: FaceObserver = _AbsentFaceObserver()
    assert observer.observe(np.zeros((10, 10, 3), dtype=np.uint8)) is None
