"""Face observation per tile.

A :class:`FaceObserver` looks at a single tile crop and returns a
:class:`FaceObservation` (landmarks, iris centres, head pose, eye blendshapes) or
``None`` when no face is visible. The MediaPipe adapter is the production
implementation; it is imported lazily so the core never depends on it.

Landmark index constants follow the MediaPipe 478-point mesh and are shared with
the geometric gaze baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .frames import Image
from .models import HeadPose

__all__ = [
    "Landmarks",
    "FaceObservation",
    "FaceObserver",
    "MediaPipeFaceObserver",
    "head_pose_from_matrix",
    "iris_center",
    "RIGHT_IRIS",
    "LEFT_IRIS",
    "RIGHT_EYE",
    "LEFT_EYE",
]

# (N, 3) normalized landmark coordinates within the crop.
Landmarks = NDArray[np.float32]

# Iris landmark indices (5 points each): centre plus four around the perimeter.
RIGHT_IRIS = (468, 469, 470, 471, 472)  # image-left eye (subject's right)
LEFT_IRIS = (473, 474, 475, 476, 477)  # image-right eye (subject's left)

# Eye corner and lid indices used by the geometric baseline.
RIGHT_EYE = {"inner": 133, "outer": 33, "top": 159, "bottom": 145}
LEFT_EYE = {"inner": 362, "outer": 263, "top": 386, "bottom": 374}


def iris_center(landmarks: Landmarks, indices: tuple[int, ...]) -> tuple[float, float]:
    """Return the mean ``(x, y)`` of the given landmark indices."""

    points = landmarks[list(indices), :2]
    center = points.mean(axis=0)
    return (float(center[0]), float(center[1]))


def head_pose_from_matrix(matrix: NDArray[np.float64]) -> HeadPose:
    """Derive head :class:`HeadPose` from a 4x4 facial transformation matrix.

    Uses the facing (canonical ``+Z``) and up (canonical ``+Y``) vectors so the
    result is stable and interpretable rather than a raw Euler decomposition.
    Signs invert the standard rotation about each axis (a right head turn gives
    ``+yaw``, an upward tilt gives ``+pitch``); the absolute up/down direction of
    the MediaPipe frame is calibrated against ground truth in #15.
    """

    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    forward = rotation @ np.array([0.0, 0.0, 1.0])
    up = rotation @ np.array([0.0, 1.0, 0.0])
    yaw = math.atan2(float(forward[0]), float(forward[2]))
    pitch = math.atan2(float(-forward[1]), math.hypot(float(forward[0]), float(forward[2])))
    roll = math.atan2(float(-up[0]), float(up[1]))
    return HeadPose(yaw=yaw, pitch=pitch, roll=roll)


@dataclass(frozen=True)
class FaceObservation:
    """What an observer sees in one tile crop.

    Coordinates are normalized to the crop. ``blendshapes`` holds MediaPipe eye
    coefficients (e.g. ``eyeLookInLeft``) when available.
    """

    landmarks: Landmarks
    head_pose: HeadPose
    blendshapes: dict[str, float]
    detection_confidence: float

    @property
    def right_iris(self) -> tuple[float, float]:
        return iris_center(self.landmarks, RIGHT_IRIS)

    @property
    def left_iris(self) -> tuple[float, float]:
        return iris_center(self.landmarks, LEFT_IRIS)


class FaceObserver(Protocol):
    """Observes a single tile crop."""

    def observe(self, image: Image) -> FaceObservation | None: ...


class MediaPipeFaceObserver:
    """Face landmarks, iris, head pose, and eye blendshapes via MediaPipe.

    Requires the ``face`` extra and a ``face_landmarker.task`` model file, which
    the user downloads explicitly; nothing is fetched automatically.
    """

    def __init__(self, model_path: str) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def observe(self, image: Image) -> FaceObservation | None:
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        points = result.face_landmarks[0]
        landmarks = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)

        if result.facial_transformation_matrixes:
            head_pose = head_pose_from_matrix(result.facial_transformation_matrixes[0])
        else:
            head_pose = HeadPose(0.0, 0.0, 0.0)

        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                if category.category_name.startswith("eyeLook"):
                    blendshapes[category.category_name] = float(category.score)

        return FaceObservation(landmarks, head_pose, blendshapes, detection_confidence=1.0)

    def close(self) -> None:
        self._landmarker.close()
