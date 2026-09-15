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
    "YuNetFaceObserver",
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

        # The detector's own score when it exposes one. Reporting a constant 1.0
        # made every observation look equally well founded.
        scores = [
            float(detection.score)
            for detection in getattr(result, "face_detections", None) or []
            if getattr(detection, "score", None) is not None
        ]
        detection_confidence = max(scores) if scores else 1.0

        if result.facial_transformation_matrixes:
            head_pose = head_pose_from_matrix(result.facial_transformation_matrixes[0])
        else:
            head_pose = HeadPose(0.0, 0.0, 0.0)

        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                if category.category_name.startswith("eyeLook"):
                    blendshapes[category.category_name] = float(category.score)

        return FaceObservation(
            landmarks, head_pose, blendshapes, detection_confidence=detection_confidence
        )

    def close(self) -> None:
        self._landmarker.close()


# Frontal faces sit with the nose a little below the eye line; subtracting that
# bias puts "looking at the camera" near zero. The scale factors turn landmark
# offsets, measured in interocular distances, into angles.
_FRONTAL_NOSE_DROP = 0.55
_YAW_PER_IOD = 0.45
_PITCH_PER_IOD = 0.35
_PITCH_PRIOR = -0.16


class YuNetFaceObserver:
    """Face observation from OpenCV's YuNet detector.

    Requires the ``cv`` extra and a YuNet ONNX model, which the user supplies.

    YuNet gives a box, a score and five landmarks: both eyes, the nose tip and
    the mouth corners. There is no iris, so this observer cannot see where the
    eyes point — only where the head does. It reports the eyes as open and
    centred, because zero is the only eye contribution it can justify, and the
    geometric estimator therefore returns head orientation alone.

    That is a real limitation rather than a tuning issue, and a run using this
    observer says so: see :func:`describe`. It exists because MediaPipe's Face
    Landmarker does not run everywhere, and a machine without it otherwise has
    no path through the pipeline at all.
    """

    role = "face"

    def __init__(self, model_path: str, score_threshold: float = 0.55) -> None:
        import cv2

        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 5000
        )

    def observe(self, image: Image) -> FaceObservation | None:
        height, width = image.shape[:2]
        if height < 2 or width < 2:
            return None

        self._detector.setInputSize((width, height))
        _, detections = self._detector.detect(image)
        if detections is None or len(detections) == 0:
            return None

        face = max(detections, key=lambda row: float(row[2] * row[3]))
        return self._observation(face, width, height)

    def _observation(
        self, face: NDArray[np.float32], width: int, height: int
    ) -> FaceObservation:
        score = float(face[14])
        right_eye = (float(face[4]), float(face[5]))
        left_eye = (float(face[6]), float(face[7]))
        nose = (float(face[8]), float(face[9]))

        eye_mid_x = 0.5 * (right_eye[0] + left_eye[0])
        eye_mid_y = 0.5 * (right_eye[1] + left_eye[1])
        iod = max(abs(left_eye[0] - right_eye[0]), 1.0)

        horizontal = (nose[0] - eye_mid_x) / iod
        vertical = (nose[1] - eye_mid_y) / iod - _FRONTAL_NOSE_DROP

        yaw = max(-0.35, min(0.35, horizontal * _YAW_PER_IOD))
        pitch = max(-0.40, min(0.05, _PITCH_PRIOR - vertical * _PITCH_PER_IOD))
        roll = math.atan2(left_eye[1] - right_eye[1], max(left_eye[0] - right_eye[0], 1.0))

        landmarks = _synthetic_landmarks(right_eye, left_eye, width, height)
        return FaceObservation(
            landmarks=landmarks,
            head_pose=HeadPose(yaw=yaw, pitch=pitch, roll=roll),
            blendshapes={},
            detection_confidence=max(0.0, min(1.0, score)),
        )

    def describe(self) -> str:
        return "yunet_head_pose"


def _synthetic_landmarks(
    right_eye: tuple[float, float],
    left_eye: tuple[float, float],
    width: int,
    height: int,
) -> Landmarks:
    """Place the landmarks the geometric estimator reads, and nothing else.

    The eye corners are positioned around the detected eye centres so the
    interocular distance is real, and each iris is placed at the centre of its
    own eye so the eye contribution to gaze is exactly zero. Anything else would
    be inventing an eye direction from data that does not contain one.
    """

    landmarks = np.zeros((478, 3), dtype=np.float32)
    # A plausible open-eye aspect: the geometric estimator reads openness from
    # the lid box, and a squint here would understate confidence for a reason
    # this observer cannot actually see.
    half_w = max(abs(left_eye[0] - right_eye[0]) / 6.0, 1.0)
    half_h = half_w * 0.45

    for centre, corners, iris in (
        (right_eye, RIGHT_EYE, RIGHT_IRIS),
        (left_eye, LEFT_EYE, LEFT_IRIS),
    ):
        cx, cy = centre[0] / width, centre[1] / height
        dx, dy = half_w / width, half_h / height
        landmarks[corners["inner"], :2] = (cx + dx, cy)
        landmarks[corners["outer"], :2] = (cx - dx, cy)
        landmarks[corners["top"], :2] = (cx, cy - dy)
        landmarks[corners["bottom"], :2] = (cx, cy + dy)
        for index in iris:
            landmarks[index, :2] = (cx, cy)  # centred: no eye contribution
    return landmarks
