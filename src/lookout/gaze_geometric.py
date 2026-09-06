"""Geometric gaze baseline (model-free).

Combines the iris offset within each eye with the head pose to produce a gaze
angle, with a confidence derived from face size, eye openness, and head-pose
extremity. It is interpretable and dependency-free, and serves as the baseline
every learned estimator must beat.

Sign convention (see ``models.py``): positive yaw -> larger screen x, positive
pitch -> up. The horizontal/vertical signs are parameters so the eventual
per-viewer calibration (#18) can correct for mirroring without touching this
code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .face import LEFT_EYE, LEFT_IRIS, RIGHT_EYE, RIGHT_IRIS, FaceObservation, iris_center
from .models import GazeDirection

__all__ = ["GeometricGazeParams", "eye_offsets", "estimate_gaze"]

_EPS = 1e-6


@dataclass(frozen=True)
class GeometricGazeParams:
    """Tunable constants for the geometric baseline."""

    max_eye_yaw: float = math.radians(25.0)
    max_eye_pitch: float = math.radians(20.0)
    horizontal_sign: float = 1.0
    vertical_sign: float = 1.0
    reference_iod: float = 0.25
    min_openness: float = 0.15
    full_openness: float = 0.40


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _single_eye(
    landmarks: np.ndarray,
    corners: dict[str, int],
    iris_indices: tuple[int, ...],
) -> tuple[float, float, float]:
    """Return ``(h_offset, v_offset, openness)`` for one eye.

    ``h_offset`` in ``[-1, 1]`` is the iris displacement toward larger image x;
    ``v_offset`` is positive when the iris is above the eye centre (looking up);
    ``openness`` is the eye height/width ratio.
    """

    inner = landmarks[corners["inner"], :2]
    outer = landmarks[corners["outer"], :2]
    top = landmarks[corners["top"], :2]
    bottom = landmarks[corners["bottom"], :2]
    iris = np.array(iris_center(landmarks, iris_indices))

    center_x = (float(inner[0]) + float(outer[0])) / 2.0
    center_y = (float(top[1]) + float(bottom[1])) / 2.0
    half_w = abs(float(outer[0]) - float(inner[0])) / 2.0
    half_h = abs(float(bottom[1]) - float(top[1])) / 2.0

    h = (float(iris[0]) - center_x) / half_w if half_w > _EPS else 0.0
    v = (center_y - float(iris[1])) / half_h if half_h > _EPS else 0.0
    openness = (2.0 * half_h) / (2.0 * half_w) if half_w > _EPS else 0.0
    return _clamp(h, -1.0, 1.0), _clamp(v, -1.0, 1.0), openness


def eye_offsets(observation: FaceObservation) -> tuple[float, float, float]:
    """Averaged ``(h_offset, v_offset, openness)`` across both eyes."""

    lm = observation.landmarks
    h_left, v_left, open_left = _single_eye(lm, RIGHT_EYE, RIGHT_IRIS)
    h_right, v_right, open_right = _single_eye(lm, LEFT_EYE, LEFT_IRIS)
    return (
        (h_left + h_right) / 2.0,
        (v_left + v_right) / 2.0,
        (open_left + open_right) / 2.0,
    )


def _confidence(
    observation: FaceObservation,
    openness: float,
    params: GeometricGazeParams,
) -> float:
    right = np.array(observation.right_iris)
    left = np.array(observation.left_iris)
    iod = float(np.linalg.norm(right - left))
    size_factor = _clamp(iod / params.reference_iod, 0.0, 1.0)

    span = max(params.full_openness - params.min_openness, _EPS)
    open_factor = _clamp((openness - params.min_openness) / span, 0.0, 1.0)

    extremity = max(abs(observation.head_pose.yaw), abs(observation.head_pose.pitch))
    head_factor = _clamp(1.0 - extremity / (math.pi / 2.0), 0.0, 1.0)

    return _clamp(
        observation.detection_confidence * size_factor * open_factor * head_factor, 0.0, 1.0
    )


def estimate_gaze(
    observation: FaceObservation,
    *,
    timestamp: float,
    person_id: str,
    params: GeometricGazeParams | None = None,
) -> GazeDirection:
    """Estimate a :class:`GazeDirection` from a face observation."""

    params = params or GeometricGazeParams()
    h_offset, v_offset, openness = eye_offsets(observation)

    eye_yaw = params.horizontal_sign * h_offset * params.max_eye_yaw
    eye_pitch = params.vertical_sign * v_offset * params.max_eye_pitch
    yaw = observation.head_pose.yaw + eye_yaw
    pitch = observation.head_pose.pitch + eye_pitch

    confidence = _confidence(observation, openness, params)
    return GazeDirection(timestamp, person_id, yaw, pitch, observation.head_pose, confidence)
