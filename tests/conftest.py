"""Shared test fixtures: synthetic face observations without any model."""

from __future__ import annotations

import numpy as np

from lookout.face import (
    LEFT_EYE,
    LEFT_IRIS,
    RIGHT_EYE,
    RIGHT_IRIS,
    FaceObservation,
)
from lookout.models import HeadPose


def make_observation(
    *,
    h_shift: float = 0.0,
    v_shift: float = 0.0,
    head_pose: HeadPose | None = None,
    left_center: tuple[float, float] = (0.35, 0.5),
    right_center: tuple[float, float] = (0.65, 0.5),
    half_w: float = 0.05,
    half_h: float = 0.05,
    detection_confidence: float = 1.0,
) -> FaceObservation:
    """Build a FaceObservation with controllable iris offsets.

    ``h_shift`` and ``v_shift`` are the desired per-eye offsets in ``[-1, 1]``:
    +h moves the iris toward larger image x, +v moves it up.
    ``left_center`` is the image-left eye (subject's right, RIGHT_* indices);
    ``right_center`` is the image-right eye (subject's left, LEFT_* indices).
    """

    landmarks = np.zeros((478, 3), dtype=np.float32)

    def place(center: tuple[float, float], corners: dict[str, int], iris: tuple[int, ...]) -> None:
        cx, cy = center
        landmarks[corners["inner"], :2] = (cx + half_w, cy)
        landmarks[corners["outer"], :2] = (cx - half_w, cy)
        landmarks[corners["top"], :2] = (cx, cy - half_h)
        landmarks[corners["bottom"], :2] = (cx, cy + half_h)
        iris_x = cx + h_shift * half_w
        iris_y = cy - v_shift * half_h
        for idx in iris:
            landmarks[idx, :2] = (iris_x, iris_y)

    place(left_center, RIGHT_EYE, RIGHT_IRIS)
    place(right_center, LEFT_EYE, LEFT_IRIS)

    return FaceObservation(
        landmarks=landmarks,
        head_pose=head_pose or HeadPose(0.0, 0.0, 0.0),
        blendshapes={},
        detection_confidence=detection_confidence,
    )
