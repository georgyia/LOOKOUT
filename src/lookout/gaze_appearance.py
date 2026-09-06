"""Appearance-based gaze adapter.

Wraps a learned gaze model (L2CS-Net) behind a small :class:`GazeEstimator`
protocol so it is interchangeable with the geometric baseline and testable with
a fake. Torch and the model are imported lazily; the core never depends on them.

Licensing: the L2CS-Net code is MIT. The published Gaze360 weights are
research/non-commercial only, which is compatible with this project's licence.
Weights are provided by the user and never downloaded automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .frames import Image
from .models import GazeDirection, HeadPose

__all__ = ["GazeEstimator", "L2CSGazeEstimator", "appearance_gaze"]


class GazeEstimator(Protocol):
    """Estimates gaze ``(yaw, pitch)`` in radians from a face crop."""

    def estimate(self, crop: Image) -> tuple[float, float]: ...


class L2CSGazeEstimator:
    """L2CS-Net adapter (requires the ``appearance`` extra and user weights)."""

    def __init__(self, weights_path: str, arch: str = "ResNet50", device: str = "cpu") -> None:
        import torch
        from l2cs import Pipeline

        self._pipeline = Pipeline(
            weights=Path(weights_path),
            arch=arch,
            device=torch.device(device),
        )

    def estimate(self, crop: Image) -> tuple[float, float]:
        results = self._pipeline.step(crop)
        return float(results.yaw[0]), float(results.pitch[0])


def appearance_gaze(
    estimator: GazeEstimator,
    crop: Image,
    *,
    timestamp: float,
    person_id: str,
    head_pose: HeadPose | None = None,
    confidence: float = 0.8,
) -> GazeDirection:
    """Run an estimator on a crop and wrap the result as a :class:`GazeDirection`.

    ``head_pose`` is carried through for downstream calibration; the appearance
    model already accounts for head pose in its angle, so it is not added again.
    """

    yaw, pitch = estimator.estimate(crop)
    return GazeDirection(
        timestamp=timestamp,
        person_id=person_id,
        yaw=yaw,
        pitch=pitch,
        head_pose=head_pose or HeadPose(0.0, 0.0, 0.0),
        confidence=confidence,
    )
