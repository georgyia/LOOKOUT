"""Frame ingestion.

Decodes a video and subsamples it to a fixed rate, yielding frames with
timestamps. Raw frames are never written to disk.

Decoding (OpenCV) is isolated behind :func:`decode_video`; the subsampling
policy in :func:`sample_frames` is pure and testable without any video file.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = ["Image", "Frame", "decode_video", "sample_frames", "read_video"]

# HxWx3 uint8 image (OpenCV BGR order).
Image = NDArray[np.uint8]


@dataclass(frozen=True, order=True)
class Frame:
    """A sampled frame and its presentation timestamp in seconds."""

    timestamp: float
    image: Image = field(compare=False)

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")


def decode_video(path: str | Path) -> Iterator[tuple[float, Image]]:
    """Yield ``(timestamp_seconds, image)`` for every frame, in order.

    Timestamps come from the frame index and the stream frame rate, which is
    stable for constant-frame-rate video and avoids the backend-dependent
    behavior of ``CAP_PROP_POS_MSEC``.
    """

    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if fps > 0:
                timestamp = index / fps
            else:
                timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            yield timestamp, cast(Image, image)
            index += 1
    finally:
        capture.release()


def sample_frames(source: Iterable[tuple[float, Image]], target_fps: float) -> Iterator[Frame]:
    """Subsample a timestamped image stream to ``target_fps``.

    The stream is partitioned into buckets of width ``1 / target_fps`` seconds;
    the first frame seen in each bucket is emitted. This keeps output roughly
    uniform regardless of the source rate and never emits two frames from the
    same bucket.
    """

    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    interval = 1.0 / target_fps
    last_bucket = -1
    for timestamp, image in source:
        bucket = int(math.floor((timestamp + 1e-9) / interval))
        if bucket > last_bucket:
            last_bucket = bucket
            yield Frame(timestamp, image)


def read_video(path: str | Path, target_fps: float = 5.0) -> Iterator[Frame]:
    """Decode ``path`` and subsample to ``target_fps``.

    The default 5 fps is chosen so that typical fixations (about 200-400 ms) are
    each sampled at least once while keeping the frame count manageable.
    """

    return sample_frames(decode_video(path), target_fps)
