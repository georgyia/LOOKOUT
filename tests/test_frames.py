from pathlib import Path

import numpy as np
import pytest

from lookout.frames import Frame, read_video, sample_frames


def _synthetic_source(rate: float, count: int) -> list[tuple[float, np.ndarray]]:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return [(i / rate, image) for i in range(count)]


def test_sample_frames_buckets_by_target_rate() -> None:
    # 30 fps source, 1 second, sampled at 5 fps -> one frame per 0.2 s bucket.
    frames = list(sample_frames(_synthetic_source(30.0, 30), target_fps=5.0))
    timestamps = [round(f.timestamp, 3) for f in frames]
    assert timestamps == [0.0, 0.2, 0.4, 0.6, 0.8]


def test_sample_frames_handles_source_slower_than_target() -> None:
    # 2 fps source is already sparser than a 5 fps target: every frame survives.
    frames = list(sample_frames(_synthetic_source(2.0, 4), target_fps=5.0))
    assert [round(f.timestamp, 3) for f in frames] == [0.0, 0.5, 1.0, 1.5]


def test_sample_frames_rejects_nonpositive_rate() -> None:
    with pytest.raises(ValueError):
        list(sample_frames(_synthetic_source(30.0, 3), target_fps=0.0))


def test_frame_rejects_negative_timestamp() -> None:
    with pytest.raises(ValueError):
        Frame(-0.1, np.zeros((2, 2, 3), dtype=np.uint8))


def test_read_video_decodes_and_subsamples(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.avi"
    width, height, fps, count = 32, 32, 30, 30
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    assert writer.isOpened()
    for i in range(count):
        frame = np.full((height, width, 3), i * 8 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frames = list(read_video(path, target_fps=5.0))
    assert len(frames) == 5
    assert [round(f.timestamp, 2) for f in frames] == [0.0, 0.2, 0.4, 0.6, 0.8]
    assert frames[0].image.shape == (height, width, 3)
