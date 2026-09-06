import math
from pathlib import Path

import numpy as np
import pytest

from lookout import artifacts
from lookout.frames import Image
from lookout.models import GazeDirection, HeadPose
from lookout.pipeline import EVENTS, GAZE_RAW, LAYOUT, AnalysisConfig, analyze, attribute
from lookout.runrecord import build_record, read_record, write_record


def _grid_frame(w: int = 640, h: int = 360, gutter: int = 10) -> np.ndarray:
    image = np.full((h, w, 3), 16, dtype=np.uint8)
    colors = [(200, 80, 80), (80, 200, 80), (80, 80, 200), (200, 200, 80)]
    k = 0
    for r in range(2):
        for c in range(2):
            x0, y0 = int(c * w / 2) + gutter, int(r * h / 2) + gutter
            x1, y1 = int((c + 1) * w / 2) - gutter, int((r + 1) * h / 2) - gutter
            image[y0:y1, x0:x1] = colors[k]
            k += 1
    return image


def _write_video(path: Path, frames: int = 30, fps: int = 30) -> None:
    cv2 = pytest.importorskip("cv2")
    # Prefer a lossless codec so decoded frames are identical; fall back to MJPG.
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (640, 360))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 360))
    assert writer.isOpened()
    frame = _grid_frame()
    for _ in range(frames):
        writer.write(frame)
    writer.release()


def _fake_stage(crop: Image, person_id: str, timestamp: float) -> GazeDirection | None:
    # Everyone looks at the top-left tile (slot_0): x~0.25, y~0.25.
    return GazeDirection(
        timestamp, person_id, math.radians(-8.0), math.radians(-4.75), HeadPose(0, 0, 0), 0.9
    )


def test_analyze_writes_artifacts_and_events(tmp_path: Path) -> None:
    video = tmp_path / "clip.avi"
    _write_video(video)
    out = tmp_path / "run"

    summary = analyze(video, out, _fake_stage, AnalysisConfig(target_fps=5.0))

    for name in (GAZE_RAW, LAYOUT, "gaze_screen.jsonl", "attribution.jsonl", EVENTS):
        assert (out / name).exists(), name

    # Segmentation itself is covered by test_layout; here we only require a
    # stable gallery to have been found and the four tiles observed per frame.
    assert summary["layouts"] >= 1
    assert summary["directions"] == 20  # 4 tiles x 5 sampled frames

    events = artifacts.read_events(out / EVENTS)
    assert len(events) == 4  # one per participant slot
    assert {e.viewer_id for e in events} == {"slot_0", "slot_1", "slot_2", "slot_3"}
    assert all(e.target == "slot_0" for e in events)


def test_attribute_rerun_is_reproducible(tmp_path: Path) -> None:
    video = tmp_path / "clip.avi"
    _write_video(video)
    out = tmp_path / "run"
    analyze(video, out, _fake_stage, AnalysisConfig(target_fps=5.0))

    first = artifacts.read_events(out / EVENTS)
    attribute(out)  # re-run from the raw store only
    second = artifacts.read_events(out / EVENTS)
    assert first == second


def test_a_run_can_be_documented_by_a_record(tmp_path: Path) -> None:
    """The record identifies the recording and the configuration behind a run."""

    video = tmp_path / "clip.avi"
    _write_video(video)
    out = tmp_path / "run"
    config = AnalysisConfig(target_fps=5.0)
    analyze(video, out, _fake_stage, config)

    events = artifacts.read_events(out / EVENTS)
    record = build_record(config, {"total_events": len(events)}, video=video)
    write_record(out / "report.json", record)
    restored = read_record(out / "report.json")

    assert restored.provenance.video is not None
    assert restored.provenance.video.sha256 is not None
    assert restored.provenance.video.width == 640
    assert restored.provenance.video.height == 360
    assert restored.config["target_fps"] == 5.0
    assert restored.config["mapping"]["off_screen_margin"] == 0.05
    assert restored.results["total_events"] == len(events)
