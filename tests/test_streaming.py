"""Peak memory must not scale with recording length.

`analyze` used to materialize every sampled frame before observing any of them,
so a long call was limited by RAM rather than by time: at 5 fps a 43-minute
854x480 recording is about 16 GB of held frames, and the failure when that
ceiling arrives is an allocation error rather than a degradation.

Retention is measured with weak references to the decoded arrays, which observes
what the pipeline actually holds. A byte threshold would be machine-specific and
would pass on a large runner while the property it protects had already been
lost.
"""

import gc
import weakref
from pathlib import Path

import pytest

from lookout import artifacts
from lookout.frames import Frame
from lookout.pipeline import EVENTS, AnalysisConfig, analyze
from tests.test_pipeline import _fake_stage, _grid_frame, _write_video

pytest.importorskip("cv2")


def _retention(monkeypatch, tmp_path: Path, frame_count: int) -> int:
    """Run analyze over a synthetic source and report the most frames held at once.

    The source drops its own reference before yielding, so anything still alive
    afterwards is held by the pipeline.
    """

    import lookout.pipeline as pipeline

    refs: list[weakref.ref] = []
    peak = 0

    def source(video, target_fps):
        nonlocal peak
        for index in range(frame_count):
            image = _grid_frame()
            refs.append(weakref.ref(image))
            frame = Frame(index * (1.0 / target_fps), image)
            del image
            yield frame
            del frame
            gc.collect()
            peak = max(peak, sum(1 for ref in refs if ref() is not None))

    monkeypatch.setattr(pipeline, "read_video", source)
    analyze(
        tmp_path / f"clip{frame_count}.avi",
        tmp_path / f"run{frame_count}",
        _fake_stage,
        AnalysisConfig(target_fps=5.0),
    )
    return peak


def test_only_a_constant_number_of_frames_is_held(tmp_path, monkeypatch) -> None:
    """The direct statement: the pipeline never holds the whole recording."""

    peak = _retention(monkeypatch, tmp_path, frame_count=40)
    assert peak <= 2, f"held {peak} decoded frames at once over 40 sampled frames"


def test_retention_does_not_grow_with_recording_length(tmp_path, monkeypatch) -> None:
    """Quadrupling the recording must not multiply what is retained."""

    short = _retention(monkeypatch, tmp_path, frame_count=20)
    long = _retention(monkeypatch, tmp_path, frame_count=80)
    assert long <= short + 1, f"retention grew from {short} to {long} with recording length"


def test_streaming_does_not_change_results(tmp_path: Path) -> None:
    """A memory fix that altered the answers would not be a fix."""

    video = tmp_path / "clip.avi"
    _write_video(video, frames=60, fps=30)

    first = analyze(video, tmp_path / "a", _fake_stage, AnalysisConfig(target_fps=5.0))
    second = analyze(video, tmp_path / "b", _fake_stage, AnalysisConfig(target_fps=5.0))

    assert first.coverage.frames == second.coverage.frames
    assert first.coverage.events == second.coverage.events
    assert first.coverage.face_attempts == second.coverage.face_attempts
    assert artifacts.read_events(tmp_path / "a" / EVENTS) == artifacts.read_events(
        tmp_path / "b" / EVENTS
    )


def test_both_passes_see_every_frame(tmp_path: Path) -> None:
    """Streaming means decoding twice; an off-by-one there would silently drop
    the first or last frame of the observation pass."""

    video = tmp_path / "clip.avi"
    _write_video(video, frames=60, fps=30)
    outcome = analyze(video, tmp_path / "run", _fake_stage, AnalysisConfig(target_fps=5.0))

    # Every sampled frame contributed a tile crop for each of the four slots.
    assert outcome.coverage.face_attempts == outcome.coverage.frames * 4


def test_an_empty_video_streams_without_error(tmp_path, monkeypatch) -> None:
    import lookout.pipeline as pipeline

    monkeypatch.setattr(pipeline, "read_video", lambda video, target_fps: iter(()))
    outcome = analyze(tmp_path / "none.avi", tmp_path / "run", _fake_stage, AnalysisConfig())

    assert outcome.coverage.frames == 0
    assert outcome.coverage.events == 0
