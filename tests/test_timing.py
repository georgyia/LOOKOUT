"""What a run costs, and the regression guard that is stable enough to assert.

A wall-clock threshold in CI is a flaky test: shared runners vary by more than
the effects worth catching. So durations are recorded for humans to compare, and
what CI asserts is the *work* done — how often each stage runs, and how that
scales. Those hold on any machine.
"""

import time
from pathlib import Path

import pytest

from lookout.pipeline import AnalysisConfig, analyze
from lookout.runrecord import timing_to_dict
from lookout.timing import RunTiming, StageTiming, Stopwatch
from tests.test_pipeline import _fake_stage, _write_video


def test_a_stopwatch_accumulates_per_stage() -> None:
    watch = Stopwatch()
    for _ in range(3):
        with watch.stage("detect"):
            time.sleep(0.001)
    with watch.stage("map"):
        time.sleep(0.001)

    result = watch.result(video_seconds=10.0)
    stages = {s.stage: s for s in result.stages}
    assert stages["detect"].calls == 3
    assert stages["map"].calls == 1
    assert stages["detect"].seconds > 0
    assert watch.calls("detect") == 3
    assert watch.calls("never ran") == 0


def test_stages_are_ordered_by_cost() -> None:
    """The dominant stage is the one worth attacking, so it leads."""

    watch = Stopwatch()
    with watch.stage("cheap"):
        pass
    with watch.stage("expensive"):
        time.sleep(0.01)

    result = watch.result()
    assert result.stages[0].stage == "expensive"
    assert result.dominant is not None
    assert result.dominant.stage == "expensive"


def test_throughput_answers_how_long_a_recording_takes() -> None:
    """A per-stage duration does not answer that; a realtime multiple does."""

    timing = RunTiming(
        stages=(StageTiming("decode", 40.0, 1),),
        wall_seconds=50.0,
        video_seconds=100.0,
    )
    assert timing.realtime_factor == 2.0
    assert timing.projected_seconds(3600.0) == 1800.0


def test_an_unmeasured_run_reports_nothing_rather_than_zero_throughput() -> None:
    empty = RunTiming()
    assert empty.realtime_factor == 0.0
    assert empty.projected_seconds(3600.0) == 0.0
    assert empty.dominant is None
    assert empty.stages == ()


def test_shares_sum_to_the_whole() -> None:
    stages = (StageTiming("a", 3.0, 1), StageTiming("b", 1.0, 1))
    total = sum(s.seconds for s in stages)
    assert sum(s.share_of(total) for s in stages) == pytest.approx(1.0)
    assert StageTiming("a", 1.0, 1).share_of(0.0) == 0.0


def test_timing_serializes_the_figures_a_reader_uses() -> None:
    timing = RunTiming(
        stages=(StageTiming("decode", 3.0, 10), StageTiming("map", 1.0, 10)),
        wall_seconds=5.0,
        video_seconds=10.0,
    )
    payload = timing_to_dict(timing)
    assert payload["realtime_factor"] == 2.0
    assert payload["stages"][0]["share"] == 0.75


# ------------------------------------------------- work, not wall clock


def test_decoding_happens_once_per_frame_not_once_per_tile(tmp_path: Path) -> None:
    """The regression that matters: a stage moving inside a per-tile loop.

    Wall clock would catch it only on a quiet machine; call counts catch it
    anywhere."""

    video = tmp_path / "clip.avi"
    _write_video(video, frames=30, fps=30)
    outcome = analyze(video, tmp_path / "run", _fake_stage, AnalysisConfig(target_fps=5.0))

    calls = {s.stage: s.calls for s in outcome.timing.stages}
    frames = outcome.coverage.frames
    assert calls["decode"] == 1
    assert calls["detect_tiles"] == frames
    assert calls["segment_layouts"] == 1
    assert calls["attribute"] == 1


def test_gaze_is_observed_once_per_tile_crop(tmp_path: Path) -> None:
    video = tmp_path / "clip.avi"
    _write_video(video, frames=30, fps=30)
    outcome = analyze(video, tmp_path / "run", _fake_stage, AnalysisConfig(target_fps=5.0))

    calls = {s.stage: s.calls for s in outcome.timing.stages}
    assert calls["observe_gaze"] == outcome.coverage.face_attempts


def test_work_scales_linearly_with_sampled_frames(tmp_path: Path) -> None:
    """Attribution being quadratic in observations would not show up on a short
    clip as a duration, but it shows up here as a ratio."""

    video = tmp_path / "clip.avi"
    _write_video(video, frames=60, fps=30)

    sparse = analyze(video, tmp_path / "a", _fake_stage, AnalysisConfig(target_fps=2.0))
    dense = analyze(video, tmp_path / "b", _fake_stage, AnalysisConfig(target_fps=8.0))

    sparse_calls = {s.stage: s.calls for s in sparse.timing.stages}
    dense_calls = {s.stage: s.calls for s in dense.timing.stages}

    assert dense.coverage.frames > sparse.coverage.frames
    ratio = dense.coverage.frames / sparse.coverage.frames
    assert dense_calls["detect_tiles"] / sparse_calls["detect_tiles"] == pytest.approx(
        ratio, rel=0.01
    )


def test_a_run_reports_its_own_throughput(tmp_path: Path) -> None:
    video = tmp_path / "clip.avi"
    _write_video(video, frames=30, fps=30)
    outcome = analyze(video, tmp_path / "run", _fake_stage, AnalysisConfig(target_fps=5.0))

    assert outcome.timing.wall_seconds > 0
    assert outcome.timing.video_seconds > 0
    assert outcome.timing.realtime_factor > 0
