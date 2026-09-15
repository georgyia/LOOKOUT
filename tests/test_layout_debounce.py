"""How long an arrangement must hold to count as a layout.

The debounce used to be `2.0 / target_fps` — two frames, expressed in seconds —
so it shrank as sampling improved. A client animating between layouts passes
through intermediate states, and at 0.4 s each qualified as stable; since
identity is not carried across a reshape, every one minted participants who were
never in the call.
"""

import pytest

from lookout.layout import LayoutInterval, Tile, segment_layouts
from lookout.models import RegionKind
from lookout.pipeline import AnalysisConfig


def _tiles(*positions: float) -> tuple[Tile, ...]:
    return tuple(Tile(RegionKind.PARTICIPANT, x, 0.0, 0.2, 0.2) for x in positions)


def _stream(fps: float, plan: list[tuple[float, tuple[Tile, ...]]]):
    """Sample a scripted arrangement timeline at ``fps``."""

    step = 1.0 / fps
    frames = []
    total = plan[-1][0]
    timestamp = 0.0
    while timestamp < total:
        current = plan[0][1]
        for until, tiles in plan:
            if timestamp < until:
                current = tiles
                break
        frames.append((timestamp, current))
        timestamp += step
    return frames


def _debounce(config: AnalysisConfig) -> float:
    return max(config.min_layout_seconds, 2.0 / config.target_fps)


def test_the_debounce_is_a_duration_not_a_frame_count() -> None:
    """The defect: sampling five times finer made the threshold five times
    shorter, so the same recording described a different meeting."""

    assert _debounce(AnalysisConfig(target_fps=1.0)) == 2.0
    assert _debounce(AnalysisConfig(target_fps=5.0)) == 2.0
    assert _debounce(AnalysisConfig(target_fps=25.0)) == 2.0


def test_two_frames_remain_the_floor() -> None:
    """A layout cannot be confirmed by fewer samples than two, so coarse
    sampling still raises the threshold above the configured duration."""

    assert _debounce(AnalysisConfig(target_fps=0.5, min_layout_seconds=1.0)) == 4.0
    assert _debounce(AnalysisConfig(target_fps=0.25, min_layout_seconds=1.0)) == 8.0


def test_a_brief_transition_does_not_become_a_layout() -> None:
    gallery = _tiles(0.0, 0.25, 0.5)
    mid_animation = _tiles(0.05, 0.3)
    share = _tiles(0.8, 0.8)

    plan = [(10.0, gallery), (10.6, mid_animation), (30.0, share)]
    intervals = segment_layouts(_stream(5.0, plan), min_stable_seconds=2.0)

    assert len(intervals) == 2, [i.tiles for i in intervals]


def test_a_layout_that_genuinely_holds_is_kept() -> None:
    gallery = _tiles(0.0, 0.25, 0.5)
    share = _tiles(0.8, 0.8)
    plan = [(10.0, gallery), (20.0, share), (30.0, gallery)]

    intervals = segment_layouts(_stream(5.0, plan), min_stable_seconds=2.0)
    assert len(intervals) == 3


@pytest.mark.parametrize("fps", [2.0, 5.0, 10.0, 25.0])
def test_the_same_recording_segments_the_same_way_at_any_rate(fps: float) -> None:
    """The property the defect broke: sampling is a cost knob, not a claim about
    the meeting."""

    gallery = _tiles(0.0, 0.25, 0.5)
    flicker = _tiles(0.05, 0.3)
    share = _tiles(0.8, 0.8)
    plan = [(10.0, gallery), (10.5, flicker), (30.0, share)]

    config = AnalysisConfig(target_fps=fps)
    intervals = segment_layouts(_stream(fps, plan), min_stable_seconds=_debounce(config))
    assert len(intervals) == 2


def test_the_threshold_is_configurable() -> None:
    gallery = _tiles(0.0, 0.25, 0.5)
    brief = _tiles(0.8, 0.8)
    plan = [(10.0, gallery), (13.0, brief), (30.0, gallery)]
    frames = _stream(5.0, plan)

    assert len(segment_layouts(frames, min_stable_seconds=2.0)) == 3
    assert len(segment_layouts(frames, min_stable_seconds=5.0)) == 1


def test_the_threshold_is_recorded_like_every_other() -> None:
    from lookout.runrecord import describe_config

    values, _ = describe_config(AnalysisConfig())
    assert values["min_layout_seconds"] == 2.0

    _, overrides = describe_config(AnalysisConfig(min_layout_seconds=3.0))
    assert overrides == ("min_layout_seconds",)


def test_an_empty_stream_yields_nothing() -> None:
    assert segment_layouts([], min_stable_seconds=2.0) == []


def test_intervals_are_contiguous_and_ordered() -> None:
    gallery = _tiles(0.0, 0.25, 0.5)
    share = _tiles(0.8, 0.8)
    intervals = segment_layouts(
        _stream(5.0, [(10.0, gallery), (30.0, share)]), min_stable_seconds=2.0
    )
    assert all(isinstance(i, LayoutInterval) for i in intervals)
    for earlier, later in zip(intervals[:-1], intervals[1:], strict=True):
        assert earlier.end_time == later.start_time
