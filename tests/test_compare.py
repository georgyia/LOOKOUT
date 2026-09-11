"""Comparing two runs.

The comparison this exists for went unmade for a milestone: both records
existed, nothing put them side by side, and the 3-minute clip's face hit rate of
82.9% was quoted throughout while the full recording's was 73.7%.
"""

from lookout.compare import compare
from lookout.coverage import Coverage, ParticipantCoverage
from lookout.diagnostics import Diagnostics
from lookout.pipeline import AnalysisConfig
from lookout.runrecord import AdapterInfo, VideoInfo, build_record
from lookout.screen_mapping import ScreenMappingParams

FIXED = "2026-01-01T00:00:00+00:00"


def _record(
    *,
    config: AnalysisConfig | None = None,
    coverage: Coverage | None = None,
    diagnostics: Diagnostics | None = None,
    evaluation: dict | None = None,
    adapters: tuple[AdapterInfo, ...] = (AdapterInfo("gaze", "geometric"),),
    results: dict | None = None,
):
    record = build_record(
        config or AnalysisConfig(),
        results or {},
        coverage=coverage,
        diagnostics=diagnostics,
        evaluation=evaluation,
        adapters=adapters,
        created_at=FIXED,
    )
    return record


def _coverage(**kwargs) -> Coverage:
    base = dict(
        frames=100,
        participants_detected=4,
        face_attempts=400,
        face_hits=332,
        directions=332,
        points=330,
        off_screen=2,
        fixations=40,
        attributions=40,
        events=12,
        layout_sources=("assumed_shared",),
        per_participant=(ParticipantCoverage("slot_0", face_attempts=100, face_hits=83),),
    )
    base.update(kwargs)
    return Coverage(**base)


def test_a_run_compared_with_itself_reports_nothing() -> None:
    record = _record(coverage=_coverage())
    result = compare(record, record)
    assert result.identical
    assert result.comparable


def test_different_adapters_block_comparison_before_any_numbers() -> None:
    """A difference between runs that answer different questions is not a finding."""

    left = _record(adapters=(AdapterInfo("gaze", "geometric"),))
    right = _record(adapters=(AdapterInfo("gaze", "L2CSGazeEstimator"),))
    result = compare(left, right)

    assert not result.comparable
    assert [d.field for d in result.blockers] == ["adapter.gaze"]
    assert "not directly comparable" in result.notes[0]


def test_a_different_recording_blocks_comparison() -> None:
    left = _record()
    right = _record()
    object.__setattr__(
        left.provenance, "video", VideoInfo(path="a.mp4", sha256="aaa")
    )
    object.__setattr__(
        right.provenance, "video", VideoInfo(path="b.mp4", sha256="bbb")
    )
    result = compare(left, right)
    assert not result.comparable
    assert any(d.field == "video" for d in result.blockers)


def test_configuration_differences_are_named_per_field() -> None:
    """"The hashes differ" is true and useless."""

    left = _record()
    right = _record(
        config=AnalysisConfig(
            target_fps=2.0, mapping=ScreenMappingParams(off_screen_margin=0.2)
        )
    )
    fields = {d.field for d in compare(left, right).configuration}
    assert "config.target_fps" in fields
    assert "config.mapping.off_screen_margin" in fields
    assert "config.attribution.confidence_floor" not in fields  # unchanged


def test_coverage_and_results_are_reported_separately() -> None:
    """A result that moved because the pipeline saw more of the recording is a
    different claim from one that moved because attribution changed its mind."""

    left = _record(
        coverage=_coverage(),
        results={"viewers": {"a": {"duration_by_target": {"slot_1": 10.0}}}},
    )
    right = _record(
        coverage=_coverage(face_hits=200, directions=200, points=198),
        results={"viewers": {"a": {"duration_by_target": {"slot_1": 6.0}}}},
    )
    result = compare(left, right)

    assert {d.field for d in result.coverage} >= {"coverage.face_hits", "coverage.face_hit_rate"}
    assert [d.field for d in result.results] == ["duration.slot_1"]


def test_a_change_in_frames_analyzed_is_called_out() -> None:
    """The trap this tool exists to avoid: reading a duration difference as a
    change in behaviour when the runs cover different amounts of recording."""

    left = _record(coverage=_coverage(frames=100))
    right = _record(
        coverage=_coverage(
            frames=900, face_attempts=3600, face_hits=2652, directions=2652,
            points=2650, off_screen=2,
        )
    )
    notes = " ".join(compare(left, right).notes)
    assert "cover different amounts of recording" in notes


def test_the_clip_against_the_full_recording_surfaces_the_hit_rate_gap() -> None:
    """The comparison that went unmade for a milestone."""

    clip = _record(coverage=_coverage(face_attempts=6648, face_hits=5510, directions=5510,
                                      points=5504, off_screen=6))
    full = _record(coverage=_coverage(frames=12803, face_attempts=96899, face_hits=71396,
                                      directions=71396, points=71390, off_screen=6))
    rates = {
        d.field: (d.left, d.right) for d in compare(clip, full).coverage
    }
    left_rate, right_rate = rates["coverage.face_hit_rate"]
    assert left_rate > right_rate
    assert round(left_rate, 3) == 0.829
    assert round(right_rate, 3) == 0.737


def test_overlapping_intervals_are_reported_as_inconclusive() -> None:
    """A difference inside the noise is the most common way a comparison misleads."""

    left = _record(evaluation={"hit_rate": 0.62, "hit_rate_ci95": [0.50, 0.74]})
    right = _record(evaluation={"hit_rate": 0.70, "hit_rate_ci95": [0.58, 0.82]})
    notes = " ".join(compare(left, right).notes)

    assert "inconclusive" in notes
    assert "overlap" in notes


def test_disjoint_intervals_are_reported_as_such() -> None:
    left = _record(evaluation={"hit_rate": 0.30, "hit_rate_ci95": [0.25, 0.35]})
    right = _record(evaluation={"hit_rate": 0.80, "hit_rate_ci95": [0.75, 0.85]})
    notes = " ".join(compare(left, right).notes)

    assert "disjoint" in notes
    assert "inconclusive" not in notes


def test_comparing_a_scored_run_with_an_unscored_one_says_so() -> None:
    left = _record(evaluation={"hit_rate": 0.7})
    right = _record()
    result = compare(left, right)

    assert any(d.field == "evaluation" for d in result.evaluation)
    assert any("Only one run was scored" in note for note in result.notes)


def test_distribution_changes_are_reported() -> None:
    left = _record(diagnostics=Diagnostics(top_target="slot_4", top_target_share=0.93))
    right = _record(diagnostics=Diagnostics(top_target="slot_1", top_target_share=0.31))
    fields = {d.field for d in compare(left, right).results}

    assert "distribution.top_target" in fields
    assert "distribution.top_target_share" in fields
