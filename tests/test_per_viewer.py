"""Per-viewer layouts and per-viewer mappings.

Two of the pipeline's largest error sources are assumptions rather than model
error: every viewer is attributed against the recording's layout, and every
viewer is mapped with one laptop's angle-to-screen prior. Both are systematic,
which is the kind of error note 13 shows the pipeline tolerates least.
"""

import math
from pathlib import Path

from lookout import artifacts, store
from lookout.calibration import calibrate_mapping
from lookout.evaluate import evaluate
from lookout.manifest import save_manifest
from lookout.models import (
    GazeDirection,
    HeadPose,
    Layout,
    LayoutSource,
    Region,
    RegionKind,
)
from lookout.pipeline import (
    EVENTS,
    GAZE_RAW,
    LAYOUT,
    SPEAKER,
    AnalysisConfig,
    attribute,
)
from lookout.screen_mapping import ScreenMappingParams
from lookout.speaker import SpeakerSegment
from tests.synthetic import angles_for, grid_layout, scripted_directions, tour, truth_intervals

CONFIG = AnalysisConfig(
    target_fps=5.0,
    min_fixation=0.3,
    max_gap=0.4,
    event_min_duration=0.4,
    min_calibration_labels=10,
)


def _write_run(out: Path, directions: list[GazeDirection], layout: Layout) -> None:
    out.mkdir(parents=True, exist_ok=True)
    store.write_gaze(out / GAZE_RAW, directions)
    artifacts.write_layouts(out / LAYOUT, [layout])


# ---------------------------------------------------------------- per-viewer layouts


def test_a_manifest_replaces_the_shared_layout_assumption(tmp_path: Path) -> None:
    """Bob's screen has alice and carol swapped relative to the recording. The
    same gaze angle therefore means a different person for him."""

    out = tmp_path / "run"
    recording = Layout(
        "recording",
        (
            Region(RegionKind.PARTICIPANT, 0.0, 0.0, 0.5, 1.0, "alice"),
            Region(RegionKind.PARTICIPANT, 0.5, 0.0, 0.5, 1.0, "carol"),
        ),
        0.0,
        LayoutSource.RECORDING,
    )
    params = ScreenMappingParams()
    yaw, pitch = angles_for(0.25, 0.5, params)  # left half of the screen
    directions = [
        GazeDirection(i * 0.2, "bob", yaw, pitch, HeadPose(0, 0, 0), 0.9) for i in range(20)
    ]
    _write_run(out, directions, recording)

    assumed, _ = attribute(out, CONFIG)
    (assumed_event,) = artifacts.read_events(out / EVENTS)
    assert assumed_event.target == "alice"
    assert assumed.layout_sources == ("assumed_shared",)

    # Bob actually saw carol on the left.
    bob = Layout(
        "bob",
        (
            Region(RegionKind.PARTICIPANT, 0.0, 0.0, 0.5, 1.0, "carol"),
            Region(RegionKind.PARTICIPANT, 0.5, 0.0, 0.5, 1.0, "alice"),
        ),
        0.0,
        LayoutSource.MANIFEST,
    )
    known, _ = attribute(out, CONFIG, [bob])
    (known_event,) = artifacts.read_events(out / EVENTS)
    assert known_event.target == "carol"
    assert known.layout_sources == ("manifest",)


def test_a_manifest_round_trips_through_a_file(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    layout = grid_layout(2, 2, viewer_id="bob", source=LayoutSource.MANIFEST)
    save_manifest(path, [layout])

    from lookout.manifest import load_manifest

    (restored,) = load_manifest(path)
    assert restored.viewer_id == "bob"
    assert restored.source is LayoutSource.MANIFEST
    assert len(restored.participant_regions()) == 4


# ------------------------------------------------------------------- calibration


def _speaker_run(tmp_path: Path, params: ScreenMappingParams) -> tuple[Path, list]:
    """A viewer whose real geometry is ``params``, looking at whoever speaks.

    Returns the run directory and the ground truth the cue schedule implies.
    """

    out = tmp_path / "run"
    layout = grid_layout(2, 2)
    targets = ["slot_0", "slot_1", "slot_2", "slot_3"] * 3
    cues = tour("slot_0", targets, dwell=2.0)
    directions = scripted_directions(layout, cues, fps=5.0, params=params)
    _write_run(out, directions, layout)

    # The speaker is whoever the viewer was cued to look at: the conversation
    # prior this calibration rests on, made exact.
    artifacts.write_speaker_segments(
        out / SPEAKER,
        [
            SpeakerSegment(cue.target, cue.start_time, cue.end_time, 0.9, "highlight")
            for cue in cues
        ],
    )
    return out, truth_intervals(cues, "2x2")


def _score(out: Path, truth: list, calibrate: bool) -> float:
    attribute(out, CONFIG, calibrate=calibrate)
    return evaluate(artifacts.read_events(out / EVENTS), truth).hit_rate


def test_calibration_recovers_a_viewer_on_different_geometry(tmp_path: Path) -> None:
    """The prior encodes one laptop. A viewer sitting closer sweeps a wider angle
    across the same screen, so every target shifts outward and the shared prior
    misses systematically. Measured, not asserted."""

    real = ScreenMappingParams(
        yaw_at_left=math.radians(-30.0),
        yaw_at_right=math.radians(30.0),
        pitch_at_top=math.radians(4.0),
        pitch_at_bottom=math.radians(-34.0),
    )
    out, truth = _speaker_run(tmp_path, real)

    before = _score(out, truth, calibrate=False)
    after = _score(out, truth, calibrate=True)

    assert before < 0.5, f"the shared prior should fail on this geometry, scored {before:.0%}"
    assert after > before
    assert after >= 0.9, f"calibration should recover the mapping, scored {after:.0%}"


def test_calibration_reports_which_viewers_were_fit(tmp_path: Path) -> None:
    """A calibrated run and an assumed one must not read alike."""

    out, _ = _speaker_run(tmp_path, ScreenMappingParams())

    _, none = attribute(out, CONFIG, calibrate=False)
    assert none == ()

    _, (report,) = attribute(out, CONFIG, calibrate=True)
    assert report.viewer_id == "slot_0"
    assert report.calibrated is True
    assert report.labels >= CONFIG.min_calibration_labels
    assert report.reason == "fit from speaker cues"


def test_calibration_does_not_harm_a_viewer_already_on_the_prior(tmp_path: Path) -> None:
    """Fitting must not be a regression when the assumption happened to be right."""

    out, truth = _speaker_run(tmp_path, ScreenMappingParams())
    before = _score(out, truth, calibrate=False)
    after = _score(out, truth, calibrate=True)
    assert after >= before


def test_calibration_falls_back_rather_than_fitting_badly(tmp_path: Path) -> None:
    """A degenerate fit is worse than a documented guess, because it looks fitted."""

    out, _ = _speaker_run(tmp_path, ScreenMappingParams())
    artifacts.write_speaker_segments(out / SPEAKER, [])

    _, (report,) = attribute(out, CONFIG, calibrate=True)
    assert report.calibrated is False
    assert report.reason == "no speaker segments"


def test_too_few_labels_keeps_the_prior(tmp_path: Path) -> None:
    """A fit over a handful of glances tracks their noise; the prior is at least
    wrong consistently."""

    out, _ = _speaker_run(tmp_path, ScreenMappingParams())
    segments = artifacts.read_speaker_segments(out / SPEAKER)
    artifacts.write_speaker_segments(out / SPEAKER, segments[:1])

    strict = AnalysisConfig(
        min_fixation=0.3, max_gap=0.4, event_min_duration=0.4, min_calibration_labels=500
    )
    _, (report,) = attribute(out, strict, calibrate=True)
    assert report.calibrated is False
    assert report.reason == "too few weak labels"


def test_a_degenerate_geometry_is_rejected() -> None:
    """calibrate_mapping refuses a fit where x does not increase with yaw."""

    import pytest

    backwards = [(0.1 * i, 0.0, 1.0 - 0.1 * i, 0.5) for i in range(10)]
    with pytest.raises(ValueError, match="x must increase with yaw"):
        calibrate_mapping(backwards)


def test_speaker_segments_round_trip(tmp_path: Path) -> None:
    """Speaker context is an observation: deriving it needs the frames, so
    attribution has to be able to read it back."""

    path = tmp_path / "speaker.jsonl"
    segments = [
        SpeakerSegment("slot_1", 0.0, 2.0, 0.8, "highlight"),
        SpeakerSegment("slot_2", 2.0, 4.5, 0.6, "mouth"),
    ]
    artifacts.write_speaker_segments(path, segments)
    assert artifacts.read_speaker_segments(path) == segments
