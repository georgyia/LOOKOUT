import math

import pytest

from lookout.calibration import calibrate_mapping, fit_linear_ransac, weak_labels_from_speaker
from lookout.models import GazeDirection, HeadPose, Layout, LayoutSource, Region, RegionKind
from lookout.screen_mapping import ScreenMappingParams, project
from lookout.speaker import SpeakerSegment

POSE = HeadPose(0, 0, 0)


def test_ransac_recovers_line_despite_outliers() -> None:
    xs = [i / 10 for i in range(20)]
    ys = [2.0 * x + 0.5 for x in xs]
    # Corrupt a quarter of the points.
    for k in (2, 7, 11, 17):
        ys[k] += 5.0
    slope, intercept = fit_linear_ransac(xs, ys, threshold=0.1)
    assert slope == pytest.approx(2.0, abs=1e-6)
    assert intercept == pytest.approx(0.5, abs=1e-6)


def test_calibrate_recovers_known_mapping() -> None:
    true = ScreenMappingParams()
    labels = []
    yaws = [math.radians(v) for v in range(-15, 16, 3)]
    pitches = [math.radians(v) for v in range(-18, 1, 3)]
    for yaw in yaws:
        for pitch in pitches:
            projected = project(yaw, pitch, true)
            assert projected is not None
            labels.append((yaw, pitch, projected[0], projected[1]))
    # Add outliers with wrong targets.
    labels.append((math.radians(10), math.radians(-5), 0.0, 1.0))
    labels.append((math.radians(-10), math.radians(-15), 1.0, 0.0))

    recovered = calibrate_mapping(labels, threshold=0.02)

    assert recovered.yaw_at_left == pytest.approx(true.yaw_at_left, abs=1e-3)
    assert recovered.yaw_at_right == pytest.approx(true.yaw_at_right, abs=1e-3)
    assert recovered.pitch_at_top == pytest.approx(true.pitch_at_top, abs=1e-3)
    assert recovered.pitch_at_bottom == pytest.approx(true.pitch_at_bottom, abs=1e-3)


def test_calibrate_rejects_degenerate_geometry() -> None:
    # x decreasing with yaw is not a valid screen mapping.
    labels = [(0.0, 0.0, 1.0, 0.5), (0.2, 0.0, 0.0, 0.5), (0.4, 0.0, -1.0, 0.5)]
    with pytest.raises(ValueError):
        calibrate_mapping(labels, threshold=0.05)


def test_weak_labels_use_speaker_tile_center() -> None:
    layout = Layout(
        "bob",
        (
            Region(RegionKind.PARTICIPANT, 0.0, 0.0, 0.5, 1.0, "alice"),
            Region(RegionKind.PARTICIPANT, 0.5, 0.0, 0.5, 1.0, "bob"),
        ),
        0.0,
        LayoutSource.ASSUMED_SHARED,
    )
    directions = [
        GazeDirection(1.0, "bob", 0.1, -0.1, POSE, 0.9),  # alice speaking -> label
        GazeDirection(5.0, "bob", 0.2, -0.2, POSE, 0.9),  # bob (self) speaking -> skip
    ]
    segments = [
        SpeakerSegment("alice", 0.0, 2.0, 0.9, "highlight"),
        SpeakerSegment("bob", 4.0, 6.0, 0.9, "highlight"),
    ]
    labels = weak_labels_from_speaker(directions, segments, layout)
    assert labels == [(0.1, -0.1, 0.25, 0.5)]
