"""The observer that works where MediaPipe does not.

YuNet gives five landmarks and no iris, so it can see where a head points but
not where its eyes do. These pin that it reports exactly that, and nothing more:
an observer that quietly invented an eye direction would be indistinguishable in
the output from one that measured it.
"""

import math

import numpy as np
import pytest

from lookout.face import LEFT_IRIS, RIGHT_IRIS, YuNetFaceObserver, _synthetic_landmarks
from lookout.gaze_geometric import estimate_gaze, eye_offsets
from lookout.models import HeadPose

pytest.importorskip("cv2")


def _face(
    *,
    right_eye: tuple[float, float] = (40.0, 50.0),
    left_eye: tuple[float, float] = (80.0, 50.0),
    nose: tuple[float, float] = (60.0, 72.0),
    score: float = 0.9,
) -> np.ndarray:
    """One YuNet detection row: box, five landmarks, score."""

    row = np.zeros(15, dtype=np.float32)
    row[0:4] = (20.0, 20.0, 80.0, 80.0)
    row[4:6] = right_eye
    row[6:8] = left_eye
    row[8:10] = nose
    row[10:12] = (48.0, 90.0)
    row[12:14] = (72.0, 90.0)
    row[14] = score
    return row


def _observe(**kwargs):
    observer = YuNetFaceObserver.__new__(YuNetFaceObserver)
    return observer._observation(_face(**kwargs), width=120, height=120)


def test_the_eye_contribution_is_zero() -> None:
    """The observer has no iris data, so zero is the only honest eye direction.

    Not bit-exact: the landmarks are float32, so placing an iris at the centre
    of a synthesized eye box leaves about 1e-7 of rounding. That is seven orders
    of magnitude below the smallest angle the mapping resolves.
    """

    observation = _observe()
    horizontal, vertical, _ = eye_offsets(observation)
    assert horizontal == pytest.approx(0.0, abs=1e-6)
    assert vertical == pytest.approx(0.0, abs=1e-6)


def test_gaze_is_head_orientation_alone() -> None:
    observation = _observe(nose=(70.0, 72.0))  # head turned
    direction = estimate_gaze(observation, timestamp=0.0, person_id="a")

    assert direction.yaw == pytest.approx(observation.head_pose.yaw, abs=1e-6)
    assert direction.pitch == pytest.approx(observation.head_pose.pitch, abs=1e-6)


def test_a_turned_head_yields_a_turned_gaze() -> None:
    straight = _observe(nose=(60.0, 72.0))
    right = _observe(nose=(74.0, 72.0))
    left = _observe(nose=(46.0, 72.0))

    assert right.head_pose.yaw > straight.head_pose.yaw > left.head_pose.yaw


def test_a_raised_nose_reads_as_looking_up() -> None:
    level = _observe(nose=(60.0, 72.0))
    raised = _observe(nose=(60.0, 62.0))
    assert raised.head_pose.pitch > level.head_pose.pitch


def test_roll_comes_from_the_eye_line() -> None:
    """Two eye points genuinely determine roll, unlike eye direction."""

    level = _observe()
    tilted = _observe(right_eye=(40.0, 44.0), left_eye=(80.0, 56.0))

    assert level.head_pose.roll == pytest.approx(0.0, abs=1e-6)
    assert tilted.head_pose.roll > math.radians(10.0)


def test_the_detector_score_becomes_the_detection_confidence() -> None:
    assert _observe(score=0.42).detection_confidence == pytest.approx(0.42)
    assert _observe(score=1.5).detection_confidence == 1.0
    assert _observe(score=-0.2).detection_confidence == 0.0


def test_no_blendshapes_are_claimed() -> None:
    """MediaPipe supplies eye coefficients; this observer has none to give."""

    assert _observe().blendshapes == {}


def test_interocular_distance_is_real_not_assumed() -> None:
    """Confidence scales with face size, so the eye spacing has to be measured."""

    near = _observe(right_eye=(30.0, 50.0), left_eye=(90.0, 50.0))
    far = _observe(right_eye=(55.0, 50.0), left_eye=(65.0, 50.0))

    near_gaze = estimate_gaze(near, timestamp=0.0, person_id="a")
    far_gaze = estimate_gaze(far, timestamp=0.0, person_id="a")
    assert near_gaze.quality is not None and far_gaze.quality is not None
    assert near_gaze.quality.size > far_gaze.quality.size


def test_eyes_are_reported_open_rather_than_squinting() -> None:
    """A squint here would understate confidence for a reason the observer
    cannot actually see."""

    direction = estimate_gaze(_observe(), timestamp=0.0, person_id="a")
    assert direction.quality is not None
    assert direction.quality.openness == 1.0
    assert direction.quality.limiting != "openness"


def test_synthetic_landmarks_put_each_iris_at_its_own_eye() -> None:
    landmarks = _synthetic_landmarks((40.0, 50.0), (80.0, 50.0), None, width=120, height=120)
    right = landmarks[list(RIGHT_IRIS), :2].mean(axis=0)
    left = landmarks[list(LEFT_IRIS), :2].mean(axis=0)

    assert right[0] == pytest.approx(40.0 / 120)
    assert left[0] == pytest.approx(80.0 / 120)
    assert right[1] == left[1] == pytest.approx(50.0 / 120)


def test_a_degenerate_crop_is_not_observed() -> None:
    observer = YuNetFaceObserver.__new__(YuNetFaceObserver)
    assert observer.observe(np.zeros((1, 1, 3), dtype=np.uint8)) is None


def test_head_pose_stays_within_its_documented_range() -> None:
    """Extreme landmark geometry must not produce angles the mapping cannot use."""

    for nose_x in (0.0, 200.0):
        for nose_y in (0.0, 200.0):
            pose = _observe(nose=(nose_x, nose_y)).head_pose
            assert -0.35 <= pose.yaw <= 0.35
            assert -0.40 <= pose.pitch <= 0.05
            assert isinstance(pose, HeadPose)


def test_mouth_corners_are_kept_rather_than_discarded() -> None:
    """YuNet detects them; #58 found the pipeline was throwing them away."""

    from lookout.face import MOUTH_CORNERS
    from lookout.speaker import mouth_open_ratio

    observation = _observe()
    corners = observation.landmarks[list(MOUTH_CORNERS), :2]
    assert corners.any(), "the mouth corners should be populated"
    assert mouth_open_ratio(observation) is not None
