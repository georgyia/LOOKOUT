import numpy as np
import pytest

from lookout.face import FaceObservation
from lookout.layout import Tile
from lookout.models import HeadPose, RegionKind
from lookout.speaker import (
    aggregate_speaker_segments,
    detect_highlighted_tile,
    mouth_open_ratio,
)


def _tile(x: float) -> Tile:
    return Tile(RegionKind.PARTICIPANT, x, 0.1, 0.3, 0.6)


def test_detect_highlighted_tile_finds_bordered_tile() -> None:
    image = np.full((200, 400, 3), 40, dtype=np.uint8)
    tiles = (_tile(0.05), _tile(0.45))
    # Fill both tiles a uniform colour; add a bright border to the second.
    for t in tiles:
        image[20:140, int(t.x * 400) : int((t.x + t.width) * 400)] = 60
    x0, x1 = int(0.45 * 400), int(0.75 * 400)
    image[20:140, x0 : x0 + 4] = (0, 255, 255)
    image[20:140, x1 - 4 : x1] = (0, 255, 255)
    image[20:24, x0:x1] = (0, 255, 255)
    image[136:140, x0:x1] = (0, 255, 255)

    assert detect_highlighted_tile(image, tiles) == 1


def test_detect_highlighted_tile_returns_none_when_uniform() -> None:
    image = np.full((200, 400, 3), 40, dtype=np.uint8)
    tiles = (_tile(0.05), _tile(0.45))
    for t in tiles:
        image[20:140, int(t.x * 400) : int((t.x + t.width) * 400)] = 60
    assert detect_highlighted_tile(image, tiles) is None


def _mouth_obs(gap: float) -> FaceObservation:
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[list(range(468, 473)), :2] = (0.35, 0.5)  # right iris
    landmarks[list(range(473, 478)), :2] = (0.65, 0.5)  # left iris
    landmarks[13, :2] = (0.5, 0.7)
    landmarks[14, :2] = (0.5, 0.7 + gap)
    return FaceObservation(landmarks, HeadPose(0, 0, 0), {}, 1.0)


def test_mouth_open_ratio_increases_with_gap() -> None:
    closed = mouth_open_ratio(_mouth_obs(0.01))
    open_ = mouth_open_ratio(_mouth_obs(0.15))
    assert open_ > closed
    # iod is 0.3, gap 0.15 -> ratio 0.5
    assert open_ == pytest.approx(0.5, abs=1e-5)


def test_mouth_open_ratio_prefers_blendshape() -> None:
    landmarks = np.zeros((478, 3), dtype=np.float32)
    obs = FaceObservation(landmarks, HeadPose(0, 0, 0), {"jawOpen": 0.42}, 1.0)
    assert mouth_open_ratio(obs) == pytest.approx(0.42)


def test_aggregate_speaker_segments_merges_and_filters() -> None:
    items = [
        (0.0, 0.2, "alice", 0.9),
        (0.2, 0.4, "alice", 0.7),
        (0.4, 0.45, "bob", 0.9),  # too short
        (0.45, 0.9, "alice", 0.8),
    ]
    segments = aggregate_speaker_segments(items, cue="mouth", min_duration=0.3, max_gap=0.2)
    assert [s.participant_id for s in segments] == ["alice", "alice"]
    assert segments[0].start_time == 0.0
    assert segments[0].end_time == 0.4
    assert segments[0].cue == "mouth"


# ------------------------------------------------------------- the dead branch

from lookout.face import MOUTH_BOTTOM, MOUTH_CORNERS, MOUTH_TOP  # noqa: E402


def _bare_observation(**kwargs) -> FaceObservation:
    """An observation with eyes but no mouth landmarks at all."""

    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[list(range(468, 473)), :2] = (0.35, 0.5)
    landmarks[list(range(473, 478)), :2] = (0.65, 0.5)
    return FaceObservation(landmarks, HeadPose(0, 0, 0), kwargs.get("blendshapes", {}), 1.0)


def test_jaw_open_survives_the_blendshape_filter() -> None:
    """The bug: MediaPipeFaceObserver kept only eyeLook* coefficients, so the
    preferred mouth signal was discarded at the source and the branch reading it
    had never run outside tests."""

    from lookout.face import _KEPT_BLENDSHAPES

    assert "jawOpen" in _KEPT_BLENDSHAPES


def test_the_blendshape_is_preferred_when_present() -> None:
    assert mouth_open_ratio(_bare_observation(blendshapes={"jawOpen": 0.42})) == pytest.approx(
        0.42
    )


def test_an_observer_without_a_mouth_declines_rather_than_guesses() -> None:
    """Measuring between two zeroed landmarks returns a number that looks like a
    measurement. The absence of a cue is not a closed mouth."""

    assert mouth_open_ratio(_bare_observation()) is None


def test_the_lip_gap_is_used_when_the_blendshape_is_absent() -> None:
    observation = _bare_observation()
    observation.landmarks[MOUTH_TOP, :2] = (0.5, 0.70)
    observation.landmarks[MOUTH_BOTTOM, :2] = (0.5, 0.85)

    ratio = mouth_open_ratio(observation)
    assert ratio is not None
    assert ratio == pytest.approx(0.5, abs=1e-5)  # gap 0.15 over iod 0.3


def test_corner_separation_is_the_last_resort() -> None:
    """Weaker than lip gap — it widens with a smile — but real, and some
    observers report only the corners."""

    observation = _bare_observation()
    observation.landmarks[MOUTH_CORNERS[0], :2] = (0.40, 0.8)
    observation.landmarks[MOUTH_CORNERS[1], :2] = (0.61, 0.8)

    ratio = mouth_open_ratio(observation)
    assert ratio is not None
    assert ratio == pytest.approx(0.7, abs=1e-4)  # separation 0.21 over iod 0.3


def test_the_lip_gap_wins_over_the_corners() -> None:
    observation = _bare_observation()
    observation.landmarks[MOUTH_TOP, :2] = (0.5, 0.70)
    observation.landmarks[MOUTH_BOTTOM, :2] = (0.5, 0.85)
    observation.landmarks[MOUTH_CORNERS[0], :2] = (0.40, 0.8)
    observation.landmarks[MOUTH_CORNERS[1], :2] = (0.61, 0.8)

    assert mouth_open_ratio(observation) == pytest.approx(0.5, abs=1e-5)


def test_a_degenerate_face_declines() -> None:
    flat = FaceObservation(np.zeros((478, 3), dtype=np.float32), HeadPose(0, 0, 0), {}, 1.0)
    assert mouth_open_ratio(flat) is None
