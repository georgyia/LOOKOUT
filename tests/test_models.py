import math

import pytest

from lookout.models import (
    Attribution,
    GazeDirection,
    GazeEvent,
    GazePoint,
    HeadPose,
    Layout,
    LayoutSource,
    Region,
    RegionKind,
)

POSE = HeadPose(0.0, 0.0, 0.0)


def test_gaze_direction_validates_confidence_and_timestamp() -> None:
    GazeDirection(1.0, "bob", 0.1, -0.2, POSE, 0.9)
    with pytest.raises(ValueError):
        GazeDirection(1.0, "bob", 0.1, -0.2, POSE, 1.5)
    with pytest.raises(ValueError):
        GazeDirection(-1.0, "bob", 0.1, -0.2, POSE, 0.9)


def test_gaze_direction_rejects_nonfinite_angles() -> None:
    with pytest.raises(ValueError):
        GazeDirection(1.0, "bob", math.inf, 0.0, POSE, 0.9)


def test_angle_sign_convention_is_pinned() -> None:
    # The convention (fixed in models.py) is: +yaw -> larger screen x,
    # +pitch -> smaller screen y. A minimal reference mapping encodes it so the
    # meaning cannot drift without breaking this test.
    def reference(yaw: float, pitch: float) -> tuple[float, float]:
        return (0.5 + yaw, 0.5 - pitch)

    right_x, _ = reference(0.2, 0.0)
    left_x, _ = reference(-0.2, 0.0)
    assert right_x > left_x

    _, up_y = reference(0.0, 0.2)
    _, down_y = reference(0.0, -0.2)
    assert up_y < down_y


def test_region_participant_id_rules() -> None:
    Region(RegionKind.PARTICIPANT, 0.0, 0.0, 0.5, 0.5, "alice")
    Region(RegionKind.SHARED_CONTENT, 0.0, 0.0, 0.5, 0.5)
    with pytest.raises(ValueError):
        Region(RegionKind.PARTICIPANT, 0.0, 0.0, 0.5, 0.5)
    with pytest.raises(ValueError):
        Region(RegionKind.UI, 0.0, 0.0, 0.5, 0.5, "alice")


def test_region_bounds_are_validated() -> None:
    with pytest.raises(ValueError):
        Region(RegionKind.SHARED_CONTENT, 0.8, 0.0, 0.5, 0.5)
    with pytest.raises(ValueError):
        Region(RegionKind.SHARED_CONTENT, 0.0, 0.0, 0.0, 0.5)


def test_region_geometry() -> None:
    region = Region(RegionKind.PARTICIPANT, 0.2, 0.4, 0.4, 0.2, "alice")
    assert region.contains(0.3, 0.5)
    assert not region.contains(0.9, 0.5)
    assert region.center == pytest.approx((0.4, 0.5))
    # Nearest edge to the center is the top/bottom (half-height 0.1).
    assert region.edge_distance(0.4, 0.5) == pytest.approx(0.1)
    assert region.edge_distance(0.9, 0.5) < 0


def test_layout_interval_and_queries() -> None:
    alice = Region(RegionKind.PARTICIPANT, 0.0, 0.0, 0.5, 1.0, "alice")
    share = Region(RegionKind.SHARED_CONTENT, 0.5, 0.0, 0.5, 1.0)
    layout = Layout("bob", (alice, share), 10.0, LayoutSource.ASSUMED_SHARED, 20.0)

    assert not layout.active_at(9.9)
    assert layout.active_at(10.0)
    assert not layout.active_at(20.0)
    assert layout.participant_regions() == (alice,)
    assert layout.regions_containing(0.25, 0.5) == (alice,)
    assert layout.regions_containing(0.75, 0.5) == (share,)


def test_attribution_and_event_validation() -> None:
    Attribution("alice", 0.8, "center hit", LayoutSource.ASSUMED_SHARED)
    with pytest.raises(ValueError):
        Attribution("", 0.8, "empty", LayoutSource.ASSUMED_SHARED)

    event = GazeEvent("bob", "alice", 1.0, 3.5, 0.7, LayoutSource.ASSUMED_SHARED)
    assert event.duration == pytest.approx(2.5)
    with pytest.raises(ValueError):
        GazeEvent("bob", "alice", 3.0, 1.0, 0.7, LayoutSource.ASSUMED_SHARED)


def test_gaze_point_still_validates() -> None:
    GazePoint(0.0, "bob", 0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        GazePoint(0.0, "bob", 1.2, 0.5, 0.5)
