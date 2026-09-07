import pytest

from lookout.attribution import attribute_point, attribute_target
from lookout.models import (
    GazePoint,
    GazeTarget,
    Layout,
    LayoutSource,
    Region,
    RegionKind,
)


def _layout(viewer: str = "bob", *, source: LayoutSource = LayoutSource.ASSUMED_SHARED) -> Layout:
    regions = (
        Region(RegionKind.PARTICIPANT, 0.0, 0.0, 0.5, 1.0, "alice"),
        Region(RegionKind.PARTICIPANT, 0.5, 0.0, 0.5, 1.0, "carol"),
    )
    return Layout(viewer, regions, 0.0, source)


def test_center_hit_attributes_to_participant() -> None:
    result = attribute_point(GazePoint(1.0, "bob", 0.25, 0.5, 0.9), _layout())
    assert result.target == "alice"
    assert result.reason == "center hit"
    assert result.layout_source is LayoutSource.ASSUMED_SHARED
    assert 0.0 < result.confidence <= 1.0


def test_point_outside_all_regions_is_unknown() -> None:
    tall = Layout(
        "bob",
        (Region(RegionKind.PARTICIPANT, 0.0, 0.0, 0.4, 0.4, "alice"),),
        0.0,
        LayoutSource.ASSUMED_SHARED,
    )
    result = attribute_point(GazePoint(1.0, "bob", 0.9, 0.9, 0.9), tall)
    assert result.target == GazeTarget.UNKNOWN.value
    assert result.reason == "no region"


def test_near_border_is_low_confidence() -> None:
    # Just inside alice's right border: tiny margin relative to tile size.
    result = attribute_point(GazePoint(1.0, "bob", 0.495, 0.5, 0.9), _layout())
    assert result.target == GazeTarget.LOW_CONFIDENCE.value
    assert "alice" in result.reason
    assert result.confidence < 0.15


def test_point_on_shared_edge_is_ambiguous() -> None:
    # x = 0.5 is contained by both adjacent tiles.
    result = attribute_point(GazePoint(1.0, "bob", 0.5, 0.5, 0.9), _layout())
    assert result.target == GazeTarget.UNKNOWN.value
    assert result.reason == "ambiguous overlap"


def test_self_view_is_flagged() -> None:
    layout = Layout(
        "alice",
        (Region(RegionKind.PARTICIPANT, 0.0, 0.0, 1.0, 1.0, "alice"),),
        0.0,
        LayoutSource.ASSUMED_SHARED,
    )
    result = attribute_point(GazePoint(1.0, "alice", 0.5, 0.5, 0.9), layout)
    assert result.target == "alice"
    assert result.reason == "self"


def test_shared_content_is_a_non_participant_target() -> None:
    layout = Layout(
        "bob",
        (Region(RegionKind.SHARED_CONTENT, 0.0, 0.0, 1.0, 1.0),),
        0.0,
        LayoutSource.ASSUMED_SHARED,
    )
    result = attribute_point(GazePoint(1.0, "bob", 0.5, 0.5, 0.9), layout)
    assert result.target == "shared_content"


def test_time_varying_layout_selection_is_the_callers_choice() -> None:
    # attribute_point uses the layout it is given; the caller selects by time.
    early = _layout(source=LayoutSource.ASSUMED_SHARED)
    result = attribute_point(GazePoint(1.0, "bob", 0.75, 0.5, 0.9), early)
    assert result.target == "carol"


def test_attribute_target_passthrough() -> None:
    result = attribute_target(GazeTarget.OFF_SCREEN, _layout(), confidence=0.8)
    assert result.target == "off_screen"
    assert result.confidence == pytest.approx(0.8)
    assert result.layout_source is LayoutSource.ASSUMED_SHARED
