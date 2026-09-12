"""Identity across layout changes.

Participant ids were assigned by reading order within each interval,
independently, so a layout change silently reused `slot_0` for a different tile.
Every per-participant number — duration, coverage, diagnostics — is keyed by that
id and summed across the run, so the result was arithmetic over two people.
"""

import pytest

from lookout.identity import build_layouts, identity_breaks
from lookout.layout import LayoutInterval, Tile, _reading_order
from lookout.models import LayoutSource, RegionKind


def _tile(x: float, y: float, w: float = 0.33, h: float = 0.33) -> Tile:
    return Tile(RegionKind.PARTICIPANT, x, y, w, h)


def _interval(start: float, end: float | None, tiles: tuple[Tile, ...]) -> LayoutInterval:
    return LayoutInterval(start, end, tiles)


def _ids(layout) -> list[str]:
    return [r.participant_id for r in layout.participant_regions()]


def _build(intervals):
    return build_layouts(intervals, viewer_id="recording", source=LayoutSource.RECORDING)


# ------------------------------------------------------------- reading order


def test_a_row_is_ordered_left_to_right_despite_pixel_noise() -> None:
    """Sorting on rounded y alone is not reading order: two tiles in a row
    detected a pixel apart order by y, so bottom-right precedes bottom-left and
    slot numbering depends on noise."""

    jittered = [
        _tile(0.01, 0.02),
        _tile(0.51, 0.03),
        _tile(0.51, 0.52),  # bottom-right, detected one pixel higher
        _tile(0.01, 0.53),  # bottom-left
    ]
    ordered = _reading_order(jittered)
    assert [round(t.x, 2) for t in ordered] == [0.01, 0.51, 0.01, 0.51]


def test_rows_are_still_separated() -> None:
    grid = [_tile(c / 3, r / 3) for r in range(3) for c in range(3)]
    ordered = _reading_order(list(reversed(grid)))
    assert [(round(t.y, 2), round(t.x, 2)) for t in ordered] == [
        (round(r / 3, 2), round(c / 3, 2)) for r in range(3) for c in range(3)
    ]


def test_reading_order_of_nothing_is_nothing() -> None:
    assert _reading_order([]) == ()


# ---------------------------------------------------------------- continuity


def test_an_id_is_carried_when_the_tile_stays_put() -> None:
    """Someone joining a gallery must not renumber everyone else."""

    before = (_tile(0.0, 0.0), _tile(0.33, 0.0))
    after = (_tile(0.0, 0.0), _tile(0.33, 0.0), _tile(0.66, 0.0))
    first, second = _build([_interval(0.0, 10.0, before), _interval(10.0, None, after)])

    assert _ids(first) == ["slot_0", "slot_1"]
    assert _ids(second)[:2] == ["slot_0", "slot_1"]
    assert _ids(second)[2] not in {"slot_0", "slot_1"}


def test_a_departing_participant_does_not_renumber_the_rest() -> None:
    before = (_tile(0.0, 0.0), _tile(0.33, 0.0), _tile(0.66, 0.0))
    after = (_tile(0.0, 0.0), _tile(0.66, 0.0))
    first, second = _build([_interval(0.0, 10.0, before), _interval(10.0, None, after)])

    assert _ids(first) == ["slot_0", "slot_1", "slot_2"]
    assert _ids(second) == ["slot_0", "slot_2"]


def test_a_reshaped_screen_mints_new_ids_rather_than_reusing_them() -> None:
    """The case from the example recording: a 3x3 gallery becomes a screen share
    with a filmstrip, and slot_0 would otherwise mean a different person."""

    gallery = tuple(_tile(c / 3, r / 3) for r in range(3) for c in range(3))
    filmstrip = tuple(_tile(0.87, 0.2 + i * 0.13, 0.13, 0.12) for i in range(4))
    first, second = _build(
        [_interval(0.0, 150.0, gallery), _interval(150.0, None, filmstrip)]
    )

    assert set(_ids(first)).isdisjoint(_ids(second))
    assert len(set(_ids(first)) | set(_ids(second))) == 13


def test_ids_are_unique_across_a_whole_run() -> None:
    """Uniqueness is what stops per-participant numbers being summed over two
    different people."""

    gallery = tuple(_tile(c / 3, r / 3) for r in range(3) for c in range(2))
    filmstrip = tuple(_tile(0.9, 0.2 + i * 0.15, 0.1, 0.14) for i in range(3))
    layouts = _build(
        [
            _interval(0.0, 10.0, gallery),
            _interval(10.0, 20.0, filmstrip),
            _interval(20.0, None, gallery),
        ]
    )
    seen: dict[str, tuple[float, float]] = {}
    for layout in layouts:
        for region in layout.participant_regions():
            assert region.participant_id is not None
            position = (round(region.x, 2), round(region.y, 2))
            if region.participant_id in seen:
                assert seen[region.participant_id] == position, (
                    f"{region.participant_id} means two different positions"
                )
            seen[region.participant_id] = position


def test_a_partial_overlap_below_threshold_is_not_a_match() -> None:
    before = (_tile(0.0, 0.0, 0.3, 0.3),)
    after = (_tile(0.25, 0.25, 0.3, 0.3),)
    first, second = _build([_interval(0.0, 10.0, before), _interval(10.0, None, after)])
    assert _ids(first) != _ids(second)


def test_continuity_threshold_is_configurable() -> None:
    before = (_tile(0.0, 0.0, 0.3, 0.3),)
    after = (_tile(0.1, 0.0, 0.3, 0.3),)
    strict = build_layouts(
        [_interval(0.0, 10.0, before), _interval(10.0, None, after)],
        viewer_id="r",
        source=LayoutSource.RECORDING,
        continuity=0.9,
    )
    lenient = build_layouts(
        [_interval(0.0, 10.0, before), _interval(10.0, None, after)],
        viewer_id="r",
        source=LayoutSource.RECORDING,
        continuity=0.3,
    )
    assert _ids(strict[0]) != _ids(strict[1])
    assert _ids(lenient[0]) == _ids(lenient[1])


# -------------------------------------------------------------------- breaks


def test_breaks_report_what_survived_a_change() -> None:
    before = (_tile(0.0, 0.0), _tile(0.33, 0.0))
    after = (_tile(0.0, 0.0), _tile(0.66, 0.0))
    layouts = _build([_interval(0.0, 10.0, before), _interval(10.0, None, after)])

    (change,) = identity_breaks(layouts)
    assert change.at_time == 10.0
    assert change.carried == ("slot_0",)
    assert change.ended == ("slot_1",)
    assert len(change.introduced) == 1
    assert not change.total


def test_a_total_break_is_marked_as_such() -> None:
    gallery = tuple(_tile(c / 3, r / 3) for r in range(3) for c in range(3))
    filmstrip = tuple(_tile(0.87, 0.2 + i * 0.13, 0.13, 0.12) for i in range(4))
    layouts = _build([_interval(0.0, 150.0, gallery), _interval(150.0, None, filmstrip)])

    (change,) = identity_breaks(layouts)
    assert change.total
    assert change.carried == ()
    assert len(change.ended) == 9
    assert len(change.introduced) == 4


def test_an_unchanged_layout_reports_no_break() -> None:
    tiles = (_tile(0.0, 0.0), _tile(0.33, 0.0))
    layouts = _build([_interval(0.0, 10.0, tiles), _interval(10.0, None, tiles)])
    assert identity_breaks(layouts) == ()


def test_a_single_interval_has_no_breaks() -> None:
    layouts = _build([_interval(0.0, None, (_tile(0.0, 0.0),))])
    assert identity_breaks(layouts) == ()
    assert isinstance(identity_breaks(layouts), tuple)


def test_named_participants_are_left_to_the_linker() -> None:
    """Name labels link across arbitrary changes; geometry must not override them."""

    from lookout.identity import SlotLinker

    before = (_tile(0.0, 0.0), _tile(0.33, 0.0))
    after = (_tile(0.66, 0.0), _tile(0.0, 0.0))
    layouts = build_layouts(
        [_interval(0.0, 10.0, before), _interval(10.0, None, after)],
        viewer_id="r",
        source=LayoutSource.RECORDING,
        names_per_interval=[["Alice", "Bob"], ["Alice", "Bob"]],
        linker=SlotLinker(),
    )
    assert _ids(layouts[0]) == ["Alice", "Bob"]
    assert _ids(layouts[1]) == ["Alice", "Bob"]


@pytest.mark.parametrize("count", [1, 2, 5])
def test_shared_content_regions_survive_renaming(count: int) -> None:
    tiles = (Tile(RegionKind.SHARED_CONTENT, 0.0, 0.0, 0.87, 1.0),) + tuple(
        _tile(0.87, 0.2 + i * 0.13, 0.13, 0.12) for i in range(count)
    )
    (layout,) = _build([_interval(0.0, None, tiles)])
    kinds = [r.kind for r in layout.regions]
    assert kinds.count(RegionKind.SHARED_CONTENT) == 1
    assert kinds.count(RegionKind.PARTICIPANT) == count
