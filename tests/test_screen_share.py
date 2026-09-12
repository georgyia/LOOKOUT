"""Screen-share layouts.

A gallery is not the only layout a call takes. When someone shares their screen
the frame becomes one large content region with a column of small participant
tiles down an edge — cells nothing like equal, so the grid detector correctly
refuses it and, before this, the whole stretch read as no layout at all.

That cost two things: the filmstrip participants were reported absent rather
than present, and the shared document was not an attributable target even though
it is the most likely thing everyone was looking at.
"""

import numpy as np
import pytest

from lookout.attribution import attribute_point
from lookout.layout import DetectionParams, detect_shared_content, detect_tiles
from lookout.models import GazePoint, Layout, LayoutSource, Region, RegionKind
from tests.test_grid_detection import _cell_content

pytest.importorskip("cv2")


def _screen_share(
    tiles: int = 5,
    side: str = "right",
    band_fraction: float = 0.13,
    w: int = 854,
    h: int = 480,
    document: bool = True,
) -> np.ndarray:
    """A shared document with a filmstrip of participant thumbnails down one edge."""

    image = np.zeros((h, w, 3), dtype=np.uint8)
    band_w = int(band_fraction * w)
    content_w = w - band_w
    content_x = 0 if side == "right" else band_w
    band_x = content_w if side == "right" else 0

    # Shared content: pale, with horizontal rules, like a document.
    image[:, content_x : content_x + content_w] = 232
    if document:
        for y in range(40, h - 20, 26):
            image[y : y + 2, content_x + 20 : content_x + content_w - 20] = 150

    tile_h = h // (tiles + 2)
    top = tile_h  # a toolbar above the strip, as clients draw
    for index in range(tiles):
        y0, y1 = top + index * tile_h, top + (index + 1) * tile_h
        tint = np.array([50 + 30 * index, 90, 140 - 15 * index], dtype=np.float32)
        image[y0:y1, band_x : band_x + band_w] = _cell_content(
            y1 - y0, band_w, tint, seed=index
        )
    return image


def _kinds(tiles) -> dict[str, int]:
    counted: dict[str, int] = {}
    for tile in tiles:
        counted[tile.kind.value] = counted.get(tile.kind.value, 0) + 1
    return counted


@pytest.mark.parametrize("side", ["right", "left"])
def test_a_screen_share_yields_content_plus_a_filmstrip(side: str) -> None:
    tiles = detect_shared_content(_screen_share(tiles=5, side=side))
    counted = _kinds(tiles)

    assert counted["shared_content"] == 1
    assert counted["participant"] >= 3

    content = next(t for t in tiles if t.kind is RegionKind.SHARED_CONTENT)
    assert content.width > 0.7
    assert content.height == 1.0


def test_the_filmstrip_sits_on_the_edge_the_content_does_not() -> None:
    right = detect_shared_content(_screen_share(side="right"))
    content = next(t for t in right if t.kind is RegionKind.SHARED_CONTENT)
    strip = [t for t in right if t.kind is RegionKind.PARTICIPANT]

    assert content.x == 0.0
    assert all(t.x > content.width - 0.01 for t in strip)
    assert all(t.width < 0.3 for t in strip)


def test_a_document_without_a_filmstrip_invents_no_participants() -> None:
    """A content region alone is still better than nothing: it makes
    shared_content an available answer. It must not manufacture people."""

    plain = _screen_share(tiles=0)
    assert detect_shared_content(plain) == ()


def test_content_rules_are_not_read_as_filmstrip_tiles() -> None:
    """A document's horizontal rules recur down a narrow margin too. Requiring a
    recurring tile size is what separates them."""

    document_only = np.full((480, 854, 3), 232, dtype=np.uint8)
    for y in range(40, 460, 26):
        document_only[y : y + 2, 20:834] = 150
    assert detect_shared_content(document_only) == ()


def test_a_gallery_is_not_read_as_a_screen_share() -> None:
    from tests.test_grid_detection import _abutting_grid

    tiles = detect_tiles(_abutting_grid(3, 3))
    assert _kinds(tiles) == {"participant": 9}


def test_detect_tiles_prefers_a_gallery_and_falls_back_to_a_share() -> None:
    from tests.test_grid_detection import _abutting_grid

    assert _kinds(detect_tiles(_abutting_grid(2, 2))) == {"participant": 4}

    shared = _kinds(detect_tiles(_screen_share(tiles=5)))
    assert shared["shared_content"] == 1
    assert shared["participant"] >= 3


def test_a_band_too_wide_to_be_a_filmstrip_is_rejected() -> None:
    assert detect_shared_content(_screen_share(tiles=5, band_fraction=0.45)) == ()


def test_the_band_width_bounds_are_configurable() -> None:
    wide = _screen_share(tiles=5, band_fraction=0.35)
    assert detect_shared_content(wide) == ()
    lenient = DetectionParams(filmstrip_max_fraction=0.4)
    assert detect_shared_content(wide, lenient) != ()


def test_gaze_on_the_shared_region_is_attributed_to_it() -> None:
    """The point of describing the layout: the document becomes an answer.

    Before, gaze landing there produced `unknown`, which is indistinguishable
    from the pipeline having failed."""

    layout = Layout(
        "bob",
        (
            Region(RegionKind.SHARED_CONTENT, 0.0, 0.0, 0.87, 1.0),
            Region(RegionKind.PARTICIPANT, 0.87, 0.2, 0.13, 0.12, "slot_0"),
        ),
        0.0,
        LayoutSource.ASSUMED_SHARED,
    )

    on_document = attribute_point(GazePoint(0.0, "bob", 0.4, 0.5, 0.9), layout)
    assert on_document.target == "shared_content"
    assert on_document.reason == "center hit"

    on_person = attribute_point(GazePoint(0.0, "bob", 0.93, 0.26, 0.9), layout)
    assert on_person.target == "slot_0"
