"""Galleries whose tiles abut.

The background-difference detector assumes visible gutters. Many clients draw
tiles edge to edge, and then the whole gallery is a single connected component —
which is why the example recording could only be analyzed by a script that
imposed a grid by hand.
"""

import numpy as np
import pytest

from lookout.layout import DetectionParams, axis_separators, detect_grid, detect_tiles
from lookout.models import RegionKind

cv2 = pytest.importorskip("cv2")


def _cell_content(height: int, width: int, tint: np.ndarray, seed: int) -> np.ndarray:
    """Something that looks like a webcam tile: smooth, with a subject in it.

    Real video is locally smooth. Filling cells with per-pixel noise instead
    would raise the gradient floor everywhere and hide the very boundaries the
    detector looks for — a property of the fixture, not of galleries.
    """

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    shade = 18.0 * (yy / max(height, 1)) + 12.0 * (xx / max(width, 1))
    cell = tint[None, None, :] + shade[:, :, None]

    # A darker blob standing in for a head and shoulders, placed differently in
    # each cell. Identical placement would align blob edges across the whole
    # frame and mimic the boundaries the detector looks for.
    cy = height * float(rng.uniform(0.42, 0.68))
    cx = width * float(rng.uniform(0.35, 0.65))
    radius = min(height, width) * 0.3
    blob = ((yy - cy) ** 2 + (xx - cx) ** 2) < radius**2
    cell[blob] -= 45.0
    cell += rng.normal(0, 2.0, size=cell.shape)  # mild sensor grain
    return np.clip(cell, 0, 255).astype(np.uint8)


def _abutting_grid(
    rows: int,
    cols: int,
    empty: set[tuple[int, int]] | None = None,
    w: int = 640,
    h: int = 360,
    seed: int = 0,
) -> np.ndarray:
    """A gallery with no gutters: each cell is distinct video content."""

    image = np.zeros((h, w, 3), dtype=np.uint8)
    empty = empty or set()
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(r * h / rows), int((r + 1) * h / rows)
            x0, x1 = int(c * w / cols), int((c + 1) * w / cols)
            if (r, c) in empty:
                image[y0:y1, x0:x1] = 8  # a dark, flat slot
                continue
            tint = np.array([40 + 60 * r, 70 + 50 * c, 150 - 30 * r], dtype=np.float32)
            image[y0:y1, x0:x1] = _cell_content(y1 - y0, x1 - x0, tint, seed + 7 * r + c)
    return image


def _kinds(tiles) -> set[RegionKind]:
    return {t.kind for t in tiles}


@pytest.mark.parametrize("rows,cols", [(2, 2), (3, 3), (2, 3), (1, 2), (3, 2)])
def test_a_gallery_without_gutters_is_recovered(rows: int, cols: int) -> None:
    tiles = detect_grid(_abutting_grid(rows, cols))
    assert len(tiles) == rows * cols
    assert _kinds(tiles) == {RegionKind.PARTICIPANT}


def test_cells_tile_the_frame_without_overlap() -> None:
    tiles = detect_grid(_abutting_grid(3, 3))
    assert sum(t.width * t.height for t in tiles) == pytest.approx(1.0, abs=0.02)
    for tile in tiles:
        assert 0.0 <= tile.x < 1.0
        assert 0.0 <= tile.y < 1.0


def test_empty_slots_are_not_reported_as_participants() -> None:
    """A 3x3 with two empty slots holds seven participants, not nine."""

    tiles = detect_grid(_abutting_grid(3, 3, empty={(2, 0), (2, 2)}))
    assert len(tiles) == 7


def test_ordinary_content_yields_no_grid() -> None:
    """An invented layout misattributes every subsequent gaze, which is worse
    than having no layout at all."""

    single = _cell_content(360, 640, np.array([90.0, 100.0, 110.0]), seed=11)
    assert detect_grid(single) == ()

    flat = np.full((360, 640, 3), 90, dtype=np.uint8)
    assert detect_grid(flat) == ()


def test_an_irregular_split_is_rejected_rather_than_accepted() -> None:
    """Gallery cells are near-equal; a lopsided split is content, not a grid."""

    image = np.zeros((360, 640, 3), dtype=np.uint8)
    # One narrow strip and one wide one: a real boundary, but not a gallery.
    image[:, :80] = _cell_content(360, 80, np.array([50.0, 60.0, 70.0]), seed=21)
    image[:, 80:] = _cell_content(360, 560, np.array([170.0, 180.0, 190.0]), seed=22)
    assert detect_grid(image) == ()


def test_uniformity_tolerance_is_configurable() -> None:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:, :240] = _cell_content(360, 240, np.array([50.0, 60.0, 70.0]), seed=31)
    image[:, 240:] = _cell_content(360, 400, np.array([170.0, 180.0, 190.0]), seed=32)

    assert detect_grid(image) == ()
    lenient = DetectionParams(cell_uniformity=0.6)
    assert len(detect_grid(image, lenient)) == 2


def test_a_border_is_not_counted_twice() -> None:
    """Boundaries are a few pixels wide; each collapses to one separator."""

    gray = cv2.cvtColor(_abutting_grid(1, 3), cv2.COLOR_BGR2GRAY).astype(np.float32)
    assert len(axis_separators(gray, 1, DetectionParams())) == 2


def test_the_frame_edge_is_not_a_separator() -> None:
    gray = cv2.cvtColor(_abutting_grid(1, 2), cv2.COLOR_BGR2GRAY).astype(np.float32)
    separators = axis_separators(gray, 1, DetectionParams())
    assert separators
    assert all(0 < s < gray.shape[1] - 1 for s in separators)


# ----------------------------------------------------------- integration


def test_detect_tiles_falls_back_only_when_it_has_to() -> None:
    """The gutter detector stays first choice: it distinguishes shared content
    from participants, which the grid detector cannot."""

    from tests.test_layout import _render_grid

    gutters, _ = _render_grid(2, 2)
    assert len(detect_tiles(gutters)) == 4

    abutting = _abutting_grid(2, 2)
    assert len(detect_tiles(abutting)) == 4


def test_detect_tiles_recovers_a_gallery_that_used_to_collapse() -> None:
    """The case that forced the example recording onto a hand-imposed grid."""

    image = _abutting_grid(3, 3)
    assert len(detect_tiles(image)) == 9


def test_a_whole_frame_region_does_not_suppress_the_fallback() -> None:
    """A collapsing frame often yields one whole-frame region called shared
    content. Counting that as a result would suppress the fallback exactly when
    it is needed, so the test is on participant tiles."""

    tiles = detect_tiles(_abutting_grid(3, 3))
    assert all(t.kind is RegionKind.PARTICIPANT for t in tiles)
    assert len(tiles) == 9
