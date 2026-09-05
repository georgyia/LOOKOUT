import numpy as np
import pytest

from lookout.layout import Tile, detect_tiles, segment_layouts
from lookout.models import RegionKind

_COLORS = [
    (200, 80, 80),
    (80, 200, 80),
    (80, 80, 200),
    (200, 200, 80),
    (200, 80, 200),
    (80, 200, 200),
    (200, 140, 60),
    (140, 60, 200),
    (60, 200, 140),
]


def _render_grid(
    rows: int,
    cols: int,
    filled: list[tuple[int, int]] | None = None,
    w: int = 640,
    h: int = 360,
    gutter: int = 10,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    image = np.full((h, w, 3), 16, dtype=np.uint8)
    cells = filled or [(r, c) for r in range(rows) for c in range(cols)]
    boxes: list[tuple[float, float, float, float]] = []
    for idx, (r, c) in enumerate(cells):
        x0 = int(c * w / cols) + gutter
        y0 = int(r * h / rows) + gutter
        x1 = int((c + 1) * w / cols) - gutter
        y1 = int((r + 1) * h / rows) - gutter
        image[y0:y1, x0:x1] = _COLORS[idx % len(_COLORS)]
        boxes.append((x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h))
    boxes.sort(key=lambda b: (round(b[1], 2), round(b[0], 2)))
    return image, boxes


def _assert_boxes_match(
    tiles: tuple[Tile, ...],
    boxes: list[tuple[float, float, float, float]],
) -> None:
    assert len(tiles) == len(boxes)
    for tile, box in zip(tiles, boxes, strict=True):
        assert (tile.x, tile.y, tile.width, tile.height) == pytest.approx(box, abs=0.02)


def test_detect_2x2_grid() -> None:
    pytest.importorskip("cv2")
    image, boxes = _render_grid(2, 2)
    tiles = detect_tiles(image)
    assert all(t.kind is RegionKind.PARTICIPANT for t in tiles)
    _assert_boxes_match(tiles, boxes)


def test_detect_3x3_grid() -> None:
    pytest.importorskip("cv2")
    image, boxes = _render_grid(3, 3)
    tiles = detect_tiles(image)
    assert len(tiles) == 9
    _assert_boxes_match(tiles, boxes)


def test_detect_uneven_last_row() -> None:
    pytest.importorskip("cv2")
    filled = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    image, boxes = _render_grid(2, 3, filled=filled)
    tiles = detect_tiles(image)
    assert len(tiles) == 5
    _assert_boxes_match(tiles, boxes)


def test_detect_shared_content_plus_tiles() -> None:
    pytest.importorskip("cv2")
    w, h = 640, 360
    image = np.full((h, w, 3), 16, dtype=np.uint8)
    # Dominant shared region on the left.
    image[20 : h - 20, 20 : int(0.7 * w)] = (180, 180, 180)
    # A column of three small participant tiles on the right.
    for i in range(3):
        y0 = 20 + i * 110
        image[y0 : y0 + 90, int(0.72 * w) : int(0.72 * w) + 150] = _COLORS[i]
    tiles = detect_tiles(image)
    kinds = sorted(t.kind.value for t in tiles)
    assert kinds.count("shared_content") == 1
    assert kinds.count("participant") == 3


def _ptiles(n: int) -> tuple[Tile, ...]:
    return tuple(Tile(RegionKind.PARTICIPANT, 0.1 * i, 0.0, 0.05, 0.05) for i in range(n))


def test_segment_layouts_ignores_flicker_and_detects_change() -> None:
    layout_a = _ptiles(2)
    layout_c = _ptiles(4)
    frames: list[tuple[float, tuple[Tile, ...]]] = []
    for k in range(10):  # 0.0 .. 0.9 stable A
        frames.append((round(0.1 * k, 1), layout_a))
    frames.append((1.0, ()))  # single-frame flicker
    for k in range(11, 20):  # 1.1 .. 1.9 back to A
        frames.append((round(0.1 * k, 1), layout_a))
    for k in range(20, 30):  # 2.0 .. 2.9 stable C
        frames.append((round(0.1 * k, 1), layout_c))

    intervals = segment_layouts(frames, min_stable_seconds=0.5)

    assert len(intervals) == 2
    assert intervals[0].tiles == layout_a
    assert intervals[0].start_time == pytest.approx(0.0)
    assert intervals[0].end_time == pytest.approx(2.0)
    assert intervals[1].tiles == layout_c
    assert intervals[1].start_time == pytest.approx(2.0)
    assert intervals[1].end_time is None
