"""Gallery layout detection and layout-change segmentation.

Detection is deliberately model-free and has two strategies.

The first assumes a gallery view is tiles of video drawn on a near-uniform
background, so tiles are the connected foreground components with a plausible
aspect ratio. A single dominant region is treated as shared content (a spotlight
or a screen share).

That assumption fails when tiles abut: with no gutter between them the whole
gallery is one component. The fallback recovers the grid from its structure
instead. A tile border is a step that recurs along *almost every* line of the
frame; an edge inside a tile appears only on the lines that cross it. Measuring
how consistently a step recurs, rather than how strong it is on average,
separates the two cleanly — a border scores above 0.9 where in-tile content
rarely clears 0.1.

Detection produces geometry only (:class:`Tile`); identity assignment happens in
:mod:`lookout.identity`. Segmentation (:func:`segment_layouts`) is pure and turns
a per-frame tile stream into stable intervals, ignoring brief flicker.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .frames import Image
from .models import RegionKind

if TYPE_CHECKING:  # numpy is imported lazily below, as cv2 is
    import numpy as np
    from numpy.typing import NDArray

__all__ = [
    "Tile",
    "LayoutInterval",
    "DetectionParams",
    "detect_tiles",
    "detect_grid",
    "axis_separators",
    "segment_layouts",
]


@dataclass(frozen=True)
class Tile:
    """A detected rectangle in normalized coordinates, without identity.

    ``kind`` is either ``PARTICIPANT`` (a person tile) or ``SHARED_CONTENT``.
    """

    kind: RegionKind
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class DetectionParams:
    """Tunable thresholds for :func:`detect_tiles`."""

    background_diff: int = 18
    min_area_fraction: float = 0.01
    fill_ratio: float = 0.6
    aspect_ratios: tuple[float, ...] = (16 / 9, 4 / 3, 1.0)
    aspect_tolerance: float = 0.25
    shared_area_fraction: float = 0.45
    separator_consistency: float = 0.5
    """Fraction of lines on which a step must recur to count as a tile border.

    A border recurs on nearly every line; content inside a tile appears only on
    the lines crossing it. The gap between the two is wide, so this threshold is
    not delicate."""

    separator_prominence: float = 6.0
    """How far a step must exceed its line's typical variation to count at all."""

    min_cell_fraction: float = 0.12
    """Smallest cell, as a fraction of the frame. Also the minimum spacing
    between separators, which keeps a tile border from being counted twice."""

    cell_uniformity: float = 0.15
    """How much cell sizes may vary before a candidate is content, not a grid.

    Gallery cells are near-equal. Without this the detector accepts high-contrast
    in-tile content as an irregular grid, which is worse than finding nothing."""

    empty_cell_variation: float = 6.0
    """Below this pixel standard deviation a cell holds no video."""


@dataclass(frozen=True)
class LayoutInterval:
    """A stable set of tiles valid over ``[start_time, end_time)``.

    ``end_time`` is ``None`` for the final interval (valid through the end of
    the stream).
    """

    start_time: float
    end_time: float | None
    tiles: tuple[Tile, ...]


def _estimate_background(image: Image) -> tuple[int, int, int]:
    import numpy as np

    border = np.concatenate(
        [
            image[0, :, :].reshape(-1, 3),
            image[-1, :, :].reshape(-1, 3),
            image[:, 0, :].reshape(-1, 3),
            image[:, -1, :].reshape(-1, 3),
        ]
    )
    median = np.median(border, axis=0)
    return (int(median[0]), int(median[1]), int(median[2]))


def _aspect_ok(ratio: float, params: DetectionParams) -> bool:
    return any(abs(ratio - r) <= params.aspect_tolerance * r for r in params.aspect_ratios)


def axis_separators(
    gray: NDArray[np.float32],
    axis: int,
    params: DetectionParams,
) -> list[int]:
    """Positions along ``axis`` where a boundary spans the whole frame.

    What distinguishes a tile border from content is not how strong the step is
    but how *consistently* it recurs: a border is present on nearly every line,
    while an edge inside one tile appears only on the lines crossing it. A large
    subject in a quiet frame can out-weigh a border on average gradient; it
    cannot out-recur one.
    """

    import numpy as np

    steps = np.abs(np.diff(gray, axis=axis))
    if steps.size == 0:
        return []

    # Judge each step against the typical variation of its own line, so a
    # high-contrast tile does not drown out a border beside a low-contrast one.
    scale = np.median(steps, axis=axis, keepdims=True) + 1.0
    energy = (steps > params.separator_prominence * scale).mean(axis=1 - axis)
    length = len(energy)

    spacing = max(1, int(params.min_cell_fraction * length))
    strong = np.where(energy > params.separator_consistency)[0]
    if not len(strong):
        return []

    # A border is a few pixels wide; collapse each run to its centre.
    groups: list[list[int]] = [[int(strong[0])]]
    for index in strong[1:]:
        if int(index) - groups[-1][-1] <= 2:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])

    candidates = [
        (int(sum(g) / len(g)), float(max(energy[i] for i in g)))
        for g in groups
        if spacing <= int(sum(g) / len(g)) <= length - spacing  # frame edge is not a separator
    ]

    # Most consistent first, not leftmost first. A weaker candidate that happens
    # to come earlier would otherwise claim the spacing budget and crowd out the
    # real border beside it.
    kept: list[int] = []
    for centre, _ in sorted(candidates, key=lambda item: -item[1]):
        if any(abs(centre - chosen) < spacing for chosen in kept):
            continue
        kept.append(centre)
    return sorted(kept)


def _uniform(cuts: list[int], length: int, tolerance: float) -> bool:
    """Whether ``cuts`` divide ``length`` into near-equal cells."""

    import numpy as np

    edges = [0, *cuts, length]
    sizes = np.diff(np.array(edges, dtype=float))
    if len(sizes) < 2:
        return True
    mean = float(sizes.mean())
    return mean > 0 and float(sizes.std()) / mean <= tolerance


def detect_grid(image: Image, params: DetectionParams | None = None) -> tuple[Tile, ...]:
    """Recover a gallery grid whose tiles abut, or return nothing.

    Returns nothing rather than a guess when the frame does not look like a
    regular grid: an invented layout misattributes every subsequent gaze, which
    is worse than having no layout at all.
    """

    import cv2
    import numpy as np

    params = params or DetectionParams()
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    rows = axis_separators(gray, 0, params)
    cols = axis_separators(gray, 1, params)
    if not rows and not cols:
        return ()
    if not _uniform(rows, height, params.cell_uniformity):
        return ()
    if not _uniform(cols, width, params.cell_uniformity):
        return ()

    y_edges = [0, *rows, height]
    x_edges = [0, *cols, width]

    tiles: list[Tile] = []
    for top, bottom in zip(y_edges[:-1], y_edges[1:], strict=True):
        for left, right in zip(x_edges[:-1], x_edges[1:], strict=True):
            # Measure emptiness inside the border, not across it. A boundary
            # row carries the neighbouring tile's brightness, and one such row
            # is enough to make an empty slot look like video.
            inset_y = max(2, int(0.03 * (bottom - top)))
            inset_x = max(2, int(0.03 * (right - left)))
            interior = gray[top + inset_y : bottom - inset_y, left + inset_x : right - inset_x]
            if interior.size == 0:
                continue
            # An empty slot is flat; a tile carrying video is not.
            if float(interior.std()) < params.empty_cell_variation:
                continue
            tiles.append(
                Tile(
                    kind=RegionKind.PARTICIPANT,
                    x=left / width,
                    y=top / height,
                    width=(right - left) / width,
                    height=(bottom - top) / height,
                )
            )
    return tuple(tiles)


def detect_tiles(image: Image, params: DetectionParams | None = None) -> tuple[Tile, ...]:
    """Detect participant tiles and shared-content regions in one frame.

    Returns tiles in reading order (top-to-bottom, then left-to-right).
    """

    import cv2
    import numpy as np

    params = params or DetectionParams()
    height, width = image.shape[:2]
    frame_area = float(height * width)

    background = np.array(_estimate_background(image), dtype=np.int16)
    diff = np.abs(image.astype(np.int16) - background).max(axis=2)
    mask = (diff > params.background_diff).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

    tiles: list[Tile] = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < params.min_area_fraction * frame_area:
            continue
        if area < params.fill_ratio * (w * h):
            continue  # ragged component, not a solid tile
        ratio = w / h
        box_fraction = (w * h) / frame_area
        if box_fraction >= params.shared_area_fraction:
            kind = RegionKind.SHARED_CONTENT
        elif _aspect_ok(ratio, params):
            kind = RegionKind.PARTICIPANT
        else:
            continue
        tiles.append(
            Tile(
                kind=kind,
                x=x / width,
                y=y / height,
                width=w / width,
                height=h / height,
            )
        )

    # With no gutter between them the gallery is one component, so the
    # background-difference strategy collapses. Recover the grid from its
    # structure instead. The test is on participant tiles: a frame that
    # collapses often yields one whole-frame region called shared content, and
    # counting that as a result would suppress the fallback exactly when it is
    # needed.
    if sum(1 for t in tiles if t.kind is RegionKind.PARTICIPANT) <= 1:
        grid = detect_grid(image, params)
        if len(grid) > 1:
            return grid

    tiles.sort(key=lambda t: (round(t.y, 2), round(t.x, 2)))
    return tuple(tiles)


def _signature(tiles: Iterable[Tile]) -> tuple[tuple[str, int, int, int, int], ...]:
    """A coarse, order-independent fingerprint of a tile arrangement."""

    def quantize(value: float) -> int:
        return round(value * 50)  # 0.02 resolution

    entries = [
        (t.kind.value, quantize(t.x), quantize(t.y), quantize(t.width), quantize(t.height))
        for t in tiles
    ]
    return tuple(sorted(entries))


def segment_layouts(
    frames: Iterable[tuple[float, tuple[Tile, ...]]],
    min_stable_seconds: float = 0.5,
) -> list[LayoutInterval]:
    """Collapse a per-frame tile stream into stable layout intervals.

    A new arrangement must persist for at least ``min_stable_seconds`` before it
    replaces the current one, so a single-frame flicker never creates an
    interval. The change is dated to when the new arrangement first appeared.
    """

    committed_sig: tuple[tuple[str, int, int, int, int], ...] | None = None
    committed_start = 0.0
    committed_tiles: tuple[Tile, ...] = ()
    pending_sig: tuple[tuple[str, int, int, int, int], ...] | None = None
    pending_since = 0.0
    pending_tiles: tuple[Tile, ...] = ()
    intervals: list[LayoutInterval] = []

    for timestamp, tiles in frames:
        sig = _signature(tiles)
        if committed_sig is None:
            committed_sig, committed_start, committed_tiles = sig, timestamp, tiles
            pending_sig = None
            continue
        if sig == committed_sig:
            pending_sig = None
            continue
        if sig != pending_sig:
            pending_sig, pending_since, pending_tiles = sig, timestamp, tiles
        if timestamp - pending_since >= min_stable_seconds:
            intervals.append(LayoutInterval(committed_start, pending_since, committed_tiles))
            committed_sig, committed_start, committed_tiles = sig, pending_since, pending_tiles
            pending_sig = None

    if committed_sig is not None:
        intervals.append(LayoutInterval(committed_start, None, committed_tiles))
    return intervals
