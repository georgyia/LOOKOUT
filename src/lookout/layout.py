"""Gallery layout detection and layout-change segmentation.

Detection is deliberately model-free: a gallery view is tiles of video drawn on
a near-uniform background, so tiles are the connected foreground components with
a plausible aspect ratio. A single dominant region is treated as shared content
(a spotlight or a screen share).

Detection produces geometry only (:class:`Tile`); identity assignment happens in
:mod:`lookout.identity`. Segmentation (:func:`segment_layouts`) is pure and turns
a per-frame tile stream into stable intervals, ignoring brief flicker.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .frames import Image
from .models import RegionKind

__all__ = [
    "Tile",
    "LayoutInterval",
    "DetectionParams",
    "detect_tiles",
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
