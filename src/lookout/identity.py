"""Tile identity: stable slot ids and optional name-label linking.

Detection (:mod:`lookout.layout`) yields anonymous tiles. Identity turns them
into :class:`~lookout.models.Region` objects with participant ids, and assembles
a viewer-relative :class:`~lookout.models.Layout` per interval.

Within one interval, identity is the reading-order slot (`slot_0`, `slot_1`, ...).
Across intervals, an optional :class:`TextReader` reads the name label so the
same participant keeps one id even when their slot moves. Face recognition is
intentionally excluded here; see the re-identification issue.
"""

from __future__ import annotations

from typing import Protocol

from .frames import Image
from .layout import LayoutInterval, Tile
from .models import Layout, LayoutSource, Region, RegionKind

__all__ = [
    "TextReader",
    "TesseractReader",
    "SlotLinker",
    "read_tile_names",
    "build_layout",
    "build_layouts",
]


class TextReader(Protocol):
    """Reads text from an image crop. Implemented by a local OCR adapter."""

    def read(self, image: Image) -> str: ...


class TesseractReader:
    """Local OCR via Tesseract (requires the ``ocr`` extra).

    Kept intentionally thin: it adapts a single name-label crop to the
    :class:`TextReader` protocol and imports the backend lazily so the core
    never depends on it.
    """

    def __init__(self, lang: str = "eng") -> None:
        self._lang = lang

    def read(self, image: Image) -> str:
        import pytesseract

        text: str = pytesseract.image_to_string(image, lang=self._lang, config="--psm 7")
        return text.strip()


class SlotLinker:
    """Maps participant names to stable ids across layout intervals.

    A name resolves to the same id wherever it appears; an empty or missing
    name falls back to the interval-local slot id, so identity never fabricates
    a link it cannot support.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, str] = {}

    def resolve(self, name: str | None, fallback: str) -> str:
        if name is None:
            return fallback
        key = name.strip().casefold()
        if not key:
            return fallback
        return self._by_name.setdefault(key, name.strip())


def read_tile_names(
    image: Image,
    tiles: tuple[Tile, ...],
    reader: TextReader,
    label_fraction: float = 0.25,
) -> list[str | None]:
    """Read a name from the bottom label strip of each participant tile.

    Returns names aligned to the participant tiles in reading order; a blank
    read becomes ``None``.
    """

    height, width = image.shape[:2]
    names: list[str | None] = []
    for tile in tiles:
        if tile.kind is not RegionKind.PARTICIPANT:
            continue
        x0 = int(round(tile.x * width))
        x1 = int(round((tile.x + tile.width) * width))
        y1 = int(round((tile.y + tile.height) * height))
        strip_top = int(round((tile.y + tile.height * (1.0 - label_fraction)) * height))
        crop = image[strip_top:y1, x0:x1]
        text = reader.read(crop).strip()
        names.append(text or None)
    return names


def build_layout(
    tiles: tuple[Tile, ...],
    *,
    viewer_id: str,
    source: LayoutSource,
    start_time: float,
    end_time: float | None = None,
    names: list[str | None] | None = None,
    linker: SlotLinker | None = None,
) -> Layout:
    """Assemble a :class:`Layout` from detected tiles.

    ``names`` (aligned to participant tiles in reading order) and ``linker`` are
    optional; without them, participants get interval-local slot ids.
    """

    regions: list[Region] = []
    slot = 0
    for tile in tiles:
        if tile.kind is not RegionKind.PARTICIPANT:
            regions.append(Region(tile.kind, tile.x, tile.y, tile.width, tile.height))
            continue
        fallback = f"slot_{slot}"
        name = names[slot] if names is not None and slot < len(names) else None
        if linker is not None:
            participant_id = linker.resolve(name, fallback)
        else:
            participant_id = name or fallback
        regions.append(
            Region(
                RegionKind.PARTICIPANT,
                tile.x,
                tile.y,
                tile.width,
                tile.height,
                participant_id,
            )
        )
        slot += 1
    return Layout(viewer_id, tuple(regions), start_time, source, end_time)


def build_layouts(
    intervals: list[LayoutInterval],
    *,
    viewer_id: str,
    source: LayoutSource,
    names_per_interval: list[list[str | None]] | None = None,
    linker: SlotLinker | None = None,
) -> list[Layout]:
    """Build one :class:`Layout` per interval, sharing a linker so names carry
    identity across layout changes."""

    layouts: list[Layout] = []
    for index, interval in enumerate(intervals):
        names = names_per_interval[index] if names_per_interval is not None else None
        layouts.append(
            build_layout(
                interval.tiles,
                viewer_id=viewer_id,
                source=source,
                start_time=interval.start_time,
                end_time=interval.end_time,
                names=names,
                linker=linker,
            )
        )
    return layouts
