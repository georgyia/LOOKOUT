"""Tile identity: stable slot ids and optional name-label linking.

Detection (:mod:`lookout.layout`) yields anonymous tiles. Identity turns them
into :class:`~lookout.models.Region` objects with participant ids, and assembles
a viewer-relative :class:`~lookout.models.Layout` per interval.

Within one interval, identity is the reading-order slot (`slot_0`, `slot_1`, ...).

Across intervals an id is carried only where something supports it. Geometry
supports it when a tile stays in nearly the same place, which covers a
participant joining or leaving a gallery. When the screen is reshaped — a
gallery becoming a screen share, say — nothing links the old tiles to the new
ones, so fresh ids are minted rather than the old ones being reused with a new
meaning. Asserting that two tiles in different layouts are the same person is an
invention unless something supports it, and the alternative is arithmetic over
two different people.

An optional :class:`TextReader` reads name labels, which links across arbitrary
changes when available. Face recognition is intentionally excluded here; see the
re-identification issue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .frames import Image
from .layout import LayoutInterval, Tile
from .models import Layout, LayoutSource, Region, RegionKind

__all__ = [
    "TextReader",
    "TesseractReader",
    "SlotLinker",
    "IdentityBreak",
    "read_tile_names",
    "build_layout",
    "build_layouts",
    "identity_breaks",
]

DEFAULT_CONTINUITY = 0.5
"""Overlap a tile needs with one in the previous layout to keep its id."""


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


@dataclass(frozen=True)
class IdentityBreak:
    """A layout change across which identity could not be carried."""

    at_time: float
    carried: tuple[str, ...]
    introduced: tuple[str, ...]
    ended: tuple[str, ...]

    @property
    def total(self) -> bool:
        """Whether nothing at all survived the change."""

        return not self.carried


def _overlap(a: Region, b: Region) -> float:
    """Intersection over union of two normalized rectangles."""

    x0, x1 = max(a.x, b.x), min(a.x + a.width, b.x + b.width)
    y0, y1 = max(a.y, b.y), min(a.y + a.height, b.y + b.height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0


def _carry_ids(
    regions: list[Region],
    previous: tuple[Region, ...],
    threshold: float,
) -> dict[int, str]:
    """Map region index to a carried id, best overlap first.

    Greedy rather than optimal: tiles rarely swap places, and a wrong match here
    is worse than no match, so each previous id is claimed once by whichever new
    tile overlaps it most.
    """

    scored = sorted(
        (
            (_overlap(region, old), index, old.participant_id)
            for index, region in enumerate(regions)
            for old in previous
            if old.participant_id is not None
        ),
        key=lambda item: -item[0],
    )
    carried: dict[int, str] = {}
    claimed: set[str] = set()
    for score, index, old_id in scored:
        if score < threshold or index in carried or old_id in claimed:
            continue
        carried[index] = old_id
        claimed.add(old_id)
    return carried


def identity_breaks(layouts: list[Layout]) -> tuple[IdentityBreak, ...]:
    """Where identity was carried across a layout change, and where it was not."""

    breaks: list[IdentityBreak] = []
    for previous, current in zip(layouts[:-1], layouts[1:], strict=True):
        before = {
            r.participant_id for r in previous.participant_regions() if r.participant_id
        }
        after = {r.participant_id for r in current.participant_regions() if r.participant_id}
        introduced = after - before
        ended = before - after
        if not introduced and not ended:
            continue
        breaks.append(
            IdentityBreak(
                at_time=current.start_time,
                carried=tuple(sorted(before & after)),
                introduced=tuple(sorted(introduced)),
                ended=tuple(sorted(ended)),
            )
        )
    return tuple(breaks)


def build_layouts(
    intervals: list[LayoutInterval],
    *,
    viewer_id: str,
    source: LayoutSource,
    names_per_interval: list[list[str | None]] | None = None,
    linker: SlotLinker | None = None,
    continuity: float = DEFAULT_CONTINUITY,
) -> list[Layout]:
    """Build one :class:`Layout` per interval, carrying identity where supported.

    An id is carried across a layout change when the tile stays in nearly the
    same place, which covers someone joining or leaving a gallery. Otherwise a
    fresh id is minted: reusing ``slot_0`` for a tile that is now somewhere else
    entirely makes every per-participant number a sum over two people.
    """

    layouts: list[Layout] = []
    previous: tuple[Region, ...] = ()
    minted = 0

    for index, interval in enumerate(intervals):
        names = names_per_interval[index] if names_per_interval is not None else None
        layout = build_layout(
            interval.tiles,
            viewer_id=viewer_id,
            source=source,
            start_time=interval.start_time,
            end_time=interval.end_time,
            names=names,
            linker=linker,
        )

        if linker is None and names is None:
            participants = list(layout.participant_regions())
            carried = _carry_ids(participants, previous, continuity)
            renamed: list[Region] = []
            for position, region in enumerate(participants):
                if position in carried:
                    participant_id = carried[position]
                else:
                    participant_id = f"slot_{minted}"
                    minted += 1
                renamed.append(
                    Region(
                        RegionKind.PARTICIPANT,
                        region.x,
                        region.y,
                        region.width,
                        region.height,
                        participant_id,
                    )
                )
            others = [r for r in layout.regions if r.kind is not RegionKind.PARTICIPANT]
            layout = Layout(
                viewer_id, tuple(renamed + others), layout.start_time, source, layout.end_time
            )

        layouts.append(layout)
        previous = layout.participant_regions()
    return layouts
