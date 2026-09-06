"""Speaker context from visual cues.

Two cues, both offline and visual: the active-speaker highlight many clients
draw around a tile, and mouth motion from face landmarks. Per-frame speaker
flags are aggregated into :class:`SpeakerSegment`s. Audio diarization is a
separate later issue; this module never uses audio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .face import FaceObservation
from .frames import Image
from .layout import Tile
from .models import RegionKind

__all__ = [
    "SpeakerSegment",
    "detect_highlighted_tile",
    "mouth_open_ratio",
    "aggregate_speaker_segments",
]

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
_EPS = 1e-9


@dataclass(frozen=True)
class SpeakerSegment:
    """A span during which a participant is estimated to be speaking."""

    participant_id: str
    start_time: float
    end_time: float
    confidence: float
    cue: str

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def _rect(image: Image, tile: Tile) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    x0 = int(round(tile.x * width))
    x1 = int(round((tile.x + tile.width) * width))
    y0 = int(round(tile.y * height))
    y1 = int(round((tile.y + tile.height) * height))
    return x0, y0, x1, y1


def _ring_and_interior_means(
    image: Image,
    tile: Tile,
    thickness_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = _rect(image, tile)
    region = image[y0:y1, x0:x1].astype(np.float64)
    h, w = region.shape[:2]
    t = max(1, int(round(min(h, w) * thickness_frac)))
    ring = np.concatenate(
        [
            region[:t, :, :].reshape(-1, 3),
            region[-t:, :, :].reshape(-1, 3),
            region[:, :t, :].reshape(-1, 3),
            region[:, -t:, :].reshape(-1, 3),
        ]
    )
    interior = region[t:-t, t:-t, :].reshape(-1, 3)
    if interior.size == 0:
        interior = region.reshape(-1, 3)
    return ring.mean(axis=0), interior.mean(axis=0)


def detect_highlighted_tile(
    image: Image,
    tiles: tuple[Tile, ...],
    thickness_frac: float = 0.06,
    threshold: float = 40.0,
) -> int | None:
    """Index (in reading order among participant tiles) of the highlighted tile.

    Scores each participant tile by how much its border ring differs from its
    interior; the active-speaker highlight stands out, uniform tiles score near
    zero. Returns ``None`` when no tile clears ``threshold``.
    """

    participants = [t for t in tiles if t.kind is RegionKind.PARTICIPANT]
    best_index: int | None = None
    best_score = threshold
    for index, tile in enumerate(participants):
        ring, interior = _ring_and_interior_means(image, tile, thickness_frac)
        score = float(np.abs(ring - interior).sum())
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def mouth_open_ratio(observation: FaceObservation) -> float:
    """Vertical mouth gap normalized by interocular distance.

    Prefers the MediaPipe ``jawOpen`` blendshape when present; otherwise uses the
    inner-lip landmark gap.
    """

    jaw = observation.blendshapes.get("jawOpen")
    if jaw is not None:
        return float(jaw)

    landmarks = observation.landmarks
    gap = abs(float(landmarks[MOUTH_BOTTOM, 1]) - float(landmarks[MOUTH_TOP, 1]))
    right = np.array(observation.right_iris)
    left = np.array(observation.left_iris)
    iod = float(np.linalg.norm(right - left))
    return gap / iod if iod > _EPS else 0.0


def aggregate_speaker_segments(
    items: list[tuple[float, float, str, float]],
    cue: str,
    min_duration: float = 0.4,
    max_gap: float = 0.5,
) -> list[SpeakerSegment]:
    """Merge per-frame speaker flags ``(start, end, participant_id, confidence)``
    into segments, dropping runs shorter than ``min_duration``."""

    ordered = sorted(items, key=lambda item: item[0])
    segments: list[SpeakerSegment] = []
    pid: str | None = None
    start = end = 0.0
    weight = weighted_conf = 0.0

    def flush() -> None:
        nonlocal pid
        if pid is not None and end - start >= min_duration:
            segments.append(SpeakerSegment(pid, start, end, weighted_conf / weight, cue))

    for item_start, item_end, participant, confidence in ordered:
        if pid == participant and item_start - end <= max_gap:
            end = max(end, item_end)
            w = max(item_end - item_start, _EPS)
            weight += w
            weighted_conf += w * confidence
            continue
        flush()
        pid = participant
        start, end = item_start, item_end
        weight = max(item_end - item_start, _EPS)
        weighted_conf = weight * confidence
    flush()
    return segments
