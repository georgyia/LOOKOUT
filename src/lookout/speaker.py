"""Speaker context from visual cues.

Two cues, both offline and visual: the active-speaker highlight many clients
draw around a tile, and mouth motion from face landmarks. The highlight is
client-specific — it depends on a client drawing a ring — so a recording from a
client that does not is left with the mouth alone. Per-frame speaker
flags are aggregated into :class:`SpeakerSegment`s. Audio diarization is a
separate later issue; this module never uses audio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .face import MOUTH_CORNERS, FaceObservation
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


def mouth_open_ratio(observation: FaceObservation) -> float | None:
    """How open the mouth is, or ``None`` when the observer cannot say.

    Three sources, in descending order of directness: MediaPipe's ``jawOpen``
    coefficient, the inner-lip landmark gap, and the separation of the mouth
    corners. All are normalized so they are comparable across face sizes.

    Returning ``None`` matters. An observer without mouth landmarks would
    otherwise measure the distance between two zeroed points and report a number
    that looks like a measurement — the absence of a cue is not a closed mouth.
    """

    jaw = observation.blendshapes.get("jawOpen")
    if jaw is not None:
        return float(jaw)

    iod = _interocular(observation)
    if iod <= _EPS:
        return None

    landmarks = observation.landmarks
    top, bottom = landmarks[MOUTH_TOP, :2], landmarks[MOUTH_BOTTOM, :2]
    if _present(top) and _present(bottom):
        return abs(float(bottom[1]) - float(top[1])) / iod

    # Corner separation is a weaker signal than lip gap — it widens with a smile
    # as well as with speech — but it is a real one, and some observers report
    # only the corners.
    left_corner = landmarks[MOUTH_CORNERS[0], :2]
    right_corner = landmarks[MOUTH_CORNERS[1], :2]
    if _present(left_corner) and _present(right_corner):
        return float(np.linalg.norm(np.asarray(right_corner) - np.asarray(left_corner))) / iod

    return None


def _present(point: np.ndarray) -> bool:
    """Whether a landmark was populated rather than left at the origin."""

    return bool(abs(float(point[0])) > _EPS or abs(float(point[1])) > _EPS)


def _interocular(observation: FaceObservation) -> float:
    right = np.asarray(observation.right_iris)
    left = np.asarray(observation.left_iris)
    return float(np.linalg.norm(right - left))


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
