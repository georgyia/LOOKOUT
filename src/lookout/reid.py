"""Opt-in face re-identification across layout changes.

When name-label OCR is unavailable, participants can still be linked across
layout changes by face similarity. This is biometric, so it is strictly opt-in
and disabled by default: a :class:`ReIdentifier` refuses to run unless
constructed with ``enabled=True``. Embeddings are held in memory only and never
written to disk. See docs/research/11-reidentification-privacy.md.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .frames import Image

__all__ = ["FaceEmbedder", "ReIdentifier"]

Embedding = NDArray[np.float64]
_EPS = 1e-12


class FaceEmbedder(Protocol):
    """Produces a face embedding vector from a crop (a local model adapter)."""

    def embed(self, image: Image) -> Embedding: ...


def _unit(vector: Embedding) -> Embedding:
    norm = float(np.linalg.norm(vector))
    if norm < _EPS:
        return vector
    return vector / norm


class ReIdentifier:
    """Links crops to stable ids by embedding similarity, in memory only.

    Disabled by default; every operating method raises unless ``enabled`` is
    True. Nothing is persisted; :meth:`reset` clears the in-memory gallery.
    """

    def __init__(
        self,
        embedder: FaceEmbedder,
        threshold: float = 0.6,
        enabled: bool = False,
    ) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._enabled = enabled
        self._gallery: list[tuple[str, Embedding]] = []
        self._next = 0

    def _require_enabled(self) -> None:
        if not self._enabled:
            raise RuntimeError(
                "face re-identification is opt-in; construct ReIdentifier(enabled=True)"
            )

    def identify(self, crop: Image) -> str:
        """Return the id of the closest known face, or mint a new one."""

        self._require_enabled()
        vector = _unit(self._embedder.embed(crop))

        best_id: str | None = None
        best_similarity = self._threshold
        for identity, known in self._gallery:
            similarity = float(np.dot(vector, known))
            if similarity >= best_similarity:
                best_similarity = similarity
                best_id = identity

        if best_id is None:
            best_id = f"person_{self._next}"
            self._next += 1
            self._gallery.append((best_id, vector))
        return best_id

    def reset(self) -> None:
        """Clear the in-memory gallery."""

        self._gallery.clear()
        self._next = 0
