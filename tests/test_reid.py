import numpy as np
import pytest

from lookout.reid import ReIdentifier


class _FakeEmbedder:
    """Embeds a crop as a one-hot vector keyed by its first pixel."""

    def embed(self, image: np.ndarray) -> np.ndarray:
        index = int(image[0, 0, 0]) % 4
        vector = np.zeros(4, dtype=np.float64)
        vector[index] = 1.0
        return vector


def _crop(tag: int) -> np.ndarray:
    return np.full((2, 2, 3), tag, dtype=np.uint8)


def test_disabled_by_default_refuses_to_run() -> None:
    reid = ReIdentifier(_FakeEmbedder())
    with pytest.raises(RuntimeError):
        reid.identify(_crop(0))


def test_links_same_face_and_separates_different() -> None:
    reid = ReIdentifier(_FakeEmbedder(), enabled=True)
    first = reid.identify(_crop(0))
    same = reid.identify(_crop(0))
    other = reid.identify(_crop(1))
    assert first == same
    assert other != first


def test_reset_clears_gallery() -> None:
    reid = ReIdentifier(_FakeEmbedder(), enabled=True)
    reid.identify(_crop(0))
    reid.reset()
    # After reset, ids start over from person_0.
    assert reid.identify(_crop(1)) == "person_0"
