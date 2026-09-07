import numpy as np

from lookout.identity import SlotLinker, build_layout, build_layouts, read_tile_names
from lookout.layout import LayoutInterval, Tile
from lookout.models import LayoutSource, RegionKind


def _participant(x: float) -> Tile:
    return Tile(RegionKind.PARTICIPANT, x, 0.0, 0.2, 0.4)


def test_slot_linker_reuses_ids_and_falls_back() -> None:
    linker = SlotLinker()
    assert linker.resolve("Alice", "slot_0") == "Alice"
    assert linker.resolve("alice", "slot_5") == "Alice"  # same person, moved slot
    assert linker.resolve("Bob", "slot_1") == "Bob"
    assert linker.resolve(None, "slot_2") == "slot_2"
    assert linker.resolve("   ", "slot_3") == "slot_3"


def test_build_layout_assigns_slots_and_keeps_shared() -> None:
    tiles = (
        _participant(0.0),
        _participant(0.4),
        Tile(RegionKind.SHARED_CONTENT, 0.6, 0.0, 0.4, 1.0),
    )
    layout = build_layout(
        tiles,
        viewer_id="recording",
        source=LayoutSource.RECORDING,
        start_time=0.0,
    )
    participants = layout.participant_regions()
    assert [r.participant_id for r in participants] == ["slot_0", "slot_1"]
    shared = [r for r in layout.regions if r.kind is RegionKind.SHARED_CONTENT]
    assert len(shared) == 1 and shared[0].participant_id is None


def test_build_layout_uses_names_when_available() -> None:
    tiles = (_participant(0.0), _participant(0.4))
    layout = build_layout(
        tiles,
        viewer_id="recording",
        source=LayoutSource.RECORDING,
        start_time=0.0,
        names=["Alice", None],
        linker=SlotLinker(),
    )
    assert [r.participant_id for r in layout.participant_regions()] == ["Alice", "slot_1"]


def test_build_layouts_links_names_across_intervals() -> None:
    linker = SlotLinker()
    intervals = [
        LayoutInterval(0.0, 5.0, (_participant(0.0), _participant(0.4))),
        LayoutInterval(5.0, None, (_participant(0.0), _participant(0.4))),
    ]
    # Bob moves from slot 1 to slot 0; the name keeps his id stable.
    layouts = build_layouts(
        intervals,
        viewer_id="recording",
        source=LayoutSource.RECORDING,
        names_per_interval=[["Alice", "Bob"], ["Bob", "Carol"]],
        linker=linker,
    )
    first = [r.participant_id for r in layouts[0].participant_regions()]
    second = [r.participant_id for r in layouts[1].participant_regions()]
    assert first == ["Alice", "Bob"]
    assert second == ["Bob", "Carol"]


def test_read_tile_names_crops_and_reads_labels() -> None:
    width, height = 200, 100
    image = np.zeros((height, width, 3), dtype=np.uint8)
    tiles = (_participant(0.0), _participant(0.5))
    # Each tile spans rows 0..40; its bottom 25% label strip is rows 30..40.
    image[30:40, 0:40] = (10, 20, 30)
    image[30:40, 100:140] = (40, 50, 60)

    class FakeReader:
        def __init__(self) -> None:
            self.mapping = {(10, 20, 30): "Alice", (40, 50, 60): "Bob"}

        def read(self, crop: np.ndarray) -> str:
            color = tuple(int(v) for v in crop.reshape(-1, 3).mean(axis=0).round())
            return self.mapping.get(color, "")

    names = read_tile_names(image, tiles, FakeReader())
    assert names == ["Alice", "Bob"]
