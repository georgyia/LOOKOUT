from pathlib import Path

from lookout.manifest import load_manifest, save_manifest
from lookout.models import Layout, LayoutSource, Region, RegionKind


def _layouts() -> list[Layout]:
    return [
        Layout(
            "bob",
            (
                Region(RegionKind.PARTICIPANT, 0.0, 0.0, 0.5, 1.0, "alice"),
                Region(RegionKind.SHARED_CONTENT, 0.5, 0.0, 0.5, 1.0),
            ),
            0.0,
            LayoutSource.MANIFEST,
            10.0,
        ),
        Layout(
            "carol",
            (Region(RegionKind.PARTICIPANT, 0.0, 0.0, 1.0, 1.0, "alice"),),
            0.0,
            LayoutSource.MANIFEST,
        ),
    ]


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    save_manifest(path, _layouts())
    restored = load_manifest(path)
    assert restored == _layouts()


def test_loaded_layouts_are_tagged_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    save_manifest(path, _layouts())
    restored = load_manifest(path)
    assert all(layout.source is LayoutSource.MANIFEST for layout in restored)
