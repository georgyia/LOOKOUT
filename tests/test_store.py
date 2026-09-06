from pathlib import Path

import pytest

from lookout.models import GazeDirection, HeadPose
from lookout.store import append_gaze, read_gaze, write_gaze


def _sample() -> list[GazeDirection]:
    return [
        GazeDirection(0.0, "bob", 0.1, -0.2, HeadPose(0.05, 0.0, 0.0), 0.9),
        GazeDirection(0.2, "bob", 0.12, -0.18, HeadPose(0.06, 0.01, 0.0), 0.8),
    ]


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "gaze.jsonl"
    write_gaze(path, _sample())
    restored = read_gaze(path)
    assert restored == _sample()


def test_append_creates_header_then_appends(tmp_path: Path) -> None:
    path = tmp_path / "gaze.jsonl"
    append_gaze(path, _sample()[:1])
    append_gaze(path, _sample()[1:])
    assert read_gaze(path) == _sample()


def test_unknown_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "gaze.jsonl"
    path.write_text('{"schema": "lookout.gaze", "version": 999}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported gaze schema version"):
        read_gaze(path)


def test_foreign_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "gaze.jsonl"
    path.write_text('{"schema": "something.else", "version": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not a lookout gaze store"):
        read_gaze(path)


def test_empty_store_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "gaze.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty gaze store"):
        read_gaze(path)
