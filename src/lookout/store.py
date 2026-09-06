"""Raw gaze store.

An append-only, versioned JSONL file of :class:`GazeDirection` records. The
store is written once by the observation stages and only ever read by the
mapping/attribution stages, so a run can be re-attributed without re-running any
model. Attribution never mutates it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .models import GazeDirection, HeadPose

__all__ = ["SCHEMA_VERSION", "write_gaze", "append_gaze", "read_gaze"]

SCHEMA_VERSION = 1
_SCHEMA_NAME = "lookout.gaze"


def _to_dict(direction: GazeDirection) -> dict[str, object]:
    return {
        "timestamp": direction.timestamp,
        "person_id": direction.person_id,
        "yaw": direction.yaw,
        "pitch": direction.pitch,
        "head_pose": {
            "yaw": direction.head_pose.yaw,
            "pitch": direction.head_pose.pitch,
            "roll": direction.head_pose.roll,
        },
        "confidence": direction.confidence,
    }


def _from_dict(record: dict[str, object]) -> GazeDirection:
    head = record["head_pose"]
    assert isinstance(head, dict)
    return GazeDirection(
        timestamp=float(record["timestamp"]),  # type: ignore[arg-type]
        person_id=str(record["person_id"]),
        yaw=float(record["yaw"]),  # type: ignore[arg-type]
        pitch=float(record["pitch"]),  # type: ignore[arg-type]
        head_pose=HeadPose(
            yaw=float(head["yaw"]),
            pitch=float(head["pitch"]),
            roll=float(head["roll"]),
        ),
        confidence=float(record["confidence"]),  # type: ignore[arg-type]
    )


def _header() -> str:
    return json.dumps({"schema": _SCHEMA_NAME, "version": SCHEMA_VERSION})


def write_gaze(path: str | Path, directions: Iterable[GazeDirection]) -> None:
    """Write a fresh store: a header line followed by one record per line."""

    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(_header() + "\n")
        for direction in directions:
            handle.write(json.dumps(_to_dict(direction)) + "\n")


def append_gaze(path: str | Path, directions: Iterable[GazeDirection]) -> None:
    """Append records, writing the header first if the file does not yet exist."""

    target = Path(path)
    exists = target.exists()
    with target.open("a", encoding="utf-8") as handle:
        if not exists:
            handle.write(_header() + "\n")
        for direction in directions:
            handle.write(json.dumps(_to_dict(direction)) + "\n")


def read_gaze(path: str | Path) -> list[GazeDirection]:
    """Read a store, validating the schema version."""

    with Path(path).open("r", encoding="utf-8") as handle:
        lines = [line for line in (raw.strip() for raw in handle) if line]

    if not lines:
        raise ValueError("empty gaze store")

    header = json.loads(lines[0])
    if header.get("schema") != _SCHEMA_NAME:
        raise ValueError("not a lookout gaze store")
    if header.get("version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported gaze schema version: {header.get('version')}")

    return [_from_dict(json.loads(line)) for line in lines[1:]]
