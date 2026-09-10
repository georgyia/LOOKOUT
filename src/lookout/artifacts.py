"""Serialization for run artifacts.

The raw gaze store (:mod:`lookout.store`) is the re-run source of truth. These
are the human-inspectable outputs of a run: the recording layouts, the derived
screen points, the per-fixation attributions, and the aggregated events. Files
are plain JSONL; provenance and versions live in ``run.json``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .models import (
    Attribution,
    GazeEvent,
    GazePoint,
    Layout,
    LayoutSource,
    Region,
    RegionKind,
)

__all__ = [
    "region_to_dict",
    "region_from_dict",
    "write_layouts",
    "read_layouts",
    "write_points",
    "write_attributions",
    "write_events",
    "read_events",
]


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def region_to_dict(region: Region) -> dict[str, object]:
    return {
        "kind": region.kind.value,
        "x": region.x,
        "y": region.y,
        "width": region.width,
        "height": region.height,
        "participant_id": region.participant_id,
    }


def region_from_dict(data: dict[str, object]) -> Region:
    pid = data["participant_id"]
    return Region(
        kind=RegionKind(str(data["kind"])),
        x=float(data["x"]),  # type: ignore[arg-type]
        y=float(data["y"]),  # type: ignore[arg-type]
        width=float(data["width"]),  # type: ignore[arg-type]
        height=float(data["height"]),  # type: ignore[arg-type]
        participant_id=str(pid) if pid is not None else None,
    )


def write_layouts(path: str | Path, layouts: Iterable[Layout]) -> None:
    _write_jsonl(
        path,
        (
            {
                "viewer_id": layout.viewer_id,
                "source": layout.source.value,
                "start_time": layout.start_time,
                "end_time": layout.end_time,
                "regions": [region_to_dict(r) for r in layout.regions],
            }
            for layout in layouts
        ),
    )


def read_layouts(path: str | Path) -> list[Layout]:
    layouts: list[Layout] = []
    for row in _read_jsonl(path):
        regions_raw = row["regions"]
        assert isinstance(regions_raw, list)
        regions = tuple(region_from_dict(r) for r in regions_raw)
        end = row["end_time"]
        layouts.append(
            Layout(
                viewer_id=str(row["viewer_id"]),
                regions=regions,
                start_time=float(row["start_time"]),  # type: ignore[arg-type]
                source=LayoutSource(str(row["source"])),
                end_time=float(end) if end is not None else None,  # type: ignore[arg-type]
            )
        )
    return layouts


def write_points(path: str | Path, points: Iterable[GazePoint]) -> None:
    _write_jsonl(
        path,
        (
            {
                "timestamp": p.timestamp,
                "person_id": p.person_id,
                "x": p.x,
                "y": p.y,
                "confidence": p.confidence,
            }
            for p in points
        ),
    )


def write_attributions(
    path: str | Path,
    rows: Iterable[tuple[str, float, float, Attribution]],
) -> None:
    _write_jsonl(
        path,
        (
            {
                "viewer_id": viewer_id,
                "start_time": start,
                "end_time": end,
                "target": attr.target,
                "confidence": attr.confidence,
                "reason": attr.reason,
                "margin_ratio": attr.margin_ratio,
                "layout_source": attr.layout_source.value,
            }
            for viewer_id, start, end, attr in rows
        ),
    )


def write_events(path: str | Path, events: Iterable[GazeEvent]) -> None:
    _write_jsonl(
        path,
        (
            {
                "viewer_id": e.viewer_id,
                "target": e.target,
                "start_time": e.start_time,
                "end_time": e.end_time,
                "confidence": e.confidence,
                "layout_source": e.layout_source.value,
                "reason": e.reason,
            }
            for e in events
        ),
    )


def read_events(path: str | Path) -> list[GazeEvent]:
    return [
        GazeEvent(
            viewer_id=str(row["viewer_id"]),
            target=str(row["target"]),
            start_time=float(row["start_time"]),  # type: ignore[arg-type]
            end_time=float(row["end_time"]),  # type: ignore[arg-type]
            confidence=float(row["confidence"]),  # type: ignore[arg-type]
            layout_source=LayoutSource(str(row["layout_source"])),
            reason=str(row.get("reason", "")),
        )
        for row in _read_jsonl(path)
    ]
