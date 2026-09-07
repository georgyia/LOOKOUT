"""Per-viewer layout manifests.

A manifest lets a user supply the true layout each participant saw, replacing the
v1 ``assumed_shared`` assumption for viewers whose screen differed from the
recording. Loaded layouts carry ``source = manifest`` so provenance stays honest.

Manifest JSON:

```json
{
  "viewers": {
    "bob": [
      {"start_time": 0.0, "end_time": null, "regions": [
        {"kind": "participant", "x": 0.0, "y": 0.0, "width": 0.5, "height": 1.0,
         "participant_id": "alice"}
      ]}
    ]
  }
}
```
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .artifacts import region_from_dict, region_to_dict
from .models import Layout, LayoutSource

__all__ = ["load_manifest", "save_manifest"]


def load_manifest(path: str | Path) -> list[Layout]:
    """Load per-viewer layouts from a manifest file, tagged ``source=manifest``."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    viewers = data["viewers"]
    layouts: list[Layout] = []
    for viewer_id, intervals in viewers.items():
        for interval in intervals:
            regions = tuple(region_from_dict(r) for r in interval["regions"])
            end = interval.get("end_time")
            layouts.append(
                Layout(
                    viewer_id=str(viewer_id),
                    regions=regions,
                    start_time=float(interval["start_time"]),
                    source=LayoutSource.MANIFEST,
                    end_time=float(end) if end is not None else None,
                )
            )
    return layouts


def save_manifest(path: str | Path, layouts: list[Layout]) -> None:
    """Write layouts to a manifest file, grouped by viewer."""

    viewers: dict[str, list[dict[str, object]]] = defaultdict(list)
    for layout in layouts:
        viewers[layout.viewer_id].append(
            {
                "start_time": layout.start_time,
                "end_time": layout.end_time,
                "regions": [region_to_dict(r) for r in layout.regions],
            }
        )
    Path(path).write_text(json.dumps({"viewers": dict(viewers)}, indent=2), encoding="utf-8")
