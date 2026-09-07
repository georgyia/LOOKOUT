"""Report export.

Turns aggregated events into shareable, offline artifacts: a JSON summary, a CSV
of events, and a static HTML timeline (no server). Wording is deliberately
constrained to estimates with confidence; the report never describes attention,
interest, intent, or emotion.
"""

from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path

from .models import GazeEvent

__all__ = ["DISCLAIMER", "summarize", "write_json", "write_csv", "write_html"]

DISCLAIMER = (
    "These are gaze-direction estimates with confidence, derived from video. "
    "They are not measurements of attention, interest, intent, or emotion."
)


def summarize(events: list[GazeEvent]) -> dict[str, object]:
    """Per-viewer totals: event count and looked-at duration per target."""

    by_viewer: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for event in events:
        by_viewer[event.viewer_id][event.target] += event.duration

    viewers = {
        viewer: {
            "events": sum(1 for e in events if e.viewer_id == viewer),
            "duration_by_target": {t: round(d, 3) for t, d in targets.items()},
        }
        for viewer, targets in by_viewer.items()
    }
    return {"viewers": viewers, "total_events": len(events)}


def write_json(path: str | Path, events: list[GazeEvent]) -> None:
    payload = {"disclaimer": DISCLAIMER, "summary": summarize(events)}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: str | Path, events: list[GazeEvent]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["viewer_id", "target", "start_time", "end_time", "confidence", "layout_source"]
        )
        for event in sorted(events, key=lambda e: (e.viewer_id, e.start_time)):
            writer.writerow(
                [
                    event.viewer_id,
                    event.target,
                    f"{event.start_time:.3f}",
                    f"{event.end_time:.3f}",
                    f"{event.confidence:.3f}",
                    event.layout_source.value,
                ]
            )


def _timeline_rows(events: list[GazeEvent]) -> str:
    if not events:
        return '<tr><td colspan="5">No events.</td></tr>'
    span_end = max(e.end_time for e in events) or 1.0
    rows: list[str] = []
    for event in sorted(events, key=lambda e: (e.viewer_id, e.start_time)):
        left = 100.0 * event.start_time / span_end
        width = max(0.5, 100.0 * event.duration / span_end)
        bar = (
            f'<div class="bar" style="margin-left:{left:.1f}%;width:{width:.1f}%" '
            f'title="{event.confidence:.2f}"></div>'
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(event.viewer_id)}</td>"
            f"<td>{html.escape(event.target)}</td>"
            f"<td>{event.start_time:.2f}</td>"
            f"<td>{event.end_time:.2f}</td>"
            f"<td>{bar}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_html(path: str | Path, events: list[GazeEvent]) -> None:
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LOOKOUT report</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  .disclaimer {{ background: #f4f4f4; padding: 0.75rem 1rem; border-radius: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
  th, td {{ border-bottom: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
  .track {{ width: 40%; }}
  .bar {{ height: 0.8rem; background: #4a7; border-radius: 3px; }}
</style>
</head>
<body>
<h1>LOOKOUT gaze report</h1>
<p class="disclaimer">{html.escape(DISCLAIMER)}</p>
<table>
<thead>
<tr>
<th>Viewer</th><th>Target</th><th>Start (s)</th><th>End (s)</th>
<th class="track">Timeline</th>
</tr>
</thead>
<tbody>
{_timeline_rows(events)}
</tbody>
</table>
</body>
</html>
"""
    Path(path).write_text(document, encoding="utf-8")
