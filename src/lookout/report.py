"""Report rendering.

Every format here is a rendering of the run record; none of them recompute
anything. The section order is the argument: provenance, coverage and
degradations come before results, so a reader who stops early stops having read
the caveats rather than the durations.

The headline is whether the run was scored at all. An unscored run and a scored
one must not read alike, which is what the previous report allowed — it showed a
duration table either way.

Wording is constrained to estimates with confidence. Nothing here describes
attention, interest, intent, or emotion, and a test enforces that.
"""

from __future__ import annotations

import csv
import html
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .coverage import Coverage, ParticipantCoverage
from .models import GazeEvent, GazeTarget
from .runrecord import DISCLAIMER, RunRecord

__all__ = [
    "DISCLAIMER",
    "Verdict",
    "summarize",
    "verdict",
    "limitations",
    "render_markdown",
    "render_html",
    "write_markdown",
    "write_html",
    "write_csv",
]

_UNRESOLVED = {target.value for target in GazeTarget}


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


@dataclass(frozen=True)
class Verdict:
    """How much weight the results below can carry."""

    scored: bool
    severity: str
    headline: str
    detail: str


def verdict(record: RunRecord) -> Verdict:
    """Decide the report's headline from the run's own state."""

    warnings = [d for d in record.degradations if d.stage in {"attribution", "mapping"}]

    if record.evaluation is None:
        detail = (
            "No ground truth was supplied, so no claim is made about how often the "
            "targets below are right. The durations describe what the pipeline "
            "produced, not what happened."
        )
        if warnings:
            detail += (
                " The distribution of those durations is also degenerate; see the "
                "warnings below."
            )
        return Verdict(
            scored=False,
            severity="unscored",
            headline="Accuracy: NOT MEASURED",
            detail=detail,
        )

    hit_rate = _number(record.evaluation.get("hit_rate"))
    baselines = record.evaluation.get("baselines") or {}
    best_baseline = max((_number(v) for v in baselines.values()), default=0.0)
    beats = hit_rate > best_baseline

    return Verdict(
        scored=True,
        severity="scored" if beats else "warning",
        headline=f"Accuracy: {hit_rate:.1%} hit rate",
        detail=(
            f"Measured against ground truth. Best baseline scores {best_baseline:.1%}; "
            + (
                "the estimator beats it."
                if beats
                else "the estimator does not beat it, so the result carries no evidence "
                "of gaze signal."
            )
        ),
    )


def limitations(record: RunRecord) -> tuple[str, ...]:
    """What this particular run cannot support, derived from its own state.

    Generated rather than written down, because the caveat that matters is
    specific to the run and a static list drifts from what the code does.
    """

    found: list[str] = []
    coverage = record.coverage

    if record.evaluation is None:
        found.append(
            "This run was never scored. Hit rate, unknown rate and off-screen recall "
            "are unknown, and no threshold in the configuration has been validated "
            "against ground truth."
        )

    if coverage and "assumed_shared" in coverage.layout_sources:
        found.append(
            "Every viewer was attributed against the recorded layout. That layout is "
            "only what the recording participant saw; everyone else saw a different "
            "arrangement, so cross-viewer targets are not credible."
        )

    if coverage and coverage.face_attempts and coverage.face_hit_rate < 0.9:
        missed = coverage.face_attempts - coverage.face_hits
        found.append(
            f"No face was resolved in {missed} of {coverage.face_attempts} tile crops "
            f"({1 - coverage.face_hit_rate:.1%}). Durations understate time spent "
            f"looking, by an amount this run cannot measure."
        )

    if coverage:
        unresolvable = [
            e.participant_id
            for e in coverage.per_participant
            if not e.observed and e.face_attempts
        ]
        if unresolvable:
            found.append(
                "No face was ever resolved for "
                + ", ".join(sorted(unresolvable))
                + ". They are reported as not visible rather than omitted: the pipeline "
                "could not see them, which is not the same as their having looked "
                "nowhere."
            )

    if record.diagnostics and record.diagnostics.unresolved_share > 0.25:
        found.append(
            f"{record.diagnostics.unresolved_share:.1%} of attributed duration is "
            f"unknown, off screen, or below the confidence floor. That is the "
            f"conservative default working as intended, not missing data."
        )

    for adapter in record.provenance.adapters:
        if adapter.model_path is None and adapter.role == "gaze":
            found.append(
                f"Gaze came from '{adapter.implementation}', which is not a trained "
                f"gaze model. Treat fine distinctions between neighbouring tiles as "
                f"unsupported."
            )

    return tuple(found)


def _number(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _evaluation_lines(evaluation: dict[str, object]) -> list[str]:
    """Scores, always next to what a trivial strategy would have scored."""

    out: list[str] = []
    hit_rate = _number(evaluation.get("hit_rate"))
    ci = evaluation.get("hit_rate_ci95")
    interval = (
        f" (95% CI {_number(ci[0]):.1%}-{_number(ci[1]):.1%})"
        if isinstance(ci, list) and len(ci) == 2
        else ""
    )
    out.append(f"- Hit rate: {hit_rate:.1%}{interval} over {evaluation.get('total', 0)} samples")

    baselines = evaluation.get("baselines")
    if isinstance(baselines, dict) and baselines:
        out.append("- Baselines:")
        best = max(_number(v) for v in baselines.values())
        for name, value in sorted(baselines.items(), key=lambda kv: -_number(kv[1])):
            marker = " <- best" if _number(value) == best else ""
            out.append(f"    - {name}: {_number(value):.1%}{marker}")
        if not evaluation.get("beats_baseline", True):
            out.append(
                "- **The estimator does not beat its best baseline**, so this run carries "
                "no evidence of gaze signal."
            )

    breakdown = evaluation.get("unknown_breakdown")
    if isinstance(breakdown, dict):
        parts = ", ".join(f"{k} {v}" for k, v in breakdown.items())
        out.append(f"- Declined to answer: {parts}")
    out.append(f"- Wrong answers: {evaluation.get('wrong', 0)}")
    out.append(f"- Off-screen recall: {_number(evaluation.get('off_screen_recall')):.1%}")

    ece = evaluation.get("expected_calibration_error")
    if ece is not None:
        out.append(f"- Expected calibration error: {_number(ece):.3f}")

    per_grid = evaluation.get("per_grid")
    if isinstance(per_grid, dict) and per_grid:
        grids = ", ".join(f"{g} {_number(v):.1%}" for g, v in sorted(per_grid.items()))
        out.append(f"- By grid: {grids}")
    out.append("")
    return out


def _outcome(entry: ParticipantCoverage) -> str:
    """Which of three cases applies, said out loud.

    Absence of a result used to read as absence of a person. These are different
    facts and the report should not leave the reader to infer which one it has.
    """

    if entry.observed:
        return "observed"
    if entry.face_attempts:
        return "present, not resolvable"
    return "not present in this layout"


def _funnel(coverage: Coverage) -> list[tuple[str, int, str]]:
    """The funnel as ordered rows: label, count, and what the step means."""

    return [
        ("Frames sampled", coverage.frames, "after subsampling to the target rate"),
        ("Tile crops examined", coverage.face_attempts, "participants x frames"),
        ("Faces resolved", coverage.face_hits, f"{coverage.face_hit_rate:.1%} of crops"),
        ("Gaze directions", coverage.directions, "raw observations stored"),
        ("Mapped on screen", coverage.points, "angle fell inside the screen"),
        ("Rejected off screen", coverage.off_screen, f"{coverage.off_screen_rate:.1%}"),
        ("Fixations", coverage.fixations, "stable dwells"),
        ("Attributions", coverage.attributions, "fixations matched to a layout"),
        ("Events", coverage.events, "sustained runs of one target"),
    ]


def _reproduce(record: RunRecord) -> str:
    command = record.provenance.command or "lookout analyze <video> --out <dir>"
    return (
        f"{command}\n\n"
        "# re-interpret without re-running any model:\n"
        "lookout attribute --out <dir>"
    )


# --------------------------------------------------------------------------- markdown


def render_markdown(record: RunRecord, events: list[GazeEvent]) -> str:
    """Render the record as plain Markdown, for a research note or a PR body."""

    v = verdict(record)
    out: list[str] = ["# LOOKOUT run report", ""]
    out += [f"**{v.headline}**", "", v.detail, "", f"_{record.disclaimer}_", ""]

    p = record.provenance
    out += ["## Provenance", ""]
    if p.video:
        size = (
            f"{p.video.width}x{p.video.height}"
            if p.video.width and p.video.height
            else "unknown size"
        )
        duration = f"{p.video.duration:.1f}s" if p.video.duration else "unknown length"
        out.append(f"- Recording: `{p.video.path}` — {size}, {duration}")
        if p.video.sha256:
            out.append(f"- SHA-256: `{p.video.sha256}`")
    commit = f" at `{p.git_commit[:12]}`" if p.git_commit else ""
    out.append(f"- LOOKOUT {p.lookout_version}{commit}")
    out.append(f"- Run at {p.created_at}")
    for adapter in p.adapters:
        version = f" ({adapter.library_version})" if adapter.library_version else ""
        out.append(f"- Adapter [{adapter.role}]: {adapter.implementation}{version}")
    out.append(f"- Configuration digest: `{p.config_hash[:16]}`")
    if record.config_overrides:
        changed = ", ".join(f"`{o}`" for o in record.config_overrides)
        out.append(f"- Non-default settings: {changed}")
    else:
        out.append("- Non-default settings: none")
    out.append("")

    if record.coverage:
        out += ["## Coverage", "", "| Stage | Count | |", "| --- | ---: | --- |"]
        for label, count, note in _funnel(record.coverage):
            out.append(f"| {label} | {count} | {note} |")
        out += [
            "",
            "### Per participant",
            "",
            "| Participant | Crops | Faces | Rate | Events | Outcome |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for entry in record.coverage.per_participant:
            out.append(
                f"| {entry.participant_id} | {entry.face_attempts} | {entry.face_hits} "
                f"| {entry.face_hit_rate:.1%} | {entry.events} | {_outcome(entry)} |"
            )
        out.append("")

    if record.degradations:
        out += ["## What this run settled for", ""]
        for settled in record.degradations:
            out += [
                f"**{settled.code}** ({settled.stage}) — {settled.detail}",
                "",
                f"> {settled.impact}",
                "",
            ]

    shape = record.diagnostics
    if shape:
        out += ["## Distribution", ""]
        if shape.top_target:
            out.append(f"- Dominant target: `{shape.top_target}` at {shape.top_target_share:.1%}")
        out.append(f"- Normalized entropy: {shape.normalized_entropy:.2f}")
        out.append(f"- Gaze in central ninth of screen: {shape.centre_mass:.1%}")
        out.append(f"- Self-view share: {shape.self_view_share:.1%}")
        out.append(f"- Unresolved share: {shape.unresolved_share:.1%}")
        if shape.limiting_factors:
            factors = ", ".join(f"{k} ({n})" for k, n in shape.limiting_factors.items())
            out.append(f"- Limiting confidence factor: {factors}")
        out.append("")

    timing = record.timing
    if timing and timing.stages:
        out += ["## Cost", ""]
        out.append(
            f"- {timing.wall_seconds:.1f}s of wall time for "
            f"{timing.video_seconds:.1f}s of video "
            f"({timing.realtime_factor:.2f}x realtime)"
        )
        hour = timing.projected_seconds(3600.0)
        if hour:
            out.append(f"- An hour of recording projects to {hour / 60.0:.0f} minutes")
        out += ["", "| Stage | Seconds | Calls | Share |", "| --- | ---: | ---: | ---: |"]
        total = sum(stage.seconds for stage in timing.stages)
        for stage in timing.stages:
            out.append(
                f"| {stage.stage} | {stage.seconds:.2f} | {stage.calls} "
                f"| {stage.share_of(total):.1%} |"
            )
        out.append("")

    out += ["## Evaluation", ""]
    if record.evaluation is None:
        out += ["Not measured. No ground truth was supplied for this recording.", ""]
    else:
        out += _evaluation_lines(record.evaluation)

    out += ["## Results", "", "These are estimates. Read them against the sections above.", ""]
    summary = summarize(events)
    viewers = summary["viewers"]
    assert isinstance(viewers, dict)
    if viewers:
        out += ["| Viewer | Target | Duration (s) |", "| --- | --- | ---: |"]
        for viewer, stats in sorted(viewers.items()):
            for target, duration in sorted(
                stats["duration_by_target"].items(), key=lambda kv: -kv[1]
            ):
                out.append(f"| {viewer} | {target} | {duration:.1f} |")
    else:
        out.append("No events were produced.")
    out.append("")

    notes = limitations(record)
    if notes:
        out += ["## Limitations", ""]
        out += [f"- {note}" for note in notes]
        out.append("")

    out += ["## Reproduce", "", "```bash", _reproduce(record), "```", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- html

_STYLE = """
  :root { --fg:#1a1a1a; --muted:#666; --line:#e2e2e2; --bg:#fff; --card:#f7f7f8;
          --warn:#8a5a00; --warnbg:#fff6e5; --bad:#8a1f1f; --badbg:#fdecec;
          --ok:#1f6b3a; --okbg:#eaf6ee; }
  body { font-family: system-ui, -apple-system, sans-serif; margin:0; padding:2rem;
         color:var(--fg); background:var(--bg); line-height:1.5; }
  main { max-width: 62rem; margin: 0 auto; }
  h1 { font-size:1.5rem; margin:0 0 1rem; }
  h2 { font-size:1.1rem; margin:2.5rem 0 .75rem; padding-bottom:.3rem;
       border-bottom:1px solid var(--line); }
  h3 { font-size:.95rem; margin:1.5rem 0 .5rem; color:var(--muted); }
  .verdict { padding:1rem 1.25rem; border-radius:8px; margin-bottom:1rem;
             border:1px solid transparent; }
  .verdict.unscored { background:var(--warnbg); border-color:#f0dcb0; color:var(--warn); }
  .verdict.warning { background:var(--badbg); border-color:#f2c9c9; color:var(--bad); }
  .verdict.scored { background:var(--okbg); border-color:#c3e3cf; color:var(--ok); }
  .verdict strong { display:block; font-size:1.15rem; margin-bottom:.35rem; }
  .disclaimer { color:var(--muted); font-size:.85rem; margin:0 0 1rem; }
  table { border-collapse:collapse; width:100%; font-size:.9rem; }
  th,td { border-bottom:1px solid var(--line); padding:.4rem .6rem; text-align:left; }
  td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; }
  .card { background:var(--card); border-radius:6px; padding:.75rem 1rem; margin:.6rem 0; }
  .card .code { font-family:ui-monospace,monospace; font-size:.8rem; color:var(--muted); }
  .card .impact { margin:.4rem 0 0; color:var(--muted); font-size:.88rem; }
  .bar { height:.7rem; background:#4a7a9b; border-radius:2px; }
  .funnel td.track { width:45%; }
  ul.notes li { margin-bottom:.5rem; }
  pre { background:var(--card); padding:.75rem 1rem; border-radius:6px; overflow-x:auto;
        font-size:.85rem; }
  .wrap { overflow-x:auto; }
  dl { display:grid; grid-template-columns:max-content 1fr; gap:.3rem 1rem; font-size:.9rem;
       margin:0; }
  dt { color:var(--muted); }
  dd { margin:0; font-family:ui-monospace,monospace; word-break:break-all; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _heatmap_svg(grid: tuple[tuple[int, ...], ...]) -> str:
    if not grid:
        return ""
    peak = max((max(row) for row in grid if row), default=0)
    if peak <= 0:
        return ""
    size, cells = 240, len(grid)
    step = size / cells
    rects: list[str] = []
    for r, row in enumerate(grid):
        for c, count in enumerate(row):
            if not count:
                continue
            alpha = 0.12 + 0.88 * (count / peak)
            rects.append(
                f'<rect x="{c * step:.2f}" y="{r * step:.2f}" width="{step:.2f}" '
                f'height="{step:.2f}" fill="#4a7a9b" fill-opacity="{alpha:.3f}"/>'
            )
    third = size / 3
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img" '
        f'aria-label="Where gaze landed on the screen">'
        f'<rect width="{size}" height="{size}" fill="#f7f7f8"/>{"".join(rects)}'
        f'<rect x="{third:.1f}" y="{third:.1f}" width="{third:.1f}" height="{third:.1f}" '
        f'fill="none" stroke="#8a1f1f" stroke-dasharray="4 3"/></svg>'
    )


def render_html(record: RunRecord, events: list[GazeEvent]) -> str:
    """Render the record as a self-contained offline page."""

    v = verdict(record)
    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>LOOKOUT run report</title>",
        f"<style>{_STYLE}</style></head><body><main>",
        "<h1>LOOKOUT run report</h1>",
        f'<div class="verdict {_esc(v.severity)}"><strong>{_esc(v.headline)}</strong>'
        f"{_esc(v.detail)}</div>",
        f'<p class="disclaimer">{_esc(record.disclaimer)}</p>',
    ]

    p = record.provenance
    parts.append("<h2>Provenance</h2><dl>")
    if p.video:
        parts.append(f"<dt>Recording</dt><dd>{_esc(p.video.path)}</dd>")
        if p.video.width and p.video.height:
            duration = f"{p.video.duration:.1f}s" if p.video.duration else "unknown"
            parts.append(
                f"<dt>Format</dt><dd>{p.video.width}x{p.video.height}, "
                f"{_esc(p.video.source_fps)} fps, {duration}</dd>"
            )
        if p.video.sha256:
            parts.append(f"<dt>SHA-256</dt><dd>{_esc(p.video.sha256)}</dd>")
    parts.append(f"<dt>Version</dt><dd>{_esc(p.lookout_version)} {_esc(p.git_commit or '')}</dd>")
    parts.append(f"<dt>Run at</dt><dd>{_esc(p.created_at)}</dd>")
    for adapter in p.adapters:
        version = f" ({adapter.library_version})" if adapter.library_version else ""
        parts.append(
            f"<dt>Adapter: {_esc(adapter.role)}</dt>"
            f"<dd>{_esc(adapter.implementation)}{_esc(version)}</dd>"
        )
    parts.append(f"<dt>Config digest</dt><dd>{_esc(p.config_hash)}</dd>")
    overrides = ", ".join(record.config_overrides) or "none"
    parts.append(f"<dt>Non-default</dt><dd>{_esc(overrides)}</dd>")
    parts.append("</dl>")

    if record.coverage:
        cov = record.coverage
        peak = max((count for _, count, _ in _funnel(cov)), default=1) or 1
        parts.append('<h2>Coverage</h2><div class="wrap"><table class="funnel">')
        parts.append(
            '<tr><th>Stage</th><th class="n">Count</th>'
            '<th class="track"></th><th></th></tr>'
        )
        for label, count, note in _funnel(cov):
            width = 100.0 * count / peak
            parts.append(
                f"<tr><td>{_esc(label)}</td><td class='n'>{count}</td>"
                f"<td class='track'><div class='bar' style='width:{width:.1f}%'></div></td>"
                f"<td>{_esc(note)}</td></tr>"
            )
        parts.append("</table></div>")

        parts.append('<h3>Per participant</h3><div class="wrap"><table>')
        parts.append(
            '<tr><th>Participant</th><th class="n">Crops</th><th class="n">Faces</th>'
            '<th class="n">Rate</th><th class="n">Events</th><th>Outcome</th></tr>'
        )
        for entry in cov.per_participant:
            parts.append(
                f"<tr><td>{_esc(entry.participant_id)}</td>"
                f"<td class='n'>{entry.face_attempts}</td>"
                f"<td class='n'>{entry.face_hits}</td>"
                f"<td class='n'>{entry.face_hit_rate:.1%}</td>"
                f"<td class='n'>{entry.events}</td>"
                f"<td>{_esc(_outcome(entry))}</td></tr>"
            )
        parts.append("</table></div>")

    if record.degradations:
        parts.append("<h2>What this run settled for</h2>")
        for settled in record.degradations:
            parts.append(
                f'<div class="card">'
                f'<span class="code">{_esc(settled.stage)} / {_esc(settled.code)}</span>'
                f"<div>{_esc(settled.detail)}</div>"
                f'<p class="impact">{_esc(settled.impact)}</p></div>'
            )

    shape = record.diagnostics
    if shape:
        parts.append("<h2>Distribution</h2><dl>")
        if shape.top_target:
            parts.append(
                f"<dt>Dominant target</dt><dd>{_esc(shape.top_target)} "
                f"({shape.top_target_share:.1%})</dd>"
            )
        parts.append(f"<dt>Normalized entropy</dt><dd>{shape.normalized_entropy:.2f}</dd>")
        parts.append(f"<dt>Central ninth</dt><dd>{shape.centre_mass:.1%}</dd>")
        parts.append(f"<dt>Self-view</dt><dd>{shape.self_view_share:.1%}</dd>")
        parts.append(f"<dt>Unresolved</dt><dd>{shape.unresolved_share:.1%}</dd>")
        parts.append("</dl>")
        heatmap = _heatmap_svg(shape.screen_histogram)
        if heatmap:
            parts.append(
                f"<h3>Where gaze landed</h3>{heatmap}"
                f'<p class="disclaimer">Dashed outline marks the central ninth of the '
                f"screen.</p>"
            )

    timing = record.timing
    if timing and timing.stages:
        total = sum(stage.seconds for stage in timing.stages)
        parts.append("<h2>Cost</h2><dl>")
        parts.append(
            f"<dt>Throughput</dt><dd>{timing.realtime_factor:.2f}x realtime "
            f"({timing.wall_seconds:.1f}s for {timing.video_seconds:.1f}s of video)</dd>"
        )
        hour = timing.projected_seconds(3600.0)
        if hour:
            parts.append(f"<dt>An hour projects to</dt><dd>{hour / 60.0:.0f} minutes</dd>")
        parts.append("</dl>")
        parts.append('<div class="wrap"><table>')
        parts.append(
            '<tr><th>Stage</th><th class="n">Seconds</th><th class="n">Calls</th>'
            '<th class="track"></th><th class="n">Share</th></tr>'
        )
        for stage in timing.stages:
            share = stage.share_of(total)
            parts.append(
                f"<tr><td>{_esc(stage.stage)}</td>"
                f"<td class='n'>{stage.seconds:.2f}</td>"
                f"<td class='n'>{stage.calls}</td>"
                f"<td class='track'><div class='bar' style='width:{100 * share:.1f}%'></div></td>"
                f"<td class='n'>{share:.1%}</td></tr>"
            )
        parts.append("</table></div>")
        parts.append(
            '<p class="disclaimer">Cost is a property of this machine and '
            "configuration, not a measure of result quality.</p>"
        )

    parts.append("<h2>Evaluation</h2>")
    if record.evaluation is None:
        parts.append(
            '<p class="disclaimer">Not measured. No ground truth was supplied for this '
            "recording, so no accuracy figure appears anywhere in this report.</p>"
        )
    else:
        parts.append("<ul class='notes'>")
        for line in _evaluation_lines(record.evaluation):
            if line.strip():
                parts.append(f"<li>{_esc(line.strip().lstrip('- '))}</li>")
        parts.append("</ul>")

    parts.append("<h2>Results</h2>")
    parts.append('<p class="disclaimer">Estimates. Read against the sections above.</p>')
    parts.append(f'<div class="wrap">{_timeline_table(events)}</div>')

    notes = limitations(record)
    if notes:
        parts.append('<h2>Limitations</h2><ul class="notes">')
        parts += [f"<li>{_esc(note)}</li>" for note in notes]
        parts.append("</ul>")

    parts.append(f"<h2>Reproduce</h2><pre>{_esc(_reproduce(record))}</pre>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def _timeline_table(events: list[GazeEvent]) -> str:
    if not events:
        return "<p>No events were produced.</p>"
    span = max(e.end_time for e in events) or 1.0
    rows = [
        "<table><tr><th>Viewer</th><th>Target</th><th>Why</th>"
        '<th class="n">Start</th><th class="n">End</th><th class="track">Timeline</th></tr>'
    ]
    for event in sorted(events, key=lambda e: (e.viewer_id, e.start_time)):
        left = 100.0 * event.start_time / span
        width = max(0.5, 100.0 * event.duration / span)
        unresolved = event.target in _UNRESOLVED
        color = "#b0b0b0" if unresolved else "#4a7a9b"
        rows.append(
            f"<tr><td>{_esc(event.viewer_id)}</td><td>{_esc(event.target)}</td>"
            f"<td>{_esc(event.reason)}</td>"
            f"<td class='n'>{event.start_time:.1f}</td><td class='n'>{event.end_time:.1f}</td>"
            f"<td class='track'><div class='bar' style='margin-left:{left:.1f}%;"
            f"width:{width:.1f}%;background:{color}' "
            f"title='confidence {event.confidence:.2f}'></div></td></tr>"
        )
    rows.append("</table>")
    return "".join(rows)


# --------------------------------------------------------------------------- writers


def write_markdown(path: str | Path, record: RunRecord, events: list[GazeEvent]) -> None:
    Path(path).write_text(render_markdown(record, events), encoding="utf-8")


def write_html(path: str | Path, record: RunRecord, events: list[GazeEvent]) -> None:
    Path(path).write_text(render_html(record, events), encoding="utf-8")


def write_csv(path: str | Path, events: list[GazeEvent]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "viewer_id",
                "target",
                "start_time",
                "end_time",
                "confidence",
                "layout_source",
                "reason",
            ]
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
                    event.reason,
                ]
            )
