# Architecture

LOOKOUT separates observation from interpretation.

```text
                    MEETING VIDEO
                          │
                          ▼
                   Frame ingestion
                          │
                          ▼
                Participant tracking
                          │
                          ▼
                Layout reconstruction
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        Raw gaze estimation       Audio later
              │
              ▼
          Raw gaze stream
              │
              ▼
       Gaze → screen region
              │
              ▼
       Region → participant
              │
              ▼
       Interaction events
```

## Key decisions

**Raw gaze is an angle, not a screen point.** A raw observation is

`person_id + timestamp + (yaw, pitch) + head_pose + confidence`

Mapping that angle to a screen point is a separate, uncertain step. Storing the
angle lets us re-run mapping and attribution without re-running any model.

**Layouts are viewer-relative.** Each participant sees their own arrangement of
tiles; the recording only shows one. A `Layout` therefore names a `viewer_id`
and a `source` recording how it was obtained (`recording`, `assumed_shared`,
`manifest`, `inferred`). Attribution uses the layout valid at that timestamp
and propagates the source into every result.

Together these support reprocessing, changing layouts, uncertainty, and
alternative attribution algorithms. See research note 01 for the rationale.

## Frame handling

The pipeline streams: one decoded frame is held at a time, and the video is
decoded twice — once to detect tiles, once to crop and observe. The second pass
needs only the first pass's geometry, not its pixels. Holding the sampled frames
instead would make peak memory scale with recording length, which capped how
long a recording could be analyzed at all. See
[research note 17](research/17-streaming.md).

## Layout detection

Two strategies, tried in order. The first treats a gallery as tiles drawn on a
near-uniform background and takes connected foreground components; it handles
gutters and separates shared content from participants. When tiles abut there
are no gutters and that collapses to one component, so the fallback recovers the
grid from its structure — the steps that recur along nearly every line of the
frame. See [research note 15](research/15-abutting-grids.md).

Neither strategy invents a layout. A frame that looks like content rather than a
grid yields no tiles, because an invented layout misattributes every subsequent
gaze while no layout attributes none.

## Reporting

Reporting is a stage over a single document, not a second summary computed from
the events. `lookout.runrecord` holds the provenance, configuration, coverage,
degradations and diagnostics of a run; `lookout.report` renders it. Nothing in
the rendering layer recomputes a number.

The order the report presents is part of the contract: evidence before results,
and an explicit verdict on whether the run was scored at all. See
[report.md](report.md).
