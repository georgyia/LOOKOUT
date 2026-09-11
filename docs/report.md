# The run report

A run writes one canonical document, `report.json`, conforming to schema
`lookout.report`. `report.html`, `report.md` and `report.csv` are renderings of
it and recompute nothing. Read the record; do not re-derive summaries from the
JSONL artifacts.

`lookout report --out DIR [--truth FILE]` re-renders or re-scores a stored run
without re-running any stage.

## Why the record exists

A duration table answers *who looked at whom*. On its own it cannot say whether
the answer means anything — what was analyzed, with which model, over how much
of the recording, under which assumptions, and whether any of it was ever
checked. Those were previously written by hand after the fact, or not at all.
See [research note 12](research/12-run-report.md).

## Section order

The order is deliberate. A reader who stops early should stop having read the
caveats, not the durations.

1. **Verdict** — was this run scored against ground truth?
2. **Provenance** — what was analyzed, by which code and which adapters.
3. **Configuration** — every parameter, and which ones were changed.
4. **Coverage** — how much of the recording survived each stage.
5. **Degradations** — what the run settled for, and what that costs.
6. **Distribution** — does the result look like signal?
7. **Evaluation** — the scores, or an explicit absence.
8. **Results** — the durations.
9. **Limitations** — what this run cannot support.
10. **Reproduce** — the commands.

## Fields

### `provenance`

| Field | Meaning |
| --- | --- |
| `lookout_version` | Package version that produced the run. |
| `created_at` | UTC timestamp, seconds resolution. |
| `git_commit` | Commit the run was made from, when inside a repository. |
| `command` | The invocation, shell-quoted. |
| `config_hash` | SHA-256 of the full serialized configuration. Runs sharing it are parameter-comparable. |
| `video` | `path`, `sha256`, `duration`, `width`, `height`, `source_fps`. All but `path` may be absent when the `cv` extra is missing. |
| `adapters` | One entry per role (`face`, `gaze`): `implementation`, `model_path`, `model_sha256`, `library_version`. |

Adapters are provenance, not configuration: two runs with identical parameters
are not comparable if different models produced the observations.

### `config` and `config_overrides`

`config` is the whole `AnalysisConfig` tree, including every nested parameter
block. `config_overrides` lists the dotted paths that differ from the defaults,
so a reader sees at a glance what this run did differently.

### `coverage`

The funnel, overall and under `per_participant`. Validated on construction:
`face_hits <= face_attempts`, and `points + off_screen == directions`. A
miscounted stage fails loudly rather than reading plausibly.

| Field | Meaning |
| --- | --- |
| `frames` | Frames sampled after subsampling to `target_fps`. |
| `face_attempts` / `face_hits` | Tile crops examined, and those in which a face was resolved. |
| `directions` | Raw observations written to the store. |
| `points` / `off_screen` | Directions mapped onto the screen, and those rejected outside it. |
| `fixations` / `attributions` / `events` | Stable dwells, dwells matched to a layout, and sustained runs of one target. |
| `layout_sources` | Which layout provenances were in play. |

A participant appearing with `face_attempts > 0` and `face_hits == 0` is a tile
the observer never resolved. They produce no directions, and the report states
which of three cases applies rather than leaving the reader to infer it:

| Outcome | Meaning |
| --- | --- |
| `observed` | A face was resolved at least once. |
| `present, not resolvable` | The tile was examined and no face was ever found. |
| `not present in this layout` | No tile; the participant was not on screen. |

`not_visible` counts the windows in which a participant held a region but
nothing could be attributed. These are emitted as events so that absence of a
result is not read as absence of a person, and are counted separately from
attributed events so the two cannot be confused.

### `degradations`

What the run settled for. Each carries `stage`, `code`, `detail` and `impact` —
the consequence for the reader, not just the cause.

Most are *derived* from coverage and the distribution, which makes them a pure
function of the run's state and therefore testable. A caller that substitutes
its own layout or estimator has to declare that itself; the core pipeline cannot
observe it.

| Code | Raised when |
| --- | --- |
| `no_layout` | No stable layout was reconstructed. |
| `no_participant_regions` | A layout was found but held no participants. |
| `assumed_shared_layout` | The recorded layout was applied to every viewer. |
| `low_face_hit_rate` | Faces were resolved in under half the crops. |
| `high_off_screen_rate` | Over half of directions fell outside the screen. |
| `no_events` | Directions were recorded but none survived. |
| `uncalibrated_viewers` | Viewers kept the shared angle-to-screen prior instead of a fitted mapping. |
| `concentrated_targets` | One target holds most of the attributed duration. |
| `low_target_entropy` | Duration carries little information about who looked at whom. |
| `centre_clustered_gaze` | Most mapped points land in the central ninth. |
| `high_self_view` | Viewers looking at their own tile dominate. |

### `diagnostics`

The shape of the results, independent of whether they are correct. These cannot
prove a result right — only measurement does that — but they catch the more
common failure, which is a result too degenerate to be worth reading.

`normalized_entropy` is scaled by the targets *available*, not those observed:
scaling by what was seen would call a nine-tile call that collapsed onto two of
them evenly spread, when collapsing the field is itself the symptom.

Concentration excludes non-participant outcomes. A run that is mostly `unknown`
is not concentrated, it is empty, and coverage already says so.

### `timing`

What the run cost, per stage, with call counts and shares. Reported as a
realtime multiple and a projection, because that is the form the question takes.

Timing sits with provenance, not with results: cost is a property of the machine
and the configuration, not a measure of quality. Nothing about a fast run makes
it more credible. See [research note 16](research/16-run-cost.md).

### `evaluation`

`null` when the run was never scored. The report's headline turns on this: an
unscored run quotes no accuracy figure anywhere.

`unknown_breakdown` separates the four ways a prediction declines to answer:
`silent` (no prediction at all), `unknown` (no region, or an ambiguous overlap),
`low_confidence` (below the attribution floor), and `not_visible` (the
participant could not be observed). None is ever counted as a wrong answer.

### `results`

Per-viewer event counts and duration by target. Estimates, to be read against
everything above.

## Related artifacts

`report.json` is the report. The JSONL artifacts beside it remain the working
data: `gaze_raw.jsonl` is the re-run source of truth, and `layout.jsonl`,
`gaze_screen.jsonl`, `attribution.jsonl` and `events.jsonl` are the
human-inspectable intermediates. See [data-model.md](data-model.md).
