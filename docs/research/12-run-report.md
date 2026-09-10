# 12 — What a run report must contain

## Question

A run produced a duration table: who looked at whom, and for how long. Nothing
in the output said whether those durations meant anything. What has to be in a
report before its numbers can be read?

The question is not cosmetic. On a real 3-minute gallery recording the pipeline
reported that about 79% of attributions pointed at the centre tile. That was an
artifact of a coarse gaze proxy combined with the shared-layout assumption, and
nothing in the generated report distinguished it from a finding. A person had to
open the frames, work out what had happened, and write it down by hand.

## Candidates

1. Keep the duration table and put the caveats in the documentation. The report
   stays small, and the reader is trusted to remember what it cannot show.
2. Add a header to the existing report: the video name, the version, a warning.
   Cheap, and covers the most common question.
3. Make the record — not the duration table — the report: one schema-versioned
   document carrying provenance, configuration, coverage and degradations, from
   which the human formats are rendered.

## Decision

Option 3.

Option 1 fails for the reason above: the caveat that mattered was specific to
one run, and no static document could have carried it. Option 2 fails for a
subtler reason — a header is written once and then diverges from what the code
actually does, because nothing forces it to stay true.

The record is generated from the run's own state, so it cannot drift:

- **Provenance** identifies the recording by hash and names the adapters and
  their versions, because two runs with identical parameters are not comparable
  if different models produced the observations.
- **Configuration** is serialized whole. The previous `run.json` echoed five
  scalars and dropped every nested parameter block, so a run could not be
  reproduced from its own metadata. Serializing a chosen subset is how that
  happens; a test now pins the entire tree.
- **Coverage** is the funnel each stage achieved, per participant as well as in
  total. The same result over 8% of frames and over 90% of them are different
  claims, and only coverage separates them. The funnel is validated on
  construction — every direction is either mapped or rejected off screen — so a
  miscounted stage fails loudly instead of reading plausibly.
- **Degradations** are derived from coverage rather than logged in passing, which
  makes them a pure function of the run's state and therefore testable. Each
  states its impact, not just its cause.

Section order is part of the decision: provenance, coverage and degradations come
before results. A reader who stops early should stop having read the caveats, not
having read the durations.

## Evidence

`tests/test_coverage.py` pins the funnel's reconciliation and every degradation
rule, including that the shared-layout assumption is always declared.
`tests/test_runrecord.py` pins the full configuration tree against the subset
that was previously recorded. `tests/test_pipeline.py` covers the case the old
counters could not represent at all: a tile in which no face was ever found
produces no directions, and so appeared nowhere in the output; it now appears in
coverage with a hit rate of zero.

Applied to the 3-minute gallery recording, the generated report reaches unaided
the conclusions that previously had to be written by hand:

| Previously hand-written | Now |
| --- | --- |
| "Layout: equal 3x3 grid (mask detection fails on abutting tiles)" | declared by the caller as `forced_equal_grid` |
| "Gaze: YuNet 5-point proxy" | declared as `landmark_proxy`, and recorded in the adapter provenance |
| "Attribution uses assumed_shared layout" | derived: `assumed_shared_layout` |
| "No cue-protocol ground truth: hit-rate cannot be computed" | the verdict: `Accuracy: NOT MEASURED` |
| "Face detections (hit rate): 5510 / 6648 → 82.9%" | the coverage funnel |
| "That is largely an artifact of the coarse YuNet gaze proxy: frontal faces map near screen CENTER" | derived: `centre_clustered_gaze`, 82.9% of points in the central ninth |
| "about 79% of attributions point at slot_4" | derived: `concentrated_targets`, 92.8%, with entropy 0.12 |

The two figures differ because the diagnostic excludes non-participant outcomes
from the denominator: a run that is mostly `unknown` is empty, not concentrated.

One participant appears in the report that appeared nowhere in the previous
output. `slot_6` was examined in 183 crops and never resolved; with no
directions it left no trace, and the earlier summary simply listed eight
participants instead of nine.

## Risks

- Degradation thresholds are priors, not measurements. They are configurable and
  should be revisited once the evaluation harness can say which thresholds
  actually predict a bad result.
- A derived degradation cannot describe a fallback the core pipeline never sees;
  a caller that substitutes its own layout or estimator has to declare it.
- More sections invite a reader to skim past them. The rendering order is the
  mitigation, and is worth re-testing on a real reader.
