# 07 — Ground-truth protocol

## Question

How do we obtain ground truth for who each participant was looking at, so that
design choices (geometric vs appearance gaze, the mapping prior, thresholds) are
settled by measurement instead of assertion?

## Candidates

1. Hand-annotate recordings frame by frame (accurate, very costly).
2. Infer truth from the conversation (speaker as a weak label) — circular with
   the calibration we want to evaluate.
3. A cue protocol: script where each participant should look, then record.

## Decision

Option 3. A cue script highlights one tile at a time on a fixed schedule while a
participant joins a real gallery call and records themselves. The schedule is
the ground truth: for each cue window we know the intended target and the grid
size. This yields `truth.jsonl` lines:

```json
{"viewer_id": "self", "start_time": 12.0, "end_time": 15.0, "target": "slot_2", "grid": "3x3"}
```

Cues also include deliberate off-screen and away windows so `off_screen` and
`unknown` are exercised, not just hits.

`lookout evaluate` scores predicted events against this timeline and reports:

- **hit-rate**, overall and per grid size (2x2, 3x3, ...),
- **unknown-rate** (silent or unknown/low-confidence predictions),
- **off-screen recall**.

A silent or unknown prediction never counts as a hit, honoring the conservative
default: preferring `unknown` to a guess is not penalized as a wrong answer, but
it is not rewarded as a right one either.

## Evidence

The harness is validated on synthetic events and truth in
`tests/test_evaluate.py`. Real numbers are produced by running the protocol on a
self-recorded call; all such recordings and `truth.jsonl` files stay local and
gitignored. Results, once collected, are summarized in notes 03, 05, and 06.

## Provisional success criteria (hypotheses, not assumptions)

- >= 80% hit-rate on a 2x2 grid, >= 60% on 3x3, reported alongside the
  unknown-rate. These are targets to test, and may be revised by the evidence.

**Revised by note 13.** A hit rate is meaningless without the baseline it is
measured against: 60% on a 3x3 grid is a strong result against chance (11%) and
a weak one against a majority-truth baseline of 55%. State these targets as a
margin over the best baseline, which `lookout evaluate` now reports.

Note 13 also measures what these targets demand of the estimator: 3x3
attribution survives about 1 degree of angular error and is unusable past 6.
Published appearance models sit near 4 degrees, so the 3x3 target is a bet on
better models rather than better tuning.

## Risks

- One self-recording is a single subject and setup; it validates the pipeline
  but not population accuracy. Mitigation: treat early numbers as directional and
  widen the subject pool before any strong claim.
