# 14 — Replacing assumptions with facts

## Question

Note 13 measured the accuracy a gallery demands: 3x3 attribution survives about
1 degree of angular error and is unusable past 6, while published appearance
models sit near 4. That reads as a dead end until you separate the sources of
error. How much of the pipeline's error is model error, and how much is
assumption?

## Candidates

1. Wait for better gaze models. The 1-degree budget is what it is.
2. Reduce the demand: attribute to screen halves or quadrants rather than tiles,
   and accept a coarser answer.
3. Remove the systematic error first, and find out what the model has to cover.

## Decision

Option 3, before either of the others is worth arguing about.

Two assumptions in the v1 pipeline contribute error that no gaze model can fix,
because the error is not in the angle:

**The shared layout.** Every viewer is attributed against the layout visible in
the recording. That layout is only what the recording participant saw. For
everyone else the tiles are elsewhere, so a *correct* angle resolves to the
wrong person. The magnitude scales with how many viewers are not the recorder,
which in a nine-tile gallery is eight of nine.

**The shared mapping prior.** `ScreenMappingParams` defaults encode roughly
+/-16 degrees horizontally and 0 to -19 vertically: one laptop, camera above
centre. A participant sitting closer sweeps a wider angle across the same
screen; one on a large display sweeps a narrower one; a camera below the screen
inverts the vertical relationship entirely. Each is a *bias*, not noise — every
target for that viewer shifts the same way — and bias is what the pipeline
tolerates least, because the conservative default converts it into confident
silence rather than visible error.

Both already had tested implementations that nothing could reach:
`lookout.manifest` loaded per-viewer layouts, `lookout.calibration` fit
per-viewer mappings from weak labels, and `lookout.speaker` produced the labels.
This wires all three and measures the result.

Speaker segments become a persisted observation artifact. Deriving them needs
the frames; attribution has neither video nor model. Writing them during
observation is what lets a stored run be re-interpreted with calibration later,
and follows the existing split — observe first, interpret later.

## Evidence

Hit rate on the synthetic 2x2 fixture, where the viewer's true geometry differs
from the prior. The cue schedule supplies both the ground truth and, through the
speaker cue, the weak labels.

| Viewer's real geometry | Shared prior | Fitted |
| --- | ---: | ---: |
| Matches the prior | 100% | 100% |
| Sitting closer (+/-30 deg) | 0% | 100% |
| Large display (+/-24 deg) | 50% | 100% |
| Camera below the screen | 0% | 100% |

Two of four plausible setups score *zero* under the shared prior and recover
completely once fitted. Fitting a viewer whose geometry already matched the
prior costs nothing, which is the regression that mattered most.

Per-viewer layouts are covered separately in `tests/test_per_viewer.py`: a
viewer whose screen has two participants swapped relative to the recording is
attributed to the wrong one under the shared assumption and the right one under
a manifest, from identical gaze angles.

The practical reading, against note 13: a meaningful share of what looked like a
model-accuracy problem is geometry the pipeline was assuming rather than
measuring. That does not rescue the 3x3 target — a fitted mapping still needs an
estimator inside its error budget — but it means the budget is being spent on
model error rather than on a guess about someone's desk.

## Risks

- The weak labels come from the conversation prior: listeners look at the
  speaker more often than at anyone else. Where that prior fails — someone
  reading shared content, a call with no dominant speaker, a participant looking
  at their own tile — the labels are wrong. RANSAC rejects a minority of bad
  labels, not a systematically wrong majority.
- The synthetic evidence above uses an *exact* speaker cue, so it measures
  whether the fit recovers a known mapping, not whether real speaker detection
  supplies usable labels. The visual highlight cue is the weakest link and is
  unmeasured on real recordings.
- A degenerate fit is worse than a documented guess, because it looks fitted.
  Calibration falls back to the prior on every failure — no layout, no segments,
  too few labels, degenerate geometry — and records which viewers were fit, on
  how many labels, and why not.
- `min_calibration_labels` is a prior, not a measurement. Too low and the fit
  tracks the noise in a handful of glances.
- A manifest is user-supplied and unverified. It replaces a stated assumption
  with a stated claim; `source = manifest` keeps the distinction visible.
