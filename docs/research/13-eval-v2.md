# 13 — Making a hit rate readable

## Question

`lookout evaluate` reported a hit rate, an unknown rate and off-screen recall.
None of those is interpretable alone. What has to accompany a hit rate before it
supports a conclusion, and how is any of it exercised without a recorded cue
session?

The concrete failure: on the recorded gallery clip the pipeline attributed most
duration to one tile. Had ground truth existed for that recording, a constant
"always the centre tile" estimator would have posted a respectable hit rate, and
nothing in the harness would have separated that from a working estimator.

## Candidates

1. Report the hit rate and rely on the reader to know the chance level.
2. Report the hit rate alongside chance (1/N for an N-tile gallery).
3. Report it alongside a family of baselines, including one computed from the
   model's own predictions, with an interval and a calibration check.

## Decision

Option 3, plus a synthetic fixture so any of it can run at all.

Chance alone (option 2) is too weak. On a real call the targets are not uniform:
people do look at the speaker more, so an estimator exploiting nothing but that
prior beats 1/N comfortably. Four baselines are reported:

| Baseline | What it rules out |
| --- | --- |
| `always_unknown` | Trivially 0; anchors the floor. |
| `uniform_random` | Guessing with no information. |
| `majority_truth` | The best possible constant answer. |
| `model_modal_target` | **The model's own most frequent answer, applied everywhere.** |

The last is the sharpest. An estimator scoring no better than its own modal
target has no per-interval discrimination, however high the raw rate looks, and
that is exactly the shape the gallery clip produced. `beats_baseline` is a
first-class property; the report renders a run that fails it as a warning rather
than a result.

Three further changes:

- **Bootstrap 95% interval** on the hit rate, so a rate over 12 cues does not
  read like one over 1200.
- **The unknown rate is split** into silent, explicit unknown, and
  low-confidence. Silence and a border rejection are different failures and were
  previously collapsed. A wrong answer is now counted separately from declining
  to answer.
- **Reliability bins and expected calibration error.** Confidence gates
  attribution and appears on every event, but had never been checked against how
  often such predictions were right.

Interval sampling becomes optional-dense. At the midpoint a 30-second window
counts the same as a one-second one, and a prediction that held throughout is
indistinguishable from one that happened to be right in the middle.

## Evidence

`tests/synthetic.py` generates a run whose answers are known by construction:
directions are the exact inverse of the angle-to-screen projection onto an
intended tile centre, so a correct pipeline must recover the tile it started
from. The cue schedule — including deliberate off-screen and no-observation
windows — is the ground truth.

Hit rate on a 3x3 gallery against injected gaussian angular noise:

| Angular noise | Hit rate | Unknown |
| ---: | ---: | ---: |
| 0.0° | 100% | 0% |
| 1.1° | 100% | 0% |
| 2.3° | 72% | 28% |
| 3.4° | 50% | 50% |
| 5.7° | 11% | 89% |
| 8.6° | 0% | 100% |

Two things this settles.

**The accuracy a 3x3 gallery demands.** Attribution survives about 1° of angular
error and is unusable past 6°. Note 05 records appearance models at roughly 4°
mean angular error on Gaze360 — mid-table on this ladder. A 3x3 gallery is
therefore at or beyond the limit of what any currently available estimator can
support, and the provisional 60% target in note 07 should be read as demanding a
model materially better than the state of the art, not as a tuning goal.

**The failure mode is silence, not error.** At 8.6° the pipeline declines to
answer rather than guessing: unknown rises to 100% while wrong answers stay
near zero. The conservative default holds under stress, which is the property
that was asserted in the architecture and is now measured.

The tests also pin that more noise never scores better and that a finer grid is
harder — properties a fixture passing only at zero noise would not establish.

## Risks

- The synthetic fixture validates mapping, smoothing, fixation detection,
  attribution, aggregation and scoring. It says nothing about face detection,
  landmarks, or any gaze model. A report generated from it names its adapter
  `synthetic` so the distinction survives into the output.
- Gaussian angular noise is a convenient error model, not a measured one. Real
  estimator error is biased and correlated over time, and correlated error will
  produce confident wrong answers where independent noise produces unknowns.
  The ladder above is therefore optimistic about the failure mode.
- `model_modal_target` is undefined when a model never names a participant; it
  reports 0 there, which is correct but uninformative.
- The success criteria in note 07 should be restated relative to a baseline.
  A 60% hit rate on 3x3 means something very different at chance 11% than
  against a majority-truth baseline of 55%.
