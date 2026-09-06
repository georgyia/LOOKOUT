# 03 — Minimum tile resolution for landmarks

## Question

Below what tile resolution do MediaPipe face landmarks (and the iris centres
the gaze baseline depends on) become too noisy to trust, so that a frame should
be reported `low_confidence` or `not_visible` rather than estimated?

## Candidates

Threshold expressed as:

1. Absolute tile pixel height (e.g. face crop >= N px).
2. Interocular distance in pixels.
3. Interocular distance normalized to the crop (resolution-independent).

## Decision (provisional prior, pending #15)

Use option 3 inside the confidence, because it composes with the layout scale:
the geometric baseline's `size_factor` is `iod / reference_iod` clamped to
`[0, 1]`, with `reference_iod = 0.25` of the crop. Frames whose eyes are closer
together than that scale down in confidence and, once combined with openness and
head-pose terms, fall below the attribution threshold rather than asserting a
target.

This threshold is a documented prior. It is confirmed or adjusted once the
resolution sweep runs against a real clip under the ground-truth protocol (#15).

## Evidence

Pending. The experiment is `experiments/resolution_sweep.py`: take a local clip,
downscale each tile crop in steps, run the MediaPipe observer, and record iris
jitter (std of the iris centre over a fixed head) versus crop size. The knee of
that curve sets the confidence scale. Numbers will be recorded here.

## Risks

- MediaPipe may still return landmarks on tiny crops with false confidence.
  Mitigation: rely on the geometry-derived confidence, not the detector's
  presence flag, for the `low_confidence` decision.
