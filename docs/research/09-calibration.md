# 09 — Implicit per-viewer calibration

## Question

Can we replace the fixed mapping prior (note 06) with per-viewer parameters
learned from the recording itself, without any explicit calibration step?

## Candidates

1. Keep the fixed prior for everyone.
2. Ask users to calibrate (not possible from a past recording).
3. Use the conversation prior: listeners look at the speaker, giving weak labels
   for where a viewer's gaze should land.

## Decision

Option 3, following Siegfried and Odobez. For each of a viewer's gaze
directions, if someone else is speaking, that speaker's tile centre is a weak
target. The mapping is linear per axis, so calibration is two robust line fits
(`x = m*yaw + b`, `y = m*pitch + b`) via RANSAC, converted back into
`ScreenMappingParams`. Degenerate geometry raises so the caller keeps the prior.

## Evidence

- Prior strength: listeners look at the speaker roughly 5-8x more than at other
  listeners; weak-label precision above ~0.75 with speaker-plus-physical
  constraints (Siegfried and Odobez). That is well within RANSAC's tolerance for
  outliers.
- `tests/test_calibration.py`: RANSAC recovers a known line with 20% outliers;
  `calibrate_mapping` recovers known edge angles to <0.001 rad despite injected
  wrong labels; degenerate geometry is rejected.

## Risks

- Too few weak labels for a quiet viewer. Mitigation: require a minimum count
  and otherwise fall back to the prior.
- Speaker detection errors feed in as label noise. Mitigation: RANSAC rejection
  plus a similarity check against the prior before adopting a fit (future work);
  the head-to-head against the prior is measured on the #15 set.
