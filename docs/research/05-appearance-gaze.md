# 05 — Appearance model vs baseline

## Question

Does a learned appearance model (L2CS-Net) estimate gaze more accurately than
the geometric baseline at meeting-tile resolution, enough to be the default?

## Candidates

1. Geometric baseline only (#7).
2. L2CS-Net appearance model.
3. Both, choosing per frame by confidence.

## Decision

Keep the baseline as the default and expose L2CS-Net behind the same
`GazeEstimator` interface. The appearance model becomes the default only if it
beats the baseline on the ground-truth set (#15); otherwise it stays optional.
Both feed the identical raw store and downstream stages, so switching is a
configuration choice, not a code change.

## Evidence

- Reported model accuracy: ~3.92 deg MAE on Gaze360, ~4.16 deg on MPIIFaceGaze
  (224x224 face crop, ImageNet-normalized). These are in-domain numbers; tile
  crops in a gallery are far smaller and compressed, so in-domain MAE is an
  upper bound on what to expect here.
- The head-to-head hit-rate comparison at tile resolution is pending and will be
  produced by `lookout evaluate` under the protocol in note 07. Numbers recorded
  there and summarized here once available.

## Risks

- Licence: L2CS-Net code is MIT; the Gaze360 weights are research/non-commercial
  only. Compatible with this project's non-commercial licence; documented, and
  weights are never auto-downloaded.
- Domain gap: tiny, compressed tiles differ from the training distribution.
  Mitigation: the baseline remains available and the choice is evidence-based.
- Sign/frame mismatch between the model output and our convention. Mitigation:
  calibration in #18; until then the raw angle is stored verbatim.
