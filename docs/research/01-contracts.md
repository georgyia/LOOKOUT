# 01 — Viewer-relative layout and raw gaze direction

## Question

What is the right shape for a raw gaze observation, and how should a layout be
represented, given that the recording shows one screen while every participant
sees their own?

## Candidates

1. Raw observation as a screen point `(x, y)`, single shared layout.
2. Raw observation as a screen point, per-viewer layout.
3. Raw observation as a camera-frame angle `(yaw, pitch)` plus head pose;
   per-viewer layout with an explicit provenance tag.

## Decision

Option 3.

- The raw unit is `GazeDirection` (angle + head pose + confidence). Mapping to a
  `GazePoint` on a screen is a separate step.
- A `Layout` names a `viewer_id` and a `source`
  (`recording | assumed_shared | manifest | inferred`).

## Evidence

- Gaze estimators output angles, not screen points. Appearance models such as
  L2CS-Net regress `(yaw, pitch)` directly (~4 deg MAE in-domain); landmark
  geometry likewise yields an angle. Turning an angle into a screen point needs
  per-person camera/screen geometry that a recording does not contain, so that
  conversion must be an explicit, replaceable step rather than baked into the
  observation.
- Prior "who looks at whom" systems (LookAtChat 2021; "Restoring Eye Contact to
  the Virtual Classroom" 2021) ran inside each client and thus had that client's
  own layout. LOOKOUT analyzes a single recording and does not, so a layout must
  be viewer-relative and must advertise how trustworthy it is. v1 uses
  `assumed_shared`; it is correct only for the recorded participant.

## Risks

- The `assumed_shared` layout is wrong for other viewers. Mitigation: the
  assumption is carried in `Attribution.layout_source`, and issue #19 replaces it
  with a manifest or an inferred layout.
- Sign conventions are easy to get wrong. Mitigation: fixed in `models.py` and
  pinned by `test_angle_sign_convention_is_pinned`; the mapping step (#10) reuses
  the same convention.
