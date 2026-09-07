# 06 — Angle-to-screen prior

## Question

Without per-person camera/screen geometry, how do we map a gaze angle to a
screen coordinate well enough to attribute it to a tile, and how do we decide a
gaze is off screen?

## Candidates

1. No mapping: attribute directly in angle space.
2. Full per-viewer calibration up front (not available from a recording).
3. A fixed linear prior for typical laptop geometry, replaceable per viewer.

## Decision

Option 3. A linear map takes the angles at the screen edges to normalized
coordinates:

```
x = (yaw - yaw_at_left) / (yaw_at_right - yaw_at_left)
y = (pitch_at_top - pitch) / (pitch_at_top - pitch_at_bottom)
```

Defaults encode the prior: horizontal about +/-16 deg, vertical 0 to -19 deg for
a camera above the screen centre. A point past an edge by more than
`off_screen_margin` is `off_screen`; a small overshoot is clamped.

## Evidence

- Geometry: a 14-inch laptop at ~55 cm subtends roughly +/-16 deg horizontally
  and ~19 deg vertically, which sets the default edge angles.
- Coarse resolution is expected and acceptable: prior video-conference work
  reports better horizontal than vertical discrimination and falling target
  hit-rate as tiles are added, which is why v1 is evaluated by hit-rate per grid
  size (note 07), not pixel error.
- The linear form is deliberately simple so #18 can fit `yaw_at_*`/`pitch_at_*`
  per viewer from weak labels without changing any downstream stage.

## Risks

- The prior is wrong for non-laptop setups (external monitors, sitting
  distance). Mitigation: parameters are per-mapping and, once fit per viewer
  (#18), replace the prior; until then the assumption is visible via
  `layout_source` downstream.
