# 04 — Geometric gaze baseline

## Question

Can a model-free estimator turn face landmarks into a usable gaze angle, to
serve as the baseline every learned model must beat?

## Candidates

1. Head pose only (ignore the eyes).
2. Iris offset only (ignore head pose).
3. Head pose plus iris offset within the eye.

## Decision

Option 3. Per eye, the horizontal iris offset is measured against the corner
line and the vertical offset against the lid line:

```
h = (iris_x - eye_center_x) / (eye_width / 2)      # + toward larger image x
v = (eye_center_y - iris_y) / (eye_height / 2)     # + when looking up
```

Offsets are averaged over both eyes, scaled to an eye-rotation range
(`max_eye_yaw`, `max_eye_pitch`), and added to the head pose:

```
yaw   = head_yaw   + sign_h * h * max_eye_yaw
pitch = head_pitch + sign_v * v * max_eye_pitch
```

Confidence is the product of three factors in `[0, 1]`: face size (interocular
distance vs `reference_iod`), eye openness (height/width vs an open-eye band),
and a head-pose extremity penalty.

## Evidence

`tests/test_gaze_geometric.py` pins the formula: a centred iris returns the head
angle alone; a +/-0.5 horizontal offset produces `+/- 0.5 * max_eye_yaw` with the
correct sign; a vertical offset produces positive pitch when looking up;
confidence is 1.0 for a clear frontal face and falls to 0 for near-closed eyes,
and drops for a small face and for extreme head pose.

## Risks

- The offset-to-angle scale (`max_eye_*`) and the mirroring signs are priors,
  not fitted. Mitigation: they are parameters; #18 fits them per viewer from the
  conversation prior, and #15 measures the baseline's hit-rate.
- Landmark noise at low tile resolution propagates into the offset. Mitigation:
  the size/openness confidence terms suppress unreliable frames; see note 03.
