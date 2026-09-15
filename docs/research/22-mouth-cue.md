# 22 — A cue that was never reached

## Question

`speaker.mouth_open_ratio` prefers MediaPipe's `jawOpen` coefficient and falls
back to a lip-landmark gap. How much of the speaker signal does the mouth cue
actually contribute?

None. The preferred branch had never run.

## What was wrong

`MediaPipeFaceObserver` filtered blendshapes to eye coefficients:

```python
if category.category_name.startswith("eyeLook"):
```

so `jawOpen` was discarded before any consumer saw it. The branch reading it was
reachable only from tests, which construct a `FaceObservation` by hand — which is
exactly why it looked covered. Every test of the preferred path passed, and the
path did not exist in production.

Note 20 then added a second observer with different landmark coverage, and the
fallback acquired a worse failure. It measures between landmark indices 13 and
14, which `YuNetFaceObserver` does not populate, so it computed the distance
between two zeroed points and returned a plausible small number. Not an error, a
fabrication: indistinguishable from a closed mouth.

## Decision

- Keep `jawOpen`. The filter exists to avoid carrying forty-odd categories, not
  to exclude a value a stage reads.
- Return `None` rather than a number when there are no usable mouth landmarks.
  The absence of a cue is not a closed mouth, and this project's whole reporting
  discipline is built on that distinction being visible.
- Add corner separation as a third source. YuNet detects both mouth corners and
  was discarding them; the signal is weaker than lip gap — it widens with a smile
  as well as with speech — but it is measured rather than assumed.

## Why it matters beyond dead code

The mouth is one of two speaker cues. The other is the active-speaker highlight,
which depends on a meeting client drawing a ring around a tile. That is
client-specific, and the example recording happens to support it; a recording
from a client that does not would yield no speaker segments at all.

Speaker segments are what `--calibrate` fits per-viewer mappings from (note 14),
and note 14 recorded the weak labels as the weakest link in that chain —
demonstrated with an exact cue, never measured with real detection. A second
independent cue is the obvious way to make that less fragile, and it had been
behind a filter since it was written.

## Evidence

The regression test asserts `jawOpen` survives the filter, so the branch cannot
silently die again. The declining path is covered against an observation with
eyes and no mouth, which previously returned a number.

Ordering is pinned: blendshape over lip gap over corners, with the lip gap
winning when both it and the corners are present.

This note deliberately reports no accuracy figure. Nothing here measures whether
the mouth cue identifies speakers correctly — only that it is reachable and that
it declines rather than fabricating. The measurement needs a recording with
known speaker turns, which is the same cue-protocol gap note 07 describes.

## Risks

- A test constructing the object a stage consumes will pass whether or not any
  producer builds that object that way. Every test here still does that, because
  the alternative needs a real model; the mitigation is the separate assertion
  about the filter, which is the only part that pins the producer.
- Corner separation conflates smiling with speaking. It is last in the ordering
  for that reason, and a run relying on it should be treated as weaker evidence
  than one with a blendshape.
- `_present` treats a landmark at the origin as absent. A real landmark at
  exactly (0, 0) would be misread, which cannot happen for a mouth inside a tile
  crop but is an assumption rather than a check.
- Nothing yet feeds the mouth cue into `analyze`. The highlight remains the only
  cue wired into observation; this makes the second one usable, not used.
