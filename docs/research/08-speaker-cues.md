# 08 — Speaker context from visual cues

## Question

Can we estimate who is speaking from video alone, to enable weak supervision
(#18) and richer analytics, without using audio?

## Candidates

1. Audio diarization (out of scope here; a separate later issue).
2. Active-speaker highlight detection (the border many clients draw).
3. Mouth-motion from landmarks/blendshapes.

## Decision

Use both visual cues and keep them behind small functions:

- `detect_highlighted_tile` scores each tile by how much its border ring differs
  from its interior; the highlighted tile stands out, uniform tiles do not.
- `mouth_open_ratio` uses the MediaPipe `jawOpen` blendshape when present, else
  the inner-lip gap normalized by interocular distance.

Per-frame flags are merged into `SpeakerSegment`s with a minimum duration and
gap tolerance.

## Evidence

`tests/test_speaker.py`: highlight detection finds a bordered tile and returns
`None` when tiles are uniform; the mouth ratio grows with the lip gap and prefers
the blendshape; aggregation merges same-speaker runs and drops short glances.

## Risks

- Not every client draws a highlight, and colors vary. Mitigation: the interior
  contrast score is color-agnostic; mouth motion is an independent fallback.
- Mouth motion also fires on non-speech (yawning, chewing). Mitigation: it is a
  weak cue used only as a prior; #18 treats its labels as noisy and robustly
  rejects outliers.
