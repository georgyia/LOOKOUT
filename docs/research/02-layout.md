# 02 — Gallery layout detection

## Question

How do we recover participant tile rectangles from a gallery-view frame without
a trained model, and how do we know when the layout changes?

## Candidates

1. Fixed grid assumption (rows x cols from participant count).
2. Template/UI matching per meeting client (Zoom/Meet/Teams chrome).
3. Background mask -> connected components -> aspect filter, with a dominant
   region treated as shared content.

## Decision

Option 3, matching the approach validated for video-conference gaze work
("Restoring Eye Contact to the Virtual Classroom", 2021): estimate the
background from the frame border, threshold the per-pixel difference, take
connected components, and keep those whose area and aspect ratio are tile-like.
A component covering more than `shared_area_fraction` of the frame is labeled
`shared_content`. Detection yields geometry only; identity is assigned later.

Layout changes are found by fingerprinting the quantized tile arrangement per
frame and requiring a new fingerprint to persist for `min_stable_seconds`
(hysteresis) before it starts a new interval, dated to its first appearance.

## Evidence

Deterministic on rendered fixtures (see `tests/test_layout.py`): exact tile
counts and rectangles for 2x2, 3x3, and an uneven last row; correct split of
one shared region plus three tiles; and a single flicker frame that does not
create a spurious interval.

## Risks

- Client chrome (toolbars, headers) can survive the background threshold and
  appear as extra regions. Mitigation: the aspect/fill filters drop most of it;
  a `ui` region kind exists for the rest; per-client tuning is future work.
- Low-contrast backgrounds or custom virtual backgrounds weaken the mask.
  Mitigation: `DetectionParams` is tunable; the eventual answer for unreliable
  frames is fewer or no tiles rather than wrong ones.
- Distinguishing a large single participant tile from shared content is
  ambiguous from geometry alone; v1 calls any dominant region shared content.
