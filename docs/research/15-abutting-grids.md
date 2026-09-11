# 15 — Detecting galleries without gutters

## Question

`lookout analyze` could not process the example recording. Tile detection
assumed a gallery is video drawn on a near-uniform background, so tiles are the
connected foreground components. Many clients draw tiles edge to edge, and then
the whole gallery is one component: the detector returned a single region.

That is why the recording was analyzed by a throwaway script that imposed an
equal 3x3 grid, and why every report from it carried a `forced_equal_grid`
degradation admitting that tile boundaries were assumed rather than measured.

How is a grid found when nothing separates the tiles?

## Candidates

1. Keep the background-difference detector and require the user to state the
   grid shape.
2. Detect tile borders as strong edges, using gradient magnitude projected onto
   each axis.
3. Detect tile borders by how *consistently* a step recurs along each axis,
   rather than by how strong it is.

## Decision

Option 3, as a fallback behind the existing detector.

Option 1 pushes a measurement onto the user and cannot handle a layout that
changes mid-call. Option 2 is the obvious version and does not survive contact
with real frames: a person in a quiet tile produces a stronger average gradient
at their silhouette than a border does, and a border beside a high-contrast tile
is drowned by it.

The property that distinguishes a border is recurrence. A border is a step
present on nearly every line of the frame; an edge inside one tile appears only
on the lines crossing it. Scoring each position by the fraction of lines on
which the step is locally prominent separates the two by an order of magnitude,
which makes the threshold uninteresting to tune.

Candidate grids are then required to have near-equal cells, because gallery
cells are. Without that check the detector accepts structured content — a table,
a slide — as an irregular grid. That failure is worse than finding nothing: an
invented layout misattributes every subsequent gaze, while no layout attributes
none.

The background-difference detector stays first in line. It handles gutters and
distinguishes shared content from participants, which the grid detector cannot.

## Evidence

Consistency scores, as the fraction of lines on which a step is locally
prominent:

| Frame | True borders | Everything else |
| --- | ---: | ---: |
| Synthetic 2x2, no gutters | 0.99, 1.00 | <= 0.06 |
| Synthetic 3x3, no gutters | 0.98 - 1.00 | <= 0.07 |
| Example recording, gallery | 0.87 - 0.99 | — |
| Single tile, no grid | — | max 0.07 |

On the example recording, sampled at 1 fps over 181 frames:

- 143 frames yield a 3x3 grid with **seven participants and two empty corner
  slots**, matching what a person reading the frames had recorded by hand.
- 38 contiguous frames at the end yield nothing. The call switches to a screen
  share there: a document fills the frame with a small participant filmstrip
  down one edge. The uniformity check rejects the document's table structure
  rather than reading it as a gallery.

That second result is the more useful one. The forced 3x3 the experiment script
imposed was attributing gaze to nine tile positions over a shared document for
the last fifth of the clip. Declining is not a gap in coverage there; it is the
removal of twenty percent of a previous run's output that should never have
existed.

Three implementation details were wrong before they were right, and each is now
pinned by a test:

- Candidates ranked by position rather than by consistency let a weak early peak
  claim the minimum-spacing budget and crowd out the true border beside it,
  turning a 2x2 into a 2x3.
- Measuring cell emptiness across the border rather than inside it let a single
  boundary row, carrying the neighbouring tile's brightness, make an empty slot
  look like video.
- Triggering the fallback on total tile count rather than participant count let
  a spurious whole-frame "shared content" region suppress it.

## Risks

- Screen-share-with-filmstrip is now *detected as absent* rather than
  misattributed, which is correct but not useful. Those small tiles are real
  participants and a later change should find them; the roadmap's hidden and
  not-visible participants item is the place for it.
- A gallery with genuinely unequal cells — some clients enlarge the active
  speaker — is rejected by the uniformity check. This is a deliberate trade: the
  alternative accepts content as a grid.
- Consistency is computed on a single frame. A static overlay spanning the frame
  (a banner, a caption bar) would score as a border. Nothing in the sampled
  recording did, but a temporal check across frames would be the defence.
- Empty-slot rejection uses pixel variation, so a participant with a perfectly
  static camera and a plain background could in principle be read as an empty
  slot. Face detection would disagree, and nothing currently reconciles them.
