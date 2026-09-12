# 18 — Describing a screen share rather than declining it

## Question

Note 15 gave the layout detector a rule it needed: a candidate grid must have
near-equal cells, or it is content rather than a gallery. That rule is right, and
it made the detector decline the stretch of the example recording where someone
shares their screen.

Declining was an improvement — a forced 3x3 had been attributing gaze to nine
tile positions over a shared document — but the stretch still held real people,
in a column of thumbnails down one edge, and the document itself is the most
likely thing any of them was looking at. What should the pipeline report there?

## Candidates

1. Keep declining. A screen share is not a gallery and the honest answer is no
   layout.
2. Detect the filmstrip as a gallery with one column, ignoring the content.
3. Detect the shape as what it is: one dominant content region plus a strip of
   equally sized participant tiles.

## Decision

Option 3.

Option 1 conflates two different states. "No layout" and "a layout in which most
of the screen is a shared document" are not the same claim, and only the second
lets `shared_content` be an answer. Gaze landing on the document currently
produces `unknown`, which is indistinguishable from the pipeline having failed.

Option 2 would find the participants but throw away the larger fact. It also
does not work directly: the filmstrip's boundaries, measured across the whole
frame, are diluted by the content beside them and never clear the recurrence
threshold.

The principle from note 15 carries over unchanged — a boundary is a step that
recurs along nearly every line — applied to a narrow edge band rather than to the
whole frame. `RegionKind.SHARED_CONTENT` and `attribute_point`'s handling of
non-participant regions both already existed; nothing was producing them for
this layout.

## Evidence

On the screen-share frames of the example recording, the detector finds a
content region spanning 87.2% of the width and a filmstrip of 2 to 5
participants, where previously it found nothing. Layout segmentation resolves
the recording into four intervals rather than a gallery followed by a gap:

| Interval | Layout |
| --- | --- |
| 0-150 s | 7 participants |
| 150-157 s | shared content + 2 participants |
| 157-166 s | shared content + 4 participants |
| 166 s-end | shared content + 4 participants |

Two implementation facts, both found by the detector returning nothing on real
frames rather than by reading the code:

**A narrow band is a candidate, not a filmstrip.** The first qualifying band in
this recording sat at 15% of the width and was a rule inside the document. Each
candidate split has to be validated by whether its band actually holds uniformly
sized tiles; taking the first one that fits the width bounds finds the wrong
edge.

**Thumbnails have internal edges.** A shoulder line inside a tile was splitting
that tile, producing a mix of 60-, 43- and 28-pixel spans that failed the
uniformity test which should have passed. Band cuts now enforce a minimum
spacing derived from the recurring gap, as frame-level separators already did.

The strip also rarely spans the full height — a toolbar above, a partial tile
below — so tiles are matched against the recurring size rather than requiring
every span to agree.

## Risks

- Tile count varies between 2 and 5 across the screen-share frames, because
  thumbnail contrast varies and low-contrast tiles fall below the
  empty-cell threshold. Layout segmentation absorbs the flicker, but the
  participant set during a share is less stable than during a gallery, and a
  participant dropping out of the strip is currently indistinguishable from one
  leaving the call.
- Identity is not carried across a layout change. The `slot_N` in a gallery
  interval and the `slot_N` in the following share interval are assigned
  independently and are probably different people. This is the existing
  reading-order assignment, now exposed by layouts that change more often.
- Only vertical edge strips are detected. Clients also draw filmstrips along the
  top or bottom, which would need the same treatment on the other axis.
- Attributing gaze to `shared_content` says where someone looked, not that they
  were reading it. The report's wording constraint already covers this, but the
  temptation is larger here than with a participant target.
