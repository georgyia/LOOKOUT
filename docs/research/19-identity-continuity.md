# 19 — Identity across a layout change

## Question

Participant ids are assigned by reading order within a layout interval. What
should happen to those ids when the layout changes?

Until recently the question did not arise, because a recording was one interval.
Note 18 made screen shares visible and a single call now segments into four, at
which point the existing answer turned out to be wrong: `slot_0` was reused for
whatever tile came first in the new layout, and every per-participant number —
duration, coverage, every diagnostic — is keyed by that id and summed across the
run. The results were arithmetic over two different people.

## Candidates

1. Keep reusing ids. Reading order is deterministic and the numbers still add up.
2. Link across changes with face re-identification, which works regardless of
   how the screen is reshaped.
3. Carry an id only where something supports it, and mint fresh ids otherwise.

## Decision

Option 3.

Option 1 is the status quo and produces confidently wrong aggregates. It is the
same failure the project's conservative default exists to prevent, in a place
nobody had looked: asserting that two tiles in different layouts are the same
person is an invention, and the rule is that a participant is never invented.

Option 2 works and is the wrong default. `lookout.reid` exists, is tested, and is
deliberately opt-in and in-memory for the reasons in note 11. Making identity
depend on a biometric comparison by default would trade a correctness problem
for a privacy one.

Geometry is the evidence available without either cost. A tile that stays in
nearly the same place across a change is the same participant; the overlap
threshold is what "nearly" means. That covers the common case — someone joining
or leaving a gallery — without renumbering everyone else. When the screen is
reshaped there is no such evidence, so new ids are minted and `identity_breaks`
records that nothing was carried.

## Evidence

On the example recording:

| Interval | Participants |
| --- | --- |
| 0-150 s | slot_0 - slot_6 |
| 150-157 s | slot_7, slot_8 |
| 157-166 s | slot_9 - slot_12 |
| 166 s-end | slot_9 - slot_12 |

Two total breaks are reported, at 150 s and 157 s. The last change carries its
ids, because the filmstrip holds still. Before this, all four intervals used
`slot_0` onward and the report summed them.

### A pre-existing bug this exposed

`detect_tiles` sorted tiles by rounded `y` then rounded `x`, which is not reading
order. Two tiles in the same row detected a pixel apart — 0.52 against 0.53 —
order by `y`, so the bottom-right tile precedes the bottom-left one. Slot
numbering therefore depended on pixel noise and changed between frames of an
unchanging layout.

It surfaced only because continuity matching made the inconsistency visible: the
same tile held `slot_2` in one frame and `slot_3` in the next. Tiles are now
grouped into rows before being ordered within them, which is what reading order
meant all along.

This is worth recording as a class of bug rather than an incident. The original
sort was correct for the case it was written against and silently wrong
everywhere else, and no test caught it because every test used tiles at exact
positions. The real frames did not.

## Risks

- Geometric continuity cannot survive a reshape, so a participant present
  throughout a call appears as two people either side of a screen share. That is
  honest rather than correct, and the report should not be read as a headcount.
- The overlap threshold is a prior. A client that animates tiles into place, or
  reflows a gallery as people join, will break continuity where a human would
  not.
- Greedy matching claims each previous id once, by best overlap. Tiles rarely
  swap places, but if two did, one would be matched wrongly rather than left
  unmatched.
- Identity is still per-viewer-layout. Nothing links a participant across
  *viewers*, which is what a manifest or re-identification would provide.
