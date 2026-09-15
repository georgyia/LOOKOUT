# 21 — What counts as a layout

## Question

`analyze` debounced layout changes with `min_stable_seconds=2.0/target_fps`.
That reads as a duration and is a frame count: two frames, expressed in seconds.
The threshold therefore shrank as sampling improved — 2.0 s at 1 fps, 0.4 s at
5 fps, 0.2 s at 10 fps.

The defect only became visible once #54 made real runs easy at 5 fps. How long
should an arrangement hold before it counts as a layout rather than a transition?

## Candidates

1. Keep the frame count. Two samples is the minimum that can confirm anything.
2. A fixed duration. A transition is a transition regardless of how often it is
   sampled.
3. Both: a duration, with the frame count as a floor.

## Decision

Option 3, with the duration at two seconds.

Option 1 is what was there, and its consequence is that sampling rate changes
what the pipeline claims about the meeting. Measured on the example recording,
which contains a genuine change when someone shares their screen:

| Sampling | Participants reported |
| ---: | ---: |
| 1 fps | 13 |
| 2 fps | 18 |
| 5 fps | 20 |
| 10 fps | 21 |

Every one of those describes the same 7-person call. The extra participants are
transition frames: a client animating between layouts passes through
intermediate arrangements, each qualifying as stable at 0.4 s, and since
identity is correctly not carried across a reshape (note 19) every one mints
people who were never present.

This set two earlier decisions against each other. Note 13 measured that
attribution wants fine temporal resolution; note 16 measured that higher rates
are affordable. Both point toward sampling more, while this made sampling more
actively degrade the participant model.

Option 2 alone fails at coarse sampling, where a two-second layout may be two
samples or one. The floor keeps that case honest.

## Evidence

Sweeping the threshold across sampling rates, reported as intervals/participants:

| Threshold | 1 fps | 2 fps | 5 fps | 10 fps |
| ---: | --- | --- | --- | --- |
| 0.5 s | 4i/13p | 6i/18p | 7i/20p | 9i/21p |
| 1.0 s | 4i/13p | 6i/18p | 5i/18p | 6i/18p |
| **2.0 s** | **4i/13p** | 5i/18p | **4i/13p** | **3i/13p** |
| 3.0 s | 4i/13p | 4i/13p | 3i/13p | 2i/9p |

Two seconds agrees with the coarse-sampling answer at three of the four rates,
which is the best available and not a complete fix.

### What is left, and why no debounce fixes it

The residue is a different effect. A debounce merges arrangements that did not
hold long enough; it cannot recover arrangements that were never sampled. At
1 fps a layout lasting 1.5 s may produce one sample or two, and at 10 fps it
produces fifteen — so coarse sampling *misses* short layouts where fine sampling
*sees* them. Those are genuinely different observations of the same recording,
and calling them the same would require inventing the intermediate states.

Three seconds would make 1, 2 and 5 fps agree and break 10 fps, which is a
worse trade: it suppresses a real layout to make a number look stable.

The honest position is that participant count is sampling-dependent at the
margins, which is why #52 reports it as an upper bound rather than a headcount.
This narrows the bound substantially without closing it.

## Risks

- Two seconds is a prior, chosen because it agrees with the coarse-sampling
  answer on one recording. A call whose layout genuinely changes faster — a
  client switching active speaker every second — would have real changes
  suppressed.
- The floor means a run at under 1 fps silently uses a longer threshold than
  configured. It is recorded in the configuration, but the effective value is
  not.
- Nothing links participants across a suppressed transition either. Merging two
  intervals into one does not merge their identities; it avoids creating a third
  set.
- This measures one recording with one transition. A call with many layout
  changes would exercise the threshold far harder.
