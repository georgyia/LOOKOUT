# 16 — What a run costs, and how to guard it

## Question

Every change to this project has been justified by a measurement of accuracy or
coverage. Performance had no equivalent: there was no timing instrumentation
anywhere, so nobody could say what a run cost, which stage dominated, or whether
a change had made things slower.

The gap had a concrete consequence. The example recording was analyzed as a
3-minute clip rather than in full, and the stated reason was that the whole
43-minute run "would be slow" — an estimate nobody had checked.

## Candidates

1. Time the run as a whole and record one number.
2. Time each stage and record durations.
3. Time each stage, report throughput in the terms the question is asked in, and
   guard regressions on work rather than on time.

## Decision

Option 3.

Option 1 cannot answer the only interesting follow-up: which stage to attack.

Option 2 records the right data in the wrong units. "4.2 s in `detect_tiles`" and
"0.8x realtime, an hour of recording projects to 75 minutes" are the same
measurement, but only the second answers what a user actually asked. Throughput
as a realtime multiple, plus a projection, is the reported form; per-stage
seconds and shares sit underneath it.

Timing is carried with provenance, not with results, and the report says so
outright: cost is a property of this machine and this configuration, not a
measure of result quality. Nothing about a fast run makes it more credible.

### The guard is on work, not on wall clock

A duration threshold in CI is a flaky test. Shared runners vary by more than the
effects worth catching, so such a test either fails randomly or is set so loose
it catches nothing.

What is stable on any machine is the *work* done. The tests assert that decoding
happens once per run rather than once per tile, that tile detection runs once per
sampled frame, that gaze observation runs once per tile crop, and that detection
work scales linearly with the sampling rate rather than faster.

Those catch the regression that actually happens — a stage migrating inside a
per-tile loop — which a wall-clock threshold would catch only on a quiet machine
and only once the clip was long enough. Absolute durations are recorded in the
run record for humans to compare across runs, and deliberately not asserted.

## Evidence

Measured on this machine, the YuNet observation path at 5 fps sampling:

| Recording | Video | Wall | Throughput | Face hit rate |
| --- | ---: | ---: | ---: | ---: |
| 3-minute clip | 180 s | 23.5 s | 7.7x realtime | 99.0% |
| **Full recording** | **2561 s (42.7 min)** | **501 s (8.3 min)** | **5.1x realtime** | **99.4%** |

The full run was measured rather than projected, which is the point: the estimate
that it "would be slow" was wrong by an order of magnitude. Forty-three minutes
of video is eight minutes of work.

Throughput falls with recording length because longer recordings contain more
layout changes, and segmentation work is not constant per frame. A projection
from a short clip is therefore optimistic by roughly a third here — enough to
matter, and an argument for recording cost per run rather than assuming it
transfers.

### A correction

An earlier version of this note reported 11.9x realtime and a face hit rate of
73.7% for the full recording, against 82.9% for the clip, and concluded that the
clip was an unrepresentative stretch. **Both figures were artifacts and the
conclusion was wrong.**

The full run had been produced by a script that froze the tile layout from the
first frame. On the full recording the first frame has no usable tiles, so the
layout was empty, every attribution afterwards was "no region", and the tile
crops being measured were the wrong rectangles. The hit-rate gap was a property
of that bug, not of the recording.

`lookout compare` found it in one command, by putting the two records side by
side and showing `unresolved_share` at 1.0 for the full run. Both records had
existed for some time. Nothing had compared them.

Re-run against the production layout detector, the two recordings agree: 99.0%
and 99.4%. The clip was representative after all, and the correct reading of the
original 82.9% is that a forced grid was cropping tiles badly enough to lose a
sixth of the faces in them.

That matters beyond the arithmetic. The clip was chosen partly for speed, and
the choice cost coverage: note 15 records that the call switches to a screen
share partway through, which a longer run would have surfaced earlier. A guess
about cost shaped what got analyzed, and the guess was off by an order of
magnitude.

## Risks

- Throughput here is dominated by a coarse face detector. A real landmark model
  is substantially more expensive per tile crop, and these figures will not
  transfer to the MediaPipe or L2CS paths. The record names the adapters beside
  the timing for exactly this reason.
- Cost scales with tile count as well as frame count: a nine-tile gallery does
  nine observations per frame. A larger call is proportionally more expensive
  and the projection should be read per-tile, not per-recording.
- The complexity assertions pin current call counts. Legitimate changes — batching
  observations, caching decoded frames — will change them, and the tests will
  need updating rather than silencing.
- Nothing here measures memory. A long recording holds all sampled frames in
  memory before observation begins, which is a limit these timings do not expose.
