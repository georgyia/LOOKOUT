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

| Recording | Video | Wall | Throughput |
| --- | ---: | ---: | ---: |
| Decode and tile detection alone | 60 s | 5.3 s | 11.2x realtime |
| 3-minute clip, end to end | 180 s | 16 s | 11.3x realtime |
| **Full recording, end to end** | **2561 s (42.7 min)** | **215 s (3.6 min)** | **11.9x realtime** |

The full run was then measured rather than projected, which is the point: the
estimate that it "would be slow" was wrong by an order of magnitude. Forty-three
minutes of video is under four minutes of work, and the 3-minute clip was never
necessary on performance grounds.

Throughput is flat across the three, which is the reassuring part: cost is
linear in sampled frames, so the projection from a short clip is trustworthy.

The full run also corrects a figure. Face hit rate over 96,899 tile crops is
73.7%, against 82.9% on the 3-minute clip — the clip was a more favourable
stretch than the call as a whole, and any number taken from it was optimistic.

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
