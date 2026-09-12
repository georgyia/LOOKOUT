# 17 — Bounding memory instead of recording length

## Question

Note 16 measured what a run costs in time and recorded that nothing measured
what it costs in memory. What does it cost, and does it limit anything?

## Candidates

1. Leave it. Recordings are short in practice, and the cost is paid once.
2. Sample fewer frames, or downscale them, so the held set stays smaller.
3. Stream: hold one frame at a time and decode the video twice.

## Decision

Option 3, once the measurement made the shape of the problem clear.

`analyze` materialized every sampled frame before observing any of them, so peak
memory grew with recording length:

| Video held | Peak RSS |
| ---: | ---: |
| 1.7 min | 0.60 GB |
| 5.0 min | 0.74 GB |
| 10.0 min | 1.53 GB |

One sampled 854x480 frame is 1.23 MB, so the full 43-minute recording is roughly
16 GB of frames held at once. Option 1 fails on that alone, but the more
compelling reason is how it scales: doubling `target_fps` doubles it, and a
1080p source is six times a 480p one, so a 40-minute 1080p call is around 90 GB.
Recording length was capped by RAM, invisibly, and the failure on arrival is an
allocation error rather than a degradation the report could explain.

Option 2 trades accuracy for memory, which is the wrong currency. Sampling rate
and resolution are quality decisions and should be made on quality grounds.

Frames are needed twice — once to detect tiles, once to crop and observe — but
the second pass needs only the first pass's *geometry*, a few hundred bytes per
frame, not its pixels. So the second pass can re-decode.

## Evidence

The cost of the extra pass, over 1500 frames:

| Work | Time |
| --- | ---: |
| Decode only | 3.3 s |
| Decode + tile detection | 32.6 s |

Decoding is about a tenth of the stage that dominates a run, so a second pass is
roughly a 10% increase on that stage — against removing a ceiling entirely.

Measured end to end over ten minutes of video:

| | Before | After |
| --- | ---: | ---: |
| Peak RSS | 1.53 GB | **0.18 GB** |
| Growth with length | linear | flat |

### On testing this

The first version of these tests passed against the *old* implementation, which
makes them worthless, and it is worth recording why. They monkeypatched the
frame source with a generator that tracked its own bookkeeping list, so they
measured the generator's behaviour rather than the pipeline's retention.

The working version takes weak references to the decoded arrays and drops the
source's own reference before yielding, so anything still alive afterwards is
held by the pipeline. Checked against the previous implementation it fails as it
should: retention grows from 20 to 80 frames as the recording does, where the
streaming version holds one.

A byte threshold would have been the obvious test and the wrong one. It is
machine-specific, and it would pass on a large runner long after the property it
protects had been lost.

## Risks

- Two decode passes assume decoding is deterministic: the same frames at the
  same timestamps both times. That holds for constant-frame-rate files, which is
  what `decode_video` already assumes for its timestamps, but a
  variable-frame-rate source could in principle desynchronise the passes. A test
  asserts every sampled frame reaches the observation pass, which would catch a
  gross mismatch but not a subtle one.
- Memory is now dominated by whatever the gaze adapter holds, which this change
  does not touch and which is model-specific.
- `per_frame_tiles` still grows with recording length. It is geometry rather
  than pixels, so it is four orders of magnitude smaller, but it is not bounded
  either — a recording long enough would eventually reach it.
