# 20 — Shipping the observer that works

## Question

Every published measurement about a real recording — the 3x3 grid recovery in
note 15, the corrected throughput in note 16, the identity breaks in note 19 —
came from `experiments/run_real_clip.py`, which is untracked. None of it is
reproducible from a clone.

The script existed because `lookout analyze` hardcoded `MediaPipeFaceObserver`
and MediaPipe's Face Landmarker does not run on this machine. What should the
pipeline do on hardware where its default adapter fails?

## Candidates

1. Nothing. MediaPipe is the right model; a machine that cannot run it is not
   supported, and the script stays a local workaround.
2. Ship the workaround as a second adapter.
3. Ship it, and make the pipeline state what it costs.

## Decision

Option 3.

Option 1 is defensible as a modelling position and untenable as an engineering
one. The untracked script has needed repair after four separate interface
changes it was not covered by, it silently froze a layout for long enough to
corrupt a published figure, and every number this project has reported about
real video depends on it. A dependency that important should not be outside the
repository.

Option 2 on its own is worse than either. YuNet gives five landmarks — both
eyes, nose tip, mouth corners — and no iris. It can see where a head points and
not where its eyes do. An adapter that quietly produced a `GazeDirection` from
that would be indistinguishable in the output from one that measured eye
direction, which is the specific confusion this project's reporting exists to
prevent.

So the adapter reports the eyes as open and centred. Zero is the only eye
contribution justifiable from data that does not contain one, and the geometric
estimator therefore returns head orientation alone. A run using it raises
`no_iris_signal`, naming what is missing and what it costs, rather than leaving
a reader to infer it from a low score.

MediaPipe stays the default. This is a fallback for hardware, not a modelling
preference.

## Evidence

`lookout analyze --observer yunet` runs the example recording end to end in 33
seconds: 901 frames, a 96.0% face hit rate, and five degradations reported.

Four of those five were already derived from run state. The fifth,
`no_iris_signal`, is declared by the adapter choice. Together they say what the
run is: a layout the pipeline reconstructed, observed by a detector with no eye
signal, attributed under a shared-layout assumption, producing a distribution
concentrated on the centre tile.

That last part is the check that matters. `centre_clustered_gaze` fires at 83.6%
on this adapter's output, as it did on the script's, and was not tuned around.
The adapter is shipped with the warning that says not to trust it for
neighbouring tiles, which is the only form in which shipping it is honest.

Roll is the one thing this observer determines better than expected: two eye
points genuinely fix it, where MediaPipe derives it from a transformation
matrix.

## Risks

- Shipping a head-pose adapter invites using it as a gaze model. The degradation
  is the mitigation and it is only as good as the report being read.
- The angle scaling constants were fitted by eye against one recording. They are
  a documented prior, like the mapping prior in note 06, and nothing has
  measured them against ground truth.
- "Eyes open" is an assumption the observer cannot verify. It avoids
  understating confidence for an unseen reason, at the cost of overstating it
  when someone blinks.
- Two adapters now produce `FaceObservation`, and only one fills `blendshapes`.
  Anything depending on those must handle their absence; the mouth cue did not,
  and [note 22](22-mouth-cue.md) records what that cost.
