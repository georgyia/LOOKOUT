# Roadmap

## Phase 0 — foundations

- [x] Repository skeleton
- [x] Core gaze/layout data contracts
- [x] Basic attribution
- [x] Tests
- [x] Issue/PR/CI automation

## Phase 1 — video

- [x] Video decoding
- [x] Frame sampling
- [x] Timestamp management
- [x] Participant (tile) detection
- [x] Slot persistence and optional name linking

## Phase 2 — layout

- [x] Meeting UI region detection
- [x] Galleries whose tiles abut
- [x] Participant-to-region assignments
- [x] Layout transitions
- [ ] Hidden/not-visible participants

## Phase 3 — gaze

- [x] Head pose
- [x] Eye gaze (geometric baseline)
- [x] Appearance estimator adapter
- [x] Confidence calibration (geometry-based)
- [x] Raw gaze stream (store)

## Phase 4 — attribution

- [x] Gaze-to-region intersection with margin confidence
- [x] Temporal smoothing and fixations
- [x] Unknown/off-screen classification
- [x] Participant attribution
- [x] Interaction events

## Phase 5 — analysis

- [ ] Speaking context
- [x] Gaze timelines
- [ ] Interaction graph
- [x] JSON/CSV export
- [x] Interactive (static HTML) report
- [x] Run record with provenance and coverage
- [x] Degradation and distribution diagnostics
- [x] Evaluation harness
- [x] Baselines, calibration, and a synthetic ground-truth fixture

## Phase 6 — research extensions

- [x] Implicit per-viewer calibration
- [x] Per-viewer layouts (manifest)
- [ ] Speaker diarization
- [ ] Opt-in face re-identification

## Phase 7 — optimization

- [ ] GPU/MPS acceleration
- [ ] Batch processing
- [ ] Model caching
- [ ] Benchmarks

Live meeting integrations are intentionally later.
