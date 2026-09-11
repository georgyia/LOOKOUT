# Research notes

Every design decision that could be settled by evidence gets a short note here,
numbered to match the issue that produced it. A note is at most one page and
answers five questions:

1. **Question** — what had to be decided, and why it matters for the pipeline.
2. **Candidates** — the options considered.
3. **Decision** — the option chosen.
4. **Evidence** — the numbers that justify it (from the evaluation harness or a
   synthetic experiment under `experiments/`), or an explicit note that the
   decision is a documented prior pending evaluation.
5. **Risks** — what could invalidate the decision, and how it is revisited.

Notes describe the technical work only. Keep prose tight; prefer a table of
numbers over paragraphs.

## Template

```markdown
# NN — Title

## Question

## Candidates

## Decision

## Evidence

## Risks
```

## Index

- `01-contracts.md` — viewer-relative layout and raw gaze direction (#2)
- `02-layout.md` — gallery layout detection (#4)
- `03-landmarks.md` — minimum tile resolution for landmarks (#6)
- `04-geometric-gaze.md` — geometric gaze baseline (#7)
- `05-appearance-gaze.md` — appearance model vs baseline (#8)
- `06-screen-mapping.md` — angle-to-screen prior (#10)
- `07-eval-protocol.md` — ground-truth protocol (#15)
- `08-speaker-cues.md` — speaker context from visual cues (#17)
- `09-calibration.md` — implicit per-viewer calibration (#18)
- `10-per-viewer-layouts.md` — manifest and inferred layouts (#19)
- `11-reidentification-privacy.md` — re-identification privacy review (#20)
- `12-run-report.md` — what a run report must contain (#27)
- `13-eval-v2.md` — making a hit rate readable (#30)
- `14-per-viewer-accuracy.md` — replacing assumptions with facts (#35)
- `15-abutting-grids.md` — detecting galleries without gutters (#38)
