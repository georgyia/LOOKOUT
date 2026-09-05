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
