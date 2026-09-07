# LOOKOUT — Repository Instructions

These rules apply to every change.

## Mission

LOOKOUT is a local-first system for analyzing meeting recordings to estimate:

- who is present and where they appear in the meeting layout;
- each participant's estimated gaze direction/location over time;
- whether a gaze location can be attributed to another participant;
- uncertainty, unknown targets, and off-screen gaze;
- optional speaking/audio context.

**Observe first. Interpret later.**

## Architecture

Prefer this pipeline:

VIDEO
→ frame ingestion
→ participant detection/tracking
→ meeting-layout reconstruction
→ raw gaze estimation
→ raw gaze storage
→ temporal normalization
→ gaze-to-region attribution
→ region-to-participant attribution
→ interaction events
→ analytics/reporting

Do not collapse these layers without a strong reason.

## Data philosophy

Never discard raw observations just because attribution is uncertain.

Valid states include:

- unknown
- off_screen
- not_visible
- low_confidence

Never manufacture a participant target.

All observations need timestamps. Layout and participant positions are time-varying.

## Privacy

- Prefer local/offline processing.
- No telemetry without explicit approval.
- Do not upload meeting content by default.
- Do not retain raw video unnecessarily.
- Do not infer psychological states, intent, agreement, interest, or emotion from gaze.
- Describe outputs as estimates with confidence.
- Avoid covert-monitoring functionality.

## Scope

Start with recorded video.

Do not add Zoom, Teams, Meet, browser automation, or live-capture integrations until a dedicated issue exists and the direction is approved.

Avoid premature model dependencies. Establish interfaces and testable data contracts first.

## Git / issue workflow

Every non-trivial change should have an issue.

Before coding:

1. Check existing issues.
2. Reuse an existing issue when appropriate.
3. Otherwise create a focused issue.
4. Apply appropriate labels.
5. Create a focused branch.

Branch naming:

- `feature/<short-name>`
- `fix/<short-name>`
- `docs/<short-name>`
- `refactor/<short-name>`
- `test/<short-name>`

Do not work directly on the default branch for normal changes.

## Pull requests

Open a PR from the working branch.

A PR should:

- explain what changed;
- explain why;
- reference the issue;
- include tests for behavior changes;
- update docs when architecture/behavior changes;
- remain focused.

Before merge:

```bash
pytest
```

Run configured lint/type checks when relevant.

Merge only after required checks pass and the change is reviewable.

## Tests

Meaningful logic requires tests.

Prioritize:

- data validation;
- coordinate normalization;
- temporal intervals;
- participant identity tracking;
- layout changes;
- gaze attribution;
- confidence thresholds;
- unknown/off-screen behavior;
- event aggregation.

Model-heavy code should have deterministic unit-test interfaces and fixtures.

## Code style

Prefer:

- typed Python;
- dataclasses or clear schemas;
- pure transformation functions;
- dependency injection for models;
- deterministic tests;
- small modules.

Avoid giant pipeline classes, hidden global state, unnecessary frameworks, and premature optimization.

## Documentation

Keep README concise. Put technical detail in `docs/`.

Architectural concepts should be documented when introduced.

## Commits

Use focused, descriptive commits:

- `feat: add gaze observation schema`
- `test: cover layout interval resolution`
- `fix: handle unknown gaze targets`
- `docs: explain attribution pipeline`

Do not mix unrelated cleanup into feature work.

## Repository presentation

Project history and documentation should describe the technical work and its purpose.

Do not mention AI agents, model names, or agent contributions in commits, PRs, issues, documentation, or release notes.

## Definition of done

For meaningful changes:

- issue exists and has labels;
- focused branch is used;
- implementation is tested;
- docs are updated when needed;
- checks pass;
- PR is opened;
- PR references the issue;
- review concerns are resolved;
- branch is merged only after required checks pass.

## Conservative default

If LOOKOUT cannot confidently determine who was being looked at, preserve:

`gaze → coordinates → unknown`

rather than guessing.

This is a core design principle.
