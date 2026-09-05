# LOOKOUT 👀

> Local-first meeting gaze analysis — observe first, interpret later.

```text
        VIDEO
          │
          ▼
   PARTICIPANTS ───► LAYOUT
          │             │
          ▼             ▼
        RAW GAZE ───► ATTRIBUTION
                         │
                         ▼
              Alice ← Bob → Charlie
```

**Raw gaze → screen coordinates → participant attribution → interaction timeline**

## What is LOOKOUT?

LOOKOUT analyzes recorded meeting video and estimates where each participant is looking.

The architecture deliberately separates:

1. **Observation:** record gaze direction/location without assuming who is being looked at.
2. **Resolution:** reconstruct the meeting layout over time and map gaze coordinates to the participant occupying that region.

Unknown and off-screen are valid results. The system must never invent a target when evidence is insufficient.

## Status

🚧 Early-stage research / engineering project.

The first milestone is a **local, recording-first pipeline**. Live Zoom/Teams/Meet integrations come later.

## Principles

- Local-first and privacy-conscious.
- Preserve raw observations.
- Confidence and uncertainty are first-class.
- Participant positions are time-dependent.
- Components should be replaceable.
- Meaningful behavior requires tests.
- Gaze estimates must not be presented as proof of attention, intent, emotion, or psychology.

## Development

Python 3.11+. Always work inside an isolated virtual environment; a global
interpreter may carry unrelated plugins that break test collection.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest        # tests
ruff check .  # lint
mypy src      # types
```

Optional model adapters are extras and are never imported by the core:

```bash
pip install -e ".[dev,face,appearance,ocr]"
```

Research decisions live in [docs/research/](docs/research); the throwaway code
behind them lives in [experiments/](experiments).

## Roadmap

- [x] Project skeleton and data contracts
- [ ] Video/frame ingestion
- [ ] Participant detection and tracking
- [ ] Meeting-layout reconstruction
- [ ] Raw gaze estimation
- [ ] Gaze → screen-region attribution
- [ ] Screen-region → participant attribution
- [ ] Off-screen / second-screen / unknown handling
- [ ] Temporal smoothing and gaze events
- [ ] Audio speaker diarization
- [ ] Interactive timeline/report
- [ ] Performance optimization
- [ ] Live meeting input experiments

## License

LOOKOUT is source-available for **personal, educational, research, and other non-commercial use**. Commercial use requires separate permission.

See [LICENSE](LICENSE).

Repository workflow and engineering rules are in [AGENTS.md](AGENTS.md).
