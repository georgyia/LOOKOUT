# Architecture

LOOKOUT separates observation from interpretation.

```text
                    MEETING VIDEO
                          │
                          ▼
                   Frame ingestion
                          │
                          ▼
                Participant tracking
                          │
                          ▼
                Layout reconstruction
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        Raw gaze estimation       Audio later
              │
              ▼
          Raw gaze stream
              │
              ▼
       Gaze → screen region
              │
              ▼
       Region → participant
              │
              ▼
       Interaction events
```

## Key decision

Raw gaze must remain independent from participant attribution.

A raw observation is:

`person_id + timestamp + normalized screen coordinate + confidence`

Attribution is a later operation using the layout valid at that timestamp.

This supports reprocessing, changing layouts, uncertainty, and alternative attribution algorithms.
