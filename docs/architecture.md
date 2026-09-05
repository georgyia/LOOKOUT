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

## Key decisions

**Raw gaze is an angle, not a screen point.** A raw observation is

`person_id + timestamp + (yaw, pitch) + head_pose + confidence`

Mapping that angle to a screen point is a separate, uncertain step. Storing the
angle lets us re-run mapping and attribution without re-running any model.

**Layouts are viewer-relative.** Each participant sees their own arrangement of
tiles; the recording only shows one. A `Layout` therefore names a `viewer_id`
and a `source` recording how it was obtained (`recording`, `assumed_shared`,
`manifest`, `inferred`). Attribution uses the layout valid at that timestamp
and propagates the source into every result.

Together these support reprocessing, changing layouts, uncertainty, and
alternative attribution algorithms. See research note 01 for the rationale.
