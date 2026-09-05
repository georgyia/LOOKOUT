# Data Model

LOOKOUT keeps raw observation separate from interpretation. Raw observations
are angles in the camera frame; interpretation maps them onto a viewer's screen
and then to a target.

## Angle conventions

Angles are in radians in the camera frame:

- `yaw` positive -> gaze toward the viewer's right -> larger screen `x`.
- `pitch` positive -> gaze upward -> smaller screen `y` (image `y` increases
  downward).

All coordinates are normalized to `[0, 1]`; all timestamps are non-negative
seconds from the start of the recording.

## Raw observation

### GazeDirection

- `timestamp`, `person_id`
- `yaw`, `pitch` — gaze angles
- `head_pose` — `HeadPose(yaw, pitch, roll)`
- `confidence`

The unit of the raw gaze stream. It never references a screen or layout, so it
can be stored once and reinterpreted whenever the mapping or layout changes.

## Interpretation

### GazePoint

- `timestamp`, `person_id`, `x`, `y`, `confidence`

A gaze estimate as a point on a viewer's screen. Derived from a `GazeDirection`
by the mapping step; not a raw observation.

### Region

- `kind` — `participant | shared_content | ui | unknown`
- `x`, `y`, `width`, `height`
- `participant_id` — required for `participant`, forbidden otherwise

A normalized rectangle. Carries no time of its own.

### Layout

- `viewer_id`
- `regions`
- `start_time`, `end_time`
- `source` — `recording | assumed_shared | manifest | inferred`

A viewer-relative layout valid over a time interval. `source` records how much
to trust it; see research note 01.

### Attribution

- `target` — a `participant_id` or a `GazeTarget` value
- `confidence`, `reason`, `layout_source`

### GazeEvent

- `viewer_id`, `target`, `start_time`, `end_time`, `confidence`, `layout_source`

A sustained attribution of a viewer's gaze to one target.

## Valid non-participant results

`GazeTarget`: `unknown`, `off_screen`, `not_visible`, `low_confidence`. A target
is never invented; when evidence is insufficient the result is one of these.

## Future entities

`ParticipantTrack`, `SpeakerSegment`, `Interaction`, `VisibilityState`.
