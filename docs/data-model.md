# Data Model

## GazePoint

- `timestamp`
- `person_id`
- `x`
- `y`
- `confidence`

`x` and `y` are normalized to `[0, 1]`.

## ScreenRegion

- `participant_id`
- `x`
- `y`
- `width`
- `height`
- `start_time`
- `end_time`

A region is valid only during its time interval.

## Future entities

Expected concepts:

- ParticipantTrack
- HeadPose
- EyeGaze
- MeetingLayout
- GazeEvent
- SpeakerSegment
- Interaction
- Screen
- VisibilityState
