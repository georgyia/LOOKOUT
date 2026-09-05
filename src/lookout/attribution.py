from .models import GazePoint, GazeTarget, ScreenRegion


def resolve_gaze(gaze: GazePoint, regions: list[ScreenRegion]) -> str:
    """Resolve raw gaze to a participant or an explicit unknown state."""
    for region in regions:
        if region.contains(gaze.x, gaze.y, gaze.timestamp):
            return region.participant_id
    return GazeTarget.UNKNOWN.value
