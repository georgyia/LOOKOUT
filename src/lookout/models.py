from dataclasses import dataclass
from enum import Enum


class GazeTarget(str, Enum):
    UNKNOWN = "unknown"
    OFF_SCREEN = "off_screen"
    NOT_VISIBLE = "not_visible"


@dataclass(frozen=True)
class GazePoint:
    timestamp: float
    person_id: str
    x: float
    y: float
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.x <= 1.0:
            raise ValueError("x must be normalized to [0, 1]")
        if not 0.0 <= self.y <= 1.0:
            raise ValueError("y must be normalized to [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")


@dataclass(frozen=True)
class ScreenRegion:
    participant_id: str
    x: float
    y: float
    width: float
    height: float
    start_time: float
    end_time: float | None = None

    def contains(self, x: float, y: float, timestamp: float) -> bool:
        if timestamp < self.start_time:
            return False
        if self.end_time is not None and timestamp >= self.end_time:
            return False
        return (
            self.x <= x <= self.x + self.width
            and self.y <= y <= self.y + self.height
        )
