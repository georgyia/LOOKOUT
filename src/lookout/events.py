"""Interaction events.

Aggregates a viewer's time-ordered attributions into :class:`GazeEvent`s: runs
of the same target are merged (across gaps up to ``max_gap``), confidence is the
duration-weighted mean, and runs shorter than ``min_duration`` are dropped so a
momentary glance does not become an event.
"""

from __future__ import annotations

from .models import Attribution, GazeEvent

__all__ = ["aggregate_events"]

_EPS = 1e-9

# One attributed span: (start_time, end_time, attribution).
TimedAttribution = tuple[float, float, Attribution]


class _OpenEvent:
    def __init__(self, start: float, end: float, attribution: Attribution) -> None:
        self.target = attribution.target
        self.layout_source = attribution.layout_source
        self.start = start
        self.end = end
        self._weight = 0.0
        self._weighted_confidence = 0.0
        self.add(start, end, attribution)

    def add(self, start: float, end: float, attribution: Attribution) -> None:
        weight = max(end - start, _EPS)
        self.end = max(self.end, end)
        self._weight += weight
        self._weighted_confidence += weight * attribution.confidence

    @property
    def confidence(self) -> float:
        return self._weighted_confidence / self._weight

    def to_event(self, viewer_id: str) -> GazeEvent:
        return GazeEvent(
            viewer_id=viewer_id,
            target=self.target,
            start_time=self.start,
            end_time=self.end,
            confidence=self.confidence,
            layout_source=self.layout_source,
        )


def aggregate_events(
    viewer_id: str,
    items: list[TimedAttribution],
    min_duration: float = 0.2,
    max_gap: float = 0.3,
) -> list[GazeEvent]:
    """Merge same-target attributions into events for one viewer."""

    ordered = sorted(items, key=lambda item: item[0])
    events: list[GazeEvent] = []
    current: _OpenEvent | None = None

    for start, end, attribution in ordered:
        if (
            current is not None
            and attribution.target == current.target
            and start - current.end <= max_gap
        ):
            current.add(start, end, attribution)
            continue
        if current is not None and current.end - current.start >= min_duration:
            events.append(current.to_event(viewer_id))
        current = _OpenEvent(start, end, attribution)

    if current is not None and current.end - current.start >= min_duration:
        events.append(current.to_event(viewer_id))
    return events
