import pytest

from lookout.events import aggregate_events
from lookout.models import Attribution, LayoutSource

SOURCE = LayoutSource.ASSUMED_SHARED


def _attr(target: str, confidence: float = 0.9) -> Attribution:
    return Attribution(target, confidence, "test", SOURCE)


def test_consecutive_same_target_merges_into_one_event() -> None:
    items = [
        (0.0, 0.3, _attr("alice", 0.8)),
        (0.3, 0.6, _attr("alice", 0.6)),
    ]
    events = aggregate_events("bob", items)
    assert len(events) == 1
    event = events[0]
    assert event.target == "alice"
    assert event.start_time == pytest.approx(0.0)
    assert event.end_time == pytest.approx(0.6)
    assert event.confidence == pytest.approx(0.7)  # duration-weighted mean


def test_target_change_splits_events() -> None:
    items = [
        (0.0, 0.3, _attr("alice")),
        (0.3, 0.6, _attr("carol")),
    ]
    events = aggregate_events("bob", items)
    assert [e.target for e in events] == ["alice", "carol"]


def test_short_run_is_dropped() -> None:
    items = [
        (0.0, 0.3, _attr("alice")),
        (0.3, 0.35, _attr("carol")),  # 0.05 s glance
        (0.35, 0.7, _attr("alice")),
    ]
    events = aggregate_events("bob", items, min_duration=0.2)
    assert [e.target for e in events] == ["alice", "alice"]


def test_large_gap_splits_same_target() -> None:
    items = [
        (0.0, 0.3, _attr("alice")),
        (1.5, 1.8, _attr("alice")),
    ]
    events = aggregate_events("bob", items, max_gap=0.3)
    assert len(events) == 2
    assert all(e.target == "alice" for e in events)
