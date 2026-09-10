"""The signal each stage computes must survive to where it can be read.

Every stage below thresholds on a quality measure and then, historically, threw
it away — leaving a single confidence float that says something is wrong but
never what. These pin the retention, not the maths.
"""

import json
import math
from pathlib import Path

from lookout import artifacts, store
from lookout.attribution import AttributionParams, attribute_point
from lookout.events import aggregate_events
from lookout.gaze_geometric import estimate_gaze
from lookout.models import (
    Attribution,
    GazeDirection,
    GazePoint,
    GazeQuality,
    HeadPose,
    Layout,
    LayoutSource,
    Region,
    RegionKind,
)
from lookout.temporal import detect_fixations
from tests.conftest import make_observation

SOURCE = LayoutSource.ASSUMED_SHARED


def _layout() -> Layout:
    return Layout(
        "bob",
        (
            Region(RegionKind.PARTICIPANT, 0.0, 0.0, 0.5, 1.0, "alice"),
            Region(RegionKind.PARTICIPANT, 0.5, 0.0, 0.5, 1.0, "carol"),
        ),
        0.0,
        SOURCE,
    )


def test_confidence_factors_are_kept_apart() -> None:
    """A blink and a turned head yield the same confidence for different reasons."""

    blinking = estimate_gaze(
        make_observation(half_h=0.005), timestamp=0.0, person_id="bob"
    )
    turned = estimate_gaze(
        make_observation(head_pose=HeadPose(math.radians(75.0), 0.0, 0.0)),
        timestamp=0.0,
        person_id="bob",
    )

    assert blinking.quality is not None and turned.quality is not None
    assert blinking.quality.limiting == "openness"
    assert turned.quality.limiting == "head"
    assert blinking.quality.openness < turned.quality.openness
    assert turned.quality.head < blinking.quality.head


def test_confidence_is_still_the_product_of_its_factors() -> None:
    direction = estimate_gaze(make_observation(h_shift=0.3), timestamp=0.0, person_id="bob")
    assert direction.quality is not None
    assert direction.confidence == direction.quality.combined


def test_detection_confidence_is_a_factor_not_an_assumption() -> None:
    weak = estimate_gaze(
        make_observation(detection_confidence=0.25), timestamp=0.0, person_id="bob"
    )
    assert weak.quality is not None
    assert weak.quality.detection == 0.25
    assert weak.quality.limiting == "detection"


def test_attribution_keeps_the_margin_it_scaled_by() -> None:
    """A central hit from a poor observation and a border hit from a good one
    produce similar confidences and are otherwise indistinguishable."""

    central = attribute_point(GazePoint(0.0, "bob", 0.25, 0.5, 0.4), _layout())
    border = attribute_point(GazePoint(0.0, "bob", 0.02, 0.5, 1.0), _layout())

    assert central.margin_ratio == 1.0
    assert border.margin_ratio is not None and border.margin_ratio < 0.2
    assert central.target == "alice"


def test_fixation_records_how_tight_it_was() -> None:
    steady = [GazePoint(i * 0.1, "bob", 0.5, 0.5, 0.9) for i in range(8)]
    loose = [
        GazePoint(i * 0.1, "bob", 0.5 + 0.02 * (i % 3), 0.5 + 0.02 * (i % 2), 0.9)
        for i in range(8)
    ]
    (tight_fixation,) = detect_fixations(steady, 0.12, 0.15, 0.3)
    (loose_fixation,) = detect_fixations(loose, 0.12, 0.15, 0.3)

    assert tight_fixation.dispersion == 0.0
    assert loose_fixation.dispersion > tight_fixation.dispersion


def test_the_reason_survives_aggregation() -> None:
    """The reason used to die at the event boundary, so a report could see that
    an event was uncertain but never why."""

    spans: list[tuple[float, float, Attribution]] = [
        (0.0, 1.0, Attribution("alice", 0.8, "center hit", SOURCE, margin_ratio=0.9)),
        (1.0, 1.2, Attribution("alice", 0.2, "near border: alice", SOURCE, margin_ratio=0.1)),
    ]
    (event,) = aggregate_events("bob", spans, min_duration=0.2, max_gap=0.3)
    assert event.reason == "center hit"  # the longest span wins


def test_the_dominant_reason_wins_not_the_first() -> None:
    spans: list[tuple[float, float, Attribution]] = [
        (0.0, 0.3, Attribution("alice", 0.8, "center hit", SOURCE)),
        (0.3, 2.0, Attribution("alice", 0.2, "near border: alice", SOURCE)),
    ]
    (event,) = aggregate_events("bob", spans, min_duration=0.2, max_gap=0.3)
    assert event.reason == "near border: alice"


def test_events_round_trip_their_reason(tmp_path: Path) -> None:
    spans: list[tuple[float, float, Attribution]] = [
        (0.0, 1.0, Attribution("alice", 0.8, "center hit", SOURCE)),
    ]
    events = aggregate_events("bob", spans, min_duration=0.2, max_gap=0.3)
    path = tmp_path / "events.jsonl"
    artifacts.write_events(path, events)
    assert artifacts.read_events(path) == events


def test_attributions_serialize_their_margin(tmp_path: Path) -> None:
    path = tmp_path / "attribution.jsonl"
    attribution = attribute_point(GazePoint(0.0, "bob", 0.25, 0.5, 0.9), _layout())
    artifacts.write_attributions(path, [("bob", 0.0, 1.0, attribution)])
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["margin_ratio"] == attribution.margin_ratio


def test_quality_round_trips_through_the_store(tmp_path: Path) -> None:
    path = tmp_path / "gaze_raw.jsonl"
    quality = GazeQuality(detection=0.9, size=0.8, openness=0.7, head=0.6)
    direction = GazeDirection(0.0, "bob", 0.1, -0.1, HeadPose(0, 0, 0), quality.combined, quality)
    store.write_gaze(path, [direction])
    assert store.read_gaze(path) == [direction]


def test_version_one_stores_stay_readable(tmp_path: Path) -> None:
    """The raw store is the re-run source of truth; adding a diagnostic field
    must not invalidate stores already on disk."""

    path = tmp_path / "gaze_raw.jsonl"
    path.write_text(
        json.dumps({"schema": "lookout.gaze", "version": 1})
        + "\n"
        + json.dumps(
            {
                "timestamp": 0.0,
                "person_id": "bob",
                "yaw": 0.1,
                "pitch": -0.1,
                "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                "confidence": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (restored,) = store.read_gaze(path)
    assert restored.confidence == 0.5
    assert restored.quality is None


def test_attribution_params_still_gate_on_confidence() -> None:
    """Retaining the margin must not change what counts as low confidence."""

    border = attribute_point(
        GazePoint(0.0, "bob", 0.005, 0.5, 1.0), _layout(), AttributionParams(0.15)
    )
    assert border.target == "low_confidence"
    assert border.margin_ratio is not None
