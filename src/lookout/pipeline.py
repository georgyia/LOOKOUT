"""Pipeline orchestration.

Composes the stages by function composition with injected adapters; there is no
pipeline class and no hidden state. ``analyze`` runs observation (video ->
layouts -> raw gaze) and then attribution; ``attribute`` re-runs only the
mapping/attribution/event stages from the raw store, so a run can be
re-interpreted without any model.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import artifacts, store
from .attribution import AttributionParams, attribute_point
from .events import aggregate_events
from .face import FaceObserver
from .frames import Image, read_video
from .gaze_appearance import GazeEstimator, appearance_gaze
from .gaze_geometric import GeometricGazeParams, estimate_gaze
from .identity import build_layouts
from .layout import DetectionParams, detect_tiles, segment_layouts
from .models import (
    Attribution,
    GazeDirection,
    GazeEvent,
    GazePoint,
    GazeTarget,
    Layout,
    LayoutSource,
    Region,
)
from .screen_mapping import ScreenMappingParams, map_direction
from .temporal import detect_fixations, median_smooth

__all__ = [
    "AnalysisConfig",
    "GazeStage",
    "geometric_stage",
    "appearance_stage",
    "analyze",
    "attribute",
    "RECORDING_VIEWER",
]

# A gaze stage turns a tile crop into a raw observation, or None if no face.
GazeStage = Callable[[Image, str, float], GazeDirection | None]

RECORDING_VIEWER = "recording"

GAZE_RAW = "gaze_raw.jsonl"
LAYOUT = "layout.jsonl"
GAZE_SCREEN = "gaze_screen.jsonl"
ATTRIBUTION = "attribution.jsonl"
EVENTS = "events.jsonl"
RUN = "run.json"


@dataclass(frozen=True)
class AnalysisConfig:
    """All tunable parameters for a run, in one place."""

    target_fps: float = 5.0
    detection: DetectionParams = field(default_factory=DetectionParams)
    geometric: GeometricGazeParams = field(default_factory=GeometricGazeParams)
    mapping: ScreenMappingParams = field(default_factory=ScreenMappingParams)
    attribution: AttributionParams = field(default_factory=AttributionParams)
    smoothing_window: int = 3
    dispersion_threshold: float = 0.12
    min_fixation: float = 0.15
    max_gap: float = 0.3
    event_min_duration: float = 0.2


def geometric_stage(
    observer: FaceObserver,
    params: GeometricGazeParams | None = None,
) -> GazeStage:
    """Gaze stage using landmarks and the geometric baseline."""

    def stage(crop: Image, person_id: str, timestamp: float) -> GazeDirection | None:
        observation = observer.observe(crop)
        if observation is None:
            return None
        return estimate_gaze(observation, timestamp=timestamp, person_id=person_id, params=params)

    return stage


def appearance_stage(
    observer: FaceObserver,
    estimator: GazeEstimator,
    confidence: float = 0.8,
) -> GazeStage:
    """Gaze stage using the appearance model; head pose comes from the observer."""

    def stage(crop: Image, person_id: str, timestamp: float) -> GazeDirection | None:
        observation = observer.observe(crop)
        if observation is None:
            return None
        return appearance_gaze(
            estimator,
            crop,
            timestamp=timestamp,
            person_id=person_id,
            head_pose=observation.head_pose,
            confidence=confidence,
        )

    return stage


def _crop_region(image: Image, region: Region) -> Image:
    height, width = image.shape[:2]
    x0 = int(round(region.x * width))
    x1 = int(round((region.x + region.width) * width))
    y0 = int(round(region.y * height))
    y1 = int(round((region.y + region.height) * height))
    return image[y0:y1, x0:x1]


def _active(layouts: list[Layout], timestamp: float) -> Layout | None:
    for layout in layouts:
        if layout.active_at(timestamp):
            return layout
    return None


def analyze(
    video: str | Path,
    out_dir: str | Path,
    gaze_stage: GazeStage,
    config: AnalysisConfig | None = None,
) -> dict[str, object]:
    """Run observation and attribution end to end, writing artifacts to ``out_dir``."""

    config = config or AnalysisConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = list(read_video(video, config.target_fps))
    per_frame_tiles = [
        (frame.timestamp, detect_tiles(frame.image, config.detection)) for frame in frames
    ]

    intervals = segment_layouts(per_frame_tiles, min_stable_seconds=2.0 / config.target_fps)
    recording_layouts = build_layouts(
        intervals,
        viewer_id=RECORDING_VIEWER,
        source=LayoutSource.RECORDING,
    )

    directions: list[GazeDirection] = []
    for frame in frames:
        layout = _active(recording_layouts, frame.timestamp)
        if layout is None:
            continue
        for region in layout.participant_regions():
            assert region.participant_id is not None
            crop = _crop_region(frame.image, region)
            observation = gaze_stage(crop, region.participant_id, frame.timestamp)
            if observation is not None:
                directions.append(observation)

    store.write_gaze(out / GAZE_RAW, directions)
    artifacts.write_layouts(out / LAYOUT, recording_layouts)

    summary = attribute(out, config)
    summary.update(
        {
            "frames": len(frames),
            "layouts": len(recording_layouts),
            "directions": len(directions),
        }
    )
    _write_run_metadata(out, config, summary)
    return summary


def _assumed_layouts(viewer: str, recording_layouts: list[Layout]) -> list[Layout]:
    return [
        Layout(
            viewer,
            layout.regions,
            layout.start_time,
            LayoutSource.ASSUMED_SHARED,
            layout.end_time,
        )
        for layout in recording_layouts
    ]


def attribute(out_dir: str | Path, config: AnalysisConfig | None = None) -> dict[str, object]:
    """Re-run mapping, attribution, and events from the stored raw gaze."""

    config = config or AnalysisConfig()
    out = Path(out_dir)

    directions = store.read_gaze(out / GAZE_RAW)
    recording_layouts = artifacts.read_layouts(out / LAYOUT)

    by_person: dict[str, list[GazeDirection]] = {}
    for direction in directions:
        by_person.setdefault(direction.person_id, []).append(direction)

    all_points: list[GazePoint] = []
    all_rows: list[tuple[str, float, float, Attribution]] = []
    all_events: list[GazeEvent] = []
    off_screen = 0

    for viewer, person_directions in sorted(by_person.items()):
        viewer_layouts = _assumed_layouts(viewer, recording_layouts)
        points: list[GazePoint] = []
        for direction in sorted(person_directions, key=lambda d: d.timestamp):
            result = map_direction(direction, config.mapping)
            if isinstance(result, GazeTarget):
                off_screen += 1
                continue
            points.append(result)
        all_points.extend(points)

        smoothed = median_smooth(points, config.smoothing_window)
        fixations = detect_fixations(
            smoothed,
            config.dispersion_threshold,
            config.min_fixation,
            config.max_gap,
        )

        timed: list[tuple[float, float, Attribution]] = []
        for fixation in fixations:
            layout = _active(viewer_layouts, fixation.start_time)
            if layout is None:
                continue
            centroid = GazePoint(
                fixation.start_time, viewer, fixation.x, fixation.y, fixation.confidence
            )
            attribution = attribute_point(centroid, layout, config.attribution)
            timed.append((fixation.start_time, fixation.end_time, attribution))
            all_rows.append((viewer, fixation.start_time, fixation.end_time, attribution))

        all_events.extend(
            aggregate_events(viewer, timed, config.event_min_duration, config.max_gap)
        )

    artifacts.write_points(out / GAZE_SCREEN, all_points)
    artifacts.write_attributions(out / ATTRIBUTION, all_rows)
    artifacts.write_events(out / EVENTS, all_events)

    return {
        "viewers": len(by_person),
        "points": len(all_points),
        "off_screen": off_screen,
        "events": len(all_events),
    }


def _write_run_metadata(
    out: Path,
    config: AnalysisConfig,
    summary: dict[str, object],
) -> None:
    from . import __version__

    metadata = {
        "schema": "lookout.run",
        "version": 1,
        "lookout_version": __version__,
        "config": {
            "target_fps": config.target_fps,
            "dispersion_threshold": config.dispersion_threshold,
            "min_fixation": config.min_fixation,
            "max_gap": config.max_gap,
            "event_min_duration": config.event_min_duration,
        },
        "summary": summary,
    }
    (out / RUN).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
