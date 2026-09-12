"""Pipeline orchestration.

Composes the stages by function composition with injected adapters; there is no
pipeline class and no hidden state. ``analyze`` runs observation (video ->
layouts -> raw gaze) and then attribution; ``attribute`` re-runs only the
mapping/attribution/event stages from the raw store, so a run can be
re-interpreted without any model.

Both return the :class:`~lookout.coverage.Coverage` they achieved rather than a
loose summary dict; assembling that into the run record that documents the run
is :mod:`lookout.runrecord`'s job.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from . import artifacts, store
from .attribution import AttributionParams, attribute_point
from .calibration import calibrate_mapping, weak_labels_from_speaker
from .coverage import Coverage, ParticipantCoverage
from .events import aggregate_events
from .face import FaceObserver
from .frames import Frame, Image, read_video
from .gaze_appearance import GazeEstimator, appearance_gaze
from .gaze_geometric import GeometricGazeParams, estimate_gaze
from .identity import IdentityBreak, build_layouts, identity_breaks
from .layout import DetectionParams, Tile, detect_tiles, segment_layouts
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
from .speaker import SpeakerSegment, aggregate_speaker_segments, detect_highlighted_tile
from .temporal import detect_fixations, median_smooth
from .timing import RunTiming, Stopwatch

__all__ = [
    "AnalysisConfig",
    "CalibrationReport",
    "RunOutcome",
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
SPEAKER = "speaker.jsonl"


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
    min_calibration_labels: int = 20
    """Weak labels a viewer needs before its mapping is fit rather than assumed.

    Too few and the fit tracks the noise in a handful of glances; the prior,
    wrong as it is, is at least wrong consistently."""


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


@dataclass(frozen=True)
class RunOutcome:
    """Everything a run produced about itself: what survived, what was fit, and
    what it cost."""

    coverage: Coverage
    calibrations: tuple[CalibrationReport, ...] = ()
    timing: RunTiming = field(default_factory=RunTiming)
    identity_breaks: tuple[IdentityBreak, ...] = ()


@dataclass(frozen=True)
class CalibrationReport:
    """Whether a viewer's mapping was fit or assumed, and why.

    A calibrated run and an assumed one must not read alike, so this is carried
    into the record rather than logged.
    """

    viewer_id: str
    calibrated: bool
    labels: int
    reason: str


def _calibrated_mapping(
    viewer: str,
    directions: list[GazeDirection],
    segments: list[SpeakerSegment],
    layout: Layout | None,
    config: AnalysisConfig,
) -> tuple[ScreenMappingParams, CalibrationReport]:
    """Fit this viewer's angle-to-screen mapping, or keep the prior.

    Falls back on every failure rather than producing a confident wrong mapping:
    a degenerate fit is worse than a documented guess, because it looks fitted.
    """

    if layout is None:
        return config.mapping, CalibrationReport(viewer, False, 0, "no layout")
    if not segments:
        return config.mapping, CalibrationReport(viewer, False, 0, "no speaker segments")

    labels = weak_labels_from_speaker(directions, segments, layout)
    if len(labels) < config.min_calibration_labels:
        return config.mapping, CalibrationReport(
            viewer, False, len(labels), "too few weak labels"
        )
    try:
        fitted = calibrate_mapping(labels)
    except ValueError as error:
        return config.mapping, CalibrationReport(viewer, False, len(labels), str(error))
    return fitted, CalibrationReport(viewer, True, len(labels), "fit from speaker cues")


def _crop_region(image: Image, region: Region) -> Image:
    height, width = image.shape[:2]
    x0 = int(round(region.x * width))
    x1 = int(round((region.x + region.width) * width))
    y0 = int(round(region.y * height))
    y1 = int(round((region.y + region.height) * height))
    return image[y0:y1, x0:x1]


def _timed(frames: Iterator[Frame], watch: Stopwatch, stage: str) -> Iterator[Frame]:
    """Yield frames one at a time, charging decode time to ``stage``.

    Wrapping the iterator rather than draining it is the whole point: the caller
    never holds more than the frame it is working on.
    """

    while True:
        with watch.stage(stage):
            frame = next(frames, None)
        if frame is None:
            return
        yield frame


def _unobservable_windows(
    layouts: list[Layout],
    events: list[GazeEvent],
    horizon: float,
    source: LayoutSource | None = None,
) -> list[GazeEvent]:
    """Windows where a participant was on screen but nothing was resolved.

    Absence of a result reads as absence of a person. These say the difference
    out loud: the participant had a tile, and the pipeline could not see them.
    """

    produced = [(e.viewer_id, e.start_time, e.end_time) for e in events]
    unseen: list[GazeEvent] = []

    for layout in layouts:
        start = layout.start_time
        end = layout.end_time if layout.end_time is not None else horizon
        if end <= start:
            continue
        for region in layout.participant_regions():
            participant = region.participant_id
            assert participant is not None
            covered = any(
                viewer == participant and event_end > start and event_start < end
                for viewer, event_start, event_end in produced
            )
            if covered:
                continue
            unseen.append(
                GazeEvent(
                    viewer_id=participant,
                    target=GazeTarget.NOT_VISIBLE.value,
                    start_time=start,
                    end_time=end,
                    confidence=1.0,
                    layout_source=source or layout.source,
                    reason="no usable observation",
                )
            )
    return unseen


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
    viewer_layouts: list[Layout] | None = None,
) -> RunOutcome:
    """Run observation and attribution end to end, writing artifacts to ``out_dir``."""

    config = config or AnalysisConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    watch = Stopwatch()

    # Two passes over the video, holding one frame at a time. Materializing the
    # sampled frames instead would make peak memory scale with recording length:
    # at 5 fps a 43-minute 854x480 call is about 16 GB of held frames, and the
    # failure when that ceiling arrives is an allocation error rather than a
    # degradation. Decoding is around a tenth of the cost of detection, so the
    # second pass is the cheaper side of that trade by a wide margin.
    frame_count = 0
    video_seconds = 0.0
    per_frame_tiles: list[tuple[float, tuple[Tile, ...]]] = []
    for frame in _timed(read_video(video, config.target_fps), watch, "decode"):
        frame_count += 1
        video_seconds = max(video_seconds, frame.timestamp)
        with watch.stage("detect_tiles"):
            per_frame_tiles.append((frame.timestamp, detect_tiles(frame.image, config.detection)))

    with watch.stage("segment_layouts"):
        intervals = segment_layouts(per_frame_tiles, min_stable_seconds=2.0 / config.target_fps)
        recording_layouts = build_layouts(
            intervals,
            viewer_id=RECORDING_VIEWER,
            source=LayoutSource.RECORDING,
        )

    directions: list[GazeDirection] = []
    # (attempts, hits) per participant: a tile the observer never resolved a
    # face in yields no directions at all, so misses have to be counted here or
    # they leave no trace anywhere downstream.
    faces: dict[str, tuple[int, int]] = {}
    participants: set[str] = set()
    # Speaker flags, collected here because only observation has the frames.
    speaking: list[tuple[float, float, str, float]] = []
    frame_step = 1.0 / config.target_fps

    for recorded in recording_layouts:
        for region in recorded.participant_regions():
            assert region.participant_id is not None
            participants.add(region.participant_id)

    tiles_at = dict(per_frame_tiles)
    for frame in _timed(read_video(video, config.target_fps), watch, "decode"):
        layout = _active(recording_layouts, frame.timestamp)
        if layout is None:
            continue

        regions = layout.participant_regions()
        with watch.stage("speaker_cue"):
            highlighted = detect_highlighted_tile(frame.image, tiles_at.get(frame.timestamp, ()))
        if highlighted is not None and highlighted < len(regions):
            speaker_id = regions[highlighted].participant_id
            if speaker_id is not None:
                speaking.append(
                    (frame.timestamp, frame.timestamp + frame_step, speaker_id, 1.0)
                )

        for region in regions:
            assert region.participant_id is not None
            crop = _crop_region(frame.image, region)
            with watch.stage("observe_gaze"):
                observation = gaze_stage(crop, region.participant_id, frame.timestamp)
            attempts, hits = faces.get(region.participant_id, (0, 0))
            faces[region.participant_id] = (attempts + 1, hits + (observation is not None))
            if observation is not None:
                directions.append(observation)

    store.write_gaze(out / GAZE_RAW, directions)
    artifacts.write_layouts(out / LAYOUT, recording_layouts)
    artifacts.write_speaker_segments(
        out / SPEAKER,
        aggregate_speaker_segments(speaking, cue="highlight", max_gap=2.0 * frame_step),
    )

    with watch.stage("attribute"):
        attributed = attribute(out, config, viewer_layouts)

    return RunOutcome(
        coverage=attributed.coverage.with_observation(
            frames=frame_count,
            layouts=len(recording_layouts),
            layout_changes=max(0, len(recording_layouts) - 1),
            participants_detected=len(participants),
            faces_per_participant=faces,
        ),
        calibrations=attributed.calibrations,
        timing=watch.result(video_seconds=video_seconds),
        identity_breaks=identity_breaks(recording_layouts),
    )


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


def attribute(
    out_dir: str | Path,
    config: AnalysisConfig | None = None,
    viewer_layouts: list[Layout] | None = None,
    calibrate: bool = False,
) -> RunOutcome:
    """Re-run mapping, attribution, and events from the stored raw gaze.

    ``viewer_layouts`` optionally supplies per-viewer layouts (e.g. from a
    manifest, ``source=manifest``); without it, each viewer's layout is the
    recording layout applied as ``assumed_shared``.

    ``calibrate`` fits each viewer's angle-to-screen mapping from stored speaker
    segments instead of using the shared prior. The prior encodes one laptop's
    geometry; a viewer on a different screen has a systematically different
    relationship, which is the error the pipeline tolerates least.
    """

    config = config or AnalysisConfig()
    out = Path(out_dir)

    directions = store.read_gaze(out / GAZE_RAW)
    segments = (
        artifacts.read_speaker_segments(out / SPEAKER)
        if calibrate and (out / SPEAKER).exists()
        else []
    )

    layouts_by_viewer: dict[str, list[Layout]] = {}
    if viewer_layouts is not None:
        for layout in viewer_layouts:
            layouts_by_viewer.setdefault(layout.viewer_id, []).append(layout)
        recording_layouts = None
    else:
        recording_layouts = artifacts.read_layouts(out / LAYOUT)

    def layouts_for(viewer: str) -> list[Layout]:
        if recording_layouts is not None:
            return _assumed_layouts(viewer, recording_layouts)
        return layouts_by_viewer.get(viewer, [])

    by_person: dict[str, list[GazeDirection]] = {}
    for direction in directions:
        by_person.setdefault(direction.person_id, []).append(direction)

    all_points: list[GazePoint] = []
    all_rows: list[tuple[str, float, float, Attribution]] = []
    all_events: list[GazeEvent] = []
    per_participant: list[ParticipantCoverage] = []
    sources: set[str] = set()
    calibrations: list[CalibrationReport] = []

    for viewer, person_directions in sorted(by_person.items()):
        active_layouts = layouts_for(viewer)
        sources.update(layout.source.value for layout in active_layouts)

        mapping = config.mapping
        if calibrate:
            mapping, calibration = _calibrated_mapping(
                viewer,
                person_directions,
                segments,
                active_layouts[0] if active_layouts else None,
                config,
            )
            calibrations.append(calibration)

        points: list[GazePoint] = []
        off_screen = 0
        for direction in sorted(person_directions, key=lambda d: d.timestamp):
            result = map_direction(direction, mapping)
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
            current_layout = _active(active_layouts, fixation.start_time)
            if current_layout is None:
                continue
            centroid = GazePoint(
                fixation.start_time, viewer, fixation.x, fixation.y, fixation.confidence
            )
            attribution = attribute_point(centroid, current_layout, config.attribution)
            timed.append((fixation.start_time, fixation.end_time, attribution))
            all_rows.append((viewer, fixation.start_time, fixation.end_time, attribution))

        events = aggregate_events(viewer, timed, config.event_min_duration, config.max_gap)
        all_events.extend(events)

        per_participant.append(
            ParticipantCoverage(
                participant_id=viewer,
                directions=len(person_directions),
                points=len(points),
                off_screen=off_screen,
                fixations=len(fixations),
                attributions=len(timed),
                events=len(events),
            )
        )

    # A participant who was on screen but never resolved has, until now, simply
    # been missing from the output — indistinguishable from one who was not in
    # the call at all. Say so instead.
    unseen = _unobservable_windows(
        recording_layouts if recording_layouts is not None else (viewer_layouts or []),
        all_events,
        horizon=max((d.timestamp for d in directions), default=0.0),
        # A recording layout is applied to each viewer as assumed_shared, so the
        # result must carry that provenance rather than the layout's own.
        source=LayoutSource.ASSUMED_SHARED if recording_layouts is not None else None,
    )
    all_events.extend(unseen)
    unseen_by_viewer: dict[str, int] = {}
    for event in unseen:
        unseen_by_viewer[event.viewer_id] = unseen_by_viewer.get(event.viewer_id, 0) + 1

    seen = {entry.participant_id for entry in per_participant}
    per_participant = [
        ParticipantCoverage(
            participant_id=entry.participant_id,
            directions=entry.directions,
            points=entry.points,
            off_screen=entry.off_screen,
            fixations=entry.fixations,
            attributions=entry.attributions,
            events=entry.events,
            not_visible=unseen_by_viewer.get(entry.participant_id, 0),
        )
        for entry in per_participant
    ] + [
        ParticipantCoverage(participant_id=viewer, not_visible=count)
        for viewer, count in sorted(unseen_by_viewer.items())
        if viewer not in seen
    ]

    artifacts.write_points(out / GAZE_SCREEN, all_points)
    artifacts.write_attributions(out / ATTRIBUTION, all_rows)
    artifacts.write_events(out / EVENTS, all_events)

    coverage = Coverage(
        directions=len(directions),
        points=len(all_points),
        off_screen=sum(entry.off_screen for entry in per_participant),
        fixations=sum(entry.fixations for entry in per_participant),
        attributions=len(all_rows),
        events=len(all_events),
        not_visible=len(unseen),
        layout_sources=tuple(sorted(sources)),
        per_participant=tuple(per_participant),
    )
    return RunOutcome(coverage=coverage, calibrations=tuple(calibrations))
