"""The run record: the canonical document describing one run.

A run's artifacts answer *what was estimated*; the record answers *whether the
estimate can be believed*. It carries the provenance of the recording and the
adapters that produced the numbers, the complete configuration behind them, and
the results themselves — in that order, because the reader needs the evidence
before the conclusion.

The record is the source of truth for reporting: ``report.json`` is this
document, and every other format is a rendering of it. Consumers should read it
rather than re-deriving summaries from the JSONL artifacts.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .coverage import Coverage, Degradation, ParticipantCoverage
from .diagnostics import Diagnostics
from .pipeline import AnalysisConfig
from .timing import RunTiming, StageTiming

__all__ = [
    "SCHEMA_VERSION",
    "DISCLAIMER",
    "VideoInfo",
    "AdapterInfo",
    "Provenance",
    "RunRecord",
    "coverage_to_dict",
    "timing_to_dict",
    "describe_config",
    "config_hash",
    "file_digest",
    "probe_video",
    "git_commit",
    "build_provenance",
    "build_record",
    "write_record",
    "read_record",
]

SCHEMA_VERSION = 1
_SCHEMA_NAME = "lookout.report"

DISCLAIMER = (
    "These are gaze-direction estimates with confidence, derived from video. "
    "They are not measurements of attention, interest, intent, or emotion."
)

_DIGEST_CHUNK = 1 << 20


@dataclass(frozen=True)
class VideoInfo:
    """What was analyzed, identified well enough to find it again.

    Every field but ``path`` is optional: probing needs the optional ``cv`` extra
    and the file itself, and a record is still worth writing when neither is
    available.
    """

    path: str
    sha256: str | None = None
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    source_fps: float | None = None


@dataclass(frozen=True)
class AdapterInfo:
    """Which model produced the observations, and which build of it.

    Two runs with identical configuration are not comparable if they ran
    different adapters, so the adapters are part of the provenance rather than
    of the configuration.
    """

    role: str
    implementation: str
    model_path: str | None = None
    model_sha256: str | None = None
    library_version: str | None = None

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("adapter role must be non-empty")
        if not self.implementation:
            raise ValueError("adapter implementation must be non-empty")


@dataclass(frozen=True)
class Provenance:
    """Where a run came from: the code, the inputs, and the invocation."""

    lookout_version: str
    created_at: str
    config_hash: str
    git_commit: str | None = None
    command: str | None = None
    video: VideoInfo | None = None
    adapters: tuple[AdapterInfo, ...] = ()


@dataclass(frozen=True)
class RunRecord:
    """One run, as a single serializable document.

    The field order is the order a reader needs: what produced the numbers, how
    much of the recording they cover, what the run had to settle for, and only
    then the numbers themselves.
    """

    provenance: Provenance
    config: dict[str, Any]
    config_overrides: tuple[str, ...] = ()
    coverage: Coverage | None = None
    diagnostics: Diagnostics | None = None
    timing: RunTiming | None = None
    evaluation: dict[str, Any] | None = None
    """Scores against ground truth, or ``None`` when the run was never scored.

    The distinction is the report's headline: an unscored run and a scored one
    must not read alike."""
    degradations: tuple[Degradation, ...] = ()
    results: dict[str, Any] = field(default_factory=dict)
    disclaimer: str = DISCLAIMER


def _plain(value: Any) -> Any:
    """Convert a value to JSON-compatible primitives, recursively."""

    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def describe_config(config: AnalysisConfig) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Serialize a configuration in full, and name the fields that were changed.

    Every nested parameter is included, not a chosen subset: a run that cannot be
    reproduced from its own record is not evidence. ``overrides`` lists the
    dotted paths that differ from the defaults, so a reader can see at a glance
    what this run did differently.
    """

    values = _plain(config)
    assert isinstance(values, dict)
    defaults = _plain(AnalysisConfig())
    assert isinstance(defaults, dict)
    return values, tuple(_diff_paths(values, defaults))


def _diff_paths(values: Any, defaults: Any, prefix: str = "") -> list[str]:
    if isinstance(values, dict) and isinstance(defaults, dict):
        changed: list[str] = []
        for key in values:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in defaults:
                changed.append(path)
            else:
                changed.extend(_diff_paths(values[key], defaults[key], path))
        return changed
    return [] if values == defaults else [prefix]


def config_hash(config: AnalysisConfig) -> str:
    """A stable digest of the full configuration, for grouping comparable runs."""

    values, _ = describe_config(config)
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_digest(path: str | Path) -> str | None:
    """SHA-256 of a file, or ``None`` if it cannot be read."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while chunk := handle.read(_DIGEST_CHUNK):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def probe_video(path: str | Path) -> VideoInfo:
    """Identify and measure the recording, degrading to just the path.

    Probing needs the optional ``cv`` extra; without it the record still names
    and hashes the file.
    """

    info = VideoInfo(path=str(path), sha256=file_digest(path))
    try:
        import cv2
    except ImportError:
        return info

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return info
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    return VideoInfo(
        path=info.path,
        sha256=info.sha256,
        duration=round(count / fps, 3) if fps > 0 and count > 0 else None,
        width=width or None,
        height=height or None,
        source_fps=round(fps, 3) if fps > 0 else None,
    )


def git_commit(cwd: str | Path | None = None) -> str | None:
    """The current commit, or ``None`` outside a repository."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def build_provenance(
    config: AnalysisConfig,
    *,
    video: str | Path | None = None,
    adapters: tuple[AdapterInfo, ...] = (),
    command: list[str] | None = None,
    created_at: str | None = None,
) -> Provenance:
    """Assemble the provenance for a run, probing what it can."""

    from . import __version__

    return Provenance(
        lookout_version=__version__,
        created_at=created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        config_hash=config_hash(config),
        git_commit=git_commit(),
        command=shlex.join(command) if command is not None else None,
        video=probe_video(video) if video is not None else None,
        adapters=adapters,
    )


def build_record(
    config: AnalysisConfig,
    results: dict[str, Any],
    *,
    coverage: Coverage | None = None,
    diagnostics: Diagnostics | None = None,
    timing: RunTiming | None = None,
    evaluation: dict[str, Any] | None = None,
    degradations: tuple[Degradation, ...] = (),
    video: str | Path | None = None,
    adapters: tuple[AdapterInfo, ...] = (),
    command: list[str] | None = None,
    created_at: str | None = None,
) -> RunRecord:
    """Assemble the record for a completed run."""

    values, overrides = describe_config(config)
    return RunRecord(
        provenance=build_provenance(
            config,
            video=video,
            adapters=adapters,
            command=command,
            created_at=created_at,
        ),
        config=values,
        config_overrides=overrides,
        coverage=coverage,
        diagnostics=diagnostics,
        timing=timing,
        evaluation=evaluation,
        degradations=degradations,
        results=results,
    )


def coverage_to_dict(coverage: Coverage) -> dict[str, Any]:
    """Serialize coverage with its derived rates spelled out.

    The rates are the numbers a reader actually reasons about, and recomputing
    them from the counts is the kind of small step that gets skipped.
    """

    payload = _plain(coverage)
    assert isinstance(payload, dict)
    payload["face_hit_rate"] = round(coverage.face_hit_rate, 4)
    payload["off_screen_rate"] = round(coverage.off_screen_rate, 4)
    payload["per_participant"] = [
        {**entry, "face_hit_rate": round(source.face_hit_rate, 4)}
        for entry, source in zip(payload["per_participant"], coverage.per_participant, strict=True)
    ]
    return payload


def _coverage_from_dict(data: dict[str, Any]) -> Coverage:
    derived = {"per_participant", "layout_sources"}
    known = {f.name for f in fields(Coverage)} - derived
    participant_fields = {f.name for f in fields(ParticipantCoverage)}
    return Coverage(
        **{key: value for key, value in data.items() if key in known},
        layout_sources=tuple(data.get("layout_sources", ())),
        per_participant=tuple(
            ParticipantCoverage(
                **{k: v for k, v in entry.items() if k in participant_fields}
            )
            for entry in data.get("per_participant", ())
        ),
    )


def timing_to_dict(timing: RunTiming) -> dict[str, Any]:
    """Serialize timing with the derived figures a reader actually uses.

    Throughput as a realtime multiple answers "how long will a 43-minute
    recording take"; a list of per-stage seconds does not.
    """

    total = sum(stage.seconds for stage in timing.stages)
    return {
        "wall_seconds": timing.wall_seconds,
        "video_seconds": timing.video_seconds,
        "realtime_factor": round(timing.realtime_factor, 4),
        "stages": [
            {
                "stage": stage.stage,
                "seconds": stage.seconds,
                "calls": stage.calls,
                "share": round(stage.share_of(total), 4),
            }
            for stage in timing.stages
        ],
    }


def _timing_from_dict(data: dict[str, Any]) -> RunTiming:
    return RunTiming(
        stages=tuple(
            StageTiming(
                stage=str(row["stage"]),
                seconds=float(row["seconds"]),
                calls=int(row["calls"]),
            )
            for row in data.get("stages", ())
        ),
        wall_seconds=float(data.get("wall_seconds", 0.0)),
        video_seconds=float(data.get("video_seconds", 0.0)),
    )


def _diagnostics_from_dict(data: dict[str, Any]) -> Diagnostics:
    known = {f.name for f in fields(Diagnostics)}
    payload = {key: value for key, value in data.items() if key in known}
    for key in ("confidence_histogram",):
        if key in payload:
            payload[key] = tuple(payload[key])
    if "screen_histogram" in payload:
        payload["screen_histogram"] = tuple(tuple(row) for row in payload["screen_histogram"])
    return Diagnostics(**payload)


def to_dict(record: RunRecord) -> dict[str, Any]:
    return {
        "schema": _SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "disclaimer": record.disclaimer,
        "provenance": asdict(record.provenance),
        "config": record.config,
        "config_overrides": list(record.config_overrides),
        "coverage": coverage_to_dict(record.coverage) if record.coverage else None,
        "diagnostics": _plain(record.diagnostics) if record.diagnostics else None,
        "timing": timing_to_dict(record.timing) if record.timing else None,
        "evaluation": record.evaluation,
        "degradations": [asdict(d) for d in record.degradations],
        "results": record.results,
    }


def from_dict(data: dict[str, Any]) -> RunRecord:
    if data.get("schema") != _SCHEMA_NAME:
        raise ValueError("not a lookout run record")
    if data.get("version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported run record version: {data.get('version')}")

    raw = dict(data["provenance"])
    video = raw.pop("video", None)
    adapters = raw.pop("adapters", []) or []
    provenance = Provenance(
        **raw,
        video=VideoInfo(**video) if video else None,
        adapters=tuple(AdapterInfo(**a) for a in adapters),
    )
    coverage = data.get("coverage")
    diagnostics = data.get("diagnostics")
    timing = data.get("timing")
    return RunRecord(
        provenance=provenance,
        config=data.get("config", {}),
        config_overrides=tuple(data.get("config_overrides", ())),
        coverage=_coverage_from_dict(coverage) if coverage else None,
        diagnostics=_diagnostics_from_dict(diagnostics) if diagnostics else None,
        timing=_timing_from_dict(timing) if timing else None,
        evaluation=data.get("evaluation"),
        degradations=tuple(Degradation(**d) for d in data.get("degradations", ())),
        results=data.get("results", {}),
        disclaimer=str(data.get("disclaimer", DISCLAIMER)),
    )


def write_record(path: str | Path, record: RunRecord) -> None:
    payload = json.dumps(to_dict(record), indent=2, sort_keys=False)
    Path(path).write_text(payload + "\n", encoding="utf-8")


def read_record(path: str | Path) -> RunRecord:
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
