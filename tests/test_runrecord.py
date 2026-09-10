import json
from pathlib import Path

import pytest

from lookout.coverage import Coverage, Degradation, ParticipantCoverage
from lookout.pipeline import AnalysisConfig
from lookout.runrecord import (
    DISCLAIMER,
    SCHEMA_VERSION,
    AdapterInfo,
    Provenance,
    RunRecord,
    build_record,
    config_hash,
    describe_config,
    file_digest,
    read_record,
    write_record,
)

FIXED_TIME = "2026-01-01T00:00:00+00:00"


def _record(config: AnalysisConfig | None = None) -> RunRecord:
    config = config or AnalysisConfig()
    return build_record(
        config,
        {"viewers": {}, "total_events": 0},
        adapters=(
            AdapterInfo("face", "FakeObserver", model_path=None, library_version="0.1"),
            AdapterInfo("gaze", "geometric"),
        ),
        command=["lookout", "analyze", "clip.mp4", "--out", "run"],
        created_at=FIXED_TIME,
    )


def _leaf_paths(value: object, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        paths: set[str] = set()
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths |= _leaf_paths(item, path)
        return paths
    return {prefix}


def test_record_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_record(path, _record())
    restored = read_record(path)

    assert restored.provenance.created_at == FIXED_TIME
    assert restored.provenance.command == "lookout analyze clip.mp4 --out run"
    assert [a.implementation for a in restored.provenance.adapters] == [
        "FakeObserver",
        "geometric",
    ]
    assert restored.disclaimer == DISCLAIMER
    assert restored.config == _record().config


def test_reading_rejects_a_foreign_document(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"schema": "something.else", "version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a lookout run record"):
        read_record(path)


def test_reading_rejects_an_unsupported_version(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    payload = json.loads(json.dumps({"schema": "lookout.report", "version": SCHEMA_VERSION + 1}))
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported run record version"):
        read_record(path)


def test_every_configuration_field_is_recorded() -> None:
    """A run that cannot be reproduced from its own record is not evidence.

    The previous run metadata echoed five scalars and silently dropped every
    nested parameter block, so this pins the whole tree rather than a sample.
    """

    values, _ = describe_config(AnalysisConfig())
    recorded = _leaf_paths(values)

    expected = {
        "target_fps",
        "smoothing_window",
        "dispersion_threshold",
        "min_fixation",
        "max_gap",
        "event_min_duration",
        "detection.background_diff",
        "detection.min_area_fraction",
        "detection.fill_ratio",
        "detection.aspect_tolerance",
        "detection.shared_area_fraction",
        "geometric.max_eye_yaw",
        "geometric.max_eye_pitch",
        "geometric.horizontal_sign",
        "geometric.vertical_sign",
        "geometric.reference_iod",
        "geometric.min_openness",
        "geometric.full_openness",
        "mapping.yaw_at_left",
        "mapping.yaw_at_right",
        "mapping.pitch_at_top",
        "mapping.pitch_at_bottom",
        "mapping.off_screen_margin",
        "attribution.confidence_floor",
    }
    assert expected <= recorded
    assert "aspect_ratios" in values["detection"]


def test_overrides_name_only_what_changed() -> None:
    _, defaults = describe_config(AnalysisConfig())
    assert defaults == ()

    _, overrides = describe_config(AnalysisConfig(target_fps=2.0))
    assert overrides == ("target_fps",)

    from lookout.attribution import AttributionParams

    _, nested = describe_config(AnalysisConfig(attribution=AttributionParams(0.4)))
    assert nested == ("attribution.confidence_floor",)


def test_config_hash_tracks_nested_parameters() -> None:
    from lookout.screen_mapping import ScreenMappingParams

    base = AnalysisConfig()
    assert config_hash(base) == config_hash(AnalysisConfig())
    changed = AnalysisConfig(mapping=ScreenMappingParams(off_screen_margin=0.2))
    assert config_hash(changed) != config_hash(base)


def test_record_is_deterministic_for_a_fixed_clock(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    write_record(first, _record())
    write_record(second, _record())
    assert first.read_bytes() == second.read_bytes()


def test_file_digest_is_none_for_a_missing_file(tmp_path: Path) -> None:
    assert file_digest(tmp_path / "absent.mp4") is None
    present = tmp_path / "present.bin"
    present.write_bytes(b"lookout")
    assert file_digest(present) is not None


def test_provenance_survives_a_record_without_video_or_adapters(tmp_path: Path) -> None:
    record = RunRecord(
        provenance=Provenance(
            lookout_version="0.1.0", created_at=FIXED_TIME, config_hash="abc"
        ),
        config={},
    )
    path = tmp_path / "report.json"
    write_record(path, record)
    restored = read_record(path)
    assert restored.provenance.video is None
    assert restored.provenance.adapters == ()


def test_coverage_and_degradations_round_trip(tmp_path: Path) -> None:
    coverage = Coverage(
        frames=10,
        layouts=1,
        participants_detected=2,
        face_attempts=20,
        face_hits=15,
        directions=15,
        points=12,
        off_screen=3,
        fixations=6,
        attributions=6,
        events=4,
        layout_sources=("assumed_shared",),
        per_participant=(
            ParticipantCoverage("slot_0", face_attempts=10, face_hits=10, directions=10,
                                points=10, events=3),
            ParticipantCoverage("slot_1", face_attempts=10, face_hits=5, directions=5,
                                points=2, off_screen=3, events=1),
        ),
    )
    record = build_record(
        AnalysisConfig(),
        {"total_events": 4},
        coverage=coverage,
        degradations=(Degradation("layout", "assumed_shared_layout", "detail", "impact"),),
        created_at=FIXED_TIME,
    )
    path = tmp_path / "report.json"
    write_record(path, record)
    restored = read_record(path)

    assert restored.coverage == coverage
    assert restored.degradations[0].code == "assumed_shared_layout"


def test_serialized_coverage_spells_out_its_rates(tmp_path: Path) -> None:
    """The rates are what a reader reasons about; recomputing them gets skipped."""

    coverage = Coverage(
        face_attempts=20,
        face_hits=15,
        directions=15,
        points=12,
        off_screen=3,
        per_participant=(
            ParticipantCoverage("slot_0", face_attempts=20, face_hits=15, directions=15,
                                points=12, off_screen=3),
        ),
    )
    path = tmp_path / "report.json"
    write_record(path, build_record(AnalysisConfig(), {}, coverage=coverage, created_at=FIXED_TIME))
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["coverage"]["face_hit_rate"] == 0.75
    assert payload["coverage"]["off_screen_rate"] == 0.2
    assert payload["coverage"]["per_participant"][0]["face_hit_rate"] == 0.75
