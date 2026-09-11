"""Command-line interface: analyze, attribute, evaluate.

Wires real adapters to the injectable pipeline functions, and records which
adapters ran so a report can be judged. Everything runs offline; model files and
weights are provided by the user.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from . import artifacts, report, runrecord, store
from .coverage import Degradation, detect_degradations
from .diagnostics import build_diagnostics, distribution_warnings
from .evaluate import EvaluationResult, evaluate, load_truth
from .manifest import load_manifest
from .pipeline import (
    EVENTS,
    GAZE_RAW,
    GAZE_SCREEN,
    AnalysisConfig,
    CalibrationReport,
    RunOutcome,
    analyze,
    appearance_stage,
    attribute,
    geometric_stage,
)
from .runrecord import AdapterInfo, RunRecord

REPORT = "report.json"


def _library_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _adapter(role: str, implementation: str, model: str | None, library: str) -> AdapterInfo:
    return AdapterInfo(
        role=role,
        implementation=implementation,
        model_path=model,
        model_sha256=runrecord.file_digest(model) if model else None,
        library_version=_library_version(library),
    )


def _write_reports(
    out: Path,
    config: AnalysisConfig,
    outcome: RunOutcome,
    *,
    video: str | None = None,
    adapters: tuple[AdapterInfo, ...] = (),
    provenance_from: RunRecord | None = None,
) -> RunRecord:
    """Write the run record and the rendered event tables.

    ``provenance_from`` carries the recording and adapter identity of an earlier
    run forward, so re-attributing a stored run does not lose the provenance of
    observations no longer being recomputed.
    """

    coverage = outcome.coverage
    calibrations = outcome.calibrations
    events = artifacts.read_events(out / EVENTS)
    results: dict[str, Any] = dict(report.summarize(events))

    points = artifacts.read_points(out / GAZE_SCREEN) if (out / GAZE_SCREEN).exists() else []
    directions = store.read_gaze(out / GAZE_RAW) if (out / GAZE_RAW).exists() else []
    diagnostics = build_diagnostics(
        events, points, directions, participants=len(coverage.per_participant)
    )
    warnings = distribution_warnings(diagnostics, coverage.layout_sources)
    warnings += _calibration_warnings(calibrations)
    results["calibration"] = [
        {
            "viewer_id": c.viewer_id,
            "calibrated": c.calibrated,
            "labels": c.labels,
            "reason": c.reason,
        }
        for c in calibrations
    ]

    if provenance_from is not None:
        video = video or (
            provenance_from.provenance.video.path if provenance_from.provenance.video else None
        )
        adapters = adapters or provenance_from.provenance.adapters

    record = runrecord.build_record(
        config,
        results,
        coverage=coverage,
        diagnostics=diagnostics,
        timing=outcome.timing,
        degradations=detect_degradations(coverage) + warnings,
        video=video,
        adapters=adapters,
        command=sys.argv,
    )
    runrecord.write_record(out / REPORT, record)
    report.write_csv(out / "report.csv", events)
    report.write_html(out / "report.html", record, events)
    report.write_markdown(out / "report.md", record, events)
    return record


def _calibration_warnings(
    calibrations: tuple[CalibrationReport, ...],
) -> tuple[Degradation, ...]:
    """A calibrated run and an assumed one must not read alike."""

    if not calibrations:
        return ()
    assumed = [c for c in calibrations if not c.calibrated]
    if not assumed:
        return ()
    reasons = ", ".join(sorted({c.reason for c in assumed}))
    return (
        Degradation(
            stage="mapping",
            code="uncalibrated_viewers",
            detail=(
                f"{len(assumed)} of {len(calibrations)} viewers kept the shared "
                f"angle-to-screen prior ({reasons})."
            ),
            impact=(
                "The prior encodes one laptop's geometry. A viewer on a different "
                "screen has a systematically different angle-to-screen relationship, "
                "which biases every target for that viewer in the same direction."
            ),
        ),
    )


def _read_existing(out: Path) -> RunRecord | None:
    path = out / REPORT
    if not path.exists():
        return None
    try:
        return runrecord.read_record(path)
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def _print_summary(record: RunRecord) -> None:
    """Print the funnel and what the run had to settle for, not just the totals."""

    print(
        json.dumps(
            {
                "coverage": runrecord.coverage_to_dict(record.coverage)
                if record.coverage
                else None,
                "degradations": [
                    {"code": d.code, "detail": d.detail} for d in record.degradations
                ],
            },
            indent=2,
        )
    )


def _cmd_analyze(args: argparse.Namespace) -> None:
    from .face import MediaPipeFaceObserver

    observer = MediaPipeFaceObserver(args.model)
    adapters = [_adapter("face", "MediaPipeFaceObserver", args.model, "mediapipe")]

    if args.gaze == "appearance":
        if not args.weights:
            raise SystemExit("--weights is required for --gaze appearance")
        from .gaze_appearance import L2CSGazeEstimator

        stage = appearance_stage(observer, L2CSGazeEstimator(args.weights))
        adapters.append(_adapter("gaze", "L2CSGazeEstimator", args.weights, "torch"))
    else:
        stage = geometric_stage(observer)
        adapters.append(_adapter("gaze", "geometric", None, "numpy"))

    config = AnalysisConfig(target_fps=args.fps)
    layouts = load_manifest(args.manifest) if args.manifest else None
    outcome = analyze(args.video, args.out, stage, config, layouts)
    record = _write_reports(
        Path(args.out),
        config,
        outcome,
        video=args.video,
        adapters=tuple(adapters),
    )
    _print_summary(record)


def _cmd_attribute(args: argparse.Namespace) -> None:
    out = Path(args.out)
    existing = _read_existing(out)
    config = AnalysisConfig()
    layouts = load_manifest(args.manifest) if args.manifest else None
    outcome = attribute(out, config, layouts, calibrate=args.calibrate)
    record = _write_reports(out, config, outcome, provenance_from=existing)
    _print_summary(record)


def _cmd_report(args: argparse.Namespace) -> None:
    """Re-render a stored run's report without re-running any stage.

    The record already holds everything the report shows, so regenerating it
    needs neither the video nor a model — only the run directory.
    """

    out = Path(args.out)
    existing = _read_existing(out)
    if existing is None:
        raise SystemExit(f"no run record found in {out}; run 'lookout analyze' first")

    events = artifacts.read_events(out / EVENTS)
    record = existing
    if args.truth:
        truth = load_truth(args.truth)
        result = evaluate(events, truth, sample_step=args.sample_step)
        record = replace(existing, evaluation=_evaluation_summary(result, args.truth))

    runrecord.write_record(out / REPORT, record)
    report.write_csv(out / "report.csv", events)
    report.write_html(out / "report.html", record, events)
    report.write_markdown(out / "report.md", record, events)
    print(report.verdict(record).headline)


def _evaluation_summary(result: EvaluationResult, truth_path: str) -> dict[str, Any]:
    """The scores, always alongside what a trivial strategy would have scored."""

    return {
        "truth_path": truth_path,
        "truth_sha256": runrecord.file_digest(truth_path),
        "total": result.total,
        "hit_rate": round(result.hit_rate, 4),
        "hit_rate_ci95": list(result.hit_rate_ci),
        "baselines": {name: round(value, 4) for name, value in result.baselines.items()},
        "beats_baseline": result.beats_baseline,
        "unknown_rate": round(result.unknown_rate, 4),
        "unknown_breakdown": {
            "silent": result.silent,
            "unknown": result.explicit_unknown,
            "low_confidence": result.low_confidence,
            "not_visible": result.not_visible,
        },
        "wrong": result.wrong,
        "off_screen_recall": round(result.off_screen_recall, 4),
        "per_grid": {grid: round(result.grid_hit_rate(grid), 4) for grid in result.per_grid},
        "expected_calibration_error": result.expected_calibration_error,
        "reliability": [
            {
                "lower": b.lower,
                "upper": b.upper,
                "count": b.count,
                "mean_confidence": b.mean_confidence,
                "hit_rate": b.hit_rate,
            }
            for b in result.reliability
        ],
        "confusion": result.confusion,
    }


def _cmd_evaluate(args: argparse.Namespace) -> None:
    events = artifacts.read_events(Path(args.out) / EVENTS)
    truth = load_truth(args.truth)
    result = evaluate(events, truth, sample_step=args.sample_step)
    print(json.dumps(_evaluation_summary(result, args.truth), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lookout", description="Local-first meeting gaze analysis."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze_parser = sub.add_parser("analyze", help="analyze a recording end to end")
    analyze_parser.add_argument("video")
    analyze_parser.add_argument("--out", required=True)
    analyze_parser.add_argument(
        "--model", required=True, help="MediaPipe face_landmarker.task path"
    )
    analyze_parser.add_argument("--gaze", choices=["geometric", "appearance"], default="geometric")
    analyze_parser.add_argument("--weights", help="gaze model weights (for --gaze appearance)")
    analyze_parser.add_argument("--fps", type=float, default=5.0)
    analyze_parser.add_argument(
        "--manifest", help="per-viewer layouts, replacing the shared-layout assumption"
    )
    analyze_parser.set_defaults(func=_cmd_analyze)

    attribute_parser = sub.add_parser("attribute", help="re-run attribution from a stored run")
    attribute_parser.add_argument("--out", required=True)
    attribute_parser.add_argument(
        "--manifest", help="per-viewer layouts, replacing the shared-layout assumption"
    )
    attribute_parser.add_argument(
        "--calibrate",
        action="store_true",
        help="fit each viewer's angle-to-screen mapping from stored speaker segments",
    )
    attribute_parser.set_defaults(func=_cmd_attribute)

    report_parser = sub.add_parser("report", help="re-render the report for a stored run")
    report_parser.add_argument("--out", required=True)
    report_parser.add_argument("--truth", help="ground truth to score the run against")
    report_parser.add_argument(
        "--sample-step",
        type=float,
        help="score truth intervals every N seconds instead of at their midpoint",
    )
    report_parser.set_defaults(func=_cmd_report)

    evaluate_parser = sub.add_parser("evaluate", help="score a run against ground truth")
    evaluate_parser.add_argument("--out", required=True)
    evaluate_parser.add_argument("--truth", required=True)
    evaluate_parser.add_argument(
        "--sample-step",
        type=float,
        help="score truth intervals every N seconds instead of at their midpoint",
    )
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args()
    args.func(args)
