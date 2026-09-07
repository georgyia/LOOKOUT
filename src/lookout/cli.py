"""Command-line interface: analyze, attribute, evaluate.

Wires real adapters to the injectable pipeline functions. Everything runs
offline; model files and weights are provided by the user.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import artifacts, report
from .evaluate import evaluate, load_truth
from .pipeline import (
    EVENTS,
    AnalysisConfig,
    analyze,
    appearance_stage,
    attribute,
    geometric_stage,
)


def _write_reports(out: Path) -> None:
    events = artifacts.read_events(out / EVENTS)
    report.write_json(out / "report.json", events)
    report.write_csv(out / "report.csv", events)
    report.write_html(out / "report.html", events)


def _cmd_analyze(args: argparse.Namespace) -> None:
    from .face import MediaPipeFaceObserver

    observer = MediaPipeFaceObserver(args.model)
    if args.gaze == "appearance":
        if not args.weights:
            raise SystemExit("--weights is required for --gaze appearance")
        from .gaze_appearance import L2CSGazeEstimator

        stage = appearance_stage(observer, L2CSGazeEstimator(args.weights))
    else:
        stage = geometric_stage(observer)

    config = AnalysisConfig(target_fps=args.fps)
    summary = analyze(args.video, args.out, stage, config)
    _write_reports(Path(args.out))
    print(json.dumps(summary, indent=2))


def _cmd_attribute(args: argparse.Namespace) -> None:
    summary = attribute(args.out)
    _write_reports(Path(args.out))
    print(json.dumps(summary, indent=2))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    events = artifacts.read_events(Path(args.out) / EVENTS)
    truth = load_truth(args.truth)
    result = evaluate(events, truth)
    report_data = {
        "total": result.total,
        "hit_rate": round(result.hit_rate, 3),
        "unknown_rate": round(result.unknown_rate, 3),
        "off_screen_recall": round(result.off_screen_recall, 3),
        "per_grid": {grid: round(result.grid_hit_rate(grid), 3) for grid in result.per_grid},
    }
    print(json.dumps(report_data, indent=2))


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
    analyze_parser.set_defaults(func=_cmd_analyze)

    attribute_parser = sub.add_parser("attribute", help="re-run attribution from a stored run")
    attribute_parser.add_argument("--out", required=True)
    attribute_parser.set_defaults(func=_cmd_attribute)

    evaluate_parser = sub.add_parser("evaluate", help="score a run against ground truth")
    evaluate_parser.add_argument("--out", required=True)
    evaluate_parser.add_argument("--truth", required=True)
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args()
    args.func(args)
