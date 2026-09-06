import csv
from pathlib import Path

from lookout.models import GazeEvent, LayoutSource
from lookout.report import DISCLAIMER, summarize, write_csv, write_html

SOURCE = LayoutSource.ASSUMED_SHARED


def _events() -> list[GazeEvent]:
    return [
        GazeEvent("bob", "alice", 0.0, 2.0, 0.9, SOURCE),
        GazeEvent("bob", "carol", 2.0, 2.5, 0.6, SOURCE),
        GazeEvent("alice", "bob", 0.0, 1.0, 0.8, SOURCE),
    ]


def test_summarize_durations_by_target() -> None:
    summary = summarize(_events())
    assert summary["total_events"] == 3
    viewers = summary["viewers"]
    assert isinstance(viewers, dict)
    assert viewers["bob"]["duration_by_target"] == {"alice": 2.0, "carol": 0.5}


def test_write_csv_has_header_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "report.csv"
    write_csv(path, _events())
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    header = ["viewer_id", "target", "start_time", "end_time", "confidence", "layout_source"]
    assert rows[0] == header
    assert len(rows) == 4  # header + 3 events
    assert rows[1][0] == "alice"  # sorted by viewer then start


def test_write_html_is_static_and_carries_disclaimer(tmp_path: Path) -> None:
    path = tmp_path / "report.html"
    write_html(path, _events())
    document = path.read_text(encoding="utf-8")
    assert "<!doctype html>" in document
    assert DISCLAIMER in document
    assert "alice" in document and "carol" in document
