"""The report is a rendering of the record, and its order is the argument.

The tests that matter here are not about formatting. They pin two promises: an
unscored run never shows an accuracy figure, and no rendered text ever describes
attention, interest, intent or emotion.
"""

import csv
import re
from pathlib import Path

import pytest

from lookout.coverage import Coverage, Degradation, ParticipantCoverage
from lookout.diagnostics import build_diagnostics
from lookout.models import GazeEvent, LayoutSource
from lookout.pipeline import AnalysisConfig
from lookout.report import (
    DISCLAIMER,
    limitations,
    render_html,
    render_markdown,
    summarize,
    verdict,
    write_csv,
    write_html,
    write_markdown,
)
from lookout.runrecord import AdapterInfo, RunRecord, build_record

SOURCE = LayoutSource.ASSUMED_SHARED

# Language that would turn an estimate into a claim about a person's mind.
FORBIDDEN = ("attention", "interest", "intent", "emotion", "engagement", "focused", "distracted")


def _events() -> list[GazeEvent]:
    return [
        GazeEvent("bob", "alice", 0.0, 2.0, 0.9, SOURCE, reason="center hit"),
        GazeEvent("bob", "carol", 2.0, 2.5, 0.6, SOURCE, reason="near border: carol"),
        GazeEvent("alice", "bob", 0.0, 1.0, 0.8, SOURCE, reason="center hit"),
    ]


def _coverage(sources: tuple[str, ...] = ("assumed_shared",)) -> Coverage:
    return Coverage(
        frames=10,
        layouts=1,
        participants_detected=3,
        face_attempts=30,
        face_hits=20,
        directions=20,
        points=19,
        off_screen=1,
        fixations=6,
        attributions=6,
        events=3,
        layout_sources=sources,
        per_participant=(
            ParticipantCoverage("alice", face_attempts=10, face_hits=10, directions=10,
                                points=10, events=1),
            ParticipantCoverage("bob", face_attempts=10, face_hits=10, directions=10,
                                points=9, off_screen=1, events=2),
            ParticipantCoverage("carol", face_attempts=10, face_hits=0, directions=0,
                                points=0, events=0, not_visible=1),
        ),
    )


def _record(evaluation: dict | None = None, sources: tuple[str, ...] = ("assumed_shared",)):
    events = _events()
    return build_record(
        AnalysisConfig(),
        dict(summarize(events)),
        coverage=_coverage(sources),
        diagnostics=build_diagnostics(events, [], participants=3),
        evaluation=evaluation,
        degradations=(
            Degradation("layout", "assumed_shared_layout", "shared layout", "not credible"),
        ),
        adapters=(AdapterInfo("gaze", "geometric"),),
        created_at="2026-01-01T00:00:00+00:00",
    )


# --------------------------------------------------------------------- promises


@pytest.mark.parametrize("render", [render_markdown, render_html])
def test_an_unscored_run_never_shows_an_accuracy_figure(render) -> None:
    """The previous report showed a duration table whether or not the run had
    ever been scored. The two must not read alike."""

    text = render(_record(), _events())
    assert "Accuracy: NOT MEASURED" in text
    # No accuracy figure anywhere: naming the metrics as unknown is fine, quoting
    # a number for them is not.
    assert re.search(r"Accuracy:\s*[\d.]+", text) is None
    assert re.search(r"(?i)hit[ _]rate\D{0,20}[\d.]+\s*%?", text) is None
    assert "%" in text  # coverage rates are still shown; only accuracy is withheld


@pytest.mark.parametrize("render", [render_markdown, render_html])
def test_no_rendered_text_interprets_the_estimate(render) -> None:
    """Enforced here rather than left to convention, because the wording rule is
    a property of every report and reviewers do not reread boilerplate."""

    text = render(_record(evaluation={"hit_rate": 0.8}), _events()).lower()
    for word in FORBIDDEN:
        assert word not in text.replace(DISCLAIMER.lower(), ""), word


def test_the_verdict_leads_with_whether_the_run_was_scored() -> None:
    unscored = verdict(_record())
    assert unscored.scored is False
    assert unscored.headline == "Accuracy: NOT MEASURED"
    assert "no claim is made" in unscored.detail


def test_a_run_that_does_not_beat_its_baseline_is_not_reported_as_good() -> None:
    beaten = verdict(_record(evaluation={"hit_rate": 0.4, "baselines": {"always_centre": 0.55}}))
    assert beaten.severity == "warning"
    assert "does not beat" in beaten.detail

    better = verdict(_record(evaluation={"hit_rate": 0.8, "baselines": {"always_centre": 0.55}}))
    assert better.severity == "scored"
    assert "beats it" in better.detail


# ------------------------------------------------------------------ limitations


def test_limitations_are_derived_from_this_run_not_written_down() -> None:
    notes = " ".join(limitations(_record()))
    assert "never scored" in notes
    assert "cross-viewer targets are not credible" in notes
    assert "carol" in notes  # examined, never resolved


def test_a_scored_run_with_known_layouts_drops_those_limitations() -> None:
    notes = " ".join(limitations(_record(evaluation={"hit_rate": 0.9}, sources=("manifest",))))
    assert "never scored" not in notes
    assert "cross-viewer" not in notes


def test_an_unresolvable_participant_is_called_out_rather_than_omitted() -> None:
    """Absence of a result is not absence of a person, and the report says which."""

    notes = " ".join(limitations(_record()))
    assert "not visible rather than omitted" in notes
    assert "could not see them" in notes


def test_the_per_participant_table_states_which_case_applies() -> None:
    text = render_markdown(_record(), _events())
    assert "| alice |" in text
    assert "observed" in text
    assert "present, not resolvable" in text


# --------------------------------------------------------------------- ordering


def test_evidence_is_rendered_before_results() -> None:
    """A reader who stops early should stop having read the caveats."""

    text = render_markdown(_record(), _events())
    assert text.index("## Coverage") < text.index("## Results")
    assert text.index("## What this run settled for") < text.index("## Results")
    assert text.index("Accuracy: NOT MEASURED") < text.index("## Results")


# ----------------------------------------------------------------------- output


def test_summarize_durations_by_target() -> None:
    summary = summarize(_events())
    assert summary["total_events"] == 3
    viewers = summary["viewers"]
    assert isinstance(viewers, dict)
    assert viewers["bob"]["duration_by_target"] == {"alice": 2.0, "carol": 0.5}


def test_html_is_self_contained_and_offline(tmp_path: Path) -> None:
    path = tmp_path / "report.html"
    write_html(path, _record(), _events())
    document = path.read_text(encoding="utf-8")
    assert "<!doctype html>" in document
    assert DISCLAIMER in document
    assert "http://" not in document and "https://" not in document
    assert "<script" not in document


def test_participant_ids_are_escaped() -> None:
    hostile = [GazeEvent("<img src=x>", "alice", 0.0, 1.0, 0.5, SOURCE, reason="center hit")]
    document = render_html(_record(), hostile)
    assert "<img src=x>" not in document
    assert "&lt;img src=x&gt;" in document


def test_markdown_carries_provenance_and_reproduction(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    write_markdown(path, _record(), _events())
    text = path.read_text(encoding="utf-8")
    assert "## Provenance" in text
    assert "## Reproduce" in text
    assert "lookout attribute" in text


def test_write_csv_has_header_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "report.csv"
    write_csv(path, _events())
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == [
        "viewer_id",
        "target",
        "start_time",
        "end_time",
        "confidence",
        "layout_source",
        "reason",
    ]
    assert len(rows) == 4  # header + 3 events
    assert rows[1][0] == "alice"  # sorted by viewer then start
    assert rows[2][6] == "center hit"


def test_a_record_with_nothing_in_it_still_renders() -> None:
    bare = RunRecord(provenance=_record().provenance, config={})
    assert "LOOKOUT run report" in render_html(bare, [])
    assert "No events were produced." in render_markdown(bare, [])


# ------------------------------------------------------------------- identity

from lookout.identity import IdentityBreak  # noqa: E402


def _broken_record(breaks: tuple[IdentityBreak, ...]):
    events = _events()
    return build_record(
        AnalysisConfig(),
        dict(summarize(events)),
        coverage=_coverage(),
        diagnostics=build_diagnostics(events, [], participants=3),
        identity_breaks=breaks,
        adapters=(AdapterInfo("gaze", "geometric"),),
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_a_run_whose_layout_never_changes_renders_as_before() -> None:
    text = render_markdown(_broken_record(()), _events())
    assert "## Identity" not in text
    assert "upper bound" not in text


@pytest.mark.parametrize("render", [render_markdown, render_html])
def test_a_total_break_is_rendered(render) -> None:
    """Correct arithmetic presented so the obvious reading is wrong would be a
    worse failure than the wrong arithmetic it replaced."""

    breaks = (
        IdentityBreak(
            at_time=150.0,
            carried=(),
            introduced=("slot_7", "slot_8"),
            ended=("slot_0", "slot_1", "slot_2"),
        ),
    )
    text = render(_broken_record(breaks), _events())
    assert "Identity" in text
    assert "150" in text
    assert "nothing" in text


def test_a_total_break_makes_the_participant_count_an_upper_bound() -> None:
    breaks = (
        IdentityBreak(at_time=150.0, carried=(), introduced=("slot_7",), ended=("slot_0",)),
    )
    notes = " ".join(limitations(_broken_record(breaks)))
    assert "upper bound" in notes
    assert "150s" in notes
    assert "appears twice" in notes


def test_a_carried_break_does_not_claim_the_count_is_unreliable() -> None:
    """A change that preserves identity is not a break in it."""

    breaks = (
        IdentityBreak(
            at_time=90.0, carried=("alice", "bob"), introduced=("carol",), ended=()
        ),
    )
    record = _broken_record(breaks)
    notes = " ".join(limitations(record))
    assert "upper bound" not in notes
    assert "## Identity" in render_markdown(record, _events())


def test_identity_breaks_round_trip_through_the_record(tmp_path: Path) -> None:
    from lookout.runrecord import read_record, write_record

    breaks = (
        IdentityBreak(at_time=150.0, carried=("a",), introduced=("b",), ended=("c",)),
    )
    path = tmp_path / "report.json"
    write_record(path, _broken_record(breaks))
    assert read_record(path).identity_breaks == breaks
