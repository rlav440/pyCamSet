"""Lint learner-facing education strings against the A1 copy contract.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pytest

# The real-copy tests below reuse the Phase 2 and Phase 3 run fixtures of their visual tests.
from test_education_phase2_visual import synthetic_run  # noqa: F401  (fixture)
from test_education_phase3_visual import pinhole_run  # noqa: F401  (fixture)

REPOSITORY = Path(__file__).resolve().parents[1]
BAN_LIST = REPOSITORY / "tests" / "test_data" / "education" / "banned_phrases.json"
A1_COPY_SHA256 = "5b1b134ff5eb68ad107ae9a6956f87ac1b18b6f8852b7b083f5a6df947dff077"


def lint_copy(text: str, filename: str, key: str, rules: Iterable[dict[str, Any]],
              evidence: Iterable[str] = ()) -> list[str]:
    """Return diagnostics with file/key, matched text and nearby source context."""
    evidence_set = set(evidence)
    findings = []
    for rule in rules:
        required = rule.get("requires_evidence")
        if required and required in evidence_set:
            continue
        match = re.search(rule["regex"], text, flags=re.IGNORECASE)
        if match:
            start, end = max(0, match.start() - 36), min(len(text), match.end() + 36)
            context = text[start:end].replace("\n", " ")
            findings.append(
                f"{filename}:{key}: {rule['key']} matched {match.group(0)!r}; context={context!r}")
    return findings


def _rules() -> list[dict[str, Any]]:
    document = json.loads(BAN_LIST.read_text(encoding="utf-8"))
    assert document["source_artifact"] == "A1-copy.md"
    assert document["source_sha256"] == A1_COPY_SHA256
    assert document["version"] == 1
    return document["patterns"]


def test_ban_list_is_pinned_to_reviewed_a1_copy_and_well_formed():
    rules = _rules()
    assert len(rules) >= 18
    assert len({rule["key"] for rule in rules}) == len(rules)
    for rule in rules:
        re.compile(rule["regex"])
        assert rule["reason"]


def test_copy_lint_reports_path_key_match_and_nearby_context():
    rules = _rules()
    text = "The target moved during playback, which is not a physical measurement."
    findings = lint_copy(text, "gui/education_phase3.py", "phase3.playback.caption", rules)
    assert len(findings) == 1
    assert "gui/education_phase3.py:phase3.playback.caption" in findings[0]
    assert "target moved" in findings[0]
    assert "nearby context" not in findings[0]
    assert "context=" in findings[0]


def test_copy_lint_is_case_insensitive_and_keeps_evidence_conditional():
    rules = _rules()
    assert lint_copy("The camera shifted between estimates.", "phase3.py", "caption", rules)
    # This phrase is only allowed when the caller can show the detector-list check.
    text = "The photo list no longer matches the photos in the folder."
    assert lint_copy(text, "phase4.py", "source.reason", rules)
    assert lint_copy(text, "phase4.py", "source.reason", rules,
                     evidence=("detector_list_comparison",)) == []


@pytest.mark.parametrize("copy", [
    "The solver objective is not measured in pixels.",
    "These are accepted optimisation steps, not physical camera movement.",
    "Phase 4 gave large errors less influence; the objectives use different losses.",
])
def test_safe_example_copy_passes(copy):
    assert lint_copy(copy, "fixture.py", "safe", _rules()) == []


# Learner copy must not leak plan-card codes (C1, N2, ...), authoring-level
# labels ("L3-only") or the trace producer's name. D-codes such as D3.5 are
# diagnostic names, allowed from L2, so a card code must not be followed by
# ".<digit>". Case-sensitive: k1/k2 distortion names stay allowed. "L" is not
# a card letter, so naming a detail level ("shown at L3") stays allowed.
INTERNAL_CODES = re.compile(r"\b[A-KMN]\d\b(?!\.\d)|\bL[123]-only\b|\bwriter(?:'s)?\b")
# A1 section 1: diagnostic codes, "seed", "detector-frame", parameter
# identifiers and gauge language are L2/L3 only. "camset" stays allowed: it is
# the GUI's own file name and A1's L1 refusal copy uses it.
L1_JARGON = re.compile(r"\bD\d\.\d|\bseed\b|fixed_pose|detector-frame|\bgauge\b", re.IGNORECASE)


def learner_copy_findings(text: str, level: int) -> list[str]:
    """Return internal codes at any level, and L2/L3-only jargon at L1."""
    found = [match.group(0) for match in INTERNAL_CODES.finditer(text)]
    if level == 1:
        found += [match.group(0) for match in L1_JARGON.finditer(text)]
    return found


@pytest.mark.parametrize("text", [
    "C1 recorded RMS (px)", "N2: the comparison is withheld", "This is an L3-only copy.",
    "paired with the writer's recorded photo name",
])
def test_internal_plan_codes_are_rejected_in_learner_copy(text):
    assert learner_copy_findings(text, 3)


@pytest.mark.parametrize("text", [
    "The solver starting point is D3.5, 1.2 px.", "k1 = 0.1, k2 = 0.01",
    "Phase 3 used the 2nd target pose (image index 1).",
])
def test_diagnostic_names_are_allowed_from_l2(text):
    assert learner_copy_findings(text, 2) == []


@pytest.mark.parametrize("text", [
    "D3.5 is 1.2 px.", "Initial seed", "Detector-frame x (px)", "fixed_pose", "gauge frame",
])
def test_l2_jargon_is_rejected_at_l1(text):
    assert learner_copy_findings(text, 1)


def test_quoted_control_labels_match_the_gui_text():
    """A control quoted in education copy (‘…’) must be the label the GUI shows.

    The quote must equal a string literal in a non-education GUI module, or be
    that literal without a trailing parenthetical note, so a learner who
    searches the window for the quoted words finds the whole control name.
    """
    gui = REPOSITORY / "pyCamSet" / "gui"
    literals = set()
    for path in gui.glob("*.py"):
        if path.name.startswith("education_"):
            continue
        source = path.read_text(encoding="utf-8")
        literals.update(re.findall(r'"([^"\n]+)"', source))
        literals.update(re.findall(r"'([^'\n]+)'", source))
    labels = {re.sub(r"\s+\([^)]*\)$", "", literal) for literal in literals} | literals
    quoted = {}
    for path in gui.glob("education_*.py"):
        for label in re.findall(r"‘([^’]+)’", path.read_text(encoding="utf-8")):
            quoted.setdefault(label, path.name)
    assert quoted
    missing = {label: name for label, name in quoted.items() if label not in labels}
    assert not missing, missing


# --- The real copy: every stage of Phases 2, 3 and 4, read from the live views -----------
#
# The fixture tests above lint sample strings. These read what a learner sees on each stage
# at each level: the caption, every stat chip (value and label) and the Numbers drawer
# (headers and rows). The banned-phrase list and the internal-code rule apply at every
# level; the L1 jargon rule (gauge, seed, D-codes, ...) applies at L1.
# The phase tests build their own views, so these tests reuse their fixtures and helpers.


def learner_units(stage_view) -> list[tuple[str, str]]:
    """Each piece of learner-visible text on one stage, as (kind, text)."""
    table = stage_view.numbers.table
    units = [("caption", stage_view.caption.text())]
    units += [(f"chip {chip.label!r}", f"{chip.value} {chip.label}")
              for chip in stage_view.chips.chips()]
    headers = [table.horizontalHeaderItem(col).text() for col in range(table.columnCount())
               if table.horizontalHeaderItem(col) is not None]
    units.append(("numbers header", " | ".join(headers)))
    for row in range(table.rowCount()):
        cells = [table.item(row, col).text() for col in range(table.columnCount())
                 if table.item(row, col) is not None]
        units.append((f"numbers row {row}", " | ".join(cells)))
    return [(kind, text) for kind, text in units if text.strip()]


def real_copy_findings(stage_view, where: str, level: int) -> list[str]:
    """Lint the copy one stage shows at one level. ``where`` names the file and stage."""
    rules = _rules()
    findings = []
    for kind, text in learner_units(stage_view):
        findings += lint_copy(text, where, f"L{level} {kind}", rules)
        findings += [f"{where}:L{level} {kind}: internal or L1-only term {hit!r}; text={text!r}"
                     for hit in learner_copy_findings(text, level)]
    return findings


def _qt(monkeypatch):
    """A QApplication with the static 3D fallback, as the phase view tests use."""
    pytest.importorskip("PySide6")
    monkeypatch.setenv("PYCAMSET_EDU_3D", "static")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.mark.gui
def test_real_phase2_copy_passes_the_ban_list_and_plain_language(synthetic_run, monkeypatch):
    """Phase 2 stages at L1-L3, from the synthetic telecentric run of its visual tests."""
    _qt(monkeypatch)
    from pyCamSet.gui.education_phase2 import STAGES, Phase2View

    view = Phase2View(synthetic_run.context)
    findings = []
    try:
        for level in (1, 2, 3):
            view.stage_view.set_level(level)
            for stage in range(len(STAGES)):
                view.stage_view.set_current_stage(stage)
                where = f"gui/education_phase2.py stage {stage} {STAGES[stage]!r}"
                findings += real_copy_findings(view.stage_view, where, level)
    finally:
        view.close()
    assert findings == [], "\n".join(findings)


@pytest.mark.gui
def test_real_phase3_copy_passes_the_ban_list_and_plain_language(pinhole_run, monkeypatch):
    """Phase 3 stages at L1-L3, from the recorded pinhole run of its visual tests."""
    _qt(monkeypatch)
    import hashlib

    from pyCamSet.gui import education_phase3
    from test_education_phase3 import _available_derivation, _context

    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _c: _available_derivation())
    context = _context(pinhole_run.run_dir.parent.parent, {
        "params": {"record_education": True},
        "artifacts": {"phase2_initial_camset_used": str(pinhole_run.phase2),
                      "initial_state": str(pinhole_run.initial),
                      "initial_state_sha256": hashlib.sha256(
                          pinhole_run.initial.read_bytes()).hexdigest()},
    })
    context.artifact = lambda _phase=None: pinhole_run.phase3
    view = education_phase3.Phase3View(context)
    findings = []
    try:
        for level in (1, 2, 3):
            view.stage_view.set_level(level)
            for key in view.stage_keys():
                view.show_stage(key)
                where = f"gui/education_phase3.py stage {key!r}"
                findings += real_copy_findings(view.stage_view, where, level)
    finally:
        view.close()
    assert findings == [], "\n".join(findings)


@pytest.mark.gui
def test_real_phase4_copy_passes_the_ban_list_and_plain_language(tmp_path, monkeypatch):
    """Phase 4 stages at L1-L3, from the synthetic four-photo run of its visual tests."""
    _qt(monkeypatch)
    from pyCamSet.workflow.education_visual_p4 import STAGES

    from test_education_phase4_visual import _view

    view, _visual = _view(tmp_path, monkeypatch)
    findings = []
    try:
        for level in (1, 2, 3):
            view.set_level(level)
            for stage in range(len(STAGES)):
                view.set_stage(stage)
                where = f"gui/education_phase4.py stage {stage} {STAGES[stage]!r}"
                findings += real_copy_findings(view.stage_view, where, level)
    finally:
        view.close()
    assert findings == [], "\n".join(findings)
