'''Share the education detail-level control, phase names and orientation copy.

A1 section 1 makes the levels additive: L1 is plain language and one
decision, L2 adds diagnostic names, quantities, units and sources, and L3 adds
method, assumptions and limitations. A frame therefore carries its plain
``copy`` for every level, an optional ``detail`` added from L2, and an
optional ``expert`` part added at L3; :func:`compose` joins them.
'''
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

LEVEL_LABELS = ("L1 — overview", "L2 — details", "L3 — full evidence")

#: Human phase names, as A1 section 3 titles each storyboard.
PHASE_NAMES = {
    "phase1": "Phase 1 — Find target",
    "phase2": "Phase 2 — Calibrate individual cameras",
    "phase3": "Phase 3 — Joint camera and target-pose solve",
    "phase4": "Phase 4 — Free target geometry",
}

#: A1 section 3 novice (L1) and expert (L3) scripts. The Phase 4 novice
#: script keeps A1's meaning but avoids the banned "target moved" wording.
_ORIENTATION = {
    "phase1": (
        "We start with the photos from each camera. The detector returns target points for "
        "some photos. Check whether each camera has enough points and whether they cover the "
        "image; a large total can still hide a weak camera or a cluster in one corner.",
        "The saved output is the detector's returned correspondence set, not a transcript of "
        "every internal candidate. OpenCV and third-party internals are opaque unless this run "
        "recorded an observable Python-side stage. Compare per-camera counts and spatial "
        "coverage; do not infer false detections, rejection counts, or pose quality from a "
        "count alone.",
    ),
    "phase2": (
        "Each camera gets its own first lens estimate, with no lens distortion yet, from the "
        "detections Phase 2 actually used. Check the points and their fit error. Phase 2 does "
        "not establish where the cameras sit as a rig.",
        "Pinhole initialisation uses pyCamSet's Zhang closed-form route; telecentric "
        "initialisation uses an affine fit. Phase 2 saves no board poses, so the closed-form "
        "residual view is recalculated from saved inputs and must reproduce D2.6. The "
        "high-distortion branch is a fresh detection route, not an iterated lens-aware detector.",
    ),
    "phase3": (
        "Phase 3 uses the camera lens estimates and detections to find camera positions and a "
        "target position for each photo together. It also refines each camera's lens: the focal "
        "length and image centre Phase 2 estimated, and the lens distortion Phase 2 left at "
        "zero, unless you lock those values. Compare the same observed points before and "
        "after, and check which images were excluded. The first estimate and the point where "
        "the solver began can differ.",
        "The seed rig is recalculated from the camset Phase 3 used and its embedded detections; "
        "it must reproduce D3.3 per image. The solver start is a separate state and reproduces "
        "D3.5. Compare residuals on identical (camera, image, point) rows; list excluded or "
        "unposed rows separately. The saved target points stay at their design geometry in "
        "this phase.",
    ),
    "phase4": (
        "Phase 4 estimates a target shape from the points and compares it with the design you "
        "entered. The display is lined up for comparison; a changed estimate does not mean the "
        "cameras or the physical target changed position. Overall size is not checked, so "
        "measure the printed pattern if its size matters.",
        "The Phase 3 state comes only from the verified phase3_camset_used source and passes "
        "start-setting, residual-segment and embedded-detection identity checks. Compare rigs "
        "directly only when the pinhole frame checks pass and both fixed_pose values match. "
        "State each objective separately: the loss can differ between Phase 3 and Phase 4 (the "
        "GUI default is linear for both), and objectives under different losses are not "
        "comparable.",
    ),
}


def phase_name(phase: str) -> str:
    """Return the human phase name, falling back to the raw slug for unknown phases."""
    return PHASE_NAMES.get(str(phase), str(phase))


def orientation(phase: str, level: int) -> str:
    """Return the 'What this phase does' line: novice script, plus expert script at L3."""
    novice, expert = _ORIENTATION.get(str(phase), ("", ""))
    if not novice:
        return ""
    text = f"What this phase does: {novice}"
    return f"{text} In detail: {expert}" if level >= 3 else text


def compose(frame: Mapping[str, Any], level: int) -> str:
    """Join a frame's plain copy with the parts the chosen level adds."""
    parts = [str(frame.get("copy") or "")]
    if level >= 2 and frame.get("detail"):
        parts.append(str(frame["detail"]))
    if level >= 3 and frame.get("expert"):
        parts.append(str(frame["expert"]))
    return " ".join(part.strip() for part in parts if part and part.strip())


#: Widest a non-last evidence column opens; longer values wrap, and every cell
#: and header keeps its full text as a tooltip.
_MAX_COLUMN_WIDTH = 320
#: Above this many rows, rows keep one line (tooltips still hold the full text)
#: so a large table opens without measuring every row.
_MAX_WRAPPED_ROWS = 300


def setup_evidence_table(table: QTableWidget) -> None:
    """Make an education table show its values instead of eliding them.

    Phases 2-4 share this so a value is never cut to an ellipsis without a
    way to read it: cells wrap, long paths keep their file name visible, and
    columns are sized to their contents up to a cap after every fill.
    """
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    header.setStretchLastSection(True)
    table.setWordWrap(True)
    table.setTextElideMode(Qt.TextElideMode.ElideMiddle)


def fill_evidence_table(table: QTableWidget, rows: Sequence[Sequence[Any]]) -> None:
    """Fill ``table`` with ``rows``; each cell and header carries its full text as a tooltip."""
    table.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row):
            text = str(value)
            item = QTableWidgetItem(text)
            item.setToolTip(text)
            table.setItem(row_index, column, item)
    fit_evidence_columns(table)


def fit_evidence_columns(table: QTableWidget) -> None:
    """Size columns to their contents (capped), then let wrapped rows grow."""
    for column in range(table.columnCount()):
        header_item = table.horizontalHeaderItem(column)
        if header_item is not None:
            header_item.setToolTip(header_item.text())
    table.resizeColumnsToContents()
    header = table.horizontalHeader()
    for column in range(max(table.columnCount() - 1, 0)):
        # Cells wrap but headers cannot, so never cap a column below its header.
        cap = max(_MAX_COLUMN_WIDTH, header.sectionSizeFromContents(column).width())
        if header.sectionSize(column) > cap:
            header.resizeSection(column, cap)
    if table.rowCount() <= _MAX_WRAPPED_ROWS:
        table.resizeRowsToContents()


def make_level_selector(parent: QWidget, on_change: Callable[[int], None]) -> QComboBox:
    """Create the shared, accessible detail-level combo box (index 0 is L1)."""
    combo = QComboBox(parent)
    combo.setAccessibleName("Explanation detail level")
    combo.setToolTip("L1: plain overview and one decision. L2: adds diagnostic names, "
                     "numbers, units and sources. L3: adds method and limitations.")
    combo.addItems(list(LEVEL_LABELS))
    combo.currentIndexChanged.connect(on_change)
    return combo


class _FollowVisibility(QObject):
    """Hide or show a label together with the control it names."""

    def __init__(self, label: QLabel, control: QWidget) -> None:
        super().__init__(control)
        self._label = label

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API name
        if event.type() == QEvent.Type.HideToParent:
            self._label.hide()
        elif event.type() == QEvent.Type.ShowToParent:
            self._label.show()
        return False


def labelled(text: str, control: QWidget, parent: QWidget) -> QWidget:
    """Return a row with a visible label (its buddy is ``control``) and the control.

    The label follows the control's visibility, so call sites can keep hiding or
    showing the control itself.
    """
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    label = QLabel(text, row)
    label.setBuddy(control)
    control.setParent(row)
    layout.addWidget(label)
    layout.addWidget(control, 1)
    control.installEventFilter(_FollowVisibility(label, control))
    label.setVisible(not control.isHidden())
    row.label = label
    return row


class StageNavigation(QWidget):
    """Shared storyboard header: labelled stage list, Previous/Next stage and position.

    Every phase moves through its storyboard the same way; per-phase inner
    controls (camera, image, board view, recorded step) stay separate.
    """

    def __init__(self, stage: QComboBox, phase_label: str, parent: QWidget) -> None:
        super().__init__(parent)
        self._combo = stage
        # Two rows (stage list, then Previous/position/Next) so the header fits
        # the dialog's minimum width.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        chooser = QHBoxLayout()
        stepper = QHBoxLayout()
        layout.addLayout(chooser)
        layout.addLayout(stepper)
        label = QLabel("Stage", self)
        label.setBuddy(stage)
        stage.setParent(self)
        stage.setToolTip("Choose which part of this phase's explanation to read.")
        self.previous = QPushButton("Previous stage", self)
        self.next = QPushButton("Next stage", self)
        self.previous.setAccessibleName(f"Previous {phase_label} storyboard stage")
        self.next.setAccessibleName(f"Next {phase_label} storyboard stage")
        self.position = QLabel(self)
        self.position.setAccessibleName(f"{phase_label} stage position")
        chooser.addWidget(label)
        chooser.addWidget(stage, 1)
        stepper.addWidget(self.previous)
        stepper.addWidget(self.position, 1)
        stepper.addWidget(self.next)
        self.position.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.previous.clicked.connect(lambda: self._move(-1))
        self.next.clicked.connect(lambda: self._move(1))
        stage.currentIndexChanged.connect(self._update)
        self._update()

    def _move(self, delta: int) -> None:
        index = self._combo.currentIndex() + delta
        if 0 <= index < self._combo.count():
            self._combo.setCurrentIndex(index)

    def _update(self, *_args) -> None:
        index, count = self._combo.currentIndex(), self._combo.count()
        self.position.setText(f"Stage {index + 1} of {count}" if count else "No stages")
        self.previous.setEnabled(index > 0)
        self.next.setEnabled(0 <= index < count - 1)
