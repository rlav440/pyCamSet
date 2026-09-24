'''Purpose: Reusable, presentation-only controls for managed 3D calibration views.
Status: Active; PyVista controls are explicit and Open3D limitations are stated.
Future: Extend persisted controls only with presentation-invariance tests.
'''
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QWidget,
)

_STYLE_SCHEMA = "pycamset.3d-visual-style"
_STYLE_VERSION = 1
_BACKGROUNDS = {"theme", "white", "charcoal"}
_VIEWS = {"isometric", "top", "front", "side"}


def _style_path(visual_id: str) -> Path:
    """Hash the fixed visual identity so paths cannot escape app config."""
    from pyCamSet.gui.preferences import config_directory

    digest = hashlib.sha256(visual_id.encode("utf-8")).hexdigest()[:16]
    return config_directory() / "visual-styles" / f"3d-{digest}.json"


def _validated_style(document: object, visual_id: str) -> dict:
    """Validate the complete versioned style before any control is changed."""
    if not isinstance(document, dict) or set(document) != {
            "schema", "version", "visual_id", "style"}:
        raise ValueError("3D style document has missing or unknown fields")
    if (document["schema"] != _STYLE_SCHEMA
            or isinstance(document["version"], bool)
            or document["version"] != _STYLE_VERSION):
        raise ValueError("Unsupported 3D style schema or version")
    if document["visual_id"] != visual_id:
        raise ValueError("3D style belongs to a different visual")
    style = document["style"]
    required = {"background", "point_size", "view", "axes", "legend"}
    if not isinstance(style, dict) or set(style) != required:
        raise ValueError("3D style has missing or unknown settings")
    if (not isinstance(style["background"], str) or style["background"] not in _BACKGROUNDS
            or not isinstance(style["view"], str) or style["view"] not in _VIEWS):
        raise ValueError("3D style contains an unsupported background or view")
    size = style["point_size"]
    if (isinstance(size, bool) or not isinstance(size, (int, float))
            or not math.isfinite(size) or not 1.0 <= size <= 20.0):
        raise ValueError("point_size must be finite and between 1 and 20")
    if not isinstance(style["axes"], bool) or not isinstance(style["legend"], bool):
        raise ValueError("axes and legend must be boolean")
    return {"background": style["background"], "point_size": float(size),
            "view": style["view"], "axes": style["axes"], "legend": style["legend"]}


class ThreeDStyleControls(QWidget):
    """Compact 3D cosmetics editor; values never enter calibration parameters."""

    def __init__(self, parent=None, visual_id: str = "assessment:phase3",
                 show_open3d_note: bool = True):
        super().__init__(parent)
        if not isinstance(visual_id, str) or not visual_id.strip():
            raise ValueError("visual_id must be a non-empty string")
        self._visual_id = visual_id
        self._style_file = _style_path(visual_id)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("3D background:"))
        self.background = QComboBox()
        self.background.addItem("Theme default", "theme")
        self.background.addItem("White", "white")
        self.background.addItem("Charcoal", "charcoal")
        layout.addWidget(self.background)
        layout.addWidget(QLabel("Point size:"))
        self.point_size = QDoubleSpinBox()
        self.point_size.setRange(1.0, 20.0)
        self.point_size.setSingleStep(0.5)
        self.point_size.setValue(3.0)
        self.point_size.setSuffix(" px")
        self.point_size.setToolTip("Display size only; point coordinates and error scalars are unchanged.")
        layout.addWidget(self.point_size)
        layout.addWidget(QLabel("View:"))
        self.view = QComboBox()
        for label, key in (("Isometric", "isometric"), ("Top", "top"),
                           ("Front", "front"), ("Side", "side")):
            self.view.addItem(label, key)
        layout.addWidget(self.view)
        self.axes = QCheckBox("Axes")
        self.axes.setChecked(True)
        self.legend = QCheckBox("Error legend")
        self.legend.setChecked(True)
        self.legend.setToolTip("Shows or hides the reprojection-error scale; it does not rescale values.")
        layout.addWidget(self.axes)
        layout.addWidget(self.legend)
        self.save_style = QPushButton("Save style")
        self.save_style.setToolTip("Save these presentation settings for this visual on this computer.")
        self.save_style.clicked.connect(self._save_style)
        layout.addWidget(self.save_style)
        self.load_style = QPushButton("Load saved")
        self.load_style.clicked.connect(self._load_style)
        layout.addWidget(self.load_style)
        self.reset_style = QPushButton("Reset style")
        self.reset_style.setToolTip("Remove this visual's saved style and inherit theme defaults.")
        self.reset_style.clicked.connect(self._reset_style)
        layout.addWidget(self.reset_style)
        self._open3d_note = QLabel(
            "Open3D limitation: managed background, point size, view, axes and legend controls are unavailable; "
            "use the native viewer's interactive camera controls."
        )
        self._open3d_note.setVisible(show_open3d_note)
        layout.addWidget(self._open3d_note)
        self._load_style(silent=True)

    def _style_values(self) -> dict:
        """Capture presentation controls without touching reconstructed data."""
        return {"background": self.background.currentData(),
                "point_size": self.point_size.value(), "view": self.view.currentData(),
                "axes": self.axes.isChecked(), "legend": self.legend.isChecked()}

    def _set_style(self, style: dict) -> None:
        """Apply validated style values to the controls as a single operation."""
        self.background.setCurrentIndex(self.background.findData(style["background"]))
        self.point_size.setValue(style["point_size"])
        self.view.setCurrentIndex(self.view.findData(style["view"]))
        self.axes.setChecked(style["axes"])
        self.legend.setChecked(style["legend"])

    def _save_style(self) -> None:
        """Persist one visual's settings in the platform app-config directory."""
        document = {"schema": _STYLE_SCHEMA, "version": _STYLE_VERSION,
                    "visual_id": self._visual_id, "style": self._style_values()}
        try:
            _validated_style(document, self._visual_id)
            self._style_file.parent.mkdir(parents=True, exist_ok=True)
            self._style_file.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "3D style not saved", str(exc))

    def _load_style(self, silent: bool = False) -> None:
        """Load and apply a saved style; reject malformed data without partial edits."""
        if not self._style_file.exists():
            return
        try:
            document = json.loads(self._style_file.read_text(encoding="utf-8"))
            self._set_style(_validated_style(document, self._visual_id))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            if not silent:
                QMessageBox.warning(self, "3D style not loaded", str(exc))

    def _reset_style(self) -> None:
        """Remove the saved override and restore inheriting defaults."""
        answer = QMessageBox.question(
            self, "Reset 3D style", "Remove this visual's saved style and inherit theme defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._style_file.unlink(missing_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "3D style not reset", str(exc))
            return
        self._set_style({"background": "theme", "point_size": 3.0,
                         "view": "isometric", "axes": True, "legend": True})

    def viewer_arguments(self) -> list[str]:
        """Return explicit viewer CLI arguments for the current cosmetic state."""
        return ["--3d-background", str(self.background.currentData()),
                "--3d-point-size", str(self.point_size.value()),
                "--3d-view", str(self.view.currentData()),
                "--3d-axes" if self.axes.isChecked() else "--no-3d-axes",
                "--3d-legend" if self.legend.isChecked() else "--no-3d-legend"]
