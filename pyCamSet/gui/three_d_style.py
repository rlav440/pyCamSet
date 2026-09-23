'''Purpose: Reusable, presentation-only controls for managed 3D calibration views.
Status: Active; PyVista controls are explicit and Open3D limitations are stated.
Future: Persist presets only with a separate, versioned style contract.
'''
from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QWidget


class ThreeDStyleControls(QWidget):
    """Compact 3D cosmetics editor; values never enter calibration parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
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
        layout.addWidget(QLabel(
            "Open3D limitation: managed background, point size, view, axes and legend controls are unavailable; "
            "use the native viewer's interactive camera controls."
        ))

    def viewer_arguments(self) -> list[str]:
        """Return explicit viewer CLI arguments for the current cosmetic state."""
        return ["--3d-background", str(self.background.currentData()),
                "--3d-point-size", str(self.point_size.value()),
                "--3d-view", str(self.view.currentData()),
                "--3d-axes" if self.axes.isChecked() else "--no-3d-axes",
                "--3d-legend" if self.legend.isChecked() else "--no-3d-legend"]
