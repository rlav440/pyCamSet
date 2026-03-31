"""
Assess Calibration utilities and widgets shared by Phase 3/4 diagnostics.

The visualisation is intentionally explicit: nothing renders until callers invoke
`render_runs(...)` after the user clicks "Assess Calibration".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pyCamSet.gui.shared_functions import make_section_label, make_separator

try:
    import matplotlib

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - handled at runtime in widget
    FigureCanvasQTAgg = None
    Figure = Any

try:
    from pyCamSet.utils.saving import load_CameraSet
except ImportError:  # pragma: no cover - handled at runtime in widget
    load_CameraSet = None


def canonical_phase_tag(value: str | None) -> str:
    """Normalise *value* to a canonical phase tag (e.g. ``"phase3"``).

    Returns ``"unknown"`` for blank input.  Legacy aliases such as ``phase5``
    or ``visualise_target`` are no longer mapped; they are returned as-is so
    the caller can see the raw value from persisted metadata.
    """
    txt = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return txt or "unknown"


def resolve_run_camset_artifact(run: dict) -> Optional[Path]:
    """Return the first existing camset path from *run*'s artifact metadata.

    Recognised keys (in priority order): ``self_calibrated_camset``,
    ``optimised_camset``, ``initial_camset``, ``camset``.

    Legacy keys such as ``phase5_camset`` are **not** supported.  If a run
    produced via an old build is passed, its ``artifacts`` dict will contain
    none of the recognised keys and ``None`` is returned.
    """
    artifacts = run.get("artifacts") or {}
    for key in (
        "self_calibrated_camset",
        "optimised_camset",
        "initial_camset",
        "camset",
    ):
        p = artifacts.get(key)
        if p:
            pp = Path(p)
            if pp.exists():
                return pp
    return None


def merge_phase3_phase4_runs(phase3_runs: list[dict], phase4_runs: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for phase_name, runs in (("phase3", phase3_runs), ("phase4", phase4_runs)):
        for run in runs:
            copied = dict(run)
            copied.setdefault("phase", phase_name)
            copied["display_name"] = f"{copied.get('phase', phase_name)} | {copied.get('run_id', 'unknown')}"
            merged.append(copied)
    merged.sort(key=lambda r: str(r.get("run_id", "")))
    return merged


def ordered_visualisation_selection(selected_runs: list[dict], all_runs: list[dict], max_runs: int = 2) -> list[dict]:
    if not selected_runs:
        return []
    index_map = {id(run): idx for idx, run in enumerate(all_runs)}
    ordered = sorted(selected_runs, key=lambda r: index_map.get(id(r), 10**9))
    return ordered[-max_runs:]


@dataclass
class RunViewData:
    run: dict
    cam_positions: np.ndarray
    target_points: np.ndarray


def _extract_run_view_data(run: dict) -> RunViewData:
    if load_CameraSet is None:
        raise RuntimeError("pyCamSet loading utilities are unavailable.")

    camset_path = resolve_run_camset_artifact(run)
    if camset_path is None:
        raise RuntimeError("Selected run has no readable camset artifact.")

    cams = load_CameraSet(camset_path)

    cam_positions = []
    for cam in cams:
        p = np.array(cam.position).reshape(-1)
        if p.size >= 3 and np.all(np.isfinite(p[:3])):
            cam_positions.append(p[:3])
    cam_positions_arr = np.array(cam_positions, dtype=float) if cam_positions else np.zeros((0, 3), dtype=float)

    handler = getattr(cams, "calibration_handler", None)
    params = getattr(cams, "calibration_params", None)
    target_points = None

    if handler is not None and params is not None:
        try:
            if hasattr(handler, "get_bundle_adjustment_inputs"):
                pts = handler.get_bundle_adjustment_inputs(np.array(params), make_points=True)
                pts = np.array(pts, dtype=float)
                if pts.ndim == 3 and pts.shape[1] > 0:
                    target_points = np.nanmean(pts, axis=0)
            if target_points is None and hasattr(handler, "target"):
                target_points = np.array(handler.target.point_data, dtype=float).reshape(-1, 3)
        except Exception:
            target_points = None

    if target_points is None:
        target_points = np.zeros((0, 3), dtype=float)

    return RunViewData(run=run, cam_positions=cam_positions_arr, target_points=target_points)


_FIGURE_TOOLTIPS: dict[str, str] = {
    "Target Reconstruction": (
        "3-D scatter of reconstructed calibration-target points.\n\n"
        "What it shows: the spatial layout of the calibration target as\n"
        "estimated by the bundle adjustment.\n\n"
        "How to interpret: points should form a tight, regular pattern\n"
        "matching the physical target geometry.  Outliers or a distorted\n"
        "cluster suggest residual calibration error."
    ),
    "Camera Layout": (
        "3-D scatter of estimated camera optical-centre positions.\n\n"
        "What it shows: where each camera is located in the world frame\n"
        "after optimisation.\n\n"
        "How to interpret: cameras should be distributed around the\n"
        "target volume.  Overlapping or wildly separated positions can\n"
        "indicate a degenerate or poorly constrained calibration."
    ),
}


class _FigureCard(QWidget):
    def __init__(self, title: str, plot_fn: Callable[[Any], None], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title
        self._plot_fn = plot_fn

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        hdr = QHBoxLayout()
        title_lbl = make_section_label(title)
        if title in _FIGURE_TOOLTIPS:
            title_lbl.setToolTip(_FIGURE_TOOLTIPS[title])
        hdr.addWidget(title_lbl)
        hdr.addStretch()
        expand = QPushButton("Expand")
        expand.clicked.connect(self._open_expanded)
        hdr.addWidget(expand)
        layout.addLayout(hdr)

        if FigureCanvasQTAgg is None or Figure is None:
            layout.addWidget(QLabel("matplotlib is unavailable."))
            return

        fig = Figure(figsize=(5.4, 4.2), tight_layout=True)
        self._plot_fn(fig)
        layout.addWidget(FigureCanvasQTAgg(fig))

    def _open_expanded(self) -> None:
        if FigureCanvasQTAgg is None or Figure is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(self._title)
        dlg.resize(960, 720)
        root = QVBoxLayout(dlg)
        fig = Figure(figsize=(9.0, 6.0), tight_layout=True)
        self._plot_fn(fig)
        root.addWidget(FigureCanvasQTAgg(fig))
        dlg.exec()


class AssessCalibrationWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.addWidget(make_section_label("Assess Calibration"))
        root.addWidget(make_separator())

        self._status = QLabel("Select one or two runs and click Assess Calibration.")
        self._status.setStyleSheet("color: #666;")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self._splitter, stretch=1)

    def _replace_splitter_widgets(self, widgets: list[QWidget]) -> None:
        while self._splitter.count():
            old = self._splitter.widget(0)
            old.setParent(None)
            old.deleteLater()
        for widget in widgets:
            self._splitter.addWidget(widget)

    def _build_run_panel(self, data: RunViewData) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        phase = canonical_phase_tag(data.run.get("phase"))
        rid = data.run.get("run_id", "unknown")
        hdr = QLabel(f"{phase} | {rid}")
        hdr.setStyleSheet("font-weight: bold;")
        layout.addWidget(hdr)

        def _plot_target(fig: Any) -> None:
            ax = fig.add_subplot(111, projection="3d")
            pts = data.target_points
            if pts.size:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, c="#1f77b4", alpha=0.85)
            else:
                ax.text(0.5, 0.5, 0.5, "No target points in this run", transform=ax.transAxes)
            ax.set_title("Target Reconstruction")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        def _plot_cams(fig: Any) -> None:
            ax = fig.add_subplot(111, projection="3d")
            cams = data.cam_positions
            if cams.size:
                ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], s=22, c="#ff7f0e")
                for idx, p in enumerate(cams):
                    ax.text(float(p[0]), float(p[1]), float(p[2]), f"cam{idx}", fontsize=8)
            else:
                ax.text(0.5, 0.5, 0.5, "No camera positions in this run", transform=ax.transAxes)
            ax.set_title("Camera Layout")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        layout.addWidget(_FigureCard("Target Reconstruction", _plot_target, parent=panel))
        layout.addWidget(_FigureCard("Camera Layout", _plot_cams, parent=panel))
        layout.addStretch()
        return panel

    def render_runs(self, runs: list[dict]) -> None:
        if not runs:
            self._status.setText("Select one or two runs and click Assess Calibration.")
            self._replace_splitter_widgets([])
            return

        if len(runs) > 2:
            QMessageBox.information(self, "Selection limit", "Select up to two runs for Assess Calibration.")
            return

        try:
            bundles = [_extract_run_view_data(run) for run in runs]
        except Exception as exc:
            self._status.setText(f"Assessment failed: {exc}")
            self._replace_splitter_widgets([])
            return

        widgets = [self._build_run_panel(bundle) for bundle in bundles]
        self._replace_splitter_widgets(widgets)

        if len(widgets) == 1:
            self._status.setText("Showing a single run.")
            self._splitter.setSizes([1])
        else:
            left = runs[0].get("run_id", "run A")
            right = runs[1].get("run_id", "run B")
            self._status.setText(f"Showing side-by-side comparison: {left} (left) vs {right} (right).")
            self._splitter.setSizes([1, 1])
