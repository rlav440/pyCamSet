"""
Assess Calibration utilities and widgets shared by Phase 3/4 diagnostics.

The visualisation is intentionally explicit: nothing renders until callers invoke
`render_runs(...)` after the user clicks "Assess Calibration".

Uses pyCamSet/utils/visualisation.py helper functions where possible and wraps
each figure in an expand / save-PNG card.  The panel is vertically scrollable so
figures are never squished.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
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
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    FigureCanvasQTAgg = None
    Figure = Any
    plt = None

try:
    from pyCamSet.utils.saving import load_CameraSet
except ImportError:  # pragma: no cover
    load_CameraSet = None

try:
    from pyCamSet.utils.visualisation import (
        plot_error_histogram,
        plot_per_camera_errors,
        plot_camera_arrangement,
    )
    _VIS_OK = True
except ImportError:  # pragma: no cover
    _VIS_OK = False
    plot_error_histogram = None  # type: ignore[assignment]
    plot_per_camera_errors = None  # type: ignore[assignment]
    plot_camera_arrangement = None  # type: ignore[assignment]


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
    cam_set: Any  # loaded CameraSet or None
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

    return RunViewData(run=run, cam_set=cams, cam_positions=cam_positions_arr, target_points=target_points)


class _FigureCard(QWidget):
    """A single labelled figure with Expand and Save-as-PNG buttons."""

    def __init__(
        self,
        title: str,
        fig: Any,
        run_dir: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._fig = fig
        self._run_dir = run_dir

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 8)

        hdr = QHBoxLayout()
        hdr.addWidget(make_section_label(title))
        hdr.addStretch()
        save_btn = QPushButton("Save as PNG")
        save_btn.setFixedWidth(100)
        save_btn.clicked.connect(self._save_png)
        hdr.addWidget(save_btn)
        expand_btn = QPushButton("Expand")
        expand_btn.setFixedWidth(70)
        expand_btn.clicked.connect(self._open_expanded)
        hdr.addWidget(expand_btn)
        layout.addLayout(hdr)

        if FigureCanvasQTAgg is None or fig is None:
            layout.addWidget(QLabel("matplotlib is unavailable."))
        else:
            canvas = FigureCanvasQTAgg(fig)
            canvas.setMinimumHeight(320)
            layout.addWidget(canvas)

    def _save_png(self) -> None:
        if self._fig is None:
            return
        safe_title = self._title.replace(" ", "_").replace("/", "_").replace("\\", "_")
        if self._run_dir is not None and self._run_dir.exists():
            save_path = self._run_dir / f"{safe_title}.png"
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Figure", f"{safe_title}.png", "PNG files (*.png)"
            )
            if not path:
                return
            save_path = Path(path)
        try:
            self._fig.savefig(save_path, dpi=150, bbox_inches="tight")
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", f"Could not save PNG:\n{exc}")

    def _open_expanded(self) -> None:
        if FigureCanvasQTAgg is None or self._fig is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(self._title)
        dlg.resize(1000, 750)
        root = QVBoxLayout(dlg)
        # Re-render a fresh figure at expanded size
        from matplotlib.figure import Figure as _Figure
        fig2 = _Figure(figsize=(10.0, 7.5), tight_layout=True)
        # Copy axes from self._fig into fig2 by re-running the same plot on a new canvas
        # (simple approach: just embed the existing figure at larger display size)
        root.addWidget(FigureCanvasQTAgg(self._fig))
        dlg.exec()


def _run_dir_from_run(run: dict) -> Optional[Path]:
    """Derive the run metadata directory from camset artifact path."""
    artifacts = run.get("artifacts") or {}
    for key in ("self_calibrated_camset", "optimised_camset", "initial_camset", "camset"):
        p = artifacts.get(key)
        if p:
            return Path(p).parent
    return None


def _build_figures_for_run(data: RunViewData) -> list[tuple[str, Any]]:
    """
    Build a list of (title, matplotlib_figure) for the given run using
    pyCamSet/utils/visualisation.py helpers where possible.

    Returns figures for:
    1. Error histogram (D3.3 per-image initial reprojection errors)
    2. Per-camera reprojection error (D3.12)
    3. Target reconstruction (3-D scatter from camset)
    4. Camera arrangement (pyvista screenshot wrapped in matplotlib, optional)
    """
    figures: list[tuple[str, Any]] = []
    if plt is None:
        return figures

    run = data.run
    diag = run.get("diagnostics") or {}
    phase_tag = canonical_phase_tag(run.get("phase"))

    # ── 1. Error histogram ────────────────────────────────────────────
    per_im_list: list[float] = []
    if phase_tag == "phase3":
        raw = diag.get("D3.3_per_image_initial_reprojection")
        if isinstance(raw, list):
            per_im_list = [float(v) for v in raw if v is not None]
    elif phase_tag == "phase4":
        raw = diag.get("D4.3_per_image_initial_reprojection")
        if isinstance(raw, list):
            per_im_list = [float(v) for v in raw if v is not None]

    if per_im_list and _VIS_OK and plot_error_histogram is not None:
        try:
            fig = plot_error_histogram(
                per_im_list,
                title="Per-image Initial Reprojection Error",
                xlabel="Reprojection Error (px)",
            )
            if fig is not None:
                figures.append(("Per-image Reprojection Error", fig))
        except Exception:
            pass

    # ── 2. Per-camera RPE ─────────────────────────────────────────────
    per_cam: dict = {}
    if phase_tag == "phase3":
        per_cam = dict(diag.get("D3.12_per_camera_mean_reprojection") or {})
    elif phase_tag == "phase4":
        per_cam = dict(diag.get("D4.12_per_camera_mean_reprojection") or {})

    if per_cam and _VIS_OK and plot_per_camera_errors is not None:
        try:
            cam_names = list(per_cam.keys())
            mean_errs = [float(per_cam[c]) for c in cam_names]
            fig = plot_per_camera_errors(
                cam_names,
                mean_errs,
                title="Per-camera Mean Reprojection Error",
            )
            if fig is not None:
                figures.append(("Per-camera Reprojection Error", fig))
        except Exception:
            pass

    # ── 3. Target reconstruction (matplotlib 3-D scatter) ─────────────
    try:
        pts = data.target_points
        fig_tgt = Figure(figsize=(7, 5.5), tight_layout=True)
        ax = fig_tgt.add_subplot(111, projection="3d")
        if pts.size:
            sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, c=pts[:, 2],
                            cmap="viridis", alpha=0.85)
            fig_tgt.colorbar(sc, ax=ax, label="Z")
        else:
            ax.text(0.5, 0.5, 0.5, "No target points", transform=ax.transAxes,
                    ha="center", va="center")
        ax.set_title("Calibration Target Reconstruction")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        figures.append(("Target Reconstruction", fig_tgt))
    except Exception:
        pass

    # ── 4. Camera arrangement ─────────────────────────────────────────
    if data.cam_set is not None and _VIS_OK and plot_camera_arrangement is not None:
        try:
            fig_cam = plot_camera_arrangement(data.cam_set, title="Camera Arrangement")
            if fig_cam is not None:
                figures.append(("Camera Arrangement", fig_cam))
        except Exception:
            pass

    # Fallback: matplotlib 3-D scatter for camera positions
    if not any(t == "Camera Arrangement" for t, _ in figures):
        try:
            cams = data.cam_positions
            fig_cp = Figure(figsize=(7, 5.5), tight_layout=True)
            ax = fig_cp.add_subplot(111, projection="3d")
            if cams.size:
                ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], s=24, c="#ff7f0e")
                for idx, p in enumerate(cams):
                    ax.text(float(p[0]), float(p[1]), float(p[2]), f"cam{idx}", fontsize=8)
            else:
                ax.text(0.5, 0.5, 0.5, "No camera positions", transform=ax.transAxes,
                        ha="center", va="center")
            ax.set_title("Camera Positions")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            figures.append(("Camera Positions", fig_cp))
        except Exception:
            pass

    return figures


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

        # Horizontal splitter for side-by-side run comparison
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
        """Build a vertically-scrollable panel of figure cards for one run."""
        phase = canonical_phase_tag(data.run.get("phase"))
        rid = data.run.get("run_id", "unknown")
        run_dir = _run_dir_from_run(data.run)

        # Outer container
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(2, 2, 2, 2)

        hdr = QLabel(f"{phase} | {rid}")
        hdr.setStyleSheet("font-weight: bold;")
        outer_layout.addWidget(hdr)

        # Scroll area wrapping all figure cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        inner_layout.setSpacing(12)

        # Build figures using visualisation.py helpers
        try:
            figures = _build_figures_for_run(data)
        except Exception as exc:
            inner_layout.addWidget(QLabel(f"Could not build figures: {exc}"))
            figures = []

        if not figures:
            inner_layout.addWidget(QLabel("No figures available for this run."))
        else:
            for title, fig in figures:
                card = _FigureCard(title, fig, run_dir=run_dir, parent=inner)
                inner_layout.addWidget(card)
                inner_layout.addWidget(make_separator())

        inner_layout.addStretch()
        scroll.setWidget(inner)
        outer_layout.addWidget(scroll, stretch=1)
        return outer

    def render_runs(self, runs: list[dict]) -> None:
        if not runs:
            self._status.setText("Select one or two runs and click Assess Calibration.")
            self._replace_splitter_widgets([])
            return

        runs = runs[-2:]  # silently limit to the last two runs

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
            self._status.setText(
                f"Side-by-side: {left} (left) vs {right} (right). Scroll each panel independently."
            )
            self._splitter.setSizes([1, 1])
