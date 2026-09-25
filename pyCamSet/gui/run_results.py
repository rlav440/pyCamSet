"""The small result view beside a phase's controls, shown once a run finishes.

Each phase gets one glance at its outcome, with the full detail a click away
in its diagnostics tab:

* Phase 1: the image with the most detections and the one with the fewest.
* Phase 2: every image's reprojection error, per camera, with each camera's
  RMS -- which camera is off, and whether a few images carry the error.
* Phase 3: the solved camera positions, each named with its mean
  reprojection error.

The figures are built from the run's own record and artifacts, never from
the live forms, so what is shown is what the run did.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

from pyCamSet.gui.theme import set_text_role

_LOGGER = logging.getLogger(__name__)


def _theme() -> str:
    application = QApplication.instance()
    return (application.property("pycamsetTheme") if application else None) or "Light"


def _canvas(figure, min_height: int = 220):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    from pyCamSet.gui.theme import apply_matplotlib_theme

    apply_matplotlib_theme(figure, _theme())
    canvas = FigureCanvasQTAgg(figure)
    canvas.setMinimumHeight(min_height)
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return canvas


class RunResultPanel(QFrame):
    """A titled card that holds one phase's result view.

    :param title: what the card shows, e.g. "Detections"
    :param open_diagnostics: called by the "See more in Diagnostics" link
    """

    def __init__(self, title: str, open_diagnostics: Callable[[], None],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("collapsibleSection")  # the themed card chrome
        self.setMinimumWidth(320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        heading = QLabel(title)
        set_text_role(heading, "subheading")
        layout.addWidget(heading)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        set_text_role(self.message, "muted")
        layout.addWidget(self.message)
        self._body = QVBoxLayout()
        self._body.setSpacing(6)
        layout.addLayout(self._body, 1)
        self.more = QPushButton("See more in Diagnostics")
        self.more.setObjectName("linkButton")
        self.more.setFlat(True)
        self.more.setCursor(Qt.CursorShape.PointingHandCursor)
        self.more.clicked.connect(open_diagnostics)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self.more)
        row.addStretch()
        layout.addLayout(row)
        self.setVisible(False)

    def clear(self) -> None:
        while self._body.count():
            item = self._body.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
            elif item.layout() is not None:
                _clear_layout(item.layout())

    def show_widget(self, widget: QWidget | None, message: str = "") -> None:
        """Replace the card's content and show it."""
        self.clear()
        self.message.setText(message)
        self.message.setVisible(bool(message))
        if widget is not None:
            self._body.addWidget(widget, 1)
        self.setVisible(True)


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().deleteLater()
        elif item.layout() is not None:
            _clear_layout(item.layout())


# ---------------------------------------------------------------------------
# Phase 1: best and worst detections


def detection_extremes(features: np.ndarray) -> Optional[tuple[tuple[int, int], tuple[int, int], int]]:
    """
    The (image, camera) cells with the most and the fewest detections.

    The worst is the fewest among images where the target was found at all:
    an image with none shows nothing to judge, so those are counted instead.

    :param features: detections per image (rows) per camera (columns)
    :return: the best cell, the worst cell, and how many cells had none;
        None when nothing was detected anywhere
    """
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or not np.any(features > 0):
        return None
    best = np.unravel_index(int(np.nanargmax(features)), features.shape)
    masked = np.where(features > 0, features, np.inf)
    worst = np.unravel_index(int(np.argmin(masked)), features.shape)
    empty = int(np.sum(features <= 0))
    return (int(best[0]), int(best[1])), (int(worst[0]), int(worst[1])), empty


def _load_detections(pickle_path: Path):
    from pyCamSet.gui.phase_1_detection import Phase1DiagnosticsTab
    from pyCamSet.workflow.workspace import as_io_path

    with open(as_io_path(pickle_path), "rb") as handle:
        payload = pickle.load(handle)
    return Phase1DiagnosticsTab._extract_detections_obj(payload)


def _image_points(detections, camera_index: int, image_index: int) -> np.ndarray:
    for camera in detections.get_cam_list():
        data = camera.get_data()
        if data is None or not len(data) or int(data[0, 0]) != camera_index:
            continue
        rows = data[data[:, 1].astype(int) == image_index]
        return rows[:, -2:]
    return np.empty((0, 2))


def _camera_images(image_folder: Path, camera: str) -> list[Path]:
    from natsort import natsorted

    from pyCamSet.utils.general_utils import glob_ims

    # The producer numbers images in this order; see the diagnostics viewer.
    return natsorted(glob_ims(image_folder / camera))


def detection_figure(image_path: Path, points: np.ndarray):
    """One image with its detections, styled like the diagnostics overlay."""
    import matplotlib.image as mpimg
    from matplotlib.figure import Figure

    from pyCamSet.gui.preferences import config_directory
    from pyCamSet.gui.visual_style import apply_visual_style, load_style_for_visual

    figure = Figure(figsize=(4, 3), layout="constrained")
    axes = figure.add_subplot(111)
    axes.axis("off")
    try:
        image = mpimg.imread(image_path)
        axes.imshow(image, cmap="gray" if getattr(image, "ndim", 3) == 2 else None)
    except (OSError, ValueError, SyntaxError) as exc:
        axes.text(0.5, 0.5, f"Image unreadable\n{type(exc).__name__}",
                  transform=axes.transAxes, ha="center", va="center")
    overlay = axes.scatter(points[:, 0], points[:, 1], s=10, c="lime", marker="o",
                           linewidths=0.4)
    # The same identity and saved style as the diagnostics viewer's overlay,
    # so a marker shape chosen there is the one shown here.
    overlay.set_gid("detection-overlay:phase1:summary")
    style, _source = load_style_for_visual(config_directory(), "phase1:detection-overlay")
    apply_visual_style(figure, style, _theme())
    return figure


def phase1_view(run: dict, workspace: Optional[Path]) -> tuple[Optional[QWidget], str]:
    """The best and worst detection images of a Phase 1 run, side by side."""
    from pyCamSet.workflow.workspace import resolve_artifact

    diagnostics = run.get("diagnostics") or {}
    names = diagnostics.get("cam_names") or []
    extremes = detection_extremes(diagnostics.get("D1.4_features_matrix") or [])
    if extremes is None or not names:
        return None, "No detections to show for this run."
    pickle_path = resolve_artifact(run, "phase1", workspace) if workspace else None
    if pickle_path is None:
        return None, "This run's detections file is missing."
    detections = _load_detections(pickle_path)
    image_folder = Path((run.get("params") or {}).get("f_loc", ""))
    features = np.asarray(diagnostics["D1.4_features_matrix"], dtype=float)

    row = QWidget()
    columns = QHBoxLayout(row)
    columns.setContentsMargins(0, 0, 0, 0)
    for label, (image_index, camera_index) in (("Best", extremes[0]), ("Worst", extremes[1])):
        camera = names[camera_index]
        images = _camera_images(image_folder, camera)
        column = QVBoxLayout()
        if image_index >= len(images):
            caption = QLabel(f"{label}: {camera}, image {image_index} is no longer on disk")
            caption.setWordWrap(True)
            column.addWidget(caption)
            columns.addLayout(column, 1)
            continue
        count = int(features[image_index, camera_index])
        caption = QLabel(f"<b>{label} detections</b><br>{camera} · {images[image_index].name}"
                         f"<br>{count} feature{'s' if count != 1 else ''} detected")
        caption.setWordWrap(True)
        caption.setToolTip(str(images[image_index]))
        column.addWidget(caption)
        points = _image_points(detections, camera_index, image_index)
        column.addWidget(_canvas(detection_figure(images[image_index], points)), 1)
        columns.addLayout(column, 1)
    empty = extremes[2]
    note = (f"{empty} camera image{'s' if empty != 1 else ''} had no detections."
            if empty else "")
    return row, note


# ---------------------------------------------------------------------------
# Phase 2: per-camera error


def _short(name: str, limit: int = 14) -> str:
    return name if len(name) <= limit else name[:limit - 1] + "…"


def phase2_figure(run: dict):
    """Every image's reprojection RMS per camera, with the camera's own RMS."""
    from matplotlib.figure import Figure

    diagnostics = run.get("diagnostics") or {}
    per_view = diagnostics.get("D2.6_per_view_reprojection") or {}
    per_camera = diagnostics.get("D2.1_per_camera_rms_reprojection") or {}
    names = [name for name in per_camera if name in per_view] or list(per_view)
    if not names:
        return None
    figure = Figure(figsize=(5, 3), layout="constrained")
    axes = figure.add_subplot(111)
    rng = np.random.default_rng(0)  # fixed jitter: the same run draws the same
    for index, name in enumerate(names):
        views = per_view.get(name) or {}
        rms = np.asarray(views.get("rms_px") or [], dtype=float)
        rms = rms[np.isfinite(rms)]
        if rms.size:
            axes.scatter(index + rng.uniform(-0.18, 0.18, rms.size), rms, s=9,
                         alpha=0.45, color="#0072b2", linewidths=0,
                         label="Each image" if index == 0 else None)
        if name in per_camera:
            axes.scatter([index], [float(per_camera[name])], marker="D", s=36,
                         color="#d55e00", zorder=3,
                         label="Camera RMS" if index == 0 else None)
    axes.set_xticks(range(len(names)))
    axes.set_xticklabels([_short(name) for name in names], rotation=35, ha="right", fontsize=8)
    axes.set_ylabel("Reprojection RMS (px)")
    axes.set_ylim(bottom=0)
    axes.set_title("Reprojection error per camera", fontsize=10)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.legend(fontsize=7, frameon=False, loc="upper right")
    return figure


def phase2_view(run: dict, workspace: Optional[Path]) -> tuple[Optional[QWidget], str]:
    del workspace
    figure = phase2_figure(run)
    if figure is None:
        return None, "No per-camera errors in this run."
    per_camera = (run.get("diagnostics") or {}).get("D2.1_per_camera_rms_reprojection") or {}
    worst = max(per_camera.items(), key=lambda item: item[1]) if per_camera else None
    note = (f"Highest camera RMS: {worst[0]} at {worst[1]:.2f} px." if worst else "")
    return _canvas(figure, 260), note


# ---------------------------------------------------------------------------
# Phase 3: camera poses with their errors


def camera_layout(cams) -> tuple[np.ndarray, np.ndarray, list[str], bool]:
    """
    Where to draw each camera, which way it looks, and its name.

    A telecentric camera is solved for its orientation alone -- its position
    is not observable -- so a rig of them all sits at the origin.  Those are
    drawn one unit back along their viewing direction, looking at the
    target, which shows the rig's geometry without inventing a distance.

    :return: positions, unit viewing directions, names, and whether the
        positions were placed rather than solved
    """
    positions, views, names = [], [], []
    for cam in cams:
        position = np.asarray(cam.position, dtype=float).reshape(-1)
        view = np.asarray(getattr(cam, "view", np.zeros(3)), dtype=float).reshape(-1)
        if position.size < 3:
            continue
        norm = np.linalg.norm(view[:3]) if view.size >= 3 else 0.0
        positions.append(position[:3])
        views.append(view[:3] / norm if norm > 0 else np.zeros(3))
        names.append(cam.name)
    positions, views = np.array(positions).reshape(-1, 3), np.array(views).reshape(-1, 3)
    placed = len(positions) > 1 and float(np.ptp(positions, axis=0).max()) < 1e-9
    if placed:
        positions = -views
    return positions, views, names, placed


def phase3_figure(cams, per_camera_error: dict[str, float], title: str = "Camera poses"):
    """Camera positions and viewing directions, each labelled with its error."""
    from matplotlib.figure import Figure

    from pyCamSet.gui.theme import THEME_TOKENS

    positions, views, names, placed = camera_layout(cams)
    if not len(positions):
        return None
    tokens = THEME_TOKENS.get(_theme(), THEME_TOKENS["Light"])
    figure = Figure(figsize=(5, 4), layout="constrained")
    axes = figure.add_subplot(111, projection="3d")
    span = float(np.ptp(positions, axis=0).max()) or 1.0
    axes.scatter(*positions.T, color="#0072b2", s=30, depthshade=False)
    for position, view in zip(positions, views):
        if np.any(view):
            axes.quiver(*position, *(view * span * 0.18), color="#0072b2",
                        linewidth=0.9, arrow_length_ratio=0.3)
    if placed:
        axes.scatter([0], [0], [0], marker="+", s=60, color=tokens["text_muted"])
    centre = positions.mean(axis=0)
    for name, position in zip(names, positions):
        # Nudge each label away from the rig's centre, off its own marker.
        outward = position - centre
        length = np.linalg.norm(outward)
        if length > 0:
            position = position + outward / length * span * 0.08
        error = per_camera_error.get(name)
        label = name if error is None else f"{name}\n{float(error):.2f} px"
        axes.text(*position, label, fontsize=7, ha="center", va="top", color=tokens["text"])
    axes.set_title(title, fontsize=10)
    for axis, name in ((axes.xaxis, "X"), (axes.yaxis, "Y"), (axes.zaxis, "Z")):
        # Open panes: filled grey ones swallow the labels in a dark theme.
        axis.pane.set_facecolor((0, 0, 0, 0))
        axis.pane.set_edgecolor(tokens["border"])
        axis.set_label_text(name, fontsize=8)
    axes.tick_params(labelsize=7)
    if placed:
        for setter in (axes.set_xticklabels, axes.set_yticklabels, axes.set_zticklabels):
            setter([])
    try:
        axes.set_box_aspect(np.maximum(np.ptp(positions, axis=0), span * 0.05))
    except Exception:  # an older matplotlib; the default aspect is fine
        pass
    figure.placed_positions = placed
    return figure


def phase3_view(run: dict, workspace: Optional[Path]) -> tuple[Optional[QWidget], str]:
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.workspace import resolve_artifact

    path = resolve_artifact(run, "phase3", workspace) if workspace else None
    if path is None:
        return None, "This run's camera set is missing."
    cams = load_CameraSet(path)
    diagnostics = run.get("diagnostics") or {}
    errors = diagnostics.get("D3.12_per_camera_mean_reprojection") or {}
    figure = phase3_figure(cams, errors, "Camera poses · mean reprojection error")
    if figure is None:
        return None, "No camera positions in this run."
    final = diagnostics.get("D3.6_final_euclid_px")
    note = f"Mean reprojection error {float(final):.2f} px over all cameras." if final is not None else ""
    if figure.placed_positions:
        note += (" These cameras are telecentric: only their direction is solved, so each is "
                 "drawn one unit back along it, looking at the target (+).")
    return _canvas(figure, 300), note


VIEWS = {"phase1": phase1_view, "phase2": phase2_view, "phase3": phase3_view}


def show_run_result(panel: RunResultPanel, phase: str, metadata: dict,
                    workspace: Optional[Path]) -> None:
    """Fill *panel* from a finished run's metadata; failures are shown, not raised."""
    if not metadata or metadata.get("error") or not metadata.get("run_id"):
        panel.setVisible(False)
        return
    try:
        widget, note = VIEWS[phase](metadata, workspace)
    except Exception as exc:
        _LOGGER.exception("Run result view failed")
        widget, note = None, f"Could not draw the result: {exc}"
    panel.show_widget(widget, note)
