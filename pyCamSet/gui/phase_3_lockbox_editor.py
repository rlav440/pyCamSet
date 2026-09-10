'''
Purpose: Phase 3 lockbox prior editor for non-destructive camera-centre repair.
Status:  Experimental v1 GUI: translation-only edits, trust flags, plane groups,
         reset/undo/redo, preview, and derived camset/metadata save.
Future:  Replace the static 3D preview with an interactive viewport and drag handles.
'''
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QEvent, QObject, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from pyCamSet.gui.lockbox_geometry import (
    CameraEditRecord,
    PlaneGroup,
    build_lockbox_metadata,
    camera_center_from_extrinsic,
    extrinsic_from_center_preserving_rotation,
    fit_plane,
    match_signed_plane_offset,
    project_point_to_plane,
    radius_from_center,
    reference_radius,
    resolve_object_centre,
    select_fit_members,
    snap_radius_to_reference,
)
from pyCamSet.utils.saving import load_CameraSet

# Try to import CameraSet for clean geometry-only clone.
try:
    from pyCamSet.cameras.camera_set import CameraSet
except Exception:
    CameraSet = None

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    _MATPLOTLIB_OK = True
except Exception:  # pragma: no cover - GUI fallback for lean environments.
    FigureCanvas = None
    Figure = None
    _MATPLOTLIB_OK = False

try:
    import open3d as _o3d
    import open3d.visualization.gui as _o3d_gui
    import open3d.visualization.rendering as _o3d_rendering
    _OPEN3D_OK = True
except Exception:  # pragma: no cover
    _o3d = None
    _o3d_gui = None
    _o3d_rendering = None
    _OPEN3D_OK = False


class _DelayedTooltipFilter(QObject):
    """Event filter that shows a tooltip after the mouse has hovered for 3 seconds.

    Install on any QWidget button to replace the OS-default instant tooltip with
    a deliberately delayed one so it does not flash on accidental hover.
    """

    _DELAY_MS = 3000

    def __init__(self, widget: "QWidget", text: str) -> None:
        super().__init__(widget)          # parent keeps filter alive with the widget
        self._text = text
        self._widget = widget
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._show_tip)

    def _show_tip(self) -> None:
        pos = self._widget.mapToGlobal(self._widget.rect().center())
        QToolTip.showText(pos, self._text, self._widget)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if event.type() == QEvent.Type.Enter:
            self._timer.start(self._DELAY_MS)
        elif event.type() == QEvent.Type.Leave:
            self._timer.stop()
            QToolTip.hideText()
        return False  # never consume the event


@dataclass(slots=True)
class LockboxEditorResult:
    """Paths returned when the user applies an edited lockbox prior."""

    original_source_camset: str
    edited_source_camset: str
    effective_lockbox_source: str
    metadata_path: str
    edited_count: int
    max_shift: float


@dataclass(slots=True)
class _CameraRowState:
    """Mutable per-camera editor state kept separate from the pyCamSet Camera."""

    name: str
    trust: str = "uncertain"
    plane_group: str = ""
    included_in_lockbox: bool = True
    is_reference: bool = False
    edited_center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    original_center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))


class Phase3LockboxEditor(QDialog):
    """Translation-only editor for deriving a Phase 3 lockbox source camset."""

    def __init__(
        self,
        source_camset_path: str | Path,
        workspace_path: str | Path,
        target: object | None = None,
        active_camera_names: list[str] | None = None,
        fixed_params: dict | None = None,
        lockbox_params: dict | None = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Phase 3 Lockbox Prior Editor")
        self.resize(1180, 720)

        self.source_camset_path = self._resolve_source_camset_path(Path(source_camset_path))
        self.workspace_path = Path(workspace_path)
        self.target = target
        self.active_camera_names = list(active_camera_names or [])
        self.fixed_params = fixed_params or {}
        self.lockbox_params = dict(lockbox_params or {})
        self.result: LockboxEditorResult | None = None

        self.source_camset = load_CameraSet(self.source_camset_path)
        self.working_camset = self._clone_camset_geometry_only(self.source_camset)
        self.centre_definition = resolve_object_centre(target)
        self.object_centre = np.asarray(self.centre_definition["point"], dtype=float)
        self.states: dict[str, _CameraRowState] = {}
        self.undo_stack: list[dict[str, dict]] = []
        self.redo_stack: list[dict[str, dict]] = []
        self.edit_history: list[str] = []
        self._open3d_auto_opened = False
        self._o3d_window = None
        self._o3d_scene_widget = None
        self._o3d_status_label = None
        self._o3d_camera_info_labels = {}
        self._o3d_select_checks = {}
        self._o3d_ref_checks = {}
        self._o3d_return_code = QDialog.DialogCode.Rejected
        self._o3d_scene_initialised = False
        self._o3d_label_ids: list[object] = []
        self._o3d_box_targets: dict[str, dict[str, np.ndarray]] = {}
        self._o3d_apply_requested = False
        self._o3d_visual_settings = {
            'view_mode': 'Planetary',
            'box_picking': False,  # left-click = S (select), right-click = R (reference)
            'show_skybox': False,
            'show_ground': True,
            'ground_plane': 'XZ floor',
            'show_axes': True,
            'background': 'Dark calibration',
            'lighting': 'Medium shadows',
            'show_lockbox': False,
        }

        self._initialise_states()
        self._build_ui()
        self._sync_reference_list_from_state()
        self.reference_list.itemChanged.connect(self._on_reference_list_changed)
        self._refresh_table()
        self._refresh_preview()
        self._refresh_status()
        self._show_initial_warnings()

    def exec(self) -> int:
        """Run the editor as a single Open3D-native window when available."""
        if not _OPEN3D_OK:
            return super().exec()
        self._o3d_return_code = QDialog.DialogCode.Rejected
        self._o3d_apply_requested = False
        self._open_open3d_view()
        if self._o3d_apply_requested:
            # Save/apply on the Qt side after the Open3D callback stack has unwound.
            # The large-shift confirmation uses QMessageBox; doing that directly from
            # the Open3D button callback has been observed to destabilise the editor.
            self._save_and_apply()
            if self.result is not None:
                self._o3d_return_code = QDialog.DialogCode.Accepted
        return int(self._o3d_return_code)

    def _show_initial_warnings(self) -> None:
        """Show one-off startup warnings for optional editor capabilities."""
        warnings: list[str] = []
        if not _OPEN3D_OK:
            warnings.append("Open3D is not available; using the Qt-only lockbox editor view.")
        if (not _MATPLOTLIB_OK) and (not _OPEN3D_OK):
            warnings.append("3D preview is unavailable because neither Open3D nor matplotlib is installed.")
        if not warnings:
            return
        QMessageBox.information(self, "Lockbox editor startup", "\n".join(warnings))

    def _resolve_source_camset_path(self, input_path: Path) -> Path:
        """Resolve a user-supplied source path to a concrete camset/json file.

        Some call paths can pass a run directory instead of a file. In that case,
        pick a sensible camset candidate rather than trying to open the directory.
        """
        path = Path(input_path)
        if path.is_file():
            return path
        if not path.is_dir():
            raise FileNotFoundError(f"Lockbox source path does not exist: {path}")

        # Prefer explicit lockbox editor outputs first when present.
        preferred_names = [
            "edited_lockbox_source.camset",
            "phase3_result.camset",
            "phase2_result.camset",
            "camset.camset",
        ]
        for name in preferred_names:
            candidate = path / name
            if candidate.is_file():
                return candidate

        # Fall back to most recent *.camset, then *.json if needed.
        try:
            camset_candidates = sorted(path.glob("*.camset"), key=lambda p: p.stat().st_mtime, reverse=True)
        except PermissionError as exc:
            raise PermissionError(f"Cannot read lockbox source directory: {path}") from exc
        if camset_candidates:
            return camset_candidates[0]
        try:
            json_candidates = sorted(path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        except PermissionError as exc:
            raise PermissionError(f"Cannot read lockbox source directory: {path}") from exc
        if json_candidates:
            return json_candidates[0]

        raise ValueError(
            "Lockbox source path points to a directory, but no .camset or .json file was found inside: "
            f"{path}"
        )

    def _clone_camset_geometry_only(self, camset) -> CameraSet:
        """Return a new CameraSet containing only camera geometry, not handler/detections.

        deepcopy fails when the source camset carries an OpenCV/ArUco detection
        handler that cannot be pickled. The lockbox editor only needs extrinsics,
        intrinsics, distortions and resolutions.
        """
        if CameraSet is None:
            raise RuntimeError("CameraSet class is not available; cannot create editor working copy")
        cam_dict = {}
        for cam in camset:
            cam_dict[cam.name] = type(cam)(
                extrinsic=np.array(cam.extrinsic, copy=True),
                intrinsic=np.array(cam.intrinsic, copy=True),
                res=np.array(cam.res, copy=True),
                distortion_coefs=np.array(cam.distortion_coefs, copy=True),
                name=cam.name,
                minimal=True,
            )
        return CameraSet(camera_dict=cam_dict)

    def _initialise_states(self) -> None:
        """Copy source camera centres into row state for non-destructive editing."""
        for cam in self.source_camset:
            centre = camera_center_from_extrinsic(cam.extrinsic)
            self.states[cam.name] = _CameraRowState(
                name=cam.name,
                edited_center=centre.copy(),
                original_center=centre.copy(),
            )

    def _sync_reference_list_from_state(self) -> None:
        """Update reference list checkboxes from _CameraRowState.is_reference."""
        self.reference_list.blockSignals(True)
        for i in range(self.reference_list.count()):
            item = self.reference_list.item(i)
            name = item.text()
            state = self.states.get(name)
            if state is not None:
                item.setCheckState(Qt.CheckState.Checked if state.is_reference else Qt.CheckState.Unchecked)
        self.reference_list.blockSignals(False)

    def _on_reference_list_changed(self, item: QListWidgetItem) -> None:
        """Sync reference list checkbox changes to _CameraRowState.is_reference."""
        name = item.text()
        if name in self.states:
            self.states[name].is_reference = (item.checkState() == Qt.CheckState.Checked)
            self._refresh_table()

    def _build_ui(self) -> None:
        """Build the three-panel editor layout."""
        root = QVBoxLayout(self)
        banner = QLabel("Editing lockbox prior centres only. Original source camset will not be modified.")
        banner.setStyleSheet("font-weight: bold; color: #9a5b00;")
        root.addWidget(banner)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "selected",
            "included",
            "camera name",
            "trust state",
            "edited",
            "delta",
            "radius",
            "plane group",
            "ref",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._refresh_status)
        splitter.addWidget(self.table)

        centre_panel = QWidget()
        centre_layout = QVBoxLayout(centre_panel)

        # Text diagnostics (always present)
        self.preview_summary = QTextEdit()
        self.preview_summary.setReadOnly(True)
        centre_layout.addWidget(self.preview_summary, stretch=1)

        # 3D view button — opens a native Open3D interactive window
        view_3d_row = QHBoxLayout()
        self._view_3d_btn = QPushButton("View in 3D (Open3D)")
        _v3d_tip = _DelayedTooltipFilter(
            self._view_3d_btn,
            "Open an interactive Open3D window with camera wireframes, viewcones,\n"
            "delta lines, and coordinate axes. Use mouse to zoom, rotate, and pan.",
        )
        self._view_3d_btn.installEventFilter(_v3d_tip)
        self._view_3d_btn.setEnabled(_OPEN3D_OK)
        if not _OPEN3D_OK:
            self._view_3d_btn.setText("View in 3D (Open3D not available)")
        self._view_3d_btn.clicked.connect(self._open_open3d_view)
        view_3d_row.addWidget(self._view_3d_btn)
        view_3d_row.addStretch()
        centre_layout.addLayout(view_3d_row)

        # Matplotlib fallback kept only when Open3D is unavailable.
        if (not _OPEN3D_OK) and _MATPLOTLIB_OK:
            self.figure = Figure(figsize=(5, 3))
            self.canvas = FigureCanvas(self.figure)
            centre_layout.addWidget(self.canvas, stretch=0)
        else:
            self.figure = None
            self.canvas = None

        splitter.addWidget(centre_panel)

        tools = QWidget()
        tools_layout = QVBoxLayout(tools)
        self.selection_label = QLabel("No cameras selected")
        self.selection_label.setWordWrap(True)
        tools_layout.addWidget(self.selection_label)

        nudge_form = QFormLayout()
        self.nudge_x = self._make_float_spin()
        self.nudge_y = self._make_float_spin()
        self.nudge_z = self._make_float_spin()
        self.nudge_x.setToolTip("X component of the translation nudge (target-frame units).")
        self.nudge_y.setToolTip("Y component of the translation nudge (target-frame units).")
        self.nudge_z.setToolTip("Z component of the translation nudge (target-frame units).")
        nudge_form.addRow("Nudge X:", self.nudge_x)
        nudge_form.addRow("Nudge Y:", self.nudge_y)
        nudge_form.addRow("Nudge Z:", self.nudge_z)
        tools_layout.addLayout(nudge_form)
        tools_layout.addWidget(self._button(
            "↵ Apply Nudge", self._apply_nudge,
            "Apply the X/Y/Z nudge values to all selected cameras (target-frame units).",
        ))

        trust_row = QHBoxLayout()
        self.trust_combo = QComboBox()
        self.trust_combo.addItems(["trusted", "uncertain", "bad"])
        self.trust_combo.setToolTip("Trust level to assign to the selected cameras.")
        trust_row.addWidget(self.trust_combo)
        trust_row.addWidget(self._button(
            "Set Trust", self._set_selected_trust,
            "Apply the chosen trust level (trusted / uncertain / bad) to all selected cameras.",
        ))
        tools_layout.addLayout(trust_row)

        include_row = QHBoxLayout()
        self.include_selected_cb = QCheckBox("Included in lockbox")
        self.include_selected_cb.setChecked(True)
        self.include_selected_cb.setToolTip("Whether selected cameras will be included in the saved lockbox prior.")
        include_row.addWidget(self.include_selected_cb)
        include_row.addWidget(self._button(
            "Set Inclusion", self._set_selected_inclusion,
            "Apply the current include/exclude state to all selected cameras.",
        ))
        tools_layout.addLayout(include_row)

        plane_row = QHBoxLayout()
        self.plane_edit = QLineEdit()
        self.plane_edit.setPlaceholderText("plane_id, e.g. top_ring")
        self.plane_edit.setToolTip(
            "Name for the plane group (e.g. 'top_ring'). Cameras in the same group share a "
            "fitted plane for projection and offset-matching operations."
        )
        plane_row.addWidget(self.plane_edit)
        plane_row.addWidget(self._button(
            "Assign Plane", self._assign_plane_group,
            "Assign selected cameras to the plane group named above.",
        ))
        tools_layout.addLayout(plane_row)
        tools_layout.addWidget(self._button(
            "→ Project to Plane", self._project_selected_to_plane,
            "Project selected cameras orthogonally onto the fitted plane of their common plane group.",
        ))

        tools_layout.addWidget(self._button(
            "Snap Radius", self._snap_radius_to_trusted,
            "Snap selected cameras to the median radial distance of checked reference cameras from the object centre.",
        ))

        # ── Radial move relative to object centre ──────────────────
        radial_form = QFormLayout()
        self.radial_delta_spin = self._make_float_spin()
        self.radial_delta_spin.setPrefix("+/- ")
        self.radial_delta_spin.setToolTip("Positive moves outward from object centre, negative inward.")
        radial_form.addRow("Radial delta:", self.radial_delta_spin)
        tools_layout.addLayout(radial_form)
        tools_layout.addWidget(self._button(
            "Apply Radial", self._apply_radial_move,
            "Move selected cameras outward (+) or inward (−) from the object centre by the radial delta amount.",
        ))

        # ── Match signed plane offset ──────────────────────────────
        tools_layout.addWidget(self._button(
            "Match Plane Offset", self._match_plane_offset_to_references,
            "Shift selected cameras along the plane normal to match the median signed offset of "
            "reference cameras in the same plane group.",
        ))

        # ── Reference camera selector ──────────────────────────────
        ref_label = QLabel("Reference cameras (for radius/offset stats):")
        tools_layout.addWidget(ref_label)
        self.reference_list = QListWidget()
        self.reference_list.setToolTip(
            "Cameras checked here are used as reference cameras for snap and offset operations.\n"
            "This is distinct from trust flags and from table selection."
        )
        for name in self.states:
            item = QListWidgetItem(name)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.reference_list.addItem(item)
        tools_layout.addWidget(self.reference_list)

        undo_redo_row = QHBoxLayout()
        undo_redo_row.addWidget(self._button("↩ Undo", self._undo, "Undo the last edit operation."))
        undo_redo_row.addWidget(self._button("↪ Redo", self._redo, "Redo the last undone operation."))
        tools_layout.addLayout(undo_redo_row)

        reset_row = QHBoxLayout()
        reset_row.addWidget(self._button(
            "Reset Selected", self._reset_selected,
            "Reset selected cameras to their original source camset positions.",
        ))
        reset_row.addWidget(self._button(
            "Reset All", self._reset_all,
            "Reset all cameras to their original source camset positions. Preserves trust and plane annotations.",
        ))
        tools_layout.addLayout(reset_row)
        tools_layout.addStretch()

        action_row = QHBoxLayout()
        action_row.addWidget(self._button(
            "✓ Apply", self._save_and_apply,
            "Save the edited lockbox copy and apply it as the Phase 3 lockbox prior source.",
        ))
        action_row.addWidget(self._button(
            "✗ Discard", self.reject,
            "Discard all edits and close the editor without saving.",
        ))
        tools_layout.addLayout(action_row)
        splitter.addWidget(tools)
        splitter.setSizes([420, 470, 290])

    def _make_float_spin(self) -> QDoubleSpinBox:
        """Return a coordinate spin box using target-frame length units."""
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(-1e6, 1e6)
        spin.setSingleStep(0.001)
        return spin

    def _button(self, text: str, callback, tip: str = "") -> QPushButton:
        """Return a button connected to *callback* with an optional 3-second hover tooltip."""
        button = QPushButton(text)
        button.clicked.connect(callback)
        if tip:
            f = _DelayedTooltipFilter(button, tip)
            button.installEventFilter(f)
        return button

    def _selected_names(self) -> list[str]:
        """Return selected camera names from the active frontend."""
        if self._o3d_select_checks:
            return [name for name, cb in self._o3d_select_checks.items() if cb.checked]
        names: list[str] = []
        for item in self.table.selectedItems():
            row = item.row()
            name_item = self.table.item(row, 2)
            if name_item is not None and name_item.text() not in names:
                names.append(name_item.text())
        return names

    def _snapshot(self) -> dict[str, dict]:
        """Capture edited centres, extrinsics, and intrinsics for undo/redo."""
        return {
            name: {
                'center': state.edited_center.copy(),
                'extrinsic': self.working_camset[name].extrinsic.copy(),
                'intrinsic': self.working_camset[name].intrinsic.copy(),
            }
            for name, state in self.states.items()
        }

    def _push_undo(self) -> None:
        """Record a state before a mutating operation."""
        self.undo_stack.append(self._snapshot())
        self.redo_stack.clear()

    def _restore_snapshot(self, snapshot: dict[str, dict]) -> None:
        """Restore edited centres, extrinsics, and intrinsics."""
        for name, snap in snapshot.items():
            cam = self.working_camset[name]
            cam.set_extrinsic(snap['extrinsic'].copy())  # refreshes derived pose state
            cam.intrinsic = snap['intrinsic'].copy()  # restores intrinsic (viewcone shape)
            cam._update_state()                        # recompute state after intrinsic restore
            self.states[name].edited_center = snap['center'].copy()
        self._refresh_table()
        self._refresh_preview()
        self._refresh_status()

    def _set_camera_center(self, name: str, centre: np.ndarray) -> None:
        """Update row state and working camset while preserving source rotation."""
        state = self.states[name]
        state.edited_center = np.asarray(centre, dtype=float).reshape(3)
        source_ext = self.source_camset[name].extrinsic
        self.working_camset[name].set_extrinsic(
            extrinsic_from_center_preserving_rotation(source_ext, state.edited_center)
        )

    def _refresh_table(self) -> None:
        """Populate the camera table from current state."""
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.states))
        for row, name in enumerate(self.states):
            state = self.states[name]
            delta_norm = float(np.linalg.norm(state.edited_center - state.original_center))
            selected = QTableWidgetItem("")
            selected.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 0, selected)

            included = QTableWidgetItem("yes" if state.included_in_lockbox else "no")
            self.table.setItem(row, 1, included)
            self.table.setItem(row, 2, QTableWidgetItem(name))
            self.table.setItem(row, 3, QTableWidgetItem(state.trust))
            self.table.setItem(row, 4, QTableWidgetItem("yes" if delta_norm > 1e-9 else "no"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{delta_norm:.6g}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{radius_from_center(state.edited_center, self.object_centre):.6g}"))
            self.table.setItem(row, 7, QTableWidgetItem(state.plane_group))
            ref_item = QTableWidgetItem("yes" if state.is_reference else "no")
            self.table.setItem(row, 8, ref_item)
        self.table.blockSignals(False)

    def _refresh_preview(self) -> None:
        """Refresh textual diagnostics and the optional matplotlib fallback preview."""
        lines = [f"Centre definition: {self.centre_definition['label']} ({self.centre_definition['mode']}) {self.centre_definition['point']}"]
        for name, state in self.states.items():
            delta = state.edited_center - state.original_center
            lines.append(f"{name}: source {state.original_center.tolist()} -> edited {state.edited_center.tolist()} | delta={np.linalg.norm(delta):.6g}")
        self.preview_summary.setPlainText("\n".join(lines))

        # Native Open3D frontend still needs refresh even when matplotlib is absent.
        if not _MATPLOTLIB_OK or self.figure is None or self.canvas is None:
            if self._o3d_scene_widget is not None:
                self._refresh_open3d_native_view()
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")
        originals = np.array([state.original_center for state in self.states.values()], dtype=float)
        edited = np.array([state.edited_center for state in self.states.values()], dtype=float)
        ax.scatter(originals[:, 0], originals[:, 1], originals[:, 2], marker="x", alpha=0.35, label="Original source camset")
        ax.scatter(edited[:, 0], edited[:, 1], edited[:, 2], marker="o", label="Edited lockbox copy")
        ax.scatter([self.object_centre[0]], [self.object_centre[1]], [self.object_centre[2]], marker="*", s=80, label="Object centre")
        for name, src, dst in zip(self.states.keys(), originals, edited):
            ax.plot([src[0], dst[0]], [src[1], dst[1]], [src[2], dst[2]], color="grey", alpha=0.55)
            ax.text(dst[0], dst[1], dst[2], name, fontsize=8)
        ax.set_xlabel("X target frame")
        ax.set_ylabel("Y target frame")
        ax.set_zlabel("Z target frame")
        ax.legend(loc="best")
        self.figure.tight_layout()
        self.canvas.draw_idle()
        if self._o3d_scene_widget is not None:
            self._refresh_open3d_native_view()

    def _pv_polydata_to_o3d_lineset(self, pv_mesh) -> object:
        """Convert a PyVista PolyData triangle mesh to an Open3D LineSet (wireframe).

        PyVista stores faces as [3, v0, v1, v2] per triangle. Open3D LineSet needs
        edge pairs and vertices. This extracts all triangle edges uniquely.
        """
        verts = np.asarray(pv_mesh.points, dtype=np.float64)
        # pv.PolyData faces are stored as a flat array: [3, v0, v1, v2, 3, v0, v1, v2, ...]
        faces_arr = np.asarray(pv_mesh.faces)
        if faces_arr.ndim == 1:
            # Strip the leading cell-size entries and reshape to (n_tri, 3)
            n_tri = len(faces_arr) // 4
            triangles = faces_arr.reshape(n_tri, 4)[:, 1:]
        else:
            triangles = faces_arr[:, 1:]

        # Build edge list from triangle edges (each triangle has 3 edges)
        edges = set()
        for tri in triangles:
            for i in range(3):
                e = tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
                edges.add(e)
        edge_arr = np.array(list(edges), dtype=np.int32)

        ls = _o3d.geometry.LineSet()
        ls.points = _o3d.utility.Vector3dVector(verts)
        ls.lines = _o3d.utility.Vector2iVector(edge_arr)
        return ls

    def _build_open3d_native_window(self) -> None:
        """Build the single-window Open3D-native editor frontend.

        Layout uses CollapsableVert sections so the panel height stays
        manageable regardless of camera count. The nested ScrollableVert
        for the camera list has been removed — nesting two ScrollableVert
        containers causes the Open3D layout engine to mis-position children.
        Unicode symbol characters (arrows, ticks) are intentionally absent
        because Open3D's built-in font does not include them.
        """
        app = _o3d_gui.Application.instance
        app.initialize()
        self._o3d_window = app.create_window("Phase 3 Lockbox Prior Editor", 1600, 980)
        self._o3d_window.set_on_layout(self._on_open3d_layout)
        self._o3d_window.set_on_close(self._on_open3d_close)

        self._o3d_scene_widget = _o3d_gui.SceneWidget()
        self._o3d_scene_widget.scene = _o3d_rendering.Open3DScene(self._o3d_window.renderer)
        self._o3d_scene_widget.set_on_mouse(self._on_open3d_mouse)

        em = self._o3d_window.theme.font_size
        margins = _o3d_gui.Margins(0.4 * em, 0.25 * em, 0.4 * em, 0.25 * em)
        tight = _o3d_gui.Margins(0, 0, 0, 0)
        self._o3d_panel = _o3d_gui.ScrollableVert(0.2 * em, margins)
        self._o3d_panel.preferred_width = 34 * em

        # ── Banner + status (always visible) ──────────────────────────
        self._o3d_panel.add_child(_o3d_gui.Label(
            "Editing lockbox prior centres only — source camset unchanged."
        ))
        self._o3d_status_label = _o3d_gui.Label("No cameras selected")
        self._o3d_panel.add_child(self._o3d_status_label)

        # ── Camera list (S = selected, R = reference) ──────────────────
        # No nested ScrollableVert here — that causes layout overlap in Open3D.
        cam_sect = _o3d_gui.CollapsableVert("Cameras  (S = select, R = reference)", 0.15 * em, margins)
        cam_sect.set_is_open(True)
        self._o3d_camera_info_labels = {}
        self._o3d_select_checks = {}
        self._o3d_ref_checks = {}
        for name in self.states:
            cam_row = _o3d_gui.Horiz(0.3 * em, tight)
            sel_cb = _o3d_gui.Checkbox("S")
            sel_cb.tooltip = "Select (S) this camera for editing operations (nudge, trust, snap, etc.)."
            ref_cb = _o3d_gui.Checkbox("R")
            ref_cb.tooltip = "Mark as Reference (R) for snap-radius and match-offset operations."
            info = _o3d_gui.Label(name)
            sel_cb.set_on_checked(lambda checked, n=name: self._on_o3d_select_changed(n, checked))
            ref_cb.set_on_checked(lambda checked, n=name: self._on_o3d_ref_changed(n, checked))
            self._o3d_select_checks[name] = sel_cb
            self._o3d_ref_checks[name] = ref_cb
            self._o3d_camera_info_labels[name] = info
            cam_row.add_child(sel_cb)
            cam_row.add_child(ref_cb)
            cam_row.add_child(info)
            cam_sect.add_child(cam_row)
        self._o3d_panel.add_child(cam_sect)

        # ── Nudge ─────────────────────────────────────────────────────
        nudge_sect = _o3d_gui.CollapsableVert("Nudge", 0.15 * em, margins)
        nudge_sect.set_is_open(True)

        self._o3d_nudge_x = _o3d_gui.NumberEdit(_o3d_gui.NumberEdit.DOUBLE)
        self._o3d_nudge_y = _o3d_gui.NumberEdit(_o3d_gui.NumberEdit.DOUBLE)
        self._o3d_nudge_z = _o3d_gui.NumberEdit(_o3d_gui.NumberEdit.DOUBLE)
        for w in (self._o3d_nudge_x, self._o3d_nudge_y, self._o3d_nudge_z):
            w.set_limits(-1e6, 1e6)
            w.decimal_precision = 6
            w.set_preferred_width(7 * em)
        self._o3d_nudge_x.tooltip = "X nudge value (target-frame units)."
        self._o3d_nudge_y.tooltip = "Y nudge value (target-frame units)."
        self._o3d_nudge_z.tooltip = "Z nudge value (target-frame units)."
        nudge_xyz_row = _o3d_gui.Horiz(0.3 * em, tight)
        nudge_xyz_row.add_child(_o3d_gui.Label("X"))
        nudge_xyz_row.add_child(self._o3d_nudge_x)
        nudge_xyz_row.add_child(_o3d_gui.Label("Y"))
        nudge_xyz_row.add_child(self._o3d_nudge_y)
        nudge_xyz_row.add_child(_o3d_gui.Label("Z"))
        nudge_xyz_row.add_child(self._o3d_nudge_z)
        nudge_sect.add_child(nudge_xyz_row)
        btn_nudge = _o3d_gui.Button("Apply Nudge")
        btn_nudge.tooltip = "Apply the X/Y/Z nudge values to all selected cameras (target-frame units)."
        btn_nudge.set_on_clicked(self._on_o3d_apply_nudge)
        nudge_sect.add_child(btn_nudge)
        self._o3d_panel.add_child(nudge_sect)

        # ── Trust / Inclusion / Plane ──────────────────────────────────
        tip_sect = _o3d_gui.CollapsableVert("Trust / Inclusion / Plane", 0.15 * em, margins)
        tip_sect.set_is_open(True)

        self._o3d_trust_combo = _o3d_gui.Combobox()
        for item in ["trusted", "uncertain", "bad"]:
            self._o3d_trust_combo.add_item(item)
        self._o3d_trust_combo.tooltip = (
            "Trust level for the selected cameras:\n"
            "  trusted   = high-confidence pose; the lockbox prior is applied with a tight weight.\n"
            "  uncertain = pose may have drifted; prior is applied with a looser weight.\n"
            "  bad       = camera is excluded from optimisation entirely."
        )
        btn_trust = _o3d_gui.Button("Set Trust")
        btn_trust.tooltip = (
            "Apply the chosen trust level to all currently selected (S) cameras.\n"
            "Trust controls how strongly the lockbox prior constrains each camera during "
            "bundle adjustment — it does not hard-lock the pose."
        )
        btn_trust.set_on_clicked(self._on_o3d_set_trust)
        trust_row = _o3d_gui.Horiz(0.3 * em, tight)
        trust_row.add_child(self._o3d_trust_combo)
        trust_row.add_child(btn_trust)
        tip_sect.add_child(trust_row)

        self._o3d_include_cb = _o3d_gui.Checkbox("In lockbox")
        self._o3d_include_cb.checked = True
        self._o3d_include_cb.tooltip = (
            "Included cameras have a lockbox prior applied during bundle adjustment — "
            "their pose is softly constrained to stay near the centre you set here.\n"
            "Excluded cameras are optimised freely with no positional prior."
        )
        btn_incl = _o3d_gui.Button("Set Inclusion")
        btn_incl.tooltip = (
            "Apply the current 'In lockbox' tick state to all selected (S) cameras. "
            "Tick the checkbox first, then click this button to include or exclude a group at once."
        )
        btn_incl.set_on_clicked(self._on_o3d_set_inclusion)
        incl_row = _o3d_gui.Horiz(0.3 * em, tight)
        incl_row.add_child(self._o3d_include_cb)
        incl_row.add_child(btn_incl)
        tip_sect.add_child(incl_row)

        self._o3d_plane_edit = _o3d_gui.TextEdit()
        self._o3d_plane_edit.placeholder_text = "plane_id, e.g. top_ring"
        self._o3d_plane_edit.tooltip = (
            "Name for the plane group (e.g. 'top_ring'). Cameras in the same group "
            "share a fitted plane for projection and offset-matching operations."
        )
        btn_plane = _o3d_gui.Button("Assign Plane")
        btn_plane.tooltip = (
            "Select 2+ cameras (S and/or R), type a group name (e.g. 'top_ring'), "
            "then click to assign them to that plane group.\n"
            "Cameras in the same group share a least-squares fitted plane, which is used "
            "by 'Project to Plane' and 'Match Plane Offset'."
        )
        btn_plane.set_on_clicked(self._on_o3d_assign_plane)
        plane_row = _o3d_gui.Horiz(0.3 * em, tight)
        plane_row.add_child(self._o3d_plane_edit)
        plane_row.add_child(btn_plane)
        tip_sect.add_child(plane_row)

        btn_proj = _o3d_gui.Button("Project to Plane")
        btn_proj.tooltip = "Project selected cameras orthogonally onto the fitted plane of their common plane group."
        btn_proj.set_on_clicked(self._on_o3d_project_to_plane)
        tip_sect.add_child(btn_proj)

        btn_mpo = _o3d_gui.Button("Match Plane Offset")
        btn_mpo.tooltip = (
            "Shift selected cameras along the plane normal to match the median signed "
            "offset of reference cameras in the same plane group."
        )
        btn_mpo.set_on_clicked(self._on_o3d_match_plane_offset)
        tip_sect.add_child(btn_mpo)
        self._o3d_panel.add_child(tip_sect)

        # ── Radius / Radial ────────────────────────────────────────────
        rad_sect = _o3d_gui.CollapsableVert("Radius / Radial", 0.15 * em, margins)
        rad_sect.set_is_open(True)

        btn_snap = _o3d_gui.Button("Snap Radius")
        btn_snap.tooltip = (
            "Snap selected cameras to the median radial distance of checked reference "
            "cameras from the object centre."
        )
        btn_snap.set_on_clicked(self._on_o3d_snap_radius)
        rad_sect.add_child(btn_snap)

        self._o3d_radial_delta = _o3d_gui.NumberEdit(_o3d_gui.NumberEdit.DOUBLE)
        self._o3d_radial_delta.set_limits(-1e6, 1e6)
        self._o3d_radial_delta.decimal_precision = 6
        self._o3d_radial_delta.set_preferred_width(9 * em)
        self._o3d_radial_delta.tooltip = "Positive = outward from object centre; negative = inward."
        btn_rad = _o3d_gui.Button("Apply Radial")
        btn_rad.tooltip = "Move selected cameras outward (+) or inward (-) from the object centre by the radial delta."
        btn_rad.set_on_clicked(self._on_o3d_apply_radial)
        radial_row = _o3d_gui.Horiz(0.3 * em, tight)
        radial_row.add_child(_o3d_gui.Label("Delta"))
        radial_row.add_child(self._o3d_radial_delta)
        radial_row.add_child(btn_rad)
        rad_sect.add_child(radial_row)
        self._o3d_panel.add_child(rad_sect)

        # ── Align to References ────────────────────────────────────────
        align_sect = _o3d_gui.CollapsableVert("Align to References", 0.15 * em, margins)
        align_sect.set_is_open(True)

        btn_reorient = _o3d_gui.Button("Orient to Centre")  # Feature 1
        btn_reorient.tooltip = (
            "Rotate selected (S) cameras so they face the target-space origin "
            "(the object centre), adopting the median roll of the reference (R) cameras.\n"
            "Position is unchanged; only the rotation component is updated."
        )
        btn_reorient.set_on_clicked(self._reorient_selected_to_centre)

        btn_match_K = _o3d_gui.Button("Match Intrinsics")  # Feature 2
        btn_match_K.tooltip = (
            "Copy the element-wise median intrinsic K matrix (and resolution) "
            "of the reference (R) cameras onto all selected (S) cameras.\n"
            "This aligns the shape of the green viewcone polygons to the references."
        )
        btn_match_K.set_on_clicked(self._match_selected_intrinsics)

        align_row = _o3d_gui.Horiz(0.25 * em, _o3d_gui.Margins(0))
        align_row.add_child(btn_reorient)
        align_row.add_child(btn_match_K)
        align_sect.add_child(align_row)
        self._o3d_panel.add_child(align_sect)

        # ── History / Finish ───────────────────────────────────────────
        hist_sect = _o3d_gui.CollapsableVert("History / Finish", 0.15 * em, margins)
        hist_sect.set_is_open(True)

        b_undo = _o3d_gui.Button("Undo")
        b_undo.tooltip = "Undo the last edit operation."
        b_undo.set_on_clicked(self._on_o3d_undo)
        b_redo = _o3d_gui.Button("Redo")
        b_redo.tooltip = "Redo the last undone operation."
        b_redo.set_on_clicked(self._on_o3d_redo)
        undo_redo_row = _o3d_gui.Horiz(0.3 * em, tight)
        undo_redo_row.add_child(b_undo)
        undo_redo_row.add_child(b_redo)
        hist_sect.add_child(undo_redo_row)

        b_reset_sel = _o3d_gui.Button("Reset Selected")
        b_reset_sel.tooltip = "Reset selected cameras to their original source camset positions."
        b_reset_sel.set_on_clicked(self._on_o3d_reset_selected)
        b_reset_all = _o3d_gui.Button("Reset All")
        b_reset_all.tooltip = "Reset all cameras to their original source camset positions. Preserves trust and plane annotations."
        b_reset_all.set_on_clicked(self._on_o3d_reset_all)
        reset_row = _o3d_gui.Horiz(0.3 * em, tight)
        reset_row.add_child(b_reset_sel)
        reset_row.add_child(b_reset_all)
        hist_sect.add_child(reset_row)

        b_apply = _o3d_gui.Button("Apply")
        b_apply.tooltip = "Save the edited lockbox copy and apply it as the Phase 3 lockbox prior source."
        b_apply.set_on_clicked(self._on_o3d_apply_and_close)
        b_discard = _o3d_gui.Button("Discard")
        b_discard.tooltip = "Discard all edits and close the editor without saving."
        b_discard.set_on_clicked(self._on_o3d_discard_and_close)
        apply_discard_row = _o3d_gui.Horiz(0.3 * em, tight)
        apply_discard_row.add_child(b_apply)
        apply_discard_row.add_child(b_discard)
        hist_sect.add_child(apply_discard_row)
        self._o3d_panel.add_child(hist_sect)

        # ── View Settings (collapsed by default — saves vertical space) ─
        vis_sect = _o3d_gui.CollapsableVert("View Settings", 0.15 * em, margins)
        vis_sect.set_is_open(False)

        self._o3d_view_mode_combo = _o3d_gui.Combobox()
        for item in ["Planetary", "Arcball", "Fly", "Model", "Sun", "Environment"]:
            self._o3d_view_mode_combo.add_item(item)
        self._o3d_view_mode_combo.selected_text = self._o3d_visual_settings['view_mode']
        self._o3d_view_mode_combo.set_on_selection_changed(self._on_o3d_view_mode_changed)
        self._o3d_view_mode_combo.tooltip = (
            "Mouse interaction mode: Planetary = sphere orbit, Arcball = camera orbit, "
            "Fly = first-person, Model = rotate object."
        )
        vis_sect.add_child(_o3d_gui.Label("Mouse / orbit mode"))
        vis_sect.add_child(self._o3d_view_mode_combo)

        self._o3d_pick_cb = _o3d_gui.Checkbox("Enable camera box picking")
        self._o3d_pick_cb.checked = bool(self._o3d_visual_settings['box_picking'])
        self._o3d_pick_cb.set_on_checked(self._on_o3d_box_picking_changed)
        self._o3d_pick_cb.tooltip = (
            "When enabled: left-click a camera box to toggle it as Selected (S); "
            "right-click a camera box to toggle it as Reference (R). "
            "Disable to use all mouse buttons for orbiting without accidentally selecting cameras."
        )
        vis_sect.add_child(self._o3d_pick_cb)

        self._o3d_ground_cb = _o3d_gui.Checkbox("Show ground plane")
        self._o3d_ground_cb.checked = bool(self._o3d_visual_settings['show_ground'])
        self._o3d_ground_cb.set_on_checked(self._on_o3d_show_ground_changed)
        self._o3d_ground_cb.tooltip = "Show a reference ground-plane grid in the 3D view."
        vis_sect.add_child(self._o3d_ground_cb)

        self._o3d_ground_combo = _o3d_gui.Combobox()
        for item in ["XZ floor", "XY backplane", "YZ sideplane"]:
            self._o3d_ground_combo.add_item(item)
        self._o3d_ground_combo.selected_text = self._o3d_visual_settings['ground_plane']
        self._o3d_ground_combo.set_on_selection_changed(self._on_o3d_ground_plane_changed)
        self._o3d_ground_combo.tooltip = "Which plane to use as the ground reference grid."
        vis_sect.add_child(self._o3d_ground_combo)

        self._o3d_skybox_cb = _o3d_gui.Checkbox("Show skybox")
        self._o3d_skybox_cb.checked = bool(self._o3d_visual_settings['show_skybox'])
        self._o3d_skybox_cb.set_on_checked(self._on_o3d_show_skybox_changed)
        self._o3d_skybox_cb.tooltip = "Show the environment image (skybox) in the scene background."
        vis_sect.add_child(self._o3d_skybox_cb)

        self._o3d_axes_cb = _o3d_gui.Checkbox("Show axes")
        self._o3d_axes_cb.checked = bool(self._o3d_visual_settings['show_axes'])
        self._o3d_axes_cb.set_on_checked(self._on_o3d_show_axes_changed)
        self._o3d_axes_cb.tooltip = "Show coordinate frame axes at the world origin."
        vis_sect.add_child(self._o3d_axes_cb)

        self._o3d_lockbox_cb = _o3d_gui.Checkbox("Show lockbox boxes")
        self._o3d_lockbox_cb.checked = bool(self._o3d_visual_settings['show_lockbox'])
        self._o3d_lockbox_cb.set_on_checked(self._on_o3d_show_lockbox_changed)
        self._o3d_lockbox_cb.tooltip = "Overlay the lockbox half-width bounding boxes for each included camera."
        vis_sect.add_child(self._o3d_lockbox_cb)

        self._o3d_background_combo = _o3d_gui.Combobox()
        for item in ["Dark calibration", "Neutral grey", "Light studio"]:
            self._o3d_background_combo.add_item(item)
        self._o3d_background_combo.selected_text = self._o3d_visual_settings['background']
        self._o3d_background_combo.set_on_selection_changed(self._on_o3d_background_changed)
        self._o3d_background_combo.tooltip = "Background colour preset for the 3D view."
        vis_sect.add_child(_o3d_gui.Label("Background"))
        vis_sect.add_child(self._o3d_background_combo)

        self._o3d_lighting_combo = _o3d_gui.Combobox()
        for item in ["Medium shadows", "Soft shadows", "Hard shadows", "Dark shadows", "No shadows"]:
            self._o3d_lighting_combo.add_item(item)
        self._o3d_lighting_combo.selected_text = self._o3d_visual_settings['lighting']
        self._o3d_lighting_combo.set_on_selection_changed(self._on_o3d_lighting_changed)
        self._o3d_lighting_combo.tooltip = "Lighting and shadow quality preset for the 3D view."
        vis_sect.add_child(_o3d_gui.Label("Lighting"))
        vis_sect.add_child(self._o3d_lighting_combo)

        vis_sect.add_child(_o3d_gui.Label(
            "Box picking: enable above, then left-click (S) or right-click (R) a camera box."
        ))
        self._o3d_panel.add_child(vis_sect)

        self._o3d_window.add_child(self._o3d_panel)
        self._o3d_window.add_child(self._o3d_scene_widget)
        self._apply_open3d_visual_settings()
        self._refresh_status()
        self._refresh_open3d_native_view(reset_camera=True)

    def _on_open3d_layout(self, ctx) -> None:
        em = ctx.theme.font_size
        panel_width = int(34 * em)
        rect = self._o3d_window.content_rect
        self._o3d_panel.frame = _o3d_gui.Rect(rect.x, rect.y, panel_width, rect.height)
        x = self._o3d_panel.frame.get_right()
        self._o3d_scene_widget.frame = _o3d_gui.Rect(x, rect.y, rect.get_right() - x, rect.height)

    def _on_open3d_close(self) -> bool:
        if self.result is None:
            self._o3d_return_code = QDialog.DialogCode.Rejected
        return True

    def _clear_o3d_labels(self) -> None:
        if self._o3d_scene_widget is None:
            return
        for label_id in self._o3d_label_ids:
            try:
                self._o3d_scene_widget.remove_3d_label(label_id)
            except Exception:
                pass
        self._o3d_label_ids.clear()

    def _camera_frame_axes(self, cam) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return camera position and an orthonormal right/up/forward frame."""
        pos = np.asarray(cam.position, dtype=float)
        forward = np.asarray(cam.view, dtype=float)
        f_norm = float(np.linalg.norm(forward))
        forward = forward / (f_norm if f_norm > 1e-12 else 1.0)
        right = np.asarray(cam.u_axis, dtype=float)
        r_norm = float(np.linalg.norm(right))
        if r_norm <= 1e-12:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            right = right / r_norm
        up = np.cross(forward, right)
        u_norm = float(np.linalg.norm(up))
        if u_norm <= 1e-12:
            up = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            up = up / u_norm
        right = np.cross(up, forward)
        right = right / max(float(np.linalg.norm(right)), 1e-12)
        return pos, right, up, forward

    def _plane_group_colour(self, plane_group: str) -> list[float]:
        palette = [
            [0.93, 0.33, 0.31],
            [0.22, 0.62, 0.89],
            [0.26, 0.74, 0.44],
            [0.92, 0.67, 0.18],
            [0.67, 0.44, 0.89],
            [0.17, 0.76, 0.75],
        ]
        plane_ids = sorted({state.plane_group for state in self.states.values() if state.plane_group})
        if plane_group not in plane_ids:
            return [0.35, 0.35, 0.35]
        return palette[plane_ids.index(plane_group) % len(palette)]

    def _create_oriented_box_lineset(
        self,
        center: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        forward: np.ndarray,
        half_extents: np.ndarray,
        color: list[float],
        name: str,
        register_pick_target: bool = True,
    ) -> object:
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append(
                        center
                        + sx * half_extents[0] * right
                        + sy * half_extents[1] * up
                        + sz * half_extents[2] * forward
                    )
        points = np.asarray(corners, dtype=np.float64)
        lines = np.asarray([
            [0, 1], [0, 2], [0, 4],
            [1, 3], [1, 5],
            [2, 3], [2, 6],
            [3, 7],
            [4, 5], [4, 6],
            [5, 7],
            [6, 7],
        ], dtype=np.int32)
        ls = _o3d.geometry.LineSet()
        ls.points = _o3d.utility.Vector3dVector(points)
        ls.lines = _o3d.utility.Vector2iVector(lines)
        ls.paint_uniform_color(color)
        if register_pick_target:
            self._o3d_box_targets[name] = {
            'center': np.asarray(center, dtype=float),
            'right': np.asarray(right, dtype=float),
            'up': np.asarray(up, dtype=float),
            'forward': np.asarray(forward, dtype=float),
            'half_extents': np.asarray(half_extents, dtype=float),
            }
        return ls

    def _add_box_face_labels(self, name: str, center: np.ndarray, right: np.ndarray, up: np.ndarray, forward: np.ndarray, half_extents: np.ndarray) -> None:
        if self._o3d_scene_widget is None:
            return
        selected = bool(self._o3d_select_checks.get(name).checked) if name in self._o3d_select_checks else False
        reference = bool(self.states[name].is_reference)
        if not selected and not reference:
            return
        face_text = 'S' if selected else 'R'  # S and R are mutually exclusive; joint state is impossible
        offsets = [
            right * half_extents[0], -right * half_extents[0],
            up * half_extents[1], -up * half_extents[1],
            forward * half_extents[2], -forward * half_extents[2],
        ]
        for offset in offsets:
            label_id = self._o3d_scene_widget.add_3d_label((center + offset).tolist(), face_text)
            self._o3d_label_ids.append(label_id)

    def _apply_open3d_visual_settings(self) -> None:
        if self._o3d_scene_widget is None:
            return
        scene = self._o3d_scene_widget.scene
        bg_map = {
            'Dark calibration': [0.08, 0.10, 0.16, 1.0],
            'Neutral grey': [0.22, 0.22, 0.24, 1.0],
            'Light studio': [0.82, 0.84, 0.88, 1.0],
        }
        plane_map = {
            'XZ floor': _o3d_rendering.Scene.GroundPlane.XZ,
            'XY backplane': _o3d_rendering.Scene.GroundPlane.XY,
            'YZ sideplane': _o3d_rendering.Scene.GroundPlane.YZ,
        }
        lighting_map = {
            'Medium shadows': _o3d_rendering.Open3DScene.LightingProfile.MED_SHADOWS,
            'Soft shadows': _o3d_rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
            'Hard shadows': _o3d_rendering.Open3DScene.LightingProfile.HARD_SHADOWS,
            'Dark shadows': _o3d_rendering.Open3DScene.LightingProfile.DARK_SHADOWS,
            'No shadows': _o3d_rendering.Open3DScene.LightingProfile.NO_SHADOWS,
        }
        view_map = {
            'Planetary': _o3d_gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE,
            'Arcball': _o3d_gui.SceneWidget.Controls.ROTATE_CAMERA,
            'Fly': _o3d_gui.SceneWidget.Controls.FLY,
            'Model': _o3d_gui.SceneWidget.Controls.ROTATE_MODEL,
            'Sun': _o3d_gui.SceneWidget.Controls.ROTATE_SUN,
            'Environment': _o3d_gui.SceneWidget.Controls.ROTATE_IBL,
        }
        scene.set_background(bg_map[self._o3d_visual_settings['background']])
        scene.show_skybox(bool(self._o3d_visual_settings['show_skybox']))
        scene.show_ground_plane(bool(self._o3d_visual_settings['show_ground']), plane_map[self._o3d_visual_settings['ground_plane']])
        scene.show_axes(bool(self._o3d_visual_settings['show_axes']))
        scene.set_lighting(lighting_map[self._o3d_visual_settings['lighting']], np.asarray([0.577, -0.577, -0.577], dtype=np.float32))
        self._o3d_scene_widget.set_view_controls(view_map[self._o3d_visual_settings['view_mode']])

    def _refresh_open3d_native_view(self, reset_camera: bool = False) -> None:
        if self._o3d_scene_widget is None:
            return
        scene = self._o3d_scene_widget.scene
        scene.clear_geometry()
        self._clear_o3d_labels()
        self._o3d_box_targets.clear()
        self._apply_open3d_visual_settings()

        def _safe_scale(values: list[np.ndarray]) -> float:
            radii = [float(np.linalg.norm(v)) for v in values]
            return max(max(radii, default=1.0) * 0.1, 0.03)

        line_mat = _o3d_rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 2.0
        mesh_mat = _o3d_rendering.MaterialRecord()
        mesh_mat.shader = "defaultLit"

        all_points: list[np.ndarray] = []
        cam_scale = _safe_scale([s.edited_center for s in self.states.values()])
        try:
            source_meshes = self.source_camset.get_camera_meshes(viewcone=None, scale=_safe_scale([s.original_center for s in self.states.values()]))
            for i, mesh in enumerate(source_meshes):
                ls = self._pv_polydata_to_o3d_lineset(mesh)
                ls.paint_uniform_color([0.6, 0.6, 0.6])
                scene.add_geometry(f"source_{i}", ls, line_mat)
        except Exception:
            pass
        try:
            cam_meshes, view_cones = self.working_camset.get_camera_meshes(viewcone=0.15, scale=cam_scale)
            for i, mesh in enumerate(cam_meshes):
                ls = self._pv_polydata_to_o3d_lineset(mesh)
                ls.paint_uniform_color([0.0, 0.0, 0.0])
                scene.add_geometry(f"edited_{i}", ls, line_mat)
            for i, vc in enumerate(view_cones):
                ls = self._pv_polydata_to_o3d_lineset(vc)
                ls.paint_uniform_color([0.0, 0.7, 0.0])
                scene.add_geometry(f"cone_{i}", ls, line_mat)
        except Exception:
            pass
        for i, (name, state) in enumerate(self.states.items()):
            cam = self.working_camset[name]
            pos, right, up, forward = self._camera_frame_axes(cam)
            all_points.append(np.asarray(state.original_center, dtype=float))
            all_points.append(np.asarray(state.edited_center, dtype=float))
            delta = state.edited_center - state.original_center
            if np.linalg.norm(delta) >= 1e-9:
                pts = np.array([state.original_center, state.edited_center], dtype=np.float64)
                ls = _o3d.geometry.LineSet()
                ls.points = _o3d.utility.Vector3dVector(pts)
                ls.lines = _o3d.utility.Vector2iVector([[0, 1]])
                ls.paint_uniform_color([0.8, 0.0, 0.0])
                scene.add_geometry(f"delta_{i}", ls, line_mat)
            if state.plane_group:
                box_color = self._plane_group_colour(state.plane_group)
            elif state.is_reference:
                box_color = [0.95, 0.62, 0.14]
            elif bool(self._o3d_select_checks.get(name).checked) if name in self._o3d_select_checks else False:
                box_color = [0.16, 0.62, 0.96]
            else:
                box_color = [0.35, 0.35, 0.35]
            half_extents = np.array([0.18 * cam_scale, 0.18 * cam_scale, 0.14 * cam_scale], dtype=float)
            box_center = pos - forward * (0.45 * cam_scale)
            selector_box = self._create_oriented_box_lineset(box_center, right, up, forward, half_extents, box_color, name)
            scene.add_geometry(f"selector_box_{i}", selector_box, line_mat)
            self._add_box_face_labels(name, box_center, right, up, forward, half_extents)
            if bool(self._o3d_visual_settings['show_lockbox']) and state.included_in_lockbox:
                t_half = float(self.lockbox_params.get('translation_half_width', 0.1))
                lockbox_half = np.array([t_half, t_half, t_half], dtype=float)
                lockbox = self._create_oriented_box_lineset(pos, right, up, forward, lockbox_half, [0.98, 0.88, 0.24], f"lockbox_{name}", register_pick_target=False)
                scene.add_geometry(f"lockbox_{i}", lockbox, line_mat)

        centre_sphere = _o3d.geometry.TriangleMesh.create_sphere(radius=max(0.01, 0.08 * cam_scale))
        centre_sphere.translate(self.object_centre)
        centre_sphere.paint_uniform_color([0.9, 0.7, 0.0])
        scene.add_geometry("object_centre", centre_sphere, mesh_mat)
        frame = _o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(0.05, 0.4 * cam_scale), origin=[0, 0, 0])
        scene.add_geometry("frame", frame, mesh_mat)
        all_points.append(np.asarray(self.object_centre, dtype=float))
        all_points.append(np.zeros(3, dtype=float))
        if all_points and (reset_camera or not self._o3d_scene_initialised):
            pts = np.vstack(all_points)
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
            bbox = _o3d.geometry.AxisAlignedBoundingBox(mn, mx)
            centre = bbox.get_center()
            extent = max(float(np.max(mx - mn)), 1.0)
            eye = centre + np.array([1.6, -1.5, 1.1]) * extent
            self._o3d_scene_widget.setup_camera(60.0, bbox, centre)
            self._o3d_scene_widget.look_at(centre, eye, [0.0, 0.0, 1.0])
            self._o3d_scene_initialised = True
        self._o3d_window.post_redraw()

    def _sync_o3d_inputs_to_qt(self) -> None:
        if not self._o3d_select_checks:
            return
        self.nudge_x.setValue(float(self._o3d_nudge_x.double_value))
        self.nudge_y.setValue(float(self._o3d_nudge_y.double_value))
        self.nudge_z.setValue(float(self._o3d_nudge_z.double_value))
        self.radial_delta_spin.setValue(float(self._o3d_radial_delta.double_value))
        self.trust_combo.setCurrentText(self._o3d_trust_combo.selected_text)
        self.include_selected_cb.setChecked(bool(self._o3d_include_cb.checked))
        self.plane_edit.setText(self._o3d_plane_edit.text_value)

    def _on_o3d_view_mode_changed(self, text: str, index: int) -> None:
        del index
        self._o3d_visual_settings['view_mode'] = text
        self._apply_open3d_visual_settings()
        self._o3d_window.post_redraw()

    def _on_o3d_box_picking_changed(self, checked: bool) -> None:
        self._o3d_visual_settings['box_picking'] = bool(checked)

    def _on_o3d_show_ground_changed(self, checked: bool) -> None:
        self._o3d_visual_settings['show_ground'] = bool(checked)
        self._apply_open3d_visual_settings()
        self._o3d_window.post_redraw()

    def _on_o3d_ground_plane_changed(self, text: str, index: int) -> None:
        del index
        self._o3d_visual_settings['ground_plane'] = text
        self._apply_open3d_visual_settings()
        self._o3d_window.post_redraw()

    def _on_o3d_show_skybox_changed(self, checked: bool) -> None:
        self._o3d_visual_settings['show_skybox'] = bool(checked)
        self._apply_open3d_visual_settings()
        self._o3d_window.post_redraw()

    def _on_o3d_show_axes_changed(self, checked: bool) -> None:
        self._o3d_visual_settings['show_axes'] = bool(checked)
        self._apply_open3d_visual_settings()
        self._o3d_window.post_redraw()

    def _on_o3d_show_lockbox_changed(self, checked: bool) -> None:
        self._o3d_visual_settings['show_lockbox'] = bool(checked)
        self._refresh_open3d_native_view()

    def _on_o3d_background_changed(self, text: str, index: int) -> None:
        del index
        self._o3d_visual_settings['background'] = text
        self._apply_open3d_visual_settings()
        self._o3d_window.post_redraw()

    def _on_o3d_lighting_changed(self, text: str, index: int) -> None:
        del index
        self._o3d_visual_settings['lighting'] = text
        self._apply_open3d_visual_settings()
        self._o3d_window.post_redraw()

    def _find_clicked_box_name(self, world: np.ndarray) -> str | None:
        best_name = None
        best_dist = None
        for name, target in self._o3d_box_targets.items():
            delta = np.asarray(world, dtype=float) - target['center']
            local = np.array([
                float(delta @ target['right']),
                float(delta @ target['up']),
                float(delta @ target['forward']),
            ])
            half = target['half_extents']
            tol = 0.35 * float(np.max(half))
            if np.all(np.abs(local) <= (half + tol)):
                dist = float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0)))
                if best_dist is None or dist < best_dist:
                    best_name = name
                    best_dist = dist
        return best_name

    def _toggle_box_pick_result(self, name: str, action: str) -> None:
        # action is 'select' (left-click) or 'reference' (right-click)
        # S and R are mutually exclusive — turning one on clears the other.
        if action == 'select':
            cb = self._o3d_select_checks[name]  # toggle the S checkbox for this camera
            new_state = not bool(cb.checked)
            cb.checked = new_state
            if new_state and name in self._o3d_ref_checks:  # S on → clear R
                self._o3d_ref_checks[name].checked = False
                self.states[name].is_reference = False
            self._refresh_status()
            self._refresh_open3d_native_view()
        elif action == 'reference':
            cb = self._o3d_ref_checks[name]  # toggle the R checkbox for this camera
            new_state = not bool(cb.checked)
            cb.checked = new_state
            self.states[name].is_reference = new_state
            if new_state and name in self._o3d_select_checks:  # R on → clear S
                self._o3d_select_checks[name].checked = False
            self._refresh_table()
            self._refresh_status()
            self._refresh_open3d_native_view()

    def _on_open3d_mouse(self, event):
        if not self._o3d_visual_settings['box_picking']:  # box picking disabled — orbit freely
            return _o3d_gui.Widget.EventCallbackResult.IGNORED
        if event.type != _o3d_gui.MouseEvent.Type.BUTTON_DOWN:
            return _o3d_gui.Widget.EventCallbackResult.IGNORED

        # Determine which action to apply based on which mouse button was pressed.
        # Open3D uses a bitmask in event.buttons: LEFT=1, MIDDLE=2, RIGHT=4.
        is_left = bool(event.buttons & _o3d_gui.MouseButton.LEFT)
        is_right = bool(event.buttons & _o3d_gui.MouseButton.RIGHT)
        if not is_left and not is_right:  # middle or unknown button — ignore
            return _o3d_gui.Widget.EventCallbackResult.IGNORED
        action = 'select' if is_left else 'reference'  # left = S, right = R

        def depth_callback(depth_image):
            x = event.x - self._o3d_scene_widget.frame.x
            y = event.y - self._o3d_scene_widget.frame.y
            depth = np.asarray(depth_image)[y, x]
            if depth == 1.0:  # background — nothing was clicked
                return
            world = self._o3d_scene_widget.scene.camera.unproject(
                x, y, depth, self._o3d_scene_widget.frame.width, self._o3d_scene_widget.frame.height
            )
            picked = self._find_clicked_box_name(np.asarray(world, dtype=float))
            if picked is None:
                return
            _o3d_gui.Application.instance.post_to_main_thread(
                self._o3d_window,
                lambda p=picked, a=action: self._toggle_box_pick_result(p, a),
            )

        self._o3d_scene_widget.scene.scene.render_to_depth_image(depth_callback)
        return _o3d_gui.Widget.EventCallbackResult.HANDLED

    def _on_o3d_select_changed(self, name: str, checked: bool) -> None:
        if checked and name in self._o3d_ref_checks:  # S on → clear R (mutually exclusive)
            self._o3d_ref_checks[name].checked = False
            self.states[name].is_reference = False
        self._refresh_status()
        self._refresh_open3d_native_view()

    def _on_o3d_ref_changed(self, name: str, checked: bool) -> None:
        self.states[name].is_reference = bool(checked)
        if checked and name in self._o3d_select_checks:  # R on → clear S (mutually exclusive)
            self._o3d_select_checks[name].checked = False
        self._refresh_table()
        self._refresh_status()
        self._refresh_open3d_native_view()

    # ------------------------------------------------------------------
    # Align-to-references operations
    # ------------------------------------------------------------------

    def _reorient_selected_to_centre(self) -> None:
        # Orient selected cameras to face the object centre with the median
        # roll of the reference cameras, then rebuild their extrinsic matrices.
        ref_names = [n for n, st in self.states.items() if st.is_reference]
        sel_names = [n for n, st in self.states.items()
                     if n in self._o3d_select_checks and bool(self._o3d_select_checks[n].checked)]
        if not ref_names or not sel_names:
            return  # nothing to do — need at least one R and one S

        # The target-space origin is the correct look-at point for all cameras.
        # Reference cameras are used only to extract the median roll convention,
        # NOT to estimate where the target is.
        object_centre = self.object_centre.copy()

        # For each reference camera compute its roll: the angle between its
        # projected up-axis and world-up when viewed along the look-at direction.
        world_up = np.array([0.0, 1.0, 0.0])  # assumed world up vector
        rolls = []
        for rn in ref_names:
            cam = self.working_camset[rn]
            fwd = object_centre - cam.position  # look-at vector toward target origin
            fwd_len = np.linalg.norm(fwd)
            if fwd_len < 1e-9:
                continue  # degenerate — skip
            fwd = fwd / fwd_len  # unit look-at direction
            # Project world_up perpendicular to fwd to get the plane's up reference.
            world_up_perp = world_up - np.dot(world_up, fwd) * fwd
            wup_len = np.linalg.norm(world_up_perp)
            if wup_len < 1e-6:
                continue  # camera pointing straight up/down — roll undefined; skip
            world_up_perp = world_up_perp / wup_len
            # Project the camera's u_axis (camera up) the same way.
            cam_up = cam.u_axis[:3] if cam.u_axis.shape[0] == 4 else cam.u_axis  # (3,)
            cam_up_perp = cam_up - np.dot(cam_up, fwd) * fwd
            cup_len = np.linalg.norm(cam_up_perp)
            if cup_len < 1e-6:
                continue
            cam_up_perp = cam_up_perp / cup_len
            # Roll = signed angle between the two projected up vectors.
            cross_val = np.dot(np.cross(world_up_perp, cam_up_perp), fwd)
            roll = np.arctan2(cross_val, np.dot(world_up_perp, cam_up_perp))
            rolls.append(roll)

        if not rolls:
            return  # all references degenerate
        median_roll = float(np.median(rolls))  # median roll in radians

        self._push_undo()  # capture state before modification

        for sn in sel_names:
            cam = self.working_camset[sn]
            fwd_new = object_centre - cam.position  # look toward the object centre
            fwd_len = np.linalg.norm(fwd_new)
            if fwd_len < 1e-9:
                continue
            fwd_new = fwd_new / fwd_len  # unit forward vector

            # Build the up vector for this camera by rotating world_up around fwd_new
            # by the median roll angle.
            world_up_perp = world_up - np.dot(world_up, fwd_new) * fwd_new
            wup_len = np.linalg.norm(world_up_perp)
            if wup_len < 1e-6:
                # Fallback: camera would point along world up; use world X as reference.
                fallback = np.array([1.0, 0.0, 0.0])
                world_up_perp = fallback - np.dot(fallback, fwd_new) * fwd_new
                wup_len = np.linalg.norm(world_up_perp)
            world_up_perp = world_up_perp / wup_len

            # Rodrigues rotation of world_up_perp around fwd_new by median_roll.
            c, s = np.cos(median_roll), np.sin(median_roll)
            up_target = (
                c * world_up_perp
                + s * np.cross(fwd_new, world_up_perp)
                + (1 - c) * np.dot(fwd_new, world_up_perp) * fwd_new
            )  # rotated up vector; norm ≈ 1 by construction

            # Construct new cam_to_world columns from (right, down, forward, pos).
            right = np.cross(-up_target, fwd_new)  # right = (-up) × fwd
            right_len = np.linalg.norm(right)
            if right_len < 1e-9:
                continue  # degenerate
            right = right / right_len

            # Re-orthogonalise up against right and forward.
            # Order matters: cross(right, fwd) satisfies right × up = -fwd,
            # which makes [right, -up, fwd] a right-handed frame (det = +1).
            up_target = np.cross(right, fwd_new)
            up_target = up_target / np.linalg.norm(up_target)

            pos = cam.position  # translation unchanged — only rotation changes
            ctw = np.eye(4, dtype=float)
            ctw[:3, 0] = right          # X column: camera right
            ctw[:3, 1] = -up_target     # Y column: camera down (OpenCV convention)
            ctw[:3, 2] = fwd_new        # Z column: camera forward
            ctw[:3, 3] = pos            # W column: camera position

            new_ext = np.linalg.inv(ctw)  # world-to-camera extrinsic
            cam.set_extrinsic(new_ext)  # recompute position, view, u_axis and cam_to_world

        self.edit_history.append(f"Oriented to centre {sel_names}")
        self._refresh_table()
        self._refresh_preview()
        self._refresh_status()
        self._refresh_open3d_native_view()

    def _match_selected_intrinsics(self) -> None:
        # Copy the element-wise median intrinsic K matrix of reference cameras
        # onto all selected cameras.  Resolution is also matched to the modal
        # resolution of the reference set (most common W×H).
        ref_names = [n for n, st in self.states.items() if st.is_reference]
        sel_names = [n for n, st in self.states.items()
                     if n in self._o3d_select_checks and bool(self._o3d_select_checks[n].checked)]
        if not ref_names or not sel_names:
            return

        # Stack reference K matrices and take the element-wise median.
        ref_Ks = np.array(
            [self.working_camset[n].intrinsic for n in ref_names], dtype=float
        )  # shape (N_ref, 3, 3)
        median_K = np.median(ref_Ks, axis=0)  # element-wise median; shape (3, 3)

        # Modal resolution from references (most common (W, H) pair).
        ref_res = [tuple(self.working_camset[n].res) for n in ref_names]
        res_counts: dict[tuple, int] = {}
        for r in ref_res:
            res_counts[r] = res_counts.get(r, 0) + 1
        modal_res = max(res_counts, key=lambda k: res_counts[k])  # (W, H) tuple

        self._push_undo()

        for sn in sel_names:
            cam = self.working_camset[sn]
            cam.intrinsic = median_K.copy()  # assign element-wise median K
            cam.res = list(modal_res)         # match resolution so viewcone matches
            cam._update_state()

        self._refresh_table()
        self._refresh_preview()
        self._refresh_status()
        self._refresh_open3d_native_view()

    def _on_o3d_apply_nudge(self) -> None:
        self._sync_o3d_inputs_to_qt()
        self._apply_nudge()

    def _on_o3d_set_trust(self) -> None:
        self._sync_o3d_inputs_to_qt()
        self._set_selected_trust()

    def _on_o3d_set_inclusion(self) -> None:
        self._sync_o3d_inputs_to_qt()
        self._set_selected_inclusion()

    def _on_o3d_assign_plane(self) -> None:
        self._sync_o3d_inputs_to_qt()
        self._assign_plane_group()

    def _on_o3d_project_to_plane(self) -> None:
        self._sync_o3d_inputs_to_qt()
        self._project_selected_to_plane()

    def _on_o3d_snap_radius(self) -> None:
        self._sync_o3d_inputs_to_qt()
        self._snap_radius_to_trusted()

    def _on_o3d_apply_radial(self) -> None:
        self._sync_o3d_inputs_to_qt()
        self._apply_radial_move()

    def _on_o3d_match_plane_offset(self) -> None:
        self._sync_o3d_inputs_to_qt()
        self._match_plane_offset_to_references()

    def _on_o3d_undo(self) -> None:
        self._undo()

    def _on_o3d_redo(self) -> None:
        self._redo()

    def _on_o3d_reset_selected(self) -> None:
        self._reset_selected()

    def _on_o3d_reset_all(self) -> None:
        self._reset_all()

    def _on_o3d_apply_and_close(self) -> None:
        self._o3d_apply_requested = True
        if self._o3d_window is not None:
            self._o3d_window.close()
        _o3d_gui.Application.instance.quit()

    def _on_o3d_discard_and_close(self) -> None:
        self.result = None
        self._o3d_return_code = QDialog.DialogCode.Rejected
        if self._o3d_window is not None:
            self._o3d_window.close()
        _o3d_gui.Application.instance.quit()

    def _open_open3d_view(self) -> None:
        """Open the single Open3D-native editor window and run its event loop."""
        if not _OPEN3D_OK:
            return
        self._build_open3d_native_window()
        _o3d_gui.Application.instance.run()

    def _refresh_status(self) -> None:
        """Update selection summary and edited-count diagnostics."""
        selected = self._selected_names()
        edited_count = sum(
            np.linalg.norm(state.edited_center - state.original_center) > 1e-9
            for state in self.states.values()
        )
        max_shift = max(
            [float(np.linalg.norm(state.edited_center - state.original_center)) for state in self.states.values()] or [0.0]
        )
        status_text = (
            f"Selected: {selected if selected else 'none'}\n"
            f"Edited lockbox copy: {edited_count} cameras edited, max shift {max_shift:.6g}"
        )
        self.selection_label.setText(status_text)
        if self._o3d_status_label is not None:
            self._o3d_status_label.text = status_text.replace('\n', '  |  ')
        if self._o3d_camera_info_labels:
            for name, label in self._o3d_camera_info_labels.items():
                state = self.states[name]
                delta_norm = float(np.linalg.norm(state.edited_center - state.original_center))
                selected_mark = 'S' if (name in self._o3d_select_checks and self._o3d_select_checks[name].checked) else '-'
                ref_mark = 'R' if state.is_reference else '-'
                label.text = (
                    f"{name} | {selected_mark}/{ref_mark} | {state.trust} | "
                    f"plane={state.plane_group or '-'} | delta={delta_norm:.4g}"
                )
    def _apply_nudge(self) -> None:
        """Translate selected cameras by the numeric target-frame nudge."""
        names = self._selected_names()
        if not names:
            QMessageBox.information(self, "No selection", "Select one or more cameras to nudge.")
            return
        delta = np.array([self.nudge_x.value(), self.nudge_y.value(), self.nudge_z.value()], dtype=float)
        if np.allclose(delta, 0.0):
            return
        self._push_undo()
        for name in names:
            self._set_camera_center(name, self.states[name].edited_center + delta)
        self.edit_history.append(f"Nudged {names} by {delta.tolist()}")
        self._refresh_table()
        self._refresh_preview()
        self._refresh_status()

    def _set_selected_trust(self) -> None:
        """Mark selected cameras as trusted, uncertain, or bad."""
        names = self._selected_names()
        if not names:
            QMessageBox.information(self, "No selection", "Select one or more cameras to mark.")
            return
        trust = self.trust_combo.currentText()
        for name in names:
            self.states[name].trust = trust
        self.edit_history.append(f"Set trust={trust} for {names}")
        self._refresh_table()

    def _assign_plane_group(self) -> None:
        """Assign selected (S) and reference (R) cameras to a persistent plane group id."""
        sel = self._selected_names()
        ref = [n for n in self._reference_names() if n not in sel]  # R cameras not already in S list
        names = sel + ref  # both S and R cameras are eligible for plane assignment
        plane_id = self.plane_edit.text().strip()
        if not names or not plane_id:
            QMessageBox.information(self, "Plane group", "Select at least one camera (S or R) and enter a plane group name.")
            return
        for name in names:
            self.states[name].plane_group = plane_id
        self.edit_history.append(f"Assigned {names} to plane group {plane_id}")
        self._refresh_table()

    def _set_selected_inclusion(self) -> None:
        """Set whether selected cameras are included in the derived lockbox metadata."""
        names = self._selected_names()
        if not names:
            QMessageBox.information(self, "No selection", "Select one or more cameras to include/exclude.")
            return
        included = bool(self.include_selected_cb.isChecked())
        for name in names:
            self.states[name].included_in_lockbox = included
        self.edit_history.append(f"Set included_in_lockbox={included} for {names}")
        self._refresh_table()

    def _reference_names(self) -> list[str]:
        """Return reference-camera names from the active frontend."""
        if self._o3d_ref_checks:
            return [name for name, cb in self._o3d_ref_checks.items() if cb.checked]
        names: list[str] = []
        for i in range(self.reference_list.count()):
            item = self.reference_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                names.append(item.text())
        return names

    def _reference_centres(self) -> list[np.ndarray]:
        """Return edited centres of cameras checked in the reference selector."""
        ref_names = self._reference_names()
        return [self.states[name].edited_center for name in ref_names if name in self.states]

    def _snap_radius_to_trusted(self) -> None:
        """Snap selected cameras to the median reference-camera radius."""
        names = self._selected_names()
        if not names:
            QMessageBox.information(self, "No selection", "Select one or more cameras to snap.")
            return
        references = self._reference_centres()
        if len(references) < 1:
            QMessageBox.warning(
                self,
                "Reference cameras",
                "Check at least one reference camera in the reference list. "
                "Reference cameras are distinct from trust flags and table selection.",
            )
            return
        if len(references) < 2:
            QMessageBox.warning(
                self,
                "Reference cameras",
                f"Only {len(references)} reference camera(s) selected. "
                "Consider adding more trusted references for robust median statistics.",
            )
        radius = reference_radius(references, self.object_centre, statistic="median")
        self._push_undo()
        try:
            for name in names:
                self._set_camera_center(name, snap_radius_to_reference(self.states[name].edited_center, self.object_centre, radius))
        except ValueError as exc:
            self._restore_snapshot(self.undo_stack.pop())
            QMessageBox.warning(self, "Snap radius", str(exc))
            return
        self.edit_history.append(f"Snapped {names} to reference median radius {radius:.6g}")
        self._refresh_table()
        self._refresh_preview()
        self._refresh_status()

    def _apply_radial_move(self) -> None:
        """Move selected cameras radially (outward/inward) relative to object centre."""
        names = self._selected_names()
        if not names:
            QMessageBox.information(self, "No selection", "Select one or more cameras to move radially.")
            return
        delta = float(self.radial_delta_spin.value())
        if abs(delta) < 1e-12:
            return
        self._push_undo()
        try:
            for name in names:  # iterate selected cameras for radial displacement
                state = self.states[name]  # row state holds the current edited centre
                vec = state.edited_center - self.object_centre  # vector from object centre to camera
                norm = float(np.linalg.norm(vec))  # scalar distance; used to normalise direction
                if norm <= 1e-12:  # camera is effectively on the object centre
                    raise ValueError(f"Camera {name} is effectively at the object centre; radial direction is undefined.")
                direction = vec / norm  # unit radial direction away from centre
                self._set_camera_center(name, state.edited_center + delta * direction)  # displace along radial
        except ValueError as exc:  # catch degenerate-position errors raised above
            self._restore_snapshot(self.undo_stack.pop())  # roll back before the failed move
            QMessageBox.warning(self, "Radial move", str(exc))
            return
        self.edit_history.append(f"Radial move {names} by {delta:.6g}")  # log the operation
        self._refresh_table()
        self._refresh_preview()
        self._refresh_status()

    def _match_plane_offset_to_references(self) -> None:
        # Move selected cameras along their plane normal to match reference signed offset.
        names = self._selected_names()
        if not names:
            QMessageBox.information(self, "No selection", "Select one or more cameras to offset-match.")
            return
        group_ids = {self.states[name].plane_group for name in names if self.states[name].plane_group}  # unique plane groups of selection
        if len(group_ids) != 1:  # must share exactly one plane group
            QMessageBox.warning(self, "Plane group", "Selected cameras must share exactly one plane group for offset matching.")
            return
        group_id = next(iter(group_ids))  # the single shared plane group name
        group = next((g for g in self._plane_groups() if g.plane_id == group_id), None)  # find the fitted PlaneGroup object
        if group is None or group.normal is None or group.point is None:  # plane not yet fittable
            QMessageBox.warning(self, "Plane group", "Plane group needs at least two non-bad cameras before offset matching.")
            return
        ref_names = self._reference_names()  # all cameras currently marked as reference
        ref_in_plane = [name for name in ref_names if name in self.states and self.states[name].plane_group == group_id]  # references that share this plane
        if not ref_in_plane:
            QMessageBox.warning(
                self,
                "Reference cameras",
                "No reference cameras are checked in this plane group. Check at least one reference camera that belongs to the same plane.",
            )
            return
        normal = np.asarray(group.normal, dtype=float)  # fitted plane normal (unit vector)
        plane_point = np.asarray(group.point, dtype=float)  # a point on the fitted plane
        d_val = float(-normal @ plane_point)  # scalar d in plane equation: normal·x + d = 0
        ref_offsets = [float(normal @ self.states[name].edited_center + d_val) for name in ref_in_plane]  # signed offset per reference
        target_offset = float(np.median(ref_offsets))  # robust target: median of reference offsets
        self._push_undo()
        try:
            for name in names:  # apply median offset to each selected camera
                self._set_camera_center(
                    name,
                    match_signed_plane_offset(self.states[name].edited_center, normal, target_offset, d=d_val),
                )
            self.edit_history.append(f"Matched plane offset for {names} to reference median {target_offset:.6g} on {group_id}")
        except Exception as exc:  # catch any geometry or state error; undo and report
            self._restore_snapshot(self.undo_stack.pop())
            QMessageBox.warning(self, "Match plane offset", f"Operation failed: {exc}")
            return
        self._refresh_table()
        self._refresh_preview()
        self._refresh_status()

    def _plane_groups(self) -> list:
        # Fit all named plane groups from non-bad members.
        groups = []
        for plane_id in sorted({state.plane_group for state in self.states.values() if state.plane_group}):  # unique non-empty group names
            members = [state for state in self.states.values() if state.plane_group == plane_id]  # all states in this group
            trusted_members = [state for state in members if state.trust == 'trusted']  # prefer trusted for fitting
            fit_members = select_fit_members(members)  # centralised trust-priority selection, kept in sync with tests
            group = PlaneGroup(
                plane_id=plane_id,
                member_camera_names=[state.name for state in members],  # names of all group members
                trusted_members=[state.name for state in trusted_members],  # names of trusted members
            )
            if len(fit_members) >= 2:  # need at least 2 points to define a plane
                try:
                    fit = fit_plane([state.edited_center for state in fit_members])  # least-squares plane fit
                    group.fit_mode = str(fit['fit_mode'])
                    group.normal = list(fit['normal'])    # fitted unit normal
                    group.point = list(fit['point'])      # centroid of fit points
                    group.rms_residual = float(fit['rms_residual'])
                    group.max_residual = float(fit['max_residual'])
                except ValueError:  # collinear or degenerate set
                    group.fit_mode = 'unfit'
            groups.append(group)
        return groups

    def _project_selected_to_plane(self) -> None:
        # Project selected cameras onto their common explicit plane group.
        names = self._selected_names()
        if not names:
            QMessageBox.information(self, "No selection", "Select one or more cameras to project.")
            return
        group_ids = {self.states[name].plane_group for name in names if self.states[name].plane_group}  # unique plane groups of selection
        if len(group_ids) != 1:  # must share exactly one plane group
            QMessageBox.warning(self, "Plane group", "Selected cameras must share exactly one explicit plane group.")
            return
        group_id = next(iter(group_ids))
        group = next((candidate for candidate in self._plane_groups() if candidate.plane_id == group_id), None)
        if group is None or group.normal is None or group.point is None:
            # Plane not yet fittable: need ≥2 trusted cameras, or ≥2 non-bad cameras if none are trusted.
            QMessageBox.warning(
                self,
                "Plane group",
                f"Plane group '{group_id}' cannot be fitted yet.\n\n"
                "Rules: if the group contains any trusted cameras, at least 2 trusted cameras are needed to fit the plane "
                "(uncertain cameras are not used when trusted ones exist). "
                "If no trusted cameras are present, at least 2 uncertain (non-bad) cameras are required.\n\n"
                "Assign more cameras to the group or adjust trust levels.",
            )
            return
        self._push_undo()
        try:
            for name in names:  # project each selected camera centre onto the fitted plane
                self._set_camera_center(
                    name,
                    project_point_to_plane(self.states[name].edited_center, group.normal, plane_point=group.point),
                )
            self.edit_history.append(f"Projected {names} onto plane group {group_id}")
        except Exception as exc:  # catch any geometry or state error; undo and report
            self._restore_snapshot(self.undo_stack.pop())
            QMessageBox.warning(self, "Project to plane", f"Operation failed: {exc}")
            return
        self._refresh_table()
        self._refresh_preview()
        self._refresh_status()

    def _undo(self) -> None:
        # Undo the last centre-edit operation.
        if not self.undo_stack:
            return
        self.redo_stack.append(self._snapshot())  # push current state to redo stack before restoring
        self._restore_snapshot(self.undo_stack.pop())

    def _redo(self) -> None:
        # Redo the last undone centre-edit operation.
        if not self.redo_stack:
            return
        self.undo_stack.append(self._snapshot())  # push current state to undo stack before restoring
        self._restore_snapshot(self.redo_stack.pop())

    def _reset_selected(self) -> None:
        # Reset selected cameras to the original source camset centres.
        names = self._selected_names()
        if not names:
            QMessageBox.information(self, "No selection", "Select one or more cameras to reset.")
            return
        self._push_undo()
        for name in names:
            self._set_camera_center(name, self.states[name].original_center)  # revert to original position
        self.edit_history.append(f"Reset selected cameras {names}")
        self._refresh_table()
        self._refresh_preview()
        self._refresh_status()

    def _reset_all(self) -> None:
        # Reset all edited centres while preserving trust/plane annotations.
        if QMessageBox.question(self, "Reset all edits", "Reset all edited lockbox copy centres to the original source camset?") != QMessageBox.StandardButton.Yes:
            return
        self._push_undo()
        for name, state in self.states.items():
            self._set_camera_center(name, state.original_center)  # revert each camera to original centre
        self.edit_history.append("Reset all camera centres")
        self._refresh_table()
        self._refresh_preview()
        self._refresh_status()

    def _save_and_apply(self) -> None:
        # Save a derived camset and metadata sidecar, then accept the dialog.
        if not self._validate_before_apply():
            return
        max_shift = max(  # largest distance any camera was moved from its source position
            [float(np.linalg.norm(state.edited_center - state.original_center)) for state in self.states.values()] or [0.0]
        )
        if max_shift > 0.25:  # warn user if any camera moved more than 25 cm (world units)
            answer = QMessageBox.question(
                self,
                "Large lockbox shift",
                f"Maximum source-to-edited shift is {max_shift:.6g}. Apply this edited lockbox copy?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # timestamp for unique output directory
        out_dir = self.workspace_path / 'lockbox_priors' / stamp  # output directory under workspace
        out_dir.mkdir(parents=True, exist_ok=True)
        edited_path = out_dir / 'edited_lockbox_source.camset'
        metadata_path = out_dir / 'edited_lockbox_source_metadata.json'
        self.working_camset.save(edited_path)  # persist the edited camera set to disk
        metadata = build_lockbox_metadata(
            original_source_camset=self.source_camset_path,
            edited_source_camset=edited_path,
            centre_definition=self.centre_definition,
            cameras={
                name: CameraEditRecord(  # per-camera record of trust, plane group, and centres
                    trust=state.trust,
                    plane_group=state.plane_group or None,
                    included_in_lockbox=state.included_in_lockbox,
                    original_center=state.original_center.tolist(),
                    edited_center=state.edited_center.tolist()
                )
                for name, state in self.states.items()
            },
            plane_groups=self._plane_groups(),
            edit_history_summary=self.edit_history,
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')  # write JSON sidecar
        reloaded = load_CameraSet(edited_path)  # reload to verify the saved file is intact
        if set(reloaded.get_names()) != set(self.source_camset.get_names()):  # camera-name mismatch indicates corruption
            raise RuntimeError('Edited lockbox copy failed camera-name validation after save')
        edited_count = sum(
            np.linalg.norm(state.edited_center - state.original_center) > 1e-9
            for state in self.states.values()
        )
        self.result = LockboxEditorResult(
            original_source_camset=str(self.source_camset_path),
            edited_source_camset=str(edited_path),
            effective_lockbox_source=str(edited_path),
            metadata_path=str(metadata_path),
            edited_count=int(edited_count),
            max_shift=float(max_shift),
        )
        self.accept()

    def _validate_before_apply(self) -> bool:
        # Run guardrail checks before saving/applying the edited copy.
        return True  # placeholder — full guardrail checks to be added
