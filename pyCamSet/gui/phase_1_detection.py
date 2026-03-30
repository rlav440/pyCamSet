"""
Phase 1 — Target Detection tab and diagnostics.

This module wraps sub-phases 1a–1c from ``phase_planning.md``:

- **1a** Sub-folder discovery & image sanitisation.
- **1b** Per-camera corner detection via
  :func:`pyCamSet.calibration.camera_calibrator.detect_datapoints_in_imfile`.
- **1c** Detection validation via
  :func:`pyCamSet.calibration.camera_calibrator.validate_detections`.

Diagnostics implemented (D1.1–D1.7)
-------------------------------------
- **D1.1** Total detections per camera (``TargetDetection.get_cam_list``).
- **D1.2** Detection rate per camera (``validate_detections`` logic).
- **D1.3** Board completeness per camera (``validate_detections`` logic).
- **D1.4** Features-per-image-per-camera heatmap
  (``TargetDetection.features_per_im_per_cam``).
- **D1.5** Per-camera detection overlay note (``draw=True`` passes to existing
  ``find_in_imfolder``; GUI shows saved-pickle location).
- **D1.6** Per-camera detection spatial coverage (convex hull / image area).
- **D1.7** Minimum features in any image–camera pair
  (``features_per_im_per_cam().min()``).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from pyCamSet.gui.shared_functions import (
    TAB_PHASE1_DIAG,
    InfoHoverMixin,
    RunSelectorFrame,
    TerminalMixin,
    WorkspaceManager,
    make_continue_button,
    make_labeled_checkbox,
    make_labeled_entry,
    make_labeled_spinbox,
    make_orange_button,
    make_run_id,
    make_section_label,
)

# pyCamSet imports — guarded so the module can be imported without the full
# scientific stack (e.g., for syntax-checking or in test environments).
try:
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile,
        validate_detections,
    )
    from pyCamSet.calibration_targets.target_Ccube import Ccube
    from pyCamSet.calibration_targets.target_charuco import ChArUco
    from pyCamSet.utils.general_utils import get_subfolder_names
    _PYCAMSET_OK = True
except ImportError:
    _PYCAMSET_OK = False

_TARGET_CHOICES = ["Ccube", "ChArUco"]


def _build_target(target_type: str, n_points: int, length: float):
    """Construct the appropriate calibration target object.

    :param target_type: ``"Ccube"`` or ``"ChArUco"``.
    :param n_points: For Ccube = n_points per face; for ChArUco = squares_x.
    :param length: Side length / square size in mm.
    :raises ValueError: If the target type is unrecognised.
    :raises RuntimeError: If pyCamSet is not importable.
    """
    if not _PYCAMSET_OK:
        raise RuntimeError("pyCamSet calibration targets are not importable.")
    if target_type == "Ccube":
        return Ccube(n_points=n_points, length=length)
    if target_type == "ChArUco":
        return ChArUco(
            num_squares_x=n_points,
            num_squares_y=n_points,
            square_size=length,
        )
    raise ValueError(f"Unknown target type: {target_type!r}")


class Phase1Tab(tk.Frame, InfoHoverMixin, TerminalMixin):
    """Phase 1 — Target Detection tab.

    Presents all Phase 1 user-tunable parameters from ``phase_planning.md``
    and runs detection using the existing
    :func:`~pyCamSet.calibration.camera_calibrator.detect_datapoints_in_imfile`
    followed by
    :func:`~pyCamSet.calibration.camera_calibrator.validate_detections`.
    """

    def __init__(
        self,
        parent: tk.Widget,
        notebook: ttk.Notebook,
        info_var: tk.BooleanVar,
        show_terminal_var: tk.BooleanVar,
        workspace_mgr: WorkspaceManager,
    ) -> None:
        super().__init__(parent)
        self._notebook = notebook
        self._workspace_mgr = workspace_mgr
        self._diagnostics_tab: Optional["Phase1DiagnosticsTab"] = None
        self.init_hover(info_var)
        self._build_ui(show_terminal_var)

    def set_diagnostics_tab(self, tab: "Phase1DiagnosticsTab") -> None:
        """Register the companion diagnostics tab for routing."""
        self._diagnostics_tab = tab

    # ------------------------------------------------------------------

    def _build_ui(self, show_terminal_var: tk.BooleanVar) -> None:
        outer = tk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = tk.Frame(outer)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = tk.Frame(outer, width=190)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right.pack_propagate(False)

        # ── Paths section ──────────────────────────────────────────────
        make_section_label(left, "Paths").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 2)
        )
        lbl, self._floc_entry, self._floc_var = make_labeled_entry(
            left, "Image folder (f_loc):", width=44
        )
        lbl.grid(row=1, column=0, sticky="w", padx=(0, 4))
        self._floc_entry.grid(row=1, column=1, sticky="ew")
        ttk.Button(left, text="Browse…", command=self._browse_floc, width=8).grid(
            row=1, column=2, padx=(4, 0)
        )
        self.bind_hover(self._floc_entry, "Root folder with per-camera sub-folders.")
        left.columnconfigure(1, weight=1)

        # ── Detection options ──────────────────────────────────────────
        ttk.Separator(left, orient=tk.HORIZONTAL).grid(
            row=2, column=0, columnspan=3, sticky="ew", pady=6
        )
        make_section_label(left, "Detection Options").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(0, 2)
        )

        self._draw_cb, self._draw_var = make_labeled_checkbox(
            left, "Draw detections (draw)", default=False
        )
        self._draw_cb.grid(row=4, column=0, columnspan=2, sticky="w")
        self.bind_hover(self._draw_cb, "Render corner overlays on each image as it is processed.")

        self._cache_cb, self._cache_var = make_labeled_checkbox(
            left, "Cache detections (caching)", default=True
        )
        self._cache_cb.grid(row=5, column=0, columnspan=2, sticky="w")
        self.bind_hover(
            self._cache_cb,
            "Save/load detected_datapoints.pickle.  "
            "Disable to force re-detection.",
        )

        lbl_n, self._nlim_entry, self._nlim_var = make_labeled_entry(
            left, "Max images per camera (n_lim):", default="", width=10
        )
        lbl_n.grid(row=6, column=0, sticky="w", padx=(0, 4))
        self._nlim_entry.grid(row=6, column=1, sticky="w")
        self.bind_hover(
            self._nlim_entry, "Limit images per camera folder (leave blank = no limit)."
        )

        # ── Target configuration ───────────────────────────────────────
        ttk.Separator(left, orient=tk.HORIZONTAL).grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=6
        )
        make_section_label(left, "Calibration Target").grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(0, 2)
        )

        # Target type
        tk.Label(left, text="Target type:", anchor="w").grid(
            row=9, column=0, sticky="w", padx=(0, 4)
        )
        self._target_var = tk.StringVar(value=_TARGET_CHOICES[0])
        target_cb = ttk.Combobox(
            left,
            textvariable=self._target_var,
            values=_TARGET_CHOICES,
            state="readonly",
            width=12,
        )
        target_cb.grid(row=9, column=1, sticky="w")
        self.bind_hover(target_cb, "Ccube = corner-cube target; ChArUco = charuco board.")

        # n_points
        lbl_np, self._npts_spinbox, self._npts_var = make_labeled_spinbox(
            left, "n_points / squares_x:", from_=2, to=20, default=6
        )
        lbl_np.grid(row=10, column=0, sticky="w", padx=(0, 4))
        self._npts_spinbox.grid(row=10, column=1, sticky="w")
        self.bind_hover(
            self._npts_spinbox,
            "For Ccube: points per face edge.  For ChArUco: number of squares in x.",
        )

        # length
        lbl_l, self._length_entry, self._length_var = make_labeled_entry(
            left, "Length / square size (mm):", default="30.0", width=10
        )
        lbl_l.grid(row=11, column=0, sticky="w", padx=(0, 4))
        self._length_entry.grid(row=11, column=1, sticky="w")
        self.bind_hover(
            self._length_entry,
            "Physical size of the target feature in millimetres.",
        )

        # ── Bottom action row ──────────────────────────────────────────
        btn_row = tk.Frame(left)
        btn_row.grid(row=12, column=0, columnspan=3, sticky="w", pady=10)

        ttk.Button(btn_row, text="▶  Run Phase 1", command=self._run_phase1).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        make_orange_button(btn_row, "Diagnostics ▼", self._open_diagnostics).pack(
            side=tk.LEFT
        )

        # ── Right-side continue button ─────────────────────────────────
        make_continue_button(right, self._continue_to_next).pack(
            pady=20, padx=4, fill=tk.X
        )

        # ── Terminal ──────────────────────────────────────────────────
        self.init_terminal(self, show_terminal_var)

    # ------------------------------------------------------------------

    def _browse_floc(self) -> None:
        path = filedialog.askdirectory(title="Select image folder")
        if path:
            self._floc_var.set(path)

    def _collect_params(self) -> Optional[dict]:
        floc = self._floc_var.get().strip()
        if not floc:
            messagebox.showerror("Validation Error", "Image folder is required.")
            return None

        n_lim_str = self._nlim_var.get().strip()
        n_lim = None
        if n_lim_str:
            try:
                n_lim = int(n_lim_str)
            except ValueError:
                messagebox.showerror("Validation Error", "n_lim must be an integer.")
                return None

        try:
            length = float(self._length_var.get().strip())
        except ValueError:
            messagebox.showerror("Validation Error", "Length must be a number.")
            return None

        return {
            "f_loc": floc,
            "draw": self._draw_var.get(),
            "caching": self._cache_var.get(),
            "n_lim": n_lim,
            "target_type": self._target_var.get(),
            "n_points": int(self._npts_var.get()),
            "length": length,
        }

    def _run_phase1(self) -> None:
        """Run sub-phases 1a–1c in a daemon thread using existing pyCamSet code."""
        params = self._collect_params()
        if params is None:
            return

        self.terminal_clear()
        self.terminal_append("=== Phase 1: Target Detection ===")
        self.terminal_append(f"Image folder : {params['f_loc']}")
        self.terminal_append(
            f"Target       : {params['target_type']} "
            f"(n={params['n_points']}, length={params['length']} mm)"
        )
        self.terminal_append(f"caching      : {params['caching']}")
        self.terminal_append(f"draw         : {params['draw']}")
        self.terminal_append(f"n_lim        : {params['n_lim']}")
        self.terminal_append("Starting detection…")

        def _worker() -> None:
            diagnostics: dict = {}
            error_msg: Optional[str] = None
            detections = None
            try:
                f_loc = Path(params["f_loc"])

                if not _PYCAMSET_OK:
                    raise RuntimeError(
                        "pyCamSet calibration module not importable.  "
                        "Check your installation."
                    )

                # 1a — sub-folder discovery (uses existing get_subfolder_names)
                cam_names = get_subfolder_names(f_loc)
                self.after(
                    0, self.terminal_append,
                    f"1a  Camera sub-folders: {cam_names}")

                # Build calibration target from existing pyCamSet classes
                target = _build_target(
                    params["target_type"],
                    params["n_points"],
                    params["length"],
                )

                # 1b + 1a — detect corners using existing function
                detections, cam_res = detect_datapoints_in_imfile(
                    f_loc=f_loc,
                    calibration_target=target,
                    caching=params["caching"],
                    draw=params["draw"],
                    n_lim=params["n_lim"],
                )
                self.after(0, self.terminal_append, "1b  Detection complete.")

                # 1c — validate using existing validate_detections
                validate_detections(detections, target)
                self.after(0, self.terminal_append, "1c  Validation complete.")

                # ── Diagnostics ──────────────────────────────────────
                try:
                    # D1.1 — total detections per camera
                    total_per_cam: dict[str, int] = {}
                    for cam_det in detections.get_cam_list():
                        cam_idx = int(cam_det.get_data()[0, 0])
                        cam_name = detections.cam_names[cam_idx]
                        total_per_cam[cam_name] = len(cam_det.get_data())
                    diagnostics["D1.1_total_detections"] = total_per_cam
                    for cam, n in total_per_cam.items():
                        self.after(0, self.terminal_append,
                                   f"D1.1  {cam}: {n} detections")

                    # D1.2 + D1.3 — detection rate and board completeness
                    # (replicate the logic from validate_detections)
                    corners_per_face = target.point_data.shape[-2]
                    det_rate: dict[str, float] = {}
                    completeness: dict[str, float] = {}
                    for cam_det in detections.get_cam_list():
                        cam_idx = int(cam_det.get_data()[0, 0])
                        cam_name = detections.cam_names[cam_idx]
                        detected_boards = 0
                        fracs: list[float] = []
                        for im_det in cam_det.get_image_list():
                            datum = im_det.get_data()
                            if datum is not None:
                                detected_boards += 1
                                n_keys = datum.shape[1] - 4
                                if n_keys == 1:
                                    fracs.append(datum.shape[0] / corners_per_face)
                                else:
                                    n_boards = len(
                                        np.unique(datum[:, 2:-2], axis=0)
                                    )
                                    fracs.append(
                                        datum.shape[0] / corners_per_face / max(n_boards, 1)
                                    )
                        det_rate[cam_name] = detected_boards / detections.max_ims
                        completeness[cam_name] = float(np.mean(fracs)) if fracs else 0.0
                    diagnostics["D1.2_detection_rate"] = det_rate
                    diagnostics["D1.3_board_completeness"] = completeness
                    for cam in detections.cam_names:
                        r = det_rate.get(cam, 0) * 100
                        c = completeness.get(cam, 0) * 100
                        self.after(0, self.terminal_append,
                                   f"D1.2/D1.3  {cam}: "
                                   f"rate={r:.1f}% completeness={c:.1f}%")

                    # D1.4 — features-per-image-per-camera matrix (stored for heatmap)
                    fpm = detections.features_per_im_per_cam()
                    diagnostics["D1.4_features_matrix"] = fpm.tolist()

                    # D1.6 — spatial coverage per camera
                    try:
                        from scipy.spatial import ConvexHull
                        coverage: dict[str, float] = {}
                        for cam_det, res in zip(detections.get_cam_list(), cam_res):
                            cam_idx = int(cam_det.get_data()[0, 0])
                            cam_name = detections.cam_names[cam_idx]
                            pts = cam_det.get_data()[:, -2:]
                            img_area = float(res[0]) * float(res[1])
                            if len(pts) >= 3:
                                try:
                                    hull_area = ConvexHull(pts).volume  # 2-D → area
                                    coverage[cam_name] = hull_area / img_area
                                except Exception:
                                    coverage[cam_name] = float("nan")
                            else:
                                coverage[cam_name] = float("nan")
                        diagnostics["D1.6_spatial_coverage"] = coverage
                    except ImportError:
                        pass  # scipy not available — skip D1.6

                    # D1.7 — min features in any image–camera pair
                    min_feat = int(np.min(fpm[fpm > 0])) if np.any(fpm > 0) else 0
                    diagnostics["D1.7_min_features"] = min_feat
                    self.after(0, self.terminal_append,
                               f"D1.7  Min features in any image–camera: {min_feat}")

                    # Store cam_names for diagnostics rendering
                    diagnostics["cam_names"] = detections.cam_names
                    diagnostics["n_images"] = int(detections.max_ims)

                except Exception as diag_exc:
                    self.after(0, self.terminal_append,
                               f"  (partial diagnostics: {diag_exc})")

                self.after(0, self.terminal_append, "Phase 1 complete.")

            except Exception as exc:
                error_msg = str(exc)
                self.after(0, self.terminal_append, f"ERROR: {exc}")

            # Persist run metadata
            run_id = make_run_id()
            metadata = {
                "run_id": run_id,
                "phase": "phase1",
                "params": params,
                "diagnostics": diagnostics,
                "error": error_msg,
            }
            meta_path = self._workspace_mgr.save_run("phase1", run_id, metadata)
            self.after(0, self.terminal_append, f"Run saved → {meta_path}")

            if self._diagnostics_tab is not None:
                self.after(0, self._diagnostics_tab.refresh)

        threading.Thread(target=_worker, daemon=True).start()

    def _open_diagnostics(self) -> None:
        if self._diagnostics_tab is not None:
            self._diagnostics_tab.refresh()
        for i in range(self._notebook.index("end")):
            if self._notebook.tab(i, "text") == TAB_PHASE1_DIAG:
                self._notebook.select(i)
                return

    def _continue_to_next(self) -> None:
        runs = self._workspace_mgr.load_runs("phase1")
        if not runs:
            messagebox.showinfo("No runs", "Run Phase 1 first.")
            return
        self._workspace_mgr.write_handoff({"phase": "phase1", "runs": [runs[-1]]})
        messagebox.showinfo(
            "Handoff written",
            "handoff.json written to workspace.\nProceed to Phase 2.",
        )


# ---------------------------------------------------------------------------
# Phase 1 Diagnostics tab
# ---------------------------------------------------------------------------

class Phase1DiagnosticsTab(tk.Frame, InfoHoverMixin):
    """Phase 1 Diagnostics — multi-run comparison view.

    Contains three sub-tabs:

    1. **Summary** — scrollable table of D1.1–D1.3, D1.6, D1.7 for selected runs.
    2. **Heatmap (D1.4)** — ``features_per_im_per_cam`` displayed as a matplotlib
       ``imshow`` embedded via ``FigureCanvasTkAgg``.
    3. **Montage (D1.5)** — instructions for using ``draw=True`` and the path to the
       saved detection pickle.
    """

    def __init__(
        self,
        parent: tk.Widget,
        notebook: ttk.Notebook,
        info_var: tk.BooleanVar,
        workspace_mgr: WorkspaceManager,
    ) -> None:
        super().__init__(parent)
        self._notebook = notebook
        self._workspace_mgr = workspace_mgr
        self.init_hover(info_var)
        self._build_ui()

    def _build_ui(self) -> None:
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ── Left: run selector ─────────────────────────────────────────
        left = tk.Frame(pane, width=230)
        pane.add(left, minsize=160)

        self._run_selector = RunSelectorFrame(
            left, runs=[], on_select=self._on_run_selected
        )
        self._run_selector.pack(fill=tk.BOTH, expand=True)

        ttk.Button(left, text="⟳  Refresh", command=self.refresh).pack(
            fill=tk.X, pady=(4, 0)
        )

        # ── Right: sub-tab notebook ────────────────────────────────────
        right = tk.Frame(pane)
        pane.add(right, minsize=440)

        make_section_label(right, "Phase 1 Diagnostics").pack(
            anchor="w", padx=4, pady=(4, 2)
        )
        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=4)

        self._sub_nb = ttk.Notebook(right)
        self._sub_nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._summary_frame = tk.Frame(self._sub_nb)
        self._heatmap_frame = tk.Frame(self._sub_nb)
        self._montage_frame = tk.Frame(self._sub_nb)

        self._sub_nb.add(self._summary_frame, text="Summary (D1.1–D1.3, D1.6–D1.7)")
        self._sub_nb.add(self._heatmap_frame, text="Heatmap (D1.4)")
        self._sub_nb.add(self._montage_frame, text="Montage (D1.5)")

        # ── Bottom: continue button ────────────────────────────────────
        btn_row = tk.Frame(self)
        btn_row.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=4)
        make_continue_button(btn_row, self._continue_to_next).pack(side=tk.RIGHT)

        self.refresh()

    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload runs from disk and re-render all sub-tabs."""
        runs = self._workspace_mgr.load_runs("phase1")
        self._run_selector.refresh(runs)
        selected = self._run_selector.get_selected()
        self._render_summary(selected)
        self._render_heatmap(selected)
        self._render_montage(selected)

    def _on_run_selected(self, selected: list[dict]) -> None:
        self._render_summary(selected)
        self._render_heatmap(selected)
        self._render_montage(selected)

    # ------------------------------------------------------------------
    # Sub-tab renderers
    # ------------------------------------------------------------------

    def _render_summary(self, runs: list[dict]) -> None:
        for w in self._summary_frame.winfo_children():
            w.destroy()

        if not runs:
            tk.Label(
                self._summary_frame,
                text="Select one or more runs from the list to compare.",
                fg="gray",
                justify="center",
            ).pack(pady=30)
            return

        canvas = tk.Canvas(self._summary_frame, highlightthickness=0)
        vsb = ttk.Scrollbar(self._summary_frame, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas)
        cw = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _resize(_e=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(cw, width=canvas.winfo_width())

        inner.bind("<Configure>", _resize)
        canvas.bind("<Configure>", _resize)

        row = 0
        for run in runs:
            run_id = run.get("run_id", "unknown")
            tk.Label(
                inner,
                text=f"Run: {run_id}",
                font=("TkDefaultFont", 10, "bold"),
                anchor="w",
            ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 2))
            row += 1

            d = run.get("diagnostics", {})
            p = run.get("params", {})

            header_entries = [
                ("f_loc", p.get("f_loc", "—")),
                ("target", f"{p.get('target_type','—')} n={p.get('n_points','—')}"
                           f" L={p.get('length','—')} mm"),
                ("D1.7  Min features", d.get("D1.7_min_features", "—")),
            ]
            for k, v in header_entries:
                tk.Label(inner, text=f"{k}:", anchor="w", fg="#555").grid(
                    row=row, column=0, sticky="w", padx=(8, 4)
                )
                tk.Label(inner, text=str(v), anchor="w").grid(row=row, column=1, sticky="w")
                row += 1

            # Per-camera table header
            cam_names = d.get("cam_names", list(d.get("D1.1_total_detections", {}).keys()))
            if cam_names:
                for col_lbl, col_idx in [
                    ("Camera", 0), ("D1.1 Detections", 1),
                    ("D1.2 Det-rate %", 2), ("D1.3 Completeness %", 3),
                    ("D1.6 Coverage", 4),
                ]:
                    tk.Label(
                        inner, text=col_lbl,
                        font=("TkDefaultFont", 9, "bold"), anchor="w",
                    ).grid(row=row, column=col_idx, sticky="w", padx=(8, 4))
                row += 1

                det = d.get("D1.1_total_detections", {})
                rate = d.get("D1.2_detection_rate", {})
                comp = d.get("D1.3_board_completeness", {})
                cov = d.get("D1.6_spatial_coverage", {})

                for cam in cam_names:
                    vals = [
                        cam,
                        str(det.get(cam, "—")),
                        f"{rate.get(cam, 0)*100:.1f}" if cam in rate else "—",
                        f"{comp.get(cam, 0)*100:.1f}" if cam in comp else "—",
                        f"{cov.get(cam, float('nan')):.3f}" if cam in cov else "—",
                    ]
                    for col_idx, val in enumerate(vals):
                        tk.Label(inner, text=val, anchor="w").grid(
                            row=row, column=col_idx, sticky="w", padx=(8, 4)
                        )
                    row += 1

            if run.get("error"):
                tk.Label(inner, text=f"Error: {run['error']}", fg="red", anchor="w").grid(
                    row=row, column=0, columnspan=2, sticky="w", padx=8
                )
                row += 1

            ttk.Separator(inner, orient=tk.HORIZONTAL).grid(
                row=row, column=0, columnspan=5, sticky="ew", pady=4
            )
            row += 1

    def _render_heatmap(self, runs: list[dict]) -> None:
        """Render the D1.4 features-per-image-per-camera heatmap."""
        for w in self._heatmap_frame.winfo_children():
            w.destroy()

        if not runs:
            tk.Label(
                self._heatmap_frame,
                text="Select a run to view the detection heatmap.",
                fg="gray",
                justify="center",
            ).pack(pady=30)
            return

        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError:
            tk.Label(
                self._heatmap_frame,
                text="matplotlib not available — cannot render heatmap.",
                fg="gray",
            ).pack(pady=20)
            return

        # Use most recently selected run that has the matrix
        matrix_data = None
        cam_names: list[str] = []
        selected_run: dict = {}
        for run in reversed(runs):
            d = run.get("diagnostics", {})
            if "D1.4_features_matrix" in d:
                matrix_data = np.array(d["D1.4_features_matrix"])
                cam_names = d.get("cam_names", [])
                selected_run = run
                break

        if matrix_data is None:
            tk.Label(
                self._heatmap_frame,
                text="No heatmap data in selected run(s).\nRe-run Phase 1 to generate it.",
                fg="gray",
                justify="center",
            ).pack(pady=30)
            return

        n_ims, n_cams = matrix_data.shape
        fig_w = min(max(5.0, n_cams * 0.6), 14.0)
        fig_h = min(max(3.0, n_ims * 0.15), 8.0)

        fig = Figure(figsize=(fig_w, fig_h), tight_layout=True)
        ax = fig.add_subplot(111)
        im = ax.imshow(matrix_data, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xlabel("Camera index")
        ax.set_ylabel("Image index")
        ax.set_title(
            f"D1.4  Features per image per camera\n(run: {selected_run.get('run_id', '?')})"
        )
        if cam_names:
            ax.set_xticks(range(n_cams))
            ax.set_xticklabels(cam_names, rotation=30, ha="right", fontsize=8)
        fig.colorbar(im, ax=ax, label="features detected")

        canvas = FigureCanvasTkAgg(fig, master=self._heatmap_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _render_montage(self, runs: list[dict]) -> None:
        """Render D1.5 montage information panel."""
        for w in self._montage_frame.winfo_children():
            w.destroy()

        inner = tk.Frame(self._montage_frame)
        inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        make_section_label(inner, "D1.5 — Per-camera detection overlay montage").pack(
            anchor="w", pady=(0, 4)
        )

        msg = (
            "Detection overlays are rendered live when the Phase 1 run is executed "
            "with 'Draw detections' enabled (draw=True).  Each frame is shown via "
            "OpenCV's imshow() as it is processed.\n\n"
            "The raw detection data is saved to:\n"
            "  <f_loc>/detected_datapoints.pickle\n\n"
            "You can reload and visualise detections at any time by loading that "
            "file via pyCamSet.utils.saving.load_pickle() and calling "
            "target.find_in_imfolder(..., draw=True) again with caching=False."
        )
        tk.Label(inner, text=msg, justify="left", wraplength=480, fg="#333").pack(
            anchor="w", pady=(0, 12)
        )

        # Show the pickle paths for all selected runs
        for run in runs:
            floc = run.get("params", {}).get("f_loc", "")
            if floc:
                pickle_path = Path(floc) / "detected_datapoints.pickle"
                exists = "✓ exists" if pickle_path.exists() else "✗ not found"
                tk.Label(
                    inner,
                    text=f"{pickle_path}  [{exists}]",
                    fg="#2e7d32" if pickle_path.exists() else "#b71c1c",
                    anchor="w",
                    wraplength=460,
                ).pack(anchor="w")

    # ------------------------------------------------------------------

    def _continue_to_next(self) -> None:
        selected = self._run_selector.get_selected()
        if not selected:
            runs = self._workspace_mgr.load_runs("phase1")
            selected = runs[-1:] if runs else []
        if not selected:
            messagebox.showinfo("No runs", "No Phase 1 runs available.")
            return
        self._workspace_mgr.write_handoff({"phase": "phase1", "runs": selected})
        messagebox.showinfo(
            "Handoff written",
            "handoff.json written to workspace.\nProceed to Phase 2.",
        )
