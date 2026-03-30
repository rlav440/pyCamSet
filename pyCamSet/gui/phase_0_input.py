"""
Phase 0 — Input Validation & Setup tab and diagnostics.

This module wraps the top section of
:func:`pyCamSet.calibration.camera_calibrator.calibrate_cameras` (lines 49–66)
into a GUI tab that validates parameters and computes diagnostics D0.1–D0.3
defined in ``phase_planning.md``.

Diagnostics implemented
-----------------------
- **D0.1** Number of camera sub-folders (``get_subfolder_names``).
- **D0.2** Number of images per camera (``glob_ims``).
- **D0.3** Image-count consistency (``sanitise_input_images``).
"""
from __future__ import annotations

import json
import threading
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from pyCamSet.gui.shared_functions import (
    TAB_PHASE0_DIAG,
    TAB_PHASE1,
    InfoHoverMixin,
    RunSelectorFrame,
    TerminalMixin,
    WorkspaceManager,
    make_continue_button,
    make_labeled_checkbox,
    make_labeled_entry,
    make_orange_button,
    make_run_id,
    make_section_label,
)

# pyCamSet utilities used for D0.1-D0.3
try:
    from pyCamSet.utils.general_utils import get_subfolder_names, glob_ims
    from pyCamSet.calibration.camera_calibrator import sanitise_input_images
    _PYCAMSET_OK = True
except ImportError:
    _PYCAMSET_OK = False


class Phase0Tab(tk.Frame, InfoHoverMixin, TerminalMixin):
    """Phase 0 — Input Validation & Setup tab.

    Presents all nine user-tunable parameters from ``phase_planning.md`` and
    runs diagnostics D0.1–D0.3 using the existing pyCamSet helpers:
    ``get_subfolder_names``, ``glob_ims``, and ``sanitise_input_images``.
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
        self._diagnostics_tab: Optional["Phase0DiagnosticsTab"] = None
        self.init_hover(info_var)
        self._build_ui(show_terminal_var)

    def set_diagnostics_tab(self, tab: "Phase0DiagnosticsTab") -> None:
        """Register the companion diagnostics tab for routing."""
        self._diagnostics_tab = tab

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, show_terminal_var: tk.BooleanVar) -> None:
        # Two-column layout: scrollable params on left, action buttons on right
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

        # f_loc
        lbl, self._floc_entry, self._floc_var = make_labeled_entry(
            left, "Image folder (f_loc):", width=44
        )
        lbl.grid(row=1, column=0, sticky="w", padx=(0, 4))
        self._floc_entry.grid(row=1, column=1, sticky="ew")
        ttk.Button(left, text="Browse…", command=self._browse_floc, width=8).grid(
            row=1, column=2, padx=(4, 0)
        )
        self.bind_hover(self._floc_entry, "Root folder containing one sub-folder per camera.")

        # save_loc
        lbl2, self._saveloc_entry, self._saveloc_var = make_labeled_entry(
            left, "Save location:", width=44
        )
        lbl2.grid(row=2, column=0, sticky="w", padx=(0, 4))
        self._saveloc_entry.grid(row=2, column=1, sticky="ew")
        ttk.Button(left, text="Browse…", command=self._browse_saveloc, width=8).grid(
            row=2, column=2, padx=(4, 0)
        )
        self.bind_hover(
            self._saveloc_entry,
            "Where to save outputs.  Defaults to f_loc when left blank.",
        )

        # workspace
        lbl3, self._ws_entry, self._ws_var = make_labeled_entry(
            left, "Workspace:", width=44
        )
        lbl3.grid(row=3, column=0, sticky="w", padx=(0, 4))
        self._ws_entry.grid(row=3, column=1, sticky="ew")
        self.bind_hover(
            self._ws_entry,
            "Run-history workspace.  Default: <f_loc>/.pycamset_workspace",
        )

        left.columnconfigure(1, weight=1)
        self._floc_var.trace_add("write", self._on_floc_change)

        # ── Options section ────────────────────────────────────────────
        ttk.Separator(left, orient=tk.HORIZONTAL).grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=6
        )
        make_section_label(left, "Options").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(0, 2)
        )

        self._save_cb, self._save_var = make_labeled_checkbox(
            left, "Save artefacts", default=True
        )
        self._save_cb.grid(row=6, column=0, columnspan=2, sticky="w")
        self.bind_hover(self._save_cb, "Cache detection pickle and camera-set files to disk.")

        self._draw_cb, self._draw_var = make_labeled_checkbox(
            left, "Draw detections", default=False
        )
        self._draw_cb.grid(row=7, column=0, columnspan=2, sticky="w")
        self.bind_hover(self._draw_cb, "Render detection overlays as each image is processed.")

        self._hd_cb, self._hd_var = make_labeled_checkbox(
            left, "High distortion mode", default=False
        )
        self._hd_cb.grid(row=8, column=0, columnspan=2, sticky="w")
        self.bind_hover(
            self._hd_cb,
            "Re-detect with undistorted images after an initial calibration pass.",
        )

        # n_lim
        lbl_n, self._nlim_entry, self._nlim_var = make_labeled_entry(
            left, "Max images per camera (n_lim):", default="", width=10
        )
        lbl_n.grid(row=9, column=0, sticky="w", padx=(0, 4))
        self._nlim_entry.grid(row=9, column=1, sticky="w")
        self.bind_hover(self._nlim_entry, "Max images to use per camera (leave blank = no limit).")

        # threads
        lbl_t, self._threads_entry, self._threads_var = make_labeled_entry(
            left, "Threads (blank = auto):", default="", width=10
        )
        lbl_t.grid(row=10, column=0, sticky="w", padx=(0, 4))
        self._threads_entry.grid(row=10, column=1, sticky="w")
        self.bind_hover(
            self._threads_entry,
            f"Thread count for optimisation.  Auto = min(max(1, cpu-2), 20)"
            f" = {min(max(1, cpu_count() - 2), 20)} on this machine.",
        )

        # fixed_params (JSON)
        lbl_fp, self._fp_entry, self._fp_var = make_labeled_entry(
            left, "Fixed params (JSON):", default="", width=32
        )
        lbl_fp.grid(row=11, column=0, sticky="w", padx=(0, 4))
        self._fp_entry.grid(row=11, column=1, sticky="ew")
        self.bind_hover(
            self._fp_entry,
            'JSON dict to lock camera params, e.g. {"cam0": "int"}.  Leave blank for none.',
        )

        # problem_options (JSON)
        lbl_po, self._po_entry, self._po_var = make_labeled_entry(
            left, "Problem options (JSON):", default="", width=32
        )
        lbl_po.grid(row=12, column=0, sticky="w", padx=(0, 4))
        self._po_entry.grid(row=12, column=1, sticky="ew")
        self.bind_hover(
            self._po_entry,
            "JSON dict of options forwarded to the optimiser.  Leave blank for defaults.",
        )

        # ── Bottom action row ──────────────────────────────────────────
        btn_row = tk.Frame(left)
        btn_row.grid(row=13, column=0, columnspan=3, sticky="w", pady=10)

        ttk.Button(btn_row, text="▶  Run Phase 0", command=self._run_phase0).pack(
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
    # Callbacks
    # ------------------------------------------------------------------

    def _browse_floc(self) -> None:
        path = filedialog.askdirectory(title="Select image folder")
        if path:
            self._floc_var.set(path)

    def _browse_saveloc(self) -> None:
        path = filedialog.askdirectory(title="Select save location")
        if path:
            self._saveloc_var.set(path)

    def _on_floc_change(self, *_) -> None:
        """Auto-fill workspace path once f_loc is set."""
        floc = self._floc_var.get().strip()
        if floc and not self._ws_var.get().strip():
            self._ws_var.set(str(Path(floc) / ".pycamset_workspace"))

    def _collect_params(self) -> Optional[dict]:
        """Validate form inputs and return a params dict, or None on error."""
        floc = self._floc_var.get().strip()
        if not floc:
            messagebox.showerror("Validation Error", "Image folder (f_loc) is required.")
            return None

        save_loc = self._saveloc_var.get().strip() or None

        n_lim_str = self._nlim_var.get().strip()
        n_lim = None
        if n_lim_str:
            try:
                n_lim = int(n_lim_str)
            except ValueError:
                messagebox.showerror("Validation Error", "n_lim must be an integer.")
                return None

        threads_str = self._threads_var.get().strip()
        threads = None
        if threads_str:
            try:
                threads = int(threads_str)
            except ValueError:
                messagebox.showerror("Validation Error", "Threads must be an integer.")
                return None

        fp_str = self._fp_var.get().strip()
        fixed_params = None
        if fp_str:
            try:
                fixed_params = json.loads(fp_str)
            except json.JSONDecodeError as exc:
                messagebox.showerror("Validation Error", f"Fixed params JSON: {exc}")
                return None

        po_str = self._po_var.get().strip()
        problem_options = None
        if po_str:
            try:
                problem_options = json.loads(po_str)
            except json.JSONDecodeError as exc:
                messagebox.showerror("Validation Error", f"Problem options JSON: {exc}")
                return None

        return {
            "f_loc": floc,
            "save_loc": save_loc,
            "save": self._save_var.get(),
            "draw": self._draw_var.get(),
            "n_lim": n_lim,
            "threads": threads,
            "high_distortion": self._hd_var.get(),
            "fixed_params": fixed_params,
            "problem_options": problem_options,
        }

    def _run_phase0(self) -> None:
        """Validate inputs and run D0.1–D0.3 diagnostics in a daemon thread."""
        params = self._collect_params()
        if params is None:
            return

        # Update workspace manager
        ws_str = self._ws_var.get().strip()
        if not ws_str:
            ws_str = str(Path(params["f_loc"]) / ".pycamset_workspace")
        self._workspace_mgr.workspace_path = Path(ws_str)
        self._workspace_mgr.ensure_dirs()

        self.terminal_clear()
        self.terminal_append("=== Phase 0: Input Validation & Setup ===")
        self.terminal_append(f"Image folder : {params['f_loc']}")
        self.terminal_append(f"Save         : {params['save']}")
        self.terminal_append(f"draw         : {params['draw']}")
        self.terminal_append(f"n_lim        : {params['n_lim']}")
        self.terminal_append(f"threads      : {params['threads'] or 'auto'}")
        self.terminal_append(f"high_distort : {params['high_distortion']}")
        self.terminal_append("Running diagnostics…")

        def _worker() -> None:
            diagnostics: dict = {}
            error_msg: Optional[str] = None
            try:
                f_loc = Path(params["f_loc"])

                if _PYCAMSET_OK:
                    # D0.1 — camera sub-folder count using existing helper
                    cam_folders = get_subfolder_names(f_loc, return_full_path=True)
                    cam_names = get_subfolder_names(f_loc)
                    diagnostics["D0.1_n_cameras"] = len(cam_folders)
                    self.after(
                        0, self.terminal_append,
                        f"D0.1  Camera sub-folders : {len(cam_folders)}")

                    # D0.2 — images per camera using existing glob_ims
                    imgs_per_cam: dict[str, int] = {}
                    for name, path in zip(cam_names, cam_folders):
                        imgs_per_cam[name] = len(glob_ims(path))
                    diagnostics["D0.2_images_per_camera"] = imgs_per_cam
                    for name, n in imgs_per_cam.items():
                        self.after(0, self.terminal_append, f"        {name}: {n} images")

                    # D0.3 — consistency check using existing sanitise_input_images
                    try:
                        sanitise_input_images(cam_folders)
                        consistent = True
                    except ValueError:
                        consistent = False
                    diagnostics["D0.3_consistent"] = consistent
                    self.after(0, self.terminal_append,
                               f"D0.3  Image count consistent : "
                               f"{'Yes ✓' if consistent else 'No ✗ (WARNING)'}")
                else:
                    # Fallback: plain filesystem scan
                    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
                    cam_folders = [p for p in Path(params["f_loc"]).iterdir() if p.is_dir()]
                    diagnostics["D0.1_n_cameras"] = len(cam_folders)
                    imgs_per_cam = {
                        cf.name: len([p for p in cf.iterdir() if p.suffix.lower() in exts])
                        for cf in cam_folders
                    }
                    diagnostics["D0.2_images_per_camera"] = imgs_per_cam
                    counts = list(imgs_per_cam.values())
                    diagnostics["D0.3_consistent"] = len(set(counts)) <= 1
                    self.after(0, self.terminal_append, "(pyCamSet not found — basic scan only)")

                self.after(0, self.terminal_append, "Phase 0 diagnostics complete.")

            except Exception as exc:
                error_msg = str(exc)
                self.after(0, self.terminal_append, f"ERROR: {exc}")

            # Persist run
            run_id = make_run_id()
            metadata = {
                "run_id": run_id,
                "phase": "phase0",
                "params": params,
                "diagnostics": diagnostics,
                "error": error_msg,
            }
            meta_path = self._workspace_mgr.save_run("phase0", run_id, metadata)
            self.after(0, self.terminal_append, f"Run saved → {meta_path}")

            if self._diagnostics_tab is not None:
                self.after(0, self._diagnostics_tab.refresh)

        threading.Thread(target=_worker, daemon=True).start()

    def _open_diagnostics(self) -> None:
        """Switch to the Phase 0 Diagnostics tab."""
        if self._diagnostics_tab is not None:
            self._diagnostics_tab.refresh()
        for i in range(self._notebook.index("end")):
            if self._notebook.tab(i, "text") == TAB_PHASE0_DIAG:
                self._notebook.select(i)
                return

    def _continue_to_next(self) -> None:
        """Write handoff.json and switch to Phase 1."""
        runs = self._workspace_mgr.load_runs("phase0")
        if not runs:
            messagebox.showinfo("No runs", "Run Phase 0 first.")
            return
        self._workspace_mgr.write_handoff({"phase": "phase0", "runs": [runs[-1]]})
        for i in range(self._notebook.index("end")):
            if self._notebook.tab(i, "text") == TAB_PHASE1:
                self._notebook.select(i)
                return


# ---------------------------------------------------------------------------
# Phase 0 Diagnostics tab
# ---------------------------------------------------------------------------

class Phase0DiagnosticsTab(tk.Frame, InfoHoverMixin):
    """Phase 0 Diagnostics — run comparison view.

    Left pane: :class:`~pyCamSet.gui.shared_functions.RunSelectorFrame` with
    multi-select.  Right pane: scrollable table of D0.1–D0.3 results for the
    selected run(s).  Defaults to the most recent 1–3 runs.
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

        # ── Right: diagnostics display ─────────────────────────────────
        right = tk.Frame(pane)
        pane.add(right, minsize=420)

        make_section_label(right, "Phase 0 Diagnostics").pack(
            anchor="w", padx=4, pady=(4, 2)
        )
        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=4)

        self._diag_frame = tk.Frame(right)
        self._diag_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── Bottom: continue button ────────────────────────────────────
        btn_row = tk.Frame(self)
        btn_row.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=4)
        make_continue_button(btn_row, self._continue_to_next).pack(side=tk.RIGHT)

        self.refresh()

    def refresh(self) -> None:
        """Reload runs from disk and re-render the diagnostics panel."""
        runs = self._workspace_mgr.load_runs("phase0")
        self._run_selector.refresh(runs)
        self._render_diagnostics(self._run_selector.get_selected())

    def _on_run_selected(self, selected: list[dict]) -> None:
        self._render_diagnostics(selected)

    def _render_diagnostics(self, runs: list[dict]) -> None:
        for w in self._diag_frame.winfo_children():
            w.destroy()

        if not runs:
            tk.Label(
                self._diag_frame,
                text="Select one or more runs from the list to compare.",
                fg="gray",
                justify="center",
            ).pack(pady=30)
            return

        # Scrollable inner content
        canvas = tk.Canvas(self._diag_frame, highlightthickness=0)
        vsb = ttk.Scrollbar(self._diag_frame, orient=tk.VERTICAL, command=canvas.yview)
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

            diag = run.get("diagnostics", {})
            params = run.get("params", {})

            entries = [
                ("f_loc", params.get("f_loc", "—")),
                ("D0.1  Camera sub-folders", diag.get("D0.1_n_cameras", "—")),
                ("D0.3  Image count consistent", diag.get("D0.3_consistent", "—")),
            ]
            for cam, n in diag.get("D0.2_images_per_camera", {}).items():
                entries.append((f"  D0.2  {cam}", f"{n} images"))

            if run.get("error"):
                entries.append(("Error", run["error"]))

            for key, val in entries:
                tk.Label(inner, text=f"{key}:", anchor="w", fg="#555").grid(
                    row=row, column=0, sticky="w", padx=(8, 4)
                )
                tk.Label(inner, text=str(val), anchor="w").grid(
                    row=row, column=1, sticky="w"
                )
                row += 1

            ttk.Separator(inner, orient=tk.HORIZONTAL).grid(
                row=row, column=0, columnspan=2, sticky="ew", pady=4
            )
            row += 1

    def _continue_to_next(self) -> None:
        selected = self._run_selector.get_selected()
        if not selected:
            runs = self._workspace_mgr.load_runs("phase0")
            selected = runs[-1:] if runs else []
        if not selected:
            messagebox.showinfo("No runs", "No Phase 0 runs available.")
            return
        self._workspace_mgr.write_handoff({"phase": "phase0", "runs": selected})
        for i in range(self._notebook.index("end")):
            if self._notebook.tab(i, "text") == TAB_PHASE1:
                self._notebook.select(i)
                return
