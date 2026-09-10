"""Assess Calibration helpers used by Phase 3/4 diagnostics.

This module intentionally avoids embedded Qt figure rendering. It selects one
run and launches native matplotlib/pyvista windows through
``pyCamSet.utils.visualisation.visualise_calibration``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import logging
import warnings

try:
    from pyCamSet.utils.saving import load_CameraSet
except ImportError:  # pragma: no cover
    load_CameraSet = None

try:
    from pyCamSet.utils.visualisation import (
        visualise_calibration,
        visualise_calibration_open3d,
        render_calibration_pyvista_png,
    )
except ImportError:  # pragma: no cover
    visualise_calibration = None
    visualise_calibration_open3d = None
    render_calibration_pyvista_png = None

from pyCamSet.gui.viewer_process import run_viewer, spawn_viewer

_LOGGER = logging.getLogger(__name__)

_IGNORED_VISUALISATION_WARNING_MODULES = (
    r"numpy\._core\.",
    r"pyvista\.core\.utilities\.",
)


def canonical_phase_tag(phase: Optional[str]) -> str:
    """Return a normalised, lower-case phase tag.

    :param phase: Raw phase string.  ``None`` or empty returns ``"unknown"``.
    :returns: Normalised phase string — no legacy aliasing is applied.
    """
    if not phase:
        return "unknown"
    return str(phase).strip().lower()


def resolve_run_camset_artifact(run: dict) -> Optional[Path]:
    """Return the first existing camset path from a run's artifacts."""
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
    """Merge runs into one ordered list (oldest-first) with a stable phase label.

    Sorts by each run's ``_recency_ts`` (set by WorkspaceManager.load_runs -- see its
    docstring) rather than a lexicographic sort of ``run_id``: run_id formats differ
    across producers (e.g. this GUI vs. a headless script) and are not reliably
    orderable as plain strings, which previously made a stale/older run look "most
    recent" whenever both formats coexisted in the same workspace. Falls back to 0.0
    for any run missing ``_recency_ts`` (e.g. a hand-built dict in a test) so sorting
    never raises -- such entries sort as oldest rather than raising.
    """
    merged: list[dict] = []
    for phase_name, runs in (("phase3", phase3_runs), ("phase4", phase4_runs)):
        for run in runs:
            copied = dict(run)
            copied.setdefault("phase", phase_name)
            copied["display_name"] = f"{copied.get('phase', phase_name)} | {copied.get('run_id', 'unknown')}"
            merged.append(copied)
    merged.sort(key=lambda r: r.get("_recency_ts", 0.0))
    return merged


def select_latest_visualisation_run(selected_runs: list[dict], all_runs: list[dict]) -> Optional[dict]:
    """Pick exactly one run for visualisation, preferring the most recent selected."""
    if not selected_runs:
        return None

    index_map = {id(run): idx for idx, run in enumerate(all_runs)}
    ordered = sorted(
        selected_runs,
        key=lambda r: (
            index_map.get(id(r), 10**9),
            str(r.get("run_id", "")),
        ),
    )
    return ordered[-1]


def _build_o_results(cam_set: Any) -> Optional[dict[str, np.ndarray]]:
    err = getattr(cam_set, "calibration_result", None)
    x = getattr(cam_set, "calibration_params", None)
    if err is None or x is None:
        return None

    err_arr = np.asarray(err, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if err_arr.size == 0 or x_arr.size == 0:
        return None
    return {"err": err_arr, "x": x_arr}


def observation_residual_xy(residuals: Any, handler: Any) -> np.ndarray:
    """Return only the two-scalar-per-observation residual segment.

    The optimisation backend may append one-dimensional lockbox-prior
    residuals after the reprojection residuals. Diagnostics must not reshape
    those priors into pixel pairs: odd camera counts would crash and even
    counts would contaminate the reported per-camera reprojection metric.
    """
    values = np.asarray(residuals, dtype=float).reshape(-1)
    base_count = int(getattr(handler, "get_base_residual_count", lambda: 0)())
    if base_count <= 0:
        base_count = values.size
    if base_count > values.size or base_count % 2:
        raise ValueError(
            "Invalid reprojection residual segment length: "
            f"base_count={base_count}, total_count={values.size}"
        )
    return values[:base_count].reshape(-1, 2)


def launch_visualise_calibration_for_run(run: dict) -> tuple[bool, str]:
    """Launch native matplotlib/pyvista windows via visualise_calibration()."""
    if load_CameraSet is None or visualise_calibration is None:
        return False, "visualise_calibration dependencies are unavailable."

    camset_path = resolve_run_camset_artifact(run)
    if camset_path is None:
        return False, "Selected run has no readable camset artifact."

    try:
        cams = load_CameraSet(camset_path)
    except Exception as exc:  # pragma: no cover
        return False, f"Could not load camset: {exc}"

    handler = getattr(cams, "calibration_handler", None)
    if handler is None:
        return False, "Loaded camset does not contain calibration handler data."

    o_results = _build_o_results(cams)
    if o_results is None:
        return False, "Loaded camset does not contain calibration optimisation results."

    run_id = run.get("run_id", "unknown")

    # Everything above is checked here, in the GUI process, so the usual
    # failures still reach the user as a dialog.  The drawing itself is not:
    # see visualise_camset for why it cannot share a process with Qt.
    ok, detail = spawn_calibration_viewer(camset_path)
    if not ok:
        return False, detail
    return True, f"Opened Assess Calibration for run {run_id} in a new window."


def spawn_calibration_viewer(camset_path: Path) -> tuple[bool, str]:
    """
    Draw a calibration in a process of its own.

    :param camset_path: the ``.camset`` file to draw
    :return: whether the viewer was started, and what to say if it was not
    """
    return spawn_viewer("pyCamSet.utils.visualise_camset", [str(camset_path)])


def launch_visualise_calibration_open3d_for_run(
    run: dict, output_widget=None
) -> tuple[bool, str]:
    """Launch Open3D calibration visualisation for *run*.

    :param run: Run metadata dict with an artifacts section containing a camset path.
    :param output_widget: Optional Qt QLabel.  When provided the result is
        rendered offscreen and embedded; otherwise an Open3D native window is opened.
    :returns: ``(success, message)`` tuple.
    """
    if load_CameraSet is None or visualise_calibration_open3d is None:
        return False, "Open3D visualisation dependencies are unavailable."

    camset_path = resolve_run_camset_artifact(run)
    if camset_path is None:
        return False, "Selected run has no readable camset artifact."

    try:
        cams = load_CameraSet(camset_path)
    except Exception as exc:  # pragma: no cover
        return False, f"Could not load camset: {exc}"

    handler = getattr(cams, "calibration_handler", None)
    if handler is None:
        return False, "Loaded camset does not contain calibration handler data."

    o_results = _build_o_results(cams)
    if o_results is None:
        return False, "Loaded camset does not contain calibration optimisation results."

    return visualise_calibration_open3d(o_results, handler, output_widget=output_widget)


def launch_save_pyvista_png_for_run(run: dict, file_path: Path) -> tuple[bool, str]:
    """Perform offscreen PyVista PNG export for a given run.

    :param run: Run metadata dict with an artifacts section containing a camset path.
    :param file_path: Target file path for the exported PNG image.
    :returns: ``(success, message)`` tuple.
    """
    if load_CameraSet is None or render_calibration_pyvista_png is None:
        return False, "PyVista PNG export dependencies are unavailable."

    camset_path = resolve_run_camset_artifact(run)
    if camset_path is None:
        return False, "Selected run has no readable camset artifact."

    try:
        cams = load_CameraSet(camset_path)
    except Exception as exc:  # pragma: no cover
        return False, f"Could not load camset: {exc}"

    handler = getattr(cams, "calibration_handler", None)
    if handler is None:
        return False, "Loaded camset does not contain calibration handler data."

    o_results = _build_o_results(cams)
    if o_results is None:
        return False, "Loaded camset does not contain calibration optimisation results."

    # Rendered out of process like everything else that touches VTK.
    # off_screen=True still builds a vtkCocoaRenderWindow on macOS, so it is
    # no safer inside the GUI than a visible one; this one is waited on
    # because the caller wants the file, not a window.
    ok, detail = run_viewer(
        "pyCamSet.utils.visualise_camset",
        [str(camset_path), "--png", str(file_path)],
    )
    if not ok:
        return False, detail
    return True, detail or f"Saved {file_path}."
