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
    from pyCamSet.utils.visualisation import visualise_calibration
except ImportError:  # pragma: no cover
    visualise_calibration = None

_LOGGER = logging.getLogger(__name__)

_IGNORED_VISUALISATION_WARNING_MODULES = (
    r"numpy\._core\.",
    r"pyvista\.core\.utilities\.",
)


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
    """Merge runs into one ordered list with a stable phase label."""
    merged: list[dict] = []
    for phase_name, runs in (("phase3", phase3_runs), ("phase4", phase4_runs)):
        for run in runs:
            copied = dict(run)
            copied.setdefault("phase", phase_name)
            copied["display_name"] = f"{copied.get('phase', phase_name)} | {copied.get('run_id', 'unknown')}"
            merged.append(copied)
    merged.sort(key=lambda r: str(r.get("run_id", "")))
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
    success_message = f"Opened Assess Calibration for run {run_id}."
    try:
        with warnings.catch_warnings():
            for module_pattern in _IGNORED_VISUALISATION_WARNING_MODULES:
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    module=module_pattern,
                )
            # Intentionally launches external matplotlib/pyvista windows.
            visualise_calibration(o_results, handler)
    except Exception as exc:
        msg = str(exc).strip()
        msg_lower = msg.lower() if msg else ""
        if not msg or ("event loop" in msg_lower and "already running" in msg_lower):
            _LOGGER.debug(
                "Suppressed visualise_calibration exception for run %s (%s).",
                run_id,
                msg or "<blank>",
                exc_info=True,
            )
            return True, success_message
        return False, f"Could not run visualise_calibration: {msg or str(exc)}"

    return True, success_message
