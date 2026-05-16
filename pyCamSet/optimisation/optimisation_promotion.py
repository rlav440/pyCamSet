"""Purpose: Bridge retained optimisation trials into normal workspace phase outputs.

Status: Active helper for Optimisation-tab full mode and retained-run promotion.

Future: Replace the lightweight metadata copies if phase tabs gain shared service APIs.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyCamSet.gui.shared_functions import WorkspaceManager

_RUN_ID_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"


@dataclass(frozen=True)
class OptimisationPhaseContext:
    """Resolved workspace inputs needed to run phase 3 from optimisation trials."""

    workspace_path: Path
    phase2_run_id: str | None
    phase2_initial_camset: Path
    threads: int = 1


def latest_phase2_context(workspace_mgr: WorkspaceManager) -> OptimisationPhaseContext | None:
    """Resolve the latest saved phase 2 run for Optimisation full mode."""
    ws_path = workspace_mgr.workspace_path  # Read the workspace selected by the GUI.
    if ws_path is None:  # Without a workspace there is no phase 2 camset to start from.
        return None
    phase2_runs = workspace_mgr.load_runs("phase2")  # Reuse the normal workspace metadata loader.
    if not phase2_runs:  # Phase 3 semantics require an initial camset from phase 2.
        return None
    phase2_run = phase2_runs[-1]  # Match the phase tabs' convention of defaulting to latest.
    camset_path = (phase2_run.get("artifacts") or {}).get("initial_camset")  # Read the phase 2 artefact.
    if not camset_path:  # Metadata without a camset cannot seed template bundle adjustment.
        return None
    path = Path(camset_path)  # Normalise the path so existence checks are reliable.
    if not path.exists():  # Avoid starting a full-mode run that will fail every trial.
        return None
    return OptimisationPhaseContext(  # Return an immutable context for worker-thread callables.
        workspace_path=Path(ws_path),
        phase2_run_id=phase2_run.get("run_id"),
        phase2_initial_camset=path,
    )


def make_phase_callables(context: OptimisationPhaseContext):
    """Create phase 3 and phase 4 callables using the existing backend handlers."""
    from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment_with_stats
    from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler
    from pyCamSet.utils.saving import load_CameraSet

    def _options(max_nfev: int, outliers: str) -> dict[str, Any]:
        outlier_mode = "y" if outliers == "ask" else (outliers or "n")  # GUI cannot prompt in a worker.
        return {  # Mirror the existing phase-tab defaults.
            "verbosity": 0,
            "fixed_pose": 0,
            "ref_cam": 0,
            "ref_pose": 0,
            "outliers": outlier_mode,
            "max_nfev": int(max_nfev),
        }

    def _phase3(detection_payload: dict[str, Any], *, controls) -> dict[str, Any]:
        cams = load_CameraSet(context.phase2_initial_camset)  # Load the phase 2 initial camset.
        detections = detection_payload["detections"]  # Use the trial-specific detections.
        target = detection_payload.get("target")  # Use the exact target used during detection.
        if target is None:  # Full mode cannot safely rebuild arbitrary target instances here.
            raise RuntimeError("Detection payload is missing the calibration target.")
        handler = TemplateBundleHandler(  # Invoke the same handler used by the Phase 3 GUI tab.
            camset=cams,
            target=target,
            detection=detections,
            fixed_params=None,
            options=_options(controls.max_nfev_phase3, controls.outliers),
        )
        optimisation, out_cams, stats = run_bundle_adjustment_with_stats(  # Run bundle adjustment.
            handler,
            threads=context.threads,
        )
        final_rpe = float(stats.get("final_euclid", float("nan")))  # Match D3.6 semantics.
        return {  # Return the payload shape consumed by run_trial and metadata writers.
            "rpe": final_rpe,
            "camset": out_cams,
            "optimisation": optimisation,
            "stats": dict(stats),
            "source_phase2_run_id": context.phase2_run_id,
            "source_phase2_camset": str(context.phase2_initial_camset),
        }

    def _phase4(detection_payload: dict[str, Any], phase3_payload: dict[str, Any], *, controls) -> dict[str, Any]:
        phase3_cams = phase3_payload.get("camset")  # Phase 4 starts from the phase 3 camera set.
        if phase3_cams is None:  # Avoid silently fabricating a phase 4 source.
            raise RuntimeError("Phase 3 payload is missing a camset.")
        handler = SelfBundleHandler(  # Invoke the same handler used by the Phase 4 GUI tab.
            camset=phase3_cams,
            target=detection_payload.get("target"),
            detection=detection_payload["detections"],
            fixed_params=None,
            options=_options(controls.max_nfev_phase4, controls.outliers),
        )
        handler.set_from_templated_camset(phase3_cams)  # Match the existing Phase 4 initialisation.
        optimisation, out_cams, stats = run_bundle_adjustment_with_stats(  # Run self-calibration.
            handler,
            threads=context.threads,
        )
        final_rpe = float(stats.get("final_euclid", float("nan")))  # Match D4.3 final error.
        return {  # Return the payload shape consumed by run_trial and metadata writers.
            "rpe": final_rpe,
            "camset": out_cams,
            "optimisation": optimisation,
            "stats": dict(stats),
        }

    return _phase3, _phase4


def promote_retained_trial(workspace_mgr: WorkspaceManager, metadata_path: str | Path) -> dict[str, str]:
    """Promote one retained optimisation trial into phase-oriented workspace runs."""
    if workspace_mgr.workspace_path is None:  # Promotion needs a destination workspace.
        raise RuntimeError("Workspace path is not set.")
    metadata_file = Path(metadata_path)  # Normalise the retained trial metadata path.
    metadata = json.loads(metadata_file.read_text())  # Load the structured optimisation metadata.
    stage = (metadata.get("identity") or {}).get("success_stage")  # Decide which phases to promote.
    if stage not in {"phase3", "phase4"}:  # Only successful retained trials have enough artefacts.
        raise RuntimeError("Only successful phase3/phase4 trials can be promoted.")
    extra = metadata.get("extra") or {}  # Read optional artefact data written by the worker.
    artifacts = extra.get("artifacts") or {}  # Pull paths for detection and camset files.
    promoted: dict[str, str] = {}  # Track the workspace run ids for caller feedback.

    phase1_run_id = f"optimisation_phase1_{_make_run_id()}"  # Use a distinct normal phase 1 run id.
    phase1_pickle = _copy_artifact(  # Copy detections into the normal phase 1 run directory.
        workspace_mgr,
        "phase1",
        phase1_run_id,
        artifacts.get("detected_datapoints_pickle"),
        "detected_datapoints.pickle",
        required=True,
    )
    workspace_mgr.save_run(  # Save metadata after the artefact path is known.
        "phase1",
        phase1_run_id,
        {
            "run_id": phase1_run_id,
            "phase": "phase1",
            "params": {
                "f_loc": (metadata.get("paths") or {}).get("f_loc"),
                "target": metadata.get("target"),
                "detector_settings": metadata.get("detector_settings"),
                "source": "optimisation",
            },
            "diagnostics": {"promoted_from": str(metadata_file)},
            "error": None,
            "artifacts": {"detected_datapoints_pickle": str(phase1_pickle)},
        },
    )
    promoted["phase1"] = phase1_run_id  # Record the promoted phase 1 id.

    phase3_run_id = f"optimisation_phase3_{_make_run_id()}"  # Use a distinct normal phase 3 run id.
    phase3_camset = _copy_artifact(  # Copy the retained phase 3 camera set.
        workspace_mgr,
        "phase3",
        phase3_run_id,
        artifacts.get("phase3_camset"),
        "optimised_cameras.camset",
        required=True,
    )
    workspace_mgr.save_run(  # Save phase 3 metadata using the standard phase keys.
        "phase3",
        phase3_run_id,
        {
            "run_id": phase3_run_id,
            "phase": "phase3",
            "params": _promotion_params(metadata),
            "diagnostics": {
                "D3.6_final_euclid_px": (metadata.get("metrics") or {}).get("phase3_rpe"),
                "promoted_from": str(metadata_file),
            },
            "error": None,
            "inputs": {
                "phase1_run_id": phase1_run_id,
                "phase2_run_id": (extra.get("phase_sources") or {}).get("phase2_run_id"),
            },
            "artifacts": {
                "optimised_camset": str(phase3_camset),
                "phase1_detection_pickle_used": str(phase1_pickle),
                "phase2_initial_camset_used": (extra.get("phase_sources") or {}).get("phase2_initial_camset"),
            },
        },
    )
    promoted["phase3"] = phase3_run_id  # Record the promoted phase 3 id.

    if stage == "phase4":  # Only phase 4 successes should create phase 4 outputs.
        phase4_run_id = f"optimisation_phase4_{_make_run_id()}"  # Use a distinct normal phase 4 run id.
        phase4_camset = _copy_artifact(  # Copy the retained phase 4 camera set.
            workspace_mgr,
            "phase4",
            phase4_run_id,
            artifacts.get("phase4_camset"),
            "self_calibrated_cameras.camset",
            required=True,
        )
        workspace_mgr.save_run(  # Save phase 4 metadata using the standard phase keys.
            "phase4",
            phase4_run_id,
            {
                "run_id": phase4_run_id,
                "phase": "phase4",
                "params": _promotion_params(metadata),
                "diagnostics": {
                    "D4.3_final_euclid_px": (metadata.get("metrics") or {}).get("phase4_rpe"),
                    "promoted_from": str(metadata_file),
                },
                "error": None,
                "inputs": {"phase3_run_id": phase3_run_id},
                "artifacts": {
                    "self_calibrated_camset": str(phase4_camset),
                    "phase3_camset_used": str(phase3_camset),
                },
            },
        )
        promoted["phase4"] = phase4_run_id  # Record the promoted phase 4 id.

    return promoted  # Let the GUI surface concise success feedback.


def _promotion_params(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build phase-run params from optimisation metadata."""
    controls = dict(metadata.get("calibration_controls") or {})  # Preserve max_nfev/outlier choices.
    return {  # Keep enough structured context for downstream tabs/exporters.
        "f_loc": (metadata.get("paths") or {}).get("f_loc"),
        "target": dict(metadata.get("target") or {}),
        "detector_settings": dict(metadata.get("detector_settings") or {}),
        "problem_options": {
            "outliers": controls.get("outliers"),
            "max_nfev": controls.get("max_nfev_phase3"),
        },
        "source": "optimisation",
    }


def _copy_artifact(
    workspace_mgr: WorkspaceManager,
    phase: str,
    run_id: str,
    source: str | None,
    filename: str,
    *,
    required: bool,
) -> Path:
    """Copy one retained-trial artefact into a workspace phase-run directory."""
    if workspace_mgr.workspace_path is None:  # Keep type-checkers and callers honest.
        raise RuntimeError("Workspace path is not set.")
    phase_dir = f"{phase}_runs"  # Reuse the repository's standard phase directory naming.
    run_dir = workspace_mgr.workspace_path / phase_dir / run_id  # Compute the normal run directory.
    run_dir.mkdir(parents=True, exist_ok=True)  # Ensure the destination exists before copying.
    dest = run_dir / filename  # Use the standard artefact name for that phase.
    if not source:  # Missing source is an error for required promotion artefacts.
        if required:
            raise RuntimeError(f"Missing retained artefact for {phase}.")
        return dest
    src = Path(source)  # Normalise the source path from metadata.
    if not src.exists():  # Avoid writing metadata that points at a missing copied artefact.
        if required:
            raise RuntimeError(f"Retained artefact does not exist: {src}")
        return dest
    shutil.copy2(src, dest)  # Copy bytes rather than reserialising unknown backend objects.
    return dest


def _make_run_id() -> str:
    """Return a compact run id without importing Qt-backed GUI helpers."""
    stamp = time.strftime(_RUN_ID_TIMESTAMP_FORMAT, time.localtime())  # Use YYYYMMDDTHHMMSS ids.
    return f"{stamp}_{uuid.uuid4().hex[:8]}"  # Add entropy so repeated saves do not collide.


__all__ = [
    "OptimisationPhaseContext",
    "latest_phase2_context",
    "make_phase_callables",
    "promote_retained_trial",
]
