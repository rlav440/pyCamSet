"""
Phase 1: finding the calibration target in every image.

Reads an image folder, detects the target in each camera's images, and saves
the detections plus the diagnostics that say whether they are worth
calibrating from.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from pyCamSet.workflow.detections import (
    detection_cache_name,
    selected_camera_folders,
    staged_camera_root,
)
from pyCamSet.workflow.logs import LogFn, captured_output, discard
from pyCamSet.workflow.targets import target_of_params
from pyCamSet.workflow.workspace import (
    WorkspaceManager,
    copy_file,
    count_images_in_folder,
    make_run_id,
    path_exists,
    workspace_path_for,
)

_LOG = logging.getLogger(__name__)

try:
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile,
        validate_detections,
    )

    BACKEND_OK = True
except (ImportError, OSError) as exc:
    detect_datapoints_in_imfile = None
    validate_detections = None
    BACKEND_OK = False
    # Said out loud: collapsing the cause into a boolean is how a renamed
    # backend symbol came to look like a broken install.
    _LOG.warning("Phase 1 detection backend unavailable: %s", exc)


def run(params: dict,
        workspace: Optional[WorkspaceManager] = None,
        log: LogFn = discard) -> dict:
    """
    Detect the calibration target through an image folder.

    :param params: the phase's settings, as :mod:`pyCamSet.workflow.params`
        validates them
    :param workspace: where to save the run; defaults to the workspace of the
        image folder in *params*
    :param log: what to call with each line of output
    :return: the run's metadata record, saved to the workspace
    """
    if workspace is None or workspace.workspace_path is None:
        workspace = WorkspaceManager(workspace_path_for(params["f_loc"]))

    diagnostics: dict = {}
    error: Optional[str] = None
    detections_source: Optional[Path] = None

    with captured_output(log):
        try:
            detections_source, diagnostics = _detect(params, log)
        except Exception as exc:
            error = str(exc)
            diagnostics["error"] = error

    run_id = make_run_id()
    metadata = {
        "run_id": run_id,
        "phase": "phase1",
        "params": params,
        "diagnostics": diagnostics,
        "error": error,
    }
    workspace.save_run("phase1", run_id, metadata)

    run_dir = workspace.run_dir("phase1", run_id)
    if detections_source is None:
        candidate = Path(params["f_loc"]) / "detected_datapoints.pickle"
        detections_source = candidate if path_exists(candidate) else None

    if detections_source is not None:
        saved = run_dir / "detected_datapoints.pickle"
        try:
            copy_file(detections_source, saved)
        except OSError as exc:
            metadata["error"] = (
                f"Could not save run-local detected_datapoints.pickle: {exc}")
            workspace.save_run("phase1", run_id, metadata)
            log(f"ERROR: {metadata['error']}")
            return metadata
        metadata.setdefault("artifacts", {})[
            "detected_datapoints_pickle"] = str(saved)
        workspace.save_run("phase1", run_id, metadata)
        log(f"Artifact saved: {saved}")

    log(f"Run saved: {run_id}")
    return metadata


def _detect(params: dict, log: LogFn) -> tuple[Optional[Path], dict]:
    """Run the detection pass and compute its diagnostics."""
    if not BACKEND_OK:
        raise RuntimeError("pyCamSet detection modules are not importable.")

    f_loc = Path(params["f_loc"])
    upscale_factor = params.get("upscale_factor", 1)

    cam_folders = selected_camera_folders(f_loc, params.get("selected_cameras"))
    cam_names = [folder.name for folder in cam_folders]
    cam_img_counts = {folder.name: count_images_in_folder(folder)
                      for folder in cam_folders}
    log(f"1a  Camera sub-folders: {cam_names}")
    if upscale_factor > 1:
        log(f"1a  Upscale factor: {upscale_factor}x")

    if not cam_folders:
        raise RuntimeError("No selected camera sub-folders found.")
    counts = list(cam_img_counts.values())
    if any(count <= 0 for count in counts) or len(set(counts)) != 1:
        raise RuntimeError(
            "Camera folders must contain equal non-zero image counts.")

    target = target_of_params(params)
    cache_name = detection_cache_name(upscale_factor)

    with staged_camera_root(f_loc, cam_folders, log, "pycamset_phase1_") as root:
        detections, cam_res = detect_datapoints_in_imfile(
            f_loc=root,
            calibration_target=target,
            caching=params["caching"],
            draw=False,
            n_lim=params["n_lim"],
            upscale_factor=upscale_factor,
        )
        log("1b  Detection complete.")

        # The cache lands beside the images the pass read, which is the
        # staging folder when there was one.  Bring it back to the image
        # folder, where the next phase and a re-run both look for it.
        cached = root / cache_name
        detections_source = None
        if path_exists(cached):
            detections_source = f_loc / cache_name
            if root != f_loc:
                copy_file(cached, detections_source)

    validate_detections(detections, target)
    log("1c  Validation complete.")

    diagnostics = _diagnostics(
        detections, cam_res, target, cam_img_counts, params["n_lim"], log)
    log("Phase 1 complete.")
    return detections_source, diagnostics


def _diagnostics(detections, cam_res, target, cam_img_counts,
                 n_lim: Optional[int], log: LogFn) -> dict:
    """The D1 series: how much of the target each camera actually saw."""
    diagnostics: dict = {}
    try:
        diagnostics["D1.1_total_detections"] = _total_detections(detections)

        detection_rate, completeness = _rate_and_completeness(
            detections, target, cam_img_counts, n_lim)
        diagnostics["D1.2_detection_rate"] = detection_rate
        diagnostics["D1.3_board_completeness"] = completeness
        for cam in detections.cam_names:
            log(f"D1.2/D1.3  {cam}: "
                f"rate={detection_rate.get(cam, 0.0) * 100.0:.1f}% "
                f"completeness={completeness.get(cam, 0.0) * 100.0:.1f}%")

        features = detections.features_per_im_per_cam()
        diagnostics["D1.4_features_matrix"] = features.tolist()

        coverage = _spatial_coverage(detections, cam_res)
        if coverage is not None:
            diagnostics["D1.6_spatial_coverage"] = coverage

        min_features = int(np.min(features[features > 0])) if np.any(features > 0) else 0
        diagnostics["D1.7_min_features"] = min_features
        log(f"D1.7  Min features in any image-camera: {min_features}")

        diagnostics["cam_names"] = detections.cam_names
        diagnostics["n_images"] = int(detections.max_ims)
    except Exception as exc:
        log(f"  (partial diagnostics: {exc})")
    return diagnostics


def _total_detections(detections) -> dict[str, int]:
    """How many points each camera contributed in total."""
    totals: dict[str, int] = {}
    for cam_detection in detections.get_cam_list():
        data = cam_detection.get_data()
        if data is None or len(data) == 0:
            continue
        totals[detections.cam_names[int(data[0, 0])]] = len(data)
    return totals


def _corners_per_face(target) -> int:
    """How many points one face of the target holds.

    PuzzleBoard's point data spans the whole 501x501 virtual code-lookup
    field, not the printed window, so its printed size has to be asked for
    directly.  Every other target's point data is the printed target.
    """
    if target.__class__.__name__ == "PuzzleBoard":
        return int(target.num_squares_x * target.num_squares_y)
    return int(target.point_data.shape[-2])


def _rate_and_completeness(
    detections, target, cam_img_counts, n_lim: Optional[int],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Per camera: the fraction of images with a detection, and how much of the
    target those detections covered.
    """
    corners_per_face = _corners_per_face(target)
    detection_rate: dict[str, float] = {}
    completeness: dict[str, float] = {}

    for cam_detection in detections.get_cam_list():
        data = cam_detection.get_data()
        if data is None or len(data) == 0:
            continue
        cam_name = detections.cam_names[int(data[0, 0])]

        expected = int(cam_img_counts.get(cam_name, detections.max_ims))
        if n_lim is not None:
            expected = min(expected, int(n_lim))
        expected = max(expected, 1)

        detected_images = 0
        per_image_fraction: list[float] = []

        for im_detection in cam_detection.get_image_list():
            datum = im_detection.get_data()
            if datum is None or len(datum) == 0:
                continue
            detected_images += 1

            id_cols = datum[:, 2:-2]
            if id_cols.ndim == 1:
                id_cols = id_cols.reshape(-1, 1)
            if id_cols.shape[1] <= 0:
                continue

            if id_cols.shape[1] == 1:
                # A point id and nothing else: one board.
                unique_points = len(np.unique(id_cols[:, 0]))
                per_image_fraction.append(unique_points / max(corners_per_face, 1))
            else:
                # Board id columns, then the point id: average over boards.
                board_cols, point_col = id_cols[:, :-1], id_cols[:, -1]
                board_fractions = []
                for board in np.unique(board_cols, axis=0):
                    mask = np.all(board_cols == board, axis=1)
                    unique_points = len(np.unique(point_col[mask]))
                    board_fractions.append(unique_points / max(corners_per_face, 1))
                if board_fractions:
                    per_image_fraction.append(float(np.mean(board_fractions)))

        detection_rate[cam_name] = float(detected_images) / float(expected)
        completeness[cam_name] = (
            float(np.mean(per_image_fraction)) if per_image_fraction else 0.0)

    return detection_rate, completeness


def _spatial_coverage(detections, cam_res) -> Optional[dict[str, float]]:
    """The fraction of each image the detected points span."""
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return None

    coverage: dict[str, float] = {}
    # Indexed by position rather than by the camera index inside the data,
    # which is unreadable for a camera that detected nothing.
    for cam_index, (cam_detection, res) in enumerate(
            zip(detections.get_cam_list(), cam_res)):
        cam_name = detections.cam_names[cam_index]
        data = cam_detection.get_data()
        if data is None or len(data) < 3:
            coverage[cam_name] = float("nan")
            continue
        try:
            hull_area = ConvexHull(data[:, -2:]).volume
        except Exception:
            coverage[cam_name] = float("nan")
            continue
        coverage[cam_name] = hull_area / (float(res[0]) * float(res[1]))
    return coverage
