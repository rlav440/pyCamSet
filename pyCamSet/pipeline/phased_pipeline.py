"""
Purpose: Phased calibration pipeline for pyCamSet.
         Wraps the existing calibration primitives (detect_datapoints_in_imfile,
         run_initial_calibration, run_stereo_calibration, SelfBundleHandler,
         run_bundle_adjustment) into six discrete, independently-cacheable phases
         plus a single run_pipeline() orchestrator.
         Each phase accepts and returns a plain dict so results can be persisted,
         inspected, and resumed at any boundary.
Status:  Working
Future:  Add CLI support via run_pipeline() once all phases are implemented.
"""

import logging                                               # for per-phase status messages
import pickle                                                # for detection cache serialisation
from pathlib import Path                                     # for filesystem path handling
from datetime import datetime                                # for run-timestamp logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple  # type annotations

import cv2 as cv                                             # for image I/O and resize
import numpy as np                                           # for numerical operations

# pyCamSet detection containers — direct import
from pyCamSet.calibration_targets import TargetDetection, ImageDetection  # detection containers
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler  # self-calibration handler
from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment  # bundle adjustment entry point

# Pipeline helpers (file I/O and plots) from the pyCamSet pipeline package
from pyCamSet.pipeline.pipeline_cache import (               # file I/O utilities
    phase_cache_path, cache_exists,                          # path building and existence check
    save_pickle, load_pickle,                                # pickle helpers for detection cache
    save_json, load_json,                                    # JSON helpers for skip indices / stats
    save_csv, load_csv,                                      # CSV helpers for error tables
    save_camset, load_camset,                                # CamSet helpers for calibration results
    title_to_filename,                                       # title → filename stem conversion
)
from pyCamSet.pipeline.pipeline_plots import (               # plot / summary utilities
    plot_error_histogram,                                    # histogram of reprojection errors
    plot_per_camera_errors,                                  # bar chart of per-camera means
    save_numeric_summary,                                    # numeric dict → .csv
    plot_residual_clusters,                                  # 2-D residual cluster plot
    plot_coverage_scatter,                                   # per-camera coverage scatter
    plot_camera_arrangement,                                 # pyvista camera-arrangement screenshot
)


# ──────────────────────────────────────────────────────────────────────────────
#  Private image helpers — replace calibria.detection equivalents
# ──────────────────────────────────────────────────────────────────────────────

def _images_in_folder(folder: Path) -> List[Path]:
    """
    Return a sorted list of image file paths directly within *folder* (non-recursive).
    Uses pyCamSet's glob_ims_local to find supported image formats.

    :param folder: Directory to search for images.
    :return:       Sorted list of Path objects for images in *folder*.
    """
    from pyCamSet.utils.general_utils import glob_ims_local  # local image glob utility
    return sorted(glob_ims_local(folder))                    # sort for deterministic ordering


def _load_and_normalize(path: Path) -> Optional[np.ndarray]:
    """
    Load an image at any bit depth and normalise to uint8 [0, 255].
    Returns None if the file cannot be read (safe wrapper around cv2.imread).

    :param path: Path to the image file.
    :return:     uint8 numpy array, or None on read failure.
    """
    img = cv.imread(str(path), cv.IMREAD_UNCHANGED)          # read at native bit depth
    if img is None:                                          # imread returns None on failure
        return None                                          # signal failure to caller
    if img.dtype == np.uint8:                                # already 8-bit — no conversion needed
        return img                                           # return as-is
    img_f = img.astype(np.float64)                           # promote to float for normalisation
    img_min, img_max = img_f.min(), img_f.max()              # dynamic range
    if img_max > img_min:                                    # avoid division by zero on flat images
        img_f = (img_f - img_min) / (img_max - img_min) * 255.0  # scale to [0, 255]
    return img_f.astype(np.uint8)                            # convert to uint8 and return


# ──────────────────────────────────────────────────────────────────────────────
#  Shared aggregation helper
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate_errors(
    per_image_errors: List[Dict],
) -> Tuple[Dict[str, float], float]:
    """
    Aggregate per-image error dicts into per-camera means and a global mean.
    Returns (per_camera_errors, global_mean_error).
    Each entry in *per_image_errors* must have 'camera' and 'mean_error' keys.

    :param per_image_errors: List of dicts with 'camera' and 'mean_error' keys.
    :return:                 (per_camera_errors, global_mean_error).
    """
    per_camera_sums: Dict[str, List[float]] = {}             # {cam: [error, ...]}
    for entry in per_image_errors:                           # group errors by camera name
        per_camera_sums.setdefault(entry['camera'], []).append(entry['mean_error'])
    per_camera_errors = {cam: float(np.mean(vals))           # mean per camera
                         for cam, vals in per_camera_sums.items()}
    all_errors = [e['mean_error'] for e in per_image_errors]  # flat list across all cameras
    global_mean = float(np.mean(all_errors)) if all_errors else float('nan')  # global mean
    return per_camera_errors, global_mean                    # return both aggregates


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helpers (edited copies of pyCamSet calibration_calibrator functions)
# ──────────────────────────────────────────────────────────────────────────────

def _sanitise_input_images(cam_folders: List[Path]) -> None:
    """
    Check that all camera folders contain the same number of images.
    Raises ValueError if counts differ.

    Edited copy of sanitise_input_images() from
    pyCamSet/calibration/camera_calibrator.py.  Change: uses _images_in_folder()
    instead of pyCamSet's glob_ims(); includes counts in error message.

    :param cam_folders: Resolved Path objects, one per camera subfolder.
    :raises ValueError: If image counts differ across any two camera folders.
    """
    counts = {f.name: len(_images_in_folder(f)) for f in cam_folders}  # images per camera
    if len(set(counts.values())) > 1:                        # unequal counts detected
        raise ValueError(                                    # descriptive error message
            f"Unequal image counts across camera folders: {counts}")


def _validate_detections(detected: TargetDetection, target: Any) -> Dict[str, Dict]:
    """
    Compute per-camera board detection rate and mean board completeness.
    Returns {cam: {detection_rate, mean_board_completeness}} for JSON caching.

    Edited copy of validate_detections() from
    pyCamSet/calibration/camera_calibrator.py.  Change: returns stats dict
    instead of None so results can be cached as JSON.

    :param detected: TargetDetection containing all per-camera, per-image data.
    :param target:   AbstractTarget instance (used for point count per face).
    :return:         {cam_name: {'detection_rate': float, 'mean_board_completeness': float}}
    """
    corners_per_face = target.point_data.shape[-2]           # max detectable corners per face
    cam_names = detected.cam_names                           # ordered camera name list
    n_detected: Dict[str, int] = {}                          # {cam: count of images with detection}
    board_fraction: Dict[str, List[float]] = {}              # {cam: [completeness per image]}

    for cam_td in detected.get_cam_list():                   # iterate per-camera sub-detections
        data = cam_td.get_data()                             # raw data array for this camera
        if data is None:                                     # no detections for this camera
            continue                                         # skip; default stats remain 0
        cam_idx = int(data[0, 0])                            # camera index in cam_names
        cam_name = cam_names[cam_idx]                        # resolve name
        board_detected = 0                                   # count of images with ≥1 detection
        for im_td in cam_td.get_image_list():                # iterate per-image sub-detections
            datum = im_td.get_data()                         # data for this image
            if datum is None:                                # no corners in this image
                continue                                     # skip to next image
            total_seen = datum.shape[0]                      # detected corners in this image
            board_detected += 1                              # at least one corner found
            n_keys = datum.shape[1] - 4                      # key columns: total cols - cam/im/x/y
            seen = board_fraction.setdefault(cam_name, [])  # get or create list
            if n_keys == 1:                                  # flat board (single key per point)
                seen.append(total_seen / corners_per_face)  # fraction of corners detected
            else:                                            # cube board (2 keys: board + corner)
                n_boards = len(np.unique(datum[:, 2:-3], axis=0))  # distinct board face IDs
                seen.append(total_seen / corners_per_face / n_boards)  # mean per-face completeness
        n_detected[cam_name] = board_detected                # raw count; divide later

    stats: Dict[str, Dict] = {}
    for cam in cam_names:                                    # assemble stats for each camera
        rate = n_detected.get(cam, 0) / max(detected.max_ims, 1)  # detection rate [0, 1]
        completeness = float(np.mean(board_fraction.get(cam, [0.0])))  # mean completeness
        stats[cam] = {'detection_rate': rate,                # fraction of images with a detection
                      'mean_board_completeness': completeness}  # mean fraction of corners seen
        if rate < 0.9:                                       # warn on high miss rate
            logging.warning('[Phase 2d] Camera "%s" detected boards in %.1f%% of images',
                            cam, rate * 100)
        if completeness < 0.5:                               # warn on sparse board coverage
            logging.warning('[Phase 2d] Camera "%s" mean board completeness %.1f%%',
                            cam, completeness * 100)
    return stats                                             # return JSON-serialisable stats dict


def _outlier_rejection_auto(
    per_image_means: List[float],
    out_thresh: float = 5.0,
) -> Optional[List[int]]:
    """
    Non-interactive MAD outlier detection on per-image mean reprojection errors.
    Returns a list of flagged image indices, or None if no outliers are found.

    Edited copy of outlier_rejection() from
    pyCamSet/calibration/camera_calibrator.py.  Changes: removes interactive
    plt.show() / input() calls; accepts pre-computed means list; returns
    List[int] instead of (TargetDetection, bool).

    :param per_image_means: Per-image mean reprojection errors (one per image slot).
    :param out_thresh:      MAD threshold multiplier (default 5.0).
    :return:                Sorted list of flagged image indices, or None if clean.
    """
    from pyCamSet.utils.general_utils import mad_outlier_detection  # direct import
    result = mad_outlier_detection(                          # run MAD detection (no draw)
        per_image_means, out_thresh=out_thresh, draw=False)
    if result is None:                                       # no outliers found
        return None                                          # signal clean dataset
    return [int(i) for i in result[0]]                       # flatten tuple of arrays → plain list


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1 — Target Construction and Per-Image Detection
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1_detection(
    parent_folder: Path,
    camera_names: Sequence[str],
    ccube_length_mm: float = 0.0,
    ccube_n_points: int = 0,
    target: Any = None,
    aruco_dict_name: str = 'DICT_4X4_1000',
    border_fraction: float = 0.1,
    detection_size: int = 1000,
    n_lim: Optional[int] = None,
    threads: int = 1,
    out_dir: Optional[Path] = None,
    save_cache: bool = True,
    load_cache: bool = True,
) -> Dict[str, Any]:
    """
    Phase 1: Build a calibration target and detect corners in every image
    for every camera.  Detections are returned as a TargetDetection container.
    Supports any AbstractTarget subclass (e.g. Ccube, ChArUco) via the
    *target* parameter; when None a Ccube is constructed from the ccube_* args.
    Detection is performed via pyCamSet's detect_datapoints_in_imfile(),
    matching the legacy phased calibration path.

    Parameters
    ----------
    parent_folder    : Root directory containing one subfolder per camera.
    camera_names     : Ordered list of camera subfolder names.
    ccube_length_mm  : Physical side length of one Ccube face in mm (Ccube only).
    ccube_n_points   : Number of ChArUco corners per side (Ccube only).
    target           : Pre-built AbstractTarget instance.  If None, a Ccube is
                       constructed from ccube_length_mm and ccube_n_points.
    aruco_dict_name  : OpenCV ArUco dictionary name string (Ccube only).
    border_fraction  : Relative border width for Ccube (Ccube only).
    detection_size   : Unused (kept for API compatibility).
    n_lim            : Maximum number of images to use per camera (None = all).
    threads          : Unused (kept for API compatibility).
    out_dir          : Directory for cache files; required when save_cache=True.
    save_cache       : If True, persist detections to a .pickle file in out_dir.
    load_cache       : If True, attempt to resume from an existing .pickle cache.

    Returns
    -------
    dict with keys:
        target       : AbstractTarget instance used for detection.
        detections   : TargetDetection containing all per-camera, per-image data.
        image_lists  : {camera_name: [Path, ...]} sorted image path lists.
        cam_res      : [(height, width), ...] one tuple per camera (same order).
        parent_folder: Path to the parent folder (passed through for Phase 3).
        n_lim        : n_lim value used (passed through for Phase 3 high_distortion).
        cache_path   : Path to the saved .pickle file, or None.
    """
    parent_folder = Path(parent_folder)                      # ensure Path object
    out_dir = Path(out_dir) if out_dir else None             # normalise optional out_dir

    cache_path = (phase_cache_path(out_dir, 'phase1_detections', '.pickle')
                  if out_dir else None)                      # determine cache file location

    # ── 1a. Image count sanity check ─────────────────────────────────────────
    cam_folders = [parent_folder / cam for cam in camera_names]  # full folder paths
    _sanitise_input_images(cam_folders)                      # raises if counts differ

    # ── 1b. Cache check ───────────────────────────────────────────────────────
    if load_cache and cache_path and cache_exists(cache_path):  # cache file found on disk
        print(f"[Phase 1b] Loading detection cache: {cache_path}")
        cached = load_pickle(cache_path)                     # deserialise cached result
        return cached                                        # return immediately, skip re-detection

    # ── 1c. Build target ──────────────────────────────────────────────────────
    if target is None:                                       # no target provided — build Ccube
        print("[Phase 1c] Building Ccube target...")
        from pyCamSet.calibration_targets.target_Ccube import Ccube  # local import avoids hard dep
        aruco_enum = getattr(cv.aruco, aruco_dict_name)      # resolve dict name to OpenCV enum
        target = Ccube(                                      # construct pyCamSet Ccube
            length=ccube_length_mm,                          # face side length in mm
            n_points=ccube_n_points,                         # corners per side
            aruco_dict=aruco_enum,                           # ArUco dictionary enum
            border_fraction=border_fraction,                 # relative border width
        )
        # Rebuild CharucoDetectors so detection works without pyCamSet's own init path
        det_params = cv.aruco.DetectorParameters()           # standard ArUco detector params
        charuco_params = cv.aruco.CharucoParameters()        # ChArUco-specific params
        charuco_params.tryRefineMarkers = True               # enable marker refinement
        target.board_detectors = [                           # assign one detector per face
            cv.aruco.CharucoDetector(board, charuco_params, det_params)
            for board in target.boards                       # iterate all Ccube faces
        ]
    else:
        print(f"[Phase 1c] Using provided target: {type(target).__name__}")

    # ── 1d. Detect via detect_datapoints_in_imfile (legacy-compatible path) ───
    print("[Phase 1d] Running detection via detect_datapoints_in_imfile across all cameras...")
    from pyCamSet.calibration.camera_calibrator import detect_datapoints_in_imfile  # detection entry point

    cam_names_list = list(camera_names)                      # ensure list for downstream loops
    image_lists: Dict[str, List[Path]] = {                   # {cam: sorted image paths}
        cam: _images_in_folder(parent_folder / cam) for cam in cam_names_list
    }
    all_detections, cam_res = detect_datapoints_in_imfile(   # robust full-dataset detection
        f_loc=parent_folder,
        calibration_target=target,
        caching=False,                                       # Phase 1 cache is handled by this wrapper
        draw=False,                                          # keep headless behaviour
        n_lim=n_lim,
    )
    if all_detections is None:                               # defensive guard for unexpected None
        raise RuntimeError(
            "Phase 1 detection failed: detect_datapoints_in_imfile returned None "
            "(check dataset contents and target compatibility)."
        )

    detected_by_cam: Dict[str, int] = {}                     # {cam_name: detected_image_count}
    for idx, cam_td in enumerate(all_detections.get_cam_list()):
        cam_name = (all_detections.cam_names[idx]
                    if idx < len(all_detections.cam_names) else str(idx))
        n_detected = sum(
            1 for im_td in cam_td.get_image_list()
            if im_td.get_data() is not None
        )
        detected_by_cam[cam_name] = int(n_detected)

    for cam in cam_names_list:                               # preserve requested camera logging order
        cam_im_list = image_lists.get(cam, [])               # sorted image paths for this camera
        n_total = min(len(cam_im_list), n_lim) if n_lim else len(cam_im_list)
        print(f"  Camera '{cam}': {detected_by_cam.get(cam, 0)}/{n_total} images with detections.")

    result = {                                               # bundle Phase 1 outputs
        'target': target,                                    # AbstractTarget instance
        'detections': all_detections,                        # TargetDetection (all cameras)
        'image_lists': image_lists,                          # {cam: [Path, ...]}
        'cam_res': cam_res,                                  # [(h, w), ...] per camera
        'parent_folder': parent_folder,                      # root folder (for high_distortion)
        'n_lim': n_lim,                                      # image cap (for high_distortion)
        'cache_path': cache_path,                            # path where cache was/will be saved
    }

    # ── 1e. Persist cache ─────────────────────────────────────────────────────
    if save_cache and cache_path:                            # caller requested cache save
        print(f"[Phase 1e] Saving detection cache: {cache_path}")
        save_pickle(result, cache_path)                      # serialise full result dict

    return result                                            # return Phase 1 outputs


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — Image Culling by Detection Count
# ══════════════════════════════════════════════════════════════════════════════

def run_phase2_culling(
    phase1_result: Dict[str, Any],
    min_corners: int = 4,
    out_dir: Optional[Path] = None,
    save_cache: bool = True,
    load_cache: bool = True,
) -> Dict[str, Any]:
    """
    Phase 2: Inspect raw detection counts and delete image indices where any
    camera has fewer than *min_corners* corners.  Uses TargetDetection.delete_row
    for clean pruning.  Runs a board-completeness validation (subphase 2d) after
    culling and saves a per-camera stats JSON.

    Parameters
    ----------
    phase1_result : Output dict from run_phase1_detection().
    min_corners   : Minimum acceptable detection count per image (default 4).
    out_dir       : Directory for cache files; required when save_cache=True.
    save_cache    : If True, persist results to cache files in out_dir.
    load_cache    : If True, attempt to resume from existing cache files.

    Returns
    -------
    dict with keys:
        skip_indices      : set of int — image indices culled across all cameras.
        pruned_detections : TargetDetection with culled images removed.
        skip_path         : Path to the saved skip_indices.json file, or None.
        validation_path   : Path to the saved validation stats JSON, or None.
    """
    out_dir = Path(out_dir) if out_dir else None             # normalise optional out_dir
    skip_path = (phase_cache_path(out_dir, 'phase2_skip_indices', '.json')
                 if out_dir else None)                       # skip index cache path
    val_path = (phase_cache_path(out_dir, 'phase2_validation_stats', '.json')
                if out_dir else None)                        # validation stats cache path

    # ── 2a. Cache check ───────────────────────────────────────────────────────
    if load_cache and skip_path and cache_exists(skip_path):  # cache exists on disk
        print(f"[Phase 2a] Loading cull cache: {skip_path}")
        loaded_skip = set(load_json(skip_path))              # restore saved skip index set
        val_stats = (load_json(val_path)                     # load validation stats if present
                     if val_path and cache_exists(val_path) else {})
        # reconstruct pruned TargetDetection from Phase 1 detections
        detections: TargetDetection = phase1_result['detections']
        pruned = (detections.delete_row(im_num=sorted(loaded_skip))  # remove culled images
                  if loaded_skip else detections)             # nothing to delete if set is empty
        return {'skip_indices': loaded_skip, 'pruned_detections': pruned,
                'skip_path': skip_path, 'validation_path': val_path}

    # ── 2b. Count corners per image per camera ────────────────────────────────
    print("[Phase 2b] Evaluating detection counts from Phase 1 results...")
    detections: TargetDetection = phase1_result['detections']

    # Build (n_ims, n_cams) count matrix via the TargetDetection data directly
    data = detections.get_data()                             # raw detection data array
    n_ims = detections.max_ims                               # total image slots
    n_cams = len(detections.cam_names)                       # number of cameras
    counts = np.zeros((n_ims, n_cams), dtype=int)            # corner counts per (image, camera)
    if data is not None:                                     # guard against empty detection
        cam_idxs = data[:, 0].astype(int)                    # camera index column
        im_nums = data[:, 1].astype(int)                     # image number column
        np.add.at(counts, (im_nums, cam_idxs), 1)           # add 1 for each detected corner

    # ── 2c. Determine skip indices ────────────────────────────────────────────
    skip_mask = np.any(counts < min_corners, axis=1)         # True if any cam below threshold
    skip_indices: Set[int] = set(int(i) for i in np.where(skip_mask)[0])  # flagged indices
    print(f"[Phase 2c] Culled {len(skip_indices)} image indices "
          f"(min_corners={min_corners}).")

    # prune TargetDetection using delete_row
    pruned: TargetDetection = (
        detections.delete_row(im_num=sorted(skip_indices))   # remove culled image rows
        if skip_indices else detections                       # nothing to remove if set is empty
    )

    # ── 2d. Validate detection quality ────────────────────────────────────────
    print("[Phase 2d] Validating detection completeness on retained images...")
    target = phase1_result['target']                         # AbstractTarget from Phase 1
    val_stats = _validate_detections(pruned, target)         # per-camera completeness stats

    result = {                                               # bundle Phase 2 outputs
        'skip_indices': skip_indices,                        # set of culled image indices
        'pruned_detections': pruned,                         # TargetDetection after culling
        'skip_path': skip_path,                              # cache path for skip indices
        'validation_path': val_path,                         # cache path for validation stats
    }

    # ── 2e. Persist cache ─────────────────────────────────────────────────────
    if save_cache and skip_path:                             # caller requested save
        print(f"[Phase 2e] Saving skip indices: {skip_path}")
        save_json(sorted(skip_indices), skip_path)           # persist sorted list as .json
    if save_cache and val_path:                              # save validation stats
        print(f"[Phase 2e] Saving validation stats: {val_path}")
        save_json(val_stats, val_path)                       # persist per-camera stats as .json

    return result                                            # return Phase 2 outputs


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3 — Two-Stage Multi-Camera Calibration
# ══════════════════════════════════════════════════════════════════════════════

def run_phase3_calibration(
    phase1_result: Dict[str, Any],
    phase2_result: Dict[str, Any],
    fixed_params: Optional[Dict] = None,
    problem_options: Optional[Dict] = None,
    threads: int = 1,
    high_distortion: bool = False,
    out_dir: Optional[Path] = None,
    save_cache: bool = True,
    load_cache: bool = True,
) -> Dict[str, Any]:
    """
    Phase 3: Two-stage multi-camera calibration using pyCamSet's
    run_initial_calibration (subphase 3a) followed by run_stereo_calibration
    (subphase 3b).  An optional iterative high-distortion subphase (3a-hd)
    re-detects with the initial cameras to improve corner localisation.

    Parameters
    ----------
    phase1_result   : Output dict from run_phase1_detection().
    phase2_result   : Output dict from run_phase2_culling().
    fixed_params    : Dict of parameter names to fix during optimisation (pyCamSet).
    problem_options : Dict of additional solver options for bundle adjustment.
                      For Ccube targets, {'max_nfev': 100, 'outliers': 'n'} is
                      known to work well (matches calibrate_ccube.py).
    threads         : Worker threads for stereo bundle adjustment (default 1).
    high_distortion : If True, re-detect with initial cameras before stereo BA
                      to improve corner localisation under heavy distortion.
    out_dir         : Directory for cache files; required when save_cache=True.
    save_cache      : If True, save calibration results as .camset files.
    load_cache      : If True, attempt to resume from existing .camset caches.

    Returns
    -------
    dict with keys:
        cam_set              : Calibrated pyCamSet CameraSet (optimised).
        initial_camset_path  : Path to initial_cameras.camset, or None.
        camset_path          : Path to optimised_cameras.camset, or None.
        detections           : TargetDetection used for the stereo bundle adjustment.
    """
    from pyCamSet.calibration.camera_calibrator import (     # direct import of calibration primitives
        detect_datapoints_in_imfile, run_initial_calibration, run_stereo_calibration)

    out_dir = Path(out_dir) if out_dir else None             # normalise optional out_dir
    init_path = (out_dir / 'initial_cameras.camset'          # subphase 3a cache
                 if out_dir else None)
    hd_path = (out_dir / 'initial_cameras_high_distortion.camset'  # subphase 3a-hd cache
               if out_dir else None)
    camset_path = (out_dir / 'optimised_cameras.camset'      # subphase 3b cache
                   if out_dir else None)

    # ── 3a. Cache check (optimised result) ───────────────────────────────────
    if load_cache and camset_path and cache_exists(camset_path):
        print(f"[Phase 3a] Loading optimised calibration cache: {camset_path}")
        cam_set = load_camset(camset_path)                   # deserialise via pyCamSet
        return {'cam_set': cam_set, 'initial_camset_path': init_path,
                'camset_path': camset_path,
                'detections': phase2_result['pruned_detections']}

    # ── 3b. Initial per-camera calibration ───────────────────────────────────
    print("[Phase 3b] Running initial per-camera calibration...")
    if problem_options is None:                              # no solver options provided by caller
        logging.info(                                        # inform user about default behaviour
            "[Phase 3] No problem_options provided; pyCamSet will use its own "
            "DEFAULT_OPTIONS. For Ccube targets, consider passing "
            "problem_options={'max_nfev': 100, 'outliers': 'n'} "
            "(as used in calibrate_ccube.py).")
    target = phase1_result['target']                         # AbstractTarget from Phase 1
    cam_res = phase1_result['cam_res']                       # [(h, w), ...] per camera
    pruned_td: TargetDetection = phase2_result['pruned_detections']  # culled detections
    stereo_td: TargetDetection = pruned_td                    # default Phase 3 detections

    # check for cached initial cameras
    if load_cache and init_path and cache_exists(init_path):
        print(f"[Phase 3b] Loading initial cameras cache: {init_path}")
        initial_cams = load_camset(init_path)                # reuse saved initial cameras
    else:
        initial_cams = run_initial_calibration(              # per-camera OpenCV calibration
            pruned_td,                                       # TargetDetection after culling
            target,                                          # calibration target instance
            cam_res,                                         # [(h, w)] per camera
            save=save_cache and bool(init_path),             # persist if requested
            save_loc=init_path if init_path else Path('initial_cameras.camset'),
            fixed_params=fixed_params,                       # optional parameter locking
        )
        print("[Phase 3b] Initial calibration complete.")

    # ── 3a-hd. High-distortion iterative re-detection ─────────────────────────
    if high_distortion:
        print("[Phase 3a-hd] High-distortion: re-detecting with initial cameras...")
        parent_folder = phase1_result['parent_folder']       # root folder from Phase 1
        n_lim = phase1_result['n_lim']                       # image cap from Phase 1
        all_hd, _ = detect_datapoints_in_imfile(             # mirror old working script path
            f_loc=parent_folder,                             # dataset root with camera subfolders
            calibration_target=target,                       # calibration target from Phase 1
            caching=False,                                   # avoid stale detection cache reuse
            draw=False,                                      # keep headless behaviour for GUI worker
            n_lim=n_lim,                                     # preserve caller's image cap
            camset=initial_cams,                             # camera-aware corner refinement input
        )
        print("[Phase 3a-hd] Re-running initial calibration with refined detections...")
        initial_cams = run_initial_calibration(              # re-calibrate with refined data
            all_hd,
            target,
            cam_res,
            save=save_cache and bool(hd_path),
            save_loc=hd_path if hd_path else Path('initial_cameras_high_distortion.camset'),
        )
        stereo_td = all_hd                                    # carry refined detections into stereo BA
        print("[Phase 3a-hd] High-distortion initial calibration complete.")

    # ── 3b-res. Set camera resolutions from actual image files ────────────────
    parent_folder = phase1_result['parent_folder']           # root folder with camera subfolders
    initial_cams.set_resolutions_from_file(floc=parent_folder)  # populate Camera.res from image dims

    # ── 3c. Stereo bundle adjustment ─────────────────────────────────────────
    print("[Phase 3c] Running stereo bundle adjustment...")
    cam_set = run_stereo_calibration(                        # multi-camera Levenberg–Marquardt
        initial_cams,                                        # initial intrinsics from 3b
        stereo_td,                                           # detections used for final BA
        target,                                              # calibration target instance
        save=save_cache and bool(camset_path),               # persist if requested
        save_loc=camset_path if camset_path else Path('optimised_cameras.camset'),
        fixed_params=fixed_params,                           # optional parameter locking
        threads=threads,                                     # worker count for BA
        problem_options=problem_options,                     # additional solver options
    )
    print("[Phase 3c] Stereo calibration complete.")

    return {                                                 # bundle Phase 3 outputs
        'cam_set': cam_set,                                  # optimised CameraSet
        'initial_camset_path': init_path,                    # initial_cameras.camset path
        'camset_path': camset_path,                          # optimised_cameras.camset path
        'detections': stereo_td,                             # detections used to calibrate cam_set
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 4 — Reprojection Error Analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_phase4_analysis(
    phase1_result: Dict[str, Any],
    phase2_result: Dict[str, Any],
    phase3_result: Dict[str, Any],
    out_dir: Optional[Path] = None,
    save_cache: bool = True,
    load_cache: bool = True,
) -> Dict[str, Any]:
    """
    Phase 4: Compute per-image and per-camera reprojection errors.  Stores
    per-image signed residuals (dx_mean, dy_mean) for cluster plotting.  Stores
    per-point (u, v, error) data for coverage scatter plots.  Applies MAD
    outlier detection (subphase 4b) and saves flagged indices as JSON.

    Parameters
    ----------
    phase1_result : Output dict from run_phase1_detection().
    phase2_result : Output dict from run_phase2_culling().
    phase3_result : Output dict from run_phase3_calibration().
    out_dir       : Directory for cache files; required when save_cache=True.
    save_cache    : If True, persist results to .csv and .json files in out_dir.
    load_cache    : If True, attempt to resume from existing cache files.

    Returns
    -------
    dict with keys:
        per_image_errors  : list of dicts {camera, image, index, mean_error,
                            dx_mean, dy_mean}.
        per_camera_errors : dict {camera_name: mean_error_across_images}.
        global_mean_error : float — mean error across all cameras and images.
        outlier_indices   : list of int — flagged image indices (MAD detection).
        csv_path          : Path to per-image .csv file, or None.
        per_point_path    : Path to per-point .csv file, or None.
        outlier_path      : Path to outlier indices .json file, or None.
    """
    out_dir = Path(out_dir) if out_dir else None             # normalise optional out_dir
    csv_path = (phase_cache_path(out_dir, 'phase4_reprojection_errors', '.csv')
                if out_dir else None)                        # per-image error table cache path
    pp_path = (phase_cache_path(out_dir, 'phase4_per_point_errors', '.csv')
               if out_dir else None)                         # per-point error table cache path
    outlier_path = (phase_cache_path(out_dir, 'phase4_outlier_indices', '.json')
                    if out_dir else None)                     # outlier indices cache path

    # ── 4a. Cache check ───────────────────────────────────────────────────────
    if load_cache and csv_path and cache_exists(csv_path):
        print(f"[Phase 4a] Loading error cache: {csv_path}")
        headers, rows = load_csv(csv_path)                   # load from CSV
        per_image_errors = []                                # reconstruct from rows
        has_residuals = 'dx_mean' in headers                 # check if new columns present
        for row in rows:                                     # one row per image
            entry = {                                        # rebuild dict
                'camera': row[0],
                'image': row[1],
                'index': int(row[2]),
                'mean_error': float(row[3]),
                'dx_mean': float(row[4]) if has_residuals else float('nan'),
                'dy_mean': float(row[5]) if has_residuals else float('nan'),
            }
            per_image_errors.append(entry)
        per_camera_errors, global_mean = _aggregate_errors(per_image_errors)
        outlier_indices = (load_json(outlier_path)           # load outlier indices if cached
                           if outlier_path and cache_exists(outlier_path) else [])
        return {
            'per_image_errors': per_image_errors,
            'per_camera_errors': per_camera_errors,
            'global_mean_error': global_mean,
            'outlier_indices': outlier_indices,
            'csv_path': csv_path,
            'per_point_path': pp_path,
            'outlier_path': outlier_path,
        }

    # ── 4b. Compute per-image reprojection errors with signed residuals ───────
    print("[Phase 4b] Computing per-image reprojection errors...")
    cam_set = phase3_result['cam_set']                       # optimised CameraSet
    target = phase1_result['target']                         # AbstractTarget (for point_data)
    pruned_td: TargetDetection = (phase3_result.get('detections')
                                  or phase2_result['pruned_detections'])  # prefer Phase 3 detections

    per_image_errors: List[Dict] = []                        # accumulate one entry per image
    per_point_rows: List[Tuple] = []                         # accumulate (cam, im, u, v, err) rows

    for cam_td in pruned_td.get_cam_list():                  # iterate per-camera sub-detections
        cam_data = cam_td.get_data()                         # raw data for this camera
        if cam_data is None:                                 # no detections for this camera
            continue                                         # skip to next camera
        cam_idx = int(cam_data[0, 0])                        # camera index
        cam_name = pruned_td.cam_names[cam_idx]              # resolve camera name
        camera_obj = cam_set[cam_name]                       # get Camera object from CameraSet
        intrinsics = camera_obj.intrinsic                    # 3x3 camera matrix
        dist = camera_obj.distortion_coefs                   # distortion coefficients

        for im_td in cam_td.get_image_list():                # iterate per-image sub-detections
            im_data = im_td.get_data()                       # data for this image
            if im_data is None:                              # no corners in this image
                continue                                     # skip to next image
            im_num = int(im_data[0, 1])                      # image index
            img_pts = im_data[:, -2:]                        # detected (x, y) pixel coords
            keys = im_data[:, 2:-2].astype(int)              # [board_id, corner_id] keys

            if len(img_pts) < 4:                             # too few points for PnP
                continue                                     # skip image

            # resolve 3-D object points from target's point_data table
            if keys.shape[1] == 2:                           # 2D keys: [board_id, corner_id]
                obj_pts = target.point_data[                 # index into (n_boards, n_corners, 3)
                    keys[:, 0], keys[:, 1]                   # board index, corner index
                ].astype(np.float64)
            else:                                            # 1D key (flat board, single key col)
                obj_pts = target.point_data.reshape(         # reshape to (N, 3) for flat board
                    -1, 3)[keys[:, 0]].astype(np.float64)   # keys[:, 0] = the only key column

            img_pts_f = img_pts.astype(np.float64).reshape(-1, 1, 2)  # OpenCV: Nx1x2

            ok, rvec, tvec = cv.solvePnP(                    # estimate camera pose
                obj_pts, img_pts_f, intrinsics, dist,
                flags=cv.SOLVEPNP_ITERATIVE)
            if not ok:                                       # solver failed
                continue                                     # skip this image

            proj, _ = cv.projectPoints(                      # reproject 3-D → 2-D
                obj_pts, rvec, tvec, intrinsics, dist)
            proj = proj.reshape(-1, 2)                       # flatten to Nx2
            residuals = img_pts.astype(np.float64) - proj    # signed (dx, dy) per corner
            errors = np.linalg.norm(residuals, axis=1)       # Euclidean error per corner
            mean_err = float(np.mean(errors))                # mean error for this image
            dx_mean = float(np.mean(residuals[:, 0]))        # mean signed x-residual
            dy_mean = float(np.mean(residuals[:, 1]))        # mean signed y-residual

            image_lists = phase1_result.get('image_lists', {})  # {cam: [Path, ...]}
            cam_imgs = image_lists.get(cam_name, [])         # sorted image paths for this cam
            img_name = (cam_imgs[im_num].name                # filename for this image index
                        if im_num < len(cam_imgs) else str(im_num))

            per_image_errors.append({                        # record per-image result
                'camera': cam_name,
                'image': img_name,
                'index': im_num,
                'mean_error': mean_err,
                'dx_mean': dx_mean,                          # mean signed x residual
                'dy_mean': dy_mean,                          # mean signed y residual
            })

            for (u, v), err in zip(img_pts, errors):        # accumulate per-point rows
                per_point_rows.append(                       # (cam, image, u, v, error)
                    (cam_name, img_name, float(u), float(v), float(err)))

    # ── 4c. Aggregate per-camera and global means ────────────────────────────
    per_camera_errors, global_mean = _aggregate_errors(per_image_errors)
    print(f"[Phase 4c] Global mean reprojection error: {global_mean:.4f} px")
    for cam, err in per_camera_errors.items():               # log per-camera summary
        print(f"  Camera '{cam}': mean error = {err:.4f} px")

    # ── 4d. MAD outlier detection on per-image means ──────────────────────────
    print("[Phase 4d] Running MAD outlier detection on per-image errors...")
    im_means = [e['mean_error'] for e in per_image_errors]   # flat list of per-image means
    raw_outliers = _outlier_rejection_auto(im_means)         # returns indices or None
    outlier_indices: List[int] = raw_outliers if raw_outliers is not None else []
    if outlier_indices:
        print(f"[Phase 4d] Flagged {len(outlier_indices)} outlier image(s): {outlier_indices}")
    else:
        print("[Phase 4d] No outlier images detected.")

    result = {                                               # bundle Phase 4 outputs
        'per_image_errors': per_image_errors,
        'per_camera_errors': per_camera_errors,
        'global_mean_error': global_mean,
        'outlier_indices': outlier_indices,                  # MAD-flagged indices
        'csv_path': csv_path,
        'per_point_path': pp_path,
        'outlier_path': outlier_path,
    }

    # ── 4e. Persist caches ────────────────────────────────────────────────────
    if save_cache and csv_path:                              # caller requested per-image save
        print(f"[Phase 4e] Saving error table: {csv_path}")
        headers = ['camera', 'image', 'index', 'mean_error', 'dx_mean', 'dy_mean']
        rows = [(e['camera'], e['image'], e['index'],
                 e['mean_error'], e['dx_mean'], e['dy_mean'])
                for e in per_image_errors]
        save_csv(rows, headers, csv_path)                    # write per-image error table

    if save_cache and pp_path and per_point_rows:            # save per-point error table
        print(f"[Phase 4e] Saving per-point error table: {pp_path}")
        pp_headers = ['camera', 'image', 'u', 'v', 'error']
        save_csv(per_point_rows, pp_headers, pp_path)        # write per-point CSV

    if save_cache and outlier_path:                          # save outlier indices
        print(f"[Phase 4e] Saving outlier indices: {outlier_path}")
        save_json(outlier_indices, outlier_path)             # persist as .json

    return result                                            # return Phase 4 outputs


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 5 — Calibration Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def run_phase5_visualisation(
    phase4_result: Dict[str, Any],
    phase3_result: Optional[Dict[str, Any]] = None,
    out_dir: Optional[Path] = None,
    save_plots: bool = True,
    save_summaries: bool = True,
    show_plots: bool = False,
) -> Dict[str, Any]:
    """
    Phase 5: Generate reprojection error visualisations and optionally save them
    as .png files with lower-case title-based filenames.  Numeric summaries are
    optionally saved as .csv files alongside the plots.

    Subphases
    ---------
    5a : Global error histogram.
    5b : Per-camera mean error bar chart.
    5c : Per-camera individual histograms.
    5d : 2-D residual cluster plot (global + per camera) from dx/dy residuals.
    5e : Per-camera image-plane coverage scatter coloured by error magnitude.
    5f : 3-D camera arrangement screenshot (pyvista; optional dependency).

    Parameters
    ----------
    phase4_result   : Output dict from run_phase4_analysis().
    phase3_result   : Output dict from run_phase3_calibration() (needed for 5f).
    out_dir         : Directory for .png and .csv outputs; required when saving.
    save_plots      : If True, save all plots as .png files in out_dir.
    save_summaries  : If True, save numeric summaries as .csv files in out_dir.
    show_plots      : If True, call plt.show() after generating all figures.

    Returns
    -------
    dict with keys:
        figures        : list of matplotlib Figure objects.
        saved_png_paths: list of Path objects for saved .png files.
        saved_csv_paths: list of Path objects for saved .csv files.
    """
    import matplotlib.pyplot as plt                          # deferred import — optional dep

    out_dir = Path(out_dir) if out_dir else None             # normalise optional out_dir
    plot_dir = out_dir if save_plots else None               # pass dir only if saving
    csv_dir = out_dir if save_summaries else None            # pass dir only if saving

    per_image_errors = phase4_result['per_image_errors']     # list of per-image error dicts
    per_camera_errors = phase4_result['per_camera_errors']   # {cam: mean_error}
    global_mean = phase4_result['global_mean_error']         # scalar global mean

    figures = []                                             # accumulate Figure objects
    saved_pngs: List[Path] = []                              # accumulate saved .png paths
    saved_csvs: List[Path] = []                              # accumulate saved .csv paths

    def _track(fig, title, is_plot=True):                    # helper: track saved paths
        """Append plot or CSV path if saving is enabled."""
        if is_plot and save_plots and plot_dir:
            saved_pngs.append(plot_dir / f"{title_to_filename(title)}.png")
        elif not is_plot and save_summaries and csv_dir:
            saved_csvs.append(csv_dir / f"{title_to_filename(title)}.csv")

    # ── 5a. Global error histogram ────────────────────────────────────────────
    all_errors = [e['mean_error'] for e in per_image_errors]  # flat list of per-image means
    title1 = 'Mean Euclidean Error'                          # plot title (sets filename)
    fig1 = plot_error_histogram(                             # create histogram figure
        all_errors, title=title1,
        xlabel='Mean Reprojection Error per Image (px)',
        out_dir=plot_dir)
    figures.append(fig1)                                     # track figure
    _track(fig1, title1, is_plot=True)
    if save_summaries:                                       # save numeric data as CSV
        csv_path = save_numeric_summary({'mean_error_px': all_errors}, title=title1, out_dir=csv_dir)
        if csv_path:
            saved_csvs.append(csv_path)

    # ── 5b. Per-camera bar chart ──────────────────────────────────────────────
    cam_names = list(per_camera_errors.keys())               # ordered camera names
    cam_means = [per_camera_errors[c] for c in cam_names]    # matching mean errors
    title2 = 'Per Camera Mean Reprojection Error'            # plot title (sets filename)
    fig2 = plot_per_camera_errors(cam_names, cam_means, title=title2, out_dir=plot_dir)
    figures.append(fig2)                                     # track figure
    _track(fig2, title2, is_plot=True)
    if save_summaries:
        csv_path = save_numeric_summary(
            {'camera': cam_names, 'mean_error_px': cam_means}, title=title2, out_dir=csv_dir)
        if csv_path:
            saved_csvs.append(csv_path)

    # ── 5c. Per-camera individual histograms ─────────────────────────────────
    for cam in cam_names:                                    # iterate each camera
        cam_errors = [e['mean_error']
                      for e in per_image_errors if e['camera'] == cam]
        if not cam_errors:                                   # no data for this camera
            continue
        title3 = f'Reprojection Error {cam}'                 # title encodes camera name
        fig3 = plot_error_histogram(cam_errors, title=title3,
                                    xlabel='Mean Reprojection Error per Image (px)',
                                    out_dir=plot_dir)
        figures.append(fig3)
        _track(fig3, title3, is_plot=True)
        if save_summaries:
            csv_path = save_numeric_summary(
                {'mean_error_px': cam_errors}, title=title3, out_dir=csv_dir)
            if csv_path:
                saved_csvs.append(csv_path)

    # ── 5d. 2-D residual cluster plots ────────────────────────────────────────
    dx_all = [e['dx_mean'] for e in per_image_errors        # global signed x residuals
              if np.isfinite(e.get('dx_mean', float('nan')))]
    dy_all = [e['dy_mean'] for e in per_image_errors        # global signed y residuals
              if np.isfinite(e.get('dy_mean', float('nan')))]
    if dx_all and dy_all:
        title_cluster_global = 'Residual Cluster Global'     # global cluster plot title
        xy_global = np.column_stack((dx_all, dy_all)).ravel()  # interleave x/y efficiently
        fig_cg = plot_residual_clusters(                     # global cluster plot
            [xy_global], titles=['Global'], out_dir=plot_dir,
            title=title_cluster_global)
        if fig_cg is not None:                               # plot_residual_clusters may return None
            figures.append(fig_cg)
            _track(fig_cg, title_cluster_global, is_plot=True)

        for cam in cam_names:                                # per-camera cluster plots
            dx_cam = [e['dx_mean'] for e in per_image_errors
                      if e['camera'] == cam and np.isfinite(e.get('dx_mean', float('nan')))]
            dy_cam = [e['dy_mean'] for e in per_image_errors
                      if e['camera'] == cam and np.isfinite(e.get('dy_mean', float('nan')))]
            if not dx_cam:                                   # no data for this camera
                continue
            title_cc = f'Residual Cluster {cam}'             # per-camera cluster title
            xy_cam = np.column_stack((dx_cam, dy_cam)).ravel()  # interleave x/y efficiently
            fig_cc = plot_residual_clusters(                 # per-camera cluster plot
                [xy_cam], titles=[cam], out_dir=plot_dir, title=title_cc)
            if fig_cc is not None:
                figures.append(fig_cc)
                _track(fig_cc, title_cc, is_plot=True)

    # ── 5e. Per-camera coverage scatter ───────────────────────────────────────
    pp_path = phase4_result.get('per_point_path')            # per-point CSV from Phase 4
    cam_set = phase3_result.get('cam_set') if phase3_result else None  # calibrated cameras
    if pp_path and cache_exists(Path(pp_path)):              # per-point data available
        pp_headers, pp_rows = load_csv(Path(pp_path))       # load per-point CSV
        for cam in cam_names:                                # one scatter per camera
            cam_pts = [(float(r[2]), float(r[3]), float(r[4]))  # (u, v, error)
                       for r in pp_rows if r[0] == cam]
            if not cam_pts:                                  # no points for this camera
                continue
            principal_pt = None                              # default: no principal point
            if cam_set is not None:
                try:
                    mtx = cam_set[cam].intrinsic             # 3x3 camera matrix
                    principal_pt = (float(mtx[0, 2]), float(mtx[1, 2]))  # (cx, cy)
                except Exception:
                    pass                                     # ignore if camera not in set
            title_cov = f'Coverage {cam}'                    # coverage plot title
            fig_cov = plot_coverage_scatter(                 # generate scatter plot
                cam_pts, cam_name=cam, principal_point=principal_pt,
                title=title_cov, out_dir=plot_dir)
            if fig_cov is not None:
                figures.append(fig_cov)
                _track(fig_cov, title_cov, is_plot=True)

    # ── 5f. 3-D camera arrangement ────────────────────────────────────────────
    if cam_set is not None:                                  # calibrated cameras available
        title_arr = 'Camera Arrangement'                     # arrangement plot title
        fig_arr = plot_camera_arrangement(                   # pyvista screenshot (optional)
            cam_set, title=title_arr, out_dir=plot_dir)
        if fig_arr is not None:                              # None when pyvista unavailable
            figures.append(fig_arr)
            _track(fig_arr, title_arr, is_plot=True)

    # ── 5g. Log results and display ───────────────────────────────────────────
    print(f"[Phase 5g] Generated {len(figures)} plots; "
          f"global mean error = {global_mean:.4f} px.")
    if saved_pngs:
        print(f"[Phase 5g] Saved {len(saved_pngs)} plot(s):")
        for p in saved_pngs:
            print(f"  {p}")
    if saved_csvs:
        print(f"[Phase 5g] Saved {len(saved_csvs)} summary CSV(s):")
        for p in saved_csvs:
            print(f"  {p}")

    if show_plots:                                           # display interactively if requested
        plt.show()                                           # block until user closes windows

    return {                                                 # bundle Phase 5 outputs
        'figures': figures,
        'saved_png_paths': saved_pngs,
        'saved_csv_paths': saved_csvs,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 6 — Self-Calibration (optional)
# ══════════════════════════════════════════════════════════════════════════════

def run_phase6_self_calibration(
    phase1_result: Dict[str, Any],
    phase2_result: Dict[str, Any],
    phase3_result: Dict[str, Any],
    enable_self_calibration: bool = False,
    fixed_params: Optional[Dict] = None,
    options: Optional[Dict] = None,
    threads: int = 1,
    out_dir: Optional[Path] = None,
    save_cache: bool = True,
    load_cache: bool = True,
    save_summaries: bool = True,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """
    Phase 6: Optional self-calibration using pyCamSet's SelfBundleHandler.
    Uses the existing detections and target to refine camera parameters and
    target geometry without re-running detection.

    Parameters
    ----------
    phase1_result          : Output dict from run_phase1_detection().
    phase2_result          : Output dict from run_phase2_culling().
    phase3_result          : Output dict from run_phase3_calibration().
    enable_self_calibration: If False, returns current camset unchanged.
    fixed_params           : Parameter locks for the self-calibration handler.
    options                : Options dict forwarded to SelfBundleHandler.
    threads                : Worker thread count for bundle adjustment.
    out_dir                : Directory for cache files; required when saving.
    save_cache             : If True, persist self-calibrated camset as .camset.
    load_cache             : If True, attempt to load cached camset.
    save_summaries         : If True, save displacement summary as .csv.
    save_plots             : If True, save displacement histogram as .png.

    Returns
    -------
    dict with keys:
        cam_set             : Self-calibrated CameraSet (or input if skipped).
        camset_path         : Path to cached .camset file, or None.
        displacement_path   : Path to displacement .csv, or None.
        displacement_plot   : Path to displacement .png, or None.
        skipped             : Bool flag indicating if phase was skipped.
    """
    out_dir = Path(out_dir) if out_dir else None             # normalise optional out_dir
    camset_path = (phase_cache_path(out_dir,                 # derive cache path (snake_case)
                                   'phase6_self_calibrated_cameras',
                                   '.camset') if out_dir else None)  # .camset cache path
    disp_csv_path = (phase_cache_path(out_dir,               # displacement .csv path
                                     'self_calibration_point_displacement',
                                     '.csv') if out_dir else None)   # .csv cache path
    disp_png_path = (out_dir /                                # plot output path (title → filename)
                     f"{title_to_filename('Self Calibration Point Displacement')}.png"
                     if out_dir else None)                   # .png output path

    if not enable_self_calibration:                          # user opted out of Phase 6
        return {                                             # return unchanged camset
            'cam_set': phase3_result['cam_set'],             # pass through Phase 3 camset
            'camset_path': None,                             # no cache path used
            'displacement_path': None,                       # no CSV saved
            'displacement_plot': None,                       # no plot saved
            'skipped': True,                                 # mark as skipped
        }

    if load_cache and camset_path and cache_exists(camset_path):  # cached camset available
        print(f"[Phase 6a] Loading self-calibration cache: {camset_path}")
        cached = load_camset(camset_path)                    # load cached CameraSet
        return {                                             # return cached results
            'cam_set': cached,                               # cached camset
            'camset_path': camset_path,                      # cached path
            'displacement_path': (disp_csv_path
                                  if (disp_csv_path and cache_exists(disp_csv_path)) else None),
            'displacement_plot': (disp_png_path
                                  if (disp_png_path and disp_png_path.exists()) else None),
            'skipped': False,                                # phase executed via cache
        }

    print("[Phase 6b] Running self-calibration bundle adjustment...")  # log start
    target = phase1_result['target']                         # AbstractTarget from Phase 1
    current_cams = phase3_result['cam_set']                  # calibrated CameraSet from Phase 3

    # Prefer the detection stored inside the calibration handler (matches calibrate_ccube.py
    # and pyCamSet's own test), falling back to pipeline detections if unavailable.
    _handler = getattr(current_cams, 'calibration_handler', None)  # may be None if loaded
    pruned_td = (getattr(_handler, 'detection', None)        # detection as seen by stereo BA
                 or phase3_result.get('detections')           # Phase 3 detections (high_distortion)
                 or phase2_result['pruned_detections'])       # Phase 2 pruned detections (last resort)

    param_handler = SelfBundleHandler(                       # instantiate self-calibration handler
        camset=current_cams,                                 # provide current camset
        target=target,                                       # provide calibration target
        detection=pruned_td,                                 # provide pruned detections
        fixed_params=fixed_params,                           # optional fixed params
        options=options,                                     # optional solver options
    )

    try:                                                     # attempt to warm-start from templated camset
        param_handler.set_from_templated_camset(current_cams)  # use previous solution as initial params
    except Exception as exc:                                 # fallback if camset not templated
        print(f"[Phase 6b] Warning: warm-start failed ({exc}); using default initial params.")

    optimisation, final_cams = run_bundle_adjustment(        # run pyCamSet bundle adjustment
        param_handler=param_handler,                         # handler defines the problem
        threads=threads,                                     # thread count
    )

    if save_cache and camset_path:                           # caller requested cache save
        print(f"[Phase 6c] Saving self-calibrated camset: {camset_path}")
        save_camset(final_cams, camset_path)                 # persist .camset via helper

    displacement_csv = None                                  # default: no CSV
    displacement_png = None                                  # default: no PNG
    if save_summaries or save_plots:                         # only compute diagnostics if needed
        try:                                                 # guard in case handler lacks API
            updated_target = param_handler.get_updated_target(optimisation.x)  # updated target
            old_pts = np.array(target.point_data, dtype=float).reshape(-1, 3)  # original points
            new_pts = np.array(updated_target.point_data, dtype=float).reshape(-1, 3)  # updated
            if old_pts.shape == new_pts.shape:               # only compute if point counts match
                diffs = new_pts - old_pts                    # per-point vector displacement
                mags = np.linalg.norm(diffs, axis=1)         # per-point magnitude
                if save_summaries and disp_csv_path:         # save per-point CSV if requested
                    print(f"[Phase 6d] Saving displacement CSV: {disp_csv_path}")
                    headers = ['point_index', 'dx', 'dy', 'dz', 'displacement']  # CSV headers
                    rows = [(int(i), float(d[0]), float(d[1]), float(d[2]), float(m))
                            for i, (d, m) in enumerate(zip(diffs, mags))]  # rows per point
                    save_csv(rows, headers, disp_csv_path)    # write CSV via helper
                    displacement_csv = disp_csv_path         # record CSV path
                if save_plots and disp_png_path:             # save histogram plot if requested
                    import matplotlib.pyplot as plt           # local import to avoid hard dep
                    fig, ax = plt.subplots(figsize=(7, 4))    # create figure
                    ax.hist(mags, bins=30, color='steelblue', alpha=0.8)  # plot histogram
                    ax.set_title('Self Calibration Point Displacement')  # plot title
                    ax.set_xlabel('Displacement (units of target model)')  # x-axis label
                    ax.set_ylabel('Count')                    # y-axis label
                    fig.tight_layout()                        # improve layout
                    fig.savefig(str(disp_png_path), dpi=150, bbox_inches='tight')  # save to .png
                    displacement_png = disp_png_path          # record PNG path
            else:
                print(f"[Phase 6d] Skipping displacement diagnostics: point counts differ "
                      f"(old: {old_pts.shape[0]}, new: {new_pts.shape[0]}).")
        except Exception as exc:
            print(f"[Phase 6d] Skipping displacement diagnostics: {exc}")

    return {                                                  # bundle Phase 6 outputs
        'cam_set': final_cams,                                # final self-calibrated camset
        'camset_path': camset_path,                           # camset cache path
        'displacement_path': displacement_csv,                # CSV path (optional)
        'displacement_plot': displacement_png,                # PNG path (optional)
        'skipped': False,                                     # phase executed
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Top-level: run all six phases sequentially
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    parent_folder: Path,
    camera_names: Sequence[str],
    out_dir: Path,
    ccube_length_mm: float = 0.0,
    ccube_n_points: int = 0,
    target: Any = None,
    aruco_dict_name: str = 'DICT_4X4_1000',
    border_fraction: float = 0.1,
    detection_size: int = 1000,
    n_lim: Optional[int] = None,
    threads: int = 1,
    min_corners: int = 4,
    fixed_params: Optional[Dict] = None,
    problem_options: Optional[Dict] = None,
    high_distortion: bool = False,
    save_phase1: bool = True,
    load_phase1: bool = True,
    save_phase2: bool = True,
    load_phase2: bool = True,
    save_phase3: bool = True,
    load_phase3: bool = True,
    save_phase4: bool = True,
    load_phase4: bool = True,
    save_plots: bool = True,
    save_summaries: bool = True,
    show_plots: bool = False,
    enable_phase6_self_calibration: bool = False,
    self_calibration_options: Optional[Dict] = None,
    save_phase6: bool = True,
    load_phase6: bool = True,
    save_phase6_summaries: bool = True,
    save_phase6_plots: bool = True,
) -> Dict[str, Any]:
    """
    Run the full six-phase calibration pipeline.
    Each phase can independently load from cache or save its results to disk,
    allowing the pipeline to resume from any phase without re-running earlier phases.

    Parameters
    ----------
    parent_folder   : Root directory with one subfolder per camera.
    camera_names    : Ordered list of camera subfolder names.
    out_dir         : Output directory; all cache and plot files go here.
    ccube_length_mm : Physical side length of one Ccube face in mm (Ccube only).
    ccube_n_points  : Number of ChArUco corners per side (Ccube only).
    target          : Pre-built AbstractTarget; if None a Ccube is constructed.
    aruco_dict_name : OpenCV ArUco dictionary name string (default DICT_4X4_1000).
    border_fraction : Relative Ccube border width (default 0.1).
    detection_size  : Unused; kept for API compatibility.
    n_lim           : Maximum images per camera for detection (None = all).
    threads         : Worker threads for detection and stereo BA (default 1).
    min_corners     : Minimum detection count for an image to be kept (default 4).
    fixed_params    : Dict of parameters to lock during calibration optimisation.
    problem_options : Dict of additional solver options for bundle adjustment.
                      For Ccube targets, {'max_nfev': 100, 'outliers': 'n'} is
                      known to work well (matches calibrate_ccube.py).
    high_distortion : If True, enable iterative high-distortion calibration.
    save_phase1..4  : If True, save the output of that phase to disk.
    load_phase1..4  : If True, try to load the cached output before recomputing.
    save_plots      : If True, save Phase 5 plots as .png files.
    save_summaries  : If True, save Phase 5 numeric summaries as .csv files.
    show_plots      : If True, display Phase 5 plots interactively.
    enable_phase6_self_calibration : Master toggle for Phase 6 self-calibration.
    self_calibration_options       : Options forwarded to SelfBundleHandler.
    save_phase6     : If True, save Phase 6 self-calibrated camset.
    load_phase6     : If True, load Phase 6 cached camset if available.
    save_phase6_summaries : If True, save Phase 6 displacement CSV.
    save_phase6_plots     : If True, save Phase 6 displacement histogram PNG.

    Returns
    -------
    dict with results from all six phases keyed by 'phase1' … 'phase6'.
    """
    out_dir = Path(out_dir)                                  # ensure Path object
    out_dir.mkdir(parents=True, exist_ok=True)               # create output dir if needed
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # human-readable run timestamp
    print(f"=== pyCamSet pipeline start: {timestamp} ===")   # log run start

    # ── Phase 1: Detection ────────────────────────────────────────────────────
    p1 = run_phase1_detection(                               # detect corners in all images
        parent_folder=parent_folder,
        camera_names=camera_names,
        ccube_length_mm=ccube_length_mm,
        ccube_n_points=ccube_n_points,
        target=target,
        aruco_dict_name=aruco_dict_name,
        border_fraction=border_fraction,
        detection_size=detection_size,
        n_lim=n_lim,
        threads=threads,
        out_dir=out_dir,
        save_cache=save_phase1,
        load_cache=load_phase1,
    )

    # ── Phase 2: Culling ──────────────────────────────────────────────────────
    p2 = run_phase2_culling(                                 # cull images with too few corners
        phase1_result=p1,
        min_corners=min_corners,
        out_dir=out_dir,
        save_cache=save_phase2,
        load_cache=load_phase2,
    )

    # ── Phase 3: Calibration ──────────────────────────────────────────────────
    p3 = run_phase3_calibration(                             # two-stage multi-camera calibration
        phase1_result=p1,
        phase2_result=p2,
        fixed_params=fixed_params,
        problem_options=problem_options,
        threads=threads,
        high_distortion=high_distortion,
        out_dir=out_dir,
        save_cache=save_phase3,
        load_cache=load_phase3,
    )

    # ── Phase 4: Error Analysis ───────────────────────────────────────────────
    p4 = run_phase4_analysis(                                # compute reprojection errors
        phase1_result=p1,
        phase2_result=p2,
        phase3_result=p3,
        out_dir=out_dir,
        save_cache=save_phase4,
        load_cache=load_phase4,
    )

    # ── Phase 5: Visualisation ────────────────────────────────────────────────
    p5 = run_phase5_visualisation(                           # generate and optionally save plots
        phase4_result=p4,
        phase3_result=p3,
        out_dir=out_dir,
        save_plots=save_plots,
        save_summaries=save_summaries,
        show_plots=show_plots,
    )

    # ── Phase 6: Self-Calibration ─────────────────────────────────────────────
    p6 = run_phase6_self_calibration(                        # optional self-calibration step
        phase1_result=p1,
        phase2_result=p2,
        phase3_result=p3,
        enable_self_calibration=enable_phase6_self_calibration,  # master toggle
        fixed_params=fixed_params,                           # reuse fixed params
        options=self_calibration_options,                    # pass-through options
        threads=threads,                                     # worker threads
        out_dir=out_dir,                                     # output directory
        save_cache=save_phase6,                              # save camset cache
        load_cache=load_phase6,                              # load camset cache
        save_summaries=save_phase6_summaries,                # save CSV
        save_plots=save_phase6_plots,                        # save PNG
    )

    print("=== pyCamSet pipeline complete ===")              # log run end
    return {                                                 # return all phase results
        'phase1': p1,
        'phase2': p2,
        'phase3': p3,
        'phase4': p4,
        'phase5': p5,
        'phase6': p6,
    }
