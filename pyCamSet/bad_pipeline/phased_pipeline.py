"""
Purpose: Phased calibration bad_pipeline for pyCamSet.
         Wraps the existing calibration primitives (detect_datapoints_in_imfile,
         run_initial_calibration, run_stereo_calibration, SelfBundleHandler,
         run_bundle_adjustment) into six discrete, independently-cacheable phases
         plus a single run_pipeline() orchestrator.
         Each phase accepts and returns a plain dict so results can be persisted,
         inspected, and resumed at any boundary.
         Camera subfolders are auto-discovered when camera_names is omitted;
         the reserved 'metadata' output folder is always excluded from discovery.
         A run-manifest is written to the output directory so that subsequent
         runs can detect configuration changes and invalidate stale caches from
         the earliest affected phase onward.
Status:  Working
Future:  Add CLI support via run_pipeline() once all phases are implemented.
"""

import logging                                               # for per-phase status messages
import pickle                                                # for detection cache serialisation
from pathlib import Path                                     # for filesystem path handling
from datetime import datetime                                # for run-timestamp logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple  # type annotations
import shutil  # import shutil to remove temporary directories after detection completes
import tempfile  # import tempfile to create isolated temporary roots for camera-only detection inputs

import cv2 as cv                                             # for image I/O and resize
import numpy as np                                           # for numerical operations

# pyCamSet detection containers — direct import
from pyCamSet.calibration_targets import TargetDetection, ImageDetection  # detection containers
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler  # self-calibration handler
from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment  # bundle adjustment entry point
from pyCamSet.calibration_targets.target_Ccube import Ccube  # import Ccube class used below to construct the Ccube target
from pyCamSet.calibration_targets.target_charuco import ChArUco  # import ChArUco class used below to construct the ChArUco target

# Pipeline helpers (file I/O and plots) from the pyCamSet bad_pipeline package
from pyCamSet.bad_pipeline.pipeline_cache import (               # file I/O utilities
    phase_cache_path, cache_exists,                          # path building and existence check
    save_pickle, load_pickle,                                # pickle helpers for detection cache
    save_json, load_json,                                    # JSON helpers for skip indices / stats
    save_csv, load_csv,                                      # CSV helpers for error tables
    save_camset, load_camset,                                # CamSet helpers for calibration results
    title_to_filename,                                       # title → filename stem conversion
)
from pyCamSet.bad_pipeline.pipeline_plots import (               # plot / summary utilities
    plot_error_histogram,                                    # histogram of reprojection errors
    plot_per_camera_errors,                                  # bar chart of per-camera means
    save_numeric_summary,                                    # numeric dict → .csv
    plot_residual_clusters,                                  # 2-D residual cluster plot
    plot_coverage_scatter,                                   # per-camera coverage scatter
    plot_camera_arrangement,                                 # pyvista camera-arrangement screenshot
)


# ──────────────────────────────────────────────────────────────────────────────
#  Private image helpers
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


# ──────────────────────────────────────────────────────────────────────────────
#  Reserved folder names — never treated as camera subfolders
# ──────────────────────────────────────────────────────────────────────────────

_RESERVED_FOLDERS: Set[str] = {'metadata'}      # folders always excluded from auto-discovery


def _discover_camera_names(parent_folder: Path) -> List[str]:
    """
    Auto-discover camera subfolder names directly under *parent_folder*.

    Discovery rules:
    - Only immediate subdirectories are considered (non-recursive).
    - Folders in *_RESERVED_FOLDERS* (e.g. 'metadata') are always excluded.
    - A folder is accepted only if it contains at least one supported image
      (as determined by pyCamSet's glob_ims_local), providing a practical safety
      filter against non-camera directories such as logs or auxiliary data.
    - Returned names are sorted for deterministic ordering across runs.

    :param parent_folder: Root directory to search for camera subfolders.
    :return:              Sorted list of discovered camera subfolder names.
    :raises ValueError:   If no suitable camera folders are found.
    """
    candidates = sorted(                                         # sort for deterministic ordering
        d.name for d in parent_folder.iterdir()                  # iterate immediate children
        if d.is_dir()                                            # directories only
        and d.name not in _RESERVED_FOLDERS                      # exclude reserved names
        and bool(_images_in_folder(d))                           # must contain ≥1 image
    )
    if not candidates:                                           # nothing found — likely wrong path
        raise ValueError(
            f"No camera folders with images found under '{parent_folder}'. "
            "Ensure each camera has a dedicated subfolder containing images, "
            f"or pass camera_names explicitly. Reserved folders {_RESERVED_FOLDERS} "
            "are always excluded."
        )
    return candidates                                            # sorted list of camera names


# ──────────────────────────────────────────────────────────────────────────────
#  Run-manifest helpers — capture and compare effective bad_pipeline configuration
# ──────────────────────────────────────────────────────────────────────────────

_MANIFEST_FILENAME = 'run_manifest.json'                         # filename for the run manifest


def _build_run_manifest(
    camera_names: List[str],
    target_type: str,
    ccube_length_mm: float,
    ccube_n_points: int,
    aruco_dict_name: str,
    border_fraction: float,
    n_lim: Optional[int],
    min_corners: int,
    high_distortion: bool,
    fixed_params: Optional[Dict],
    problem_options: Optional[Dict],
) -> Dict[str, Any]:
    """
    Build a JSON-serialisable dict that describes the effective bad_pipeline
    configuration for the current run.  This manifest is saved alongside the
    phase caches so that subsequent runs can compare configurations and detect
    incompatible checkpoints.

    :param camera_names:     Effective (post-discovery) sorted camera name list.
    :param target_type:      Class name of the calibration target (e.g. 'Ccube').
    :param ccube_length_mm:  Physical side length of one Ccube face in mm.
    :param ccube_n_points:   Number of ChArUco corners per side.
    :param aruco_dict_name:  OpenCV ArUco dictionary name string.
    :param border_fraction:  Relative Ccube border width.
    :param n_lim:            Maximum images per camera (None = all).
    :param min_corners:      Minimum detection count for culling.
    :param high_distortion:  Whether high-distortion iterative path was used.
    :param fixed_params:     Parameter locks (may be None).
    :param problem_options:  Solver options (may be None).
    :return:                 Dict suitable for JSON serialisation.
    """
    return {
        'camera_names': sorted(camera_names),                    # sorted for stable comparison
        'target_type': target_type,                              # discriminates target class
        'ccube_length_mm': ccube_length_mm,                      # physical target dimension
        'ccube_n_points': ccube_n_points,                        # grid density
        'aruco_dict_name': aruco_dict_name,                      # ArUco dictionary identifier
        'border_fraction': border_fraction,                      # relative border width
        'n_lim': n_lim,                                          # image cap (None = all)
        'min_corners': min_corners,                              # culling threshold
        'high_distortion': high_distortion,                      # iterative re-detection flag
        'fixed_params': fixed_params,                            # parameter locks (may be None)
        'problem_options': problem_options,                      # solver options (may be None)
    }


def _manifest_first_invalid_phase(current: Dict, saved: Dict) -> int:
    """
    Compare the *current* run manifest against a *saved* one and return the
    index of the first bad_pipeline phase whose cached results are no longer valid.

    Phase numbering:
    - 1 = Detection      (Phase 1 cache)
    - 2 = Culling        (Phase 2 cache)
    - 3 = Calibration    (Phase 3 cache; also invalidates Phase 4 and 6)
    - 7 = All compatible (no invalidation needed)

    Keys that invalidate each phase:
    - Phase 1: camera_names, target_type, ccube_length_mm, ccube_n_points,
               aruco_dict_name, border_fraction, n_lim
    - Phase 2: min_corners
    - Phase 3: high_distortion, fixed_params, problem_options

    :param current: Manifest dict for the current run.
    :param saved:   Manifest dict loaded from the previous run.
    :return:        Index of the first invalid phase (1–3), or 7 if all valid.
    """
    p1_keys = [                                                  # keys that affect detection
        'camera_names', 'target_type', 'ccube_length_mm',
        'ccube_n_points', 'aruco_dict_name', 'border_fraction', 'n_lim',
    ]
    if any(current.get(k) != saved.get(k) for k in p1_keys):    # any Phase 1 key changed
        return 1                                                 # invalidate from Phase 1

    p2_keys = ['min_corners']                                    # keys that affect culling
    if any(current.get(k) != saved.get(k) for k in p2_keys):    # any Phase 2 key changed
        return 2                                                 # invalidate from Phase 2

    p3_keys = ['high_distortion', 'fixed_params', 'problem_options']  # keys for calibration
    if any(current.get(k) != saved.get(k) for k in p3_keys):    # any Phase 3 key changed
        return 3                                                 # invalidate from Phase 3

    return 7                                                     # all phases compatible


def _build_camera_only_root(parent_folder: Path, camera_names: Sequence[str]) -> Path:
    """
    Create a temporary root containing only the selected camera subfolders.

    This prevents downstream detection internals from re-discovering reserved
    folders such as 'metadata' under the original parent_folder.

    :param parent_folder: Original dataset root containing cameras and metadata.
    :param camera_names:  Explicit camera folder names selected by the bad_pipeline.
    :return:              Temporary directory path exposing only camera folders.
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="pycamset_camroot_"))  # create unique temporary directory and wrap as Path
    for cam in camera_names:  # iterate each selected camera folder name to populate temporary root
        src = parent_folder / cam  # resolve source camera folder path under original dataset root
        dst = tmp_root / cam  # resolve destination folder path under temporary root
        try:  # attempt symlink first to avoid expensive directory copies
            dst.symlink_to(src, target_is_directory=True)  # create directory symlink so detector sees camera folder without duplication
        except Exception:  # fall back when symlinks are disallowed (common on Windows without developer mode/admin)
            shutil.copytree(src, dst)  # copy folder recursively so bad_pipeline remains functional without symlink support
    return tmp_root  # return temporary root path containing only selected camera folders


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
    if load_cache and cache_path and cache_exists(
            cache_path):  # only try loading when cache use is enabled and file exists
        print(f"[Phase 1b] Loading detection cache: {cache_path}")  # log cache read for traceability
        cached = load_pickle(cache_path)  # load previously cached phase-1 dictionary from disk
        if cached.get(
                'target') is None:  # handle cache-safe format where non-pickle-safe target was intentionally omitted
            if target is None:  # only rebuild target when caller did not provide one explicitly
                aruco_enum = getattr(cv.aruco,
                                     aruco_dict_name)  # resolve dictionary enum from configured dictionary name
                target = Ccube(  # reconstruct Ccube target from run configuration values
                    length=ccube_length_mm,  # restore configured physical cube side length
                    n_points=ccube_n_points,  # restore configured points-per-side
                    aruco_dict=aruco_enum,  # restore configured aruco dictionary enum
                    border_fraction=border_fraction,  # restore configured border fraction
                )
                det_params = cv.aruco.DetectorParameters()  # create detector parameters required by CharucoDetector
                charuco_params = cv.aruco.CharucoParameters()  # create charuco-specific detector parameters
                charuco_params.tryRefineMarkers = True  # enable marker refinement as in normal target construction path
                target.board_detectors = [cv.aruco.CharucoDetector(board, charuco_params, det_params) for board in
                                          target.boards]  # rebuild face detectors
            cached['target'] = target  # inject reconstructed/provided target back into loaded phase-1 result
        return cached  # return repaired cache payload so downstream phases receive required target object

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
    detect_root = _build_camera_only_root(parent_folder,
                                          cam_names_list)  # build temporary root that contains only camera folders
    try:  # ensure temporary root is always removed after detection, even if detection raises
        all_detections, cam_res = detect_datapoints_in_imfile(
            # run detection on camera-only root to exclude metadata from internal folder discovery
            f_loc=detect_root,  # pass isolated temporary root instead of original parent folder
            calibration_target=target,  # pass selected calibration target unchanged
            caching=False,  # keep wrapper-level caching behaviour unchanged
            draw=False,  # preserve headless detection mode
            n_lim=n_lim,  # preserve caller-specified image limit
        )
    finally:  # always execute cleanup for temporary root resources
        shutil.rmtree(detect_root, ignore_errors=True)  # remove temporary directory tree and ignore cleanup errors
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

    result = {  # build full in-memory phase output dictionary
        'target': target,  # keep live target object for downstream phases in this run
        'detections': all_detections,  # keep full detection container for calibration
        'image_lists': image_lists,  # keep per-camera sorted image lists for reporting/analysis
        'cam_res': cam_res,  # keep camera resolution tuples for initial calibration
        'parent_folder': parent_folder,  # keep dataset root for optional high-distortion re-detection
        'n_lim': n_lim,  # keep image cap setting for reproducibility
        'cache_path': cache_path,  # keep resolved cache path for transparency/logging
    }

    if save_cache and cache_path:  # save cache only when explicitly enabled and path exists
        print(f"[Phase 1e] Saving detection cache: {cache_path}")  # log where phase-1 cache is being written
        cacheable_result = dict(result)  # copy full result so we can remove non-pickle-safe fields for disk cache
        cacheable_result[
            'target'] = None  # remove OpenCV-backed target object (contains non-pickle-safe aruco dictionary)
        save_pickle(cacheable_result, cache_path)  # serialise cache-safe subset of phase-1 results to disk

    return result  # return full in-memory result (with live target) to subsequent phases


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
        hd_camera_names = list(phase1_result[
                                   'image_lists'].keys())  # recover effective camera names used in phase 1 for consistent re-detection
        hd_root = _build_camera_only_root(parent_folder,
                                          hd_camera_names)  # build camera-only temporary root for high-distortion re-detection
        try:  # ensure temporary root cleanup even when high-distortion detection fails
            all_hd, _ = detect_datapoints_in_imfile(  # run high-distortion re-detection on camera-only root
                f_loc=hd_root,  # pass isolated temporary root so metadata is excluded from internal discovery
                calibration_target=target,  # pass same target as phase 1
                caching=False,  # disable detector-level caching to avoid stale results
                draw=False,  # keep headless behaviour
                n_lim=n_lim,  # preserve image limit
                camset=initial_cams,  # provide initial cameras for camera-aware refinement
            )
        finally:  # always clean up temporary root
            shutil.rmtree(hd_root,
                          ignore_errors=True)  # remove temporary directory tree created for high-distortion re-detection
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
    # Set resolutions using explicit camera image lists from Phase 1 to avoid folder-name mismatch caused by reserved folders.
    image_lists = phase1_result['image_lists']  # get per-camera image paths already aligned to discovered camera names
    for cam in initial_cams:  # iterate cameras currently in the CameraSet
        cam_name = cam.name  # read current camera name to index phase1 image lists deterministically
        cam_imgs = image_lists.get(cam_name, [])  # fetch this camera's image path list from phase-1 outputs
        if not cam_imgs:  # fail fast if no images are available for this camera
            raise ValueError(f"No images available for camera '{cam_name}' when setting resolution.")
        im0 = cv.imread(str(cam_imgs[0]),
                        cv.IMREAD_UNCHANGED)  # read first image at native depth to get reliable dimensions
        if im0 is None:  # fail fast on unreadable image path
            raise ValueError(f"Could not read image '{cam_imgs[0]}' for camera '{cam_name}' resolution assignment.")
        h, w = im0.shape[:2]  # extract image height and width from loaded array
        cam.res = np.array([w, h],
                           dtype=int)  # assign [width, height] as expected by pyCamSet camera resolution convention

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
    phase1_result: Dict[str, Any],  # accept phase-1 outputs containing target and image filename mappings
    phase2_result: Dict[str, Any],  # accept phase-2 outputs containing pruned detections after culling
    phase3_result: Dict[str, Any],  # accept phase-3 outputs containing calibrated camera parameters
    out_dir: Optional[Path] = None,  # optional directory for phase-4 cache files
    save_cache: bool = True,  # if True, write computed analysis artefacts to disk
    load_cache: bool = True,  # if True, reuse existing cached analysis artefacts when available
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
    out_dir = Path(out_dir) if out_dir else None  # normalise optional output directory to Path if provided
    csv_path = (phase_cache_path(out_dir, 'phase4_reprojection_errors', '.csv')
                if out_dir else None)  # resolve cache path for per-image error table
    pp_path = (phase_cache_path(out_dir, 'phase4_per_point_errors', '.csv')
               if out_dir else None)  # resolve cache path for per-point error table
    outlier_path = (phase_cache_path(out_dir, 'phase4_outlier_indices', '.json')
                    if out_dir else None)  # resolve cache path for outlier index list

    if load_cache and csv_path and cache_exists(csv_path):  # use cached phase-4 output when enabled and available
        print(f"[Phase 4a] Loading error cache: {csv_path}")  # log cache load location for traceability
        headers, rows = load_csv(csv_path)  # load cached per-image CSV headers and data rows
        per_image_errors = []  # allocate list to reconstruct per-image dictionaries from CSV rows
        has_residuals = 'dx_mean' in headers  # detect whether cache includes signed residual columns
        for row in rows:  # iterate each cached row representing one image observation summary
            entry = {  # reconstruct in-memory per-image entry in current expected schema
                'camera': row[0],  # restore camera name string
                'image': row[1],  # restore image filename string
                'index': int(row[2]),  # restore image index as integer
                'mean_error': float(row[3]),  # restore mean reprojection error in pixels
                'dx_mean': float(row[4]) if has_residuals else float('nan'),  # restore mean signed x residual if present
                'dy_mean': float(row[5]) if has_residuals else float('nan'),  # restore mean signed y residual if present
            }
            per_image_errors.append(entry)  # append reconstructed row dict to list
        per_camera_errors, global_mean = _aggregate_errors(per_image_errors)  # recompute aggregate statistics from cached per-image rows
        outlier_indices = (load_json(outlier_path)
                           if outlier_path and cache_exists(outlier_path) else [])  # load cached outlier list if available
        return {  # return cache-derived phase-4 result payload in standard structure
            'per_image_errors': per_image_errors,  # provide per-image table reconstructed from cache
            'per_camera_errors': per_camera_errors,  # provide per-camera means recomputed from per-image rows
            'global_mean_error': global_mean,  # provide global mean reprojection error
            'outlier_indices': outlier_indices,  # provide cached outlier indices if present
            'csv_path': csv_path,  # provide path to per-image cache file
            'per_point_path': pp_path,  # provide path to per-point cache file
            'outlier_path': outlier_path,  # provide path to outlier-index cache file
        }

    print("[Phase 4b] Computing per-image reprojection errors...")  # log start of fresh phase-4 computation
    cam_set = phase3_result['cam_set']  # read calibrated CameraSet generated by phase 3
    target = phase1_result['target']  # read calibration target used to resolve object-space points
    pruned_td: TargetDetection = (phase3_result.get('detections')
                                  or phase2_result['pruned_detections'])  # prefer phase-3 detections; fallback to phase-2 detections

    per_image_errors: List[Dict] = []  # allocate list for per-image summary records
    per_point_rows: List[Tuple] = []  # allocate list for per-point rows used by coverage scatter plots
    skipped_too_few_points = 0  # count images skipped because correspondence count was insufficient for stable PnP
    skipped_pnp_failure = 0  # count images skipped because solvePnP failed or raised an OpenCV error

    for cam_td in pruned_td.get_cam_list():  # iterate per-camera detection containers
        cam_data = cam_td.get_data()  # get raw detection array for this camera view
        if cam_data is None:  # skip cameras with no detections
            continue  # continue to next camera because there is nothing to analyse
        cam_idx = int(cam_data[0, 0])  # read camera index from detection payload
        cam_name = pruned_td.cam_names[cam_idx]  # map camera index to camera name string
        camera_obj = cam_set[cam_name]  # fetch calibrated camera object by name
        intrinsics = camera_obj.intrinsic  # read camera intrinsic matrix for reprojection
        dist = camera_obj.distortion_coefs  # read camera distortion coefficients for reprojection

        for im_td in cam_td.get_image_list():  # iterate all image detections for this camera
            im_data = im_td.get_data()  # get per-image detection data array
            if im_data is None:  # skip image entries with no detected points
                continue  # move to next image because no correspondences exist
            im_num = int(im_data[0, 1])  # read image index for logging and keying
            img_pts = im_data[:, -2:]  # extract detected image points as Nx2 pixel coordinates
            keys = im_data[:, 2:-2].astype(int)  # extract object-point lookup keys from detection table

            if len(img_pts) < 4:  # retain original conservative pre-check for clearly underdetermined cases
                skipped_too_few_points += 1  # increment sparse-point skip counter for diagnostics
                continue  # skip this image because PnP is not meaningful with very few points

            if keys.shape[1] == 2:  # branch for targets keyed by [board_id, corner_id]
                obj_pts = target.point_data[
                    keys[:, 0], keys[:, 1]
                ].astype(np.float64)  # resolve 3-D object points for each detected key pair
            else:  # branch for flat-board style targets with one key column
                obj_pts = target.point_data.reshape(
                    -1, 3)[keys[:, 0]].astype(np.float64)  # flatten point table then index by key column

            img_pts_f = img_pts.astype(np.float64).reshape(-1, 1, 2)  # convert image points to OpenCV-required Nx1x2 float format
            n_corr = int(obj_pts.shape[0])  # compute number of 3D-2D correspondences for this image

            if n_corr < 6:  # guard for OpenCV iterative solvePnP path that can fail below six correspondences in this workflow
                skipped_too_few_points += 1  # increment sparse-point skip counter
                continue  # skip this image rather than raising and aborting full analysis

            try:  # isolate OpenCV failures so one problematic image does not terminate phase 4
                ok, rvec, tvec = cv.solvePnP(  # estimate per-image camera pose from 3D-2D correspondences
                    obj_pts,  # pass object-space points resolved from target model
                    img_pts_f,  # pass measured image-space points
                    intrinsics,  # pass camera matrix from calibrated camera
                    dist,  # pass distortion coefficients from calibrated camera
                    flags=cv.SOLVEPNP_ITERATIVE)  # use iterative PnP method consistent with existing implementation
            except cv.error as exc:  # catch OpenCV exceptions (e.g. degeneracy or solver precondition failure)
                skipped_pnp_failure += 1  # increment PnP-failure counter for summary diagnostics
                logging.warning(  # emit warning with camera and image identifiers for later debugging
                    "[Phase 4b] solvePnP failed for camera '%s', image index %d, correspondences=%d: %s",
                    cam_name, im_num, n_corr, exc)  # include critical context in log message
                continue  # skip this frame and continue analysing remaining frames

            if not ok:  # handle explicit solvePnP failure return without exception
                skipped_pnp_failure += 1  # increment PnP-failure counter for summary diagnostics
                continue  # skip frame if solver reports failure

            proj, _ = cv.projectPoints(
                obj_pts, rvec, tvec, intrinsics, dist)  # project object points with estimated pose for residual calculation
            proj = proj.reshape(-1, 2)  # reshape projected points to Nx2 for vectorised arithmetic
            residuals = img_pts.astype(np.float64) - proj  # compute signed 2-D residual vectors per point
            errors = np.linalg.norm(residuals, axis=1)  # compute Euclidean reprojection error magnitude per point
            mean_err = float(np.mean(errors))  # compute mean reprojection error for this camera/image pair
            dx_mean = float(np.mean(residuals[:, 0]))  # compute mean signed x residual for bias diagnostics
            dy_mean = float(np.mean(residuals[:, 1]))  # compute mean signed y residual for bias diagnostics

            image_lists = phase1_result.get('image_lists', {})  # read camera->image-list mapping saved in phase 1
            cam_imgs = image_lists.get(cam_name, [])  # get ordered image path list for current camera
            img_name = (cam_imgs[im_num].name
                        if im_num < len(cam_imgs) else str(im_num))  # resolve human-readable image label safely

            per_image_errors.append({  # append one per-image summary record for this camera/image pair
                'camera': cam_name,  # store camera name for grouping and plotting
                'image': img_name,  # store image filename for traceability
                'index': im_num,  # store image index for deterministic referencing
                'mean_error': mean_err,  # store mean Euclidean reprojection error
                'dx_mean': dx_mean,  # store mean signed x residual
                'dy_mean': dy_mean,  # store mean signed y residual
            })

            for (u, v), err in zip(img_pts, errors):  # iterate point-level observations to build coverage/error table
                per_point_rows.append(
                    (cam_name, img_name, float(u), float(v), float(err)))  # store point record tuple for CSV export

    per_camera_errors, global_mean = _aggregate_errors(per_image_errors)  # compute per-camera and global mean errors
    print(f"[Phase 4c] Global mean reprojection error: {global_mean:.4f} px")  # report global aggregate error
    for cam, err in per_camera_errors.items():  # iterate camera aggregates for logging
        print(f"  Camera '{cam}': mean error = {err:.4f} px")  # report per-camera mean reprojection error

    print(f"[Phase 4c] Skipped images: too_few_points={skipped_too_few_points}, "
          f"pnp_failures={skipped_pnp_failure}")  # report skip counters so sparse/unstable images are visible to user

    print("[Phase 4d] Running MAD outlier detection on per-image errors...")  # log start of outlier analysis
    # Build one robust error value per dataset frame index so outlier indices map to real image indices.
    errs_by_frame: Dict[int, List[float]] = {}  # map frame index -> list of camera-specific mean errors for that frame
    for e in per_image_errors:  # iterate all per-camera per-image entries collected above
        idx = int(e['index'])  # read dataset frame index from entry
        val = float(e['mean_error'])  # read scalar mean reprojection error for this camera/frame pair
        if np.isfinite(val):  # ignore non-finite values so MAD is computed on valid numbers only
            errs_by_frame.setdefault(idx, []).append(val)  # accumulate valid error into frame bucket

    frame_indices = sorted(errs_by_frame.keys())  # deterministic ordering of real frame indices present in analysis
    frame_scores = [float(np.median(errs_by_frame[i])) for i in frame_indices]  # robust per-frame score across cameras
    raw_outliers = _outlier_rejection_auto(
        frame_scores)  # run MAD on per-frame scores (not flattened camera-image list)
    outlier_indices = [frame_indices[i] for i in
                       raw_outliers] if raw_outliers is not None else []  # map MAD result back to true frame indices
    if outlier_indices:  # branch for detected outlier image indices
        print(f"[Phase 4d] Flagged {len(outlier_indices)} outlier image(s): {outlier_indices}")  # report detected outliers
    else:  # branch when no outliers are detected
        print("[Phase 4d] No outlier images detected.")  # report clean outlier result

    result = {  # assemble phase-4 return payload in standard schema
        'per_image_errors': per_image_errors,  # include per-image summary records
        'per_camera_errors': per_camera_errors,  # include per-camera aggregate means
        'global_mean_error': global_mean,  # include global mean error across all analysed images
        'outlier_indices': outlier_indices,  # include MAD outlier indices list
        'csv_path': csv_path,  # include path to per-image CSV cache
        'per_point_path': pp_path,  # include path to per-point CSV cache
        'outlier_path': outlier_path,  # include path to outlier JSON cache
    }

    if save_cache and csv_path:  # save per-image cache only when requested and path is available
        print(f"[Phase 4e] Saving error table: {csv_path}")  # log per-image CSV output path
        headers = ['camera', 'image', 'index', 'mean_error', 'dx_mean', 'dy_mean']  # define per-image CSV headers
        rows = [(e['camera'], e['image'], e['index'],
                 e['mean_error'], e['dx_mean'], e['dy_mean'])
                for e in per_image_errors]  # convert per-image dicts to tuple rows for CSV writer
        save_csv(rows, headers, csv_path)  # write per-image error table to disk

    if save_cache and pp_path and per_point_rows:  # save per-point cache only when requested and non-empty
        print(f"[Phase 4e] Saving per-point error table: {pp_path}")  # log per-point CSV output path
        pp_headers = ['camera', 'image', 'u', 'v', 'error']  # define per-point CSV headers
        save_csv(per_point_rows, pp_headers, pp_path)  # write per-point error table to disk

    if save_cache and outlier_path:  # save outlier index cache when requested
        print(f"[Phase 4e] Saving outlier indices: {outlier_path}")  # log outlier JSON output path
        save_json(outlier_indices, outlier_path)  # write outlier index list to JSON file

    return result  # return phase-4 results for downstream plotting and bad_pipeline summary


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 5 — Calibration Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def run_phase5_visualisation(
    phase4_result: Dict[str, Any],  # receive phase-4 outputs containing per-image/per-camera error summaries and cache paths
    phase3_result: Optional[Dict[str, Any]] = None,  # optionally receive phase-3 outputs so camera arrangement and principal points can be plotted
    out_dir: Optional[Path] = None,  # optionally receive output directory where plots/CSVs should be written
    save_plots: bool = True,  # toggle writing plot PNG files to disk
    save_summaries: bool = True,  # toggle writing numeric CSV summaries to disk
    show_plots: bool = False,  # toggle interactive display of generated matplotlib figures
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
    import matplotlib.pyplot as plt  # import plotting backend locally so non-plot phases do not require matplotlib at import time

    out_dir = Path(out_dir) if out_dir else None  # normalise optional output directory into a Path object when provided
    plot_dir = out_dir if save_plots else None  # choose plot output directory only when plot saving is enabled
    csv_dir = out_dir if save_summaries else None  # choose CSV output directory only when summary saving is enabled

    per_image_errors = phase4_result['per_image_errors']  # unpack per-image error table produced by phase 4
    per_camera_errors = phase4_result['per_camera_errors']  # unpack per-camera mean errors produced by phase 4
    global_mean = phase4_result['global_mean_error']  # unpack global mean reprojection error for logging at end of phase

    figures = []  # accumulate matplotlib figure objects for return and optional interactive display
    saved_pngs: List[Path] = []  # accumulate paths of plot images written to disk
    saved_csvs: List[Path] = []  # accumulate paths of numeric summary CSVs written to disk

    def _track(fig, title, is_plot=True):  # define helper that mirrors file naming logic so saved paths are easy to inspect
        """Append plot or CSV path if saving is enabled."""
        if is_plot and save_plots and plot_dir:  # branch when this artefact is a plot and plot saving is enabled
            saved_pngs.append(plot_dir / f"{title_to_filename(title)}.png")  # store expected PNG path generated by helper plotting functions
        elif not is_plot and save_summaries and csv_dir:  # branch when this artefact is a numeric summary and CSV saving is enabled
            saved_csvs.append(csv_dir / f"{title_to_filename(title)}.csv")  # store expected CSV path generated by summary helper

    # ── 5a. Global error histogram ────────────────────────────────────────────
    all_errors = [e['mean_error'] for e in per_image_errors if np.isfinite(e.get('mean_error', float('nan')))]  # collect only finite global per-image errors to avoid plot failures
    title1 = 'Mean Euclidean Error'  # define canonical title used both for plot label and deterministic output filename
    if all_errors:  # plot only when at least one finite error exists
        fig1 = plot_error_histogram(  # create global histogram of per-image mean reprojection errors
            all_errors, title=title1,
            xlabel='Mean Reprojection Error per Image (px)',
            out_dir=plot_dir)
        figures.append(fig1)  # store created figure for return/show
        _track(fig1, title1, is_plot=True)  # track expected saved PNG path when saving is enabled
        if save_summaries:  # optionally save numeric values behind this histogram
            csv_path = save_numeric_summary({'mean_error_px': all_errors}, title=title1, out_dir=csv_dir)  # write raw global error list as CSV
            if csv_path:  # guard against helper returning None
                saved_csvs.append(csv_path)  # record saved CSV path for return/logging

    # ── 5b. Per-camera bar chart ──────────────────────────────────────────────
    cam_names = list(per_camera_errors.keys())  # get deterministic camera ordering from per-camera error dictionary keys
    cam_means = [per_camera_errors[c] for c in cam_names if np.isfinite(per_camera_errors[c])]  # keep finite per-camera means only for stable plotting
    cam_names_finite = [c for c in cam_names if np.isfinite(per_camera_errors[c])]  # keep matching finite camera names aligned to filtered means
    title2 = 'Per Camera Mean Reprojection Error'  # define title and output filename stem for per-camera bar chart
    if cam_names_finite:  # plot only when at least one finite per-camera value exists
        fig2 = plot_per_camera_errors(cam_names_finite, cam_means, title=title2, out_dir=plot_dir)  # draw per-camera mean reprojection bar chart
        figures.append(fig2)  # store created figure
        _track(fig2, title2, is_plot=True)  # track expected saved PNG path when saving is enabled
        if save_summaries:  # optionally save bar-chart source values to CSV
            csv_path = save_numeric_summary(
                {'camera': cam_names_finite, 'mean_error_px': cam_means}, title=title2, out_dir=csv_dir)  # write finite per-camera means only
            if csv_path:  # guard against helper returning None
                saved_csvs.append(csv_path)  # record saved CSV path

    # ── 5c. Per-camera individual histograms ─────────────────────────────────
    for cam in cam_names:  # iterate all camera names to make one histogram per camera where possible
        cam_errors = [e['mean_error']
                      for e in per_image_errors
                      if e['camera'] == cam and np.isfinite(e.get('mean_error', float('nan')))]  # collect finite per-image errors for this camera only
        if not cam_errors:  # skip camera when no finite per-image values are available
            continue  # move to next camera
        title3 = f'Reprojection Error {cam}'  # define camera-specific title and filename stem
        fig3 = plot_error_histogram(cam_errors, title=title3,  # draw histogram of per-image mean error for current camera
                                    xlabel='Mean Reprojection Error per Image (px)',
                                    out_dir=plot_dir)
        figures.append(fig3)  # store figure for return/show
        _track(fig3, title3, is_plot=True)  # track expected PNG output path
        if save_summaries:  # optionally save numeric values behind this camera histogram
            csv_path = save_numeric_summary(
                {'mean_error_px': cam_errors}, title=title3, out_dir=csv_dir)  # write finite camera-specific errors as CSV
            if csv_path:  # guard against helper returning None
                saved_csvs.append(csv_path)  # record saved CSV path

    # ── 5d. 2-D residual cluster plots ────────────────────────────────────────
    dx_all_raw = [e.get('dx_mean', float('nan')) for e in per_image_errors]  # collect raw global dx means from per-image entries
    dy_all_raw = [e.get('dy_mean', float('nan')) for e in per_image_errors]  # collect raw global dy means from per-image entries
    global_pairs = [(float(x), float(y)) for x, y in zip(dx_all_raw, dy_all_raw) if np.isfinite(x) and np.isfinite(y)]  # keep only finite (dx,dy) pairs for robust plotting
    if len(global_pairs) >= 2:  # require at least two finite points before attempting cluster histogram
        title_cluster_global = 'Residual Cluster Global'  # define title and filename stem for global residual cluster
        xy_global = np.array(global_pairs, dtype=float).ravel()  # convert finite residual pairs to flattened vector expected by plotting helper
        fig_cg = plot_residual_clusters(  # generate global residual cluster plot from finite residual pairs
            [xy_global], titles=['Global'], out_dir=plot_dir,
            title=title_cluster_global)
        if fig_cg is not None:  # guard in case helper returns None when plotting backend has issues
            figures.append(fig_cg)  # store returned figure
            _track(fig_cg, title_cluster_global, is_plot=True)  # track expected global cluster PNG path

    for cam in cam_names:  # generate per-camera residual clusters with strict finite filtering
        dx_cam_raw = [e.get('dx_mean', float('nan')) for e in per_image_errors if e['camera'] == cam]  # collect raw dx means for current camera
        dy_cam_raw = [e.get('dy_mean', float('nan')) for e in per_image_errors if e['camera'] == cam]  # collect raw dy means for current camera
        cam_pairs = [(float(x), float(y)) for x, y in zip(dx_cam_raw, dy_cam_raw) if np.isfinite(x) and np.isfinite(y)]  # retain only finite residual pairs to avoid pcolormesh non-finite crash
        if len(cam_pairs) < 2:  # skip plotting if insufficient finite points remain
            continue  # move to next camera to keep bad_pipeline running
        title_cc = f'Residual Cluster {cam}'  # define title and filename stem for this camera cluster plot
        xy_cam = np.array(cam_pairs, dtype=float).ravel()  # flatten finite residual pairs into helper input format
        fig_cc = plot_residual_clusters(  # generate per-camera residual cluster on cleaned finite values
            [xy_cam], titles=[cam], out_dir=plot_dir, title=title_cc)
        if fig_cc is not None:  # guard helper return value
            figures.append(fig_cc)  # store figure for return/show
            _track(fig_cc, title_cc, is_plot=True)  # track expected per-camera cluster PNG path

    # ── 5e. Per-camera coverage scatter ───────────────────────────────────────
    pp_path = phase4_result.get('per_point_path')  # resolve per-point CSV path produced by phase 4
    cam_set = phase3_result.get('cam_set') if phase3_result else None  # extract calibrated camera set when available for principal-point overlays
    if pp_path and cache_exists(Path(pp_path)):  # proceed only when per-point CSV exists on disk
        pp_headers, pp_rows = load_csv(Path(pp_path))  # load per-point CSV rows for scatter plotting
        for cam in cam_names:  # generate one coverage scatter per camera
            cam_pts = [(float(r[2]), float(r[3]), float(r[4]))  # parse per-point row fields as (u,v,error)
                       for r in pp_rows
                       if r[0] == cam and np.isfinite(float(r[2])) and np.isfinite(float(r[3])) and np.isfinite(float(r[4]))]  # keep only finite triplets to avoid downstream plotting issues
            if not cam_pts:  # skip camera if no finite per-point rows are available
                continue  # move to next camera
            principal_pt = None  # default principal-point marker is disabled unless camera matrix is available
            if cam_set is not None:  # only try principal-point extraction when calibrated camera set is provided
                try:  # guard against missing camera names or malformed intrinsics
                    mtx = cam_set[cam].intrinsic  # read camera intrinsic matrix for current camera
                    principal_pt = (float(mtx[0, 2]), float(mtx[1, 2]))  # extract principal point (cx,cy) for optional overlay marker
                except Exception:  # ignore errors silently to keep plotting robust
                    pass  # leave principal point as None when extraction fails
            title_cov = f'Coverage {cam}'  # define title and filename stem for coverage scatter
            fig_cov = plot_coverage_scatter(  # generate scatter showing image-plane sampling coverage and error magnitude
                cam_pts, cam_name=cam, principal_point=principal_pt,
                title=title_cov, out_dir=plot_dir)
            if fig_cov is not None:  # guard helper return value
                figures.append(fig_cov)  # store figure for return/show
                _track(fig_cov, title_cov, is_plot=True)  # track expected coverage PNG path

    # ── 5f. 3-D camera arrangement ────────────────────────────────────────────
    if cam_set is not None:  # proceed only when calibrated cameras are available
        title_arr = 'Camera Arrangement'  # define title and filename stem for arrangement visualisation
        fig_arr = plot_camera_arrangement(  # request pyvista-based 3-D camera arrangement screenshot
            cam_set, title=title_arr, out_dir=plot_dir)
        if fig_arr is not None:  # helper may return None if pyvista is unavailable
            figures.append(fig_arr)  # store arrangement figure for return/show
            _track(fig_arr, title_arr, is_plot=True)  # track expected arrangement PNG path

    # ── 5g. Log results and display ───────────────────────────────────────────
    print(f"[Phase 5g] Generated {len(figures)} plots; "
          f"global mean error = {global_mean:.4f} px.")  # print concise phase summary with global error context
    if saved_pngs:  # print saved plot list only when one or more PNG paths were recorded
        print(f"[Phase 5g] Saved {len(saved_pngs)} plot(s):")  # report number of saved PNG files
        for p in saved_pngs:  # iterate tracked plot paths for transparency
            print(f"  {p}")  # print each saved plot path
    if saved_csvs:  # print saved CSV list only when one or more summary files were recorded
        print(f"[Phase 5g] Saved {len(saved_csvs)} summary CSV(s):")  # report number of saved summary CSV files
        for p in saved_csvs:  # iterate tracked summary paths for transparency
            print(f"  {p}")  # print each saved summary CSV path

    if show_plots:  # display figures interactively only when caller enabled this option
        plt.show()  # block and render all generated figures in interactive windows

    return {  # return structured phase-5 outputs for downstream bad_pipeline reporting
        'figures': figures,  # return list of generated figure objects
        'saved_png_paths': saved_pngs,  # return list of PNG paths tracked during this phase
        'saved_csv_paths': saved_csvs,  # return list of CSV paths tracked during this phase
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
    # and pyCamSet's own test), falling back to bad_pipeline detections if unavailable.
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
    camera_names: Optional[Sequence[str]] = None,
    out_dir: Optional[Path] = None,
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
    Run the full six-phase calibration bad_pipeline.
    Each phase can independently load from cache or save its results to disk,
    allowing the bad_pipeline to resume from any phase without re-running earlier phases.

    Auto-discovery and default paths
    ---------------------------------
    When *camera_names* is ``None`` (the default), camera subfolders are
    auto-discovered directly under *parent_folder*.  Discovery excludes the
    reserved ``'metadata'`` folder and any subfolder that contains no supported
    images.  Discovered names are sorted for deterministic ordering.

    When *out_dir* is ``None`` (the default), all cache and plot files are
    written to ``parent_folder / 'metadata'``.  This folder is always excluded
    from camera auto-discovery so subsequent reruns never confuse it for a
    camera folder.

    Checkpoint compatibility and auto-resume
    ----------------------------------------
    A run-manifest (``run_manifest.json``) is saved in *out_dir* after each
    successful run.  On the next run the manifest is loaded and compared against
    the current effective configuration.  If the configs are compatible the
    bad_pipeline resumes from cached checkpoints transparently.  If any parameter
    that affects a cached phase has changed, the cache for that phase (and all
    subsequent phases) is bypassed and recomputed from scratch.  A warning is
    logged for each phase whose cache is invalidated so the behaviour is always
    transparent.

    Parameters
    ----------
    parent_folder   : Root directory with one subfolder per camera.
    camera_names    : Ordered list of camera subfolder names, or ``None`` to
                      auto-discover from *parent_folder* (default ``None``).
    out_dir         : Output directory for all cache and plot files.  Defaults
                      to ``parent_folder / 'metadata'`` when ``None``.
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
    parent_folder = Path(parent_folder)                          # ensure Path object

    # ── Resolve output directory ───────────────────────────────────────────────
    if out_dir is None:                                          # default to metadata subfolder
        out_dir = parent_folder / 'metadata'                     # reserved output folder
        print(f"[Pipeline] out_dir not provided; defaulting to '{out_dir}'.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)                   # create output dir if needed

    # ── Resolve camera names ───────────────────────────────────────────────────
    if camera_names is None:                                     # auto-discover from parent_folder
        print("[Pipeline] camera_names not provided; auto-discovering camera folders...")
        camera_names = _discover_camera_names(parent_folder)     # excludes _RESERVED_FOLDERS
        print(f"[Pipeline] Discovered cameras: {camera_names}")
    else:
        camera_names = list(camera_names)                        # normalise to plain list

    # ── Build current effective configuration manifest ─────────────────────────
    target_type = type(target).__name__ if target is not None else 'Ccube'  # target class name
    current_manifest = _build_run_manifest(                      # capture current config
        camera_names=camera_names,
        target_type=target_type,
        ccube_length_mm=ccube_length_mm,
        ccube_n_points=ccube_n_points,
        aruco_dict_name=aruco_dict_name,
        border_fraction=border_fraction,
        n_lim=n_lim,
        min_corners=min_corners,
        high_distortion=high_distortion,
        fixed_params=fixed_params,
        problem_options=problem_options,
    )

    # ── Compare against saved manifest to determine cache validity ─────────────
    manifest_path = out_dir / _MANIFEST_FILENAME                 # standard manifest filename
    first_invalid = 7                                            # 7 = all phases compatible
    if manifest_path.is_file():                                  # saved manifest found on disk
        try:
            saved_manifest = load_json(manifest_path)            # load previous run config
            first_invalid = _manifest_first_invalid_phase(       # compare configurations
                current_manifest, saved_manifest)
            if first_invalid <= 6:                               # some phase(s) invalidated
                logging.warning(
                    "[Pipeline] Manifest mismatch detected: cached phases %s–6 are "
                    "incompatible with current configuration and will be recomputed.",
                    first_invalid,
                )
            else:
                print("[Pipeline] Manifest matches saved config; "
                      "compatible checkpoints will be reused.")
        except Exception as exc:                                 # corrupted / unreadable manifest
            logging.warning(
                "[Pipeline] Could not read saved manifest (%s); "
                "all phase caches will be treated as incompatible.", exc)
            first_invalid = 1                                    # force full recomputation
    else:
        print("[Pipeline] No saved manifest found; starting fresh run.")

    # ── Override cache-load flags for invalidated phases ──────────────────────
    # For each phase whose index >= first_invalid, disable cache loading so
    # stale checkpoints are not silently reused.
    if first_invalid <= 1:
        load_phase1 = False                                      # Phase 1 cache invalid
    if first_invalid <= 2:
        load_phase2 = False                                      # Phase 2 cache invalid
    if first_invalid <= 3:
        load_phase3 = False                                      # Phase 3 cache invalid
    if first_invalid <= 4:
        load_phase4 = False                                      # Phase 4 cache invalid
    if first_invalid <= 6:
        load_phase6 = False                                      # Phase 6 cache invalid

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')     # human-readable run timestamp
    print(f"=== pyCamSet bad_pipeline start: {timestamp} ===")       # log run start

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

    print("=== pyCamSet bad_pipeline complete ===")              # log run end

    # ── Save updated run manifest ─────────────────────────────────────────────
    # Overwrite (or create) the manifest so the next run has an accurate
    # record of what configuration produced the current cached results.
    try:
        save_json(current_manifest, manifest_path)               # persist effective config
        print(f"[Pipeline] Run manifest saved to '{manifest_path}'.")
    except Exception as exc:                                     # non-fatal; warn and continue
        logging.warning("[Pipeline] Could not save run manifest: %s", exc)

    return {                                                 # return all phase results
        'phase1': p1,
        'phase2': p2,
        'phase3': p3,
        'phase4': p4,
        'phase5': p5,
        'phase6': p6,
    }

if __name__ == "__main__":  # only execute this block when running this file directly in an IDE or terminal
    from pathlib import Path  # import Path locally so the run block is self-contained for user editing

    # ── User-editable run configuration (PyCharm-friendly) ───────────────────────
    parent_folder = Path(r"E:\R_pan\Calibration\12h-33m-07s\filtered-per-image_cut_tiffs_12h-33m-07s")  # set root directory containing one subfolder per camera
    camera_names = None  # leave as None to auto-discover camera folders; or set e.g. ["cam0", "cam1", "cam2"]
    out_dir = None  # leave as None to default to parent_folder / "metadata"; or set Path("custom/output/path")
    threads = 6  # set optimisation thread count; 6 is a reasonable starting point on an 8-core CPU
    n_lim = None  # set integer to limit images per camera during detection; keep None to use all images
    min_corners = 4  # set minimum corners required per image in phase-2 culling
    high_distortion = False  # enable iterative high-distortion refinement path when True
    enable_phase6_self_calibration = True  # enable optional phase-6 self-calibration when True
    show_plots = True  # set True to display plots interactively at end of run

    # ── Target selection flags (single-source-of-truth) ──────────────────────────
    use_ccube = True  # default selection: Ccube enabled
    use_charuco = False  # default selection: ChArUco disabled

    # Enforce exactly one active target selection to avoid ambiguous configuration.
    if use_ccube == use_charuco:  # True/True or False/False are both invalid states
        raise ValueError(  # raise explicit configuration error for the user
            "Invalid target selection: set exactly one of "
            "'use_ccube' or 'use_charuco' to True."
        )

    # ── Target-specific parameters ────────────────────────────────────────────────
    ccube_n_points = 6  # Ccube: number of ChArUco corners per side of each face
    ccube_length_mm = 30.0  # Ccube: physical face side length in millimetres
    ccube_aruco_dict_name = "DICT_4X4_1000"  # Ccube: OpenCV ArUco dictionary enum name
    ccube_border_fraction = 0.2  # Ccube: marker border fraction used by the target model

    charuco_squares_x = 20  # ChArUco: number of squares along x-axis
    charuco_squares_y = 20  # ChArUco: number of squares along y-axis
    charuco_square_size = 4  # ChArUco: class-specific sizing argument (units per your target convention)

    # Build exactly one target object from the validated boolean selection.
    if use_ccube:  # create a Ccube target when Ccube flag is active
        aruco_enum = getattr(cv.aruco, ccube_aruco_dict_name)  # map dictionary name string to OpenCV enum value
        target = Ccube(  # instantiate Ccube target object consumed by bad_pipeline detection/calibration
            n_points=ccube_n_points,  # pass Ccube corners-per-side parameter
            length=ccube_length_mm,  # pass physical face length parameter
            aruco_dict=aruco_enum,  # pass OpenCV ArUco enum used by this target
            border_fraction=ccube_border_fraction,  # pass border fraction for marker layout
        )
    else:  # create a ChArUco target when ChArUco flag is active
        target = ChArUco(  # instantiate ChArUco target object consumed by bad_pipeline
            charuco_squares_x,  # pass board width in squares
            charuco_squares_y,  # pass board height in squares
            charuco_square_size,  # pass target sizing argument expected by ChArUco class
        )

    # ── Execute bad_pipeline ───────────────────────────────────────────────────────────
    results = run_pipeline(  # run the full phased bad_pipeline with explicit keyword arguments
        parent_folder=parent_folder,  # provide dataset root path
        camera_names=camera_names,  # provide explicit camera names or None for auto-discovery
        out_dir=out_dir,  # provide explicit output directory or None for metadata default
        target=target,  # provide the user-selected target object constructed above
        threads=threads,  # provide optimisation thread count used by BA phases
        n_lim=n_lim,  # provide optional image-limit parameter for detection
        min_corners=min_corners,  # provide culling threshold for phase 2
        high_distortion=high_distortion,  # provide optional high-distortion refinement toggle
        enable_phase6_self_calibration=enable_phase6_self_calibration,  # provide optional phase-6 toggle
        show_plots=show_plots,  # provide optional interactive plotting flag
    )

    print("Pipeline completed.")  # print completion message for quick IDE feedback
    print(f"Completed phases: {list(results.keys())}")  # print top-level result keys for quick verification