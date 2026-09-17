from copy import copy
import hashlib
import json
import multiprocessing
import cv2
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import reduce
import re
import os
from typing import Optional


from pyCamSet.cameras import CameraSet, Camera
from pyCamSet.calibration_targets import TargetDetection, AbstractTarget
from pyCamSet.calibration_targets.core.target_registry import spec_of
# from pyCamSet.optimisation.base_optimiser import run_bundle_adjustment, TemplateBundleHandler
from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment
from pyCamSet.optimisation.template_handler import (
    TemplateBundleHandler, DEFAULT_OPTIONS)
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
from pyCamSet.utils.saving import (
    save_pickle, load_CameraSet, _normalise_windows_open_path,
)
from pyCamSet.utils.general_utils import average_tforms, get_subfolder_names, glob_ims, mad_outlier_detection

import logging

from pyCamSet.utils.logs import setup_logging_from_verbosity
from pyCamSet.utils.intrinsics_report import IntrinsicsReport
from pyCamSet.utils.setup_reports import DetectionReport

logger = logging.getLogger(__name__)


def calibrate_cameras(
    f_loc: Path|str,
    calibration_target: AbstractTarget,
    save: bool = True,
    save_loc: Path|None = None,
    draw=False,
    n_lim=None,
    fixed_params: dict | None =None,
    high_distortion=False,
    threads=None,
    problem_options: dict|None = None,
    initial_cams: CameraSet | None= None,
    min_detections_per_board: int = 12,
    optimise_target: bool = False,
    model: str = "pinhole",
    ) -> CameraSet:
    """
    This function coordinates the calibration process, from detection to outputing a final camset.

    :param f_loc: the folder containing the nested cam images
    :param calibration_target: the calibration target
    :param save: should the final camset be saved
    :param save_loc: where should the final camset be saved
    :param draw: should the detection be drawn as the detections are completed
    :param n_lim: the maximum number of images to use for detection
    :param fixed_params: a dictionary of fixed parameters for the optimisation, which will not be changed
    :param high_distortion: Implements an iterative scheme for high distortion cameras.
    :param model: the lens model to calibrate, "pinhole" or "telecentric". Every
        camera in a run shares one model, because one kernel is compiled per
        calibration.
    :param min_detections_per_board: Minimum number of detected corners required
        for a board observation to contribute to the initial per-camera calibration.
    :param optimise_target: solve the target's own geometry as well, in a second
        bundle adjustment started from the first.
    """

    setup_logging_from_verbosity(
        (problem_options or {}).get("verbosity", DEFAULT_OPTIONS["verbosity"]))

    if isinstance(f_loc, str):
        f_loc = Path(f_loc)

    if save_loc is None:
        save_loc = f_loc

    if threads is None:
        threads = min(max(1, cpu_count()-2), 20) # don't DOS servers

    detections, camera_res = detect_datapoints_in_imfile(
        f_loc=f_loc,
        caching=save,
        calibration_target=calibration_target,
        draw=draw,
        n_lim=n_lim,
        threads=threads,
    )

    validate_detections(detections, calibration_target,
                        image_counts=images_per_camera(f_loc), n_lim=n_lim)

    string_tail = '.camset'
    if initial_cams is None:

        initial_cams = run_initial_calibration(
            detections,
            calibration_target,
            camera_res,
            save=save,
            save_loc=save_loc / ('initial_cameras' + string_tail),
            fixed_params=fixed_params,
            min_detections_per_board=min_detections_per_board,
            model=model,
        )



        if high_distortion:
            detections, _ = detect_datapoints_in_imfile(
                f_loc=f_loc,
                calibration_target=calibration_target,
                draw=draw,
                n_lim=n_lim,
                camset=initial_cams
            )

            initial_cams = run_initial_calibration(
                detections,
                calibration_target,
                camera_res,
                save=save,
                save_loc=save_loc / ('initial_cameras_high_distortion' + string_tail),
                min_detections_per_board=min_detections_per_board,
                model=model,
                )

            # as outlier_rejection already does: only open a window when
            # someone is there to close it
            initial_cams.draw_camera_distortions(show=draw)
    else:
        logger.info("Using the provided initial cameras.")

    initial_cams.set_resolutions_from_file(floc=f_loc)
    report_initial_calibration(initial_cams, detections, calibration_target,
                               min_detections_per_board)
    if len(initial_cams) == 1:
        logger.warning("Only found and calibrated one camera - returning single camera calibration")
        return initial_cams



    calibrated_cameras = run_stereo_calibration(
        initial_cams,
        detections,
        calibration_target,
        save=False, #TODO REMOVE
        save_loc=save_loc/('optimised_cameras' + string_tail),
        fixed_params=fixed_params,
        threads = threads,
        problem_options = problem_options,
    )

    if optimise_target:
        calibrated_cameras = run_self_calibration(
            calibrated_cameras,
            detections,
            calibration_target,
            fixed_params=fixed_params,
            threads=threads,
            problem_options=problem_options,
        )

    return calibrated_cameras


def run_self_calibration(
    cams: CameraSet,
    detections: TargetDetection,
    target: AbstractTarget,
    fixed_params: dict|None = None,
    threads: int = 1,
    problem_options: dict|None = None,
) -> CameraSet:
    """
    Solves the target's own geometry, starting from a finished calibration.

    The same observations are solved a second time with every feature
    coordinate free, so what was being blamed on the cameras but is really the
    printing and assembly of the target comes out of the reprojection error.
    The scale of the result is then gauged by three of the target's own points
    rather than by the target as drawn.

    :param cams: the cameras a fixed target calibration produced
    :param detections: the detections that calibration was solved against
    :param target: the calibration target, as drawn
    :param fixed_params: parameters the optimisation is not allowed to move
    :param threads: evaluation threads for the compiled kernels
    :param problem_options: options passed through to the solver
    :return: the camera set that minimises the free target problem
    """
    logger.info("Running the calibration again with the target's geometry free")
    param_handler = SelfBundleHandler(
        camset=cams, target=target, detection=detections,
        fixed_params=fixed_params, options=problem_options,
    )
    param_handler.set_from_templated_camset(cams)
    _, self_calibrated = run_bundle_adjustment(
        param_handler=param_handler, threads=threads)
    return self_calibrated


def run_initial_calibration(detection: TargetDetection,
                            calibration_target: AbstractTarget,
                            cam_res: list[tuple],
                            save=True, save_loc: Path = Path('initial_estimate.camset'),
                            ref_cam: int|str = 0,
                            fixed_params: dict|None = None,
                            return_poses_and_costs=False,
                            min_detections_per_board: int = 12,
                            model: str = "pinhole") -> CameraSet | tuple[CameraSet, np.ndarray, np.ndarray]:
    """
    For all of the cameras, runs the calibration method provided by an abstract target.
    The default is an opencv calibration but may be overwritten.

    :param detection: the detection data to use for the calibration
    :param calibration_target: the calibration target to use for the calibration
    :param save: should the result be saved
    :param save_loc: where should the result be saved
    :param fixed_params: a dictionary of fixed parameters for the optimisation, which will not be changed
    :param model: the lens model to fit, "pinhole" or "telecentric"
    :param min_detections_per_board: Minimum number of detected corners required
        for a board observation to contribute to the initial per-camera calibration.
    :return: the camera set with the initial calibration
    """

    if save_loc.exists() and save:
        logger.info(f"Loading a previously saved initial calib from {save_loc}")
        cams = load_CameraSet(save_loc)
        if return_poses_and_costs is False:
            return cams
        pose_path = save_loc.parent/'calib_pose_data.npy'
        per_im_path = save_loc.parent/'calib_im_data.npy'
        poses = np.load(pose_path) if pose_path.exists() else np.array([], dtype=float)
        per_im = np.load(per_im_path) if per_im_path.exists() else np.array([], dtype=float)
        return cams, poses, per_im

    # define the input structure to the
    # inp = data, target, intial_estimate, camera_res
    c_m = detection.features_per_im_per_cam()
    if c_m.size == 0:
        raise ValueError(
            "No detection features were found for any camera/image. "
            "Check that the calibration target matches the detected board "
            "type and that the test data contains valid images."
        )
    mask = ~np.any(c_m < 6, axis=1)
    score = np.sum(c_m, axis=1)
    pose_im = np.argmax(score * mask)
    # create a lambda based on the inputs

    logger.info("Pulling calibration method from target")
    work_fn = lambda datum: calibration_target.initial_calibration(
            cam_name=datum[0],
            detection=datum[1],
            res=datum[2],
            pose_im=pose_im,
            fixed_params=fixed_params,
            return_poses=True,
            model=model,
        )
    cam_names = detection.cam_names
    cam_detections = detection.get_cam_list()
    work_data = zip(cam_names, cam_detections, cam_res)
    if return_poses_and_costs:
        work_fn = lambda datum: calibration_target.initial_calibration(
                cam_name=datum[0],
                detection=datum[1],
                res=datum[2],
                pose_im=pose_im,
                fixed_params=fixed_params,
                return_poses=True,
                min_detections_per_board=min_detections_per_board,
                model=model,
            )
        results = [work_fn(datum) for datum in work_data]
        raw_calibration = [res[0] for res in results]
        poses = [res[1] for res in results]
        per_im = [res[2] for res in results]
    else:
        work_fn = lambda datum: calibration_target.initial_calibration(
                cam_name=datum[0],
                detection=datum[1],
                res=datum[2],
                pose_im=pose_im,
                fixed_params=fixed_params,
                return_poses=False,
                min_detections_per_board=min_detections_per_board,
                model=model,
            )
        raw_calibration = [work_fn(datum) for datum in work_data]
        poses = []
        per_im = []
    cam_dict = {cam_name: cam for cam_name, cam in zip(cam_names, raw_calibration)}
    cams = CameraSet(camera_dict=cam_dict)

    if save:
        cams.save(save_loc)
        # np.save(save_loc.parent/'calib_pose_data.npy', poses)
        # np.save(save_loc.parent/'calib_im_data.npy', per_im)

    if not return_poses_and_costs:
        return cams
    return cams, poses, per_im


def outlier_rejection(results, params, draw: bool = True) -> tuple[TargetDetection | None, bool]:
    """
    Takes a set of results from the optimisation and performs outlier rejection on them.
    Will identify which images are outliers, raise a warning, and return a detection set without this data.

    :param results: the per detection residuals of the optimisation
    :param params: the parameter handler the residuals came from
    :param draw: whether to show the per image error boxplot
    :return: A target detection without the outliers, and whether any were found.
    """
    # outliers = mad_outlier_detection(results)

    detection = params.get_detection_data()
    # plot this as a boxplot
    d_list = [[] for _ in range(params.detection.max_ims)]
    for global_im_num, errs in zip(detection[:, 1], results):
        d_list[int(global_im_num)].append(errs)

    per_im_outliers = mad_outlier_detection([np.mean(datum) for datum in d_list if datum],
                                            draw=False,
                                            out_thresh=5)
    # Only draw when someone is there to look: plt.show() from a batch run or
    # a test is at best wasted work and at worst a blocking window.
    if draw:
        plt.boxplot(d_list)
        plt.ylabel("Average Pixels Reprojection error")
        if per_im_outliers is not None:
            plt.title(f"Images {list([per_im_outliers][0])} are likely outliers")
        else:
            plt.title("Reprojection error per image")
        plt.show()

    if per_im_outliers is None:
        return None, False
    logger.info("deleting datum associated with the above outliers")
    data = params.detection
    return data.delete_row(global_im_num=per_im_outliers), True

def run_stereo_calibration(
    cams: CameraSet,
    detections: TargetDetection,
    target: AbstractTarget,
    param_handler = None,
    save: bool=True,
    save_loc: Path|None = None,
    fixed_params: dict|None=None,
    floc: Path|None=None,
    threads: int = 1,
    problem_options: dict|None = None,
    est_poses: Optional[np.ndarray] = None,
    pose_errors: Optional[np.ndarray] = None,
) -> CameraSet:
    """
    This code runs a multi camera stereo calibration.
    The default behaviour is to run a standard object pose based bundle adjustment.

    :param param_handler: The parameter handler to use. If none is provided, a standard one will be created.
    :param save: should the result be saved
    :param save_loc: where should the result be saved
    :param fixed_params: a dictionary of fixed parameters for the optimisation, which will not be changed
    :param floc: the location of the images, used to update the camera resolutions.
                 When supplied, set_resolutions_from_file() is always called regardless
                 of the save flag, so that the returned CameraSet has correct resolution
                 data even when save=False.
    """
    logger.info("Running the full multiview calibration")

    if save_loc is None:
        save_loc = Path('optimised_cameras.camset')

    if param_handler is None:
        param_handler = TemplateBundleHandler(
            detection=detections, target=target, camset=cams,
            fixed_params=fixed_params,
            options=problem_options,
            # TODO - USE THE EXISTING CALIBRATION RESULTS TO SOLVE THIS.
        )

    optimisation, optimised_cams = run_bundle_adjustment(
        param_handler=param_handler,
        threads = threads,
    )

    param_handler.camset = optimised_cams
    # optimised_cams = param_handler.camset
    # outlier_rejection(optimisation.fun.reshape((-1,2)), param_handler)


    if floc is not None:
        optimised_cams.set_resolutions_from_file(floc)
    if save:
        optimised_cams.save(save_loc)
    return optimised_cams


def detector_backend_of(target) -> str | None:
    """
    The detector a target object is read with, when it says.

    ChArUco and Ccube carry the one they were built with as
    ``marker_backend``; a target that can only be read one way (ChArUco2,
    PuzzleBoard) has no such attribute, and is read with its only detector.

    :param target: any calibration target
    :return: a key of the target's ``DETECTOR_BACKENDS``, or None when it
        cannot be told
    """
    backend = getattr(target, "marker_backend", None)
    if backend:
        return str(backend)
    backends = tuple(getattr(type(target), "DETECTOR_BACKENDS", None) or ())
    return backends[0] if len(backends) == 1 else None


def cache_identity_path(cache_path: Path) -> Path:
    """The identity sidecar written beside a detection cache."""
    return cache_path.with_name(cache_path.stem + ".identity.json")


def _target_identity(calibration_target, cam_names: list[str],
                      n_lim: int | None,
                      camset: CameraSet | None = None) -> dict | None:
    """The identity a detection cache is checked against.

    None when the target cannot be described (not a registered type, e.g. a
    hand-built test double) -- callers must then never trust, and never
    write, an identity for it. Also unconditionally None whenever *camset*
    was passed at all.

    A camset is only ever passed on the ``high_distortion`` redetection path
    (detection biased toward a specific, evolving calibration), and an
    earlier version of this function tried to fingerprint that camset's per
    camera parameters (intrinsic/extrinsic/distortion_coefs/res) so two
    different camsets on the same target/cam_names/n_lim would not share a
    cache slot. That fingerprint kept missing model-specific state -- it
    took a P1 finding about ``TelecentricCamera.telecentricity`` to notice
    the gap, and every future camera model with its own extra parameter
    would reopen it the same way. Rather than chase Camera subclasses one at
    a time, a camset-bearing identity is simply unconfirmable, exactly like
    an unregistered target: such a call is always a cache miss (a safe,
    if occasionally redundant, redetection) and, via ``write_cache_identity``
    reading this same None, never gets an identity sidecar written for it
    either -- so a camset-bearing cache can never be read back as a hit,
    under its own camset or anyone else's.

    :param camset: the camera set the caller is biasing detection with, when
        there is one. Passing one always makes the identity unconfirmable
        (see above). ``None`` (the ordinary case) leaves the identity
        exactly as before -- a cache written or checked without a camset is
        unaffected either way.
    """
    if camset is not None:
        return None
    try:
        spec = spec_of(calibration_target)
    except ValueError:
        return None
    return {"target_spec": spec, "cam_names": sorted(cam_names), "n_lim": n_lim}


def _identity_text(payload: dict) -> str:
    # Strict equality only: a spurious MISS just costs one extra, safe
    # redetection; a spurious HIT is the exact defect being removed. No
    # numeric/float tolerance, unlike workflow/targets.py's _same_value,
    # which answers an unrelated question over a narrower field set.
    #
    # Also used to canonicalise the identity half of a sidecar that has
    # already been round-tripped through JSON (cache_matches), so the same
    # comparison applies whether *payload* still holds live target objects
    # or plain JSON-native values read back off disk.
    return json.dumps(payload, sort_keys=True, default=str)


def _long_path(path: Path | str) -> Path:
    """*path* as a ``Path`` safe for Windows long-path I/O.

    Windows caps a path at 260 characters unless it is given in
    extended-length (``\\\\?\\``) form -- see
    :func:`pyCamSet.utils.saving._normalise_windows_open_path`, reused here
    for the identical prefixing, so ``open()``, ``.exists()``,
    ``.write_text()`` and ``.unlink()`` on the detection cache and its
    identity sidecar all engage at the same path lengths
    ``save_pickle``/``load_pickle`` (also in that module) already do.
    Without this, every one of those calls silently degrades to "cache
    unreadable/unwritable" once the cache path crosses ~248 characters --
    realistic for a deeply nested image-folder layout -- and caching never
    engages for that folder, even though the underlying file is fine.
    """
    return Path(_normalise_windows_open_path(path))


def file_sha256(path: Path) -> str | None:
    """SHA-256 of *path*'s current bytes, or None when it cannot be read.

    Pairs a sidecar with the exact pickle bytes it was written for, so a
    hand-copied sidecar next to someone else's pickle, or a pickle rewritten
    mid-crash while an old sidecar happened to survive, is always a MISS
    rather than a silently trusted mismatch. Public (not ``_file_sha256``)
    because :func:`cache_matches` and :mod:`pyCamSet.workflow.phase1`'s own
    identity checks (``matching_image_folder_cache``) both call it directly,
    as a standalone digest of whatever is currently on disk -- unlike
    :func:`_load_cache_if_verified`'s hot-path hash, which is taken from
    bytes already held in memory rather than a second read of the file.
    """
    try:
        digest = hashlib.sha256()
        with open(_long_path(path), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _sidecar_record(cache_path: Path) -> dict | None:
    """The sidecar's parsed identity and digest, or None when it cannot be
    trusted at all -- missing, unreadable, not JSON, or missing either field.

    Shared by :func:`cache_matches` (a standalone predicate reused by
    :mod:`pyCamSet.workflow.phase1`'s resolvers, and by tests) and by
    :func:`_load_cache_if_verified` (the hot detection path's own,
    single-read check), so the two agree on what counts as a well-formed
    sidecar.
    """
    sidecar = cache_identity_path(cache_path)
    if not _long_path(sidecar).exists():
        return None
    try:
        with open(_long_path(sidecar), encoding="utf-8") as fh:
            recorded = json.load(fh)
    except (OSError, ValueError):
        # ValueError also covers json.JSONDecodeError and the UnicodeDecodeError
        # of a sidecar that is not valid UTF-8 -- both are just an unreadable
        # sidecar, so both are a miss, not a crash.
        return None
    if not isinstance(recorded, dict):
        return None
    recorded_identity = recorded.get("identity")
    recorded_sha256 = recorded.get("cache_sha256")
    if not isinstance(recorded_identity, dict) or not isinstance(recorded_sha256, str):
        return None
    return recorded


def cache_matches(cache_path: Path, calibration_target, cam_names: list[str],
                   n_lim: int | None, camset: CameraSet | None = None) -> bool:
    """Whether *cache_path* was produced for this exact target, camera
    selection and image cap. False whenever this cannot be confirmed --
    cache missing, sidecar missing/unreadable, target unregistered, a
    genuine identity mismatch, the pickle's bytes no longer matching the
    sidecar's recorded digest, or *camset* being given at all -- so an
    unverifiable cache is always a miss, never a silent hit.

    A standalone predicate: reads the sidecar and hashes the whole pickle
    itself, independently of any load.  Used where nothing is about to be
    deserialised anyway (:mod:`pyCamSet.workflow.phase1`'s own identity
    checks, the GUI's last-resort resolver, and the tests below) -- never on
    the cache-hit path inside :func:`detect_datapoints_in_imfile` itself,
    which reads and verifies the pickle's bytes exactly once via
    :func:`_load_cache_if_verified` instead of paying for this function's
    separate re-read on top of the load.

    :param camset: the camera set this call is biasing detection with, when
        there is one. A camset-bearing identity is always unconfirmable
        (see :func:`_target_identity`), so passing any camset here always
        makes this return False -- a camset-bearing cache is never read
        back, whether or not one with a matching target/cam_names/n_lim
        identity exists.
    """
    if not _long_path(cache_path).exists():
        return False
    identity = _target_identity(calibration_target, cam_names, n_lim, camset=camset)
    if identity is None:
        return False
    recorded = _sidecar_record(cache_path)
    if recorded is None:
        return False
    if _identity_text(recorded["identity"]) != _identity_text(identity):
        return False
    current_sha256 = file_sha256(cache_path)
    return current_sha256 is not None and current_sha256 == recorded["cache_sha256"]


def _load_cache_if_verified(cache_path: Path, calibration_target,
                             cam_names: list[str], n_lim: int | None,
                             camset: CameraSet | None = None):
    """A cache hit, read exactly once.

    The hot path inside :func:`detect_datapoints_in_imfile`: identity is
    checked first (cheap -- no read of the pickle itself), then, only once
    that confirms, the pickle's bytes are read ONE time, hashed in memory,
    and -- only once that hash matches the sidecar's recorded digest --
    deserialised from those SAME bytes.  No second read, no second hash, no
    post-load re-check: the digest that confirms the bytes IS the digest of
    the bytes handed to the unpickler, so there is no gap between "verified"
    and "used" for a concurrent writer to land in, and nothing here costs
    more than one read-and-hash of the file on top of the load it was always
    going to pay for.

    :return: ``(detected, cam_res)`` on a confirmed hit, or ``None`` for any
        reason :func:`cache_matches` itself would call a miss (missing
        cache/sidecar, an unreadable or malformed sidecar, a target/camera/
        n_lim identity that does not match, *camset* being given at all,
        bytes that do not pair with the recorded digest, or bytes that fail
        to unpickle).
    """
    if not _long_path(cache_path).exists():
        return None
    identity = _target_identity(calibration_target, cam_names, n_lim, camset=camset)
    if identity is None:
        return None
    recorded = _sidecar_record(cache_path)
    if recorded is None:
        return None
    if _identity_text(recorded["identity"]) != _identity_text(identity):
        return None
    try:
        with open(_long_path(cache_path), "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    if hashlib.sha256(raw).hexdigest() != recorded["cache_sha256"]:
        return None
    try:
        import dill as pickler
    except ImportError:
        import pickle as pickler
    try:
        return pickler.loads(raw)
    except Exception:
        # A hash match with an unpickle failure would mean dill/pickle
        # itself is misbehaving -- treat it the same as every other
        # unconfirmable cache: a safe miss, never a crash.
        return None


def load_verified_cache(cache_path: Path, calibration_target,
                         cam_names: list[str], n_lim: int | None,
                         camset: CameraSet | None = None):
    """Public entry point to :func:`_load_cache_if_verified`, for a caller
    outside this module that needs a confirmed, single-read load of a cache
    pickle.

    Exists because that verify-then-read gap is not just this module's
    problem: the GUI's Draw Detections (``pyCamSet.gui.phase_1_detection.
    _draw_detections_for_run``) resolves a run's last-resort cache path
    once (via ``phase1_workflow.matching_image_folder_cache``, which
    confirms identity at THAT instant) and used to read it raw a moment
    later with no re-check -- exactly the check-to-load race
    :func:`_load_cache_if_verified`'s own docstring describes, just at a
    different call site. Routing that read through here closes it the same
    way: identity confirmed, bytes read once, hashed, and only then
    deserialised from those same bytes -- see
    ``pyCamSet.workflow.phase1.load_matching_image_folder_cache``, which
    wraps this for that caller (not linked: the ``workflow`` API page
    renders the package only, not its submodules, so mkdocstrings/autorefs
    cannot resolve a cross-reference into ``phase1`` and a strict
    ``mkdocs build`` would abort on it).

    :return: ``(detected, cam_res)`` on a confirmed hit, or ``None`` -- see
        :func:`_load_cache_if_verified` for the exact conditions.
    """
    return _load_cache_if_verified(
        cache_path, calibration_target, cam_names, n_lim, camset=camset)


def write_cache_identity(cache_path: Path, calibration_target,
                          cam_names: list[str], n_lim: int | None,
                          cache_sha256: str | None = None,
                          camset: CameraSet | None = None) -> None:
    """Write the sidecar identity beside a freshly written cache.

    Always starts by discarding whatever sidecar is already there, so an
    identity :func:`cache_matches` could never confirm (unregistered target,
    any camset at all, or the cache itself unreadable straight after being
    written) is left with no sidecar at all rather than a stale one -- a
    bare cache with no sidecar is always a miss, which is the safe state.

    :param camset: the camera set this call is biasing detection with, when
        there is one. A camset-bearing identity is always unconfirmable
        (see :func:`_target_identity`), so passing any camset here always
        leaves *cache_path* with no sidecar -- a camset-bearing cache is
        never given one to be read back by.
    :param cache_sha256: the SHA-256 of the exact bytes this call's own
        write put at *cache_path* (e.g. :func:`~pyCamSet.utils.saving.save_pickle`'s
        return value, hashed), when the caller has them. Passing this closes
        a write-write race between two concurrent detection passes sharing
        one cache slot: without it, this function would re-read whatever
        bytes happen to be at *cache_path* right now, which -- if a second
        writer's save_pickle lands between this caller's own save_pickle and
        this call -- are the OTHER writer's bytes, and the sidecar would then
        pair this call's identity with that other pickle. A caller with no
        such bytes on hand (e.g. a test that wrote the pickle directly)
        leaves this ``None`` and the digest is read off disk, as before --
        still correct when nothing else is racing the write.
    """
    sidecar = cache_identity_path(cache_path)
    try:
        _long_path(sidecar).unlink(missing_ok=True)
    except OSError as exc:
        # A transient lock/sharing failure here (a concurrent reader has the
        # sidecar open via cache_matches()'s own open() call, antivirus/
        # backup software briefly holding it, a read-only sidecar) must
        # never propagate out of this call and discard the save_pickle it is
        # meant to record: write_text() below overwrites the stale sidecar's
        # content in place regardless of whether this unlink succeeded, so
        # continue rather than raising past an already-successful write.
        logger.warning(
            "Could not remove the stale detection cache identity sidecar "
            "%s before rewriting it: %s; continuing to write the fresh "
            "identity over it.", sidecar, exc)
    identity = _target_identity(calibration_target, cam_names, n_lim, camset=camset)
    if identity is None:
        return
    if cache_sha256 is None:
        cache_sha256 = file_sha256(cache_path)
    if cache_sha256 is None:
        return
    payload = {"identity": identity, "cache_sha256": cache_sha256}
    try:
        _long_path(sidecar).write_text(_identity_text(payload), encoding="utf-8")
    except OSError as exc:
        # A sidecar path can exceed Windows' MAX_PATH even when the pickle
        # beside it (7 characters shorter) fit -- and this write happens
        # after a successful detection, so raising here would throw away
        # good results over a bookkeeping file. Degrade to "no sidecar"
        # instead: the next run reads that as an unconfirmable cache (a safe
        # miss, per cache_matches), never a silently wrong hit.
        logger.warning(
            "Could not write the detection cache identity sidecar %s: %s; "
            "the cache is left without one, so the next run redetects "
            "rather than trusting an unrecorded identity.", sidecar, exc)


def detect_datapoints_in_imfile(
    f_loc: Path,
    calibration_target: AbstractTarget,
    caching=True,
    cache_name='detected_datapoints.pickle',
    draw=False,
    n_lim=None,
    camset:CameraSet|None = None,
    subfolder_string: str|None = None,
    threads=1,
    upscale_factor:int=1,
    cam_names: list[str] | None = None,
) -> tuple[TargetDetection, list[tuple]]:
    """
    This function organises the detection of the image datapoints in a folder of images.

    :param f_loc: the file location to find the images in
    :param calibration_target: the calibration target to use for the detection
    :param caching:  should the result be cached
    :param cache_name: The name of the cache file
    :param draw: Should the detection be drawn
    :param n_lim: The maximum number of images to use for the detection
    :param camset: Optional, a camera set to use for the detections (for high distortion cameras)
    :param subfolder_string: Optional, the name of an intermediate folder bewtween the camera name folder and the image data.
    :param cam_names: the exact camera folder names this pass must read, when
        the caller already knows them (phase1.py passes its own selection,
        fixed before any staging decision and any later race window). Given
        this, the folders read -- for the cache identity AND for a redetect
        -- come from this list alone, never from a fresh scan of *f_loc*
        taken here: a camera folder that appears in (or vanishes from)
        *f_loc* after the caller made its selection can then never silently
        join or leave this pass, whether or not staging happened to isolate
        it. ``None`` (the default) keeps the original behaviour: both the
        cache identity's camera names and which folders a redetect reads
        come from *f_loc* scanned here, unfiltered.
    :return: A target detection.
    """

    logger.info('starting image detection')

    if camset is not None:
        cache_name = cache_name.split('.')[0] + "_with_calib.pickle"

    # incorporate upscale factor into cache filename so different upscale
    # settings get independent caches. When upscale_factor == 1 (default),
    # keep the original cache name unchanged so existing caches stay valid.
    if upscale_factor != 1:
        base = cache_name.split('.')[0]
        cache_name = f"{base}_upscale{upscale_factor}x.pickle"

    # Likewise the detector: a run read with ArUco 2 must never load the
    # detections an ArUco 1 run cached in the same folder, or the reverse.
    # ArUco 1 keeps the original name, so existing caches stay valid.
    if detector_backend_of(calibration_target) == "aruco2":
        base = cache_name.split('.')[0]
        cache_name = f"{base}_aruco2.pickle"

    # Absolute before any cache/sidecar I/O below: every read goes through
    # this module's own _long_path(), and the pickle write further down
    # goes straight through save_pickle() -- both, in the end, through
    # pyCamSet.utils.saving._normalise_windows_open_path(), which only adds
    # its Windows long-path prefix to a path that is ALREADY absolute. A
    # relative f_loc (calibrate_cameras' own docstring, and docs/how-to/
    # calibrate.md, both use one) would otherwise stay relative all the way
    # through, so caching silently degrades to always-redetect -- and a
    # cache genuinely written elsewhere via an absolute path is never read
    # back either -- once the resolved path crosses Windows' MAX_PATH.
    # f_loc itself is left exactly as given: get_subfolder_names() below has
    # this same, separate, already-accepted long-path limitation regardless
    # of this fix (see tests/test_detection_cache_identity.py's own module
    # docstring), so resolving only cache_path is the minimal change that
    # closes the cache-specific gap without touching that one.
    cache_path = Path(os.path.abspath(f_loc / cache_name))
    # scan_cam_names is False exactly when the caller passed its own,
    # already-fixed cam_names -- see this function's own :param above for
    # why that must then also govern detected_sub_folders below, not just
    # the identity check here.
    scan_cam_names = cam_names is None
    cam_names = get_subfolder_names(f_loc=f_loc) if scan_cam_names else list(cam_names)

    # A cache hit reads and verifies cache_path's bytes exactly once -- see
    # _load_cache_if_verified's own docstring for why that alone already
    # closes the check-to-load race the round-3/round-4 fixes used to patch
    # with a second, separate cache_matches() call after load_pickle(): with
    # only one read, there is no gap between "confirmed" and "used" left for
    # a concurrent writer to land in.
    cache_hit = (_load_cache_if_verified(
        cache_path, calibration_target, cam_names, n_lim, camset=camset)
        if caching else None)
    if cache_hit is not None:
        logger.info('loading cached detection')
        detected, cam_res = cache_hit
        return detected, cam_res

    if not caching:
        logger.info('Not caching, starting detection')
    elif not _long_path(cache_path).exists():
        logger.info('No cached detection, starting detection')
    elif not _long_path(cache_identity_path(cache_path)).exists():
        # Distinct from an outright mismatch below: this cache was never
        # given an identity to check at all (written before this scheme
        # existed, or a previous write_cache_identity degraded to "no
        # sidecar" -- see its own OSError handling) -- not that it was
        # checked and found to disagree.
        logger.info('Cached detection has no identity record (made by '
                    'an earlier version, or its record could not be '
                    'written); redetecting')
    else:
        logger.info('Cached detection does not match this target, camera '
                    'selection or image cap; redetecting')
    detected_sub_folders = (
        get_subfolder_names(f_loc, return_full_path=True) if scan_cam_names
        else [f_loc / name for name in cam_names])

    if not detected_sub_folders:
        raise ValueError(f'no subfolders were found in {f_loc}')

    # checking for uneven image numbers
    sanitise_input_images(detected_sub_folders)

    use_cams = camset is not None

    work_fn = lambda file, cam=None: \
        calibration_target.find_in_imfolder(
            file if subfolder_string is None else file/subfolder_string,
            cam_names=cam_names,
            draw=draw,
            n_lim=n_lim,
            camera=cam,
            threads=threads,
            upscale_factor=upscale_factor,
        )

    if use_cams:
        cam_zip = [camset[f.parts[-1]] for f in detected_sub_folders]
        detections = [work_fn(file, cam) for file, cam in zip(tqdm(detected_sub_folders), cam_zip)]
    else:
        detections = [work_fn(file) for file in tqdm(detected_sub_folders)]
    detected = reduce(lambda x, y: x + y, detections)

    # cam_res must reflect the upscaled coordinate frame, not native.
    # When upscale_factor > 1, detected 2D pixel coords are in the upscaled
    # frame, so cam_res must match. Multiply native .shape[:2] by the factor
    # (cheaper than re-reading the image and resizing it).
    cam_res = [tuple(int(d * upscale_factor) for d in cv2.imread(str(glob_ims(f_loc/cname)[0])).shape[:2]) for cname in cam_names]

    if caching:
        # A1: pairing integrity across the write -- no sidecar can be
        # read as matching a pickle that is mid-rewrite, and the new
        # sidecar is only written once the new pickle is fully on disk.
        try:
            _long_path(cache_identity_path(cache_path)).unlink(missing_ok=True)
        except OSError as exc:
            # A transient lock/sharing failure on the OLD sidecar (a
            # concurrent reader has it open via cache_matches()'s own
            # open() call, antivirus/backup software briefly holding it, a
            # read-only sidecar) must not discard the detection pass that
            # has already completed by this point -- log and continue to
            # save_pickle regardless; write_cache_identity() below still
            # unlinks (and, on its own failure, degrades the same way)
            # before writing the fresh sidecar.
            logger.warning(
                "Could not remove the stale detection cache identity "
                "sidecar for %s: %s; continuing to write the fresh cache "
                "anyway.", cache_path, exc)
        try:
            written = save_pickle((detected, cam_res), cache_path)
        except OSError as exc:
            # This write happens after a successful detection, so raising
            # here would throw away good results over a caching speed-up --
            # the same rationale write_cache_identity() and the sidecar
            # unlink() above already apply to their own writes (e.g. a
            # sidecar/pickle path exceeding Windows' MAX_PATH, disk-full, an
            # AV lock, a read-only folder). There is nothing valid at
            # cache_path to hash, so skip write_cache_identity entirely --
            # the stale sidecar was already unlinked above, so this leaves
            # the cache in the same safe "no sidecar means a miss" state as
            # write_cache_identity's own degraded path -- and return the
            # detections exactly as the caching=False path does.
            logger.warning(
                "Could not write the detection cache %s: %s; returning "
                "this pass's detections without caching them.",
                cache_path, exc)
            return detected, cam_res
        # Hash the exact bytes THIS call wrote, in memory -- never a
        # re-read of cache_path, which by the time write_cache_identity
        # ran could hold a concurrent writer's bytes instead (see that
        # function's cache_sha256 docstring). A monkeypatched save_pickle
        # that returns nothing (some tests replace it with a no-op) falls
        # back to write_cache_identity's own disk read, same as before.
        cache_sha256 = (hashlib.sha256(written).hexdigest()
                        if isinstance(written, (bytes, bytearray)) else None)
        write_cache_identity(cache_path, calibration_target, cam_names, n_lim,
                             cache_sha256=cache_sha256, camset=camset)
    return detected, cam_res

def validate_detections(detected: TargetDetection,
                        target: AbstractTarget,
                        image_counts: dict[str, int] | None = None,
                        n_lim: int | None = None) -> DetectionReport:
    """
    Reports how well each camera saw the target, before anything is solved.

    A calibration can only be as good as its detections, and this is the
    earliest point at which a bad set is visible, so it is worth reading
    before waiting for the per camera calibration that follows.

    :param detected: the detections to describe
    :param target: the calibration target they were found with
    :param image_counts: how many images each camera's folder holds
    :param n_lim: the per camera image cap the detection ran under
    :return: the report, which is also logged
    """
    report = DetectionReport.from_detection(
        detected, target, image_counts=image_counts, n_lim=n_lim)
    logger.info("\n" + report.summary())
    return report


def images_per_camera(f_loc: Path) -> dict[str, int]:
    """
    How many images each camera folder under *f_loc* holds.

    The denominator of the detection rate: a camera is measured against what
    it was given, not against the images another camera happened to have.

    :param f_loc: the folder holding the per camera sub folders
    """
    return {folder.name: len(glob_ims(folder))
            for folder in get_subfolder_names(f_loc, return_full_path=True)}


def report_initial_calibration(cams: CameraSet, detection: TargetDetection,
                               target: AbstractTarget,
                               min_detections_per_board: int = 12,
                               ) -> IntrinsicsReport:
    """
    Reports what each camera's own calibration came out as.

    The stage between the detections and the bundle adjustment: every camera
    has been solved on its own, and a camera whose intrinsics are already
    wrong here will not be rescued by solving them all together.

    :param cams: the per camera calibration to describe
    :param detection: the detections it was solved from
    :param target: the calibration target they were found with
    :param min_detections_per_board: the per board minimum it was solved under
    :return: the report, which is also logged
    """
    report = IntrinsicsReport.from_calibration(
        cams, detection, target, min_detections_per_board)
    logger.info("\n" + report.summary())
    return report


def sanitise_input_images(detected_sub_folders:list[Path], optmode:str='na'):

    """
    Takes a list of detected sub folders and checks that they all have the same number of images.
    :param detected_sub_folders: A list of detected subfolders in the current location.
    :return:
    """
    equal_ims = [len(glob_ims(fol)) for fol in detected_sub_folders]
    if not len(set(equal_ims)) <= 1:
        raise ValueError("An unequal number of calibration images were passed in the input folders.")
