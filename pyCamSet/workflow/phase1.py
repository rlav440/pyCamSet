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
    save_detections,
    selected_camera_folders,
    staged_camera_root,
)
from pyCamSet.workflow.logs import LogFn, captured_output, discard
from pyCamSet.workflow.targets import (
    detector_backend_of_spec,
    target_of_params,
    target_spec_of,
)
from pyCamSet.workflow.workspace import (
    WorkspaceManager,
    copy_file,
    count_images_in_folder,
    get_camera_subfolders,
    make_run_id,
    path_exists,
    workspace_path_for,
)

_LOG = logging.getLogger(__name__)

try:
    from pyCamSet.calibration.camera_calibrator import detect_datapoints_in_imfile
    from pyCamSet.calibration.detection_cache import (
        cache_matches,
        load_verified_cache,
    )
    from pyCamSet.utils.setup_reports import validate_detections

    BACKEND_OK = True
except (ImportError, OSError) as exc:
    cache_matches = None
    detect_datapoints_in_imfile = None
    load_verified_cache = None
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

    The run's artifact (``artifacts['detected_datapoints_pickle']``) is
    always written from the ``(detections, cam_res)`` this call itself
    computed in memory -- never copied from the image folder's own
    detection cache, the staged root's cache, or any other file on disk.
    That holds whether ``caching`` is on or off, and whether this run
    staged its images or not: nothing external is ever adopted as a run's
    own output, so a stale or concurrently-written cache elsewhere can
    never end up recorded as this run's result. A run whose detection pass
    itself failed records its error and produces no artifact at all -- see
    :func:`_detect`.

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
    report: Optional[dict] = None
    error: Optional[str] = None
    detections = None
    cam_res = None

    with captured_output(log):
        try:
            detections, cam_res, diagnostics, report = _detect(params, log)
        except Exception as exc:
            error = str(exc)
            diagnostics["error"] = error

    run_id = make_run_id()
    metadata = {
        "run_id": run_id,
        "phase": "phase1",
        "params": params,
        "diagnostics": diagnostics,
        "report": report,
        "error": error,
    }
    workspace.save_run("phase1", run_id, metadata)

    if detections is not None:
        run_dir = workspace.run_dir("phase1", run_id)
        saved = run_dir / "detected_datapoints.pickle"
        try:
            save_detections(saved, detections, cam_res)
        except OSError as exc:
            metadata["error"] = (
                f"Could not save run-local detected_datapoints.pickle: {exc}")
            workspace.save_run("phase1", run_id, metadata)
            log(f"ERROR: {metadata['error']}")
            return metadata

        metadata.setdefault("artifacts", {})[
            "detected_datapoints_pickle"] = str(saved)
        log(f"Artifact saved: {saved}")
        workspace.save_run("phase1", run_id, metadata)

    log(f"Run saved: {run_id}")
    return metadata


def _cache_name_of(params: dict) -> str:
    """The cache a detection pass with these parameters reads and writes.

    Named for the detector as well as the upscale, so a run never picks up
    detections another detector cached in the same folder.  A spec that
    cannot even be built -- an unregistered target, or one whose module
    fails to import (e.g. a reconstruction-only install missing an optional
    graphics dependency; see docs/troubleshooting.md) -- falls back to the
    detector-less name, the same way a spec :func:`matching_image_folder_cache`
    cannot build falls back to "unconfirmable" rather than raising: an
    import failure here must never crash a whole phase 1 run over an
    optional cache lookup.
    """
    try:
        backend = detector_backend_of_spec(target_spec_of(params))
    except Exception:
        backend = None
    return detection_cache_name(params.get("upscale_factor", 1), backend)


def _confirmed_cache_source(candidate: Path, params: dict) -> bool:
    """Whether *candidate* still pairs with *params*' own identity, right now.

    :func:`matching_image_folder_cache`'s own identity check, factored out:
    ``False`` for every reason :func:`cache_matches` itself gives one, plus
    a target/camera selection that cannot even be built (e.g. a legacy run
    with no target recorded at all, or one whose module fails to import) --
    never adopt sight unseen.

    :param candidate: the cache pickle to check
    :param params: a phase's parameters, or a saved run's ``params``
    """
    if not BACKEND_OK or cache_matches is None:
        return False
    f_loc = params.get("f_loc")
    if not f_loc:
        return False
    try:
        target = target_of_params(params)
        cam_names = [folder.name for folder in selected_camera_folders(
            Path(f_loc), params.get("selected_cameras"))]
    except Exception:
        return False
    return cache_matches(candidate, target, cam_names, params.get("n_lim"))


def matching_image_folder_cache(params: dict) -> Optional[Path]:
    """The image folder's own detection cache, verified against *params*.

    ``None`` when there is no cache under the name these params compute, or
    when its identity sidecar does not confirm it was produced for this
    exact target, camera selection and image cap -- the cache is then
    unusable even though a file of that name exists, e.g. a different
    target or an older run left one sharing this detector's cache name.

    Used only by the GUI's last-resort resolver (``phase_1_detection.py``'s
    ``_resolve_pickle_path_for_run``), to find something to *draw* for a run
    with no artifact of its own -- never to decide what a run's own
    permanent artifact is.  :func:`run` never calls this: a run's artifact
    is always written from the detections it computed itself (see its own
    docstring), so there is nothing here for it to adopt.

    :param params: a phase's parameters, or a saved run's ``params``
    """
    f_loc = params.get("f_loc")
    if not f_loc:
        return None
    candidate = Path(f_loc) / _cache_name_of(params)
    if not path_exists(candidate):
        return None
    return candidate if _confirmed_cache_source(candidate, params) else None


def load_matching_image_folder_cache(params: dict) -> Optional[tuple]:
    """The image folder's own detection cache, verified AND read together.

    Unlike :func:`matching_image_folder_cache` (a path, confirmed only at
    the moment this call happens), this re-confirms identity and reads the
    pickle's bytes at the instant of the load, then deserialises those same
    bytes -- so there is no gap, between resolving a path and a caller
    reading it later, for a concurrent Phase 1 run sharing this cache slot
    to overwrite it in. Used by the GUI's Draw Detections
    (``pyCamSet.gui.phase_1_detection._draw_detections_for_run``), which
    must never trust a path it resolved a moment earlier without
    re-verifying it right before the read -- see
    :func:`~pyCamSet.calibration.camera_calibrator.load_verified_cache`.

    :param params: a phase's parameters, or a saved run's ``params``
    :return: ``(detected, cam_res)`` on a confirmed hit, or ``None`` for
        every reason :func:`matching_image_folder_cache` itself would --
        including the identity having changed since an earlier resolution.
    """
    if not BACKEND_OK or load_verified_cache is None:
        return None
    f_loc = params.get("f_loc")
    if not f_loc:
        return None
    candidate = Path(f_loc) / _cache_name_of(params)
    if not path_exists(candidate):
        return None
    try:
        target = target_of_params(params)
        cam_names = [folder.name for folder in selected_camera_folders(
            Path(f_loc), params.get("selected_cameras"))]
    except Exception:
        return None
    return load_verified_cache(candidate, target, cam_names, params.get("n_lim"))


def _detect(params: dict, log: LogFn) -> tuple[object, list, dict, dict]:
    """Run the detection pass and compute its diagnostics.

    Returns the ``(detections, cam_res)`` this call itself computed --
    :func:`run` writes those straight to the run's artifact.  The image
    folder's own detection cache (and, when staging, the staged root's
    copy of it) is read and written only as :func:`~pyCamSet.calibration.
    camera_calibrator.detect_datapoints_in_imfile`'s own speed-up, gated
    throughout on ``params['caching']``: every cache file touch here is a
    copy of that speed-up's cache home to and from the staged root, and
    every one of those copies is best-effort -- it can make a later run
    redetect instead of hitting the cache, but it can never change what
    THIS call returns.
    """
    if not BACKEND_OK:
        raise RuntimeError("pyCamSet detection modules are not importable.")

    f_loc = Path(params["f_loc"])
    upscale_factor = params.get("upscale_factor", 1)
    caching = bool(params["caching"])

    # Scanned fresh here for the camera-subset selection below -- NOT reused
    # for staged_camera_root's own "is staging needed" check further down.
    # Round-9 P1: this same scan used to be threaded through to that check
    # too, but real work happens between the two (image counting,
    # target_of_params(), cache-name computation), and if a new, unselected
    # camera folder appears on disk in that window, a scan taken here would
    # be stale by the time staged_camera_root uses it -- wrongly concluding
    # nothing needs staging and letting the new folder slip into this run's
    # own detections, unfiltered. staged_camera_root takes its own,
    # separately fresh scan immediately before it needs one instead (below).
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
    cache_name = _cache_name_of(params)

    # Scanned again, as late as possible -- immediately before the
    # containment check this feeds -- rather than reusing the selection
    # scan above (see the comment there). A folder that appears after THIS
    # scan is still a gap in principle, but it is now as small as the
    # single call below, not the whole of the slow work above.
    all_camera_folders = get_camera_subfolders(f_loc)
    with staged_camera_root(f_loc, cam_folders, log, "pycamset_phase1_",
                             all_camera_folders=all_camera_folders) as root:
        if root != f_loc and caching:
            # staged_camera_root hands a strict camera subset a fresh
            # TemporaryDirectory on every call, so detect_datapoints_in_imfile's
            # own cache lookup (against root / cache_name) can never see a
            # cache: even a previous run of this EXACT subset only ever left
            # one beside the images at f_loc, via the copy-back below.
            # Bringing that cache into the staged root first is what lets a
            # rerun of the same subset hit it at all. cam_names
            # is passed to detect_datapoints_in_imfile explicitly below (this
            # run's own selection, pinned above), so a different subset's
            # identity still fails to match and correctly redetects. Best-effort only
            # -- see this function's own docstring -- a failure here can
            # only cost a cache hit, never this call's own detections.
            cached_source = f_loc / cache_name
            if path_exists(cached_source):
                try:
                    copy_file(cached_source, root / cache_name)
                except OSError as exc:
                    # A race with a concurrent writer at f_loc (the source
                    # unlinked between the path_exists() checks above and
                    # this copy, a transient sharing violation) must not
                    # abort the whole detection pass over what a same-subset
                    # rerun would merely have missed as a cache hit. Skip
                    # the pre-warm and let detect_datapoints_in_imfile
                    # redetect into an empty staged root instead -- it
                    # writes its own fresh cache there, which the copy-back
                    # below still brings home to f_loc.
                    _LOG.warning(
                        "Could not pre-warm the staged root with the "
                        "existing cache at %s: %s; proceeding to redetect.",
                        cached_source, exc)

        detections, cam_res = detect_datapoints_in_imfile(
            f_loc=root,
            calibration_target=target,
            caching=caching,
            draw=False,
            n_lim=params["n_lim"],
            upscale_factor=upscale_factor,
            # This run's own selection, fixed above before any staging
            # decision or later race window -- passed through explicitly so
            # detect_datapoints_in_imfile never re-derives it from a fresh
            # scan of root/f_loc. Without this, a camera folder that
            # appeared on disk after cam_folders was selected (staged: not
            # possible, since root is this call's own fresh, filtered temp
            # dir; unstaged: f_loc is the real, persistent image folder)
            # would still be picked up by that scan and silently folded
            # into this run's own detections, regardless of how fresh
            # staged_camera_root's own containment check was.
            cam_names=cam_names,
        )
        log("1b  Detection complete.")

        if root != f_loc and caching:
            # The cache lands beside the images the pass read, i.e. the
            # staging folder -- bring it back to the image folder, where a
            # rerun of this same subset looks for it. Purely a speed-up for a LATER run: this call's own
            # (detections, cam_res) below are already final and returned
            # regardless of whether this copy-back succeeds, so the whole
            # thing is best-effort -- any failure here is logged and
            # skipped, never raised past a successful detection.
            cached = root / cache_name
            if path_exists(cached):
                destination = f_loc / cache_name
                try:
                    copy_file(cached, destination)
                except OSError as exc:
                    _LOG.warning(
                        "Could not copy the freshly written detection "
                        "cache back to %s: %s; a later run of this image "
                        "folder redetects rather than trusting a stale or "
                        "partial copy -- this run's own detections are "
                        "unaffected.", destination, exc)

    report = validate_detections(
        detections, target, image_counts=cam_img_counts, n_lim=params["n_lim"])

    diagnostics = _diagnostics(report, detections, cam_res, log)
    log("Phase 1 complete.")
    return detections, cam_res, diagnostics, report.to_dict()


def _diagnostics(report, detections, cam_res, log: LogFn) -> dict:
    """The D1 series: how much of the target each camera actually saw.

    The per camera detection rate and board completeness are the detection
    summary's own numbers, taken off the report that has just been printed
    rather than measured a second time here.
    """
    diagnostics: dict = {}
    try:
        diagnostics["D1.1_total_detections"] = {
            cam.name: cam.n_features for cam in report.per_camera}
        diagnostics["D1.2_detection_rate"] = {
            cam.name: cam.detection_rate for cam in report.per_camera}
        diagnostics["D1.3_board_completeness"] = {
            cam.name: cam.completeness for cam in report.per_camera}

        features = detections.features_per_im_per_cam()
        diagnostics["D1.4_features_matrix"] = features.tolist()

        coverage = _spatial_coverage(detections, cam_res)
        if coverage is not None:
            diagnostics["D1.6_spatial_coverage"] = coverage

        min_features = int(np.min(features[features > 0])) if np.any(features > 0) else 0
        diagnostics["D1.7_min_features"] = min_features

        diagnostics["cam_names"] = list(report.camera_names)
        diagnostics["n_images"] = report.n_images
    except Exception as exc:
        log(f"  (partial diagnostics: {exc})")
    return diagnostics


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
