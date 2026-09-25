"""
Reading a saved detection, staging the folder a detection pass reads, and
leaving observations out of a re-run.
"""
from __future__ import annotations

import contextlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Optional

from pyCamSet.workflow.logs import LogFn, discard
from pyCamSet.utils.paths import long_path
from pyCamSet.workflow.workspace import get_camera_subfolders


@dataclass(frozen=True)
class DetectionFilter:
    """
    Observations to leave out of a re-run, and the record of why.

    A diagnostics plot puts a threshold on a run's reprojection error and asks
    for the same phase again without whatever sits above it.  That is the same
    phase, so it is the same runner, with this in front of the detections
    rather than a second copy of the solve.

    Exactly one of the two selections is used: ``images`` drops an image from
    every camera, which is what phase 3 offers, and ``camera_images`` drops it
    from the named camera only, which is what phase 2 offers.

    :param images: image indices to drop across all cameras
    :param camera_images: per camera, the image indices to drop
    :param record: what to write into the run as ``threshold_pruning``
    """

    images: Optional[list[int]] = None
    camera_images: Optional[dict[str, list[int]]] = None
    record: dict[str, Any] = field(default_factory=dict)

    def apply(self, detections, log: LogFn = discard):
        """Return *detections* without the observations this leaves out."""
        log(f"Removing {self.count} {self.unit} via {self.mode}…")
        if self.camera_images is not None:
            filtered = detections.delete_row(cam_im_num=self.camera_images)
        else:
            filtered = detections.delete_row(global_im_num=list(self.images or []))
        log("Filtering complete.")
        return filtered

    @property
    def mode(self) -> str:
        """The filter the detections are asked for, by its own name."""
        return "cam_im_num filter" if self.camera_images is not None else "global_im_num filter"

    @property
    def unit(self) -> str:
        return ("(camera, image) pair(s)" if self.camera_images is not None
                else "image(s)")

    @property
    def count(self) -> int:
        if self.camera_images is not None:
            return sum(len(images) for images in self.camera_images.values())
        return len(self.images or [])


def extract_detection_and_cam_res(payload):
    """
    Return ``(TargetDetection, cam_res)`` from a saved detection payload.

    Two shapes are written: a bare ``TargetDetection``, and the
    ``(TargetDetection, cam_res)`` pair ``detect_datapoints_in_imfile``
    returns.

    :param payload: the unpickled contents of a detections file
    :raises ValueError: for a payload from an earlier build, which has to be
        re-detected rather than read
    """
    if hasattr(payload, "get_cam_list"):
        return payload, None

    if isinstance(payload, (tuple, list)) and len(payload) >= 1:
        detections = payload[0]
        cam_res = payload[1] if len(payload) > 1 else None
        if hasattr(detections, "get_cam_list"):
            return detections, cam_res

    if isinstance(payload, dict):
        raise ValueError(
            "Legacy dict-format detection pickle detected.  "
            "Re-run Phase 1 to produce a current-format "
            "detected_datapoints.pickle."
        )

    raise ValueError(
        f"Unrecognised detection payload type: {type(payload).__name__!r}.  "
        "Re-run Phase 1 to produce a current-format detected_datapoints.pickle."
    )


def extract_detection(payload):
    """Return just the ``TargetDetection`` from a saved detection payload."""
    detections, _ = extract_detection_and_cam_res(payload)
    return detections


def save_detections(path: Path, detections, cam_res=None) -> Path:
    """
    Write *detections* where a later run can read them back.

    Uses dill when it is installed, because a detection holds objects plain
    pickle refuses; pyCamSet depends on dill, so the fallback is for an
    install that has been taken apart.

    :param path: where to write
    :param detections: the detection to save
    :param cam_res: the camera resolutions, when the caller has them
    :return: the path written
    """
    try:
        import dill as pickler
    except ImportError:
        import pickle as pickler

    payload = detections if cam_res is None else (detections, cam_res)
    with open(long_path(path), "wb") as fh:
        pickler.dump(payload, fh)
    return path


def detection_cache_name(upscale_factor: int = 1,
                         marker_backend: str | None = None) -> str:
    """
    The filename a detection pass caches its results under.

    Mirrors :func:`pyCamSet.calibration.camera_calibrator.detect_datapoints_in_imfile`,
    which writes it.

    :param upscale_factor: the upscale the pass ran at
    :param marker_backend: the detector it read the markers with; only
        ``"aruco2"`` changes the name
    """
    name = "detected_datapoints"
    if upscale_factor != 1:
        name += f"_upscale{upscale_factor}x"
    if marker_backend == "aruco2":
        name += "_aruco2"
    return f"{name}.npz"


@contextlib.contextmanager
def staged_camera_root(
    image_folder: Path,
    cam_folders: list[Path],
    log: LogFn = discard,
    prefix: str = "pycamset_",
    all_camera_folders: Optional[list[Path]] = None,
) -> Iterator[Path]:
    """
    Yield a folder holding only *cam_folders*, for a detection pass to read.

    A detection pass treats every sub-folder of its root as a camera.  An
    image folder that also holds the cameras not selected for this run
    therefore cannot be handed over as it stands.  Staging is triggered by an
    unselected *candidate camera folder* -- a directory with images of its
    own that is not one of the selected cameras -- never by any other kind of
    entry: a workspace directory, a detection cache pickle, its identity
    sidecar, a `.camset` file, and any other non-candidate entry are all left
    in place and the folder is yielded unchanged, nothing copied.

    The staged folder is symlinks where the platform allows them, and copies
    where it does not, and is removed on the way out.

    :param image_folder: the root someone chose
    :param cam_folders: the camera folders this run should see
    :param log: what to call with each line of output
    :param prefix: distinguishes one phase's staging folder from another's
    :param all_camera_folders: *image_folder*'s full candidate-camera-folder
        scan, when the caller already has it (e.g. from its own earlier
        :func:`selected_camera_folders` call) -- reused here instead of
        repeating :func:`~pyCamSet.workflow.workspace.get_camera_subfolders`'s
        full per-folder image glob a second time. Scanned fresh when
        omitted, exactly as before.
    """
    allowed = {folder.name for folder in cam_folders}
    present_folders = (get_camera_subfolders(image_folder)
                       if all_camera_folders is None else all_camera_folders)
    present = {folder.name for folder in present_folders}
    if present <= allowed:
        yield image_folder
        return

    with TemporaryDirectory(prefix=prefix) as staged:
        root = Path(staged)
        log("Using filtered staging folder (camera subfolders only).")
        for camera in cam_folders:
            destination = root / camera.name
            try:
                destination.symlink_to(camera, target_is_directory=True)
            except OSError:
                shutil.copytree(camera, destination)
        yield root


def selected_camera_folders(
        image_folder: Path, selected: Optional[list[str]],
        all_camera_folders: Optional[list[Path]] = None) -> list[Path]:
    """
    Return the camera folders of *image_folder*, narrowed to *selected*.

    :param image_folder: the root holding one folder per camera
    :param selected: the camera names to keep, or empty for all of them
    :param all_camera_folders: *image_folder*'s full candidate-camera-folder
        scan, when the caller already has it -- reused instead of a second
        :func:`~pyCamSet.workflow.workspace.get_camera_subfolders` scan.
        Scanned fresh when omitted, exactly as before.
    """
    folders = (get_camera_subfolders(image_folder)
              if all_camera_folders is None else all_camera_folders)
    if selected:
        wanted = set(selected)
        folders = [folder for folder in folders if folder.name in wanted]
    return folders
