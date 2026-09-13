"""
Reading a saved detection, and staging the folder a detection pass reads.
"""
from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Optional

from pyCamSet.workflow.logs import LogFn, discard


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


def detection_cache_name(upscale_factor: int = 1) -> str:
    """The filename a detection pass caches its results under."""
    if upscale_factor != 1:
        return f"detected_datapoints_upscale{upscale_factor}x.pickle"
    return "detected_datapoints.pickle"


@contextlib.contextmanager
def staged_camera_root(
    image_folder: Path,
    cam_folders: list[Path],
    log: LogFn = discard,
    prefix: str = "pycamset_",
) -> Iterator[Path]:
    """
    Yield a folder holding only *cam_folders*, for a detection pass to read.

    A detection pass treats every sub-folder of its root as a camera.  An
    image folder that also holds a workspace, a camset, or the cameras not
    selected for this run therefore cannot be handed over as it stands.  When
    it holds nothing else, it is yielded unchanged and nothing is copied.

    The staged folder is symlinks where the platform allows them, and copies
    where it does not, and is removed on the way out.

    :param image_folder: the root someone chose
    :param cam_folders: the camera folders this run should see
    :param log: what to call with each line of output
    :param prefix: distinguishes one phase's staging folder from another's
    """
    allowed = {folder.name for folder in cam_folders}
    if all(entry.name in allowed for entry in image_folder.iterdir()):
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
        image_folder: Path, selected: Optional[list[str]]) -> list[Path]:
    """
    Return the camera folders of *image_folder*, narrowed to *selected*.

    :param image_folder: the root holding one folder per camera
    :param selected: the camera names to keep, or empty for all of them
    """
    from pyCamSet.workflow.workspace import get_camera_subfolders

    folders = get_camera_subfolders(image_folder)
    if selected:
        wanted = set(selected)
        folders = [folder for folder in folders if folder.name in wanted]
    return folders
