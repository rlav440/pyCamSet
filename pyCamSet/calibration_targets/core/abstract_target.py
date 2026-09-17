from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import logging

logger = logging.getLogger(__name__)
import numpy as np
import time
from pathlib import Path
from copy import copy
import cv2
from natsort import natsorted
import multiprocessing
import dill
import os
import signal

from pyCamSet.utils.general_utils import ask_yes_no, glob_ims, h_tform, make_4x4h_tform, mad_outlier_detection, plane_fit
from pyCamSet.cameras import CameraSet, Camera
from pyCamSet.cameras.telecentric_calibration import calibrate_telecentric
from pyCamSet.cameras.telecentric_camera import TelecentricCamera
from pyCamSet.cameras.telecentric_calibration import pose_from_affine
from pyCamSet.calibration_targets.core.parameters import (
    NO_PARAMETERS,
    DetectorParameterisation,
    Parameterisation,
    combine,
)
from pyCamSet.calibration_targets.core.target_detections import TargetDetection, ImageDetection


#: The printable formats a target can be written as, by the name the
#: interface and the generators know them by.
EXPORT_KINDS = ("svg", "pdf_vector", "pdf_raster")

#: The file a printable format is written to.
EXPORT_SUFFIXES = {"svg": ".svg", "pdf_vector": ".pdf", "pdf_raster": ".pdf"}


def get_keys(data):
    """
    Returns keys, of the data, padding with zeros if necessary to maintain a 2nd order descriptor.
    """

    keys = data[:, 2:-2]  # this slicing is always 2d.
    if keys.shape[1] == 1:
        keys = np.concatenate((np.zeros_like(keys), keys), axis=1)
    return keys

# A global variable for the worker process. Each worker will have its own instance.
worker_detector: Optional['AbstractTarget'] = None

def init_worker(detector_class, input_args): #: Optional['AbstractTarget']):
    """
    Initializer for each worker process.
    This function receives the detector object and stores it in a global variable
    for the lifetime of the worker process.
    """
    multiprocessing.freeze_support()
    global worker_detector
    worker_detector = detector_class(**input_args)

def _looks_like_native_opencv_error(exc: BaseException) -> bool:
    """
    Whether ``exc`` carries OpenCV's own C++ exception message shape --
    ``OpenCV(<version>) <file>:<line>: error: (<code>:<name>) <what> in
    function '<func>'`` -- rather than a Python-level message pyCamSet or a
    target itself wrote.

    The rare, clustered OpenCV-internal failure documented in
    ``tests/test_detection_error_isolation.py`` ("a rare, clustered
    cv2.error ... that is not caused by aruco2 and can hit any cv2 aruco
    call in an affected process") normally surfaces as ``cv2.error``, which
    is isolated per-image below. aruco2's own grid-board detector
    (``markers/aruco2_gridboard.py``) can translate the SAME class of
    native OpenCV assertion into a Python ``ValueError`` instead -- that
    module's ``_is_cornersubpix_edge_failure`` documents one specific such
    message; this is the general form of that same check, used here to
    isolate any OTHER native-OpenCV-shaped ``ValueError`` the same way a
    ``cv2.error`` already is. A hand-written ``ValueError`` -- a malformed
    board, a bad argument, an unsupported image dtype -- never has this
    shape, so it is left to propagate and abort the folder/Pool: it is a
    programming or configuration error, not a transient detection failure.
    """
    return str(exc).lstrip().startswith("OpenCV(")


def _process_image(im_file:Path, cam_name:str, idx:int, draw:bool, camera:Camera, upscale_factor:int=1):
    """
    Helper function to process a single image.

    ``cv2.imread`` does not raise on an unreadable or undecodable file; it
    returns ``None``. That is checked explicitly, before anything is done
    with the array, so a corrupt file is a per-image failure rather than an
    ``AttributeError``/``TypeError`` out of ``resize`` or ``find_in_image``
    (which would not be isolated and would abort the Pool). Beyond that,
    ``cv2.error`` is caught here, matching ``find_in_imfolder``'s
    single-process path below, and so is a ``ValueError`` that looks like a
    native OpenCV assertion (see :func:`_looks_like_native_opencv_error`) --
    aruco2's grid-board detector can hand the SAME rare, OpenCV-internal
    failure (see the module docstring-adjacent note on ``find_in_imfolder``)
    back as either exception type. Either way it must not lose the rest of
    the Pool's ``starmap``, so the image is reported back as having no
    detections instead of raising out of the worker. Any other exception --
    including a ``ValueError`` that does not have that native-OpenCV shape --
    is a programming error and is left to propagate, which aborts the Pool
    the same way it always did. The caller does the logging, once all the
    workers' results are in hand, rather than each worker logging on its
    own -- worker process log records do not reach the main process.

    The fifth element of the return tuple (``unreadable``) tells the caller
    which of the two isolated failure kinds ``error`` describes: a file
    ``cv2.imread`` could not decode at all, versus a ``cv2.error`` raised by
    detection itself. The caller (the ``Pool`` aggregator in
    ``find_in_imfolder``) needs this to format the same message the
    single-process path below uses, instead of nesting a pre-formatted
    "could not read image ..." sentence inside a second "detection failed
    ..." wrapper.

    The sixth element of the return tuple carries the same idea for a
    target's own legacy-pattern-mismatch warning (round-2 review, P1):
    ``worker_detector.find_in_image`` may call ``logger.warning`` on its own
    (e.g. ChArUco's/Ccube's ``_warn_legacy_once``), but that call runs in
    THIS worker process and its log record never reaches the main process
    either -- there is no log-forwarding set up between them. A target that
    supports this stashes the fired message on its own
    ``legacy_warning_message`` attribute (None until/unless it fires); read
    back here and handed to the main process alongside the detection, so
    ``find_in_imfolder`` can log it once there. A target with no such
    attribute (most targets) simply reports None, unchanged from before.
    """
    im = cv2.imread(im_file)
    if im is None:
        return cam_name, idx, ImageDetection(), "unreadable or undecodable", True, None
    # Use the globally available detector in this worker process
    try:
        if upscale_factor > 1:
            im = cv2.resize(im, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
        detection = worker_detector.find_in_image(im, draw=draw, camera=camera)
    except (cv2.error, ValueError) as exc:
        if isinstance(exc, ValueError) and not _looks_like_native_opencv_error(exc):
            raise
        return cam_name, idx, ImageDetection(), str(exc), False, None
    warning_message = getattr(worker_detector, "legacy_warning_message", None)
    return cam_name, idx, detection, None, False, warning_message


def _report_folder_detection_failures(
    cam_name: str, folder: Path, n_unreadable: int, n_detect_error: int, n_total: int
) -> None:
    """
    Logs a per-folder summary of isolated per-image detection failures, and
    refuses to hand back a folder's worth of silently empty detections.

    The two failure kinds are tracked and worded separately: an unreadable
    or undecodable file (``cv2.imread`` returning ``None``; no exception is
    ever raised for this) is a file-I/O problem, not an OpenCV detection
    bug, so lumping it into "failed detection with an OpenCV error" would
    misdirect anyone reading the log towards the detector rather than their
    image files.

    :param cam_name: the camera (folder) the images were detected for
    :param folder: the image folder, for the error message
    :param n_unreadable: how many images in the folder ``cv2.imread`` could
        not read or decode (returned ``None``; no exception was raised)
    :param n_detect_error: how many images in the folder raised
        ``cv2.error`` during detection itself
    :param n_total: how many images were attempted
    :raises RuntimeError: if every image in the folder failed
    """
    n_failed = n_unreadable + n_detect_error
    if n_failed == 0:
        return
    parts = []
    if n_unreadable:
        parts.append(f"{n_unreadable} of {n_total} images could not be read")
    if n_detect_error:
        parts.append(
            f"{n_detect_error} of {n_total} images failed detection with an OpenCV error"
        )
    logger.warning(
        f"{cam_name}: " + "; ".join(parts) + " and were recorded as having no detections."
    )
    if n_failed == n_total:
        if n_detect_error == 0:
            raise RuntimeError(
                f"{cam_name}: every one of {n_total} images in {folder} could not "
                f"be read; no usable detections were produced for this camera."
            )
        if n_unreadable == 0:
            raise RuntimeError(
                f"{cam_name}: every one of {n_total} images in {folder} failed "
                f"detection with an OpenCV error; no usable detections were "
                f"produced for this camera."
            )
        raise RuntimeError(
            f"{cam_name}: every one of {n_total} images in {folder} failed "
            f"({n_unreadable} could not be read, {n_detect_error} failed detection "
            f"with an OpenCV error); no usable detections were produced for this "
            f"camera."
        )


class AbstractTarget(ABC):
    """
    This is an abstract calibration target. It implements most of the functionality
    required to calibrate an object with a calibration target.
    Inheriting targets must define:
    1. A function to detect themselves in an input image
    2. the self.point_data array. For a target this is an array of the generic form
    (u,v ... w, n, 3) where u ... w are relevant dimensions, and n,3 is "n" 3d
    and co planar points. As an example, a cube forms a (6,n,3) point_data array, as there are 6 co-planar faces with n points on each face.

    Saving is achieved by calling the __init__ of the super and passing a dictionary of
    the required parameters to initialise the target.

    Inheriting targets may also define: a printable shape, which is either the target, or can then be folded onto a base shape and a plotting function which draws some visualisation of the target.

    A target is also detected somehow, and what that detection can be told is
    described rather than hardcoded: :data:`DETECTOR_BACKENDS` names the
    detectors this target can read itself with and
    :meth:`own_detector_parameters` any settings of its own
    ``find_in_image``, which are the ones that are nobody else's business --
    a cube's face-consistency gate is not the detector's. Together they are
    :meth:`detector_parameterisation`, which is what a form builds itself
    from and what a study sweeps. A target that is never detected declares
    neither and gets an empty parameterisation, which works.

    What the target *is* is described the same way, by
    :meth:`construction_parameters`: the arguments that decide where its
    points are, each with the bounds and the prose a form needs to offer
    it. An interface builds its target controls from that rather than from
    a list of which widget each target reads, so a target it has never
    heard of gets a form. The prose is the constructor's own docstring,
    read back from the ``:param:`` entry that documents each argument, so
    a target is described in the one place a person maintaining it is
    already looking.
    """

    #: The detectors this target can read itself with, by the name its
    #: constructor selects them by. The first is the default. A target that
    #: is never detected leaves this empty.
    DETECTOR_BACKENDS: dict[str, DetectorParameterisation] = {}

    def __init__(self, inputs: dict, backend: str | None = None):
        inputs.pop('self', None)
        inputs.pop('__class__', None)
        for k,v in inputs.items():
            if isinstance(v, np.ndarray):
                inputs[k] = v.tolist()

        self.point_data: np.ndarray = None # An

        self.point_local = None #: np.ndarray = self.make_local()
        self.original_points = None # = self.point_data.copy()
        self.valid_map = True

        # The backend is settled first because the arguments are described in
        # terms of it -- a ChArUco's dictionary names are whichever library
        # reads it -- so asking what this target is, with a detector it cannot
        # be read with, would refuse in the vocabulary of that library rather
        # than in the target's own.
        self.detection_parameters = self.detector_parameterisation(backend)

        problems = self.construction_parameters(backend).validate(inputs)
        if problems:
            # Before anything is built from them.  OpenCV, for one, does not
            # refuse a board too small to exist; it corrupts its own state.
            raise ValueError(" ".join(problems))

        self.detection_options = self.detection_parameters.resolve(
            inputs.get("detection_options"))
        problems = self.detection_parameters.validate(self.detection_options)
        if problems:
            # A form reads these through parse, which checks them; a study
            # and a saved spec reach the constructor directly.
            raise ValueError(" ".join(problems))
        if "detection_options" in inputs:
            # Written back so that a spec taken from this target rebuilds it
            # exactly, whatever subset of the settings it was given.
            inputs["detection_options"] = self.detection_options
        self.input_args = inputs

    @classmethod
    def construction_parameters(cls, backend: str | None = None) -> Parameterisation:
        """
        The arguments that decide what this target is.

        Its geometry, and how it is drawn: everything a form asks for
        besides which detector to read it with.  Declared rather than
        written into each interface, so that a new target gets a form
        without one being written for it.

        Declared by the constructor that takes them, through
        :class:`~pyCamSet.calibration_targets.core.parameters.DocumentedParameters`:
        what each argument is called, defaults to and means is read from
        the signature and the docstring rather than written out again.

        :param backend: which detector the target will be read with, for
            the arguments whose choices depend on it -- a marker dictionary
            is named differently by each marker library
        """
        return NO_PARAMETERS

    @classmethod
    def export_parameters(cls) -> Parameterisation:
        """
        The options that change how this target is drawn, not what it is.

        A cut outline on a net, the border around a board, the resolution a
        raster page is rendered at: none of them change where a point is,
        so they are not part of the target and do not belong in its spec.
        """
        return NO_PARAMETERS

    @classmethod
    def printable_name(cls, values: dict, kind: str = "svg") -> str:
        """
        A filename that says what a target is.

        Taken from the values rather than from a built target, because a
        form shows it as it is typed into and building is not free.

        :param values: the construction parameters, as declared
        :param kind: one of :data:`EXPORT_KINDS`
        """
        raise NotImplementedError

    def save_printable(self, path: Path, kind: str = "svg", **options) -> Path:
        """
        Write this target as a file to print.

        :param path: where to write it
        :param kind: one of :data:`EXPORT_KINDS`
        :param options: the values of :meth:`export_parameters`
        :raises ValueError: for a format this target cannot be written as
        """
        raise NotImplementedError

    @classmethod
    def own_detector_parameters(cls) -> DetectorParameterisation:
        """
        The settings this target's own ``find_in_image`` takes.

        Not the detector's: what the target does with what the detector
        hands back. Targets that do nothing of their own declare nothing.
        """
        return NO_PARAMETERS

    @classmethod
    def detector_parameterisation(cls, backend: str | None = None) -> DetectorParameterisation:
        """
        Everything that alters this target's detection, described as data.

        This target's own settings beside those of the detector it is read
        with, which is what a phase 1 form shows, what a study sweeps, and
        what a target resolves its ``detection_options`` against.

        :param backend: which of :data:`DETECTOR_BACKENDS` to describe;
            defaults to the first
        :raises ValueError: for a backend this target cannot be read with
        """
        if not cls.DETECTOR_BACKENDS:
            return cls.own_detector_parameters()
        if backend is None:
            backend = next(iter(cls.DETECTOR_BACKENDS))
        if backend not in cls.DETECTOR_BACKENDS:
            raise ValueError(
                f"{cls.__name__} cannot be detected with {backend!r}; "
                f"expected one of {', '.join(cls.DETECTOR_BACKENDS)}.")
        return combine(cls.own_detector_parameters(),
                       cls.DETECTOR_BACKENDS[backend])

    def _process_data(self):
        """
        A function called at the end of the __init__ of any inhereting class
        """
        self.point_local = self.make_local()
        self.original_points = self.point_data.copy()

    def plot(self):
        """
        Notes: A function to plot and view the object instance.
        """

        raise NotImplementedError

    def save_to_pdf(self):
        """
        Notes: Implemented by a method that saves a printable version of the
        target to a pdf. Printable version contains information necessary to
        recreate
        """
        raise NotImplementedError

    @abstractmethod
    def find_in_image(self, image, draw=False, camera: Camera=None, wait_len = 1) -> ImageDetection:
        """
        Notes: Detects the calibration target in an image

        :param image: a mxn or mxnx3 image input
        :param draw: whether to draw the target
        :param camera: A camera object for use in camera aware detections
        :return: An ImageDetection object, containing the detected data
        """
        raise NotImplementedError

    def find_in_imfolder(self, file:Path, cam_names, draw=False, n_lim=None, camera: Camera=None, threads=12, upscale_factor:int=1) -> TargetDetection:
        """
        Notes: A function to detect the camera results in the image folder.
        generally a process wrapper around the previous function

        :param file: the top level folder containing the input images
        :param cam_names: the names of the cameras we expect to find
        :param draw: whether to draw the detected images
        :param n_lim: limit on the number of images used
        :param camera: A Camera object, used to optionally increase detection accuracy if the intrinsics are known.

        :return detections: a TargetDetection container for the detection.
                This function is responsible for giving image numbers and camera type to the detector.

        """
        cam_name = file.parts[-1]
        im_locs = [str(x) for x in glob_ims(file)]

        if len(im_locs) == 0:
            raise ValueError(f"No images were found in the given folder {file}")


        im_locs = natsorted(im_locs)
        if n_lim is not None:
            im_locs = im_locs[:n_lim]

        if cam_names is None:
            cam_names = [cam_name]

        # breakpoint()

        detections = TargetDetection(cam_names=cam_names)
       
        # threads=1
        if threads == 1:
            detections = TargetDetection(cam_names=cam_names)
            n_unreadable = 0
            n_detect_error = 0
            for idx, im_file in enumerate(im_locs):
                im = cv2.imread(im_file)
                if im is None:
                    # cv2.imread does not raise on an unreadable or
                    # undecodable file, it returns None -- checked explicitly
                    # here, before resize or find_in_image ever see it, so a
                    # corrupt file is a per-image failure rather than an
                    # AttributeError/TypeError that would abort the folder.
                    logger.warning(
                        f"{cam_name}: could not read image {im_file} "
                        f"(unreadable or undecodable); recording it as "
                        f"having no detections."
                    )
                    n_unreadable += 1
                    detections.add_detection(cam_name, idx, ImageDetection())
                    continue
                # cv2.error is isolated here: a rare, OpenCV-internal
                # failure on one frame (clustered, not caused by aruco2 --
                # see stage A investigation) must not abort the whole folder.
                # aruco2's grid-board detector can hand the same failure back
                # as a ValueError instead (see
                # _looks_like_native_opencv_error), so that is isolated too,
                # but only when it has OpenCV's own native message shape. A
                # programming error -- including a ValueError without that
                # shape -- must still surface, so nothing else is caught.
                try:
                    if upscale_factor > 1:
                        im = cv2.resize(im, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
                    detection = self.find_in_image(im, draw=draw, camera=camera)
                except (cv2.error, ValueError) as exc:
                    if isinstance(exc, ValueError) and not _looks_like_native_opencv_error(exc):
                        raise
                    logger.warning(
                        f"{cam_name}: detection failed for image {im_file} "
                        f"({exc}); recording it as having no detections."
                    )
                    n_detect_error += 1
                    detection = ImageDetection()
                detections.add_detection(cam_name, idx, detection)
            _report_folder_detection_failures(cam_name, file, n_unreadable, n_detect_error, len(im_locs))
            return detections

        if not multiprocessing.current_process().name == "MainProcess":
            logger.error("Multiprocessing Image Detections requires use of the __name__ = '__main__': idiom in your script")
            os.kill(int(os.environ['Detection_PID']), signal.SIGTERM)
            raise RuntimeError

        os.environ['Detection_PID'] = str(os.getpid())

        # prepare arguments for the worker processes
        tasks = [(im_file, cam_name, idx, draw, camera, upscale_factor) for idx, im_file in enumerate(im_locs)]
        # use a Pool of worker processes.
        if not (processname := multiprocessing.current_process().name) == "MainProcess":
            # print(processname)
            logger.critical("Python multiprocessing attempted to start an infinite loop. Use the if __name__ == '__main__' idiom in your calling script to prevent this")
            # multiprocessing.parent_process().terminate()
            raise RuntimeError()

        with multiprocessing.Pool(processes=threads, initializer=init_worker, initargs=(self.__class__, self.input_args)) as pool:
            results = pool.starmap(_process_image, tasks)
        # add the detections from the results in the main process
        n_unreadable = 0
        n_detect_error = 0
        # A worker's own logger.warning calls (e.g. a legacy-pattern-mismatch
        # warning fired inside find_in_image) never reach this process, so
        # _process_image hands the message back with its image's result
        # instead (see its own docstring). Deduplicated by message text
        # rather than logged per-image: several workers can each
        # independently fire the SAME once-per-(worker-)target warning
        # (every worker rebuilds its own fresh target instance), and a
        # multi-face target (Ccube) can legitimately raise more than one
        # DISTINCT message (different faces) -- both are logged, each once.
        warning_messages: dict[str, None] = {}
        for cam, idx, detection, error, unreadable, warning_message in results:
            detections.add_detection(cam, idx, detection)
            if error is not None:
                if unreadable:
                    # Matches the threads==1 message below verbatim: `error`
                    # is already just "unreadable or undecodable" here, so
                    # this is not doubling that reason inside a second,
                    # differently-worded "detection failed" sentence.
                    logger.warning(
                        f"{cam}: could not read image {im_locs[idx]} "
                        f"(unreadable or undecodable); recording it as "
                        f"having no detections."
                    )
                    n_unreadable += 1
                else:
                    logger.warning(
                        f"{cam}: detection failed for image {im_locs[idx]} "
                        f"({error}); recording it as having no detections."
                    )
                    n_detect_error += 1
            if warning_message is not None:
                # A plain dict, not a set: insertion-ordered, so the
                # messages log in the order their images were dispatched --
                # deterministic and easier to read than set iteration order.
                warning_messages.setdefault(warning_message, None)
        # Deduplicated again against messages already logged by an EARLIER
        # find_in_imfolder call on this SAME (main-process) target instance
        # (round-4 review, P2): the real multi-camera workflow calls
        # find_in_imfolder once per camera subfolder on one target instance
        # (camera_calibrator.py), and every one of those calls rebuilds its
        # own fresh Pool workers from self.input_args, so a worker can never
        # know a previous camera folder already reported the same message.
        # Kept as a dedicated instance attribute rather than reusing
        # given_legacy_warning: that flag specifically means "self's OWN
        # find_in_image call saw a mismatch", and several tests rely on it
        # staying False on this main-process instance under threads>1 (self
        # never runs detection itself on that path -- see
        # test_legacy_warning_reaches_main_process_under_multiprocessing).
        already_logged = getattr(self, "_imfolder_logged_legacy_warnings", None)
        if already_logged is None:
            already_logged = set()
            self._imfolder_logged_legacy_warnings = already_logged
        for msg in warning_messages:
            if msg not in already_logged:
                logger.warning(msg)
                already_logged.add(msg)
        _report_folder_detection_failures(cam_name, file, n_unreadable, n_detect_error, len(im_locs))
        return detections

    



    def additional_params(self, x: np.ndarray) -> np.ndarray:
        """
        An object may have additional parameters that are, as yet,
        undefined.
        This method provides a way to pull those results, and transform the state
        of the calibration target.
        """
        return x

    def parametise_features(self, detections: TargetDetection, camset:CameraSet, ref_cam=0):
        """
        A function to parametise any non pose related parameters of the object.
        if there are no such parameters, the function returns none.
        """
        return None

    def pose_in_detections(self, detections: TargetDetection, camset: CameraSet, ref_cam=0
                           ) -> tuple[list[np.ndarray], list[bool]]:
        """
        Returns a list of the found poses of the object in each image of a detection.
        If a pose cannot be found,indicates that the pose is not good in an additional output array

        :param detections: the detections in which to find the pose
        :param camset: The camset to use to detect the poses
        :param ref_cam: if the pose defaults to a reference camera, this is the camera


        :return poses: a list of poses [4x4 homogenous transforms],
        :return pose_d: a boolean list indicating if a pose was found in an image number.

        """
        other_cams = set(range(camset.get_n_cams())) - {0}
        cam = camset[ref_cam]
        poses = []
        for im_list in detections.get_image_list():
            # first try to get the pose with the reference cam
            try:
                pose = self.target_pose_in_cam_image(im_list, cam)
                pose = (cam.cam_to_world @  #cam -> world
                    pose # cube-> cam
                )
            except:
                for other_cam in other_cams:
                    try:
                        pose = self.target_pose_in_cam_image(im_list, camset[other_cam])
                        pose = camset[other_cam].cam_to_world @ pose
                        break
                    except:
                        continue
                else:
                    pose = None
            poses.append(pose)
            # then try to get the pose with any other cams.
        p_detected = np.array([False if p is None else True for p in poses])
        poses = [p for p in poses if p is not None]
        mloc = np.mean([p[:3, 3] for p in poses], axis=0)

        cyclic_outlier_detection = True
        num_loops = 0
        logger.info("Begining outlier detection")
        while cyclic_outlier_detection and num_loops < 10:
            ans = mad_outlier_detection([np.linalg.norm(p[:3,3] - mloc) for p in poses], out_thresh=5)
            inds = np.arange(len(p_detected))[p_detected][ans]
            if ans is not None:
                user_in = "g"
                while not (user_in == 'y' or user_in == 'n'):
                    user_in = ask_yes_no(
                        f"Outliers detected in iteration {num_loops}.",
                        "Do you wish to remove these outliers?",
                        default='n',
                    )

                if user_in == 'y':
                    def del_list_numpy(l, id_to_del):
                        arr = np.array(l)
                        return list(np.delete(arr, id_to_del, axis=0))

                    poses = del_list_numpy(poses, ans)
                    p_detected[inds] = False
                if user_in == 'n':
                    cyclic_outlier_detection = False
            else:
                logger.info(f"No outliers detected in iteration {num_loops}.")
                cyclic_outlier_detection = False
            num_loops += 1

        return poses, p_detected

    def make_local(self):
        """
        The self point data is of the general form: (u,v, ... w, n, 3)
        Calibration approaches assume that each face is locally flat with z = 0
        This computes, for every sub structure (u,v, ..., w) a locally flat sub structure
        representation with the z axis = 0;
        """

        if self.point_data is None:
            raise AttributeError("The self.point_data variable should be set during initialisation")

        if self.point_data.ndim == 2:
            self.point_data = self.point_data[None, ...]

        init_shape = self.point_data.shape

        n = init_shape[-2]
        local_view = np.reshape(self.point_data, (-1, n, 3))

        if local_view.shape[0] == 1:
            return copy(self.point_data)
        ref_point = local_view[:, 0, :]
        init_dir = local_view[:, 1, :] - ref_point

        normals = []
        for face in local_view:
            normals.append(plane_fit(face.T)[1])
        normals = np.array(normals)

        # create the change of basis maxtrixes
        v_3 = np.array([np.cross(v_d, v_n) for v_d, v_n in zip(init_dir, normals)])

        v_3 /= np.linalg.norm(v_3, axis=1, keepdims=True)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        init_dir /= np.linalg.norm(init_dir, axis=1, keepdims=True)

        cob_mats = [np.linalg.inv(a) for a in np.stack(
            (v_3, init_dir, normals)
        ).transpose((1, 0, 2))]
        cob_mats = np.array(cob_mats)
        ref_0 = local_view - ref_point[:, None, :]

        local_coords = (
            ref_0 @ cob_mats
        )
        return np.reshape(local_coords, init_shape)

    @staticmethod
    def _apply_fixed_params_to(init_cam, fixed_params, fixed_param):
        """Overwrites a freshly seeded camera with whatever the caller pinned."""
        if fixed_params is None:
            return init_cam, False
        if "int" in fixed_param:
            init_cam.intrinsic = fixed_param['int']
        if 'dst' in fixed_param:
            init_cam.distortion_coefs = fixed_param['dst']
        if "ext" in fixed_param:
            init_cam.set_extrinsic(fixed_param['ext'])
            return init_cam, True
        return init_cam, False

    def initial_calibration(self, cam_name, detection: TargetDetection,
                            res: list, pose_im: int=0,
                            fixed_params: dict|None =None,
                            return_poses=False,
                            min_detections_per_board: int = 12,
                            model: str = "pinhole") -> Camera | tuple[Camera, np.ndarray, np.ndarray]:
        """
        Takes a single camera's detections, and performs an initial
        calibration on them.
        If the object has a special model of calibration associated, this can
        be overwritten.

        :param cam_name: The name of the camera being calibrated
        :param detection: A TargetDetection of only the detections of the currently
            being calibrated camera
        :param res: The resolution of the camera being calibrated.
        :param pose_im: The image in which the Target's pose sets the coordinate system
        :param fixed_params: A dict containing any fixed params of the camera to calibrate
            accepted options are "ext", "int", and "dst" respectively.
        :param min_detections_per_board: Minimum number of detected corners required
            for a board observation to contribute to the initial OpenCV calibration.
        :param model: the lens model to fit, "pinhole" or "telecentric"
        :return: A camera object.
        """
        if model not in ("pinhole", "telecentric"):
            raise ValueError(
                f"Unknown lens model {model!r}; expected 'pinhole' or 'telecentric'")
        telecentric = model == "telecentric"

        detections_in_image = detection.get(cam=cam_name).get_image_list()
        object_points = []
        image_points = []

        pre_defined_camera = False
        init_cam = Camera()
        fixed_param = {}
        if fixed_params is not None:
            fixed_param = fixed_params.get(cam_name, {})
            if "int" in fixed_param and "dst" in fixed_param:
                cam_class = TelecentricCamera if telecentric else Camera
                init_cam = cam_class(intrinsic=fixed_param['int'], distortion_coefs=fixed_param['dst'], res=res, name=cam_name)
                logger.info(f'Camera {cam_name} was pre determined. Skipping opencv calibration')
                return init_cam


        # A board too sparsely seen to calibrate from is an ordinary outcome --
        # every session has boards caught edge on -- and one line per board
        # buried the reports that matter under dozens of them.  The individual
        # boards stay on the debug record; what is logged is how many.
        n_boards = 0
        dropped: list[int] = []
        sparse: list[int] = []

        for im_detect in detections_in_image:

            data = im_detect.get_data()
            if data is None:
                continue  # no data here, so don't add it to the optimisation
            keys = get_keys(data) # this slicing is always 2d.
            boards, board_id, b_counts = np.unique(keys[:, :-1], return_inverse=True,   return_counts=True)
            mask = boards < np.prod(self.point_local.shape[:-2])

            for board in boards[mask]:
                key_mask = np.squeeze(keys[:, :-1] == board)
                num_detections = int(np.sum(key_mask))
                n_boards += 1
                if num_detections >= min_detections_per_board:
                    if num_detections < 12:
                        sparse.append(num_detections)
                        logger.debug(
                            f"{cam_name}: calibrating from a board with "
                            f"{num_detections} detections. <12 may be an issue."
                        )
                    board_obj = self.point_local[tuple(keys[key_mask].astype(int).T)][None, ...].astype('float32')
                    board_im = data[key_mask, -2:][None, ...].astype('float32')
                    object_points.append(board_obj)
                    image_points.append(board_im)
                else:
                    dropped.append(num_detections)
                    logger.debug(
                        f"{cam_name}: dropping a board with {num_detections} "
                        f"detections, under the {min_detections_per_board} minimum."
                    )

        if dropped:
            logger.info(
                f"{cam_name}: dropped {len(dropped)} of {n_boards} board "
                f"observations under the {min_detections_per_board} detection "
                f"minimum ({min(dropped)}-{max(dropped)} detections each)"
            )
        if sparse:
            logger.warning(
                f"{cam_name}: {len(sparse)} of {n_boards} board observations "
                f"calibrated from fewer than 12 detections, which may be an issue"
            )

        start = time.time()
        if len(object_points) == 0:
            raise ValueError(
                f"Camera {cam_name} has zero valid board detections "
                f"(after the {min_detections_per_board}-detection-per-board "
                f"minimum) — cannot run initial calibration. Check Phase 1 "
                f"detection results for this camera."
            )
        if telecentric:
            # OpenCV has no telecentric model, and the affine seed needs none:
            # with the distortion and the telecentricity error set aside the
            # projection is linear, and the bundle adjustment refines both.
            magnification, principal, tele_poses, tele_rms = calibrate_telecentric(
                [np.reshape(o, (-1, 3)) for o in object_points],
                [np.reshape(i, (-1, 2)) for i in image_points],
                res,
            )
            logger.info(
                f'{cam_name} seeded as telecentric at '
                f'{magnification[0]:.1f}, {magnification[1]:.1f} px per unit'
                f', leftover error of {np.mean(tele_rms):.2f} pixels')
            init_cam = TelecentricCamera(
                intrinsic=np.array([[magnification[0], 0, principal[0]],
                                    [0, magnification[1], principal[1]],
                                    [0, 0, 1.0]]),
                res=res, distortion_coefs=np.array([0.0]),
                telecentricity=0.0, name=cam_name)
            init_cam, ext_was_fixed = self._apply_fixed_params_to(
                init_cam, fixed_params, fixed_param)
            if ext_was_fixed or not return_poses:
                return init_cam
            return init_cam, np.stack(tele_poses).astype(float), np.asarray(tele_rms, dtype=float)

        ic = cv2.calibrateCameraExtended(
            object_points,
            image_points,
            tuple(res[::-1]),
            None,
            None,
            # flags=(
            #     cv2.CALIB_RATIONAL_MODEL + \
            #     cv2.CALIB_THIN_PRISM_MODEL +\
            #     cv2.CALIB_TILTED_MODEL
            # ),
            None,
        )
        end = time.time()

        logger.info(f'{cam_name} took {end - start:.1f} seconds'
            f', leftover error of {ic[0]:.2f} pixels')

        # perform an initial pose estimate on the first images

        

        init_cam = Camera(intrinsic=ic[1], distortion_coefs=np.array(ic[2]), res=res, name=cam_name)
        if fixed_params is not None:
            if "int" in fixed_param:
                init_cam.intrinsic = fixed_param['int']
            if 'dst' in fixed_param:
                init_cam.distortion_coefs = fixed_param['dst']
            if "ext" in fixed_param:
                init_cam.set_extrinsic(fixed_param['ext'])
                return init_cam
        if not return_poses:
            return init_cam

        poses = np.stack([make_4x4h_tform(rot, tran) for rot, tran in zip(ic[3], ic[4])]).astype(float)
        per_im_reproj = np.asarray(ic[-1], dtype=float)
        return init_cam, poses, per_im_reproj

    def target_pose_in_cam_image(
            self, detection: TargetDetection, cam: Camera, 
            refine:bool = False, mode="throw", give_error=False) -> np.ndarray | tuple[np.ndarray, float]:
        """
        This function gives a pose estimate of the cube in an image as seen by a camera.

        :param detection: a detection containing data from a single image.
        :param cam: a camera model to use
        :param refine: Whether to use LM refinement of the estimate.
        :param mode: whether to throw an error or return nan arrays.

        :return: a 4x4 transformation of the target giving the transformation from target to camera coordinates
        """
        
        if not detection.has_data():
            if mode == "nan":
                if give_error:
                    return np.ones((4,4)) * np.nan, np.nan
                return np.ones((4,4)) * np.nan
            raise ValueError(f"The detection had no data  at all, including for camera {cam.name}")


        datum = detection.get(cam=cam.name).get_data()
        if datum is None:
            if mode == "nan":
                if give_error:
                    return np.ones((4,4)) * np.nan, np.nan
                return np.ones((4,4)) * np.nan
            raise ValueError(f"The detection had no data for camera {cam.name}")

        n_im = np.unique(datum[:, 0])  # check that only one camera and one image is here.
        if len(n_im) > 1:
            if mode == "nan":
                if give_error:
                    return np.ones((4,4)) * np.nan, np.nan
                return np.ones((4,4)) * np.nan
            raise ValueError(f"passed detection contained info from {n_im} ims. \n"
                "Pose estimation only works with 1 image")

        keys = get_keys(datum)
        object_points = self.point_data[tuple(keys.astype(int).T)]
        image_points = datum[:, -2:]
        if len(object_points) < 8:
            if mode == "nan":
                if give_error:
                    return np.ones((4,4)) * np.nan, np.nan
                return np.ones((4,4)) * np.nan
            raise ValueError("Inadequate number of corners for pose estimation")

        if len(object_points) < 12:
            logger.warning("Low number of points used for pose estimation")

        if isinstance(cam, TelecentricCamera):
            # solvePnP assumes a perspective camera, and filters its solutions
            # on the sign of a depth a telecentric camera does not have.  The
            # affine fit that replaces it has no alternative solutions to pick
            # between, so it short circuits the rest of this.
            try:
                ext, rms = pose_from_affine(
                    object_points, cam.undistort_points(image_points),
                    cam.magnification, cam.principal_point)
            except (ValueError, np.linalg.LinAlgError):
                if mode == "nan":
                    if give_error:
                        return np.ones((4, 4)) * np.nan, np.nan
                    return np.ones((4, 4)) * np.nan
                raise
            if rms > 20:
                logger.warning(
                    f"Past 20 pixel error for failed detection - counting detection "
                    f"as a failure (camera={cam.name}, pose={n_im[0]}) ")
                if mode == "nan":
                    if give_error:
                        return np.ones((4, 4)) * np.nan, np.nan
                    return np.ones((4, 4)) * np.nan
                raise ValueError("Failed a detection")
            if give_error:
                return ext, rms
            return ext

        try:
            _, rvec, tvec, err_list = cv2.solvePnPGeneric(object_points.astype("float32"),
                                                          image_points.astype("float32"),
                                                          cam.intrinsic,
                                                          cam.distortion_coefs
                                                          )

            passing_t = []
            passing_r = []
            for e, t in enumerate(tvec):
                if t[-1] < 0:
                    continue
                passing_t.append(tvec[e])
                passing_r.append(rvec[e])
            tvec, rvec = passing_t, passing_r
            if not passing_t:
                if mode == "nan":
                    if give_error:
                        return np.ones((4,4)) * np.nan, np.nan
                    return np.ones((4,4)) * np.nan
                raise ValueError("Opencv failed to find a non-negative pose")


        except cv2.error as e:
            if mode == "nan":
                if give_error:
                    return np.ones((4,4)) * np.nan, np.nan
                return np.ones((4,4)) * np.nan
            raise ValueError("Opencv failed to find a pose")

        
        max_err = np.argmax(err_list)
        min_err = np.argmin(err_list)
        if (err := err_list[max_err].squeeze()) > 5:
            logger.warning(f"Initial error of {err: .2f} found for a pose detection (camera={cam.name}, pose={n_im[0]}).")

        if (err := err_list[max_err].squeeze()) > 20:
            logger.warning(f"Past 20 pixel error for failed detection - counting detection as a failure (camera={cam.name}, pose={n_im[0]}) ")
            if mode == "nan":
                if give_error:
                    return np.ones((4,4)) * np.nan, np.nan
                return np.ones((4,4)) * np.nan
            raise ValueError("Failed a detection")

        ext = make_4x4h_tform(
            rvec[min_err],
            tvec[min_err],
        )
        
        if not refine:
            if give_error:
                return ext, np.mean(err_list)
            return ext # from target -> cam coordinates
        else: 
            raise NotImplementedError
