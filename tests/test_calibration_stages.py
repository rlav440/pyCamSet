"""The calibration pipeline, one stage at a time.

``test_calibration_charuco.py`` and ``test_calibration_ccube.py`` run
``calibrate_cameras`` end to end and assert a reprojection baseline.  That is
the right acceptance test, but it is the only thing exercising detection
gathering, input validation, initial calibration and outlier rejection, so a
fault in any of them arrives as a single drifted number.

These call the same stages directly.  The ones that need real detections share
the session-scoped fixtures rather than re-detecting, so the whole file costs
little more than the detection it already pays for elsewhere.
"""

from __future__ import annotations

import logging
import shutil

import numpy as np
import pytest

from pyCamSet import CameraSet
from pyCamSet.calibration.camera_calibrator import (
    detect_datapoints_in_imfile,
    outlier_rejection,
    run_initial_calibration,
    sanitise_input_images,
    validate_detections,
)
from pyCamSet.calibration_targets import ImageDetection, TargetDetection

# --------------------------------------------------------------------------
# sanitise_input_images -- pure, no images needed beyond empty files
# --------------------------------------------------------------------------


def _make_camera_folders(root, counts, suffix=".png"):
    """Folders of empty image files, one folder per camera."""
    folders = []
    for index, count in enumerate(counts):
        folder = root / f"cam{index}"
        folder.mkdir()
        for image in range(count):
            (folder / f"{image:04}{suffix}").write_bytes(b"")
        folders.append(folder)
    return folders


def test_equal_image_counts_are_accepted(tmp_path):
    sanitise_input_images(_make_camera_folders(tmp_path, [3, 3, 3]))


def test_unequal_image_counts_are_rejected(tmp_path):
    """Unequal counts mean the images are not synchronised across cameras.

    Every downstream stage indexes poses by image number, so a camera with a
    different count silently pairs the wrong frames together.
    """
    with pytest.raises(ValueError, match="unequal number of calibration images"):
        sanitise_input_images(_make_camera_folders(tmp_path, [3, 2, 3]))


def test_a_single_camera_is_accepted(tmp_path):
    sanitise_input_images(_make_camera_folders(tmp_path, [5]))


def test_non_image_files_do_not_count(tmp_path):
    """Only the recognised image suffixes are counted, so a stray file is fine."""
    folders = _make_camera_folders(tmp_path, [3, 3])
    (folders[0] / "notes.txt").write_text("not an image")
    (folders[0] / "cache.pickle").write_bytes(b"")

    sanitise_input_images(folders)


# --------------------------------------------------------------------------
# detect_datapoints_in_imfile
# --------------------------------------------------------------------------


def test_detection_needs_camera_subfolders(tmp_path, charuco_target):
    """Images must be nested one folder per camera; a flat folder is a mistake."""
    (tmp_path / "0000.png").write_bytes(b"")

    with pytest.raises(ValueError, match="no subfolders were found"):
        detect_datapoints_in_imfile(
            f_loc=tmp_path, calibration_target=charuco_target, caching=False
        )


@pytest.mark.data
def test_detection_caches_and_reloads(tmp_path, session_data_dir, charuco_target):
    """The pickle cache must reload as the detection that produced it.

    A small copy of the corpus, so this writes its cache into tmp_path rather
    than into tests/test_data.
    """
    source = session_data_dir / "calibration_charuco"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for cam_folder in sorted(p for p in source.iterdir() if p.is_dir()):
        target_folder = corpus / cam_folder.name
        target_folder.mkdir()
        for image in sorted(cam_folder.glob("*.jpg"))[:3]:
            shutil.copy(image, target_folder / image.name)

    first, res_first = detect_datapoints_in_imfile(
        f_loc=corpus, calibration_target=charuco_target, caching=True, threads=1
    )
    assert (corpus / "detected_datapoints.pickle").is_file()

    second, res_second = detect_datapoints_in_imfile(
        f_loc=corpus, calibration_target=charuco_target, caching=True, threads=1
    )

    assert res_first == res_second
    assert np.array_equal(first.get_data(), second.get_data())


@pytest.mark.data
def test_n_lim_bounds_the_images_used(tmp_path, session_data_dir, charuco_target):
    """n_lim exists to make a quick pass over a big corpus, so it must bite."""
    source = session_data_dir / "calibration_charuco"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for cam_folder in sorted(p for p in source.iterdir() if p.is_dir()):
        target_folder = corpus / cam_folder.name
        target_folder.mkdir()
        for image in sorted(cam_folder.glob("*.jpg"))[:4]:
            shutil.copy(image, target_folder / image.name)

    limited, _ = detect_datapoints_in_imfile(
        f_loc=corpus, calibration_target=charuco_target, caching=False, n_lim=2, threads=1
    )

    assert limited.max_ims <= 2


@pytest.mark.data
def test_detection_reports_the_camera_resolutions(charuco_detections):
    """The resolutions come from the images, and feed the initial calibration."""
    detections, camera_res = charuco_detections

    assert len(camera_res) == len(detections.cam_names)
    for height, width in camera_res:
        assert height > 0 and width > 0


# --------------------------------------------------------------------------
# validate_detections
# --------------------------------------------------------------------------


def _detection_with_completeness(fraction, n_images=10, corners=16):
    """A synthetic detection where each camera sees *fraction* of the board."""
    seen = max(1, int(corners * fraction))
    detection = TargetDetection(cam_names=["a", "b"])
    for name in ("a", "b"):
        for im_num in range(n_images):
            detection.add_detection(
                name,
                im_num,
                ImageDetection(
                    keys=np.arange(seen),
                    image_points=np.tile([100.0, 100.0], (seen, 1)),
                ),
            )
    return detection


class _FlatTarget:
    """A target with a known number of corners per face."""

    def __init__(self, corners=16):
        self.point_data = np.zeros((1, corners, 3))


@pytest.mark.parametrize("fraction", [1.0, 0.75])
def test_good_detections_pass_quietly(fraction, caplog):
    detection = _detection_with_completeness(fraction)

    with caplog.at_level(logging.WARNING):
        validate_detections(detection, _FlatTarget())

    assert "struggled" not in caplog.text


def test_incomplete_boards_are_reported(caplog):
    """Under half a board per image is worth warning about."""
    detection = _detection_with_completeness(0.25)

    with caplog.at_level(logging.WARNING):
        validate_detections(detection, _FlatTarget())

    assert "struggled to detect full complete boards" in caplog.text


def test_failed_detections_are_reported(caplog):
    """A camera that only saw the target in a few images is flagged."""
    detection = TargetDetection(cam_names=["a"])
    detection.max_ims = 20
    for im_num in range(3):  # 3 of 20 images
        detection.add_detection(
            "a",
            im_num,
            ImageDetection(keys=np.arange(16), image_points=np.tile([1.0, 2.0], (16, 1))),
        )

    with caplog.at_level(logging.WARNING):
        validate_detections(detection, _FlatTarget())

    assert "high number of failed detections" in caplog.text


@pytest.mark.data
def test_the_real_corpus_reports_per_camera_metrics(charuco_detections, charuco_target, caplog):
    """Validation must produce a line per camera on the real corpus.

    This corpus does trip the completeness warning -- the boards are seen at
    an angle and no camera resolves a full 19x19 in every frame -- and camera
    "1" trips the failed-detection one too.  That is the expected state of
    these images, not a fault: the calibration still converges inside its
    reprojection baseline.  Pinned so the warnings appearing or vanishing is a
    visible change rather than a silent one.
    """
    detections, _ = charuco_detections

    with caplog.at_level(logging.INFO):
        validate_detections(detections, charuco_target)

    for name in detections.cam_names:
        assert f'Camera "{name}" detected boards' in caplog.text
    assert "struggled to detect full complete boards" in caplog.text


# --------------------------------------------------------------------------
# run_initial_calibration
# --------------------------------------------------------------------------


@pytest.mark.data
def test_initial_calibration_produces_a_camera_per_folder(charuco_problem):
    _, detections, cams = charuco_problem

    assert isinstance(cams, CameraSet)
    assert cams.get_names() == detections.cam_names


@pytest.mark.data
def test_initial_cameras_have_plausible_intrinsics(charuco_problem):
    """A sanity floor on the initial estimate, before any bundle adjustment.

    The principal point should sit inside the sensor and the focal length
    should be the same order as the image width; anything else means the
    per-camera OpenCV calibration failed rather than merely being imprecise.
    """
    _, _, cams = charuco_problem

    for cam in cams:
        height, width = cam.res
        assert 0 < cam.intrinsic[0, 2] < width * 2
        assert 0 < cam.intrinsic[1, 2] < height * 2
        assert 0 < cam.intrinsic[0, 0] < width * 10
        assert np.all(np.isfinite(cam.extrinsic))
        assert np.all(np.isfinite(cam.distortion_coefs))


@pytest.mark.data
def test_initial_calibration_can_return_poses_and_costs(
    session_data_dir, charuco_target, charuco_detections
):
    detections, camera_res = charuco_detections

    cams, poses, per_im = run_initial_calibration(
        detections, charuco_target, camera_res, save=False, return_poses_and_costs=True
    )

    assert isinstance(cams, CameraSet)
    assert len(poses) == len(cams)
    assert len(per_im) == len(cams)


@pytest.mark.data
def test_initial_calibration_saves_and_reloads(
    tmp_path, charuco_target, charuco_detections
):
    """The save path short-circuits a rerun, so it must reload the same cameras."""
    detections, camera_res = charuco_detections
    save_loc = tmp_path / "initial_cameras.camset"

    first = run_initial_calibration(
        detections, charuco_target, camera_res, save=True, save_loc=save_loc
    )
    assert save_loc.is_file()

    second = run_initial_calibration(
        detections, charuco_target, camera_res, save=True, save_loc=save_loc
    )

    assert second == first


# --------------------------------------------------------------------------
# outlier_rejection
# --------------------------------------------------------------------------


class _StubHandler:
    """Just the two attributes outlier_rejection reaches for."""

    def __init__(self, detection):
        self.detection = detection

    def get_detection_data(self):
        return self.detection.get_data()


def _even_error_problem(n_images=8, per_image=5):
    detection = TargetDetection(cam_names=["a"])
    for im_num in range(n_images):
        detection.add_detection(
            "a",
            im_num,
            ImageDetection(
                keys=np.arange(per_image),
                image_points=np.tile([10.0, 20.0], (per_image, 1)),
            ),
        )
    return detection, n_images * per_image


def test_outlier_rejection_finds_nothing_in_even_errors():
    detection, n_rows = _even_error_problem()
    residuals = np.ones(n_rows)

    data, found = outlier_rejection(residuals, _StubHandler(detection), draw=False)

    assert data is None
    assert found is False


def test_outlier_rejection_removes_a_bad_image():
    """One image with a wild error must be dropped, and only that one."""
    detection, n_rows = _even_error_problem(n_images=8, per_image=5)
    residuals = np.ones(n_rows)
    residuals[10:15] = 500.0  # image 2

    data, found = outlier_rejection(residuals, _StubHandler(detection), draw=False)

    assert found is True
    assert 2 not in np.unique(data.get_data()[:, 1])
    assert len(data.get_data()) == n_rows - 5


def test_outlier_rejection_does_not_draw_when_asked_not_to(monkeypatch):
    """Regression: both branches called plt.show() unconditionally.

    That made the function unusable from an unattended run, and is why it had
    no coverage at all.
    """
    import matplotlib.pyplot as plt

    def explode(*args, **kwargs):
        raise AssertionError("plt.show() was called with draw=False")

    monkeypatch.setattr(plt, "show", explode)

    detection, n_rows = _even_error_problem()
    outlier_rejection(np.ones(n_rows), _StubHandler(detection), draw=False)
