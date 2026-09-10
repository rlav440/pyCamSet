"""Detection itself, pinned against the checked-in corpus.

Everything downstream -- initial calibration, pose graph, bundle adjustment --
consumes a TargetDetection, and the staged tests share one per session rather
than re-detecting.  That makes detection a single point of failure that no
other test looks at directly: if OpenCV changes what it returns, the
calibration tests notice only as a drifting reprojection error, which reads
like a solver regression rather than a detection one.

opencv-python is deliberately unpinned within ``>=4.8,<5``, and the CI matrix
resolves it differently per platform and per Python version, so this is the
test that says which layer moved.

The structural assertions are exact: shapes, key ranges, column counts and
camera names cannot change without something being broken.  The counts and
pixel statistics carry a tolerance, because corner refinement legitimately
differs a little between OpenCV patch releases and platforms -- wide enough
not to cry wolf, narrow enough that a real API change blows through it.
"""

from __future__ import annotations

import numpy as np
import pytest

# Reference values measured on the checked-in corpus with opencv-python 4.13.0,
# macOS/arm64, Python 3.12.  Regenerate deliberately, never to make a red test
# green: a change here means the detected geometry moved.
CHARUCO_REFERENCE = {
    "cam_names": ["1", "2", "3"],
    "resolutions": [(1024, 1280), (1024, 1280), (1024, 1280)],
    "n_detections": 4674,
    "per_cam": [1257, 1675, 1742],
    "max_ims": 22,
    "n_key_columns": 1,
    "n_target_points": 361,
    "uv_mean": (640.12, 521.65),
}
CCUBE_REFERENCE = {
    "cam_names": ["cam0", "cam1", "cam2"],
    "resolutions": [(1080, 1920), (1080, 1920), (1080, 1920)],
    "n_detections": 4887,
    "per_cam": [1579, 1832, 1476],
    "max_ims": 24,
    "n_key_columns": 2,
    "n_target_points": 6 * 81,
    "uv_mean": (1271.05, 542.86),
}

# A detection count this far from the reference is a change worth looking at.
COUNT_TOLERANCE = 0.10
# The mean detected pixel should not wander by more than a corner or two.
UV_MEAN_TOLERANCE_PX = 5.0


def _check_structure(detections, camera_res, reference, target):
    """The assertions that must hold exactly, whatever OpenCV is installed."""
    data = detections.get_data()

    assert detections.cam_names == reference["cam_names"]
    assert [tuple(r) for r in camera_res] == reference["resolutions"]
    assert detections.max_ims == reference["max_ims"]

    # | cam | im_num | key ... | x | y |
    assert data.shape[1] == 4 + reference["n_key_columns"]

    # every camera index addresses a real camera, every image a real image
    assert set(np.unique(data[:, 0])) <= set(range(len(reference["cam_names"])))
    assert data[:, 1].min() >= 0
    assert data[:, 1].max() < reference["max_ims"]

    # every key addresses a real target point
    keys = data[:, 2:-2].astype(int)
    assert keys.min() >= 0
    flat_points = target.point_data.reshape(-1, 3)
    assert len(flat_points) == reference["n_target_points"]

    # The keys index point_data's leading dimensions, aligned to the right:
    # get_keys pads a single column key with a leading zero, so a ChArUco key
    # addresses point_data (1, n, 3) on its second axis, while a Ccube's two
    # column key addresses (6, 81, 3) on both.
    extents = target.point_data.shape[:-1][-keys.shape[1]:]
    for axis, extent in enumerate(extents):
        assert keys[:, axis].max() < extent, (
            f"key column {axis} reaches {keys[:, axis].max()}, past the {extent} "
            f"entries of point_data{target.point_data.shape}"
        )

    # every detected point landed on a sensor
    for cam_index, (height, width) in enumerate(reference["resolutions"]):
        rows = data[data[:, 0] == cam_index]
        assert rows[:, -2].min() >= 0 and rows[:, -2].max() < width
        assert rows[:, -1].min() >= 0 and rows[:, -1].max() < height

    assert np.all(np.isfinite(data))


def _check_counts(detections, reference):
    """The assertions that carry a tolerance for OpenCV version drift."""
    data = detections.get_data()
    expected = reference["n_detections"]
    low, high = expected * (1 - COUNT_TOLERANCE), expected * (1 + COUNT_TOLERANCE)

    assert low <= len(data) <= high, (
        f"detected {len(data)} points, expected about {expected}. A change this "
        f"large usually means the detector's behaviour moved: check the "
        f"installed opencv-python version before adjusting this reference."
    )

    for name, count, want in zip(
        reference["cam_names"],
        [len(d.get_data()) for d in detections.get_cam_list()],
        reference["per_cam"],
    ):
        assert want * (1 - COUNT_TOLERANCE) <= count <= want * (1 + COUNT_TOLERANCE), (
            f"camera {name} detected {count} points, expected about {want}"
        )

    uv_mean = data[:, -2:].mean(axis=0)
    assert np.allclose(uv_mean, reference["uv_mean"], atol=UV_MEAN_TOLERANCE_PX), (
        f"the mean detected pixel moved to {np.round(uv_mean, 2)} from "
        f"{reference['uv_mean']}: the detected geometry has shifted."
    )


# --------------------------------------------------------------------------
# ChArUco
# --------------------------------------------------------------------------


@pytest.mark.data
def test_charuco_detection_structure(charuco_detections, charuco_target):
    detections, camera_res = charuco_detections
    _check_structure(detections, camera_res, CHARUCO_REFERENCE, charuco_target)


@pytest.mark.data
def test_charuco_detection_counts(charuco_detections):
    detections, _ = charuco_detections
    _check_counts(detections, CHARUCO_REFERENCE)


@pytest.mark.data
def test_charuco_board_geometry_is_what_the_images_show():
    """The board definition itself, independent of any image.

    CharucoBoard.getChessboardCorners and setLegacyPattern are the two OpenCV
    entry points the ChArUco target depends on; a change in either silently
    re-indexes every detection.
    """
    from pyCamSet import ChArUco

    from conftest import CHARUCO_ARGS

    target = ChArUco(**CHARUCO_ARGS)
    corners = target.point_data.reshape(-1, 3)

    # a 20x20 square board has 19x19 interior chessboard corners
    assert len(corners) == 19 * 19
    assert np.allclose(corners[:, 2], 0), "the board must be planar in z"

    # 4mm squares, given to the target in mm and held in metres
    spacing = np.diff(np.unique(np.round(corners[:, 0], 6)))
    assert np.allclose(spacing, 0.004, atol=1e-9)


@pytest.mark.data
def test_the_legacy_flag_changes_the_board():
    """legacy=True is load bearing for this corpus, so it must still do something."""
    from pyCamSet import ChArUco

    from conftest import CHARUCO_ARGS

    legacy = ChArUco(**CHARUCO_ARGS)
    modern_args = dict(CHARUCO_ARGS, legacy=False)
    modern = ChArUco(**modern_args)

    assert legacy.board.getLegacyPattern() is True
    assert modern.board.getLegacyPattern() is False


# --------------------------------------------------------------------------
# Ccube
# --------------------------------------------------------------------------


@pytest.mark.data
@pytest.mark.slow
def test_ccube_detection_structure(ccube_detections, ccube_target):
    detections, camera_res = ccube_detections
    _check_structure(detections, camera_res, CCUBE_REFERENCE, ccube_target)


@pytest.mark.data
@pytest.mark.slow
def test_ccube_detection_counts(ccube_detections):
    detections, _ = ccube_detections
    _check_counts(detections, CCUBE_REFERENCE)


@pytest.mark.data
@pytest.mark.slow
def test_ccube_keys_carry_a_face_and_a_point(ccube_detections, ccube_target):
    """A cube key is (face, point), which is what makes it 2-d.

    A flattening change here would put every detection on face zero, which the
    bundle adjustment would absorb as a large but plausible error rather than
    an obvious failure.
    """
    detections, _ = ccube_detections
    keys = detections.get_data()[:, 2:-2].astype(int)

    assert keys.shape[1] == 2
    assert set(np.unique(keys[:, 0])) <= set(range(6))
    assert keys[:, 1].max() < 81
    # more than one face must actually be seen, or the split is not happening
    assert len(np.unique(keys[:, 0])) > 1


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


@pytest.mark.data
def test_detection_is_deterministic(session_data_dir, charuco_target):
    """Detecting the same folder twice must give the same answer.

    A shared session fixture is only sound if detection is a pure function of
    the images; anything order or state dependent would make one test's
    detections differ from another's.
    """
    from conftest import _detect

    first, res_a = _detect(session_data_dir / "calibration_charuco", charuco_target)
    second, res_b = _detect(session_data_dir / "calibration_charuco", charuco_target)

    assert res_a == res_b
    assert np.array_equal(first.get_data(), second.get_data())


@pytest.mark.data
def test_threaded_detection_matches_serial(session_data_dir, charuco_target):
    """The multiprocessing path must not reorder or drop detections.

    find_in_imfolder forks a pool when threads > 1 and reassembles the results
    by camera and index; a mismatch there would scramble which image a
    detection belongs to.
    """
    from pyCamSet.calibration.camera_calibrator import detect_datapoints_in_imfile

    loc = session_data_dir / "calibration_charuco"
    serial, _ = detect_datapoints_in_imfile(
        f_loc=loc, caching=False, calibration_target=charuco_target, threads=1
    )
    threaded, _ = detect_datapoints_in_imfile(
        f_loc=loc, caching=False, calibration_target=charuco_target, threads=2
    )

    # sort both, since the pool may return cameras in any order
    order = ["cam", "im_num", "key"]
    assert np.allclose(
        serial.sort(order).get_data(), threaded.sort(order).get_data()
    )


@pytest.mark.data
def test_the_shared_fixture_is_the_detection_the_corpus_gives(
    charuco_detections, session_data_dir, charuco_target
):
    """The session fixture must not drift from a fresh detection.

    This is the guard on sharing: every staged test below trusts the fixture,
    so it has to be the same thing a test doing its own detection would get.
    """
    from conftest import _detect

    shared, shared_res = charuco_detections
    fresh, fresh_res = _detect(session_data_dir / "calibration_charuco", charuco_target)

    assert shared_res == fresh_res
    assert np.array_equal(shared.get_data(), fresh.get_data())
