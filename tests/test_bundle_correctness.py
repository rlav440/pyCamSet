"""Cross-checks pyCamSet's compiled projection against OpenCV's.

The bundle adjuster evaluates its residuals through ``bundle_adj_parrallel_solver``
and ``nb_distort_prealloc``.  This test calibrates a camera with OpenCV, then
projects the same points twice -- once through ``cv2.projectPoints`` and once
through pyCamSet's own compiled path -- and asserts the two agree.  Any drift
between the analytic projection/distortion model and OpenCV's shows up here.
"""

import cv2
import numpy as np
import pytest

from pyCamSet.optimisation.compiled_helpers import (
    bundle_adj_parrallel_solver,
    nb_distort_prealloc,
)
from pyCamSet.utils.general_utils import h_tform, make_4x4h_tform

# Board geometry of the checked-in calibration images.
SQUARES = (20, 20)
SQUARE_SIZE = 0.004
MARKER_SIZE = 0.0032
MIN_CORNERS_PER_VIEW = 4


def _detect_board(image_dir):
    """Detect ChArUco corners in every image of *image_dir*.

    :param image_dir: folder holding the calibration JPEGs for one camera.
    :return: (list of (N,1,2) corner arrays, list of (N,1) id arrays, (w, h)).
    """
    a_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    board = cv2.aruco.CharucoBoard(SQUARES, SQUARE_SIZE, MARKER_SIZE, a_dict)
    detector = cv2.aruco.CharucoDetector(board)

    all_corners, all_ids = [], []
    im_size = None
    for frame in sorted(image_dir.glob("*.jpg")):
        im = cv2.imread(str(frame))
        assert im is not None, f"OpenCV could not read {frame}"
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_size = (gray.shape[1], gray.shape[0])  # OpenCV wants (width, height)

        corners, ids, _, _ = detector.detectBoard(gray)
        if corners is not None and len(corners) > MIN_CORNERS_PER_VIEW:
            all_corners.append(corners)
            all_ids.append(ids)

    return all_corners, all_ids, im_size, board


@pytest.mark.data
def test_bundle_projection_matches_opencv(data_dir):
    """pyCamSet's compiled projection must match cv2.projectPoints to 1e-4 px."""
    all_corners, all_ids, im_size, board = _detect_board(
        data_dir / "calibration_charuco" / "1"
    )
    assert len(all_corners) > 10, (
        f"Only {len(all_corners)} usable views detected; the test data or the "
        "OpenCV detection defaults have changed."
    )

    chessboard = board.getChessboardCorners().squeeze()

    # cv2.aruco.calibrateCameraCharucoExtended was removed in OpenCV 4.7, so
    # feed the interpolated corners to the generic calibrator instead.
    object_points = [
        chessboard[ids.squeeze(-1)].astype(np.float32) for ids in all_ids
    ]
    image_points = [corners.astype(np.float32) for corners in all_corners]

    camera_matrix_init = np.array(
        [
            [1000.0, 0.0, im_size[0] / 2.0],
            [0.0, 1000.0, im_size[1] / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO

    (
        _ret,
        camera_matrix,
        distortion_coefficients,
        rotation_vectors,
        translation_vectors,
        *_,
    ) = cv2.calibrateCameraExtended(
        objectPoints=object_points,
        imagePoints=image_points,
        imageSize=im_size,
        cameraMatrix=camera_matrix_init,
        distCoeffs=np.zeros((5, 1)),
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1e-9),
    )

    # Reshape the detections into the (thread, datum, 6) layout the solver wants.
    dct = [
        [0, idim, 0, idx, corner[0], corner[1]]
        for idim, (corners, ids) in enumerate(zip(all_corners, all_ids))
        for corner, idx in zip(corners.squeeze(1), ids.squeeze(1))
    ]
    dct = np.array(dct)
    dct = dct[: (len(dct) // 2) * 2].reshape((2, -1, 6))

    im_points = [
        h_tform(chessboard, make_4x4h_tform(r, t, mode="opencv"))
        for r, t in zip(rotation_vectors, translation_vectors)
    ]
    im_points = np.array(im_points).reshape(len(all_corners), -1, 1, 3)
    proj_mat = np.concatenate((camera_matrix, np.zeros((3, 1))), axis=1).reshape((1, 3, 4))

    # Exercising the solver guards against shape regressions in the residual path.
    bundle_adj_parrallel_solver(
        dct,
        im_points,
        proj_mat,
        camera_matrix.reshape((1, 3, 3)),
        np.reshape(distortion_coefficients, (1, -1)),
    )

    errors = []
    for idim, (corners, ids) in enumerate(zip(all_corners, all_ids)):
        for _corner, idx in zip(corners.squeeze(1), ids.squeeze(1)):
            opencv_projection = cv2.projectPoints(
                chessboard[idx],
                rotation_vectors[idim],
                translation_vectors[idim],
                camera_matrix,
                distortion_coefficients,
            )[0].squeeze()

            point = np.ones(4)
            point[:3] = im_points[idim, idx, 0, :]
            proj_p = proj_mat[0] @ point
            pycamset_projection = (proj_p[:-1] / proj_p[-1]).copy()
            nb_distort_prealloc(
                pycamset_projection,
                camera_matrix,
                distortion_coefficients.squeeze(),
            )

            errors.append(np.linalg.norm(opencv_projection - pycamset_projection))

    mean_err = np.mean(errors)
    assert mean_err < 1e-4, (
        f"pyCamSet's projection drifted from OpenCV's by {mean_err:.3e} px "
        "(tolerance 1e-4). The distortion or projection model has changed."
    )
