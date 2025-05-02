import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle

from pyCamSet.optimisation.compiled_helpers import bundle_adj_parrallel_solver, nb_distort_prealloc
from pyCamSet.utils.general_utils import h_tform, make_4x4h_tform



def test_bundle_correctness():
    # for an input file, use opencv to detect a charuco board and calibrate a camera
    data_loc = Path('tests/test_data/calibration_charuco/1')
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    board = cv2.aruco.CharucoBoard((20, 20), 0.004, 0.0032, adict)
    all_corners = []
    all_ids = []
    cache_loc = Path('tests/test_data/cam_test_cache.pkl')
    if cache_loc.exists():
        with open(cache_loc, 'rb') as f:
            all_corners, all_ids, imsize = pickle.load(f)
    else:
        images = list(data_loc.glob("*.jpg"))
        for frame in tqdm(images):
            im = cv2.imread(str(frame))
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # im_ida = im.copy()
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, adict)
            if len(corners) > 0:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                if ret > 4:
                    all_corners.append(charuco_corners)
                    all_ids.append( charuco_ids)

        imsize = gray.shape
        with open(cache_loc, 'wb') as f:
            pickle.dump((all_corners, all_ids, imsize), f)
    # now we have a list of detections, we can calibrate the camera
    camera_matrix_init = np.array([[ 1000.,    0., imsize[0]/2.],
                                     [    0., 1000., imsize[1]/2.],
                                     [    0.,    0.,           1.]])

    dist_coeffs_init = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=all_corners,
                      charucoIds=all_ids,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=camera_matrix_init,
                      distCoeffs=dist_coeffs_init,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
    # so now we turn this into an input for the cost function.

    dct = []
    for idim, (corners, ids) in enumerate(zip(all_corners, all_ids)):
        for corner, id in zip(corners.squeeze(), ids.squeeze()):
            blank_dct = [0, idim, 0, id, corner[0], corner[1]]
            dct.append(blank_dct)

    dct = np.array(dct)[:-1]
    dct = dct.reshape((2, -1, 6))
    points = board.getChessboardCorners().squeeze()
    im_points = [h_tform(points, make_4x4h_tform(r, t, mode='opencv')) for r, t in zip(rotation_vectors, translation_vectors)]
    im_points = np.array(im_points).reshape(len(all_corners), -1, 1, 3)
    proj_mat = np.concatenate((camera_matrix, np.zeros((3,1))), axis=1).reshape((1,3,4))
    intrinsics = camera_matrix.reshape((1,3,3))
    dists = np.reshape(distortion_coefficients0, (1, -1))
    cost = bundle_adj_parrallel_solver(dct, im_points, proj_mat, intrinsics, dists)
    euclid_cost = np.sqrt(np.sum(cost.T**2, axis=1))
    perIm = []
    scratch_cost = euclid_cost.copy()
    for id in all_ids:
        perIm.append(np.mean(scratch_cost[:len(id)]))
        scratch_cost = scratch_cost[len(id):]

    errors = []
    for idim, (corners, ids) in enumerate(zip((all_corners), all_ids)):
        for corner, id in zip(corners.squeeze(), ids.squeeze()):
            pro_loc_0 = cv2.projectPoints(
                board.getChessboardCorners()[id], rotation_vectors[idim],
                translation_vectors[idim], camera_matrix, distortion_coefficients0
            )[0].squeeze()

            #now we perform a similar check to the cost function.
            point = np.ones(4)
            point[:3] = im_points[idim, id, 0, :]
            proj_p = proj_mat[0]@point
            proj_loc_1 = (proj_p[:-1]/proj_p[-1]).copy()
            nb_distort_prealloc(proj_loc_1, camera_matrix, distortion_coefficients0.squeeze())

            errors.append(np.linalg.norm(pro_loc_0 - proj_loc_1))
    
    mean_err  =np.mean(errors)
    # print(f"found a mean difference of {mean_err:.5f}")

    assert mean_err < 1e-4

if __name__ == "__main__":
    test_bundle_correctness()
