import math
from os import walk
import numba
from numba import njit
import numpy as np
import cv2


@njit(cache=True)
def fill_pose(pose_data, poses, poses_unfixed):
    """
    Fills a pose array with data from an input param array.
    
    :param pose_data: The pose parameters, an nx6 array
    :param poses: The pose data with missing blocks to fill
    :param poses_unfixed: A boolean array describing which poses are unfixed
    :return:
    """
    k = 0
    n_poses = len(poses)
    for i in range(n_poses):
        if poses_unfixed[i]:
            n_e4x4_flat_INPLACE(pose_data[k], poses[i])
            k += 1
    return

@njit(cache=True)
def fill_extr(extr_data, extr, extr_unfixed):
    """
    Fills an extrinsics array with data from an input param array.

    :param extr_data: The extrinsic parameters, an nx6 array
    :param extr: The extrinsic data with missing blocks to fill
    :param extr_unfixed: A boolean array describing which extrinsics are unfixed
    :return:
    """
    k = 0
    n_extr = len(extr)
    for i in range(n_extr):
        if extr_unfixed[i]:
            n_e4x4(extr_data[k], extr[i])
            k += 1
    return


@njit(cache=True)
def fill_intr(intr_data, intr, intr_unfixed):
    """
    Fills an intrinsics array with data from an input param array.

    :param intr_data: The intrinsic parameters, an nx4 array
    :param intr: The intrinsic data with missing blocks to fill
    :param intr_unfixed: A boolean array describing which intrinsics are unfixed
    :return:
    """
    k = 0
    n_intr = len(intr)
    for i in range(n_intr):
        if intr_unfixed[i]:
            intr[i, 0, 0] = intr_data[k, 0]
            intr[i, 0, 2] = intr_data[k, 1]
            intr[i, 1, 1] = intr_data[k, 2]
            intr[i, 1, 2] = intr_data[k, 3]
            intr[i, 2, 2] = 1
            k += 1
    return


@njit(cache=True)
def fill_dst(dst_data, dst, dst_unfixed):
    """
    Fills a distortion array with data from an input param array.
    
    :param dst_data: The distortion parameters, an nx5 array
    :param dst: The distortion data with missing blocks to fill
    :param dst_unfixed: A boolean array describing which distortions are unfixed
    :return:
    """
    k = 0
    n_dst = len(dst)
    for i in range(n_dst):
        if dst_unfixed[i]:
            dst[i, :] = dst_data[k, :]
            k += 1
    return


@njit(cache=True)
def n_e4x4(rog_vec: np.ndarray, output: np.ndarray):
    """
    Converts a 6dof pose vector into a 4x4 homogenous transform

    :param rog_vec: A 3 axis_angle opencv representation of rotation, followed by a translation
    :param output: the 4x4 output to write the data too
    """
    angles = rog_vec[:3].reshape((3, 1))
    blank_rot = np.empty((3, 3))
    numba_flat_rodrigues_INPLACE(angles, blank_rot)
    output[:-1, :] = 0
    output[:-1, :-1] = blank_rot
    output[-1, -1] = 1
    output[:-1, -1] = rog_vec[3:]


@njit(fastmath=True, cache=True)
def numba_flat_rodrigues_INPLACE(r, blank_rot):
    """
    Converts a 3dof axis angle representation of rotation into a 3x3 rotation matrix

    :param r: The rotation vector
    :param blank_rot: the output location
    """
    theta = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2 + r[2, 0] ** 2)
    if theta == 0.0:
        blank_rot[:] = 0
        blank_rot[0, 0] = 1
        blank_rot[1, 1] = 1
        blank_rot[2, 2] = 1
        return
    scalar = 1 / theta
    s2 = scalar ** 2
    ct = math.cos(theta)
    st = math.sin(theta) * scalar

    np.dot(r, r.T, blank_rot)
    blank_rot *= (1 - ct) * s2
    blank_rot[0, 0] += ct
    blank_rot[1, 1] += ct
    blank_rot[2, 2] += ct
    blank_rot[0, 1] -= r[2, 0] * st
    blank_rot[1, 0] += r[2, 0] * st
    blank_rot[0, 2] += r[1, 0] * st
    blank_rot[2, 0] -= r[1, 0] * st
    blank_rot[1, 2] -= r[0, 0] * st
    blank_rot[2, 1] += r[0, 0] * st
    return


@njit(cache=True)
def n_e4x4_flat_INPLACE(rog_vec: np.ndarray, blank_tform: np.ndarray) -> None:
    """
    Converts a 6dof pose vector into a 12x1 homogenous transform representation
    
    :param rog_vec: A 3 axis_angle opencv representation of rotation, followed by a translation
    :param blank_tform: the output transform array
    """
    # returns a flattened memory contigous version
    numba_flat_rodrigues_INPLACE(
        rog_vec[:3].reshape((3, 1)),
        blank_tform[:9].reshape((3, 3))
    )
    blank_tform[9:] = rog_vec[3:]


def n_flat_tform_from_aa(pt, vec, ang) -> np.ndarray:
    """
    Converts a 6dof pose vector into a 12x1 homogenous transform representation

    :param pt: The point to rotate around
    :param vec: The axis to rotate around
    :param ang: the angle to rotate.
    :return:
    """
    vec /= np.linalg.norm(vec)  # needs to be normalised
    [u_x, u_y, u_z] = vec  # assume vec is norm
    c_a = np.cos(-ang)
    s_a = np.sin(-ang)
    u_crs = np.array([[0, -u_z, u_y],
                      [u_z, 0, -u_x],
                      [-u_y, u_x, 0]
                      ])
    uTT = vec[..., None] @ vec[None, ...]
    rot = c_a * np.eye(3) + s_a * u_crs + (1 - c_a) * uTT
    # check = rot@rot.T
    rotation, t_0, t_1 = np.eye(4), np.eye(4), np.eye(4)
    t_0[:3, -1] = -pt
    t_1[:3, -1] = pt
    rotation[:-1, :-1] = rot
    t_form = t_1 @ rotation @ t_0
    return np.concatenate((t_form[:-1, :-1].flatten(), t_form[:-1, -1]), axis=0)


@njit(cache=True)
def n_htform_broadcast_prealloc(points: np.ndarray, t_numba: np.ndarray, out, fill=True) -> None:
    """
    Fastest way to do a homogenous transform is not to do a homogenous transform.
    This function is designed to be used with a preallocated output array.

    :param points: The points to transform, a nx3 array
    :param t_numba: The transform to apply. a 12x1 array
    :param out: The output array, a nx3 array
    :param fill: Whether to add the translation component of the transform (if false only rotates)
    """
    if points.ndim > 2:
        lenp = 1
        for i in range(len(points.shape) - 1):
            lenp = lenp * points.shape[i]

        temp_pts = points.reshape((lenp, 3))
        temp_blnk = out.reshape((lenp, 3))
        n_htform_prealloc(
            temp_pts, t_numba, temp_blnk, fill
        )
    else:
        n_htform_prealloc(points, t_numba, out, fill)


@njit(cache=True)
def n_htform_prealloc(points, t_numba, out, fill=True):
    """
    Does a homogenous transform on a single point to a an ouput array

    :param points: the point to transform
    :param t_numba: the transform to apply, as a 12x1 array
    :param out: the output array, a 3x1 array view
    :param fill: Whether to add the translation component of the transform (if false only rotates)
    :return:
    """
    np.dot(points, t_numba[:9].reshape((3, 3)).T, out)
    if fill:
        out += t_numba[9:]


@njit(cache=True)
def nb_undistort_prealloc(pt: np.ndarray, intrinsics: np.ndarray, k: np.ndarray, blank):
    """
    this fuction takes points from as detected to the idealised linear pinhole model.

    :param pt: the single point to undistort.
    :param intrinsics: The intrinsics of the imaging camera
    :param k: The Brown Conway distortion model of the distorting camera
    :param blank: the output array, a 2 array.
    """
    centre_0, centre_1 = intrinsics[0, -1], intrinsics[1, -1]
    focal_0, focal_1 = intrinsics[0, 0], intrinsics[1, 1]
    x0, y0 = (pt[0] - centre_0) / focal_0, (pt[1] - centre_1) / focal_1

    x, y = x0, y0
    for _ in range(5):
        r2 = x ** 2 + y ** 2
        k_inv = 1 / (1 + k[0] * r2 + k[1] * (r2 ** 2) + k[4] * (r2 ** 3))
        xD = 2 * k[2] * x * y + k[3] * (r2 + 2 * (x ** 2))
        yD = k[2] * (r2 + 2 * (y ** 2)) + 2 * k[3] * x * y
        # back to absolute
        x = (x0 - xD) * k_inv
        y = (y0 - yD) * k_inv

    blank[0] = x * focal_0 + centre_0
    blank[1] = y * focal_1 + centre_1


@njit(cache=True)
def nb_undistort(pts: np.ndarray, intrinsics: np.ndarray, dist_coef: np.ndarray) -> np.ndarray:
    """
    This function takes points from as detected to the idealised linear pinhole model.
    It has some inherent allocations, so is slower than the preallocated version.

    :param pts: The input poitns to transform
    :param intrinsics: The intrinsics of the imaging camera
    :param dist_coef: The Brown Conway distortion model of the distorting camera
    :return: A distorted array of points.
    """
    # relative coordinates and distances.
    centre = intrinsics[:2, -1]
    focal = np.diag(intrinsics)[:2]
    x0, y0 = (pts - centre) / focal
    k = np.reshape((dist_coef), (-1))
    x, y = x0, y0
    for _ in range(5):
        r2 = x ** 2 + y ** 2
        k_inv = 1 / (1 + k[0] * r2 + k[1] * (r2 ** 2) + k[4] * (r2 ** 3))
        xD = 2 * k[2] * x * y + k[3] * (r2 + 2 * (x ** 2))
        yD = k[2] * (r2 + 2 * (y ** 2)) + 2 * k[3] * x * y
        # back to absolute
        x = (x0 - xD) * k_inv
        y = (y0 - yD) * k_inv
    locs = np.array([x, y]) * focal + centre
    return locs


@njit(cache=True)
def nb_distort_prealloc(pts: np.ndarray, intrinsics: np.ndarray, k: np.ndarray):
    """
    This function distorts points based on the input values, going from the mathematical ideal to detections.

    :params pts: points to distort, which are overwritten
    :params intrinsics. The intrinsics of the imaging camera
    :params k: Brown Conway model of the distorting camera
    """
    # relative coordinates and distances.
    centre_0, centre_1 = intrinsics[0, -1], intrinsics[1, -1]
    focal_0, focal_1 = intrinsics[0, 0], intrinsics[1, 1]
    x, y = (pts[0] - centre_0) / focal_0, (pts[1] - centre_1) / focal_1
    r2 = x ** 2 + y ** 2
    kup = (1 + k[0] * r2 + k[1] * (r2 ** 2) + k[4] * (r2 ** 3))
    # distort radially
    xD = x * kup
    yD = y * kup
    # distort tangentially
    xD += 2 * k[2] * x * y + k[3] * (r2 + 2 * (x ** 2))
    yD += k[2] * (r2 + 2 * (y ** 2)) + 2 * k[3] * x * y
    # back to absolute
    pts[0] = xD * focal_0 + centre_0
    pts[1] = yD * focal_1 + centre_1


@njit(cache=True)
def nb_distort(pts: np.ndarray, intrinsics: np.ndarray, dist_coef: np.ndarray) -> np.ndarray:
    """
    This function distorts points based on the input values, going from the mathematical ideal to detections.
    It has some inherent allocations, so is slower than the preallocated version.

    :params pts: points to distort, which are overwritten
    :params intrinsics. The intrinsics of the imaging camera
    :params k: Brown Conway model of the distorting camera
    :returns locs: double numpy array of distorted coordinates
    """
    # relative coordinates and distances.
    centre = intrinsics[:2, -1]
    focal = np.diag(intrinsics)[:2]
    x, y = (pts - centre) / focal
    r2 = x ** 2 + y ** 2

    k = np.reshape((dist_coef), (-1))
    kup = (1 + k[0] * r2 + k[1] * (r2 ** 2) + k[4] * (r2 ** 3))
    # distort radially
    xD = x * kup
    yD = y * kup
    # distort tangentially
    xD += 2 * k[2] * x * y + k[3] * (r2 + 2 * (x ** 2))
    yD += k[2] * (r2 + 2 * (y ** 2)) + 2 * k[3] * x * y
    # back to absolute
    locs = np.array([xD, yD]) * focal + centre
    return locs


@njit(parallel=True, cache=True)
def bundle_adj_parrallel_solver(dct: np.ndarray, im_points: np.ndarray,
              projection_matrixes: np.ndarray, intrinsics: np.ndarray, dists: np.ndarray):
    """
    This function calculates the bundle adjustment function for a set of points.
    The first dimension of the dct indicates the number of threads that will be used in parrallel.

    :param dct: The data to use, which is obtained by a TargetDetection.get_data() call, then reshaped to have
        an additional dimension describing the number of threads to use.
    :param im_points: An (ix(u,v,...n)x3) array of image points, where i is the number of images,
        (u,v, .. n) is the number of points given each points (u,v, ... n) key.
    :param projection_matrixes: The projection matrix of all cameras, a (cx3x4) array
    :param intrinsics: The intrinsics of all cameras, a (cx3x3) array
    :param dists: The distortion parameters of all cameras, a (cx5) array
    :return errors: The cost of the bundle adjustment based on the input data.
    """
    errors = np.empty((dct.shape[0], dct.shape[1] * 2))

    for idt in numba.prange(dct.shape[0]):
        errors[idt, :] = bundle_adjustment_costfn(
            dct[idt], im_points, projection_matrixes, intrinsics, dists,
        )
    return errors


@njit(fastmath=True, cache=True)
def bundle_adjustment_costfn(dct: np.ndarray, im_points: np.ndarray,
              projection_matrixes: np.ndarray, intrinsics: np.ndarray, dists: np.ndarray):
    """
    This function calculates the bundle adjustment function for a set of points.

    :param dct: The detection data
    :param im_points: An (i, x, 3) array of image points, where i is the number of images, x = prod(u,v, ... n) and contains flattened key locations.
    :param projection_matrixes: The projection matrix of all cameras, a (cx3x4) array
    :param intrinsics: The intrinsics of all cameras, a (cx3x3) array
    :param dists: The distortion parameters of all cameras, a (cx5) array
    :return errors: The cost of the bundle adjustment based on the input data.
    """
    error = np.empty(len(dct) * 2)
    measured_uv = np.empty(2)
    proj_uv = np.empty((3, 1))
    work_pos = np.empty((4, 1))
    work_pos[-1, 0] = 1.0

    for idx in range(len(dct)):
        cam = int(dct[idx, 0])
        measured_uv[:] = dct[idx, -2:]
        work_pos[:-1, 0] = im_points[  # just automatically account for homogenous
            int(dct[idx, 1]), int(dct[idx, 2]),
        ]
        np.dot(projection_matrixes[cam], work_pos, out=proj_uv)
        proj_uv[0, 0], proj_uv[1, 0] = proj_uv[0, 0] / proj_uv[-1, 0], proj_uv[1, 0] / proj_uv[-1, 0]
        nb_distort_prealloc(proj_uv[:, 0], intrinsics[cam], dists[cam])
        error[idx * 2] = proj_uv[0, 0] - measured_uv[0]
        error[idx * 2 + 1] = proj_uv[1, 0] - measured_uv[1]

    return error


@numba.njit(cache=True)
def make_cartesian(lat, lng):
    """
    converts points from spherical coordinates to cartesian coordinates
    The length is a unit normal by definition.

    :param lat: The latitude of the point
    :param lng: The longitude of the point
    :return x: A cartesian x coordinate
    """
    lat += np.pi / 2  # sign convention transforms
    x = np.sin(lat) * np.cos(lng)  # @0 lng, pi/2 lat = 1
    y = np.sin(lat) * np.sin(lng)  # @0 lng, pi/2 lat = 0
    z = np.cos(lat)
    return np.array([x, y, z])


def make_polar(vec):
    """
    converts points from cartesian coordinates to spherical coordinates

    :param vec: The cartesian input points
    :return gamma, theta: the polar coordinates of the input points.
    """
    vec = vec / np.linalg.norm(vec)
    [x, y, z] = vec
    theta = np.arctan2(y, x)
    gamma = np.arccos(z) - np.pi / 2
    return gamma, theta


@njit(cache=True)
def nb_triangulate_full(data, proj, start_inds, intr, dist):
    """
    Triangulates a set of points from a set of images, using numba acceleration.
    Data is sorted by point, by image, where at least two cameras see the point

    :param data: The data to reconstruct
    :param proj: The camera projection matrices
    :param start_inds: The starting index of each reconstructable point (enabling parrallelisation)
    :param intr: The intrinsics of each camera
    :param dist: The distortion parameters of each camera
    :return:
    """
    pts = np.empty((len(start_inds) - 1, 3))

    for idx in numba.prange(len(start_inds) - 1):
        start_pt, end_pt = start_inds[idx], start_inds[idx + 1]
        diff = end_pt - start_pt

        input_uv = np.empty((diff, 3))
        input_proj = np.empty((diff, 3, 4))
        M = np.empty((3 * diff, 4 + diff))
        M[:] = 0

        for idt in range(diff):
            datum = data[start_pt + idt]
            cam = int(datum[0])
            ud_uv = nb_undistort(datum[-2:], intr[cam], dist[cam])
            input_uv[idt] = [ud_uv[0], ud_uv[1], 1]
            input_proj[idt] = proj[cam]

        pts[idx] = nb_triangulate_nviews(input_proj, input_uv, M)
    return pts


@numba.njit(fastmath=True, cache=True)
def nb_triangulate_nviews(P, ip, M):
    """
    Triangulate a point visible in n camera views, using numba acceleration

    :param P: a 3d array of  camera projection matrices.
    :param ip: a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ] in homegenous image points
    :M A preallocated array of shape (3*n, 4+n) for math reasons
    len of ip must be the same as len of P
    :return X: the 3d point in space
    """
    M[:] = 0

    for i, (x, p) in enumerate(zip(ip, P)):
        M[3 * i:3 * i + 3, :4] = p
        M[3 * i:3 * i + 3, 4 + i] = -x
    V = np.linalg.svd(M, full_matrices=False)[-1]
    X = V[-1, :4]
    return X[:3] / X[3]


@njit(cache=True)
def n_inv_pose(inp, out):
    """
    Inverts a pose transform given a flat 12x1 representation

    :param inp: The input pose to invert
    :param out: The output location.
    :return:
    """
    # a pose transform is defined as R*x0 + t = x1
    # tf x0 = R-1 *x1 - R-1-t

    out[0] = inp[0]
    out[1] = inp[3]
    out[2] = inp[6]
    out[3] = inp[1]
    out[4] = inp[4]
    out[5] = inp[7]
    out[6] = inp[2]
    out[7] = inp[5]
    out[8] = inp[8]
    np.dot(-inp[9:].reshape(1, 3), inp[:9].reshape(3, 3), out[9:].reshape(1, 3))


@njit(cache=True)
def n_dist(x):
    """
    Calculates the distance between a set of points

    :param x: The points to calculate the internal distances
    :return:
    """
    l = len(x)
    block_array = -2 * x @ x.T
    s_array = x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2
    block_array += s_array.reshape((1, l))
    block_array += s_array.reshape((l, 1))
    return np.sqrt(np.abs(block_array))


@njit(cache=True)
def n_dist_prealloc(x, out):
    """
    Calculates the distance between a set of points

    :param x: The points to calculate the internal distances of
    :param out: The output location
    """
    np.dot(x, x.T, out)
    out *= -2
    for i in range(x.shape[0]):
        s = x[i, 0] ** 2 + x[i, 1] ** 2 + x[i, 2] ** 2
        for j in range(x.shape[0]):
            out[j, i] += s
            out[i, j] += s

    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            out[i, j] = abs(out[i, j]) ** 0.5


# @njit(cache=True)
def n_estimate_rigid_transform(v0:np.ndarray, v1:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the rigid transform between two sets of points using an svd

    :params v0: The first set of points
    :params v1: The second set of points
    """
    ndim = v0.shape[1]

    t0 = np.zeros((ndim))
    t1 = np.zeros((ndim))

    matR = np.zeros((ndim,ndim))
    for i in range(v1.shape[0]): #gets the mean vectors of both
        t0 += v0[i, :]
        t1 += v1[i, :]
    t0 /= v0.shape[0]
    t1 /= v1.shape[0]
    lv0 = v0 - t0
    lv1 = v1 - t1

    matR =  lv0.T @ lv1
    u, _ , vh = np.linalg.svd(matR) 
    matR = vh.T @ u.T
    inp = np.eye(ndim)
    inp[-1,-1] = np.linalg.det(matR)
    matR = vh.T @ inp @ u.T
    # the process described here is a transformation from 
    t = - matR @ t0 + t1

    # error = np.sum(np.abs((matR @ v0.T).T + t - v1))
    # print(f"default={np.sum(np.abs(v0 - v1))}, tformed = {error}")
    
    return matR, t
