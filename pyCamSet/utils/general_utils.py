import functools
import logging
import math as m
import sys
import time
from itertools import zip_longest, chain
from pathlib import Path
import os

import cv2
import numba
import numpy as np
from cv2 import aruco
from matplotlib import pyplot as plt
from natsort import natsorted
from numpy.linalg import svd
from tqdm import tqdm
from uniplot import histogram

from scipy.spatial.transform import Rotation as R

def ask_yes_no(context: str, question: str, default: str = 'n') -> str:
    """
    Asks a yes/no question, falling back to *default* when nobody can answer.

    A bare input() call blocks forever when there is no terminal attached,
    which turns an outlier detected during an automated run -- a test suite, a
    CI job, a batch script -- into a hang that only ends at the timeout.  This
    checks for an interactive stdin first and returns the default otherwise,
    saying so in the log rather than silently.

    :param context: what has been found, logged either way
    :param question: the question to put to the user
    :param default: the answer to assume when running unattended
    :return: 'y' or 'n'
    """
    logging.info(context)
    try:
        interactive = sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):  # closed or replaced stdin
        interactive = False

    if not interactive:
        logging.warning(
            f"{context} Not running interactively, so assuming '{default}'. "
            f"Set the handler's 'outliers' option to 'y' or 'n' to choose."
        )
        return default

    answer = ''
    while answer not in ('y', 'n'):
        answer = input(f"{question}: \n y/n: ").strip().lower()
    return answer


def list_dict_to_np_array(d) -> dict:
    if isinstance(d, dict):
        for key, val in d.items():
            if isinstance(val, dict):
                list_dict_to_np_array(val)
            elif isinstance(val, list):
                d[key] = np.array(val)
            else:
                pass
    return d

def approx_average_quaternion(quats:list[np.ndarray]) -> np.ndarray:
    q = np.array([q for q in quats if not np.any(np.isnan(q))])
    w = np.ones(len(q))/len(q)
    eig = np.linalg.eigh(np.einsum('ij,ik,i->...jk', q, q, w))[1][:, -1]
    return eig

def average_tforms(tforms: list[np.ndarray]):
    tforms = [t for t in tforms if not np.any(np.isnan(t))]
    if len(tforms) == 0:
        return np.ones((4,4)) * np.nan
    if len(tforms) == 1:
        return tforms[0]
    average_translation = np.mean([t[:3,-1] for t in tforms], axis=0)
    quats = [R.from_matrix(t[:3,:3]).as_quat(canonical=True) for t in tforms]
        
    average_quat = approx_average_quaternion(quats)
    average_rot = R.from_quat(average_quat).as_matrix() 
    return np.block([[average_rot, average_translation.reshape((3,1))],[0,0,0,1]])


def flatten_pose_list(pose_list):
    """
    A handy function to flatten a list of poses into a single array
    :param pose_list: a list of 4x4 poses to flatten
    :return:
    """
    params = [ext_4x4_to_rod(t) for t in pose_list]
    return np.concatenate(list(chain(*params)), axis=0)


def benchmark(func, repeats=100, mode="ms", timer=time.time_ns, max_runtime=100):
    """
    A handy function to benchmark a function. Tracks the execution time, and also the numba allocations.
    :param func: The function to benchmark as a lambda
    :param repeats: The number of times to repeat the function call
    :param mode: The mode to display the results in. Can be "us", "ms", or "s"
    :param timer: The timer to use. Can be time.time_ns, or time.perf_counter_ns
    """

    ranges = {
        "us":1e-3,
        "ms":1e-6,
        "s":1e-9,
    }
    # starting_alloc = numba.core.runtime.rtsys.get_allocation_stats()[0]
    times = []
    loop_start = timer()
    for _ in range(repeats):
        start = timer()
        func()
        end=timer()
        times.append(end-start)
        total_time = end - loop_start
        if (total_time * ranges['s'] )> max_runtime:
            print(
            f"Exceeded given max_runtime of {max_runtime} seconds."
            )
            break

    times = np.array(times)
    mean = np.mean(times) * ranges[mode]
    stdev = np.std(times * ranges[mode])
    median = np.median(times) * ranges[mode]
    max_t = min(mean + 3*stdev,np.amax(times) * ranges[mode])
    print(f"Mean: {mean:.2f} {mode}, median: {median:.2f} {mode}, stdev: {stdev:.2f} {mode}")
    histogram(times*ranges[mode], bins=20,
              bins_min=max(mean- 3*stdev, 0),
              x_max = min(mean + 5*stdev, max_t),
              height = 3,
              color = True,
              y_unit=" freq",
              x_unit=mode,
              )
    # final_alloc = numba.core.runtime.rtsys.get_allocation_stats()[0]


def mad_outlier_detection(data: np.ndarray|list, out_thresh = 3, draw=True) -> np.ndarray or None:
    """
    Implemenents Median Absolute Deviation outlier detection.
    :param data: The data to process
    :param out_thresh: The outlier threshold to reject
    :param draw: Whether to draw the results
    :return: The indices of the outliers, or None when there are none.
    """
    n_mdn = np.median(data)
    n_mad = np.median(np.absolute(np.array(data) - n_mdn))
    outliers = np.abs(np.array(data) - n_mdn) / n_mad > out_thresh

    if np.any(outliers):
        # [0] to unwrap np.nonzero's tuple into a plain index array.  Callers
        # that index with the result tolerate either, but delete_row does not:
        # handed the tuple it fell through to a broadcast comparison and
        # deleted a single arbitrary row instead of the flagged images.
        w_out = np.nonzero(outliers)[0]
        listout = functools.reduce(lambda x, y: x+y, [f" {w}" for w in w_out])

        logging.critical(f'found outliers in indicies:{listout}')
        logging.critical(f'These may prevent calibration conversion')
        if draw:
            fig, ax = plt.subplots(1, 1)
            ax.plot(np.abs(np.array(data) - n_mdn) / n_mad, '.')
            ax.set_title("Found outliers: displaying mad outlier threshold as red line")
            ax.axhline(out_thresh, color='r')
            plt.show()
        return w_out
    return None



def glob_ims(loc: Path):
    """
    Returns a list of all images one folder below the input path
    :param loc:
    :return:
    """
    imlocs = [p.resolve() for p in loc.glob("**/*") if p.suffix in {".png", '.bmp', '.tiff', '.jpeg', '.jpg'}]
    return imlocs


def glob_ims_local(loc: Path):
    """
    Returns a list of all images in this folder
    :param loc:
    :return:
    """
    imlocs = [p.resolve() for p in loc.glob("*") if p.suffix in {".png", '.bmp', '.tiff', '.jpeg', '.jpg'}]
    return imlocs

def plane_fit(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
    """

    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.
    return ctr, svd(M)[0][:,-1]


def write_colour_ply(f_name, verts, cols):
    """
    Writes a colour point cloud. This is a basic wrapper and is very slow.
    Potential to replace this with an open3d or pyvista implementation.
    :param f_name: The file location to write to
    :param verts: the points of the cloud in 3D space
    :param cols: the colours of the cloud.
    """
    with open(f_name, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for vert, col in zip(tqdm(verts), cols):
            f.write(f'{vert[0]:.8f} {vert[1]:.8f} {vert[2]:.8f} {col[0]}'
                    f' {col[1]} {col[2]} \n')
        f.write("")


def get_subfolder_names(f_loc: Path, return_full_path = False) -> list[Path] | list[str]:
    """
    Notes: This is a function that returns a list of subfolder names.
        for the purposes of this library, it's usually used to grab camera names
    Args:
        f_loc:

    Returns:

    """
    detected_sub_folders = [p for p in f_loc.glob('*/') if p.is_dir()]
    detected_sub_folders= natsorted(detected_sub_folders)
    if return_full_path:
        return detected_sub_folders

    subfolder_names = [p.parts[-1] for p in detected_sub_folders]
    return subfolder_names


def get_close_square_tuple(n):
    """

    Args:
        n:

    Returns:
        (x,y)

    """

    x = m.ceil(m.sqrt(n))
    y = m.ceil(n/x)
    return (x,y)

def h_tform(points: np.ndarray, transform:np.ndarray, fill=1) -> np.ndarray:

    """
    Performms a homogenous transformation on data
    
    :param points: the points to transform
    :param transform: the 4x4 transformation
    :param fill: 1 for points, 0 for vectors.
    :return pts: the transformed points

    """
    if points.ndim == 1:
        points = points[None, ...]

    homogenous_points = np.concatenate(
        [points, np.ones((len(points), 1))*fill], axis=-1
    )[..., None]
    new_points = (transform[None, ...] @ homogenous_points)[..., 0] #always 0 on this axis
    if fill==1:
        new_points = (
                new_points[:, :-1] / new_points[:, -1][..., None]
        )
    else:
        new_points = new_points[:,:-1]
    return new_points.squeeze()

def ext_4x4_to_rod(h4):
    """
    Converts a 4x4 extrinsic matrix to a rotation vector
    
    :param h4: the input 4x4 matrix
    :return rot, trans: the rotation and translation vectors
    """
    rot_m = h4[:3, :3]
    trans = h4[:3, -1]
    rot, m = cv2.Rodrigues(rot_m)
    return rot.squeeze(), trans


def colourmap_to_colour_list(len, colourmap):
    pts = np.linspace(0,1,len)
    return [np.array(colourmap(pt, bytes=True))[:3] for pt in pts]


def distort_points(
        pts: np.ndarray, intrinsics: np.ndarray, dist_coef: np.ndarray
) -> tuple[float, float]:
    """
    Distorts a point using the Brown Conrady model

    Takes one point at a time: x, y = (pts - centre)/focal unpacks the leading
    axis, so pts must be a single pixel.  Note the layout is the transpose of
    Camera.project_points, which returns (n, 2); map over its rows to distort
    many points.

    :param pts: the pixel to distort, as (u, v)
    :param intrinsics: The intrinsics of the imaging camera
    :param dist_coef: 5 parameter Brown Conrady model of the distorting camera
    :return: the distorted (u, v) as a pair of scalars
    """
    #relative coordinates and distances.
    centre = intrinsics[:2, -1]
    focal = np.diag(intrinsics)[:2]
    x, y = (pts - centre)/focal
    r2 = x**2 + y**2

    [k1,k2,p1,p2,k3] = np.reshape((dist_coef), (-1))
    #distort radially
    xD = x * (1 + k1*r2 + k2*(r2**2) + k3*(r2**3))
    yD = y * (1 + k1*r2 + k2*(r2**2) + k3*(r2**3))
    #distort tangentially
    xD += 2*p1*x*y + p2 *(r2 + 2*(x**2))
    yD += p1*(r2 + 2*(y**2)) + 2 * p2 * x * y
    #back to absolute
    xN, yN = [xD, yD] * focal + centre
    return xN, yN


def split_aruco_dictionary(
        split_size: int,
        a_dict = aruco.DICT_6X6_250
    ):
    """
    Splits an aruco dictionary into multiple smaller dictionaries.

    :param split_size: The size of the output dictionaries
    :param a_dict: The input dictionary to split.
    """

    i = 0

    if isinstance(a_dict, int):
        base = aruco.getPredefinedDictionary(a_dict)
    else:
        base = a_dict
    markers = base.bytesList
    n_markers = markers.shape[0]

    split_markers = grouper(markers, split_size)

    limit = int(np.floor(n_markers/split_size))
    aruco_dicts = []
    for set in split_markers:

        temp = np.array(set)
        aruco_dict = aruco.Dictionary(0, base.markerSize,)
        aruco_dict.bytesList = np.empty(shape=(split_size, markers.shape[1],
                                               markers.shape[2]),
                                        dtype=np.uint8)
        aruco_dict.bytesList = temp
        aruco_dicts.append(aruco_dict)

        i += 1
        if i == limit:
            break
    return aruco_dicts


def grouper(iterable, n, fillvalue=None):
    """
    Returns an iterable of n items at a time from some originally iterable object.

    :param iterable: The iterable object to group
    :param n: The number of items to group
    :param fillvalue: The value to fill the last group with if it is not full.
    :return The iterable:
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def make_4x4h_tform(euler_angles, trans, mode='opencv'):
    """
    :param euler_angles: opencv axis angle representation of rotation
        these angles default to being in radians
    :param trans: the translation taken by the camera from the 0,0,0
    :param mode: the convention used for assembling the transform.
    :return: 4x4 transformation
    """

    euler_angles = np.array(euler_angles).squeeze()
    trans = np.array(trans).squeeze()
    if euler_angles.ndim < 2:
        Rot, _ = cv2.Rodrigues(euler_angles)

    else:
        Rot = euler_angles
    if mode == 'mvg':
        transform = np.block([[Rot, -(Rot @ trans)[..., np.newaxis]],
                              [0, 0, 0, 1]])
    elif mode == 'opencv':
        transform = np.block([[Rot, trans[..., np.newaxis]],
                              [0, 0, 0, 1]])
    else:
        raise ValueError(mode + ' is an invalid 4x4 type')
    return transform


def px_array(res=[32, 32], startZero=False,):
    """
    creates the index grid once during the full ingest pipeline
    
    :param res: The resolution of the camera
    :param startZero: whether to start the grid at zero or have zero be the middle
    :return:
    """
    # only works with even numbers
    if startZero:
        x = range(res[0])
        y = range(res[1])
    else:
        x = range(res[0] // 2, -res[0] // 2, -1)
        y = range(-res[1] // 2, res[1] // 2)
    y, x = np.meshgrid(y, x)
    h = np.ones(res)
    return x, y, h

def downsample_valid(inp: np.ndarray, d_factor: int) -> np.ndarray:
    """
    An averaging downsample using numpy array indexing.

    Averages each d_factor x d_factor block of the first two axes.  Rows and
    columns past the last whole block are cropped rather than padded, so a
    (5, 7) input at d_factor 2 returns (2, 3).

    Note that no value is treated as invalid: a NaN in a block propagates to
    that block's mean.

    :param inp: The input to be downsampled
    :param d_factor: The factor to downsample by; 1 returns the input unchanged
    :return: The block averaged array
    """
    if d_factor == 1:
        return inp

    shape = np.array(inp.shape)
    rem = shape % d_factor
    up_to = shape - rem

    im = inp[:up_to[0], :up_to[1]]

    return np.mean(im.reshape(im.shape[0]//d_factor, d_factor, im.shape[
        1]//d_factor, d_factor), axis=(1, 3))


def vector_cam_points(type, pts, intrinsics, cam_to_world):
    """
    Makes a sensor map or a smaller amount of points, designed to
    make running a few smaller things easier

    :param type: normalised or linear sensor map
    :param pts: the points to give vectors too
    :param intrinsics: the camera intrinsics
    :param cam_to_world: the transformation from camera coordinates to world coordinates.
    """

    if (type != 'normalised') and (type != 'linear'):
        raise ValueError("Invalid sensor map type")

    c_int = np.linalg.inv(intrinsics)
    cords = np.concatenate([pts, np.ones_like(pts[:, 0])[:, None]], axis=-1).T
    s_map = (c_int @ cords).T

    # do things to the data for specific sensor types
    if type == "normalised":
        s_map /= np.linalg.norm(s_map, axis=-1, keepdims=True)

    return h_tform(s_map, cam_to_world, fill=0) #.transpose(s_map, (1,0,2))

def sensor_map(type, intrinsics, res=(1600, 1200), dist_coefs=None):
    """
    This creates a sensor map for a representation of a sensor. A sensor map is the ray vector associated with
    each pixel. It is essentially a precomputed ray cast.

    :param type: normalised or linear sensor map. Normalised has length 1, linear has z==1.
    :param intrinsics: The camera intrinsics
    :param res: the nominal resolution of the input camera THIS USES OPENCV (y,x) PIXEL ORDER.
    :param dist_coefs: The distortion coefficients of the camera.
    :returns : A sensor map of the appropriate resolution,
    """
    if (type != 'normalised') and (type != 'linear'):
        raise ValueError("Invalid sensor map type")

    c_int = np.linalg.inv(intrinsics)
    u, v, h = px_array(res=res, startZero=True)
    c = np.stack((u.flatten(), v.flatten()))
    if dist_coefs is not None:
        c = cv2.undistortImagePoints(c.T.astype(float), intrinsics, dist_coefs).squeeze().T
    [u, v] = c
    cords = np.stack((u.flatten(), v.flatten(), h.flatten()))
    s_map = (c_int @ cords).T.reshape(res[0], res[1], 3)

    # do things to the data for specific sensor types
    if type == "normalised":
        s_map /= np.linalg.norm(s_map, axis=-1, keepdims=True)

    return s_map #.transpose(s_map, (1,0,2))


def adaptive_decimated_charuco_detection_stereo(frame_L, charuco_board, aruco_dict, rescale_corners_to_original=True):
    '''
    This function runs through a series of decimation numbers on the input images and runs corner detections on the decimated image.
    It will find the optimal decimation factor that would yield the most number of corners detected on either images

    :param frame_L: original res image
    :param charuco_board: the cv2.aruco.CharucoBoard to detect.
    :param aruco_dict: retained for backwards compatibility. The board already
        carries its own dictionary, which is what the detector uses.
    :param rescale_corners_to_original: if True will corners will multiply with optimal decimation factor (default = True)
    :return charuco_corners_L - charuco corners detected from downsampled image
    :return optimal_decimation - optimal downsampled factor
    '''

    # This used cv2.aruco.detectMarkers + interpolateCornersCharuco, which
    # OpenCV removed in 4.7 -- below the supported floor, so the old body
    # raised AttributeError on every version this package allows.
    # CharucoDetector.detectBoard replaces both: it detects the markers and
    # interpolates the chessboard corners in one call.
    detector = cv2.aruco.CharucoDetector(charuco_board)

    # Initialize variables to store the best decimation factor and maximum corners detected in either image and the charuco corners detected in both the left and right image
    optimal_decimation = 1
    max_corners_detected = 0
    charuco_corners_L = None
    charuco_ids = None
    # adaptive down sampling to account for large resolution images
    for decimation_factor in range(1, 12):
        decimated_image_L = frame_L[::decimation_factor, ::decimation_factor]
        all_charuco_corners_L, all_charuco_ids_L, _, _ = detector.detectBoard(decimated_image_L)
        if all_charuco_corners_L is None:
            continue
        num_corners_detected_L = len(all_charuco_corners_L)
        if num_corners_detected_L > max_corners_detected:
            max_corners_detected = num_corners_detected_L
            optimal_decimation = decimation_factor
            # OpenCV 5 squeezes the singleton axis off both returns; reshape so
            # this function keeps handing back the (N, 1, 2) / (N, 1) arrays it
            # always has.
            charuco_corners_L = np.asarray(all_charuco_corners_L).reshape(-1, 1, 2)
            charuco_ids = np.asarray(all_charuco_ids_L).reshape(-1, 1)

    if charuco_corners_L is not None and rescale_corners_to_original:
        charuco_corners_L = charuco_corners_L * optimal_decimation

    return charuco_corners_L, charuco_ids, optimal_decimation
