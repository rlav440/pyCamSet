import functools
import logging
import math as m
import time
from itertools import zip_longest, chain
from pathlib import Path

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


def benchmark(func, repeats=100, mode="ms", timer=time.time_ns):
    """
    A handy function to benchmark a function. Tracks the execution time, and also the numba allocations.
    :param func: The function to benchmark as a lambda
    :param repeats: The number of times to repeat the function call
    :param mode: The mode to display the results in. Can be "us", "ms", or "s"
    :param timer: The timer to use. Can be time.time_ns, or time.perf_counter_ns
    """

    def run_benchmark():
        ranges = {
            "us":1e-3,
            "ms":1e-6,
            "s":1e-9,
        }
        starting_alloc = numba.core.runtime.rtsys.get_allocation_stats()[0]
        times = []
        for _ in range(repeats):
            start = timer()
            func()
            end=timer()
            times.append(end-start)

        times = np.array(times)
        mean = np.mean(times) * ranges[mode]
        stdev = np.std(times * ranges[mode])
        median = np.median(times) * ranges[mode]
        max_t = min(mean + 3*stdev,np.amax(times) * ranges[mode])
        print(f"Mean: {mean:.2f} {mode}, median: {median:.2f} {mode}, stdev: {stdev:.2f} {mode}")
        histogram(times*ranges[mode], bins=50,
                  bins_min=max(mean- 3*stdev, 0),
                  x_max= max_t,
                  height = 3,
                  color = True,
                  y_unit=" freq",
                  x_unit=mode,
                  )
        final_alloc = numba.core.runtime.rtsys.get_allocation_stats()[0]
        print(f"Mean numba allocations: {(final_alloc - starting_alloc)/repeats:.0f}")
    run_benchmark()


def mad_outlier_detection(data: np.ndarray|list, out_thresh = 3, draw=True) -> np.ndarray or None:
    """
    Implemenents Median Absolute Deviation outlier detection.
    :param data: The data to process
    :param out_thresh: The outlier threshold to reject
    :param draw: Whether to draw the results
    :return: A boolean array of the outliers.
    """
    n_mdn = np.median(data)
    n_mad = np.median(np.absolute(np.array(data) - n_mdn))
    outliers = np.abs(np.array(data) - n_mdn) / n_mad > out_thresh

    if np.any(outliers):
        w_out = np.nonzero(outliers)
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
    imlocs = [p.resolve() for p in loc.glob("**/*") if p.suffix in {".png", '.bmp', '.tiff', '.jpg'}]
    return imlocs


def glob_ims_local(loc: Path):
    """
    Returns a list of all images in this folder
    :param loc:
    :return:
    """
    imlocs = [p.resolve() for p in loc.glob("*") if p.suffix in {".png", '.bmp', '.tiff', '.jpg'}]
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


def distort_points(pts:np.ndarray, intrinsics: np.ndarray, dist_coef:np.ndarray) -> np.ndarray:
    """
    Distorts points using the Brown Conway model

    :param pts: points to distort
    :param intrinsics. The intrinsics of the imaging camera
    :param dist_coef: Brown Conway model of the distorting camera
    :return pts: double numpy array of distorted coordinates
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
    xD += 2*p1*x*y + p2 * (r2 + 2*(x**2))
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
        aruco_dict = aruco.custom_dictionary(0, base.markerSize,)
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

def downsample_valid(inp, d_factor, invalid=None):
    """
    An averaging downsample using a numpy array indexing.

    :param inp: The input to be be downsampled
    :param d_factor: The factor to downsample
    :param invalid: The value of points to be excluded from the downsampling in the function

    :return  For a point with inputs, returns the average of the valid inputs
        For a point without valid inputs, returns the value of the invalid inputs
        Returned object has no singleton dimensions

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

