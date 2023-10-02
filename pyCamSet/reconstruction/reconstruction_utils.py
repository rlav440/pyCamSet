import logging

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pyvista as pv
from functools import reduce

from pyCamSet.cameras import Camera
from pyCamSet.utils.general_utils import ext_4x4_to_rod

def undistort_im(image, cam: Camera) -> np.ndarray:
    """
    A function to undistort an image using a camera object.

    :param image: The image to undistort.
    :param cam: The camera object to undistort with.
    :return:
    """
    im = cv2.undistort(image, cam.intrinsic, cam.distortion_coefs, None, cam.intrinsic)
    return im


def depth_image_ptcloud_mask(depth_im, mind, maxd):
    """
    A function to filter a point cloud based on its associated depth image, applying a minimum and maximum depth.

    :param depth_im: A point cloud to filter.
    :param mind: The min depth
    :param maxd: The max depth.
    :return: The mask to filter the cloud
    """
    t_nan = np.any(np.isnan(depth_im), axis=-1)
    t_inf = np.any(np.isinf(depth_im), axis=-1)
    t_hgh = depth_im[:, -1] > maxd
    t_low = depth_im[:, -1] < mind
    mask = ~ reduce(np.logical_or, [t_nan, t_inf, t_hgh, t_low])
    return mask

def remap_im(im, cam: Camera, new_rot, new_proj, new_size) -> np.ndarray:
    """
    A function to remap an image using a camera object.

    :param im: The impage to remap.
    :param cam: The associated camera.
    :param new_rot: The new rotation matrix.
    :param new_proj: The new projection matrix.
    :param new_size: The new image size.
    :return: A remappeed image.
    """
    map = cv2.initUndistortRectifyMap(
        cam.intrinsic, cam.distortion_coefs,
        new_rot, new_proj,
        new_size, #[new_size[2], new_size[3]],
        cv2.CV_32FC1,
    )
    new_im0 = cv2.remap(im, *map, cv2.INTER_CUBIC)
    return new_im0


def rectify_camera_images(
        cam_0:Camera, cam_1:Camera, im_0, im_1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A function to rectify a pair of images using a pair of cameras.

    :param cam_0: The reference camera.
    :param cam_1: The comparison camera
    :param im_0: The image of the reference camera.
    :param im_1: The image of the comparison camera.
    :return: The remapped images and the associated q matrix
    """

    zero_flag = True
    p0, p1, q, r0, r1, s0 = rectify_camera_pair(cam_0, cam_1, zero_flag=zero_flag)
    new_im0 = remap_im(
        undistort_im(im_0, cam_0) if zero_flag else im_0, cam_0, r0, p0, cam_0.res
    )
    new_im1 = remap_im(
        undistort_im(im_1, cam_1) if zero_flag else im_1, cam_1, r1, p1, cam_1.res
    )
    return new_im0, new_im1, q


def rectify_camera_pair(cam_0: Camera, cam_1:Camera, zero_flag = False):
    """
    Generates the rectification matrices for a pair of cameras.

    :param cam_0: The reference Camera
    :param cam_1: The comparison Camera
    :param zero_flag: a flag to force zero distortion.
    :return: The recitification matrices.
    """
    rot, trans = ext_4x4_to_rod(cam_1.extrinsic @ cam_0.cam_to_world)
    # it's looking for world -> cam in reference space
    #

    r0, r1, p0, p1, q, s0, s1 = cv2.stereoRectify(
        cam_0.intrinsic, np.zeros(5) if zero_flag else cam_0.distortion_coefs,
        cam_1.intrinsic, np.zeros(5) if zero_flag else cam_1.distortion_coefs,
        cam_0.res,
        rot, trans,
        cv2.CALIB_ZERO_DISPARITY,
        alpha=1,
        newImageSize=cam_0.res
    )
    return p0, p1, q, r0, r1, s0


def disparity_to_ptcld(disp, q) -> tuple[pv.PolyData, np.ndarray]:
    """
    Converts a disparity image to a point cloud using the associated q matrix.

    :param disp: the disparity image from the refence camera.
    :param q: The q matrix.
    :return:
    """
    pt_cloud = cv2.reprojectImageTo3D((disp / 16).astype('float32'), q)

    test = np.reshape(pt_cloud, (-1, 3)) * [1, 1, 1]

    t_nan = np.any(np.isnan(test), axis=-1)
    t_inf = np.any(np.isinf(test), axis=-1)
    t_hgh = test[:, -1] > 2.5
    t_low = test[:, -1] < 0.5

    mask = ~ reduce(np.logical_or,
                    [
                        t_nan,
                        t_inf,
                        t_hgh,
                        t_low,
                    ]
                    )

    cloud = pv.PolyData(test[mask])
    return cloud, mask


def matlab_stereo(im0, im1, disp_range = (128, 256), uniqueness_thresh=25, plot=False):
    """
    A function to run the matlab stereo algorithm, as it seems to perform better than the opencv one.

    :param im0: The reference image, rectified
    :param im1: The comparison image, rectified
    :param disp_range: The disparity range to search
    :param uniqueness_thresh: The uniqueness threshold, used to reject noise.
    :param plot: Whether to plot the disparity image.
    :return: The disparity image
    """

    try:
        from matlab.engine import start_matlab
    except ImportError as e:
        logging.info("matlab engine not installed")
        raise e

    m = start_matlab()
    disp = np.array(m.disparitySGM(
        im0, im1, 'DisparityRange', np.array(disp_range).astype('int32'), 'UniquenessThreshold', uniqueness_thresh
    ))
    disp -= 1
    if plot:
        plt.imshow(disp)
        plt.colorbar()
        plt.show()
    return disp


def stereo_reconstruct( cam_0:Camera, cam_1:Camera, im_0, im_1, num_disp=256, blockSize=25, matlab=False, plot=False):
    """
    A function to reconstruct a point cloud from a pair of images, using the opencv stereosgbm algorithm.

    :param cam_0: the reference camera
    :param cam_1: the comparison camera
    :param im_0: the reference image
    :param im_1: the comparison image
    :param num_disp: the number of disparities to search
    :param blockSize: the block size to use, larger is smoother and more likely to match.
    :param matlab: whether to use the matlab stereo algorithm instead of the opencv one.
    :param plot: Whether to plot the point cloud.
    :return: the output point cloud.
    """

    r0, r1, q = rectify_camera_images(cam_0, cam_1, im_0, im_1)
    # stacked_im = np.stack([r0, np.zeros_like(r0), r1]).transpose([1,2,0])
    # plt.imshow(stacked_im)
    # plt.show()
    if matlab:
        disp = matlab_stereo(r0, r1, disp_range=(num_disp-128, num_disp), plot=plot)
    else:
        stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=blockSize
                                     # uniquenessRatio=15
                                     )
        disp = (stereo.compute(r0.astype(np.uint8), r1.astype(np.uint8))/16)
        plt.imshow(disp)
        plt.show()
    """
    default disp = 0-128, uniqueness_thresh = 50"""

    pt_cloud = cv2.reprojectImageTo3D((disp).astype('float32'), q)

    test = np.reshape(pt_cloud, (-1, 3)) * [1, 1, 1]

    t_nan = np.any(np.isnan(test), axis=-1)
    t_inf = np.any(np.isinf(test), axis=-1)
    t_hgh = test[:,-1] > 2
    t_low = test[:, -1] < 0

    mask = ~reduce(np.logical_or, [
        t_nan,
        t_inf,
        t_hgh,
        t_low,
        ]
    )
    cloud = pv.PolyData(test[mask])
    cloud['i'] = r0.flatten()[mask]
    if plot:
        pv.set_plot_theme("Document")

        cloud.plot(cmap='gray', point_size=0.75)
    return cloud
