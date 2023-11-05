from __future__ import annotations
import logging
import math as m
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from numpy.linalg import norm

from pyCamSet.utils.general_utils import distort_points, h_tform, vector_cam_points
from pyCamSet.utils.general_utils import sensor_map

DEFAULT_RES = [1000, 1000]
DEFAULT_CAMERA_MATRIX = np.array(
    [[1000, 0.0, DEFAULT_RES[0] / 2],
     [0.0, 1000, DEFAULT_RES[1] / 2],
     [0.0, 0.0, 1.0]])  # mm


class Camera:
    """
    An object oriented camera model.
    Using the pinhole camera model this represents the position and imaging characteristics
    of a camera.
    """

    def __init__(self,
                 extrinsic=np.eye(4),
                 intrinsic=None,
                 res=None,
                 distortion_coefs=np.array([0,0,0,0,0]),
                 name = None,
                 minimal = True):
        """
        Initialises a camera
        
        :param extrinsic: the camera extrinsic matrix
        :param intrinsic: the camera intrinsic matrix
        :param res: the camera resoluiton
        :param distortion_coefs: 5 parameter distortion model from camera
        :param name: the camera name
        :param minimal: lazy generation of sensor map creation for high resolution cameras intensive work loads
        """
        if res is None:
            res = DEFAULT_RES
        self.res = res
        self.extrinsic = extrinsic

        if intrinsic is None:
            intrinsic = DEFAULT_CAMERA_MATRIX

        self.intrinsic = intrinsic
        self.original_matrix = deepcopy(self.intrinsic)  # stored for reference #TODO rename to original_intrinsic_matrix
        self.distortion_coefs = distortion_coefs
        self.cam_to_world = None

        self.down_scale_factor = 0
        self.name = name
        self.minimal = minimal
        self._update_state()

    def __eq__(self, other: Camera):
        """
        Override the equal to basically say the cameras are mathematically identical, ignoring name

        :param other: The other object. if it's not a camera, returns false.
        :return: True or false.
        """
        if not isinstance(other, Camera):
            return False
        equal_int = np.isclose(self.intrinsic, other.intrinsic)
        equal_ext = np.isclose(self.extrinsic, other.extrinsic)
        equal_dst = np.isclose(self.distortion_coefs, other.distortion_coefs)
        return all([equal_dst, equal_ext, equal_int])

    def set_minimal(self, minimal: bool):
        """
        Either removes or recalculates the sensor map and world sensor map of a camera

        :param minimal: If a new camera is a minimal model
        """
        self.minimal = minimal
        self.sensor_map = None
        self.world_sensor_map = None
        self._update_state()

    def to_MVSnet_txt(self, f_loc: Path, depth_range: tuple[float, float], depth_steps: int) -> None:
        """
        Writes the camera data to the standard format used by MVSnet and its derivatives.

        :param f_loc: The file location
        :param depth_range: The depth range to use.
        :param depth_steps: The number of discrete steps
        """
        with open(f_loc, 'w') as f:
            f.write('extrinsic' + '\n')
            for row in self.extrinsic:
                f.write(f"{row[0]} {row[1]} {row[2]} {row[3]}" + '\n')
            f.write('\n')
            f.write("intrinsic" + '\n')
            for row in self.intrinsic:
                f.write(f"{row[0]} {row[1]} {row[2]}" + '\n')
            f.write('\n')
            f.write(
                f"{depth_range[0]} {(depth_range[1] - depth_range[0]) / depth_steps} {depth_steps} {depth_range[1]}" + '\n')


    def _compute_world_sensor_map(self):
        '''
        transforms the returned sensormap to be in world coordinates relative to position and view, rather than camera
        gives the sensor map as an offset from the prinicple position because this is much easier.
        '''
        pts = np.ones(self.res)[..., np.newaxis]
        # big boy intrinsic multiplication

        temp_ext = self.cam_to_world[np.newaxis, np.newaxis, ...]
        temp_map = np.concatenate((self.sensor_map, pts), axis=-1)[
            ..., np.newaxis]

        s_map = (temp_ext @ temp_map)[:, :, :3, 0]
        s_map -= self.position

        return s_map

    def set_extrinsic(self, new_extrinsic):
        """
        Sets the extrinsic matrix, and updates the camera in place

        :param new_extrinsic: A newly defined extrinsic calibration
        """
        self.extrinsic = new_extrinsic
        self._update_state()


    def set_distortion_coefs(self, dist_coefs: np.ndarray):
        """
        Sets the distortions coefs and updates in place

        :param dist_coefs: A 5 parameter distortion model
        """
        self.distortion_coefs = dist_coefs
        self._update_state()

    def view_sensor_distortion(self, ax=None):
        """
        Draws the distortion of the current camera sensor

        :param ax: A matplotlib plotting axis to draw the distortion too.
        """
        grid = np.meshgrid(
            np.arange(0, self.res[0], 100),
            np.arange(0, self.res[1], 100),
        )
        grid = np.c_[
            grid[0].ravel(),
            grid[1].ravel(),
        ]
        n_grid = np.array([distort_points(g, self.intrinsic, self.distortion_coefs) for g in grid])
        shift = (n_grid - grid)

        if ax is None:
            plt.quiver(grid[:, 0], grid[:, 1], shift[:, 0], shift[:, 1], angles='xy', scale_units='xy', scale=1)
            plt.set_aspect('equal')
            plt.title(f'Distortion in camera {self.name}')
            plt.show()
        else:
            ax.quiver(grid[:, 0], grid[:, 1], shift[:, 0], shift[:, 1], angles='xy', scale_units='xy', scale=1)
            ax.set_aspect('equal')
            ax.set_title(f"Distortion in camera {self.name}")

    def _cam_fov(self):
        """
        Calculates the fov of the camera

        :return the fov
        """
        return 180 / m.pi * (2 * np.arctan2(self.res[1] / 2,
                                            self.intrinsic[0, 0]))

    def _calc_projection_matrix(self):
        """
        calcs the 3,4 projection matrix of a camera

        :return matrix: the projection matrix of the camera
        """
        return self.intrinsic @ self.extrinsic[:3, :4]

    def project_points(self, points, mode="opencv", distort=True):
        """
        Projects a list of points onto camera coordinates

        :param points: The points to project
        :param mode: by default, returns in opencv coordinates.
                    with mode == opencv
                    alternate method is mode == "image"
                    which returns v,u coordinates
        :return points: points in the uv coordinates
        """

        centered = h_tform(points, self.proj)
        if centered.ndim == 1:
            centered = centered[None, ...]
        if mode == "image":
            return centered[:, ::-1]
        if distort and not np.any(np.logical_not(np.isclose(self.distortion_coefs, np.zeros(5)))):
            distorted = [distort_points(
                pt,
                self.intrinsic,
                self.distortion_coefs
            ) for pt in centered]
            return np.array(distorted)
        return centered

    def _is_in_image(self, cords) -> bool:
        """
        A bounds check on image points

        :param cords: Some coordinate in UV, int
        :return view: BOOL indicating if point is on image array
        """
        truths = []
        for cord, res in zip(cords.squeeze(), self.res):
            truths.append(0 < cord < res)
            truths.append(0 < cord < res)
        return np.all(truths)

    def can_image(self, pt) -> bool:
        """
        Uses projection to work out if a camera images a point

        :param pt: a point in world space coordinates
        :return True if the camera images that world coordinate

        """
        uv = self.project_points(pt[None, ...])
        return self._is_in_image(uv)

    def get_mesh(self, scale=0.025):
        """
        Creates a pyvista mesh of the camera in world coordinates.

        :param scale: the display length of the camera in world coordinates.
        :return mesh: A PV mesh detailing the camera
        """
        cam_len = max(scale, 0.03)
        p1 = self.position

        pts = np.array([
            [0,             0],
            [0,             self.res[1]],
            [self.res[0],   0],
            [self.res[0],   self.res[1]],
        ])

        pt_100 = np.array([
            [0,     0],
            [100,   0],
            [0,     100],
            [100,   100],
        ])

        vs = vector_cam_points('linear', pts, self.intrinsic, self.cam_to_world)
        v100 = vector_cam_points('linear', pt_100, self.intrinsic, self.cam_to_world)

        [p2, p3, p4, p5] = vs * cam_len + p1
        [p6, p7, p8, p9] = v100 * cam_len + p1

        pn = p2 + (p2 - p3) / 3 + (p4 - p2) / 2

        verts = np.stack((p1, p2, p3, p4, p5, pn, p6, p7, p8, p9))
        faces = np.array([[3, 0, 1, 2],
                          [3, 0, 2, 4],
                          [3, 0, 4, 3],
                          [3, 0, 3, 1],
                          [3, 1, 3, 5],
                          [3, 6, 7, 8],
                          [3, 7, 8, 9]])

        return pv.PolyData(verts, faces)

    def get_viewcone(self, view_len=1, triangle=False):
        """
        Calculates a viewcone, indicating the region which projects to the viewed camera.

        :param view_len: the view length of the camera
        :param triangle: forces the mesh to only use triangular faces.
        :returns mesh: A PV mesh showing the camera object's viewcone
        """
        if triangle:
            p1 = self.position

            pts = np.array([
                [0, 0],
                [0, self.res[1]],
                [self.res[0], 0],
                [self.res[0], self.res[1]],
            ])

            vs = vector_cam_points('linear', pts, self.intrinsic, self.cam_to_world)

            [p6, p7, p8, p9] = vs * view_len + p1

            verts = np.stack((p6, p7, p8, p9, p1))

            faces = np.array([[3, 0, 1, 2],
                              [3, 2, 3, 0],
                              [3, 4, 1, 0],
                              [3, 4, 2, 1],
                              [3, 4, 3, 2],
                              [3, 4, 0, 3]])

            return pv.PolyData(verts, faces)
        else:
            cam_len = 0.025
            p1 = self.position

            pts = np.array([
                [0, 0],
                [0, self.res[1]],
                [self.res[0], 0],
                [self.res[0], self.res[1]],
            ])

            vs = vector_cam_points('linear', pts, self.intrinsic, self.cam_to_world)

            [p2, p3, p4, p5] = vs * cam_len + p1
            [p6, p7, p8, p9] = vs * view_len + p1

            verts = np.stack((p2, p3, p4, p5, p6, p7, p8, p9))

            faces = np.array([[3, 0, 1, 2],
                              [3, 2, 3, 0],
                              [3, 4, 5, 6],
                              [3, 6, 7, 4],
                              [3, 0, 1, 5],
                              [3, 5, 4, 0],
                              [3, 1, 2, 6],
                              [3, 6, 5, 1],
                              [3, 2, 3, 7],
                              [3, 7, 6, 2],
                              [3, 3, 0, 4],
                              [3, 4, 7, 3]])

            return pv.PolyData(verts, faces)

    def get_image_cord_sensor_map(self):
        """
        Returns a version of the sensor map in image coordinates, rather than the default opencv
        
        :returns sensor_map: a model of the intrinsic directions of the camera rays
        """
        return np.transpose(self.world_sensor_map, (1, 0, 2))

    def _update_state(self):
        """
        Recalculates all variables depending on the camera parameters after they have changed.
        """
        self.cam_to_world = np.linalg.inv(self.extrinsic)
        self.position = (self.cam_to_world @ [0, 0, 0, 1])[:3]
        self.view = (self.cam_to_world @ [0, 0, 1, 0])[
                    :3]  # might need to change this to -1 to reflect the standard

        self.u_axis = (self.cam_to_world @ [0, -1, 0, 0])[:3]  # should this
        # really be minus 1 here?
        if not self.minimal:
            self._make_sensormap()
        else:
            self.sensor_map = None
            self.world_sensor_map = None
        self.focal_point = self.position + self.intrinsic[
            0, 0] / 1000 * self.view  # focal length along principle axis in mm
        self.fov = self._cam_fov()
        self.proj = self._calc_projection_matrix()

    def _make_sensormap(self, mode='linear', distort=True):
        """
        Calculates a model of the ray direction of each pixel of the camera.

        :param mode: "linear" or "normalised".
            linear mode produces constant z components of each direction (good for depth maps)
            normalised produces constant length vectors (this is the blender convention as of writing).
        :param distort:
        :return:
        """
        self.sensor_map = sensor_map(
            mode,
            self.intrinsic,
            self.res,
            dist_coefs=self.distortion_coefs if distort else None)
        self.world_sensor_map = self._compute_world_sensor_map()

    def undistort(self, image: np.array) -> np.array:
        """
        Wraps the opencv undistort function with the camera parameters for convenience
        :param image: An input image
        :return An undistorted image

        """
        return cv2.undistort(image, self.intrinsic, np.array(self.distortion_coefs))

    def im_to_world_ray(self, cord: np.array or list, depth_im=None, distort=True):
        """
        Given an image coordinate in opencv cords, nx2, returns a normalised ray.
        If the depth image is given, uses the depth at the coordinate to set the
        length instead.

        :param cord: points to project
        :param depth_im: the depht image, if not none
        :param distort: whether the coordinate needs to be distorted before projection.
        :return dist
        """
        if isinstance(cord, list):
            cord = np.array(cord)
        if cord.ndim == 1:
            cord = cord[None, ...]


        self._make_sensormap(distort=distort)

        rays = self.world_sensor_map[cord[:, 0], cord[:, 1], :]

        if depth_im is not None:
            length = depth_im[cord[:, 1], cord[:, 0]]
            if np.any(np.isnan(length)):
                logging.warning('Nan length found in depth image used for ray')
            rays *= length[:, None]

        return rays + self.position

    def scale_self_2n(self, down_scale_factor=1):
        """
        Downscales the intrinsics of the camera

        :param down_scale_factor: the power of two by which to scale
        """
        self.down_scale_factor = down_scale_factor
        # need to change the camera calibration
        sf = float(-down_scale_factor)

        scale_mat = np.array(
            [[2.0 ** sf, 0.0, 2.0 ** (sf - 1.0) - 0.5],
             [0.0, 2.0 ** sf, 2.0 ** (sf - 1.0) - 0.5],
             [0.0, 0.0, 1.0]]
        )
        self.res = [
            int(self.res[0] * 2 ** sf),
            int(self.res[1] * 2 ** sf),
        ]
        self.intrinsic = scale_mat @ self.intrinsic
        self._update_state()

    def crop_to_roi(self, roi):
        """
        This function alters the intrinsics to mimic camera that takes
        a subset of the calibrated image size.

        :param roi: [xmin, xmax, ymin, ymax]
        """
        [ymin, xmin, xmax, ymax] = roi
        if xmax > self.res[0] or ymax > self.res[1]:
            raise ValueError('crop bounds outside of camera viewpoint')

        self.intrinsic -= [[0,0,xmin],
                           [0,0,ymin],
                           [0,0,0]]

        self._update_state()

    def reset_to_original_params(self):
        """
        Returns the camera to the original intrinsic matrix
        """
        self.intrinsic = self.original_matrix
        self._update_state()

    def transform(self, transformation_matrix):
        """
        Transforms the camera with a homogenous transformation

        :param transformation_matrix: a homogenous 4x4 transform
        """
        self.extrinsic = self.extrinsic @ transformation_matrix
        self._update_state()
