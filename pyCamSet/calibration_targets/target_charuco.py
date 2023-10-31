from __future__ import annotations

import cv2
import numpy as np
from cv2 import aruco
from matplotlib import pyplot as plt

from pyCamSet.calibration_targets.abstractTarget import AbstractTarget
from pyCamSet.calibration_targets.targetDetections import ImageDetection
from pyCamSet.cameras import Camera
from pyCamSet.utils.general_utils import downsample_valid


class ChArUco(AbstractTarget):
    def __init__(self, num_squares_x, num_squares_y, square_size, marker_fraction = 0.8, a_dict=cv2.aruco.DICT_4X4_1000):
        super().__init__(inputs=locals())

        # define checker and marker size
        square_size = square_size
        marker_size = marker_fraction * square_size  # 80% of the square size
        # convert to meters
        squares_length = square_size / 1000
        marker_length = marker_size / 1000

        # Create the dictionary for the Charuco board
        self.a_dict = cv2.aruco.Dictionary_get(a_dict)
        # Create the Charuco board
        self.board = cv2.aruco.CharucoBoard_create(num_squares_x, num_squares_y, squares_length, marker_length,
                                                   self.a_dict)
        self.point_data = self.board.chessboardCorners.squeeze().astype(np.float64)

        self._process_data()

    def find_in_image(self, image, draw=False, camera: Camera = None, wait_len=1) -> ImageDetection:
        params = aruco.DetectorParameters_create()
        params.minMarkerPerimeterRate = 0.01
        corners, ids, _ = aruco.detectMarkers(image, self.a_dict, parameters=params)

        if len(corners) == 0:
            return ImageDetection() # return an empty detection

        if draw:
            display_im = image.copy()
            target_size = [1080, 1920]
            d_f = int(min(np.array(display_im.shape[:2]) / target_size))
            display_im = downsample_valid(display_im, d_f).astype(np.uint8)
            # d_f=1
            if display_im.ndim == 2:
                display_im = np.tile(display_im[..., None], (1, 1, 3))

            # aruco.drawDetectedMarkers(display_im, np.array(corners)/d_f, ids)
        use_cam = camera is not None

        n, c_corners, c_ids = aruco.interpolateCornersCharuco(
            corners,
            ids,
            image,
            self.board,
            camera.intrinsic if use_cam else None,
            camera.distortion_coefs if use_cam else None,
        )
        if n == 0:
            return ImageDetection()

        if draw:
            aruco.drawDetectedCornersCharuco(
                display_im,
                np.array(c_corners) / d_f,
            )

            cv2.imshow('detections', display_im)
            cv2.waitKey(wait_len)

        return ImageDetection(c_ids[:, 0], c_corners[:, 0])
        
    def plot(self,imres=(1000,1000)):                           
        plt.imshow(self.board.draw(imres))
        plt.show()  
