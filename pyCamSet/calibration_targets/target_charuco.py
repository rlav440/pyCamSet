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
    def __init__(self, num_squares_x, num_squares_y, square_size):
        super().__init__(inputs=locals())

        # define checker and marker size
        square_size = square_size
        marker_size = 0.8 * square_size  # 80% of the square size
        # convert to meters
        squares_length = square_size / 1000
        marker_length = marker_size / 1000

        # Create the dictionary for the Charuco board
        self.a_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
        # Create the Charuco board
        self.board = cv2.aruco.CharucoBoard_create(num_squares_x, num_squares_y, squares_length, marker_length,
                                                   self.a_dict)
        self.point_data = self.board.chessboardCorners.squeeze().astype(np.float64)

        self._process_data()

    def find_in_image(self, image, draw=False, camera: Camera = None, wait_len=1) -> ImageDetection:
        params = aruco.DetectorParameters_create()
        params.minMarkerPerimeterRate = 0.01
        # params.adaptiveThreshConstant = 1 # for low light, but lowers accuracy

        corners, ids, rejected = aruco.detectMarkers(image, self.a_dict, parameters=params)
        draw_rejected = False

        if draw:
            if corners:
                im_idea = image.copy()
                target_size = [640, 480]
                d_f = int(min(np.array(im_idea.shape[:2]) / target_size))
                im_idea = downsample_valid(im_idea, d_f).astype(np.uint8)
                if im_idea.ndim == 2:
                    im_idea = np.tile(im_idea[..., None], (1, 1, 3))

                # aruco.drawDetectedMarkers(im_idea, np.array(corners)/d_f, ids)
                if draw_rejected:
                    for point in rejected:
                        tmp = point.squeeze()
                        plt.plot(tmp[:, 0], tmp[:, 1], 'y')

        use_cam = camera is not None
        if len(corners) == 0:
            return ImageDetection() # return an empty detection

        n, c_corners, c_ids = aruco.interpolateCornersCharuco(
            corners,
            ids,
            image,
            self.board,
            camera.intrinsic if use_cam else None,
            camera.distortion_coefs if use_cam else None,
        )
        if n > 0:
            if draw:
                aruco.drawDetectedCornersCharuco(
                    im_idea,
                    np.array(c_corners) / d_f,
                    c_ids,
                )

            seen_points = ImageDetection(c_ids[:, 0], c_corners[:, 0])
            if draw:
                if corners:
                    cv2.imshow('detections', im_idea)
                    cv2.waitKey(1)

            return seen_points
        return ImageDetection() # return an empty detection
