from __future__ import annotations

import cv2
import numpy as np
from cv2 import aruco
from matplotlib import pyplot as plt
import logging

from pyCamSet.calibration_targets.abstract_target import AbstractTarget
from pyCamSet.calibration_targets.target_detections import ImageDetection
from pyCamSet.cameras import Camera
from pyCamSet.utils.general_utils import downsample_valid, adaptive_decimated_charuco_detection_stereo


class ChArUco(AbstractTarget):
    def __init__(self, num_squares_x, num_squares_y, square_size, marker_fraction = 0.8, a_dict=cv2.aruco.DICT_4X4_1000, legacy=False):
        """
        Initialises a ChArUco board in mm.

        :param num_squares_x: number of squares in the x direction
        :param num_squares_y: number of squares in the y direction
        :param square_size: the size of a square in mm
        :param marker_fraction: the percentage of a chessboard square occupied by a marker
        :param a_dict: the aruco dictionairy to use.
        """
        super().__init__(inputs=locals())

        # define checker and marker size
        square_size = square_size
        self.square_size=square_size
        marker_size = marker_fraction * square_size  # 80% of the square size
        # convert to meters
        squares_length = square_size / 1000
        marker_length = marker_size / 1000

        # Create the dictionary for the Charuco board
        self.a_dict = cv2.aruco.getPredefinedDictionary(a_dict)
        # Create the Charuco board
        self.board = cv2.aruco.CharucoBoard((num_squares_x, num_squares_y), squares_length, marker_length, self.a_dict)
        if legacy:
            self.board.setLegacyPattern(True)
        self.point_data = self.board.getChessboardCorners().squeeze().astype(np.float64)

        self.detection_params = aruco.CharucoParameters()
        self.detection_params.tryRefineMarkers = True
        # params.minMarkerPerimeterRate = 0.01
        #params.adaptiveThreshConstant = 1 # for low light, but lowers accuracy
        self.board_detectors = aruco.CharucoDetector(self.board, self.detection_params)
        self.given_legacy_warning = False

        self._process_data()


    def find_in_image(self, image, draw=False, camera: Camera|None = None, wait_len=1) -> ImageDetection:
        """
        Detects features of this target in the input image.

        :param image: The image to detect in.
        :param draw: Whether or not the detected corners should be drawn.
        :param camera: optional. A camera target for more accurate detection.
        :param wait_len: time to pause to allow drawing of detections. -1 waits for key press.
        
        :return ImageDetection: a data class wrapping the data detected in the image.
        """
        # c_corners, c_ids, od = adaptive_decimated_charuco_detection_stereo(image, charuco_board=self.board, aruco_dict=self.a_dict)
        # _, _, mloc, mid = self.board_detectors.detectBoard(image)
        c_corners, c_ids, mloc, mid = self.board_detectors.detectBoard(image) #, markerCorners=mloc, markerIds=mid)
        if c_corners is None and mloc is not None:
            if not self.given_legacy_warning:
                logging.warning("Found markers, but no corners, trying using alternative board detection")
                self.given_legacy_warning = True
            am_legacy = self.board.getLegacyPattern()
            self.board.setLegacyPattern(not am_legacy)
            c_corners, c_ids, mloc, mid = self.board_detectors.detectBoard(image, markerCorners=mloc, markerIds=mid)

        od = 1
        c_corners
        # breakpoint()

        if c_corners is None:
            return ImageDetection() # return an empty detection

            # aruco.drawDetectedMarkers(display_im, np.array(corners)/d_f, ids)

        if draw:           
            display_im = image.copy()
            target_size = [480, 640]
            d_f = int(max((min(np.array(display_im.shape[:2]) / target_size)), 1))
            display_im = downsample_valid(display_im, d_f).astype(np.uint8)
            # d_f=1
            if display_im.ndim == 2:
                display_im = np.tile(display_im[..., None], (1, 1, 3))
            aruco.drawDetectedCornersCharuco(
                display_im,
                np.array(c_corners) / d_f,
                c_ids,
            )

            cv2.imshow('detections', display_im)
            cv2.waitKey(wait_len)


        return ImageDetection(c_ids[:, 0], c_corners[:, 0])
        
    def plot(self,imres=(1000,1000)):                           
        """
        Draws the target as a matplotlib plot.
        """
        plt.imshow(self.board.generateImage(imres), cmap='gray')
        plt.show()  
