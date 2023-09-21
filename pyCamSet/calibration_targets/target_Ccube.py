from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyvista as pv
from cv2 import aruco
import cv2
from PIL import Image

from pyCamSet.calibration_targets import AbstractTarget, ImageDetection, FaceToShape
from pyCamSet.cameras import Camera
from pyCamSet.utils.general_utils import split_aruco_dictionary, e_4x4, downsample_valid


class Ccube(AbstractTarget):
    """
    Notes: This class defines a calibration that is the result of multiple
    """

    def __init__(self, length=20, n_points=5,
                 aruco_dict=aruco.DICT_6X6_250,
                 draw_res=(1000, 1000),
                 border_fraction=0.2,
                 line_fraction=0.01,
                 ):
        """


        """
        super().__init__(inputs=locals())
        self.input_border_fraction = border_fraction
        self.actual_border_fraction = None
        self.line_fraction = line_fraction
        self.aruco_dict = aruco_dict
        self.length = length/1000
        self.square_size = self.length * (1 - border_fraction) / n_points
        if n_points % 2 == 0:
            split = int(n_points ** 2 / 2)
        else:
            split = int((n_points - 1) * (n_points + 1) / 2)
        self.markers_per_face = split
        self.a_dicts = split_aruco_dictionary(split, self.aruco_dict)
        if len(self.a_dicts) < 6:
            raise ValueError("Input dictionary of marker didn't contain enough "
                             "markers for this cube")

        self.boards = [aruco.CharucoBoard_create(n_points, n_points, self.square_size,
                                                 markerLength=0.75 * self.square_size, dictionary=a_dict)
                       for a_dict in self.a_dicts()[:6]]

        self.n_points = n_points
        self.draw_res = draw_res
        self.dpi = self.draw_res[0] / self.length / 39.3701  # inch conversion
        self.textures = [board.draw(draw_res) for board in self.boards]

        self.faceData = FaceToShape(
            face_local_coords=[board.ChessboardCorners for board in self.boards],
            face_transforms=[
                e_4x4(), e_4x4(), e_4x4(),
                e_4x4(), e_4x4(), e_4x4()
            ]
        )
        self.point_data = self.faceData.point_data
        self._process_data()

    def plot(self):
        scene = pv.Scene()
        faces = self.faceData.draw_meshes(self.textures, BLANK)
        for face, im in faces:
            scene.add_mesh(face, texture=im)

    def save_to_pdf(
            self,
            f_out: Path = None,
            border_width: float = 10,
    ):

        im_board = self.texture

        blank_f = np.int(border_width * 0.0393701 * self.dpi)
        dims = np.array(im_board.shape) + blank_f * 2
        full_im = np.ones((dims)) * 255
        full_im[blank_f:-blank_f, blank_f:-blank_f] = im_board

        if f_out is None:
            f_out = f'Ccube_length_{self.length * 1000:.2f}mm' \
                    f'_{self.n_points}_points_at' \
                    f'_{self.square_size * 1000:.2f}mm.pdf'
        full_im = full_im.astype(np.uint8)
        with Image.fromarray(full_im) as im:
            im.save(fp=f_out, resolution=self.dpi)

    def find_in_image(self, image, draw=False, camera: Camera = None, wait_len=1) -> ImageDetection:
        """
        An implementation of the find in image function for
        :param image: the iamge
        :param self: a charuco cube that has been made
        :param draw:

        Returns:

        """

        params = aruco.DetectorParameters_create()
        params.minMarkerPerimeterRate = 0.01
        #params.adaptiveThreshConstant = 1 # for low light, but lowers accuracy
        a_dict = self.aruco_dict() if callable(self.aruco_dict) else aruco.getPredefinedDictionary(self.aruco_dict)
        corners, ids, rejected = aruco.detectMarkers(image, a_dict, parameters=params)

        if draw:
            if corners:
                im_idea = image.copy()
                target_size = [640, 480]
                d_f = int(min(np.array(im_idea.shape[:2])/target_size))
                im_idea = downsample_valid(im_idea, d_f).astype(np.uint8)
                if im_idea.ndim == 2:
                    im_idea = np.tile(im_idea[..., None], (1, 1, 3))

        seen_keys = []
        seen_data = []

        if ids is not None:
            # then split the detections by floor div
            board_marker_id = ids % self.markers_per_face
            board_origin = np.floor(np.array(ids) / self.markers_per_face).astype(np.int)
            seen_boards = np.unique(board_origin).astype(np.int)

            if np.any(seen_boards >= 6):
                logging.warning(
                    "A marker was detected with an unfeasibly high board number."
                )

            for n_board in seen_boards[seen_boards < 6]:
                board = self.boards[n_board]
                input_index, _ = np.where(board_origin == n_board)

                board_corners = np.array([corners[ind] for ind in input_index])
                board_ids = np.array([board_marker_id[ind] for ind in input_index])
                use_cam = camera is not None

                scale = np.eye(3)
                n, c_corners, c_ids = aruco.interpolateCornersCharuco(
                    board_corners,
                    board_ids,
                    image,
                    board,
                    scale @ camera.intrinsic if use_cam else None,
                    camera.distortion_coefs if use_cam else None,
                )

                if n > 0:
                    if draw:
                        pass
                        aruco.drawDetectedCornersCharuco(im_idea,
                                        np.array(c_corners)/d_f, c_ids)

                    for cid, corner in zip(c_ids[:, 0], c_corners[:, 0, :]):
                        seen_keys.append([n_board, cid])
                        seen_data.append(corner)

        if draw:
            if corners:
                cv2.imshow('detections', im_idea)
                cv2.waitKey(wait_len)

        return ImageDetection(keys=seen_keys, image_points=seen_data)



if __name__ == '__main__':
    test = Ccube(n_points=7, length=4)
    test.plot()
    # test.get_printable_texture()
