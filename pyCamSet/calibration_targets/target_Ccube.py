from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
from cv2 import aruco
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from pyCamSet.calibration_targets import AbstractTarget, ImageDetection, FaceToShape
from pyCamSet.cameras import Camera
from pyCamSet.utils.general_utils import split_aruco_dictionary, make_4x4h_tform, downsample_valid

# TFORMS = [
# 	([-1.209,-1.209, 1.209],[ 0.5,-0.5, 0.5]),
# 	([ 1.209,-1.209, 1.209],[ 0.5, 0.5,-0.5]),
# 	([0.   ,0.   ,1.571],[ 0.5,-0.5,-0.5]),
# 	([2.221,0.   ,2.221],[-0.5, 0.5,-0.5]),
# 	([3.142,0.   ,0.   ],[-0.5, 0.5, 0.5]),
# 	([-1.571, 0.   , 0.   ],[-0.5,-0.5, 0.5]),
# ]

TFORMS = [
    (([2.22144147, 2.22144147, 0.        ]), ([-0.5, -0.5,  0.5])),
    (([-1.57079633,  0.        ,  0.        ]), ([-0.5, -0.5,  0.5])),
    (([-1.20919958, -1.20919958,  1.20919958]), ([ 0.5, -0.5,  0.5])),
    (([ 0.        ,  2.22144147, -2.22144147]), ([0.5, 0.5, 0.5])),
    (([0.        , 0.        , 1.57079633]), ([ 0.5, -0.5, -0.5])),
    (([1.20919958, 1.20919958, 1.20919958]), ([-0.5, -0.5, -0.5])),
]

NET_FORMS=[
	[[1.0,0.0,0.0], [0.0,1.0,0.0],[0.0,0.0,1.0]],
	[[0.0,1.0,0.0], [-1.0,0.0,0.0],[0.0,0.0,1.0]],
	[[1.0,0.0,1.0], [0.0,1.0,0.0],[0.0,0.0,1.0]],
	[[0.0,-1.0,1.0], [1.0,0.0,1.0],[0.0,0.0,1.0]],
	[[1.0,0.0,2.0], [0.0,1.0,0.0],[0.0,0.0,1.0]],
	[[1.0,0.0,-1.0], [0.0,1.0,0.0],[0.0,0.0,1.0]],
]

def make_blank_square(draw_res, line_fraction, border_fraction):
    """
    This function makes a blank face of a square, with surrounding edge lines set to black.
    As convinience, it also returns the array offset used for the border fraction.
    
    :param draw_res: the drawing resolution of the cube face
    :param line_fraction: the thickness of the line as a fraction of the face width
    :param border_fraction: the size of the fraction to use as the border of the cube.
    """
    canvas = np.ones(draw_res)*255
    int_line = int(draw_res[0] * line_fraction)
    canvas[:, :int_line] = 0
    canvas[:int_line, :] = 0
    canvas[:, -int_line:] = 0
    canvas[-int_line:, :] = 0
    return canvas, int(border_fraction * draw_res[0]/2)

class Ccube(AbstractTarget):
    """
    This class defines a calibration target that consists of a Cube of ChArUco boards.
    """

    def __init__(self, length=20, n_points=5,
                 aruco_dict=aruco.DICT_4X4_1000,
                 draw_res=(1000, 1000),
                 border_fraction=0.1,
                 line_fraction=0.003,
                 ):
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

        self.boards = [
            aruco.CharucoBoard(
                (n_points, n_points), self.square_size,
                markerLength=0.75 * self.square_size,
                dictionary=a_dict
            )
            for a_dict in self.a_dicts][:6] #only need 6 of them!
        [b.setLegacyPattern(True) for b in self.boards]

        self.n_points = n_points
        self.draw_res = draw_res
        self.dpi = self.draw_res[0] / self.length / 39.3701  # inch conversion
        blank_face, board_offset = make_blank_square(draw_res, line_fraction, border_fraction)
        sub_res = (draw_res[0] - 2 * board_offset, draw_res[1] - 2*board_offset)
        self.textures = [blank_face.copy() for _ in range(6)]
        # debug_t = []
        for idb, (t, board) in enumerate(zip(self.textures, self.boards)):
            t[board_offset:-board_offset, board_offset:-board_offset] = board.generateImage(sub_res)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = int(t.shape[0]/500)
            cv2.putText(t, f"{idb}", (t.shape[0]//100, t.shape[0]//100 * 99 ), font, font_scale, 0, thickness)

            # debug_t.append(board.draw(draw_res)) #DEBUG
        # self.textures = debug_t

        bd = np.array([board.getChessboardCorners() for board in self.boards])
        coord_bump = self.length*border_fraction/2
        board_coords = bd + np.array([coord_bump, coord_bump, 0])
        # board_coords = bd #DEBUG
        self.base_face = np.array([
                            [0, self.length,0],
                            [self.length, self.length,0],
                            [self.length, 0,0],
                            [0, 0,0],
                        ])

        # breakpoint()
        self.faceData = FaceToShape(
            face_local_coords=board_coords,
            face_transforms=[make_4x4h_tform(*t) for t in TFORMS],
            scale_factor= self.length,
        )
        self.point_data = self.faceData.point_data
        self._process_data()

        self.board_detectors = None

    def plot(self, return_scene = False):
        """
        Draws a 3D model of the calibration target using pyVista

        """
        scene = self.faceData.draw_meshes(self.base_face, self.textures, return_scene=return_scene)
        if return_scene:
            return scene

    def save_to_pdf(
            self,
            f_out: Path | None = None,
            border_width: float = 10,
            individual_faces = False,

    ):
        if individual_faces:
            for idf, face in enumerate(tqdm(self.textures)):
                blank_f = int(border_width * 0.0393701 * self.dpi)
                dims = np.array(face.shape) + blank_f * 2
                full_im = np.ones((dims)) * 255
                full_im[blank_f:-blank_f, blank_f:-blank_f] = face

                if f_out is None:
                    f_out = Path(f'Ccube_length_{self.length * 1000:.2f}mm' \
                            f'_{self.n_points}_points_at' \
                            f'_{self.square_size * 1000:.2f}mm_face_{idf}.png')
                full_im = full_im.astype(np.uint8)
                with Image.fromarray(full_im) as im:
                    im.save(fp=f_out, resolution=self.dpi)
                f_out = None
            return

        # raise NotImplementedError
        im_board = self.faceData.draw_net(self.textures, NET_FORMS)

        blank_f = int(border_width * 0.0393701 * self.dpi)
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

    def find_in_image(self, image, draw=False, camera: Camera|None = None, wait_len=1) -> ImageDetection:
        """
        An implementation of the find in image function for

        :param image: the image
        :param draw: whether to draw the visualisation of the cube,
        :param camera: an optional camera model, for more accurate detections.
        :param wait_len: the wait time for the visualisation. -1 waits for user keypress.

        Returns:

        """

        # params = aruco.DetectorParameters()
        params = aruco.CharucoParameters()
        params.tryRefineMarkers = True
        # params.minMarkerPerimeterRate = 0.01
        #params.adaptiveThreshConstant = 1 # for low light, but lowers accuracy

        if self.board_detectors is None:
            self.board_detectors = [aruco.CharucoDetector(b, params) for b in self.boards]


        if draw:
            im_idea = image.copy()
            target_size = [640, 480]
            d_f = int(min(np.array(im_idea.shape[:2])/target_size))
            im_idea = downsample_valid(im_idea, d_f).astype(np.uint8)
            if im_idea.ndim == 2:
                im_idea = np.tile(im_idea[..., None], (1, 1, 3))


        seen_keys = []
        seen_data = []
        for idb, bd in enumerate(self.board_detectors):
            c_corners, c_ids, _, _ =  bd.detectBoard(image)
            if c_ids is not None:
                for cid, corner in zip(c_ids[:, 0], c_corners[:, 0, :]):
                    seen_keys.append([idb, cid])
                    seen_data.append(corner)
                if draw:
                    aruco.drawDetectedCornersCharuco(im_idea, c_corners/d_f, c_ids/d_f)

        if draw:
            cv2.imshow('detections', im_idea)
            cv2.waitKey(wait_len)

        return ImageDetection(keys=seen_keys, image_points=seen_data)



if __name__ == '__main__':
    test = Ccube(n_points=7, length=4)
    test.plot()
    # test.get_printable_texture()
