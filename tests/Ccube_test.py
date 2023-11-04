from pyCamSet import Ccube
import numpy as np
from pyCamSet.calibration_targets.shape_by_faces import make_tforms
import cv2

def test_cube():

    test = Ccube(length=40, n_points=10, aruco_dict=cv2.aruco.DICT_6X6_1000)
    test.plot()

if __name__ == "__main__":
    
    test_cube()
      
