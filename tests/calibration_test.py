from pathlib import Path
from matplotlib import pyplot as plt
import cv2

from pyCamSet import load_CameraSet, calibrate_cameras, ChArUco

# we want to load some n camera calibration model

def test_calibration():
    data_loc = Path("./tests/test_data/calibration_charuco")
    ref_loc = Path("")
    target = ChArUco(20, 20, 4)

    camera_model = calibrate_cameras(f_loc=data_loc, calibration_target=target, draw=True)
    camera_model.visualise_calibration()
    reference_cams = None #load_CameraSet(ref_loc)
    assert camera_model == reference_cams


if __name__ == '__main__':
    test_calibration()
    
