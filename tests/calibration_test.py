from pathlib import Path
from matplotlib import pyplot as plt
import cv2
import numpy as np

from pyCamSet import calibrate_cameras, ChArUco

# we want to load some n camera calibration model

def test_calibration_charuco():
    data_loc = Path("./tests/test_data/calibration_charuco")
    target = ChArUco(20, 20, 4)

    cams = calibrate_cameras(f_loc=data_loc, calibration_target=target, 
                             # draw=True,
                             save=False,
                             )
    cams.visualise_calibration()
    
    final_euclid = np.mean(np.linalg.norm(np.reshape(
        cams.calibration_result, (-1, 2)
    ), axis=1))
    assert final_euclid < 1.3, "The calibration accuracy did not pass for ChArUco targets" 

if __name__ == '__main__':
    test_calibration_charuco()
    
