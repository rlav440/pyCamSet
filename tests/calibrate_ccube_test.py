from cv2 import aruco
import numpy as np

from pyCamSet import calibrate_cameras, Ccube, load_CameraSet

def test_calib_ccube():
    target = Ccube(n_points=10, length=40,  aruco_dict=aruco.DICT_6X6_1000, border_fraction=0.2)
    loc="tests/test_data/calibration_ccube"
    cams = calibrate_cameras(loc, target, 
                      # draw=True,
                      # save=False
                      )

    final_euclid = np.mean(np.linalg.norm(np.reshape(
        cams.calibration_result, (-1, 2)
    ), axis=1))
    assert final_euclid < 3, "The calibration accuracy did not pass for ccube targets" 

if __name__ == "__main__":
    test_calib_ccube()
