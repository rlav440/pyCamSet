from pathlib import Path
from matplotlib import pyplot as plt
import cv2
import numpy as np

from pyCamSet import calibrate_cameras, ChArUco, load_CameraSet
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
from pyCamSet.calibration.camera_calibrator import run_stereo_calibration, run_bundle_adjustment

# we want to load some n camera calibration model

def test_self_calibration_charuco():
    data_loc = Path("./tests/test_data/calibration_charuco")
    target = ChArUco(20, 20, 4)
    

    cams = calibrate_cameras(f_loc=data_loc, calibration_target=target, 
                             # draw=True,
                             # save=False,
                             )
    param_handler = SelfBundleHandler(
        detection=cams.calibration_handler.detection, target=target, camset=cams,
    )

    param_handler.set_from_templated_camset(cams)
    # final_cams = run_stereo_calibration(
    _, final_cams = run_bundle_adjustment(
        param_handler=param_handler,
        threads = 16,
    )
    final_euclid = np.mean(np.linalg.norm(np.reshape(
        final_cams.calibration_result, (-1, 2)
    ), axis=1)) 
    assert (final_euclid < 1.07), "regression found in self calibration using a charuco board"


if __name__ == '__main__':
    test_self_calibration_charuco()
    
