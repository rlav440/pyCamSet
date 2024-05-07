from cv2 import aruco
from pathlib import Path
import numpy as np

from pyCamSet import calibrate_cameras, Ccube, load_CameraSet
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
from pyCamSet.calibration.camera_calibrator import run_bundle_adjustment

def test_calib_ccube():
    target = Ccube(n_points=10, length=40,  aruco_dict=aruco.DICT_6X6_1000, border_fraction=0.2)
    loc=Path("tests/test_data/calibration_ccube")
    test_cams = loc/'self_calib_test.camset'
    debug = True
    if test_cams.exists() and debug:
        cams = load_CameraSet(test_cams)
    else:
        cams = calibrate_cameras(loc, target, 
                          # draw=True,
                          # save=False,
                          )
        cams.save(test_cams)
    param_handler = SelfBundleHandler(
        detection=cams.calibration_handler.detection, target=target, camset=cams,
        options={'max_nfev':100}
    )
    param_handler.set_from_templated_camset(cams)
    op, final_cams = run_bundle_adjustment(
        param_handler=param_handler,
        threads = 16,
    )

    # final_cams.visualise_calibration()
    final_euclid = np.mean(np.linalg.norm(np.reshape(
        final_cams.calibration_result, (-1, 2)
    ), axis=1))
    assert (final_euclid < 0.50), "regression found in self calibration using a ccube"

if __name__ == "__main__":
    test_calib_ccube()
