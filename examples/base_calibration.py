from multiprocessing import cpu_count

import numpy as np
import pytest
from cv2 import aruco

from pyCamSet import Ccube, calibrate_cameras
from pyCamSet.calibration.camera_calibrator import run_bundle_adjustment
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler

if __name__ == "__main__":

    target = Ccube(
        n_points=10, length=40, aruco_dict=aruco.DICT_6X6_1000, border_fraction=0.2
    )

    cams = calibrate_cameras("tests/test_data/calibration_ccube", target, save=False)
    print()
    print("Beginning the full self bundler")
    param_handler = SelfBundleHandler(
        detection=cams.calibration_handler.detection,
        target=target,
        camset=cams,
        options={"max_nfev": 1000},
    )
    param_handler.set_from_templated_camset(cams)

    _, final_cams = run_bundle_adjustment(
        param_handler=param_handler, threads=cpu_count()
    )

    mean_reprojection = np.mean(
        np.linalg.norm(np.reshape(final_cams.calibration_result, (-1, 2)), axis=1)
    )

