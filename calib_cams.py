from cv2 import aruco
from pathlib import Path
import numpy as np
from multiprocessing import cpu_count

from pyCamSet import calibrate_cameras, Ccube, load_CameraSet
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
from pyCamSet.calibration.camera_calibrator import run_bundle_adjustment

target = Ccube(n_points=12, length=80)
loc="bin/calib_images_process"
cams = calibrate_cameras(loc, target, 
                  draw=True,
                  )
param_handler = SelfBundleHandler(
    detection=cams.calibration_handler.detection, target=target, camset=cams,
    options={'max_nfev':300}
)
param_handler.set_from_templated_camset(cams)
op, final_cams = run_bundle_adjustment(
    param_handler=param_handler,
    threads = cpu_count(),
)

final_cams.visualise_calibration()
