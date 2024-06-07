from pathlib import Path
from cv2 import aruco
from multiprocessing import cpu_count

from pyCamSet import Ccube, load_CameraSet
from pyCamSet.utils.general_utils import benchmark

from pyCamSet.optimisation.template_handler import TemplateBundleHandler
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment, make_optimisation_function

# load scripts that represent all of the calibration examples
# then, drag race the functions!

target = Ccube(n_points=10, length=40,  aruco_dict=aruco.DICT_6X6_1000, border_fraction=0.2)
loc=Path("tests/test_data/calibration_ccube")
test_cams = loc/'self_calib_test.camset'

# target = Ccube(n_points=4, length=400,  aruco_dict=aruco.DICT_6X6_1000, border_fraction=0.1)
# test_cams = Path("/home/rlav440/Projects/breast_imaging/bin/Alex_hbi/calib/ok_cams.camset")

cams = load_CameraSet(test_cams)

param_handler = SelfBundleHandler(
    detection=cams.calibration_handler.detection, target=target, camset=cams,
    options={'max_nfev':100}
)

param_handler.set_from_templated_camset(cams)

loss, jac, init = make_optimisation_function(
    param_handler=param_handler,
    threads = cpu_count(),
)
print("initialising self calib")
loss(init)
jac(init)

print("Testing self calib loss")
benchmark(lambda :loss(init), repeats=100, mode='ms')

print("Testing self calib jac")
benchmark(lambda :jac(init), repeats=100, mode='ms')


param_handler: TemplateBundleHandler = cams.calibration_handler
param_handler.set_initial_params(cams.calibration_params)

loss, jac, init = make_optimisation_function(
    param_handler=param_handler,
    threads = cpu_count(),
)

print("initialising templated calib")
loss(init)
jac(init)

print("Testing templated loss")
benchmark(lambda :loss(init), repeats=100, mode='ms')

print("Testing templated jac")
benchmark(lambda :jac(init), repeats=100, mode='ms')
