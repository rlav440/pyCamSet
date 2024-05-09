from petsc4py.PETSc import TAO
import petsc4py.PETSc as pc

from pyCamSet import load_CameraSet
from pyCamSet.calibration_targets.target_Ccube import Ccube
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler

cams = load_CameraSet("tests/test_data/calibration_ccube/self_calib_test.camset")






#get the param handler
detections = cams.calibration_handler.detection
target = Ccube()
param_handler = SelfBundleHandler(camset=cams, target=target, detection=detections)
param_handler.set_from_templated_camset(cams)

#recalibrate!





#get both functions from the param handlers.
loss_fn = param_handler.make_loss_fun(threads=16)
jaco_fn = param_handler.make_loss_jac(threads=16)
init_params = param_handler.get_initial_params()

def sparse_petsc_jac(x):
    scipy_sparse = jaco_fn(x)
    mat = pc.Mat()
    mat.convert_AIJ(scipy_sparse)
    return mat

solver = TAO
solver.setJacobianResidual(jaco_fn)
solver.setResidual(loss_fn)
solver.setInitial(init_params)

new_params = init_params.copy()
solver.solve(new_params)


