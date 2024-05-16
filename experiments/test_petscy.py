from petsc4py.PETSc import TAO
import petsc4py.PETSc as PETSc
import petsc4py
import petsc

import sys
petsc4py.init(sys.argv)

import numpy as np
from cv2 import aruco
from scipy.sparse import csr_array, linalg
from scipy.optimize import least_squares, minimize, approx_fprime
from matplotlib import pyplot as plt

from pyCamSet import load_CameraSet
from pyCamSet.calibration_targets.target_Ccube import Ccube
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
from pyCamSet.calibration_targets.target_detections import TargetDetection

from pyCamSet.optimisation.function_block_implementations import projection, rigidTform3d, free_point

cams = load_CameraSet("tests/test_data/calibration_ccube/self_calib_test.camset")

#get the param handler
detections: TargetDetection = cams.calibration_handler.detection 
# d_data = detections.get_data()[400:800, :]
# detections = TargetDetection(data=d_data, max_ims=detections.max_ims, cam_names=detections.cam_names) 
target = Ccube(n_points=10, length=40,  aruco_dict=aruco.DICT_6X6_1000, border_fraction=0.2)
param_handler = SelfBundleHandler(camset=cams, target=target, detection=detections)
res_out_size = 2 * param_handler.get_detection_data().shape[0]


param_handler.set_from_templated_camset(cams)

#recalibrate!

#get both functions from the param handlers.
loss_fn = param_handler.make_loss_fun(threads=16)
jaco_fn = param_handler.make_loss_jac(threads=16)
init_params = param_handler.get_initial_params()

#
# jac = (jaco_fn(init_params))[:200, :].todense()
# print(jac.shape)
#
# num_jac = approx_fprime(init_params, loss_fn)[:200, :]
# errs = jac - num_jac
# rel_errs = (errs)/(jac + 1e-8)
# print(np.argmax(rel_errs[:200]))


# def s(n, o, ax):
#     ax[o].imshow(n)
#     # l = n.shape[1]//3
#     # ax[o + 0].imshow(n[:, :l])
#     # ax[o + 1].imshow(n[:, l:2*l])
#     # ax[o + 2].imshow(n[:, 2*l:3*l])
#
# fig, axs = plt.subplots(2,2)
# ax = axs.flatten()
# s(jac, 0, ax)
# s(num_jac, 1, ax)
# s(np.log(np.abs(errs) + 1e-50), 2, ax)
# s((np.abs(errs/(jac + 1e-8)) + 1e-50), 3, ax)
# plt.show()

jac_size = (res_out_size, init_params.shape[0])

def sparse_petsc_jac(TAO, x, mat, _mat):
    # print("ran jac")
    intermediate = jaco_fn(x)
    print(intermediate.shape)
    vals = linalg.norm(intermediate, axis=0)
    print(vals.shape)
    intermediate /= vals[None, :]         
    mat.setValuesCSR(I=intermediate.indptr.astype("int32"), J=intermediate.indices.astype("int32"), V=intermediate.data, addv=None)
    mat.assemble()

def petsc_residual(TAO, x, res):
    loss = loss_fn(x.getArray())
    res[:] = loss

def petsc_objective(TAO, x):
    return np.sum(loss_fn(x.getArray(readonly=True))**2)

def scipy_objective(x):
    return np.sum(loss_fn(x)**2)


# optim = minimize(scipy_objective, init_params, method='Nelder-Mead', options={"maxiter":2000, "disp":True})
# optim2 = least_squares(loss_fn, init_params, max_nfev=10, verbose=2)
#
# print(f"found a final error of {np.mean(np.abs(loss_fn(optim2.x)))}")
# optim2 = least_squares(loss_fn, optim2.x, jac=jaco_fn, max_nfev=10, verbose=2)
# print(f"found a final error of {np.mean(np.abs(loss_fn(optim2.x)))}")

# options = PETSc.Options()
# options.setValue("-tao_monitor", "")
# options.setValue("-tao_view", "")
# options.setValue("-tao_brgn_regularization_type", "lm")
# options.setFromOptions()

# options = pc.Options()
# options.create()
# options['tao_brgn_regularization_type'] = 'lm'
# options['tao_monitor'] = 1
# print(options.getAll())
intermediate = jaco_fn(init_params)
J = PETSc.Mat().createAIJ(size=jac_size, csr=(intermediate.indptr.astype("int32"), intermediate.indices.astype("int32"), intermediate.data))
residual = PETSc.Vec().createWithArray(np.zeros(res_out_size))
result_vec = PETSc.Vec().createWithArray(init_params)
solver = TAO().create(PETSc.COMM_WORLD)
# solver.setOptionsPrefix("tao_")
# pc.PETScOptions().set('tao_brgn_regularization_type', 'lm')

print(f"found an initial error of {np.mean(np.abs(loss_fn(result_vec.getArray())))}")
# solver.setType(TAO.Type.NM)
# solver.setType(TAO.Type.POUNDERS)
# solver.setMaximumFunctionEvaluations(1e6)
# solver.setMaximumIterations(1e6)
# solver.setObjective(petsc_objective)
solver.setResidual(petsc_residual, R=residual)
solver.setJacobianResidual(sparse_petsc_jac, J=J)
solver.setSolution(result_vec)
# solver.computeResidual(init, residual)
# solver.setUp()
solver.setType(TAO.Type.BRGN)
solver.setFromOptions()

solver.solve()

solver.view()


print(f"found a residual error of {np.mean(np.abs(loss_fn(solver.getSolution().getArray())))}")


print("I finished")
