import numpy as np
from matplotlib import pyplot as plt

from pyCamSet.utils.general_utils import benchmark
import pyCamSet.optimisation.function_block_implementations as fb
import pyCamSet.optimisation.abstract_function_blocks as afb

import numba


op_fun: afb.optimisation_function = fb.projection() + fb.extrinsic3D() + fb.template_points()


jac_line = op_fun.block_string_to_compiled_jacobian_line()

# for block in op_fun.function_blocks:
#     test = block.compute_jac
#     print(type(block))
#     print("trying to run the test")
#     test(
#         np.ones(27)[:block.params.n_params], #params
#         np.ones(27), #input
#         np.ones(27), #output
#         np.ones(27), #memory
#     )
#     print("test passed")



params = np.ones(op_fun.param_line_length)
inp = np.ones(op_fun.inp_mem_req)
out = np.ones(100)
working_mem = np.ones(op_fun.wrk_mem_req)

a0 = np.eye(op_fun.param_line_length + np.sum(op_fun.n_outs))
a1 = np.eye(op_fun.param_line_length + np.sum(op_fun.n_outs))
a2 = np.eye(op_fun.param_line_length + np.sum(op_fun.n_outs))


mat = jac_line(params, a0, a1, a2, inp, out, working_mem, op_fun.loss_jac, op_fun.loss_fun)
l = lambda : jac_line(params, a0, a1, a2, inp, out, working_mem, op_fun.loss_jac, op_fun.loss_fun)
# mat = l()

benchmark(l, repeats =10000, mode = "us")
plt.imshow(np.abs(mat)>0)
plt.show()


print(op_fun.wrk_mem_req)

