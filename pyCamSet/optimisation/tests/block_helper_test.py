import pyCamSet.optimisation.function_block_implementations as fb
import pyCamSet.optimisation.abstract_function_blocks as afb


op_fun = fb.projection() + fb.extrinsic3D() + fb.template_points()


jac_line = op_fun.block_string_to_compiled_jacobian_line()

print(op_fun.wrk_mem_req)

