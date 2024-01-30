from pyCamSet.optimisation.compiled_helpers import numba_rodrigues_jac
from pyCamSet.optimisation.compiled_helpers import n_htform_prealloc
import numpy as np
from numba import njit
from time import sleep
from pyCamSet.optimisation.abstract_function_blocks import abstract_function_block
from pyCamSet.optimisation.abstract_function_blocks import key_type
from pyCamSet.optimisation.abstract_function_blocks import param_type
from numba import gdb_init
from pyCamSet.optimisation.compiled_helpers import n_e4x4_flat_INPLACE
import numba


from numba import prange
from pyCamSet.optimisation.abstract_function_blocks import make_param_struct
 
def make_full_loss(op_fun, detections, template, threads):
    param_slices = op_fun.param_slices
    n_outs = op_fun.n_outs
    working_memories = np.array(op_fun.working_memories)
    n_blocks = op_fun.n_blocks
    n_threads = threads
    op_fun._prep_for_computation()
    d_shape = detections.shape
    p_shape = (threads, int(np.ceil(d_shape[0]/threads)), *d_shape[1:])
    detection_data = np.resize(detections, p_shape)
    n_lines = detection_data.shape[1]
    inp_mem = op_fun.inp_mem_req
    out_mem = op_fun.out_mem_req
    wrk_mem = op_fun.wrk_mem_req
    starts, block_n_params, param_inds, key_type = make_param_struct(op_fun.function_blocks, detections)
    param_len = np.sum(block_n_params)
    param_slices = op_fun.param_slices
    use_template = op_fun.templated and (template is not None)
    if op_fun.templated and not use_template:
        raise ValueError("A templated optimisation was defined, but no template data was given to create the loss function")
    t_data: np.ndarray = template if use_template else np.zeros(3)
    @njit
    def full_loss(inp_params):
        losses = np.empty((n_threads, n_lines, 2))
        for i in prange(n_threads):
            #make the memory components required
            inp = np.empty(inp_mem)
            output = np.empty(out_mem)
            memory = np.empty(wrk_mem)
            dense_param_arr = np.empty(param_len)
            for ii in range(n_lines):
                datum = detection_data[i, ii]
                for idb in range(n_blocks):
                    s_num = datum[key_type[idb]] #the index value of the associated parameter
                    p_ind = param_inds[idb] # maps the param to it's index in the unique params
                    start = starts[p_ind] + s_num * block_n_params[p_ind] #and the associated change in the start location
                    dense_param_arr[param_slices[2*idb]:param_slices[2*idb + 1]] = inp_params[int(start):int(start + block_n_params[p_ind])]
                if use_template:
                    inp[:3] = t_data[int(datum[2])] 
                params = dense_param_arr[param_slices[2*2]:param_slices[2*2+1]]
                n_e4x4_flat_INPLACE(params, memory[:12])
                n_htform_prealloc(inp, memory[:12], out=output[:3])
                inp[:n_outs[2]] = output[:n_outs[2]]
                params = dense_param_arr[param_slices[2*1]:param_slices[2*1+1]]
                n_e4x4_flat_INPLACE(params, memory[:12])
                n_htform_prealloc(inp, memory[:12], out=output[:3])
                inp[:n_outs[1]] = output[:n_outs[1]]
                params = dense_param_arr[param_slices[2*0]:param_slices[2*0+1]]
                x, y, inv_z = inp[0], inp[1], 1/inp[2]
                #params have order fx,px,fy,py k0,k1, p0, p1
                u = (params[0] *x + params[1]* inp[2]) * inv_z
                v = (params[2] *y + params[3]* inp[2]) * inv_z
                k = params[4:]
                x, y = (u - params[1]) / params[0], (v - params[3]) / params[2]
                r2 = x ** 2 + y ** 2
                kup = (1 + k[0] * r2 + k[1] * (r2 ** 2) + k[4] * (r2 ** 3))
                # distort radially
                xD = x * kup
                yD = y * kup
                # distort tangentially
                xD += 2 * k[2] * x * y + k[3] * (r2 + 2 * (x ** 2))
                yD += k[2] * (r2 + 2 * (y ** 2)) + 2 * k[3] * x * y
                # back to absolute
                output[0] =  xD * params[0] + params[1]
                output[1] =  yD * params[2] + params[3]
                inp[:n_outs[0]] = output[:n_outs[0]]
                losses[i, ii] = [output[0] - datum[3], output[1] - datum[4]]
        return np.resize(losses, (d_shape[0], 2))
    return full_loss