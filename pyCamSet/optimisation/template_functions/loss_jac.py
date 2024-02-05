import numpy as np
from pyCamSet.optimisation.abstract_function_blocks import abstract_function_block
import numba
from numba import njit
from pyCamSet.optimisation.abstract_function_blocks import param_type
from time import sleep
from pyCamSet.optimisation.compiled_helpers import n_htform_prealloc
from pyCamSet.optimisation.compiled_helpers import n_e4x4_flat_INPLACE
from pyCamSet.optimisation.compiled_helpers import numba_rodrigues_jac
from numba import gdb_init
from pyCamSet.optimisation.abstract_function_blocks import optimisation_function
from pyCamSet.optimisation.abstract_function_blocks import key_type


from numba import prange
from pyCamSet.optimisation.abstract_function_blocks import make_param_struct
 
def make_full_jac(op_fun, detections, template, threads):
    param_slices = op_fun.param_slices
    n_outs = op_fun.n_outs
    working_memories = np.array(op_fun.working_memories)
    n_blocks = op_fun.n_blocks
    n_params  = np.array(op_fun.n_params)
    n_outs = np.array(op_fun.n_outs)
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
    grad_outputsize = np.max(op_fun.grad_outputsize)
    starts, block_n_params, param_inds, key_type = make_param_struct(op_fun.function_blocks, detections)
    param_len = np.sum(block_n_params)
    param_slices = op_fun.param_slices
    out_param_start = param_len
    jac_size = param_slices[-1]
    f_outs = op_fun.n_outs[0]
    use_template = op_fun.templated and (template is not None)
    if op_fun.templated and not use_template:
        raise ValueError("A templated optimisation was defined, but no template data was given to create the loss function")
    t_data: np.ndarray = template if use_template else np.zeros(3)
    #workingtag 16:38:32
    @njit(parallel=True,fastmath=True)
    def full_jac(inp_params):
        p_size = len(inp_params)
        out_jac = np.zeros((n_threads, n_lines * f_outs, p_size))
        for i in prange(n_threads):
            #make the memory components required
            inp = np.empty(inp_mem)
            output = np.empty(grad_outputsize)
            memory = np.empty(wrk_mem)
            dense_param_arr = np.empty(param_len)
            base = np.eye(jac_size)
            jac = np.eye(jac_size)
            per_block = np.eye(jac_size)
            for ii in range(n_lines):
                datum = detection_data[i, ii]
                jac[:] = base[:]
                for idb in range(n_blocks):
                    s_num = datum[key_type[idb]]
                    p_ind = param_inds[idb]
                    start = starts[p_ind] + s_num * block_n_params[p_ind]
                    dense_param_arr[param_slices[2*idb]:param_slices[2*idb + 1]] = inp_params[int(start):int(start + block_n_params[p_ind])]
                if use_template:
                    inp[:3] = t_data[int(datum[2])] 
                ################# BLOCK 2 #################
                params = dense_param_arr[param_slices[2*2]:param_slices[2*2+1]]
                per_block[:] = base[:]
                numba_rodrigues_jac(params[:3], memory) #will use 27 points
                output[:18] = 0 
                for op in range(3):
                    for ang_comp in range(3):
                        k = op * 6 + ang_comp
                        output[k] = (
                            memory[9 * ang_comp + op * 3 + 0] * inp[0] + 
                            memory[9 * ang_comp + op * 3 + 1] * inp[1] + 
                            memory[9 * ang_comp + op * 3 + 2] * inp[2]
                        )
                # do the translations 
                output[0 * 6 + 3] = 1
                output[1 * 6 + 4] = 1
                output[2 * 6 + 5] = 1
                ############### POPULATING THE JACOBEAN ###########
                out_ind_start = param_slices[2*n_blocks + 2*(2)] 
                out_ind_end = param_slices[2*n_blocks + 2*(2) + 1]
                n_outputs = out_ind_end - out_ind_start
                

                inp_ind_start = param_slices[2*n_blocks + 2*2 + 2] 
                inp_ind_end = param_slices[2*n_blocks + 2*2 + 1 + 2]
                n_inputs = inp_ind_end - inp_ind_start
                #write the derivatives of the parameters
                n_param  = n_params[2]
                param_start = param_slices[2*2]
                param_end = param_slices[2*2 + 1]
                ll = n_inputs + n_param
                for idc, output_var in enumerate(range(out_ind_start, out_ind_end)):
                    #write the derivative with respect to the controlling params
                    per_block[output_var, param_start:param_end] = output[idc*ll:idc*ll + n_param] #envisions this as a dense array
                    #write the derivative with respect to the inputs
                    if n_inputs != 0:
                        per_block[output_var, inp_ind_start:inp_ind_end] = output[idc*ll + n_param:(idc + 1)*ll] #envisions this as a dense array
                jac = per_block @ jac
                params = dense_param_arr[param_slices[2*2]:param_slices[2*2+1]]
                n_e4x4_flat_INPLACE(params, memory[:12])
                n_htform_prealloc(inp, memory[:12], out=output[:3])
                inp[:n_outs[2]] = output[:n_outs[2]]
                ################# BLOCK 1 #################
                params = dense_param_arr[param_slices[2*1]:param_slices[2*1+1]]
                per_block[:] = base[:]
                numba_rodrigues_jac(params[:3], memory) #will use 27 points
                output[:] = 0 
                for op in range(3):
                    for ang_comp in range(3):
                        k = op * 9 + ang_comp
                        output[k] = (
                            memory[9 * ang_comp + op * 3 + 0] * inp[0] + 
                            memory[9 * ang_comp + op * 3 + 1] * inp[1] + 
                            memory[9 * ang_comp + op * 3 + 2] * inp[2]
                        )
                # do the translations 
                output[0 * 9 + 3] = 1
                output[1 * 9 + 4] = 1
                output[2 * 9 + 5] = 1
                #do the change with the input variables
                n_e4x4_flat_INPLACE(params, memory[:12])
                for op in range(3):
                    for inval in range(3):
                        k = op*9 + 6 + inval
                        output[k] = memory[inval + 3 * op]
                ############### POPULATING THE JACOBEAN ###########
                out_ind_start = param_slices[2*n_blocks + 2*(1)] 
                out_ind_end = param_slices[2*n_blocks + 2*(1) + 1]
                n_outputs = out_ind_end - out_ind_start
                

                inp_ind_start = param_slices[2*n_blocks + 2*1 + 2] 
                inp_ind_end = param_slices[2*n_blocks + 2*1 + 1 + 2]
                n_inputs = inp_ind_end - inp_ind_start
                #write the derivatives of the parameters
                n_param  = n_params[1]
                param_start = param_slices[2*1]
                param_end = param_slices[2*1 + 1]
                ll = n_inputs + n_param
                for idc, output_var in enumerate(range(out_ind_start, out_ind_end)):
                    #write the derivative with respect to the controlling params
                    per_block[output_var, param_start:param_end] = output[idc*ll:idc*ll + n_param] #envisions this as a dense array
                    #write the derivative with respect to the inputs
                    if n_inputs != 0:
                        per_block[output_var, inp_ind_start:inp_ind_end] = output[idc*ll + n_param:(idc + 1)*ll] #envisions this as a dense array
                jac = per_block @ jac
                params = dense_param_arr[param_slices[2*1]:param_slices[2*1+1]]
                n_e4x4_flat_INPLACE(params, memory[:12])
                n_htform_prealloc(inp, memory[:12], out=output[:3])
                inp[:n_outs[1]] = output[:n_outs[1]]
                ################# BLOCK 0 #################
                params = dense_param_arr[param_slices[2*0]:param_slices[2*0+1]]
                per_block[:] = base[:]
                f_x, p_x, f_y, p_y, k_0, k_1, p_0, p_1, k_2 = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]
                x, y, z = inp[0], inp[1], inp[2]
                #derivative of x with respect to p_x
                dxdp_x = 1
                #derivative of x with respect to p_y
                dxdp_y = 0
                #derivative of x with respect to f_x
                dxdf_x = (
                    x*(k_0*z**4*(x**2 + y**2) + k_1*z**2*(x**2 + y**2)**2 + k_2*(x**2 + y**2)**3 + z**6)
                        + z**5*(2*p_0*x*y + p_1*(3*x**2 + y**2))
                )/z**7
                #derivative of x with respect to f_y
                dxdf_y = 0
                #derivative of x with respect to k_0
                dxdk_0 = f_x*x*(x**2 + y**2)/z**3
                #derivative of x with respect to k_1
                dxdk_1 = f_x*x*(x**2 + y**2)**2/z**5
                #derivative of x with respect to k_2
                dxdk_2 = f_x*x*(x**2 + y**2)**3/z**7
                #derivative of x with respect to p_0
                dxdp_0 = 2*f_x*x*y/z**2
                #derivative of x with respect to p_1
                dxdp_1 = f_x*(3*x**2 + y**2)/z**2
                #derivative of x with respect to xw
                dxdxw = f_x*(k_0*z**4*(x**2 + y**2)\
                    + k_1*z**2*(x**2 + y**2)**2\
                    + k_2*(x**2 + y**2)**3\
                    + 2*x**2*(k_0*z**4 + 2*k_1*z**2*(x**2 + y**2) + 3*k_2*(x**2 + y**2)**2)\
                    + z**6\
                    + 2*z**5*(p_0*y + 3*p_1*x))/z**7
                #derivative of x with respect to yw
                dxdyw = 2*f_x*(x*y*(k_0*z**4 + 2*k_1*z**2*(x**2 + y**2)\
                    + 3*k_2*(x**2 + y**2)**2)\
                    + z**5*(p_0*x + p_1*y))/z**7
                #derivative of x with respect to zw
                dxdzw = -f_x*(
                    4*p_0*x*y*z**5\
                    + 2*p_1*z**5*(3*x**2 + y**2) \
                    + 2*x*(x**2 + y**2)*(k_0*z**4 + 2*k_1*z**2*(x**2 + y**2) + 3*k_2*(x**2 + y**2)**2)\
                    + x*(k_0*z**4*(x**2 + y**2)\
                    + k_1*z**2*(x**2 + y**2)**2\
                    + k_2*(x**2 + y**2)**3 + z**6)
                    )/z**8
                #derivative of y with respect to p_x
                dydp_x = 0
                #derivative of y with respect to p_y
                dydp_y = 1
                #derivative of y with respect to f_x
                dydf_x = 0
                #derivative of y with respect to f_y
                dydf_y = (
                    y*(k_0*z**4*(x**2 + y**2) + k_1*z**2*(x**2 + y**2)**2 + k_2*(x**2 + y**2)**3 + z**6)
                    + z**5*(p_0*(x**2 + 3*y**2) + 2*p_1*x*y)
                )/z**7
                #derivative of y with respect to k_0
                dydk_0 = f_y*y*(x**2 + y**2)/z**3
                #derivative of y with respect to k_1
                dydk_1 = f_y*y*(x**2 + y**2)**2/z**5
                #derivative of y with respect to k_2
                dydk_2 = f_y*y*(x**2 + y**2)**3/z**7
                #derivative of y with respect to p_0
                dydp_0 = f_y*(x**2 + 3*y**2)/z**2
                #derivative of y with respect to p_1
                dydp_1 = 2*f_y*x*y/z**2
                #derivative of y with respect to xw
                dydxw = 2*f_y*(x*y*(k_0*z**4 + 2*k_1*z**2*(x**2 + y**2) + 3*k_2*(x**2 + y**2)**2) + z**5*(p_0*x + p_1*y))/z**7
                #derivative of y with respect to yw
                dydyw = f_y*(k_0*z**4*(x**2 + y**2) \
                    + k_1*z**2*(x**2 + y**2)**2\
                    + k_2*(x**2 + y**2)**3\
                    + 2*y**2*(k_0*z**4 + 2*k_1*z**2*(x**2 + y**2) + 3*k_2*(x**2 + y**2)**2)\
                    + z**6\
                    + 2*z**5*(3*p_0*y + p_1*x))/z**7
                #derivative of y with respect to zw
                dydzw = -f_y*(
                    2*p_0*z**5*(x**2 + 3*y**2) \
                    + 4*p_1*x*y*z**5\
                    + 2*y*(x**2 + y**2)*(k_0*z**4 + 2*k_1*z**2*(x**2 + y**2) + 3*k_2*(x**2 + y**2)**2)\
                    + y*(k_0*z**4*(x**2 + y**2) + k_1*z**2*(x**2 + y**2)**2 + k_2*(x**2 + y**2)**3 + z**6)
                )/z**8
                derive_list =[
                    dxdf_x,dxdp_x,dxdf_y,dxdp_y,dxdk_0,dxdk_1,dxdp_0,dxdp_1,dxdk_2,dxdxw,dxdyw,dxdzw, 
                    dydf_x,dydp_x,dydf_y,dydp_y,dydk_0,dydk_1,dydp_0,dydp_1,dydk_2,dydxw,dydyw,dydzw]
                for i_local in range(24):
                    output[i_local] = derive_list[i_local]
                ############### POPULATING THE JACOBEAN ###########
                out_ind_start = param_slices[2*n_blocks + 2*(0)] 
                out_ind_end = param_slices[2*n_blocks + 2*(0) + 1]
                n_outputs = out_ind_end - out_ind_start
                

                inp_ind_start = param_slices[2*n_blocks + 2*0 + 2] 
                inp_ind_end = param_slices[2*n_blocks + 2*0 + 1 + 2]
                n_inputs = inp_ind_end - inp_ind_start
                #write the derivatives of the parameters
                n_param  = n_params[0]
                param_start = param_slices[2*0]
                param_end = param_slices[2*0 + 1]
                ll = n_inputs + n_param
                for idc, output_var in enumerate(range(out_ind_start, out_ind_end)):
                    #write the derivative with respect to the controlling params
                    per_block[output_var, param_start:param_end] = output[idc*ll:idc*ll + n_param] #envisions this as a dense array
                    #write the derivative with respect to the inputs
                    if n_inputs != 0:
                        per_block[output_var, inp_ind_start:inp_ind_end] = output[idc*ll + n_param:(idc + 1)*ll] #envisions this as a dense array
                jac = per_block @ jac
                for idb in range(n_blocks):
                    s_num = datum[key_type[idb]] #the index value of the associated parameter
                    p_ind = param_inds[idb] # maps the param to it's index in the unique params
                    start = starts[p_ind] + s_num * block_n_params[p_ind]
                    for i_out in range(f_outs):
                        out_jac[i, ii*f_outs + i_out, int(start):int(start + block_n_params[p_ind])] = jac[                        out_param_start + i_out, param_slices[2*idb]:param_slices[2*idb + 1]                    ]
        return np.resize(out_jac, (2*d_shape[0], p_size))
    return full_jac