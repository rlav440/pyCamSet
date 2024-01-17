from numba import njit, gdb_init
from time import sleep
import numpy as np

from .abstract_function_blocks import abstract_function_block, param_type, key_type

from .compiled_helpers import n_htform_prealloc, n_e4x4_flat_INPLACE, numba_rodrigues_jac
import numba  


ftemplate = "void(float64[::1],float64[::1],float64[::1],float64[::1])"
numba.types.FunctionType(
    numba.void(
        numba.types.Array(numba.float64, 1, "C"),
        numba.types.Array(numba.float64, 1, "C"),
        numba.types.Array(numba.float64, 1, "C"),
        numba.types.Array(numba.float64, 1, "C"),
    )
)

class projection(abstract_function_block):
    num_inp = 3
    num_out = 2
    params = param_type(key_type.PER_CAM, 9)
    array_memory = 1

    @staticmethod
    @njit(ftemplate) 
    def compute_fun(params, inp, output, memory): 
        x, y, inv_z = inp[0], inp[1], 1/inp[2]
        #params have order fx,px,fy,py k0,k1, p0, p1
        u = (params[0] *x + params[1]) * inv_z
        v = (params[2] *y + params[3]) * inv_z
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
        return


    @staticmethod
    @njit(ftemplate)
    def compute_jac(params, inp, output, memory):

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
        output[:24] = [
            dxdp_x,dxdp_y,dxdf_x,dxdf_y,dxdk_0,dxdk_1,dxdk_2,dxdp_0,dxdp_1,dxdxw,dxdyw,dxdzw, 
            dydp_x,dydp_y,dydf_x,dydf_y,dydk_0,dydk_1,dydk_2,dydp_0,dydp_1,dydxw,dydyw,dydzw]
        return


class rigidTform3d(abstract_function_block):
    num_inp = 3
    num_out = 3
    params = param_type(key_type.PER_IMG, 6)

    array_memory = 27 # 12 for the normal, but 27 for calcing the deritaves

    @staticmethod
    @njit(ftemplate)
    def compute_fun(params, inp, output, memory):
        n_e4x4_flat_INPLACE(params, memory[:12])
        n_htform_prealloc(inp, memory[:12], out=output[:3])
        return

    @staticmethod
    @njit(ftemplate)
    def compute_jac(params, inp, output, memory): 
        numba_rodrigues_jac(params[:3], memory) #will use 27 points
        output[:] = 0 

        for op in range(3):
            for ang_comp in range(3):
                output[op * 12 + ang_comp] = memory[9 * ang_comp + 0] * inp[0] + \
                                             memory[9 * ang_comp + 1] * inp[1] + \
                                             memory[9 * ang_comp + 2] * inp[2]
        # do the translations 
        output[0 * 9 + 3] = 1
        output[1 * 9 + 4] = 1
        output[2 * 9 + 5] = 1

        #do the change with the input variables
        n_e4x4_flat_INPLACE(params, memory[:12])
        for op in range(3):
            for inval in range(3):
                output[op*9 + 6 + inval] = memory[inval + 4 * op]
        return 

class extrinsic3D(rigidTform3d):
    params = param_type(key_type.PER_CAM, 6)

# I need some way to grab data from the function block and return it's points.
class template_points(rigidTform3d):
    template = True
    num_inp = 0
    num_out = 3
    params = param_type(key_type.PER_KEY, 6)
  
    @staticmethod
    @njit(ftemplate)
    def compute_jac(params, inp, output, memory): 
        numba_rodrigues_jac(params[:3], memory) #will use 27 points
        output[:18] = 0 
        for op in range(3):
            for ang_comp in range(3):
                output[op * 6 + ang_comp] = memory[6 * ang_comp + 0] * inp[0] + \
                                            memory[6 * ang_comp + 1] * inp[1] + \
                                            memory[6 * ang_comp + 2] * inp[2]    
        # do the translations 
        output[0 * 6 + 3] = 1
        output[1 * 6 + 4] = 1
        output[2 * 6 + 5] = 1
        return




class free_point(abstract_function_block):
    """
    Implements a 3D point that is parameterised by it's x,y and z locations.
    """

    num_inp = 0
    num_out = 3
    params = param_type(key_type.PER_KEY, 3)
    array_memory = 0

    @staticmethod
    @njit(ftemplate)
    def compute_fun(params, inp, output, memory=0):
        output[0] = params[0]
        output[1] = params[1]
        output[2] = params[2]


    @staticmethod
    @njit(ftemplate)
    def compute_jac(params, inp, output, memory=0):
        output[:] = 0
        output[0] = 1
        output[4] = 1
        output[8] = 1


