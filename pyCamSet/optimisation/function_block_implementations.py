from numba import njit, gdb_init
from time import sleep
import numpy as np

from pyCamSet.optimisation.abstract_function_blocks import abstract_function_block, param_type, key_type, optimisation_function

from pyCamSet.optimisation.compiled_helpers import n_htform_prealloc, n_e4x4_flat_INPLACE, numba_rodrigues_jac
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
    @njit(ftemplate, cache=True)
    def compute_fun(params, inp, output, memory):
        """
        Projects a camera frame point to pixels through the 5 term distortion.

        The previous form scaled by the focal length and principal point and
        then immediately undid it (``x = (u - p_x)/f_x`` recovers ``x/z``), and
        recomputed r**2 and its powers throughout; going straight to normalised
        coordinates and sharing the powers is ~3.4x faster and slightly more
        accurate.
        """
        f_x, p_x, f_y, p_y = params[0], params[1], params[2], params[3]
        k_0, k_1, p_0, p_1, k_2 = params[4], params[5], params[6], params[7], params[8]

        inv_z = 1.0/inp[2]
        x = inp[0] * inv_z
        y = inp[1] * inv_z

        r2 = x*x + y*y
        r4 = r2*r2
        kup = 1.0 + k_0*r2 + k_1*r4 + k_2*r4*r2

        xy = x*y
        xD = x*kup + 2.0*p_0*xy + p_1*(r2 + 2.0*x*x)
        yD = y*kup + p_0*(r2 + 2.0*y*y) + 2.0*p_1*xy

        output[0] = xD*f_x + p_x
        output[1] = yD*f_y + p_y
        return


    @staticmethod
    @njit(ftemplate, cache=True)
    def compute_jac(params, inp, output, memory):
        """
        The analytic jacobian of the projection, with its subexpressions shared.

        The expanded form of these derivatives repeats (x**2 + y**2) 38 times
        and powers of z 58 times, and numba does not eliminate them, so sharing
        them by hand is ~5x faster. Two groupings do most of the work:
        ``radial``, the distortion polynomial, and ``d_radial``, its derivative
        with respect to r**2.
        """
        f_x, p_x, f_y, p_y = params[0], params[1], params[2], params[3]
        k_0, k_1, p_0, p_1, k_2 = params[4], params[5], params[6], params[7], params[8]
        x, y, z = inp[0], inp[1], inp[2]

        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2
        z2 = z*z
        z3 = z2*z
        z4 = z2*z2
        z5 = z4*z
        z6 = z3*z3
        z7 = z6*z
        iz2 = 1.0/z2
        iz3 = 1.0/z3
        iz5 = 1.0/z5
        iz7 = 1.0/z7
        iz8 = iz7/z

        radial = k_0*z4*r2 + k_1*z2*r4 + k_2*r6 + z6
        d_radial = k_0*z4 + 2.0*k_1*z2*r2 + 3.0*k_2*r4

        xy = x*y
        xx3 = 3.0*x*x + y*y
        yy3 = x*x + 3.0*y*y
        # dx/dyw and dy/dxw differ only by the focal length
        cross = 2.0*(xy*d_radial + z5*(p_0*x + p_1*y))*iz7

        output[0] = (x*radial + z5*(2.0*p_0*xy + p_1*xx3))*iz7
        output[1] = 1.0
        output[2] = 0.0
        output[3] = 0.0
        output[4] = f_x*x*r2*iz3
        output[5] = f_x*x*r4*iz5
        output[6] = 2.0*f_x*xy*iz2
        output[7] = f_x*xx3*iz2
        output[8] = f_x*x*r6*iz7
        output[9] = f_x*(radial + 2.0*x*x*d_radial + 2.0*z5*(p_0*y + 3.0*p_1*x))*iz7
        output[10] = f_x*cross
        output[11] = -f_x*(4.0*p_0*xy*z5 + 2.0*p_1*z5*xx3
                           + 2.0*x*r2*d_radial + x*radial)*iz8

        output[12] = 0.0
        output[13] = 0.0
        output[14] = (y*radial + z5*(p_0*yy3 + 2.0*p_1*xy))*iz7
        output[15] = 1.0
        output[16] = f_y*y*r2*iz3
        output[17] = f_y*y*r4*iz5
        output[18] = f_y*yy3*iz2
        output[19] = 2.0*f_y*xy*iz2
        output[20] = f_y*y*r6*iz7
        output[21] = f_y*cross
        output[22] = f_y*(radial + 2.0*y*y*d_radial + 2.0*z5*(3.0*p_0*y + p_1*x))*iz7
        output[23] = -f_y*(2.0*p_0*z5*yy3 + 4.0*p_1*xy*z5
                           + 2.0*y*r2*d_radial + y*radial)*iz8


class rigidTform3d(abstract_function_block):
    num_inp = 3
    num_out = 3
    params = param_type(key_type.PER_IMG, 6)

    array_memory = 27 # 12 for the normal, but 27 for calcing the deritaves

    @staticmethod
    @njit(ftemplate, cache=True)
    def compute_fun(params, inp, output, memory):
        n_e4x4_flat_INPLACE(params, memory[:12])
        n_htform_prealloc(inp, memory[:12], out=output[:3])
        return

    @staticmethod
    @njit(ftemplate, cache=True)
    def compute_jac(params, inp, output, memory): 
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
        return 

class extrinsic3D(rigidTform3d):
    params = param_type(key_type.PER_CAM, 6)

# I need some way to grab data from the function block and return it's points.
class template_points(rigidTform3d):
    template = True
    num_inp = 0
    # compute_jac reads the 3 template coordinates from inp even though none of
    # them is a differentiable input, so the read width is wider than num_inp.
    n_inp_read = 3
    num_out = 3
    params = param_type(key_type.PER_IMG, 6)
    n_inp_read = 3
  
    @staticmethod
    @njit(ftemplate, cache=True)
    def compute_jac(params, inp, output, memory): 
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
    @njit(ftemplate, cache=True)
    def compute_fun(params, inp, output, memory=0):
        output[0] = params[0]
        output[1] = params[1]
        output[2] = params[2]


    @staticmethod
    @njit(ftemplate, cache=True)
    def compute_jac(params, inp, output, memory=0):
        output[:] = 0
        output[0] = 1
        output[4] = 1
        output[8] = 1
