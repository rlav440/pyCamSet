from sympy import simplify, symbols, diff
import numpy as np
import sympy as sy
from numba import njit


from pyCamSet.utils.general_utils import benchmark

#define the projection and distortion functions

x_out, y_out, p_x, p_y, f_x, f_y, k_0, k_1, k_2, p_0, p_1, xw, yw, zw = symbols(
    "x_out y_out p_x p_y f_x f_y k_0 k_1 k_2 p_0 p_1 xw yw zw"
)

x_m_0, y_m_0, x_m_1, y_m_1, r, k_up, x_in, y_in = symbols(
    "x_m_0 y_m_0 x_m_1 y_m_1 r k_up x_in y_in"
)

x_in = (f_x * xw +  0 * yw + p_x * zw)/zw
y_in = (0 * xw +  f_y * yw + p_y * zw)/zw


x_m_0, y_m_0 = (x_in - p_x)/f_x, (y_in - p_y)/f_y
r = x_m_0**2 + y_m_0**2


k_up = (1 + k_0 * r + k_1 * (r ** 2) + k_2 * (r ** 3))
tang_x = 2 * p_0 * x_m_0 * y_m_0 + p_1 * (r + 2 * (x_m_0 ** 2))
tang_y = p_0 * (r + 2 * (y_m_0 ** 2)) + 2 * p_1 * x_m_0 * y_m_0

x_m_1 = k_up * x_m_0 + tang_x
y_m_1 = k_up * y_m_0 + tang_y

x_out = x_m_1 * f_x + p_x
y_out = y_m_1 * f_y + p_y

# for num, sym in enumerate([x_out, y_out]):
#     n = "x" if num == 0 else "y"
#     for partial in [p_x, p_y, f_x, f_y, k_0, k_1, k_2, p_0, p_1, xw, yw, zw]:
#         print(f"#derivative of {n} with respect to {partial}")
#         print(f"d{n}d{partial} = ", end="")
#         print(simplify(diff(sym, partial)))
#         print()


# matrix = sy.Matrix([x_out, y_out])
# J = matrix.jacobian( [p_x, p_y, f_x, f_y, k_0, k_1, k_2, p_0, p_1, x_in, y_in])
# print(sy.simplify(J))

@njit(cache=True, fastmath=True)
def cpd_brown_conrady(params, inp, output, memory):

    f_x, p_x, f_y, p_y, k_0, k_1, p_0, p_1, k_2 = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]
    xw, yw, zw = inp[0], inp[1], inp[2]

    #derivative of x with respect to p_x
    dxdp_x = 1
    #derivative of x with respect to p_y
    dxdp_y = 0
    #derivative of x with respect to f_x
    dxdf_x = (
        xw*(k_0*zw**4*(xw**2 + yw**2) + k_1*zw**2*(xw**2 + yw**2)**2 + k_2*(xw**2 + yw**2)**3 + zw**6)
            + zw**5*(2*p_0*xw*yw + p_1*(3*xw**2 + yw**2))
    )/zw**7
    #derivative of x with respect to f_y
    dxdf_y = 0
    #derivative of x with respect to k_0
    dxdk_0 = f_x*xw*(xw**2 + yw**2)/zw**3
    #derivative of x with respect to k_1
    dxdk_1 = f_x*xw*(xw**2 + yw**2)**2/zw**5
    #derivative of x with respect to k_2
    dxdk_2 = f_x*xw*(xw**2 + yw**2)**3/zw**7
    #derivative of x with respect to p_0
    dxdp_0 = 2*f_x*xw*yw/zw**2
    #derivative of x with respect to p_1
    dxdp_1 = f_x*(3*xw**2 + yw**2)/zw**2
    #derivative of x with respect to xw
    dxdxw = f_x*(k_0*zw**4*(xw**2 + yw**2)\
        + k_1*zw**2*(xw**2 + yw**2)**2\
        + k_2*(xw**2 + yw**2)**3\
        + 2*xw**2*(k_0*zw**4 + 2*k_1*zw**2*(xw**2 + yw**2) + 3*k_2*(xw**2 + yw**2)**2)\
        + zw**6\
        + 2*zw**5*(p_0*yw + 3*p_1*xw))/zw**7
    #derivative of x with respect to yw
    dxdyw = 2*f_x*(xw*yw*(k_0*zw**4 + 2*k_1*zw**2*(xw**2 + yw**2)\
        + 3*k_2*(xw**2 + yw**2)**2)\
        + zw**5*(p_0*xw + p_1*yw))/zw**7
    #derivative of x with respect to zw
    dxdzw = -f_x*(
        4*p_0*xw*yw*zw**5\
        + 2*p_1*zw**5*(3*xw**2 + yw**2) \
        + 2*xw*(xw**2 + yw**2)*(k_0*zw**4 + 2*k_1*zw**2*(xw**2 + yw**2) + 3*k_2*(xw**2 + yw**2)**2)\
        + xw*(k_0*zw**4*(xw**2 + yw**2)\
        + k_1*zw**2*(xw**2 + yw**2)**2\
        + k_2*(xw**2 + yw**2)**3 + zw**6)
        )/zw**8
    #derivative of y with respect to p_x
    dydp_x = 0
    #derivative of y with respect to p_y
    dydp_y = 1
    #derivative of y with respect to f_x
    dydf_x = 0
    #derivative of y with respect to f_y
    dydf_y = (
        yw*(k_0*zw**4*(xw**2 + yw**2) + k_1*zw**2*(xw**2 + yw**2)**2 + k_2*(xw**2 + yw**2)**3 + zw**6)
        + zw**5*(p_0*(xw**2 + 3*yw**2) + 2*p_1*xw*yw)
    )/zw**7
    #derivative of y with respect to k_0
    dydk_0 = f_y*yw*(xw**2 + yw**2)/zw**3
    #derivative of y with respect to k_1
    dydk_1 = f_y*yw*(xw**2 + yw**2)**2/zw**5
    #derivative of y with respect to k_2
    dydk_2 = f_y*yw*(xw**2 + yw**2)**3/zw**7
    #derivative of y with respect to p_0
    dydp_0 = f_y*(xw**2 + 3*yw**2)/zw**2
    #derivative of y with respect to p_1
    dydp_1 = 2*f_y*xw*yw/zw**2
    #derivative of y with respect to xw
    dydxw = 2*f_y*(xw*yw*(k_0*zw**4 + 2*k_1*zw**2*(xw**2 + yw**2) + 3*k_2*(xw**2 + yw**2)**2) + zw**5*(p_0*xw + p_1*yw))/zw**7
    #derivative of y with respect to yw
    dydyw = f_y*(k_0*zw**4*(xw**2 + yw**2) \
        + k_1*zw**2*(xw**2 + yw**2)**2\
        + k_2*(xw**2 + yw**2)**3\
        + 2*yw**2*(k_0*zw**4 + 2*k_1*zw**2*(xw**2 + yw**2) + 3*k_2*(xw**2 + yw**2)**2)\
        + zw**6\
        + 2*zw**5*(3*p_0*yw + p_1*xw))/zw**7
    #derivative of y with respect to zw
    dydzw = -f_y*(
        2*p_0*zw**5*(xw**2 + 3*yw**2) \
        + 4*p_1*xw*yw*zw**5\
        + 2*yw*(xw**2 + yw**2)*(k_0*zw**4 + 2*k_1*zw**2*(xw**2 + yw**2) + 3*k_2*(xw**2 + yw**2)**2)\
        + yw*(k_0*zw**4*(xw**2 + yw**2) + k_1*zw**2*(xw**2 + yw**2)**2 + k_2*(xw**2 + yw**2)**3 + zw**6)
    )/zw**8
    output[:] = [dxdp_x,dxdp_y,dxdf_x,dxdf_y,dxdk_0,dxdk_1,dxdk_2,dxdp_0,dxdp_1,dxdxw,dxdyw,dxdzw, 
                 dydp_x,dydp_y,dydf_x,dydf_y,dydk_0,dydk_1,dydk_2,dydp_0,dydp_1,dydxw,dydyw,dydzw]

params = np.zeros(9)
inp = np.ones(3)
output = np.empty(24)
cpd_brown_conrady(params, inp, output, 0)
benchmark(lambda :cpd_brown_conrady(params, inp, output, 0), repeats=100000, mode="us")
