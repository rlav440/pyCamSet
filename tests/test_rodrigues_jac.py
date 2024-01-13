import numpy as np
import cv2
from numba import njit

import time
import math

from uniplot import histogram


def benchmark(func, repeats=100, mode="ms", timer=time.time_ns):
    """
    A handy function to benchmark a function. Tracks the execution time, and also the numba allocations.
    :param func: The function to benchmark as a lambda
    :param repeats: The number of times to repeat the function call
    :param mode: The mode to display the results in. Can be "us", "ms", or "s"
    :param timer: The timer to use. Can be time.time_ns, or time.perf_counter_ns
    """

    def run_benchmark():
        ranges = {
            "us":1e-3,
            "ms":1e-6,
            "s":1e-9,
        }
        # starting_alloc = numba.core.runtime.rtsys.get_allocation_stats()[0]
        times = []
        for _ in range(repeats):
            start = timer()
            func()
            end=timer()
            times.append(end-start)

        times = np.array(times)
        mean = np.mean(times) * ranges[mode]
        stdev = np.std(times * ranges[mode])
        median = np.median(times) * ranges[mode]
        max_t = min(mean + 3*stdev,np.amax(times) * ranges[mode])
        print(f"Mean: {mean:.2f} {mode}, median: {median:.2f} {mode}, stdev: {stdev:.2f} {mode}")
        histogram(times*ranges[mode], bins=50,
                  bins_min=max(mean- 3*stdev, 0),
                  x_max= max_t,
                  height = 3,
                  color = True,
                  y_unit=" freq",
                  x_unit=mode,
                  )
        # final_alloc = numba.core.runtime.rtsys.get_allocation_stats()[0]
        # print(f"Mean numba allocations: {(final_alloc - starting_alloc)/repeats:.0f}")
    run_benchmark()

@njit(fastmath = True, cache=True)
def numba_rodrigues_jac(r, out):
    """
    A numba remplementation of the opencv method of defining the jacobean, from:
    https://github.com/opencv/opencv/blob/be1373f01a6bcdc40e4a397cfb266338050cc195/modules/calib3d/src/calibration.cpp#L251
    """

    theta = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2 + r[2, 0] ** 2)

    if theta < 1e-10:
        out[:] = 0 
        out[5] = out[15] = out[19] = -1
        out[7] = out[11] = out[21] = 1
        return

    i_theta = 0 if theta == 0 else 1/theta

    ct = math.cos(theta)
    ct_1 = 1 - ct
    st = math.sin(theta) 

    x,y,z = r[0,0]*i_theta, r[1,0]*i_theta, r[2,0]* i_theta

    rrt = [x*x, x*y, x*z, x*y, y*y, y*z, x*z, y*z, z*z]
    r_x = [  0, -z,  y, # 
             z,  0, -x, #
            -y,  x,  0] #

    eye = [1, 0, 0, 0, 1, 0, 0, 0, 1]

    drrt = [x+x, y, z, y, 0, 0, z, 0, 0,
          0,   x, 0, x, y+y, z, 0, z, 0,
          0,   0, x, 0, 0, y, x, y, z+z]

    d_r_x_ = [0, 0, 0, 0, 0, -1, 0, 1, 0,
              0, 0, 1, 0, 0, 0, -1, 0, 0,
              0, -1, 0, 1, 0, 0, 0, 0, 0]

    for i, ri in enumerate([x,y,z]):
        a0 = -st*ri
        a1 = (st - 2*ct_1*i_theta)*ri
        a2 = ct_1*i_theta
        a3 = (ct - st*i_theta)*ri 
        a4 = st*i_theta
        for k in range(9):
            out[i*9+k] = a0*eye[k]  +  a1*rrt[k] + a2*drrt[i*9+k] + a3*r_x[k] + a4*d_r_x_[i*9+k];

memory = np.zeros(27)
val0 = np.ones((3,1))
val1 = np.ones(3)

m0 = lambda :numba_rodrigues_jac(val0, memory)
m1 = lambda :cv2.Rodrigues(val1)
# print("manual computation")
# print(np.array2string(memory.reshape((3,9)), precision=2))
#
# print("\n Opencv Standard")
# _, jac = cv2.Rodrigues(np.ones(3))
# print(np.array2string(jac, precision=2))
m0()

benchmark(m0, repeats=10000, mode="us")
benchmark(m1, repeats=10000, mode="us")

