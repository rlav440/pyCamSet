import numpy as np
import cv2
from numba import njit


from pyCamSet.optimisation.compiled_helpers import numba_rodrigues_jac, numba_flat_rodrigues_INPLACE
from pyCamSet.utils.general_utils import benchmark


memory = np.zeros(27)
val0 = np.ones((3,1))
val1 = np.random.random(3)

# m0 = lambda :numba_rodrigues_jac(val0, memory)
m1 = lambda :cv2.Rodrigues(val1)

numba_flat_rodrigues_INPLACE(val1, memory[:9])

print(np.array2string(memory[:9].reshape((3,3)), precision=2))
# print("manual computation")
# print(np.array2string(memory.reshape((3,9)), precision=2))
#
# print("\n Opencv Standard")
val, jac = cv2.Rodrigues(val1)
# print(np.array2string(jac, precision=2))
print(np.array2string(val, precision=2))
# m0()

benchmark(m0, repeats=10000, mode="us")
benchmark(m1, repeats=10000, mode="us")

