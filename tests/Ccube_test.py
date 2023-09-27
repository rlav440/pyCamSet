from pyCamSet import Ccube
import numpy as np
from pyCamSet.calibration_targets.shape_by_faces import make_tforms





def test_cube():

    test = Ccube(length=40, n_points=5)
    test.plot()

if __name__ == "__main__":
    
    #test_cube()
    make_tforms(
        np.array([
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
        ]).astype(float),
        "cube"
    )
    print("Hi")