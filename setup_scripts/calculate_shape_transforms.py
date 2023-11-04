import numpy as np

from pyCamSet.calibration_targets.shape_by_faces import make_tforms, print_formatted_transforms



def make_cube_tforms():
      
    faces = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ]).astype("double")
    tforms = make_tforms(faces, "cube")
    order = [1, 3, 4, 0, 5, 2]
    tforms = [tforms[a] for a in order]
    print_formatted_transforms(tforms)

if __name__ == "__main__":
    make_cube_tforms() 


