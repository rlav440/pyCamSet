import numpy as np

from pyCamSet.calibration_targets.target_Ccube import TFORMS
from pyCamSet.calibration_targets.shape_by_faces import print_formatted_transforms
from pyCamSet.utils import make_4x4h_tform
from pyCamSet.utils.general_utils import ext_4x4_to_rod
from pyCamSet.optimisation.compiled_helpers import n_estimate_rigid_transform



#define the initial rotaiton that we will be useing 

p_pycamset = np.array([
    [0,0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]
])

p_pycamera = np.array([
    [0,1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0]
])


tform_est = n_estimate_rigid_transform(p_pycamset, p_pycamera)
tforms  = []
for t in TFORMS:
    h4 = make_4x4h_tform(*t)
    new_h4 = h4 @ np.block([[tform_est[0], tform_est[1][:, None]], [0,0,0,1]])
    tforms.append(ext_4x4_to_rod(new_h4))

print_formatted_transforms(tforms)





