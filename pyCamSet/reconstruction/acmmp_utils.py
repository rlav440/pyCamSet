from dataclasses import dataclass
from io import TextIOWrapper
import numpy as np

# make and print a Ccube target
@dataclass
class ReconParams:
    def __init__(
            self, mindist=0.1, maxdist=0.8, steps=192, minangle=3, maxangle=45,
            max_n_view=9
    ):
        self.mindist = mindist
        self.maxdist = maxdist
        self.steps = steps
        self.minangle = minangle
        self.maxangle = maxangle
        self.max_n_view = max_n_view


def write_pair_file(f: TextIOWrapper, pair_list):
    f.write(f"{int(len(pair_list))}" + '\n')
    for idi, list_vals in enumerate(pair_list):
        f.write(f"{idi}" + '\n')
        line_string = f"{len(list_vals)} "
        line_string += " ".join([f"{cam_id} 1" for cam_id in list_vals])
        f.write(line_string + '\n')
    return

def get_v_vec(ext):
    return ext[:3,:3] @ np.array([0,0,1])

def calc_pairs(c_vec, r_param: ReconParams, rng=None, pick_closest=False):
    if rng is None:
        rng = np.random.default_rng()
    c_vec /= np.linalg.norm(c_vec, axis=1, keepdims=True)
    t = c_vec[None, ...] * c_vec[:, None]
    ang = np.arccos(np.sum(t, axis=-1)) * 180 / np.pi
    mask = np.logical_and(
        ang > r_param.minangle, ang < r_param.maxangle
    )
    returned_pairs = []
    for idx, masklet in enumerate(mask):
        valid_points = np.where(masklet)[0]
        if len(valid_points) < r_param.max_n_view:
            returned_pairs.append(valid_points)
        else:
            if not pick_closest:
                returned_pairs.append(
                    rng.choice(valid_points, r_param.max_n_view)
                )
            else:
                # pick the closest
                dists_sorted = np.argsort(ang[idx, valid_points])
                returned_pairs.append(
                    valid_points[dists_sorted][:r_param.max_n_view]
                )
    return returned_pairs
