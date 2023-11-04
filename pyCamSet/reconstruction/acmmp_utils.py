from dataclasses import dataclass
from io import TextIOWrapper
import numpy as np

# make and print a Ccube target
@dataclass
class ReconParams:
    """
    This is a data class that contains the expected parameters for ACMMP/mvsnet.

    """
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
    """
    Given a list of pairs and a file handler, writes that pair list to the file.

    :param f: the handle of the file to write.
    :param pair_list: a list of camera pairs to write to the file.
    """
    f.write(f"{int(len(pair_list))}" + '\n')
    for idi, list_vals in enumerate(pair_list):
        f.write(f"{idi}" + '\n')
        line_string = f"{len(list_vals)} "
        line_string += " ".join([f"{cam_id} 1" for cam_id in list_vals])
        f.write(line_string + '\n')
    return

def get_v_vec(ext):
    """
    Gets the view vector of a camera given the extrinsic

    :param ext: The extrinsic matrix of a camera.
    """
    return ext[:3,:3] @ np.array([0,0,1])

def calc_pairs(c_vec, r_param: ReconParams, rng=None, pick_closest=False):
    """
    Calclulates the likely pairs from camera view vectors.

    :param c_vec: the camera view vectors.
    :param r_param: the parameters of the reconsturction. Places limits on
    acceptable pairs.
    :param rng: an rng seed for reproducibility.
    :param pick_closest: whether to sort the cameras by angle, or use random selection
    :return pairs: a list of lists of the acceptable pairs for each camera.

    """
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
