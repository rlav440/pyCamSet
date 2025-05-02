import pyvista as pv        
import numpy as np
from pyCamSet.optimisation.compiled_helpers import n_estimate_rigid_transform


def make_h(rot, trans):
    """
    Turns a 2D transformation and rotation vector into the stacked homogenous transform.

    :return: homogenous transform matrix
    """
    return np.block([[rot, trans[..., None]], [ 0, 0, 1]])

def make_ragged_faces(face_array):
    """
    unrolls a 1d face array to a ragged list of the points of each face

    :param face_array: the array defining the faces of the geometry
    :return ragged_faces: a list of lists containing the points associated with each face
    """
    ragged_faces = []
    parsing = True
    array_pointer = 0
    while parsing:
        n_faces = face_array[0]
        array_pointer += 1
        ragged_faces.append(face_array[array_pointer:(array_pointer + n_faces)])
        array_pointer += n_faces
        if array_pointer >= len(face_array):
            parsing = False
    return ragged_faces

def print_net_tforms(tforms):
    
    print("NET_FORMS=[")
    for t in tforms:
        tt = t.flatten()
        print(f"\t[[{tt[0]},{tt[1]},{tt[2]}], [{tt[3]},{tt[4]},{tt[5]}],[{tt[6]},{tt[7]},{tt[8]}]],")
    print("]")

def make_net_tforms(base_shape, face_connectivity, connections):
    """
    Defines a net by unfolding the object face by face.
    The first fast is taken as a starting face, then every other face must be connected.
    The structure is:
        [n connnected, base, connected_face_0, ... connected_face_n, ....]

    :param base_shape: a definition of a single face (asuming all are the same) or definitions of each face
    :param face_connectivitiy: the connectivity of the points defining the faces.
    :param draw_res: the drawing resolution of the single face
    :param connections: the connection structure defining the desired net unwinding
    :return tforrms: the desired transform for each face.
    """
    parsing = True
    cpoint = 0
    added_faces = set()
    ragged = make_ragged_faces(face_connectivity)
    for rag in ragged:
        print(rag)

    tforms: list = [None for _ in range(len(ragged))]

    while parsing:
        num_connect = connections[cpoint]
        base_face = connections[cpoint+1]
        bfp = list(ragged[base_face])
        if cpoint == 0:
            added_faces.add(base_face)
            tforms[base_face] = np.eye(3)
        if not base_face in added_faces:
            raise ValueError(f"Attempted to add faces to {base_face} without first defining the location of {base_face}.")
        cpoint += 2
        for _ in range(num_connect):

            connected_face = connections[cpoint]
            cfp = list(ragged[connected_face])
            print(set(bfp))
            print(set(cfp))
            print(set(bfp).intersection(cfp))
            shared_points = list(set(bfp).intersection(set(cfp)))
            if len(shared_points) == 2:
                if connected_face in added_faces:
                    raise ValueError(f"Face {connected_face} was added to the tree multiple times.")
                added_faces.add(connected_face) 
                pb = np.array([base_shape[bfp.index(shared_points[0])],base_shape[bfp.index(shared_points[1])]])
                cb = np.array([base_shape[cfp.index(shared_points[0])],base_shape[cfp.index(shared_points[1])]])
                rtform = n_estimate_rigid_transform(cb, pb)
                rtform = make_h(*rtform)
                tforms[connected_face] = tforms[base_face] @ rtform
            else:
                raise ValueError(f"faces {base_face} and {connected_face} do not share enough points to estimate the transformation between them")
            cpoint += 1
            if cpoint >= len(connections):
                parsing = False
    return tforms


if __name__ == "__main__":
    import pyvista as pv
    from matplotlib import pyplot as plt
    
    from pyCamSet.utils.general_utils import h_tform, make_4x4h_tform

    #use an existing set of transforms for the cube faces to define the connectivity of the net creation
    TFORMS = [
        ([2.22144147, 2.22144147, 0.        ], [-0.5, -0.5,  0.5]),
        ([-1.57079633,  0.        ,  0.        ], [-0.5, -0.5,  0.5]),
        ([-1.20919958, -1.20919958,  1.20919958], [ 0.5, -0.5,  0.5]),
        ([ 0.        ,  2.22144147, -2.22144147], [0.5, 0.5, 0.5]),
        ([0.        , 0.        , 1.57079633], [ 0.5, -0.5, -0.5]),
        ([1.20919958, 1.20919958, 1.20919958], [-0.5, -0.5, -0.5]),
    ]
    converted_tforms = []
    points_already_existing = []
    bf = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ]).astype("double")

    known_points = []
    faces = []
    for T in TFORMS:
        #each tform defines a face
        local_face = [4]
        tform = make_4x4h_tform(*T)
        points = h_tform(bf, tform) 
        for point in points:
            if len(known_points) == 0:
                known_points.append(point)
                local_face.append(0)
                continue
            close_mask = np.linalg.norm(np.array(known_points) - point, axis=1) < 1e-5
            if not np.any(close_mask):
                local_face.append(len(known_points))
                known_points.append(point)
                continue
            local_face.append(np.where(close_mask)[0][0])
        faces.extend(local_face)

    base_shape = np.array([
        [1.,1.],
        [1.,0.],
        [0.,0.],
        [0.,1.],
    ])

    connections = [ 
        4, 0, 1, 2, 3, 5, # define faces 2,3,4 and 5 by relation to face 0
        1, 2, 4, # define face 4 by it's connection to face 2
        ]
    
    tforms = make_net_tforms(base_shape, faces, connections) 

    mp  = np.mgrid[2:9,2:9]
    points = np.stack([mp[0].flatten(), mp[1].flatten()]).T/10
    ctrs = [h_tform(points, t) for t in tforms]

    for c in ctrs:
        clabel = np.arange(0, len(c))
        plt.scatter(c[:, 0], c[:, 1], c=clabel, cmap='jet')

    print_net_tforms(tforms)

    plt.gca().set_aspect("equal")
    plt.show()





