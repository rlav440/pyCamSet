import numpy as np
import pyvista as pv
from pyCamSet.optimisation.compiled_helpers import n_htform_broadcast_prealloc

class FaceToShape:
    """
    This class defines a geometric shape by taking (u,v, ... w) faces, 6dof transformations for them
    It also handles how these faces can then be used to draw objects.
    """

    def __init__(self, face_local_coords, face_transforms):

        assert np.array(face_transforms.shape)[:-1] == np.array(face_transforms.shape)[:-2] (
            "The dimensionality of the number of transforms and faces should be equal")

        self.face_local_coords = face_local_coords
        self.face_transforms = face_transforms
        self.point_data = np.empty_like(face_local_coords)
        n_htform_broadcast_prealloc(self.face_transforms, self.point_data)


    def draw_meshes(self, face_images, face_corners: list or np.array):
        """
        Draws a mesh given the current face set up
        :param face_images: The face images.
        :param face_corners: the location of the face corners, in clockwise order.
        :param bounds: optional bounds to override defaults.
        :return: created mesh models.
        """

        # get the bounds from the min and the max of the face corners
        meshes = []
        for face_corner, face_transform in zip(face_corners, self.face_transforms):
            num_elems = len(face_corner)
            connections = [num_elems] + list(range(num_elems))
            new_mesh = pv.PolyData(face_corner, faces=connections)
            new_mesh.texture_map_to_plane(use_bounds=True, inplace=True)
            new_mesh.transform(face_transform)
            meshes.append(new_mesh)

        scene = pv.Scene()
        for mesh, texture in zip(meshes, face_images):
            scene.add_mesh(mesh, pv.np_to_texture(face_images))
        scene.show()

