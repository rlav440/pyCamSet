import numpy as np
from PIL import Image, ImageDraw
import cv2
from pyCamSet.utils.general_utils import h_tform
import pyvista as pv
from pyCamSet.optimisation.compiled_helpers import n_htform_broadcast_prealloc, n_estimate_rigid_transform

make_shape = {
    "cube":lambda size: pv.Cube(x_length=size, y_length=size, z_length=size),
    "Icosahedron":pv.Icosahedron, 
}

def bound_pts(face, res):
    max_bounds = np.max(face, axis=0)
    return face/max_bounds * res

def print_formatted_transforms(tforms):
    """
    formats ain input array of (rotation, translation tuples) as printed strings
    
    :param tforms: the input transformations to print
    """
    print("TFORMS = [")
    for tform in tforms:
        s0 = np.array2string(tform[0].squeeze(), precision=3, separator=",")
        s1 = np.array2string(tform[1].squeeze(), precision=3, separator=",")
        print(f"\t({s0},{s1}),") 
    print("]")


def make_tforms(base_face, shape):
    """
    Takes an input face and a shape name, then recovers the transformation
    to each of the faces of the object. 

    :param base_face: the basic face of the object. Points oriented clockwise from bottom left corner.
    """
    #take the face corners and add them 
    #try assume the face is the right shape: push the  
    size = np.max(base_face[:, 1]) -  np.min(base_face[:, 1])
    polyhedra: pv.PolyData = make_shape[shape](size)
    poly_points = polyhedra.points
 
    tforms = []
    for face in polyhedra.regular_faces:
        local_face = np.array([poly_points[f] for f in face])
        r, t = n_estimate_rigid_transform(base_face, local_face)
        rvec, _ = cv2.Rodrigues(r) 
        tforms.append((rvec,t))
    return tforms

class FaceToShape:
    """
    This class defines a geometric shape by taking (u,v, ... w) faces, 6dof transformations for them
    It also handles how these faces can then be used to draw objects.
    """

    def __init__(self, face_local_coords, face_transforms, scale_factor=1.):
        """
        A function that takes some base faces, the transformations of the faces, and an optional scale factor.
        The scale factor is used to scale before and after applying the transformations.
        This is a convineience factor that allows the same transformations to define a scaled family of polygons.

        :param face_local_coords: the local coordinates of the corners of the face in anticlockwise order.
            If there is one nx3 face given, it is used as the base face for all transformations.
        :param face_transforms: 4x4 representations of the transformations for a face to the output shape
        :param scale_factor: An optional scaling factor that transforms the result.
        """

        if isinstance(face_local_coords, list):
            face_local_coords = np.array(face_local_coords)
            
        if isinstance(face_transforms, list):
            face_transforms = np.array(face_transforms)

        self.same_face = False

        if face_local_coords.ndim == 2: # support a single face input for easy use 
            nfaces = np.prod(face_transforms.shape[:-2])
            face_local_coords = np.tile(face_local_coords[None, ...], [nfaces, 1, 1]).reshape(
                (*face_transforms.shape[:-2], *face_local_coords.shape)
            )
            self.same_face = True

        
        ppf = face_local_coords.shape[-2] #points per face
        
        self.sf = scale_factor
        self.face_local_coords = face_local_coords
        self.ur_flocal = self.face_local_coords.reshape((-1, ppf, 3))
        self.face_transforms = face_transforms
        self.ur_ftform = self.face_transforms.reshape((-1, 4, 4))
        self.point_data = np.empty_like(face_local_coords)
        self.ur_pdata = self.point_data.reshape((-1, ppf, 3))
        for i, (tform, points) in enumerate(zip(self.ur_ftform, self.ur_flocal)):
            self.ur_pdata[i] = h_tform(transform=tform, points=points/scale_factor) * scale_factor
        return

    def draw_meshes(self, face_corners, face_images: list[np.ndarray]):
        """
        Draws a mesh given the current face set up

        :param face_images: The face images.
        :return: created mesh models.
        """
        if isinstance(face_corners, list):
            face_corners = np.array(face_corners)

        if face_corners.ndim == 2: # support a single face input for easy use 
            nfaces = self.ur_ftform.shape[0]
            face_corners = np.tile(face_corners[None, ...], [nfaces, 1, 1])
        # get the bounds from the min and the max of the face corners
        meshes = []
        for face_corner, face_transform in zip(face_corners, self.ur_ftform):
            num_elems = len(face_corner)
            connections = [num_elems] + list(range(num_elems))
            new_mesh = pv.PolyData(face_corner, faces=connections)

            new_mesh.scale(1/self.sf, inplace=True)
            new_mesh.transform(face_transform, inplace=True)
            new_mesh.scale(self.sf, inplace=True)
            #todo make this generic.
            new_mesh.texture_map_to_plane(
                origin=new_mesh.points[0],
                point_u=new_mesh.points[1],
                point_v=new_mesh.points[3], inplace=True)
            meshes.append(new_mesh)
         
        scene = pv.Plotter()
        for mesh, texture in zip(meshes, face_images):
            scene.add_mesh(mesh, 
                           texture = pv.numpy_to_texture(texture.astype(np.uint8))
                           )
        scene.add_mesh(pv.PolyData(self.point_data.reshape((-1,3))), color='r')
        # a = self.point_data.reshape((-1, 3))
        # ls = [str(i) for i in range(a.shape[0])]
        # scene.add_point_labels(a, ls)
        scene.add_axes()
        scene.show()

    def draw_net(self, net_images, net_transforms):
        """
        Draws a net to a numpy array based o the input images and the input textures.
        """
        
        # take the image
        locs = []
        nbf_list = [] #normalised bounded faces
        for face, im, tform in zip(self.face_local_coords, net_images, net_transforms):
            bf = bound_pts(face, im.shape) #normalise by height by default
            n_bf = bf/im.shape[1]
            nbf_list.append(n_bf)
            locs.append(h_tform(n_bf, tform)*im.shape[1])

        shift_to_align = 0 - np.min(locs)
        max_shape = np.max(locs)
        
        blank_canvas = np.zeros(max_shape)
        for n_bf, im, tform in zip(nbf_list, net_images, net_tforms):
            # rotate the points.
            r_bf = h_tform(n_bf, tform, fill=0) * im.shape[1] 
            bounds = np.max(r_bf), np.min(r_bf)
            p_im = im #padd the image here
            r_im = cv2.warpAffine(im, tform[:2,:2]) 
            r_im = None #crop down to the max and min containing the array.
            img = Image.new('L', im.shape, 0) # some shapes might use a sub window of the image
            ImageDraw.Draw(img).polygon(bf, outline=1, fill=1)
            r_im *= numpy.array(img)
            # map the face shape over the image, making a mask and doing a bounds check for all faces
            blank_canvas[0] += r_im
    
        return blank_canvas
