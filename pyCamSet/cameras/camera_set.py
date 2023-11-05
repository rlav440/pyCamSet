from __future__ import annotations

import numbers
import cv2
import numpy as np
from numpy.linalg import norm
import pyvista as pv
from pathlib import Path
from copy import deepcopy
from matplotlib import pyplot as plt


from pyCamSet.cameras.camera import Camera
from pyCamSet.utils.visualisation import visualise_calibration
from pyCamSet.utils.general_utils import get_subfolder_names
from pyCamSet.utils.general_utils import get_close_square_tuple, glob_ims_local


from pyCamSet.optimisation.compiled_helpers import nb_triangulate_full
from pyCamSet.utils.saving import save_camset
from pyCamSet.reconstruction.acmmp_utils import ReconParams, write_pair_file, calc_pairs


def make_cam_dict(camera_names:list, extrinsic_matrices:list, intrinsic_matrices:list,
                  dist_coefs=None, res=None):
    """
    A function that takes a list of camera parameters, and creates a dictionary
    of Camera objects from those parameters., filing in defaults.
    
    :param camera_names: The names of the cameras to use as params for the dict
    :param extrinsic_matrices: A list of matrices representing the camera positions
    :param intrinsic_matrices: A list of intrinsic matrices of the camera as per the pinhole model
    :param dist_coefs: A list of parameter arrays containing 5 parameter Brown-Conrady models.
    :param res: a list of the represent at
    :return: A dictionary of Camera objects.
    """

    if res is None:
        res = [768, 1024]
    camera_dict = {}
    if dist_coefs is None:
        dist_coefs = [np.array([0,0,0,0,0])] * len(camera_names)

    for name, ext, int, dist, r in zip(camera_names,
                                    extrinsic_matrices,
                                    intrinsic_matrices,
                                    dist_coefs,
                                    res,
                                       ):
        camera_dict[name] = Camera(extrinsic=ext,
                                   intrinsic=int,
                                   res=r,
                                   distortion_coefs=dist,
                                   name=name,
                                   )

    return camera_dict

class CameraSet:
    """
    This base class represents a set of fixed cameras, typically a multi camera imaging rig.
    It provides methods that are used for calibration, reconstruction and detection.
    """

    def __init__(self,
                 camera_names:list[str]|None=None,
                 extrinsic_matrices:list[np.ndarray]|None=None,
                 intrinsic_matrices:list[np.ndarray]|None=None,
                 distortion_coefs:list[np.ndarray]|None=None,
                 res:list[list or np.ndarray]|None=None,
                 camera_dict: dict[str|int,Camera]|None=None,
                 ):
        """
        :param camera_names: Names for cameras
        :param extrinsic_matrices: Extrinsic list
        :param intrinsic_matrices: Intrinsic parameters as per pinhole camera model.
        :param distortion_coefs:
        :param res: resolution of the used cameras.
        :param camera_dict: A preset up dictionary of cameras.

        """

        self.calibration_result = None
        self.calibration_handler = None
        self.calibration_jac = None
        self.calibration_params = None 
        self._cam_list: list|None = None
        self._cam_dict: dict|None = None
        self.n_cams = None

        all_none = all(
            [v is None for v in [camera_names, extrinsic_matrices, res, intrinsic_matrices, distortion_coefs, camera_dict]]
        )

        all_good = all(
            [v is not None for v in [camera_names, extrinsic_matrices, res, intrinsic_matrices, distortion_coefs]]
        )

        if all_none:
            return
        if camera_dict is None:
            if not all_good:
                raise ValueError("Initialising a CameraSet requires names, extrinsic, intrinsic, distortion coefficents and resolutions")
            self._cam_dict = make_cam_dict(camera_names,
                                           extrinsic_matrices,
                                           intrinsic_matrices,
                                           dist_coefs=distortion_coefs,
                                           res=res,
                                           )
        else:
            self._cam_dict = camera_dict

        self.ind = 0
        self.__update()

    def __update(self):
        """
        updates the camera set parameters after a change in the added Cameras
        """
        self._cam_list = list(self._cam_dict.values())
        self.n_cams = len(self._cam_list)

    def get_n_cams(self):
        """
        Gets the number of cameras in the CameraSet
        """
        return len(self._cam_list)

    def make_subset(self, inp, cam_key = None) -> CameraSet:
        """
        Returns a subset of the Cameras in the CameraSet as a new CameraSet

        :param inp: the slice or list to run over
        :param cam_key: a key for the camera names, optional. If used, the dictionary and camera lists are reduced only to the names containing this key.
        Returns: A sliced camset

        """
        new_camset = CameraSet()
        if cam_key is None:

            if isinstance(inp, slice):
                cam_list = self._cam_list[inp]
                cam_names = list(self._cam_dict.keys())[inp]
            elif isinstance(inp, list):
                cam_list = [self._cam_list[idx] for idx in inp]
                names = list(self._cam_dict.keys())
                cam_names = [names[idx] for idx in inp]
            else:
                raise ValueError(f"{inp} is not a valid subset identifier")
            cam_dict = {key:cam for key, cam in zip(cam_names, cam_list)}
            new_camset._cam_dict=cam_dict
            new_camset._cam_list=cam_list
            new_camset.n_cams = new_camset.get_n_cams()
            return new_camset

        else:
            subset = [[key, value] for key, value in self._cam_dict.items() if
                      cam_key in key]
            if not subset:
                raise ValueError(f"{cam_key} found no matching camera names")

            init_key, init_list = map(list, zip(*subset))

            if isinstance(inp, slice):
                cam_list = init_list[inp]
                cam_names = init_key[inp]
            elif isinstance(inp, list):
                cam_list = [init_list[idx] for idx in inp]
                cam_names = [init_key[idx] for idx in inp]
            else:
                raise ValueError(f"{inp} is not a valid subset identifier")

            cam_dict = {key: cam for key, cam in zip(cam_names, cam_list)}
            new_camset._cam_dict = cam_dict
            new_camset.__update()
            return new_camset

    def __getitem__(self, input) -> Camera|CameraSet:
        if isinstance(input, list) or isinstance(input, slice):
            return self.make_subset(input)
        if isinstance(input, numbers.Number):
            if input in self._cam_dict: # preferentially go to dict if given a key number that exists as a key
                return self._cam_dict[input]
            return self._cam_list[input]
        return self._cam_dict[input]

    def __setitem__(self, key, value: Camera):
        self._cam_dict[key] = value
        self._cam_list = list(self._cam_dict.values())
        self.n_cams = self.get_n_cams()

    def __iter__(self):
        self.ind = 0
        return self

    def __next__(self) -> Camera:
        if self.ind < len(self._cam_list):
            cam = self._cam_list[self.ind]
            self.ind += 1
            return cam
        else:
            raise StopIteration

    def __eq__(self, other: CameraSet):
        """
        Implements equality for camera sets. Uses the camera names in the camera, then compares
        cameras with the same name to check for equality.

        :param other: The other object to compare. False if not a camera.
        """
        if not isinstance(other, CameraSet):
            return False
        #check the camera names, without ordering
        if not set(self.get_names()) == set(other.get_names()):
            return False
        sub_camera_comps = [self[cam] == other[cam] for cam in self.get_names()]
        if not all(sub_camera_comps):
            return False
        return True

    def write_to_txt(self, loc: Path, r: ReconParams, ims:list[np.ndarray]|None = None, mode='MVSnet'):
        """
        Writes an entire camera set to some form of defined camera structure.
        Currently only MVSnet is defined.

        :param loc: the file location to write to
        :param r: the reconstruction parameters to follow
        """
        if not mode == 'MVSnet':
            raise NotImplementedError

        for cam_n, cam in enumerate(self):
            cam_loc = loc/f"{cam_n:08}_cam.txt"
            cam.to_MVSnet_txt(cam_loc, (r.mindist, r.maxdist), r.steps)

        if ims is not None:
            im_loc = loc.parent/'images'
            im_loc.mkdir(exist_ok=True)
            for idx, im in enumerate(ims):
                cv2.imwrite(str(im_loc/f"{idx:08}.jpg"), im)

        cvwc = np.array(
            [cam.view for cam in self]
        )
        pairs = calc_pairs(cvwc, r)
        with open((loc.parent) / "pair.txt", 'w') as f:
            write_pair_file(f, pairs)


    def return_view_overlaps(self):
        """
        :return: A meshed, overlapping set of view cones
        """
        raise NotImplementedError('Mesh intersections fail on large camera '
                                  'numbers')

        pv.set_plot_theme("Document")
        v_cones = [cam.get_viewcone(view_len=0.5, triangle=True)
                   for cam in self]

        for i in range(len(v_cones) - 1):
            logging.info(f"performing {i + 1}th intersection of {len(v_cones)}")

            plot_intersection = True
            if plot_intersection:
                scene = pv.Plotter()
                scene.add_mesh(v_cones[i], opacity=0.5)
                scene.add_mesh(v_cones[i + 1], opacity=0.5)
                scene.show()

            mesh = v_cones[i + 1].boolean_union(v_cones[i])

            if mesh.n_points == 0:
                logging.info(f"Zero point mesh output, skipping viewcone "
                             f"{i - 1}")
                v_cones[i + 1] = v_cones[i - 1]
            else:
                v_cones[i + 1] = mesh

        scene = pv.Plotter
        for v in v_cones:
            scene.add_mesh(v)
        scene.show()
        final_mesh = v_cones[-1]

        final_mesh.show()

        return final_mesh

    def project_points_to_all_cams(self, points) -> list[dict[str|int,np.ndarray]]|dict[str|int, np.ndarray]:
        """
        Projects a point or list of points to all cameras.

        :param points: a list of points nx3 in world reference coordinates
        :return cam_cords: a list of camera:projection dictionaries
        """
        single_flag = False

        if isinstance(points, list):
            points = np.array(points)

        if points.ndim == 1:
            points = points[None, ...]
            single_flag = True

        all_projections = [cam.project_points(points) for cam in self._cam_list]
        projection_dictionary_list = [{} for _ in range(points.shape[0])]

        for cam_proj, cam_name in zip(all_projections, self._cam_dict.keys()):
            for dict, result in zip(projection_dictionary_list, cam_proj):
                dict[cam_name] = result

        if single_flag:
            return projection_dictionary_list[0]

        return projection_dictionary_list

    def multi_cam_triangulate(self, to_reconstruct: list[dict] or dict or np.ndarray,
                              return_used = False):
        """
        A lsq minimised triangulation of camera point locations to reconstruct.
         Automatically identifies points with shared visibility

        :param points: dictioniry of cam name key and camera coordinate values
                    or a list of dictionairies
                    Alternatively, the returned data from a TargetDetection.get_data method
        :return: world projected point or array of points

        """
        names = self.get_names()
        if isinstance(to_reconstruct, dict):
            to_reconstruct = [to_reconstruct]
        if isinstance(to_reconstruct, list):
            #make it like the internal data structure of a detection
            data = []
            for idx, reconstructlet in enumerate(to_reconstruct):
                bulklet = []
                for  cam_name, datum in reconstructlet.items():
                    cam_ind = names.index(cam_name)
                    array = [cam_ind, 0, idx, datum[0], datum[1]]
                    bulklet.append(array)
                data.append(bulklet)
            data = np.concatenate(data, axis=0)

        else:
            data = to_reconstruct
        _, inv, count = np.unique(
            data[:, 1:-2], axis=0, return_inverse=True, return_counts=True
        )
        viable_mask = count > 1
        reconstructable_data = data[viable_mask[inv]]
        _, im_index, im_counts = np.unique(reconstructable_data[:, 1:-2], axis=0, return_index=True, return_counts=True)
        start_ind = np.append(0, np.cumsum(im_counts[np.argsort(im_index)]))

        #build the projection matricies
        proj = np.array([cam.proj for cam in self])
        dists = np.array([cam.distortion_coefs for cam in self])
        intr = np.array([cam.intrinsic for cam in self])
        reconstructed = nb_triangulate_full(reconstructable_data, proj, start_ind, intr, dists)
    
        if return_used:
            # for every final point find the inds it pulled from in the final array
            where_mask = np.where(viable_mask[inv])[0]
            working_array = []
            for idx in range(len(start_ind) - 1):
                len_pts = start_ind[idx+1] - start_ind[idx]
                grabbed, where_mask = where_mask[:len_pts], where_mask[len_pts:]
                working_array.append(grabbed)

            return reconstructed, reconstructable_data, working_array
        return reconstructed


    def plot_np_array(self, points: np.ndarray | list):
        """
        A shorthand plotting function for plotting raw np arrays with reference to the cameras.

        :param points: a numpy array or list of numpy arrays with dimension nx3 to draw.
        :return:
        """
        if not isinstance(points, list):
            points = [points]
        pt = [pv.PolyData(point) for point in points]
        self.plot(additional_mesh=pt)

    def get_camera_meshes(self, viewcone=None, scale=None):
        """
        :param viewcone: whether to give camera viewcones as a mesh
        :param scale: the scale of the camera models.
        :returns: A list of pyvista mesh objects for every camera in the camera set
        """
        if scale is None:
            scale = np.max([np.linalg.norm(cam.position) for cam in self]) * 0.1

        cam_meshes = [cam.get_mesh(scale) for cam in self]

        if viewcone is None:
            return cam_meshes

        else:
            view_cones = [cam.get_viewcone(view_len=viewcone) for cam in self]

        return cam_meshes, view_cones

    def get_scene(self, scale_factor=0.3/8, view_cones=None, scene: pv.Scene=None) -> pv.Scene:
        """
        Returns a pyvista scene containing the camera meshes.

        :param scale_factor: the scale of the camera meshes to use
        :param view_cones: whether to draw camera viewcones.
        :param scene: optionally a scene to which the camera meshes will be added.
        :return: A scene containing the camera meshes.
        """
        cam_meshes, v_cones = self.get_camera_meshes(viewcone=0.15, scale=scale_factor)
        positions = np.array([cam.position for cam in self])
        pv.set_plot_theme('Document')
        if scene is None:
            scene = pv.Plotter()
        for mesh in cam_meshes:
            scene.add_mesh(mesh, style='wireframe', reset_camera=True)
        if view_cones is not None:
            for v_con in v_cones:
                scene.add_mesh(v_con, opacity=0.05, color='g')
        scene.add_point_labels(positions, list(self._cam_dict.keys()))

        # also visualise the origin of the coordinate system
        p0 = np.array([0, 0, 0])
        px = np.array([0.05, 0, 0])
        py = np.array([0, 0.05, 0])
        pz = np.array([0, 0, 0.05])

        connect = np.hstack(([2, 0, 1],
                             [2, 1, 2]))

        lx = np.hstack((p0, px))
        ly = np.hstack((p0, py))
        lz = np.hstack((p0, pz))

        polyx = pv.PolyData(lx)
        polyx.lines = connect

        polyy = pv.PolyData(ly)
        polyy.lines = connect

        polyz = pv.PolyData(lz)
        polyz.lines = connect

        cols = ['red', 'green', 'blue']

        for mesh, col in zip([polyx, polyy, polyz], cols):
            scene.add_mesh(mesh, color=col)

        return scene

    def plot(self, scale_factor=None,
             additional_mesh: pv.PolyData|list[pv.PolyData]|None=None,
             view_cones=False):
        """
        Draws a 3D plot of the cameras and any additional meshes

        :param scale_factor: the scale of the camera meshes to use
        :param additional_mesh: a mesh or list of meshes to add to the scene
        :param view_cones: whether to draw camera viewcones.
        """

        cam_meshes, v_cones = self.get_camera_meshes(viewcone=0.15, scale=scale_factor)
        positions = np.array([cam.position for cam in self])
        # view_vectors = np.array([cam.view for cam in self.cam_list])
        # view_pos = view_vectors * scale_factor # + positions

        pv.set_plot_theme('Document')
        scene = pv.Plotter()

        for mesh in cam_meshes:
            scene.add_mesh(mesh, style='wireframe', line_width=1, color='k')
        if view_cones:
            for v_con in v_cones:
                scene.add_mesh(v_con, opacity=0.05, color='g')

        scene.add_point_labels(positions, list(self._cam_dict.keys()))
        # scene.add_arrows(cent=positions, direction=view_pos)

        # also visualise the origin of the coordinate system
        p0 = np.array([0, 0, 0])
        px = np.array([0.05, 0, 0])
        py = np.array([0, 0.05, 0])
        pz = np.array([0, 0, 0.05])

        connect = np.hstack(([2, 0, 1],
                             [2, 1, 2]))

        lx = np.hstack((p0, px))
        ly = np.hstack((p0, py))
        lz = np.hstack((p0, pz))

        polyx = pv.PolyData(lx)
        polyx.lines = connect

        polyy = pv.PolyData(ly)
        polyy.lines = connect

        polyz = pv.PolyData(lz)
        polyz.lines = connect

        cols = ['red', 'green', 'blue']

        for mesh, col in zip([polyx, polyy, polyz], cols):
            scene.add_mesh(mesh, color=col)

        if additional_mesh is not None:
            if not isinstance(additional_mesh, list):
                additional_mesh = [additional_mesh]

            #create a colourscheme for the additional meshes.
            cls = len(additional_mesh)
            #colours = colourmap_to_colour_list(cls, plt.get_cmap('Set1'))
            colours = ['r', 'g', 'b'] + ['b'] * 100
            colours = colours[:cls]
            for mesh, col in zip(additional_mesh, colours):
                if not isinstance(mesh, CameraSet):
                    # if mesh has no colour
                    if mesh.active_scalars is None:
                        scene.add_mesh(mesh, col, opacity=0.1)
                    else:
                        scene.add_mesh(mesh,
                                       render_lines_as_tubes=True,
                                       line_width=4,
                                       point_size=0.7,
                                       rgb=True)
                else:
                    cams = mesh.get_camera_meshes()
                    for mini_mesh in cams:
                        scene.add_mesh(mini_mesh,
                                       style='wireframe',
                                       line_width=2,
                                       color=col)
                    if view_cones:
                        for v_con in v_cones:
                            scene.add_mesh(v_con,
                                           opacity=0.05,
                                           color=col)

        scene.show()

    def draw_camera_distortions(self):
        """
        Draws a quiver plot of the distortion of all cameras in the camera set.
        """
        to_draw = get_close_square_tuple(self.n_cams)
        fix, axes = plt.subplots(*to_draw)
        for ax, cam in zip(axes.flatten(), self):
            cam.view_sensor_distortion(ax)
        plt.show()

    def get_cam_dict(self):
        """
        :return: the underlying camera dictionary.
        """
        return self._cam_dict

    def get_cam_list(self):
        """
        :return: the underlying camera list.
        """
        return self._cam_list

    def get_names(self):
        """
        :return: a list of the camera names in the camera set.
        """
        return list(self._cam_dict.keys())

    def save(self, floc: Path|str="saved_cameras.camset"):
        """
        Saves the camera set to a file using .json esque encoding.

        :param floc: The file location to save to
        """
        if isinstance(floc, str):
            floc = Path(floc)
        save_camset(self, floc)

    def set_resolutions_from_file(self, floc: Path):
        """
        Populates the resolutions of the CameraSet given a folder of images from
            the cameras of the CameraSet.

        :param floc: the file location of the images to use.
        """

        file_names = get_subfolder_names(f_loc=floc)
        cam_names = self.get_names()

        if not set(file_names) == set(cam_names):
            raise ValueError(f'Subfolders of the file {floc} do not match the '
                             f'current camera names')

        for cam_name in cam_names:
            im_locs = glob_ims_local(floc/cam_name)
            temp_im = cv2.imread(str(im_locs[0]))
            self[cam_name].res = np.array((temp_im.shape[1], temp_im.shape[0])) #CV2 ordering



    def scale_set_2n(self, d_factor):
        """
        Scales all cameras in the set to 2^d_factor

        :param d_factor: the power of 2 to used in the downsampling
        """
        for cam in self._cam_list:
            cam.scale_self_2n(d_factor)

    def transform(self, transformation_matrix, in_place=True) -> None|CameraSet:
        """
        Transforms all cameras in the set by a transformation matrix.

        :param transformation_matrix: A 4x4 homogenous transformation matrix
        :param in_place: whether to transform the camera set in place or not.
        :return: Optionally returns a new camera set with the transformation applied.
        """
        if not in_place:
            temp_camset = deepcopy(self)
            return temp_camset.transform(transformation_matrix)

        for cam in self._cam_list:
            cam.transform(transformation_matrix)

    def set_reference_cam(self, cam_id):
        """
        Sets a reference camera to be the centre of world coordinates and transforms the camera set around this

        :param cam_id: the camera to use as the reference
        """

        ref_cam: Camera = self[cam_id]
        ref_tform = np.linalg.inv(ref_cam.extrinsic)
        self.transform(ref_tform)

    def __add__(self, other: CameraSet) -> CameraSet:
        if not isinstance(other, CameraSet):
            raise ValueError('Can only add together camera sets')
        intersection = self._cam_dict.keys() & other._cam_dict
        if intersection:
            raise ValueError('Camera sets share camera names so cannot be added')
        self._cam_dict = {**self._cam_dict, **other._cam_dict}
        self._cam_list = list(self._cam_dict.values())
        return self

    def set_calibration_history(self,
                                optimisation_results,
                                param_handler,
        ):
        """
        Camera sets are representations of data, so provides methods to store the data
        needed to generate a camera set into the camera set itself.

        :param optimisation_results: the results of the optimisation from scipy lsq
        :param param_handler: the parameter handler used to manage the optimisation
        """
        self.calibration_params = optimisation_results['x']
        self.calibration_result = optimisation_results['fun']
        self.calibration_jac = optimisation_results['jac']
        self.calibration_handler = param_handler

    def visualise_calibration(self):
        """
        Displays the calibration results of the camera set.
        """
        if self.calibration_params is None:
            raise ValueError('The camera set has no calibration data saved')
        optim_results = {
                'x':self.calibration_params,
                'err':self.calibration_result
            }

        visualise_calibration(
            optim_results,
            self.calibration_handler,
        )



