
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
import time
from copy import copy
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_array
try:
    import pyvista as pv
    _PYVISTA_OK = True
except ImportError:  # pragma: no cover - exercised in headless installs
    pv = None
    _PYVISTA_OK = False
from typing import TYPE_CHECKING

from scipy.spatial.distance import cdist
from pyCamSet.optimisation.template_handler import TemplateBundleHandler, DEFAULT_OPTIONS
import pyCamSet.utils.general_utils as gu
import pyCamSet.optimisation.compiled_helpers as ch
import pyCamSet.optimisation.function_block_implementations as fb
from pyCamSet.optimisation.template_handler import _extrinsic_from_params
from pyCamSet.optimisation.numba_schur import ParamGroup

from pyCamSet.calibration_targets import TargetDetection
    
if TYPE_CHECKING:
    from pyCamSet.calibration_targets import AbstractTarget
    from pyCamSet.cameras import CameraSet, Camera


def find_gauge_points(points, candidates=None):
    """
    Chooses the coordinates to hold fixed to remove the seven gauge freedoms
    of a self calibration: three translations, three rotations and the scale.

    Two points, held in all three of their coordinates, remove six of them:
    the translation, the scale, and every rotation but the one about the line
    AB between them.  A single coordinate of a third point C removes that
    last one.  Rotating about AB carries C along the normal of the plane ABC,
    so the coordinate held is the axis most nearly parallel to that normal;
    a coordinate lying in the plane of ABC is unchanged by the rotation and
    would constrain nothing.  C is taken as far from AB as the data allows,
    and A and B as far apart, so that each held coordinate is the one the
    residuals respond to most strongly.

    A point that no camera saw is not a parameter of the optimisation, so
    holding it fixed removes no freedom at all; ``candidates`` is the set of
    points that may be used, which makes the gauge choice depend on the
    detections rather than on the target alone.

    :param points: the (n, 3) geometry of the target.
    :param candidates: indices of the points that may be held fixed. The
        default allows all of them.
    :return: the three point indices, and the axis of the third point.
    :raises ValueError: if the points available cannot fix a gauge.
    """
    inds = np.arange(len(points)) if candidates is None else np.asarray(candidates)
    if len(inds) < 3:
        raise ValueError(
            f"Fixing the gauge needs three points, and only {len(inds)} are available.")

    pts = points[inds]
    i0 = inds[np.argmax(np.linalg.norm(pts - np.mean(pts, axis=0), axis=1))]
    span = np.linalg.norm(pts - points[i0], axis=1)
    i1 = inds[np.argmax(span)]
    if np.max(span) == 0:
        raise ValueError("Every point available to fix the gauge is at the same location.")

    ab = (points[i1] - points[i0]) / np.max(span)
    normals = np.cross(ab, pts - points[i0])
    distance = np.linalg.norm(normals, axis=1)  # each point's distance from AB
    furthest = np.argmax(distance)
    if distance[furthest] < 1e-6 * np.max(span):
        raise ValueError(
            "The points available to fix the gauge are colinear, so no rotation "
            "about them can be constrained.")

    return (int(i0), int(i1), int(inds[furthest])), int(np.argmax(np.abs(normals[furthest])))

class StandardBundlePrimitive:
    """
    A class that contains a set of base arrays.
    These arrays contain the pose, extrinsic, intrinsic and distortion params
    that will be used to create the bundle adjustment problem.
    If a param is fixed, it can be marked as fixed in the ``*_fixed`` data structure.
    A fixed value will not be dependent on the standard parameters.
    """

    def __init__(self, poses:np.ndarray, bundle_points: np.ndarray, extr: np.ndarray, intr: np.ndarray,
                 poses_unfixed=None, bundle_points_unfixed=None, extr_unfixed=None, intr_unfixed=None, 
                 always_correct_gauge=False
                 ):

        self.extr = extr
        self.extr_unfixed = extr_unfixed if extr_unfixed is not None else np.ones(extr.shape[0], dtype=bool)
        self.intr = intr
        self.intr_unfixed = intr_unfixed if intr_unfixed is not None else np.ones(intr.shape[0], dtype=bool)
        self.bundle_pts = bundle_points
        self.bdpt_unfixed = bundle_points_unfixed if bundle_points_unfixed is not None else np.ones(bundle_points.shape[0], dtype=bool)
        #we fix the bundle points on a per point basis

        self.correct_gauge = True
        self.poses = poses
        self.poses_unfixed = poses_unfixed if poses_unfixed is not None else np.ones(poses.shape[0], dtype=bool)
        self.n_intr = intr.shape[1]
        self.n_extr = extr.shape[1]
        self.calc_type_inds()

    def calc_type_inds(self):
        """
        Updates the internal indicies where different params are stored internally.
        """

        self.free_extr = np.sum(self.extr_unfixed)
        self.free_intr = np.sum(self.intr_unfixed)
        self.free_pose = np.sum(self.poses_unfixed)
        self.free_bdpt = np.sum(self.bdpt_unfixed)

        self.intr_end = self.n_intr * self.free_intr
        self.extr_end = self.n_extr * self.free_extr + self.intr_end
        self.pose_end = 6 * self.free_pose + self.extr_end
        self.bdpt_end = 1 * self.free_bdpt + self.pose_end

    def return_bundle_primitives(self, params):
        """
        Takes an array of parameters and populates all unfixed parameters.
        :param params: The input parameters
        :return: The intrinsics, extrinsics, poses and feature points of the calibration.
        """


        intr_data = params[:self.intr_end].reshape((self.free_intr, self.n_intr))
        extr_data = params[self.intr_end:self.extr_end].reshape((self.free_extr, self.n_extr))
        pose_data = params[self.extr_end:self.pose_end].reshape((self.free_pose, 6))
        bdpt_data = params[self.pose_end:self.bdpt_end]

        ch.fill_flat(pose_data, self.poses, self.poses_unfixed)
        ch.fill_flat(extr_data, self.extr, self.extr_unfixed)
        ch.fill_flat(intr_data, self.intr, self.intr_unfixed)

        ch.fill_flat(bdpt_data, self.bundle_pts, self.bdpt_unfixed)

        return self.intr, self.extr, self.poses, self.bundle_pts.reshape((-1, 3))

class SelfBundleHandler(TemplateBundleHandler):
    """
    The standard bundle handler is a class that handles the optimisation of camera parameters.
    It is designed to be used with the numba implentation of the bundle adjustment cost function.
    It takes a CameraSet, a Target and the associated TargetDetection.
    Given these, it will return a function that takes a parameter array and returns data structures ready for
    evaluation with the bundle adjustment cost function.
    The implementation given in the standard param handler implements a target based, but feature unconstrained
    pose based bundle adjustment.

    Two functions provide the ability to add extra parameters and functionality to the optimisation.
    - add_extra_params: this can be overriden to add initial estimates of additional parameters.
    - parse_extra_params_and_setup: this can be overriden to parse additional parameters given to the optimisation.
    Manipulations of the object data/state can be done here, and will be reflected in the cost function.
    As an example: if a higher level structure for camera poses is defined, self.extr_unfixed can be set to all
    false. The parameters can then be parsed, translated into specific extrinsics for each camera, written
    to self.extr, and the cost function will use these extrinsics to define the camera.

    """

    def __init__(self,
                 camset: CameraSet, target: AbstractTarget, detection: TargetDetection,
                 fixed_params: dict|None = None,
                 options: dict | None = None,
                 missing_poses: list | None =None
                 ):
        super().__init__(camset, target, detection, fixed_params, options, missing_poses) 

        self.flat_point_data = np.copy(self.point_data.reshape((-1)))

        # a feature no camera saw cannot be solved for, so it is held fixed
        n_points = int(np.prod(self.point_data.shape[:-1]))
        dd = self.detection.return_flattened_keys(self.target.point_data.shape[:-1]).get_data()[:, 2]
        self.visible_feature_mask = np.isin(np.arange(n_points), dd)
        self.feat_unfixed = np.repeat(self.visible_feature_mask, 3)

        # fix the gauge of the optimisation by fixing 7 coordinates of the
        # target. They are chosen from the features that were seen: a fixed
        # coordinate of an unseen feature is not a parameter of this
        # optimisation and so removes no freedom from it.
        self.fixed_inds, self.gauge_axis = find_gauge_points(
            self.flat_point_data.reshape((-1, 3)),
            candidates=np.flatnonzero(self.visible_feature_mask),
        )
        i0, i1, i2 = self.fixed_inds
        self.feat_unfixed[3*i0:3*i0+3] = False
        self.feat_unfixed[3*i1:3*i1+3] = False
        self.feat_unfixed[3*i2 + self.gauge_axis] = False

        superBundlePrimitive = self.bundlePrimitive

        self.bundlePrimitive = StandardBundlePrimitive(
            superBundlePrimitive.poses, self.flat_point_data, superBundlePrimitive.extr, superBundlePrimitive.intr,
            extr_unfixed=superBundlePrimitive.extr_unfixed, intr_unfixed=superBundlePrimitive.intr_unfixed, poses_unfixed=superBundlePrimitive.poses_unfixed, bundle_points_unfixed=self.feat_unfixed
        )

        self.param_len = None
        self.jac_mask = None
        self.missing_poses: list | None = missing_poses
        self.op_fun: fb.optimisation_function = self._intr_block() + self._extr_block() + fb.rigidTform3d() +  fb.free_point()

    def _kernel_extra_args(self) -> tuple:
        # the target geometry is a parameter here, not a fixed template
        return ()

    def _kernel_maximums(self):
        return None

    def parameter_groups(self) -> list[ParamGroup]:
        """The blocks of this problem; the free target points are eliminated."""
        dd = self._flat_detections()
        cam = dd[:, 0].astype(np.int64)
        img = dd[:, 1].astype(np.int64)
        key = dd[:, 2].astype(np.int64)
        bp = self.bundlePrimitive
        return [
            ParamGroup("intr", bp.intr, bp.intr_unfixed, cam),
            ParamGroup("extr", bp.extr, bp.extr_unfixed, cam),
            ParamGroup("pose", bp.poses, bp.poses_unfixed, img),
            ParamGroup("point", np.asarray(bp.bundle_pts).reshape((-1, 3)),
                       bp.bdpt_unfixed, key),
        ]

    def make_loss_fun(self, threads):
        """
        Describes and writes the loss function of the loss function represented by self.
        Wraps the loss function to account for the fixed parameters of the optimisation.

        :params threads: the number of threads to use.

        """
        target_shape = self.target.point_data.shape
        dd = self.detection.return_flattened_keys(target_shape[:-1]).get_data()
        temp_loss = self.op_fun.make_full_loss_fn(dd, threads)
        def loss_fun(params):
            inps = self.get_bundle_adjustment_inputs(params) #return proj, extr, poses
            param_str = self.op_fun.build_param_list(*inps)
            return temp_loss(param_str).flatten()
        return loss_fun

    def make_loss_jac(self, threads): 
        """
        Describes and writes the jacobian of the loss function described in self.
        Wraps the jacobian of the loss function to deal with the fixed parameters of the optimisation.

        :params threads: the number of threads to use for the optimisation.
        :returns jac_fn: a callable jacobian function that returns the jacobian of the given paramaters.
        """
        target_shape = self.target.point_data.shape
        dd = self.detection.return_flattened_keys(target_shape[:-1]).get_data()
        temp_loss = self.op_fun.make_jacobean(
            dd, threads, unfixed_params=self.parameter_mask())
        def jac_fn(params):
            inps = self.get_bundle_adjustment_inputs(params) #return proj, extr, poses
            param_str = self.op_fun.build_param_list(*inps)
            d, c, rp = temp_loss(param_str)
            return csr_array((d,c,rp), shape=(2*dd.shape[0], params.shape[0]))
        return jac_fn

    def get_bundle_adjustment_inputs(self, x, make_points=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function uses the state of the parameter handler, and the given params
        to build np arrays that describe:

            - the projection matrices of the cameras
            - the distortion parameters of the cameras
            - the pose of every object point in every time poin

        These are direct inputs to a bundle adjustment based loss.

        :param x: The input optimisation parameters.
        """
        proj, extr, poses, bundle_points = self.bundlePrimitive.return_bundle_primitives(x)

        if make_points:
            im_points = np.empty((len(poses), *self.point_data.shape))
            for idx, pose in enumerate(poses):
                blank = np.zeros((12))
                ch.n_e4x4_flat_INPLACE(pose, blank)
                ch.n_htform_broadcast_prealloc(bundle_points.reshape(self.point_data.shape), blank, im_points[idx])

            im_points = np.reshape(im_points, (len(poses), -1, 3))
            return im_points

        return proj, extr, poses, bundle_points

    def set_initial_params(self,x):
        #get the number of params to set on the super
        self.initial_params = x

    def set_from_templated_camset(self, prev_cams: CameraSet):
        """
        Sets the initial values of the calibration from a previous calibration of the same system.
        The previous system must have used a TemplateBundleHandler.
        :param prev_cams: The calibrated camseet to use.
        """
        self.initial_params = np.empty(self.bundlePrimitive.bdpt_end)

        if not isinstance(prev_cams.calibration_handler, TemplateBundleHandler):
            raise ValueError("Previous camera set was not a templated adjustment")
        self.missing_poses =  prev_cams.calibration_handler.missing_poses
        self.initial_params[:self.bundlePrimitive.pose_end] = prev_cams.calibration_params.copy()
        self.initial_params[ 
            self.bundlePrimitive.pose_end:
        ] = prev_cams.calibration_handler.target.point_data.copy().flatten()[self.feat_unfixed]
        # print(prev_cams.calibration_handler.target.point_data.flatten()[:20])

    def get_initial_params(self) -> np.ndarray:
        """
        Returns initial parameters if they exist, or starts calculating them
        and returns the result if they do not.

        :return: the params
        """

        if self.initial_params is not None:
            return self.initial_params
        start_params = self.calc_initial_params()

        self.initial_params = np.empty(self.bundlePrimitive.bdpt_end)
        self.initial_params[:self.bundlePrimitive.pose_end] = start_params
        self.initial_params[ 
            self.bundlePrimitive.pose_end:
        ] = self.target.point_data.copy().flatten()[self.feat_unfixed]
        return self.initial_params

    def get_updated_target(self, x):
        standard_model = self.bundlePrimitive.return_bundle_primitives(x)
        proj, extr, poses, ps = self.apply_gauge_transform(*standard_model)
        return ps

    def get_camset(self, x, return_pose=False) -> CameraSet | tuple[CameraSet, np.ndarray]:
        """
        Given a set of parameters, returns a camera set.

        :param x: the optimisation parameters.
        :param return_pose: Optionally also return the poses of the target.
        :return: Either a CameraSet, or a CameraSet and a list of object poses.
        """


        new_cams = copy(self.camset)

        standard_model = self.bundlePrimitive.return_bundle_primitives(x)
        proj, extr, poses, ps = self.apply_gauge_transform(*standard_model)

        for idc, cam_name in enumerate(self.cam_names):
            temp_cam: Camera = new_cams[cam_name]
            temp_cam.extrinsic = _extrinsic_from_params(extr[idc], temp_cam.extrinsic)
            temp_cam.from_param_vector(proj[idc])
            temp_cam._update_state()
        if not return_pose:
            return new_cams

        ps = np.empty((len(poses), 12))
        for pn, p in zip(ps, poses):
            ch.n_e4x4_flat_INPLACE(p, pn)

        return new_cams, ps

    def apply_gauge_transform(self, proj, extr, poses, point_estimate) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Maps a set of parameters from an existing representation to the scale and transformation that best matches the provided model.
        The transformation is garunteed to preserve the calibration result.

        Where that gauge lands depends on what the camera's pose can hold. A
        camera with a translation takes the whole of it, its intrinsics never
        move, and the first pose stays as the identity. A camera without one --
        a telecentric lens -- cannot absorb a translation, so the poses take
        that part and the scale goes into the magnification instead; the first
        pose then keeps its rotation but carries the gauge translation. Both
        preserve every predicted pixel, which is the property that matters and
        the one the tests check.

        :param proj: The array describing the intrinsic + distortion of the camera. Untouched for a camera whose extrinsic carries a translation; rescaled for one whose does not.
        :param extr: The array containing the extrinsics of the camera system.
        :param poses: The array containing the extimated poses of the calibration target.
        :param point_estimate: The array containing the estimated locations of the calibration target features.
        :return: A tuple containing updated proj, extr, poses, and point estimate.
        """
        # the arrays of the bundle primitive are refilled from the parameters
        # on every call, so the transform is applied to copies: a plot or a
        # second call for the camera set must not leave a transformed pose
        # behind for the next one to transform again. The projection is copied
        # for the same reason -- a rotation only camera takes the scale there.
        poses = poses.copy()
        extr = extr.copy()
        proj = proj.copy()

        ref_points = self.target.point_data.reshape((-1,3))
        valid_map = self.target.valid_map
        vm = self.visible_feature_mask

        if isinstance(valid_map, bool):
            if valid_map == False:
                raise ValueError("Target has given a valid map of False, which indicates no distance comparisons are valid.")
            #use cdist, take the upper
            inds = np.triu_indices(point_estimate[vm].shape[0], k=1)
            new_map = cdist(point_estimate[vm], point_estimate[vm])[inds]
            ref_map = cdist(ref_points[vm], ref_points[vm])[inds]
            dt = self.target.square_size
            mask = np.isclose(ref_map, dt)
            new_map = new_map[mask]
            ref_map = ref_map[mask]
            
            if len(ref_map) == 0:
                raise ValueError("The mask of valid distance pairs was empty, indicating an issue with the square size of the target.")

        elif isinstance(valid_map, np.ndarray):
            new_map = ch.calc_distance_subset(point_estimate, point_estimate, valid_map[:,:2])
            ref_map = ch.calc_distance_subset(ref_points, ref_points, valid_map[:,:2])
        else:
            raise ValueError("The target.valid_map property either needs to be true, for all comparisons being valid, or a nx2 list of index pairs.")
        s = np.mean(ref_map/new_map)
        logger.info(f"Scale factor found {s}")

        new_points = s * point_estimate
        if np.isnan(s):
            raise ValueError("Found S as nan, indicating that the requisite mappings did not exist")
        try:
            update_tform = gu.make_4x4h_tform(*ch.n_estimate_rigid_transform(
                new_points[vm], ref_points[vm])
            ) #this mapping from used points to a reference space
        except Exception as e:
            logger.critical("Failed to find an acceptable gauge transform, returning the identity")
            logger.critical(f"Gave error: {e}")
            update_tform = np.eye(4)

        inv_update = np.linalg.inv(update_tform)
        new_points = gu.h_tform(new_points, update_tform)
        #proj matricies never change: scale invariance!

        # Which world frame the cameras end up in depends on what their pose
        # can hold.  A camera with a translation absorbs the whole gauge
        # change, so the frame is the reference one.  A camera without one --
        # a telecentric lens, whose translation is unidentifiable -- cannot,
        # so the frame keeps the rotation and the scale and leaves the
        # translation to the poses, which are rigid and can carry it.
        rotation_only = self._extr_block.params.n_params == 3
        if rotation_only:
            left = np.eye(4)
            left[:3, :3] = update_tform[:3, :3]
        else:
            left = update_tform

        for i in range(len(poses)):
            ### scale change
            poses[i][3:] = poses[i][3:] * s
            ### rigid change
            pose = gu.make_4x4h_tform(poses[i][:3], poses[i][3:])
            new_pose = left @ pose @ inv_update
            poses[i][:3], poses[i][3:] = gu.ext_4x4_to_rod(new_pose)

        if rotation_only:
            proj, extr = self._regauge_rotation_only(proj, extr, s, left)
        else:
            for i in range(len(extr)):
                ### scale change
                extr[i][3:] = extr[i][3:] * s
                ### rigid change
                og_tform = gu.make_4x4h_tform(extr[i][:3], extr[i][3:])
                new_tform = og_tform @ inv_update
                extr[i][:3], extr[i][3:] = gu.ext_4x4_to_rod(new_tform)
        return proj, extr, poses, new_points

    def _regauge_rotation_only(self, proj, extr, s, left):
        """
        The gauge change for a camera whose pose carries no translation.

        A pinhole camera absorbs the gauge in its extrinsic: the pixel is
        ``K*(Y_xy/Y_z)``, unchanged by scaling the camera-frame point, so the
        scale rides on ``t`` and the intrinsics never move.  A telecentric
        camera has no ``t`` to put it in -- the translation is unidentifiable,
        which is why its extrinsic block is rotation only -- and its pixel,
        ``m*Y_x/(1 + eps*Y_z)``, is not scale invariant.  So for this model the
        gauge lands in the intrinsics instead.

        Because the poses took the translation, the world this camera sees has
        changed by ``R_T`` and ``s`` alone.  With ``R' = R @ R_T^T`` the
        camera-frame point becomes exactly ``s*Y``, so

            m'   = m   / s
            eps' = eps / s

        and nothing else moves: ``c`` is untouched because there is no
        in-plane shift left to absorb, and ``k`` acts on a radius built from
        ``m*Y*w``, which is preserved.  This is exact, not a small-angle or
        small-``eps`` argument -- see
        ``test_the_gauge_transform_leaves_a_telecentric_projection_alone``.
        """
        inv_left = np.linalg.inv(left)
        for i in range(len(extr)):
            # No translation to scale and none to carry: the pose is the
            # rotation alone, and stays that way.
            og_tform = gu.make_4x4h_tform(extr[i][:3], np.zeros(3))
            new_tform = og_tform @ inv_left
            rod, _ = gu.ext_4x4_to_rod(new_tform)
            extr[i][:3] = rod

            if s == 0.0 or not np.isfinite(s):
                raise ValueError(
                    f"The gauge scale came out as {s}, so the solve has no "
                    "usable scale to re-express the magnification against.")
            proj[i][0] = proj[i][0] / s   # m_x
            proj[i][2] = proj[i][2] / s   # m_y
            proj[i][5] = proj[i][5] / s   # eps
        return proj, extr

    
    def special_plots(self, x):
        """
        An additional plot called to visualise the calibration. 
        Visualises the error in the calibration target that was recovered.
        """
        if not _PYVISTA_OK:
            raise ImportError(
                "PyVista is required for self-calibration visualisation. "
                "Install it with: pip install pyCamSet[viz]"
            )
        og_data = self.target.point_data.reshape((-1,3))
        vm = self.visible_feature_mask

        _, _, _, final_data = self.apply_gauge_transform(*self.get_bundle_adjustment_inputs(x))
        diff = (final_data - og_data) * 1000

        scale = 5
        descale = 1000//scale
        logger.info(f"found a mean difference of {np.mean(np.linalg.norm(diff[vm], axis=-1)):.2f} mm")
        s = pv.Plotter()
        s.title = "Target Self-calibration Results."
        s.add_arrows(
            (og_data*descale)[vm], diff[vm], label = f"Recovered shape change ({scale}x mag)", 
                cmap="Greens",
        )
        s.remove_scalar_bar()
        s.add_scalar_bar(title="Euclidean displacement from initial model (mm).")
        s.add_mesh(pv.PolyData(og_data*descale), color='k', label = "Original Model", point_size=0.3)

        # a lattice for every face of the target: point_data's (u, ... w, n, 3)
        # shape groups the n coplanar points of each face together, so the
        # faces are the rows of that reshape whatever the shape of the object.
        for face in og_data.reshape((-1, self.point_data.shape[-2], 3)):
            lattice = pv.PolyData(face*descale, lines=make_connectivity(face))
            s.add_mesh(lattice, style='wireframe', line_width=2, color='k', opacity=0.1)

        s.add_legend(bcolor='w', border=True)

        # the direction the target is viewed from is a choice; the distance is
        # not, and a fixed one draws a small target as a speck. reset_camera
        # keeps the direction and fits the distance to what is being drawn.
        camera = s.camera
        camera.position = (-60, -60, -36)
        camera.focal_point = (0,0,0)
        camera.up = (0,0,-1)
        s.reset_camera()

        s.show()


def make_connectivity(pts):
    """
    The lines of the lattice a face's points sit on, as pyvista reads them.

    :param pts: the (n, 3) points of one face, in the row major order a board
        numbers its corners in.
    :return: a flat list of (2, start, end) line entries.
    """
    n_pts = pts.shape[0]
    n_points_per_line = row_length(pts)
    connectivity = []
    for idp, _ in enumerate(pts):
        if (idp + n_points_per_line < n_pts):
            connectivity.extend([2, idp, idp + n_points_per_line])
        if not ((idp +1) % n_points_per_line == 0):
            connectivity.extend([2, idp, idp + 1])
    return connectivity


def row_length(pts):
    """
    How many of a face's points lie on one row of its lattice.

    The points of a face are numbered row major, so the first step that turns
    away from the direction of the first one has left the first row. A square
    face gives the square root; a rectangular board gives its own width, which
    assuming the square root would have drawn a lattice at a diagonal to.

    :param pts: the (n, 3) points of one face.
    :return: the number of points in a row.
    """
    step = pts[1] - pts[0]
    step = step / np.linalg.norm(step)
    for idp in range(2, len(pts)):
        next_step = pts[idp] - pts[idp - 1]
        next_step = next_step / np.linalg.norm(next_step)
        if np.linalg.norm(np.cross(step, next_step)) > 1e-6:
            return idp
    return len(pts)



def rms_plane(c, n, data):
    norms = np.sum((data - c) * n, axis=1)
    return np.mean(np.abs(norms), axis=0)


def angle_between_planes(normal1, normal2):
    # Normalize the vectors
    normal1_unit = normal1 / np.linalg.norm(normal1)
    normal2_unit = normal2 / np.linalg.norm(normal2)
    
    # Calculate the dot product
    dot_product = np.dot(normal1_unit, normal2_unit)
    
    # Calculate the angle in radians and then convert to degrees
    angle = np.arccos(dot_product)
    return np.degrees(angle)
