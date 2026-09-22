
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
        self.super_primitive = self.bundlePrimitive

        self.param_len = None
        self.jac_mask = None
        self.missing_poses: list | None = missing_poses
        self._setup_free_points()
        self.op_fun: fb.optimisation_function = self._intr_block() + self._extr_block() + fb.rigidTform3d() +  fb.free_point()

        # The kernels index a dense parameter array whose blocks are as long
        # as the arrays build_param_list packs: one entry per camera, per
        # image and per target feature. Left to infer them, make_param_struct
        # takes each count from the largest index in the detections it is
        # given, so a trailing image or feature that nothing detected makes
        # its block one element short -- which shifts every block after it.
        # The per image poses come before the per key geometry, so an
        # undetected last image has the kernels reading the target's
        # coordinates six parameters early, and the residuals then measure
        # nothing to do with the reprojection they stand for. Stating the
        # counts keeps the kernels' layout the one the parameter mask and
        # build_param_list describe.
        self.problem_maximums = {
            "max_cams": self.camset.get_n_cams(),
            "max_imgs": self.detection.max_ims,
            "max_keys": int(np.prod(self.point_data.shape[:-1])),
        }

    def _setup_free_points(self):
        """
        Decides which feature coordinates this solve may move, and rebuilds
        the bundle primitive around them.

        Which features are solvable depends on which detections the optimiser
        is allowed to fit, so this has to be redone whenever that changes --
        when ``missing_poses`` arrives from a previous solve, for instance.  A
        feature seen only in a pose that solve discarded has no residual here
        either, and leaving it free puts three all-zero columns in the
        jacobian; taking the gauge from one is worse still, since it fixes the
        seven freedoms against a point nothing constrains.
        """
        n_points = int(np.prod(self.point_data.shape[:-1]))
        seen_keys = self._flat_detections()[:, 2]
        # a feature no camera saw cannot be solved for, so it is held fixed
        self.visible_feature_mask = np.isin(np.arange(n_points), seen_keys)
        self.feat_unfixed = np.repeat(self.visible_feature_mask, 3)

        self.fixed_inds, self.gauge_axis = find_gauge_points(
            self.flat_point_data.reshape((-1, 3)),
            candidates=np.flatnonzero(self.visible_feature_mask),
        )
        i0, i1, i2 = self.fixed_inds
        self.feat_unfixed[3*i0:3*i0+3] = False
        self.feat_unfixed[3*i1:3*i1+3] = False
        self.feat_unfixed[3*i2 + self.gauge_axis] = False

        # the same arrays the super primitive holds, so a value written to
        # either -- a fixed parameter, say -- is seen by the cost function
        super_primitive = self.super_primitive
        self.bundlePrimitive = StandardBundlePrimitive(
            super_primitive.poses, self.flat_point_data, super_primitive.extr, super_primitive.intr,
            extr_unfixed=super_primitive.extr_unfixed, intr_unfixed=super_primitive.intr_unfixed, poses_unfixed=super_primitive.poses_unfixed, bundle_points_unfixed=self.feat_unfixed
        )

    def _kernel_extra_args(self) -> tuple:
        # the target geometry is a parameter here, not a fixed template
        return ()

    def _kernel_maximums(self):
        return self.problem_maximums

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
        dd = self._flat_detections()
        self._base_residual_count = 2 * int(dd.shape[0])  # two per observation
        temp_loss = self.op_fun.make_full_loss_fn(dd, threads, self.problem_maximums)
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
        # the same rows the loss uses, or the jacobian describes a
        # different problem from the residuals being minimised
        dd = self._flat_detections()
        temp_loss = self.op_fun.make_jacobean(
            dd, threads, unfixed_params=self.parameter_mask(),
            problem_maximums=self.problem_maximums)
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

        A parameter vector only carries what its own solve left free, so the
        previous vector cannot be copied into this one: anything that solve
        held fixed is absent from it, and every parameter after the gap lands
        one slot early. What the two solves fix need not even agree -- a
        camera pinned there may be free here, and the reverse -- so the
        previous solution is expanded through the masks that produced it,
        giving a value for every camera, and then re-packed against the masks
        of this problem.

        A parameter this solve fixes keeps the value it was fixed at, which is
        what ``fixed_params`` asked for; one it leaves free starts from the
        previous solve's answer, fixed there or not.

        :param prev_cams: The calibrated camseet to use.
        :raises ValueError: if the previous calibration was not a templated
            adjustment, or describes a different rig to this one.
        """
        prev_handler = prev_cams.calibration_handler
        if not isinstance(prev_handler, TemplateBundleHandler):
            raise ValueError("Previous camera set was not a templated adjustment")
        if prev_cams.calibration_params is None:
            raise ValueError(
                "The previous camera set holds no calibration parameters, so "
                "there is no solution to start this one from.")

        prev_primitive = prev_handler.bundlePrimitive
        prev_params = np.asarray(prev_cams.calibration_params, dtype=float)
        if prev_params.shape[0] != prev_primitive.pose_end:
            raise ValueError(
                f"The previous calibration's {prev_params.shape[0]} parameters "
                f"do not fill its own intrinsic, extrinsic and pose blocks, "
                f"which take {prev_primitive.pose_end}. A parameter vector from "
                "a self calibration carries a target geometry block as well, "
                "and cannot be read as a templated one.")

        self._adopt_missing_poses(prev_handler.missing_poses)

        # Expanding through the previous solve's masks fills in whatever it
        # held fixed from the arrays it fixed them in, so every camera and
        # pose comes back whole however that solve was parameterised.
        prev_intr, prev_extr, prev_poses = (
            np.copy(a) for a in prev_primitive.return_bundle_primitives(prev_params))

        bundle = self.bundlePrimitive
        for name, prev_vals, vals, unfixed in (
            ("intrinsic", prev_intr, bundle.intr, bundle.intr_unfixed),
            ("extrinsic", prev_extr, bundle.extr, bundle.extr_unfixed),
            ("pose", prev_poses, bundle.poses, bundle.poses_unfixed),
        ):
            if prev_vals.shape != vals.shape:
                raise ValueError(
                    f"The previous calibration's {name} block is "
                    f"{prev_vals.shape}, and this one's is {vals.shape}. The "
                    "two calibrations describe different problems, so one "
                    "cannot seed the other.")
            vals[unfixed] = prev_vals[unfixed]

        prev_points = prev_handler.target.point_data.copy().flatten()
        if prev_points.shape != self.flat_point_data.shape:
            raise ValueError(
                f"The previous calibration's target has "
                f"{prev_points.shape[0] // 3} features, and this one's has "
                f"{self.flat_point_data.shape[0] // 3}.")

        self.initial_params = np.empty(bundle.bdpt_end)
        self.initial_params[:bundle.intr_end] = bundle.intr[bundle.intr_unfixed].flatten()
        self.initial_params[bundle.intr_end:bundle.extr_end] = (
            bundle.extr[bundle.extr_unfixed].flatten())
        self.initial_params[bundle.extr_end:bundle.pose_end] = (
            bundle.poses[bundle.poses_unfixed].flatten())
        self.initial_params[bundle.pose_end:] = prev_points[self.feat_unfixed]

    def _adopt_missing_poses(self, missing_poses):
        """
        Takes on the poses a previous solve gave up on.

        A pose it could not estimate -- unposed by the initial estimate, or
        thrown out as an outlier -- is absent from its parameter vector,
        because ``calc_initial_params`` holds such poses fixed.  Its value in
        that solve is therefore the identity it was initialised to, not an
        estimate of anything.  Carrying that identity over while still fitting
        the detections it came from asks this solve to reproject a target
        sitting in the camera's centre, which is where the enormous initial
        error comes from.  So the pose is held fixed here too and its
        detections are dropped, exactly as the previous solve had them.

        Rebuilding the free points afterwards matters as much: without it the
        gauge could be fixed on a feature only those discarded poses saw.

        :param missing_poses: the previous solve's mask, or None
        """
        if missing_poses is None:
            return

        missing = np.asarray(missing_poses, dtype=bool)
        n_poses = self.super_primitive.poses.shape[0]
        if missing.shape != (n_poses,):
            raise ValueError(
                f"The previous calibration marks {missing.size} poses missing, "
                f"and this problem has {n_poses} poses.")
        self.missing_poses = missing

        if not np.any(missing):
            return
        logger.info(
            f"{int(missing.sum())} poses the previous calibration could not "
            "estimate are held fixed, and their detections excluded")
        self.super_primitive.poses_unfixed = (
            self.super_primitive.poses_unfixed & ~missing)
        self.super_primitive.calc_free_poses()
        # which features are solvable, and which may take the gauge, both
        # follow from the detections that are left
        self._setup_free_points()

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

    def _scale_against_target(self, point_estimate, ref_points) -> float:
        """
        How much the solved points must grow to match the target as drawn.

        Read from the distances between features rather than their positions,
        which is what makes it independent of where the solve put them.

        :param point_estimate: the solved feature positions
        :param ref_points: the same features, as the target is drawn
        :raises ValueError: when the target offers no valid distance pair
        """
        valid_map = self.target.valid_map
        vm = self.visible_feature_mask

        if isinstance(valid_map, np.ndarray):
            new_map = ch.calc_distance_subset(point_estimate, point_estimate, valid_map[:,:2])
            ref_map = ch.calc_distance_subset(ref_points, ref_points, valid_map[:,:2])
        elif valid_map is True:
            inds = np.triu_indices(point_estimate[vm].shape[0], k=1)
            new_map = cdist(point_estimate[vm], point_estimate[vm])[inds]
            ref_map = cdist(ref_points[vm], ref_points[vm])[inds]
            # One square's edge only: every other distance is some multiple of
            # it, and a pair a whole board apart is the least well solved.
            mask = np.isclose(ref_map, self.target.square_size)
            new_map, ref_map = new_map[mask], ref_map[mask]
            if len(ref_map) == 0:
                raise ValueError(
                    "No pair of visible features was one square apart, so the "
                    "target's square size does not match its geometry.")
        elif valid_map is False:
            raise ValueError(
                "The target reports no valid distance comparisons, so its "
                "scale cannot be recovered.")
        else:
            raise ValueError(
                "target.valid_map must be True for all comparisons, or an "
                f"(n, 2) array of index pairs; got {type(valid_map).__name__}.")

        scale = float(np.mean(ref_map / new_map))
        if np.isnan(scale):
            raise ValueError(
                "The scale came out NaN, so the distances it was read from "
                "did not exist.")
        return scale

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
        poses = poses.copy()
        extr = extr.copy()
        proj = proj.copy()

        ref_points = self.target.point_data.reshape((-1,3))
        vm = self.visible_feature_mask

        s = self._scale_against_target(point_estimate, ref_points)
        logger.info(f"Scale factor found {s}")
        new_points = s * point_estimate
        try:
            update_tform = gu.make_4x4h_tform(*ch.n_estimate_rigid_transform(
                new_points[vm], ref_points[vm])
            ) #this mapping from used points to a reference space
        except Exception as exc:
            # Returning the identity here would hand back an ungauged result
            # that this method's contract says is gauged, and no caller can
            # tell the two apart.
            raise ValueError(
                "Could not find a gauge transform against the target, so the "
                "solved geometry cannot be put back in the target's frame."
            ) from exc

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

        for face in og_data.reshape((-1, self.point_data.shape[-2], 3)):
            lattice = pv.PolyData(face*descale, lines=make_connectivity(face))
            s.add_mesh(lattice, style='wireframe', line_width=2, color='lightgrey')

        s.add_legend(bcolor='w', border=True)

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
