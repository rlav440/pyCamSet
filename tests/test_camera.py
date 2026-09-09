"""``Camera``: the pinhole model, its state updates and its accessors.

``test_coordinate_system.py`` already covers the sensor-map round trip; this
covers the rest of the class, which sat at 44%.  The interesting cases are the
ones where a setter has to invalidate derived state (``proj``, ``cam_to_world``,
``position``) -- a camera that keeps a stale projection matrix after being
moved projects to the wrong place with no error at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyCamSet import Camera
from pyCamSet.utils.general_utils import h_tform, make_4x4h_tform

from conftest import REF_INTRINSIC, REF_RES, make_camera

# --------------------------------------------------------------------------
# Construction and derived state
# --------------------------------------------------------------------------


def test_defaults_are_usable():
    """A bare Camera() must be a valid camera, since the defaults are public."""
    cam = Camera()
    assert cam.intrinsic is not None
    assert cam.res is not None
    assert np.allclose(cam.extrinsic, np.eye(4))
    assert np.allclose(cam.position, 0)


def test_derived_state_is_populated_at_construction():
    cam = make_camera("a", translation=(0.1, 0.2, 0.3))

    assert np.allclose(cam.cam_to_world, np.linalg.inv(cam.extrinsic))
    # extrinsic translates the world, so the camera sits at its negation
    assert np.allclose(cam.position, [-0.1, -0.2, 0.3 * -1])
    assert np.allclose(cam.proj, REF_INTRINSIC @ cam.extrinsic[:3, :4])


def test_original_matrix_is_a_copy_not_a_reference():
    """The reference copy must not alias the live intrinsic."""
    cam = make_camera("a")
    cam.intrinsic[0, 0] = 1.0
    assert cam.original_matrix[0, 0] == REF_INTRINSIC[0, 0]


def test_projection_matrix_is_intrinsic_times_extrinsic():
    cam = make_camera("a", translation=(0.05, -0.02, 0.3))
    assert np.allclose(cam._calc_projection_matrix(), cam.intrinsic @ cam.extrinsic[:3, :4])


# --------------------------------------------------------------------------
# Setters, and the derived state they must invalidate
# --------------------------------------------------------------------------


def test_set_extrinsic_refreshes_the_derived_state():
    cam = make_camera("a")
    moved = make_4x4h_tform(np.zeros(3), np.array([0.1, 0.0, 0.0]))

    cam.set_extrinsic(moved)

    assert np.allclose(cam.extrinsic, moved)
    assert np.allclose(cam.cam_to_world, np.linalg.inv(moved))
    assert np.allclose(cam.position, (np.linalg.inv(moved) @ [0, 0, 0, 1])[:3])
    assert np.allclose(cam.proj, cam.intrinsic @ moved[:3, :4])


def test_set_distortion_coefs_takes_effect_on_projection():
    cam = make_camera("a")
    points = np.array([[0.2, 0.15, 1.0]])
    undistorted = cam.project_points(points).copy()

    cam.set_distortion_coefs(np.array([-0.3, 0.1, 0.0, 0.0, 0.0]))
    distorted = cam.project_points(points)

    assert not np.allclose(undistorted, distorted)


def test_transform_composes_onto_the_extrinsic():
    """``transform`` right-multiplies, so it moves the world, not the camera."""
    cam = make_camera("a")
    shift = make_4x4h_tform(np.zeros(3), np.array([0.1, 0.0, 0.0]))
    before = cam.extrinsic.copy()

    cam.transform(shift)

    assert np.allclose(cam.extrinsic, before @ shift)
    assert np.allclose(cam.cam_to_world, np.linalg.inv(cam.extrinsic))


def test_transforming_a_point_and_the_camera_together_is_a_no_op():
    """The invariant behind set_reference_cam: E T inv(T) p == E p."""
    cam = make_camera("a", translation=(0.02, -0.01, 0.2))
    points = np.array([[0.0, 0.0, 1.0], [0.05, -0.03, 1.4]])
    before = cam.project_points(points).copy()

    shift = make_4x4h_tform(np.array([0.1, -0.2, 0.05]), np.array([0.3, 0.1, -0.2]))
    cam.transform(shift)
    moved_points = h_tform(points, np.linalg.inv(shift))

    assert np.allclose(cam.project_points(moved_points), before)


def test_set_minimal_toggles_the_flag():
    """Sensor maps are lazily built for large sensors; the flag drives that."""
    cam = make_camera("a")
    cam.set_minimal(False)
    assert cam.minimal is False
    cam.set_minimal(True)
    assert cam.minimal is True


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------


def test_a_point_on_the_optical_axis_lands_on_the_principal_point():
    cam = make_camera("a")
    uv = cam.project_points(np.array([[0.0, 0.0, 1.0]]))
    assert np.allclose(uv[0], [REF_INTRINSIC[0, 2], REF_INTRINSIC[1, 2]])


def test_projection_uses_separate_x_and_y_focal_lengths():
    """fx != fy in the fixture, so a transposed intrinsic fails here."""
    cam = make_camera("a")
    at_depth = 2.0
    offset = 0.1

    u_only = cam.project_points(np.array([[offset, 0.0, at_depth]]))[0]
    v_only = cam.project_points(np.array([[0.0, offset, at_depth]]))[0]

    assert np.isclose(u_only[0], REF_INTRINSIC[0, 2] + REF_INTRINSIC[0, 0] * offset / at_depth)
    assert np.isclose(v_only[1], REF_INTRINSIC[1, 2] + REF_INTRINSIC[1, 1] * offset / at_depth)


def test_projection_accepts_a_single_flat_point():
    """A (3,) input is promoted to (1, 3) rather than rejected."""
    cam = make_camera("a")
    flat = cam.project_points(np.array([0.0, 0.0, 1.0]))
    shaped = cam.project_points(np.array([[0.0, 0.0, 1.0]]))
    assert np.allclose(flat, shaped)


def test_image_mode_swaps_the_coordinate_order():
    """mode="image" returns v,u where the default returns u,v."""
    cam = make_camera("a")
    points = np.array([[0.05, -0.02, 1.0]])

    opencv = cam.project_points(points, mode="opencv")
    image = cam.project_points(points, mode="image")

    assert np.allclose(image, opencv[:, ::-1])


def test_distortion_is_skipped_when_the_coefficients_are_zero():
    """The zero-distortion fast path must agree with asking for no distortion."""
    cam = make_camera("a")
    points = np.array([[0.1, 0.1, 1.0]])
    assert np.allclose(cam.project_points(points, distort=True), cam.project_points(points, distort=False))


def test_distortion_can_be_disabled_on_a_distorted_camera():
    cam = make_camera("a", distortion=[-0.3, 0.1, 1e-3, -1e-3, 0.02])
    points = np.array([[0.2, 0.15, 1.0]])
    assert not np.allclose(
        cam.project_points(points, distort=True),
        cam.project_points(points, distort=False),
    )


def test_image_mode_still_swaps_when_distorting():
    """The distorted branch has its own mode handling, so check it too."""
    cam = make_camera("a", distortion=[-0.3, 0.1, 0.0, 0.0, 0.0])
    points = np.array([[0.2, 0.15, 1.0]])

    opencv = cam.project_points(points, mode="opencv").copy()
    image = cam.project_points(points, mode="image")

    assert np.allclose(image, opencv[:, ::-1])


# --------------------------------------------------------------------------
# Visibility
# --------------------------------------------------------------------------


def test_a_point_in_front_of_the_camera_can_be_imaged():
    cam = make_camera("a")
    assert cam.can_image(np.array([0.0, 0.0, 1.0]))


def test_a_point_outside_the_sensor_cannot_be_imaged():
    cam = make_camera("a")
    # far off to the side at close range: projects well outside 640x480
    assert not cam.can_image(np.array([5.0, 0.0, 0.1]))


def test_can_image_takes_one_point_at_a_time():
    """Pinning the signature: ``can_image`` adds the leading axis itself.

    It calls ``project_points(pt[None, ...])``, so a single (3,) point is the
    contract and an (n, 3) array becomes (1, n, 3) and fails inside h_tform.
    Callers wanting many points project them and use ``_is_in_image``.
    """
    cam = make_camera("a")
    with pytest.raises(ValueError):
        cam.can_image(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]))


def test_is_in_image_at_the_sensor_bounds():
    """0 is inside, res is outside: the check is a half-open interval."""
    cam = make_camera("a")
    inside = np.array([[0.0, 0.0], [REF_RES[0] - 1.0, REF_RES[1] - 1.0]])
    outside = np.array([[-1.0, 0.0], [float(REF_RES[0]), 0.0]])

    assert np.all(cam._is_in_image(inside))
    assert not np.any(cam._is_in_image(outside))


# --------------------------------------------------------------------------
# Resolution changes
# --------------------------------------------------------------------------


def test_scale_self_2n_halves_the_resolution_and_the_focal_length():
    cam = make_camera("a")
    cam.scale_self_2n(1)

    assert list(cam.res) == [REF_RES[0] // 2, REF_RES[1] // 2]
    assert np.isclose(cam.intrinsic[0, 0], REF_INTRINSIC[0, 0] / 2)
    assert np.isclose(cam.intrinsic[1, 1], REF_INTRINSIC[1, 1] / 2)
    assert cam.down_scale_factor == 1


def test_scale_self_2n_keeps_the_image_centre_fixed():
    """Downscaling must be equivalent to resampling the same view."""
    cam = make_camera("a")
    centre_ray = np.array([[0.0, 0.0, 1.0]])
    full = cam.project_points(centre_ray)[0]

    cam.scale_self_2n(1)
    half = cam.project_points(centre_ray)[0]

    # the principal point moves to the halved sensor's centre, within the
    # half-pixel convention the scale matrix encodes
    assert np.allclose(half, (full - 0.5) / 2, atol=1e-9)


def test_scale_self_2n_by_zero_is_a_no_op():
    cam = make_camera("a")
    cam.scale_self_2n(0)
    assert list(cam.res) == REF_RES
    assert np.allclose(cam.intrinsic, REF_INTRINSIC)


def test_crop_to_roi_shifts_the_principal_point():
    """The roi is [xmin, xmax, ymin, ymax], grouped by axis.

    The implementation used to destructure ``[ymin, xmin, xmax, ymax]`` while
    the docstring promised this order, so a caller following the docs shifted
    the principal point by the wrong offsets.  Distinct values in every slot
    here, so a reordering fails rather than coincidentally passing.
    """
    cam = make_camera("a")
    xmin, xmax, ymin, ymax = 10, 200, 30, 150

    cam.crop_to_roi([xmin, xmax, ymin, ymax])

    assert np.isclose(cam.intrinsic[0, 2], REF_INTRINSIC[0, 2] - xmin)
    assert np.isclose(cam.intrinsic[1, 2], REF_INTRINSIC[1, 2] - ymin)
    # focal lengths are unchanged by a crop
    assert np.isclose(cam.intrinsic[0, 0], REF_INTRINSIC[0, 0])
    assert np.isclose(cam.intrinsic[1, 1], REF_INTRINSIC[1, 1])


def test_crop_to_roi_refreshes_the_projection_matrix():
    cam = make_camera("a")
    cam.crop_to_roi([10, 200, 30, 150])
    assert np.allclose(cam.proj, cam.intrinsic @ cam.extrinsic[:3, :4])


def test_crop_outside_the_sensor_is_refused():
    cam = make_camera("a")
    # xmax past the sensor width
    with pytest.raises(ValueError, match="outside of camera viewpoint"):
        cam.crop_to_roi([0, REF_RES[0] + 1, 0, 10])
    # ymax past the sensor height
    with pytest.raises(ValueError, match="outside of camera viewpoint"):
        cam.crop_to_roi([0, 10, 0, REF_RES[1] + 1])


def test_reset_to_original_params_restores_the_intrinsic():
    cam = make_camera("a")
    cam.crop_to_roi([10, 200, 30, 150])
    assert not np.allclose(cam.intrinsic, REF_INTRINSIC)

    cam.reset_to_original_params()

    assert np.allclose(cam.intrinsic, REF_INTRINSIC)
    assert np.allclose(cam.proj, REF_INTRINSIC @ cam.extrinsic[:3, :4])


def test_reset_to_original_params_survives_repeated_use():
    """Regression: reset handed out the stored original by reference.

    ``crop_to_roi`` subtracts from ``self.intrinsic`` in place, so once reset
    had aliased the two arrays together the next crop overwrote the reference
    copy.  The second reset then restored the cropped values instead of the
    calibrated ones, and the camera could never be recovered.
    """
    cam = make_camera("a")

    for _ in range(3):
        cam.crop_to_roi([10, 200, 30, 150])
        cam.reset_to_original_params()
        assert np.allclose(cam.intrinsic, REF_INTRINSIC)

    assert np.allclose(cam.original_matrix, REF_INTRINSIC)


def test_reset_does_not_alias_the_reference_copy():
    cam = make_camera("a")
    cam.reset_to_original_params()
    assert cam.intrinsic is not cam.original_matrix


# --------------------------------------------------------------------------
# Rays
# --------------------------------------------------------------------------


def test_im_to_world_ray_accepts_a_list():
    cam = make_camera("a")
    from_list = cam.im_to_world_ray([320, 240])
    from_array = cam.im_to_world_ray(np.array([[320, 240]]))
    assert np.allclose(from_list, from_array)


def test_im_to_world_ray_round_trips_through_projection():
    """A ray from a pixel must project back to that pixel."""
    cam = make_camera("a", translation=(0.03, -0.01, 0.2))
    pixels = np.array([[100, 200], [320, 240], [500, 400]])

    rays = cam.im_to_world_ray(pixels)
    assert np.allclose(cam.project_points(rays), pixels)


def test_geometry_helpers_return_meshes():
    """Smoke test: the pyvista helpers must build without a display."""
    cam = make_camera("a")
    assert cam.get_mesh(scale=0.01).n_points > 0
    assert cam.get_viewcone(view_len=0.5).n_points > 0
    assert cam.get_viewcone(view_len=0.5, triangle=True).n_points > 0


def test_cam_fov_is_a_plausible_angle():
    cam = make_camera("a")
    fov = cam._cam_fov()
    assert 0 < fov < 180


def test_get_image_cord_sensor_map_transposes_the_world_map():
    cam = make_camera("a")
    cam._make_sensormap()
    image_cords = cam.get_image_cord_sensor_map()
    assert image_cords.shape == (cam.world_sensor_map.shape[1], cam.world_sensor_map.shape[0], 3)
    assert np.allclose(image_cords, np.transpose(cam.world_sensor_map, (1, 0, 2)))
