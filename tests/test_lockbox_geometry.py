'''
Purpose: Tests for Phase 3 lockbox editor geometry helpers.
Status:  Active coverage for translation-only centre editing semantics.
Future:  Add GUI save/apply tests once Qt test fixtures are available.
'''
from __future__ import annotations

import numpy as np
import pytest

from pyCamSet.gui.lockbox_geometry import (
    camera_center_from_extrinsic,
    extrinsic_from_center_preserving_rotation,
    fit_plane,
    match_signed_plane_offset,
    project_point_to_plane,
    reference_radius,
    snap_radius_to_reference,
)


def test_camera_center_round_trip_preserves_rotation() -> None:
    ext = np.eye(4)
    ext[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ext[:3, 3] = np.array([1.0, 2.0, 3.0])

    original = camera_center_from_extrinsic(ext)
    assert np.allclose(original, -ext[:3, :3].T @ ext[:3, 3])

    edited = np.array([4.0, 5.0, 6.0])
    out = extrinsic_from_center_preserving_rotation(ext, edited)
    assert np.allclose(out[:3, :3], ext[:3, :3])
    assert np.allclose(camera_center_from_extrinsic(out), edited)
    assert np.allclose(out[:3, 3], -ext[:3, :3] @ edited)


def test_reference_radius_uses_median_by_default() -> None:
    points = [np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0])]
    centre = np.zeros(3)
    assert reference_radius(points, centre, statistic="median") == pytest.approx(2.0)
    assert reference_radius(points, centre, statistic="mean") == pytest.approx(103.0 / 3.0)


def test_snap_radius_refuses_camera_at_centre() -> None:
    with pytest.raises(ValueError, match="object centre"):
        snap_radius_to_reference([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 3.0)


def test_snap_radius_preserves_direction() -> None:
    snapped = snap_radius_to_reference([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], 5.0)
    assert np.allclose(snapped, [5.0, 0.0, 0.0])


def test_fit_plane_canonicalises_normal_and_reports_residuals() -> None:
    fit = fit_plane([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [1.0, 1.0, 2.0]])
    assert fit["fit_mode"] == "fitted"
    assert np.allclose(fit["normal"], [0.0, 0.0, 1.0])
    assert fit["rms_residual"] == pytest.approx(0.0)
    assert fit["max_residual"] == pytest.approx(0.0)


def test_project_point_to_plane() -> None:
    projected = project_point_to_plane([1.0, 2.0, 5.0], [0.0, 0.0, 1.0], plane_point=[0.0, 0.0, 2.0])
    assert np.allclose(projected, [1.0, 2.0, 2.0])


def test_match_signed_plane_offset() -> None:
    moved = match_signed_plane_offset([1.0, 2.0, 5.0], [0.0, 0.0, 1.0], 0.5, plane_point=[0.0, 0.0, 2.0])
    assert np.allclose(moved, [1.0, 2.0, 2.5])


def test_radial_move_outward_from_centre() -> None:
    """Radial move: p=[3,0,0], centre=[0,0,0], delta=+2 -> [5,0,0]."""
    centre = np.zeros(3)
    p_test = np.array([3.0, 0.0, 0.0])
    vec = p_test - centre
    direction = vec / np.linalg.norm(vec)
    moved = p_test + 2.0 * direction
    assert np.allclose(moved, [5.0, 0.0, 0.0])


def test_reference_offset_match_with_d_parameter() -> None:
    """Match signed offset: ref at z=3, target at z=5, plane z=2 -> target moves to z=3."""
    normal = np.array([0.0, 0.0, 1.0])
    plane_point = np.array([0.0, 0.0, 2.0])
    d_val = float(-normal @ plane_point)
    ref_offset = float(normal @ np.array([0.0, 0.0, 3.0]) + d_val)  # = 1.0
    result = match_signed_plane_offset([1.0, 2.0, 5.0], normal, ref_offset, d=d_val)
    assert np.allclose(result, [1.0, 2.0, 3.0])
