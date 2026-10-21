'''Verify Phase 4 saved-run derivations and recorded playback.
'''
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyCamSet.workflow.education_derive import (
    DerivationUnavailable,
    _phase4_counts,
    _phase4_point_classes,
    _phase4_start_state,
    derive_phase4,
    phase4_scale_consistency_ratio,
)


def _context(camset_path: Path, run_dir: Path, metadata: dict):
    return SimpleNamespace(
        phase="phase4", status="complete", run_id="fixture-phase4",
        run_dir=run_dir, metadata=metadata,
        artifact=lambda phase: camset_path if phase == "phase4" else None,
    )


def test_phase4_scale_ratio_only_uses_measured_one_square_edges():
    nominal = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    solved = nominal.copy()
    solved[1, 0] = 1.1
    measured = np.array([True, True, True, False])
    ratio = phase4_scale_consistency_ratio(nominal, solved, measured, 1.0)
    assert ratio == pytest.approx((1 / 1.1 + 1.0) / 2)
    assert abs(ratio - 1.0) > 0.01
    assert abs(1.005 - 1.0) <= 0.01


def test_phase4_display_refit_is_rigid_and_keeps_the_solved_size():
    """SCI-1 (round 1): the display alignment removes turn + shift only. A shape that is 3 %
    larger than the design stays 3 % larger (its corrections are real); only the similarity
    fit, kept for reporting, would scale it back (and shrink every face)."""
    from pyCamSet.workflow.education_derive import _phase4_display_refit
    rng = np.random.default_rng(1)
    nominal = rng.normal(0, 0.005, (40, 3))
    a = np.radians(7.0)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    solved = 1.03 * nominal @ R.T + np.array([0.01, -0.02, 0.003])
    mask = np.ones(len(nominal), dtype=bool)
    rigid, scale = _phase4_display_refit(nominal, solved, mask)
    assert scale == 1.0
    # Distances inside the shape are untouched: still 3 % larger than the design.
    d_rigid = np.linalg.norm(rigid[1:] - rigid[0], axis=1)
    d_nom = np.linalg.norm(nominal[1:] - nominal[0], axis=1)
    np.testing.assert_allclose(d_rigid, 1.03 * d_nom, rtol=1e-12)
    # Turn and shift are removed: the centroid and orientation match the design.
    np.testing.assert_allclose(rigid.mean(axis=0), nominal.mean(axis=0), atol=1e-15)
    sized, sim_scale = _phase4_display_refit(nominal, solved, mask, with_scale=True)
    assert sim_scale == pytest.approx(1 / 1.03, rel=1e-12)
    np.testing.assert_allclose(sized, nominal, atol=1e-15)


def test_phase4_recorded_start_distinguishes_verified_and_held_parameter_states():
    from pyCamSet.workflow.education_trace import Verification

    confirmed = SimpleNamespace(
        confirmed=True, verification=Verification.VERIFIED,
        effective_fixed_parameter_differences=("camera 1 focal length",))
    unconfirmed = SimpleNamespace(
        confirmed=False, verification=Verification.UNCHECKABLE,
        effective_fixed_parameter_differences=())
    assert _phase4_start_state(SimpleNamespace(
        phase3_state_source=lambda: confirmed)) == (True, True)
    assert _phase4_start_state(SimpleNamespace(
        phase3_state_source=lambda: unconfirmed)) == (False, False)


def test_phase4_point_classes_ignore_excluded_pose_rows_and_partition_visible_points():
    from pyCamSet.cameras.camera import Camera
    rows = np.array([[0, 0, 0, 1., 2.], [0, 1, 1, 3., 4.]])
    target = SimpleNamespace(point_data=np.array([[0., 0., 0.], [1., 0., 0.]]))
    primitive = SimpleNamespace(return_bundle_primitives=lambda _params: (
        np.zeros((1, 6)), np.zeros((1, 6)), np.zeros((2, 6))))
    handler = SimpleNamespace(
        _flat_detections=lambda: rows, target=target,
        cam_names=["cam0"], camset={"cam0": Camera(name="cam0")},
        visible_feature_mask=np.array([True, False]),
        missing_poses=np.array([False, True]), bundlePrimitive=primitive,
        get_initial_params=lambda: np.empty(0),
    )
    assert _phase4_counts(handler).tolist() == [1, 0]
    classes = _phase4_point_classes(handler)
    assert classes.unseen == (1,)
    assert classes.one_direction == (0,)
    assert classes.several_directions == ()


@pytest.mark.parametrize("rotation_deg,has_diversity", [(0.0, False), (10.0, True)])
def test_telecentric_translation_does_not_invent_viewing_ray_diversity(rotation_deg, has_diversity):
    from pyCamSet.cameras.telecentric_camera import TelecentricCamera
    rows = np.array([[0, 0, 0, 0., 0.], [0, 1, 0, 0., 0.]])
    primitive = SimpleNamespace(return_bundle_primitives=lambda _params: (
        np.empty((0, 0)), np.array([[0., 0., 0., 0., 0., 0.]]),
        np.array([[0., 0., 0., -1., 0., 0.],
                  [0., np.deg2rad(rotation_deg), 0., 1., 0., 0.]])))
    handler = SimpleNamespace(
        _flat_detections=lambda: rows,
        target=SimpleNamespace(point_data=np.array([[0., 0., 5.]])),
        cam_names=["cam0"], camset={"cam0": TelecentricCamera(name="cam0")},
        visible_feature_mask=np.array([True]), missing_poses=np.array([False, False]),
        bundlePrimitive=primitive, get_initial_params=lambda: np.empty(0))
    classes = _phase4_point_classes(handler)
    assert (classes.several_directions == (0,)) is has_diversity
    assert (classes.one_direction == (0,)) is not has_diversity
    assert classes.ray_separation_deg[0] == pytest.approx(rotation_deg, abs=1e-9)


@pytest.mark.slow
@pytest.mark.data
def test_generated_phase4_camset_recalculates_and_builds_trace_frames(tmp_path):
    from education_p4_fixtures import generate_phase4_run

    camset_path, metadata = generate_phase4_run(tmp_path)
    run_dir = camset_path.parent
    before = hashlib.sha256(camset_path.read_bytes()).hexdigest()
    result = derive_phase4(_context(camset_path, run_dir, metadata))
    assert not isinstance(result, DerivationUnavailable), getattr(result, "reason", None)
    assert result.status == "available"
    assert result.d41_visible_points == metadata["diagnostics"]["D4.1_n_free_target_points"]
    assert result.d43_final_px == pytest.approx(
        metadata["diagnostics"]["D4.3_final_euclid_px"], rel=1e-9)
    assert result.d47_mean_displacement_mm == pytest.approx(
        metadata["diagnostics"]["D4.7_mean_target_displacement_mm"], rel=1e-9)
    assert abs(result.scale_consistency_ratio - 1.0) <= 0.01
    assert result.display_refit_points is not None
    assert result.display_refit_scale is not None
    classes = result.point_classes
    assert classes is not None
    assert sorted((*classes.unseen, *classes.one_direction, *classes.several_directions)) == list(
        range(len(classes.ray_separation_deg)))
    if classes.several_directions:
        # The L1 shape headline: several-direction points only, after display alignment.
        assert result.several_directions_mean_displacement_mm is not None
        assert np.isfinite(result.several_directions_mean_displacement_mm)
        assert result.several_directions_mean_displacement_mm >= 0
    assert result.frames and result.frames[-1].status == "available"
    assert result.frames[-1].shape_points is not None
    assert result.frames[0].recorded_start is True
    assert result.frames[-1].target_poses is not None
    from pyCamSet.utils.saving import load_CameraSet
    saved_camset = load_CameraSet(camset_path)
    for name in saved_camset.get_names():
        np.testing.assert_allclose(
            result.frames[-1].rig_cameras[name].extrinsic,
            saved_camset[name].extrinsic, rtol=0.0, atol=1e-6)
    assert hashlib.sha256(camset_path.read_bytes()).hexdigest() == before


@pytest.mark.slow
@pytest.mark.data
def test_playback_frames_from_one_handler_equal_fresh_rebuilds(tmp_path):
    """derive_phase4 replays every recorded step through ONE rebuilt handler (it used to
    rebuild two per frame: ~3 minutes on a real 8-camera rig). Each frame must equal what a fresh
    handler on a freshly loaded camset gives, and frames must not share Camera objects."""
    from education_p4_fixtures import generate_phase4_run
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_derive import _phase4_rebuild
    from pyCamSet.workflow.education_trace import read_trace

    camset_path, metadata = generate_phase4_run(tmp_path)
    result = derive_phase4(_context(camset_path, camset_path.parent, metadata))
    assert result.status == "available"
    trace = read_trace(camset_path.parent, "fixture-phase4", "phase4")
    steps = next(record["payload"]["steps"] for record in trace["records"]
                 if record["kind"] == "accepted_step")
    assert len(steps) == len(result.frames) >= 2
    for step, frame in zip(steps, result.frames):
        assert frame.status == "available"
        assert frame.point_classes == result.point_classes
        params = np.asarray(step["x"], dtype=np.float64)
        fresh = _phase4_rebuild(load_CameraSet(camset_path), metadata["diagnostics"])
        np.testing.assert_array_equal(
            frame.shape_points,
            np.asarray(fresh.get_updated_target(params.copy()), dtype=np.float64).reshape(-1, 3))
        rig, poses = _phase4_rebuild(load_CameraSet(camset_path),
                                     metadata["diagnostics"]).get_camset(params.copy(),
                                                                         return_pose=True)
        for name in rig.get_names():
            for attribute in ("extrinsic", "intrinsic", "distortion_coefs"):
                np.testing.assert_array_equal(np.asarray(getattr(frame.rig_cameras[name], attribute)),
                                              np.asarray(getattr(rig[name], attribute)))
    first, last = result.frames[0].rig_cameras, result.frames[-1].rig_cameras
    assert all(first[name] is not last[name] for name in first.get_names())
    # Not aliased: the first frame still shows the start, not the last step's cameras.
    assert any(not np.array_equal(np.asarray(first[name].extrinsic),
                                  np.asarray(last[name].extrinsic))
               for name in first.get_names())


@pytest.mark.slow
@pytest.mark.data
def test_phase4_recorded_trace_endpoint_mismatch_fails_closed(tmp_path):
    from education_p4_fixtures import generate_phase4_run
    from pyCamSet.workflow.education_trace import read_trace, write_trace

    camset_path, metadata = generate_phase4_run(tmp_path)
    run_dir = camset_path.parent
    trace = read_trace(run_dir, "fixture-phase4", "phase4")
    assert trace["status"] == "available"
    payload = next(record["payload"] for record in trace["records"]
                   if record["kind"] == "accepted_step")
    payload["steps"][-1]["x"][0] += 0.25
    write_trace(run_dir, "fixture-phase4", "phase4", "accepted_step",
                payload, enabled=True)
    result = derive_phase4(_context(camset_path, run_dir, metadata))
    assert not isinstance(result, DerivationUnavailable)
    assert result.frames
    assert result.frames[0].status == "unavailable"
    assert "do not match" in (result.frames[0].reason or "")


@pytest.mark.slow
@pytest.mark.data
def test_phase4_once_seen_displacement_drives_scale_qualifier(tmp_path):
    from education_p4_fixtures import generate_phase4_run

    try:
        camset_path, metadata = generate_phase4_run(
            tmp_path, remove_e1_marker=True, displace_once_seen=True)
    except ValueError as exc:
        reason = "fixture has too few displaceable once-seen points to drive the ratio past tolerance"
        if str(exc) == reason:
            pytest.skip(reason)
        raise
    result = derive_phase4(_context(camset_path, camset_path.parent, metadata),
                           include_playback=False)
    assert not isinstance(result, DerivationUnavailable), getattr(result, "reason", None)
    assert abs(result.scale_consistency_ratio - 1.0) > 0.01
    assert result.metric_qualifier_required is True
    assert result.d47_mean_displacement_mm == pytest.approx(
        metadata["diagnostics"]["D4.7_mean_target_displacement_mm"], rel=1e-9)


@pytest.mark.data
def test_325fae4_phase4_camset_is_refused_by_layout_check():
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_derive import _phase4_rebuild

    camset_path = (Path(__file__).parent / "test_data" / "education" / "legacy"
                   / "phase4_325fae4.camset.json")
    diagnostics = {
        "D4.1_n_free_target_points": 302,
        "D4.2_gauge_fixed_points": {"count": 3, "indices": [0, 1, 19]},
        "D4.3_final_euclid_px": 0.6838,
        "D4.5_gauge_scale_factor": 1.0,
        "D4.7_mean_target_displacement_mm": 0.1903,
    }
    camset = load_CameraSet(camset_path)
    rebuilt = _phase4_rebuild(camset, diagnostics)
    assert rebuilt.fixed_inds != (0, 1, 19)
    assert len(camset.calibration_params) != len(rebuilt.get_initial_params())
    assert rebuilt.problem_opts["target_scale_rule"] == "legacy_mean_all_edges"
    result = derive_phase4(_context(
        camset_path, camset_path.parent,
        {"diagnostics": diagnostics, "phase": "phase4"}), include_playback=False)
    assert isinstance(result, DerivationUnavailable)
    assert "parameter count" in result.reason


def test_module_import_is_qt_free():
    import pyCamSet.workflow.education_derive as module
    assert "PySide6" not in module.__dict__
    assert callable(module.derive_phase4)


def test_several_direction_mean_ignores_a_once_seen_point_displaced_along_its_ray():
    """A once-seen point's depth is unconstrained; its drift along the ray is not shape."""
    from pyCamSet.workflow.education_derive import _phase4_class_mean_displacement_mm

    nominal = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    solved = nominal.copy()
    solved[0, 2] += 0.1  # several directions: a real 0.1 unit shape difference
    camera = np.array([1., 1., 10.])
    ray = (nominal[3] - camera) / np.linalg.norm(nominal[3] - camera)
    solved[3] += 5.0 * ray  # seen once: slid 5 units along its only line of sight
    several = (0, 1, 2)
    value = _phase4_class_mean_displacement_mm(solved, nominal, several, 2.0)
    assert value == pytest.approx(0.1 / 3 * 2.0)
    everything = _phase4_class_mean_displacement_mm(solved, nominal, (0, 1, 2, 3), 2.0)
    assert everything > 10 * value
    assert _phase4_class_mean_displacement_mm(solved, nominal, (), 2.0) is None
    assert _phase4_class_mean_displacement_mm(solved, nominal, several, None) is None
    assert _phase4_class_mean_displacement_mm(None, nominal, several, 2.0) is None
