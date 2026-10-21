'''Check the Phase 3 visual data (pose graph, 3D frames, Schur picture) against the run.
'''
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def pinhole_run(tmp_path_factory):
    """A real recorded Phase 3 solve of the ChArUco corpus (pinhole cameras)."""
    from pyCamSet import ChArUco
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile, run_initial_calibration,
    )
    from pyCamSet.workflow import phase3
    from pyCamSet.workflow.education_trace import read_trace, write_trace

    corpus = Path(__file__).parent / "test_data" / "calibration_charuco"
    target = ChArUco(num_squares_x=20, num_squares_y=20, square_size=4.0, legacy=False)
    detections, resolutions = detect_datapoints_in_imfile(
        f_loc=corpus, caching=False, calibration_target=target, threads=1)
    phase2_set = run_initial_calibration(detections, target, resolutions, save=False)
    phase2_set.set_resolutions_from_file(floc=corpus)
    handler, _result, solved, _stats = phase3.solve(
        phase2_set, detections, target, fixed_params={},
        options={"outliers": "n", "fixed_pose": 0}, threads=1, record_education=True)
    payload = handler.education_step_trace
    run_dir = tmp_path_factory.mktemp("p3") / "phase3_runs" / "phase3-run"
    run_dir.mkdir(parents=True)
    phase2_path, phase3_path = run_dir / "phase2.camset", run_dir / "phase3.camset"
    phase2_set.save(phase2_path)
    solved.save(phase3_path)
    payload.update({"run_id": "phase3-run", "phase": "phase3"})
    write_trace(run_dir, "phase3-run", "phase3", "accepted_step", payload, enabled=True)
    write_trace(run_dir, "phase3-run", "phase3", "seed_join_order",
                {"events": list(handler.seed_join_order)}, enabled=True)
    initial = run_dir / "phase3_initial_state.npz"
    np.savez_compressed(initial, calibration_params=np.asarray(payload["steps"][0]["x"]))
    return SimpleNamespace(run_dir=run_dir, phase2=phase2_path, phase3=phase3_path,
                           initial=initial, trace=read_trace(run_dir, "phase3-run", "phase3"),
                           target=target)


def _visual(run):
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_playback import build_phase3_frames
    from pyCamSet.workflow.education_visual_p3 import build_phase3_visual
    held = []

    def factory():
        handler = load_CameraSet(run.phase3).calibration_handler
        held.append(handler)
        return handler
    with np.load(run.initial) as initial:
        start = initial["calibration_params"]
    frames = build_phase3_frames(factory, run.trace, expected_start=start,
                                 saved_params=load_CameraSet(run.phase3).calibration_params)
    assert frames["status"] == "available", frames
    return build_phase3_visual(held[0], frames["arrays"], frames["nits"], frames["costs"]), \
        held[0], frames


def test_states_reproduce_the_saved_result_and_camera_positions(pinhole_run):
    from pyCamSet.utils.saving import load_CameraSet
    visual, handler, frames = _visual(pinhole_run)
    saved = load_CameraSet(pinhole_run.phase3)
    assert not visual.telecentric
    assert len(visual.states) == len(frames["arrays"])
    assert [s.nit for s in visual.states] == frames["nits"]
    # Final per-observation errors are the saved residual segment, row for row.
    rows = len(visual.rows_camera)
    saved_xy = np.asarray(saved.calibration_result, dtype=float)[:2 * rows].reshape(-1, 2)
    assert np.allclose(visual.states[-1].row_error_px, np.linalg.norm(saved_xy, axis=1),
                       rtol=1e-9, atol=1e-9)
    assert np.allclose(visual.final_residual_xy, saved_xy, rtol=1e-9, atol=1e-9)
    # Per-camera means are the means of those rows.
    for cam in range(visual.n_cameras):
        mask = visual.rows_camera == cam
        assert visual.states[-1].camera_mean_px[cam] == pytest.approx(
            float(np.linalg.norm(saved_xy[mask], axis=1).mean()))
    # Pinhole camera centres are the saved cameras' positions.
    centres = visual.states[-1].centres(visual.scene_centre, visual.camera_distance)
    assert np.allclose(centres, [cam.position[:3] for cam in saved], atol=1e-6)
    # The error shrinks from the start to the result, and the start is the recorded start.
    assert visual.states[-1].mean_px < visual.states[0].mean_px
    # Rebuilt corners: the fixed reference photo's corners are the target's own points.
    ref = visual.point_image == 0
    template = handler.target.point_data.reshape(-1, 3)
    keys = np.unique(np.asarray(handler._flat_detections())[:, 2].astype(int)[
        np.asarray(handler._flat_detections())[:, 1].astype(int) == 0])
    assert np.allclose(np.sort(visual.states[-1].point_xyz[ref], axis=0),
                       np.sort(template[keys], axis=0), atol=1e-9)


def test_pose_graph_counts_shared_photos_and_reads_the_join_order(pinhole_run):
    from pyCamSet.workflow.education_visual_p3 import graph_layout, join_events
    visual, _handler, _frames = _visual(pinhole_run)
    shared = visual.shared_photos()
    assert np.array_equal(shared, shared.T)
    counts = visual.camera_image_counts()
    assert np.array_equal(np.diag(shared), (counts > 0).sum(axis=1))
    for a in range(visual.n_cameras):
        for b in range(visual.n_cameras):
            both = np.intersect1d(np.unique(visual.rows_image[visual.rows_camera == a]),
                                  np.unique(visual.rows_image[visual.rows_camera == b]))
            assert shared[a, b] == len(both)
    events = join_events(pinhole_run.trace, {})
    assert events and all({"joining", "joined_from"} <= set(e) for e in events)
    assert join_events({"records": []}, {"seed_join_order": events}) == events
    assert join_events(None, None) == []
    layout = graph_layout(visual, visual.n_cameras)
    assert layout.shape == (visual.n_cameras, 2) and np.all(np.isfinite(layout))


def test_reduced_camera_system_is_the_schur_complement_of_the_jacobian(pinhole_run):
    """S from the solver's own assembly equals Jc'Jc - Jc'Jp (Jp'Jp)^-1 Jp'Jc from the CSR J."""
    pytest.importorskip("PySide6")
    from pyCamSet.workflow.education_visual_p3 import reduced_camera_system, schur_structure
    visual, handler, frames = _visual(pinhole_run)
    x = np.asarray(frames["arrays"][1], dtype=float)
    system = reduced_camera_system(handler, x)
    S = system["S"]
    assert np.allclose(S, S.T, rtol=1e-8, atol=1e-8 * np.abs(S).max())
    J = handler.make_loss_jac(1)(x).toarray()[:2 * len(visual.rows_camera)]
    primitive = handler.bundlePrimitive
    n_cam = primitive.extr.shape[0]
    n_keep = n_cam * (primitive.n_intr + primitive.n_extr)
    Jc, Jp = J[:, :n_keep], J[:, n_keep:]
    expected = Jc.T @ Jc - Jc.T @ Jp @ np.linalg.solve(Jp.T @ Jp, Jp.T @ Jc)
    # The view reorders the camera block so each camera's intrinsics and extrinsics sit together.
    order = []
    for cam in range(n_cam):
        order += list(range(cam * primitive.n_intr, (cam + 1) * primitive.n_intr))
        order += list(range(n_cam * primitive.n_intr + cam * primitive.n_extr,
                            n_cam * primitive.n_intr + (cam + 1) * primitive.n_extr))
    expected = expected[np.ix_(order, order)]
    free = ~system["pinned"]
    assert np.allclose(S[np.ix_(free, free)], expected[np.ix_(free, free)],
                       rtol=1e-6, atol=1e-8 * np.abs(expected).max())
    assert system["n_kept"] == int(free.sum())
    assert system["n_eliminated"] == len(x) - n_keep
    # The view's per-camera-pair blocks are non-zero exactly where two cameras shared a photo.
    from pyCamSet.gui.education_phase3 import reduced_system_blocks
    blocks = reduced_system_blocks(system)
    assert blocks.shape == (n_cam, n_cam) and blocks.max() == pytest.approx(1.0)
    assert np.array_equal(blocks > 1e-12, visual.shared_photos() > 0)
    structure = schur_structure(visual, primitive.n_intr, primitive.n_extr, len(x), 0)
    assert structure.pair_points.sum() == len(visual.rows_camera)
    assert structure.n_camera_params == n_keep and structure.n_pose_params == len(x) - n_keep


def test_block_updates_and_scene_frames(pinhole_run):
    from pyCamSet.workflow.education_visual_p3 import (
        block_updates, point_selection, ray_selection, scene_frame)
    visual, _handler, _frames = _visual(pinhole_run)
    start = block_updates(visual, 0)
    assert not start["camera_deg"].any() and not np.nanmax(start["pose_deg"])
    step = block_updates(visual, 1)
    assert step["camera_deg"].shape == (visual.n_cameras,) and np.nanmax(step["pose_deg"]) > 0
    selection = point_selection(visual, 50)
    rays = ray_selection(visual, 1, limit=20)
    assert len(selection) == 50 and np.array_equal(selection, point_selection(visual, 50))
    frame = scene_frame(visual, 1, points=selection, ray_camera=1, rays=rays)
    assert set(frame["cameras"]) == {f"cam{i}" for i in range(visual.n_cameras)}
    assert frame["points"]["corners"]["xyz"].shape == (50, 3)
    assert {f"axis{i}" for i in range(visual.n_cameras)} | {"rays"} == set(frame["rays"])
    # Pinhole rays start at the camera centre and end on the rebuilt corners.
    centre = visual.states[1].centres(visual.scene_centre, visual.camera_distance)[1]
    assert np.allclose(frame["rays"]["rays"]["origins"], centre)
    assert np.allclose(frame["rays"]["rays"]["ends"],
                       visual.states[1].point_xyz[visual.row_point[rays]])
    assert np.allclose(frame["rays"]["rays"]["scalars"], visual.states[1].row_error_px[rays])


def test_telecentric_cameras_are_drawn_at_a_fixed_distance_along_their_axis():
    from scipy.spatial.transform import Rotation
    from pyCamSet.workflow.education_visual_p3 import Phase3Visual, StepState, camera_pose
    R = Rotation.from_rotvec([[0.0, 0.0, 0.0], [0.0, 0.4, 0.0]]).as_matrix()
    state = StepState(0, 1.0, R, None, np.zeros((2, 6)), np.zeros((1, 6)), np.zeros((3, 3)),
                      np.zeros(3), np.zeros(3), np.zeros(2), 0.0)
    visual = Phase3Visual(("a", "b"), True, ((0, 0), (0, 0)), np.zeros(3, int), np.zeros(3, int),
                          np.zeros((3, 2)), np.arange(3), np.zeros(3, int), 1, [state],
                          "recorded steps", scene_centre=np.array([1.0, 2.0, 3.0]),
                          scene_radius=2.0)
    centres = state.centres(visual.scene_centre, visual.camera_distance)
    for cam in range(2):
        offset = visual.scene_centre - centres[cam]
        assert np.linalg.norm(offset) == pytest.approx(visual.camera_distance)
        assert np.allclose(offset / np.linalg.norm(offset), R[cam][2])  # looks at the centre
        pose = camera_pose(visual, 0, cam)
        assert np.allclose(-pose["R"].T @ pose["t"], centres[cam])






# --- Round-1 review: one target in the rig, the photo view, the lens and agreement pictures ---


def test_one_target_in_the_rig_and_the_exaggerated_solver_motion(pinhole_run):
    """UX-2/ED-2: the target is drawn once, in the fixed photo's frame; ED-1: the solver's
    changes can be exaggerated, the result staying exactly where it is."""
    from pyCamSet.workflow.education_visual_p3 import (
        corner_errors, motion_factor, reference_pose, rig_motion, target_edges,
        target_world_points)
    visual, handler, _frames = _visual(pinhole_run)
    template = handler.target.point_data.reshape(-1, 3)
    R, t = reference_pose(visual)
    assert np.allclose(target_world_points(visual), template @ R.T + t)
    # A flat board is outlined by its 4 edges, a cube by 12.
    origins, ends = target_edges(template)
    assert len(origins) == 4 and np.all(np.linalg.norm(ends - origins, axis=1) > 0)
    cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    assert len(target_edges(cube)[0]) == 12
    # Each corner's colour is its mean error over every observation of it.
    errors = corner_errors(visual, -1)
    key = int(np.flatnonzero(np.isfinite(errors))[0])
    rows = visual.row_key == key
    assert errors[key] == pytest.approx(visual.states[-1].row_error_px[rows].mean())
    # At x1 the drawn pinhole centres are the solver's own; the result is never moved.
    plain = rig_motion(visual, 1.0)
    for k, state in enumerate(visual.states):
        assert np.allclose(plain["C"][k], state.centres(visual.scene_centre,
                                                        visual.camera_distance), atol=1e-9)
    factor = 5.0
    big = rig_motion(visual, factor)
    assert np.allclose(big["C"][-1], plain["C"][-1]) and np.allclose(big["R"][-1], plain["R"][-1])
    assert np.allclose(big["C"][0] - big["C"][-1], factor * (plain["C"][0] - plain["C"][-1]))
    assert motion_factor(visual) in (1, 2, 5, 10, 20)


def test_photo_view_dots_are_the_solver_predictions(pinhole_run):
    """ED-1/ED-21: the dots are where the model predicts each found corner (observed +
    residual, the solver's own sign), and they move between recorded states only by
    interpolation."""
    from pyCamSet.workflow.education_visual_p3 import (
        photo_choices, photo_points, photo_rows, typical_photo)
    visual, _handler, _frames = _visual(pinhole_run)
    camera = 1
    photo = typical_photo(visual, camera)
    assert photo in photo_choices(visual, camera)
    rows = photo_rows(visual, camera, photo)
    assert len(rows) == photo_choices(visual, camera)[photo]
    observed, predicted, errors = photo_points(visual, len(visual.states) - 1, rows)
    assert np.allclose(predicted - observed, visual.final_residual_xy[rows], atol=1e-4)
    assert np.allclose(errors, visual.states[-1].row_error_px[rows], atol=1e-4)
    _, start, _ = photo_points(visual, 0, rows)
    _, half, _ = photo_points(visual, 0.5, rows)
    _, one, _ = photo_points(visual, 1, rows)
    assert np.allclose(half, (start + one) / 2, atol=1e-6)


def test_schur_picture_blocks_and_camera_coupling(pinhole_run):
    """ED-16/SCI-11: the normal matrix's blocks (cameras, couplings, one 6x6 block per photo)
    and the camera system's ties normalised so the mixed units cancel."""
    from pyCamSet.workflow.education_visual_p3 import (
        coupling_matrix, normal_matrix_blocks, reduced_camera_system, schur_structure)
    visual, handler, frames = _visual(pinhole_run)
    primitive = handler.bundlePrimitive
    x = np.asarray(frames["arrays"][-1], dtype=float)
    structure = schur_structure(visual, primitive.n_intr, primitive.n_extr, len(x), 0)
    blocks = normal_matrix_blocks(structure)
    codes = blocks["codes"]
    assert blocks["n_kept"] == structure.n_camera_params
    assert blocks["n_kept"] + blocks["n_eliminated"] == codes.shape[0]
    assert np.array_equal(codes, codes.T)
    # Photo blocks never touch each other: C is block-diagonal.
    keep = blocks["n_kept"]
    pose = codes[keep:, keep:]
    off_diagonal = np.ones_like(pose, dtype=bool)
    for k in range(blocks["n_poses"]):
        off_diagonal[6 * k:6 * k + 6, 6 * k:6 * k + 6] = False
    assert np.all(pose[off_diagonal] == -1)
    # Each (camera, photo) pair couples exactly one camera block with one photo block.
    n_pairs = int(np.sum([p in set(structure.pose_order) for p in structure.pair_image]))
    coupled = (codes[:keep, keep:] == blocks["n_cameras"]).sum()
    assert coupled == n_pairs * blocks["per_camera"] * 6
    tie = coupling_matrix(reduced_camera_system(handler, x))
    assert np.allclose(np.diag(tie), 1.0) and np.allclose(tie, tie.T)
    assert np.all((tie >= 0) & (tie <= 1))


def test_lens_changes_and_error_ladder_read_the_saved_camsets(pinhole_run):
    """ED-5/SCI-5: Phase 3's lenses against Phase 2's, read from the two saved camsets; ED-4:
    the ladder's Phase 3 rung is the RMS of the solve's own residuals per camera."""
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_visual_p3 import error_ladder, lens_changes
    visual, _handler, _frames = _visual(pinhole_run)
    before, after = load_CameraSet(pinhole_run.phase2), load_CameraSet(pinhole_run.phase3)
    changes = lens_changes(before, after, visual.image_sizes)
    assert len(changes) == visual.n_cameras
    for change, name in zip(changes, after.get_names()):
        assert not change.telecentric
        assert change.scale_after[0] == pytest.approx(after[name].intrinsic[0, 0])
        assert change.scale_before[0] == pytest.approx(before[name].intrinsic[0, 0])
        assert change.centre_shift_px == pytest.approx(np.hypot(
            *(np.asarray(after[name].intrinsic)[:2, 2] - np.asarray(before[name].intrinsic)[:2, 2])))
        assert change.distortion_shift_px >= 0
    ladder = error_ladder(visual, {"diagnostics": {"D2.1_per_camera_rms_reprojection": {
        visual.camera_names[0]: 1.25}}}, {"run_id": "p4", "diagnostics": {"D4.3_final_euclid_px": 0.3}})
    assert ladder.phase2_rms == {0: 1.25} and ladder.phase4_mean == 0.3
    err = visual.states[-1].row_error_px[visual.rows_camera == 2]
    assert ladder.phase3_rms[2] == pytest.approx(np.sqrt(np.mean(err ** 2)))


def test_camera_agreement_ghosts_are_each_cameras_own_pose(pinhole_run):
    """ED-4: a camera's ghost is the target pose it found on its own, placed with the first
    guess's rig (world_from_camera @ target_in_camera); the shared pose is the first guess's."""
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_visual_p3 import camera_agreement, seed_arrays
    visual, _handler, _frames = _visual(pinhole_run)
    tic, ext, poses = seed_arrays(load_CameraSet(pinhole_run.phase3),
                                  load_CameraSet(pinhole_run.phase2))
    agreement = camera_agreement(tic, ext, poses, visual.template, telecentric=False)
    assert agreement is not None and len(agreement.alone) >= 2
    for cam, ghost in agreement.alone.items():
        assert np.allclose(ext[cam] @ ghost, tic[cam, agreement.photo], atol=1e-9)
        assert agreement.offset[cam] >= 0 and agreement.angle_deg[cam] >= 0
    assert np.allclose(agreement.shared, poses[agreement.photo])


# --- Round-2 review: lens effect, join criterion, one ladder statistic, trails, agreement ---


def test_coupling_is_jacobi_scaled_so_per_parameter_units_cancel(pinhole_run):
    """SCI2-19: rescaling any one parameter (a unit change) leaves the camera ties unchanged."""
    from pyCamSet.workflow.education_visual_p3 import coupling_matrix, reduced_camera_system
    _visual_, handler, frames = _visual(pinhole_run)
    system = reduced_camera_system(handler, np.asarray(frames["arrays"][-1], dtype=float))
    tie = coupling_matrix(system)
    rng = np.random.default_rng(4)
    d = np.exp(rng.normal(0, 3, len(system["S"])))
    scaled = dict(system, S=d[:, None] * np.asarray(system["S"]) * d[None, :])
    assert np.allclose(coupling_matrix(scaled), tie, atol=1e-9)


def test_lens_bending_effect_is_the_solves_residual_with_and_without_the_bending(pinhole_run):
    """SCI2-1/ED2-2: 'on' is the solve's own final error; 'off' zeroes every camera's
    distortion and nothing else; the handler is left exactly as it was."""
    from pyCamSet.workflow.education_visual_p3 import BENDING_FIRST_COLUMN, lens_bending_effect
    visual, handler, frames = _visual(pinhole_run)
    x = np.asarray(frames["arrays"][-1], dtype=float)
    before = np.asarray(handler.bundlePrimitive.intr).copy()
    effect = lens_bending_effect(handler, x)
    assert np.array_equal(np.asarray(handler.bundlePrimitive.intr), before)
    assert np.allclose(np.linalg.norm(effect.on_xy, axis=1), visual.states[-1].row_error_px)
    zeroed = x.copy()
    primitive = handler.bundlePrimitive
    block = zeroed[:primitive.intr_end].reshape(-1, primitive.n_intr)
    block[:, BENDING_FIRST_COLUMN:] = 0
    expected = np.asarray(handler.make_loss_fun(1)(zeroed))[:2 * len(visual.rows_camera)]
    assert np.allclose(effect.off_xy.ravel(), expected)
    assert np.allclose(effect.moved_xy, effect.on_xy - effect.off_xy)
    assert set(effect.camera_mean("on")) == set(range(visual.n_cameras))


def test_join_candidates_reproduce_the_recorded_grow_order(pinhole_run):
    """ED2-11: the recomputed links reproduce the grow order recorded from the same per-view
    poses, each join took the smallest disagreement open at that moment, and a record that
    does not match is reported as not reproduced (the view then draws no candidates)."""
    from pyCamSet.optimisation.region_growing import estimate_rig
    from pyCamSet.optimisation.template_handler import target_radius
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_visual_p3 import full_seed, join_candidates
    p3, p2 = load_CameraSet(pinhole_run.phase3), load_CameraSet(pinhole_run.phase2)
    tic = np.asarray(full_seed(p3, p2)[0], dtype=float)
    radius = target_radius(p3.calibration_handler.target)
    events: list[dict] = []
    estimate_rig(tic, radius, record_join_order=True, join_order=events)
    found = join_candidates(tic, radius, events)
    assert found is not None and found.reproduced and len(found.chosen) == len(events)
    for chosen, options in zip(found.chosen, found.options):
        assert chosen == pytest.approx(min(options))
    tampered = [dict(e, spread=float(e["spread"]) * 2) for e in events]
    assert not join_candidates(tic, radius, tampered).reproduced


def test_ladder_uses_one_statistic_and_maps_phase4_by_camera_name(pinhole_run):
    """SCI2-4: the Phase 4 rung is per-camera RMS (matched by name), never the mean."""
    from pyCamSet.workflow.education_visual_p3 import error_ladder, phase4_camera_rms
    visual, _handler, _frames = _visual(pinhole_run)
    names = visual.camera_names
    ladder = error_ladder(visual, None, {"run_id": "p4", "diagnostics": {
        "D4.3_final_euclid_px": 0.3}}, {names[1]: 0.6, "unknown camera": 9.0})
    assert ladder.phase4_rms == {1: 0.6} and ladder.phase4_mean == 0.3
    assert phase4_camera_rms(None) == {} and phase4_camera_rms({"artifacts": {}}) == {}


def test_trail_points_grow_with_the_playback_position():
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_phase3 import trail_points
    path = np.array([[0.0, 0, 0], [1, 0, 0], [1, 2, 0]])
    assert np.allclose(trail_points(path, 0), path[:1])
    assert np.allclose(trail_points(path, 1.5), [[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    assert np.allclose(trail_points(path, 9), path)


def test_eps_is_shown_in_phase4s_unit():
    """ED2-4: ε per target unit in 1/m (metre targets keep their value, mm targets x1000)."""
    from pyCamSet.workflow.education_visual_p3 import eps_per_metre
    assert eps_per_metre(-3.13, 1000.0) == pytest.approx(-3.13)
    assert eps_per_metre(-0.00313, 1.0) == pytest.approx(-3.13)
    assert eps_per_metre(1.0, None) is None


