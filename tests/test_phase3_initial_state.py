"""Verify Phase 3 solver-start snapshots and fixed-pose seed anchoring.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyCamSet.optimisation.region_growing import estimate_rig
from pyCamSet.workflow.phase3 import _write_initial_state

EQUIVALENCE_DECLARATIONS = (
    "artifacts.initial_state_sha256",
    "diagnostics.initial_state_sha256",
)


def _synthetic_target_views() -> np.ndarray:
    """Build a connected three-camera rig with repeatable target views."""
    camera_offsets = np.array([0.0, 1.0, 2.0])
    image_offsets = np.array([0.0, 0.4, -0.3, 0.8])
    views = np.broadcast_to(np.eye(4), (3, 4, 4, 4)).copy()
    views[:, :, 0, 3] = camera_offsets[:, None] + image_offsets[None, :]
    views[:, :, 1, 3] = image_offsets[None, :]
    return views


def test_initial_state_snapshot_is_run_private_atomic_and_exact(tmp_path: Path):
    vector = np.linspace(-2.0, 3.0, 31, dtype=np.float64)
    missing = np.array([False, True, False, False], dtype=bool)

    path = _write_initial_state(tmp_path / "phase3_runs" / "run-a", vector, missing, 4.1973)

    assert path.parent == tmp_path / "phase3_runs" / "run-a"
    with np.load(path, allow_pickle=False) as state:
        np.testing.assert_array_equal(state["calibration_params"], vector)
        np.testing.assert_array_equal(state["missing_poses"], missing)
        assert state["calibration_params"].dtype == np.float64
        assert float(state["initial_error_px"]) == 4.1973


def test_initial_state_over_its_bound_is_skipped_without_replacing_existing_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Too large to keep: not written, and never a reason to fail the run."""
    import pyCamSet.workflow.phase3 as phase3

    run_dir = tmp_path / "run-b"
    run_dir.mkdir()
    path = run_dir / "phase3_initial_state.npz"
    path.write_bytes(b"previous")
    monkeypatch.setattr(phase3, "MAX_INITIAL_STATE_BYTES", 1)

    assert _write_initial_state(run_dir, np.arange(10.0), np.zeros(2, dtype=bool), 1.0) is None
    assert path.read_bytes() == b"previous"


def test_fixed_pose_anchor_handles_image_not_seen_by_every_camera_and_records_opt_in():
    target_in_camera = _synthetic_target_views()
    target_in_camera[0, 0] = np.nan  # N1: fixed image lacks camera 0's rows.
    target_in_camera[2, 1:] = np.nan  # The remaining graph is still connected.
    target_in_camera[1, 2:] = np.nan

    plain_ext, plain_poses = estimate_rig(
        target_in_camera, 1.0, reference_image=0)
    events: list[dict] = []
    recorded_ext, recorded_poses = estimate_rig(
        target_in_camera, 1.0, reference_image=0,
        record_join_order=True, join_order=events)

    np.testing.assert_allclose(plain_ext, recorded_ext, equal_nan=True)
    np.testing.assert_allclose(plain_poses, recorded_poses, equal_nan=True)
    np.testing.assert_allclose(recorded_poses[0], np.eye(4), atol=1e-12)
    assert len(events) == target_in_camera.shape[0] - 1
    for event in events:
        assert set(event) == {"joined_from", "joining", "spread", "shared_views", "spread_source"}
        first, second = event["joined_from"], event["joining"]
        shared = int(np.count_nonzero(
            np.isfinite(target_in_camera[first, :, 0, 0])
            & np.isfinite(target_in_camera[second, :, 0, 0])))
        assert event["shared_views"] == shared
        assert event["spread_source"] == ("imputed_single_view" if shared == 1 else "measured")


def test_default_rig_estimation_does_not_collect_join_events():
    target_in_camera = _synthetic_target_views()
    events: list[dict] = []

    estimate_rig(target_in_camera, 1.0, reference_image=2, join_order=events)

    assert events == []


def test_excluded_fixed_pose_cannot_become_the_world_anchor():
    target_in_camera = _synthetic_target_views()
    target_in_camera[:, 3] = np.nan

    _, poses = estimate_rig(target_in_camera, 1.0, reference_image=3)

    assert np.all(np.isnan(poses[3]))
    assert np.allclose(poses[0], np.eye(4), atol=1e-12)


@pytest.mark.parametrize("case", ["n1", "fixed_pose_5"])
def test_live_phase3_seed_reconciles_per_image_error_with_solver_start(
        case: str, charuco_problem, tmp_path: Path):
    from pyCamSet.calibration_targets.core.target_detections import TargetDetection
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, initial_cams = charuco_problem
    if case == "n1":
        data = detections.get_data()
        data = data[~((data[:, 0] == 0) & (data[:, 1] == 0))]
        detections = TargetDetection(
            cam_names=detections.cam_names,
            data=data,
            max_ims=detections.max_ims,
        )
    fixed_pose = 0 if case == "n1" else 5
    handler = TemplateBundleHandler(
        camset=initial_cams,
        target=target,
        detection=detections,
        options={"outliers": "n", "fixed_pose": fixed_pose},
    )

    start = handler.get_initial_params()
    assert handler.record_education is False
    active_rows = handler.get_detection_data(flatten=True)
    active_counts = np.bincount(
        active_rows[:, 1].astype(int), minlength=detections.max_ims)
    per_image = np.asarray(handler.initial_per_im_error, dtype=float)
    selected = (active_counts > 0) & np.isfinite(per_image)
    weighted_d33 = float(np.average(per_image[selected], weights=active_counts[selected]))
    snapshot_path = _write_initial_state(
        tmp_path / case, start, handler.missing_poses, 0.0)
    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        saved_start = snapshot["calibration_params"].copy()
        np.testing.assert_array_equal(snapshot["missing_poses"], handler.missing_poses)
    np.testing.assert_array_equal(saved_start, start)
    residuals = np.asarray(handler.make_loss_fun(1)(saved_start), dtype=float)
    d35 = float(np.mean(np.linalg.norm(residuals[:2 * len(active_rows)].reshape(-1, 2), axis=1)))

    assert d35 == pytest.approx(weighted_d33, rel=1e-9, abs=1e-12)
    assert handler.starting_state_disposition == "consistent"
    assert np.allclose(handler.bundlePrimitive.poses[fixed_pose], 0.0)


def test_recomputing_live_seed_replaces_join_events(charuco_problem):
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, initial_cams = charuco_problem
    handler = TemplateBundleHandler(
        camset=initial_cams, target=target, detection=detections,
        options={"outliers": "n"}, record_education=True,
    )
    handler.get_initial_params()
    first = list(handler.seed_join_order)
    assert len(first) == len(initial_cams) - 1
    handler.calc_initial_params()
    assert handler.seed_join_order == first


def test_phase3_run_records_join_order_only_with_opt_in(charuco_problem, tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from pyCamSet.workflow import phase3
    from pyCamSet.workflow.education_trace import read_trace, trace_path
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, cams = charuco_problem
    monkeypatch.setattr(phase3, "load_CameraSet", lambda _: cams)
    monkeypatch.setattr(phase3, "load_pickle", lambda _: detections)
    monkeypatch.setattr(phase3, "extract_detection", lambda value: value)
    monkeypatch.setattr(phase3, "target_of_params", lambda _: target)
    monkeypatch.setattr(phase3, "_lockbox", lambda *args: (None, None, {}))
    monkeypatch.setattr(phase3, "_diagnostics", lambda *args: {})
    saved = SimpleNamespace(save=lambda path: path.write_bytes(b"test camset"),
                            calibration_report=None)

    def fake_solve(_cams, _detections, _target, **options):
        handler = TemplateBundleHandler(
            camset=_cams, target=_target, detection=_detections,
            options=options["options"], record_education=options["record_education"])
        handler.get_initial_params()
        return handler, object(), saved, {"initial_euclid": 0.0, "final_euclid": 0.0}

    monkeypatch.setattr(phase3, "solve", fake_solve)
    diagnostics = []
    for enabled in (False, True):
        run_dir = tmp_path / ("on" if enabled else "off")
        run_dir.mkdir()
        params = {"fixed_params": {}, "problem_options": {"outliers": "n"},
                  "threads": 1, "record_education": enabled}
        _, diag, _, _ = phase3._solve(
            params, run_dir, tmp_path / "seed.camset", tmp_path / "detections.pickle",
            None, lambda _: None)
        diagnostics.append(diag)
        if not enabled:
            # Nothing education-only is written for a run that does not record.
            assert "initial_state_artifact" not in diag
            assert not (run_dir / "phase3_initial_state.npz").exists()
            assert "seed_join_order" not in diag and not trace_path(run_dir).exists()
        else:
            snapshot = Path(diag["initial_state_artifact"])
            assert diag["initial_state_sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
            trace = read_trace(run_dir, run_dir.name, "phase3")
            assert trace["status"] == "available"
            assert len(diag["seed_join_order"]) == len(cams) - 1
            assert trace["records"][0]["payload"]["events"] == diag["seed_join_order"]



def test_phase3_records_an_unwritable_trace_as_unavailable(charuco_problem, tmp_path, monkeypatch):
    """A trace too large to write is teaching material lost, not a failed solve."""
    from types import SimpleNamespace
    from pyCamSet.workflow import phase3
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, cams = charuco_problem
    monkeypatch.setattr(phase3, "load_CameraSet", lambda _: cams)
    monkeypatch.setattr(phase3, "load_pickle", lambda _: detections)
    monkeypatch.setattr(phase3, "extract_detection", lambda value: value)
    monkeypatch.setattr(phase3, "target_of_params", lambda _: target)
    monkeypatch.setattr(phase3, "_lockbox", lambda *args: (None, None, {}))
    monkeypatch.setattr(phase3, "_diagnostics", lambda *args: {})
    saved = SimpleNamespace(save=lambda path: path.write_bytes(b"test camset"),
                            calibration_report=None)

    def fake_solve(_cams, _detections, _target, **options):
        handler = TemplateBundleHandler(
            camset=_cams, target=_target, detection=_detections,
            options=options["options"], record_education=options["record_education"])
        handler.get_initial_params()
        return handler, object(), saved, {"initial_euclid": 0.0, "final_euclid": 0.0}

    def too_large(*_args, **_kwargs):
        raise ValueError("education trace record exceeds its bound")

    monkeypatch.setattr(phase3, "solve", fake_solve)
    monkeypatch.setattr(phase3, "write_trace", too_large)
    params = {"fixed_params": {}, "problem_options": {"outliers": "n"},
              "threads": 1, "record_education": True}
    camset_out, diag, _, _ = phase3._solve(
        params, tmp_path, tmp_path / "seed.camset", tmp_path / "detections.pickle",
        None, lambda _: None)

    assert Path(camset_out).exists()
    assert diag["education_trace"].startswith("unavailable:")

def test_live_phase3_marks_unobserved_fixed_pose_as_world_frame_unfixed(
        charuco_problem):
    from pyCamSet.calibration_targets.core.target_detections import TargetDetection
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, initial_cams = charuco_problem
    data = detections.get_data()
    data = data[data[:, 1] != 5]
    reduced = TargetDetection(
        cam_names=detections.cam_names, data=data,
        max_ims=detections.max_ims)
    handler = TemplateBundleHandler(
        camset=initial_cams, target=target, detection=reduced,
        options={"outliers": "n", "fixed_pose": 5},
    )

    handler.get_initial_params()

    assert handler.starting_state_disposition == (
        "reference image excluded; world frame not fixed")


def test_outlier_exclusion_keeps_d31_and_d32_counts_distinct(
        monkeypatch: pytest.MonkeyPatch):
    from pyCamSet.optimisation import template_handler

    handler = object.__new__(template_handler.TemplateBundleHandler)
    handler.missing_poses = np.zeros(3, dtype=bool)
    handler.problem_opts = {"outliers": "y"}
    responses = iter((np.array([1]), None))
    monkeypatch.setattr(
        template_handler.gu, "mad_outlier_detection",
        lambda *args, **kwargs: next(responses),
    )

    handler.find_and_exclude_transform_outliers(np.array([1.0, 90.0, 2.0]))

    assert int(np.sum(handler.missing_poses_before_outlier_rejection)) == 0
    assert int(np.sum(handler.missing_poses_after_outlier_rejection)) == 1
    assert int(np.sum(handler.missing_poses_after_outlier_rejection)
               - np.sum(handler.missing_poses_before_outlier_rejection)) == 1
