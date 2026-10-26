'''Verify saved Phase 3 derivations and fail-closed provenance rules.
'''
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyCamSet.workflow.education_derive import (
    DerivationUnavailable,
    attribute_residuals,
    derive_phase3,
    derive_rig_spreads,
)


EQUIVALENCE_DECLARATIONS = ()


class _FakeDetection:
    """Small detection stand-in for exact row-attribution tests."""

    max_ims = 3

    def get_data(self):
        return np.array([
            [0, 0, 2, 10.0, 12.0],
            [1, 0, 2, 20.0, 22.0],
            [0, 2, 1, 30.0, 32.0],
        ], dtype=float)


class _FakeHandler:
    """Only the persisted handler surface used by the pure derivation helper."""

    def __init__(self):
        self.detection = _FakeDetection()
        self.cam_names = ["cam0", "cam1"]

    def _flat_detections(self):
        return self.detection.get_data()


def _fake_camset(residuals):
    return SimpleNamespace(calibration_handler=_FakeHandler(),
                           calibration_result=np.asarray(residuals, dtype=float))


def test_attribution_uses_two_values_per_optimised_row_and_ignores_prior_tail():
    saved = [3.0, 4.0, 0.0, 2.0, 5.0, 12.0, 777.0, 999.0]
    observations, counts, mean, residual_xy = attribute_residuals(_fake_camset(saved))

    assert [item.key for item in observations] == [(2,), (2,), (1,)]
    assert [item.image for item in observations] == [0, 0, 2]
    assert [item.magnitude_px for item in observations] == [5.0, 2.0, 13.0]
    assert counts == (2, 0, 1)
    assert mean == pytest.approx(20.0 / 3.0)
    assert residual_xy == ((3.0, 4.0), (0.0, 2.0), (5.0, 12.0))


def test_attribution_refuses_short_or_nonfinite_saved_reprojection_segment():
    with pytest.raises(ValueError, match="shorter"):
        attribute_residuals(_fake_camset([1.0, 2.0]))
    with pytest.raises(ValueError, match="non-finite"):
        attribute_residuals(_fake_camset([1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0]))


def test_keyed_seed_refuses_duplicate_rows_and_escaped_or_corrupt_snapshot(tmp_path):
    from pyCamSet.workflow.education_derive import ResidualObservation, _keyed_seed_residuals
    inside = tmp_path / "initial.npz"
    np.savez_compressed(inside, calibration_params=np.array([0.]),
                        missing_poses=np.array([False]), initial_error_px=np.array(0.))
    context = SimpleNamespace(run_dir=tmp_path,
                              metadata={"artifacts": {
                                  "initial_state": str(inside),
                                  "initial_state_sha256": hashlib.sha256(inside.read_bytes()).hexdigest()}})
    row = ResidualObservation(camera=0, image=0, key=(1,),
                              observed_xy=(0., 0.), residual_xy=(0., 0.), magnitude_px=0.)
    saved_digest = context.metadata["artifacts"].pop("initial_state_sha256")
    values, reason = _keyed_seed_residuals(context, None, (row,), 0.)
    assert values is None and "no file digest" in reason
    context.metadata["artifacts"]["initial_state_sha256"] = saved_digest
    values, reason = _keyed_seed_residuals(context, None, (row, row), 0.)
    assert values is None and "duplicate" in reason
    outside = tmp_path.parent / f"{tmp_path.name}-escaped.npz"
    outside.write_bytes(inside.read_bytes())
    context.metadata["artifacts"]["initial_state"] = str(outside)
    values, reason = _keyed_seed_residuals(context, None, (row,), 0.)
    assert values is None and "outside this run" in reason
    inside.write_bytes(b"not a valid npz")
    context.metadata["artifacts"]["initial_state"] = str(inside)
    context.metadata["artifacts"]["initial_state_sha256"] = hashlib.sha256(
        inside.read_bytes()).hexdigest()
    values, reason = _keyed_seed_residuals(context, None, (row,), 0.)
    assert values is None and "corrupt" in reason


def _pose(rotation, translation):
    """Build a homogeneous transform without relying on pyCamSet helpers."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def _reference_spreads(tforms):
    """Independent NumPy/SciPy reference using atan2 geodesic angles."""
    relative = tforms[1] @ np.linalg.inv(tforms[0])
    # Camera 1's centre in camera 0's frame is the translation of the inverse
    # relative transform T_{0<-1}, computed here by matrix inversion rather
    # than the production -R^T t formula.
    centres = np.linalg.inv(relative)[:, :3, 3]
    mean_c = centres.mean(axis=0)
    position = np.sqrt(np.mean(np.sum((centres - mean_c) ** 2, axis=1)))
    mean_r = Rotation.from_matrix(relative[:, :3, :3]).mean()
    angles = (mean_r.inv() * Rotation.from_matrix(relative[:, :3, :3])).magnitude()
    rotation = np.degrees(np.sqrt(np.mean(angles ** 2)))
    return position, rotation


def test_rig_spreads_use_camera_zero_chordal_mean_and_photo_deviations():
    # Two camera-in-target pose stacks generate nonzero translation and angle
    # scatter; an arbitrary ref_cam option is deliberately not an input here.
    reference = np.array([_pose([0.0, 0.0, 0.0], [0.1 * i, 0.0, 1.0])
                          for i in range(10)])
    relative = np.array([_pose([0.0, 0.0, np.deg2rad(0.4 * (i - 4.5))],
                               [0.001 * i, 0.0002 * i * i, 0.0])
                         for i in range(10)])
    other = relative @ reference
    stack = np.stack([reference, other])
    target = SimpleNamespace(point_data=np.zeros((2, 2, 3)), square_size=1.0)
    result = derive_rig_spreads(stack, target, minimum_n=8, declared_range=(8, 80))[0]
    expected_position, expected_rotation = _reference_spreads(stack)

    assert result.camera == 1
    assert result.position_scatter_units == pytest.approx(expected_position, abs=1e-12)
    assert result.rotation_scatter_deg == pytest.approx(expected_rotation, abs=1e-9)
    assert result.usable_image_count == 10
    assert len(result.per_image_position_units) == 10
    assert result.worst_position_image is not None
    assert result.worst_rotation_image is not None


def test_rig_spreads_fixed_centre_with_pure_rotation_jitter_has_no_position_spread():
    """Rotation jitter about a fixed camera centre must not read as millimetres.

    T_{1<-0} = [R | t] with t = -R C: the vector t moves by about baseline x
    angle while camera 1's centre C stays put (Hartley & Zisserman, sec. 6.1).
    """
    from pyCamSet import PuzzleBoard

    target = PuzzleBoard(num_squares_x=5, num_squares_y=6, square_size=2.0)
    rng = np.random.default_rng(20261006)
    reference = np.array([_pose(rng.normal(0.0, 0.2, 3), [0.0, 0.0, 800.0])
                          for _ in range(20)])
    centre = np.array([1000.0, 0.0, 0.0])
    relative = []
    for _ in range(20):
        rotation = Rotation.from_rotvec(np.deg2rad(rng.normal(0.0, 0.9, 3) / np.sqrt(3)))
        matrix = rotation.as_matrix()
        relative.append(_pose(rotation.as_rotvec(), -matrix @ centre))
    other = np.array(relative) @ reference
    result = derive_rig_spreads(np.stack([reference, other]), target)[0]
    translation_spread = np.sqrt(np.mean(np.sum(
        (np.array(relative)[:, :3, 3] - np.array(relative)[:, :3, 3].mean(axis=0)) ** 2,
        axis=1)))
    assert translation_spread > 10.0  # the old t-based measure would say No
    assert result.rotation_scatter_deg < 5.0
    assert result.position_scatter_mm == pytest.approx(0.0, abs=1e-6)
    assert result.position_scatter_mm < 10.0
    assert not result.position_photo_outliers


def test_rig_spreads_rotation_near_zero_uses_stable_zero_not_arccos_floor():
    poses = np.array([_pose([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]) for _ in range(10)])
    stack = np.stack([poses, poses])
    target = SimpleNamespace(point_data=np.zeros((2, 2, 3)), square_size=1.0)
    result = derive_rig_spreads(stack, target)[0]
    assert result.rotation_scatter_deg == 0.0
    assert result.position_scatter_units == 0.0


def test_rig_spreads_keep_missing_pose_nan_at_original_image_index():
    poses = np.array([
        [_pose([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]) for _ in range(10)],
        [_pose([0.0, 0.0, 0.01 * i], [0.001 * i, 0.0, 1.0])
         for i in range(10)],
    ])
    poses[1, 4] = np.nan
    target = SimpleNamespace(point_data=np.zeros((2, 2, 3)), square_size=1.0)
    result = derive_rig_spreads(poses, target)[0]
    assert result.usable_image_count == 9
    assert len(result.per_image_position_units) == 10
    assert len(result.per_image_rotation_deg) == 10
    assert result.per_image_position_units[4] is None
    assert result.per_image_rotation_deg[4] is None


def test_flat_puzzle_board_scatter_uses_2mm_squares_and_respects_legacy_units():
    from pyCamSet import PuzzleBoard

    target = PuzzleBoard(num_squares_x=5, num_squares_y=6, square_size=2.0)
    reference = np.array([_pose([0, 0, 0], [0, 0, 100]) for _ in range(10)])
    other = np.array([
        _pose([0, 0, 0], [10.0 + image, 0.0, 100.0])
        for image in range(10)
    ])
    stack = np.stack([reference, other])
    verified = derive_rig_spreads(stack, target)[0]
    assert verified.position_scatter_units > 0.0
    assert verified.position_scatter_squares == pytest.approx(
        verified.position_scatter_units / 2.0,
    )
    assert verified.position_scatter_mm == pytest.approx(verified.position_scatter_units)
    legacy = derive_rig_spreads(stack, target, metric_units_verified=False)[0]
    assert legacy.position_scatter_mm is None
    assert legacy.position_scatter_squares == pytest.approx(verified.position_scatter_squares)
    assert not legacy.position_photo_outliers


def test_rig_spread_vector_shift_cases_and_rotation_threshold():
    from pyCamSet import PuzzleBoard

    target = PuzzleBoard(num_squares_x=5, num_squares_y=6, square_size=2.0)
    reference = np.array([_pose([0, 0, 0], [0, 0, 100]) for _ in range(20)])

    def derive(offsets, turns=None):
        turns = np.zeros(20) if turns is None else turns
        other = np.array([
            _pose([0, 0, np.deg2rad(turns[i])],
                  [100.0 + offsets[i, 0], offsets[i, 1], 100.0])
            for i in range(20)
        ])
        return derive_rig_spreads(np.stack([reference, other]), target)[0]

    late = np.zeros((20, 2))
    late[19, 1] = 30.0
    one = derive(late)
    assert one.position_scatter_mm == pytest.approx(30.0 * np.sqrt(19) / 20)
    assert one.position_scatter_mm < 10.0
    assert one.position_photo_outliers == (19,)
    assert one.rotation_scatter_deg == 0.0

    opposite = np.zeros((20, 2))
    opposite[:, 1] = [30.0, -30.0] * 10
    alternating = derive(opposite)
    assert alternating.position_scatter_mm == pytest.approx(30.0)
    assert np.std(np.linalg.norm(
        np.column_stack([100.0 + opposite[:, 0], opposite[:, 1]]), axis=1,
    )) == pytest.approx(0.0)
    assert alternating.rotation_scatter_deg == 0.0

    one_sided = np.zeros((20, 2))
    one_sided[10:, 1] = 30.0
    assert derive(one_sided).position_scatter_mm == pytest.approx(15.0)
    along_baseline = np.zeros((20, 2))
    along_baseline[:, 0] = [30.0, -30.0] * 10
    assert derive(along_baseline).position_scatter_mm == pytest.approx(30.0)
    turns = np.array([10.0, -10.0] * 10)
    assert derive(np.zeros((20, 2)), turns).rotation_scatter_deg > 5.0


def test_fixed_pose_rank_uses_production_flattened_keys_and_checks_collinearity():
    from pyCamSet import ChArUco
    from pyCamSet.workflow.education_derive import _fixed_pose_valid

    target = ChArUco(num_squares_x=8, num_squares_y=8, square_size=40.0)
    point_dims = target.point_data.shape[:-1]
    assert len(point_dims) == 2
    handler = SimpleNamespace(
        problem_opts={"fixed_pose": 0}, missing_poses=np.zeros(1, dtype=bool),
        target=target,
    )

    def flat_rows(keys):
        # TemplateHandler._flat_detections uses return_flattened_keys:
        # [camera, image, one linear point index, observed x, observed y].
        return np.array([[0, 0, key, 10.0, 12.0] for key in keys], dtype=float)

    assert _fixed_pose_valid(handler, flat_rows([0, 1, 7])) == (0, True)
    assert _fixed_pose_valid(handler, flat_rows([0, 1, 2])) == (0, False)


def test_per_photo_rule_has_minimum_n_and_declared_range_fail_closed():
    deviations = np.array([0.1] * 9 + [12.0])
    from pyCamSet.workflow.education_derive import _photo_outlier_indices

    indices, status = _photo_outlier_indices(
        deviations, 10.0, minimum_n=8, multiplier=5.0, declared_range=(8, 80))
    assert indices == (9,)
    assert status == "applied"
    indices, status = _photo_outlier_indices(
        deviations[:4], 1.0, minimum_n=8, multiplier=5.0, declared_range=(8, 80))
    assert indices == ()
    assert "fewer than 8" in status
    late_at_four = np.array([0.1, 0.1, 0.1, 30.0])
    indices, status = _photo_outlier_indices(
        late_at_four, 10.0, minimum_n=8, multiplier=5.0,
        declared_range=(8, 80),
    )
    assert indices == () and "fewer than 8" in status
    indices, status = _photo_outlier_indices(
        np.ones(81), 0.5, minimum_n=8, multiplier=5.0, declared_range=(8, 80))
    assert indices == ()
    assert "outside declared range" in status
    # N means *usable* images; missing-pose slots preserve original image indices.
    with_missing = np.full(100, np.nan)
    with_missing[:19] = 0.1
    with_missing[59] = 12.0
    indices, status = _photo_outlier_indices(
        with_missing, 10.0, minimum_n=8, multiplier=5.0,
        declared_range=(8, 80),
    )
    assert indices == (59,)
    assert status == "applied"


def _fake_context(tmp_path, *, phase="phase3", status="complete", verification="verified"):
    """Give derive_phase3 the narrow context contract, avoiding GUI imports."""
    from pyCamSet.workflow.education_trace import Verification

    phase3_path = tmp_path / "phase3.camset"
    phase2_path = tmp_path / "phase2.camset"
    phase3_path.write_bytes(b"phase3")
    phase2_path.write_bytes(b"phase2")
    metadata = {
        "run_id": "p3-1",
        "status": status,
        "report": {"blocking_flags": []},
        "diagnostics": {
            "D3.11_residual_xy_scatter": [[3.0, 4.0], [0.0, 2.0], [5.0, 12.0],
                                          [3.0, 4.0], [0.0, 2.0], [5.0, 12.0]],
            "D3.6_final_euclid_px": 20.0 / 3.0,
            "D3.12_per_camera_mean_reprojection": {"cam0": 9.0, "cam1": 2.0},
            "D3.3_per_image_initial_reprojection": [1.0, float("nan"), 3.0],
            "D3.5_initial_euclid_px": 2.0,
        },
        "artifacts": {
            "optimised_camset": str(phase3_path),
            "phase2_initial_camset_used": str(phase2_path),
        },
    }

    class Context:
        def __init__(self):
            self.phase = phase
            self.run_id = "p3-1"
            self.metadata = metadata
            self.status = status
            self.quality_check = "checked"

        def artifact(self, _phase=None):
            return phase3_path

        def verify_input(self, _parent):
            return SimpleNamespace(verification=Verification(verification), reason=None)

    return Context(), phase3_path, phase2_path


def test_saved_run_derivation_reproduces_metrics_and_marks_fixed_pose(monkeypatch, tmp_path):
    context, _, _ = _fake_context(tmp_path)
    handler = _FakeHandler()
    # Use an unverified legacy unit marker: values remain square-unit-only.
    target = SimpleNamespace(point_data=np.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    ]), square_size=1.0)
    handler.target = target
    handler.problem_opts = {"fixed_pose": 0, "ref_cam": 1}
    handler.missing_poses = np.zeros(3, dtype=bool)
    p3 = SimpleNamespace(calibration_handler=handler,
                         calibration_result=np.array([3, 4, 0, 2, 5, 12,
                                                      3, 4, 0, 2, 5, 12, 44, 55], float),
                         _saved_point_data_units="legacy/unverified")
    p2 = SimpleNamespace(calibration_handler=SimpleNamespace(), get_names=lambda: ["0", "1"])
    # Three non-collinear design points make fixed_pose 0 valid.
    handler.detection.get_data = lambda: np.array([
        [0, 0, 0, 0, 1, 2], [0, 0, 0, 1, 1, 2], [1, 0, 1, 0, 1, 2],
        [0, 2, 0, 0, 1, 2], [0, 2, 0, 1, 1, 2], [1, 2, 1, 0, 1, 2],
    ], dtype=float)
    handler._flat_detections = handler.detection.get_data
    monkeypatch.setattr("pyCamSet.workflow.education_derive._load_camset",
                        lambda path: p2 if Path(path).name == "phase2.camset" else p3)
    seed_stack = np.tile(np.eye(4), (2, 3, 1, 1))
    seed_stack[1, :, 0, 3] = [0.0, 1.0, 2.0]
    monkeypatch.setattr("pyCamSet.workflow.education_derive._seed_from_inputs",
                        lambda _p3, _p2: (seed_stack, np.eye(4),
                                          np.tile(np.eye(4), (3, 1, 1)),
                                          np.array([1.0, np.nan, 3.0])))
    # The fixture's embedded row counts weight the two finite seed values to 2.0.
    context.metadata["diagnostics"]["D3.5_initial_euclid_px"] = 2.0
    context.metadata["diagnostics"]["D3.12_per_camera_mean_reprojection"] = {
        "cam0": 3.5, "cam1": 13.0,
    }
    # A known rebuilt target does not verify the units of legacy saved poses.
    monkeypatch.setattr("pyCamSet.workflow.education_derive.target_point_data_units",
                        lambda _target: "m")
    monkeypatch.setattr("pyCamSet.workflow.education_derive.millimetres_per_point_data_unit",
                        lambda _target: 1000.0)

    result = derive_phase3(context)
    assert not isinstance(result, DerivationUnavailable)
    assert result.seed_matches_saved is True
    assert result.seed_weighted_mean_px == 2.0
    assert result.per_camera_residual_mean_px == (("cam0", 3.5), ("cam1", 13.0))
    assert result.fixed_pose == 0
    assert result.fixed_pose_valid is True
    # The recorded ref_cam=1 does not override the defined camera-0 reference.
    assert result.rig_spreads[0].camera == 1
    assert result.rig_spreads[0].position_scatter_units == pytest.approx(np.sqrt(2.0 / 3.0))
    assert "legacy saved unit unverified" in result.unit_disposition
    assert all(spread.position_scatter_mm is None for spread in result.rig_spreads)
    assert all(not spread.position_photo_outliers for spread in result.rig_spreads)
    assert result.input_verification == "verified"
    context.metadata["diagnostics"]["D3.12_per_camera_mean_reprojection"]["cam0"] = 3.501
    mismatch = derive_phase3(context)
    assert isinstance(mismatch, DerivationUnavailable)
    assert "does not match D3.12" in mismatch.reason


def test_saved_run_derivation_refuses_changed_edge_or_bad_scalar(monkeypatch, tmp_path):
    context, _, phase2_path = _fake_context(tmp_path, verification="differs")
    phase2_path.write_bytes(b"edited Phase 2 override camset")
    result = derive_phase3(context)
    assert isinstance(result, DerivationUnavailable)
    assert "input is not verified" in result.reason

    context, _, _ = _fake_context(tmp_path)
    from pyCamSet.workflow.education_trace import Verification
    context.verify_input = lambda _parent: SimpleNamespace(
        verification=Verification.VERIFIED, reason=None)
    context.metadata["diagnostics"]["D3.6_final_euclid_px"] = 1.0
    handler = _FakeHandler()
    handler.target = SimpleNamespace(point_data=np.zeros((2, 2, 3)), square_size=1.0)
    handler.detection.get_data = lambda: np.array([
        [0, 0, 2, 1, 2], [1, 0, 2, 1, 2], [0, 0, 1, 1, 2],
        [0, 1, 2, 1, 2], [1, 1, 2, 1, 2], [0, 1, 1, 1, 2],
    ], dtype=float)
    handler._flat_detections = handler.detection.get_data
    p3 = SimpleNamespace(calibration_handler=handler,
                         calibration_result=np.array([3, 4, 0, 2, 5, 12,
                                                      3, 4, 0, 2, 5, 12, 44, 55], float),
                         _saved_point_data_units="legacy/unverified")
    p2 = SimpleNamespace(calibration_handler=SimpleNamespace(), get_names=lambda: ["0", "1"])
    monkeypatch.setattr("pyCamSet.workflow.education_derive._load_camset",
                        lambda path: p2 if Path(path).name == "phase2.camset" else p3)
    result = derive_phase3(context)
    assert isinstance(result, DerivationUnavailable)
    assert "does not match D3.6" in result.reason


def test_derive_refuses_an_edited_phase2_override_artifact(tmp_path):
    from pyCamSet.workflow.education_trace import RunContext, Verification
    from pyCamSet.workflow.workspace import WorkspaceManager

    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    phase2_dir = workspace / "phase2_runs" / "p2"
    phase2_dir.mkdir(parents=True)
    phase2_artifact = phase2_dir / "initial_cameras.camset"
    phase2_artifact.write_bytes(b"original phase2 camset")
    manager.save_run("phase2", "p2", {
        "run_id": "p2", "artifacts": {"initial_camset": str(phase2_artifact)},
    })

    phase3_dir = workspace / "phase3_runs" / "p3"
    phase3_dir.mkdir(parents=True)
    output_path = phase3_dir / "optimised_cameras.camset"
    output_path.write_bytes(b"phase3 camset")
    used_override = workspace / "overrides" / "initial_cameras.camset"
    used_override.parent.mkdir()
    used_override.write_bytes(b"edited Phase 2 override camset")
    context = RunContext.from_run(manager, {
        "run_id": "p3", "_run_dir": str(phase3_dir),
        "status": "complete", "report": {"blocking_flags": []},
        "inputs": {"phase2_run_id": "p2"},
        "artifacts": {
            "optimised_camset": str(output_path),
            "phase2_initial_camset_used": str(used_override),
        },
    })

    edge = context.verify_input("phase2")
    assert edge.verification is Verification.DIFFERS
    refusal = derive_phase3(context)
    assert isinstance(refusal, DerivationUnavailable)
    assert "Phase 2 input is not verified" in refusal.reason


def test_frozen_a0_countercases_keep_n1_and_fixed_pose_negative_controls():
    fixture = Path(__file__).parent / "test_data" / "education" / "phase3_reconciliation_1db6067.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    assert payload["base_sha"] == "1db60672627d1b54a5f895b3eef190f6f361e986"
    fixed = payload["cases"]["fixed_pose_5"]
    n1 = payload["cases"]["N1_camera_0_rows_removed_image_0"]
    assert len(fixed["D3.3_per_image_px"]) == len(fixed["N_per_image"]) == 22
    assert fixed["D3.3_weighted_px"] == pytest.approx(4.243247485264933)
    assert fixed["D3.5_initial_euclid_px"] == pytest.approx(26.16966266536118)
    assert n1["N_per_image"][0] == 158
    assert n1["D3.3_weighted_px"] == pytest.approx(4.197327579639883)
    assert n1["D3.5_initial_euclid_px"] == pytest.approx(31.571436640072285)
    assert abs(fixed["D3.5_initial_euclid_px"] - fixed["D3.3_weighted_px"]) > 1.0
    assert abs(n1["D3.5_initial_euclid_px"] - n1["D3.3_weighted_px"]) > 1.0


def test_module_is_qt_free_at_import():
    import pyCamSet.workflow.education_derive as module
    assert "PySide6" not in module.__dict__
    assert callable(module.derive_phase3)


@pytest.mark.slow
@pytest.mark.data
@pytest.mark.skipif(
    int(cv2.__version__.split(".")[0]) < 5,
    reason=("OpenCV 5 measurement: OpenCV 4 places sub-pixel ChArUco corners slightly "
            "differently, so its results are not numerically comparable."),
)
def test_real_saved_camsets_rederive_phase3_metrics(tmp_path):
    """Exercise Phase 2->3, disk round-trips, and full derivation on corpus data."""
    from pyCamSet import ChArUco
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile,
        run_initial_calibration,
    )
    from pyCamSet.optimisation.camera_lockbox import CameraLockboxConfig
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow import phase3
    from pyCamSet.workflow.education_trace import RunContext
    from pyCamSet.workflow.workspace import WorkspaceManager

    fixture = Path(__file__).parent / "test_data" / "calibration_charuco"
    target = ChArUco(num_squares_x=20, num_squares_y=20,
                     square_size=4.0, legacy=False)
    detection, resolutions = detect_datapoints_in_imfile(
        f_loc=fixture, caching=False, calibration_target=target, threads=1,
    )
    phase2 = run_initial_calibration(detection, target, resolutions, save=False)
    phase2.set_resolutions_from_file(floc=fixture)
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    phase2_dir = workspace / "phase2_runs" / "p2"
    phase2_dir.mkdir(parents=True)
    phase2_path = phase2_dir / "initial_cameras.camset"
    phase2.save(phase2_path)
    manager.save_run("phase2", "p2", {
        "run_id": "p2", "status": "complete",
        "artifacts": {"initial_camset": str(phase2_path)},
    })

    baseline_handler, baseline_optimisation, lockbox_source, _ = phase3.solve(
        phase2, detection, target, fixed_params={},
        options={"outliers": "n", "fixed_pose": 0}, threads=1,
    )
    baseline_path = tmp_path / "phase3_lockbox_source.camset"
    lockbox_source.save(baseline_path)
    baseline_saved = load_CameraSet(baseline_path)
    phase2_for_solve = load_CameraSet(phase2_path)
    handler, optimisation, solved, stats = phase3.solve(
        phase2_for_solve, detection, target, fixed_params={},
        options={"outliers": "y", "fixed_pose": 0}, threads=1,
        lockbox_config=CameraLockboxConfig(enabled=True),
        lockbox_source_camset=lockbox_source, lockbox_warm_start=True,
        record_education=True,
    )
    phase3_dir = workspace / "phase3_runs" / "p3"
    phase3_dir.mkdir(parents=True)
    initial_path = phase3._write_initial_state(
        phase3_dir, handler.get_initial_params(), handler.missing_poses,
        float(stats["initial_euclid"]))
    phase3_path = phase3_dir / "optimised_cameras.camset"
    solved.save(phase3_path)
    reloaded = load_CameraSet(phase3_path)
    diagnostics = phase3._diagnostics(
        optimisation, handler, stats, float(stats["initial_euclid"]),
        float(stats["final_euclid"]), lambda _message: None,
    )

    manager.save_run("phase3", "p3", {
        "run_id": "p3", "status": "complete",
        "report": {"blocking_flags": []},
        "inputs": {"phase2_run_id": "p2"},
        "artifacts": {
            "optimised_camset": str(phase3_path),
            "initial_state": str(initial_path),
            "initial_state_sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
            "phase2_initial_camset_used": str(phase2_path),
        },
        "diagnostics": diagnostics,
    })
    p3_record = manager.find_run("phase3", "p3")
    context = RunContext.from_run(manager, p3_record)
    assert context.verify_input("phase2").verification.value == "verified"
    result = derive_phase3(context)
    assert not isinstance(result, DerivationUnavailable), getattr(result, "reason", None)
    assert result.status == "available"
    assert result.seed_observations is not None
    assert [(r.camera, r.image, r.key) for r in result.seed_observations] == [
        (r.camera, r.image, r.key) for r in result.observations]
    assert np.mean([r.magnitude_px for r in result.seed_observations]) == pytest.approx(
        diagnostics["D3.5_initial_euclid_px"], rel=1e-9)
    assert result.seed_observation_status == "verified on the same ordered optimised rows"
    from pyCamSet.workflow.education_playback import build_phase3_frames
    def fresh_playback_handler():
        replay = load_CameraSet(phase3_path).calibration_handler
        replay.lockbox_config = CameraLockboxConfig(enabled=True)
        replay.lockbox_source_camset = load_CameraSet(baseline_path)
        replay.lockbox_prior = None
        return replay

    with np.load(initial_path, allow_pickle=False) as snapshot:
        recorded_start = np.asarray(snapshot["calibration_params"], dtype=float)
    playback = build_phase3_frames(
        fresh_playback_handler,
        {"status": "available", "records": [
            {"kind": "accepted_step", "payload": handler.education_step_trace}]},
        expected_start=recorded_start, saved_params=solved.calibration_params)
    assert playback["status"] == "available", playback.get("reason")
    assert len(playback["frames"]) == len(handler.education_step_trace["steps"])
    assert result.input_verification == "verified"
    assert result.saved_point_units == result.rebuilt_point_units == "m"
    assert result.fixed_pose == 0 and result.fixed_pose_valid is True
    assert all(spread.position_scatter_mm is not None for spread in result.rig_spreads)
    # This report inequality is only valid for metre-family point data.
    from pyCamSet.utils.setup_reports import RigConsistencyReport
    from pyCamSet.workflow.education_derive import _seed_from_inputs

    seed_poses, _, _, _ = _seed_from_inputs(reloaded, load_CameraSet(phase2_path))
    report = RigConsistencyReport.from_transforms(seed_poses, ref_cam=0)
    for spread, prior in zip(result.rig_spreads, report.per_camera):
        assert spread.camera == prior.index
        assert spread.position_scatter_mm + 1e-9 >= prior.translation_stdev_mm
        assert spread.rotation_scatter_deg + 1e-9 >= prior.angle_stdev_deg
    assert sum(result.per_image_observation_count) == len(result.observations)
    assert result.seed_matches_saved is True
    assert result.weighted_seed_matches_d35 is False
    assert result.d35_convention == "unknown: weighted mean does not reconcile with D3.5"
    assert np.array_equal(
        np.asarray(result.residual_xy),
        np.asarray(diagnostics["D3.11_residual_xy_scatter"]),
    )
    assert result.residual_mean_px == pytest.approx(
        diagnostics["D3.6_final_euclid_px"], rel=1e-12,
    )
    assert len(result.per_camera_residual_mean_px) == len(diagnostics[
        "D3.12_per_camera_mean_reprojection"])
    assert np.any(handler.missing_poses)
    from pyCamSet.workflow.education_derive import _keyed_seed_residuals
    wrong_rows, reason = _keyed_seed_residuals(
        context, reloaded.calibration_handler, tuple(reversed(result.observations)),
        float(diagnostics["D3.5_initial_euclid_px"]))
    assert wrong_rows is None and "row keys" in reason
    with np.load(initial_path, allow_pickle=False) as saved_initial:
        altered_x = np.asarray(saved_initial["calibration_params"], dtype=float).copy()
    altered_x[0] += 1.0
    phase3._write_initial_state(phase3_dir, altered_x, handler.missing_poses,
                                float(stats["initial_euclid"]))
    altered = derive_phase3(context)
    assert not isinstance(altered, DerivationUnavailable)
    assert altered.seed_observations is None
    assert "digest mismatch" in altered.seed_observation_status
    context.metadata["artifacts"]["initial_state_sha256"] = hashlib.sha256(
        initial_path.read_bytes()).hexdigest()
    forged_but_inconsistent = derive_phase3(context)
    assert not isinstance(forged_but_inconsistent, DerivationUnavailable)
    assert forged_but_inconsistent.seed_observations is None
    assert "D3.5" in forged_but_inconsistent.seed_observation_status
    initial_path.unlink()
    missing = derive_phase3(context)
    assert not isinstance(missing, DerivationUnavailable)
    assert missing.seed_observations is None
    assert "missing" in missing.seed_observation_status

    # Independently project the saved solved points; the solver's convention is
    # projected minus observed, not observed minus projected.
    rows = baseline_handler._flat_detections()
    points_by_image = baseline_handler.get_bundle_adjustment_inputs(
        baseline_optimisation.x, make_points=True,
    )
    point_keys = rows[:, 2:-2].astype(int)
    point_indices = (
        point_keys[:, 0] if point_keys.shape[1] == 1
        else np.ravel_multi_index(
            point_keys.T, baseline_handler.target.point_data.shape[:-1],
        )
    )
    expected = np.empty((len(rows), 2))
    cam_dict = baseline_saved.get_cam_dict()
    for camera_index, name in enumerate(baseline_handler.cam_names):
        selected = np.flatnonzero(rows[:, 0].astype(int) == camera_index)
        for row_index in selected:
            image_index = int(rows[row_index, 1])
            projected = cam_dict[name].project_points(
                points_by_image[image_index, point_indices[row_index]],
            )
            expected[row_index] = projected - rows[row_index, -2:]
    saved_segment = np.asarray(
        baseline_saved.calibration_result[:2 * len(rows)],
    ).reshape(-1, 2)
    np.testing.assert_allclose(expected, saved_segment, rtol=1e-9, atol=1e-9)


@pytest.mark.slow
def test_a0_generator_healthy_noise_campaign_has_bounded_pair_triggers():
    """Exercise declared position/rotation triggers across all A0 camera pairs."""
    from itertools import combinations

    from equivalence.synthetic_rig import make_synthetic_rig
    from pyCamSet import ChArUco
    from pyCamSet.workflow.education_derive import _photo_outlier_indices

    # A0 supplies a real projected three-camera geometry; pose errors below are
    # injected in camera space at the measured healthy-run RMS levels.
    rig = make_synthetic_rig(
        ChArUco(num_squares_x=8, num_squares_y=8, square_size=40.0),
        camera_count=3, image_count=8,
    )
    assert len(rig.camera_names) == 3
    pairs = tuple(combinations(range(len(rig.camera_names)), 2))
    rng = np.random.default_rng(20260926)
    covariance_shapes = {
        "isotropic": ([1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]),
        "moderate_anisotropy": ([0.63, 0.33, 0.04], [0.63, 0.33, 0.04]),
        "strong_anisotropy": ([0.77, 0.18, 0.05], [0.77, 0.18, 0.05]),
        "planar_rotation": ([1 / 3, 1 / 3, 1 / 3], [0.494, 0.494, 0.012]),
    }
    for shape, (position_variances, rotation_variances) in covariance_shapes.items():
        position_root = np.diag(np.sqrt(position_variances))
        rotation_root = np.diag(np.sqrt(rotation_variances))
        pair_trials = {pair: 0 for pair in pairs}
        pair_triggers = {pair: 0 for pair in pairs}
        run_trials = 0
        run_triggers = 0
        triggers_by_n = {n: 0 for n in range(8, 81)}
        for n in range(8, 81):
            for _ in range(1000):
                run_flagged = False
                translation = rng.normal(size=(3, n, 3)) @ position_root.T
                rotation = rng.normal(size=(3, n, 3)) @ rotation_root.T
                translation -= translation.mean(axis=1, keepdims=True)
                rotation -= rotation.mean(axis=1, keepdims=True)
                # Per-camera perturbations yield the requested pair-RMS scales.
                for camera in range(3):
                    pos_rms = np.sqrt(np.mean(np.sum(translation[camera] ** 2, axis=1)))
                    rot_rms = np.sqrt(np.mean(np.sum(rotation[camera] ** 2, axis=1)))
                    translation[camera] *= (0.006 / np.sqrt(2)) / pos_rms
                    rotation[camera] *= (np.deg2rad(2.5) / np.sqrt(2)) / rot_rms
                for pair in pairs:
                    pos = np.linalg.norm(translation[pair[0]] - translation[pair[1]], axis=1)
                    rot = np.rad2deg(np.linalg.norm(
                        rotation[pair[0]] - rotation[pair[1]], axis=1,
                    ))
                    position_bad, _ = _photo_outlier_indices(
                        pos, 0.010, minimum_n=8, multiplier=5.0,
                        declared_range=(8, 80),
                    )
                    rotation_bad, _ = _photo_outlier_indices(
                        rot, 5.0, minimum_n=8, multiplier=5.0,
                        declared_range=(8, 80),
                    )
                    pair_trials[pair] += 1
                    flagged = bool(position_bad or rotation_bad)
                    pair_triggers[pair] += flagged
                    run_flagged |= flagged
                run_trials += 1
                run_triggers += run_flagged
                triggers_by_n[n] += run_flagged
        assert run_trials == 73_000
        assert run_triggers / run_trials < 0.02, (shape, run_triggers, run_trials)
        assert max(triggers_by_n[n] for n in range(8, 81)) <= 20, (shape, triggers_by_n)
        for pair in pairs:
            assert pair_trials[pair] == 73_000
            assert pair_triggers[pair] / pair_trials[pair] < 0.02, (shape, pair)
        worst_n = max(range(8, 81), key=lambda n: triggers_by_n[n])
        print(f"A2r3 healthy {shape}: run={run_triggers}/{run_trials}; "
              f"worst_N={worst_n}:{triggers_by_n[worst_n]}/1000; " + ", ".join(
            f"{pair[0]}-{pair[1]}={pair_triggers[pair]}/{pair_trials[pair]}"
            for pair in pairs
        ))


@pytest.mark.slow
def test_projected_detection_noise_and_late_position_rotation_triggers():
    """Check noisy A0 projections and isolated late-trigger controls."""
    import cv2

    from equivalence.synthetic_rig import make_synthetic_rig
    from pyCamSet import ChArUco
    from pyCamSet.workflow.education_derive import derive_rig_spreads

    target = ChArUco(num_squares_x=8, num_squares_y=8, square_size=40.0)
    rig = make_synthetic_rig(target, camera_count=3, image_count=40)
    points = np.asarray(target.point_data, dtype=float)
    rows = rig.detections.get_data()
    rng = np.random.default_rng(20260927)
    poses = np.full((3, 40, 4, 4), np.nan)
    for camera in range(3):
        for image in range(40):
            selected = rows[(rows[:, 0] == camera) & (rows[:, 1] == image)]
            keys = selected[:, 2:-2].astype(int)
            object_points = points[tuple(keys.T)]
            image_points = selected[:, -2:] + rng.normal(0.0, 0.25, (len(selected), 2))
            ok, rvec, tvec = cv2.solvePnP(
                object_points.astype(np.float64), image_points.astype(np.float64),
                rig.intrinsic, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE,
            )
            assert ok
            poses[camera, image, :3, :3] = cv2.Rodrigues(rvec)[0]
            poses[camera, image, :3, 3] = tvec.reshape(3)
            poses[camera, image, 3, :] = [0.0, 0.0, 0.0, 1.0]
    healthy = derive_rig_spreads(poses, target)
    assert all(not spread.position_photo_outliers for spread in healthy), healthy
    assert all(not spread.rotation_photo_outliers for spread in healthy), healthy

    late_image = 20
    shifted = poses.copy()
    shifted[1, late_image, :3, 3] += [0.05, 0.0, 0.0]
    position_case = derive_rig_spreads(shifted, target)
    assert late_image in position_case[0].position_photo_outliers
    assert not position_case[0].rotation_photo_outliers

    rotated = poses.copy()
    turn = Rotation.from_rotvec(np.deg2rad([10.0, 0.0, 0.0])).as_matrix()
    # Turn camera 2 about its own centre: left-multiplying the whole
    # target-in-camera transform by [turn | 0] leaves the camera centre fixed.
    rotated[2, late_image, :3, :3] = turn @ rotated[2, late_image, :3, :3]
    rotated[2, late_image, :3, 3] = turn @ poses[2, late_image, :3, 3]
    rotation_case = derive_rig_spreads(rotated, target)
    assert late_image in rotation_case[1].rotation_photo_outliers
    assert not rotation_case[1].position_photo_outliers


def test_a0_generator_late_30mm_position_triggers_for_every_noisy_n20_trial():
    """The declared late-position case must trigger per run, not on average."""
    from equivalence.synthetic_rig import make_synthetic_rig
    from pyCamSet import ChArUco
    from pyCamSet.workflow.education_derive import _photo_outlier_indices

    rig = make_synthetic_rig(
        ChArUco(num_squares_x=8, num_squares_y=8, square_size=40.0),
        camera_count=3, image_count=20,
    )
    assert len(rig.camera_names) == 3
    rng = np.random.default_rng(20260928)
    late_image = 19
    triggers = {(0, 1): 0, (1, 2): 0}
    for _ in range(1000):
        # Two millimetres is the 3-D position RMS per camera, not the
        # standard deviation of each Cartesian component.
        noise = rng.normal(0.0, 0.002 / np.sqrt(3.0), size=(3, 20, 3))
        noise[1, late_image, 0] += 0.030
        for pair in triggers:
            displacement = np.linalg.norm(noise[pair[0]] - noise[pair[1]], axis=1)
            flagged, _ = _photo_outlier_indices(
                displacement, 0.010, minimum_n=8, multiplier=5.0,
                declared_range=(8, 80),
            )
            triggers[pair] += late_image in flagged
    assert triggers == {(0, 1): 1000, (1, 2): 1000}, triggers


def test_rig_spread_and_pycamset_rig_check_can_straddle_the_10mm_level():
    """The education spread is not pyCamSet's own rig statistic: sideways jitter
    leaves the translation length (and so RigConsistencyReport) nearly unchanged,
    while the camera-centre RMS spread is over 10 mm. The copy must not attribute
    the 10 mm level on this spread to pyCamSet's own check."""
    from pyCamSet import PuzzleBoard
    from pyCamSet.utils.setup_reports import RigConsistencyReport

    target = PuzzleBoard(num_squares_x=5, num_squares_y=6, square_size=2.0)
    rng = np.random.default_rng(20261006)
    reference = np.array([_pose([0, 0, 0], [0, 0, 800.0]) for _ in range(30)])
    other = np.array([
        _pose([0, 0, 0], [100.0, *rng.normal(0.0, 10.0, 2)]) @ reference[i]
        for i in range(30)
    ])
    stack = np.stack([reference, other])
    education = derive_rig_spreads(stack, target)[0]
    report = RigConsistencyReport.from_transforms(stack)
    # from_transforms assumes metres (x1000); these poses are in mm.
    pycamset_mm = report.per_camera[0].translation_stdev_mm / 1000.0
    assert pycamset_mm < 10.0 < education.position_scatter_mm


def test_rig_spreads_name_a_camera_with_too_few_common_poses_instead_of_failing():
    """On a real PuzzleBoard cube set, one sparse PuzzleBoard camera sharing fewer than two posed
    photos with camera 0 made the whole Phase 3 derivation unavailable."""
    import numpy as np
    from types import SimpleNamespace
    from pyCamSet.workflow import education_derive as D

    eye = np.eye(4)
    nan = np.full((4, 4), np.nan)
    shifted = np.eye(4)
    shifted[0, 3] = 1.0
    # 3 cameras x 3 photos of target-in-camera poses; camera 2 only sees photo 0
    target_in_camera = np.array([[eye, shifted, eye], [eye, shifted, eye], [eye, nan, nan]])
    target = SimpleNamespace()
    try:
        D.derive_rig_spreads(target_in_camera, target, metric_units_verified=False)
        raised = False
    except ValueError:
        raised = True
    except Exception:
        raised = None
    if raised is None:
        import pytest
        pytest.skip("synthetic target lacks attributes this derivation needs")
    assert raised  # unchanged default: a caller that gives no list still fails closed
    unmeasured = []
    spreads = D.derive_rig_spreads(target_in_camera, target, metric_units_verified=False,
                                   unmeasured=unmeasured)
    assert unmeasured == [2]
    assert [s.camera for s in spreads] == [1]
