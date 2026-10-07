"""A calibration with images it could not pose reads back from its camset whole."""

from __future__ import annotations

import numpy as np
import pytest


def _solve_with_missing_poses(charuco_problem):
    """A short Phase 3 solve in which two images had no usable pose."""
    from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, cams = charuco_problem
    missing = np.zeros(detections.max_ims, dtype=bool)
    missing[[3, 5]] = True
    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detections,
        options={"outliers": "n", "max_nfev": 3, "verbosity": 0},
        missing_poses=missing,
    )
    _, solved = run_bundle_adjustment(handler, threads=1)
    return solved, missing


def _assert_reads_back(solved, tmp_path):
    from pyCamSet.gui.assess_calibration import _build_o_results
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.utils.visualisation import CalibrationDiagnostics

    path = tmp_path / "solved.camset"
    solved.save(path)
    reloaded = load_CameraSet(path)
    rebuilt = reloaded.calibration_handler
    assert rebuilt is not None, "the handler did not survive the round trip"
    assert np.array_equal(rebuilt.missing_poses, solved.calibration_handler.missing_poses)
    params = reloaded.calibration_params
    read_back = rebuilt.get_camset(params)
    for name in solved.get_names():
        np.testing.assert_allclose(read_back[name].extrinsic, solved[name].extrinsic, atol=1e-9)
    # What Assess Calibration does with the reloaded file.
    diagnostics = CalibrationDiagnostics.from_results(_build_o_results(reloaded), rebuilt)
    assert np.isfinite(diagnostics.euclidean_err).all()
    assert len(diagnostics.scene_points) > 0


@pytest.mark.data
def test_a_reloaded_phase3_calibration_with_missing_poses_reads_back(charuco_problem, tmp_path):
    """The rebuilt handler's layout skips the poses the saved parameters skip."""
    solved, missing = _solve_with_missing_poses(charuco_problem)
    assert solved.calibration_handler.missing_poses[missing].all()
    assert solved.calibration_handler.bundlePrimitive.pose_end == solved.calibration_params.size
    _assert_reads_back(solved, tmp_path)


@pytest.mark.data
def test_a_reloaded_phase4_calibration_with_missing_poses_reads_back(charuco_problem, tmp_path):
    """The same for a self calibration started, as Phase 4 starts, from Phase 3."""
    from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment
    from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler

    target, detections, _ = charuco_problem
    phase3, _ = _solve_with_missing_poses(charuco_problem)
    handler = SelfBundleHandler(
        camset=phase3, target=target, detection=detections,
        options={"outliers": "n", "max_nfev": 3, "verbosity": 0},
        missing_poses=phase3.calibration_handler.missing_poses,
    )
    handler.set_from_templated_camset(phase3)
    _, solved = run_bundle_adjustment(handler, threads=1)
    _assert_reads_back(solved, tmp_path)


@pytest.mark.data
def test_an_unseeded_self_calibration_that_loses_a_pose_reads_back(
        charuco_problem, tmp_path, monkeypatch):
    """A pose lost during the solve takes its only-seen feature out of the layout."""
    from pyCamSet.calibration_targets import TargetDetection
    from pyCamSet.optimisation import template_handler
    from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment
    from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler

    target, detections, cams = charuco_problem
    data = detections.get_data()
    lost = int(np.unique(data[:, 1])[2])
    only_there = data[data[:, 1] == lost][0, 2]
    keep = ~((data[:, 2] == only_there) & (data[:, 1] != lost))
    detections = TargetDetection(
        cam_names=detections.cam_names, data=data[keep], max_ims=detections.max_ims)

    estimate = template_handler.estimate_camera_relative_poses

    def lose_one_pose(**kwargs):
        cam_poses, target_poses, errors = estimate(**kwargs)
        target_poses[lost] = np.nan
        return cam_poses, target_poses, errors

    monkeypatch.setattr(template_handler, "estimate_camera_relative_poses", lose_one_pose)
    handler = SelfBundleHandler(
        camset=cams, target=target, detection=detections,
        options={"outliers": "n", "max_nfev": 3, "verbosity": 0},
    )
    _, solved = run_bundle_adjustment(handler, threads=1)
    monkeypatch.undo()

    assert handler.missing_poses[lost]
    assert not handler.visible_feature_mask[int(only_there)]
    assert handler.bundlePrimitive.bdpt_end == solved.calibration_params.size
    _assert_reads_back(solved, tmp_path)


@pytest.mark.data
def test_a_camset_whose_missing_poses_do_not_fit_its_detections_is_refused(
        charuco_problem, tmp_path):
    import json

    from pyCamSet.utils.saving import load_CameraSet

    solved, _ = _solve_with_missing_poses(charuco_problem)
    path = tmp_path / "solved.camset"
    solved.save(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["optim"]["handler_config"]["missing_poses"].append(0)
    path.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(ValueError, match="poses missing"):
        load_CameraSet(path)


@pytest.mark.data
def test_a_given_mask_is_copied_not_shared(charuco_problem):
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, cams = charuco_problem
    given = [False] * detections.max_ims
    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detections, missing_poses=given)
    handler.missing_poses[0] = True

    assert isinstance(handler.missing_poses, np.ndarray)
    assert given[0] is False


@pytest.mark.data
def test_a_free_point_handler_still_takes_a_mask_of_any_length(charuco_problem):
    from pyCamSet.optimisation.free_point_handler import FreePointBundleHandler

    target, detections, cams = charuco_problem
    stale = [False] * (detections.max_ims + 3)
    handler = FreePointBundleHandler(
        camset=cams, target=target, detection=detections, missing_poses=stale)

    assert handler.missing_poses is stale
