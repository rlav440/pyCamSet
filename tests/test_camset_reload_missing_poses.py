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


def _assert_reads_back(solved, missing, tmp_path):
    from pyCamSet.gui.assess_calibration import _build_o_results
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.utils.visualisation import CalibrationDiagnostics

    path = tmp_path / "solved.camset"
    solved.save(path)
    reloaded = load_CameraSet(path)
    rebuilt = reloaded.calibration_handler
    assert rebuilt is not None, "the handler did not survive the round trip"
    assert np.array_equal(np.asarray(rebuilt.missing_poses, dtype=bool)[:missing.size], missing)
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
    """The saved parameters skip the missing poses; the rebuilt handler must too.

    Before the fix the handler rebuilt from the file kept every missing pose
    in its layout, so reading the saved vector back failed with "cannot
    reshape array of size ... into shape (n, 6)" -- the Assess Calibration
    error on any run with an image the solve could not pose.
    """
    solved, missing = _solve_with_missing_poses(charuco_problem)
    assert solved.calibration_handler.bundlePrimitive.pose_end == solved.calibration_params.size
    _assert_reads_back(solved, missing, tmp_path)


@pytest.mark.data
def test_a_reloaded_phase4_calibration_with_missing_poses_reads_back(charuco_problem, tmp_path):
    """The same for a self calibration started, as Phase 4 starts, from Phase 3."""
    from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment
    from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler

    target, detections, _ = charuco_problem
    phase3, missing = _solve_with_missing_poses(charuco_problem)
    handler = SelfBundleHandler(
        camset=phase3, target=target, detection=detections,
        options={"outliers": "n", "max_nfev": 3, "verbosity": 0},
        missing_poses=phase3.calibration_handler.missing_poses,
    )
    handler.set_from_templated_camset(phase3)
    _, solved = run_bundle_adjustment(handler, threads=1)
    _assert_reads_back(solved, missing, tmp_path)
