"""The seam between the workflow phases and the optimisation backend.

Each phase reaches into the library through a small, unenforced contract:
a guarded import block at the top of the phase module, and a statistics
dictionary whose keys it reads by name.  Nothing held either end in place,
and both ends drifted -- ``run_bundle_adjustment_with_stats`` was removed
from ``optimisation_handling`` while three call sites still imported it.
Because the import sits in ``try: ... except ImportError``, the failure did
not surface as a missing name.  It set the backend flag false, and Phase 3
told the user "pyCamSet optimisation modules are unavailable" -- a message
about the install, for what was a rename.

So these tests are about the seam rather than about the solve:

* the guarded imports actually resolve, named one at a time, so a removal
  reports the symbol rather than flipping a boolean;
* the statistics dictionary carries every key the phases and the study
  driver read out of it;
* the two public entry points describe the same run, because they are the
  same solve.

The guards sit in :mod:`pyCamSet.workflow`, which imports no Qt, so nothing
here needs a GUI toolkit -- and nothing here may grow one.  The widgets
that read this contract are covered by
:mod:`tests.test_gui_phase_contracts`, and the windows they open by
:mod:`tests.test_out_of_process_viewers`.

The unit tests here use stubs and run in the fast suite.  The ones that
need a real solve share the session-scoped ChArUco fixtures.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from pyCamSet.calibration_targets.markers.aruco2 import ARUCO2_AVAILABLE

# The module rather than the names: a backend symbol that goes missing is
# what these tests are here to report, and importing it by name at module
# scope would turn that into a collection error that takes the whole file
# down before any test can say which symbol it was.
from pyCamSet.optimisation import optimisation_handling as backend

from pyCamSet.calibration_targets.ccube.target import Ccube
from pyCamSet.workflow.targets import (
    READING_ONLY_FIELDS,
    describe_target_mismatch,
    target_mismatch_message,
    target_params_of_run,
)

BACKEND_ENTRY_POINTS = (
    "get_bundle_adjustment_stats",
    "run_bundle_adjustment",
    "run_bundle_adjustment_with_stats",
)


@pytest.mark.parametrize("name", BACKEND_ENTRY_POINTS)
def test_the_backend_exposes_its_entry_points(name):
    """The library half of the contract, before anything calls into it."""
    assert callable(getattr(backend, name, None)), (
        f"optimisation_handling.{name} is missing; every phase tab guards "
        f"its import of it, so the tabs will report the whole backend as "
        f"unavailable rather than naming this symbol."
    )


# --------------------------------------------------------------------------
# The guarded imports at the top of each phase module
# --------------------------------------------------------------------------

# Mirrors the ``except ImportError`` branch in each phase, which sets exactly
# these names to None.  Listing them here means a backend symbol that goes
# away is reported as itself, not as "the modules are unavailable".
PHASE_BACKEND_NAMES = {
    "pyCamSet.workflow.phase1": (
        "detect_datapoints_in_imfile",
        "validate_detections",
    ),
    "pyCamSet.workflow.phase2": (
        "detect_datapoints_in_imfile",
        "run_initial_calibration",
        "load_pickle",
    ),
    "pyCamSet.workflow.phase3": (
        "CameraLockboxConfig",
        "run_bundle_adjustment_with_stats",
        "TemplateBundleHandler",
        "load_CameraSet",
        "load_pickle",
    ),
    "pyCamSet.workflow.phase4": (
        "run_bundle_adjustment_with_stats",
        "SelfBundleHandler",
        "load_CameraSet",
    ),
}


def _phase_module(name: str):
    return importlib.import_module(name)


@pytest.mark.parametrize("module_name", sorted(PHASE_BACKEND_NAMES))
def test_the_phase_finds_its_backend(module_name):
    """``BACKEND_OK`` false here is the dialog the user actually sees."""
    module = _phase_module(module_name)

    assert module.BACKEND_OK is True, (
        f"{module_name} fell back to its ImportError branch, so the phase "
        f"will refuse to run with 'pyCamSet optimisation modules are "
        f"unavailable'. Import the module directly to see the underlying "
        f"error."
    )


@pytest.mark.parametrize(
    ("module_name", "symbol"),
    [(m, s) for m, names in sorted(PHASE_BACKEND_NAMES.items()) for s in names],
)
def test_every_guarded_backend_name_resolves(module_name, symbol):
    """Named one at a time, so a removal says which symbol went."""
    module = _phase_module(module_name)

    assert getattr(module, symbol) is not None, (
        f"{module_name} could not import {symbol}; it is left as the None "
        f"sentinel from the ImportError branch."
    )


def test_a_broken_backend_says_why(monkeypatch, caplog):
    """The guard must name the cause, not just disable the phase.

    Reloading with the backend module poisoned takes the ImportError branch
    the same way a renamed symbol would.
    """
    import sys

    module_name = "pyCamSet.workflow.phase3"
    module = _phase_module(module_name)

    monkeypatch.setitem(
        sys.modules, "pyCamSet.optimisation.optimisation_handling", None)
    with caplog.at_level("WARNING"):
        reloaded = importlib.reload(module)

    try:
        assert reloaded.BACKEND_OK is False
        assert "Phase 3 optimisation backend unavailable" in caplog.text
    finally:
        # leave the module importable for everything after this test
        monkeypatch.undo()
        importlib.reload(module)


def test_the_study_driver_can_import_its_backend():
    """``optimisation_worker`` imports inside the function, with no guard.

    Nothing sets a flag there, so a missing backend symbol surfaces only
    once a trial is already running.
    """
    from pyCamSet.workflow.tuning import worker as optimisation_worker

    assert callable(optimisation_worker.default_phase3_fn)
    assert callable(optimisation_worker.default_phase4_fn)

    from pyCamSet.optimisation.optimisation_handling import (  # noqa: F401
        run_bundle_adjustment_with_stats as _worker_dependency,
    )
    from pyCamSet.optimisation.template_handler import (  # noqa: F401
        TemplateBundleHandler as _worker_handler,
    )


# --------------------------------------------------------------------------
# The statistics contract
# --------------------------------------------------------------------------

# Every key read out of the stats dict by a phase tab or by the study
# driver.  ``final_euclid`` is the one the study ranks trials on.
PHASE_STATS_KEYS = frozenset({
    "initial_euclid",
    "final_euclid",
    "elapsed_sec",
    "success",
    "message",
    "status",
    "nfev",
    "param_count",
    "observation_count",
})


class _StubResult:
    """The parts of an OptimizeResult the statistics read."""

    def __init__(self, fun, status=0, success=True, message="converged", nfev=7):
        self.fun = np.asarray(fun, dtype=float)
        self.x = np.zeros(4)
        self.status = status
        self.success = success
        self.message = message
        self.nfev = nfev


class _StubHandler:
    """A handler that reports a reprojection block of ``base_count``."""

    def __init__(self, base_count: int, **extras):
        self._base_count = base_count
        for name, value in extras.items():
            setattr(self, name, value)

    def get_base_residual_count(self) -> int:
        return self._base_count


def _stats(fun, init=None, handler=None, elapsed=1.5):
    init = fun if init is None else init
    return backend.get_bundle_adjustment_stats(
        _StubResult(fun), np.zeros(4), np.asarray(init, dtype=float),
        elapsed, param_handler=handler,
    )


def test_stats_carries_every_key_the_phases_read():
    """The contract the tabs and the study driver read by name."""
    stats = _stats(np.zeros(8))

    assert PHASE_STATS_KEYS <= set(stats), (
        f"missing from the stats contract: "
        f"{sorted(PHASE_STATS_KEYS - set(stats))}"
    )


def test_the_error_figures_are_mean_euclidean_pixels():
    """Residuals are (x, y) pairs; the figure is the mean of their norms."""
    # three points, each 3-4-5 off, so every norm is 5.0
    stats = _stats(np.tile([3.0, 4.0], 3), init=np.tile([6.0, 8.0], 3))

    assert stats["final_euclid"] == pytest.approx(5.0)
    assert stats["initial_euclid"] == pytest.approx(10.0)
    assert stats["observation_count"] == 3


def test_solver_fields_are_carried_through():
    stats = _stats(np.zeros(4))

    assert stats["status"] == 0
    assert stats["success"] is True
    assert stats["message"] == "converged"
    assert stats["nfev"] == 7
    assert stats["elapsed_sec"] == pytest.approx(1.5)
    assert stats["param_count"] == 4


def test_lockbox_priors_are_kept_out_of_the_error():
    """A prior residual is not a reprojection and must not be averaged in.

    Without the split the trailing prior inflates the reported error and,
    worse, the study driver ranks trials on it.
    """
    residuals = np.concatenate([np.tile([3.0, 4.0], 3), [100.0, 100.0]])

    split = _stats(residuals, handler=_StubHandler(6))
    unsplit = _stats(residuals, handler=_StubHandler(0))

    assert split["final_euclid"] == pytest.approx(5.0)
    assert split["observation_count"] == 3
    assert split["prior_residual_count"] == 2
    assert unsplit["final_euclid"] > split["final_euclid"]
    assert unsplit["prior_residual_count"] == 0


def test_a_handler_without_per_pose_diagnostics_omits_them():
    """The per-pose keys are additive; their absence is not a failure."""
    stats = _stats(np.zeros(8), handler=_StubHandler(0))

    assert "per_pose_initial_error_px" not in stats
    assert "outlier_poses_before_rejection" not in stats
    assert PHASE_STATS_KEYS <= set(stats)


def test_per_pose_diagnostics_are_reported_when_the_handler_has_them():
    handler = _StubHandler(
        0,
        initial_per_im_error=np.array([0.5, 2.0]),
        missing_poses_before_outlier_rejection=np.array([False, True]),
        missing_poses_after_outlier_rejection=np.array([True, True]),
    )

    stats = _stats(np.zeros(8), handler=handler)

    assert stats["per_pose_initial_error_px"] == [
        {"pose": 0, "initial_error_px": 0.5},
        {"pose": 1, "initial_error_px": 2.0},
    ]
    assert stats["outlier_poses_before_rejection"] == [1]
    assert stats["outlier_poses_after_rejection"] == [0, 1]


def test_no_handler_still_produces_the_contract():
    """The study driver's stub phases pass no handler."""
    stats = backend.get_bundle_adjustment_stats(
        _StubResult(np.zeros(8)), np.zeros(4), np.zeros(8), 0.25)

    assert PHASE_STATS_KEYS <= set(stats)
    assert stats["prior_residual_count"] == 0


def test_a_solver_that_reports_no_status_still_produces_ints():
    """The metadata writers cast these; None would raise there instead."""
    result = _StubResult(np.zeros(4))
    del result.status
    del result.nfev

    stats = backend.get_bundle_adjustment_stats(
        result, np.zeros(4), np.zeros(4), 0.1)

    assert stats["status"] == 0
    assert stats["nfev"] == 0


# --------------------------------------------------------------------------
# Both entry points, against a real solve
# --------------------------------------------------------------------------


@pytest.fixture
def short_charuco_handler(charuco_problem):
    """A real problem, stopped early: this is about plumbing, not accuracy."""
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, cams = charuco_problem
    return TemplateBundleHandler(
        camset=cams, target=target, detection=detections,
        options={"outliers": "n", "max_nfev": 2, "verbosity": 0},
    )


@pytest.mark.data
def test_a_real_solve_fills_in_the_whole_contract(short_charuco_handler):
    _optimisation, camset, stats = backend.run_bundle_adjustment_with_stats(
        short_charuco_handler, threads=1)

    assert PHASE_STATS_KEYS <= set(stats)
    assert camset is not None
    assert stats["observation_count"] > 0
    assert stats["param_count"] > 0
    assert np.isfinite(stats["initial_euclid"])
    assert np.isfinite(stats["final_euclid"])
    assert stats["elapsed_sec"] > 0
    assert isinstance(stats["message"], str)


@pytest.mark.data
def test_robust_loss_runs_on_the_schur_solver(short_charuco_handler, monkeypatch):
    """A non-linear loss stays on the Schur solver, which honours it.

    It used to fall back to scipy's trust region solver, which is far slower
    on a real rig; the Schur path now minimises the same robust objective.
    """
    short_charuco_handler.problem_opts.update({"loss": "soft_l1", "f_scale": 1.0})
    seen = []
    real = backend.run_schur_bundle_adjustment

    def recording(*args, **kwargs):
        seen.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(backend, "run_schur_bundle_adjustment", recording)
    optimisation, _camset, stats = backend.run_bundle_adjustment_with_stats(
        short_charuco_handler, threads=1)

    assert seen, "the robust loss fell back to the trust region solver"
    assert np.isfinite(stats["final_euclid"])
    # fun is the raw reprojection residual, and cost the robust objective,
    # which is below the plain half sum of squares for soft_l1.
    raw = 0.5 * float(optimisation.fun @ optimisation.fun)
    assert optimisation.cost <= raw + 1e-9


@pytest.mark.data
def test_the_solve_improves_on_where_it_started(short_charuco_handler):
    """Two iterations is not convergence, but it must not be worse."""
    _optimisation, _camset, stats = backend.run_bundle_adjustment_with_stats(
        short_charuco_handler, threads=1)

    assert stats["final_euclid"] <= stats["initial_euclid"]


@pytest.mark.data
def test_both_entry_points_describe_the_same_run(charuco_problem):
    """They share one solve; only the amount handed back differs.

    Each run gets its own deep copy of the cameras: a handler optimises the
    camera set it is given in place, so a second handler built on the same
    one would start from the first run's answer.
    """
    from copy import deepcopy

    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, cams = charuco_problem
    options = {"outliers": "n", "max_nfev": 2, "verbosity": 0}

    def handler():
        return TemplateBundleHandler(
            camset=deepcopy(cams), target=target, detection=detections,
            options=options,
        )

    plain_result, plain_cams = backend.run_bundle_adjustment(handler(), threads=1)
    stats_result, stats_cams, stats = backend.run_bundle_adjustment_with_stats(
        handler(), threads=1)

    assert np.allclose(plain_result.x, stats_result.x)
    assert plain_cams.get_names() == stats_cams.get_names()
    assert stats["nfev"] == stats_result.nfev
    assert stats["message"] == str(stats_result.message)


@pytest.mark.data
def test_the_stats_agree_with_the_calibration_report(short_charuco_handler):
    """The report is what a person reads; the stats are the same run."""
    optimisation, camset, stats = backend.run_bundle_adjustment_with_stats(
        short_charuco_handler, threads=1)

    report = camset.calibration_report
    assert report is not None
    assert stats["final_euclid"] == pytest.approx(report.mean_px, rel=1e-9)
    assert stats["initial_euclid"] == pytest.approx(
        report.initial_error_px, rel=1e-9)
    assert stats["param_count"] == report.n_parameters
    assert stats["observation_count"] == report.n_control_points
    assert stats["nfev"] == report.n_function_evals


# --------------------------------------------------------------------------
# The target a phase uses, against the run whose detections it reads
# --------------------------------------------------------------------------
#
# Phases 2 and 3 build their own target from their own fields and pair it
# with detections made by an earlier Phase 1 run.  Nothing tied the two
# together, so selecting a run detected with a different target produced
#
#     IndexError: index 80 is out of bounds for axis 1 with size 25
#
# from inside the target: a detection stores its keys as indices into the
# target's points, and key 80 is real in an 11x11 face and absent from a
# 5x5 one.  The tabs now adopt the run's target when the run changes, and
# refuse the run when the two still disagree.


# The two Ccube targets behind the reported failure.
CCUBE_12 = {"target": {"type": "Ccube", "n_points": 12, "length": 80.0,
                       "border_fraction": 0.1, "marker_backend": "aruco1"}}
CCUBE_6 = {"target": {"type": "Ccube", "n_points": 6, "length": 30.0,
                      "border_fraction": 0.1, "marker_backend": "aruco1"}}


def _with(params, **changes):
    """The same parameters, with the target's spec altered."""
    return {"target": {**params["target"], **changes}}


def _run(params, run_id="20260910_180530_fe4431"):
    return {"run_id": run_id, "phase": "phase1", "params": dict(params)}


def test_a_run_and_matching_settings_do_not_disagree():
    assert describe_target_mismatch(CCUBE_12, dict(CCUBE_12)) == []


def test_the_reported_mismatch_is_caught():
    """The exact case: n_points=12 detections against an n_points=6 target."""
    differences = describe_target_mismatch(CCUBE_12, CCUBE_6)

    assert len(differences) == 2
    assert any("n_points" in d and "12" in d and "6" in d for d in differences)
    assert any("length" in d and "80" in d and "30" in d for d in differences)


def test_a_different_target_type_is_reported_on_its_own():
    """No point listing field differences between incomparable targets."""
    differences = describe_target_mismatch(
        CCUBE_12, _with(CCUBE_12, type="ChArUco"))

    assert len(differences) == 1
    assert "target type" in differences[0]
    # Named as the interface shows them, since a person reads this.
    assert differences[0] == ("target type: the run used ChArUco1 ccube, "
                              "these settings say ChArUco1")


def test_a_target_is_described_by_the_label_it_is_shown_by():
    from pyCamSet.workflow.targets import describe_target

    assert describe_target(CCUBE_12).startswith("ChArUco1 ccube(")
    assert describe_target(
        {"target": {"type": "Ccube2", "n_points": 5, "length": 20.0}}
    ) == "ChArUco2 ccube(length=20.0, n_points=5)"


def test_detection_only_settings_are_not_a_mismatch():
    """The backend changes which points are found, not what a key means."""
    other_backend = _with(CCUBE_12, marker_backend="aruco2")

    assert describe_target_mismatch(CCUBE_12, other_backend) == []


def test_numbers_are_compared_as_numbers():
    """Run metadata round-trips through JSON, so 80 may arrive as "80.0"."""
    as_text = _with(CCUBE_12, n_points="12", length="80")

    assert describe_target_mismatch(CCUBE_12, as_text) == []


def test_a_spec_carries_its_own_targets_arguments_and_no_others():
    """A run used to record every field whichever target it used, so the
    comparison had to name the ones that mattered per type.  A spec is the
    target's own constructor arguments, so there is nothing else in it."""
    assert "num_squares_x" not in CCUBE_12["target"]
    assert set(CCUBE_12["target"]) - {"type"} <= set(
        Ccube.__init__.__code__.co_varnames)


def test_what_is_ignored_is_named_rather_than_what_is_compared():
    """Stated as an exclusion so that a new target, or a new argument on an
    existing one, is compared by default instead of quietly left out."""
    assert "marker_backend" in READING_ONLY_FIELDS
    assert "aruco_dict" in READING_ONLY_FIELDS
    assert "n_points" not in READING_ONLY_FIELDS


def test_a_run_whose_target_cannot_be_read_is_refused():
    """Under the shape this replaced, a run with no recognisable target made
    the check return early -- so it did not fail, it was skipped, and the
    index error it exists to prevent came back."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="no target spec"):
        describe_target_mismatch({}, CCUBE_6)
    with _pytest.raises(ValueError, match="no target spec"):
        describe_target_mismatch(CCUBE_12, {})

    assert target_params_of_run(None) == {}
    assert target_params_of_run({}) == {}


def test_the_run_parameters_come_off_the_run():
    assert target_params_of_run(_run(CCUBE_12)) == CCUBE_12


def test_the_message_names_the_run_and_every_difference():
    differences = describe_target_mismatch(CCUBE_12, CCUBE_6)

    message = target_mismatch_message("20260910_180530_fe4431", differences)

    assert "20260910_180530_fe4431" in message
    for difference in differences:
        assert difference in message
    assert "choose a run detected with this target" in message


def test_a_detector_setting_does_not_make_it_a_different_target():
    """``min_width`` was a PuzzleBoard constructor argument, so two boards
    detected at different widths read as different point layouts -- and the
    detections of one were refused to the other, though every key means the
    same thing in both."""
    def spec(min_width):
        return {"target": {"type": "PuzzleBoard", "num_squares_x": 10,
                           "num_squares_y": 10,
                           "detection_options": {"min_width": min_width}}}

    assert describe_target_mismatch(spec(4), spec(9)) == []
    assert describe_target_mismatch(
        spec(4), {"target": {**spec(4)["target"], "num_squares_x": 11}})


@pytest.mark.data
def test_the_mismatch_is_exactly_what_breaks_the_solve():
    """Ties the check to the failure it stands in for.

    The two targets' point layouts differ, so a key valid in one indexes
    off the end of the other -- which is the IndexError the guard exists
    to pre-empt.
    """
    from pyCamSet import Ccube

    big = Ccube(n_points=12, length=80.0, border_fraction=0.1)
    small = Ccube(n_points=6, length=30.0, border_fraction=0.1)

    assert big.point_local.shape[1] == 121   # 11 x 11 per face
    assert small.point_local.shape[1] == 25  # 5 x 5 per face

    # a key the big target produces, indexed into the small one
    with pytest.raises(IndexError):
        small.point_local[(np.array([0]), np.array([80]))]

    assert describe_target_mismatch(CCUBE_12, CCUBE_6) != []


# --------------------------------------------------------------------------
# The camera lockbox, as Phase 3 constructs it
# --------------------------------------------------------------------------
#
# Phase 3 passes lockbox_config, lockbox_source_camset and
# lockbox_warm_start to TemplateBundleHandler.  The handler's whole lockbox
# integration went out in the same merge that took
# run_bundle_adjustment_with_stats, so the tab failed with
#
#     TypeError: TemplateBundleHandler.__init__() got an unexpected
#     keyword argument 'lockbox_config'
#
# while camera_lockbox.py, and every defensive getattr for
# get_base_residual_count, stayed behind.

LOCKBOX_HANDLER_KWARGS = (
    "lockbox_config", "lockbox_source_camset", "lockbox_warm_start")


def test_the_handler_accepts_the_lockbox_arguments():
    """Signature-level, so it reports even without a problem to solve."""
    import inspect

    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    parameters = inspect.signature(TemplateBundleHandler.__init__).parameters
    missing = [k for k in LOCKBOX_HANDLER_KWARGS if k not in parameters]

    assert not missing, (
        f"TemplateBundleHandler does not accept {missing}; Phase 3 passes "
        f"these on every run and fails with TypeError without them."
    )


@pytest.fixture
def lockbox_source_camset(charuco_problem):
    """A solved camset, which is what Phase 3 locks onto.

    ``run_initial_calibration`` fits intrinsics only and leaves every
    extrinsic at the identity, so its output is not a usable source -- a
    lockbox needs poses to hold cameras near.  A short bundle adjustment
    is the cheapest thing that produces them.
    """
    from copy import deepcopy

    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, cams = charuco_problem
    _optimisation, solved = backend.run_bundle_adjustment(
        TemplateBundleHandler(
            camset=deepcopy(cams), target=target, detection=detections,
            options={"outliers": "n", "max_nfev": 2, "verbosity": 0},
        ),
        threads=1,
    )
    return solved


@pytest.fixture
def lockbox_problem(charuco_problem, lockbox_source_camset):
    """A handler factory over a real problem, lockbox on or off."""
    from copy import deepcopy

    from pyCamSet.optimisation.camera_lockbox import CameraLockboxConfig
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, cams = charuco_problem

    def make(enabled: bool, warm_start: bool = True):
        return TemplateBundleHandler(
            camset=deepcopy(cams), target=target, detection=detections,
            fixed_params=None,
            options={"outliers": "n", "max_nfev": 2, "verbosity": 0},
            lockbox_config=CameraLockboxConfig(enabled=enabled),
            # not copied: the handler only reads extrinsics off the source,
            # and a solved camset carries a target holding a cv2 aruco
            # dictionary, which cannot be deep-copied.
            lockbox_source_camset=lockbox_source_camset if enabled else None,
            lockbox_warm_start=warm_start,
        )

    return make


@pytest.mark.data
def test_an_intrinsics_only_camset_is_refused_as_a_source(charuco_problem):
    """Its cameras are all still at the identity, so it pins them to origin.

    With the warm start on it also overwrites the estimated extrinsics the
    solve would have started from, and the run dies on a NaN residual
    instead of on anything that names the source.
    """
    from pyCamSet.optimisation.camera_lockbox import CameraLockboxConfig
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, cams = charuco_problem
    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detections,
        options={"outliers": "n", "verbosity": 0},
        lockbox_config=CameraLockboxConfig(enabled=True),
        lockbox_source_camset=cams,  # straight out of Phase 2
    )

    with pytest.raises(ValueError, match="no extrinsics"):
        handler.get_initial_params()


@pytest.mark.data
def test_no_lockbox_leaves_the_residuals_alone(lockbox_problem):
    handler = lockbox_problem(enabled=False)
    loss = handler.make_loss_fun(1)

    residuals = loss(handler.get_initial_params())

    assert handler.get_base_residual_count() == residuals.size
    assert not handler.has_lockbox_priors()


@pytest.mark.data
def test_a_lockbox_appends_priors_after_the_reprojections(lockbox_problem):
    """One residual per constrained parameter, after the reprojection block."""
    handler = lockbox_problem(enabled=True)
    loss = handler.make_loss_fun(1)
    params = handler.get_initial_params()

    residuals = loss(params)
    base = handler.get_base_residual_count()
    prior = handler._ensure_lockbox_prior(len(params))

    assert handler.has_lockbox_priors()
    assert base % 2 == 0                       # whole (x, y) pairs
    assert residuals.size == base + prior.indices.size
    assert prior.indices.size > 0


@pytest.mark.data
def test_the_jacobian_grows_with_the_residuals(lockbox_problem):
    """A mismatch here is a shape error inside the solver."""
    handler = lockbox_problem(enabled=True)
    params = handler.get_initial_params()

    residuals = handler.make_loss_fun(1)(params)
    jacobian = handler.make_loss_jac(1)(params)

    assert jacobian.shape == (residuals.size, params.size)


@pytest.mark.data
def test_the_schur_solver_refuses_a_lockbox(lockbox_problem):
    """It is built from the reprojection blocks; priors are invisible to it.

    Left usable, it would optimise a different problem from the one the
    residuals report.
    """
    usable, reason = backend.can_use_schur(lockbox_problem(enabled=True))
    assert usable is False
    assert "lockbox" in reason

    usable_without, _ = backend.can_use_schur(lockbox_problem(enabled=False))
    assert usable_without is True


@pytest.mark.data
def test_the_bounds_reach_the_solver(lockbox_problem):
    """A lockbox constrains by bounding; unbounded, the priors only pull."""
    handler = lockbox_problem(enabled=True)
    params = handler.get_initial_params()

    lower, upper = handler.get_lockbox_bounds(len(params))
    prior = handler._ensure_lockbox_prior(len(params))

    assert lower.shape == upper.shape == params.shape
    assert np.all(np.isfinite(lower[prior.indices]))
    assert np.all(np.isfinite(upper[prior.indices]))
    assert np.all(lower[prior.indices] <= params[prior.indices])
    assert np.all(params[prior.indices] <= upper[prior.indices])
    # everything else stays free
    free = np.setdiff1d(np.arange(params.size), prior.indices)
    assert np.all(np.isneginf(lower[free]))
    assert np.all(np.isposinf(upper[free]))


@pytest.mark.data
def test_the_warm_start_begins_at_the_source_calibration(lockbox_problem):
    """The centre of a bound is the safest place to start inside it."""
    warm = lockbox_problem(enabled=True, warm_start=True)
    warm_params = warm.get_initial_params()
    prior = warm._ensure_lockbox_prior(len(warm_params))

    assert np.allclose(warm_params[prior.indices], prior.centres)


@pytest.mark.data
def test_priors_stay_out_of_the_reported_error(lockbox_problem):
    """The study ranks trials on final_euclid; a prior is not a reprojection."""
    _optimisation, _camset, stats = backend.run_bundle_adjustment_with_stats(
        lockbox_problem(enabled=True), threads=1)

    assert stats["prior_residual_count"] > 0
    assert stats["observation_count"] * 2 + stats["prior_residual_count"] > 0
    assert np.isfinite(stats["final_euclid"])
    # the error is the mean over reprojection pairs alone
    assert stats["observation_count"] > stats["prior_residual_count"]


@pytest.mark.data
def test_an_enabled_lockbox_without_a_source_says_so(charuco_problem):
    from pyCamSet.optimisation.camera_lockbox import CameraLockboxConfig
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, cams = charuco_problem
    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detections,
        options={"outliers": "n", "verbosity": 0},
        lockbox_config=CameraLockboxConfig(enabled=True),
        lockbox_source_camset=None,
    )

    with pytest.raises(ValueError, match="no source CameraSet"):
        handler._ensure_lockbox_prior(10)


@pytest.mark.data
def test_the_diagnostics_name_what_was_constrained(lockbox_problem):
    handler = lockbox_problem(enabled=True)
    params = handler.get_initial_params()

    diagnostics = handler.get_lockbox_diagnostics(params)

    assert diagnostics["enabled"] is True
    assert diagnostics["constrained_parameter_count"] > 0
    assert diagnostics["constrained_camera_names"]
    assert "final_parameter_deltas" in diagnostics


@pytest.mark.data
def test_a_disabled_lockbox_reports_itself_disabled(lockbox_problem):
    diagnostics = lockbox_problem(enabled=False).get_lockbox_diagnostics()

    assert diagnostics["enabled"] is False
    assert diagnostics["constrained_parameter_count"] == 0


# --------------------------------------------------------------------------
# Marking a pose missing must change the solve, not just the record
# --------------------------------------------------------------------------
#
# missing_poses reached exactly two places: find_and_exclude_transform_outliers
# set it, and get_detection_data dropped the rows.  The loss and its jacobian
# were built from self.detection unfiltered, so marking a pose changed the
# reported detection count and nothing else -- outlier rejection detected
# outliers, logged them, wrote them into run metadata, and then fitted them
# anyway.  calc_initial_params also assigned over whatever the caller had
# passed in, so a marking made at construction never survived to be ignored.


@pytest.fixture
def marking_problem(charuco_problem):
    """A handler factory over a real problem, with poses marked or not."""
    from copy import deepcopy

    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, cams = charuco_problem

    def make(marked=None, solver="schur", max_nfev=3):
        missing = None
        if marked is not None:
            flags = np.zeros(int(detections.max_ims), dtype=bool)
            flags[marked] = True
            missing = list(flags)
        return TemplateBundleHandler(
            camset=deepcopy(cams), target=target, detection=detections,
            options={"outliers": "n", "max_nfev": max_nfev,
                     "verbosity": 0, "solver": solver},
            missing_poses=missing,
        )

    return make


def _worst_pose(handler):
    """The image contributing the most error, which is worth excluding."""
    params = handler.get_initial_params()
    residuals = handler.make_loss_fun(1)(params).reshape(-1, 2)
    rows = handler._flat_detections()
    per_image = {
        int(i): float(np.mean(np.linalg.norm(residuals[rows[:, 1] == i], axis=1)))
        for i in np.unique(rows[:, 1])
    }
    return max(per_image, key=per_image.get)


@pytest.mark.data
def test_marking_a_pose_reduces_the_calibration_loss(marking_problem):
    """The point of marking a pose: the solve stops fitting it."""
    worst = _worst_pose(marking_problem())

    _op, _cams, unmarked = backend.run_bundle_adjustment_with_stats(
        marking_problem(), threads=1)
    _op, _cams, marked = backend.run_bundle_adjustment_with_stats(
        marking_problem(marked=worst), threads=1)

    assert marked["observation_count"] < unmarked["observation_count"]
    assert marked["final_euclid"] < unmarked["final_euclid"], (
        f"marking pose {worst} left the error at "
        f"{marked['final_euclid']:.4f} px against "
        f"{unmarked['final_euclid']:.4f} px unmarked: the marking is being "
        f"recorded but not applied to the residuals."
    )


@pytest.mark.data
def test_a_marking_made_at_construction_survives(marking_problem):
    """calc_initial_params assigned the NaN scan straight over the top."""
    handler = marking_problem(marked=3)

    handler.get_initial_params()

    assert handler.missing_poses[3], (
        "the marking passed to the constructor was discarded by "
        "calc_initial_params"
    )


@pytest.mark.data
def test_the_scan_can_still_add_to_what_the_caller_marked(marking_problem):
    """Merged, not replaced, in either direction."""
    handler = marking_problem(marked=3)
    handler.get_initial_params()

    # nothing in this corpus is unposed, so the merge is the caller's set
    assert int(np.sum(handler.missing_poses)) >= 1


@pytest.mark.data
def test_the_excluded_pose_leaves_the_residuals(marking_problem):
    """Its rows, and only its rows."""
    plain = marking_problem()
    plain.get_initial_params()
    before = plain._flat_detections()

    handler = marking_problem(marked=3)
    handler.get_initial_params()
    after = handler._flat_detections()

    assert not np.any(after[:, 1] == 3)
    assert after.shape[0] == int(np.sum(before[:, 1] != 3))


@pytest.mark.data
def test_an_excluded_pose_stops_being_a_free_parameter(marking_problem):
    """Six parameters with no observations are six all-zero jacobian columns.

    The degeneracy check rejects those outright, and the Schur elimination
    would divide by the empty block, so excluding a pose has to fix it.
    """
    plain = marking_problem()
    free_before = plain.get_initial_params().size

    handler = marking_problem(marked=3)
    free_after = handler.get_initial_params().size

    assert free_after == free_before - 6
    assert not handler.bundlePrimitive.poses_unfixed[3]


@pytest.mark.data
def test_the_jacobian_stays_square_with_the_residuals_when_marking(marking_problem):
    handler = marking_problem(marked=3)
    params = handler.get_initial_params()

    residuals = handler.make_loss_fun(1)(params)
    jacobian = handler.make_loss_jac(1)(params)

    assert jacobian.shape == (residuals.size, params.size)


@pytest.mark.data
@pytest.mark.parametrize("solver", ["schur", "trf"])
def test_both_solvers_handle_a_marked_pose(marking_problem, solver):
    """The block solver is the one that would divide by the empty block."""
    worst = _worst_pose(marking_problem())

    optimisation, _cams, stats = backend.run_bundle_adjustment_with_stats(
        marking_problem(marked=worst, solver=solver), threads=1)

    assert np.all(np.isfinite(optimisation.x))
    assert np.isfinite(stats["final_euclid"])


@pytest.mark.data
def test_what_the_handler_reports_is_what_it_fitted(marking_problem):
    """get_detection_data and the loss must agree on the row count.

    They disagreed: the handler reported the exclusion while the optimiser
    fitted everything, so the run metadata described a solve that never
    happened.
    """
    handler = marking_problem(marked=3)
    params = handler.get_initial_params()

    reported = handler.get_detection_data().shape[0]
    fitted = handler.make_loss_fun(1)(params).size // 2

    assert reported == fitted


@pytest.mark.data
def test_marking_nothing_changes_nothing(marking_problem):
    """The unmarked path has to be untouched by all of this."""
    handler = marking_problem()
    params = handler.get_initial_params()

    assert not np.any(handler.missing_poses)
    assert handler._flat_detections().shape[0] * 2 == \
        handler.make_loss_fun(1)(params).size


def test_a_marking_of_the_wrong_length_is_refused(charuco_problem):
    """Silently mismatched flags would exclude the wrong images."""
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, cams = charuco_problem
    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detections,
        options={"outliers": "n", "verbosity": 0},
        missing_poses=[True, False],  # far too short
    )

    with pytest.raises(ValueError, match="missing_poses has 2 entries"):
        handler.get_initial_params()


# ---------------------------------------------------------------------------
# A detection cache is the detector's own
# ---------------------------------------------------------------------------
#
# The detector is chosen per Phase 1 run now, and a run caches its
# detections beside the images.  Under one cache name, a run read with
# ArUco 2 would silently load what an ArUco 1 run cached in the same folder
# -- detections of the other detector, reported as its own.


class _CountingTarget:
    """Stands in for a target: every camera folder is one detection."""

    DETECTOR_BACKENDS = {"aruco1": None, "aruco2": None}

    def __init__(self, marker_backend):
        self.marker_backend = marker_backend
        self.folders_read = 0

    def find_in_imfolder(self, *_args, **_kwargs):
        self.folders_read += 1
        return 1


def _image_folder(root):
    import cv2

    for camera in ("cam0", "cam1"):
        (root / camera).mkdir(parents=True)
        cv2.imwrite(str(root / camera / "im0.png"),
                    np.zeros((8, 12, 3), dtype=np.uint8))
    return root


@pytest.mark.parametrize(("upscale", "marker_backend", "expected"), [
    (1, None, "detected_datapoints.pickle"),
    (1, "aruco1", "detected_datapoints.pickle"),
    (1, "puzzle_board", "detected_datapoints.pickle"),
    (1, "aruco2", "detected_datapoints_aruco2.pickle"),
    (3, "aruco1", "detected_datapoints_upscale3x.pickle"),
    (3, "aruco2", "detected_datapoints_upscale3x_aruco2.pickle"),
])
def test_the_cache_is_named_for_its_detector_and_aruco1_keeps_the_old_name(
        upscale, marker_backend, expected):
    from pyCamSet.workflow.detections import detection_cache_name

    assert detection_cache_name(upscale, marker_backend) == expected


@pytest.mark.parametrize(("marker_backend", "upscale"), [
    ("aruco1", 1), ("aruco2", 1), ("aruco2", 2)])
def test_the_detection_pass_writes_the_name_phase_1_looks_for(
        tmp_path, monkeypatch, marker_backend, upscale):
    """Two statements of one name: the pass that writes it, and the phase
    that copies it into the run."""
    from pyCamSet.calibration import camera_calibrator as calibrator
    from pyCamSet.workflow.phase1 import _cache_name_of

    written = []
    monkeypatch.setattr(calibrator, "save_pickle",
                        lambda _data, path: written.append(path.name))
    calibrator.detect_datapoints_in_imfile(
        _image_folder(tmp_path / "images"), _CountingTarget(marker_backend),
        caching=True, upscale_factor=upscale)

    params = {"target": {"type": "Ccube", "marker_backend": marker_backend},
              "upscale_factor": upscale}
    assert written == [_cache_name_of(params)]


def _registered_counting_target(spec, per_folder):
    """A real, registered target whose detection is stubbed.

    Registered, so its cache identity is genuinely computed: an unregistered
    stub such as :class:`_CountingTarget` has no identity at all, which makes
    every cache a miss whatever its name, and a test built on it proves
    nothing about how caches are named.
    """
    from pyCamSet.calibration_targets.core.target_registry import build_target

    target = build_target(spec)
    target.folders_read = 0

    def find_in_imfolder(*_args, **_kwargs):
        target.folders_read += 1
        return per_folder

    target.find_in_imfolder = find_in_imfolder
    return target


@pytest.mark.skipif(not ARUCO2_AVAILABLE,
                    reason="reads a target that only ArUco 2 detects")
def test_an_aruco2_run_never_loads_an_aruco1_runs_cache(tmp_path):
    """Two detectors reading one image folder keep two caches: the ArUco 2
    run neither loads nor overwrites what the ArUco 1 run cached, and the
    ArUco 1 run still finds its own cache afterwards."""
    from pyCamSet.calibration import camera_calibrator as calibrator

    images = _image_folder(tmp_path / "images")
    aruco1_spec = {"type": "Ccube", "marker_backend": "aruco1"}
    aruco2_spec = {"type": "Ccube", "marker_backend": "aruco2"}

    aruco1 = _registered_counting_target(aruco1_spec, per_folder=1)
    detected, _ = calibrator.detect_datapoints_in_imfile(images, aruco1, caching=True)
    assert (detected, aruco1.folders_read) == (2, 2)
    aruco1_cache = images / "detected_datapoints.pickle"
    aruco1_bytes = aruco1_cache.read_bytes()

    aruco2 = _registered_counting_target(aruco2_spec, per_folder=10)
    detected, _ = calibrator.detect_datapoints_in_imfile(images, aruco2, caching=True)
    assert aruco2.folders_read == 2, "detected afresh"
    assert detected == 20, "loaded another detector's cache"
    assert (images / "detected_datapoints_aruco2.pickle").exists()
    assert aruco1_cache.read_bytes() == aruco1_bytes, \
        "the ArUco 2 run wrote over the ArUco 1 run's cache"

    again = _registered_counting_target(aruco1_spec, per_folder=1)
    detected, _ = calibrator.detect_datapoints_in_imfile(images, again, caching=True)
    assert (detected, again.folders_read) == (2, 0), \
        "the ArUco 1 run lost its own cache"


_needs_aruco2 = pytest.mark.skipif(
    not ARUCO2_AVAILABLE, reason="reads a target that only ArUco 2 detects")


@pytest.mark.parametrize("target", [
    {"type": "Ccube", "marker_backend": "aruco1"},
    pytest.param({"type": "Ccube", "marker_backend": "aruco2"}, marks=_needs_aruco2),
    pytest.param({"type": "ChArUco2"}, marks=_needs_aruco2),
])
def test_a_failed_phase_1_run_records_its_error_and_no_artifact_even_with_a_matching_cache(
        tmp_path, monkeypatch, target):
    """OVERSEER structural fix (round 5): phase1.run()'s old fallback --
    adopting the image folder's own cache as the run's artifact once
    _detect() raised -- is removed entirely, not just guarded further. A run
    whose detection failed records its error and has no detections artifact,
    even when a cache that WOULD have matched this run's own identity (by
    the cache's identity sidecar, not just its filename) is sitting right
    there -- there is no longer any code path in run() that looks at it."""
    from pyCamSet.calibration.camera_calibrator import write_cache_identity
    from pyCamSet.calibration_targets.core.target_registry import build_target
    from pyCamSet.workflow import phase1
    from pyCamSet.workflow.workspace import WorkspaceManager

    images = _image_folder(tmp_path / "images")
    cam_names = ["cam0", "cam1"]

    matching_cache = images / phase1._cache_name_of({"target": target})
    matching_cache.write_bytes(b"a cache that matches this run's own identity")
    write_cache_identity(matching_cache, build_target(target), cam_names, None)

    def failing_detect(*_args):
        raise RuntimeError("detection failed")

    monkeypatch.setattr(phase1, "_detect", failing_detect)
    metadata = phase1.run({"f_loc": str(images), "target": target},
                          WorkspaceManager(tmp_path / "workspace"))

    assert metadata["error"] == "detection failed"
    assert "detected_datapoints_pickle" not in metadata.get("artifacts", {})
    # The matching cache is untouched -- proves this is not adopted, not
    # merely that some OTHER file was adopted instead.
    assert matching_cache.read_bytes() == b"a cache that matches this run's own identity"


@pytest.mark.skipif(not ARUCO2_AVAILABLE,
                    reason="reads a target that only ArUco 2 detects")
def test_a_phase_1_run_adopts_only_its_own_detectors_cache(tmp_path, monkeypatch):
    """The detector-isolation guarantee itself, exercised through a REAL,
    non-raising phase1.run() pass -- round-10 review, P3: the version this
    replaced (``test_a_phase_1_run_adopts_no_other_detectors_cache``)
    monkeypatched ``phase1._detect`` to raise immediately, before any
    cache-name code ran at all, so its assertion held unconditionally --
    confirmed empirically: it still passed with ``detector_backend_of``
    itself disabled (``lambda target: None``).

    ``find_in_imfolder`` and ``validate_detections`` are stubbed, the way
    this file's other ``_CountingTarget``-based tests already stub
    detection, so this stays a fast unit test -- but the target itself is a
    real, registered ``Ccube``, built through ``target_of_params`` exactly
    as a real run would, so its cache identity is genuinely computed and
    genuinely written.  That is the one thing that actually exercises
    detector-cache isolation: a stub target (unregistered) always reads as
    an unconfirmable identity regardless of its cache's name, which is why
    the version this replaces could not have caught the regression even
    had it reached the cache-name code at all.
    """
    from pyCamSet.calibration.camera_calibrator import cache_identity_path, cache_matches
    from pyCamSet.calibration_targets.core.target_registry import build_target
    from pyCamSet.workflow import phase1
    from pyCamSet.workflow.workspace import WorkspaceManager

    images = _image_folder(tmp_path / "images")
    (images / "detected_datapoints.pickle").write_bytes(b"ArUco 1's detections")

    target_spec = {"type": "Ccube", "marker_backend": "aruco2"}
    real_target = build_target(target_spec)
    folders_read = {"count": 0}

    def fake_find_in_imfolder(*_args, **_kwargs):
        folders_read["count"] += 1
        return 1

    real_target.find_in_imfolder = fake_find_in_imfolder
    monkeypatch.setattr(phase1, "target_of_params", lambda _params: real_target)

    class _FakeReport:
        per_camera: list = []
        camera_names: list = []
        n_images = 0

        def to_dict(self):
            return {}

    monkeypatch.setattr(phase1, "validate_detections",
                        lambda *_a, **_k: _FakeReport())

    metadata = phase1.run(
        {"f_loc": str(images), "target": target_spec, "caching": True,
         "n_lim": None, "selected_cameras": [], "upscale_factor": 1},
        WorkspaceManager(tmp_path / "workspace"))

    assert metadata["error"] is None, metadata["error"]
    assert folders_read["count"] == 2, "a real, non-raising detection pass ran"
    assert "detected_datapoints_pickle" in metadata.get("artifacts", {})

    # The ArUco 1 cache sitting in the same folder must never have been read
    # as this ArUco 2 run's own result -- proven by the fact it is still
    # exactly the (invalid) bytes seeded above.
    assert (images / "detected_datapoints.pickle").read_bytes() == b"ArUco 1's detections"

    # And this run's own cache lands under its own, detector-specific name,
    # with an identity that genuinely confirms it -- the guarantee itself:
    # two detectors sharing an image folder never share a cache slot.
    aruco2_cache = images / "detected_datapoints_aruco2.pickle"
    assert aruco2_cache.exists()
    assert cache_identity_path(aruco2_cache).exists()
    assert cache_matches(aruco2_cache, real_target, ["cam0", "cam1"], None)


def test_a_target_without_a_choice_is_named_for_its_only_detector():
    """ChArUco2 has no ``marker_backend`` attribute: it is only ever read
    with ArUco 2, and its cache must say so."""
    from pyCamSet.calibration.camera_calibrator import detector_backend_of
    from pyCamSet.workflow.targets import detector_backend_of_spec

    class OnlyAruco2:
        DETECTOR_BACKENDS = {"aruco2": None}

    class NoneDeclared:
        DETECTOR_BACKENDS = {}

    assert detector_backend_of(OnlyAruco2()) == "aruco2"
    assert detector_backend_of(NoneDeclared()) is None
    assert detector_backend_of(_CountingTarget("aruco2")) == "aruco2"

    assert detector_backend_of_spec({"type": "ChArUco2"}) == "aruco2"
    assert detector_backend_of_spec({"type": "PuzzleBoard"}) == "puzzle_board"
    assert detector_backend_of_spec({"type": "ChArUco"}) == "aruco1", \
        "a spec naming no detector is read with the constructor's default"
    assert detector_backend_of_spec(
        {"type": "Ccube", "marker_backend": "aruco2"}) == "aruco2"
