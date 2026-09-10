"""The seam between the GUI phase tabs and the optimisation backend.

The phase tabs reach into the library through a small, unenforced contract:
a guarded import block at the top of each tab, and a statistics dictionary
whose keys the tabs read by name.  Nothing held either end in place, and
both ends drifted -- ``run_bundle_adjustment_with_stats`` was removed from
``optimisation_handling`` while three call sites still imported it.  Because
the import sits in ``try: ... except ImportError``, the failure did not
surface as a missing name.  It set ``_PYCAMSET_OK = False``, and Phase 3
told the user "pyCamSet optimisation modules are unavailable" -- a message
about the install, for what was a rename.

So these tests are about the seam rather than about the solve:

* the guarded imports actually resolve, named one at a time, so a removal
  reports the symbol rather than flipping a boolean;
* the statistics dictionary carries every key the tabs and the study driver
  read out of it;
* the two public entry points describe the same run, because they are the
  same solve.

The unit tests here use stubs and run in the fast suite.  The one test that
needs a real solve shares the session-scoped ChArUco fixtures.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# The module rather than the names: a backend symbol that goes missing is
# what these tests are here to report, and importing it by name at module
# scope would turn that into a collection error that takes the whole file
# down before any test can say which symbol it was.
from pyCamSet.optimisation import optimisation_handling as backend

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
# The guarded imports at the top of each phase tab
# --------------------------------------------------------------------------

# Mirrors the ``except ImportError`` branch in each tab, which sets exactly
# these names to None.  Listing them here means a backend symbol that goes
# away is reported as itself, not as "the modules are unavailable".
PHASE_BACKEND_NAMES = {
    "pyCamSet.gui.phase_3_bundle_adjustment": (
        "CameraLockboxConfig",
        "run_bundle_adjustment_with_stats",
        "TemplateBundleHandler",
        "load_CameraSet",
        "load_pickle",
    ),
    "pyCamSet.gui.phase_4_self_calibration": (
        "run_bundle_adjustment_with_stats",
        "SelfBundleHandler",
        "load_CameraSet",
    ),
}


def _phase_module(name: str):
    pytest.importorskip("PySide6", reason="the phase tabs are Qt widgets")
    return importlib.import_module(name)


@pytest.mark.gui
@pytest.mark.parametrize("module_name", sorted(PHASE_BACKEND_NAMES))
def test_the_phase_tab_finds_its_backend(module_name):
    """``_PYCAMSET_OK`` false here is the dialog the user actually sees."""
    module = _phase_module(module_name)

    assert module._PYCAMSET_OK is True, (
        f"{module_name} fell back to its ImportError branch, so the tab will "
        f"refuse to run with 'pyCamSet optimisation modules are unavailable'. "
        f"Import the module directly to see the underlying error."
    )


@pytest.mark.gui
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


@pytest.mark.gui
def test_a_broken_backend_says_why(monkeypatch, caplog):
    """The guard must name the cause, not just disable the tab.

    Reloading with the backend module poisoned takes the ImportError branch
    the same way a renamed symbol would.
    """
    import sys

    module_name = "pyCamSet.gui.phase_3_bundle_adjustment"
    module = _phase_module(module_name)

    monkeypatch.setitem(
        sys.modules, "pyCamSet.optimisation.optimisation_handling", None)
    with caplog.at_level("WARNING"):
        reloaded = importlib.reload(module)

    try:
        assert reloaded._PYCAMSET_OK is False
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
    from pyCamSet.optimisation import optimisation_worker

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
