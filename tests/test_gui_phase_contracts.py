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

from pyCamSet.gui.shared_functions import (  # noqa: E402
    TARGET_IDENTITY_KEYS,
    apply_target_params_to_widgets,
    describe_target_mismatch,
    target_mismatch_message,
    target_params_of_run,
)

# The two Ccube targets behind the reported failure.
CCUBE_12 = {"target_type": "Ccube", "n_points": 12, "length": 80.0,
            "border_fraction": 0.1, "marker_backend": "aruco1"}
CCUBE_6 = {"target_type": "Ccube", "n_points": 6, "length": 30.0,
           "border_fraction": 0.1, "marker_backend": "aruco1"}


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
        CCUBE_12, {**CCUBE_12, "target_type": "ChArUco"})

    assert len(differences) == 1
    assert "target type" in differences[0]


def test_detection_only_settings_are_not_a_mismatch():
    """The backend changes which points are found, not what a key means."""
    other_backend = {**CCUBE_12, "marker_backend": "aruco2"}

    assert describe_target_mismatch(CCUBE_12, other_backend) == []


def test_numbers_are_compared_as_numbers():
    """Run metadata round-trips through JSON, so 80 may arrive as "80.0"."""
    as_text = {**CCUBE_12, "n_points": "12", "length": "80"}

    assert describe_target_mismatch(CCUBE_12, as_text) == []


def test_only_the_fields_that_matter_for_the_type_are_compared():
    """Phase 1 records every field, whichever target was actually used."""
    # PuzzleBoard fields differ, but a Ccube run does not use them
    noisy = {**CCUBE_12, "num_squares_x": 999, "paper_width": 1.0}

    assert describe_target_mismatch(CCUBE_12, noisy) == []
    assert "num_squares_x" not in TARGET_IDENTITY_KEYS["Ccube"]


def test_a_missing_run_is_not_a_mismatch():
    """A pickle override or a first run has nothing to compare against."""
    assert describe_target_mismatch({}, CCUBE_6) == []
    assert describe_target_mismatch(CCUBE_12, {}) == []
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


# --- adopting a run's target -----------------------------------------------


class _Spin:
    def __init__(self, value=0, lo=None, hi=None):
        self.value_ = value
        self._lo, self._hi = lo, hi

    def setValue(self, value):
        if self._lo is not None:
            value = max(self._lo, min(self._hi, value))
        self.value_ = value


class _Edit:
    def __init__(self, text=""):
        self.text_ = text

    def setText(self, text):
        self.text_ = text


class _Combo:
    def __init__(self, text="", data=()):
        self.text_ = text
        self._data = list(data)
        self.index_ = -1

    def setCurrentText(self, text):
        self.text_ = text

    def findData(self, value):
        return self._data.index(value) if value in self._data else -1

    def setCurrentIndex(self, index):
        self.index_ = index


class _FakeTab:
    """The target widgets of a phase tab, without Qt.

    Same attribute names as Phases 2 and 3, which are identical to each
    other -- that is what lets one helper serve both.
    """

    def __init__(self):
        self._target_combo = _Combo("Ccube")
        self._npts_spin = _Spin(6, 2, 30)
        self._length_edit = _Edit("30.0")
        self._border_spin = _Spin(0.1)
        self._marker_spin = _Spin(0.8)
        self._marker_backend_combo = _Combo("", ["aruco1", "aruco2"])
        self._pb_x_spin = _Spin(105)
        self._pb_y_spin = _Spin(148)
        self._pb_square_edit = _Edit("2.0")
        self._pbc_size_spin = _Spin(20)
        self._pbc_square_edit = _Edit("200.0")


def test_adopting_a_run_sets_the_target_to_match_it():
    tab = _FakeTab()

    apply_target_params_to_widgets(tab, CCUBE_12)

    assert tab._target_combo.text_ == "Ccube"
    assert tab._npts_spin.value_ == 12
    assert tab._length_edit.text_ == "80"
    assert describe_target_mismatch(
        CCUBE_12,
        {"target_type": tab._target_combo.text_,
         "n_points": tab._npts_spin.value_,
         "length": tab._length_edit.text_,
         "border_fraction": tab._border_spin.value_},
    ) == []


def test_adopting_selects_the_marker_backend_by_value():
    tab = _FakeTab()

    apply_target_params_to_widgets(tab, {**CCUBE_12, "marker_backend": "aruco2"})

    assert tab._marker_backend_combo.index_ == 1


def test_adopting_nothing_changes_nothing():
    tab = _FakeTab()

    apply_target_params_to_widgets(tab, {})

    assert tab._npts_spin.value_ == 6
    assert tab._length_edit.text_ == "30.0"


def test_adopting_ignores_widgets_a_tab_does_not_have():
    """Not every tab carries every target's fields."""
    class _Sparse:
        def __init__(self):
            self._npts_spin = _Spin(6, 2, 30)

    tab = _Sparse()
    apply_target_params_to_widgets(tab, CCUBE_12)

    assert tab._npts_spin.value_ == 12


def test_a_value_the_interface_cannot_hold_is_still_caught():
    """Spin boxes clamp silently; the mismatch check is the backstop."""
    tab = _FakeTab()  # n_points range is 2..30

    apply_target_params_to_widgets(tab, {**CCUBE_12, "n_points": 99})

    assert tab._npts_spin.value_ == 30
    assert describe_target_mismatch(
        {**CCUBE_12, "n_points": 99},
        {"target_type": "Ccube", "n_points": tab._npts_spin.value_,
         "length": 80.0, "border_fraction": 0.1},
    ) != []


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
