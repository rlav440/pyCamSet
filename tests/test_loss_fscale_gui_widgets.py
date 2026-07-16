"""Purpose: Headless widget tests for the Phase 4 'loss' + 'f_scale' bundle-adjustment
controls (scipy least_squares robust-loss parameters, previously only reachable via
headless scripts). Phase 3 never got usable equivalents: main_window.py's
_normalize_outlier_combos() matches on token overlap (its outlier_tokens set includes
"loss"/"robust"/"soft_l1"/"huber"/"cauchy"/"arctan") rather than widget identity, so it
clobbered Phase 3's loss combo into a nonfunctional Yes/No toggle -- and since
_collect_params() read the combo's text straight into problem_options['loss'] with no
validation, that Yes/No value hit scipy's least_squares() as an invalid loss name and
crashed Phase 3 outright. The Phase 3 loss/f_scale controls were removed entirely
rather than fixing the collision, since the project's own validated recipe never uses a
non-linear loss in Phase 3 anyway (see phase3-removal note in project memory).

Status: Active regression coverage for the Phase 4 loss/f_scale GUI exposure, plus a
guard that Phase 3 stays free of these widgets.

Future: Extend if additional problem_options widgets are added.
"""
from __future__ import annotations

import os

# Must be set before any PySide6/Qt import happens anywhere in the process.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget

import numpy as np

from pyCamSet.gui.phase_3_bundle_adjustment import Phase3Tab
from pyCamSet.gui.phase_4_self_calibration import Phase4Tab
from pyCamSet.gui.shared_functions import WorkspaceManager
from pyCamSet.optimisation.template_handler import DEFAULT_OPTIONS, TemplateBundleHandler


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_phase3_tab(tmp_path: Path, qapp) -> Phase3Tab:
    mgr = WorkspaceManager(tmp_path / ".pycamset_workspace")
    return Phase3Tab(
        notebook=QTabWidget(),
        info_cb=QCheckBox(),
        terminal_cb=QCheckBox(),
        workspace_mgr=mgr,
    )


def _make_phase4_tab(tmp_path: Path, qapp) -> Phase4Tab:
    mgr = WorkspaceManager(tmp_path / ".pycamset_workspace")
    return Phase4Tab(
        notebook=QTabWidget(),
        info_cb=QCheckBox(),
        terminal_cb=QCheckBox(),
        workspace_mgr=mgr,
    )


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------


def test_phase3_tab_has_no_loss_or_f_scale_widgets(tmp_path: Path, qapp):
    # Regression guard: Phase 3 loss/f_scale controls were removed entirely (they
    # collided with main_window.py's outlier-combo normalizer and blocked Phase 3
    # from running at all -- see module docstring). Must not silently come back.
    tab = _make_phase3_tab(tmp_path, qapp)
    assert not hasattr(tab, "_loss_combo")
    assert not hasattr(tab, "_f_scale_spin")


def test_phase4_tab_has_loss_and_f_scale_widgets(tmp_path: Path, qapp):
    tab = _make_phase4_tab(tmp_path, qapp)
    assert hasattr(tab, "_loss_combo")
    assert hasattr(tab, "_f_scale_spin")


# ---------------------------------------------------------------------------
# Choices offered
# ---------------------------------------------------------------------------


_EXPECTED_LOSS_CHOICES = ["linear", "soft_l1", "huber", "cauchy", "arctan"]


def test_phase4_loss_combo_offers_all_scipy_loss_choices(tmp_path: Path, qapp):
    tab = _make_phase4_tab(tmp_path, qapp)
    items = [tab._loss_combo.itemText(i) for i in range(tab._loss_combo.count())]
    assert items == _EXPECTED_LOSS_CHOICES


# ---------------------------------------------------------------------------
# Defaults -- per phase-specific validated recipe
# (run_dataset1_phase1_to_phase4_headless.py: P3_PROBLEM_OPTIONS omits loss/f_scale
#  -> falls back to scipy's own 'linear' default; P4_PROBLEM_OPTIONS explicitly sets
#  loss='soft_l1', f_scale=1.0.)
# ---------------------------------------------------------------------------


def test_phase4_loss_default_is_soft_l1(tmp_path: Path, qapp):
    tab = _make_phase4_tab(tmp_path, qapp)
    assert tab._loss_combo.currentText() == "soft_l1"


def test_phase4_f_scale_default_is_1_0(tmp_path: Path, qapp):
    tab = _make_phase4_tab(tmp_path, qapp)
    assert tab._f_scale_spin.value() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# f_scale enabled/disabled in step with the loss selection
# ---------------------------------------------------------------------------


def test_phase4_f_scale_enabled_by_default_since_default_loss_is_soft_l1(tmp_path: Path, qapp):
    tab = _make_phase4_tab(tmp_path, qapp)
    assert tab._f_scale_spin.isEnabled() is True


def test_phase4_f_scale_disabled_when_switched_to_linear(tmp_path: Path, qapp):
    tab = _make_phase4_tab(tmp_path, qapp)
    tab._loss_combo.setCurrentText("linear")
    assert tab._f_scale_spin.isEnabled() is False


@pytest.mark.parametrize("loss_name", ["huber", "cauchy", "arctan"])
def test_phase4_f_scale_enabled_for_every_robust_loss(tmp_path: Path, qapp, loss_name):
    tab = _make_phase4_tab(tmp_path, qapp)
    tab._loss_combo.setCurrentText(loss_name)
    assert tab._f_scale_spin.isEnabled() is True


# ---------------------------------------------------------------------------
# Value read-back into problem_options -- the actual wiring under test
# ---------------------------------------------------------------------------


def _fill_required_numeric_fields(tab) -> None:
    # _collect_params requires floc + a few numeric text fields to be parseable;
    # populate them with harmless values so collection succeeds and we can inspect
    # the resulting problem_options dict in isolation.
    tab._floc_edit.setText("dummy_folder")
    tab._fixed_pose_edit.setText("0")
    tab._ref_cam_edit.setText("0")
    tab._ref_pose_edit.setText("0")
    tab._threads_edit.setText("1")


def test_phase3_collect_params_omits_loss_and_f_scale(tmp_path: Path, qapp):
    # Regression guard: Phase 3's problem_options must not carry a 'loss'/'f_scale' key
    # at all now -- the handler's own .get("loss", "linear") fallback (matching the
    # validated recipe) takes over instead of a GUI-supplied value.
    tab = _make_phase3_tab(tmp_path, qapp)
    _fill_required_numeric_fields(tab)
    params = tab._collect_params()
    assert params is not None
    assert "loss" not in params["problem_options"]
    assert "f_scale" not in params["problem_options"]


def test_phase4_collect_params_reads_back_default_loss_and_f_scale(tmp_path: Path, qapp):
    tab = _make_phase4_tab(tmp_path, qapp)
    _fill_required_numeric_fields(tab)
    params = tab._collect_params()
    assert params is not None
    assert params["problem_options"]["loss"] == "soft_l1"
    assert params["problem_options"]["f_scale"] == pytest.approx(1.0)


def test_phase4_collect_params_reads_back_changed_loss_and_f_scale(tmp_path: Path, qapp):
    tab = _make_phase4_tab(tmp_path, qapp)
    _fill_required_numeric_fields(tab)
    tab._loss_combo.setCurrentText("huber")
    tab._f_scale_spin.setValue(0.75)
    params = tab._collect_params()
    assert params is not None
    assert params["problem_options"]["loss"] == "huber"
    assert params["problem_options"]["f_scale"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# f_scale numeric range sanity
# ---------------------------------------------------------------------------


def test_f_scale_spin_range_excludes_zero_and_negative(tmp_path: Path, qapp):
    # Phase 3 has no f_scale_spin (see module docstring) -- only Phase 4 to check.
    tab4 = _make_phase4_tab(tmp_path, qapp)
    # Regression guard: a minimum that isn't representable at the configured
    # decimals (e.g. 1e-6 with decimals=4) silently rounds to 0.0 in Qt, which
    # would let the widget accept f_scale=0 despite scipy requiring > 0.
    assert tab4._f_scale_spin.minimum() > 0.0
    tab4._f_scale_spin.setValue(-5.0)
    assert tab4._f_scale_spin.value() == pytest.approx(tab4._f_scale_spin.minimum())
    assert tab4._f_scale_spin.value() > 0.0


# ---------------------------------------------------------------------------
# Handler-level wiring -- the actual failure mode the earlier widget-only tests
# (everything above this point) could not catch: they only ever call
# _collect_params() and inspect the resulting dict. They never construct a real
# TemplateBundleHandler/SelfBundleHandler, so a bug in how that class *consumes*
# problem_options -- as opposed to how the widget *produces* it -- was completely
# unexercised. Real GUI usage is a single long-running process in which many
# handlers get constructed over the session (repeated Phase 3 runs, the
# Diagnostics "rerun with images above threshold removed" path replaying an
# older saved run, Phase 4, ...): these tests reproduce that pattern.
# ---------------------------------------------------------------------------


class _FakeCamsetForHandler:
    """Minimal CameraSet stand-in -- only what TemplateBundleHandler.__init__ touches."""

    def __init__(self, n_cams: int = 1):
        self._n_cams = n_cams

    def get_names(self):
        return [f"cam{i}" for i in range(self._n_cams)]

    def get_n_cams(self):
        return self._n_cams


class _FakeTargetForHandler:
    """Minimal AbstractTarget stand-in -- only .point_data is touched by __init__."""

    def __init__(self):
        self.point_data = np.zeros((2, 2, 3))


class _FakeDetectionForHandler:
    """Minimal TargetDetection stand-in -- only .max_ims is touched by __init__, and
    the object must be deepcopy-able (a plain object with no special state is)."""

    def __init__(self, max_ims: int = 1):
        self.max_ims = max_ims


def _make_real_handler(problem_options: dict) -> TemplateBundleHandler:
    """Construct a real TemplateBundleHandler, exactly as Phase3Tab's work_fn does,
    with the minimum fake camset/target/detection needed to reach self.problem_opts."""
    return TemplateBundleHandler(
        camset=_FakeCamsetForHandler(),
        target=_FakeTargetForHandler(),
        detection=_FakeDetectionForHandler(),
        options=problem_options,
    )


def test_default_options_module_dict_is_never_mutated_by_construction():
    """Regression test for the exact bug found: TemplateBundleHandler.__init__ used to
    do `self.problem_opts = DEFAULT_OPTIONS` (aliasing the shared module-level dict)
    instead of copying it, so `.update(options)` permanently mutated DEFAULT_OPTIONS
    itself for the rest of the process."""
    baseline = dict(DEFAULT_OPTIONS)
    handler = _make_real_handler({"loss": "cauchy", "f_scale": 3.0})
    assert handler.problem_opts["loss"] == "cauchy"
    assert handler.problem_opts is not DEFAULT_OPTIONS
    assert DEFAULT_OPTIONS == baseline, (
        "Constructing a TemplateBundleHandler must not mutate the shared "
        "DEFAULT_OPTIONS module dict."
    )
    assert "loss" not in DEFAULT_OPTIONS


def test_two_sequential_phase3_runs_do_not_leak_loss_between_handlers():
    """Simulates two consecutive 'Run Phase 3' clicks in the same live GUI process
    with different loss selections -- the first run must not influence the second,
    and (more subtly) a later run whose options dict omits 'loss' entirely (e.g. a
    legacy saved run predating the loss/f_scale controls, replayed via the
    Diagnostics 'rerun with images above threshold removed' path) must fall back to
    scipy's own 'linear' default rather than silently inheriting a previous run's
    selection."""
    handler_cauchy = _make_real_handler(
        {"verbosity": 0, "max_nfev": 10, "loss": "cauchy", "f_scale": 3.0}
    )
    assert handler_cauchy.problem_opts["loss"] == "cauchy"

    # A later handler in the same process whose caller explicitly requests 'linear'.
    handler_linear = _make_real_handler(
        {"verbosity": 0, "max_nfev": 10, "loss": "linear", "f_scale": 1.0}
    )
    assert handler_linear.problem_opts["loss"] == "linear"
    assert handler_cauchy.problem_opts["loss"] == "cauchy"  # unaffected by the later construction

    # A later handler whose caller's options dict has no 'loss' key at all (this is
    # exactly what src_params.get("problem_options") looks like for any Phase 3 run
    # saved before the loss/f_scale controls existed). Must fall back to scipy's own
    # 'linear' default, not inherit handler_cauchy's 'cauchy'.
    legacy_options = {"verbosity": 0, "max_nfev": 10}  # no 'loss' / 'f_scale' keys
    handler_legacy = _make_real_handler(legacy_options)
    assert handler_legacy.problem_opts.get("loss", "linear") == "linear"
    assert "loss" not in handler_legacy.problem_opts


def test_handler_construction_does_not_leak_interactive_draw_backend_safe():
    """DEFAULT_OPTIONS also carries 'interactive'/'draw'/'backend_safe' keys that
    Phase3Tab._collect_params() never sets explicitly. A handler built with those
    keys overridden (mirroring the Optimisation tab's non-interactive trials, see
    test_bundle_options_create_non_interactive_settings in test_optimisation_tab.py)
    must not silently disable interactive drawing for a later, ordinary Phase 3 GUI
    run in the same process."""
    _make_real_handler({"interactive": False, "draw": False, "backend_safe": True})
    later = _make_real_handler({"verbosity": 2, "max_nfev": 300})
    assert later.problem_opts.get("interactive", True) is True
    assert later.problem_opts.get("draw", True) is True


# ---------------------------------------------------------------------------
# main_window._normalize_outlier_combos() must match the outliers combo by
# widget identity (objectName), never by scanning tooltip/item text -- direct
# regression coverage for the bug this module's docstring describes: the old
# free-text token match also caught Phase 3/4's unrelated loss combo (whose own
# items are soft_l1/huber/cauchy/arctan and whose tooltip discusses "outlier
# target points/poses"), silently corrupting it into a Yes/No toggle on every
# tab visit and crashing scipy once that value reached problem_options["loss"].
# ---------------------------------------------------------------------------


def _make_app_for_normalize():
    from pyCamSet.gui.main_window import PyCamSetApp
    return PyCamSetApp.__new__(PyCamSetApp)


def test_normalize_outlier_combos_leaves_phase4_loss_combo_untouched(tmp_path: Path, qapp):
    tab = _make_phase4_tab(tmp_path, qapp)
    _make_app_for_normalize()._normalize_outlier_combos(tab)
    items = [tab._loss_combo.itemText(i) for i in range(tab._loss_combo.count())]
    assert items == _EXPECTED_LOSS_CHOICES
    assert tab._loss_combo.currentText() == "soft_l1"


def test_normalize_outlier_combos_still_normalizes_phase3_outliers_combo(tmp_path: Path, qapp):
    tab = _make_phase3_tab(tmp_path, qapp)
    _make_app_for_normalize()._normalize_outlier_combos(tab)
    items = [tab._outliers_combo.itemText(i) for i in range(tab._outliers_combo.count())]
    assert items == ["No", "Yes"]


def test_normalize_outlier_combos_still_normalizes_phase4_outliers_combo(tmp_path: Path, qapp):
    tab = _make_phase4_tab(tmp_path, qapp)
    _make_app_for_normalize()._normalize_outlier_combos(tab)
    items = [tab._outliers_combo.itemText(i) for i in range(tab._outliers_combo.count())]
    assert items == ["No", "Yes"]


def test_normalize_outlier_combos_survives_repeated_calls_on_phase4(tmp_path: Path, qapp):
    # Regression guard: main_window.py calls this on every tab-switch, not just once.
    tab = _make_phase4_tab(tmp_path, qapp)
    app = _make_app_for_normalize()
    for _ in range(3):
        app._normalize_outlier_combos(tab)
    assert [tab._loss_combo.itemText(i) for i in range(tab._loss_combo.count())] == _EXPECTED_LOSS_CHOICES
    assert [tab._outliers_combo.itemText(i) for i in range(tab._outliers_combo.count())] == ["No", "Yes"]


def test_phase4_loss_selection_survives_into_problem_options_after_normalize(tmp_path: Path, qapp):
    # End-to-end: GUI tab-switch normalization must not silently break the actual
    # value that reaches problem_options["loss"], the exact failure mode that used
    # to block Phase 4 from running once the corrupted combo hit scipy.
    tab = _make_phase4_tab(tmp_path, qapp)
    _make_app_for_normalize()._normalize_outlier_combos(tab)
    _fill_required_numeric_fields(tab)
    tab._loss_combo.setCurrentText("cauchy")
    tab._f_scale_spin.setValue(2.0)
    params = tab._collect_params()
    assert params is not None
    assert params["problem_options"]["loss"] == "cauchy"
    assert params["problem_options"]["f_scale"] == pytest.approx(2.0)
