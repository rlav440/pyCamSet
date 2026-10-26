"""F1b label reconciliation: Diagnostics and settings labels must not contradict the education copy.

Contract: labels match the data keys and units, and the Phase 4 ``outliers`` combo is disabled
with its visible label, with a tooltip that neither claims an effect nor carries the guidance
sentence.  No Export gate is required here.

Qt convention: ``importorskip`` at module top (tests/conftest.py).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QFormLayout  # noqa: E402

GUI = Path(__file__).resolve().parents[1] / "pyCamSet" / "gui"


def _source(name: str) -> str:
    return (GUI / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    from pyCamSet.gui.main_window import PyCamSetApp

    win = PyCamSetApp()
    yield win
    win.close()
    win.deleteLater()
    application.processEvents()


# --- Phase 3: D3.3 is the per-image mean; D3.4 is only the plot key ---------------------------

def test_phase3_initial_error_tab_is_titled_by_the_d33_quantity(window):
    tabs = window.phase3_diag_tab._sub_tabs
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    initial = [t for t in titles if "Initial" in t and "image" in t.lower()]
    assert initial, f"no initial per-image error tab among {titles}"
    assert all("D3.3" in t and "D3.4" not in t for t in initial), initial


def test_phase3_initial_error_axis_says_mean_error_per_image():
    src = _source("phase_3_bundle_adjustment.py")
    assert "aggregated px" not in src, "the D3.3 axis must read 'mean error per image (px)', not 'aggregated px'"
    assert "mean error per image (px)" in src.lower()
    assert 'D3.4 Per-image initial reprojection error' not in src, "re-title the D3.3 figure, do not rename keys"


def test_phase3_outlier_tooltip_names_the_mad_rule_not_a_fixed_20_px():
    src = _source("phase_3_bundle_adjustment.py")
    assert "median + 20 px" not in src, "the outlier threshold is median + 20·MAD, not median + 20 px"


@pytest.mark.parametrize("module", ["phase_3_bundle_adjustment.py", "phase_4_self_calibration.py"])
def test_ref_cam_and_ref_pose_are_not_called_the_metric_reference(module):
    assert "metric reference" not in _source(module), (
        f"{module}: ref_cam/ref_pose do not act as the metric reference (plan §1); say they have no effect")


def test_phase3_ref_tooltips_say_the_settings_have_no_effect(window):
    tab = window.phase3_tab
    assert "no effect" in tab._ref_cam_edit.toolTip().lower()
    assert "camera index 0" in tab._ref_cam_edit.toolTip().lower() or "camera 0" in tab._ref_cam_edit.toolTip().lower()
    assert "no effect" in tab._ref_pose_edit.toolTip().lower()
    assert "fixed_pose" in tab._ref_pose_edit.toolTip()


# --- Phase 4: outliers has no effect; gauge labels follow the copy ---------------------------

def test_phase4_outliers_combo_is_disabled_with_a_visible_no_effect_label(window):
    tab = window.phase4_tab
    combo = tab._outliers_combo
    assert combo.isEnabled() is False, "a tooltip reaches neither keyboard nor touch users: disable the combo"
    label_text = ""
    for form in tab.findChildren(QFormLayout):
        label = form.labelForField(combo)
        if label is not None:
            label_text = label.text()
            break
    assert "no effect in phase 4" in label_text.lower(), f"visible label is {label_text!r}"
    tip = combo.toolTip().lower()
    assert "before self-calibration" not in tip
    assert "can discard useful" not in tip


def test_phase4_gauge_rows_do_not_use_the_contradicted_labels():
    src = _source("phase_4_self_calibration.py")
    assert "D4.2 gauge-fixed points:" not in src
    assert "can discard useful target-shape information" not in src


def test_phase4_empty_per_image_list_is_labelled_not_counted_as_zero():
    src = _source("phase_4_self_calibration.py")
    assert 'f"{len(per_image)} images" if isinstance(per_image, dict) else' not in src, (
        "an empty D4.13 per-image list must read as 'not recorded', not '0 images'")
