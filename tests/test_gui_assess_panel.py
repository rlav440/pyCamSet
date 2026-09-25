"""Assess Calibration: figures by default, 3D views on request.

The diagnostics computation and the embedded VTK views are replaced with
recorders, so this runs offscreen and needs no calibration data; the real
drawing is exercised by the data-backed visualisation tests.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from matplotlib.figure import Figure  # noqa: E402


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _run(tmp_path, name):
    camset = tmp_path / f"{name}.camset"
    camset.write_bytes(b"not read")
    return {"run_id": name, "phase": "phase3", "artifacts": {"optimised_camset": str(camset)}}


@pytest.fixture
def panel_env(qapp, tmp_path, monkeypatch):
    """A panel whose diagnostics and 3D views are recorded, not computed."""
    from PySide6.QtWidgets import QWidget

    from pyCamSet.gui import assess_panel
    from pyCamSet.utils import visualisation

    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path / "config"))
    run = _run(tmp_path, "r1")
    record = {"computed": 0, "views": [], "scenes": [], "launched": []}
    fake_diagnostics = object()

    def fake_run(worker):
        record["computed"] += 1
        worker.ready.emit(worker._key, fake_diagnostics)

    monkeypatch.setattr(assess_panel._DiagnosticsWorker, "run", fake_run)
    monkeypatch.setattr(assess_panel.AssessCalibrationPanel, "_figures", lambda self, d: [
        (key, title, Figure()) for key, title in assess_panel.FIGURES])
    monkeypatch.setattr(visualisation, "_assessment_csv_payloads", lambda d: {})

    class FakeView(QWidget):
        def clear(self):
            pass

        def render(self):
            pass

    def make_view(parent):
        view = FakeView(parent)
        record["views"].append(view)
        return view

    def scene(kind):
        return lambda d, *a, plotter=None, **k: record["scenes"].append((kind, plotter))

    monkeypatch.setattr(assess_panel, "_make_interactor", make_view)
    monkeypatch.setattr(visualisation, "reconstruction_scene", scene("scene"))
    monkeypatch.setattr(visualisation, "target_space_scene", scene("target"))
    monkeypatch.setattr(visualisation, "_apply_3d_cosmetics", lambda *a, **k: None)
    monkeypatch.setattr(
        assess_panel.assess, "launch_visualise_calibration_for_run",
        lambda r, **k: record["launched"].append(("pyvista", k)) or (True, "opened"))
    monkeypatch.setattr(
        assess_panel.assess, "launch_visualise_calibration_open3d_for_run",
        lambda r: record["launched"].append(("open3d", {})) or (True, "opened"))
    panel = assess_panel.AssessCalibrationPanel("phase3", lambda: run)
    panel.resize(1400, 900)
    yield panel, run, record
    panel.close()
    panel.deleteLater()
    qapp.processEvents()


def _settle(qapp, panel):
    if panel._worker is not None:
        panel._worker.wait(5000)
    for _ in range(20):
        qapp.processEvents()


def test_figures_draw_when_the_page_opens_and_3d_waits(qapp, panel_env):
    panel, run, record = panel_env
    panel.set_run(run)
    assert record["computed"] == 0  # hidden: nothing computed yet
    panel.show()
    _settle(qapp, panel)
    assert record["computed"] == 1
    titles = [card._title for card in panel.figure_grid.widgets()]
    assert titles == ["Error distribution", "Camera coverage", "Accuracy vs precision"]
    assert record["views"] == []  # the 3D views wait for the button
    # Re-selecting the same run neither recomputes nor redraws.
    panel.set_run(run)
    _settle(qapp, panel)
    assert record["computed"] == 1


def test_visualise_embeds_both_scenes_in_the_tab(qapp, panel_env, tmp_path):
    from pyCamSet.gui import assess_panel

    panel, run, record = panel_env
    if panel.backend.findData(assess_panel.BACKEND_EMBEDDED) < 0:
        pytest.skip("pyvistaqt is not installed")
    panel.set_run(run)
    panel.show()
    _settle(qapp, panel)
    panel.backend.setCurrentIndex(panel.backend.findData(assess_panel.BACKEND_EMBEDDED))
    panel.visualise_button.click()
    _settle(qapp, panel)
    assert len(record["views"]) == 2
    assert [kind for kind, _ in record["scenes"]] == ["scene", "target"]
    assert [view for _, view in record["scenes"]] == record["views"]
    assert record["computed"] == 1  # the figures' diagnostics were reused
    # A style change redraws the embedded views in place.
    record["scenes"].clear()
    panel.three_d_style.point_size.setValue(6.0)
    assert len(record["scenes"]) == 2
    # Another run drops the heavy views until they are asked for again.
    panel.set_run(_run(tmp_path, "r2"))
    _settle(qapp, panel)
    assert panel.view_grid.widgets() == []


@pytest.mark.parametrize("backend,expected", [
    ("pyvista-window", "pyvista"), ("open3d-window", "open3d")])
def test_separate_window_backends_open_a_viewer(qapp, panel_env, backend, expected):
    panel, run, record = panel_env
    panel.set_run(run)
    panel.backend.setCurrentIndex(panel.backend.findData(backend))
    panel.visualise_3d()
    assert [name for name, _ in record["launched"]] == [expected]
    if expected == "pyvista":
        # the figures are already in the tab, so the viewer shows only 3D
        assert "--3d-only" in record["launched"][0][1]["three_d_arguments"]


def test_no_run_says_so(qapp, panel_env):
    panel, _run_, _record = panel_env
    panel.set_run(None)
    assert "Select a run" in panel.figure_status.text()
    assert panel.figure_grid.widgets() == []


def test_grid_columns_follow_the_width(qapp):
    from PySide6.QtWidgets import QLabel

    from pyCamSet.gui.assess_panel import ResponsiveGrid

    grid = ResponsiveGrid(min_column_width=400, max_columns=3)
    grid.set_widgets([QLabel(str(i)) for i in range(3)])
    assert grid.columns_for(390) == 1
    assert grid.columns_for(850) == 2
    assert grid.columns_for(2000) == 3


def test_both_diagnostics_tabs_use_the_shared_panel(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    from pyCamSet.gui.assess_panel import AssessCalibrationPanel
    from pyCamSet.gui.main_window import PyCamSetApp

    window = PyCamSetApp()
    try:
        for tab, phase in ((window.phase3_diag_tab, "phase3"), (window.phase4_diag_tab, "phase4")):
            assert isinstance(tab._assess, AssessCalibrationPanel)
            assert tab._assess._phase == phase
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_closing_the_page_mid_assessment_leaves_the_worker_to_finish(qapp, tmp_path, monkeypatch):
    """A running diagnostics thread must outlive its panel, not be destroyed with it."""
    import threading

    from pyCamSet.gui import assess_panel

    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path / "config"))
    gate = threading.Event()

    def slow_run(worker):
        gate.wait(5)
        worker.ready.emit(worker._key, object())

    monkeypatch.setattr(assess_panel._DiagnosticsWorker, "run", slow_run)
    run = _run(tmp_path, "slow")
    panel = assess_panel.AssessCalibrationPanel("phase3", lambda: run)
    panel.set_run(run)
    panel.show()
    worker = panel._worker
    assert worker is not None and worker.isRunning() and worker.parent() is None
    panel.close()
    panel.deleteLater()
    qapp.processEvents()  # the panel is gone; the thread is not
    gate.set()
    worker.wait(5000)
    for _ in range(20):
        qapp.processEvents()
    assert worker not in assess_panel._RUNNING
