'''Purpose: Test managed 3D cosmetic controls and scientific-data invariants.
Status: Active; offscreen synthetic PyVista and Qt checks.
Future: Add native multi-platform rendering checks when available.
'''
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvista")

from pyCamSet.gui import three_d_style
from pyCamSet.gui.three_d_style import ThreeDStyleControls
from pyCamSet.utils.visualisation import _apply_3d_cosmetics, reconstruction_scene


@pytest.fixture
def application():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_controls_emit_explicit_safe_viewer_options(application):
    controls = ThreeDStyleControls()
    controls.background.setCurrentIndex(2)
    controls.point_size.setValue(7.5)
    controls.view.setCurrentIndex(1)
    controls.axes.setChecked(False)
    controls.legend.setChecked(False)
    assert controls.viewer_arguments() == [
        "--3d-background", "charcoal", "--3d-point-size", "7.5",
        "--3d-view", "top", "--no-3d-axes", "--no-3d-legend",
    ]
    from PySide6.QtWidgets import QLabel
    assert any("Open3D" in label.text() for label in controls.findChildren(QLabel))


def test_per_visual_style_round_trip_and_theme_inheritance(application, tmp_path, monkeypatch):
    style_path = tmp_path / "visual-styles" / "3d-fixed-digest.json"
    monkeypatch.setattr(three_d_style, "_style_path", lambda visual_id: style_path)
    controls = ThreeDStyleControls(visual_id="assessment:phase3")
    controls.background.setCurrentIndex(1)
    controls.point_size.setValue(6.5)
    controls.view.setCurrentIndex(2)
    controls.axes.setChecked(False)
    controls.legend.setChecked(False)
    controls._save_style()

    reopened = ThreeDStyleControls(visual_id="assessment:phase3")
    assert reopened._style_values() == {
        "background": "white", "point_size": 6.5, "view": "front",
        "axes": False, "legend": False,
    }
    monkeypatch.setattr(three_d_style.QMessageBox, "question",
                        lambda *args: three_d_style.QMessageBox.StandardButton.Yes)
    reopened._reset_style()
    assert not style_path.exists()
    assert reopened._style_values() == {
        "background": "theme", "point_size": 3.0, "view": "isometric",
        "axes": True, "legend": True,
    }


def test_malformed_saved_style_is_rejected_without_partial_apply(application, tmp_path, monkeypatch):
    style_path = tmp_path / "style.json"
    monkeypatch.setattr(three_d_style, "_style_path", lambda visual_id: style_path)
    style_path.write_text('{"schema":"wrong"}', encoding="utf-8")
    controls = ThreeDStyleControls(visual_id="assessment:phase4")
    before = controls._style_values()
    controls.background.setCurrentIndex(2)
    changed = controls._style_values()
    controls._load_style(silent=True)
    assert controls._style_values() == changed
    assert changed != before


def test_load_without_saved_preference_is_a_noop(application, tmp_path, monkeypatch):
    style_path = tmp_path / "not-created.json"
    monkeypatch.setattr(three_d_style, "_style_path", lambda visual_id: style_path)
    controls = ThreeDStyleControls(visual_id="assessment:phase3")
    controls.background.setCurrentIndex(2)
    before = controls._style_values()
    controls._load_style()
    assert controls._style_values() == before


def test_reset_cancellation_preserves_saved_style_and_controls(application, tmp_path, monkeypatch):
    style_path = tmp_path / "saved.json"
    monkeypatch.setattr(three_d_style, "_style_path", lambda visual_id: style_path)
    controls = ThreeDStyleControls(visual_id="assessment:phase3")
    controls._save_style()
    controls.background.setCurrentIndex(2)
    before = controls._style_values()
    monkeypatch.setattr(three_d_style.QMessageBox, "question",
                        lambda *args: three_d_style.QMessageBox.StandardButton.No)
    controls._reset_style()
    assert style_path.exists()
    assert controls._style_values() == before


def test_style_contract_rejects_foreign_visual_and_nonfinite_size():
    valid = {"schema": "pycamset.3d-visual-style", "version": 1,
             "visual_id": "assessment:phase3", "style": {
                 "background": "theme", "point_size": 3, "view": "isometric",
                 "axes": True, "legend": True}}
    with pytest.raises(ValueError, match="different visual"):
        three_d_style._validated_style(valid, "assessment:phase4")
    valid["style"]["point_size"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        three_d_style._validated_style(valid, "assessment:phase3")


def test_cosmetic_scene_controls_leave_coordinates_and_error_scalars_unchanged(application):
    import pyvista as pv

    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    errors = np.array([0.2, 0.4, 0.6])
    diagnostics = SimpleNamespace(
        scene_points=points.copy(), point_error=errors.copy(), e_lim=1.0,
        cams=SimpleNamespace(get_scene=lambda scene, labels=False: None),
        object_points=points.copy(), rejected=0,
    )
    plotter = reconstruction_scene(diagnostics, point_size=7.0, show_legend=False)
    try:
        _apply_3d_cosmetics(plotter, "Sepia", "theme", 7.0, "top", True)
        assert plotter.background_color == "#f3ecdf"
        assert np.array_equal(diagnostics.scene_points, points)
        assert np.array_equal(diagnostics.point_error, errors)
        actor = next(actor for actor in plotter.renderer.actors.values()
                     if actor.GetMapper() is not None
                     and actor.GetMapper().GetInput().GetNumberOfPoints() == len(points))
        dataset = actor.GetMapper().GetInput()
        assert np.array_equal(np.asarray(dataset.GetPoints().GetData()), points)
        scalar = dataset.GetPointData().GetArray("Reprojection error (px)")
        assert np.array_equal(np.asarray(scalar), errors)
        assert actor.GetProperty().GetPointSize() == pytest.approx(7.0)
        assert not plotter.scalar_bars
        assert plotter.camera_position[0][2] > plotter.camera_position[1][2]
    finally:
        plotter.close()


def test_cosmetics_are_applied_to_each_viewport():
    import pyvista as pv

    plotter = pv.Plotter(shape=(1, 2), off_screen=True)
    try:
        for index in range(2):
            plotter.subplot(0, index)
            plotter.add_mesh(pv.PolyData(np.array([[0., 0., 0.], [1., 1., 1.]])))
        _apply_3d_cosmetics(plotter, "Dark", "theme", 3.0, "front", True)
        assert len(plotter.renderers) == 2
        assert all(renderer.axes_actor is not None for renderer in plotter.renderers)
    finally:
        plotter.close()


@pytest.mark.parametrize("kwargs", [
    {"background": "red"}, {"point_size": 0}, {"point_size": 21},
    {"view": "oblique"}, {"theme_name": "unknown"},
])
def test_invalid_cosmetic_options_fail_closed(kwargs):
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True)
    try:
        with pytest.raises(ValueError):
            _apply_3d_cosmetics(plotter, **kwargs)
    finally:
        plotter.close()
