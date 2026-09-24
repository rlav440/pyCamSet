'''Purpose: Test managed 3D cosmetic controls and scientific-data invariants.
Status: Active; offscreen synthetic PyVista and Qt checks.
Future: Add native multi-platform rendering checks when available.
'''
from __future__ import annotations

import json
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


def test_create_target_adapter_uses_shared_cosmetics_and_rejects_scalar_legend(monkeypatch):
    from pyCamSet.utils import visualise_target
    from pyCamSet.utils import visualisation

    calls = []
    monkeypatch.setattr(visualisation, "_apply_3d_cosmetics",
                        lambda *args: calls.append(args))
    scene = object()
    visualise_target._apply_target_style(scene, {
        "background": "white", "point_size": 6.0,
        "view": "top", "axes": False, "legend": False,
    })
    assert calls == [(scene, "Light", "white", 6.0, "top", False)]
    with pytest.raises(ValueError, match="no scalar legend"):
        visualise_target._apply_target_style(scene, {"legend": True})


def test_phase3_open3d_style_round_trip_and_fail_closed(tmp_path, monkeypatch):
    from pyCamSet.gui.phase_3_lockbox_editor import Phase3LockboxEditor

    editor = Phase3LockboxEditor.__new__(Phase3LockboxEditor)
    editor._o3d_visual_settings = {
        "view_mode": "Planetary", "box_picking": False, "show_skybox": False,
        "show_ground": True, "ground_plane": "XZ floor", "show_axes": True,
        "background": "Dark calibration", "lighting": "Medium shadows",
        "show_lockbox": False,
    }
    style_path = tmp_path / "visual-styles" / "open3d-style.json"
    monkeypatch.setattr(editor, "_open3d_style_path", lambda: style_path)
    monkeypatch.setattr(editor, "_set_open3d_export_status", lambda _message: None)
    editor._o3d_visual_settings.update({
        "view_mode": "Arcball", "background": "Light studio", "show_axes": False,
    })
    editor._save_open3d_style()
    editor._o3d_visual_settings["background"] = "Dark calibration"
    loaded = editor._load_saved_open3d_style()
    assert loaded["background"] == "Light studio"
    assert loaded["view_mode"] == "Arcball"
    assert loaded["show_axes"] is False
    assert loaded["box_picking"] is False

    document = json.loads(style_path.read_text(encoding="utf-8"))
    document["style"]["point_size"] = 9.0
    style_path.write_text(json.dumps(document), encoding="utf-8")
    assert editor._load_saved_open3d_style()["background"] == "Dark calibration"


def test_phase3_load_saved_open3d_style_syncs_controls_without_edit_callbacks(tmp_path, monkeypatch):
    from pyCamSet.gui import phase_3_lockbox_editor as lockbox_editor
    from pyCamSet.gui.phase_3_lockbox_editor import Phase3LockboxEditor

    class Control:
        """Small widget double that emits callbacks for programmatic changes."""

        def __init__(self, value, selection=False):
            self._value = value
            self._callback = None
            self._selection = selection

        @property
        def selected_text(self):
            return self._value

        @selected_text.setter
        def selected_text(self, value):
            self._value = value
            if self._selection and self._callback:
                self._callback(value, 0)

        @property
        def checked(self):
            return self._value

        @checked.setter
        def checked(self, value):
            self._value = value
            if not self._selection and self._callback:
                self._callback(value)

    class Scene:
        """Capture renderer calls made by the actual Open3D style applier."""

        def __init__(self):
            self.calls = {}

        def set_background(self, value):
            self.calls["background"] = value

        def show_skybox(self, value):
            self.calls["skybox"] = value

        def show_ground_plane(self, enabled, plane):
            self.calls["ground"] = (enabled, plane)

        def show_axes(self, value):
            self.calls["axes"] = value

        def set_lighting(self, profile, direction):
            self.calls["lighting"] = profile

    class SceneWidget:
        def __init__(self):
            self.scene = Scene()

        def set_view_controls(self, value):
            self.scene.calls["view_mode"] = value

    monkeypatch.setattr(lockbox_editor, "_o3d_rendering", SimpleNamespace(
        Scene=SimpleNamespace(GroundPlane=SimpleNamespace(XZ="xz", XY="xy", YZ="yz")),
        Open3DScene=SimpleNamespace(LightingProfile=SimpleNamespace(
            MED_SHADOWS="medium", SOFT_SHADOWS="soft", HARD_SHADOWS="hard",
            DARK_SHADOWS="dark", NO_SHADOWS="none")),
    ))
    monkeypatch.setattr(lockbox_editor, "_o3d_gui", SimpleNamespace(
        SceneWidget=SimpleNamespace(Controls=SimpleNamespace(
            ROTATE_CAMERA_SPHERE="planetary", ROTATE_CAMERA="arcball", FLY="fly",
            ROTATE_MODEL="model", ROTATE_SUN="sun", ROTATE_IBL="environment")),
    ))

    editor = Phase3LockboxEditor.__new__(Phase3LockboxEditor)
    saved = {
        "view_mode": "Arcball", "box_picking": True, "show_skybox": True,
        "show_ground": False, "ground_plane": "YZ sideplane", "show_axes": False,
        "background": "Light studio", "lighting": "No shadows", "show_lockbox": True,
    }
    editor._o3d_visual_settings = dict(saved)
    style_path = tmp_path / "visual-styles" / "open3d-style.json"
    monkeypatch.setattr(editor, "_open3d_style_path", lambda: style_path)
    monkeypatch.setattr(editor, "_set_open3d_export_status", lambda _message: None)
    editor._save_open3d_style()

    callbacks = {
        "_o3d_view_mode_combo": "_on_o3d_view_mode_changed",
        "_o3d_ground_combo": "_on_o3d_ground_plane_changed",
        "_o3d_background_combo": "_on_o3d_background_changed",
        "_o3d_lighting_combo": "_on_o3d_lighting_changed",
        "_o3d_ground_cb": "_on_o3d_show_ground_changed",
        "_o3d_skybox_cb": "_on_o3d_show_skybox_changed",
        "_o3d_axes_cb": "_on_o3d_show_axes_changed",
        "_o3d_lockbox_cb": "_on_o3d_show_lockbox_changed",
    }
    controls = {}
    for name, callback_name in callbacks.items():
        selection = name.endswith("_combo")
        initial = "stale" if selection else not saved[{
            "_o3d_ground_cb": "show_ground", "_o3d_skybox_cb": "show_skybox",
            "_o3d_axes_cb": "show_axes", "_o3d_lockbox_cb": "show_lockbox",
        }[name]]
        control = Control(initial, selection=selection)
        control._callback = getattr(editor, callback_name)
        controls[name] = control
        setattr(editor, name, control)
    editor._o3d_pick_cb = Control(False)
    editor._o3d_scene_widget = SceneWidget()
    editor._o3d_syncing_style_controls = False
    editor._refresh_open3d_native_view = lambda reset_camera=False: None
    editor._o3d_visual_settings["box_picking"] = False
    editor.states = {"camera": object()}
    editor.working_camset = object()
    states_before = editor.states
    camset_before = editor.working_camset

    editor._o3d_visual_settings.update({
        "view_mode": "Fly", "show_skybox": False, "show_ground": True,
        "ground_plane": "XY backplane", "show_axes": True,
        "background": "Dark calibration", "lighting": "Hard shadows",
        "show_lockbox": False,
    })
    before_non_style = editor._o3d_visual_settings["box_picking"]
    editor._load_open3d_style_from_ui()

    assert {key: editor._o3d_visual_settings[key] for key in saved if key != "box_picking"} == {
        key: value for key, value in saved.items() if key != "box_picking"
    }
    assert controls["_o3d_view_mode_combo"].selected_text == saved["view_mode"]
    assert controls["_o3d_ground_combo"].selected_text == saved["ground_plane"]
    assert controls["_o3d_background_combo"].selected_text == saved["background"]
    assert controls["_o3d_lighting_combo"].selected_text == saved["lighting"]
    assert controls["_o3d_ground_cb"].checked is saved["show_ground"]
    assert controls["_o3d_skybox_cb"].checked is saved["show_skybox"]
    assert controls["_o3d_axes_cb"].checked is saved["show_axes"]
    assert controls["_o3d_lockbox_cb"].checked is saved["show_lockbox"]
    assert editor._o3d_pick_cb.checked is False
    assert editor._o3d_visual_settings["box_picking"] == before_non_style
    assert editor.states is states_before
    assert editor.working_camset is camset_before
    assert editor._o3d_syncing_style_controls is False
    assert editor._o3d_scene_widget.scene.calls["background"] == [0.82, 0.84, 0.88, 1.0]
    assert editor._o3d_scene_widget.scene.calls["skybox"] is True
    assert editor._o3d_scene_widget.scene.calls["ground"][0] is False
    assert editor._o3d_scene_widget.scene.calls["ground"][1] == "yz"
    assert editor._o3d_scene_widget.scene.calls["axes"] is False
    assert editor._o3d_scene_widget.scene.calls["lighting"] == "none"
    assert editor._o3d_scene_widget.scene.calls["view_mode"] == "arcball"
