"""
Purpose: Bounded tests for standalone target and Phase 3 lockbox exports.
Status: Synthetic file-output and provenance checks; no native Open3D GUI claim.
Future: Add native Open3D window capture only after a supported renderer API is verified.
"""
from pathlib import Path

import numpy as np


def test_target_export_saves_figure_png(tmp_path):
    import matplotlib.pyplot as plt

    from pyCamSet.utils.visualise_target import _export_target

    class Target:
        def plot(self):
            figure = plt.figure()
            figure.add_subplot(111).plot([0, 1], [1, 0])
            plt.show()

    out = tmp_path / "target.png"
    _export_target(Target(), out, None)
    assert out.is_file() and out.stat().st_size > 0
    plt.close("all")


def test_target_export_fails_closed_for_non_scene_geometry(tmp_path):
    from pyCamSet.utils.visualise_target import _export_target

    class FigureOnlyTarget:
        def plot(self):
            raise AssertionError("geometry refusal must precede drawing")

    import pytest
    with pytest.raises(ValueError, match="Reusable scene geometry is unavailable"):
        _export_target(FigureOnlyTarget(), None, tmp_path / "target.pdf")
    assert not (tmp_path / "target.pdf").exists()


def test_pyvista_png_initialises_actual_renderer_before_capture(tmp_path):
    import pyvista as pv

    from pyCamSet.utils.visualise_target import _export_target

    geometry_path = tmp_path / "target.obj"
    png_path = tmp_path / "target.png"

    class Plotter:
        meshes = [pv.PolyData(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0]]))]

        def screenshot(self, *_args):
            raise AssertionError("screenshot requires a render window")

        def show(self, *, screenshot, interactive, auto_close):
            assert geometry_path.is_file()  # Geometry must be exported before auto-close.
            assert interactive is False and auto_close is True
            Path(screenshot).write_bytes(b"actual renderer image")

    class Target:
        def plot(self, return_scene=False):
            assert return_scene
            return Plotter()

    _export_target(Target(), png_path, geometry_path)
    assert png_path.read_bytes() == b"actual renderer image"
    assert geometry_path.is_file() and geometry_path.stat().st_size > 0


def test_lockbox_csv_has_provenance_and_does_not_mutate_coordinates(tmp_path):
    from pyCamSet.gui.phase_3_lockbox_editor import Phase3LockboxEditor, _CameraRowState

    editor = Phase3LockboxEditor.__new__(Phase3LockboxEditor)
    editor.source_camset_path = Path("source.camset")
    editor.workspace_path = tmp_path
    editor.centre_definition = {"label": "synthetic centre"}
    editor.object_centre = np.array([1.0, 2.0, 3.0])
    original = np.array([4.0, 5.0, 6.0])
    edited = np.array([4.5, 5.0, 6.0])
    editor.states = {"cam A": _CameraRowState(
        name="cam A", original_center=original.copy(), edited_center=edited.copy()
    )}

    before = (original.copy(), edited.copy())
    output = tmp_path / "nested" / "centres.csv"
    editor._write_centres_csv(output)
    text = output.read_text(encoding="utf-8")

    assert "source.camset" in text
    assert "physical unit unspecified" in text
    assert "original_camera_centre,cam A,4.0,5.0,6.0" in text
    assert "edited_camera_centre,cam A,4.5,5.0,6.0" in text
    np.testing.assert_array_equal(original, before[0])
    np.testing.assert_array_equal(edited, before[1])

    ply_path = tmp_path / "nested" / "centres.ply"
    editor._write_centres_ply(ply_path)
    ply_text = ply_path.read_text(encoding="ascii")
    assert "element vertex 3" in ply_text
    assert "physical unit unspecified" in ply_text
    assert "1 2 3" in ply_text and "4.5 5 6" in ply_text
    np.testing.assert_array_equal(original, before[0])
    np.testing.assert_array_equal(edited, before[1])


def test_open3d_lockbox_png_uses_scene_render_and_refuses_overwrite(tmp_path, monkeypatch):
    import types

    import pyCamSet.gui.phase_3_lockbox_editor as editor_module

    class Scene:
        def __init__(self):
            self.calls = 0

        def render_to_image(self, callback):
            self.calls += 1
            callback(object())

    scene = Scene()
    editor = editor_module.Phase3LockboxEditor.__new__(editor_module.Phase3LockboxEditor)
    editor._o3d_window = types.SimpleNamespace(close_dialog=lambda: None, post_redraw=lambda: None)
    editor._o3d_scene_widget = types.SimpleNamespace(
        scene=types.SimpleNamespace(scene=scene))
    editor._o3d_status_label = types.SimpleNamespace(text="")
    output = tmp_path / "scene.png"

    def write_image(path, _image):
        Path(path).write_bytes(b"rendered scene")
        return True

    monkeypatch.setattr(editor_module, "_o3d", types.SimpleNamespace(
        io=types.SimpleNamespace(write_image=write_image)))
    monkeypatch.setattr(editor_module, "_o3d_gui", types.SimpleNamespace(
        Application=types.SimpleNamespace(instance=types.SimpleNamespace(
            post_to_main_thread=lambda _window, callback: callback()))))

    editor._save_open3d_scene_png(str(output))
    assert output.read_bytes() == b"rendered scene"
    assert scene.calls == 1
    assert "Saved Open3D scene PNG" in editor._o3d_status_label.text

    output.write_bytes(b"keep existing")
    editor._save_open3d_scene_png(str(output))
    assert output.read_bytes() == b"keep existing"
    assert scene.calls == 1
    assert "already exists" in editor._o3d_status_label.text


def test_create_target_window_surfaces_png_and_supported_geometry_actions():
    import inspect
    from pyCamSet.gui import create_target

    source = inspect.getsource(create_target.CreateTargetDialog._build_ui)
    assert "Save Target View PNG" in source
    assert "Export Target Geometry" in source
    assert "self._save_target_view_png" in source
    assert "self._save_target_geometry" in source


def test_create_target_png_action_confirms_overwrite_and_uses_renderer(tmp_path, monkeypatch):
    import types

    from pyCamSet.gui import create_target

    existing = tmp_path / "target.png"
    existing.write_bytes(b"old image")
    calls = {}
    monkeypatch.setattr(create_target.QFileDialog, "getSaveFileName",
                        lambda *_args: (str(existing), "PNG image (*.png)"))
    monkeypatch.setattr(create_target.QMessageBox, "question",
                        lambda *_args: create_target.QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(create_target, "run_viewer",
                        lambda module, args: calls.update(module=module, args=args) or (True, ""))

    class Status:
        def setText(self, text):
            calls["status"] = text

        def repaint(self):
            pass

    dialog = types.SimpleNamespace(
        _collect=lambda: {"spec": {"type": "Ccube", "n_points": 4},
                          "out_dir": tmp_path, "file_name": "target"},
        _status=Status(),
        _terminal=types.SimpleNamespace(append_line=lambda line: calls.update(terminal=line)),
    )
    create_target.CreateTargetDialog._export_target_view(dialog, ".png", "Save PNG")

    assert calls["module"] == "pyCamSet.utils.visualise_target"
    assert calls["args"][1:3] == ["--save-png", str(existing)]
    assert calls["args"][-1] == "--overwrite"
    assert existing.read_bytes() == b"old image"  # The viewer owns replacement after confirmation.
