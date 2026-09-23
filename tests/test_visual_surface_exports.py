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
