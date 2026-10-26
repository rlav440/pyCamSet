'''Verify Phase 1 overlays map detector pixels onto native source images.
'''
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

pytest.importorskip("PySide6")

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from PySide6.QtWidgets import QCheckBox, QTabWidget  # noqa: E402

from pyCamSet.calibration_targets.core.target_detections import (  # noqa: E402
    ImageDetection, TargetDetection,
)
from pyCamSet.gui.phase_1_detection import Phase1DiagnosticsTab  # noqa: E402
from pyCamSet.gui.run_results import detection_figure  # noqa: E402
from pyCamSet.workflow.detections import save_detections  # noqa: E402
from pyCamSet.workflow.workspace import WorkspaceManager  # noqa: E402


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _detection_fixture(tmp_path):
    """Build two camera folders with exact detector-frame points."""
    import cv2

    cameras = ["camA", "camB"]
    points = {
        "camA": np.array([[4.0, 6.0], [10.0, 12.0]]),
        "camB": np.array([[8.0, 2.0], [14.0, 16.0]]),
    }
    detections = TargetDetection(cameras)
    for camera_index, camera in enumerate(cameras):
        folder = tmp_path / camera
        folder.mkdir()
        assert cv2.imwrite(str(folder / "image0.png"),
                           np.zeros((20, 20, 3), dtype=np.uint8))
        detections.add_detection(
            camera, 0,
            ImageDetection(keys=np.array([0, 1]), image_points=points[camera]),
        )
    artifact = tmp_path / "run-detections.pickle"
    save_detections(artifact, detections)
    return cameras, points, detections, artifact


@pytest.mark.parametrize("upscale_factor", [1, 2])
def test_phase1_overlay_mapping_matches_native_pixels_in_both_views(
        tmp_path, monkeypatch, qapp, upscale_factor):
    from pyCamSet.gui import phase_1_detection
    from PySide6.QtWidgets import QFileDialog

    cameras, points, detections, artifact = _detection_fixture(tmp_path)
    original_rows = detections.get_data().copy()
    params = {"f_loc": str(tmp_path), "upscale_factor": upscale_factor,
              "rescale_and_gamma": False}
    run = {"run_id": f"mapping-{upscale_factor}", "params": params,
           "artifacts": {"detected_datapoints_pickle": str(artifact)}}

    # Exercise the Draw Detections tab's real renderer for every camera.
    tab = Phase1DiagnosticsTab(QTabWidget(), QCheckBox(), WorkspaceManager(None))
    tab._draw_detections_for_run(run, show_errors=False)
    assert tab._draw_state
    tab._draw_state["canvas"].draw()
    for camera in cameras:
        displayed = tab._draw_state["sc_art"][camera].get_offsets()
        np.testing.assert_array_equal(displayed, points[camera] / upscale_factor)

    # Exercise the run-result preview renderer on the same detector-frame data.
    for camera in cameras:
        figure = detection_figure(tmp_path / camera / "image0.png",
                                  points[camera], params)
        FigureCanvasAgg(figure).draw()
        np.testing.assert_array_equal(
            figure.axes[0].collections[0].get_offsets(),
            points[camera] / upscale_factor,
        )

    # Exported CSV retains scientific detector coordinates and declares the
    # exact factor needed to compare them with the native-frame overlays.
    csv_path = tmp_path / "coordinates.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *_args, **_kwargs: (str(csv_path), "CSV Files (*.csv)"))
    tab._save_detection_coordinates_csv()
    with csv_path.open(encoding="utf-8", newline="") as stream:
        metadata = json.loads(stream.readline()[2:])
        rows = list(csv.DictReader(stream))
    assert metadata["native_overlay_scale"] == 1.0 / upscale_factor
    assert [(float(row["x_px"]), float(row["y_px"])) for row in rows] == [
        tuple(point) for camera in cameras for point in points[camera]
    ]
    np.testing.assert_array_equal(detections.get_data(), original_rows)


def test_rescale_and_gamma_coordinates_are_already_native():
    from pyCamSet.gui.run_results import _points_in_native_frame

    points = np.array([[8.0, 12.0]])
    np.testing.assert_array_equal(
        _points_in_native_frame(points, {"upscale_factor": 2, "rescale_and_gamma": True}),
        points,
    )
