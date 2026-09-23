'''Purpose: Verify GUI-managed Matplotlib export contracts and source-backed CSV metadata.
Status: Active; offscreen export regression tests.
Future: Exercise native file-dialog workflows on Windows, Linux and macOS runners.
'''
from __future__ import annotations

import csv
import json

import pytest

pytest.importorskip("PySide6")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PIL import Image
from PySide6.QtWidgets import QApplication, QFileDialog

from pyCamSet.gui.shared_functions import MatplotlibFigureCard


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _card(qapp, csv_export=None):
    figure = Figure(figsize=(4, 2))
    axes = figure.add_subplot(111)
    axes.plot([0, 1], [2, 3])
    return MatplotlibFigureCard(
        "Test figure", figure, FigureCanvasQTAgg, csv_export=csv_export,
    )


def test_png_preset_controls_physical_width_dpi_and_restores_figure(tmp_path, monkeypatch, qapp):
    card = _card(qapp)
    original = card._fig.get_size_inches().copy()
    target = tmp_path / "figure.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), "PNG"))
    card._save_png()
    assert target.is_file()
    with Image.open(target) as image:
        assert image.size == (945, 472)
    assert card._fig.get_size_inches().tolist() == original.tolist()
    card._preset.setCurrentIndex(1)
    card._save_png()
    with Image.open(target) as image:
        assert image.size == (1004, 502)
    assert card._fig.get_size_inches().tolist() == original.tolist()


def test_detection_montage_presets_set_pixel_dimensions_and_cancel_is_noop(tmp_path, monkeypatch, qapp):
    from types import SimpleNamespace
    from pyCamSet.gui import phase_1_detection

    figure = Figure(figsize=(4, 2))
    figure.add_subplot(111).plot([0, 1], [2, 3])
    original = figure.get_size_inches().copy()
    target = tmp_path / "montage.png"
    tab = SimpleNamespace(
        _draw_state={"fig": figure},
        _montage_export_preset=SimpleNamespace(currentData=lambda: (160.0, 150)),
    )
    monkeypatch.setattr(phase_1_detection.QFileDialog, "getSaveFileName", lambda *args: (str(target), "PNG"))
    phase_1_detection.Phase1DiagnosticsTab._save_detection_montage_png(tab)
    with Image.open(target) as image:
        assert image.size == (945, 472)
    assert figure.get_size_inches().tolist() == original.tolist()

    publication_target = tmp_path / "montage-publication.png"
    tab._montage_export_preset.currentData = lambda: (85.0, 300)
    monkeypatch.setattr(phase_1_detection.QFileDialog, "getSaveFileName",
                        lambda *args: (str(publication_target), "PNG"))
    phase_1_detection.Phase1DiagnosticsTab._save_detection_montage_png(tab)
    with Image.open(publication_target) as image:
        assert image.size == (1004, 502)
    assert figure.get_size_inches().tolist() == original.tolist()

    target.unlink()
    monkeypatch.setattr(phase_1_detection.QFileDialog, "getSaveFileName", lambda *args: ("", ""))
    phase_1_detection.Phase1DiagnosticsTab._save_detection_montage_png(tab)
    assert not target.exists()


def test_csv_source_adapter_preserves_values_and_metadata(tmp_path, monkeypatch, qapp):
    adapter = {
        "columns": ["view", "error_px"],
        "rows": [(2, 0.125), (3, 0.25)],
        "metadata": {"run_id": "r1", "units": {"error_px": "px"}, "data_kind": "observed"},
    }
    card = _card(qapp, adapter)
    target = tmp_path / "source.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), "CSV"))
    card._save_csv()
    text = target.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(text[0].partition(": ")[2])
    assert metadata == adapter["metadata"]
    assert list(csv.reader(text[1:])) == [["view", "error_px"], ["2", "0.125"], ["3", "0.25"]]


def test_csv_is_disabled_without_adapter_and_cancel_is_noop(tmp_path, monkeypatch, qapp):
    card = _card(qapp)
    assert not card._csv_btn.isEnabled()
    assert "no tabular source adapter" in card._csv_btn.toolTip()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: ("", ""))
    card._save_png()
    card._save_csv()
    assert list(tmp_path.iterdir()) == []


def test_svg_and_pdf_exports_are_written(tmp_path, monkeypatch, qapp):
    card = _card(qapp)
    for output_format in ("svg", "pdf"):
        target = tmp_path / f"figure.{output_format}"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, p=target: (str(p), output_format.upper()))
        card._save_vector(output_format.upper())
        assert target.is_file() and target.stat().st_size > 100


def test_assess_calibration_child_receives_selected_theme(monkeypatch, tmp_path):
    from pyCamSet.gui import assess_calibration

    captured = {}
    monkeypatch.setattr(assess_calibration, "spawn_viewer",
                        lambda module, arguments: captured.update(module=module, arguments=arguments) or (True, ""))
    assess_calibration.spawn_calibration_viewer(
        tmp_path / "run.camset", theme_name="Sepia", figure_themes=("Light", "Sepia", "Dark"))
    assert captured["module"] == "pyCamSet.utils.visualise_camset"
    assert captured["arguments"] == [
        str(tmp_path / "run.camset"), "--theme", "Sepia",
        "--figure-themes", "Light", "Sepia", "Dark",
    ]


@pytest.mark.parametrize("module_name,class_name", [
    ("pyCamSet.gui.phase_3_bundle_adjustment", "Phase3DiagnosticsTab"),
    ("pyCamSet.gui.phase_4_self_calibration", "Phase4DiagnosticsTab"),
])
def test_assess_calibration_action_forwards_per_figure_themes(
    monkeypatch, qapp, module_name, class_name,
):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module(module_name)
    selected_run = {"run_id": "run"}
    captured = {}
    monkeypatch.setattr(module, "select_latest_visualisation_run", lambda selected, runs: selected_run)
    monkeypatch.setattr(
        module, "launch_visualise_calibration_for_run",
        lambda run, **kwargs: captured.update(run=run, kwargs=kwargs) or (True, ""),
    )
    qapp.setProperty("pycamsetTheme", "Dark")
    tab = SimpleNamespace(
        _run_selector=SimpleNamespace(get_selected=lambda: [selected_run]),
        _all_runs=[selected_run],
        _current_run_label=SimpleNamespace(setText=lambda text: None),
        _open3d_cb=SimpleNamespace(isChecked=lambda: False),
        _assessment_figure_themes=[
            SimpleNamespace(currentText=lambda value=value: value)
            for value in ("Light", "Inherit", "Sepia")
        ],
    )
    getattr(getattr(module, class_name), "_run_visualise_target")(tab)
    assert captured["run"] is selected_run
    assert captured["kwargs"] == {
        "theme_name": "Dark", "figure_themes": ("Light", "Dark", "Sepia"),
    }


def test_assessment_figure_batch_formats_sizes_and_refuses_overwrite(tmp_path):
    from pyCamSet.utils.visualisation import save_figure

    figure = Figure(figsize=(4, 2))
    figure.add_subplot(111).plot([0, 1], [1, 4])
    original_size = figure.get_size_inches().copy()
    save_figure(figure, "assessment", tmp_path, width_mm=85, dpi=300,
                formats=("png", "svg", "pdf"))
    assert {"assessment.png", "assessment.svg", "assessment.pdf"} <= {
        path.name for path in tmp_path.iterdir()
    }
    with Image.open(tmp_path / "assessment.png") as image:
        assert image.size == (1004, 502)
    assert (tmp_path / "assessment.svg").read_text(encoding="utf-8").startswith("<?xml")
    assert (tmp_path / "assessment.pdf").read_bytes().startswith(b"%PDF")
    assert figure.get_size_inches().tolist() == original_size.tolist()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_figure(figure, "assessment", tmp_path, width_mm=180, dpi=300,
                    formats=("png", "svg", "pdf"))
    assert figure.get_size_inches().tolist() == original_size.tolist()


def test_gui_assessment_export_passes_preset_and_vector_formats(monkeypatch, tmp_path):
    from pyCamSet.gui import assess_calibration

    captured = {}
    monkeypatch.setattr(assess_calibration, "resolve_run_camset_artifact", lambda run: tmp_path / "r.camset")
    monkeypatch.setattr(assess_calibration, "run_viewer",
                        lambda module, arguments: captured.update(module=module, arguments=arguments) or (True, "ok"))
    ok, _ = assess_calibration.launch_save_assessment_pngs_for_run(
        {"run_id": "r"}, tmp_path, "Dark", 85, 300, ("Light", "Sepia", "Dark"))
    assert ok
    assert captured["arguments"] == [
        str(tmp_path / "r.camset"), "--save-dir", str(tmp_path), "--no-show",
        "--theme", "Dark", "--figure-width-mm", "85", "--figure-dpi", "300",
        "--figure-formats", "png", "svg", "pdf", "--matplotlib-only", "--export-csv",
        "--figure-themes", "Light", "Sepia", "Dark",
    ]


def test_assessment_csv_payloads_are_source_arrays_with_declared_units():
    import numpy as np
    from types import SimpleNamespace
    from pyCamSet.utils.visualisation import _assessment_csv_payloads

    class Detection:
        def get_data(self):
            return np.array([[0, 0, 0, 10.0, 20.0], [0, 0, 1, 12.0, 25.0]])

    class DetectionSet:
        cam_names = ["cam0"]

        def get_cam_list(self):
            return [Detection()]

    camera = SimpleNamespace(intrinsic=np.array([[1, 0, 11], [0, 1, 22], [0, 0, 1]]))
    diagnostic = SimpleNamespace(
        residuals=np.array([[1.0, 2.0], [-1.0, 0.0]]),
        euclidean_err=np.array([np.sqrt(5), 1.0]),
        detection=DetectionSet(), cams=[camera],
        accuracy=np.array([0.2]), precision=np.array([0.1]),
        feature_error=np.array([0.4]),
    )
    payloads = _assessment_csv_payloads(diagnostic)
    assert payloads["error_distribution"][1][0] == (0, 1.0, 2.0, pytest.approx(np.sqrt(5)))
    assert payloads["per_camera_coverage"][1][0][:5] == (0, "cam0", 0, 10.0, 20.0)
    assert payloads["accuracy_precision"][1] == [(0, 0.2, 0.1, 0.4)]


def test_assessment_csv_serialisation_preserves_metadata_rows(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from pyCamSet.utils.visualisation import _write_assessment_csvs

    class Detection:
        def get_data(self):
            return np.array([[0, 0, 0, 10.0, 20.0]])

    class DetectionSet:
        cam_names = ["cam0"]

        def get_cam_list(self):
            return [Detection()]

    diagnostic = SimpleNamespace(
        residuals=np.array([[3.0, 4.0]]), euclidean_err=np.array([5.0]),
        detection=DetectionSet(),
        cams=[SimpleNamespace(intrinsic=np.array([[1, 0, 11], [0, 1, 22], [0, 0, 1]]))],
        accuracy=np.array([0.2]), precision=np.array([0.1]), feature_error=np.array([0.4]),
    )
    paths = _write_assessment_csvs(diagnostic, tmp_path, "run.camset")
    assert len(paths) == 3
    lines = paths[0].read_text(encoding="utf-8").splitlines()
    metadata = json.loads(lines[0][2:])
    assert metadata["camset_source"] == "run.camset"
    assert metadata["units"]["x_error_px"] == "px"
    assert list(csv.reader(lines[1:]))[1] == ["0", "3.0", "4.0", "5.0"]

    blocked_dir = tmp_path / "blocked"
    blocked_dir.mkdir()
    existing = blocked_dir / "error_distribution.csv"
    existing.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _write_assessment_csvs(diagnostic, blocked_dir, "run.camset")
    assert not (blocked_dir / "per_camera_coverage.csv").exists()


def test_matplotlib_only_does_not_require_pyvista(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import numpy as np
    import pyCamSet.utils.visualisation as visualisation

    camera = SimpleNamespace(
        intrinsic=np.array([[10.0, 0.0, 50.0], [0.0, 10.0, 50.0], [0.0, 0.0, 1.0]]),
        res=(100, 100),
    )
    class CameraCollection(list):
        def get_n_cams(self):
            return len(self)

    detection_data = np.array([
        [0, 0, 0, 20.0, 30.0], [0, 0, 1, 50.0, 40.0], [0, 0, 2, 70.0, 80.0],
    ])
    detection = SimpleNamespace(
        cam_names=["cam0"],
        get_cam_list=lambda: [SimpleNamespace(get_data=lambda: detection_data)],
    )
    diagnostics = SimpleNamespace(
        residuals=np.array([1.0, 2.0, -1.0, 0.5, 0.1, -0.3]),
        euclidean_err=np.array([np.hypot(1, 2), np.hypot(1, 0.5), np.hypot(0.1, 0.3)]),
        e_lim=4.0, cams=CameraCollection([camera]), detection=detection,
        accuracy=np.array([0.2, 0.3]), precision=np.array([0.1, 0.2]),
        feature_error=np.array([0.4, 0.8]),
    )
    monkeypatch.setattr(visualisation, "_PYVISTA_OK", False)
    monkeypatch.setattr(visualisation.CalibrationDiagnostics, "from_results", lambda *args: diagnostics)
    monkeypatch.setattr(visualisation, "reconstruction_scene", lambda *args: pytest.fail("3D scene requested"))
    result = visualisation.visualise_calibration({}, object(), show=False,
                                                save_dir=tmp_path, matplotlib_only=True,
                                                export_csv=True, provenance="synthetic.camset")
    assert [path.name for path in result] == [
        "error_distribution.png", "per_camera_coverage.png", "accuracy_precision.png",
        "error_distribution.csv", "per_camera_coverage.csv", "accuracy_precision.csv",
    ]
    assert all(path.is_file() and path.stat().st_size > 100 for path in result)
    csv_metadata = json.loads(result[3].read_text(encoding="utf-8").splitlines()[0][2:])
    assert csv_metadata["camset_source"] == "synthetic.camset"


def test_3d_request_still_requires_pyvista(monkeypatch):
    import pyCamSet.utils.visualisation as visualisation

    monkeypatch.setattr(visualisation, "_PYVISTA_OK", False)
    with pytest.raises(ImportError, match="pyvista is required"):
        visualisation.visualise_calibration({}, object(), show=False, matplotlib_only=False)
