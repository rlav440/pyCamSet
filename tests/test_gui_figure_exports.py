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
    assess_calibration.spawn_calibration_viewer(tmp_path / "run.camset", theme_name="Sepia")
    assert captured["module"] == "pyCamSet.utils.visualise_camset"
    assert captured["arguments"] == [str(tmp_path / "run.camset"), "--theme", "Sepia"]


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
        {"run_id": "r"}, tmp_path, "Dark", 85, 300)
    assert ok
    assert captured["arguments"] == [
        str(tmp_path / "r.camset"), "--save-dir", str(tmp_path), "--no-show",
        "--theme", "Dark", "--figure-width-mm", "85", "--figure-dpi", "300",
        "--figure-formats", "png", "svg", "pdf", "--matplotlib-only",
    ]
