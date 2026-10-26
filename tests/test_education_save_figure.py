'''Verify that an education stage's figure saves through the GUI's export presets, and that a failed save leaves the learner's files as they were.
'''
from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("PYCAMSET_EDU_3D", "static")

from PIL import Image  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

# The synthetic fixtures the phase visual tests build their views from.
from test_education_phase2_visual import synthetic_run  # noqa: E402,F401
from test_education_phase4_visual import _view  # noqa: E402


def _app():
    return QApplication.instance() or QApplication([])


def _shown(view, width: int = 1380, height: int = 820):
    view.resize(width, height)
    view.show()
    for _ in range(3):
        QApplication.processEvents()
    return view


def _choose(monkeypatch, tmp_path: Path, recorded: list):
    """Answer the save dialog with ``tmp_path`` plus the default name it offered."""
    def dialog(_parent, _title, default_name, _filter):
        recorded.append(default_name)
        return str(tmp_path / default_name), "filter"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", dialog)


def _png_text(path: Path) -> dict:
    with Image.open(path) as image:
        image.load()
        return dict(image.info), image.size


def _no_temporaries(folder: Path) -> bool:
    return not list(folder.glob("*.tmp-*"))


def test_2d_stage_saves_png_named_by_phase_stage_and_level(tmp_path, monkeypatch):
    """A 2D stage (Phase 4, stage 4 "Result") saves phase4_stage4_L2.png at the preset width."""
    _app()
    from pyCamSet.gui.education_export import stage_description
    from pyCamSet.gui.shared_functions import EXPORT_SIZE_TEMPLATES

    view, _visual = _view(tmp_path / "run", monkeypatch)
    try:
        stage_view = view.stage_view
        stage_view.set_level(2)
        stage_view.set_current_stage(3)
        _shown(view)
        recorded: list[str] = []
        _choose(monkeypatch, tmp_path, recorded)
        written = stage_view.save_control.save("PNG")
        assert recorded == ["phase4_stage4_L2.png"]
        assert written == [tmp_path / "phase4_stage4_L2.png"]
        target = written[0]
        assert target.is_file() and target.stat().st_size > 0
        info, size = _png_text(target)
        width_mm, dpi = EXPORT_SIZE_TEMPLATES[0][1]
        assert size[0] == round(width_mm / 25.4 * dpi)
        # The GUI's alt text (caption, with provenance where the stage has any) is kept in the file.
        assert info.get("Description") == stage_description(stage_view)
        assert info["Description"]
        assert _no_temporaries(tmp_path)
    finally:
        view.close()


def test_3d_stage_saves_its_static_picture_and_the_stage_figure(tmp_path, monkeypatch):
    """Phase 4 stage 1 is a 3D panel beside a side figure; the static backend saves both panels."""
    _app()
    from pyCamSet.gui.education_export import default_stem

    view, _visual = _view(tmp_path / "run", monkeypatch)
    try:
        stage_view = view.stage_view
        stage_view.set_level(1)
        stage_view.set_current_stage(0)
        _shown(view)
        recorded: list[str] = []
        _choose(monkeypatch, tmp_path, recorded)
        written = stage_view.save_control.save("PNG")
        stem = default_stem(stage_view)
        assert stem == "phase4_stage1_L1"
        assert recorded == [f"{stem}.png"]
        assert written, "a 3D stage must write at least one file"
        for path in written:
            assert path.parent == tmp_path and path.name.startswith(stem)
            assert path.suffix == ".png" and path.stat().st_size > 0
            with Image.open(path) as image:
                assert image.format == "PNG"
    finally:
        view.close()


def test_phase2_2d_and_3d_stages_save_and_pick_the_matching_names(synthetic_run, tmp_path, monkeypatch):
    """Phase 2 has a 2D stage (Board views) and a 3D stage (Board in 3D); both save."""
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    view = Phase2View(synthetic_run.context)
    _shown(view, 1400, 900)
    try:
        stage_view = view.stage_view
        stage_view.set_level(1)
        recorded: list[str] = []
        _choose(monkeypatch, tmp_path, recorded)

        stage_view.set_current_stage(0)
        flat = stage_view.save_control.save("PNG")
        assert recorded[-1] == "phase2_stage1_L1.png"
        assert flat == [tmp_path / "phase2_stage1_L1.png"]
        assert flat[0].stat().st_size > 0

        stage_view.set_current_stage(1)
        spatial = stage_view.save_control.save("SVG")
        assert recorded[-1] == "phase2_stage2_L1.svg"
        assert spatial, "the 3D stage must write its picture"
        for path in spatial:
            assert path.name.startswith("phase2_stage2_L1") and path.suffix == ".svg"
            assert b"<svg" in path.read_bytes()
    finally:
        view.close()


def test_pdf_and_svg_formats_are_written_from_a_2d_stage(tmp_path, monkeypatch):
    _app()
    view, _visual = _view(tmp_path / "run", monkeypatch)
    try:
        stage_view = view.stage_view
        stage_view.set_current_stage(3)
        _shown(view)
        recorded: list[str] = []
        _choose(monkeypatch, tmp_path, recorded)
        svg = stage_view.save_control.save("SVG")
        pdf = stage_view.save_control.save("PDF")
        assert recorded == ["phase4_stage4_L1.svg", "phase4_stage4_L1.pdf"]
        assert svg[0].suffix == ".svg" and b"<svg" in svg[0].read_bytes()[:2000]
        assert pdf[0].suffix == ".pdf" and pdf[0].read_bytes().startswith(b"%PDF")
    finally:
        view.close()


def test_svg_and_pdf_keep_the_alt_text_in_their_metadata(tmp_path, monkeypatch):
    """SVG keeps the description in dc:description; PDF keeps it in the /Subject field."""
    _app()
    from pyCamSet.gui.education_export import stage_description

    view, _visual = _view(tmp_path / "run", monkeypatch)
    try:
        stage_view = view.stage_view
        stage_view.set_current_stage(3)
        _shown(view)
        description = stage_description(stage_view)
        assert description
        # Punctuation is escaped differently in each format; the words themselves survive.
        needle = " ".join(re.sub(r"[^A-Za-z0-9 ]", " ", description).split()[:3]).encode("ascii")
        _choose(monkeypatch, tmp_path, [])
        svg = stage_view.save_control.save("SVG")[0].read_bytes()
        assert b"<dc:description>" in svg and needle in svg
        pdf = stage_view.save_control.save("PDF")[0].read_bytes()
        # matplotlib writes a Subject with non-ASCII characters as UTF-16BE, so accept either form.
        assert b"/Subject" in pdf
        assert needle in pdf or needle.decode("ascii").encode("utf-16-be") in pdf
    finally:
        view.close()


def test_pyvista_render_is_saved_beside_the_figure_at_the_preset_size(tmp_path, monkeypatch):
    """With the PyVista backend the 3D panel is also saved, captured and resampled to the preset width.

    No OpenGL context is available offscreen, so the backend is switched on the class and the
    capture is taken by the static picture. The scale the export asks for is passed through, and
    the file's width, metadata and heading are checked on the result.
    """
    _app()
    from pyCamSet.gui import education_scene3d
    from pyCamSet.gui.education_export import plan_outputs, stage_description, stage_heading, write_outputs
    from pyCamSet.gui.shared_functions import EXPORT_SIZE_TEMPLATES

    view, _visual = _view(tmp_path / "run", monkeypatch)
    try:
        stage_view = view.stage_view
        stage_view.set_current_stage(0)
        _shown(view)
        monkeypatch.setattr(education_scene3d.Scene3D, "backend", property(lambda _self: "pyvista"))
        scales: list = []
        original = education_scene3d.Scene3D.screenshot

        def recording(self, path, scale=None):
            scales.append(scale)
            return original(self, path, scale=scale)

        monkeypatch.setattr(education_scene3d.Scene3D, "screenshot", recording)
        outputs = plan_outputs(stage_view, tmp_path / "phase4_stage1_L1.png", "PNG")
        scene_outputs = [o for o in outputs if o.scene is not None]
        assert len(scene_outputs) == 1
        assert scene_outputs[0].path == tmp_path / "phase4_stage1_L1_3d.png"
        width_mm, dpi = EXPORT_SIZE_TEMPLATES[0][1]
        target_px = round(width_mm / 25.4 * dpi)
        notes: list[str] = []
        written = write_outputs(stage_view, outputs, width_mm, dpi, notes=notes)
        assert tmp_path / "phase4_stage1_L1_3d.png" in written
        final = tmp_path / "phase4_stage1_L1_3d.png"
        info, size = _png_text(final)
        assert size[0] == target_px, "the 3D picture must be exactly the preset width"
        assert info.get("Description") == stage_description(stage_view) and info["Description"]
        assert stage_heading(stage_view), "the 3D panel heading must be known for the stage"
        assert scales and all(isinstance(s, int) and s >= 1 for s in scales)
        assert notes == [], "no enlargement note when the capture is within budget"
        assert _no_temporaries(tmp_path)
    finally:
        view.close()


def test_capture_scale_is_whole_and_capped_by_the_pixel_budget(monkeypatch):
    """The capture is an integer enlargement that reaches the width, within MAX_SCENE_PIXELS."""
    from pyCamSet.gui import education_export

    def scene(width, height):
        return SimpleNamespace(width=lambda: width, height=lambda: height, interactor=None)

    assert education_export._capture_scale(scene(1400, 900), 945) == (1, False)
    assert education_export._capture_scale(scene(300, 200), 945) == (4, False)
    monkeypatch.setattr(education_export, "MAX_SCENE_PIXELS", 10_000)
    assert education_export._capture_scale(scene(300, 200), 945) == (1, True)


def test_failed_export_leaves_an_existing_file_of_the_same_name_intact(tmp_path, monkeypatch):
    """Two panels, the second write fails: the earlier same-named file is untouched and nothing new is left."""
    _app()
    from pyCamSet.gui import education_export
    from pyCamSet.gui.education_export import default_stem, plan_outputs
    from pyCamSet.utils import visualisation

    view, _visual = _view(tmp_path / "run", monkeypatch)
    try:
        stage_view = view.stage_view
        stage_view.set_level(1)
        stage_view.set_current_stage(0)
        _shown(view)
        chosen = tmp_path / f"{default_stem(stage_view)}.png"
        outputs = plan_outputs(stage_view, chosen, "PNG")
        assert len(outputs) >= 2, "the test needs two files to fail on the second"
        earlier = outputs[0].path
        earlier.write_bytes(b"previous export")

        real_save = visualisation.save_figure_at_width
        calls = {"count": 0}

        def failing_second(figure, path, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("disk full (simulated)")
            return real_save(figure, path, *args, **kwargs)

        warnings: list = []
        monkeypatch.setattr(visualisation, "save_figure_at_width", failing_second)
        monkeypatch.setattr(education_export.QMessageBox, "warning",
                            lambda *args, **kwargs: warnings.append(args))
        monkeypatch.setattr(education_export, "confirm_replace", lambda _parent, _existing: True)
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(chosen), "PNG"))

        assert stage_view.save_control.save("PNG") == []
        assert calls["count"] == 2
        assert earlier.read_bytes() == b"previous export"
        assert not outputs[1].path.exists()
        assert _no_temporaries(tmp_path)
        assert stage_view.save_control.last_written == []
        assert warnings and "No file was written" in warnings[0][2]
    finally:
        view.close()


def test_cancelled_dialog_writes_nothing_for_any_format(tmp_path, monkeypatch):
    _app()
    view, _visual = _view(tmp_path / "run", monkeypatch)
    try:
        stage_view = view.stage_view
        stage_view.set_current_stage(3)
        _shown(view)
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: ("", ""))
        for fmt in ("PNG", "SVG", "PDF"):
            assert stage_view.save_control.save(fmt) == []
        written = [p for p in tmp_path.rglob("*") if p.suffix.lower() in (".png", ".svg", ".pdf")]
        assert written == []
        assert stage_view.save_control.last_written == []
    finally:
        view.close()


def test_existing_derived_files_are_replaced_only_on_confirmation(tmp_path, monkeypatch):
    """Two panels: the second file already exists; answering Keep keeps it and writes nothing."""
    _app()
    view, _visual = _view(tmp_path / "run", monkeypatch)
    try:
        stage_view = view.stage_view
        stage_view.set_current_stage(0)
        _shown(view)
        from pyCamSet.gui import education_export
        from pyCamSet.gui.education_export import default_stem, plan_outputs
        chosen = tmp_path / f"{default_stem(stage_view)}.png"
        outputs = plan_outputs(stage_view, chosen, "PNG")
        if len(outputs) < 2:
            pytest.skip("this stage shows a single figure")
        existing = outputs[1].path
        existing.write_bytes(b"keep me")
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(chosen), "PNG"))
        asked: list = []

        def keep(_parent, files):
            asked.append([p.name for p in files])
            return False

        monkeypatch.setattr(education_export, "confirm_replace", keep)
        assert stage_view.save_control.save("PNG") == []
        assert asked == [[existing.name]]
        assert existing.read_bytes() == b"keep me"
        assert not chosen.exists()
    finally:
        view.close()


def test_saved_name_takes_the_extension_of_the_format(tmp_path, monkeypatch):
    """A file offered as .png but saved as SVG is named .svg, so the name never lies about the content."""
    _app()
    view, _visual = _view(tmp_path / "run", monkeypatch)
    try:
        stage_view = view.stage_view
        stage_view.set_current_stage(3)
        _shown(view)
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            lambda *_args: (str(tmp_path / "figure.png"), "filter"))
        written = stage_view.save_control.save("SVG")
        assert written == [tmp_path / "figure.svg"]
        assert b"<svg" in written[0].read_bytes()[:2000]
    finally:
        view.close()


def test_control_is_keyboard_reachable_and_named(tmp_path, monkeypatch):
    _app()
    view, _visual = _view(tmp_path / "run", monkeypatch)
    try:
        stage_view = view.stage_view
        stage_view.set_current_stage(3)
        _shown(view)
        control = stage_view.save_control
        for button in (control._save, control._more):
            assert button.isVisible() and button.isEnabled()
            assert button.focusPolicy() == Qt.FocusPolicy.StrongFocus
            assert button.accessibleName()
            assert button.toolTip()
        assert control._save.accessibleName() == "Save figure for this stage"
        # Space on the focused Save button saves the default file.
        recorded: list[str] = []
        _choose(monkeypatch, tmp_path, recorded)
        control._save.setFocus()
        QTest.keyClick(control._save, Qt.Key.Key_Space)
        QApplication.processEvents()
        assert recorded == ["phase4_stage4_L1.png"]
        assert (tmp_path / "phase4_stage4_L1.png").is_file()
    finally:
        view.close()


def test_tab_order_runs_from_save_to_the_menu_button(tmp_path, monkeypatch):
    """Tab order: the Save button is followed by its menu button, with nothing in between."""
    _app()
    view, _visual = _view(tmp_path / "run", monkeypatch)
    try:
        stage_view = view.stage_view
        stage_view.set_current_stage(3)
        _shown(view)
        control = stage_view.save_control
        reached = False
        widget = control._save
        for _ in range(6):
            widget = widget.nextInFocusChain()
            if widget is control._more:
                reached = True
                break
        assert reached, "the menu button must follow the Save button in the focus chain"
    finally:
        view.close()


def test_save_control_fits_the_stage_header_at_1280_by_800(tmp_path, monkeypatch):
    """The header keeps one row at 1280x800: the Save control sits level with the stepper and inside the window."""
    _app()
    view, _visual = _view(tmp_path / "run", monkeypatch)
    try:
        stage_view = view.stage_view
        stage_view.set_current_stage(3)
        _shown(view, 1280, 800)
        control = stage_view.save_control
        assert control.isVisible()
        assert control._more.width() >= 24
        stepper_top = stage_view.stepper.mapTo(stage_view, stage_view.stepper.rect().topLeft()).y()
        control_top = control.mapTo(stage_view, control.rect().topLeft()).y()
        assert abs(stepper_top - control_top) < stage_view.stepper.height(), "Save is not on the stepper's row"
        right_edge = control.mapTo(stage_view, control.rect().topRight()).x()
        assert right_edge <= stage_view.width(), "the Save control is cut off at 1280 px"
    finally:
        view.close()
