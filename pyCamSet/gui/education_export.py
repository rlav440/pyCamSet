"""Save an education stage's figure, and its PyVista 3D render, through the GUI's export presets.

The planning (:func:`plan_outputs`) and the writing (:func:`write_outputs`) are plain functions
over the stage view, so the tests can drive them without a dialog.  The control
(:class:`StageFigureSaveControl`) reuses the figure card's export path: the same size templates
(``EXPORT_SIZE_TEMPLATES``), the same per-visual preset (``bind_export_preset``), the same
``save_figure_at_width`` and the same file-name check as ``MatplotlibFigureCard``.

Outputs for the stage on screen:

* every visible matplotlib figure (the 2D panel, or the static 3D picture) in the chosen
  format, named ``<name>.<ext>`` when there is one and ``<name>_panel<n>.<ext>`` when there are
  several;
* when the 3D panel is the PyVista backend, its real render as ``<name>_3d.png``. A PyVista
  render is a raster image, so it is saved as PNG whatever format was chosen. It is captured at
  an integer enlargement, then resampled to the exact export width, and it carries the alt text
  in its ``Description`` and the panel heading (the 3D title and subtitle) above the picture.

Writing is all or nothing: every output is first written to a temporary name beside its target,
and only when all of them succeed are they moved onto the final names (``os.replace``). A failure
deletes the temporaries and leaves any file with the same name as it was.

A PyVista capture runs in the GUI process. A native OpenGL fault during that capture cannot be
caught as an exception, so it can still end the process; this module does not guard against it.

The default name is ``phase<n>_stage<k>_L<level>``, for example ``phase4_stage4_L2``.
"""
from __future__ import annotations

import contextlib
import html
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QMenu, QMessageBox,
                               QPushButton, QToolButton, QWidget)

#: The formats a stage figure can be saved in (the card's PNG, SVG and PDF choices).
FIGURE_FORMATS = ("PNG", "SVG", "PDF")

#: The most pixels a PyVista capture may take. Above this the capture is made smaller and then
#: enlarged to the export width, and the learner is told so.
MAX_SCENE_PIXELS = 16_000_000


@dataclass(frozen=True)
class StageOutput:
    """One file to write: a matplotlib ``figure`` or a PyVista/static ``scene`` to screenshot."""

    path: Path
    output_format: str
    figure: Any = None
    scene: Any = None


def phase_slug(label: str | None) -> str:
    """``"Phase 4"`` -> ``"phase4"``; any other label -> a lower-case slug (``"stage"`` if empty)."""
    match = re.search(r"\d+", label or "")
    if match:
        return f"phase{match.group()}"
    slug = re.sub(r"[^a-z0-9]+", "-", (label or "").casefold()).strip("-")
    return slug or "stage"


def default_stem(view) -> str:
    """The default file name for the view's stage at its level, e.g. ``phase4_stage4_L2``."""
    return f"{phase_slug(view.phase_label)}_stage{view.current_stage() + 1}_L{view.level()}"


def _visible_of(view, kind: type) -> list:
    """Widgets of ``kind`` shown in the stage's visual area (not on a hidden stack page)."""
    host = view.visual.currentWidget()
    if host is None:
        return []
    pool = ([host] if isinstance(host, kind) else []) + list(host.findChildren(kind))
    return [widget for widget in pool if widget.isVisibleTo(view.visual)]


def stage_description(view) -> str:
    """The alt text the GUI attaches to the stage, with its provenance where it has one.

    The same wording as ``MatplotlibFigureCard``: the canvas's accessible description (the
    caption where a stage sets none), then ``Provenance: ...`` from the visual panel's tooltip.
    """
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    description = ""
    for canvas in _visible_of(view, FigureCanvasQTAgg):
        description = canvas.accessibleDescription()
        if description:
            break
    description = description or view.caption.text()
    provenance = view.visual_panel.toolTip()
    if provenance:
        description = f"{description} Provenance: {provenance}"
    return description.strip()


def _plain_text(text: str) -> str:
    """Visible text of a label that may hold rich text (tags removed, entities decoded)."""
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def stage_heading(view) -> list[str]:
    """The 3D panel heading lines (title, then subtitle) of the stage's visual, if it has one.

    The static 3D picture has no title of its own; the heading is the panel widget beside it
    (``PanelHeading`` lines in Phase 4, the title label in Phase 2).
    """
    host = view.visual.currentWidget()
    lines: list[str] = []
    for attribute in ("title", "subtitle"):
        widget = getattr(host, attribute, None)
        reader = getattr(widget, "text", None)
        text = _plain_text(reader()) if callable(reader) else ""
        if text:
            lines.append(text)
    return lines


def plan_outputs(view, chosen: str | Path, output_format: str) -> list[StageOutput]:
    """The files that saving the stage now would write, named from the path ``chosen``.

    Raises ``ValueError`` when the stage shows nothing that can be saved.
    """
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from pyCamSet.gui.education_scene3d import Scene3D

    fmt = output_format.lower()
    chosen = Path(chosen)
    stem = chosen.with_suffix("")
    canvases = _visible_of(view, FigureCanvasQTAgg)
    pyvista_scenes = [scene for scene in _visible_of(view, Scene3D) if scene.backend == "pyvista"]
    if not canvases and not pyvista_scenes:
        raise ValueError("This stage has no figure to save.")
    outputs: list[StageOutput] = []
    if len(canvases) == 1:
        outputs.append(StageOutput(chosen, fmt, figure=canvases[0].figure))
    else:
        for number, canvas in enumerate(canvases, start=1):
            outputs.append(StageOutput(Path(f"{stem}_panel{number}.{fmt}"), fmt,
                                       figure=canvas.figure))
    for scene in pyvista_scenes:
        outputs.append(StageOutput(Path(f"{stem}_3d.png"), "png", scene=scene))
    return outputs


def _metadata(output_format: str, description: str) -> dict[str, str] | None:
    """The alt text in the file's own metadata field for its format (None when there is none)."""
    if not description:
        return None
    if output_format == "pdf":
        return {"Subject": description}
    if output_format in ("png", "svg"):
        return {"Description": description}
    return None


def _capture_scale(scene: Any, target_px: int) -> tuple[int, bool]:
    """Whole-number capture scale for a render at least ``target_px`` wide.

    Returns ``(scale, capped)``: ``capped`` is True when the memory budget (``MAX_SCENE_PIXELS``)
    made the capture smaller than the width needs.
    """
    source = getattr(scene, "interactor", None) or scene
    width = max(1, int(source.width()))
    height = max(1, int(source.height()))
    wanted = max(1, math.ceil(target_px / width))
    # The largest scale whose capture stays inside the pixel budget.
    largest = max(1, math.isqrt(MAX_SCENE_PIXELS // (width * height)))
    scale = min(wanted, largest)
    return scale, scale < wanted


def _heading_image(image, lines: list[str], theme: str | None):
    """``image`` with the panel heading lines in a band above it, in the stage's colours."""
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    from pyCamSet.gui import education_style as es

    tokens = es.stage_tokens(theme)
    size = max(12, round(image.width * 0.028))
    regular = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    fonts = [ImageFont.truetype(bold, size + 4)] + [ImageFont.truetype(regular, size)] * (len(lines) - 1)
    probe = ImageDraw.Draw(image)
    pad = size
    gap = round(size * 0.4)
    usable = image.width - 2 * pad
    fitted: list[str] = []
    for line, font in zip(lines, fonts):
        text = line
        while text and probe.textlength(text, font=font) > usable:
            text = text[:-2].rstrip() + "…"
            if len(text) <= 1:
                break
        fitted.append(text)
    heights = [probe.textbbox((0, 0), line, font=font)[3] for line, font in zip(fitted, fonts)]
    band = pad + sum(heights) + gap * (len(fitted) - 1) + pad
    canvas = Image.new("RGB", (image.width, image.height + band), tokens["stage"])
    draw = ImageDraw.Draw(canvas)
    y = pad
    for line, font, height in zip(fitted, fonts, heights):
        draw.text((pad, y), line, font=font, fill=tokens["text"])
        y += height + gap
    canvas.paste(image, (0, band))
    return canvas


def _write_scene_png(scene: Any, path: Path, target_px: int, description: str,
                     heading: list[str], notes: list[str] | None) -> None:
    """Capture a PyVista scene to ``path`` at exactly ``target_px`` wide, with its metadata."""
    from PIL import Image, PngImagePlugin

    scale, capped = _capture_scale(scene, target_px)
    scene.screenshot(path, scale=scale)
    if capped and notes is not None:
        notes.append("The 3D picture was enlarged from a smaller render to reach the export size, "
                     "because a full-size render would exceed 16 million pixels.")
    resample = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(path) as raw:
        image = raw.convert("RGB")
    height = max(1, round(image.height * target_px / image.width))
    image = image.resize((target_px, height), resample)
    if heading:
        image = _heading_image(image, heading, getattr(getattr(scene, "_impl", None), "theme", None))
    info = PngImagePlugin.PngInfo()
    if description:
        info.add_text("Description", description)
    image.save(path, format="PNG", pnginfo=info)


def _temporary_for(path: Path) -> Path:
    """A unique name beside ``path`` (same folder, so the final move is a rename)."""
    return path.with_name(f"{path.stem}.tmp-{uuid.uuid4().hex[:12]}{path.suffix}")


def write_outputs(view, outputs: list[StageOutput], width_mm: float, dpi: int,
                  notes: list[str] | None = None) -> list[Path]:
    """Write ``outputs`` at the export size, all or nothing; returns the final paths.

    Every output is written to a temporary name first. The files are moved onto their final
    names only after all of them are written; if any write fails the temporaries are deleted and
    the exception is raised with the existing files unchanged. ``notes`` collects any remark the
    learner should see (for example a capped PyVista capture).
    """
    from pyCamSet.utils.visualisation import save_figure_at_width

    description = stage_description(view)
    heading = stage_heading(view)
    target_px = round(float(width_mm) / 25.4 * int(dpi))
    staged: list[tuple[Path, Path]] = []
    try:
        for output in outputs:
            temporary = _temporary_for(output.path)
            staged.append((temporary, output.path))
            if output.figure is not None:
                save_figure_at_width(output.figure, temporary, width_mm, dpi, output.output_format,
                                     metadata=_metadata(output.output_format, description))
            else:
                _write_scene_png(output.scene, temporary, target_px, description, heading, notes)
        for temporary, final in staged:
            os.replace(temporary, final)
    except BaseException:
        for temporary, _final in staged:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise
    return [final for _temporary, final in staged]


def confirm_replace(parent: QWidget, existing: list[Path]) -> bool:
    """Ask before replacing ``existing``; "Keep" is the default, so Enter keeps the files."""
    names = "\n".join(path.name for path in existing)
    box = QMessageBox(QMessageBox.Icon.Question, "Replace existing files?",
                      f"These files already exist and will be replaced:\n{names}", parent)
    replace = box.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
    keep = box.addButton("Keep", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(keep)
    box.exec()
    return box.clickedButton() is replace


class StageFigureSaveControl(QWidget):
    """The stage's "Save figure" button and its menu (other formats, export size).

    The default format and size come from the export preset this phase's figures share, bound
    per visual id as the figure cards bind theirs.  Both buttons take the keyboard.
    """

    def __init__(self, view, parent: QWidget | None = None) -> None:
        super().__init__(parent or view)
        from pyCamSet.gui.action_icons import set_action_icon
        from pyCamSet.gui.preferences import bind_export_preset
        from pyCamSet.gui.shared_functions import EXPORT_SIZE_TEMPLATES

        self._view = view
        self.last_written: list[Path] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._preset = QComboBox(self)
        for label, size in EXPORT_SIZE_TEMPLATES:
            self._preset.addItem(label, size)
        bind_export_preset(self._preset, f"figure:education-{phase_slug(view.phase_label)}")
        # Hidden, as in MatplotlibFigureCard: it holds the bound size that the menu mirrors.
        self._preset.setVisible(False)

        self._save = QPushButton("Save figure", self)
        set_action_icon(self._save, "snapshot")
        self._save.setToolTip("Save this stage's figure at its current level and step as PNG, "
                              "at the export size chosen in the menu beside it")
        self._save.setAccessibleName("Save figure for this stage")
        self._save.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._save.clicked.connect(lambda: self.save("PNG"))
        layout.addWidget(self._save)

        self._more = QToolButton(self)
        self._more.setText("▾")
        self._more.setObjectName("saveMenuButton")
        # 24 px wide is the smallest target the accessibility guidance (WCAG 2.2, 2.5.8) accepts.
        self._more.setFixedSize(24, 28)
        self._more.setToolTip("More ways to save this figure: SVG, PDF and the export size")
        self._more.setAccessibleName("More save options for this stage")
        self._more.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self._more)
        menu.addAction("Save PNG", lambda: self.save("PNG"))
        for fmt in FIGURE_FORMATS[1:]:
            menu.addAction(f"Save {fmt}", lambda output_format=fmt: self.save(output_format))
        menu.addSeparator()
        size_menu = menu.addMenu("Export size")
        size_group = QActionGroup(size_menu)
        for index in range(self._preset.count()):
            action = size_menu.addAction(self._preset.itemText(index))
            action.setCheckable(True)
            action.setChecked(index == self._preset.currentIndex())
            action.triggered.connect(lambda _checked=False, i=index: self._preset.setCurrentIndex(i))
            size_group.addAction(action)
        self._preset.currentIndexChanged.connect(
            lambda i: size_group.actions()[i].setChecked(True)
            if 0 <= i < len(size_group.actions()) else None)
        self._more.setMenu(menu)
        layout.addWidget(self._more)

    def save(self, output_format: str) -> list[Path]:
        """Ask for a file name, then save the stage; a cancelled dialog writes nothing."""
        from PySide6.QtWidgets import QFileDialog
        from pyCamSet.gui.visual_style import _validate_user_style_filename

        fmt = output_format.upper()
        extension = fmt.lower()
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save Figure as {fmt}", f"{default_stem(self._view)}.{extension}",
            f"{fmt} Files (*.{extension})")
        self.last_written = []
        if not path:
            return []
        # The name always carries the extension of the format it is saved in.
        chosen = Path(path).with_suffix(f".{extension}")
        try:
            _validate_user_style_filename(chosen.name)
            outputs = plan_outputs(self._view, chosen, fmt)
            existing = [output.path for output in outputs if output.path.exists()]
            if existing and not confirm_replace(self, existing):
                return []
            width_mm, dpi = self._preset.currentData()
            notes: list[str] = []
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.last_written = write_outputs(self._view, outputs, width_mm, dpi, notes=notes)
            finally:
                QApplication.restoreOverrideCursor()
            if notes:
                QMessageBox.information(self, f"{fmt} saved", "\n".join(notes))
        except Exception as exc:
            QMessageBox.warning(
                self, f"{fmt} export failed",
                "No file was written: the figure could not be saved, and any file with the same "
                f"name was left as it was.\n\nTechnical detail: {exc}")
        return list(self.last_written)
