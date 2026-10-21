"""Render review screenshots of the REAL education dialog for saved runs (offscreen).

Opens ``EducationDialog`` on ``EducationRunView.from_run_dir(run)`` at the requested size(s) and
theme(s), walks every stage x detail level (L1-L3) plus every stage-internal step-through, and
grabs the dialog to PNG. Each screenshot is also measured (vertical scroll needed, the visual
area's share of the height, clipped/elided/overlapping Qt labels, text problems inside the
matplotlib figures and caption-length problems; best effort) into a JSON index.
Optionally saves the real PyVista renders of the 3D stages (``--pyvista``) by pointing the
view's scene at an ``OffscreenScene`` for one redraw.

Usage (one run)::

    python tools/edu_release/edu_screenshots.py --run <phaseN_runs/<id>> --dataset <name> \
        --out <scratch>/round1/<name>/p3 --size-spec 1400x900=Light,Dark,Sepia \
        --size-spec 1280x800=Light --pyvista

Batch (one subprocess per job, BelowNormal priority), then contact sheets + README::

    python tools/edu_release/edu_screenshots.py --batch plan.json
    python tools/edu_release/edu_screenshots.py --summarise <scratch>/round1

``plan.json``: ``{"root": "...", "jobs": [{"run": ..., "dataset": ..., "size_spec":
["1400x900=Light"], "levels": "1,2,3", "pyvista": true}, ...]}`` (``out`` defaults to
``<root>/<dataset>/p<phase>``).

Names: ``<dataset>_p<phase>_s<stage>_L<level>_<theme>[_step<k>][_<W>x<H>].png`` (size suffix
only when not 1400x900). Step meanings are written to the index (``step_label``).

The tool never writes into the run or dataset folders; it reads them only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIZE = (1400, 900)


def _below_normal() -> None:
    """Run this process (and its children) at BelowNormal priority."""
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001 - priority is best effort
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),
                                                    0x4000)
        except Exception:  # noqa: BLE001
            pass


def _offscreen_env() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if sys.platform == "win32":  # point Qt at the system fonts on Windows only
        os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if os.environ["QT_QPA_PLATFORM"].casefold() in {"offscreen", "minimal"}:
        os.environ.setdefault("PYCAMSET_EDU_3D", "static")


def _parse_size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def _size_suffix(size: tuple[int, int]) -> str:
    return "" if tuple(size) == DEFAULT_SIZE else f"_{size[0]}x{size[1]}"


# --------------------------------------------------------------------------------------
# Qt helpers


def settle(app: Any, ms: int = 0, loops: int = 6) -> None:
    """Process events (incl. deferred deletes) for at least ``loops`` passes and ``ms``."""
    from PySide6.QtCore import QCoreApplication, QEvent
    start = time.monotonic()
    n = 0
    while n < loops or (time.monotonic() - start) * 1000 < ms:
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        n += 1
        if ms:
            time.sleep(0.01)


def stage_view_of(view: Any) -> Any:
    from pyCamSet.gui.education_stage import EducationStageView
    if isinstance(view, EducationStageView):
        return view
    return getattr(view, "stage_view", None)


def wait_ready(app: Any, view: Any, timeout: float = 60.0) -> bool:
    """Wait for async work: the phase view's background preparation (education_worker),
    Phase 1's live detector re-run and its debounced redraws."""
    deadline = time.monotonic() + timeout
    settle(app, 60)
    while time.monotonic() < deadline:
        busy = False
        waiter = getattr(view, "wait_until_ready", None)
        if callable(waiter):
            waiter(timeout=max(1.0, deadline - time.monotonic()))
        worker = getattr(view, "_worker", None)
        if worker is not None and callable(getattr(worker, "is_busy", None)) and worker.is_busy():
            busy = True
        if callable(getattr(view, "wait_for_explanation", None)):
            view.wait_for_explanation(timeout=max(1.0, deadline - time.monotonic()))
            busy = busy or getattr(view, "_future", None) is not None
        for name in ("_resize_timer", "_poll"):
            timer = getattr(view, name, None)
            if timer is not None and timer.isActive():
                busy = True
        settle(app, 220 if busy else 40)
        if not busy:
            return True
    return False


def draw_canvases(app: Any, dialog: Any) -> None:
    """Draw every visible matplotlib canvas now (they otherwise draw lazily)."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    for canvas in dialog.findChildren(FigureCanvasQTAgg):
        if canvas.isVisible():
            try:
                canvas.draw()
            except Exception:  # noqa: BLE001 - a broken figure still gets a screenshot
                pass
    settle(app, 0, 3)


# --------------------------------------------------------------------------------------
# Layout measurements


def _global_rect(widget: Any) -> Any:
    from PySide6.QtCore import QRect
    top_left = widget.mapTo(widget.window(), widget.rect().topLeft())
    return QRect(top_left, widget.size())


def _label_needs(label: Any) -> tuple[float, float, float, float]:
    """(needed width, needed height, available width, available height) for the text."""
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QTextDocument
    from PySide6.QtGui import Qt as QtGuiQt
    cr = label.contentsRect()
    margin = label.margin()
    avail_w, avail_h = cr.width() - 2 * margin, cr.height() - 2 * margin
    # The transport bar's counter paints a short form; text() is the full sentence.
    display = getattr(label, "displayText", None)
    text = display() if callable(display) else label.text()
    fmt = label.textFormat()
    rich = fmt == Qt.TextFormat.RichText or (fmt == Qt.TextFormat.AutoText
                                            and QtGuiQt.mightBeRichText(text))
    if rich:
        doc = QTextDocument()
        doc.setDefaultFont(label.font())
        doc.setDocumentMargin(0)
        doc.setHtml(text)
        if label.wordWrap():
            doc.setTextWidth(max(avail_w, 1))
            return doc.idealWidth(), doc.size().height(), avail_w, avail_h
        return doc.idealWidth(), doc.size().height(), avail_w, avail_h
    metrics = label.fontMetrics()
    if label.wordWrap():
        rect = metrics.boundingRect(QRect(0, 0, max(avail_w, 1), 1_000_000),
                                    int(Qt.TextFlag.TextWordWrap), text)
    else:
        rect = metrics.boundingRect(QRect(0, 0, 1_000_000, 1_000_000), 0, text)
    return rect.width(), rect.height(), avail_w, avail_h


def _clipped_by_ancestors(widget: Any, stop: Any) -> bool:
    rect = _global_rect(widget)
    area = rect.width() * rect.height()
    if area <= 0:
        return False
    parent = widget.parentWidget()
    clip = rect
    while parent is not None:
        clip = clip.intersected(_global_rect(parent))
        if parent is stop:
            break
        parent = parent.parentWidget()
    return clip.width() * clip.height() < area - 2


def _short(text: str, n: int = 90) -> str:
    import html
    import re
    plain = re.sub(r"<br\s*/?>", " ", text)
    plain = html.unescape(re.sub(r"<[^>]+>", "", plain)).replace("\xa0", " ").replace("\n", " ")
    return plain if len(plain) <= n else plain[:n - 1] + "…"


def measure(dialog: Any, view: Any) -> dict[str, Any]:
    """Scroll, visual-area fraction and text problems of the dialog as shown."""
    from PySide6.QtWidgets import QAbstractButton, QLabel, QWidget
    from pyCamSet.gui.education_stage import _StagePill
    scroll = dialog._phase_scroll.verticalScrollBar()
    sv = stage_view_of(view)
    out: dict[str, Any] = {
        "scroll_max_px": int(scroll.maximum()),
        "needs_vertical_scroll": scroll.maximum() > 0,
        "dialog_size": [dialog.width(), dialog.height()],
    }
    if sv is not None:
        out["visual_height_px"] = sv.visual_panel.height()
        out["visual_fraction"] = round(sv.visual_panel.height() / max(dialog.height(), 1), 3)
        out["visual_fraction_of_stage"] = round(
            sv.visual_panel.height() / max(dialog._phase_scroll.viewport().height(), 1), 3)
        out["stage_min_height_px"] = sv.minimumSizeHint().height()
        out["caption_lines_reserved"] = getattr(sv, "_caption_lines", None)
        # Rendered caption lines at the narrowest supported width (1280 px, UX2-5); the
        # line limit per level is part of caption_issues.
        update = getattr(sv, "_update_caption_issues", None)
        if callable(update):
            update()
        out["caption_lines_at_1280"] = getattr(sv, "caption_lines_at_narrowest", None)
        # UX3-2: the limits hold at every supported width (1280 and 1400) and no caption
        # ends in a one-word line; also the lines as rendered at this shot's own width.
        by_width = dict(getattr(sv, "caption_lines_by_width", {}) or {})
        out["caption_lines_by_width"] = {str(k): v for k, v in sorted(by_width.items())}
        out["caption_last_lines"] = {str(k): v for k, v in sorted(
            (getattr(sv, "caption_last_lines", {}) or {}).items())}
        rendered = getattr(sv, "caption_lines_text", None)
        if callable(rendered) and sv.caption.text() and sv.caption.width() > 160:
            out["caption_lines_rendered"] = len(rendered(sv.caption.width()))
        out["caption_issues"] = list(getattr(sv, "caption_issues", []) or [])
        try:
            out.update(visual_content(sv))
        except Exception as exc:  # noqa: BLE001 - the metric is best effort
            out["visual_content_error"] = repr(exc)
    issues: list[dict[str, Any]] = []
    labels = [lab for lab in dialog.findChildren(QLabel)
              if lab.isVisible() and lab.text() and lab.width() > 0]
    for lab in labels:
        try:
            need_w, need_h, w, h = _label_needs(lab)
        except Exception as exc:  # noqa: BLE001
            issues.append({"widget": "QLabel", "name": lab.objectName(), "text": _short(lab.text()),
                           "problem": f"could not measure ({exc!r})"})
            continue
        kind = []
        if need_w > w + 1.5:
            kind.append("too narrow (elided/clipped horizontally)")
        if need_h > h + 1.5:
            kind.append("too short (clipped vertically)")
        if _clipped_by_ancestors(lab, dialog):
            kind.append("cut off by its container")
        if kind:
            issues.append({"widget": "QLabel", "name": lab.objectName() or lab.accessibleName(),
                           "text": _short(lab.text()), "problem": "; ".join(kind),
                           "need": [round(need_w), round(need_h)], "has": [w, h]})
    # overlap between visible siblings (label vs any visible sibling widget)
    seen = set()
    for lab in labels:
        parent = lab.parentWidget()
        if parent is None:
            continue
        for sib in parent.children():
            if (sib is lab or not isinstance(sib, QWidget) or not sib.isVisible()
                    or sib.isWindow()):
                continue
            key = tuple(sorted((id(lab), id(sib))))
            if key in seen:
                continue
            seen.add(key)
            inter = lab.geometry().intersected(sib.geometry())
            if inter.width() > 1 and inter.height() > 1:
                issues.append({"widget": "QLabel", "name": lab.objectName() or lab.accessibleName(),
                               "text": _short(lab.text()),
                               "problem": f"overlaps sibling {type(sib).__name__} "
                                          f"'{sib.objectName() or getattr(sib, 'accessibleName', lambda: '')()}'",
                               "overlap_px": [inter.width(), inter.height()]})
    for pill in dialog.findChildren(_StagePill):
        if pill.isVisible() and pill.width() + 1 < pill.sizeHint().width():
            issues.append({"widget": "StagePill", "name": pill.name, "text": pill.name,
                           "problem": "name elided/hidden (pill narrower than its text)",
                           "need": [pill.sizeHint().width(), pill.height()],
                           "has": [pill.width(), pill.height()]})
    for button in dialog.findChildren(QAbstractButton):
        if isinstance(button, _StagePill) or not button.isVisible() or not button.text():
            continue
        if button.width() + 1 < button.sizeHint().width():
            issues.append({"widget": type(button).__name__, "name": button.accessibleName(),
                           "text": button.text(),
                           "problem": "button narrower than its text (clipped)",
                           "need": [button.sizeHint().width(), button.sizeHint().height()],
                           "has": [button.width(), button.height()]})
    issues += combo_issues(dialog)
    out["text_issues"] = issues
    out["figure_text_issues"] = figure_text_issues(dialog)
    out["figure_cmap_issues"] = figure_cmap_issues(dialog)
    out["figure_elided"] = figure_elisions(dialog)
    return out


def combo_issues(dialog: Any) -> list[dict[str, Any]]:
    """Combo boxes whose shown text is elided or clipped (UX2-3: "Camera 8 · carr").

    A shared camera picker (``CameraCombo``) paints an elided form itself: reported as
    "elided" with what it shows. Any other combo is measured: its current text (plus icon)
    against the style's edit-field rectangle; wider = "clipped"."""
    from PySide6.QtWidgets import QComboBox, QStyle, QStyleOptionComboBox
    found: list[dict[str, Any]] = []
    for combo in dialog.findChildren(QComboBox):
        if not combo.isVisible() or combo.count() == 0:
            continue
        full = combo.currentText()
        name = combo.accessibleName() or combo.objectName()
        if callable(getattr(combo, "is_elided", None)):
            if combo.is_elided():
                found.append({"widget": "QComboBox", "name": name, "text": full,
                              "problem": f"elided: shows '{combo.displayText()}'",
                              "has": [combo.width(), combo.height()],
                              "need": [combo._chrome_width()
                                       + combo.fontMetrics().horizontalAdvance(full) + 6,
                                       combo.height()]})
            continue
        opt = QStyleOptionComboBox()
        combo.initStyleOption(opt)
        field = combo.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, opt,
                                             QStyle.SubControl.SC_ComboBoxEditField, combo)
        icon = combo.itemIcon(combo.currentIndex())
        icon_w = combo.iconSize().width() + 4 if not icon.isNull() else 0
        need = combo.fontMetrics().horizontalAdvance(full) + icon_w
        if need > field.width() + 1:
            found.append({"widget": "QComboBox", "name": name, "text": full,
                          "problem": "current text clipped (combo narrower than its text)",
                          "need": [need, combo.height()], "has": [field.width(),
                                                                  combo.height()]})
    return found


def figure_elisions(dialog: Any) -> list[dict[str, Any]]:
    """Panel titles and subtitles the layout engine shortened with "…" (UX2-23: an elided
    subtitle often loses the key fact). A warning: write them to fit."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    found: list[dict[str, Any]] = []
    for canvas in dialog.findChildren(FigureCanvasQTAgg):
        if not canvas.isVisible():
            continue
        for ax in canvas.figure.axes:
            for text in [ax.title, getattr(ax, "_left_title", None)] + list(ax.texts):
                if text is None or not getattr(text, "_edu_elided", False):
                    continue
                if not str(text.get_text()).endswith("…") and "…" not in str(text.get_text()):
                    continue
                full = getattr(text, "_edu_full", None) or getattr(text, "_edu_head", "")
                found.append({"canvas": canvas.accessibleName(), "shows":
                              " ".join(str(text.get_text()).split()), "full": full})
    for heading in dialog.findChildren(__import__("PySide6.QtWidgets", fromlist=["QWidget"])
                                       .QWidget, "eduPanelHeading"):
        if not heading.isVisible():
            continue
        title, sub = heading.lines_shown()
        for shown in (title, sub):
            if shown.endswith("…"):
                found.append({"canvas": "3D heading", "shows": shown,
                              "full": heading.toolTip()})
    return found


def figure_text_issues(dialog: Any) -> list[dict[str, Any]]:
    """Text problems *inside* every visible matplotlib figure (UX-1): titles, subtitles,
    tick labels, badges, legends and annotations clipped by the figure edge or overlapping
    each other (``education_style.text_collisions``), and text running out of the card it is
    written on (``education_style.card_overflows``, UX3-1: P4 gate cards)."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from pyCamSet.gui.education_style import card_overflows, text_collisions
    found: list[dict[str, Any]] = []
    for canvas in dialog.findChildren(FigureCanvasQTAgg):
        if not canvas.isVisible():
            continue
        name = canvas.accessibleName() or type(canvas.parentWidget()).__name__
        try:
            for issue in text_collisions(canvas.figure, draw=True):
                found.append({"canvas": name, **issue})
            for issue in card_overflows(canvas.figure, draw=False):
                found.append({"canvas": name, **issue})
        except Exception as exc:  # noqa: BLE001 - a broken figure is reported, not fatal
            found.append({"canvas": name, "problem": f"could not check ({exc!r})"})
    return found


def figure_cmap_issues(dialog: Any) -> list[dict[str, Any]]:
    """Panels that draw non-error data (counts, density, ties, shares) in the error colour
    map (``education_style.colour_map_misuse``, UX3-3)."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from pyCamSet.gui.education_style import colour_map_misuse
    found: list[dict[str, Any]] = []
    for canvas in dialog.findChildren(FigureCanvasQTAgg):
        if not canvas.isVisible():
            continue
        name = canvas.accessibleName() or type(canvas.parentWidget()).__name__
        try:
            found += [{"canvas": name, **issue} for issue in colour_map_misuse(canvas.figure)]
        except Exception as exc:  # noqa: BLE001
            found.append({"canvas": name, "problem": f"could not check ({exc!r})"})
    return found


#: Grid cell (pixels) of the drawn-content estimate and the blank-area flag.
CONTENT_CELL_PX = 16
#: Cells a drawn cell is grown by (2 x 16 px: gaps up to ~64 px inside a chart are chart).
CONTENT_GROW_CELLS = 2
#: A blank rectangle larger than this share of the visual area is flagged (UX3-5).
EMPTY_AREA_LIMIT = 0.15


def _largest_empty_rectangle(empty: Any) -> tuple[int, int, int, int, int]:
    """(area, row, col, height, width) of the largest all-True rectangle in a 2D bool grid
    (histogram / stack method)."""
    import numpy as np
    rows, cols = empty.shape
    heights = np.zeros(cols, dtype=int)
    best = (0, 0, 0, 0, 0)
    for r in range(rows):
        heights = np.where(empty[r], heights + 1, 0)
        stack: list[int] = []
        for c in range(cols + 1):
            h = heights[c] if c < cols else 0
            while stack and heights[stack[-1]] >= h:
                top = stack.pop()
                height = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                area = height * (c - left)
                if area > best[0]:
                    best = (area, r - height + 1, left, height, c - left)
            stack.append(c)
    return best


def content_grid(rgb: Any, background: Any, cell: int) -> Any:
    """Cells (``cell`` px) of an H x W x 3 image that hold any pixel off ``background``
    (RGB 0-255), grown by ``CONTENT_GROW_CELLS`` cells so the gaps inside a chart (between
    bars, inside the axes) count as the chart: an estimate of the drawn-content area."""
    import numpy as np
    h, w = rgb.shape[:2]
    ink = np.abs(rgb.astype(int) - np.asarray(background, dtype=float)).sum(axis=-1) > 30
    rows, cols = h // cell, w // cell
    grown = ink[:rows * cell, :cols * cell].reshape(rows, cell, cols, cell).any(axis=(1, 3))
    for _ in range(CONTENT_GROW_CELLS):
        occ = grown.copy()
        grown[1:] |= occ[:-1]
        grown[:-1] |= occ[1:]
        grown[:, 1:] |= occ[:, :-1]
        grown[:, :-1] |= occ[:, 1:]
    return grown


def visual_content(sv: Any) -> dict[str, Any]:
    """How much of the visual area is drawn on (UX3-5): the visual widget is grabbed and cut
    into :data:`CONTENT_CELL_PX` cells (:func:`content_grid`). Returns
    ``visual_content_fraction`` (drawn area / dialog area: ``visual_fraction`` without the
    blank figure area), ``visual_content_of_visual`` (drawn / visual area),
    ``empty_area_fraction`` (largest blank rectangle / visual area), ``empty_area_px``
    ([x, y, w, h] in dialog pixels) and ``empty_area`` (above :data:`EMPTY_AREA_LIMIT`)."""
    import numpy as np
    from PySide6.QtGui import QImage
    from pyCamSet.gui.education_style import _rgb, stage_tokens
    widget = sv.visual.currentWidget() or sv.visual
    image = widget.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    w, h = image.width(), image.height()
    cell = CONTENT_CELL_PX
    if w < cell * 2 or h < cell * 2:
        return {}
    raw = np.frombuffer(image.constBits(), np.uint8, image.sizeInBytes())
    rgb = raw.reshape(h, image.bytesPerLine())[:, :w * 4].reshape(h, w, 4)[..., 2::-1]
    background = np.array(_rgb(stage_tokens(sv.theme())["stage"])) * 255
    grown = content_grid(rgb, background, cell)
    rows, cols = grown.shape
    area, r, c, eh, ew = _largest_empty_rectangle(~grown)
    origin = widget.mapTo(widget.window(), widget.rect().topLeft())
    dialog_area = max(widget.window().width() * widget.window().height(), 1)
    empty_fraction = area / max(rows * cols, 1)
    return {"visual_content_fraction": round(float(grown.sum()) * cell * cell / dialog_area, 3),
            "visual_content_of_visual": round(float(grown.mean()), 3),
            "empty_area_fraction": round(float(empty_fraction), 3),
            "empty_area_px": [int(origin.x() + c * cell), int(origin.y() + r * cell),
                              int(ew * cell), int(eh * cell)],
            "empty_area": bool(empty_fraction > EMPTY_AREA_LIMIT)}


# --------------------------------------------------------------------------------------
# Step-throughs


Step = tuple[str, Callable[[], None]]


def _transport_steps(bar: Any) -> tuple[list[Step], Callable[[], None]]:
    """Positions of a foundation ``TransportBar``: all of them up to 10, else start /
    middle / end. Labels are the bar's own full sentences."""
    n = bar.count()
    positions = list(range(n)) if n <= 10 else [0, n // 2, n - 1]
    start = bar.position()
    steps = [(f"step {p + 1}/{n}: {bar.describe(p)}", lambda p=p: bar.go_to(p))
             for p in positions]
    return steps, lambda: bar.go_to(start)


def step_plan(phase: str, view: Any, sv: Any) -> tuple[list[Step], Callable[[], None] | None]:
    """The sub-steps of the current stage, and a reset to the stage's default position.

    Generic first: when the stage's controls row is a ``ScenePlayer`` or a foundation
    ``TransportBar`` (or a widget exposing one as ``.transport``), its positions are the
    steps. The per-phase branches below cover the round-1 controls until phases adopt it.
    """
    from pyCamSet.gui.education_dialog import RecordedStepControls, TransportBar
    from pyCamSet.gui.education_scene3d import ScenePlayer
    stage = sv.current_stage()
    controls = getattr(sv, "_controls", None)
    if phase == "phase3" and isinstance(controls, ScenePlayer):
        # Phase 3's solver does almost all its work in the first accepted step: show the
        # start, half-way through that step (trails growing, dots mid-slide), the step
        # itself, and the end.
        last = len(controls.frames) - 1
        marks = [("playback: start", 0.0)]
        if last >= 1:
            marks += [("playback: half-way through step 1 (interpolated)", 0.5),
                      ("playback: after step 1", 1.0)]
        if last > 1:
            marks.append((f"playback: end (position {last} of {last})", float(last)))
        def show(p: float) -> None:
            controls.go_to(int(p), animate=False)  # the transport bar follows (p >= 0)
            if p != int(p):
                # UX3-6: Phase 3 redraws its 2D panels on a fractional frame only when 50 ms
                # have passed since the last frame (a playback throttle that *drops* the
                # frame). go_to() above has just shown the whole position, so wait before
                # the half-way frame, as real playback does, or B/C and the title stay at
                # the start while only the 3D moves.
                from PySide6.QtWidgets import QApplication
                settle(QApplication.instance(), 80)
                controls.show_position(p)

        steps = [(label, lambda p=p: show(p)) for label, p in marks]
        return steps, lambda: controls.go_to(0, animate=False)
    if isinstance(controls, ScenePlayer):
        return _player_steps(controls)
    if isinstance(controls, TransportBar) and not isinstance(controls, RecordedStepControls):
        return _transport_steps(controls)
    bar = getattr(controls, "transport", None)
    if isinstance(bar, TransportBar):
        return _transport_steps(bar)
    if phase == "phase1":
        # Phase 1's detector steps use the shared TransportBar (view.transport), handled by
        # the generic branch above whenever it is the stage's controls row.
        return [], None
    if phase == "phase3":
        # The joins use the shared TransportBar (``view.transport``) and the solver a
        # ScenePlayer; both are handled above. A joins bar with more than 10 positions is cut
        # to start / middle / end by ``_transport_steps``.
        return [], None
    if phase == "phase4":
        # The shape morph is a TransportBar (handled above when it is the controls row):
        # printed (0), faces moved as plates (middle), refined (end).
        if stage == 1 and isinstance(getattr(view, "morph", None), TransportBar) \
                and view.morph.isVisible():
            return _transport_steps(view.morph)
        if stage == 2 and view.player is not None and controls is view.player:
            return _player_steps(view.player)
    return [], None


def _player_steps(player: Any) -> tuple[list[Step], Callable[[], None]]:
    last = len(player.frames) - 1
    positions = [("playback: start", 0)]
    if last >= 2:
        positions.append((f"playback: middle (position {last // 2} of {last})", last // 2))
    positions.append((f"playback: end (position {last} of {last})", last))
    steps = [(label, lambda p=p: player.go_to(p, animate=False)) for label, p in positions]
    return steps, lambda: player.go_to(0, animate=False)


# --------------------------------------------------------------------------------------
# PyVista renders


def pyvista_render(phase: str, view: Any, sv: Any, theme: str, path: Path,
                   reapply: Callable[[], None] | None) -> str | None:
    """Render the current 3D visual with a real PyVista OffscreenScene; returns a status."""
    from pyCamSet.gui.education_scene3d import OffscreenScene
    current = sv.visual.currentWidget()

    def window_for(widget: Any) -> tuple[int, int]:
        w, h = max(widget.width(), 200), max(widget.height(), 150)
        scale = 1200 / w
        return 1200, max(300, int(h * scale))

    if phase == "phase2":
        sw = getattr(view, "_scene_widget", None)
        if sw is None or current is not sw or sw._args is None:
            return None
        from pyCamSet.gui.education_phase2 import build_board_scene
        off = OffscreenScene(theme=theme, window_size=window_for(sw.scene))
        try:
            build_board_scene(off, *sw._args, theme)
            off.screenshot(path)
        finally:
            off.close()
        return "ok"
    if phase == "phase3":
        if view._split is None or current is not view._split or view._scene is None:
            return None
        holders = [(view, "_scene")] + ([(view._player, "scene")] if view._player else [])
    elif phase == "phase4":
        if current is not view.split:
            return None
        holders = [(view.split, "scene")] + ([(view.player, "scene")] if view.player else [])
    else:
        return None
    widget_scene = getattr(*holders[0])
    off = OffscreenScene(theme=theme, window_size=window_for(widget_scene))
    saved = [(obj, attr, getattr(obj, attr)) for obj, attr in holders]
    try:
        for obj, attr, _ in saved:
            setattr(obj, attr, off)
        view._show_stage(sv.current_stage())
        if reapply is not None:
            reapply()
        off.screenshot(path)
    finally:
        for obj, attr, value in saved:
            setattr(obj, attr, value)
        off.close()
        view._show_stage(sv.current_stage())
        if reapply is not None:
            reapply()
    return "ok"


# --------------------------------------------------------------------------------------
# One run


def render_run(run: Path, dataset: str, out: Path, size_spec: list[tuple[tuple[int, int],
               list[str]]], levels: list[int], stages: list[int] | None, pyvista: bool,
               log: Callable[[str], None] = print) -> dict[str, Any]:
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setFont(QFont("Segoe UI", 10))
    from pyCamSet.gui.education_dialog import EducationDialog, EducationRunView
    from pyCamSet.gui.theme import apply_theme

    import pyCamSet
    log(f"pyCamSet from {Path(pyCamSet.__file__).parent}")
    out.mkdir(parents=True, exist_ok=True)
    context = EducationRunView.from_run_dir(run)
    phase = context.phase
    pnum = phase.removeprefix("phase")
    record: dict[str, Any] = {"dataset": dataset, "run_dir": str(context.run_dir),
                              "run_id": context.run_id, "phase": phase,
                              "status": context.status, "shots": [], "pyvista": [],
                              "errors": []}
    pv_dir = out / "pyvista"
    for size, themes in size_spec:
        for theme in themes:
            apply_theme(app, theme)
            app.setFont(QFont("Segoe UI", 10))
            t0 = time.time()
            try:
                dialog = EducationDialog([context], [context.run_id])
            except Exception as exc:  # noqa: BLE001
                record["errors"].append({"theme": theme, "size": list(size),
                                         "where": "open dialog", "error": repr(exc),
                                         "trace": traceback.format_exc()})
                log(f"FAILED to open dialog: {exc!r}")
                continue
            view = dialog._view
            sv = stage_view_of(view)
            dialog.resize(*size)
            dialog.show()
            settle(app, 150)
            wait_ready(app, view)
            log(f"{dataset} {phase} {theme} {size[0]}x{size[1]}: opened in "
                f"{time.time() - t0:.1f}s")
            if sv is None:
                record["errors"].append({"theme": theme, "where": "no stage view",
                                         "error": type(view).__name__})
                dialog.close()
                continue
            names = [sv.stepper.stage_name(i) for i in range(sv.stepper.count())]
            record["stage_names"] = names
            wanted = [i for i in range(len(names)) if not stages or (i + 1) in stages]
            for index in wanted:
                for level in levels:
                    try:
                        sv.set_level(level)
                        sv.set_current_stage(index)
                        if sv.current_stage() != index or sv.level() != level:
                            raise RuntimeError("stage/level did not take")
                        wait_ready(app, view)
                        steps, reset = step_plan(phase, view, sv)
                        shots: list[tuple[int | None, str, Callable[[], None] | None]] = (
                            [(k + 1, label, action) for k, (label, action) in enumerate(steps)]
                            or [(None, "", None)])
                        for k, label, action in shots:
                            if action is not None:
                                action()
                                wait_ready(app, view)
                            draw_canvases(app, dialog)
                            name = (f"{dataset}_p{pnum}_s{index + 1}_L{level}_{theme}"
                                    + (f"_step{k}" if k is not None else "")
                                    + _size_suffix(size) + ".png")
                            if not dialog.grab().save(str(out / name)):
                                # e.g. a path over MAX_PATH: QImage.save only returns False
                                raise OSError(f"could not save {out / name}")
                            info = {"file": name, "stage": index + 1, "stage_name": names[index],
                                    "level": level, "theme": theme, "size": list(size),
                                    "step": k, "step_label": label}
                            info.update(measure(dialog, view))
                            if pyvista and tuple(size) == DEFAULT_SIZE:
                                pv_dir.mkdir(exist_ok=True)
                                pv_name = name.replace(".png", "_pyvista.png")
                                try:
                                    status = pyvista_render(phase, view, sv, theme,
                                                            pv_dir / pv_name, action)
                                    if status:
                                        draw_canvases(app, dialog)
                                        info["pyvista"] = f"pyvista/{pv_name}"
                                        record["pyvista"].append(f"pyvista/{pv_name}")
                                except Exception as exc:  # noqa: BLE001
                                    info["pyvista_error"] = repr(exc)
                                    record["errors"].append({
                                        "file": name, "where": "pyvista render",
                                        "error": repr(exc), "trace": traceback.format_exc()})
                            record["shots"].append(info)
                            flag = (" SCROLL" if info["needs_vertical_scroll"] else "") + (
                                f" text_issues={len(info['text_issues'])}"
                                if info["text_issues"] else "") + (
                                f" figure_text={len(info['figure_text_issues'])}"
                                if info["figure_text_issues"] else "") + (
                                f" elided={len(info['figure_elided'])}"
                                if info.get("figure_elided") else "") + (
                                " CAPTION" if info.get("caption_issues") else "")
                            log(f"  wrote {name} vis={info.get('visual_fraction')}{flag}")
                        if reset is not None:
                            reset()
                            wait_ready(app, view)
                    except Exception as exc:  # noqa: BLE001 - keep rendering the rest
                        record["errors"].append({"stage": index + 1, "level": level,
                                                 "theme": theme, "size": list(size),
                                                 "where": "render", "error": repr(exc),
                                                 "trace": traceback.format_exc()})
                        log(f"  FAILED s{index + 1} L{level} {theme}: {exc!r}")
            stop = getattr(view, "stop_for_context_change", None)
            if callable(stop):
                stop()
            dialog.close()
            dialog.deleteLater()
            settle(app, 50)
    index_path = out / f"index_{dataset}_p{pnum}.json"
    index_path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"index: {index_path} ({len(record['shots'])} shots, {len(record['pyvista'])} pyvista, "
        f"{len(record['errors'])} errors)")
    return record


# --------------------------------------------------------------------------------------
# Batch, contact sheets, README


def run_batch(plan_path: Path) -> int:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    root = Path(plan["root"])
    root.mkdir(parents=True, exist_ok=True)
    failures = 0
    for job in plan["jobs"]:
        run = Path(job["run"])
        phase = run.parent.name.removesuffix("_runs").removeprefix("phase")
        out = Path(job.get("out") or root / job["dataset"] / f"p{phase}")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--run", str(run),
               "--dataset", job["dataset"], "--out", str(out),
               "--levels", str(job.get("levels", "1,2,3"))]
        for spec in job.get("size_spec", ["1400x900=Light"]):
            cmd += ["--size-spec", spec]
        if job.get("pyvista"):
            cmd.append("--pyvista")
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / f"render_{job['dataset']}_p{phase}.log"
        print("job:", job["dataset"], f"p{phase}", run.name, "->", out, flush=True)
        t0 = time.time()
        with log_path.open("w", encoding="utf-8") as handle:
            flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
            code = subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                   creationflags=flags, cwd=str(ROOT))
        print(f"   exit {code} in {time.time() - t0:.0f}s (log {log_path})", flush=True)
        failures += code != 0
    summarise(root)
    return 1 if failures else 0


def _font(size: int):
    from PIL import ImageFont
    for name in ("segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(shots: list[dict], folder: Path, title: str, path: Path,
                  thumb_w: int = 300) -> None:
    """Rows = stage (+ step), columns = levels; thumbnails with labels; scroll/text flags."""
    from PIL import Image, ImageDraw
    levels = sorted({s["level"] for s in shots})
    rows: list[tuple[int, int | None]] = []
    for s in sorted(shots, key=lambda s: (s["stage"], s["step"] or 0)):
        key = (s["stage"], s["step"])
        if key not in rows:
            rows.append(key)
    by = {(s["stage"], s["step"], s["level"]): s for s in shots}
    first = Image.open(folder / shots[0]["file"])
    thumb_h = int(first.height * thumb_w / first.width)
    label_w, head_h, cap_h, pad = 230, 62, 34, 10
    width = label_w + len(levels) * (thumb_w + pad) + pad
    height = head_h + len(rows) * (thumb_h + cap_h + pad) + pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    big, small = _font(18), _font(12)
    draw.text((pad, 10), title, fill="black", font=big)
    for c, level in enumerate(levels):
        draw.text((label_w + c * (thumb_w + pad) + thumb_w // 2 - 10, head_h - 18), f"L{level}",
                  fill="black", font=small)
    for r, (stage, step) in enumerate(rows):
        y = head_h + r * (thumb_h + cap_h + pad)
        any_shot = next((by[(stage, step, lv)] for lv in levels if (stage, step, lv) in by), None)
        name = any_shot["stage_name"] if any_shot else ""
        draw.text((pad, y + 4), f"s{stage}  {name}", fill="black", font=small)
        if step is not None and any_shot:
            label = any_shot["step_label"]
            for i in range(0, len(label), 30):
                draw.text((pad, y + 22 + 16 * (i // 30)), label[i:i + 30], fill="#444",
                          font=small)
        for c, level in enumerate(levels):
            s = by.get((stage, step, level))
            x = label_w + c * (thumb_w + pad)
            if s is None:
                draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline="#bbb")
                draw.text((x + 8, y + 8), "missing", fill="red", font=small)
                continue
            img = Image.open(folder / s["file"]).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            sheet.paste(img, (x, y))
            flags = []
            if s.get("needs_vertical_scroll"):
                flags.append(f"SCROLL {s['scroll_max_px']}px")
            if s.get("text_issues"):
                flags.append(f"{len(s['text_issues'])} text issue(s)")
            if s.get("figure_text_issues"):
                flags.append(f"{len(s['figure_text_issues'])} fig text")
            if s.get("caption_issues"):
                flags.append("caption")
            if s.get("figure_elided"):
                flags.append(f"{len(s['figure_elided'])} elided")
            if s.get("figure_cmap_issues"):
                flags.append("error map misuse")
            if s.get("empty_area"):
                flags.append(f"blank {s.get('empty_area_fraction', 0):.0%}")
            vis = s.get("visual_fraction")
            caption = s["file"].removeprefix(s["file"].split("_s")[0] + "_")
            draw.text((x, y + thumb_h + 2), caption[:48], fill="black", font=small)
            content = s.get("visual_content_fraction")
            draw.text((x, y + thumb_h + 17),
                      f"visual {vis:.0%}" + (f" (drawn {content:.0%})" if content else "")
                      + ("  " + " · ".join(flags) if flags else ""),
                      fill="red" if flags else "#2a7", font=small)
    sheet.save(path)


def summarise(root: Path) -> dict[str, Any]:
    """Contact sheets per dataset/phase/theme(/size) and README.md for ``root``."""
    root = Path(root)
    sheets_dir = root / "contact_sheets"
    sheets_dir.mkdir(exist_ok=True)
    records = []
    for index_path in sorted(root.glob("*/p*/index_*.json")):
        record = json.loads(index_path.read_text(encoding="utf-8"))
        record["_folder"] = index_path.parent
        records.append(record)
    sheet_rows = []
    for rec in records:
        folder = rec["_folder"]
        pnum = rec["phase"].removeprefix("phase")
        groups: dict[tuple[str, tuple[int, int]], list[dict]] = {}
        for s in rec["shots"]:
            groups.setdefault((s["theme"], tuple(s["size"])), []).append(s)
        for (theme, size), shots in sorted(groups.items()):
            name = f"{rec['dataset']}_p{pnum}_{theme}{_size_suffix(size)}_sheet.png"
            contact_sheet(shots, folder,
                          f"{rec['dataset']}  phase {pnum}  run {rec['run_id']}  {theme}  "
                          f"{size[0]}x{size[1]}  ({len(shots)} shots)", sheets_dir / name)
            sheet_rows.append(name)
    _write_readme(root, records, sheet_rows)
    return {"records": len(records), "sheets": len(sheet_rows)}


def _write_readme(root: Path, records: list[dict], sheets: list[str]) -> None:
    lines = [f"# Education mode review set ({root.name})", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by "
             "`tools/edu_release/edu_screenshots.py` (real `EducationDialog`, offscreen Qt, "
             "Segoe UI 10, `PYCAMSET_EDU_3D=static` so 3D panels show the static fallback; "
             "real PyVista renders are in each folder's `pyvista/`).", "",
             "Names: `<dataset>_p<phase>_s<stage>_L<level>_<theme>[_step<k>][_<W>x<H>].png` "
             "(no size suffix = 1400x900). Step meanings are in each folder's "
             "`index_*.json` (`step_label`) and on the contact sheets.", "",
             "Each `index_*.json` records per screenshot: `needs_vertical_scroll` / "
             "`scroll_max_px`, `visual_fraction` (visual panel height / dialog height), "
             "`visual_fraction_of_stage` (/ scroll viewport height), `text_issues` "
             "(labels whose text does not fit, is cut off by its container or overlaps a "
             "sibling; stage pills and buttons narrower than their text), "
             "`figure_text_issues` (matplotlib text clipped by the figure edge or overlapping "
             "other text: titles, subtitles, ticks, badges, legends, annotations), "
             "`figure_elided` (panel titles/subtitles shortened with \"…\"; write them to "
             "fit), combo boxes whose shown text is elided or clipped (in `text_issues`), "
             "`caption_lines_at_1280`, `caption_lines_by_width` (1280 and 1400), "
             "`caption_lines_rendered` (at the shot's own width) and `caption_issues` (more "
             "sentences than the level allows, more rendered lines at 1280 or 1400 px than "
             "L1 2 / L2 3 / L3 4, or a one-word last line); `figure_text_issues` also lists "
             "text running out of its card (\"outside its card\"); `figure_cmap_issues` "
             "(non-error data drawn in the error colour map, viridis); "
             "`visual_content_fraction` (the drawn part of the visual area / dialog area: "
             "blank figure area excluded); `empty_area_fraction` / `empty_area_px` / "
             "`empty_area` (largest blank rectangle in the visual area; flagged above "
             "15 %). Best effort.", "",
             "## Start here", "",
             "`contact_sheets/` has one PNG per dataset / phase / theme (/ size): rows = stage "
             "(and sub-step), columns = L1-L3; red text under a thumbnail flags scrolling or "
             "text issues.", "",
             "## Folders", "",
             "| Folder | Run | Status | Shots | PyVista | Scroll | Text issues | "
             "Figure text | Elided | Captions | Max caption lines | Min visual | "
             "Min drawn | Blank areas | Error-map misuse | Errors |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    problems = []
    for rec in records:
        folder = rec["_folder"].relative_to(root).as_posix()
        shots = rec["shots"]
        scroll = [s for s in shots if s.get("needs_vertical_scroll")]
        texty = [s for s in shots if s.get("text_issues")]
        figtext = [s for s in shots if s.get("figure_text_issues")]
        captions = [s for s in shots if s.get("caption_issues")]
        elided = [s for s in shots if s.get("figure_elided")]
        most = max((max([s.get("caption_lines_at_1280") or 0]
                        + list((s.get("caption_lines_by_width") or {}).values()))
                    for s in shots), default=0)
        vis = min((s.get("visual_fraction") or 0 for s in shots), default=0)
        drawn = min((s.get("visual_content_fraction") or 0 for s in shots), default=0)
        blank = [s for s in shots if s.get("empty_area")]
        cmaps = [s for s in shots if s.get("figure_cmap_issues")]
        lines.append(f"| `{folder}/` | `{rec['run_id']}` | {rec.get('status')} | {len(shots)} | "
                     f"{len(rec['pyvista'])} | {len(scroll)} | {len(texty)} | {len(figtext)} | "
                     f"{len(elided)} | {len(captions)} | {most} | {vis:.0%} | {drawn:.0%} | "
                     f"{len(blank)} | {len(cmaps)} | {len(rec['errors'])} |")
        for s in cmaps:
            first = s["figure_cmap_issues"][0]
            problems.append(f"- `{folder}/{s['file']}`: error colour map on non-error data: "
                            f"{first.get('panel', '')} ({first.get('reason', '')})")
        for s in blank:
            problems.append(f"- `{folder}/{s['file']}`: blank area "
                            f"{s.get('empty_area_fraction', 0):.0%} of the visual at "
                            f"{s.get('empty_area_px')}")
        for s in texty:
            combos = [i for i in s["text_issues"] if i.get("widget") == "QComboBox"]
            if combos:
                problems.append(f"- `{folder}/{s['file']}`: combo box: {combos[0]['problem']} "
                                f"(\"{combos[0]['text']}\")")
        for s in elided:
            first = s["figure_elided"][0]
            problems.append(f"- `{folder}/{s['file']}`: {len(s['figure_elided'])} elided "
                            f"title(s), e.g. \"{first.get('shows')}\"")
        for s in figtext:
            first = s["figure_text_issues"][0]
            problems.append(f"- `{folder}/{s['file']}`: {len(s['figure_text_issues'])} figure "
                            f"text problem(s), e.g. {first.get('problem')} "
                            f"\"{first.get('text', '')}\" ({first.get('panel', '')})")
        for s in captions:
            problems.append(f"- `{folder}/{s['file']}`: caption: "
                            + "; ".join(s["caption_issues"]))
        for s in scroll:
            problems.append(f"- `{folder}/{s['file']}`: needs vertical scroll "
                            f"({s['scroll_max_px']} px)")
        for e in rec["errors"]:
            where = " ".join(str(e.get(k)) for k in ("file", "stage", "level", "theme")
                             if e.get(k) is not None)
            problems.append(f"- `{folder}` {where}: {e.get('where')}: {e.get('error')}")
    lines += ["", "## Datasets", "",
              "- `mouse1/` — mouse1 telecentric rig (Autoregulation poster), phases 1-4: Light, "
              "Dark, Sepia at 1400x900, plus Light at 1280x800.",
              "- `pcube/` — pcube PuzzleBoard, latest phase 1-4 runs: Light.",
              "- `ccube1a1/` — ccube1 ArUco-1 Phase 1 run: Light.",
              "- `pinhole_p2/` — pinhole example made by the Phase 2 agent "
              "(a scratch folder), phases 1-2: Light.",
              "- `pinhole_p3/` — pinhole example made by the Phase 3 agent "
              "(a scratch folder), phases 2-3: Light.", "",
              "## Problems found", ""]
    lines += problems or ["None."]
    notes = root / "notes.md"
    if notes.is_file():
        lines += ["", "## Reviewer notes", "", notes.read_text(encoding="utf-8").strip()]
    lines += ["", "## Contact sheets", ""] + [f"- `contact_sheets/{s}`" for s in sheets]
    lines += ["", "## Re-render", "",
              "```", "python tools/edu_release/edu_screenshots.py --batch "
              f"{(root / 'plan.json').as_posix()}", "```", ""]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, help="phaseN_runs/<run_id> directory")
    parser.add_argument("--dataset", default="run", help="name prefix, e.g. rig1")
    parser.add_argument("--out", type=Path, help="output folder")
    parser.add_argument("--themes", default="Light,Dark,Sepia",
                        help="themes for the default size when no --size-spec is given")
    parser.add_argument("--size-spec", action="append", default=[],
                        help="WxH=Theme1,Theme2 (repeatable); default 1400x900=<--themes>")
    parser.add_argument("--levels", default="1,2,3")
    parser.add_argument("--stages", default="", help="1-based stage numbers, default all")
    parser.add_argument("--pyvista", action="store_true",
                        help="also save real PyVista renders of 3D stages (1400x900 only)")
    parser.add_argument("--batch", type=Path, help="plan.json: run several jobs")
    parser.add_argument("--summarise", type=Path, help="build contact sheets + README for root")
    args = parser.parse_args(argv)
    _below_normal()
    if args.batch:
        return run_batch(args.batch)
    if args.summarise:
        print(summarise(args.summarise))
        return 0
    if not args.run or not args.out:
        parser.error("--run and --out are required (or --batch / --summarise)")
    _offscreen_env()
    sys.path.insert(0, str(ROOT))
    specs = args.size_spec or [f"{DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]}={args.themes}"]
    size_spec = []
    for spec in specs:
        size, _, themes = spec.partition("=")
        size_spec.append((_parse_size(size), [t for t in (themes or args.themes).split(",") if t]))
    record = render_run(args.run, args.dataset, args.out, size_spec,
                        [int(v) for v in args.levels.split(",")],
                        [int(v) for v in args.stages.split(",") if v] or None, args.pyvista,
                        log=lambda m: print(m, flush=True))
    return 1 if record["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
