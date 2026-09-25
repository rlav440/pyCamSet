"""Semantic application and Matplotlib themes for the PySide6 GUI.

The palettes and the stylesheet follow the lab's shared GUI design language,
first built for the optical-mapping analysis application: an indigo accent,
white card surfaces on a cool grey window, hairline card borders, tinted
hover states, underlined tabs and slim scrollbars.  Values that failed this
module's stricter contrast gate (disabled text, white text on green or amber
fills) were darkened rather than the gate relaxed.

Qt is imported only inside the functions that drive widgets, so the tokens
and the Matplotlib theming also serve processes without a GUI toolkit, such as
the out-of-process viewers on a lean install.

Three layers are applied in a fixed order by :func:`apply_theme`: the Fusion
widget style (Qt's flat, cross-platform style, which renders stylesheet
colours exactly as declared instead of through a native bitmap skin), then a
palette for whatever Fusion draws natively, then the stylesheet.
"""
from __future__ import annotations

from collections.abc import Mapping
from weakref import WeakSet


THEME_TOKENS: dict[str, dict[str, str]] = {
    "Light": {
        "background": "#f4f6fb",
        "surface": "#ffffff",
        "surface_alt": "#f7f8fc",
        # Decorative hairline for cards and separators; component edges use
        # border_strong, which is held to the 3:1 non-text floor.
        "border": "#e3e6ef",
        "border_strong": "#868a93",
        "text": "#1e2432",
        "text_muted": "#6b7280",
        "text_disabled": "#80858f",
        "accent": "#4f46e5",
        "accent_hover": "#4338ca",
        "accent_pressed": "#3730a3",
        "accent_tint": "#eef0ff",
        "accent_tint_strong": "#e0e4ff",
        "on_accent": "#ffffff",
        "success": "#15803d",
        "success_hover": "#166534",
        "success_pressed": "#14532d",
        "on_success": "#ffffff",
        "danger": "#dc2626",
        "on_danger": "#ffffff",
        "warning": "#b45309",
        "warning_tint": "#fdf1e2",
        "on_warning": "#ffffff",
        "selection": "#4f46e5",
        "selection_text": "#ffffff",
        "focus": "#4f46e5",
    },
    "Dark": {
        "background": "#1e1f26",
        "surface": "#2a2c36",
        "surface_alt": "#23252e",
        "border": "#3a3d4a",
        "border_strong": "#717890",
        "text": "#e8eaf2",
        "text_muted": "#a6aab8",
        "text_disabled": "#838899",
        "accent": "#818cf8",
        "accent_hover": "#93a0ff",
        "accent_pressed": "#b4bcff",
        "accent_tint": "#33364a",
        "accent_tint_strong": "#3b3f57",
        "on_accent": "#0b1220",
        "success": "#34d399",
        "success_hover": "#2bb98a",
        "success_pressed": "#1f9d74",
        "on_success": "#0b1220",
        "danger": "#f87171",
        "on_danger": "#0b1220",
        "warning": "#f59e0b",
        "warning_tint": "#3a3226",
        "on_warning": "#0b1220",
        "selection": "#818cf8",
        "selection_text": "#0b1220",
        "focus": "#a5b4fc",
    },
    "Sepia": {
        "background": "#f3e9d7",
        "surface": "#fbf5e8",
        "surface_alt": "#f5eddd",
        "border": "#e0d3b8",
        "border_strong": "#8f8063",
        "text": "#3e2f1d",
        "text_muted": "#6f5f46",
        "text_disabled": "#877860",
        "accent": "#8c5a2b",
        "accent_hover": "#7a4d24",
        "accent_pressed": "#6b421e",
        "accent_tint": "#f0e2cc",
        "accent_tint_strong": "#e8d6ba",
        "on_accent": "#ffffff",
        "success": "#4e6b31",
        "success_hover": "#425c29",
        "success_pressed": "#374d22",
        "on_success": "#ffffff",
        "danger": "#a63d2f",
        "on_danger": "#ffffff",
        "warning": "#94561a",
        "warning_tint": "#f3e0c4",
        "on_warning": "#ffffff",
        "selection": "#8c5a2b",
        "selection_text": "#ffffff",
        "focus": "#8c5a2b",
    },
}

#: Application font size in points.  Qt picks the platform's default sans
#: family (no proprietary family is requested by name), so only the size is set.
UI_FONT_PT = 9

#: Semantic text roles for labels, set with :func:`set_text_role`.  Each maps
#: to a token so a status message follows the live theme instead of carrying
#: a hard-coded hex value that turns unreadable on the other palettes.
TEXT_ROLES = ("muted", "hint", "success", "warning", "danger", "subheading", "mono")

_MANAGED_FIGURES = WeakSet()


def contrast_ratio(foreground: str, background: str) -> float:
    """Return WCAG relative-luminance contrast ratio for two sRGB colours."""
    def luminance(value: str) -> float:
        channels = [int(value[index:index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [channel / 12.92 if channel <= 0.04045
                  else ((channel + 0.055) / 1.055) ** 2.4 for channel in channels]
        return sum(weight * channel for weight, channel in zip((0.2126, 0.7152, 0.0722), linear))

    bright, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (bright + 0.05) / (dark + 0.05)


def validate_theme_tokens(themes: Mapping[str, Mapping[str, str]]) -> None:
    """Reject incomplete token sets and insufficient text/fill contrast."""
    names = set(next(iter(themes.values()))) if themes else set()
    if not themes or any(set(tokens) != names for tokens in themes.values()):
        raise ValueError("Every theme must define the same non-empty token key set")
    required = {"background", "surface", "surface_alt", "border", "border_strong",
                "text", "text_muted", "text_disabled", "accent", "accent_hover",
                "accent_pressed", "accent_tint", "accent_tint_strong", "on_accent",
                "success", "success_hover", "success_pressed", "on_success",
                "danger", "on_danger", "warning", "warning_tint", "on_warning",
                "selection", "selection_text", "focus"}
    if names != required:
        raise ValueError("Theme token keys differ from the required semantic schema")
    for theme_name, tokens in themes.items():
        for foreground, background in (("text", "background"), ("text", "surface"),
                                       ("text_muted", "surface"),
                                       ("on_accent", "accent"),
                                       ("on_accent", "accent_hover"),
                                       ("on_success", "success"),
                                       ("on_success", "success_hover"),
                                       ("on_danger", "danger"),
                                       ("on_warning", "warning"),
                                       ("selection_text", "selection"),
                                       # Hover/checked buttons and selected
                                       # tabs draw accent-family text on tints.
                                       ("accent_pressed", "accent_tint"),
                                       ("text", "accent_tint_strong"),
                                       ("warning", "warning_tint"),
                                       # Status labels draw these as text.
                                       ("success", "surface"),
                                       ("warning", "surface"),
                                       ("danger", "surface")):
            if contrast_ratio(tokens[foreground], tokens[background]) < 4.5:
                raise ValueError(f"{theme_name}: {foreground} contrast on {background} is below 4.5:1")
        # Disabled controls are exempt from WCAG 1.4.3, but 3:1 preserves a
        # discernible disabled label without competing with active body text.
        for background in ("background", "surface", "surface_alt"):
            if contrast_ratio(tokens["text_disabled"], tokens[background]) < 3.0:
                raise ValueError(
                    f"{theme_name}: text_disabled contrast on {background} is below 3:1"
                )
        # Component edges and the focus ring are non-text UI: 3:1 (WCAG 1.4.11).
        for edge in ("border_strong", "focus", "accent"):
            if contrast_ratio(tokens[edge], tokens["surface"]) < 3.0:
                raise ValueError(f"{theme_name}: {edge} contrast is below 3:1")


#: Bump when the drawing below changes, so cached images from an older
#: release are not reused under the same colour-based file name.
_INDICATOR_ASSET_VERSION = 2


def _paint_indicator(kind: str, colour: str):
    """Paint one indicator image (chevron, tick or dot) at 4x display size."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen

    image = QImage(36, 36, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor(colour), 5.0, Qt.PenStyle.SolidLine,
               Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    if kind in ("up", "down"):
        painter.setPen(pen)
        ends, tip = (26.0, 11.0) if kind == "up" else (11.0, 26.0)
        painter.drawPolyline([QPointF(5.0, ends), QPointF(18.0, tip), QPointF(31.0, ends)])
    elif kind == "tick":
        pen.setWidthF(5.5)
        painter.setPen(pen)
        painter.drawPolyline([QPointF(8.0, 19.0), QPointF(15.0, 26.0), QPointF(28.0, 10.0)])
    elif kind == "dot":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(colour)))
        painter.drawEllipse(QPointF(18.0, 18.0), 7.5, 7.5)
    painter.end()
    return image


def _indicator_images(tokens: Mapping[str, str]) -> dict[str, str]:
    """Draw the arrow, tick and dot images for a palette; return their paths.

    Styling a combo, spin box or check box's frame in QSS stops Fusion drawing
    its arrows and check marks, and QSS takes those shapes only from image
    files.  Small antialiased PNGs are therefore painted once per colour into
    the per-user cache directory and reused.  An unwritable cache falls back
    to the temporary directory; if that fails too, the missing images are left
    out and the stylesheet omits their rules rather than breaking start-up.
    Paths use forward slashes, which QSS ``url()`` accepts on every platform.
    """
    import tempfile
    from pathlib import Path
    from PySide6.QtCore import QStandardPaths

    wanted = {
        "down": ("down", tokens["text_muted"]),
        "up": ("up", tokens["text_muted"]),
        "down_disabled": ("down", tokens["text_disabled"]),
        "up_disabled": ("up", tokens["text_disabled"]),
        "tick": ("tick", tokens["on_accent"]),
        "dot": ("dot", tokens["on_accent"]),
    }
    cache = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericCacheLocation)
    candidates = [Path(cache) / "pyCamSet" / "gui-indicators"] if cache else []
    candidates.append(Path(tempfile.gettempdir()) / "pyCamSet-gui-indicators")
    for folder in candidates:
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        paths = {}
        for name, (kind, colour) in wanted.items():
            path = folder / f"v{_INDICATOR_ASSET_VERSION}-{kind}-{colour.lstrip('#').lower()}.png"
            if path.is_file() or _paint_indicator(kind, colour).save(str(path), "PNG"):
                paths[name] = path.as_posix()
        if len(paths) == len(wanted):
            return paths
    return {}


def _indicator_rules(t: Mapping[str, str]) -> str:
    """QSS for combo/spin arrows and check/radio marks, from painted images."""
    images = _indicator_images(t)
    if not images:
        return ""
    return f"""
        /* Styling these frames stops Fusion drawing the arrows and check
           marks, so themed images are supplied (see _indicator_images):
           a combo without an arrow does not read as a choice, and a filled
           square without a tick does not read as "checked". */
        QComboBox::down-arrow {{ image: url("{images['down']}"); width: 9px; height: 9px; }}
        QComboBox::down-arrow:disabled {{ image: url("{images['down_disabled']}"); }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{images['up']}");
            width: 7px; height: 7px; }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{images['down']}");
            width: 7px; height: 7px; }}
        QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{
            image: url("{images['up_disabled']}"); }}
        QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
            image: url("{images['down_disabled']}"); }}
        QToolButton#colourPickerButton::menu-indicator {{ image: url("{images['down']}");
            subcontrol-origin: padding; subcontrol-position: right center;
            width: 9px; height: 9px; right: 8px; }}
        QCheckBox::indicator:checked {{ image: url("{images['tick']}"); }}
        QRadioButton::indicator:checked {{ image: url("{images['dot']}"); }}
    """


def _stylesheet(tokens: Mapping[str, str]) -> str:
    """Build Qt chrome styles from semantic tokens, including visible states.

    There is deliberately no bare ``QWidget {{ background-color }}`` rule: it
    would paint every container opaque and flatten the card surfaces (tab
    pane, group boxes, collapsible sections) into the window background.
    Top-level windows take their background from the palette instead.
    """
    t = tokens
    return _indicator_rules(t) + f"""
        QMainWindow, QDialog {{ background-color: {t['background']}; }}
        QWidget {{ color: {t['text']}; }}
        QToolTip {{ background-color: {t['text']}; color: {t['surface']};
                    border: 1px solid {t['text']}; border-radius: 4px; padding: 4px 8px; }}
        QLabel {{ background: transparent; }}
        QWidget:disabled {{ color: {t['text_disabled']}; }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
        QDoubleSpinBox:disabled, QTextEdit:disabled, QListWidget:disabled,
        QCheckBox:disabled, QRadioButton:disabled, QLabel:disabled,
        QPushButton:disabled {{ color: {t['text_disabled']}; }}

        /* ---- Cards ---------------------------------------------------- */
        QGroupBox {{ background-color: {t['surface']}; border: 1px solid {t['border']};
                     border-radius: 8px; margin-top: 14px; padding: 10px 6px 6px 6px;
                     font-weight: 600; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
                            color: {t['text']}; }}
        QGroupBox QGroupBox {{ background-color: {t['surface_alt']}; margin-top: 10px; }}
        QFrame#hairline {{ background-color: {t['border']}; border: none;
                           min-height: 1px; max-height: 1px; }}
        QFrame#collapsibleSection {{ background-color: {t['surface_alt']};
                                     border: 1px solid {t['border']}; border-radius: 8px; }}
        QPushButton#sectionToggle {{ text-align: left; font-weight: 600; color: {t['text']};
                                     background: transparent; border: none;
                                     border-radius: 6px; padding: 5px 8px; min-height: 0; }}
        QPushButton#sectionToggle:hover {{ color: {t['accent']}; background: {t['accent_tint']}; }}
        QPushButton#sectionToggle:focus {{ border: 1px solid {t['focus']}; }}

        /* ---- Buttons -------------------------------------------------- */
        QPushButton {{ background-color: {t['surface']}; color: {t['text']};
                       border: 1px solid {t['border_strong']}; border-radius: 6px;
                       padding: 5px 12px; min-height: 18px; }}
        QPushButton:hover {{ background-color: {t['accent_tint']}; border-color: {t['accent']};
                             color: {t['accent_pressed']}; }}
        QPushButton:pressed {{ background-color: {t['accent_tint_strong']};
                               border-color: {t['accent_hover']}; color: {t['text']}; }}
        QPushButton:checked {{ background-color: {t['accent_tint']}; border-color: {t['accent']}; }}
        QPushButton:disabled {{ background-color: {t['surface_alt']};
                                color: {t['text_disabled']}; border-color: {t['border']}; }}
        QPushButton:focus {{ outline: none; border: 1.5px solid {t['focus']}; }}
        QPushButton[designRole="primary"] {{ background-color: {t['accent']};
            color: {t['on_accent']}; border: 1px solid {t['accent']}; font-weight: 600; }}
        QPushButton[designRole="primary"]:hover {{ background-color: {t['accent_hover']};
            border-color: {t['accent_hover']}; color: {t['on_accent']}; }}
        QPushButton[designRole="primary"]:pressed {{ background-color: {t['accent_pressed']};
            border-color: {t['accent_pressed']}; color: {t['on_accent']}; }}
        QPushButton[designRole="success"] {{ background-color: {t['success']};
            color: {t['on_success']}; border: 1px solid {t['success']}; font-weight: 600; }}
        QPushButton[designRole="success"]:hover {{ background-color: {t['success_hover']};
            border-color: {t['success_hover']}; color: {t['on_success']}; }}
        QPushButton[designRole="success"]:pressed {{ background-color: {t['success_pressed']};
            border-color: {t['success_pressed']}; color: {t['on_success']}; }}
        QPushButton[designRole="danger"] {{ background-color: {t['danger']};
            color: {t['on_danger']}; border: 1px solid {t['danger']}; font-weight: 600; }}
        /* Warning is a tinted outline, not a fill: it marks navigation to
           diagnostics and runs that should not be carried forward, and must
           read as "look here" without competing with the primary action. */
        QPushButton[designRole="warning"] {{ background-color: {t['warning_tint']};
            color: {t['warning']}; border: 1px solid {t['warning']}; font-weight: 600; }}
        QPushButton[designRole="warning"]:hover {{ background-color: {t['warning']};
            color: {t['on_warning']}; border-color: {t['warning']}; }}
        QPushButton[designRole="primary"]:disabled, QPushButton[designRole="success"]:disabled,
        QPushButton[designRole="danger"]:disabled, QPushButton[designRole="warning"]:disabled {{
            background-color: {t['surface_alt']}; color: {t['text_disabled']};
            border-color: {t['border']}; }}
        QPushButton[designRole="primary"]:focus, QPushButton[designRole="success"]:focus,
        QPushButton[designRole="danger"]:focus, QPushButton[designRole="warning"]:focus {{
            border: 2px solid {t['focus']}; }}
        /* Compact glyph buttons (options, snapshot, CSV) carry their own
           fixed size; zero padding keeps the glyph from being clipped. */
        QPushButton[designRole="icon"] {{ padding: 0; min-height: 0; border-radius: 6px; }}

        /* ---- Tabs ----------------------------------------------------- */
        QTabWidget::pane {{ background-color: {t['surface']}; border: 1px solid {t['border']};
                            border-radius: 8px; top: -1px; }}
        QTabBar::tab {{ background: transparent; color: {t['text_muted']};
                        padding: 6px 16px; margin-right: 2px;
                        border-top-left-radius: 6px; border-top-right-radius: 6px;
                        border: 1px solid transparent; }}
        QTabBar::tab:selected {{ background-color: {t['surface']}; color: {t['accent']};
                                 font-weight: 600; border: 1px solid {t['border']};
                                 border-bottom: 2px solid {t['accent']}; }}
        QTabBar::tab:!selected:hover {{ background-color: {t['accent_tint']}; color: {t['text']}; }}
        QTabBar::tab:focus {{ color: {t['accent_pressed']}; border: 1px solid {t['focus']}; }}
        QTabBar::tab:selected:focus {{ border: 1px solid {t['focus']};
                                       border-bottom: 2px solid {t['accent']}; }}
        QTabWidget QTabWidget QTabBar::tab {{ padding: 4px 12px; }}
        /* Overflow: visible scroll arrows and the corner "All tabs" menu. */
        QTabBar QToolButton {{ background-color: {t['surface']}; border: 1px solid {t['border_strong']};
                               border-radius: 4px; margin: 2px 1px; }}
        QTabBar QToolButton:hover {{ background-color: {t['accent_tint']}; border-color: {t['accent']}; }}
        /* Colour picker: an input-like swatch button. */
        QToolButton#colourPickerButton {{ background-color: {t['surface']};
            border: 1px solid {t['border_strong']}; border-radius: 6px;
            padding: 3px 26px 3px 6px; text-align: left; }}
        QToolButton#colourPickerButton:hover {{ border-color: {t['accent']}; }}
        QToolButton#colourPickerButton:focus {{ border: 1.5px solid {t['focus']}; }}
        QToolButton#colourPickerButton:disabled {{ background-color: {t['surface_alt']};
            border-color: {t['border']}; color: {t['text_disabled']}; }}
        QToolButton#colourSwatch {{ border: 1px solid transparent; border-radius: 4px; padding: 2px; }}
        QToolButton#colourSwatch:hover, QToolButton#colourSwatch:focus {{
            border-color: {t['accent']}; background-color: {t['accent_tint']}; }}
        QToolButton#allTabsButton {{ background-color: transparent; color: {t['text_muted']};
            border: 1px solid transparent; border-radius: 6px; padding: 4px 8px; margin: 0 0 3px 6px; }}
        QToolButton#allTabsButton:hover {{ background-color: {t['accent_tint']}; color: {t['accent_pressed']};
            border-color: {t['accent']}; }}
        QToolButton#allTabsButton:focus {{ border-color: {t['focus']}; }}
        QToolButton#allTabsButton::menu-indicator {{ image: none; width: 0; }}

        /* ---- Scroll areas / scrollbars -------------------------------- */
        QScrollArea {{ border: none; background: transparent; }}
        QScrollArea > QWidget > QWidget {{ background: transparent; }}
        QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
        QScrollBar::handle:vertical {{ background: {t['border_strong']}; border-radius: 3px;
                                       min-height: 24px; }}
        QScrollBar::handle:vertical:hover {{ background: {t['text_muted']}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; border: none; }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
        QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
        QScrollBar::handle:horizontal {{ background: {t['border_strong']}; border-radius: 3px;
                                         min-width: 24px; }}
        QScrollBar::handle:horizontal:hover {{ background: {t['text_muted']}; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; border: none; }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}

        /* ---- Inputs --------------------------------------------------- */
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit,
        QListWidget, QTreeWidget {{
            background-color: {t['surface']}; color: {t['text']};
            border: 1px solid {t['border_strong']}; border-radius: 6px; padding: 3px 6px;
            selection-background-color: {t['selection']}; selection-color: {t['selection_text']};
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
        QTextEdit:focus, QPlainTextEdit:focus, QListWidget:focus, QTreeWidget:focus {{
            border: 1.5px solid {t['focus']}; }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
            background-color: {t['surface_alt']}; border-color: {t['border']}; }}
        QLineEdit:read-only {{ background-color: {t['surface_alt']}; color: {t['text_muted']}; }}
        QComboBox::drop-down {{ border: none; width: 20px; }}
        QComboBox QAbstractItemView {{ background-color: {t['surface']}; color: {t['text']};
            border: 1px solid {t['border']}; border-radius: 6px;
            selection-background-color: {t['accent_tint']}; selection-color: {t['text']};
            outline: none; }}
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            width: 16px; border: none; background: transparent; }}
        QTextEdit#terminalOutput {{ background-color: {t['surface_alt']}; color: {t['text']};
            border: 1px solid {t['border']}; border-radius: 8px; padding: 6px;
            font-family: "Consolas", "Menlo", "DejaVu Sans Mono", monospace; }}

        /* ---- Checkboxes / radio buttons ------------------------------- */
        QCheckBox, QRadioButton {{ spacing: 8px; background: transparent; }}
        QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px;
            border: 1.5px solid {t['border_strong']}; background: {t['surface']}; }}
        QCheckBox::indicator {{ border-radius: 4px; }}
        QRadioButton::indicator {{ border-radius: 8px; }}
        QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {t['accent']}; }}
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
            background-color: {t['accent']}; border-color: {t['accent']}; }}
        QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
            background-color: {t['surface_alt']}; border-color: {t['border']}; }}
        QCheckBox:focus, QRadioButton:focus {{ color: {t['accent_pressed']}; }}

        /* ---- Sliders -------------------------------------------------- */
        QSlider::groove:horizontal {{ height: 4px; background: {t['border']}; border-radius: 2px; }}
        QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
        QSlider::add-page:horizontal {{ background: {t['border']}; border-radius: 2px; }}
        QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -6px 0;
            border-radius: 7px; background: {t['accent']}; border: 2px solid {t['surface']}; }}
        QSlider::handle:horizontal:hover {{ background: {t['accent_hover']}; }}
        QSlider::handle:horizontal:disabled {{ background: {t['text_disabled']}; }}

        /* ---- Menus ---------------------------------------------------- */
        QMenuBar {{ background-color: {t['surface']}; border-bottom: 1px solid {t['border']};
                    padding: 2px; }}
        QMenuBar::item {{ padding: 4px 10px; border-radius: 4px; background: transparent; }}
        QMenuBar::item:selected {{ background-color: {t['accent_tint']}; color: {t['accent_pressed']}; }}
        QMenu {{ background-color: {t['surface']}; color: {t['text']};
                 border: 1px solid {t['border']}; border-radius: 6px; padding: 4px; }}
        QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 4px; }}
        QMenu::item:selected {{ background-color: {t['accent_tint']}; color: {t['accent_pressed']}; }}
        QMenu::item:disabled {{ color: {t['text_disabled']}; }}
        QMenu::separator {{ height: 1px; background: {t['border']}; margin: 4px 8px; }}
        QMenu QCheckBox, QMenu QComboBox {{ margin: 3px 8px; }}

        /* ---- Splitter / progress / tables / status bar ---------------- */
        QSplitter::handle {{ background-color: {t['background']}; }}
        QSplitter::handle:hover {{ background-color: {t['accent_tint']}; }}
        QSplitter::handle:vertical {{ height: 6px; }}
        QSplitter::handle:horizontal {{ width: 6px; }}
        QProgressBar {{ background-color: {t['surface_alt']}; border: 1px solid {t['border']};
                        border-radius: 6px; text-align: center; }}
        QProgressBar::chunk {{ background-color: {t['accent']}; border-radius: 5px; }}
        QTableWidget, QTableView {{ background-color: {t['surface']};
            alternate-background-color: {t['surface_alt']}; gridline-color: {t['border']};
            border: 1px solid {t['border']}; border-radius: 6px; }}
        QTableWidget::item:selected, QTableView::item:selected, QListWidget::item:selected,
        QTreeWidget::item:selected {{ background-color: {t['accent_tint']}; color: {t['text']}; }}
        QHeaderView::section {{ background-color: {t['surface_alt']}; color: {t['text']};
            padding: 5px; border: none; border-bottom: 1px solid {t['border']};
            border-right: 1px solid {t['border']}; font-weight: 600; }}
        QStatusBar {{ color: {t['text_muted']}; }}

        /* ---- Text roles (set_text_role) ------------------------------- */
        QLabel[designRole="section"] {{ color: {t['text']}; font-weight: 600; font-size: 10pt; }}
        QLabel[textRole="muted"] {{ color: {t['text_muted']}; }}
        QLabel[textRole="hint"] {{ color: {t['text_muted']}; font-size: 8pt; }}
        QLabel[textRole="success"] {{ color: {t['success']}; }}
        QLabel[textRole="warning"] {{ color: {t['warning']}; font-weight: 600; }}
        QLabel[textRole="danger"] {{ color: {t['danger']}; font-weight: 600; }}
        QLabel[textRole="subheading"] {{ font-weight: 600; margin-top: 8px; }}
        QLabel#viewportPlaceholder {{ background-color: {t['surface_alt']}; color: {t['text_muted']};
            border: 1px dashed {t['border_strong']}; border-radius: 8px; padding: 12px; }}
        QLabel[textRole="mono"] {{ font-family: "Consolas", "Menlo", "DejaVu Sans Mono", monospace; }}
    """


def theme_tokens(theme_name: str | None = None) -> Mapping[str, str]:
    """Return the tokens of *theme_name*, or of the live application theme."""
    if theme_name is None:
        from PySide6.QtWidgets import QApplication
        application = QApplication.instance()
        theme_name = (application.property("pycamsetTheme") if application else None) or "Light"
    return THEME_TOKENS[theme_name]


def set_text_role(label, role: str | None) -> None:
    """Give *label* a themed text role, re-polishing so the change shows now.

    Use this instead of an inline ``setStyleSheet("color: ...")``: the role
    is resolved by the application stylesheet, so it follows live theme
    switches.  ``None`` clears the role.
    """
    if role is not None and role not in TEXT_ROLES:
        raise ValueError(f"Unknown text role: {role!r}")
    label.setProperty("textRole", role)
    style = label.style()
    style.unpolish(label)
    style.polish(label)
    label.update()


def apply_theme(application, theme_name: str) -> None:
    """Apply the named theme to a QApplication without changing application data."""
    from PySide6.QtGui import QColor, QFont, QPalette

    validate_theme_tokens(THEME_TOKENS)
    if theme_name not in THEME_TOKENS:
        raise ValueError(f"Unknown theme: {theme_name}")
    tokens = THEME_TOKENS[theme_name]
    # Fusion first, then palette, then stylesheet: each layer only overrides
    # what the previous one left unset.
    if application.style().name().casefold() != "fusion":
        application.setStyle("Fusion")
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: "background",
        QPalette.ColorRole.Base: "surface",
        QPalette.ColorRole.AlternateBase: "surface_alt",
        QPalette.ColorRole.WindowText: "text",
        QPalette.ColorRole.Text: "text",
        QPalette.ColorRole.Button: "surface",
        QPalette.ColorRole.ButtonText: "text",
        QPalette.ColorRole.ToolTipBase: "text",
        QPalette.ColorRole.ToolTipText: "surface",
        QPalette.ColorRole.Link: "accent",
        QPalette.ColorRole.Highlight: "selection",
        QPalette.ColorRole.HighlightedText: "selection_text",
        QPalette.ColorRole.PlaceholderText: "text_muted",
        QPalette.ColorRole.Mid: "border",
        QPalette.ColorRole.Dark: "border_strong",
    }
    for role, token in roles.items():
        palette.setColor(role, QColor(tokens[token]))
    disabled = QPalette.ColorGroup.Disabled
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        palette.setColor(disabled, role, QColor(tokens["text_disabled"]))
    # Each call below re-polishes every live widget, so a layer that would
    # not change is skipped: re-applying the current theme (every new main
    # window does) then costs nothing, however many widgets exist.
    if application.palette() != palette:
        application.setPalette(palette)
    if application.font().pointSize() != UI_FONT_PT:
        font = QFont(application.font())
        font.setPointSize(UI_FONT_PT)
        application.setFont(font)
    sheet = _stylesheet(tokens)
    if application.styleSheet() != sheet:
        application.setStyleSheet(sheet)
    application.setProperty("pycamsetTheme", theme_name)
    # The native title bar follows too: exact colours on Windows 11,
    # light/dark elsewhere (see window_chrome).
    from pyCamSet.gui.window_chrome import refresh_window_chrome
    refresh_window_chrome(application, theme_name)


def apply_matplotlib_theme(figure, theme_name: str | None = None) -> None:
    """Theme Matplotlib chrome only; retain all data and scientific colours.

    Figures sit on the card surface (the tab pane), so the figure face uses
    ``surface`` and blends into the card instead of showing a window-grey slab.
    """
    theme_name = theme_name or "Light"
    if theme_name not in THEME_TOKENS:
        raise ValueError(f"Unknown theme: {theme_name}")
    tokens = THEME_TOKENS[theme_name]
    _MANAGED_FIGURES.add(figure)
    figure.patch.set_facecolor(tokens["surface"])
    for text in figure.texts:
        text.set_color(tokens["text"])
    for axes in figure.axes:
        axes.set_facecolor(tokens["surface"])
        # Only explicitly tagged presentation labels are themed; arbitrary
        # Axes.text annotations may encode scientific meaning or data.
        for text in axes.texts:
            if text.get_gid() == "phase1:unreadable-placeholder":
                text.set_color(tokens["text"])
        axes.title.set_color(tokens["text"])
        axes.xaxis.label.set_color(tokens["text"])
        axes.yaxis.label.set_color(tokens["text"])
        axes.tick_params(axis="both", colors=tokens["text_muted"])
        # Grid lines are existing neutral chrome; recolour them without
        # enabling grids or touching data-series line/marker encodings.
        for axis in (axes.xaxis, axes.yaxis):
            for gridline in axis.get_gridlines():
                gridline.set_color(tokens["border"])
        for spine in axes.spines.values():
            spine.set_color(tokens["border_strong"])
        legend = axes.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(tokens["surface"])
            legend.get_frame().set_edgecolor(tokens["border"])
            for label in legend.get_texts():
                label.set_color(tokens["text"])
    canvas = getattr(figure, "canvas", None)
    if canvas is not None:
        canvas.draw_idle()


def refresh_matplotlib_theme(theme_name: str) -> None:
    """Re-theme registered GUI figures after a live application theme switch."""
    for figure in tuple(_MANAGED_FIGURES):
        apply_matplotlib_theme(figure, theme_name)
        # Per-visual explicit overrides outrank the newly selected theme;
        # importing lazily keeps the theme module independent at startup.
        from pyCamSet.gui.visual_style import refresh_visual_style
        refresh_visual_style(figure, theme_name)


validate_theme_tokens(THEME_TOKENS)
