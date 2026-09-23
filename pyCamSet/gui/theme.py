"""Semantic application and Matplotlib themes for the PySide6 GUI."""
from __future__ import annotations

from collections.abc import Mapping
from weakref import WeakSet

from PySide6.QtGui import QColor, QPalette

THEME_TOKENS: dict[str, dict[str, str]] = {
    "Light": {
        "background": "#f5f7fa",
        "surface": "#ffffff",
        "surface_alt": "#e9eef5",
        "border": "#8b98a8",
        "border_strong": "#596779",
        "text": "#17212b",
        "text_muted": "#46576a",
        "text_disabled": "#596779",
        "accent": "#145ea8",
        "accent_hover": "#0e4c8b",
        "accent_pressed": "#0a3b70",
        "on_accent": "#ffffff",
        "success": "#176b3a",
        "on_success": "#ffffff",
        "danger": "#a52a2a",
        "on_danger": "#ffffff",
        "warning": "#805000",
        "on_warning": "#ffffff",
        "selection": "#145ea8",
        "selection_text": "#ffffff",
        "focus": "#8a3ffc",
    },
    "Dark": {
        "background": "#171b21",
        "surface": "#222831",
        "surface_alt": "#303946",
        "border": "#687587",
        "border_strong": "#aab6c5",
        "text": "#f1f4f8",
        "text_muted": "#c2ccd8",
        "text_disabled": "#aab6c5",
        "accent": "#76b7ff",
        "accent_hover": "#99caff",
        "accent_pressed": "#b8d9ff",
        "on_accent": "#10243a",
        "success": "#53c98a",
        "on_success": "#10271b",
        "danger": "#ff8585",
        "on_danger": "#351313",
        "warning": "#ffd17a",
        "on_warning": "#35260a",
        "selection": "#315e8c",
        "selection_text": "#ffffff",
        "focus": "#d6a8ff",
    },
    "Sepia": {
        "background": "#f3ecdf",
        "surface": "#fffaf0",
        "surface_alt": "#e9ddc8",
        "border": "#9a876b",
        "border_strong": "#6d5940",
        "text": "#30271e",
        "text_muted": "#594832",
        "text_disabled": "#6d5940",
        "accent": "#70451f",
        "accent_hover": "#5b3719",
        "accent_pressed": "#472a13",
        "on_accent": "#fffaf0",
        "success": "#315f3d",
        "on_success": "#fffaf0",
        "danger": "#963b30",
        "on_danger": "#fffaf0",
        "warning": "#87470f",
        "on_warning": "#fffaf0",
        "selection": "#70451f",
        "selection_text": "#fffaf0",
        "focus": "#704a91",
    },
}

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
                "accent_pressed", "on_accent", "success", "on_success", "danger", "on_danger", "warning",
                "on_warning", "selection", "selection_text", "focus"}
    if names != required:
        raise ValueError("Theme token keys differ from the required semantic schema")
    for theme_name, tokens in themes.items():
        for foreground, background in (("text", "background"), ("text", "surface"),
                                       ("text_muted", "surface"),
                                       ("on_accent", "accent"),
                                       ("on_success", "success"),
                                       ("on_danger", "danger"),
                                       ("on_warning", "warning"),
                                       ("selection_text", "selection")):
            if contrast_ratio(tokens[foreground], tokens[background]) < 4.5:
                raise ValueError(f"{theme_name}: {foreground} contrast on {background} is below 4.5:1")
        # Disabled controls are exempt from WCAG 1.4.3, but 3:1 preserves a
        # discernible disabled label without competing with active body text.
        for background in ("background", "surface", "surface_alt"):
            if contrast_ratio(tokens["text_disabled"], tokens[background]) < 3.0:
                raise ValueError(
                    f"{theme_name}: text_disabled contrast on {background} is below 3:1"
                )
        if contrast_ratio(tokens["border_strong"], tokens["surface"]) < 3.0:
            raise ValueError(f"{theme_name}: border_strong contrast is below 3:1")


def _stylesheet(tokens: Mapping[str, str]) -> str:
    """Build Qt chrome styles from semantic tokens, including visible states."""
    return f"""
        QWidget {{ background-color: {tokens['background']}; color: {tokens['text']};
                   font-family: sans-serif; font-size: 10pt; }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QListWidget {{
            background-color: {tokens['surface']}; color: {tokens['text']};
            border: 1px solid {tokens['border_strong']}; border-radius: 3px; padding: 3px;
        }}
        QWidget:disabled {{ color: {tokens['text_disabled']}; }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
        QDoubleSpinBox:disabled, QTextEdit:disabled, QListWidget:disabled,
        QCheckBox:disabled, QRadioButton:disabled, QLabel:disabled,
        QPushButton:disabled {{ color: {tokens['text_disabled']}; }}
        QPushButton {{ background-color: {tokens['surface_alt']}; color: {tokens['text']};
                       border: 1px solid {tokens['border_strong']}; border-radius: 4px;
                       padding: 5px 10px; min-height: 24px; }}
        QPushButton:hover {{ border-color: {tokens['accent']}; }}
        QPushButton:pressed {{ background-color: {tokens['selection']};
                               color: {tokens['selection_text']}; }}
        QPushButton:disabled {{ background-color: {tokens['surface_alt']};
                                color: {tokens['text_disabled']}; border-color: {tokens['border']}; }}
        QPushButton:focus, QLineEdit:focus, QComboBox:focus, QSpinBox:focus,
        QDoubleSpinBox:focus, QTextEdit:focus, QListWidget:focus {{
            border: 2px solid {tokens['focus']}; }}
        QPushButton[designRole="primary"] {{ background-color: {tokens['accent']};
            color: {tokens['on_accent']}; font-weight: 600; }}
        QPushButton[designRole="success"] {{ background-color: {tokens['success']};
            color: {tokens['on_success']}; font-weight: 600; }}
        QPushButton[designRole="danger"] {{ background-color: {tokens['danger']};
            color: {tokens['on_danger']}; font-weight: 600; }}
        QPushButton[designRole="warning"] {{ background-color: {tokens['warning']};
            color: {tokens['on_warning']}; font-weight: 600; }}
        QTabBar::tab {{ background: {tokens['surface_alt']}; color: {tokens['text']};
                        border: 1px solid {tokens['border']}; padding: 6px 9px; }}
        QTabBar::tab:selected {{ color: {tokens['selection_text']};
            background: {tokens['selection']}; border-bottom: 2px solid {tokens['focus']};
            font-weight: 600; }}
        QTabBar::tab:focus {{ border: 2px solid {tokens['focus']}; }}
        QLabel[designRole="section"] {{ color: {tokens['accent']}; font-weight: 600; }}
        QMenu {{ background: {tokens['surface']}; color: {tokens['text']};
                 border: 1px solid {tokens['border_strong']}; }}
        QMenu::item:selected {{ background: {tokens['selection']};
                                color: {tokens['selection_text']}; }}
    """


def apply_theme(application, theme_name: str) -> None:
    """Apply the named theme to a QApplication without changing application data."""
    validate_theme_tokens(THEME_TOKENS)
    if theme_name not in THEME_TOKENS:
        raise ValueError(f"Unknown theme: {theme_name}")
    tokens = THEME_TOKENS[theme_name]
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: "background",
        QPalette.ColorRole.Base: "surface",
        QPalette.ColorRole.AlternateBase: "surface_alt",
        QPalette.ColorRole.WindowText: "text",
        QPalette.ColorRole.Text: "text",
        QPalette.ColorRole.Button: "surface_alt",
        QPalette.ColorRole.ButtonText: "text",
        QPalette.ColorRole.Highlight: "selection",
        QPalette.ColorRole.HighlightedText: "selection_text",
        QPalette.ColorRole.PlaceholderText: "text_muted",
    }
    for role, token in roles.items():
        palette.setColor(role, QColor(tokens[token]))
    application.setPalette(palette)
    application.setStyleSheet(_stylesheet(tokens))
    application.setProperty("pycamsetTheme", theme_name)


def apply_matplotlib_theme(figure, theme_name: str | None = None) -> None:
    """Theme Matplotlib chrome only; retain all data and scientific colours."""
    theme_name = theme_name or "Light"
    if theme_name not in THEME_TOKENS:
        raise ValueError(f"Unknown theme: {theme_name}")
    tokens = THEME_TOKENS[theme_name]
    _MANAGED_FIGURES.add(figure)
    figure.patch.set_facecolor(tokens["background"])
    for text in figure.texts:
        text.set_color(tokens["text"])
    for axes in figure.axes:
        axes.set_facecolor(tokens["surface"])
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
            legend.get_frame().set_edgecolor(tokens["border_strong"])
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
