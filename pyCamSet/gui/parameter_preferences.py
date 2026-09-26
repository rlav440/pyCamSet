"""Purpose: Save editable GUI inputs in per-user config, never calibration runs.
Status: Versioned, atomic page snapshots with explicit reset-to-default controls.
Future: Add fields explicitly when new parameter widgets are introduced.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QLineEdit, QPushButton, QSpinBox)

from pyCamSet.gui.preferences import config_directory
from pyCamSet.gui.shared_functions import read_parameter_widget

_SCHEMA = "pycamset.gui-parameters"
_VERSION = 1
# Run selectors, output/status fields and diagnostic-only switches are intentionally
# absent: they represent a particular run, not a durable input parameter.
_FIELDS = {
    "phase0": ("_floc_edit", "_ws_edit"),
    "phase1": ("_cache_cb", "_upscale_combo", "_rescale_gamma_cb",
               "_preprocessing_scale_edit", "_preprocessing_gamma_edit",
               "_hd_cb", "_nlim_edit", "_threads_edit", "_fp_edit", "_po_edit"),
    "phase2": ("_det_pickle_edit", "_cache_cb", "_lens_combo", "_hd_cb",
               "_nlim_edit", "_min_dtct_spin", "_fp_edit"),
    "phase3": ("_threads_edit", "_max_nfev_spin", "_verbosity_spin",
               "_outliers_combo", "_fixed_pose_edit", "_ref_cam_edit",
               "_ref_pose_edit", "_fp_edit", "_lockbox_enabled_cb",
               "_lockbox_source_edit", "_lockbox_warm_start_cb",
               "_lockbox_rotation_half_spin", "_lockbox_translation_half_spin",
               "_lockbox_rotation_sigma_spin", "_lockbox_translation_sigma_spin",
               "_lockbox_center_sigma_spin"),
    "phase4": ("_phase3_camset_edit", "_threads_edit", "_max_nfev_spin",
               "_verbosity_spin", "_loss_combo", "_f_scale_spin",
               "_outliers_combo", "_fixed_pose_edit", "_ref_cam_edit",
               "_ref_pose_edit", "_fp_edit"),
    "export": ("_ws_edit", "_format_combo", "_depth_min_edit", "_depth_max_edit", "_depth_num_edit"),
    "optimisation": ("_floc_edit", "_outdir_edit", "_mode_combo", "_trials_spin",
                     "_seed_edit", "_sampler_combo", "_outliers_combo",
                     "_max_nfev3_spin", "_max_nfev4_spin", "_target_rpe_spin",
                     "_retain_spin", "_trial_gating_profile_combo",
                     "_min_gating_cameras_spin", "_min_gating_images_spin",
                     "_min_point_ratio_spin", "_detection_profile_combo"),
    "detection_cost": ("_folder_edit", "_out_edit", "_max_frames_spin",
                       "_fps_spin", "_single_thread_cb"),
}


def parameter_path() -> Path:
    return config_directory() / "parameters.json"


def _value(widget):
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QLineEdit):
        return widget.text()
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return widget.value()
    raise TypeError("Not an editable parameter")


def _apply(widget, value) -> None:
    if isinstance(widget, QCheckBox) and type(value) is bool:
        widget.setChecked(value)
    elif isinstance(widget, QComboBox) and type(value) is str and widget.findText(value) >= 0:
        widget.setCurrentIndex(widget.findText(value))
    elif isinstance(widget, QLineEdit) and type(value) is str and len(value) <= 4096:
        widget.setText(value)
    elif isinstance(widget, QSpinBox) and type(value) is int and widget.minimum() <= value <= widget.maximum():
        widget.setValue(value)
    elif isinstance(widget, QDoubleSpinBox) and type(value) in (int, float) and widget.minimum() <= value <= widget.maximum():
        widget.setValue(value)
    else:
        raise ValueError("Saved parameter has invalid type or is outside its current choices/range")


#: Spellings an outliers combo held before it was normalised to No/Yes at
#: start-up; files saved then still carry them.  "ask" was treated as Yes by
#: the normaliser, so it is here too.
_LEGACY_OUTLIER_CHOICES = {"n": "No", "no": "No", "y": "Yes", "yes": "Yes", "ask": "Yes"}


def _upgrade_saved_choice(widget, value):
    """Map a legacy saved outliers spelling onto the combo's current items."""
    if (isinstance(widget, QComboBox) and widget.objectName() == "outliers_combo"
            and isinstance(value, str) and widget.findText(value) < 0):
        upgraded = _LEGACY_OUTLIER_CHOICES.get(value.strip().lower())
        if upgraded is not None and widget.findText(upgraded) >= 0:
            return upgraded
    return value


class ParameterPreferences:
    """One validated snapshot for each parameter tab; reset uses startup defaults."""

    def __init__(self, window) -> None:
        self.window = window
        self.pages = {name: getattr(window, "export_calibration_tab" if name == "export"
                                    else f"{name}_tab") for name in _FIELDS}
        self.defaults = {name: self._capture(name) for name in self.pages}
        self.path = parameter_path()
        self.load_error = None
        self._preserve_corrupt = False
        if self.path.exists():
            try:
                document = json.loads(self.path.read_text(encoding="utf-8"))
                if (not isinstance(document, dict)
                        or set(document) != {"schema", "version", "pages"}
                        or document["schema"] != _SCHEMA
                        or type(document["version"]) is not int
                        or document["version"] != _VERSION
                        or not isinstance(document["pages"], dict)
                        or set(document["pages"]) - set(_FIELDS)):
                    raise ValueError("Unsupported parameter file schema")
                for name, values in document["pages"].items():
                    if not isinstance(values, dict) or set(values) - set(self.defaults[name]):
                        raise ValueError("Unknown saved parameter")
                    for key in values.keys() & set(_FIELDS[name]):
                        values[key] = _upgrade_saved_choice(getattr(self.pages[name], key), values[key])
                    # Validate every value against its actual widget before applying any.
                    for key, value in values.items():
                        if key in _FIELDS[name]:
                            self._validate(self.pages[name], key, value)
                        elif key not in {"target", "detection_options"}:
                            raise ValueError("Unknown saved parameter")
                for name, values in document["pages"].items():
                    self._restore(name, values)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self.load_error = str(exc)
                self._preserve_corrupt = True
                # A valid page may have been applied before a later page
                # failed. Never leave a half-restored parameter profile.
                for name, defaults in self.defaults.items():
                    self._restore(name, defaults)
        for name, page in self.pages.items():
            reset = QPushButton("Reset Parameters to Default", page)
            reset.setAccessibleName(f"Reset {name} parameters to default")
            reset.setToolTip("Restore this page's inputs to their defaults")
            reset.clicked.connect(lambda _checked=False, phase=name: self.reset(phase))
            # A compact, right-aligned secondary action rather than a
            # full-width bar that reads as the page's main call to action.
            page.layout().insertWidget(0, reset, 0, Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _validate(page, key, value) -> None:
        widget = getattr(page, key)
        if isinstance(widget, QComboBox):
            if type(value) is not str or widget.findText(value) < 0:
                raise ValueError("Unknown saved choice")
        elif isinstance(widget, QLineEdit):
            if type(value) is not str or len(value) > 4096:
                raise ValueError("Invalid saved text")
        elif isinstance(widget, QCheckBox):
            if type(value) is not bool:
                raise ValueError("Invalid saved checkbox")
        elif isinstance(widget, QSpinBox):
            if type(value) is not int or not widget.minimum() <= value <= widget.maximum():
                raise ValueError("Invalid saved integer")
        elif isinstance(widget, QDoubleSpinBox):
            if type(value) not in (int, float) or not widget.minimum() <= value <= widget.maximum():
                raise ValueError("Invalid saved number")
        else:
            raise ValueError("Unsupported input control")

    def _capture(self, name: str) -> dict:
        page = self.pages[name]
        values = {key: _value(getattr(page, key)) for key in _FIELDS[name]}
        if name in {"phase1", "optimisation"}:
            values["target"] = page._target_form.spec()
        if name == "phase1":
            # Only user-edited options are durable. Persisting all the current
            # detector's defaults would incorrectly pin them when switching
            # back from another target or after a backend default changes.
            options = dict(page._detection_option_retained)
            options.update({
                key: read_parameter_widget(page._detection_option_widgets[key])
                for key in page._detection_option_edited
                if key in page._detection_option_widgets})
            values["detection_options"] = options
        return values

    def _restore(self, name: str, values: dict) -> None:
        page = self.pages[name]
        if name == "phase1" and "detection_options" in values:
            options = values["detection_options"]
            if (not isinstance(options, dict) or len(options) > 128
                    or any(type(key) is not str or len(key) > 128 for key in options)):
                raise ValueError("Invalid saved detection options")
            page._detection_option_retained = dict(options)
            page._detection_option_edited = set()
        if "target" in values and name in {"phase1", "optimisation"}:
            page._target_form.apply_spec(values["target"])
        if name == "phase1":
            # Rebuild even if the target type did not change. Resetting must
            # recreate pristine detector widgets, not leave edited values up.
            if "detection_options" in values:
                page._detection_option_edited = set(options)
                # The rebuild normally snapshots outgoing user edits. These
                # widgets are fresh startup defaults, not outgoing edits; do
                # not let them overwrite the values just loaded from disk.
                page._detection_option_widgets = {}
            page._rebuild_detection_options()
        if "detection_options" in values and name == "phase1":
            for key, value in options.items():
                widget = page._detection_option_widgets.get(key)
                if widget is not None and read_parameter_widget(widget) != value:
                    raise ValueError("Saved detector option is outside its current range")
        for key in _FIELDS[name]:
            if key in values:
                _apply(getattr(page, key), values[key])

    def reset(self, name: str) -> None:
        self._restore(name, self.defaults[name])
        # Reset affects inputs only; a previously confirmed folder or loaded run
        # must not be mistaken for a newly validated one.
        self.save()

    def save(self) -> None:
        document = {"schema": _SCHEMA, "version": _VERSION,
                    "pages": {name: self._capture(name) for name in self.pages}}
        payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._preserve_corrupt and self.path.exists():
            backup = self.path.with_name("parameters.json.corrupt")
            if not backup.exists():
                shutil.copy2(self.path, backup)
            self._preserve_corrupt = False
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             dir=self.path.parent, prefix=".parameters-",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
