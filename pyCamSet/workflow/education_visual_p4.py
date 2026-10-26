'''Qt-free preparation of everything Phase 4's visual education view draws.
"unavailable"/"not recorded", never a guess.

Two parts:

* :class:`Phase4Narrative` - the honest statements about provenance, start settings, the shape
  estimate, playback and the decision. They used to be the whole Phase 4 view; they now back
  its captions, chips and the "Numbers" drawer. Semantics are unchanged (A1/A2 rules: unconfirmed
  starts are never attributed, missing evidence is never rendered as zero, recovery steps name
  the Phase 3 buttons).
* :func:`prepare_phase4_visual` - the data the five stages draw: the rig and target as Phase 4
  started (Phase 3's result), per-image/per-camera errors before and after, the images left out,
  the target-shape correction, the recorded solver steps (replayed through ONE rebuilt handler,
  each checked against the stored cost), and the quality gate as traffic-light checks.

Geometry convention: everything is drawn in the *target frame at the reference photo* (the
fixed pose). Cameras are world-to-camera ``(R, t)``. A telecentric camera has no position (only
its rotation is estimated), so it is drawn at :data:`TELECENTRIC_DISPLAY_DISTANCE` target sizes
from the target along its viewing direction; the view says so.
'''
from __future__ import annotations

import hashlib
import json
import logging
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from pyCamSet.workflow.education_derive import (
    SCALE_RATIO_TOLERANCE, DerivationUnavailable,
)
from pyCamSet.workflow.run_quality import blocking_reasons

#: A1 section 9: the one recovery route when Phase 4's starting camset cannot be confirmed.
#: The quoted labels are the Phase 3 GUI's own button texts.
START_FROM_PHASE3 = ("In Phase 3, open ‘Diagnostics ▼’, then press ‘Phase 4 - Self-Calibration ▶’ "
                     "to start Phase 4 from a traceable Phase 3 result; run Phase 4 again.")

#: Short stage names (stage stepper pills).
STAGES = ("Start from Phase 3", "Freeing the shape", "Watching the solver",
          "Can we trust it?", "Result", "What it is for")

#: The quality gate's internal blocking flags in the learner's words (UX3-4): (what it
#: means, what to do). ``{n}`` is the number of photos left out. A flag not listed here is
#: shown as "Recorded reason: <flag>".
PLAIN_BLOCKING = {
    # Written by pyCamSet versions before 2026-10-08: a photo without a target position
    # held the whole result back. Since then such photos are left out and only noted.
    "image observation graph has missing image indices": (
        "{n} photos show no target corners, so the target's position in them is unknown, and "
        "the quality gate holds this result back.",
        "Press ‘Phase 4 - Self-Calibration ▶’ in Phase 3 and run it again."),
    "camera observation graph does not cover every active camera": (
        "A camera saw no corner, so it would leave Phase 4 uncalibrated.",
        "Check that camera's photos in Phase 1, then run Phases 3 and 4 again."),
    "target gauge does not fix three non-collinear points": (
        "The solver could not fix three corners, so the shape's own symmetries stayed in the "
        "solve.",
        "Run Phase 4 again; if it repeats, the target is seen too little."),
    "pose gauge has no fixed reference pose": (
        "No photo's pose was fixed, so the rig's overall position and turn stayed in the solve.",
        "Run Phase 4 again from a complete Phase 3 result."),
    "Phase 3 input lacks an identified run with complete status": (
        "The Phase 3 result it started from cannot be traced to a complete run.",
        START_FROM_PHASE3),
    "solver did not report successful termination": (
        "The solver stopped before it settled.",
        "Run Phase 4 again; if it repeats, record the solver steps and look at them."),
    "final reprojection error did not improve finitely": (
        "The error did not go down.",
        "Run Phase 4 again from a complete Phase 3 result."),
    "camera intrinsics, distortion, or extrinsics are implausible": (
        "A camera's solved lens or pose is physically implausible.",
        "Check the camera models chosen in Phase 2, then run Phases 3 and 4 again."),
}


#: Above this many parameters the Jacobian summary is not decomposed (a dense JᵀJ of
#: 3000 parameters is about 72 MB; its eigendecompositions take seconds).
MAX_DENSE_SUMMARY_PARAMS = 3000


def plain_blocking(reason: Any, n_excluded: int = 0) -> tuple[str, str]:
    """(what the blocking ``reason`` means, what to do) in plain words (UX3-4)."""
    text = str(reason)
    if text in PLAIN_BLOCKING:
        what, todo = PLAIN_BLOCKING[text]
        return what.format(n=n_excluded or "Some"), todo
    if text.startswith("Phase 3 input disposition was"):
        return ("The Phase 3 result it started from was not complete.",
                "Run Phase 3 again, then Phase 4.")
    return f"Recorded reason: {text}.", "Fix what the reason names, then run Phase 4 again."

#: Telecentric cameras are drawn this many target sizes from the target centre.
TELECENTRIC_DISPLAY_DISTANCE = 1.35

#: A point needs at least this angle between its viewing rays to be "seen from several
#: directions" (education_derive._phase4_point_classes' default).
THETA_MIN_DEG = 5.0

#: Exaggeration factors offered for the shape correction (display only).
EXAGGERATIONS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)

#: Problem options that shape only a phase's own start or log, never the Phase 4 solve
#: (SCI-4): not counted as "other settings differed".
SEEDING_ONLY_OPTIONS = frozenset({"recover_single_face_telecentric", "verbosity",
                                  "target_scale_rule", "recorded_gauge_indices"})



# --------------------------------------------------------------------------------------
# Small helpers


def plain(value: Any) -> Any:
    """Convert the dialog's immutable JSON values (MappingProxy, tuples) to dicts/lists."""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    return value


def fmt(value: Any, digits: int = 4) -> str:
    """Format only finite numbers; missing evidence is never rendered as zero."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "unavailable"
    return f"{number:.{digits}g}" if np.isfinite(number) else "unavailable"


def finite(value: Any) -> float | None:
    """``float(value)`` when it is a finite number, else None."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def loss_text(loss: Any, scale: Any) -> str:
    """Name the objective and include its scale only for a robust loss."""
    name = str(loss or "linear")
    if name == "linear":
        return name
    return f"{name} (f_scale={fmt(scale)})" if scale is not None else f"{name} (f_scale unavailable)"


def ordinal(value: Any) -> str:
    """Format only a positive saved 1-based integer pose ordinal."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return "cannot be established from the saved data"
    suffix = "th" if 10 <= value % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def camset_options(camset: Any) -> Mapping[str, Any]:
    """Read the persisted handler's options without substituting UI defaults."""
    handler = getattr(camset, "calibration_handler", None)
    options = getattr(handler, "problem_opts", {})
    return options if isinstance(options, Mapping) else {}


def nice_exaggeration(target_size: float, typical: float, share: float = 0.15) -> int:
    """Pick an exaggeration so a typical correction is drawn about ``share`` of the target.

    Always one of :data:`EXAGGERATIONS` (display only; the saved result is unchanged).
    """
    if not (math.isfinite(target_size) and math.isfinite(typical)) or typical <= 0:
        return 1
    wanted = share * target_size / typical
    chosen = EXAGGERATIONS[0]
    for factor in EXAGGERATIONS:
        if factor <= wanted:
            chosen = factor
    return int(chosen)


def length_unit(mm_values: np.ndarray | float | None) -> tuple[str, float]:
    """Unit to show millimetre lengths in: ("µm", 1000) below 1 mm, else ("mm", 1)."""
    if mm_values is None:
        return "mm", 1.0
    values = np.asarray(mm_values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size and float(np.percentile(values, 95)) < 1.0:
        return "µm", 1000.0
    return "mm", 1.0


# --------------------------------------------------------------------------------------
# Narrative: the honest statements (moved from the previous text-first view)


class Phase4Narrative:
    """Provenance, start, shape, playback and decision statements for one Phase 4 run.

    Built from the run context (``phase3_state_source()``, ``artifact()``, ``status``,
    ``quality_disposition``), its plain metadata, the saved trace and ``derive_phase4``'s result.
    """

    def __init__(self, run_context: Any, metadata: Mapping[str, Any], state: Any,
                 trace: Mapping[str, Any], derivation: Any, run_id: str) -> None:
        self.run_context = run_context
        self.metadata = metadata
        self.state = state
        self.trace = trace
        self.derivation = derivation
        self.run_id = run_id
        self._cache: dict[str, Any] = {}

    # -- saved settings -----------------------------------------------------------------

    def value(self, name: str) -> Any:
        """An attribute of the derivation, None when it is unavailable."""
        return getattr(self.derivation, name, None)

    def phase4_loss(self) -> tuple[str, Any]:
        """Read Phase 4 loss values from its own saved camset."""
        if "phase4_loss" in self._cache:
            return self._cache["phase4_loss"]
        try:
            from pyCamSet.utils.saving import load_CameraSet
            camset = load_CameraSet(self.run_context.artifact("phase4"))
            options = camset_options(camset)
            result = (str(options.get("loss", "linear")), options.get("f_scale", 1.0))
        except Exception:
            options = (self.metadata.get("params") or {}).get("problem_options") or {}
            result = ((str(options.get("loss", "linear")), options.get("f_scale", 1.0))
                      if isinstance(options, Mapping) else ("linear", 1.0))
        self._cache["phase4_loss"] = result
        return result

    def phase3_loss(self) -> tuple[str, Any]:
        """Use only Phase 3 state-source details, never the recorded run ID's camset."""
        return self.state.loss or "linear", self.state.f_scale

    def phase3_options(self) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Read saved start settings only from the path selected by A2's state source."""
        path = self.state.path
        if path is None:
            return {}, {}
        try:
            from pyCamSet.utils.saving import load_CameraSet
            source = load_CameraSet(path)
            options = camset_options(source)
            handler = getattr(source, "calibration_handler", None)
            fixed = getattr(handler, "fixed_params", {})
            return options, fixed if isinstance(fixed, Mapping) else {}
        except Exception:
            return {}, {}

    def setting_differences(self) -> list[str]:
        """Name the effective solve settings, other than the loss, that differ between phases.

        Seeding and bookkeeping keys are not solve settings (SCI-4, round 1): Phase 3's
        ``recover_single_face_telecentric`` only shapes Phase 3's own start, ``verbosity`` only
        prints, and the gauge/scale-rule keys are Phase 4's own. (On a real telecentric rig the explanation said
        "other start settings also differed" when the one difference was the Phase 3 seeding
        switch.) A different step limit still counts: a Phase 3 stopped early leaves room
        that more steps alone would fill.
        """
        source_options, _source_fixed = self.phase3_options()
        current = (self.metadata.get("params") or {}).get("problem_options") or {}
        current = current if isinstance(current, Mapping) else {}
        differs = []
        for key in set(source_options) | set(current):
            if key in {"outliers", "loss", "f_scale"} | SEEDING_ONLY_OPTIONS:
                continue
            if source_options.get(key) != current.get(key):
                differs.append(key)
        # A2 has already compared effective held values. Do not compare the
        # serialised fixed_params containers: equivalent forms may differ.
        if self.state.effective_fixed_parameter_differences:
            differs.append("fixed parameters")
        return differs

    @staticmethod
    def loss_differs(phase3_loss: str, phase3_scale: Any, phase4_loss: str,
                     phase4_scale: Any) -> bool:
        return phase3_loss != phase4_loss or phase3_scale != phase4_scale

    # -- fit comparison -----------------------------------------------------------------

    def phase3_end_px(self) -> float | None:
        """Phase 3's final error, the baseline Phase 4 is compared against.

        A changed start uses Phase 3's own D3.6 (``phase3_before_px``). A confirmed start
        with no mismatch began exactly where Phase 3 ended, so the camset's stored mean is
        that baseline; ``phase3_before_px`` is deliberately None there. (On a real telecentric rig the
        explanation showed 'Phase 3 ended at unavailable px' for the ordinary, matching
        hand-off.) Never shown unconfirmed.
        """
        state = self.state
        if not state.confirmed:
            return None
        if state.phase3_before_px is not None:
            return state.phase3_before_px
        if state.start_mismatch is None:
            return state.stored_mean_px
        return None

    def fit_direction(self, before: Any = None) -> str | None:
        """'lower', 'higher' or 'same' for Phase 4's error against Phase 3's, or None."""
        if before is None:
            before = self.phase3_end_px()
        before_value, after_value = finite(before), finite(self.value("d43_final_px"))
        if before_value is None or after_value is None:
            return None
        if after_value < before_value:
            return "lower"
        return "higher" if after_value > before_value else "same"

    def comparison_note(self) -> str:
        """Avoid attributing a change to shape when other saved settings differ."""
        phase3_loss, phase3_scale = self.phase3_loss()
        phase4_loss, phase4_scale = self.phase4_loss()
        if not self.state.confirmed:
            return "The starting camset is unconfirmed, so the error change cannot be attributed."
        differs = self.setting_differences()
        loss_differs = self.loss_differs(phase3_loss, phase3_scale, phase4_loss, phase4_scale)
        losses = (f"({loss_text(phase3_loss, phase3_scale)} versus "
                  f"{loss_text(phase4_loss, phase4_scale)})")
        if loss_differs and differs:
            return (f"The different losses affect the comparison {losses}, and other settings also differed "
                    f"({', '.join(sorted(differs))}); the effects cannot be separated. "
                    "The change is not attributed to the shape alone.")
        if loss_differs:
            return (f"The different losses affect the comparison {losses}; the checked start and other "
                    "effective settings match. The change is not attributed to the shape alone.")
        if differs:
            return ("The starting point or other settings differed "
                    f"({', '.join(sorted(differs))}); the change is not attributed to the shape alone.")
        return ("The checked start and other effective settings match. The error change comes from "
                "freeing the target shape while the lenses and cameras were refined with it; it is "
                "not a measurement of the printed board's size.")

    def plain_comparison(self) -> str:
        """L1 reading of the fit comparison without loss names or f_scale (A1 section 1)."""
        phase3_loss, phase3_scale = self.phase3_loss()
        phase4_loss, phase4_scale = self.phase4_loss()
        direction = self.fit_direction()
        if not self.state.confirmed:
            return "The starting camset is unconfirmed, so the error change cannot be attributed."
        if not self.usable_run():
            # A held or failed run is not a calibration, so no cause is credited for its change.
            return "Not credited to any cause: this run is not a usable calibration, so its error change is not attributed."
        other = (" Other start settings also differed, so the error change is not attributed to the "
                 "shape alone." if self.setting_differences() else "")
        if self.loss_differs(phase3_loss, phase3_scale, phase4_loss, phase4_scale):
            return ("Phase 3 and Phase 4 scored the fit with different rules, so read the two error "
                    "numbers as a rough guide; the shape part of the answer above matters more." + other)
        if other:
            return other.strip()
        if direction == "lower":
            return ("Both phases scored the fit with the same rule and the checked start matches, so the "
                    "error change comes from freeing the target shape together with the lenses and "
                    "cameras that were re-tuned alongside it.")
        if direction in {"higher", "same"}:
            # Freeing more parameters from a matching start should not leave the
            # error at or above where it began; say so instead of calling it expected.
            lead = ("Both phases scored the fit with the same rule and the checked start matches, yet the "
                    "error did not go down. Freeing the target shape should lower it, so do not rely on "
                    "this run.")
            if blocking_reasons(plain(self.metadata)):
                return lead + " Read the reasons under 'Can we trust it?', then run Phase 4 again."
            termination = (self.metadata.get("report") or {}).get("termination_reason") or "not recorded"
            if (self.metadata.get("params") or {}).get("record_education") is True:
                step = ("Next: open 'Watching the solver' and check whether the cost was still falling "
                        "at the last recorded step.")
            else:
                step = ("Next: tick ‘Record solver steps for this run’ in this phase's settings, run Phase 4 "
                        "again, and check in 'Watching the solver' whether the cost was still falling "
                        "at the last recorded step.")
            return (f"{lead} No warning was recorded; the solver's recorded reason for stopping is: "
                    f"{termination}. A solver that stopped on a step or time limit, before the cost "
                    "stopped falling, can leave the error at or above where it began. " + step)
        return ("Both phases scored the fit with the same rule and the checked start matches, but the "
                "error change cannot be checked for this run.")

    # -- status and provenance ------------------------------------------------------------

    def status_copy(self) -> str:
        """Describe failure and attribution before any measurement or comparison."""
        run_context = self.run_context
        if run_context.status == "failed":
            error = str(self.metadata.get("error") or "details unavailable").rstrip().rstrip(".")
            return (f"Quality check: failed. Run {self.run_id} records an error: {error}. "
                    "No saved Phase 4 result is presented as a successful calibration. "
                    "Next: fix the recorded error above, then run Phase 4 again.")
        disposition = getattr(run_context, "quality_disposition", None)
        if disposition == "not checked":
            return ("Quality check: not checked. This saved run has no checkable quality result; "
                    "do not treat its completion status as a verified hand-off. "
                    "Next: run Phase 4 again to get a checkable result before exporting it.")
        if disposition == "failed":
            return ("Quality check: failed. This run is not offered as a clean hand-off. "
                    "Next: fix the recorded error, then run Phase 4 again.")
        if disposition == "quality check not attributable to the camset used":
            if self.state.verification.value == "differs" and self.state.origin_run_id:
                return ("Quality check not attributable to the camset used. The run started from a camset "
                        f"other than Phase 3 run {self.state.origin_run_id}. Recovery: in Phase 3, open "
                        "‘Diagnostics ▼’ and select that run alone, then press ‘Phase 4 - Self-Calibration ▶’; "
                        "run Phase 4 again.")
            return ("Quality check not attributable to the camset used: the starting camset cannot be "
                    "confirmed. Phase 3 states are not substituted. Recovery: in Phase 3, open "
                    "‘Diagnostics ▼’ and select the intended verified Phase 3 run, then press "
                    "‘Phase 4 - Self-Calibration ▶’; run Phase 4 again.")
        if self.state.verification.value != "verified" or not self.state.confirmed:
            return ("Quality check not attributable to the camset used. "
                    "The starting camset could not be confirmed. " + START_FROM_PHASE3)
        reasons = blocking_reasons(plain(self.metadata))
        if run_context.status != "complete":
            if reasons:
                return (f"Quality check: {run_context.status}. This run is not offered as a clean "
                        f"hand-off because: {'; '.join(map(str, reasons))}. "
                        "Next: fix what the reasons above name, then run Phase 4 again.")
            return (f"Quality check: {run_context.status}. This run is not offered as a clean "
                    "hand-off; its record names no reason. "
                    "Next: run Phase 4 again to get a checkable result before exporting it.")
        termination = (self.metadata.get("report") or {}).get("termination_reason", "not recorded")
        return (f"Quality check: {disposition or 'not checked'}. Run status: {run_context.status}. "
                f"Termination: {termination}. "
                f"Recorded warnings or blocking flags: {', '.join(map(str, reasons)) if reasons else 'none recorded'}.")

    def trusted_start(self) -> bool:
        """True when the start is confirmed and the run is a clean, attributable hand-off."""
        return (self.run_context.status != "failed" and bool(self.state.confirmed)
                and getattr(self.state.verification, "value", None) == "verified")

    def usable_run(self) -> bool:
        """The run passed its gate: only then may a verdict (fit better, shape backed, an
        attribution to the shape) be stated. A held-back or failed run gets neutral facts only."""
        return (self.run_context.status == "complete"
                and not blocking_reasons(plain(self.metadata)))

    def headline_warning(self, on_trust_stage: bool = False) -> str:
        """One short sentence shown first on every stage when the run cannot be trusted as is
        (on 'Can we trust it?' itself it does not point back at that stage, UX3-4)."""
        if self.run_context.status == "failed":
            return "This run failed; nothing here is a usable calibration."
        if not self.trusted_start():
            return "The camset this run started from could not be confirmed, so changes are not attributed."
        if self.run_context.status != "complete":
            if on_trust_stage:
                return f"This run is {self.run_context.status}: it is held back, not handed on."
            return f"This run is {self.run_context.status}; see 'Can we trust it?'."
        return ""

    def input_copy(self) -> str:
        """Show only provenance claims carried by A2's Phase 3 state source."""
        state = self.state
        status = self.status_copy()
        if state.reason and "another image folder" in state.reason:
            origin = "a camset from another image folder"
        elif state.origin_run_id:
            origin = f"Phase 3 run {state.origin_run_id}"
        else:
            origin = "a camset whose source cannot be traced"
        copy = f"Run {self.run_id} started from {origin}. {status}"
        if state.verification.value == "differs" and state.origin_run_id:
            copy += (f" The recorded Phase 3 run {state.origin_run_id} is not the camset "
                     "this run started from.")
        if state.source_check:
            copy += f" Source check: {state.source_check}."
        if (self.run_context.status == "complete"
                and (state.verification.value != "verified"
                     or "another copy of this workspace" in str(state.reason or ""))):
            copy += " The Export tab still offers this run; check which Phase 3 camset it started from first."
        return copy

    def start_copy(self) -> str:
        """Explain N2 or effective fixed-parameter differences without misattribution."""
        state = self.state
        metric = (self.metadata.get("diagnostics") or {}).get("D4.3_initial_euclid_px")
        if state.start_mismatch == "fixed_pose":
            return (f"{self.pose_ordinal_copy()} The two phases used different reference photos, so Phase 4 "
                    f"started from an inconsistent position. Phase 4's starting error was {fmt(metric)} px. "
                    "The Phase 3 and Phase 4 camera positions use different reference photos, so they are "
                    "not directly comparable in this view. In Phase 3, open ‘Diagnostics ▼’, then press "
                    "‘Phase 4 - Self-Calibration ▶’ to start Phase 4 from that Phase 3 result.")
        if state.start_mismatch == "fixed_parameters":
            labels = ", ".join(state.effective_fixed_parameter_differences) or "recorded parameters"
            return (f"Phase 4 held {labels} fixed at values different from the starting camset; "
                    f"its starting error was {fmt(metric)} px. This is a changed starting setting, "
                    "not a measure of Phase 3's fit.")
        if not state.confirmed:
            return "Unavailable: the starting camset cannot be confirmed. " + START_FROM_PHASE3
        return ("The Phase 4 start is confirmed against the camset it used. "
                f"Phase 3 ended at {fmt(self.phase3_end_px())} px; "
                f"Phase 4 started at {fmt(metric)} px.")

    def pose_ordinal_copy(self) -> str:
        """Describe only ordinals supplied by A2's saved state source."""
        state = self.state

        def describe(phase: str, value: Any) -> str:
            text = ordinal(value)
            if not text.endswith(("st", "nd", "rd", "th")):
                return f"{phase}'s reference photo {text}"
            return (f"{phase} used the {text} photo of the detection run's list "
                    f"(image index {int(value) - 1}) as its reference photo")

        phase3_copy = describe("Phase 3", getattr(state, "phase3_fixed_pose_ordinal1", None))
        phase4_copy = describe("Phase 4", getattr(state, "phase4_fixed_pose_ordinal1", None))
        return f"{phase3_copy}; {phase4_copy}."

    def gauge_copy(self) -> str:
        """Use photo names only when A2's verified source provides them."""
        state = self.state
        if not state.confirmed:
            return ("Which photo's pose was fixed for the position frame cannot be checked because the "
                    "starting camset or its photo source could not be confirmed. " + START_FROM_PHASE3)
        names = tuple(state.photo_names)
        source_check = str(state.source_check or "")
        # Legacy cache reuse removes the index-to-file link. Preserve A2's
        # recorded reason verbatim and never substitute a current folder name.
        if "may have reused saved detections" in source_check:
            phase3_ordinal = ordinal(getattr(state, "phase3_fixed_pose_ordinal1", None))
            mapped_pose = (f"the {phase3_ordinal} photo in the detection run's list, not necessarily "
                           f"the {phase3_ordinal} file in the folder now"
                           if phase3_ordinal.endswith(("st", "nd", "rd", "th")) else
                           "the saved detection-run photo ordinal, whose current file cannot be established")
            photo_copy = ("the detection run may have reused saved detections, so which photo each came from "
                          f"was not recorded; {mapped_pose}; no current file name is assigned to that pose")
        elif "earlier photo set" in source_check:
            photo_copy = ("the detections may have come from an earlier photo set; no current file name is "
                          "assigned to that pose")
        elif "photo list" in source_check and "no longer matches" in source_check:
            photo_copy = ("the detection run's photo list no longer matches the current folder; no current "
                          "file name is assigned to that pose")
        elif names and len(set(names)) == 1:
            photo_copy = f"the reference photo is {names[0]} in each camera folder"
        elif names:
            photo_copy = "the reference-photo names differ by camera folder: " + ", ".join(names)
        else:
            photo_copy = "the reference photo cannot be named from the saved source"
        mismatch_copy = (" The rig comparison is withheld because the two phases used different reference photos."
                         if state.rig_comparison_withheld else "")
        return (f"{self.pose_ordinal_copy()} Positions are expressed relative to where the target was in "
                f"{photo_copy}. The solved shape is then scaled to the square size you entered and lined "
                "up with the design you entered; overall size is not checked. Use callipers if physical "
                f"size matters.{mismatch_copy}")

    def gauge_detail(self) -> str:
        """L2: why the position frame cannot be checked for an unconfirmed start."""
        if self.state.confirmed:
            return ""
        return ("The position-frame fix cannot be checked; telecentric behaviour is unverified "
                "here.")

    def rig_comparison_rows(self) -> list[tuple[str, str, str]]:
        """Compare pinhole rigs directly only after both saved fixed-pose checks pass."""
        state = self.state
        if state.rig_comparison_withheld:
            return [("Rig comparison", "withheld", "the two phases used different reference photos")]
        if not state.confirmed or state.path is None:
            return [("Rig comparison", "unavailable", "the Phase 3 state source is unconfirmed")]
        try:
            from pyCamSet.cameras.telecentric_camera import TelecentricCamera
            from pyCamSet.utils.saving import load_CameraSet
            from pyCamSet.workflow.education_derive import _fixed_pose_valid, _observation_rows

            before = load_CameraSet(state.path)
            after_path = self.run_context.artifact("phase4")
            if after_path is None:
                return [("Rig comparison", "unavailable", "saved Phase 4 camset is missing")]
            after = load_CameraSet(after_path)
            before_pose, before_valid = _fixed_pose_valid(
                before.calibration_handler, _observation_rows(before.calibration_handler))
            after_pose, after_valid = _fixed_pose_valid(
                after.calibration_handler, _observation_rows(after.calibration_handler))
            if before_pose != after_pose:
                return [("Rig comparison", "withheld", "the two phases used different reference photos")]
            if not before_valid or not after_valid:
                return [("Rig comparison", "unavailable", "fixed-pose validity check failed")]
            before_cameras = before.get_cam_dict()
            after_cameras = after.get_cam_dict()
            if list(before_cameras) != list(after_cameras):
                return [("Rig comparison", "unavailable", "camera identities or order differ")]
            if any(isinstance(camera, TelecentricCamera)
                   for camera in (*before_cameras.values(), *after_cameras.values())):
                return [("Rig comparison", "unverified", "the telecentric comparison is not established")]
            before_units = getattr(before, "_saved_point_data_units", None)
            after_units = getattr(after, "_saved_point_data_units", None)
            units_match = before_units == after_units and before_units in {"m", "mm"}
            rows = []
            for name in before_cameras:
                left, right = before_cameras[name], after_cameras[name]
                delta = float(np.linalg.norm(np.asarray(right.position) - np.asarray(left.position)))
                if units_match:
                    if before_units == "m":
                        delta *= 1000.0
                    delta_text = f"{fmt(delta)} mm"
                else:
                    delta_text = f"{fmt(delta)} target units (mm unverified)"
                left_fx = np.asarray(left.intrinsic, dtype=np.float64)[0, 0]
                right_fx = np.asarray(right.intrinsic, dtype=np.float64)[0, 0]
                rows.append((str(name), delta_text,
                             f"fx: {fmt(left_fx)} → {fmt(right_fx)} px; same fixed pose, no fitted alignment"))
            if getattr(self.derivation, "metric_qualifier_required", False):
                return [(name, f"{delta} (scale-qualified)", detail) for name, delta, detail in rows]
            return rows or [("Rig comparison", "unavailable", "no cameras in saved camsets")]
        except Exception:
            return [("Rig comparison", "unavailable", "saved rig values could not be checked")]

    # -- units ---------------------------------------------------------------------------

    def nominal_points(self) -> tuple[np.ndarray | None, str | None]:
        """Read design points and saved units from this Phase 4 artifact only."""
        try:
            from pyCamSet.utils.saving import load_CameraSet
            path = self.run_context.artifact("phase4")
            if path is None:
                return None, None
            cached = self._cache.get("nominal")
            if cached is not None and cached[0] == path:
                return cached[1]
            camset = load_CameraSet(path)
            points = np.asarray(camset.calibration_handler.target.point_data,
                                dtype=np.float64).reshape(-1, 3)
            units = getattr(camset, "_saved_point_data_units", None)
            result = (points, units if units in {"m", "mm"} else None)
            self._cache["nominal"] = (path, result)
            return result
        except Exception:
            return None, None

    def display_unit(self) -> tuple[str, float | None]:
        """("mm", factor from saved units to mm) once the units are verified, else
        ("target units", None)."""
        _nominal, units = self.nominal_points()
        if units == "m":
            return "mm", 1000.0
        if units == "mm":
            return "mm", 1.0
        return "target units", None

    # -- shape ---------------------------------------------------------------------------

    def refit_headline_shown(self) -> bool:
        """True when L1 leads with the several-direction mean after display refit."""
        return finite(self.value("several_directions_mean_displacement_mm")) is not None

    def scale_ratio_out_of_tolerance(self) -> bool:
        """Mirror derive_phase4's ratio half of metric_qualifier_required."""
        ratio = finite(self.value("scale_consistency_ratio"))
        return ratio is not None and abs(ratio - 1.0) > SCALE_RATIO_TOLERANCE

    def saved_units_unverified(self) -> bool:
        """The units half of metric_qualifier_required, independent of the ratio."""
        recorded = self.value("saved_units_unverified")
        if recorded is not None:
            return bool(recorded)
        # Not recorded: only infer it when the ratio cannot explain the qualifier.
        return (bool(self.value("metric_qualifier_required"))
                and not self.scale_ratio_out_of_tolerance())

    def qualifier_reasons(self) -> str:
        """Name which half (or both) of metric_qualifier_required applies."""
        ratio_out = self.scale_ratio_out_of_tolerance()
        recorded = self.value("saved_units_unverified")
        ratio_text = f"that ratio is outside 1 ± {SCALE_RATIO_TOLERANCE:g}"
        if ratio_out and recorded is None:
            return f"{ratio_text} or the saved length units are unverified"
        units_out = self.saved_units_unverified()
        if ratio_out and units_out:
            return f"{ratio_text} and the saved length units are unverified"
        if ratio_out:
            return ratio_text
        if units_out:
            return "the saved length units are unverified"
        return "that ratio is outside tolerance or the saved units are unverified"

    def once_seen_gauge_count(self, classes: Any) -> int | None:
        """Count single-detection D4.2 points only when saved indices and classes exist."""
        diagnostics = self.metadata.get("diagnostics") or {}
        gauge = diagnostics.get("D4.2_gauge_fixed_points") or {}
        indices = gauge.get("indices") if isinstance(gauge, Mapping) else None
        if not isinstance(indices, (list, tuple)) or len(indices) != 3:
            return None
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            return None
        try:
            once_seen = set(classes.one_direction)
            return sum(index in once_seen for index in indices)
        except (AttributeError, TypeError):
            return None

    def shape_copy(self) -> str:
        """Describe the shape change with qualified units and separated observation classes."""
        result = self.derivation
        if isinstance(result, DerivationUnavailable):
            return f"Shape change unavailable: {result.reason}. No geometry is inferred."
        classes = result.point_classes
        if classes is None:
            return "Point classes unavailable; no measurement is inferred."
        ratio = fmt(result.scale_consistency_ratio)
        qualifier = (f" The scale-consistency ratio is {ratio}; this millimetre value is qualified "
                     f"because {self.qualifier_reasons()}."
                     if result.metric_qualifier_required else "")
        several = getattr(result, "several_directions_mean_displacement_mm", None)
        single = len(classes.one_direction)
        if self.refit_headline_shown():
            headline = (f"Points seen from several directions differ from the design you entered by "
                        f"{fmt(several)} mm on average, after the refined shape is turned and "
                        "shifted (not resized) onto the design.")
            # The display alignment is rigid (SCI-1): the solved shape keeps the scale Phase 4
            # hands on, so a scale-consistency ratio outside tolerance qualifies this value
            # like D4.7. Unverified units still affect its mm conversion.
            qualifier = ""
            if self.scale_ratio_out_of_tolerance():
                qualifier += (f" The scale-consistency ratio is {ratio}, outside tolerance, so this "
                              "millimetre value is qualified, as are D4.7 and the camera distances.")
            if self.saved_units_unverified():
                qualifier += (" The saved length units are unverified, so this millimetre value "
                              "is qualified.")
        elif not classes.several_directions:
            headline = ("No point was seen from several directions, so no shape difference is backed "
                        f"by the photos. The average over all seen points is "
                        f"{fmt(result.d47_mean_displacement_mm)} mm, but every one of them was seen "
                        "from one direction only, so their positions are poorly determined.")
        else:
            headline = (f"The estimated shape differs from the design you entered by "
                        f"{fmt(result.d47_mean_displacement_mm)} mm on average over all seen points"
                        + (f"; this average includes the {single} point(s) seen from one direction "
                           "only, whose positions are poorly determined." if single else "."))
        once_seen_gauge = self.once_seen_gauge_count(classes)
        pinned = ("the reference photo only sets the frame the positions are measured in, and the "
                  "final alignment sets the shape's position (see status).")
        gauge_qualifier = {
            1: " One of the three points pyCamSet chose to pin the shape was seen from one direction "
               f"only, so it does not pin the result down; {pinned}",
            2: " Two of the three points pyCamSet chose to pin the shape were seen from one direction "
               f"only, so they do not pin the result down; {pinned}",
            3: " Each of the three points pyCamSet chose to pin the shape was seen from one direction "
               f"only, so it does not pin the result down; {pinned}",
        }.get(once_seen_gauge, "")
        return (f"{headline} Points seen from several directions, "
                "points seen from one direction only, and points not seen are separate classes; not-seen "
                f"points are not estimated.{qualifier}{gauge_qualifier} "
                "The display only turns and shifts the shape: the printed squares keep the size you "
                "entered (that fixes the scale, it is not a measurement of the print), while the "
                "faces themselves are free to sit further in or out than drawn.")

    def shape_detail(self) -> str:
        """L2: name the saved diagnostic and the scale effect behind the plain mean."""
        result = self.derivation
        if isinstance(result, DerivationUnavailable):
            return ""
        classes = getattr(result, "point_classes", None)
        several = getattr(result, "several_directions_mean_displacement_mm", None)
        if classes is None or several is None:
            return "This mean is D4.7, recalculated from saved results."
        seen = len(classes.several_directions) + len(classes.one_direction)
        # SCI2-10: the display keeps D4.7's scale; the two means differ only in which
        # corners they average.
        detail = (f"D4.7 averages all {seen} seen points ({fmt(result.d47_mean_displacement_mm)} mm, "
                  f"recalculated from saved results), including the {len(classes.one_direction)} "
                  "seen from one direction only, whose positions are poorly determined; same scale "
                  "and alignment as the several-direction mean.")
        if self.refit_headline_shown() and getattr(result, "metric_qualifier_required", False):
            detail += (f" The scale-consistency ratio is {fmt(result.scale_consistency_ratio)}; "
                       f"D4.7 is a qualified millimetre value because {self.qualifier_reasons()}.")
        return detail

    def refit_scale_note(self) -> str:
        """L3: what a size-fitted alignment would hide (SCI-1), or "" when unknown.

        The display keeps the printed square size (the scale the exported calibration uses)
        and removes only turn + shift. A free size would shrink every face and draw inward
        corrections the run never estimated, so it is only reported, never drawn.
        """
        scale = finite(self.value("display_similarity_scale"))
        if scale is None or abs(scale - 1.0) < 5e-4:
            return ""
        change = abs(1.0 - scale) * 100
        word = "shrink" if scale < 1 else "grow"
        # ED2-5: plain words; the printed squares fix the size, so a resizing fit is not used.
        return (f"An alignment that may also resize would {word} the whole solved target by "
                f"{change:.1f} % to split the difference between the moved faces and the printed "
                "squares; the printed square size fixes the scale, so it is not used.")

    def refit_scale_row(self) -> tuple[str, str, str] | None:
        """L3 Numbers row for the size-fitted alignment that is *not* used (never D4.5)."""
        scale = finite(self.value("display_similarity_scale"))
        if scale is None:
            return None
        return ("Resizing alignment (not used)", f"×{fmt(scale, 4)}",
                f"would {'shrink' if scale < 1 else 'grow'} the whole solved target by "
                f"{abs(1 - scale) * 100:.1f} % to split the difference between the moved faces "
                "and the printed squares; the display keeps the printed square size (the scale) "
                "and removes only the best turn + shift")

    def class_rows(self) -> list[tuple[str, str, str]]:
        classes = getattr(self.derivation, "point_classes", None)
        if classes is None:
            return [("Point classes", "unavailable", "no display geometry")]
        return [
            ("Several viewing directions", str(len(classes.several_directions)),
             "measured points; display refit; arrows exaggerated"),
            ("Seen once / one direction", str(len(classes.one_direction)), "not exaggerated"),
            ("Not seen", str(len(classes.unseen)), "not estimated"),
        ]

    # -- playback ------------------------------------------------------------------------

    def playback_copy(self, playable: bool) -> str:
        """Name the true Phase 4 objective and state why a trace is unavailable."""
        result = self.derivation
        if isinstance(result, DerivationUnavailable):
            return f"Recorded-step playback unavailable: {result.reason}."
        params = self.metadata.get("params") or {}
        if params.get("record_education") is not True:
            return ("Steps were not recorded for this run. To record steps next time, tick ‘Record solver steps for this run’ "
                    "in this phase's settings before you run it. This run and its calibration are not deficient; "
                    "the steps were simply not requested.")
        if self.trace.get("status") != "available" or not playable:
            return ("Static before/after only. The recorded steps could not be read; "
                    "the run's saved start and result are not connected by an invented path.")
        loss, scale = self.phase4_loss()
        return ("Recorded accepted solver steps; not physical camera motion or target deformation. "
                "The cost axis is the Phase 4 solver objective "
                f"({loss_text(loss, scale)}). The first state is the stored starting point, not an accepted step. "
                "Nothing plays until you press Play or step.")

    def failed_playback_copy(self) -> str:
        """Why a failed run offers no playback, and where its recorded costs are."""
        text = "Playback is not offered: the saved Phase 4 endpoint failed verification. "
        if self.trace.get("status") == "available":
            return text + ("The Numbers table is the text alternative, with recorded step numbers "
                           "and costs.")
        return text + "No readable Phase 4 trace is available for the table."

    def failed_trace_rows(self) -> list[tuple[str, str, str]]:
        """Expose recorded step numbers and costs without claiming failed endpoints."""
        if self.trace.get("status") != "available":
            return [("Recorded steps", "unavailable", "no readable Phase 4 trace")]
        rows: list[tuple[str, str, str]] = []
        for nit, cost in trace_costs(self.trace):
            label = "Starting point (not an accepted step)" if nit == 0 else f"Step {nit}"
            rows.append((label, fmt(cost), "recorded trace; failed run"))
        if not rows:
            return [("Recorded steps", "unavailable", "trace has no usable step/cost records")]
        return rows

    # -- decision ------------------------------------------------------------------------

    def fit_answer(self) -> tuple[str, str]:
        """(chip value, tone) for 'does Phase 4 fit better?'."""
        if not self.usable_run():  # SCI5-1: no verdict on a held or failed run
            return "Not judged", "warn"
        direction = self.fit_direction()
        return {"lower": ("Yes, lower", "good"), "higher": ("No, higher", "bad"),
                "same": ("No change", "warn")}.get(direction, ("Cannot check", "warn"))

    def shape_answer(self) -> tuple[str, str]:
        """(chip value, tone) for 'is the shape change backed by enough views?'."""
        if not self.usable_run():  # SCI5-1: a held run does not back or refute a shape
            return "Not judged", "warn"
        classes = getattr(self.derivation, "point_classes", None)
        if classes is None:
            return "Cannot judge", "warn"
        several = len(classes.several_directions)
        total = several + len(classes.one_direction) + len(classes.unseen)
        if several and not (classes.one_direction or classes.unseen):
            return f"Backed {several}/{total}", "good"
        if several:
            return f"Partly {several}/{total}", "warn" if several < 0.8 * total else "good"
        return "Not backed", "bad"

    def decision_copy(self) -> str:
        """Answer A1's Phase 4 question from this run's own values, or say why it cannot."""
        before = self.phase3_end_px()
        lines = ["Does Phase 4 fit your points better, and are the target-shape differences backed "
                 "by enough views to be believable?"]
        after = self.value("d43_final_px")
        direction = self.fit_direction(before)
        classes = getattr(self.derivation, "point_classes", None)
        if not self.usable_run():
            # SCI5-1: a held or failed run gets its neutral numbers and no verdict (no "yes",
            # no "backed", no "lower"): the Decision row must not read as an answer.
            held = [lines[0], "Answer: not judged, this run is not a usable calibration."]
            held.append(f"Fit: the average point error was {fmt(before)} px at the end of Phase 3 and "
                        f"{fmt(after)} px after Phase 4; not judged on this run.")
            if classes is None:
                held.append("Shape: not judged on this run; the point classes are unavailable.")
            else:
                held.append(f"Shape: not judged on this run; {len(classes.several_directions)} point(s) "
                            f"were seen from several directions, {len(classes.one_direction)} from one "
                            f"direction only, {len(classes.unseen)} not seen.")
            held.append(self.plain_comparison())
            return "\n".join(held)
        fit_answer = {"lower": "Fit: yes, the average point error is lower than Phase 3's",
                      "higher": "Fit: no, the average point error is higher than Phase 3's",
                      "same": "Fit: no, the average point error is not lower than Phase 3's",
                      }.get(direction, "Fit: cannot be checked for this run")
        if classes is None:
            shape_answer = "Shape: cannot be judged, the point classes are unavailable"
        elif classes.several_directions and not (classes.one_direction or classes.unseen):
            shape_answer = (f"Shape: backed, all {len(classes.several_directions)} point(s) were seen from "
                            "several directions")
        elif classes.several_directions:
            total = (len(classes.several_directions) + len(classes.one_direction)
                     + len(classes.unseen))
            shape_answer = (f"Shape: partly backed, only the {len(classes.several_directions)} of {total} "
                            "points seen from several directions back a shape difference; the rest are "
                            "not backed")
        else:
            shape_answer = "Shape: not backed, no point was seen from several directions"
        lines.append(f"Answer: {fit_answer}. {shape_answer}.")
        if direction is None:
            lines.append("The fit comparison cannot be checked for this run: the Phase 3 result or the "
                         "recalculated Phase 4 result is unavailable or unconfirmed. In Phase 3, open "
                         "‘Diagnostics ▼’, then press ‘Phase 4 - Self-Calibration ▶’ to start Phase 4 from "
                         "the Phase 3 result you intend to use; run Phase 4 again.")
        else:
            relation = {"lower": "lower than", "higher": "higher than", "same": "the same as"}[direction]
            lines.append(f"Fit: the average point error was {fmt(before)} px at the end of Phase 3 and "
                         f"{fmt(after)} px after Phase 4, {relation} Phase 3's. A lower error alone does "
                         "not show a more accurate calibration.")
        if classes is None:
            lines.append("Shape: the point classes are unavailable, so no shape difference is judged.")
        else:
            several = len(classes.several_directions)
            single = len(classes.one_direction)
            lines.append(f"Shape: {several} point(s) were estimated from several directions; {single} were "
                         f"seen from one direction only, so their distance along the line of sight is poorly "
                         f"determined; {len(classes.unseen)} were not seen and stay at the design. Only the "
                         "points seen from several directions back a shape difference.")
            if single or classes.unseen:
                lines.append("What to do next: to back more of the shape, take photos that show the target "
                             "from other directions, then run the phases again from Phase 1.")
        lines.append(self.plain_comparison())
        return "\n".join(lines)

    def fit_credit(self) -> str:
        """One sentence saying what the Phase 3 -> 4 error drop can be credited to (ED-8,
        SCI-4): the shape, when the start, scoring rule and held parameters all match;
        otherwise the first sentence of :meth:`plain_comparison` (its honest hedge)."""
        if not self.usable_run():  # SCI5-1: a held run's drop is not credited to any part
            return "Not credited to any cause: this run is not a usable calibration."
        if self.fit_direction() == "lower" and self.state.confirmed:
            p3, s3 = self.phase3_loss()
            p4, s4 = self.phase4_loss()
            if not self.loss_differs(p3, s3, p4, s4) and not self.setting_differences():
                # ED2-1: the corners were freed, but the lenses and camera turns kept being
                # refined too, and they changed with the shape; never credit the shape alone.
                return ("Phase 4 freed the corners and kept refining the lenses and cameras; "
                        "the drop comes from the shape together with re-tuned lenses.")
        text = self.plain_comparison()
        return text.split(". ")[0].rstrip(".") + "."

    def loss_detail(self) -> str:
        """L2: both objectives, plus the attribution note when it adds something."""
        p3, s3 = self.phase3_loss()
        p4, s4 = self.phase4_loss()
        note = self.comparison_note()
        return (f"Phase 3's loss: {loss_text(p3, s3)}. Phase 4's loss: {loss_text(p4, s4)}."
                + ("" if note in self.plain_comparison() else f" {note}"))


def step_rows(frames: Sequence[Any], change: Any = None) -> list[tuple[str, str, str]]:
    """Text alternative for derive_phase4's recorded frames: solver step, cost and the largest
    rig change (``change(left, right)``), plus the step numbers thinning left out."""
    frames = tuple(frames)
    if not frames:
        return [("Playback", "unavailable", "no buildable recorded frames")]
    rows: list[tuple[str, str, str]] = []
    previous = None
    for index, frame in enumerate(frames):
        label = "Starting point (not an accepted step)" if index == 0 else f"Step {frame.nit}"
        if previous is None:
            move = "—"
        elif getattr(previous, "status", None) != "available" or frame.status != "available":
            move = "unavailable"
        else:
            move = change(previous, frame) if change is not None else "unavailable"
        rows.append((label, fmt(frame.cost), move))
        previous = frame
    nits = [int(frame.nit) for frame in frames]
    missing = [step for left, right in zip(nits, nits[1:]) for step in range(left + 1, right)]
    if missing:
        rows.append(("Thinning", "steps not kept", ", ".join(map(str, missing))))
    return rows


def trace_costs(trace: Mapping[str, Any]) -> list[tuple[int, float]]:
    """(nit, cost) of every finite recorded step in a trace (any run status)."""
    out: list[tuple[int, float]] = []
    for record in trace.get("records", ()) or ():
        if not isinstance(record, Mapping) or record.get("kind") != "accepted_step":
            continue
        payload = record.get("payload")
        steps = payload.get("steps", ()) if isinstance(payload, Mapping) else ()
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            nit = step.get("nit")
            if isinstance(nit, bool) or not isinstance(nit, int):
                continue
            cost = finite(step.get("cost"))
            if cost is None:
                continue
            out.append((nit, cost))
    return out


# --------------------------------------------------------------------------------------
# Visual data


@dataclass(frozen=True)
class CameraPose:
    """A camera as drawn: world-to-camera rotation ``R`` and translation ``t`` in the target
    frame at the reference photo. ``placed`` is True when the position is a display placement
    (telecentric), not a solved position."""

    name: str
    index: int
    R: np.ndarray
    t: np.ndarray
    telecentric: bool
    placed: bool = False

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass(frozen=True)
class ExcludedImage:
    """An image the solve left out (no usable target pose), with who saw it."""

    image: int
    cameras: tuple[int, ...]
    corners: tuple[int, ...]

    @property
    def total(self) -> int:
        return int(sum(self.corners))


@dataclass(frozen=True)
class FaceMove:
    """One face of a multi-face target, fitted as a rigid plate (design -> refined).

    Lengths in mm (None without verified units: then in target units, see ``unit``).
    ``offset`` is the move of the face centre along the design's outward normal (+ = further
    out than designed), ``shift`` the in-plane slide, ``tilt_deg`` the angle between the
    design and refined face normals; ``warp_*`` is what is left inside the face once the
    plate move is removed (corner by corner).
    """

    face: int                    # 0-based face index (shown 1-based)
    points: tuple[int, ...]      # every point of the face
    fitted: int                  # several-direction corners the plate was fitted to
    offset: float
    shift: float
    tilt_deg: float
    warp_mean: float
    warp_max: float
    normal: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(3))
    centre: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(3))
    #: the solved plate's outward normal (the design normal turned by the plate's rotation)
    refined_normal: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(3))
    #: the plate's rigid move (design -> solved): ``x @ R.T + t`` (target units)
    R: np.ndarray = field(repr=False, default_factory=lambda: np.eye(3))
    t: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(3))


@dataclass(frozen=True)
class FacePair:
    """Two opposite faces of a cube target (SCI2-6, ED2-5): how much further apart the solved
    plates sit than the design's. Unlike each face's own offset, this total does not depend
    on how the solved shape is turned and shifted onto the design."""

    faces: tuple[int, int]       # 0-based face indices
    design: float                # design distance between the two face planes (mm, or units)
    change: float                # solved minus design distance (+ = wider than designed)
    angle_deg: float             # how far the two solved plates are from parallel


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0 or nv == 0:
        return math.nan
    return math.degrees(math.acos(float(np.clip((u @ v) / (nu * nv), -1.0, 1.0))))


def face_pairs(faces: Sequence[FaceMove], to_mm: float = 1.0) -> tuple[FacePair, ...]:
    """Opposite faces (design normals anti-parallel) and their alignment-free width change
    (the sum of the two outward offsets; ``design`` is ``* to_mm`` like the offsets)."""
    pairs = []
    used: set[int] = set()
    for a in faces:
        if a.face in used:
            continue
        for b in faces:
            if b.face == a.face or b.face in used or float(a.normal @ b.normal) > -0.9:
                continue
            design = abs(float((a.centre - b.centre) @ a.normal)) * to_mm
            pairs.append(FacePair((a.face, b.face), design, a.offset + b.offset,
                                  _angle_deg(a.refined_normal, -b.refined_normal)))
            used.update((a.face, b.face))
            break
    return tuple(pairs)


def face_squareness(faces: Sequence[FaceMove]) -> tuple[tuple[int, int, float], ...]:
    """Neighbouring faces (design normals at right angles): (face a, face b, solved angle
    minus 90°). Alignment-free like :func:`face_pairs` (SCI2-6)."""
    out = []
    for i, a in enumerate(faces):
        for b in faces[i + 1:]:
            if abs(float(a.normal @ b.normal)) < 0.1:
                out.append((a.face, b.face,
                            _angle_deg(a.refined_normal, b.refined_normal) - 90.0))
    return tuple(out)


def _rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rotation R and shift t with ``dst ~ src @ R.T + t`` (Kabsch)."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    u, _s, vt = np.linalg.svd((dst - md).T @ (src - ms))
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt)) or 1.0
    rotation = u @ d @ vt
    return rotation, md - ms @ rotation.T


def target_faces(point_data: Any) -> tuple[np.ndarray, ...]:
    """Point indices of each flat face of a multi-face target (cube), or () when the target
    is not made of several flat faces.

    A face is a slice of ``point_data`` along its first axis that is planar with a 2D spread;
    at least two faces must face different ways (a single board, or a board stored as rows
    of points, gives ())."""
    pd = np.asarray(point_data, dtype=float)
    if pd.ndim < 3 or pd.shape[0] < 2:
        return ()
    blocks = pd.reshape(pd.shape[0], -1, 3)
    per = blocks.shape[1]
    faces, normals = [], []
    for k, block in enumerate(blocks):
        ok = np.all(np.isfinite(block), axis=1)
        pts = block[ok]
        if len(pts) < 4:
            return ()
        s = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
        if s[0] <= 0 or s[1] < 0.1 * s[0] or s[2] > 0.02 * s[0]:
            return ()
        _u, _s, vt = np.linalg.svd(pts - pts.mean(axis=0))
        normals.append(vt[2])
        faces.append(k * per + np.flatnonzero(ok))
    normals = np.asarray(normals)
    if np.all(np.abs(normals @ normals[0]) > 0.9):
        return ()
    return tuple(np.asarray(f, dtype=int) for f in faces)


def decompose_faces(nominal: np.ndarray, refined: np.ndarray, several: Sequence[int],
                    faces: Sequence[np.ndarray], *, to_mm: float = 1.0,
                    min_fitted: int = 4) -> tuple[tuple[FaceMove, ...], np.ndarray]:
    """Split the shape change into whole-face moves and the change left inside each face.

    Each face is fitted as a rigid plate to its several-direction corners (only those back a
    shape difference). Returns the faces and ``plate`` (N x 3, target units): each point's
    position after its face's plate move (nan for points on no fitted face); the change
    left inside a face is ``refined - plate``. Lengths in the FaceMove are ``* to_mm``.
    """
    nominal = np.asarray(nominal, dtype=float)
    refined = np.asarray(refined, dtype=float)
    plate = np.full_like(nominal, np.nan)
    chosen = set(int(i) for i in several)
    all_face = np.concatenate([np.asarray(f, dtype=int) for f in faces]) if faces else np.zeros(0, int)
    if not len(all_face):
        return (), plate
    middle = np.nanmean(nominal[all_face], axis=0)
    moves = []
    for k, idx in enumerate(faces):
        idx = np.asarray(idx, dtype=int)
        use = np.array([i for i in idx if i in chosen and np.all(np.isfinite(refined[i]))],
                       dtype=int)
        if len(use) < min_fitted:
            continue
        R, t = _rigid_fit(nominal[use], refined[use])
        centre = nominal[idx].mean(axis=0)
        _u, _s, vt = np.linalg.svd(nominal[idx] - centre)
        normal = vt[2] * (np.sign(vt[2] @ (centre - middle)) or 1.0)
        moved = centre @ R.T + t
        delta = moved - centre
        offset = float(delta @ normal)
        shift = float(np.linalg.norm(delta - offset * normal))
        tilt = math.degrees(math.acos(float(np.clip((R @ normal) @ normal, -1.0, 1.0))))
        plate[idx] = nominal[idx] @ R.T + t
        warp = np.linalg.norm(refined[use] - plate[use], axis=1)
        moves.append(FaceMove(k, tuple(int(i) for i in idx), int(len(use)), offset * to_mm,
                              shift * to_mm, tilt, float(warp.mean()) * to_mm,
                              float(warp.max()) * to_mm, normal, centre, R @ normal, R, t))
    return tuple(moves), plate


def telecentricity_per_metre(camset: Any, mm_per_unit: float | None) -> dict[int, float]:
    """Each telecentric camera's depth-dependent magnification ε in 1/m (SCI-5); cameras of
    other models are left out. Empty without verified units."""
    out: dict[int, float] = {}
    if camset is None or mm_per_unit is None:
        return out
    try:
        from pyCamSet.cameras.telecentric_camera import TelecentricCamera
        for index in range(camset.get_n_cams()):
            camera = camset[index]
            if isinstance(camera, TelecentricCamera):
                value = finite(getattr(camera, "telecentricity", None))
                if value is not None:
                    out[index] = value * 1000.0 / float(mm_per_unit)
    except Exception:  # noqa: BLE001 - an unreadable camset just shows nothing
        return {}
    return out


def distortion_shifts(camset: Any) -> dict[int, float]:
    """Largest shift (px) each camera's fitted lens distortion makes inside its own photo,
    measured as Phase 3's "Lens tuning" stage measures it (the same grid, about the point
    the lens model bends around). {} when the camset cannot be read."""
    out: dict[int, float] = {}
    if camset is None:
        return out
    try:
        from pyCamSet.cameras.telecentric_camera import TelecentricCamera
        from pyCamSet.workflow.education_visual_p3 import _distortion_shift
        for index, name in enumerate(camset.get_names()):
            camera = camset[name]
            size = tuple(int(v) for v in np.asarray(camera.res, dtype=float).ravel()[:2])
            coefs = np.asarray(getattr(camera, "distortion_coefs", ()), dtype=float).ravel()
            if not all(size):
                continue
            out[index] = (0.0 if not np.any(coefs) else
                          float(_distortion_shift(camera, size,
                                                  isinstance(camera, TelecentricCamera))))
    except Exception:  # noqa: BLE001 - context only; an unreadable lens shows nothing
        return {}
    return out


@dataclass(frozen=True)
class GateCheck:
    """One traffic-light check of the quality gate."""

    key: str
    title: str
    tone: str      # "good" | "warn" | "bad" | "unknown"
    value: str
    line: str
    #: what would go wrong if this check failed (ED-8): one short line under the card
    guard: str = ""


def solver_stop_words(message: Any) -> str:
    """The solver's stop message in plain words ("ftol reached" -> "stopped: the error
    stopped falling"); unknown messages are returned as recorded."""
    text = str(message or "").strip()
    low = text.casefold()
    if "ftol" in low or "`ftol`" in low:
        return "stopped: the error stopped falling"
    if "xtol" in low:
        return "stopped: steps became tiny"
    if "gtol" in low:
        return "stopped: no downhill direction left"
    if "nfev" in low or "maximum number" in low:
        return "stopped at the step limit"
    return text or "no message recorded"


@dataclass(frozen=True)
class SolverState:
    """One solver state replayed through the rebuilt handler (target frame at the reference
    photo)."""

    nit: int
    cost: float
    cameras: tuple[CameraPose, ...]
    shape: np.ndarray            # solved target points (gauge-transformed), target units
    display_shape: np.ndarray | None   # similarity-aligned to the design (display only)
    pose_centres: np.ndarray     # target centre in each posed image, (M, 3)
    norms: np.ndarray            # per-observation reprojection error (px)
    per_camera: dict[int, float]
    mean_px: float
    point_error: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))  # px
    params: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))


@dataclass(frozen=True)
class Playback:
    """Recorded solver steps, or why there are none."""

    status: str                  # "available" | "not recorded" | "unavailable" | "cost only"
    reason: str
    states: tuple[SolverState, ...] = ()
    costs: tuple[tuple[int, float], ...] = ()
    #: (n_states, 4) share of each parameter block's total movement made at each step
    #: (lens, camera rotation/position, target poses, target points); row 0 is zeros.
    block_share: np.ndarray | None = None
    block_sizes: tuple[int, int, int, int] | None = None

    @property
    def playable(self) -> bool:
        return self.status == "available" and len(self.states) >= 2

    @property
    def missing_steps(self) -> list[int]:
        nits = [state.nit for state in self.states]
        return [step for left, right in zip(nits, nits[1:]) for step in range(left + 1, right)]


@dataclass
class ErrorTable:
    """Per-observation errors of one solver state grouped by camera / image / target point."""

    norms: np.ndarray
    cameras: np.ndarray
    images: np.ndarray
    points: np.ndarray
    n_points: int

    @property
    def mean_px(self) -> float:
        return float(self.norms.mean()) if self.norms.size else math.nan

    def per_camera(self) -> dict[int, float]:
        return {int(c): float(self.norms[self.cameras == c].mean())
                for c in np.unique(self.cameras)}

    def per_image(self) -> dict[int, float]:
        return {int(i): float(self.norms[self.images == i].mean()) for i in np.unique(self.images)}

    def by_camera_image(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """camera -> (images, mean error of that camera in each image)."""
        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for cam in np.unique(self.cameras):
            mask = self.cameras == cam
            imgs = self.images[mask]
            uniq = np.unique(imgs)
            means = np.array([self.norms[mask][imgs == i].mean() for i in uniq])
            out[int(cam)] = (uniq.astype(int), means)
        return out

    def per_point(self) -> np.ndarray:
        """Mean error per target point (nan for points never observed)."""
        sums = np.bincount(self.points, weights=self.norms, minlength=self.n_points)
        counts = np.bincount(self.points, minlength=self.n_points)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


@dataclass
class Phase4Visual:
    """Everything the five Phase 4 stages draw for one run (see module docstring)."""

    status: str
    reason: str
    camera_names: tuple[str, ...] = ()
    telecentric: tuple[bool, ...] = ()
    unit: str = "target units"                 # "mm" when verified
    mm_per_unit: float | None = None
    target_size: float = 1.0                   # bounding-box diagonal of the design, target units
    printed: np.ndarray | None = None          # design points, target units
    gauge_indices: tuple[int, ...] = ()
    #: the start (Phase 3's result as Phase 4 began) and the saved result, replayed
    start: SolverState | None = None
    final: SolverState | None = None
    start_label: str = ""
    start_errors: ErrorTable | None = None
    final_errors: ErrorTable | None = None
    errors_reason: str = ""
    excluded: tuple[ExcludedImage, ...] = ()
    coverage: np.ndarray | None = None         # (n_images, n_cameras) corners seen
    n_images: int = 0
    phase3_camera_mean: dict[int, float] | None = None   # stored D3.12 (Phase 3's own record)
    phase3_camera_source: str = ""
    phase4_camera_mean: dict[int, float] | None = None   # stored D4.12
    checks: tuple[GateCheck, ...] = ()
    notices: tuple[str, ...] = ()
    blocking: tuple[str, ...] = ()
    nullity: Mapping[str, Any] | None = None
    #: target-shape correction (display fit minus design), target units
    refined: np.ndarray | None = None
    correction: np.ndarray | None = None
    ray_separation_deg: np.ndarray | None = None
    several: tuple[int, ...] = ()
    one_direction: tuple[int, ...] = ()
    unseen: tuple[int, ...] = ()
    exaggeration: int = 1
    #: whole-face (rigid plate) moves of a multi-face target and each point's position after
    #: its face's plate move (SCI-1 / ED-7); () / None for single-board targets
    faces: tuple[FaceMove, ...] = ()
    plate: np.ndarray | None = None
    #: the solver's internal gauge: corners held during the solve (D4.2) and the axis of the
    #: third one held in one coordinate only (3 + 3 + 1 numbers)
    gauge_axis: int | None = None
    #: what a size-fitted alignment would scale by (reported, never drawn)
    similarity_scale: float | None = None
    #: unknowns per parameter block (lens, camera turns/positions, target poses, corners)
    block_sizes: tuple[int, int, int, int] | None = None
    loss: str = "linear"
    #: depth-dependent magnification ε (1/m) of telecentric cameras: Phase 3 and Phase 4
    telecentricity_p3: dict[int, float] = field(default_factory=dict)
    telecentricity_p4: dict[int, float] = field(default_factory=dict)
    #: largest shift (px) each camera's fitted lens distortion makes in its photo, Phase 3 and
    #: Phase 4 (ED2-4 / SCI2-1: the lens-vs-shape trade-off); {} when not computable
    distortion_p3: dict[int, float] = field(default_factory=dict)
    distortion_p4: dict[int, float] = field(default_factory=dict)
    #: "swap one part back" (ED2-1, SCI7-1): the mean error (px) of the saved Phase 4 result
    #: with one part put back to Phase 3 and the other kept at Phase 4: "lens_swap" (Phase 3's
    #: intrinsics) and "shape_swap" (Phase 3's corner positions). Each is the lower of two
    #: photo-pose re-fits ("*_p4poses", "*_p3poses"). One frame, no gauge or size fit. Nothing
    #: is re-solved: each is one pose re-fit of the solve's own error function.
    swap_px: dict[str, float] = field(default_factory=dict)
    _swap_source: Any = field(default=None, repr=False)
    #: What the result is for (the closing stage): µm per pixel, ray gaps, one example corner
    closing: ClosingSummary | None = field(default=None, repr=False)
    _closing_source: Any = field(default=None, repr=False)
    _playback: Playback | None = field(default=None, repr=False)
    _replay: Any = field(default=None, repr=False)
    _precision: dict[str, Any] | None = field(default=None, repr=False)

    def swap_results(self) -> dict[str, float]:
        """The swap test (``swap_px``), re-fitted on first use and kept on this object."""
        source, self._swap_source = self._swap_source, None
        if source is not None:
            self.swap_px = swap_errors(*source)
        return self.swap_px

    def closing_results(self) -> ClosingSummary | None:
        """The closing stage's ray and µm numbers (about a second), computed on first use."""
        source, self._closing_source = self._closing_source, None
        if source is not None:
            try:
                self.closing = closing_summary(source[0], source[1], source[2], source[3],
                                               file=source[4])
            except Exception:  # noqa: BLE001 - the stage then says it could not be computed
                self.closing = None
        return self.closing

    @property
    def available(self) -> bool:
        return self.status == "available"

    @property
    def n_cameras(self) -> int:
        return len(self.camera_names)

    def correction_mm(self) -> np.ndarray | None:
        """|correction| per point in mm (None without verified units)."""
        if self.correction is None or self.mm_per_unit is None:
            return None
        return np.linalg.norm(self.correction, axis=1) * self.mm_per_unit

    def correction_display(self) -> tuple[np.ndarray | None, str]:
        """|correction| per point in a readable unit: (values, "µm"|"mm"|"target units")."""
        if self.correction is None:
            return None, self.unit
        size = np.linalg.norm(self.correction, axis=1)
        if self.mm_per_unit is None:
            return size, "target units"
        unit, factor = length_unit(size[list(self.several)] * self.mm_per_unit
                                   if self.several else size * self.mm_per_unit)
        return size * self.mm_per_unit * factor, unit

    def playback(self) -> Playback:
        """The recorded solver steps (replayed lazily, once)."""
        if self._playback is None:
            self._playback = Playback("unavailable", "playback was not prepared for this run")
        return self._playback

    def lengths_in(self, unit: str) -> float | None:
        """Factor from target units to ``unit`` ("µm" or "mm"); None without verified units."""
        if self.mm_per_unit is None:
            return None
        return self.mm_per_unit * (1000.0 if unit == "µm" else 1.0)

    def face_summary(self) -> dict[str, float] | None:
        """Ranges over the fitted faces (mm): offset min/max, tilt max, inside-face mean/max
        (the inside mean is over every fitted corner, not a mean of face means)."""
        if not self.faces or self.plate is None or self.refined is None:
            return None
        to_mm = self.mm_per_unit or 1.0
        fitted = [i for f in self.faces for i in f.points
                  if i in set(self.several) and np.all(np.isfinite(self.plate[i]))]
        inside = np.linalg.norm(self.refined[fitted] - self.plate[fitted], axis=1) * to_mm
        return {"offset_min": min(f.offset for f in self.faces),
                "offset_max": max(f.offset for f in self.faces),
                "tilt_max": max(f.tilt_deg for f in self.faces),
                "inside_mean": float(inside.mean()) if inside.size else math.nan,
                "inside_max": float(inside.max()) if inside.size else math.nan}

    def pairs(self) -> tuple[FacePair, ...]:
        """Opposite faces and how much wider (mm, or target units) the solved cube is across
        them than designed: independent of the display alignment (SCI2-6)."""
        return face_pairs(self.faces, self.mm_per_unit or 1.0)

    def squareness(self) -> tuple[tuple[int, int, float], ...]:
        """(face a, face b, solved angle - 90°) for neighbouring faces (alignment-free)."""
        return face_squareness(self.faces)

    def freedom_drop(self) -> float | None:
        """How much (fraction) extra freedom alone could lower the RMS error by fitting noise:
        1 - sqrt((m - p) / m), m measured numbers, p the effective unknowns (rank) (SCI2-9)."""
        nullity = self.nullity or {}
        shape = nullity.get("jacobian_shape") or ()
        rank = nullity.get("rank")
        if len(shape) != 2 or not isinstance(rank, (int, np.integer)) or int(shape[0]) <= 0:
            return None
        m, p = int(shape[0]), int(rank)
        if p >= m:
            return None
        return 1.0 - math.sqrt((m - p) / m)

    def noise_3d(self) -> tuple[float | None, float | None]:
        """(1σ 3D spread of one corner, the mean 3D distance pure noise of that size gives),
        mm, from the per-coordinate formal precision (SCI2-8): σ3 = √3 σ, mean = σ3 √(8/3π)."""
        sd, _how = self.corner_precision()
        if sd is None:
            return None, None
        sd3 = math.sqrt(3.0) * sd
        return sd3, sd3 * math.sqrt(8.0 / (3.0 * math.pi))

    def corner_precision(self) -> tuple[float | None, str]:
        """Formal 1σ precision of the solved corner coordinates (median, mm) and how it was
        computed, or (None, why not); see :meth:`jacobian_summary`."""
        summary = self.jacobian_summary()
        return summary.get("precision_mm"), str(summary.get("precision_note", ""))

    def jacobian_summary(self) -> dict[str, Any]:
        """What the solution's Jacobian says, computed once (SCI-3 context):

        * ``precision_mm`` / ``precision_note``: formal 1σ precision of the corner coordinates
          (median), σ²(JᵀJ)⁺ at the saved result with σ from its own residual and the
          recorded rank. Only for a plain (linear) loss: a robust loss reweights residuals.
        * ``weak_shares``: where the 8 weakest non-null directions of the *column-normalised*
          JᵀJ lie, as shares of {"lenses", "camera turns", "photo poses", "corners"} (SCI2-3:
          on the raw matrix the lens columns are tiny by their units, so the raw share says
          nothing about the shape).
        """
        if self._precision is not None:
            return dict(self._precision)
        out: dict[str, Any] = {"precision_mm": None,
                               "precision_note": "the saved result was not replayed"}
        try:
            replay = self._replay
            if replay is None or self.final is None:
                raise LookupError
            import scipy.sparse as sp
            x = np.asarray(self.final.params, dtype=np.float64)
            if x.size > MAX_DENSE_SUMMARY_PARAMS:
                # Dense JᵀJ and two eigendecompositions cost O(n²) memory and O(n³) time.
                out["precision_note"] = (
                    f"not computed: {x.size} parameters is more than the "
                    f"{MAX_DENSE_SUMMARY_PARAMS} this summary decomposes densely")
                raise LookupError
            jac = sp.csr_matrix(replay.handler.make_loss_jac(1)(x.copy()))
            normal = (jac.T @ jac).toarray()
            rank = (self.nullity or {}).get("rank")
            rank = int(rank) if isinstance(rank, (int, np.integer)) else int(
                np.linalg.matrix_rank(normal, hermitian=True))
            values, vectors = np.linalg.eigh(normal)
            null = max(0, normal.shape[0] - rank)
            diag = np.sqrt(np.clip(np.diag(normal), 1e-300, None))
            _vals_n, vecs_n = np.linalg.eigh(normal / diag[:, None] / diag[None, :])
            weak = vecs_n[:, null:null + 8]
            if weak.size:
                bounds = (0,) + tuple(replay.blocks)
                names = ("lenses", "camera turns", "photo poses", "corners")
                out["weak_shares"] = {
                    name: float(np.mean(np.sum(weak[bounds[b]:bounds[b + 1]] ** 2, axis=0)))
                    for b, name in enumerate(names)}
            if str(self.loss or "linear") != "linear":
                out["precision_note"] = f"not computed for the robust loss ({self.loss})"
                raise LookupError
            if self.mm_per_unit is None:
                out["precision_note"] = "not computed: the saved length units are unverified"
                raise LookupError
            residual = np.asarray(replay.loss(x.copy()), dtype=np.float64)[:jac.shape[0]]
            dof = jac.shape[0] - rank
            if dof <= 0:
                raise LookupError
            sigma2 = float(residual @ residual) / dof
            keep = values > values.max() * 1e-15
            inv_diag = (vectors[:, keep] ** 2) @ (1.0 / values[keep])
            lo, hi = replay.blocks[2], replay.blocks[3]
            sd = np.sqrt(np.clip(sigma2 * inv_diag[lo:hi], 0, None)) * self.mm_per_unit
            sd = sd[np.isfinite(sd) & (sd > 0)]
            if not sd.size:
                raise LookupError
            out["precision_mm"] = float(np.median(sd))
            out["precision_note"] = (f"formal 1σ, median over {sd.size} corner coordinates: "
                                     f"σ²(JᵀJ)⁺ at the saved result, σ = {math.sqrt(sigma2):.3g} "
                                     "px per image coordinate")
        except LookupError:
            pass
        except Exception as exc:  # noqa: BLE001 - context only; say why it is missing
            out["precision_note"] = f"could not be computed: {exc}"
        self._precision = out
        return dict(out)


# --------------------------------------------------------------------------------------
# Replay through one rebuilt handler


class Replay:
    """One rebuilt Phase 4 handler that turns a parameter vector into drawable state.

    ``education_derive._phase4_rebuild`` builds and seeds it (``get_initial_params()``), as
    derive_phase4 does, so a replayed cost is the solve's own cost.
    """

    def __init__(self, camset: Any, diagnostics: Mapping[str, Any], *, path: Any = None) -> None:
        from pyCamSet.cameras.telecentric_camera import TelecentricCamera
        from pyCamSet.workflow.education_derive import _phase4_handler
        # With the saved file's path the seeded handler comes from the derivation cache.
        self.handler = _phase4_handler(path, camset, diagnostics)
        self.rows = np.asarray(self.handler._flat_detections(), dtype=np.float64)
        self.loss = self.handler.make_loss_fun(1)
        self.names = tuple(str(n) for n in self.handler.cam_names)
        self.cameras = [self.handler.camset[name] for name in self.names]
        self.telecentric = tuple(isinstance(c, TelecentricCamera) for c in self.cameras)
        self.n_points = int(np.prod(self.handler.target.point_data.shape[:-1]))
        missing = getattr(self.handler, "missing_poses", None)
        self.missing = (np.asarray(missing, dtype=bool).reshape(-1) if missing is not None
                        else np.zeros(0, dtype=bool))
        self.fixed_pose = int((getattr(self.handler, "problem_opts", {}) or {}).get("fixed_pose", 0))
        primitive = self.handler.bundlePrimitive
        self.blocks = (int(primitive.intr_end), int(primitive.extr_end), int(primitive.pose_end),
                       int(getattr(primitive, "bdpt_end", primitive.pose_end)))

    def errors(self, params: np.ndarray) -> ErrorTable:
        residuals = np.asarray(self.loss(np.asarray(params, dtype=np.float64).copy()),
                               dtype=np.float64)
        norms = np.linalg.norm(residuals[:2 * len(self.rows)].reshape(-1, 2), axis=1)
        return ErrorTable(norms, self.rows[:, 0].astype(int), self.rows[:, 1].astype(int),
                          self.rows[:, 2].astype(int), self.n_points)

    def geometry(self, params: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """World-to-camera 4x4 per camera, target-to-world 4x4 per image, target points;
        all gauge-transformed exactly as ``get_camset``/``get_updated_target`` do."""
        from pyCamSet.optimisation.template_handler import _extrinsic_from_params
        from pyCamSet.utils.general_utils import make_4x4h_tform
        primitive = self.handler.bundlePrimitive
        names = ("intr", "extr", "poses", "bundle_pts")
        saved = {n: np.asarray(getattr(primitive, n)).copy() for n in names if hasattr(primitive, n)}
        try:
            values = primitive.return_bundle_primitives(np.asarray(params, dtype=np.float64).copy())
            proj, extr, poses, pts = self.handler.apply_gauge_transform(
                *[np.asarray(v, dtype=np.float64).copy() for v in values])
        finally:
            for n, v in saved.items():  # the primitive's arrays are shared with the handler
                getattr(primitive, n)[...] = v
        world_to_cam = [np.asarray(_extrinsic_from_params(
            np.asarray(extr[i], dtype=np.float64),
            np.asarray(self.cameras[i].extrinsic, dtype=np.float64)), dtype=np.float64)
            for i in range(len(self.cameras))]
        target_to_world = np.stack([make_4x4h_tform(p[:3], p[3:]) for p in
                                    np.asarray(poses, dtype=np.float64)])
        return world_to_cam, target_to_world, np.asarray(pts, dtype=np.float64).reshape(-1, 3)

    def state(self, nit: int, cost: float, params: np.ndarray, *, display_size: float,
              nominal: np.ndarray, measured: np.ndarray | None) -> SolverState:
        from pyCamSet.workflow.education_derive import _phase4_display_refit
        world_to_cam, target_to_world, pts = self.geometry(params)
        ref = target_to_world[self.fixed_pose]
        # Express everything in the target frame at the reference photo.
        cams = []
        centre = np.nanmean(nominal, axis=0)
        for i, (name, ext) in enumerate(zip(self.names, world_to_cam)):
            cam_from_target = ext @ ref
            R = cam_from_target[:3, :3]
            if self.telecentric[i]:
                position = centre - TELECENTRIC_DISPLAY_DISTANCE * display_size * R[2]
                cams.append(CameraPose(name, i, R, -R @ position, True, placed=True))
            else:
                cams.append(CameraPose(name, i, R, cam_from_target[:3, 3], False))
        inv_ref = np.linalg.inv(ref)
        posed = np.flatnonzero(~self.missing) if self.missing.size else np.arange(len(target_to_world))
        local_centre = np.append(np.nanmean(pts, axis=0), 1.0)
        pose_centres = np.array([(inv_ref @ target_to_world[i] @ local_centre)[:3] for i in posed])
        # This state's shape turned and shifted (never resized) onto the design, on the
        # several-direction corners: the same alignment derive_phase4 uses (SCI-1). Each
        # state gets its own turn + shift; the playback says so (SCI-17).
        display = None
        if measured is not None:
            try:
                display, _ = _phase4_display_refit(nominal, pts, measured)
            except ValueError:
                display = None
        table = self.errors(params)
        return SolverState(int(nit), float(cost), tuple(cams), pts, display, pose_centres,
                           table.norms, table.per_camera(), table.mean_px,
                           point_error=table.per_point(),
                           params=np.asarray(params, dtype=np.float64))


# --------------------------------------------------------------------------------------
# Quality gate


# What the "No free directions" card says would follow if the check failed. The guard line is
# one line on the card, so it stays short.
NULLSPACE_ELSE = "else: the rig's position or turn stays free"


def gate_checks(metadata: Mapping[str, Any], run_status: str) -> tuple[GateCheck, ...]:
    """The quality gate as traffic-light checks, read only from the saved record."""
    diagnostics = metadata.get("diagnostics") or {}
    gate = diagnostics.get("quality_gate") if isinstance(diagnostics, Mapping) else None
    report = metadata.get("report") or {}
    if not isinstance(gate, Mapping):
        return (GateCheck("gate", "Quality gate", "unknown", "not recorded",
                          "this run saved no quality gate"),)
    checks: list[GateCheck] = []
    observed = list(gate.get("observed_cameras") or [])
    expected = gate.get("expected_camera_count")
    cams_ok = gate.get("camera_coverage")
    checks.append(GateCheck(
        "cameras", "All cameras used",
        "good" if cams_ok is True else ("bad" if cams_ok is False else "unknown"),
        f"{len(observed)} / {expected}" if expected is not None else "not recorded",
        "every camera saw the target" if cams_ok else "a camera has no corners"
        if cams_ok is False else "camera coverage not recorded",
        "else: a camera goes uncalibrated"))
    images = list(gate.get("observed_images") or [])
    expected_images = gate.get("expected_image_count")
    excluded = list(gate.get("excluded_images") or gate.get("missing_images") or [])
    if expected_images is None:
        checks.append(GateCheck("images", "Photos used", "unknown", "not recorded",
                                "photo coverage not recorded",
                                "else: photos could drop out unseen"))
    else:
        # A blocking flag about the photos (e.g. pcube: "image observation graph has missing
        # image indices") turns this card red: here the left-out photos hold the result back.
        blocks = any("image" in str(flag).casefold() for flag in gate.get("blocking_flags") or ())
        checks.append(GateCheck(
            "images", "Photos used", "bad" if blocks else ("warn" if excluded else "good"),
            f"{len(images)} / {expected_images}",
            f"{len(excluded)} left out: no target position" if excluded else "every photo placed",
            "here this holds the result back" if blocks else "a notice: these photos add nothing"))
    disposition = gate.get("constraint_disposition") or {}
    nullity = gate.get("jacobian_nullity")
    global_nullity = disposition.get("global_nullity")
    # SCI-3: a numerical-rank check (float64 threshold), not a stiffness or a statistic.
    if not isinstance(nullity, Mapping) or isinstance(global_nullity, str) or global_nullity is None:
        checks.append(GateCheck("nullspace", "No free directions", "bad",
                                "not measured", str(disposition.get("reason") or
                                                    "the null-space check did not run"),
                                NULLSPACE_ELSE))
    else:
        local = disposition.get("point_local_nullity", nullity.get("point_local_nullity"))
        ok = disposition.get("status") == "constrained"
        # Only a corner with one detection (one camera, one photo) can slide; two cameras in one
        # photo fix it. With the fixed numbers removed, no other rig move leaves the photos alike.
        line = (f"{local} one-detection corner{'s' if local != 1 else ''} may slide"
                if local else "other moves all change a photo")
        if not disposition.get("fixed_pose_observed", True):
            line = "the reference photo was not observed"
        checks.append(GateCheck(
            "nullspace", "No free directions", "good" if ok else "warn",
            f"{int(global_nullity)} free direction{'s' if int(global_nullity) != 1 else ''}", line,
            NULLSPACE_ELSE))
    initial = finite(diagnostics.get("D4.3_initial_euclid_px"))
    final = finite(diagnostics.get("D4.3_final_euclid_px"))
    reduced = gate.get("error_reduced")
    # ED2-6 / SCI5-4: this row tests the MEAN error only (final mean below initial mean). It is
    # a sanity check, not evidence that the shape is real (the noise comparison is that). The
    # title says what it tests; the RMS and the solver's own objective are not tested here.
    # ED4-1: a held-back run is not a success, so its check is grey and says "not judged".
    usable = run_status == "complete"
    reduced_tone = ("good" if reduced is True else ("bad" if reduced is False else "unknown"))
    if not usable and reduced_tone == "good":
        reduced_tone = "unknown"
    checks.append(GateCheck(
        "error", "Mean error fell?" if usable else "Mean error (not judged)",
        reduced_tone,
        (f"{initial:.2f} → {final:.2f} px" + ("" if usable else " (not verified)"))
        if initial is not None and final is not None else "not recorded",
        "a sanity check on the mean error only" if usable else "held back: not verified",
        "else: the mean error did not fall"))
    success = gate.get("solver_success")
    plausible = (gate.get("camera_quality") or {}).get("physically_plausible")
    message = solver_stop_words(report.get("solver_message"))
    tone = ("good" if success is True and plausible is not False
            else "bad" if success is False or plausible is False else "unknown")
    value = {True: "Converged", False: "Not converged"}.get(success, "not recorded")
    checks.append(GateCheck(
        "solver", "Solver finished", tone, value,
        message + ("" if plausible else " · cameras implausible" if plausible is False
                   else " · plausibility not recorded"),
        "else: the answer may be half-way"))
    if not usable:
        # ED5-6: on a held run no check is a pass; a green tick would read as one.
        checks = [replace(c, tone="unknown") if c.tone == "good" else c for c in checks]
        checks = [replace(c, value=f"{c.value} (not judged)") if c.key == "solver"
                  and c.value == "Converged" else c for c in checks]
    return tuple(checks)


# --------------------------------------------------------------------------------------
# Preparation


def _coverage(handler: Any, n_cameras: int) -> tuple[np.ndarray | None, int]:
    """(images x cameras) corners detected, from the raw detections (excluded images too)."""
    detection = getattr(handler, "detection", None)
    data = getattr(detection, "_data", None)
    if data is None and hasattr(detection, "get_data"):
        data = detection.get_data()
    try:
        rows = np.asarray(data, dtype=np.float64)
        n_images = int(getattr(detection, "max_ims", 0))
        if rows.ndim != 2 or rows.shape[1] < 2 or n_images <= 0:
            return None, n_images
        grid = np.zeros((n_images, n_cameras), dtype=int)
        cams, ims = rows[:, 0].astype(int), rows[:, 1].astype(int)
        keep = (cams >= 0) & (cams < n_cameras) & (ims >= 0) & (ims < n_images)
        np.add.at(grid, (ims[keep], cams[keep]), 1)
        return grid, n_images
    except (TypeError, ValueError):
        return None, 0


def _camera_means(record: Mapping[str, Any] | None, key: str,
                  names: Sequence[str]) -> dict[int, float] | None:
    diagnostics = (record or {}).get("diagnostics") or {}
    values = diagnostics.get(key) if isinstance(diagnostics, Mapping) else None
    if not isinstance(values, Mapping):
        return None
    out = {}
    for index, name in enumerate(names):
        value = finite(values.get(name))
        if value is None:
            return None
        out[index] = value
    return out


def prepare_phase4_visual(run_context: Any, metadata: Mapping[str, Any], state: Any,
                          derivation: Any, trace: Mapping[str, Any], *,
                          playback: bool = True) -> Phase4Visual:
    """Prepare the five stages' data for one run (Qt-free; see the module docstring).

    Every replayed state is checked against the saved record: the start against D4.3's
    initial error, the result against D4.3's final error and D4.13 per image, recorded steps
    against their stored costs (``read_accepted_step_trace``). A failed check makes that part
    unavailable with the reason; nothing is substituted.
    """
    diagnostics = metadata.get("diagnostics") or {}
    gate = (diagnostics.get("quality_gate") or {}) if isinstance(diagnostics, Mapping) else {}
    checks = gate_checks(metadata, run_context.status)
    notices = tuple(str(n) for n in (gate.get("notices") or ()))
    blocking = tuple(blocking_reasons(plain(metadata)))
    nullity = gate.get("jacobian_nullity") if isinstance(gate.get("jacobian_nullity"), Mapping) else None
    base = dict(checks=checks, notices=notices, blocking=blocking, nullity=nullity)
    if run_context.status == "failed":
        error = str(metadata.get("error") or "details unavailable")
        visual = Phase4Visual("failed", f"the run failed: {error}", **base)
        costs = tuple(trace_costs(trace)) if trace.get("status") == "available" else ()
        visual._playback = Playback("cost only" if costs else "unavailable",
                                    "the saved Phase 4 result failed, so only recorded costs are shown"
                                    if costs else "no readable Phase 4 trace", costs=costs)
        return visual
    if isinstance(derivation, DerivationUnavailable):
        visual = Phase4Visual("unavailable", str(derivation.reason), **base)
        visual._playback = Playback("unavailable", str(derivation.reason))
        return visual
    path = run_context.artifact("phase4")
    try:
        from pyCamSet.utils.target_units import millimetres_per_point_data_unit
        from pyCamSet.workflow.education_derive import _load_camset as load_CameraSet
        camset = load_CameraSet(path)
        replay = Replay(camset, diagnostics, path=path)
    except Exception as exc:  # noqa: BLE001 - shown to the learner as the reason
        visual = Phase4Visual("unavailable", f"the saved result could not be replayed: {exc}", **base)
        visual._playback = Playback("unavailable", visual.reason)
        return visual

    handler = replay.handler
    nominal = np.asarray(handler.target.point_data, dtype=np.float64).reshape(-1, 3).copy()
    saved_units = getattr(camset, "_saved_point_data_units", None)
    mm_per_unit = (millimetres_per_point_data_unit(handler.target)
                   if saved_units in {"m", "mm"} else None)
    finite_nominal = nominal[np.all(np.isfinite(nominal), axis=1)]
    target_size = float(np.linalg.norm(np.ptp(finite_nominal, axis=0))) if len(finite_nominal) else 1.0
    classes = derivation.point_classes
    measured = None
    try:
        from pyCamSet.workflow.education_derive import _phase4_counts
        measured = _phase4_counts(handler) >= 2
        align = np.zeros(len(nominal), dtype=bool)
        align[list(classes.several_directions)] = True
        if align.sum() >= 3:  # derive_phase4's alignment points
            measured = align
    except Exception:
        measured = None
    coverage, n_images = _coverage(handler, len(replay.names))
    excluded_ids = [int(i) for i in (gate.get("excluded_images") or gate.get("missing_images") or [])]
    excluded = []
    for image in excluded_ids:
        if coverage is not None and 0 <= image < len(coverage):
            seen = [c for c in range(coverage.shape[1]) if coverage[image, c] > 0]
            excluded.append(ExcludedImage(image, tuple(seen),
                                          tuple(int(coverage[image, c]) for c in seen)))
        else:
            excluded.append(ExcludedImage(image, (), ()))
    phase3_record = None
    if getattr(state, "origin_run_id", None):
        try:
            phase3_record = run_context.manager.find_run("phase3", state.origin_run_id)
        except Exception:
            phase3_record = None
    phase3_means = _camera_means(phase3_record, "D3.12_per_camera_mean_reprojection", replay.names)

    visual = Phase4Visual(
        "available", "", camera_names=replay.names, telecentric=replay.telecentric,
        unit="mm" if mm_per_unit is not None else "target units", mm_per_unit=mm_per_unit,
        target_size=target_size, printed=nominal,
        gauge_indices=tuple(int(i) for i in getattr(handler, "fixed_inds", ()) or ()),
        excluded=tuple(excluded), coverage=coverage, n_images=n_images,
        phase3_camera_mean=(phase3_means if getattr(state, "confirmed", False) else None),
        phase3_camera_source=("Phase 3 run's saved D3.12" if phase3_means is not None
                              else "not recorded"),
        phase4_camera_mean=_camera_means(metadata, "D4.12_per_camera_mean_reprojection",
                                         replay.names),
        several=tuple(classes.several_directions), one_direction=tuple(classes.one_direction),
        unseen=tuple(classes.unseen),
        ray_separation_deg=np.array([np.nan if v is None else float(v)
                                     for v in classes.ray_separation_deg], dtype=float),
        **base)
    visual._replay = replay

    visual.gauge_axis = (int(handler.gauge_axis) if isinstance(
        getattr(handler, "gauge_axis", None), (int, np.integer)) else None)
    visual.similarity_scale = finite(getattr(derivation, "display_similarity_scale", None))
    bounds = (0,) + replay.blocks
    visual.block_sizes = tuple(int(bounds[b + 1] - bounds[b]) for b in range(4))
    visual.loss = str((getattr(handler, "problem_opts", {}) or {}).get("loss", "linear"))
    visual.telecentricity_p4 = telecentricity_per_metre(camset, mm_per_unit)
    if getattr(state, "confirmed", False) and getattr(state, "path", None) is not None:
        try:
            visual.telecentricity_p3 = telecentricity_per_metre(
                load_CameraSet(state.path), mm_per_unit)
        except Exception:  # noqa: BLE001 - Phase 3's values are context only
            visual.telecentricity_p3 = {}
        try:
            visual.distortion_p3 = distortion_shifts(load_CameraSet(state.path))
        except Exception:  # noqa: BLE001 - context only
            visual.distortion_p3 = {}
    visual.distortion_p4 = distortion_shifts(camset)
    visual._closing_source = (camset, replay.rows, replay.names, saved_units, str(path))

    # Shape correction: the refined shape turned and shifted onto the design (its size is
    # the printed square size, SCI-1). Unseen corners are not estimated: no correction.
    if derivation.display_refit_points is not None:
        refined = np.asarray(derivation.display_refit_points, dtype=np.float64).reshape(-1, 3)
        if refined.shape == nominal.shape:
            visual.refined = refined
            visual.correction = refined - nominal
            if visual.unseen:
                visual.correction[list(visual.unseen)] = np.nan
            sizes = np.linalg.norm(visual.correction, axis=1)
            typical = sizes[list(visual.several)] if visual.several else sizes
            typical = typical[np.isfinite(typical)]
            visual.exaggeration = nice_exaggeration(
                target_size, float(np.percentile(typical, 90)) if typical.size else math.nan)
            faces = target_faces(handler.target.point_data)
            if faces:
                visual.faces, visual.plate = decompose_faces(
                    nominal, refined, visual.several, faces, to_mm=mm_per_unit or 1.0)
                if not visual.faces:
                    visual.plate = None

    # The start (Phase 3's result as Phase 4 began) and the saved result.
    params = np.asarray(camset.calibration_params, dtype=np.float64).reshape(-1)
    initial_px = finite(diagnostics.get("D4.3_initial_euclid_px"))
    final_px = finite(diagnostics.get("D4.3_final_euclid_px"))
    steps = _trace_steps(trace)
    start_params = None
    if steps:
        start_params = np.asarray(steps[0]["x"], dtype=steps[0].get("dtype", "float64"))
        visual.start_label = "Phase 4's recorded start"
    else:
        try:
            start_params = np.asarray(handler.get_initial_params(), dtype=np.float64).copy()
            visual.start_label = "the start Phase 4 computed"
        except Exception:
            start_params = None
    reasons = []
    kwargs = dict(display_size=target_size, nominal=nominal, measured=measured)
    if start_params is not None and start_params.size == params.size:
        try:
            start = replay.state(0, math.nan, start_params, **kwargs)
            if initial_px is not None and math.isclose(start.mean_px, initial_px,
                                                       rel_tol=1e-6, abs_tol=1e-9):
                visual.start = start
                visual.start_errors = replay.errors(start_params)
            else:
                reasons.append("the replayed start does not reproduce D4.3's starting error")
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"the start could not be replayed: {exc}")
    else:
        reasons.append("no starting parameters were recorded")
    try:
        final = replay.state(-1, math.nan, params, **kwargs)
        stored_images = diagnostics.get("D4.13_per_image_mean_reprojection")
        final_errors = replay.errors(params)
        ok = final_px is not None and math.isclose(final.mean_px, final_px, rel_tol=1e-6,
                                                    abs_tol=1e-9)
        if ok and isinstance(stored_images, Mapping):
            mine = final_errors.per_image()
            ok = all(math.isclose(mine.get(int(k), math.nan), float(v), rel_tol=1e-6,
                                  abs_tol=1e-9) for k, v in stored_images.items())
        if ok:
            visual.final = final
            visual.final_errors = final_errors
        else:
            reasons.append("the replayed result does not reproduce the saved D4.3/D4.13 errors")
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"the result could not be replayed: {exc}")
    visual.errors_reason = "; ".join(reasons)
    if visual.start is not None and visual.final is not None:
        # the re-fit takes seconds: done on first use (Phase4Visual.swap_results)
        visual._swap_source = (replay, visual.start.params, visual.final.params)
    if playback:
        visual._playback = build_playback(visual, metadata, trace, params, steps, kwargs)
    return visual


_LOG = logging.getLogger(__name__)

#: Version of the swap re-fit. Bump it when the method changes: it keys the on-disk cache.
SWAP_METHOD = "poses-refit-solver-frame-v4"


def _set_corners(replay: Replay, x: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``x`` with the target points set to ``points`` (flat, fixed ones too) in the parameter
    vector and in the handler's fixed coordinates; the caller restores the handler."""
    bp = replay.handler.bundlePrimitive
    unfixed = np.asarray(replay.handler.feat_unfixed, dtype=bool)
    flat = np.asarray(points, dtype=np.float64).reshape(-1)
    bp.bundle_pts[...] = flat.reshape(np.shape(bp.bundle_pts))
    x = np.asarray(x, dtype=np.float64).copy()
    x[replay.blocks[2]:] = flat[unfixed]
    return x


def _refit_poses(replay: Replay, x: np.ndarray) -> np.ndarray:
    """Re-fit only the photo poses (the target's pose per photo), everything else held, with
    the solve's own residuals and Jacobian. A pose-only problem: a fraction of a second."""
    from scipy.optimize import least_squares
    n2 = 2 * len(replay.rows)
    cols = slice(replay.blocks[1], replay.blocks[2])
    jac_fun = replay.handler.make_loss_jac(1)

    def resid(p: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[cols] = p
        return np.asarray(replay.loss(y), dtype=np.float64)[:n2]

    def jac(p: np.ndarray) -> Any:
        y = x.copy()
        y[cols] = p
        return sp.csr_matrix(jac_fun(y))[:n2, cols]

    res = least_squares(resid, x[cols].copy(), jac=jac, method="trf", tr_solver="lsmr",
                        x_scale="jac", max_nfev=60)
    y = x.copy()
    y[cols] = res.x
    return y


def _swap_cache_file(start: np.ndarray, final: np.ndarray) -> Path:
    """Where the swap re-fit is kept between sessions (a per-user temp folder, never the run)."""
    key = hashlib.sha1(np.ascontiguousarray(start).tobytes()
                       + np.ascontiguousarray(final).tobytes()
                       + SWAP_METHOD.encode("ascii")).hexdigest()
    return Path(tempfile.gettempdir()) / "pycamset_education" / f"swap_{key}.json"


def _swap_refit(replay: Replay, start: np.ndarray, final: np.ndarray) -> dict[str, float]:
    """The two one-part swaps, in the solver's own frame (no gauge step: Phase 4 starts from
    Phase 3's parameters, so both results share one frame and a pixel error does not depend
    on the printed-square gauge). Each swapped state keeps every other block at its saved
    Phase 4 value, then re-fits the photo poses from two starts (the saved Phase 4 poses and
    Phase 3's poses). The lower converged value of the two is reported, with the start that
    gave it; the starts that were tried are kept too. Mean px."""
    L, E, P, B = replay.blocks
    bp = replay.handler.bundlePrimitive
    saved = np.asarray(bp.bundle_pts).copy()
    try:
        # every corner position of each saved state (the full point array, fixed ones too)
        start_pts = np.asarray(bp.return_bundle_primitives(start.copy())[3], dtype=np.float64).copy()
        bp.bundle_pts[...] = saved
        final_pts = np.asarray(bp.return_bundle_primitives(final.copy())[3], dtype=np.float64).copy()
        bp.bundle_pts[...] = saved
        # (1) lens swap: Phase 3's lens numbers (intrinsics) back; the rest as saved in Phase 4.
        lens_x = _set_corners(replay, final, final_pts)
        lens_x[:L] = start[:L]
        # (2) shape swap: Phase 3's corner positions back; the lenses and camera turns as saved.
        shape_x = _set_corners(replay, final, start_pts)
        out: dict[str, float] = {}
        for name, x in (("lens", lens_x), ("shape", shape_x)):
            for tag, src in (("p4poses", final), ("p3poses", start)):
                y = x.copy()
                y[E:P] = np.asarray(src, dtype=np.float64)[E:P]
                refit = _refit_poses(replay, y)
                out[f"{name}_{tag}"] = float(replay.errors(refit).mean_px)
                bp.bundle_pts[...] = saved
    finally:
        bp.bundle_pts[...] = saved
    out["lens_swap"] = min(out["lens_p4poses"], out["lens_p3poses"])
    out["shape_swap"] = min(out["shape_p4poses"], out["shape_p3poses"])
    return out


def swap_errors(replay: Replay, start: np.ndarray, final: np.ndarray) -> dict[str, float]:
    """ED2-1 / SCI3-1: what each part contributes, with the other part held. Each swapped
    state has its photo poses re-fitted (the other part's lenses or corners held as they are);
    the inserted corners also get one overall scale, fitted. Mean px; {} on failure.

    Cached on disk per pair of parameter vectors (a few seconds the first time)."""
    try:
        L = replay.blocks[0]
        if L <= 0 or start.shape != final.shape:
            return {}
        path = _swap_cache_file(start, final)
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if {"lens_swap", "shape_swap"} <= set(cached):
                return {k: float(v) for k, v in cached.items()}
        except (OSError, ValueError):
            pass
        out = _swap_refit(replay, np.asarray(start, dtype=np.float64),
                          np.asarray(final, dtype=np.float64))
        out = {k: float(v) for k, v in out.items() if math.isfinite(v)}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out), encoding="utf-8")
        except OSError:
            pass
        return out
    except Exception as exc:  # noqa: BLE001 - the panel is then left out, said in the log
        _LOG.warning("swap test not drawn: %s", exc)
        return {}


@dataclass(frozen=True)
class CornerRays:
    """One corner seen by two cameras in one photo: each camera's ray through its measured
    pixel (undistorted, then the camera's projection inverted), and where the two rays come
    closest. Lengths are in the saved camset's world units."""
    photo: int
    corner: int
    cameras: tuple[int, int]
    pixels: tuple[tuple[float, float], tuple[float, float]]
    origins: tuple[np.ndarray, np.ndarray]     # each camera's centre (the ray's start)
    directions: tuple[np.ndarray, np.ndarray]  # unit rays, pointing from the corner to the camera
    angle_deg: float                           # between the two rays (0-90)
    gap: float                                 # closest distance between the rays
    point: np.ndarray                          # midpoint of the closest approach: the corner


@dataclass(frozen=True)
class ClosingSummary:
    """What Phase 4's result is for: µm per pixel per camera, the ray gaps over every corner two
    cameras saw in one photo (medians), and one example corner for the picture."""
    um_per_px: dict[int, float]
    n_pairs: int
    gap_um_median: float | None
    angle_deg_median: float | None
    example: CornerRays | None
    file: str = ""
    #: world units to µm (1e6 for metres, 1e3 for millimetres)
    um_per_unit: float = 1e6


def _projection_3x4(camera: Any) -> np.ndarray:
    """The camera's 3x4 projection of world points to undistorted pixels (both lens kinds)."""
    if hasattr(camera, "_calc_projection_matrix"):
        return np.asarray(camera._calc_projection_matrix(), dtype=np.float64)
    return np.asarray(camera.intrinsic, dtype=np.float64) @ np.asarray(
        camera.extrinsic, dtype=np.float64)[:3]


def _ray_of(camera: Any, pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(centre, unit direction) of the world ray through ``pixel`` (distorted, as detected)."""
    undistorted = np.asarray(camera.undistort_points(np.asarray(pixel, float)[None, :]),
                             dtype=np.float64).reshape(2)
    P = _projection_3x4(camera)
    M, m = P[:, :3], P[:, 3]
    if not np.isfinite(np.linalg.cond(M)) or np.linalg.cond(M) > 1e10:
        return None
    inverse = np.linalg.inv(M)
    origin = -inverse @ m
    direction = inverse @ np.array([undistorted[0], undistorted[1], 1.0])
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm == 0.0 or not np.all(np.isfinite(origin)):
        return None
    return origin, direction / norm


def _closest_approach(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
                      ) -> tuple[float, np.ndarray, float] | None:
    """(gap, midpoint, angle in degrees) of the two lines o + t d (unit d) closest approach."""
    (o1, d1), (o2, d2) = a, b
    cross = float(d1 @ d2)
    denom = 1.0 - cross * cross
    if denom < 1e-10:
        return None  # parallel: no corner to place
    w0 = o1 - o2
    t = (cross * float(d2 @ w0) - float(d1 @ w0)) / denom
    s = (float(d2 @ w0) - cross * float(d1 @ w0)) / denom
    p1, p2 = o1 + t * d1, o2 + s * d2
    angle = math.degrees(math.acos(min(1.0, abs(cross))))
    return float(np.linalg.norm(p1 - p2)), 0.5 * (p1 + p2), angle


def closing_summary(camset: Any, rows: np.ndarray, names: Sequence[str],
                    saved_units: str | None, *, file: str = "") -> ClosingSummary | None:
    """Ray gaps over every corner that two or more cameras saw in one photo, and µm per pixel
    per camera. Telecentric: 1e6 / fx µm per pixel for ``saved_units == "m"`` (fx is pixels per
    metre of the target, as Phase 2 reads it); pinhole: object size grows with depth. {} when
    the units are not known (no µm then)."""
    factor = {"m": 1e6, "mm": 1e3}.get(str(saved_units))
    rows = np.asarray(rows, dtype=np.float64)
    if rows.size == 0 or factor is None:
        return None
    cameras = {i: camset[name] for i, name in enumerate(names)}
    um_per_px: dict[int, float] = {}
    for i, cam in cameras.items():
        fx = float(np.asarray(cam.intrinsic, dtype=float)[0, 0])
        if fx and math.isfinite(fx):
            um_per_px[i] = factor / fx if getattr(cam, "telecentricity", None) is not None \
                else float("nan")
    rays: dict[tuple[int, int], tuple[int, tuple[float, float], tuple[np.ndarray, np.ndarray]]] = {}
    for cam_i, image, point, u, v in rows:  # columns: camera, image, point, u, v
        ray = _ray_of(cameras[int(cam_i)], np.array([u, v]))
        if ray is not None:
            rays[(int(image), int(point)), int(cam_i)] = ray  # type: ignore[index]
    by_corner: dict[tuple[int, int], list[int]] = {}
    for (key, cam_i) in rays:
        by_corner.setdefault(key, []).append(cam_i)
    gaps: list[tuple[float, float, tuple]] = []
    for (image, point), seen in by_corner.items():
        seen = sorted(seen)
        for x in range(len(seen)):
            for y in range(x + 1, len(seen)):
                a, b = rays[(image, point), seen[x]], rays[(image, point), seen[y]]
                found = _closest_approach(a, b)
                if found is not None:
                    gaps.append((found[0], found[2], (image, point, seen[x], seen[y], found[1])))
    if not gaps:
        return ClosingSummary(um_per_px, 0, None, None, None, file, factor)
    gap_values = np.array([g[0] for g in gaps])
    angles = np.array([g[1] for g in gaps])
    # Example: a pair at a usable angle (>= 20 deg) whose gap is nearest the median gap.
    median_gap = float(np.median(gap_values))
    usable = [k for k, g in enumerate(gaps) if g[1] >= 20.0]
    example = None
    if usable:
        k = min(usable, key=lambda j: abs(gaps[j][0] - median_gap))
        image, point, ca, cb, mid = gaps[k][2]
        oa, da = rays[(image, point), ca]
        ob, db = rays[(image, point), cb]
        to_cam_a = (oa - mid) / max(float(np.linalg.norm(oa - mid)), 1e-300)
        to_cam_b = (ob - mid) / max(float(np.linalg.norm(ob - mid)), 1e-300)
        pix = rows[(rows[:, 1] == image) & (rows[:, 2] == point)]
        pa = tuple(float(x) for x in pix[pix[:, 0] == ca][0, 3:5])
        pb = tuple(float(x) for x in pix[pix[:, 0] == cb][0, 3:5])
        example = CornerRays(image, point, (ca, cb), (pa, pb), (oa, ob),
                             (to_cam_a, to_cam_b), float(gaps[k][1]), float(gaps[k][0]), mid)
    return ClosingSummary(um_per_px, len(gaps), float(np.median(gap_values)) * factor,
                          float(np.median(angles)), example, file, factor)


def _trace_steps(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if trace.get("status") != "available":
        return []
    for record in trace.get("records", ()) or ():
        if isinstance(record, Mapping) and record.get("kind") == "accepted_step":
            payload = record.get("payload")
            steps = payload.get("steps") if isinstance(payload, Mapping) else None
            if isinstance(steps, list) and steps:
                return steps
    return []


def build_playback(visual: Phase4Visual, metadata: Mapping[str, Any], trace: Mapping[str, Any],
                   params: np.ndarray, steps: Sequence[Mapping[str, Any]],
                   state_kwargs: Mapping[str, Any]) -> Playback:
    """Validate the recorded steps (endpoints, order, stored costs) and replay each one."""
    from pyCamSet.workflow.education_playback import read_accepted_step_trace
    if (metadata.get("params") or {}).get("record_education") is not True:
        return Playback("not recorded", "steps were not recorded for this run")
    if not steps:
        return Playback("unavailable", "the recorded steps could not be read")
    replay: Replay = visual._replay
    initial = np.asarray(steps[0]["x"], dtype=steps[0].get("dtype", "float64"))
    validated = read_accepted_step_trace(trace, expected_start=initial, saved_params=params,
                                         handler=replay.handler)
    if validated["status"] != "available":
        return Playback("unavailable", str(validated.get("reason", "recorded steps unavailable")))
    states = []
    for nit, cost, x in zip(validated["nits"], validated["costs"], validated["arrays"]):
        if np.asarray(x).size != params.size:
            return Playback("unavailable", "recorded frame parameter layout differs")
        states.append(replay.state(int(nit), float(cost), np.asarray(x, dtype=np.float64),
                                   **state_kwargs))
    if len(states) < 2:
        return Playback("unavailable", "only the starting state was recorded", tuple(states))
    # Share of each block's total movement made at each step (unit-free per block).
    bounds = (0,) + replay.blocks
    moves = np.zeros((len(states), 4))
    for k in range(1, len(states)):
        delta = states[k].params - states[k - 1].params
        for b in range(4):
            moves[k, b] = float(np.linalg.norm(delta[bounds[b]:bounds[b + 1]]))
    totals = moves.sum(axis=0)
    share = np.divide(moves, totals, out=np.zeros_like(moves), where=totals > 0)
    sizes = tuple(int(bounds[b + 1] - bounds[b]) for b in range(4))
    return Playback("available", "", tuple(states),
                    tuple((s.nit, s.cost) for s in states), share, sizes)


__all__ = [
    "STAGES", "TELECENTRIC_DISPLAY_DISTANCE", "THETA_MIN_DEG", "EXAGGERATIONS",
    "START_FROM_PHASE3", "Phase4Narrative", "Phase4Visual", "CameraPose", "ExcludedImage",
    "GateCheck", "SolverState", "Playback", "ErrorTable", "Replay", "gate_checks", "FaceMove",
    "target_faces", "decompose_faces", "telecentricity_per_metre", "solver_stop_words",
    "FacePair", "face_pairs", "face_squareness", "distortion_shifts", "swap_errors",
    "SEEDING_ONLY_OPTIONS",
    "prepare_phase4_visual", "build_playback", "trace_costs", "step_rows", "nice_exaggeration",
    "length_unit", "plain", "fmt", "finite", "loss_text", "ordinal",
]
