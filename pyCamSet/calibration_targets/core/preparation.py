"""
Phase 1's optional "Rescale and apply gamma" detector preparation, for any target.

The preparation itself (bit-depth reduction, one gamma, one area resize) lives in
:func:`~pyCamSet.calibration_targets.markers.puzzleboard.preprocess_puzzleboard_image`,
where it began as a PuzzleBoard-only path.  This module decides *whether* it runs for a
target and with *which* values:

* It runs only when it is explicitly requested (``params["rescale_and_gamma"]``).
* Scale, gamma and the 16-bit loader default per target family; an explicit value in
  ``params`` always wins over the default.
* Params written before the preparation applied to every target do not carry
  :data:`ANY_TARGET_KEY`; for those, a ticked box still applies to PuzzleBoard targets only,
  so an old run re-detects (Phase 2, the education explainer) exactly as it first did.

Why the defaults differ (measured per target on 311 real images, counts of detected
corners per image):

* PuzzleBoard family: scale 0.25, gamma 0.5, upper byte -- pyCamSet's pcube preparation,
  unchanged.  It gives about 3.3x the unprepared count on the pcube data.
* ChArUco family (ChArUco, ChArUco2, Ccube, Ccube2, CIco, CIco2): scale 1.0, gamma 0.5,
  min/max loader.  The pcube values give zero points on both ChArUco cubes; gamma 0.5 at
  full resolution helps, and every downscale below about 0.8 hurts.  CIco/CIco2 and the flat
  boards were not swept: they share the detector, so they share the defaults (assumption).
* Any other target: no measurements, so scale 1.0 and gamma 0.5 (no
  downscale, a mild lift of dark tones) with the upper-byte loader.

The loader matters for 16-bit files.  ``upper_byte`` keeps bits 8-15, which is what pyCamSet
has always done.  A 12-bit camera stored in uint16 only uses values up to 4095, so its upper
byte holds at most 16 grey levels: on the two 12-bit Ccube sets the upper byte gave zero
points and ``min_max`` (a per-image stretch of the darkest to the brightest value onto
0..255) gave 5.2-5.4 per image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

#: Loader that keeps the upper byte of a uint16 image (pyCamSet's historical behaviour).
UPPER_BYTE = "upper_byte"
#: Loader that stretches each image's own darkest..brightest value onto 0..255.
MIN_MAX = "min_max"
#: Every loader the preparation accepts.
LOADERS: tuple[str, ...] = (UPPER_BYTE, MIN_MAX)

#: Params key that says the preparation applies to every target type.  The GUI always
#: writes it; params without it (saved before 2026-10-13) keep the PuzzleBoard-only rule.
ANY_TARGET_KEY = "rescale_and_gamma_any_target"

PUZZLEBOARD = "puzzleboard"
CHARUCO = "charuco"
OTHER = "other"

#: Target modules (last dotted component) of the ChArUco family.
_CHARUCO_MODULES = frozenset({"charuco", "charuco2", "ccube", "ccube2", "cico", "cico2"})

#: Below this scale ChArUco detections collapse (measured on 311 real images: 0.5 already loses about
#: half the points, 0.25 detects nothing); Phase 1 warns when a ChArUco target is asked for it.
CHARUCO_MIN_SCALE = 0.8


@dataclass(frozen=True)
class PreparationDefaults:
    """One target family's default preparation and where the values came from."""

    scale: float
    gamma: float
    loader: str
    source: str


#: Per-family defaults.  Edit here, and only here, when new measurements arrive.
PREPARATION_DEFAULTS: dict[str, PreparationDefaults] = {
    PUZZLEBOARD: PreparationDefaults(
        0.25, 0.5, UPPER_BYTE, "pyCamSet's pcube preparation (unchanged)"),
    CHARUCO: PreparationDefaults(
        1.0, 0.5, MIN_MAX,
        "measured on 311 real images: gamma 0.5 at full resolution; scale must stay >= 0.8; "
        "min/max loading for 12-bit Ccube sets"),
    OTHER: PreparationDefaults(
        1.0, 0.5, UPPER_BYTE, "unmeasured: no downscale, mild gamma, pyCamSet's loader"),
}


def family_of_module(module_name: str) -> str:
    """The preparation family of a target class, from the name of its module."""
    name = str(module_name).lower()
    if PUZZLEBOARD in name:
        return PUZZLEBOARD
    if name.rsplit(".", 1)[-1] in _CHARUCO_MODULES:
        return CHARUCO
    return OTHER


def family_of_target_type(type_name: str) -> str:
    """The preparation family of a registry target name (e.g. ``"Ccube"``); KeyError if unknown."""
    from pyCamSet.calibration_targets.core.target_registry import TARGET_CLASSES
    return family_of_module(TARGET_CLASSES[str(type_name)][0])


def target_family(target) -> str:
    """The preparation family of a target object (``"puzzleboard"``, ``"charuco"`` or ``"other"``)."""
    return family_of_module(type(target).__module__)


def validate_loader(loader: Any) -> str:
    """Return *loader* when it is one of :data:`LOADERS`, or explain the error."""
    if loader not in LOADERS:
        raise ValueError(f"preprocessing loader must be one of {', '.join(LOADERS)}; got {loader!r}")
    return str(loader)


@dataclass(frozen=True)
class Preparation:
    """The preparation one detection pass applies, with every value resolved."""

    scale: float
    gamma: float
    loader: str
    family: str
    #: Names of the values taken from the family default rather than from params.
    defaulted: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("scale", "gamma"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"preprocessing {name} must be finite and greater than zero")
        validate_loader(self.loader)

    def detect_kwargs(self) -> dict:
        """The keyword arguments the detection entry points take for this preparation."""
        return {"rescale_and_gamma": True, "preprocessing_scale": float(self.scale),
                "preprocessing_gamma": float(self.gamma), "preprocessing_loader": self.loader}

    def describe(self) -> str:
        """One line for the run log."""
        tags = {name: " (target default)" if name in self.defaulted else ""
                for name in ("scale", "gamma", "loader")}
        return (f"scale={self.scale:g}{tags['scale']}, gamma={self.gamma:g}{tags['gamma']}, "
                f"16-bit loader={self.loader}{tags['loader']}")

    def record(self) -> dict:
        """What Phase 1 diagnostics record about the preparation that ran."""
        return {"scale": float(self.scale), "gamma": float(self.gamma), "loader": self.loader,
                "family": self.family, "defaulted": list(self.defaulted),
                "defaults_source": PREPARATION_DEFAULTS[self.family].source}


def cache_identity(rescale_and_gamma: bool, scale: float, gamma: float,
                   loader: str = UPPER_BYTE) -> Optional[dict]:
    """The preprocessing part of a detection cache's identity, or None when off.

    The loader is recorded only when it is not the historical upper byte, so a cache written
    before the loader option existed still matches the same upper-byte pass.
    """
    if not rescale_and_gamma:
        return None
    identity = {"rescale_and_gamma": True, "scale": float(scale), "gamma": float(gamma)}
    if validate_loader(loader) != UPPER_BYTE:
        identity["loader"] = loader
    return identity


def applies_to_family(params: Mapping[str, Any], family: str) -> bool:
    """Whether a ticked "Rescale and apply gamma" applies to a target of *family*."""
    if not bool(params.get("rescale_and_gamma", False)):
        return False
    return family == PUZZLEBOARD or bool(params.get(ANY_TARGET_KEY, False))


def resolve_for_family(params: Mapping[str, Any], family: str) -> Optional[Preparation]:
    """The preparation *params* ask for on a target of *family*, or None when it is off.

    ``preprocessing_scale``, ``preprocessing_gamma`` and ``preprocessing_loader`` that are
    absent or None take the family default; any other value is used as given.
    """
    if not applies_to_family(params, family):
        return None
    defaults = PREPARATION_DEFAULTS[family]
    defaulted = []
    values = {}
    for name, key in (("scale", "preprocessing_scale"), ("gamma", "preprocessing_gamma"),
                      ("loader", "preprocessing_loader")):
        given = params.get(key)
        if given is None:
            defaulted.append(name)
            values[name] = getattr(defaults, name)
        else:
            values[name] = given if name == "loader" else float(given)
    return Preparation(scale=float(values["scale"]), gamma=float(values["gamma"]),
                       loader=values["loader"], family=family, defaulted=tuple(defaulted))


def resolve_preparation(params: Mapping[str, Any], target) -> Optional[Preparation]:
    """The preparation *params* ask for on *target*, or None when it is off (see module doc)."""
    return resolve_for_family(params, target_family(target))


def detect_kwargs_of(params: Mapping[str, Any], target) -> dict:
    """Detection keyword arguments for *params* on *target*; ``rescale_and_gamma=False`` when off."""
    prep = resolve_preparation(params, target)
    return {"rescale_and_gamma": False} if prep is None else prep.detect_kwargs()
