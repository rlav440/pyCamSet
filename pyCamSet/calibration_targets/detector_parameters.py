"""
What a detector can be told, described as data.

A target's ``find_in_image`` is not one implementation: it picks a detector
backend, and each backend takes its own settings.  Those settings were
previously ChArUco's alone, written down once as a table and imported by name
from the phase 1 form, the optimisation tab, the tuning worker and the optuna
adapter -- so a target that read its markers some other way had nowhere to say
what it could be told, and the twelve settings PuzzleBoardCube already takes
were reachable from nothing.

So a backend describes itself.  :class:`DetectorParameter` is one setting --
what it is called, what it defaults to, the bounds a study may search between
and the prose a person reads while typing it in.  :class:`DetectorParameterisation`
is a backend's set of them, plus the two things that genuinely differ between
backends: rules that span more than one parameter, and whether the backend's
optional dependency is installed.

A target composes these.  Its own ``find_in_image`` may take settings of its
own -- PuzzleBoardCube's plane gate is not the PuzzleBoard detector's
business -- so what the interface and the tuner see is the target's own
parameters beside the selected backend's, joined by :func:`combine`.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Choice:
    """One of the fixed values a parameter may take."""

    label: str
    value: Any


#: The kinds of value a parameter holds.  A parameter with ``choices`` is
#: still one of these -- OpenCV's corner refinement method is an ``int`` that
#: happens to have four names -- because what it is typed as and what it is
#: applied as are the same question.
DTYPES = ("int", "float", "json_matrix_3x3", "json_vector")


@dataclass(frozen=True)
class DetectorParameter:
    """
    One setting of one detector, and everything said about it anywhere.

    A parameter is here for one of two reasons, and may be here for both:
    ``tunable`` says a study may search over it, which needs
    ``minimum``/``maximum``; ``settable`` says a person may type it on the
    phase 1 form, which needs the prose.
    """

    key: str
    label: str
    default: Any
    dtype: str
    group: str = ""
    tunable: bool = False
    settable: bool = True
    minimum: Any | None = None
    maximum: Any | None = None
    step: Any | None = None
    decimals: int | None = None
    odd: bool = False
    search_order: int = 0
    choices: tuple[Choice, ...] = ()
    priority: str = ""
    concept: str = ""
    range_text: str = ""
    range_source: str = ""
    suggested: str = ""
    drop_if_none: bool = False

    def __post_init__(self):
        if self.dtype not in DTYPES:
            raise ValueError(
                f"{self.key}: dtype {self.dtype!r} is not one of {DTYPES}.")
        if self.tunable and (self.minimum is None or self.maximum is None):
            raise ValueError(
                f"{self.key}: a study samples between bounds, so a tunable "
                f"parameter needs both a minimum and a maximum.")

    # -- reading a value -------------------------------------------------

    @property
    def numeric(self) -> bool:
        """Whether this parameter holds a number, and so can be clamped."""
        return self.dtype in ("int", "float")

    def cast(self, value: Any) -> Any:
        """
        *value* as this parameter's dtype, without bounds or rounding.

        :raises ValueError: for a value the dtype cannot hold
        """
        if self.dtype == "int":
            return int(round(float(value)))
        if self.dtype == "float":
            return float(value)
        return value

    def coerce(self, value: Any) -> Any:
        """
        *value* as this parameter will actually be applied.

        Cast to the dtype, rounded up to odd where the parameter is one of
        OpenCV's window sizes, and clamped into its bounds.  A study samples
        inside the bounds and a form refuses values outside them, so this
        only has work to do for a value arrived at some third way.

        An optional parameter left empty comes back as None, which is how a
        detector is told to keep its own default for it.
        """
        if self.drop_if_none and value in ("", None):
            return None
        if not self.numeric:
            return value
        out = self.cast(value)
        if self.odd:
            out |= 1
        if self.minimum is not None:
            out = max(out, self.cast(self.minimum))
        if self.maximum is not None:
            out = min(out, self.cast(self.maximum))
        return out

    def parse(self, raw: Any) -> Any:
        """
        Read one typed value, as a person entered it.

        An empty entry means the default, which for an optional parameter is
        itself empty and reads back as ``None``.

        :raises ValueError: for a value this parameter cannot take
        """
        value = raw.strip() if isinstance(raw, str) else raw
        if value is None or value == "":
            value = self.default
        if value == "" and self.drop_if_none:
            return None

        if self.choices:
            for choice in self.choices:
                if value in (choice.label, choice.value):
                    return choice.value
            raise ValueError(
                f"{self.key} must be one of {', '.join(self.choice_labels())}.")

        if self.dtype in ("int", "float"):
            try:
                out = self.cast(value)
            except (TypeError, ValueError) as exc:
                kind = "an integer" if self.dtype == "int" else "a number"
                raise ValueError(f"{self.key} must be {kind}.") from exc
            return self._checked(out)

        if self.dtype == "json_matrix_3x3":
            data = self._as_json(value)
            if data is None:
                return None
            if not isinstance(data, list) or len(data) != 3:
                raise ValueError(f"{self.key} must be a 3x3 JSON array.")
            for row in data:
                if not isinstance(row, list) or len(row) != 3:
                    raise ValueError(f"{self.key} must be a 3x3 JSON array.")
                self._require_numbers(row)
            return data

        data = self._as_json(value)
        if data is None:
            return None
        if not isinstance(data, list):
            raise ValueError(f"{self.key} must be a JSON array.")
        for element in data:
            self._require_numbers(element if isinstance(element, list) else [element])
        return data

    def _checked(self, value: Any) -> Any:
        """*value*, refused rather than silently clamped, when out of bounds."""
        low = None if self.minimum is None else self.cast(self.minimum)
        high = None if self.maximum is None else self.cast(self.maximum)
        if low is not None and high is not None and not low <= value <= high:
            raise ValueError(f"{self.key} must be between {low} and {high}.")
        if low is not None and value < low:
            raise ValueError(f"{self.key} must be >= {low}.")
        if high is not None and value > high:
            raise ValueError(f"{self.key} must be <= {high}.")
        return value

    def _as_json(self, value: Any):
        """The JSON *value* stands for, or None when it stands for nothing."""
        if value in ("", None):
            return None
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{self.key} must be valid JSON.") from exc

    def _require_numbers(self, values: Iterable[Any]) -> None:
        for element in values:
            try:
                float(element)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{self.key} entries must be numeric.") from exc

    # -- naming a value --------------------------------------------------

    def choice_labels(self) -> list[str]:
        """The names this parameter's choices go by, in order."""
        return [choice.label for choice in self.choices]

    def label_for(self, value: Any) -> str:
        """Render *value* the way a person picks it."""
        for choice in self.choices:
            if choice.value == value:
                return str(choice.label)
        return str(value)

    # -- checking what a study was asked to do ---------------------------

    def validate_row(
        self,
        *,
        fixed: Any,
        optimise: bool,
        lower: Any | None = None,
        upper: Any | None = None,
    ) -> list[str]:
        """
        Check one row of a study's parameter table against this parameter.

        Pure: every problem is returned rather than raised, because a form
        shows all of them at once.
        """
        errors: list[str] = []
        choice_values = {self.cast(choice.value) for choice in self.choices}
        try:
            value = self.cast(fixed)
        except (TypeError, ValueError):
            return [f"{self.label}: fixed value is not a valid {self.dtype}."]

        low, high = self.cast(self.minimum), self.cast(self.maximum)
        if not low <= value <= high:
            errors.append(
                f"{self.label}: fixed value {value} is outside allowed "
                f"bounds [{low}, {high}].")
        if choice_values and value not in choice_values:
            errors.append(
                f"{self.label}: fixed value {value} is not one of the allowed "
                f"choices {sorted(choice_values)}.")

        if not optimise or choice_values:
            return errors
        if lower is None or upper is None:
            errors.append(
                f"{self.label}: lower and upper bounds must be provided when "
                f"optimising.")
            return errors
        try:
            lo, hi = self.cast(lower), self.cast(upper)
        except (TypeError, ValueError):
            errors.append(f"{self.label}: bounds are not valid {self.dtype}s.")
            return errors
        if lo < low or hi > high:
            errors.append(
                f"{self.label}: bounds [{lo}, {hi}] exceed allowed range "
                f"[{low}, {high}].")
        if lo > hi:
            errors.append(f"{self.label}: lower bound {lo} exceeds upper bound {hi}.")
        return errors

    # -- construction ----------------------------------------------------

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "DetectorParameter":
        """Build a parameter from its stored row."""
        fields = dict(row)
        fields["choices"] = tuple(
            Choice(str(choice["label"]), choice["value"])
            for choice in row.get("choices", ()))
        return cls(**fields)


def parameters_from_json(path: Path) -> tuple[DetectorParameter, ...]:
    """Every parameter described by a stored table, in the table's order."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(DetectorParameter.from_row(row) for row in rows)


class DetectorParameterisation(ABC):
    """
    What one detector can be told.

    A backend subclasses this to describe its settings; a target subclasses
    it too when its own ``find_in_image`` takes settings that are not the
    backend's.  The two are joined by :func:`combine`, and everything that
    varies a detection -- the phase 1 form, the optimisation tab, the tuning
    worker -- reads the result rather than knowing which detector it has.
    """

    #: What this detector is called, where one is named.
    name: str = ""

    @property
    @abstractmethod
    def parameters(self) -> tuple[DetectorParameter, ...]:
        """Every parameter this detector takes, in the order a form shows them."""

    # -- looking parameters up -------------------------------------------

    def __len__(self) -> int:
        return len(self.parameters)

    def __contains__(self, key: str) -> bool:
        return any(parameter.key == key for parameter in self.parameters)

    def parameter(self, key: str) -> DetectorParameter:
        """
        The parameter *key* names.

        :raises KeyError: for a key this detector does not take
        """
        for parameter in self.parameters:
            if parameter.key == key:
                return parameter
        raise KeyError(
            f"{self.name or type(self).__name__} has no parameter {key!r}.")

    def tunable(self) -> list[DetectorParameter]:
        """The parameters a study may search over, in the order it sweeps them."""
        return sorted(
            (p for p in self.parameters if p.tunable),
            key=lambda parameter: parameter.search_order)

    def settable(self) -> list[DetectorParameter]:
        """The parameters a person may type a value for, in form order."""
        return [p for p in self.parameters if p.settable]

    # -- values ----------------------------------------------------------

    def defaults(self) -> dict[str, Any]:
        """Every parameter at its default."""
        return {p.key: p.default for p in self.parameters}

    def resolve(self, values: dict[str, Any] | None) -> dict[str, Any]:
        """
        The settings a detection will actually run with.

        Defaults, overridden by *values*, each coerced to what its parameter
        can be applied as.  A key this detector does not take is dropped
        rather than raised on: a target rebuilt from a saved spec should not
        fail because a parameter has since been renamed.
        """
        overrides = values or {}
        return {p.key: p.coerce(overrides.get(p.key, p.default))
                for p in self.parameters}

    def parse(self, raw_values: dict[str, Any]) -> dict[str, Any]:
        """
        Read typed values for the settable parameters.

        :param raw_values: what was typed, keyed by parameter
        :raises ValueError: for a value a parameter cannot take, or a
            combination :meth:`validate` rejects
        """
        parsed = {p.key: p.parse(raw_values.get(p.key)) for p in self.settable()}
        problems = self.validate(parsed)
        if problems:
            raise ValueError(" ".join(problems))
        return {key: value for key, value in parsed.items()
                if not (value is None and self.parameter(key).drop_if_none)}

    def validate_rows(self, rows: Iterable[Any]) -> list[str]:
        """
        Check a study's parameter rows against what each parameter allows.

        Each row carries at least ``key``, ``fixed`` and ``optimise``; an
        optimised one also carries ``lower`` and ``upper``.

        :return: every problem found, as a flat list of sentences
        """
        errors: list[str] = []
        for row in rows:
            key = row["key"] if isinstance(row, dict) else row.key
            get = (row.get if isinstance(row, dict)
                   else lambda name, default=None: getattr(row, name, default))
            try:
                parameter = self.parameter(key)
            except KeyError:
                errors.append(f"Unknown parameter key {key!r}.")
                continue
            if not parameter.tunable:
                # Named apart from an unknown key: a parameter can be real,
                # and settable on a form, and still have no bounds to search
                # between -- a camera matrix is not a number a study samples.
                errors.append(
                    f"Parameter {key!r} has no bounds, so a study cannot "
                    f"search over it.")
                continue
            errors.extend(parameter.validate_row(
                fixed=get("fixed", parameter.default),
                optimise=bool(get("optimise", False)),
                lower=get("lower"),
                upper=get("upper"),
            ))
        return errors

    # -- what differs between detectors ----------------------------------

    def validate(self, values: dict[str, Any]) -> list[str]:
        """
        Rules that span more than one parameter.

        :return: every problem found, empty when the settings are usable
        """
        return []

    def unavailable_reason(self, values: dict[str, Any] | None = None) -> str | None:
        """
        Why this detector cannot run here, when it cannot.

        :return: what to tell someone, or None when the detector is available
        """
        return None


class NoDetectorParameters(DetectorParameterisation):
    """A detector that takes no settings, and a target that is never detected."""

    name = "none"

    @property
    def parameters(self) -> tuple[DetectorParameter, ...]:
        return ()


#: The parameterisation of a target with nothing to tune.  Shared rather than
#: built per target, because it holds nothing to tell apart.
NO_DETECTOR_PARAMETERS = NoDetectorParameters()


class CompositeParameterisation(DetectorParameterisation):
    """
    Several parameterisations read as one.

    A target's own detection settings beside its backend's.  Problems and
    unavailability come from whichever part reported them.
    """

    def __init__(self, parts: Sequence[DetectorParameterisation], name: str = ""):
        self._parts = tuple(parts)
        self.name = name or "+".join(p.name for p in self._parts if p.name)
        seen: set[str] = set()
        for part in self._parts:
            for parameter in part.parameters:
                if parameter.key in seen:
                    raise ValueError(
                        f"{self.name}: {parameter.key!r} is described by more "
                        f"than one of its parameterisations.")
                seen.add(parameter.key)

    @property
    def parameters(self) -> tuple[DetectorParameter, ...]:
        return tuple(p for part in self._parts for p in part.parameters)

    def validate(self, values: dict[str, Any]) -> list[str]:
        return [problem for part in self._parts
                for problem in part.validate(values)]

    def unavailable_reason(self, values: dict[str, Any] | None = None) -> str | None:
        for part in self._parts:
            reason = part.unavailable_reason(values)
            if reason is not None:
                return reason
        return None


def combine(*parts: DetectorParameterisation) -> DetectorParameterisation:
    """
    The parameterisations *parts* read as one.

    Only the one that stands for nothing drops out, so composing a target
    that adds nothing of its own with a backend gives back the backend
    itself.  A part with no parameters is not nothing: aruco2 takes no
    settings and still says whether it is installed.
    """
    kept = [part for part in parts if part is not NO_DETECTOR_PARAMETERS]
    if not kept:
        return NO_DETECTOR_PARAMETERS
    if len(kept) == 1:
        return kept[0]
    return CompositeParameterisation(kept)
