"""
The ChArUco detector parameters, and the two ways they are arrived at.

One table, beside :mod:`~pyCamSet.calibration_targets.charuco_detection`,
which is what consumes the options it describes.  There were two: one in the
optimisation package giving each parameter bounds for a sampler, one in the
workflow giving the same parameters rules for reading a typed value, and
twelve parameters described in both.

A row always says what the parameter is called, which OpenCV sub-object it
belongs to, what it defaults to, and what it does.  It additionally carries

* ``dtype``/``min``/``max``/``step``/``odd``/``decimals`` when a study may
  search over it -- sixteen of the twenty-one do;
* ``parser_type``/``value_type`` and the prose around them when a person may
  type it in -- seventeen of them do.

Values are native throughout: ``cornerRefinementMethod`` defaults to ``0``,
not to the name of the constant, and its choices carry OpenCV's own names
against OpenCV's own values.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

CHARUCO_PARAMETERS: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "charuco_parameters.json").read_text(encoding="utf-8")
)


def by_key() -> dict[str, dict[str, Any]]:
    """Return ``{key: row}``, for looking one parameter up."""
    return {row["key"]: row for row in CHARUCO_PARAMETERS}


def searchable() -> list[dict[str, Any]]:
    """
    The parameters a study may sample: those with bounds to sample within.

    In the study's own order, which is not the form's -- the table is stored
    in form order so that its priority groups stay contiguous.
    """
    rows = [row for row in CHARUCO_PARAMETERS if "min" in row and "max" in row]
    return sorted(rows, key=lambda row: row["search_order"])


def typeable() -> list[dict[str, Any]]:
    """The parameters a person may type a value for, in form order."""
    return [row for row in CHARUCO_PARAMETERS if "parser_type" in row]


def numeric_keys() -> list[str]:
    """The keys of every searchable parameter, in order."""
    return [row["key"] for row in searchable()]


def default_fixed_settings() -> dict[str, dict[str, Any]]:
    """Every searchable parameter at its default, in OpenCV's sub-dict shape."""
    return assemble_detection_options(
        {row["key"]: row["default"] for row in searchable()})


def assemble_detection_options(values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Group a flat ``{key: value}`` mapping into OpenCV's sub-dict shape.

    A key this build of OpenCV does not know is dropped rather than raised
    on, which is how ``build_charuco_parameters`` treats one too.
    """
    grouped: dict[str, dict[str, Any]] = {}
    rows = by_key()
    for key, value in values.items():
        row = rows.get(key)
        if row is None:
            continue
        grouped.setdefault(row["group"], {})[key] = value
    return grouped


def choice_labels(row: dict[str, Any]) -> list[str]:
    """The names a choice parameter offers, in order."""
    return [choice["label"] for choice in row.get("choices", [])]


def label_for(row: dict[str, Any], value: Any) -> str:
    """Render *value* the way a person picks it, for a choice parameter."""
    for choice in row.get("choices", []):
        if choice["value"] == value:
            return str(choice["label"])
    return str(value)


def collect_detection_options(raw_values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Read typed values for the typeable parameters into detector options.

    :param raw_values: what was typed, keyed by parameter
    :raises ValueError: for a value the parameter cannot take
    """
    parsed = {row["key"]: parse_value(row, raw_values.get(row["key"]))
              for row in typeable()}

    if parsed["adaptiveThreshWinSizeMax"] < parsed["adaptiveThreshWinSizeMin"]:
        raise ValueError(
            "adaptiveThreshWinSizeMax must be >= adaptiveThreshWinSizeMin.")

    rows = by_key()
    kept = {key: value for key, value in parsed.items()
            if not (value is None and rows[key].get("drop_if_none", False))}
    return assemble_detection_options(kept)


def validate_all_rows(rows: Iterable[dict[str, Any]]) -> list[str]:
    """
    Check a study's parameter rows against what each parameter allows.

    Each row carries at least ``key``, ``fixed`` and ``optimise``; an
    optimised one also carries ``lower`` and ``upper``.

    :return: every problem found, as a flat list of sentences
    """
    known = {row["key"]: row for row in searchable()}
    errors: list[str] = []
    for row in rows:
        entry = known.get(row["key"])
        if entry is None:
            # Named separately from an unknown key: a parameter can be real,
            # and settable on a form, and still have no bounds to search
            # between -- a camera matrix is not a number a study can sample.
            if row["key"] in by_key():
                errors.append(
                    f"Parameter {row['key']!r} has no bounds, so a study "
                    f"cannot search over it.")
            else:
                errors.append(f"Unknown parameter key {row['key']!r}.")
            continue
        errors.extend(
            validate_parameter_row(
                entry,
                fixed_value=row.get("fixed", entry["default"]),
                optimise=bool(row.get("optimise", False)),
                lower=row.get("lower"),
                upper=row.get("upper"),
            )
        )
    return errors


def coerce_value(entry: dict[str, Any], value: Any) -> Any:
    """Coerce *value* to the dtype declared by *entry*.

    Raises :class:`ValueError` when coercion is impossible.
    """
    dtype = entry["dtype"]
    if dtype == "int":
        return int(round(float(value)))
    if dtype == "float":
        return float(value)
    raise ValueError(f"Unsupported dtype {dtype!r} for parameter {entry['key']!r}")


def clamp_to_bounds(entry: dict[str, Any], value: Any) -> Any:
    """Coerce and clamp *value* to ``[entry['min'], entry['max']]``."""
    v = coerce_value(entry, value)
    lo = coerce_value(entry, entry["min"])
    hi = coerce_value(entry, entry["max"])
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def validate_parameter_row(
    entry: dict[str, Any],
    *,
    fixed_value: Any,
    optimise: bool,
    lower: Any | None = None,
    upper: Any | None = None,
) -> list[str]:
    """Validate one parameter row.

    Returns a list of human-readable error strings.  An empty list means the
    row is valid.  This function is pure — it does not raise.
    """
    errors: list[str] = []
    label = entry.get("label", entry["key"])
    choice_values = {coerce_value(entry, choice["value"]) for choice in entry.get("choices", [])}
    try:
        fv = coerce_value(entry, fixed_value)
    except (TypeError, ValueError):
        errors.append(f"{label}: fixed value is not a valid {entry['dtype']}.")
        return errors
    abs_lo = coerce_value(entry, entry["min"])
    abs_hi = coerce_value(entry, entry["max"])
    if fv < abs_lo or fv > abs_hi:
        errors.append(
            f"{label}: fixed value {fv} is outside allowed bounds [{abs_lo}, {abs_hi}]."
        )
    if choice_values and fv not in choice_values:
        errors.append(
            f"{label}: fixed value {fv} is not one of the allowed choices {sorted(choice_values)}."
        )

    if optimise:
        if choice_values:
            return errors
        if lower is None or upper is None:
            errors.append(f"{label}: lower and upper bounds must be provided when optimising.")
            return errors
        try:
            lo = coerce_value(entry, lower)
            hi = coerce_value(entry, upper)
        except (TypeError, ValueError):
            errors.append(f"{label}: bounds are not valid {entry['dtype']}s.")
            return errors
        if lo < abs_lo or hi > abs_hi:
            errors.append(
                f"{label}: bounds [{lo}, {hi}] exceed allowed range [{abs_lo}, {abs_hi}]."
            )
        if lo > hi:
            errors.append(f"{label}: lower bound {lo} exceeds upper bound {hi}.")
    return errors



def parse_value(meta: dict[str, Any], raw_value: Any):
    """Read one typed value as the parameter *meta* describes it."""
    key = meta["key"]
    parser = meta["parser_type"]
    value = raw_value
    if isinstance(value, str):
        value = value.strip()
    if value in (None, ""):
        value = meta["default"]

    if parser == "enum":
        for choice in meta.get("choices", []):
            if value in (choice["label"], choice["value"]):
                return choice["value"]
        raise ValueError(
            f"{key} must be one of {choice_labels(meta)}.")

    if parser in {"int", "positive_int", "int_range"}:
        try:
            out = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer.") from exc
        if parser == "positive_int" and out <= 0:
            raise ValueError(f"{key} must be >= 1.")
        if parser == "int_range":
            min_value = int(meta.get("min_value", out))
            max_value = int(meta.get("max_value", out))
            if out < min_value or out > max_value:
                raise ValueError(f"{key} must be between {min_value} and {max_value}.")
        return out

    if parser in {"float", "positive_float", "non_negative_float"}:
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number.") from exc
        if parser == "positive_float" and out <= 0:
            raise ValueError(f"{key} must be > 0.")
        if parser == "non_negative_float" and out < 0:
            raise ValueError(f"{key} must be >= 0.")
        return out

    if parser == "optional_json_matrix_3x3":
        if value in ("", None):
            return None
        try:
            data = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError as exc:
            raise ValueError(f"{key} must be valid JSON.") from exc
        if not isinstance(data, list) or len(data) != 3:
            raise ValueError(f"{key} must be a 3x3 JSON array.")
        for row in data:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(f"{key} must be a 3x3 JSON array.")
            for elem in row:
                try:
                    float(elem)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} entries must be numeric.") from exc
        return data

    if parser == "optional_json_vector":
        if value in ("", None):
            return None
        try:
            data = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError as exc:
            raise ValueError(f"{key} must be valid JSON.") from exc
        if not isinstance(data, list):
            raise ValueError(f"{key} must be a JSON array.")
        for elem in data:
            if isinstance(elem, list):
                for sub_elem in elem:
                    try:
                        float(sub_elem)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{key} entries must be numeric.") from exc
            else:
                try:
                    float(elem)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} entries must be numeric.") from exc
        return data

    raise ValueError(f"Unsupported parser type {parser!r} for {key}.")
