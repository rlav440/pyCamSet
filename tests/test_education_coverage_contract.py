"""Fail when a registered target, lens model, detector backend, solver or loss has no education coverage entry, or when an entry's proof does not prove it.

Nothing here skips. The target and lens-model names are read without importing the target
modules, so the check runs on every machine. The detector backends are read from the target
classes, which needs the optional extras; if one is missing the backend test fails and names the
module, rather than passing by skipping.
"""
from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

from pyCamSet.workflow.education_coverage import COVERAGE, NOT_YET, PARTIAL, TAUGHT

REPOSITORY = Path(__file__).resolve().parents[1]

#: The kinds whose registry names need no optional import.
_NAME_KINDS = ("target", "lens_model", "solver", "loss")


def registered_names() -> dict[str, set[str]]:
    """The registry names for each kind that can be read without optional dependencies."""
    from pyCamSet.calibration_targets.core.target_registry import TARGET_CLASSES
    from pyCamSet.gui.shared_functions import SOLVER_CHOICES
    from pyCamSet.optimisation.camera_models import MODEL_BLOCKS
    from pyCamSet.optimisation.robust_loss import ROBUST_LOSSES

    return {
        "target": set(TARGET_CLASSES),
        "lens_model": {model.__name__ for model in MODEL_BLOCKS},
        "solver": {value for _label, value in SOLVER_CHOICES},
        "loss": set(ROBUST_LOSSES) | {"linear"},
    }


def detector_backends() -> set[str]:
    """The DETECTOR_BACKENDS keys of every registered target.

    Imports each target module, so an optional extra that is not installed raises here with the
    module's name. That is a failure with a message, not a skip.
    """
    from pyCamSet.calibration_targets.core.target_registry import TARGET_CLASSES

    backends: set[str] = set()
    for name, (module_name, class_name) in TARGET_CLASSES.items():
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise AssertionError(
                f"cannot read the detector backends of target {name}: module {module_name} needs "
                f"{exc.name!r}, which is not installed. Install the optional extra that provides it "
                f"so the coverage guard can run.") from exc
        backends |= set(getattr(module, class_name).DETECTOR_BACKENDS)
    return backends


def _target_labels() -> dict[str, str]:
    from pyCamSet.calibration_targets.core.target_registry import TARGET_LABELS

    return dict(TARGET_LABELS)


def _resolve(reference: str):
    """Import ``package.module:Name[.attr]`` and return the object it names."""
    module_name, _, attribute = reference.partition(":")
    obj = importlib.import_module(module_name)
    for part in attribute.split("."):
        obj = getattr(obj, part)
    return obj


def _split_proof(reference: str) -> tuple[Path, str, str | None]:
    """``tests/<file>.py::<name>[<id>]`` -> (file, name, id)."""
    path_text, separator, rest = reference.partition("::")
    assert separator, f"a proof must be tests/<file>.py::<test_name>: {reference!r}"
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[([^\]]+)\])?", rest)
    assert match, f"a proof's test name must be a plain identifier, with an optional [id]: {reference!r}"
    return REPOSITORY / path_text, match.group(1), match.group(2)


def _test_function(reference: str) -> tuple[str, str | None]:
    """The source of the named test function (decorators included), and the bracketed id."""
    path, name, param_id = _split_proof(reference)
    assert path.parent == REPOSITORY / "tests" and path.suffix == ".py", f"proof must be under tests/: {reference!r}"
    assert path.is_file(), f"proof file missing: {reference!r}"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for node in ast.parse(text, filename=str(path)).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
            source = "\n".join(lines[start - 1:node.end_lineno])
            if param_id is not None:
                assert "parametrize" in source, f"{reference!r} gives an id, but {name} is not parametrised"
            return source, param_id
    raise AssertionError(f"no test function {name!r} in {path.relative_to(REPOSITORY).as_posix()}")


def _mentions(text: str, word: str, *, ignore_case: bool) -> bool:
    """True when ``word`` appears as a whole name (not inside a longer identifier)."""
    flags = re.IGNORECASE if ignore_case else 0
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", text, flags) is not None


def _item_words(kind: str, name: str) -> list[str]:
    """The names that identify an item in a test: its registry name, and a one-word label."""
    words = [name]
    if kind == "target":
        label = _target_labels().get(name, "")
        if label and " " not in label:
            words.append(label)
    return words


def _cases(*statuses: str):
    for kind, entries in COVERAGE.items():
        for name, entry in entries.items():
            if entry.status in statuses:
                yield kind, name, entry


def test_every_registered_name_has_a_coverage_entry():
    pytest.importorskip("PySide6")
    names = registered_names()
    assert set(COVERAGE) == set(names) | {"detector_backend"}
    for kind, registered in names.items():
        missing = sorted(registered - set(COVERAGE[kind]))
        assert not missing, f"{kind}: registered but not in education_coverage.COVERAGE: {missing}"


def test_every_coverage_entry_still_exists_in_its_registry():
    pytest.importorskip("PySide6")
    names = registered_names()
    for kind in _NAME_KINDS:
        stale = sorted(set(COVERAGE[kind]) - names[kind])
        assert not stale, f"{kind}: in education_coverage.COVERAGE but no longer registered: {stale}"


def test_every_camera_subclass_is_a_registered_lens_model():
    """A new Camera subclass that is not in MODEL_BLOCKS would escape the guard, so find them by ast."""
    pytest.importorskip("PySide6")
    subclasses: set[str] = set()
    for path in sorted((REPOSITORY / "pyCamSet" / "cameras").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.ClassDef) and any(
                    (isinstance(base, ast.Name) and base.id == "Camera")
                    or (isinstance(base, ast.Attribute) and base.attr == "Camera")
                    for base in node.bases):
                subclasses.add(node.name)
    assert "TelecentricCamera" in subclasses, "the scan found no Camera subclasses; the scan is broken"
    unregistered = sorted(subclasses - registered_names()["lens_model"])
    assert not unregistered, f"Camera subclasses missing from MODEL_BLOCKS: {unregistered}"


def test_every_detector_backend_has_a_coverage_entry():
    backends = detector_backends()
    missing = sorted(backends - set(COVERAGE["detector_backend"]))
    assert not missing, f"detector_backend: registered but not in COVERAGE: {missing}"
    stale = sorted(set(COVERAGE["detector_backend"]) - backends)
    assert not stale, f"detector_backend: in COVERAGE but no longer registered: {stale}"


@pytest.mark.parametrize(("kind", "name", "entry"), list(_cases(TAUGHT, PARTIAL, NOT_YET)))
def test_entry_is_well_formed(kind, name, entry):
    assert entry.status in (TAUGHT, PARTIAL, NOT_YET), f"{kind} {name}: unknown status"
    if entry.status != TAUGHT:
        assert entry.reason.strip(), f"{kind} {name}: a {entry.status} entry must give its reason"
    if entry.status == NOT_YET:
        assert not entry.where and not entry.proof, f"{kind} {name}: a not_yet_taught entry names no code or test"
    if entry.status == TAUGHT:
        assert entry.where, f"{kind} {name}: a taught entry must name the education code"
        assert entry.proof, f"{kind} {name}: a taught entry must name the test that proves it"


@pytest.mark.parametrize(("kind", "name", "reference"), [
    pytest.param(kind, name, reference, id=f"{kind}-{name}-{reference.rsplit('::', 1)[-1]}")
    for kind, name, entry in _cases(TAUGHT, PARTIAL)
    for reference in entry.proof
])
def test_proof_is_a_real_test_function(kind, name, reference):
    _test_function(reference)


@pytest.mark.parametrize(("kind", "name", "reference"), [
    pytest.param(kind, name, reference, id=f"{kind}-{name}-{reference.rsplit('::', 1)[-1]}")
    for kind, name, entry in _cases(TAUGHT)
    for reference in entry.proof
])
def test_taught_proof_names_the_item(kind, name, reference):
    source, param_id = _test_function(reference)
    words = _item_words(kind, name)
    named_in_source = any(_mentions(source, word, ignore_case=False) for word in words)
    named_in_id = param_id is not None and any(
        _mentions(param_id, word, ignore_case=True) for word in words)
    assert named_in_source or named_in_id, (
        f"{kind} {name}: proof {reference!r} does not name the item (looked for {words} in the "
        f"test's source or its id), so it does not prove the teaching")


@pytest.mark.parametrize(("kind", "name", "reference"), [
    pytest.param(kind, name, reference, id=f"{kind}-{name}-{reference.rsplit(':', 1)[-1]}")
    for kind, entries in COVERAGE.items()
    for name, entry in entries.items()
    for reference in entry.where
])
def test_where_names_real_education_code(kind, name, reference):
    # _resolve raises ImportError or AttributeError for a name that has gone; None would be a bad entry.
    pytest.importorskip("PySide6")
    assert _resolve(reference) is not None, f"{kind} {name}: {reference} resolves to None"
