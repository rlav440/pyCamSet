"""requirements_core.txt is generated, so it can go stale.

It is the only description of the lean install -- ``pip install pyCamSet
--no-deps`` plus this file -- and nothing else would notice if a dependency
were added to pyproject.toml and not to it.
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "setup_scripts"))

from write_core_requirements import EXCLUDED, core_requirements, rendered  # noqa: E402


def test_the_file_matches_pyproject():
    actual = (ROOT / "requirements_core.txt").read_text(encoding="utf-8")

    assert actual == rendered(), (
        "requirements_core.txt is out of step with pyproject.toml; "
        "regenerate it with:\n"
        "    python setup_scripts/write_core_requirements.py"
    )


def test_one_gui_capable_contrib_opencv_distribution():
    """Every install, extras included, names one GUI-capable cv2 wheel."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    deps = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        deps.extend(extra)

    assert [dep for dep in deps if dep.startswith("opencv-")] == ["opencv-contrib-python>=4.8"]


def test_a_second_cv2_distribution_is_warned_about(monkeypatch):
    import pyCamSet

    monkeypatch.setattr(
        pyCamSet, "_installed_cv2_distributions",
        lambda: ["opencv-python 4.13.0", "opencv-contrib-python 5.0.0"])
    with pytest.warns(UserWarning, match="opencv-python 4.13.0"):
        pyCamSet._warn_on_duplicate_cv2()


def test_one_cv2_distribution_is_not_warned_about(monkeypatch, recwarn):
    import pyCamSet

    monkeypatch.setattr(
        pyCamSet, "_installed_cv2_distributions",
        lambda: ["opencv-contrib-python 5.0.0"])
    pyCamSet._warn_on_duplicate_cv2()

    assert not recwarn.list


def test_it_leaves_out_the_gui_toolkit():
    """The whole point of the lean install: no Qt."""
    assert "PySide6" not in core_requirements()


def test_the_excluded_packages_are_really_base_dependencies():
    """A typo in EXCLUDED would silently stop excluding anything."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = {
        dep.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip()
        for dep in data["project"]["dependencies"]
    }

    assert EXCLUDED <= names


def test_plotting_survives_the_lean_install():
    """Plotting is meant to work without the GUI, so it must be in there."""
    assert "pyvista" in core_requirements()
    assert "matplotlib" in core_requirements()
