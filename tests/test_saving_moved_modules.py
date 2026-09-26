"""A camset saved before the target packages were flattened still loads whole."""

from __future__ import annotations

import importlib

import pytest

from pyCamSet.utils import saving


@pytest.mark.parametrize("old,new", sorted(saving.MOVED_MODULES.items()))
def test_every_moved_module_imports_at_its_new_path(old, new):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old)
    importlib.import_module(new)


def test_a_target_recorded_under_its_old_path_is_rebuilt(monkeypatch):
    built = saving.instance_obj("pyCamSet.calibration_targets.ccube.target", "Ccube",
                                n_points=5, length=0.04)
    assert type(built).__name__ == "Ccube"
    assert type(built).__module__ == "pyCamSet.calibration_targets.ccube"
