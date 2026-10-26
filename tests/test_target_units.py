"""Test target-coordinate and saved-solution unit contracts.

"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from pyCamSet.utils.target_units import (
    LEGACY_UNVERIFIED,
    MILLIMETRES,
    METRES,
    saved_solution_units,
    square_size_in_point_data_units,
    point_data_unit_metres,
    target_point_data_units,
)


def _family(name: str):
    """Build a lightweight target-shaped value without importing targets."""
    return type(name, (), {"__module__": "pyCamSet.calibration_targets.synthetic"})


def test_lookup_reports_rebuilt_and_saved_solution_units_separately():
    puzzle = _family("PuzzleBoard")()
    charuco = _family("ChArUco")()

    assert target_point_data_units(puzzle) == MILLIMETRES
    assert target_point_data_units(charuco) == METRES
    assert saved_solution_units({"point_data_unit_m": 1.0}) == METRES
    assert saved_solution_units({"point_data_unit_m": 0.001}) == MILLIMETRES
    assert saved_solution_units({}) == LEGACY_UNVERIFIED
    assert point_data_unit_metres(puzzle) == 0.001
    assert point_data_unit_metres(charuco) == 1.0

    for family in ("CIco", "CIco2", "PuzzleBoardIco"):
        assert target_point_data_units(_family(family)()) == METRES


def test_square_size_contract_handles_both_sides_of_the_one_mm_boundary():
    puzzle_type = _family("PuzzleBoard")
    for side_length in (0.5, 1.0, 2.0):
        target = puzzle_type()
        target.square_size = side_length
        target.point_data = np.array([[0.0, 0.0, 0.0], [side_length, 0.0, 0.0]])
        assert square_size_in_point_data_units(target) == side_length

    cube_type = _family("PuzzleBoardCube")
    cube = cube_type()
    cube.square_size = 10.0  # Constructor attribute is mm; coordinates are m.
    assert square_size_in_point_data_units(cube) == 0.01


@pytest.mark.parametrize("square_size_mm", [2.0, 0.5])
def test_real_puzzleboard_units_match_mm_contract_on_both_sides(square_size_mm):
    pytest.importorskip("puzzle_board")
    from pyCamSet import PuzzleBoard, PuzzleBoardCube

    target = PuzzleBoard(num_squares_x=5, num_squares_y=6, square_size=square_size_mm)
    assert target_point_data_units(target) == MILLIMETRES
    assert point_data_unit_metres(target) == 0.001
    assert square_size_in_point_data_units(target) == square_size_mm

    # The cube's declared square is mm but its target point coordinates are m.
    cube = PuzzleBoardCube(n_points=5, length=40)
    assert target_point_data_units(cube) == METRES
    assert square_size_in_point_data_units(cube) == cube.square_size / 1000.0


def test_target_units_module_first_import_is_qt_and_target_module_free():
    code = (
        "import sys; import pyCamSet.utils.target_units; "
        "assert not any(k.startswith('PySide6') for k in sys.modules); "
        "allowed = {'pyCamSet', 'pyCamSet.utils', "
        "'pyCamSet.utils.general_utils', 'pyCamSet.utils.lazy_imports', "
        "'pyCamSet.utils.target_units'}; "
        "assert {k for k in sys.modules if k == 'pyCamSet' or "
        "k.startswith('pyCamSet.')} <= allowed"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=Path(__file__).resolve().parents[1],
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("module_name", ["pyCamSet.cameras", "pyCamSet.utils.saving"])
def test_unit_lookup_consumers_import_cleanly_as_the_first_import(module_name):
    code = (
        "import sys; "
        f"__import__({module_name!r}); "
        "assert 'pyCamSet.utils.target_units' in sys.modules; "
        "assert not any(name.startswith('PySide6') for name in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=Path(__file__).resolve().parents[1],
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "family,units,marker",
    [(name, unit, scale) for name, unit, scale in (
        ("ChArUco", METRES, 1.0), ("ChArUco2", METRES, 1.0),
        ("Ccube", METRES, 1.0), ("Ccube2", METRES, 1.0),
        ("CIco", METRES, 1.0), ("CIco2", METRES, 1.0),
        ("PuzzleBoard", MILLIMETRES, 0.001),
        ("PuzzleBoardCube", METRES, 1.0),
        ("PuzzleBoardIco", METRES, 1.0),
    )],
)
def test_saved_unit_marker_round_trips_outside_constructor_input(
    tmp_path, monkeypatch, family, units, marker
):
    from pyCamSet import Camera, CameraSet
    from pyCamSet.calibration_targets.core.target_detections import TargetDetection
    from pyCamSet.utils.saving import save_camset

    extrinsic = np.eye(4)
    extrinsic[0, 3] = 100.0 if units == MILLIMETRES else 0.1
    camera = Camera(
        extrinsic=extrinsic, intrinsic=np.eye(3),
        distortion_coefs=np.zeros(5), res=np.array([100, 100]), name="cam0")
    camset = CameraSet(camera_dict={"cam0": camera})
    target_type = _family(family)
    target = target_type()
    target.input_args = {"square_size": 2.0}
    detection = TargetDetection(cam_names=["cam0"], data=np.empty((0, 5)))
    camset.calibration_handler = SimpleNamespace(
        target=target,
        detection=detection,
        fixed_params={},
        problem_opts={"target_scale_rule": "measured_edges_v1"},
        missing_poses=None,
    )
    camset.calibration_params = np.array([], dtype=float)
    camset.calibration_result = np.array([], dtype=float)
    camset.calibration_jac = np.array([], dtype=float)

    path = tmp_path / "units.camset"
    save_camset(camset, path)
    target_config = json.loads(path.read_text(encoding="utf-8"))["optim"][
        "target_config"]
    stored_options = json.loads(path.read_text(encoding="utf-8"))["optim"][
        "handler_config"]["options"]

    assert target_config["point_data_unit_m"] == marker
    assert target_config["input"] == {"square_size": 2.0}
    assert "point_data_units" not in target_config["input"]
    assert stored_options["target_scale_rule"] == "measured_edges_v1"

    from pyCamSet.utils import saving

    target_kwargs = {}
    real_instance_obj = saving.instance_obj

    def construct(module, class_name, **kwargs):
        if module == "pyCamSet.calibration_targets.synthetic":
            target_kwargs.update(kwargs)
            return SimpleNamespace(**kwargs)
        if module == "types" and class_name == "SimpleNamespace":
            handler = SimpleNamespace(**kwargs)
            if "options" in kwargs:
                handler.problem_opts = kwargs["options"]
                handler.missing_poses = kwargs.get("missing_poses")
            return handler
        return real_instance_obj(module, class_name, **kwargs)

    monkeypatch.setattr(saving, "instance_obj", construct)
    loaded = saving.load_CameraSet(path)

    assert "point_data_unit_m" not in target_kwargs
    assert loaded._saved_point_data_units == units
    assert loaded.calibration_handler is not None
    assert loaded.calibration_handler.target.square_size == 2.0
    assert loaded.calibration_handler.problem_opts[
        "target_scale_rule"] == "measured_edges_v1"
    assert loaded["cam0"].extrinsic[0, 3] == extrinsic[0, 3]

    saved_again = tmp_path / "units-resaved.camset"
    save_camset(loaded, saved_again)
    resaved_config = json.loads(saved_again.read_text(encoding="utf-8"))[
        "optim"]["target_config"]
    assert resaved_config["point_data_unit_m"] == marker

    legacy_payload = json.loads(path.read_text(encoding="utf-8"))
    legacy_payload["optim"]["target_config"].pop("point_data_unit_m")
    legacy_path = tmp_path / "legacy.camset"
    legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
    legacy_loaded = saving.load_CameraSet(legacy_path)
    assert legacy_loaded.calibration_handler is not None
    assert legacy_loaded._saved_point_data_units == LEGACY_UNVERIFIED
    legacy_resaved = tmp_path / "legacy-resaved.camset"
    save_camset(legacy_loaded, legacy_resaved)
    legacy_config = json.loads(legacy_resaved.read_text(encoding="utf-8"))[
        "optim"]["target_config"]
    assert legacy_config["point_data_unit_m"] == LEGACY_UNVERIFIED