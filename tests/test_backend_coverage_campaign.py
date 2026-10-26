'''Deterministic headless synthetic multi-camera coverage for every target/backend row.
'''
from __future__ import annotations

import pytest

from pyCamSet.calibration_targets.core.target_registry import TARGET_NAMES, target_class
from equivalence.synthetic_rig import make_synthetic_rig


# Each row is one declared target/backend combination in the release matrix.
# The 12 rows cover nine families; three families expose two detector choices.
_TARGET_BACKEND_ROWS = (
    ("ChArUco", "aruco1"),
    ("ChArUco", "aruco2"),
    ("ChArUco2", "aruco2"),
    ("Ccube", "aruco1"),
    ("Ccube", "aruco2"),
    ("Ccube2", "aruco2"),
    ("CIco", "aruco1"),
    ("CIco", "aruco2"),
    ("CIco2", "aruco2"),
    ("PuzzleBoard", "puzzle_board"),
    ("PuzzleBoardCube", "puzzle_board"),
    ("PuzzleBoardIco", "puzzle_board"),
)

# Small, deterministic targets keep all projected detections inside the camera
# frame while still providing enough geometry for the A0 multi-camera generator.
_TARGET_SPECS = {
    "ChArUco": {"num_squares_x": 8, "num_squares_y": 8, "square_size": 4.0},
    "ChArUco2": {"num_squares_x": 8, "num_squares_y": 8},
    "Ccube": {"length": 40.0, "n_points": 5},
    "Ccube2": {"length": 40.0, "n_points": 5},
    "CIco": {"length": 100.0, "n_points": 10},
    "CIco2": {"length": 100.0, "n_points": 12},
    "PuzzleBoard": {
        "num_squares_x": 8,
        "num_squares_y": 8,
        "square_size": 0.5,
        "start_x": 128,
        "start_y": 128,
    },
    "PuzzleBoardCube": {"n_points": 6},
    "PuzzleBoardIco": {"n_points": 6},
}


def test_backend_campaign_declares_all_nine_families_and_twelve_rows():
    """Fail visibly if a registered family or backend row is omitted."""
    assert len(_TARGET_BACKEND_ROWS) == 12
    covered = {family for family, _backend in _TARGET_BACKEND_ROWS}
    assert covered == set(TARGET_NAMES)
    assert len(set(_TARGET_BACKEND_ROWS)) == 12


@pytest.mark.parametrize("family, backend", _TARGET_BACKEND_ROWS)
def test_a0_projects_multicamera_detections_for_each_backend_row(family, backend):
    """A0-style projected rows cover phases 2–4 without invoking a detector.

    These are synthetic-detection coverage, not image-detection tests: the
    generator projects target geometry directly into six poses and three
    cameras. Optional detector packages are gated per row so their absence
    is reported as a skip, never as a passing backend.
    """
    if backend == "aruco2":
        pytest.importorskip("aruco2")
    if backend == "puzzle_board":
        pytest.importorskip("puzzle_board")

    target_type = target_class(family)
    backend_arg = (
        {"marker_backend": backend}
        if family in {"ChArUco", "Ccube", "CIco"}
        else {}
    )
    target = target_type(**_TARGET_SPECS[family], **backend_arg)
    assert backend in target.DETECTOR_BACKENDS, (
        f"{family} does not declare the requested {backend} backend")
    assert target.detector_parameterisation(backend).name.endswith(backend)

    rig = make_synthetic_rig(target, camera_count=3, image_count=6)
    rows = rig.detections.get_data()
    assert rows is not None and rows.ndim == 2 and len(rows) > 0
    assert set(rows[:, 0].astype(int)) == {0, 1, 2}, (
        f"{family}/{backend}: not all three cameras contributed projected rows")
    assert set(rows[:, 1].astype(int)) == {0, 1, 2, 3, 4, 5}, (
        f"{family}/{backend}: not all six synthetic poses contributed rows")
    assert rig.detections.cam_names == list(rig.camera_names)
