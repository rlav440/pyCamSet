"""ChArUco accuracy and headless target-factory regressions."""

from pathlib import Path

import numpy as np
import pytest

from pyCamSet import ChArUco, calibrate_cameras

MAX_MEAN_REPROJECTION_PX = 1.8


@pytest.mark.data
@pytest.mark.slow
def test_calibration_charuco(data_dir):
    """A full ChArUco calibration must stay within its reprojection baseline."""
    target = ChArUco(20, 20, 4)
    cams = calibrate_cameras(
        f_loc=data_dir / "calibration_charuco",
        calibration_target=target,
        save=False,
        problem_options={"outliers": "n"},
    )
    mean_reprojection = np.mean(
        np.linalg.norm(np.reshape(cams.calibration_result, (-1, 2)), axis=1)
    )
    assert mean_reprojection < MAX_MEAN_REPROJECTION_PX


def test_charuco_renders_detects_and_returns_saved_path(tmp_path: Path) -> None:
    target = ChArUco(num_squares_x=5, num_squares_y=7, square_size=4)
    assert target.point_data.ndim == 3
    assert target.point_data.shape[0] == 1
    assert target.point_data.shape[2] == 3
    detection = target.find_in_image(target._render_board(px_per_mm=20))
    assert detection.has_data
    assert detection.data_len >= 4

    saved = target.save_printable(tmp_path / "nested/charuco.txt", "svg")
    assert saved == (tmp_path / "nested/charuco.svg").resolve()
    assert saved.exists()
    assert saved.stat().st_size > 0


@pytest.mark.parametrize("backend", ["aruco1", "aruco2"])
def test_charuco_construction_parameters_excludes_apriltag(backend: str) -> None:
    """AprilTag dictionaries are the only ones aruco1/aruco2 disagree on --
    they must never be offered, on either backend, in Create Target.  The
    detector is chosen in the detection phase, so the offer is the same
    18 names whichever backend is asked about."""
    choices = ChArUco.construction_parameters(backend).parameter("a_dict").choices
    values = [choice.value for choice in choices]
    assert not any(value.startswith("DICT_APRILTAG_") for value in values)
    assert len(values) == 18


def test_charuco_validates_backend_and_dictionary() -> None:
    """A target names the detectors it can be read with, and the dictionary
    it is printed from, and refuses anything else."""
    with pytest.raises(ValueError, match="cannot be detected with 'unknown'"):
        ChArUco(5, 7, 4, marker_backend="unknown")
    with pytest.raises(ValueError, match="Unknown ArUco dictionary"):
        ChArUco(5, 7, 4, a_dict="NOT_A_DICTIONARY")


def test_puzzleboard_factory_is_headless_when_dependency_is_available(tmp_path: Path) -> None:
    pytest.importorskip("puzzle_board")
    from pyCamSet import PuzzleBoard

    board = PuzzleBoard(num_squares_x=5, num_squares_y=6, square_size=10)
    saved = board.save_printable(tmp_path / "puzzle.txt", "svg")
    assert saved == (tmp_path / "puzzle.svg").resolve()
    assert saved.exists()
    assert saved.stat().st_size > 0


def test_puzzleboard_cube_factory_is_headless_when_dependency_is_available(tmp_path: Path) -> None:
    pytest.importorskip("puzzle_board")
    from pyCamSet import PuzzleBoardCube

    cube = PuzzleBoardCube(n_points=5, length=40)
    saved = cube.save_printable(tmp_path / "cube.txt", "svg")
    assert saved == (tmp_path / "cube.svg").resolve()
    assert saved.exists()
    assert saved.stat().st_size > 0