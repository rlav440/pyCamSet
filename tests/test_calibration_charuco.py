"""ChArUco accuracy and headless target-factory regressions."""

from pathlib import Path

import numpy as np
import pytest

from pyCamSet import ChArUco, calibrate_cameras
from pyCamSet.calibration_targets.create_charuco import (
    build_charuco,
    generate_charuco_target,
)

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
    target = build_charuco(num_squares_x=5, num_squares_y=7, square_size=4)
    assert target.point_data.ndim == 3
    assert target.point_data.shape[0] == 1
    assert target.point_data.shape[2] == 3
    detection = target.find_in_image(target._render_board(px_per_mm=20))
    assert detection.has_data
    assert detection.data_len >= 4

    _, saved = generate_charuco_target(
        num_squares_x=5,
        num_squares_y=7,
        square_size=4,
        output_dir=tmp_path,
        file_name="nested/charuco.txt",
        export_kind="svg",
    )
    assert saved == (tmp_path / "nested/charuco.svg").resolve()
    assert saved.exists()
    assert saved.stat().st_size > 0


def test_charuco_factory_validates_backend_and_dictionary() -> None:
    with pytest.raises(ValueError, match="marker_backend"):
        build_charuco(5, 7, 4, marker_backend="unknown")
    with pytest.raises(ValueError, match="Unknown ArUco dictionary"):
        build_charuco(5, 7, 4, aruco_dict="NOT_A_DICTIONARY")


def test_puzzleboard_factory_is_headless_when_dependency_is_available(tmp_path: Path) -> None:
    pytest.importorskip("puzzle_board")
    from pyCamSet.calibration_targets.create_puzzleboard import generate_puzzleboard_target

    _, saved = generate_puzzleboard_target(
        num_squares_x=5,
        num_squares_y=6,
        square_size=10,
        output_dir=tmp_path,
        file_name="puzzle.txt",
        export_kind="svg",
    )
    assert saved == (tmp_path / "puzzle.svg").resolve()
    assert saved.exists()
    assert saved.stat().st_size > 0


def test_puzzleboard_cube_factory_is_headless_when_dependency_is_available(tmp_path: Path) -> None:
    pytest.importorskip("puzzle_board")
    from pyCamSet.calibration_targets.create_puzzleboard_cube import generate_puzzleboard_cube_target

    _, saved = generate_puzzleboard_cube_target(
        n_points=5,
        length=40,
        output_dir=tmp_path,
        file_name="cube.txt",
        export_kind="svg",
    )
    assert saved == (tmp_path / "cube.svg").resolve()
    assert saved.exists()
    assert saved.stat().st_size > 0