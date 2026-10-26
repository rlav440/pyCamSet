"""D2.6 per-view error for telecentric cameras is measured with the telecentric model.

Found while testing on real rigs: D2.6 ran OpenCV's pinhole pose-only solve on
telecentric cameras, a different model from the one they were calibrated under.
"""
import numpy as np
import pytest

from pyCamSet.cameras.telecentric_camera import TelecentricCamera
from pyCamSet.utils.intrinsics_report import (
    _board_errors, telecentric_project, telecentric_view_rms)


def _camera():
    K = np.array([[120.0, 0.0, 320.0], [0.0, 121.0, 240.0], [0.0, 0.0, 1.0]])
    return TelecentricCamera(intrinsic=K, res=[640, 480], distortion_coefs=np.array([1e-7]))


def _board():
    grid = np.stack(np.meshgrid(np.arange(6), np.arange(5)), -1).reshape(-1, 2).astype(float)
    return np.column_stack([grid, np.zeros(len(grid))])


@pytest.mark.parametrize("seed", range(6))
def test_an_exact_telecentric_view_has_zero_error_from_either_tilt(seed):
    rng = np.random.default_rng(seed)
    cam, board = _camera(), _board()
    pixels = telecentric_project(cam, board, rng.normal(size=3) * 0.6,
                                 np.array([rng.normal(), rng.normal(), 3.0]))
    assert telecentric_view_rms(board, pixels, cam) == pytest.approx(0.0, abs=1e-6)


def test_noise_gives_its_expected_rms_and_the_error_is_never_below_the_fit_floor():
    rng = np.random.default_rng(1)
    cam, board = _camera(), _board()
    pixels = telecentric_project(cam, board, np.array([0.3, -0.2, 0.1]), np.array([0.5, -0.4, 0.0]))
    noisy = pixels + rng.normal(scale=0.5, size=pixels.shape)
    rms = telecentric_view_rms(board, noisy, cam)
    assert 0.4 < rms < 0.8  # 0.5 px per axis -> ~0.71 px, less the five fitted pose terms


def test_board_errors_route_telecentric_cameras_away_from_the_pinhole_solve(monkeypatch):
    import cv2

    def forbidden(*_a, **_k):
        raise AssertionError("telecentric cameras must not use cv2.calibrateCameraExtended")
    monkeypatch.setattr(cv2, "calibrateCameraExtended", forbidden)
    cam, board = _camera(), _board()
    pixels = telecentric_project(cam, board, np.array([0.2, 0.1, 0.0]), np.zeros(3))
    rms, counts = _board_errors([board[None].astype(np.float32)],
                                [pixels[None].astype(np.float32)], cam)
    assert counts.tolist() == [len(board)] and rms[0] == pytest.approx(0.0, abs=1e-4)
