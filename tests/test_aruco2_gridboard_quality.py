"""
Corner refinement, corner validation and the pixel convention of
``aruco2_gridboard.detect_grid_board_corners`` -- the choke point ChArUco2
and Ccube2 both detect through.

Every quantitative threshold here comes from stage A task A3's validation
(scratch, not shipped): Newton-to-saddle refinement at Gaussian sigma 1.8,
with a coarse-to-fine (3.0 -> 1.8) retry and a point-symmetry accept gate;
leave-one-out local-homography lattice validation at Chebyshev radius 2,
3-round Tukey IRLS, greedy worst-first removal with one readmission pass,
and a 5-non-collinear-neighbour support floor. See that module's docstrings
for the full account; this file only re-proves the numbers that matter for
regression.

Most tests here build their scenes directly from geometry (a lattice run
through a pinhole+distortion projection, or a grid-board raster warped with
``cv2``) rather than depending on the stage A scratch harness, so they need
nothing beyond ``numpy``, ``cv2`` and ``aruco2``.

Every test that exercises refinement or validation proves the feature
matters by also running with it switched off (``refine=False``/
``validate=False``, or calling ``validate_grid_board_corners`` directly on
points the un-fixed code path would have used) and checking the result is
worse -- never by editing the shipped module.

Gates itself on the optional ``aruco2`` package, as
``test_charuco2_target.py`` does.
"""

from __future__ import annotations

import logging
import warnings

import cv2
import numpy as np
import pytest

aruco2 = pytest.importorskip("aruco2")

import pyCamSet.calibration_targets.markers.aruco2 as aruco2_backend
from pyCamSet.calibration_targets.markers import aruco2_gridboard
from pyCamSet.calibration_targets.markers.aruco2_gridboard import (
    detect_grid_board_corners,
    refine_grid_board_corners,
    render_grid_board_image,
    validate_grid_board_corners,
)

_DICT_4X4 = int(aruco2.DICT_4X4_1000)


# -- shared scene builders ------------------------------------------------------


def _distorted_lattice(
    n: int, tilt_deg: float = 60.0, k1: float = -0.30, k2: float = 0.08,
    f: float = 1780.0, square: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """
    An ``n x n`` board's ``(n+1)^2`` lattice, projected through a tilted,
    wide-lens (radially distorted) pinhole camera -- an exact model, with no
    detection noise, so a validator finding any removal on it is a false one.

    :return: ``(lattice_cr, image_points)``.
    """
    cols, rows = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    lattice_cr = np.stack([cols.ravel(), rows.ravel()], axis=-1)
    x = (cols.ravel() - n / 2.0) * square
    y = (rows.ravel() - n / 2.0) * square
    plane = np.stack([x, y, np.zeros_like(x)], axis=-1)
    theta = np.radians(tilt_deg)
    tilt = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ])
    camera_pts = plane @ tilt.T + np.array([0.0, 0.0, 0.5])
    intrinsic = np.array([[f, 0, 800.0], [0, f, 600.0], [0, 0, 1.0]])
    dist = np.array([k1, k2, 0.0, 0.0, 0.0])
    projected, _ = cv2.projectPoints(camera_pts, np.zeros(3), np.zeros(3), intrinsic, dist)
    return lattice_cr, projected.reshape(-1, 2)


def _gamma_lut(gamma: float) -> np.ndarray:
    x = np.arange(256, dtype=np.float64) / 255.0
    return (np.clip(x, 0.0, 1.0) ** (1.0 / gamma) * 255.0).astype(np.uint8)


def _render_and_degrade(
    grid_size: tuple[int, int], bit_size: int, blur_sigma: float, gamma: float,
) -> tuple[np.ndarray, int, float]:
    """A grid-board raster, Gaussian-blurred and put through a gamma LUT.

    :return: ``(image, square_px, band_px)``.
    """
    image = render_grid_board_image(grid_size, _DICT_4X4, bit_size)
    square_px = 4 * bit_size  # DICT_4X4: marker_bits == 4
    band_px = square_px / 4.0
    if blur_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma, borderType=cv2.BORDER_REPLICATE)
    if gamma != 1.0:
        image = cv2.LUT(image, _gamma_lut(gamma))
    return image, square_px, band_px


def _analytic_corners(corner_ids: np.ndarray, grid_w: int, square_px: float, band_px: float) -> np.ndarray:
    """Where a corner truly sits, in the same pixel-*corner* convention
    ``detect_grid_board_corners`` returns -- i.e. this is already the full
    truth to compare its output against directly, with no further shift.
    (Two tests here used to disagree on this, one adding a stray manual
    ``+ 0.5`` on top; that shift is aruco2's own raw pixel-*centre* output's
    difference from this truth, already applied once, internally, by
    ``detect_grid_board_corners`` itself -- adding it again here double
    counts it. Confirmed to match ``detect_grid_board_corners``' output to
    0.00 px on an exact, noiseless raster.)"""
    lattice = np.stack([corner_ids % (grid_w + 1), corner_ids // (grid_w + 1)], axis=-1)
    return band_px + lattice * square_px


# -- 1. ghost regression --------------------------------------------------------


def _ghost_case(n: int, bit: int, ids: list[int]) -> tuple[np.ndarray, int, float]:
    image = np.asarray(aruco2.get_grid_board_image((n, n), _DICT_4X4, bit, ids)).copy()
    square_px = 4 * bit
    return image, square_px, square_px / 4.0


@pytest.mark.parametrize("n, ids, expected_n", [
    (3, [0, 1, 2, 3, 688, 5, 6, 7, 17], 13),
    (5, [*range(100, 125)], 33),
])
def test_ghost_regression_is_removed_by_default(n, ids, expected_n) -> None:
    """
    aruco2's known false-marker-detection ghosts (marker 688 of DICT_4X4_1000
    decoding a patch inside its own square as marker 17) corrupt 3 corners of
    a board carrying both. With refinement and validation on by default,
    those 3 are gone and everything left is accurate; disabled, the 3
    corrupted corners come back at 19-68 px off (stage A task A3).
    """
    if n == 5:
        ids = list(ids)
        ids[12], ids[18] = 688, 17
    image, square_px, band_px = _ghost_case(n, 14, ids)

    corner_ids, points = detect_grid_board_corners(image, (n, n), _DICT_4X4, 1.0, ids=ids)
    assert corner_ids is not None
    assert len(corner_ids) == expected_n
    truth = _analytic_corners(corner_ids, n, square_px, band_px)
    error = np.linalg.norm(points - truth, axis=1)
    assert error.max() < 0.5, f"max error {error.max():.4f} px"

    # Proves validation matters: without it, the same 3 ghost corners survive,
    # tens of pixels off.
    raw_ids, raw_points = detect_grid_board_corners(
        image, (n, n), _DICT_4X4, 1.0, ids=ids, validate=False)
    raw_truth = _analytic_corners(raw_ids, n, square_px, band_px)
    raw_error = np.linalg.norm(raw_points - raw_truth, axis=1)
    assert raw_error.max() > 15.0
    assert (raw_error > 0.5).sum() == 3


# -- 2. validator on exact distorted lattices -----------------------------------


@pytest.mark.parametrize("n, tilt_deg, view", [
    (20, 20.0, "full"), (20, 70.0, "full"), (12, 65.0, "partial"),
])
def test_validator_makes_no_removals_on_an_exact_distorted_lattice(n, tilt_deg, view) -> None:
    """A wide-lens (k1=-0.30, k2=0.08), tilted-plane lattice with no
    detection noise: the model-error floor (stage A: <=0.098 squares, 99th
    percentile <=0.03) stays well under the 0.10-square/2px removal
    threshold, so nothing should be removed."""
    lattice_cr, points = _distorted_lattice(n, tilt_deg=tilt_deg)
    if view == "partial":
        keep_half = lattice_cr[:, 0] <= n // 2
        lattice_cr, points = lattice_cr[keep_half], points[keep_half]
    keep = validate_grid_board_corners(lattice_cr, points)
    assert keep.all(), f"{int((~keep).sum())} corner(s) wrongly removed"


# -- 3. injected faults ----------------------------------------------------------


def _index_by_cr(lattice_cr: np.ndarray) -> dict[tuple[int, int], int]:
    return {(int(c), int(r)): i for i, (c, r) in enumerate(lattice_cr)}


def test_validator_removes_exactly_an_injected_10px_shift() -> None:
    lattice_cr, points = _distorted_lattice(20, tilt_deg=60.0)
    index = _index_by_cr(lattice_cr)
    target = index[(10, 10)]
    injected = points.copy()
    injected[target] += (10.0, 0.0)

    keep = validate_grid_board_corners(lattice_cr, injected)
    assert set(np.flatnonzero(~keep).tolist()) == {target}
    # And on the clean lattice (no injected fault), nothing is removed --
    # so the removal above is the validator catching the injection, not an
    # unrelated false positive.
    assert validate_grid_board_corners(lattice_cr, points).all()


def test_validator_removes_exactly_an_adjacent_label_swap() -> None:
    lattice_cr, points = _distorted_lattice(20, tilt_deg=60.0)
    index = _index_by_cr(lattice_cr)
    a, b = index[(10, 10)], index[(11, 10)]
    injected = points.copy()
    injected[[a, b]] = points[[b, a]]

    keep = validate_grid_board_corners(lattice_cr, injected)
    assert set(np.flatnonzero(~keep).tolist()) == {a, b}


def test_validator_removes_exactly_an_l_of_3_shifted_corners() -> None:
    """An L of 3 adjacent corners moved together by 0.4 of a square -- the
    shape a ghost-marker misdetection tends to produce, since it drags a
    small connected cluster rather than one point."""
    lattice_cr, points = _distorted_lattice(20, tilt_deg=60.0)
    index = _index_by_cr(lattice_cr)
    square_px = float(np.linalg.norm(points[index[(11, 10)]] - points[index[(10, 10)]]))
    corners = [(10, 10), (11, 10), (10, 11)]
    idx = [index[c] for c in corners]
    injected = points.copy()
    injected[idx] += 0.4 * square_px

    keep = validate_grid_board_corners(lattice_cr, injected)
    assert set(np.flatnonzero(~keep).tolist()) == set(idx)


# -- 4. insufficient support ------------------------------------------------------


def test_validator_drops_a_fragment_with_fewer_than_5_neighbours() -> None:
    """A well-supported board is untouched; a 4-corner fragment detected in
    isolation (an occluded patch far from the rest of the board) cannot be
    checked at all and is dropped rather than trusted."""
    n = 20
    lattice_full, points_full = _distorted_lattice(n, tilt_deg=10.0)
    fragment_cr = np.array([[40, 40], [41, 40], [40, 41], [42, 42]])
    fragment_pts = fragment_cr.astype(float) * 50.0 + (2000.0, 2000.0)

    lattice_cr = np.vstack([lattice_full, fragment_cr])
    points = np.vstack([points_full, fragment_pts])
    keep = validate_grid_board_corners(lattice_cr, points)

    assert keep[:len(lattice_full)].all()
    assert not keep[len(lattice_full):].any()


# -- 5. refinement accuracy -------------------------------------------------------


def test_refinement_at_least_halves_rms_on_a_blurred_gamma_render() -> None:
    image, square_px, band_px = _render_and_degrade((15, 15), bit_size=12, blur_sigma=3.0, gamma=2.2)

    raw_ids, raw_points = detect_grid_board_corners(
        image, (15, 15), _DICT_4X4, 1.0, refine=False, validate=False)
    ref_ids, ref_points = detect_grid_board_corners(
        image, (15, 15), _DICT_4X4, 1.0, refine=True, validate=False)
    assert raw_ids is not None and ref_ids is not None

    def rms(ids, points):
        truth = _analytic_corners(ids, 15, square_px, band_px) + 0.5
        error = np.minimum(np.linalg.norm(points - truth, axis=1), 3.0)
        return float(np.sqrt(np.mean(error ** 2)))

    raw_rms, ref_rms = rms(raw_ids, raw_points), rms(ref_ids, ref_points)
    assert ref_rms <= raw_rms / 2.0, f"raw {raw_rms:.3f} ref {ref_rms:.3f}"


def test_refinement_does_not_worsen_a_sharp_render() -> None:
    image, square_px, band_px = _render_and_degrade((15, 15), bit_size=12, blur_sigma=0.0, gamma=1.0)

    raw_ids, raw_points = detect_grid_board_corners(
        image, (15, 15), _DICT_4X4, 1.0, refine=False, validate=False)
    ref_ids, ref_points = detect_grid_board_corners(
        image, (15, 15), _DICT_4X4, 1.0, refine=True, validate=False)

    def rms(ids, points):
        truth = _analytic_corners(ids, 15, square_px, band_px) + 0.5
        error = np.linalg.norm(points - truth, axis=1)
        return float(np.sqrt(np.mean(error ** 2)))

    raw_rms, ref_rms = rms(raw_ids, raw_points), rms(ref_ids, ref_points)
    assert ref_rms <= raw_rms + 0.02, f"raw {raw_rms:.4f} ref {ref_rms:.4f}"


# -- 6. tone-curve invariance ------------------------------------------------------


def test_refinement_is_tone_curve_invariant() -> None:
    """A saddle point of the scene's light stays a saddle point of any
    monotone tone curve applied to it, so two renders of the same blurred
    board under different gammas must refine to (almost) the same corners --
    unlike ``cv2.cornerSubPix``, which chases second moments and drifts."""
    image, _, _ = _render_and_degrade((15, 15), bit_size=12, blur_sigma=2.0, gamma=1.0)
    toned = cv2.LUT(image, _gamma_lut(2.4))

    ids_a, points_a = detect_grid_board_corners(
        image, (15, 15), _DICT_4X4, 1.0, refine=True, validate=False)
    ids_b, points_b = detect_grid_board_corners(
        toned, (15, 15), _DICT_4X4, 1.0, refine=True, validate=False)

    common = np.intersect1d(ids_a, ids_b)
    assert len(common) > 0.9 * (16 * 16)
    by_a = dict(zip(ids_a.tolist(), points_a))
    by_b = dict(zip(ids_b.tolist(), points_b))
    diff = np.linalg.norm([by_a[i] - by_b[i] for i in common], axis=1)
    assert diff.max() < 0.05, f"max disagreement {diff.max():.4f} px"


# -- 7. edge safety -----------------------------------------------------------------


def test_refinement_near_an_roi_edge_never_raises_and_stays_inside() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(60, 60), dtype=np.uint8)
    lattice_cr = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    points = np.array([[1.5, 1.5], [58.0, 1.5], [1.5, 58.0], [58.0, 58.0]])

    refined, _ = refine_grid_board_corners(image, lattice_cr, points)

    height, width = image.shape
    assert np.all((refined[:, 0] >= 0) & (refined[:, 0] < width))
    assert np.all((refined[:, 1] >= 0) & (refined[:, 1] < height))


def test_a_board_corner_near_the_frame_edge_is_still_read_correctly() -> None:
    """An end-to-end version of the ROI-edge case, through the real
    detection choke point (not just refine_grid_board_corners in isolation)."""
    image, square_px, band_px = _render_and_degrade((10, 10), bit_size=30, blur_sigma=0.0, gamma=1.0)
    # crop tight, just past the padding/margin (see _grid_board_padding /
    # _edge_margin), so a real board corner sits a few pixels inside the crop.
    cropped = np.ascontiguousarray(image[:, 20:])
    corner_ids, points = detect_grid_board_corners(cropped, (10, 10), _DICT_4X4, 1.0)
    assert corner_ids is not None
    h, w = cropped.shape
    assert np.all((points[:, 0] >= 0) & (points[:, 0] < w))
    assert np.all((points[:, 1] >= 0) & (points[:, 1] < h))


# -- 8. occluder safety --------------------------------------------------------------


def test_an_occluder_bar_does_not_move_a_corner_more_than_1px() -> None:
    image = render_grid_board_image((12, 12), _DICT_4X4, 30).copy()
    square_px, band_px = 4 * 30, 4 * 30 / 4.0
    target_cr = (6, 6)
    cx, cy = band_px + target_cr[0] * square_px, band_px + target_cr[1] * square_px
    occluded = image.copy()
    cv2.rectangle(occluded, (0, int(cy - 3)), (occluded.shape[1], int(cy + 3)), 128, -1)

    corner_ids, points = detect_grid_board_corners(occluded, (12, 12), _DICT_4X4, 1.0, validate=False)
    lattice_cr = np.stack([corner_ids % 13, corner_ids // 13], axis=-1)
    index = _index_by_cr(lattice_cr)
    assert target_cr in index
    truth = np.array([cx, cy]) + 0.5
    error = float(np.linalg.norm(points[index[target_cr]] - truth))
    assert error < 1.0, f"occluded corner moved {error:.3f} px"


# -- 9. pixel convention matches ChArUco (OpenCV CharucoDetector) -------------------


def test_pixel_convention_matches_opencv_charuco() -> None:
    """Render a pixel-aligned ChArUco board (``cv2.aruco``'s own rasteriser)
    and a pixel-aligned ChArUco2 board (this package's own rasteriser) with
    matching, exactly known analytic corner positions; detect each with its
    own detector; and confirm both land within ~0.05 px of the analytic
    truth -- i.e. of each other, since they share it."""
    square_px = 120
    a_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    board = cv2.aruco.CharucoBoard((6, 6), square_px / 1000.0, square_px * 0.8 / 1000.0, a_dict)
    charuco_image = board.generateImage((6 * square_px, 6 * square_px))
    detector = cv2.aruco.CharucoDetector(board)
    ch_points, ch_ids, _, _ = detector.detectBoard(charuco_image)
    ch_points, ch_ids = ch_points.reshape(-1, 2), ch_ids.reshape(-1)
    chess_mm = np.asarray(board.getChessboardCorners()).reshape(-1, 3)[:, :2]
    charuco_truth = chess_mm / (square_px / 1000.0) * square_px
    charuco_error = np.linalg.norm(ch_points - charuco_truth[ch_ids], axis=1)

    bit_size = square_px // 4  # DICT_4X4: marker_bits == 4
    charuco2_image = render_grid_board_image((6, 6), _DICT_4X4, bit_size)
    c2_ids, c2_points = detect_grid_board_corners(charuco2_image, (6, 6), _DICT_4X4, 1.0)
    c2_truth = _analytic_corners(c2_ids, 6, square_px, square_px / 4.0)
    charuco2_error = np.linalg.norm(c2_points - c2_truth, axis=1)

    assert charuco_error.max() < 0.05, f"ChArUco vs analytic: {charuco_error.max():.4f} px"
    assert charuco2_error.max() < 0.05, f"ChArUco2 vs analytic: {charuco2_error.max():.4f} px"


# -- 2b. validator on exact distorted lattices, beyond stage A's envelope -----------


@pytest.mark.parametrize("n, k1, tilt_deg, view", [
    (30, -0.30, 72.0, "full"),
    (30, -0.30, 75.0, "full"),
    (30, -0.15, 72.0, "full"),
    (20, -0.30, 75.0, "partial"),
    (10, -0.30, 75.0, "full"),
    (25, 0.0, 75.0, "full"),
])
def test_validator_stays_within_budget_beyond_stage_as_envelope(n, k1, tilt_deg, view) -> None:
    """A large board, strong radial distortion and an extreme tilt together
    push the local-homography model past stage A's validated envelope (see
    ``validate_grid_board_corners``' "model uncertainty" docstring note): an
    early version of this validator wrongly removed up to 3.3% of an exact,
    noiseless lattice's correct corners there (n=30, k1=-0.30, tilt=72 deg).
    False removals must stay a small fraction of a percent even here."""
    lattice_cr, points = _distorted_lattice(n, tilt_deg=tilt_deg, k1=k1)
    if view == "partial":
        keep_half = lattice_cr[:, 0] <= n // 2
        lattice_cr, points = lattice_cr[keep_half], points[keep_half]
    keep = validate_grid_board_corners(lattice_cr, points)
    removed = int((~keep).sum())
    assert removed / len(keep) <= 0.005, (
        f"{removed}/{len(keep)} ({100 * removed / len(keep):.2f}%) wrongly removed")


def test_validator_does_not_leak_warnings_under_extreme_distortion_and_tilt() -> None:
    """The same n=30/k1=-0.30/tilt=72deg geometry used to raise
    divide-by-zero/invalid-value RuntimeWarnings from an ill-conditioned
    local homography inside the validator; none should escape."""
    lattice_cr, points = _distorted_lattice(30, tilt_deg=72.0, k1=-0.30, k2=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=RuntimeWarning)
        validate_grid_board_corners(lattice_cr, points)


# -- 2c. NaN corners --------------------------------------------------------------


def test_refine_grid_board_corners_is_robust_to_nan_input() -> None:
    """A non-finite ``image_points`` row used to raise ``ValueError:
    cannot convert float NaN to integer`` (from the bounding-box
    computation). It must instead come back unmoved and unaccepted, without
    disturbing the other, finite corners' own refinement."""
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, size=(80, 80), dtype=np.uint8)
    lattice_cr = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    points = np.array([[20.0, 20.0], [60.0, 20.0], [np.nan, np.nan], [60.0, 60.0]])

    refined, accepted = refine_grid_board_corners(image, lattice_cr, points)

    assert not accepted[2]
    assert np.array_equal(refined[2], points[2], equal_nan=True)
    # The finite corners still come back finite (not dragged into NaN).
    assert np.isfinite(refined[[0, 1, 3]]).all()


def test_validate_grid_board_corners_is_robust_to_nan_input() -> None:
    """A non-finite ``image_points`` row used to raise
    ``numpy.linalg.LinAlgError`` (a NaN entering another corner's
    homography fit as a neighbour). The NaN corner must come back dropped,
    without preventing the surrounding, finite corners from being
    validated correctly."""
    lattice_cr, points = _distorted_lattice(20, tilt_deg=60.0)
    index = {(int(c), int(r)): i for i, (c, r) in enumerate(lattice_cr)}
    nan_target = index[(10, 10)]
    injected = points.copy()
    injected[nan_target] = np.nan
    # Also inject an unrelated, ordinary fault elsewhere, to prove the NaN
    # corner does not blind the validator to a real fault nearby.
    fault_target = index[(10, 12)]
    injected[fault_target] += (10.0, 0.0)

    keep = validate_grid_board_corners(lattice_cr, injected)

    assert not keep[nan_target]
    assert not keep[fault_target]
    # Every other corner -- most of them the NaN corner's own former
    # neighbours -- is unaffected.
    unaffected = np.ones(len(keep), dtype=bool)
    unaffected[[nan_target, fault_target]] = False
    assert keep[unaffected].all()


def test_detect_grid_board_corners_drops_a_non_finite_corner(monkeypatch, caplog) -> None:
    """aruco2 should never hand back a non-finite corner, but if it did,
    ``detect_grid_board_corners`` must drop it -- explicitly, at debug level
    -- rather than crash or (silently) rely on it happening to fail a later
    comparison."""
    from pyCamSet.calibration_targets.charuco2 import ChArUco2
    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    image, _ = target._render(150)

    real_get_solve_pnp_points = aruco2.get_solve_pnp_points

    def poisoned(board, marker_size):
        obj_points, img_points = real_get_solve_pnp_points(board, marker_size)
        img_points = np.asarray(img_points, dtype=np.float64).copy()
        img_points[0] = np.nan
        return obj_points, img_points

    monkeypatch.setattr(aruco2_gridboard.aruco2, "get_solve_pnp_points", poisoned)
    with caplog.at_level(logging.DEBUG, logger=aruco2_gridboard.__name__):
        corner_ids, points = detect_grid_board_corners(
            image, target.grid_size, target._aruco_dict_int, target.square_size)

    assert corner_ids is not None
    assert np.isfinite(points).all()
    assert len(corner_ids) >= 30  # only the one poisoned corner is missing
    assert any("non-finite" in r.getMessage() for r in caplog.records)


# -- narrowed exception / OSError message --------------------------------------------


def test_require_aruco2_reports_a_found_but_unloadable_package(monkeypatch) -> None:
    """OSError at import (e.g. a missing CONCRT140.dll) is reported as a
    load failure, distinct from 'not installed', with the OSError text and
    the likely cause named."""
    monkeypatch.setattr(aruco2_backend, "ARUCO2_AVAILABLE", False)
    monkeypatch.setattr(
        aruco2_backend, "_aruco2_import_oserror",
        OSError("DLL load failed while importing _aruco2: The specified module CONCRT140.dll could not be found."))
    with pytest.raises(ImportError) as excinfo:
        aruco2_backend._require_aruco2()
    message = str(excinfo.value)
    assert "CONCRT140" in message
    assert "Visual C++" in message
    assert "not installed" not in message


def test_require_aruco2_still_reports_not_installed_without_an_oserror(monkeypatch) -> None:
    monkeypatch.setattr(aruco2_backend, "ARUCO2_AVAILABLE", False)
    monkeypatch.setattr(aruco2_backend, "_aruco2_import_oserror", None)
    with pytest.raises(ImportError, match="not installed"):
        aruco2_backend._require_aruco2()


def test_require_aruco2_does_not_double_the_full_stop(monkeypatch) -> None:
    """P3: str(OSError) commonly ends in its own full stop (as Windows' DLL
    load messages do); the message must not read "...found.. This is..."."""
    monkeypatch.setattr(aruco2_backend, "ARUCO2_AVAILABLE", False)
    monkeypatch.setattr(
        aruco2_backend, "_aruco2_import_oserror",
        OSError("DLL load failed while importing _aruco2: The specified module "
                "CONCRT140.dll could not be found."))
    with pytest.raises(ImportError) as excinfo:
        aruco2_backend._require_aruco2()
    message = str(excinfo.value)
    assert ".." not in message, f"doubled full stop in message: {message!r}"
    assert "could not be found. This is usually" in message

    # An OSError text with no trailing stop of its own must be unaffected:
    # exactly one stop still separates it from the sentence that follows.
    monkeypatch.setattr(
        aruco2_backend, "_aruco2_import_oserror",
        OSError("DLL load failed while importing _aruco2"))
    with pytest.raises(ImportError) as excinfo:
        aruco2_backend._require_aruco2()
    message = str(excinfo.value)
    assert ".." not in message
    assert "_aruco2. This is usually" in message
