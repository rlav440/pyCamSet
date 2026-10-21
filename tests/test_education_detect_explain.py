"""Tests for the live Phase 1 detection explainer (pyCamSet/workflow/education_detect_explain.py).

Synthetic rendered boards run everywhere; the real-dataset checks skip unless
PYCAMSET_REAL_DATA points at the datasets.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from pyCamSet.calibration_targets.core.abstract_target import (
    DETECT_EXPLAIN_ATTRIBUTE,
    _restore_native_coordinates,
    _prepare_detection_image,
    detect_in_image,
)
from pyCamSet.calibration_targets.core.target_detections import ImageDetection
from pyCamSet.calibration_targets.markers.backend_registry import marker_backend_available
from pyCamSet.workflow.education_detect_explain import (
    MAX_DISPLAY_SIDE,
    check_consistency,
    explain_detection,
    load_phase1_photo,
)

ARUCO_STEPS = ["input", "grayscale", "detector_input", "threshold", "contours", "candidates",
               "decoding", "markers", "board_corners", "subpixel"]

needs_aruco2 = pytest.mark.skipif(not marker_backend_available("aruco2"),
                                  reason="aruco2 is not installed")

#: The explainer runs on a click in the GUI; one photo must explain well inside this budget
#: (it takes ~0.02-0.35 s on a typical development machine).
EXPLAIN_BUDGET_S = 2.0


def _warm_cpu_seconds(explain) -> float:
    """CPU seconds the calling thread spends in a *warm* ``explain()`` call.

    Not wall time and not the explanation's own ``seconds`` (also wall time): in a full-suite
    run the machine and this process are shared, and a first call also pays imports, JIT and a
    cold file read, so wall time once measured 11 s for a 0.2 s explanation. The caller has
    already made one call (the warm-up); the detector does its work on this thread
    (thread time ~= wall time when run alone), so this measures the explanation's own cost.
    """
    started = time.thread_time()
    explain()
    return time.thread_time() - started


def _pad(image: np.ndarray, border: int = 40) -> np.ndarray:
    return cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=255)


def _tilt(image: np.ndarray) -> np.ndarray:
    """A mild perspective warp, so sub-pixel refinement has something to do."""
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * 0.04, h * 0.02], [w * 0.97, 0], [w, h * 0.96], [0, h]])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (w, h),
                               borderValue=255, flags=cv2.INTER_LINEAR)


def _bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _charuco(backend: str):
    from pyCamSet.calibration_targets.charuco import ChArUco
    target = ChArUco(num_squares_x=5, num_squares_y=4, square_size=10.0, marker_backend=backend)
    return target, _bgr(_tilt(_pad(target._render_board(10.0))))


def _charuco2():
    from pyCamSet.calibration_targets.charuco2 import ChArUco2
    target = ChArUco2(num_squares_x=5, num_squares_y=4, square_size=10.0)
    image, _ = target._render(200, border_width=6.0)
    return target, _bgr(_tilt(image))


def _ccube(backend: str):
    from pyCamSet.calibration_targets.ccube import Ccube
    target = Ccube(n_points=5, length=20.0, marker_backend=backend)
    face = np.asarray(target.textures[1], dtype=np.uint8)
    face = cv2.resize(face, (500, 500), interpolation=cv2.INTER_AREA)
    return target, _bgr(_tilt(_pad(face)))


def _ccube2():
    from pyCamSet.calibration_targets.ccube2 import Ccube2
    target = Ccube2(n_points=5, length=20.0)
    face = cv2.resize(target.face_texture(2), (520, 520), interpolation=cv2.INTER_AREA)
    return target, _bgr(_tilt(_pad(face)))


CASES = {
    "charuco_aruco1": (lambda: _charuco("aruco1"), False),
    "charuco_aruco2": (lambda: _charuco("aruco2"), True),
    "charuco2": (_charuco2, True),
    "ccube_aruco1": (lambda: _ccube("aruco1"), False),
    "ccube_aruco2": (lambda: _ccube("aruco2"), True),
    "ccube2": (_ccube2, True),
}


def _case(name):
    make, aruco2 = CASES[name]
    if aruco2 and not marker_backend_available("aruco2"):
        pytest.skip("aruco2 is not installed")
    return make()


def _same(a: ImageDetection, b: ImageDetection) -> None:
    assert a.has_data == b.has_data
    assert np.array_equal(np.asarray(a.keys), np.asarray(b.keys))
    assert np.array_equal(np.asarray(a.image_points), np.asarray(b.image_points))
    assert np.asarray(a.image_points).dtype == np.asarray(b.image_points).dtype


@pytest.mark.parametrize("name", list(CASES))
def test_explain_hooks_do_not_change_the_detection(name):
    """Default-off: the explained run returns exactly what the plain run returns."""
    target, image = _case(name)
    plain = target.find_in_image(image.copy())
    assert plain.has_data, "the synthetic board must be detectable"
    shared = detect_in_image(target, image.copy())
    events: list = []
    explained = detect_in_image(target, image.copy(), explain=events)
    _same(plain, shared)
    _same(plain, explained)
    assert events, "the instrumented detector recorded nothing"
    assert getattr(target, DETECT_EXPLAIN_ATTRIBUTE, None) is None
    # And again after an explained call: no state leaks into the next plain run.
    _same(plain, target.find_in_image(image.copy()))


@pytest.mark.parametrize("name", list(CASES))
def test_explanation_steps_and_consistency(name):
    target, image = _case(name)
    saved = target.find_in_image(image.copy())
    ex = explain_detection(image, target, saved_detection=saved)
    assert ex.error is None
    assert ex.step_keys[:len(ARUCO_STEPS)] == ARUCO_STEPS
    assert ex.step_keys[-1] == "final"
    if name in ("charuco2", "ccube2"):
        assert ex.backend == "aruco2_gridboard"
        assert "grid_check" in ex.step_keys
    assert ex.consistency.available and ex.consistency.ok, ex.consistency.message
    assert ex.consistency.max_shift_px == 0.0
    assert ex.seconds >= 0.0
    assert _warm_cpu_seconds(
        lambda: explain_detection(image, target, saved_detection=saved)) < EXPLAIN_BUDGET_S

    # thresholding really shows the markers: candidates land on the accepted markers
    markers = ex.step("markers").overlays[0]
    candidates = ex.step("candidates")
    assert len(markers) >= 2
    assert candidates.counts["became markers"] == len(markers)
    # the gallery reads the markers' bit grids (reproduction agrees with the detector)
    decoding = ex.step("decoding")
    agree, total = map(int, decoding.extra["reproduced_agrees"].split("/"))
    assert total == len(markers) and agree >= 0.8 * total
    for crop in decoding.crops:
        assert crop.image.shape == (64, 64)
        if crop.bits is not None:
            assert crop.bits.shape == (crop.marker_bits + 2,) * 2

    # geometry is in the detection frame: the final overlay is the detection itself
    final = ex.step("final")
    assert np.allclose(final.overlays[0].data, np.asarray(saved.image_points, dtype=float))
    sub = ex.step("subpixel")
    assert len(sub.overlays[2]) == len(np.asarray(ex.step("board_corners").overlays[1].data))
    for step in ex.steps:
        if step.image is not None:
            assert step.image.dtype == np.uint8
            assert max(step.image.shape[:2]) <= MAX_DISPLAY_SIDE


def test_threshold_window_sizes_follow_detector_parameters():
    target, image = _case("charuco_aruco1")
    ex = explain_detection(image, target)
    params = target.board_detectors.getDetectorParameters()
    sizes = ex.step("threshold").extra["window_sizes"]
    assert sizes[0] == params.adaptiveThreshWinSizeMin | 1
    assert all(s % 2 == 1 for s in sizes)
    assert len(ex.step("threshold").images) == len(sizes)


def test_consistency_check_reports_differences():
    keys = np.array([[0, 1], [0, 2], [1, 3]])
    points = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    same = check_consistency(keys, points, ImageDetection(keys, points))
    assert same.ok and same.matched == 3 and same.max_shift_px == 0.0
    moved = points.copy()
    moved[1] += [0.0, 0.5]
    other = check_consistency(keys[:2], points[:2], (np.vstack([keys[:2], [[2, 9]]]),
                                                     np.vstack([moved[:2], [[1.0, 1.0]]])))
    assert not other.ok
    assert other.missing_keys == [[2, 9]] and other.extra_keys == []
    assert other.max_shift_px == pytest.approx(0.5)
    none = check_consistency(keys, points, None)
    assert not none.available


def test_rescale_and_gamma_geometry_maps_to_native_pixels():
    """With a rescaled detector input, steps carry their own image scale; geometry stays native."""
    puzzle = pytest.importorskip("puzzle_board")
    del puzzle
    try:
        from pyCamSet.utils.cairo_dll_helper import cairosvg_or_explain
        cairosvg = cairosvg_or_explain()
    except Exception as exc:  # noqa: BLE001 - cairo is optional
        pytest.skip(f"cairo unavailable: {exc}")
    from io import BytesIO

    from PIL import Image
    from pyCamSet.calibration_targets.puzzleboard_cube import PuzzleBoardCube

    target = PuzzleBoardCube(n_points=5, length=40.0)
    png = cairosvg.svg2png(bytestring=target._face_svg(0).encode("utf-8"), output_width=1600)
    with Image.open(BytesIO(png)) as rendered:
        gray = np.asarray(rendered.convert("L"))
    dark16 = (gray.astype(np.uint16) * 64)  # a dark 16-bit photo
    image = cv2.cvtColor(dark16, cv2.COLOR_GRAY2BGR)

    prepared, scale = _prepare_detection_image(image, rescale_and_gamma=True,
                                               preprocessing_scale=0.5, preprocessing_gamma=0.5)
    saved = _restore_native_coordinates(target.find_in_image(prepared), scale)
    events: list = []
    _same(saved, detect_in_image(target, image, rescale_and_gamma=True, preprocessing_scale=0.5,
                                 preprocessing_gamma=0.5, explain=events))
    ex = explain_detection(image, target, rescale_and_gamma=True, preprocessing_scale=0.5,
                           preprocessing_gamma=0.5, saved_detection=saved)
    assert ex.backend == "puzzleboard"
    assert ex.consistency.ok, ex.consistency.message
    for key in ("rescale_gamma", "puzzle_grid", "puzzle_bits", "puzzle_decoded",
                "puzzle_faces", "puzzle_duplicates"):
        assert key in ex.step_keys
    assert ex.step("detector_input").image_scale == pytest.approx(0.5)
    assert ex.step("input").counts["bit depth"] == 16
    decoded = ex.step("puzzle_decoded").overlays[0].data
    assert len(decoded) >= len(saved.keys)


@needs_aruco2
def test_load_phase1_photo_from_a_real_phase1_run(tmp_path):
    """End to end: Phase 1 runs on a small folder; the loader finds photo, target and result."""
    from pyCamSet.workflow import phase1
    from pyCamSet.workflow.workspace import WorkspaceManager, workspace_path_for

    from pyCamSet.calibration_targets.charuco2 import ChArUco2
    board = ChArUco2(num_squares_x=5, num_squares_y=4, square_size=10.0)
    image, _ = board._render(200, border_width=6.0)
    folder = tmp_path / "images"
    for cam, warp in (("cam0", False), ("cam1", True)):
        (folder / cam).mkdir(parents=True)
        for index in range(2):
            shot = _tilt(image) if warp else image
            cv2.imwrite(str(folder / cam / f"{cam}_{index:03d}.png"), np.roll(shot, 7 * index, axis=1))
    params = {"target": {"type": "ChArUco2", "num_squares_x": 5, "num_squares_y": 4,
                         "square_size": 10.0},
              "f_loc": str(folder), "caching": False, "high_distortion": False, "n_lim": None,
              "threads": 1, "upscale_factor": 1, "fixed_params": None, "problem_options": None,
              "selected_cameras": []}
    workspace = WorkspaceManager(workspace_path_for(folder))
    run = phase1.run(dict(params), workspace)
    assert run["error"] is None, run["error"]
    run_dir = workspace.run_dir("phase1", run["run_id"])

    photo = load_phase1_photo(run_dir, "cam1", 1)
    assert photo.photo_path.name == "cam1_001.png"
    assert photo.mapping == "manifest" and photo.verified
    assert photo.cameras == ("cam0", "cam1")
    assert photo.saved_detection is not None and photo.saved_detection.has_data
    ex = photo.explain()
    assert ex.consistency.ok, ex.consistency.message

    class _Context:  # the GUI's run context shape
        pass
    context = _Context()
    context.run_dir = run_dir
    context.metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert load_phase1_photo(context, "cam0", 0).photo_path.name == "cam0_000.png"
    missing = load_phase1_photo(run_dir, "cam0", 5)
    assert missing.photo_path is None and missing.reason


# ---------------------------------------------------------------------------------------------
# Real datasets (read-only; skipped unless PYCAMSET_REAL_DATA is set)
# ---------------------------------------------------------------------------------------------

# The data root is read from PYCAMSET_REAL_DATA (a folder holding "1 Data/pyCamSet_paper" and
# "2 Pilot Data/Autoregulation_Poster"); the real-dataset cases skip when it is unset.
_REAL_DATA = os.environ.get("PYCAMSET_REAL_DATA")
_ROOT = Path(_REAL_DATA) if _REAL_DATA else Path("PYCAMSET_REAL_DATA-not-set")
_PAPER = _ROOT / "1 Data" / "pyCamSet_paper"
_MOUSE = (_ROOT / "2 Pilot Data" / "Autoregulation_Poster" / "mouse1" / "calibration" / "experiment_001"
          / ".pycamset_workspace" / "phase1_runs" / "20261008_083008_4927f7")


def _phase1_run(*parts: str) -> Path:
    return _PAPER.joinpath(*parts, "experiment_001", ".pycamset_workspace", "phase1_runs")


REAL_RUNS = {
    "mouse1_ccube2_aruco2": _MOUSE,
    "ccube1_aruco1": _phase1_run("ccube1", "5_squares") / "20261008_154313_abd3c6",
    "ccube1_aruco2": _phase1_run("ccube1", "5_squares") / "20261008_154404_a69f1d",
    "ccube2_aruco2": _phase1_run("ccube2", "5_corners") / "20261008_155246_e901ca",
    "pcube_puzzleboard": _phase1_run("pcube", "5_squares") / "20261008_161036_51a0ed",
}


@pytest.mark.parametrize("name", list(REAL_RUNS))
def test_real_dataset_recompute_matches_saved_phase1(name):
    if not _REAL_DATA:
        pytest.skip("PYCAMSET_REAL_DATA is not set (folder holding the real calibration datasets)")
    run_dir = REAL_RUNS[name]
    if not (run_dir / "metadata.json").is_file():
        pytest.skip(f"dataset not available: {run_dir}")
    if name.startswith(("ccube1", "charuco1")) and int(cv2.__version__.split(".")[0]) < 5:
        # Detected under OpenCV 5; OpenCV 4's CharucoDetector places sub-pixel corners a
        # few hundredths of a pixel differently, so no exact recompute is possible.
        pytest.skip("saved under OpenCV 5; an exact ChArUco recompute needs OpenCV 5")
    if "ccube2" in name:
        # Saved before GRID_BOARD_CORNER_REVISION 2, which re-seeds the corners the saddle
        # search refused (measured: one moved 4.9 px on ccube2 cam5 #7), so a live recompute
        # is not expected to match these runs exactly.
        pytest.skip("saved before the grid-board corner re-seeding; no exact recompute")
    metadata =json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    cameras = metadata["params"]["selected_cameras"]
    checked = 0
    for camera in (cameras[0], cameras[len(cameras) // 2]):
        for index in (0, 7, 33):
            photo = load_phase1_photo(run_dir, camera, index)
            if photo.photo_path is None:
                continue
            ex = photo.explain()
            check = ex.consistency
            assert check.available
            assert check.ok, f"{camera} #{index}: {check.message}"
            assert check.missing_keys == [] and check.extra_keys == []
            assert not (check.matched and check.max_shift_px > 1e-6)
            assert ex.seconds >= 0.0
            assert _warm_cpu_seconds(photo.explain) < EXPLAIN_BUDGET_S
            checked += 1
    assert checked


def test_consistency_names_a_saved_point_decoded_twice():
    """A run saved before duplicate keys were dropped holds one key twice; say so."""
    from pyCamSet.workflow.education_detect_explain import check_consistency

    saved = {"keys": [(0, 1), (4, 3), (4, 3)],
             "image_points": [(1.0, 1.0), (5.0, 5.0), (9.0, 9.0)]}
    check = check_consistency([(0, 1)], [(1.0, 1.0)], saved)
    assert not check.ok
    assert check.missing_keys == [[4, 3]] and check.saved_duplicate_keys == [[4, 3]]
    assert "decoded at two places" in check.message
