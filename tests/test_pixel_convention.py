"""
One pixel frame for every detector, whatever OpenCV is installed.

pyCamSet returns image points in the pixel-centre frame
(``aruco_opencv.PIXEL_CONVENTION``): integer coordinates are pixel centres, so
a corner on the boundary between pixels n-1 and n is at n - 0.5.  The
``CharucoDetector.detectBoard`` of OpenCV 4.14 and later returns ChArUco corners
in it; earlier versions return them half a pixel further on.  ``aruco_opencv.detect_board`` takes the
difference out, so ChArUco, Ccube and CIco agree with ChArUco2, Ccube2 and
each other under any version.  The analytic-render
check through whole targets and the rescale back-mapping is
``tests/test_pixel_centre_frame.py``; the simulated-capture one is
``tests/test_simulation.py::test_every_detector_reports_where_opencv_projects``.

Caches and Phase 1 runs record the frame, so points saved in the earlier
pixel-corner frame are never mixed with these.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from pyCamSet.calibration_targets.markers import aruco_opencv
from pyCamSet.calibration_targets.markers.aruco2 import ARUCO2_AVAILABLE


def _board_image():
    """A generated ChArUco board whose inner corners fall exactly on pixel edges."""
    squares_x, squares_y, square_px, margin = 8, 6, 100, 50
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), 1.0, 0.7, dictionary)
    image = board.generateImage(
        (squares_x * square_px + 2 * margin, squares_y * square_px + 2 * margin),
        marginSize=margin)
    image = cv2.GaussianBlur(image, (0, 0), 1.0)

    def pixel_centre_truth(ids):
        """Where corners ``ids`` are in the pixel-centre frame: on pixel edges, so at n - 0.5."""
        column = ids % (squares_x - 1) + 1
        row = ids // (squares_x - 1) + 1
        return np.stack([margin + column * square_px, margin + row * square_px], axis=1) - 0.5

    return board, dictionary, image, pixel_centre_truth


def test_the_frame_is_pixel_centre():
    assert aruco_opencv.PIXEL_CONVENTION == "pixel-centre"


def test_detect_board_reports_charuco_corners_in_the_pixel_centre_frame():
    board, _dictionary, image, truth = _board_image()
    corners, ids, _, _ = aruco_opencv.detect_board(cv2.aruco.CharucoDetector(board), image)
    ids = np.asarray(ids).reshape(-1)
    assert len(ids) == 35
    error = np.asarray(corners).reshape(-1, 2) - truth(ids)
    assert np.abs(error).max() < 0.02, (cv2.__version__, error.mean(axis=0))


def test_detect_board_moves_corners_found_from_given_markers_the_same_way():
    """The aruco2 marker backend hands its markers to detectBoard."""
    board, dictionary, image, truth = _board_image()
    markers, marker_ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
    corners, ids, _, _ = aruco_opencv.detect_board(
        cv2.aruco.CharucoDetector(board), image, markerCorners=markers, markerIds=marker_ids)
    ids = np.asarray(ids).reshape(-1)
    error = np.asarray(corners).reshape(-1, 2) - truth(ids)
    assert np.abs(error).max() < 0.02, (cv2.__version__, error.mean(axis=0))


def test_marker_quads_are_pixel_centre_and_left_as_opencv_gives_them():
    """No OpenCV version moved them, so nothing moves them; their corners sit on pixel edges too."""
    board, dictionary, image, _ = _board_image()
    direct, direct_ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
    _, _, markers, marker_ids = aruco_opencv.detect_board(cv2.aruco.CharucoDetector(board), image)
    order = np.argsort(np.asarray(marker_ids).reshape(-1))
    direct_order = np.argsort(np.asarray(direct_ids).reshape(-1))
    assert np.allclose(np.asarray(markers)[order], np.asarray(direct)[direct_order], atol=1e-4)
    # Every square, and so every marker, is centred on a pixel edge (100 px squares from a
    # 50 px margin), at a half-integer in the pixel-centre frame and an integer in the
    # pixel-corner one. OpenCV traces a quad through its outermost pixels' centres, half a
    # pixel inside the printed edge on every side, so its corners are no test of the frame,
    # but that inset is symmetric and leaves the quad's centre where the marker's is.
    centres = np.asarray(markers, dtype=np.float64).reshape(-1, 4, 2).mean(axis=1)
    assert np.abs(centres - (np.floor(centres) + 0.5)).max() < 0.05, (cv2.__version__, centres)


@pytest.mark.parametrize("version, offset", [
    ("4.11.0", 0.5), ("4.13.0", 0.5), ("4.14.0", 0.0), ("5.0.0", 0.0), ("5.1.0-dev", 0.0), ("6.0.0", 0.0),
])
def test_the_offset_follows_the_opencv_version(version, offset):
    assert aruco_opencv.opencv_charuco_pixel_offset(version) == offset
    native = np.array([[[10.5, 20.5]], [[123.75, 7.5]]], dtype=np.float32)  # as detectBoard returns them
    moved = aruco_opencv.charuco_corners_to_pixel_centre(native, version)
    if offset == 0.0:
        assert moved is native                       # untouched where OpenCV is already there
    else:
        assert moved.dtype == np.float32 and moved.shape == native.shape
        np.testing.assert_array_equal(moved, native - np.float32(offset))
    assert aruco_opencv.charuco_corners_to_pixel_centre(None, version) is None


# -- detection caches ----------------------------------------------------------

def _write_cache(tmp_path, spec):
    import pyCamSet.calibration.detection_cache as cache_module
    from pyCamSet.calibration_targets import TargetDetection
    from pyCamSet.calibration_targets.core.target_registry import build_target

    target = build_target(spec)
    path = tmp_path / "detected_datapoints.npz"
    cache_module.save_to_cache(
        TargetDetection(cam_names=["cam0"], data=np.array([[0, 0, 0, 1.0, 2.0]])),
        [(8, 12)], path, target, ["cam0"], None)
    return target, path


def _recorded_identity(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["identity"]))


def _rewrite_identity(path, identity):
    """The same cache with *identity* in place of its own, as an older pyCamSet would have written it."""
    with np.load(path, allow_pickle=False) as archive:
        members = {name: archive[name] for name in archive.files}
    members["identity"] = np.array(json.dumps(identity, sort_keys=True))
    with open(path, "wb") as fh:
        np.savez(fh, **members)


#: Every target family whose points moved into the pixel-centre frame.
MOVED = [
    {"type": "ChArUco", "marker_backend": "aruco1"},
    {"type": "Ccube", "n_points": 4, "length": 20.0},
]
MOVED_IDS = ["ChArUco", "Ccube"]


@pytest.mark.parametrize("spec", MOVED, ids=MOVED_IDS)
def test_a_cache_records_the_pixel_centre_frame_and_matches(tmp_path, spec):
    from pyCamSet.calibration.detection_cache import cache_matches
    target, path = _write_cache(tmp_path, spec)
    assert _recorded_identity(path)["pixel_convention"] == "pixel-centre"
    assert cache_matches(path, target, ["cam0"], None) is True


@pytest.mark.parametrize("spec", MOVED, ids=MOVED_IDS)
def test_a_cache_from_the_pixel_corner_frame_is_a_miss(tmp_path, spec):
    """Written before the frame was fixed: no pixel_convention field (ChArUco-family ones
    carried charuco_corner_convention instead), its points half a pixel further on."""
    from pyCamSet.calibration.detection_cache import cache_matches
    target, path = _write_cache(tmp_path, spec)
    identity = _recorded_identity(path)
    del identity["pixel_convention"]
    if spec["type"] in aruco_opencv.CHARUCO_DETECTOR_TARGETS:
        identity["charuco_corner_convention"] = "pixel-corner-v1"
    _rewrite_identity(path, identity)
    assert cache_matches(path, target, ["cam0"], None) is False


def test_a_puzzleboard_cache_does_not_depend_on_the_frame(tmp_path):
    """PuzzleBoard points were always pixel-centre, so its caches stay valid."""
    pytest.importorskip("puzzle_board")
    from pyCamSet.calibration.detection_cache import _target_identity
    from pyCamSet.calibration_targets.core.target_registry import build_target
    assert "pixel_convention" not in _target_identity(build_target({"type": "PuzzleBoard"}), ["a"], None)


@pytest.mark.skipif(not ARUCO2_AVAILABLE, reason="aruco2 is not installed")
@pytest.mark.parametrize("spec", [{"type": "ChArUco2"}, {"type": "ChArUco", "marker_backend": "aruco2"}],
                         ids=["ChArUco2", "ChArUco-aruco2"])
def test_aruco2_caches_record_the_frame_too(tmp_path, spec):
    target, path = _write_cache(tmp_path, spec)
    assert _recorded_identity(path)["pixel_convention"] == "pixel-centre"


# -- Phase 1 runs and the education re-check ------------------------------------

def _photo(tmp_path, saved, unknown, corner_offset=0.5):
    from pyCamSet.calibration_targets.charuco import ChArUco
    from pyCamSet.workflow.education_detect_explain import Phase1Photo

    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0)
    image = board._render_board(px_per_mm=8.0).astype(np.uint8)
    path = tmp_path / "photo.png"
    cv2.imwrite(str(path), image)
    live = board.find_in_image(cv2.imread(str(path), cv2.IMREAD_COLOR))
    return Phase1Photo(
        photo_path=path, camera="cam", image_index=0, target=board, rescale_and_gamma=False,
        preprocessing_scale=1.0, preprocessing_gamma=0.5, upscale_factor=1,
        saved_detection=saved(live), cameras=("cam",), image_count=1, mapping="listing",
        verified=False, charuco_convention_unknown=unknown, pixel_corner_offset=corner_offset)


def test_an_old_run_in_the_pixel_corner_frame_is_compared_half_a_pixel_back(tmp_path):
    """A ChArUco run saved under OpenCV 4 before any frame was recorded is exactly half a
    pixel on; the education check says so instead of calling it a different result."""
    from pyCamSet.calibration_targets.core.target_detections import ImageDetection

    def old(live):
        return ImageDetection(live.keys, np.asarray(live.image_points, float) + 0.5)

    check = _photo(tmp_path, old, unknown=True).explain().consistency
    assert check.ok and "pixel-corner frame" in check.message
    # A run whose frame is known gets no such allowance.
    assert not _photo(tmp_path, old, unknown=False).explain().consistency.ok


def test_any_other_offset_is_still_a_difference(tmp_path):
    from pyCamSet.calibration_targets.core.target_detections import ImageDetection

    def moved(live):
        return ImageDetection(live.keys, np.asarray(live.image_points, float) + 0.3)

    assert not _photo(tmp_path, moved, unknown=True).explain().consistency.ok


def _phase1_run(tmp_path, spec, image):
    from pyCamSet.workflow import phase1
    from pyCamSet.workflow.workspace import WorkspaceManager, workspace_path_for

    folder = tmp_path / "images"
    for camera in ("cam0", "cam1"):
        (folder / camera).mkdir(parents=True)
        cv2.imwrite(str(folder / camera / "000.png"), image)
    params = {"target": spec, "f_loc": str(folder), "caching": False, "n_lim": None,
              "threads": 1, "upscale_factor": 1, "selected_cameras": []}
    return phase1.run(params, WorkspaceManager(workspace_path_for(folder)))


def _charuco_spec_and_image():
    from pyCamSet.calibration_targets.charuco import ChArUco
    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0)
    spec = {"type": "ChArUco", "num_squares_x": 5, "num_squares_y": 5, "square_size": 10.0}
    return spec, board._render_board(px_per_mm=6.0).astype(np.uint8)


def test_a_phase1_run_records_its_frame_and_opencv_version(tmp_path):
    """So a later re-detection knows whether it can match the run exactly."""
    spec, image = _charuco_spec_and_image()
    run = _phase1_run(tmp_path, spec, image)
    assert run["pixel_convention"] == "pixel-centre"
    assert "charuco_corner_convention" not in run
    assert run["opencv_version"] == cv2.__version__


def _rewrite_run_as_pixel_corner(run_dir, metadata_edit):
    """The run as the previous pyCamSet saved it: points half a pixel on, no pixel_convention."""
    from pyCamSet.utils.saving import load_pickle, save_pickle

    meta_path = run_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata.pop("pixel_convention", None)
    metadata_edit(metadata)
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    pickle_path = run_dir / "detected_datapoints.pickle"
    payload = load_pickle(pickle_path)
    detections = payload[0] if isinstance(payload, tuple) else payload
    data = detections.get_data()
    data[:, -2:] += 0.5                      # get_data() is the detection's own array
    save_pickle(payload, pickle_path)


def test_a_known_pixel_corner_run_is_moved_back_and_matches(tmp_path):
    """A ChArUco run saved with charuco_corner_convention 'pixel-corner-v1' (the frame the
    previous pyCamSet fixed every target to) is moved half a pixel back and matches."""
    from pyCamSet.workflow.education_detect_explain import load_phase1_photo

    spec, image = _charuco_spec_and_image()
    run = _phase1_run(tmp_path, spec, image)
    run_dir = _run_dir(tmp_path, run)
    _rewrite_run_as_pixel_corner(
        run_dir, lambda m: m.update({"charuco_corner_convention": "pixel-corner-v1"}))
    photo = load_phase1_photo(run_dir, "cam0", 0)
    assert not photo.charuco_convention_unknown
    check = photo.explain().consistency
    assert check.ok and "pixel-corner frame" in check.message, check.message


def _run_dir(tmp_path, run):
    from pathlib import Path
    return Path(run["artifacts"]["detected_datapoints_pickle"]).parent
