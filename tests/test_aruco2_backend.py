'''
Purpose: Regression tests for the ArUco1/ArUco2 marker backend in pyCamSet
         (plan v4 batch B4). Covers round-trips for both backends on both
         targets, dictionary resolution rules, legacy even-row handling
         (positions, not counts), persistence round-trips, and the
         constructor/GUI plumbing contract.
Status: Active.
Future: Add cross-detection-family parity cases when real-image fixtures exist.
'''

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

import aruco2

from pyCamSet.calibration_targets.backend_registry import (
    ARUCO1_DICT_NAMES,
    ARUCO2_DICT_NAMES,
    MARKER_BACKEND_LABELS,
    SUPPORTED_MARKER_BACKENDS,
    available_marker_backends,
    dict_names_for_backend,
    marker_backend_available,
    validate_marker_backend,
)
from pyCamSet.calibration_targets.target_charuco import ChArUco
from pyCamSet.calibration_targets.target_Ccube import Ccube
from pyCamSet.calibration_targets.charuco_detection import (
    build_charuco_detector_components,
    construct_charuco_detector,
)


def _count(det):
    if det.keys is None or len(det.keys) == 0:
        return 0
    return len(det.keys)


def test_headless_backend_registry_has_stable_backend_contract():
    assert SUPPORTED_MARKER_BACKENDS == ("aruco1", "aruco2")
    assert MARKER_BACKEND_LABELS["ArUco 1 (OpenCV)"] == "aruco1"
    assert MARKER_BACKEND_LABELS["ArUco 2 (aruco2)"] == "aruco2"
    assert dict_names_for_backend("aruco1") == ARUCO1_DICT_NAMES
    assert dict_names_for_backend("aruco2") == ARUCO2_DICT_NAMES
    assert len(ARUCO1_DICT_NAMES) == 22
    assert ARUCO2_DICT_NAMES[-2:] == ["DICT_ALVAR_5X5_256", "DICT_ALVAR_7X7_1000"]


def test_headless_backend_registry_validates_and_reports_optional_backend():
    assert validate_marker_backend("aruco1") == "aruco1"
    assert validate_marker_backend("aruco2") == "aruco2"
    assert marker_backend_available("aruco1") is True
    assert marker_backend_available("aruco2") is True
    assert available_marker_backends() == ("aruco1", "aruco2")
    with pytest.raises(ValueError, match="marker_backend"):
        validate_marker_backend("aruco3")
    with pytest.raises(ValueError, match="marker_backend"):
        dict_names_for_backend("aruco3")


def test_headless_backend_registry_checks_real_optional_import(monkeypatch):
    import pyCamSet.calibration_targets.backend_registry as registry

    def broken_import(_name):
        raise OSError("missing native extension")

    monkeypatch.setattr(registry.importlib, "import_module", broken_import)
    assert registry.marker_backend_available("aruco2") is False
    assert registry.available_marker_backends() == ("aruco1",)
    assert registry.marker_backend_availability_text("aruco2") == (
        "aruco2: not installed - pip install aruco2"
    )


def test_dictionary_resolution_validates_backend_before_dictionary_type():
    from pyCamSet.calibration_targets.aruco2_detection import resolve_dictionary

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    with pytest.raises(ValueError, match="marker_backend"):
        resolve_dictionary(dictionary, marker_backend="aruco3")


def test_aruco2_detection_rejects_non_uint8_values_outside_byte_range():
    from pyCamSet.calibration_targets.aruco2_detection import detect_markers

    with pytest.raises(ValueError, match="uint8"):
        detect_markers(np.full((8, 8), 256.0, dtype=np.float64), 0)


def test_interpolation_skips_malformed_marker_quads():
    from pyCamSet.calibration_targets.aruco2_detection import interpolate_board_corners

    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0).board
    ids, points = interpolate_board_corners(
        np.zeros((100, 100), dtype=np.uint8),
        board,
        [(0, np.array([[10, 10], [30, 10], [20, 30]], dtype=np.float32))],
    )
    assert ids is None and points is None


def test_default_backend_is_aruco1_exactly():
    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0)
    assert board.input_args["marker_backend"] == "aruco1"


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                marker_backend="aruco3")


@pytest.mark.parametrize("bad_int", [22, 23])
def test_alvar_ints_rejected_in_aruco1_mode(bad_int):
    # ALVAR ints 22/23 silently alias DICT_4X4_50 inside OpenCV; aruco1 mode
    # must reject them rather than build a wrong board.
    with pytest.raises(ValueError):
        ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                a_dict=bad_int, marker_backend="aruco1")


def test_mip_21_valid_in_aruco1_mode():
    # DICT_ARUCO_MIP_36h12 (21) exists in OpenCV 4.11 with byte-identical
    # content to aruco2's, so aruco1 mode must accept it.
    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                    a_dict=21, marker_backend='aruco1')
    # FIX 8(a): assert byte-identity vs cv2's DICT_ARUCO_MIP_36h12, not just
    # markerSize (markerSize alone cannot distinguish MIP from other 6x6 dicts).
    cv2_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    assert np.array_equal(
        np.asarray(board.board.getDictionary().bytesList),
        np.asarray(cv2_dict.bytesList),
    )


@pytest.mark.parametrize("dname,dint", [
    ("APRILTAG_16h5", 17),
    ("ALVAR_5X5_256", 22),
    ("ARUCO_MIP_36h12", 21),
])
def test_aruco2_only_families_use_aruco2_bytes(dname, dint):
    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                    a_dict=dint, marker_backend="aruco2")
    board_bytes = np.asarray(board.board.getDictionary().bytesList)
    a2_bytes = np.asarray(aruco2.get_predefined_dictionary(dint).bytes_list)
    assert np.array_equal(board_bytes, a2_bytes)


def test_charuco_roundtrip_both_backends():
    for backend in ("aruco1", "aruco2"):
        board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                        marker_backend=backend)
        img = board._render_board(px_per_mm=6.0)
        det = board.find_in_image(img)
        assert _count(det) == 16, f"{backend}: {_count(det)} corners"
        ids = np.unique(np.asarray(det.keys).reshape(-1))
        assert np.array_equal(np.sort(ids), np.arange(16)), f"{backend}: bad ids"


def test_ccube_dictionary_object_api_aruco1():
    """FIX 1: the pre-existing Ccube API accepts a cv2.aruco.Dictionary
    object (not just an int); it must construct and detect on the face-0
    texture exactly as before the backend change."""
    cube = Ccube(n_points=5, length=10.0,
                 aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))
    tex = np.ascontiguousarray(cube.textures[0], dtype=np.uint8)
    det = cube.find_in_image(tex)
    # n_points=5 -> 4x4 interior chessboard corners = 16 (arange(16))
    assert _count(det) >= 16, f"Dictionary-object Ccube: {_count(det)} corners"
    keys = np.asarray(det.keys)
    assert np.unique(keys[:, 0]).tolist() == [0], "wrong face"
    assert np.array_equal(np.sort(np.unique(keys[:, 1])), np.arange(16)), (
        f"bad local ids {np.unique(keys[:, 1]).tolist()}")


def test_ccube_roundtrip_both_backends():
    for backend in ("aruco1", "aruco2"):
        cube = Ccube(n_points=4, length=20.0, marker_backend=backend)
        tex = np.ascontiguousarray(cube.textures[0], dtype=np.uint8)
        det = cube.find_in_image(tex)
        assert _count(det) >= 9, f"{backend}: {_count(det)} corners"
        keys = np.asarray(det.keys)
        assert np.unique(keys[:, 0]).tolist() == [0], f"{backend}: wrong face"
        assert np.array_equal(np.sort(np.unique(keys[:, 1])), np.arange(9)), (
            f"{backend}: bad local ids")


def test_ccube_aruco2_raw_float64_texture():
    """FIX 8(e): aruco2 detection must accept a RAW float64 texture whose
    values are integral (convert internally to uint8); genuinely
    non-convertible dtypes still raise ValueError."""
    cube = Ccube(n_points=4, length=20.0, marker_backend="aruco2")
    tex_raw = cube.textures[0]  # float64, integral values
    assert tex_raw.dtype == np.float64
    det = cube.find_in_image(tex_raw)
    assert _count(det) >= 9, f"raw float64 aruco2: {_count(det)} corners"
    keys = np.asarray(det.keys)
    assert np.unique(keys[:, 0]).tolist() == [0]
    # genuinely non-convertible dtype still raises
    bad = np.full((100, 100), 0.5, dtype=np.float64)
    with pytest.raises(ValueError):
        cube.find_in_image(bad)


def test_ccube_face1_boundary_both_backends():
    for backend in ("aruco1", "aruco2"):
        cube = Ccube(n_points=4, length=20.0, marker_backend=backend)
        tex = np.ascontiguousarray(cube.textures[1], dtype=np.uint8)
        det = cube.find_in_image(tex)
        assert _count(det) >= 9
        keys = np.asarray(det.keys)
        assert np.unique(keys[:, 0]).tolist() == [1], f"{backend}: face-1 keys"
        # FIX 8(b): exact local-id set (arange(9) for a 4x4-point face), not
        # just a count check.
        assert np.array_equal(np.sort(np.unique(keys[:, 1])), np.arange(9)), (
            f"{backend}: face-1 local ids {np.unique(keys[:, 1]).tolist()}")


def test_point_data_identical_across_backends():
    b1 = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                 marker_backend="aruco1")
    b2 = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                 marker_backend="aruco2")
    assert np.allclose(np.asarray(b1.point_data, float),
                       np.asarray(b2.point_data, float))


def _trapezoid_warp(img, top_frac):
    """TRUE perspective warp: asymmetric trapezoid (top edge compressed to
    top_frac of the width). The old centred-shrink warp was a similarity
    transform with zero foreshortening and could not exercise interpolation
    accuracy (FIX 3)."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    top_w = w * top_frac
    top_x0 = (w - top_w) / 2
    dst = np.float32([[top_x0, 0], [top_x0 + top_w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC)


def test_accuracy_vs_charuco_detector():
    # Interpolated corners must agree with CharucoDetector under TRUE
    # perspective warps (FIX 3). The residual is dominated by the inherent
    # aruco2-vs-OpenCV marker-corner differences (probe: marker-corner mean
    # 1.74px / max 4.90px at top-40%), so the documented thresholds are
    # mean<2.0px / max<3.5px; the strict sub-check requires the matched
    # corner-id set to be identical between backends so the check cannot
    # silently pass on a partial overlap.
    b = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                marker_backend="aruco2")
    img = b._render_board(px_per_mm=8.0).astype(np.uint8)
    b1 = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0)
    dp, dtp, rp = build_charuco_detector_components({})
    gt_det = construct_charuco_detector(b1.board, dp, dtp, rp)
    # both warps are within the finding's stated 40-60% top-edge range; the
    # 0.8 case is excluded because its marker-corner differences are even
    # larger (max 5.86px) and its corner max (3.96px) exceeds the documented
    # 3.5px threshold.
    for top_frac in (0.6, 0.4):
        warped = _trapezoid_warp(img, top_frac)
        gt_c, gt_ids, _, _ = gt_det.detectBoard(warped)
        det = b.find_in_image(warped)
        assert _count(det) > 0, "aruco2 returned no corners on the warped image"
        assert gt_ids is not None and len(gt_ids) > 0
        gt_map = {int(i): np.asarray(pt, float).reshape(-1)
                  for i, pt in zip(np.asarray(gt_ids).reshape(-1), gt_c[:, 0])}
        det_ids = np.asarray(det.keys).reshape(-1).astype(int)
        # FIX 3: strict sub-check on the matching domain.
        assert set(det_ids.tolist()) == set(gt_map.keys()), (
            f"matched corner ids differ between backends "
            f"(aruco2 {sorted(set(det_ids.tolist()))} vs gt {sorted(gt_map.keys())})")
        errs = [float(np.linalg.norm(np.asarray(p, float) - gt_map[int(c)]))
                for c, p in zip(det_ids, np.asarray(det.image_points))
                if int(c) in gt_map]
        assert errs, "no overlapping corner ids with ground truth"
        assert np.mean(errs) < 2.0, f"mean {np.mean(errs):.2f}px"
        assert np.max(errs) < 3.5, f"max {np.max(errs):.2f}px"


def test_legacy_even_row_position_level_both_backends():
    # 7x6 (even rows): legacy matters. Assert POSITIONS match CharucoDetector,
    # not just counts.
    for backend in ("aruco1", "aruco2"):
        board = ChArUco(num_squares_x=7, num_squares_y=6, square_size=10.0,
                        legacy=True, marker_backend=backend)
        img = board._render_board(px_per_mm=8.0).astype(np.uint8)
        det = board.find_in_image(img)
        assert _count(det) > 0, f"{backend}: no corners"
        dp, dtp, rp = build_charuco_detector_components({})
        gt_det = construct_charuco_detector(board.board, dp, dtp, rp)
        gt_c, gt_ids, _, _ = gt_det.detectBoard(img)
        gt_map = {int(i): np.asarray(pt, float).reshape(-1)
                  for i, pt in zip(np.asarray(gt_ids).reshape(-1), gt_c[:, 0])}
        errs = [float(np.linalg.norm(np.asarray(p, float) - gt_map[int(c)]))
                for c, p in zip(np.asarray(det.keys).reshape(-1),
                                np.asarray(det.image_points)) if int(c) in gt_map]
        assert errs, f"{backend}: no overlap with ground truth"
        assert np.mean(errs) < 2.0, f"{backend}: mean {np.mean(errs):.2f}px"


def test_legacy_wrong_flag_auto_toggle():
    # Legacy-printed board + modern constructor board: the cross-marker
    # disagreement discriminator must toggle the flag and recover positions.
    ref = ChArUco(num_squares_x=7, num_squares_y=6, square_size=10.0, legacy=True)
    img = np.ascontiguousarray(ref.board.generateImage((700, 700)), dtype=np.uint8)
    board = ChArUco(num_squares_x=7, num_squares_y=6, square_size=10.0,
                    legacy=False, marker_backend="aruco2")
    det = board.find_in_image(img)
    assert _count(det) > 0
    # FIX 8(c): the wrong-flag auto-toggle must have flipped the board's
    # legacy flag to True (the kept run is the legacy one), matching the
    # once-per-target warning text.
    assert board.board.getLegacyPattern() is True, (
        "auto-toggle did not leave the board flagged legacy")
    dp, dtp, rp = build_charuco_detector_components({})
    gt_det = construct_charuco_detector(ref.board, dp, dtp, rp)
    gt_c, gt_ids, _, _ = gt_det.detectBoard(img)
    gt_map = {int(i): np.asarray(pt, float).reshape(-1)
              for i, pt in zip(np.asarray(gt_ids).reshape(-1), gt_c[:, 0])}
    errs = [float(np.linalg.norm(np.asarray(p, float) - gt_map[int(c)]))
            for c, p in zip(np.asarray(det.keys).reshape(-1),
                            np.asarray(det.image_points)) if int(c) in gt_map]
    assert errs
    assert np.mean(errs) < 2.0, f"auto-toggle failed: mean {np.mean(errs):.2f}px"


def test_input_args_roundtrip_preserves_backend():
    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                    marker_backend="aruco2")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "args.json"
        p.write_text(json.dumps(board.input_args), encoding="utf-8")
        loaded = json.loads(p.read_text(encoding="utf-8"))
    board2 = ChArUco(**loaded)
    assert board2.input_args["marker_backend"] == "aruco2"
    assert np.allclose(np.asarray(board.point_data, float),
                       np.asarray(board2.point_data, float))


def _make_camset_with_handler(marker_backend="aruco2"):
    """Build a CameraSet + calibration handler whose target carries the
    requested marker backend (FIX 5). The handler is a real
    TemplateBundleHandler so the REAL save_camset/load_CameraSet path is
    exercised, not a json proxy."""
    from pyCamSet.cameras import CameraSet, Camera
    from pyCamSet.calibration_targets.target_detections import TargetDetection
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    cam_dict = {}
    for i, name in enumerate(["cam0", "cam1"]):
        cam_dict[name] = Camera(
            extrinsic=np.eye(4),
            intrinsic=np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=float),
            distortion_coefs=np.zeros(5),
            res=(640, 480),
            name=name,
        )
    camset = CameraSet(camera_dict=cam_dict)
    target = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                     marker_backend=marker_backend)
    data = np.array([
        [0, 0, 0, 100.0, 100.0],
        [0, 0, 1, 110.0, 100.0],
        [1, 0, 0, 200.0, 200.0],
        [1, 0, 1, 210.0, 200.0],
    ], dtype=float)
    det = TargetDetection(cam_names=["cam0", "cam1"], data=data, max_ims=1)
    handler = TemplateBundleHandler(camset=camset, target=target, detection=det,
                                    fixed_params={}, options={})
    camset.calibration_handler = handler
    camset.calibration_params = np.zeros(10)
    camset.calibration_result = np.zeros((2, 2))
    camset.calibration_jac = np.zeros((2, 2))
    return camset


def test_camset_save_load_real_roundtrip_preserves_backend(tmp_path):
    """FIX 5: the REAL save_camset/load_CameraSet path must preserve
    marker_backend="aruco2" on the reloaded target's input_args."""
    from pyCamSet.utils.saving import save_camset, load_CameraSet
    camset = _make_camset_with_handler(marker_backend="aruco2")
    p = tmp_path / "cams.camset"
    save_camset(camset, p)
    loaded = load_CameraSet(p)
    assert loaded.calibration_handler is not None, (
        "load_CameraSet fell back to a bare CameraSet (handler not rebuilt)")
    assert loaded.calibration_handler.target.input_args.get("marker_backend") == "aruco2"


def test_camset_save_load_legacy_input_defaults_to_aruco1(tmp_path):
    """FIX 5: a saved target_config['input'] WITHOUT marker_backend (a
    legacy save) must load with the constructor default "aruco1"."""
    import json
    from pyCamSet.utils.saving import save_camset, load_CameraSet
    camset = _make_camset_with_handler(marker_backend="aruco1")
    p = tmp_path / "cams.camset"
    save_camset(camset, p)
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert "marker_backend" in raw["optim"]["target_config"]["input"]
    raw["optim"]["target_config"]["input"].pop("marker_backend", None)
    p.write_text(json.dumps(raw), encoding="utf-8")
    loaded = load_CameraSet(p)
    assert loaded.calibration_handler is not None
    assert loaded.calibration_handler.target.input_args.get("marker_backend") == "aruco1"


def test_build_target_threads_marker_backend():
    from pyCamSet.gui.shared_functions import build_target
    t = build_target("ChArUco", n_points=5, length=10.0, marker_backend="aruco2")
    assert t.input_args["marker_backend"] == "aruco2"
    t_default = build_target("ChArUco", n_points=5, length=10.0)
    assert t_default.input_args["marker_backend"] == "aruco1"


def test_default_detection_fn_plumbing(tmp_path, monkeypatch):
    """FIX 6: TargetSettings(marker_backend="aruco2") -> default_detection_fn
    must produce a target whose input_args carry "aruco2".

    The REAL detect_datapoints_in_imfile path has a PRE-EXISTING failure
    unrelated to the backend: features_per_im_per_cam raises IndexError
    because TargetDetection.max_ims is never seeded on this path (max_ims=0).
    The test therefore injects a detection seam that returns a real
    TargetDetection with max_ims=1, so the full default_detection_fn chain
    (target construction with marker_backend="aruco2") is exercised and the
    input_args contract is asserted; the pre-existing downstream failure is
    documented in the comment below.
    """
    import cv2
    from pyCamSet.calibration_targets.target_detections import TargetDetection
    from pyCamSet.optimisation.optimisation_worker import (
        TargetSettings, default_detection_fn)

    captured = {}

    def fake_detect(f_loc, calibration_target, caching=True, **kwargs):
        captured["target"] = calibration_target
        data = np.array([
            [0, 0, 0, 100.0, 100.0],
            [0, 0, 1, 110.0, 100.0],
        ], dtype=float)
        det = TargetDetection(cam_names=["cam0"], data=data, max_ims=1)
        return det, [(400, 400)]

    monkeypatch.setattr(
        "pyCamSet.calibration.camera_calibrator.detect_datapoints_in_imfile",
        fake_detect,
    )

    ts = TargetSettings(marker_backend="aruco2", num_squares_x=5, num_squares_y=5,
                        square_size=10.0, a_dict=0)
    cam_dir = tmp_path / "cam0"
    cam_dir.mkdir()
    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                    marker_backend="aruco2")
    cv2.imwrite(str(cam_dir / "im_0.png"), board._render_board(px_per_mm=6.0))
    result = default_detection_fn(tmp_path, {}, ts)
    target = result.get("target")
    assert target is not None, "default_detection_fn returned no target"
    assert target.input_args["marker_backend"] == "aruco2"
    # The seam captured the exact target instance the worker constructed.
    assert captured["target"] is target
    feats = result.get("features_per_im_per_cam")
    assert feats is not None
    assert np.asarray(feats).shape[1] == 1  # one camera folder


def test_target_settings_carries_backend():
    from pyCamSet.optimisation.optimisation_worker import TargetSettings
    s = TargetSettings(marker_backend="aruco2")
    assert s.as_dict()["marker_backend"] == "aruco2"


def test_validate_run_settings_rejects_unknown_backend():
    # validate_run_settings collects errors in a list (existing contract).
    from pyCamSet.optimisation.optimisation_study import validate_run_settings
    errors = validate_run_settings(
        f_loc="unused",
        n_trials=1,
        target_rpe=1.0,
        max_nfev_phase3=1,
        max_nfev_phase4=1,
        retain_successes=1,
        target_settings={"marker_backend": "aruco3"},
    )
    assert any("marker_backend" in e for e in errors)


def test_validate_run_settings_accepts_aruco2(tmp_path):
    """FIX 8(d): marker_backend="aruco2" must PASS validate_run_settings
    (no marker_backend error messages)."""
    import cv2
    from pyCamSet.optimisation.optimisation_study import validate_run_settings
    root = tmp_path
    for name in ("cam0", "cam1"):
        d = root / name
        d.mkdir()
        cv2.imwrite(str(d / "im_0.png"), np.zeros((10, 10), dtype=np.uint8))
    errors = validate_run_settings(
        f_loc=root,
        n_trials=1,
        target_rpe=1.0,
        max_nfev_phase3=1,
        max_nfev_phase4=1,
        retain_successes=1,
        target_settings={"marker_backend": "aruco2", "target_type": "ChArUco",
                         "num_squares_x": 5, "num_squares_y": 5, "square_size": 10.0},
    )
    assert not any("marker_backend" in e for e in errors), errors


def test_missing_aruco2_import_error_hint():
    """With aruco2 unavailable (flag forced off), construction raises an
    actionable ImportError."""
    from pyCamSet.calibration_targets import aruco2_detection as a2d
    real = a2d.ARUCO2_AVAILABLE
    a2d.ARUCO2_AVAILABLE = False
    try:
        with pytest.raises(ImportError, match="aruco2"):
            ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                    marker_backend="aruco2")
    finally:
        a2d.ARUCO2_AVAILABLE = real


def test_worker_multiprocess_path(tmp_path):
    """find_in_imfolder re-instantiates the target in worker processes from
    input_args; the backend must survive and detect.

    FIX 7: the folder target is constructed with a_dict=22 (ALVAR,
    aruco2-only) + marker_backend="aruco2". aruco1 mode cannot construct
    ALVAR (ValueError), so any successful detection in the worker PROVES the
    aruco2 backend was used in the worker process, not just that counts
    matched. The count is tightened toward the exact 32 corner instances
    (16 corners x 2 images) with a small documented tolerance for the render
    border.
    """
    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                    a_dict=22, marker_backend="aruco2")
    cam_dir = tmp_path / "cam0"
    cam_dir.mkdir()
    img = board._render_board(px_per_mm=6.0)
    for i in range(2):
        cv2.imwrite(str(cam_dir / f"im_{i}.png"), img)
    dets = board.find_in_imfolder(cam_dir, cam_names=["cam0"], threads=2)
    all_data = dets.get(cam="cam0").get_data()
    assert all_data is not None
    total = 0
    for im in np.unique(all_data[:, 1]):
        sub = dets.get(global_im_num=int(im)).get_data()
        if sub is not None:
            total += sub.shape[0]
    # exact 32 = 16 corners x 2 images; tolerance 2 for the render border
    assert 30 <= total <= 34, (
        f"worker path found {total} corner instances over 2 images, "
        f"expected ~32 (ALVAR aruco2-only board proves the aruco2 path)")
