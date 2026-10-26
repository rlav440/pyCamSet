'''Lock "Rescale and apply gamma" for every target type: per-target defaults, explicit
         values winning, the legacy PuzzleBoard-only gate, the 16-bit loader option and the
         detection-cache identity, with a real ChArUco target end to end.
'''
from __future__ import annotations

import cv2
import numpy as np
import pytest

from pyCamSet.calibration_targets.core.preparation import (
    ANY_TARGET_KEY,
    CHARUCO,
    MIN_MAX,
    OTHER,
    PREPARATION_DEFAULTS,
    PUZZLEBOARD,
    UPPER_BYTE,
    cache_identity,
    family_of_target_type,
    resolve_preparation,
    target_family,
)
from pyCamSet.calibration_targets.markers.puzzleboard import (
    min_max_to_uint8,
    preprocess_puzzleboard_image,
)


def _stub(module: str):
    """A target stand-in: only its class's module decides the family."""
    return type("Stub", (), {"__module__": module})()


PUZZLE = _stub("pyCamSet.calibration_targets.puzzleboard_cube")
# A target of no family with measured evidence (e.g. a user's own target class).
OTHER_T = _stub("my_lab.targets")
CCUBE = _stub("pyCamSet.calibration_targets.ccube")
CHARUCO_T = _stub("pyCamSet.calibration_targets.charuco")

ON = {"rescale_and_gamma": True, ANY_TARGET_KEY: True}


@pytest.mark.parametrize(("type_name", "family"), [
    ("PuzzleBoard", PUZZLEBOARD), ("PuzzleBoardCube", PUZZLEBOARD), ("PuzzleBoardIco", PUZZLEBOARD),
    ("ChArUco", CHARUCO), ("ChArUco2", CHARUCO), ("Ccube", CHARUCO), ("Ccube2", CHARUCO),
    ("CIco", CHARUCO), ("CIco2", CHARUCO),
])
def test_every_registry_target_has_a_family(type_name, family):
    assert family_of_target_type(type_name) == family


def test_defaults_are_per_target_family():
    """PuzzleBoard keeps 0.25/0.5/upper byte; ChArUco takes the measured 1.0/0.5/min-max."""
    puzzle = resolve_preparation(ON, PUZZLE)
    assert (puzzle.scale, puzzle.gamma, puzzle.loader) == (0.25, 0.5, UPPER_BYTE)
    for target in (CCUBE, CHARUCO_T):
        prep = resolve_preparation(ON, target)
        assert (prep.scale, prep.gamma, prep.loader) == (1.0, 0.5, MIN_MAX)
        assert prep.scale >= 0.8  # measured: ChArUco detections collapse below this
    other = resolve_preparation(ON, OTHER_T)
    assert (other.scale, other.gamma, other.loader) == (1.0, 0.5, UPPER_BYTE)
    assert set(resolve_preparation(ON, CCUBE).defaulted) == {"scale", "gamma", "loader"}


def test_explicit_values_always_win():
    params = dict(ON, preprocessing_scale=0.5, preprocessing_gamma=0.7,
                  preprocessing_loader=UPPER_BYTE)
    for target in (PUZZLE, CCUBE, OTHER_T):
        prep = resolve_preparation(params, target)
        assert (prep.scale, prep.gamma, prep.loader) == (0.5, 0.7, UPPER_BYTE)
        assert prep.defaulted == ()
    # one explicit value, the rest defaulted
    prep = resolve_preparation(dict(ON, preprocessing_gamma=0.4), CCUBE)
    assert (prep.scale, prep.gamma, prep.loader) == (1.0, 0.4, MIN_MAX)
    assert prep.defaulted == ("scale", "loader")


def test_off_is_off_for_every_target():
    for target in (PUZZLE, CCUBE, OTHER_T):
        assert resolve_preparation({"rescale_and_gamma": False, ANY_TARGET_KEY: True}, target) is None
        assert resolve_preparation({}, target) is None


def test_legacy_params_keep_the_puzzleboard_only_gate():
    """Params saved before the any-target change re-detect exactly as they first did."""
    legacy = {"rescale_and_gamma": True, "preprocessing_scale": 0.25, "preprocessing_gamma": 0.5}
    assert resolve_preparation(legacy, CCUBE) is None
    assert resolve_preparation(legacy, OTHER_T) is None
    prep = resolve_preparation(legacy, PUZZLE)
    assert (prep.scale, prep.gamma, prep.loader) == (0.25, 0.5, UPPER_BYTE)


def test_bad_loader_is_refused():
    with pytest.raises(ValueError, match="loader"):
        resolve_preparation(dict(ON, preprocessing_loader="lower_byte"), CCUBE)
    with pytest.raises(ValueError, match="loader"):
        preprocess_puzzleboard_image(np.zeros((4, 4), np.uint8), enabled=True, loader="x")


def test_cache_identity_is_unchanged_for_upper_byte_and_distinct_for_min_max():
    """Caches written before the loader option still match the same upper-byte pass."""
    assert cache_identity(False, 0.25, 0.5) is None
    assert cache_identity(True, 0.25, 0.5, UPPER_BYTE) == {
        "rescale_and_gamma": True, "scale": 0.25, "gamma": 0.5}
    min_max = cache_identity(True, 1.0, 0.5, MIN_MAX)
    assert min_max["loader"] == MIN_MAX
    assert min_max != cache_identity(True, 1.0, 0.5, UPPER_BYTE)


def test_twelve_bit_upper_byte_has_sixteen_levels_and_min_max_restores_them():
    """The 12-bit Ccube issue: uint16 data below 4096 has at most 16 upper-byte levels."""
    twelve_bit = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) % 4096)
    upper = preprocess_puzzleboard_image(twelve_bit, enabled=True, scale=1.0, gamma=1.0)
    stretched = preprocess_puzzleboard_image(twelve_bit, enabled=True, scale=1.0, gamma=1.0,
                                             loader=MIN_MAX)
    assert len(np.unique(upper)) <= 16 and upper.max() <= 15
    assert stretched.min() == 0 and stretched.max() == 255
    assert len(np.unique(stretched)) == 256
    # Same rule as a per-image min/max stretch (cv2.normalize NORM_MINMAX, then uint8).
    np.testing.assert_array_equal(
        stretched, cv2.normalize(twelve_bit, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))


def test_min_max_keeps_alpha_out_of_the_stretch():
    rgba = np.zeros((4, 6, 4), dtype=np.uint16)
    rgba[..., :3] = np.arange(4 * 6 * 3, dtype=np.uint16).reshape(4, 6, 3) * 10
    rgba[..., 3] = 0x8000
    out = min_max_to_uint8(rgba)
    assert out.dtype == np.uint8
    assert out[..., :3].min() == 0 and out[..., :3].max() == 255
    np.testing.assert_array_equal(out[..., 3], np.full((4, 6), 128, dtype=np.uint8))


def test_upper_byte_default_is_byte_identical_to_the_historical_preparation():
    image = np.arange(16 * 24, dtype=np.uint16).reshape(16, 24) * 257
    np.testing.assert_array_equal(
        preprocess_puzzleboard_image(image, enabled=True),
        preprocess_puzzleboard_image(image, enabled=True, loader=UPPER_BYTE))


def test_phase1_warns_when_a_charuco_target_is_downscaled():
    from pyCamSet.workflow.phase1 import preparation_warnings
    small = resolve_preparation(dict(ON, preprocessing_scale=0.25), CCUBE)
    assert preparation_warnings(CCUBE, small) and "0.25" in preparation_warnings(CCUBE, small)[0]
    assert preparation_warnings(CCUBE, resolve_preparation(ON, CCUBE)) == []
    assert preparation_warnings(PUZZLE, resolve_preparation(ON, PUZZLE)) == []
    assert preparation_warnings(CCUBE, None) == []


def test_puzzleboard_notice_is_unchanged():
    from pyCamSet.workflow.phase1 import PUZZLEBOARD_RESCALE_NOTICE, puzzleboard_rescale_notices
    assert puzzleboard_rescale_notices(PUZZLE, False) == [PUZZLEBOARD_RESCALE_NOTICE]
    assert puzzleboard_rescale_notices(CCUBE, False) == []


# ---------------------------------------------------------------------------------------------
# A real ChArUco target, end to end
# ---------------------------------------------------------------------------------------------

def _charuco_target():
    from pyCamSet.calibration_targets.charuco import ChArUco
    return ChArUco(num_squares_x=5, num_squares_y=4, square_size=10.0, marker_backend="aruco1")


def _dark_twelve_bit(target) -> np.ndarray:
    """The rendered board as a dark 12-bit camera stored in uint16 (values 200..1400)."""
    board = target._render_board(10.0)
    board = cv2.copyMakeBorder(board, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    return (200 + board.astype(np.float64) / 255.0 * 1200).astype(np.uint16)


def _clean(target) -> np.ndarray:
    board = target._render_board(10.0)
    return cv2.copyMakeBorder(board, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)


def test_a_real_charuco_target_is_prepared_with_its_defaults():
    from pyCamSet.calibration_targets.core.abstract_target import detect_in_image

    target = _charuco_target()
    assert target_family(target) == CHARUCO
    prep = resolve_preparation(ON, target)
    reference = target.find_in_image(cv2.cvtColor(_clean(target), cv2.COLOR_GRAY2BGR))
    assert reference.has_data
    events: list = []
    detection = detect_in_image(target, _dark_twelve_bit(target), explain=events,
                                **prep.detect_kwargs())
    assert detection.has_data
    assert len(detection.keys) == len(reference.keys)
    prepared = next(e for e in events if e["event"] == "prepared_image")
    assert prepared["preparation_scale"] == 1.0 and prepared["loader"] == MIN_MAX
    assert prepared["image"].shape == _clean(target).shape  # scale 1.0: native frame
    order = np.lexsort(np.asarray(reference.keys).T[::-1])
    got = np.lexsort(np.asarray(detection.keys).T[::-1])
    np.testing.assert_allclose(np.asarray(detection.image_points)[got],
                               np.asarray(reference.image_points)[order], atol=0.5)


def test_charuco_detection_cache_records_the_loader(tmp_path):
    """Phase 1's production entry with a ChArUco target: the min/max pass gets its own slot."""
    from pyCamSet.calibration.camera_calibrator import detect_datapoints_in_imfile
    from pyCamSet.calibration.detection_cache import cache_matches

    target = _charuco_target()
    cam = tmp_path / "cam0"
    cam.mkdir()
    assert cv2.imwrite(str(cam / "a.tiff"), _dark_twelve_bit(target))
    prep = resolve_preparation(ON, target)
    detections, cam_res = detect_datapoints_in_imfile(
        tmp_path, target, caching=True, threads=1, cam_names=["cam0"], **prep.detect_kwargs())
    assert detections.get_data() is not None and len(detections.get_data()) > 0
    assert cam_res == [_clean(target).shape[:2]]
    cache = next(tmp_path.glob("*.npz"))
    assert cache_matches(cache, target, ["cam0"], None,
                         preprocessing=cache_identity(True, 1.0, 0.5, MIN_MAX))
    assert not cache_matches(cache, target, ["cam0"], None,
                             preprocessing=cache_identity(True, 1.0, 0.5, UPPER_BYTE))
    assert not cache_matches(cache, target, ["cam0"], None)


def test_explainer_shows_the_preparation_for_a_charuco_target():
    from pyCamSet.calibration_targets.core.abstract_target import detect_in_image
    from pyCamSet.workflow.education_detect_explain import explain_detection

    target = _charuco_target()
    photo = _dark_twelve_bit(target)
    saved = detect_in_image(target, photo, **resolve_preparation(ON, target).detect_kwargs())
    ex = explain_detection(photo, target, rescale_and_gamma=True, saved_detection=saved)
    assert "rescale_gamma" in ex.step_keys
    step = ex.step("rescale_gamma")
    assert step.counts["scale"] == 1.0 and step.counts["gamma"] == 0.5
    assert step.counts["16-bit loader"] == MIN_MAX
    assert "full size" in step.caption
    assert ex.consistency.ok, ex.consistency.message


def test_load_phase1_photo_resolves_the_run_preparation(tmp_path):
    """A new ChArUco run re-explains with its defaults; a legacy one stays unprepared."""
    import json

    from pyCamSet.workflow.education_detect_explain import load_phase1_photo

    spec = {"type": "ChArUco", "num_squares_x": 5, "num_squares_y": 4, "square_size": 10.0,
            "marker_backend": "aruco1"}
    for params, expected in (
            (dict(ON, target=spec), (True, 1.0, 0.5, MIN_MAX)),
            ({"rescale_and_gamma": True, "target": spec}, (False, None, None, None))):
        run = tmp_path / ("new" if expected[0] else "old")
        run.mkdir()
        (run / "metadata.json").write_text(json.dumps({"params": params}), encoding="utf-8")
        photo = load_phase1_photo(run, "cam0", 0)
        assert photo.rescale_and_gamma is expected[0]
        if expected[0]:
            assert (photo.preprocessing_scale, photo.preprocessing_gamma,
                    photo.preprocessing_loader) == expected[1:]






@pytest.mark.gui
def test_phase1_form_writes_blank_values_as_target_defaults(monkeypatch):
    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget

    from pyCamSet.gui.phase_1_detection import Phase1Tab
    from pyCamSet.workflow.workspace import WorkspaceManager

    QApplication.instance() or QApplication([])
    tab = Phase1Tab(QTabWidget(), QCheckBox(), QCheckBox(), WorkspaceManager(None))
    monkeypatch.setattr(tab, "get_selected_cameras", lambda: ["cam0"])
    monkeypatch.setattr("pyCamSet.gui.phase_1_detection.require_image_folder", lambda text: "x")
    tab._target_form.set_target_type("ChArUco")
    tab._rescale_gamma_cb.setChecked(True)
    assert tab._preprocessing_scale_edit.text() == ""
    params = tab._read_params()
    assert params[ANY_TARGET_KEY] is True
    assert params["preprocessing_scale"] is None and params["preprocessing_gamma"] is None
    assert params["preprocessing_loader"] is None
    tab._preprocessing_scale_edit.setText("0.9")
    tab._preprocessing_loader_combo.setCurrentText("Upper byte")
    params = tab._read_params()
    assert params["preprocessing_scale"] == 0.9 and params["preprocessing_loader"] == UPPER_BYTE
    assert PREPARATION_DEFAULTS[CHARUCO].scale == 1.0
