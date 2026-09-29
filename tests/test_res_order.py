"""``Camera.res`` loads as ``(width, height)`` whichever order a file holds it in.

Until 2026-09-29 the calibration stored ``image.shape[:2]``, ``(height, width)``,
while ``set_resolutions_from_file`` stored ``(width, height)``; COLMAP's
cameras.txt, written from ``res``, then had WIDTH and HEIGHT swapped.  The
repository's own fixtures hold both: ``calibration_charuco``'s initial cameras
are ``(height, width)``, the ccube self-calibration is ``(width, height)``.
(``calibration_ccube``'s initial cameras are not a fixture: the ccube
calibration test writes that file, in the current order.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from pyCamSet import CameraSet, load_CameraSet
from pyCamSet.cameras.telecentric_camera import TelecentricCamera
from pyCamSet.utils.saving import (
    RES_ORDER, RES_ORDER_KEY, camset_to_colmap, compress, export_cameras_txt,
    image_sizes_from_folder, save_camset,
)

from conftest import make_camera

DATA = Path(__file__).parent / "test_data"


def _colmap_rows(folder):
    return [line.split() for line in (folder / "cameras.txt").read_text(encoding="utf-8").splitlines()
            if not line.startswith("#")]


def _unmark(path, detections=None, cam_names=None):
    """Rewrite a saved camset as a file from before the marker, optionally with detections."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    del saved["cam_config"][RES_ORDER_KEY]
    if detections is not None:
        saved["optim"]["dtct_config"] = {"compressed_data": compress(np.asarray(detections, dtype=float)),
                                        "cam_names": cam_names, "max_ims": 1}
    path.write_text(json.dumps(saved), encoding="utf-8")


def _telecentric(res):
    return TelecentricCamera(
        intrinsic=np.array([[80.0, 0, res[0] / 2], [0, 80.0, res[1] / 2], [0, 0, 1.0]]),
        res=list(res), distortion_coefs=np.array([0.0]), telecentricity=0.01, name="cam")


# --------------------------------------------------------------------------
# The repository's legacy fixtures
# --------------------------------------------------------------------------


@pytest.mark.parametrize("folder,width,height", [("calibration_charuco", 1280, 1024)])
def test_a_legacy_height_width_file_loads_as_width_height(folder, width, height, caplog):
    """No marker, no detections: the pinhole principal points settle it."""
    with caplog.at_level(logging.INFO):
        cams = load_CameraSet(DATA / folder / "initial_cameras.camset")
    for cam in cams:
        assert tuple(int(v) for v in cam.res) == (width, height)
    assert "stored (height, width)" in caplog.text


@pytest.mark.parametrize("folder", ["calibration_charuco"])
def test_the_images_agree_with_the_order_chosen(folder):
    """The fixtures' own images are the ground truth the heuristics stand in for."""
    sizes = image_sizes_from_folder(DATA / folder)
    cams = load_CameraSet(DATA / folder / "initial_cameras.camset", image_sizes=sizes)
    for name in cams.get_names():
        assert tuple(int(v) for v in cams[name].res) == sizes[name]


def test_a_legacy_width_height_file_is_left_alone(caplog):
    """Detections reach x = 1902 on a 1920 x 1080 file: it cannot be (1080, 1920)."""
    with caplog.at_level(logging.INFO):
        cams = load_CameraSet(DATA / "calibration_ccube" / "self_calib_test.camset")
    for cam in cams:
        assert tuple(int(v) for v in cam.res) == (1920, 1080)
    assert "res is stored" not in caplog.text


def test_a_legacy_file_exports_width_then_height_to_colmap(tmp_path):
    cams = load_CameraSet(DATA / "calibration_charuco" / "initial_cameras.camset")
    camset_to_colmap(cams, tmp_path)
    for row in _colmap_rows(tmp_path):
        assert (row[2], row[3]) == ("1280", "1024")


# --------------------------------------------------------------------------
# The marker, and what overrides it
# --------------------------------------------------------------------------


def test_a_marked_file_is_read_as_stored(tmp_path, caplog):
    """An off-centre principal point must not move a marked file's res."""
    cam = make_camera("cam", res=(640, 480))
    cam.intrinsic = np.array([[800.0, 0.0, 250.0], [0.0, 750.0, 310.0], [0.0, 0.0, 1.0]])
    path = tmp_path / "new.camset"
    save_camset(CameraSet(camera_dict={"cam": cam}), path)

    assert json.loads(path.read_text(encoding="utf-8"))["cam_config"][RES_ORDER_KEY] == RES_ORDER
    with caplog.at_level(logging.INFO):
        back = load_CameraSet(path)
    assert tuple(back["cam"].res) == (640, 480)
    assert "res is stored" not in caplog.text and "settles" not in caplog.text


def test_a_legacy_file_resaved_is_marked_and_stays_put(tmp_path, caplog):
    path = tmp_path / "resaved.camset"
    save_camset(load_CameraSet(DATA / "calibration_charuco" / "initial_cameras.camset"), path)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        back = load_CameraSet(path)
    assert all(tuple(int(v) for v in cam.res) == (1280, 1024) for cam in back)
    assert "res is stored" not in caplog.text and "settles" not in caplog.text


def test_an_unsettled_legacy_file_warns_and_is_read_as_stored(tmp_path, caplog):
    """A telecentric principal point proves nothing, and there are no detections."""
    path = tmp_path / "tele.camset"
    save_camset(CameraSet(camera_dict={"cam": _telecentric((448, 375))}), path)
    _unmark(path)
    with caplog.at_level(logging.INFO):
        back = load_CameraSet(path)
    assert tuple(back["cam"].res) == (448, 375)
    assert any(r.levelno == logging.WARNING and "nothing in it settles" in r.getMessage()
               for r in caplog.records)


def test_an_unsettled_legacy_file_resaved_stays_unmarked(tmp_path):
    """Saving it again must not vouch for an order nothing established."""
    path = tmp_path / "tele.camset"
    save_camset(CameraSet(camera_dict={"cam": _telecentric((448, 375))}), path)
    _unmark(path)
    resaved = tmp_path / "resaved.camset"
    save_camset(load_CameraSet(path), resaved)
    saved = json.loads(resaved.read_text(encoding="utf-8"))
    assert RES_ORDER_KEY not in saved["cam_config"]
    # so the images can still settle it later
    assert tuple(load_CameraSet(resaved, image_sizes={"cam": (375, 448)})["cam"].res) == (375, 448)


def test_image_sizes_follow_the_exif_orientation_detection_sees(tmp_path):
    """cv2.imread, which detection uses, turns an image by its EXIF orientation."""
    Image = pytest.importorskip("PIL.Image")
    cam_folder = tmp_path / "cam"
    cam_folder.mkdir()
    exif = Image.Exif()
    exif[0x0112] = 6  # stored 400 wide, 300 high; shown turned a quarter
    Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8)).save(cam_folder / "im.jpg", exif=exif)
    assert image_sizes_from_folder(tmp_path) == {"cam": (300, 400)}


def test_image_sizes_settle_an_unsettled_file(tmp_path):
    path = tmp_path / "tele.camset"
    save_camset(CameraSet(camera_dict={"cam": _telecentric((448, 375))}), path)
    _unmark(path)
    back = load_CameraSet(path, image_sizes={"cam": (375, 448)})
    assert tuple(back["cam"].res) == (375, 448)


def test_detections_past_the_short_side_settle_a_legacy_file(tmp_path):
    """As on mouse1: 375 x 448 portrait images, res stored [448, 375], y up to 438."""
    path = tmp_path / "tele.camset"
    save_camset(CameraSet(camera_dict={"cam": _telecentric((448, 375))}), path)
    # | cam | image | key | x | y |
    _unmark(path, detections=[[0, 0, 0, 12.0, 20.0], [0, 0, 1, 360.0, 438.0]], cam_names=["cam"])
    assert tuple(load_CameraSet(path)["cam"].res) == (375, 448)


def test_detections_correct_a_marked_file_that_carried_the_error_forward(tmp_path, caplog):
    """A transposed res loaded unsettled and saved again is marked, but still wrong."""
    path = tmp_path / "tele.camset"
    save_camset(CameraSet(camera_dict={"cam": _telecentric((448, 375))}), path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["optim"]["dtct_config"] = {"compressed_data": compress(np.array([[0, 0, 0, 360.0, 438.0]])),
                                    "cam_names": ["cam"], "max_ims": 1}
    path.write_text(json.dumps(saved), encoding="utf-8")
    with caplog.at_level(logging.INFO):
        back = load_CameraSet(path)
    assert tuple(back["cam"].res) == (375, 448)
    # a marked file should never need correcting, so this one says so loudly
    assert any(r.levelno == logging.WARNING and "where the target was detected" in r.getMessage()
               for r in caplog.records)


def test_an_unknown_marker_is_refused(tmp_path):
    path = tmp_path / "odd.camset"
    save_camset(CameraSet(camera_dict={"cam": make_camera("cam", res=(640, 480))}), path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["cam_config"][RES_ORDER_KEY] = "sideways"
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="sideways"):
        load_CameraSet(path)


# --------------------------------------------------------------------------
# COLMAP's pixel convention
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["FULL_OPENCV", "PINHOLE"])
def test_cameras_txt_moves_the_principal_point_half_a_pixel(tmp_path, model):
    """COLMAP's top-left pixel centre is (0.5, 0.5); pyCamSet's is (0, 0)."""
    cams = CameraSet(camera_dict={name: make_camera(name, res=(640, 480))
                                  for name in ("cam_a", "cam_b")})
    export_cameras_txt(cams, tmp_path, model=model)
    for name, row in zip(cams.get_names(), _colmap_rows(tmp_path)):
        intrinsic = cams[name].intrinsic
        assert (row[2], row[3]) == ("640", "480")
        assert float(row[4]) == intrinsic[0, 0] and float(row[5]) == intrinsic[1, 1]
        assert float(row[6]) - intrinsic[0, 2] == 0.5
        assert float(row[7]) - intrinsic[1, 2] == 0.5


def test_the_apde_cams_files_keep_opencvs_convention(tmp_path):
    """APD-MVS samples pixel p at texel p + 0.5, so its K is OpenCV's, unshifted."""
    cam = make_camera("cam", res=(640, 480))
    cam.to_MVSnet_txt(tmp_path / "cam.txt", (0.1, 0.8), 64)
    lines = (tmp_path / "cam.txt").read_text(encoding="utf-8").split("\n")
    intrinsic = np.array([[float(v) for v in lines[k].split()] for k in range(7, 10)])
    np.testing.assert_array_equal(intrinsic, cam.intrinsic)
