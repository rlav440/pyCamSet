'''Purpose: Lock the optional PuzzleBoard rescale-and-gamma detector contract.
Status: Active regression coverage for the pcube preprocessing path.
Future: Add bounded real-dataset evidence alongside this deterministic suite.
'''
from __future__ import annotations

import cv2
import numpy as np
import pytest

from pyCamSet.calibration_targets.core.abstract_target import AbstractTarget
from pyCamSet.calibration_targets.core.target_detections import ImageDetection
from pyCamSet.calibration_targets.markers.puzzleboard import (
    preprocess_puzzleboard_image,
)


def _frozen_preparation(image: np.ndarray, scale: float, gamma: float) -> np.ndarray:
    """The frozen diagnostic comparator, kept independent of production code."""
    if image.dtype == np.uint16:
        base = (image >> 8).astype(np.uint8)
    elif image.dtype == np.uint8:
        base = image
    else:
        raise AssertionError(f"unexpected test dtype: {image.dtype}")
    corrected = np.power(base.astype(np.float32) / 255.0, gamma) * 255.0
    corrected = corrected.astype(np.uint8)
    return cv2.resize(corrected, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA)


def test_uint16_default_preparation_matches_frozen_comparator_exactly():
    image = np.arange(16 * 24, dtype=np.uint16).reshape(16, 24) * 257

    actual = preprocess_puzzleboard_image(image, enabled=True)

    np.testing.assert_array_equal(
        actual, _frozen_preparation(image, scale=0.25, gamma=0.5))
    assert actual.dtype == np.uint8
    assert actual.shape == (4, 6)


def test_uint8_is_not_promoted_before_gamma_and_area_resize():
    image = np.arange(16 * 24, dtype=np.uint8).reshape(16, 24)

    actual = preprocess_puzzleboard_image(image, enabled=True,
                                           scale=0.5, gamma=2.0)

    np.testing.assert_array_equal(
        actual, _frozen_preparation(image, scale=0.5, gamma=2.0))
    assert actual.dtype == np.uint8


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_disabled_preparation_is_a_byte_for_byte_noop(dtype):
    image = np.arange(16 * 24, dtype=dtype).reshape(16, 24)

    actual = preprocess_puzzleboard_image(image, enabled=False,
                                           scale=0.25, gamma=0.5)

    assert actual is image
    np.testing.assert_array_equal(actual, image)


def test_rgb_and_rgba_keep_channel_contract_and_alpha_is_not_gamma_corrected():
    rgb = np.zeros((8, 12, 3), dtype=np.uint16)
    rgb[..., 0] = 0x4000
    rgba = np.concatenate([rgb, np.full((8, 12, 1), 0x8000, dtype=np.uint16)], axis=2)

    rgb_out = preprocess_puzzleboard_image(rgb, enabled=True,
                                           scale=0.5, gamma=0.5)
    rgba_out = preprocess_puzzleboard_image(rgba, enabled=True,
                                            scale=0.5, gamma=0.5)

    assert rgb_out.shape == (4, 6, 3)
    assert rgba_out.shape == (4, 6, 4)
    np.testing.assert_array_equal(rgba_out[..., 3], np.full((4, 6), 128, dtype=np.uint8))
    assert np.all(rgb_out[..., 0] == 127)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
@pytest.mark.parametrize("gamma", [0.5])
def test_preparation_rejects_non_positive_or_non_finite_scale(scale, gamma):
    with pytest.raises(ValueError, match="scale"):
        preprocess_puzzleboard_image(np.zeros((8, 8), dtype=np.uint16),
                                     enabled=True, scale=scale, gamma=gamma)


@pytest.mark.parametrize("gamma", [0.0, -1.0, np.nan, np.inf])
def test_preparation_rejects_non_positive_or_non_finite_gamma(gamma):
    with pytest.raises(ValueError, match="gamma"):
        preprocess_puzzleboard_image(np.zeros((8, 8), dtype=np.uint16),
                                     enabled=True, scale=0.25, gamma=gamma)


class _CountingTarget(AbstractTarget):
    """Small real-folder target double for the production entry point."""

    def find_in_image(self, image, draw=False, camera=None, wait_len=1):
        self.calls += 1
        self.seen_shapes.append(tuple(image.shape))
        return ImageDetection(
            keys=np.asarray([7], dtype=np.int64),
            image_points=np.asarray([[2.0, 3.0]], dtype=np.float64),
        )


def test_find_in_imfolder_runs_one_detector_pass_and_inverse_maps_points(tmp_path):
    image_dir = tmp_path / "frame"
    image_dir.mkdir()
    image_path = image_dir / "image.tiff"
    source = np.full((16, 24), 0x8000, dtype=np.uint16)
    assert cv2.imwrite(str(image_path), source)

    target = object.__new__(_CountingTarget)
    target.calls = 0
    target.seen_shapes = []
    target.input_args = {}

    detections = target.find_in_imfolder(
        image_dir, cam_names=["frame"], threads=1,
        rescale_and_gamma=True, preprocessing_scale=0.25,
        preprocessing_gamma=0.5,
    )

    assert target.calls == 1
    assert target.seen_shapes == [(4, 6)]
    data = detections.get_data()
    np.testing.assert_allclose(data[:, -2:], [[8.0, 12.0]])
    assert int(data[0, 2]) == 7


def test_production_detection_entry_threads_preparation_to_target_once(tmp_path):
    from pyCamSet.calibration.camera_calibrator import detect_datapoints_in_imfile

    image_dir = tmp_path / "frame"
    image_dir.mkdir()
    assert cv2.imwrite(
        str(image_dir / "image.tiff"),
        np.full((16, 24), 0x8000, dtype=np.uint16))
    target = object.__new__(_CountingTarget)
    target.calls = 0
    target.seen_shapes = []
    target.input_args = {}

    detections, cam_res = detect_datapoints_in_imfile(
        tmp_path, target, caching=False, threads=1, cam_names=["frame"],
        rescale_and_gamma=True, preprocessing_scale=0.25,
        preprocessing_gamma=0.5)

    assert target.calls == 1
    assert target.seen_shapes == [(4, 6)]
    assert cam_res == [(16, 24)]
    np.testing.assert_allclose(detections.get_data()[:, -2:], [[8.0, 12.0]])


def test_pcube_cache_identity_includes_preprocessing_settings(tmp_path):
    from pyCamSet.calibration.camera_calibrator import (
        cache_matches,
        cache_identity_path,
        write_cache_identity,
    )
    from pyCamSet.calibration_targets.core.target_registry import build_target

    cache = tmp_path / "detected_datapoints.pickle"
    cache.write_bytes(b"same detections")
    target = build_target({"type": "PuzzleBoardCube"})
    write_cache_identity(cache, target, ["cam0"], None,
                         preprocessing={"rescale_and_gamma": True,
                                        "scale": 0.25, "gamma": 0.5})

    assert cache_matches(
        cache, target, ["cam0"], None,
        preprocessing={"rescale_and_gamma": True, "scale": 0.25, "gamma": 0.5})
    assert not cache_matches(
        cache, target, ["cam0"], None,
        preprocessing={"rescale_and_gamma": True, "scale": 0.5, "gamma": 0.5})
    assert json_identity(cache_identity_path(cache))["identity"]["preprocessing"]["scale"] == 0.25


def json_identity(path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))
