"""Verify bounded Phase 1 trace capture for face-based targets.
"""
from __future__ import annotations

import pickle

import cv2
import numpy as np

from pyCamSet.calibration_targets.ccube import Ccube


def _render_ccube(target, face_index: int = 0) -> np.ndarray:
    """Use the target's actual printable face texture, not a mock detector input."""
    return cv2.cvtColor(target.textures[face_index].astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _render_cico(target, face_index: int = 0) -> np.ndarray:
    """Use CIco's actual printable face raster, not a polygon-only sketch."""
    return cv2.cvtColor(target._face_texture(face_index, 1600), cv2.COLOR_RGB2GRAY)


def _render_cico2(target, face_index: int = 0) -> np.ndarray:
    """Use CIco2's actual printable face raster, not a synthetic polygon sketch."""
    return cv2.cvtColor(target._face_texture(face_index, 1600), cv2.COLOR_RGB2GRAY)


def _assert_face_trace_parity(target, image: np.ndarray) -> list[dict]:
    """Tracing may record IDs, but must not change detector results."""
    target.phase1_trace_enabled = False
    baseline = target.find_in_image(image)
    assert getattr(target, "_last_phase1_trace", None) is None

    target.phase1_trace_enabled = True
    traced = target.find_in_image(image)
    np.testing.assert_array_equal(traced.keys, baseline.keys)
    np.testing.assert_array_equal(traced.image_points, baseline.image_points)
    assert isinstance(target._last_phase1_trace, list)
    face_count = target.n_faces if hasattr(target, "n_faces") else len(target.boards)
    assert len(target._last_phase1_trace) <= face_count
    assert len(pickle.dumps(target._last_phase1_trace)) < 64 * 1024
    captured = target._last_phase1_trace
    for stage in target._last_phase1_trace:
        assert 0 <= stage["face_index"] < face_count
        assert set(stage).issubset({
            "name", "face_index", "marker_ids", "corner_ids",
            "unprinted_corner_ids",
        })
        assert all(isinstance(value, int)
                   for key in ("marker_ids", "corner_ids", "unprinted_corner_ids")
                   for value in stage.get(key, []))

    # A blank negative control clears the prior frame's trace rather than
    # carrying accepted IDs forward from the last detection.
    target.find_in_image(np.full_like(image, 255))
    assert all(not stage.get("marker_ids") and not stage.get("corner_ids")
               for stage in target._last_phase1_trace)
    return captured


def test_ccube_face_trace_is_opt_in_and_matches_real_rendered_face():
    target = Ccube(length=40.0, n_points=5, draw_res=(900, 900))
    face_stages = _assert_face_trace_parity(target, _render_ccube(target))
    assert face_stages
    assert all(stage["name"] == "opencv_face" for stage in face_stages)
    assert any(stage.get("marker_ids") for stage in face_stages)
    assert any(stage.get("corner_ids") for stage in face_stages)


def test_ccube_pool_returns_bounded_face_traces_without_cross_image_carryover(tmp_path):
    target = Ccube(length=40.0, n_points=5, draw_res=(900, 900))
    target.phase1_trace_enabled = True
    folder = tmp_path / "cam0"
    folder.mkdir()
    image = _render_ccube(target)
    assert cv2.imwrite(str(folder / "frame0.png"), image)
    assert cv2.imwrite(str(folder / "frame1.png"), np.full_like(image, 255))

    detections = target.find_in_imfolder(folder, cam_names=["cam0"], threads=2)

    assert detections is not None
    records = target.phase1_trace_records
    assert [(item["camera"], item["image_index"])
            for item in records] == [("cam0", 0), ("cam0", 1)]
    assert all(item["trace_status"] == "captured" for item in records)
    assert records[0]["stages"]
    assert all(not stage.get("marker_ids") and not stage.get("corner_ids")
               for stage in records[1]["stages"])
    assert all(len(item["stages"]) <= len(target.boards) for item in records)
