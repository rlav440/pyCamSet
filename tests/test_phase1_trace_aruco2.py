"""Verify opt-in Phase 1 trace capture for aruco2 GridBoard.
"""
from __future__ import annotations

import numpy as np
import pytest

# Traces the aruco2 GridBoard detection, which needs the optional aruco2 package; skip
# where it is absent, as test_charuco2_target.py does.
pytest.importorskip("aruco2")

from pyCamSet import ChArUco2  # noqa: E402
from pyCamSet.calibration_targets.markers.aruco2_gridboard import (  # noqa: E402
    _PHASE1_TRACE_MAX_IDS,
    _append_phase1_trace,
    detect_grid_board_corners,
)


def test_real_gridboard_trace_is_opt_in_and_detection_identical():
    target = ChArUco2(num_squares_x=3, num_squares_y=3, square_size=12.0)
    image, _ = target._render(dpi=600)
    untraced_ids, untraced_points = detect_grid_board_corners(
        image, target.grid_size, target._aruco_dict_int, target.square_size)
    assert untraced_ids is not None and len(untraced_ids) > 0

    stages = []
    traced_ids, traced_points = detect_grid_board_corners(
        image, target.grid_size, target._aruco_dict_int, target.square_size,
        trace_stages=stages)
    np.testing.assert_array_equal(traced_ids, untraced_ids)
    np.testing.assert_array_equal(traced_points, untraced_points)
    assert [stage["name"] for stage in stages] == ["aruco_markers", "grid_board_corners"]
    assert sorted(stages[0]["marker_ids"]) == list(range(9))
    assert stages[1]["corner_ids"] == traced_ids.tolist()


def test_gridboard_trace_miss_clears_stale_stages():
    target = ChArUco2(num_squares_x=3, num_squares_y=3, square_size=12.0)
    stages = [{"name": "stale", "corner_ids": [1]}]
    ids, points = detect_grid_board_corners(
        np.full((400, 400), 255, dtype=np.uint8), target.grid_size,
        target._aruco_dict_int, target.square_size, trace_stages=stages)
    assert ids is None and points is None
    assert stages == []


def test_trace_id_cap_is_explicit_and_does_not_exceed_limit():
    stages = []
    _append_phase1_trace(stages, "grid_board_corners", np.arange(_PHASE1_TRACE_MAX_IDS + 7))
    assert len(stages[0]["corner_ids"]) == _PHASE1_TRACE_MAX_IDS
    assert stages[0]["corner_ids"] == list(range(_PHASE1_TRACE_MAX_IDS))
    assert stages[0]["truncated"] is True


def test_trace_capture_preserves_invalid_board_id_error():
    target = ChArUco2(num_squares_x=3, num_squares_y=3, square_size=12.0)
    image, _ = target._render(dpi=600)
    with pytest.raises(ValueError, match="needs that many ids"):
        detect_grid_board_corners(
            image, target.grid_size, target._aruco_dict_int, target.square_size,
            ids=[0, 1], trace_stages=[])
