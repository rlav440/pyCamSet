"""Verify opt-in Phase 1 PuzzleBoard stage recording.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from pyCamSet.calibration_targets import puzzleboard_cube
from pyCamSet.calibration_targets.puzzleboard_cube import PuzzleBoardCube


def _stage(trace, name):
    return next(stage for stage in trace if stage["name"] == name)


def test_cube_trace_records_window_drops_and_preserves_detection(monkeypatch):
    target = PuzzleBoardCube(n_points=16, length=100.0)
    (origin_x, origin_y) = target.face_origins[0]
    valid_id = [origin_y, origin_x]
    outside_id = [10_000, 10_000]
    returned_ids = np.asarray([valid_id, outside_id], dtype=np.int64)
    returned_coords = np.asarray([[20.0, 30.0], [40.0, 50.0]])
    monkeypatch.setattr(
        puzzleboard_cube, "detect_puzzleboard_image",
        lambda _image, min_width: (returned_ids, returned_coords),
    )

    target.phase1_trace_enabled = False
    untraced = target.find_in_image(np.zeros((80, 80), dtype=np.uint8))
    assert getattr(target, "_last_phase1_trace", None) is None
    target.phase1_trace_enabled = True
    traced = target.find_in_image(np.zeros((80, 80), dtype=np.uint8))

    np.testing.assert_array_equal(traced.keys, untraced.keys)
    np.testing.assert_array_equal(traced.image_points, untraced.image_points)
    stages = target._last_phase1_trace
    assert _stage(stages, "puzzleboard_decoded")["corner_ids"]["ids"] == returned_ids.tolist()
    assignment = _stage(stages, "puzzleboard_window_assignment")
    assert assignment["dropped_window_ids"]["ids"] == [outside_id]
    assert assignment["assigned_keys"]["ids"] == traced.keys.tolist()


def test_puzzleboard_trace_ids_are_bounded():
    from pyCamSet.calibration_targets.markers.puzzleboard import _trace_id_values

    result = _trace_id_values([[index, index + 1] for index in range(5000)])
    assert result["total"] == 5000
    assert result["truncated"] is True
    assert len(result["ids"]) == 4096
    assert result["ids"][-1] == [4095, 4096]


def _skip_without_detector():
    try:
        import puzzle_board  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError) as error:
        pytest.skip(f"optional PuzzleBoard detector unavailable: {error}")
    try:
        import cairosvg  # noqa: F401
    except (ImportError, OSError) as error:
        pytest.skip(f"native cairo unavailable for the real target render: {error}")


@pytest.mark.parametrize("threads", [1, 2])
def test_real_cube_printed_face_trace_preserves_slots(tmp_path, threads):
    _skip_without_detector()
    import cairosvg
    target = PuzzleBoardCube(n_points=16, length=100.0)
    target.phase1_trace_enabled = True
    png = cairosvg.svg2png(bytestring=target._face_svg(0).encode("utf-8"),
                           output_width=800, output_height=800)
    image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    folder = tmp_path / "cam0"
    folder.mkdir()
    for frame in range(2):
        assert cv2.imwrite(str(folder / f"frame{frame}.png"), image)
    detections = target.find_in_imfolder(folder, cam_names=["cam0"], threads=threads)
    assert detections.max_ims == 2
    assert [(r["camera"], r["image_index"]) for r in target.phase1_trace_records] == [
        ("cam0", 0), ("cam0", 1)]
    for record in target.phase1_trace_records:
        assert record["trace_status"] == "captured"
        assigned = _stage(record["stages"], "puzzleboard_window_assignment")["assigned_keys"]
        assert assigned["total"] > 0
        assert assigned["truncated"] is False
