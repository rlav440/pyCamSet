'''Verify opt-in Phase 1 trace capture for ChArUco2.
'''
from __future__ import annotations

import pickle

import cv2
import numpy as np
import pytest

# Traces ChArUco2 detection, which needs the optional aruco2 package; skip
# where it is absent, as test_charuco2_target.py does.
pytest.importorskip("aruco2")

from pyCamSet import ChArUco2  # noqa: E402


_SUPPORTED_STAGES = {"aruco_markers", "grid_board_corners"}
_SUPPORTED_STAGE_KEYS = {
    "name", "marker_ids", "corner_ids", "truncated",
}


def _render_board(target: ChArUco2) -> np.ndarray:
    """Render the target's actual printable board for a detector input."""
    image, _ = target._render(dpi=600)
    return image


def _assert_supported_stages(stages: list[dict]) -> None:
    """Reject undocumented detector stages rather than silently accepting them."""
    assert [stage["name"] for stage in stages] == [
        "aruco_markers", "grid_board_corners",
    ]
    for stage in stages:
        assert set(stage).issubset(_SUPPORTED_STAGE_KEYS)
        id_key = "marker_ids" if stage["name"] == "aruco_markers" else "corner_ids"
        assert id_key in stage
        assert all(isinstance(value, int) for value in stage[id_key])
        if "truncated" in stage:
            assert stage["truncated"] is True


def test_rendered_board_trace_is_opt_in_and_semantically_identical():
    target = ChArUco2(num_squares_x=3, num_squares_y=3, square_size=12.0)
    image = _render_board(target)

    target.phase1_trace_enabled = False
    baseline = target.find_in_image(image)
    assert getattr(target, "_last_phase1_trace", None) is None

    target.phase1_trace_enabled = True
    traced = target.find_in_image(image)

    assert traced.has_data == baseline.has_data
    np.testing.assert_array_equal(traced.keys, baseline.keys)
    np.testing.assert_array_equal(traced.image_points, baseline.image_points)
    stages = target._last_phase1_trace
    _assert_supported_stages(stages)
    assert sorted(stages[0]["marker_ids"]) == list(range(9))
    assert stages[1]["corner_ids"] == traced.keys.tolist()


def test_blank_rendered_input_clears_trace_instead_of_reusing_prior_frame():
    target = ChArUco2(num_squares_x=3, num_squares_y=3, square_size=12.0)
    target.phase1_trace_enabled = True
    target.find_in_image(_render_board(target))
    assert target._last_phase1_trace

    detection = target.find_in_image(np.full((500, 500), 255, dtype=np.uint8))

    assert not detection.has_data
    assert target._last_phase1_trace == []


@pytest.mark.parametrize("threads", [1, 2])
def test_workflow_run_persists_charuco2_stages_only_when_opted_in(tmp_path, threads):
    from pyCamSet.workflow import phase1
    from pyCamSet.workflow.workspace import WorkspaceManager
    from pyCamSet.workflow.education_trace import read_trace
    images = tmp_path / "images"
    folder = images / "cam0"
    folder.mkdir(parents=True)
    target = ChArUco2(num_squares_x=3, num_squares_y=3, square_size=12.0)
    assert cv2.imwrite(str(folder / "frame0.png"), _render_board(target))
    assert cv2.imwrite(str(folder / "frame1.png"), _render_board(target))
    workspace = WorkspaceManager(tmp_path / "workspace")
    base = {"f_loc": str(images), "target": {"type": "ChArUco2",
            "num_squares_x": 3, "num_squares_y": 3, "square_size": 12.0},
            "threads": threads, "n_lim": None, "caching": False,
            "manifest_digest": True}
    off = phase1.run(dict(base, record_phase1_trace=False), workspace)
    on = phase1.run(dict(base, record_phase1_trace=True), workspace)
    assert off["error"] is None and on["error"] is None
    from pyCamSet.utils.saving import load_pickle
    from pyCamSet.workflow.detections import extract_detection
    det = lambda run: extract_detection(load_pickle(
        run["artifacts"]["detected_datapoints_pickle"]))
    before, after = det(off), det(on)
    np.testing.assert_array_equal(before.get_data(), after.get_data())
    assert "education_trace" not in off["artifacts"]
    trace = read_trace(workspace.run_dir("phase1", on["run_id"]),
                       on["run_id"], "phase1")
    assert trace["status"] == "available"
    assert len(trace["records"]) == 2
    assert all(item["payload"]["trace_status"] == "captured"
               for item in trace["records"])


@pytest.mark.parametrize("threads", [1, 2])
def test_phase1_recording_preserves_image_slots_and_stays_under_8_mb(tmp_path, threads):
    target = ChArUco2(num_squares_x=3, num_squares_y=3, square_size=12.0)
    target.phase1_trace_enabled = True
    camera_folder = tmp_path / "cam0"
    camera_folder.mkdir()
    image = _render_board(target)
    assert cv2.imwrite(str(camera_folder / "frame0.png"), image)
    assert cv2.imwrite(str(camera_folder / "frame1.png"), image)

    detections = target.find_in_imfolder(
        camera_folder, cam_names=["cam0"], threads=threads)

    assert detections.max_ims == 2
    records = target.phase1_trace_records
    assert [(item["camera"], item["image_index"]) for item in records] == [
        ("cam0", 0), ("cam0", 1),
    ]
    assert all(item["trace_status"] == "captured" for item in records)
    _assert_supported_stages(records[0]["stages"])
    _assert_supported_stages(records[1]["stages"])
    assert records[1]["stages"] == records[0]["stages"]
    assert len(pickle.dumps(records)) < 8 * 1024 * 1024
