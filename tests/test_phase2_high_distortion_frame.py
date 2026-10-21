"""Regress Phase 2 coordinate-frame and import-contract blockers.

"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyCamSet.workflow import phase2


@pytest.fixture(autouse=True)
def close_pyplot_figures_after_test():
    """Prevent visualisation tests leaving Qt-backed figures for later tests."""
    yield
    # Import on teardown so this fixture does not select a backend at collection.
    from matplotlib import pyplot as plt

    plt.close("all")

EQUIVALENCE_DECLARATIONS = (
    # The producer object and its persisted per-pass schema are intentional additions.
    "artifacts.education_used_detections",
    "artifacts.education_used_detections.schema_version",
    # The last high-distortion pass intentionally supplies the camera frame.
    "camera.res",
    # Phase 2 now records which producer supplied its calibrated detections.
    "artifacts.education_used_detections.source",
    "artifacts.education_used_detections.path",
    "artifacts.education_used_detections.fresh_pass_reasons",
    "artifacts.education_used_detections.cache_hit",
    "artifacts.education_used_detections.passes",
)


class _Detection:
    """Small detection stand-in with the camera names Phase 2 expects."""

    cam_names = ["camA", "camB"]


class _Cameras:
    """Capture the resolution handed to calibration and accept a save call."""

    def save(self, path):
        Path(path).write_text("synthetic camset", encoding="utf-8")


@pytest.mark.parametrize("upscale", [1, 2])
def test_high_distortion_calibrates_with_second_pass_resolution(
        tmp_path, monkeypatch, upscale):
    """The detections and resolution used by the last pass share one frame."""
    image_root = tmp_path / "images"
    for camera in ("camA", "camB"):
        folder = image_root / camera
        folder.mkdir(parents=True)
        (folder / "frame.png").write_bytes(b"synthetic")

    resolutions = [[(480 * upscale, 640 * upscale)] * 2,
                   [(480, 640)] * 2]
    detection_calls = []
    calibration_calls = []

    def detect(**kwargs):
        detection_calls.append(kwargs)
        return _Detection(), resolutions[len(detection_calls) - 1]

    def calibrate(_detections, cam_res, _target, **kwargs):
        calibration_calls.append(list(cam_res))
        return _Cameras()

    monkeypatch.setattr(phase2, "target_of_params", lambda _params: object())
    monkeypatch.setattr(phase2, "detect_datapoints_in_imfile", detect)
    monkeypatch.setattr(phase2, "calibrate", calibrate)
    monkeypatch.setattr(phase2, "diagnostics_of",
                        lambda *_args: ({}, {"per_camera": []}))

    run_dir = tmp_path / "phase2_runs" / "synthetic"
    run_dir.mkdir(parents=True)
    params = {
        "f_loc": str(image_root), "selected_cameras": [],
        "high_distortion": True, "n_lim": None, "caching": False,
        "fixed_params": None, "min_detections_per_board": 12,
        "upscale_factor": upscale,
    }
    used_record_holder = {}
    phase2._calibrate(params, run_dir, None, None, lambda _line: None,
                      used_record_holder=used_record_holder)

    assert len(detection_calls) == 2
    assert len(calibration_calls) == 2
    assert calibration_calls[-1] == resolutions[-1]
    assert calibration_calls[-1] == [(480, 640)] * 2
    assert used_record_holder["record"]["source"] == "fresh_pass_not_saved"
    assert used_record_holder["record"]["fresh_pass_reasons"] == [
        "pickle_missing", "high_distortion_second_pass"]
    assert used_record_holder["record"]["cache_hit"] is False
    assert used_record_holder["record"]["passes"] == [
        {"pass": "initial", "cache_hit": None},
        {"pass": "high_distortion_second_pass", "cache_hit": False},
    ]


def test_visualisation_and_cameras_are_first_importable_in_fresh_process(repo_root):
    """The E1 unit contract import must not reintroduce the cameras cycle."""
    for statement in ("import pyCamSet.utils.visualisation",
                      "import pyCamSet.cameras"):
        code = (
            f"{statement}; from pathlib import Path; import pyCamSet; "
            "root = Path.cwd().resolve(); "
            "assert root in Path(pyCamSet.__file__).resolve().parents; "
            "print('fresh-import-ok')"
        )
        finished = subprocess.run(
            [sys.executable, "-c", code], cwd=repo_root,
            capture_output=True, text=True, check=False)
        assert finished.returncode == 0, finished.stderr
        assert "fresh-import-ok" in finished.stdout


def test_camera_coverage_axes_use_native_width_and_height():
    """A non-square camera, res (width, height), plots against native image axes."""
    from pyCamSet.utils.visualisation import per_camera_coverage

    class CameraDetection:
        def get_data(self):
            return np.array([[0, 0, 0, 15.0, 25.0]])

    class DetectionSet:
        cam_names = ["camA"]

        def get_cam_list(self):
            return [CameraDetection()]

    class CameraSet(list):
        def get_n_cams(self):
            return len(self)

    camera = SimpleNamespace(
        intrinsic=np.array([[500.0, 0.0, 320.0],
                            [0.0, 500.0, 240.0],
                            [0.0, 0.0, 1.0]]),
        res=(640, 480),  # Camera.res is (width, height)
    )
    diagnostic = SimpleNamespace(
        cams=CameraSet([camera]), detection=DetectionSet(),
        euclidean_err=np.array([2.0]), residuals=np.array([[1.0, 1.0]]),
        e_lim=3.0,
    )
    figure = per_camera_coverage(diagnostic)
    axis = figure.axes[0]
    assert axis.get_xlim() == (0.0, 640.0)
    assert axis.get_ylim() == (480.0, 0.0)


def test_accuracy_precision_uses_puzzleboard_units_not_fixed_metre_scale():
    """One target-unit offset is 1 mm for PuzzleBoard, not 1000 mm."""
    from pyCamSet.utils.visualisation import _feature_consistency

    class PuzzleBoard:
        __module__ = "pyCamSet.calibration_targets.puzzleboard"

        original_points = {(0, 0): np.zeros(3)}

    class ChArUco:
        __module__ = "pyCamSet.calibration_targets.charuco"

        original_points = {(0, 0): np.zeros(3)}

    locations = {(0,): [np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]),
                        np.array([0.0, 1.0, 0])]}
    errors = {(0,): [0.1, 0.2, 0.3]}
    puzzle_accuracy, *_ = _feature_consistency(PuzzleBoard(), locations, errors)
    charuco_accuracy, *_ = _feature_consistency(ChArUco(), locations, errors)

    assert puzzle_accuracy[0] == pytest.approx(1.0)
    assert charuco_accuracy[0] == pytest.approx(1000.0)


def test_calibrate_requests_pose_details_only_when_requested_and_safe(monkeypatch):
    """Fixed extrinsics avoid the legacy bare-Camera tuple-index crash."""
    captured = []

    def run_initial_calibration(**kwargs):
        captured.append(kwargs["return_poses_and_costs"])
        if kwargs["return_poses_and_costs"]:
            return "cams", [[np.eye(4)]], [[0.25]]
        return "cams"

    monkeypatch.setattr(phase2, "BACKEND_OK", True)
    monkeypatch.setattr(phase2, "run_initial_calibration", run_initial_calibration)
    ordinary = phase2.calibrate([], [], object())
    details = phase2.calibrate([], [], object(), return_details=True)
    fixed = phase2.calibrate(
        [], [], object(), fixed_params={"camA": {"ext": np.eye(4)}},
        return_details=True)
    fixed_intrinsics = phase2.calibrate(
        [], [], object(),
        fixed_params={"camA": {"int": np.eye(3), "dst": np.zeros(5)}},
        return_details=True)

    assert ordinary == "cams"
    assert details[0] == "cams" and details[1] and details[2]
    assert fixed == ("cams", None, None)
    assert fixed_intrinsics == ("cams", None, None)
    assert captured == [False, True, False, False]


def test_pruned_high_distortion_rerun_fails_before_using_phase1_pickle(
        tmp_path, monkeypatch):
    """Never prune stale Phase 1 data when the authoritative pass was unsaved."""
    run_dir = tmp_path / "phase2_runs" / "pruned"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(phase2, "BACKEND_OK", True)
    monkeypatch.setattr(phase2, "target_of_params", lambda _params: object())
    params = {"high_distortion": True}

    with pytest.raises(RuntimeError, match=(
            "authoritative second-pass detections were not persisted")):
        phase2._calibrate(params, run_dir, tmp_path / "phase1.pickle",
                          object(), lambda _line: None)


def test_fresh_detection_cache_hit_is_recorded_from_side_channel(monkeypatch):
    """A cache hit on the actual fresh Phase 2 pass is retained in provenance."""
    detections_path = Path("phase1.pickle")
    monkeypatch.setattr(phase2, "path_exists", lambda _path: True)
    monkeypatch.setattr(phase2, "load_pickle", lambda _path: object())
    monkeypatch.setattr(phase2, "extract_detection_and_cam_res",
                        lambda _payload: (_Detection(), None))

    def detect(**kwargs):
        kwargs["side_channel"]["cache_hit"] = True
        return _Detection(), [(480, 640), (480, 640)]

    monkeypatch.setattr(phase2, "detect_datapoints_in_imfile", detect)
    pass_info = {}
    detections, cam_res = phase2._load_or_detect(
        {"caching": True, "n_lim": None}, object(), Path("images"),
        detections_path, set(), ["camA", "camB"], lambda _line: None,
        pass_info=pass_info)

    assert detections.cam_names == ["camA", "camB"]
    assert cam_res == [(480, 640), (480, 640)]
    assert pass_info == {
        "fresh_pass": True, "reasons": ["no_cam_res"], "cache_hit": True}


def test_zhang_fallback_warning_is_persisted_as_a_calibration_note(
        tmp_path, monkeypatch):
    """C2 can distinguish Zhang's image-centre fallback from an ordinary fit."""
    class Workspace:
        workspace_path = tmp_path

        def run_dir(self, _phase, run_id):
            return tmp_path / run_id

        def save_run(self, _phase, _run_id, metadata):
            self.metadata = metadata

    workspace = Workspace()
    monkeypatch.setattr(phase2, "make_run_id", lambda: "phase2-test")
    monkeypatch.setattr(phase2, "_detections_path", lambda *_args: None)

    def calibrate(_params, _run_dir, _path, _prune, log, **_kwargs):
        log("[WARNING] pyCamSet.calibration.zhang: fitting the focal lengths alone")
        return None, {}, None, {}

    monkeypatch.setattr(phase2, "_calibrate", calibrate)
    metadata = phase2.run({}, workspace)

    assert metadata["calibration_notes"] == [
        "Zhang closed-form principal point was outside the sensor; "
        "focals_at_principal held it at image centre before refinement."]
    assert workspace.metadata is metadata


def test_phase2_source_check_inherits_mismatch_and_fresh_pass_is_uncheckable(
        tmp_path):
    """A2's manifest edge survives Phase 2 only when its saved pickle was used."""
    from pyCamSet.calibration_targets.core.abstract_target import (
        enumerate_detection_images,
    )
    from pyCamSet.workflow.education_manifest import describe_enumeration
    from pyCamSet.workflow.education_trace import RunContext
    from pyCamSet.workflow.workspace import WorkspaceManager

    image_root = tmp_path / "images"
    photo_paths = {}
    for camera in ("camA", "camB"):
        folder = image_root / camera / "nested"
        folder.mkdir(parents=True)
        photo = folder / "frame.png"
        photo.write_bytes(camera.encode("utf-8"))
        photo_paths[camera] = tuple(enumerate_detection_images(folder))
    manifest = describe_enumeration(photo_paths, image_root, "nested")

    workspace_path = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace_path)
    phase1_dir = workspace_path / "phase1_runs" / "phase1"
    phase1_dir.mkdir(parents=True)
    pickle_path = phase1_dir / "detected_datapoints.pickle"
    pickle_path.write_bytes(b"saved phase1 detections")
    phase1_run = {
        "run_id": "phase1",
        "params": {"f_loc": str(image_root), "n_lim": None,
                   "subfolder_string": "nested",
                   "selected_cameras": ["camA", "camB"]},
        "diagnostics": {"education_manifest": manifest,
                        "education_cache": {"reused": False}},
        "artifacts": {"detected_datapoints_pickle": str(pickle_path)},
    }
    manager.save_run("phase1", "phase1", phase1_run)

    def phase2_context(run_id, used_record):
        phase2_run = {
            "run_id": run_id,
            "inputs": {"phase1_run_id": "phase1"},
            "params": {},
            "artifacts": {"education_used_detections": used_record},
        }
        manager.save_run("phase2", run_id, phase2_run)
        return RunContext.from_run(manager, {
            "run_id": run_id,
            "_run_dir": workspace_path / "phase2_runs" / run_id,
            **phase2_run,
        })

    source_backed = phase2_context("phase2-saved", {
        "schema_version": 1, "source": "phase1_pickle",
        "path": str(pickle_path), "fresh_pass_reasons": [], "cache_hit": None,
    })
    source_edge = source_backed.verify_input("phase1")
    assert source_edge.verification.value == "verified", source_edge
    (image_root / "camA" / "nested" / "frame.png").write_bytes(b"edited")
    assert source_backed.phase1_source_check().status == "mismatch"

    fresh = phase2_context("phase2-fresh", {
        "schema_version": 1, "source": "fresh_pass_not_saved",
        "path": None, "fresh_pass_reasons": ["pickle_missing"],
        "cache_hit": False,
    })
    assert fresh.phase1_source_check().status == "could not be checked"


def test_opted_in_phase2_writes_per_view_costs_and_poses(tmp_path, monkeypatch):
    """The sidecar records the final seed's per-view poses and RMS costs."""
    image_root = tmp_path / "images"
    for camera in ("camA", "camB"):
        folder = image_root / camera
        folder.mkdir(parents=True)
        (folder / "frame.png").write_bytes(b"synthetic")
    run_dir = tmp_path / "phase2_runs" / "run-opted-in"
    run_dir.mkdir(parents=True)
    trace_calls = []

    def detect(**kwargs):
        if kwargs.get("side_channel") is not None:
            kwargs["side_channel"]["cache_hit"] = False
        return _Detection(), [(480, 640)] * 2

    def calibrate(_detections, _cam_res, _target, **kwargs):
        assert kwargs["return_details"] is True
        poses = [[np.eye(4), np.eye(4)], [np.eye(4)]]
        costs = [[0.25, 0.5], [0.75]]
        return _Cameras(), poses, costs

    monkeypatch.setattr(phase2, "target_of_params", lambda _params: object())
    monkeypatch.setattr(phase2, "detect_datapoints_in_imfile", detect)
    monkeypatch.setattr(phase2, "calibrate", calibrate)
    monkeypatch.setattr(phase2, "diagnostics_of",
                        lambda *_args: ({}, {"per_camera": []}))
    monkeypatch.setattr(phase2, "write_trace",
                        lambda *args, **kwargs: trace_calls.append((args, kwargs)))
    params = {
        "f_loc": str(image_root), "selected_cameras": [],
        "high_distortion": False, "n_lim": None, "caching": False,
        "fixed_params": None, "record_phase2_per_view": True,
    }

    phase2._calibrate(params, run_dir, None, None, lambda _line: None)

    assert len(trace_calls) == 1
    args, kwargs = trace_calls[0]
    assert args[3] == "phase2_per_view"
    assert kwargs["enabled"] is True
    payload = args[4]
    assert payload["cost_semantics"].endswith("per-view RMS (px)")
    assert [len(camera["views"]) for camera in payload["cameras"]] == [2, 1]


def test_subset_and_missing_cam_res_reasons_survive_a_fresh_cache_hit(
        monkeypatch):
    """Both rejected Phase 1 causes remain visible when fallback hits cache."""
    fresh = _Detection()
    fresh.cam_names = ["camA"]
    monkeypatch.setattr(phase2, "path_exists", lambda _path: True)
    monkeypatch.setattr(phase2, "load_pickle", lambda _path: object())
    monkeypatch.setattr(phase2, "extract_detection_and_cam_res",
                        lambda _payload: (_Detection(), None))

    def detect(**kwargs):
        kwargs["side_channel"]["cache_hit"] = True
        return fresh, [(480, 640)]

    monkeypatch.setattr(phase2, "detect_datapoints_in_imfile", detect)
    pass_info = {}
    detections, cam_res = phase2._load_or_detect(
        {"caching": True, "n_lim": None}, object(), Path("images"),
        Path("phase1.pickle"), {"camA"}, ["camA"], lambda _line: None,
        pass_info=pass_info)

    assert detections is fresh
    assert cam_res == [(480, 640)]
    assert pass_info == {
        "fresh_pass": True,
        "reasons": ["camera_subset", "no_cam_res"],
        "cache_hit": True,
    }


def test_distortion_arrows_match_geometry_csv_and_pixel_units(
        tmp_path, monkeypatch):
    """Non-square vector signs, unit scale and CSV values share camera geometry."""
    pytest.importorskip("PySide6")
    from pyCamSet import Camera
    from pyCamSet.gui import phase_2_intrinsics as gui

    camera = Camera(
        extrinsic=np.eye(4),
        intrinsic=np.array([[200.0, 0.0, 320.0],
                            [0.0, 200.0, 240.0],
                            [0.0, 0.0, 1.0]]),
        res=(640, 480),  # Camera.res is (width, height)
        distortion_coefs=np.array([0.1, 0.0, 0.0, 0.0, 0.0]),
        name="camA",
    )

    class Layout:
        def __init__(self):
            self.widgets = []

        def count(self):
            return 0

        def addWidget(self, widget):
            self.widgets.append(widget)

    class Card:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    captured = {}
    monkeypatch.setattr(gui, "resolve_artifact", lambda *_args: tmp_path / "camset")
    monkeypatch.setattr(gui, "MatplotlibFigureCard",
                        lambda *args, **kwargs: captured.setdefault(
                            "card", Card(*args, **kwargs)))

    fake = SimpleNamespace(
        _distortion_layout=Layout(), _distortion_widget=None,
        _workspace_mgr=SimpleNamespace(workspace_path=tmp_path),
        _load_camset_cached=lambda *_args: ([camera], None, None),
    )
    monkeypatch.setattr(gui.Phase2DiagnosticsTab, "_clear_layout",
                        lambda *_args: None, raising=False)
    # The tab switches matplotlib to QtAgg, which would make every later
    # test's plt.show() block on a Qt event loop.
    import matplotlib
    monkeypatch.setattr(matplotlib, "use", lambda *_args, **_kwargs: None)
    gui.Phase2DiagnosticsTab._render_distortion(
        fake, [{"run_id": "geometry-probe", "params": {}}])

    card = captured["card"]
    figure = card.args[1]
    axes = figure.axes[0]
    quiver = axes.collections[0]
    csv = card.kwargs["csv_export"]
    rows = csv["rows"]
    chosen = next(row for row in rows if row[1:3] == (324.0, 252.0))
    index = next(i for i, row in enumerate(rows) if row is chosen)

    ray = np.array([[(324.0 - 320.0) / 200.0,
                     (252.0 - 240.0) / 200.0, 1.0]])
    world = (np.linalg.inv(camera.extrinsic)
             @ np.column_stack([ray, np.ones(1)]).T).T[:, :3]
    expected = camera.project_points(world).reshape(-1, 2)[0] - [324.0, 252.0]

    assert axes.get_xlim() == (0.0, 640.0)
    assert axes.get_ylim() == (480.0, 0.0)
    assert quiver.scale == 1.0
    assert quiver.scale_units == "xy"
    assert quiver.angles == "xy"
    assert quiver.U[index] == pytest.approx(expected[0], abs=5e-7)
    assert quiver.V[index] == pytest.approx(expected[1], abs=5e-7)
    assert chosen[3] == pytest.approx(expected[0], abs=5e-7)
    assert chosen[4] == pytest.approx(expected[1], abs=5e-7)
    assert chosen[5] == pytest.approx(np.hypot(chosen[3], chosen[4]))
    assert expected[0] > 0 and expected[1] > 0
    assert csv["metadata"]["units"] == {"coordinates": "px", "displacement": "px"}
    assert axes.get_xlabel() == "x (px)"
    assert axes.get_ylabel() == "y (px)"


@pytest.mark.data
@pytest.mark.slow
def test_real_phase1_phase2_provenance_after_image_edit(data_dir, tmp_path):
    """The real ChArUco producer flags edited Phase 1 inputs and fresh passes."""
    import shutil

    from pyCamSet.workflow import phase1
    from pyCamSet.workflow.education_trace import RunContext
    from pyCamSet.workflow.workspace import WorkspaceManager, workspace_path_for

    image_root = tmp_path / "calibration_charuco"
    shutil.copytree(data_dir / "calibration_charuco", image_root)
    workspace = WorkspaceManager(workspace_path_for(image_root))
    target = {
        "type": "ChArUco", "num_squares_x": 20, "num_squares_y": 20,
        "square_size": 4, "legacy": False,
    }
    phase1_params = {
        "f_loc": str(image_root), "target": target,
        "selected_cameras": [], "n_lim": None, "caching": False,
        "upscale_factor": 1, "rescale_and_gamma": False,
        "manifest_digest": True,
    }
    phase1_run = phase1.run(phase1_params, workspace, log=lambda _line: None)
    assert phase1_run["error"] is None

    # Replace an image only in this private copy; the tracked fixture stays intact.
    shutil.copyfile(image_root / "1" / "cal1.jpg",
                    image_root / "1" / "cal2.jpg")
    phase2_params = {
        "f_loc": str(image_root), "target": target,
        "selected_cameras": [], "n_lim": None, "caching": False,
        "high_distortion": False, "fixed_params": None,
        "min_detections_per_board": 12, "lens_model": "pinhole",
    }
    saved_run = phase2.run(
        phase2_params, workspace, log=lambda _line: None,
        phase1_run=phase1_run)
    assert saved_run["error"] is None
    saved_context = RunContext.from_run(
        workspace,
        {**saved_run, "_run_dir": workspace.run_dir("phase2", saved_run["run_id"])},
        "phase2",
    )
    assert saved_run["artifacts"]["education_used_detections"]["source"] == (
        "phase1_pickle")
    assert saved_context.phase1_source_check().status == "mismatch"

    fresh_run = phase2.run(
        dict(phase2_params, high_distortion=True), workspace,
        log=lambda _line: None)
    assert fresh_run["error"] is None
    fresh_context = RunContext.from_run(
        workspace,
        {**fresh_run, "_run_dir": workspace.run_dir("phase2", fresh_run["run_id"])},
        "phase2",
    )
    assert fresh_run["artifacts"]["education_used_detections"]["source"] == (
        "fresh_pass_not_saved")
    assert fresh_context.phase1_source_check().status == "could not be checked"


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.parametrize("upscale_factor", [1, 2])
def test_real_charuco_high_distortion_uses_final_detections_and_resolution(
        data_dir, tmp_path, monkeypatch, upscale_factor):
    """The tracked corpus calibrates the authoritative pass in its own frame."""
    from pyCamSet.workflow import phase1
    from pyCamSet.workflow.detections import extract_detection_and_cam_res
    from pyCamSet.workflow.workspace import WorkspaceManager
    from pyCamSet.utils.saving import load_pickle

    image_root = data_dir / "calibration_charuco"
    workspace = WorkspaceManager(tmp_path / "workspace")
    target = {
        "type": "ChArUco", "num_squares_x": 20, "num_squares_y": 20,
        "square_size": 4, "legacy": False,
    }
    phase1_params = {
        "f_loc": str(image_root), "target": target,
        "selected_cameras": [], "n_lim": None, "caching": False,
        "upscale_factor": upscale_factor, "rescale_and_gamma": False,
        "manifest_digest": False,
    }
    phase1_run = phase1.run(phase1_params, workspace, log=lambda _line: None)
    assert phase1_run["error"] is None

    initial_detections, initial_cam_res = extract_detection_and_cam_res(
        load_pickle(phase1_run["artifacts"]["detected_datapoints_pickle"]))
    original_detect = phase2.detect_datapoints_in_imfile
    original_calibrate = phase2.calibrate
    second_pass = []
    calibration_inputs = []

    def detect(**kwargs):
        result = original_detect(**kwargs)
        second_pass.append(result)
        return result

    def calibrate(detections, cam_res, calibration_target, **kwargs):
        calibration_inputs.append((detections, cam_res))
        return original_calibrate(detections, cam_res, calibration_target, **kwargs)

    monkeypatch.setattr(phase2, "detect_datapoints_in_imfile", detect)
    monkeypatch.setattr(phase2, "calibrate", calibrate)
    phase2_params = {
        "f_loc": str(image_root), "target": target,
        "selected_cameras": [], "n_lim": None, "caching": False,
        "high_distortion": True, "fixed_params": None,
        "min_detections_per_board": 12, "lens_model": "pinhole",
    }
    phase2_run = phase2.run(
        phase2_params, workspace, log=lambda _line: None,
        phase1_run=phase1_run)

    assert phase2_run["error"] is None
    assert len(second_pass) == 1
    assert len(calibration_inputs) == 2
    final_detections, final_cam_res = second_pass[0]
    assert calibration_inputs[-1][0] is final_detections
    assert calibration_inputs[-1][1] == final_cam_res
    assert initial_detections.cam_names == final_detections.cam_names
    assert [tuple(res) for res in initial_cam_res] == [
        tuple(value * upscale_factor for value in final_cam_res[0])
    ] * len(final_cam_res)
    assert phase2_run["artifacts"]["education_used_detections"]["source"] == (
        "fresh_pass_not_saved")
