"""Images every telecentric camera saw only one flat face of are placed from several cameras at once.

On a real telecentric rig, 17 of 99 images were left out of Phase 3 although up to seven
cameras saw each - every camera saw exactly one face, whose orthographic tilt (towards or away from
the camera) and depth no single camera can resolve.
"""
import json
import re
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import pyCamSet.utils.general_utils as gu
from pyCamSet.cameras.telecentric_camera import TelecentricCamera
from pyCamSet.optimisation.template_handler import (
    SINGLE_FACE_REASON_OFF, SINGLE_FACE_REASON_ONE_CAMERA, SINGLE_FACE_RECOVERY_MAX_RMS_PX,
    recover_single_face_telecentric_poses,
)
from pyCamSet.utils.intrinsics_report import telecentric_project


def _face(n=4, spacing=1.0):
    g = np.stack(np.meshgrid(np.arange(n), np.arange(n)), -1).reshape(-1, 2) * spacing
    return g


def _problem(n_cams, seed=0, seen_by=None, n_images=1):
    """A telecentric rig of n_cams seeing face 0 only; seen_by limits which cameras saw it."""
    rng = np.random.default_rng(seed)
    g = _face()
    face0 = np.column_stack([g, np.zeros(len(g))])            # z = 0 plane
    face1 = np.column_stack([np.zeros(len(g)), g])            # x = 0 plane (never seen here)
    point_data = np.stack([face0, face1])                     # (2, 16, 3)
    K = np.array([[80.0, 0, 320], [0, 80.0, 240], [0, 0, 1]])
    names = [f"cam{i}" for i in range(n_cams)]
    cams = {name: TelecentricCamera(intrinsic=K, res=[640, 480], distortion_coefs=np.array([0.0]),
                                    name=name) for name in names}
    # cameras look at the origin from different directions, all seeing face 0 from its front
    extrinsics = []
    for i in range(n_cams):
        rvec = np.array([0.35 * np.cos(2.1 * i), 0.35 * np.sin(2.1 * i), 0.3 * i])
        extrinsics.append(gu.make_4x4h_tform(rvec, rng.normal(size=3)))
    extrinsics = np.array(extrinsics)
    truth = gu.make_4x4h_tform(np.array([0.2, -0.15, 0.4]), np.array([1.0, -2.0, 3.0]))
    rows = []
    for c in (range(n_cams) if seen_by is None else seen_by):
        in_camera = extrinsics[c] @ truth
        rvec, _ = cv2.Rodrigues(in_camera[:3, :3])
        uv = telecentric_project(cams[names[c]], face0, rvec.reshape(3), in_camera[:3, 3])
        for image in range(n_images):
            for k, (u, v) in enumerate(uv):
                rows.append([c, image, k, u, v])  # flattened key k = face 0 point k
    data = np.array(rows, dtype=float)
    detection = SimpleNamespace(return_flattened_keys=lambda shape: SimpleNamespace(get_data=lambda: data),
                                max_ims=n_images)
    camset = type("Cams", (), {"get_names": lambda self: names, "__getitem__": lambda self, k: cams[k],
                               "get_n_cams": lambda self: n_cams})()
    target = SimpleNamespace(point_data=point_data)
    return camset, target, detection, extrinsics, truth


def test_a_single_face_image_seen_by_three_cameras_is_placed_exactly():
    cams, target, detection, extrinsics, truth = _problem(3)
    poses = np.full((1, 4, 4), np.nan)
    out = recover_single_face_telecentric_poses(poses, extrinsics, cams, target, detection)
    assert np.all(np.isfinite(out[0]))
    assert np.allclose(out[0][:3, :3], truth[:3, :3], atol=1e-6)  # the true tilt, not its mirror
    assert np.allclose(out[0][:3, 3], truth[:3, 3], atol=1e-5)    # depth fixed by the other cameras
    assert np.all(np.isnan(poses))  # the input is not modified in place


def test_one_camera_cannot_resolve_the_tilt_and_the_image_stays_unplaced():
    cams, target, detection, extrinsics, _ = _problem(1)
    out = recover_single_face_telecentric_poses(np.full((1, 4, 4), np.nan), extrinsics, cams, target, detection)
    assert np.all(np.isnan(out[0]))


def test_placed_images_and_pinhole_rigs_are_left_alone():
    cams, target, detection, extrinsics, truth = _problem(3)
    placed = np.array([truth])
    assert np.array_equal(
        recover_single_face_telecentric_poses(placed, extrinsics, cams, target, detection), placed)



# --- optional recovery, its D3.14 record, and its plumbing (optional,
# --- on by default in the library and the GUI since 2026-10-08) -----------------------

def test_recovery_off_classifies_without_fitting_and_leaves_the_image_unplaced():
    cams, target, detection, extrinsics, _ = _problem(3)
    record = {}
    out = recover_single_face_telecentric_poses(
        np.full((1, 4, 4), np.nan), extrinsics, cams, target, detection, enabled=False, record=record)
    assert np.all(np.isnan(out[0]))
    assert record["enabled"] is False and record["telecentric"] is True
    assert record["recovered"] == [] and record["teaching"] == {}
    assert record["not_recovered"] == [{"image": 0, "reason": SINGLE_FACE_REASON_OFF,
                                        "cameras": ["cam0", "cam1", "cam2"]}]


def test_recovery_on_records_the_fit_and_each_candidate_tilt():
    cams, target, detection, extrinsics, _ = _problem(3)
    record = {}
    recover_single_face_telecentric_poses(
        np.full((1, 4, 4), np.nan), extrinsics, cams, target, detection, record=record)
    assert record["enabled"] is True and record["not_recovered"] == []
    [entry] = record["recovered"]
    assert entry["image"] == 0 and entry["cameras"] == ["cam0", "cam1", "cam2"]
    assert entry["rms_px"] < 1e-6
    candidates = record["teaching"]["0"]
    assert [(c["camera"], c["tilt"]) for c in candidates] == [
        (name, tilt) for name in ("cam0", "cam1", "cam2") for tilt in "+-"]
    # In its own camera the two tilts fit equally: the ambiguity.
    for plus, minus in zip(candidates[::2], candidates[1::2]):
        assert plus["own_camera_rms_px"] < 1e-6 and minus["own_camera_rms_px"] < 1e-6
        # With the tilt held, exactly one of each camera's pair fits every camera.
        kept = sorted([plus["all_cameras_tilt_kept_rms_px"], minus["all_cameras_tilt_kept_rms_px"]])
        assert kept[0] < 1e-6 and kept[1] > 1.0
    assert min(c["all_cameras_rms_px"] for c in candidates) == pytest.approx(entry["rms_px"])
    json.dumps(record)  # saved in the run's metadata


def test_one_camera_and_a_hopeless_fit_are_left_out_with_their_reasons():
    cams, target, detection, extrinsics, _ = _problem(3, seen_by=[1])
    record = {}
    recover_single_face_telecentric_poses(
        np.full((1, 4, 4), np.nan), extrinsics, cams, target, detection, record=record)
    assert record["not_recovered"] == [{"image": 0, "reason": SINGLE_FACE_REASON_ONE_CAMERA,
                                        "cameras": ["cam1"]}]
    # Off, a single camera still gets the one-camera reason, not 'recovery off'.
    recover_single_face_telecentric_poses(
        np.full((1, 4, 4), np.nan), extrinsics, cams, target, detection, enabled=False, record=record)
    assert record["not_recovered"][0]["reason"] == SINGLE_FACE_REASON_ONE_CAMERA

    cams, target, detection, extrinsics, _ = _problem(3)
    extrinsics = extrinsics.copy()
    extrinsics[2] = gu.make_4x4h_tform(np.array([0.6, -0.6, 0.0]), np.zeros(3)) @ extrinsics[2]
    out = recover_single_face_telecentric_poses(
        np.full((1, 4, 4), np.nan), extrinsics, cams, target, detection, record=record)
    assert np.all(np.isnan(out[0])) and record["recovered"] == []
    [entry] = record["not_recovered"]
    assert re.fullmatch(r"best fit \d+\.\d\d px exceeds the 20 px limit", entry["reason"])
    assert float(entry["reason"].split()[2]) > SINGLE_FACE_RECOVERY_MAX_RMS_PX


def test_an_image_a_camera_saw_two_faces_of_is_not_a_single_face_image():
    cams, target, detection, extrinsics, _ = _problem(3)
    data = detection.return_flattened_keys(None).get_data()
    # camera 0 also saw four off-plane points of face 1: unplaced for another reason
    extra = np.array([[0, 0, 16 + k, 300.0 + k, 200.0] for k in (4, 5, 8, 9)], dtype=float)
    both = np.vstack([data, extra])
    detection = SimpleNamespace(return_flattened_keys=lambda shape: SimpleNamespace(get_data=lambda: both))
    record = {}
    out = recover_single_face_telecentric_poses(
        np.full((1, 4, 4), np.nan), extrinsics, cams, target, detection, record=record)
    assert np.all(np.isnan(out[0]))
    assert record["recovered"] == [] and record["not_recovered"] == []


def _stub_rig(monkeypatch, truth, extrinsics):
    """estimate_initial_rig with the rig growth and seed scoring stubbed: image 0 placed, 1 not."""
    from pyCamSet.optimisation import template_handler

    monkeypatch.setattr(template_handler, "estimate_rig", lambda *a, **k: (
        extrinsics, np.array([truth, np.full((4, 4), np.nan)])))
    monkeypatch.setattr(template_handler, "seed_reprojection_error",
                        lambda cams, flat, placed, ext: np.zeros(2 * len(flat)))
    return template_handler


@pytest.mark.parametrize("option", [None, False, True])
def test_estimate_initial_rig_recovers_unless_turned_off(monkeypatch, option):
    cams, target, detection, extrinsics, truth = _problem(3, n_images=2)
    template_handler = _stub_rig(monkeypatch, truth, extrinsics)
    record = {}
    kwargs = {} if option is None else {"recover_single_face_telecentric": option}
    _, poses, _ = template_handler.estimate_initial_rig(
        np.full((3, 2, 4, 4), np.nan), cams, 0, target, detection,
        single_face_record=record, **kwargs)
    assert record["enabled"] is (option is not False)  # the library default is on
    if option is not False:
        assert np.allclose(poses[1], truth, atol=1e-5)
        assert [entry["image"] for entry in record["recovered"]] == [1]
    else:
        assert np.all(np.isnan(poses[1]))
        assert record["not_recovered"][0]["reason"] == SINGLE_FACE_REASON_OFF


def test_the_handler_passes_its_option_to_the_seed(monkeypatch):
    from pyCamSet.optimisation import template_handler

    seen = {}

    class Stop(Exception):
        pass

    def capture(**kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(template_handler, "estimate_camera_relative_poses", capture)
    for options, expected in (({}, True), ({"recover_single_face_telecentric": False}, False)):
        handler = object.__new__(template_handler.TemplateBundleHandler)
        handler.problem_opts = {**template_handler.DEFAULT_OPTIONS, **options}
        handler.camset = handler.detection = handler.target = None
        handler.record_education = False
        handler.seed_join_order = []
        handler.single_face_recovery = {}
        with pytest.raises(Stop):
            handler.calc_initial_params()
        assert seen["recover_single_face_telecentric"] is expected
        assert seen["single_face_record"] is handler.single_face_recovery


def test_phase3_puts_the_param_in_the_handler_options_and_records_d314(tmp_path, monkeypatch):
    from pyCamSet.workflow import phase3

    monkeypatch.setattr(phase3, "load_CameraSet", lambda _: SimpleNamespace(get_names=lambda: []))
    monkeypatch.setattr(phase3, "load_pickle", lambda _: object())
    monkeypatch.setattr(phase3, "extract_detection", lambda value: value)
    monkeypatch.setattr(phase3, "target_of_params", lambda _: None)
    monkeypatch.setattr(phase3, "_lockbox", lambda *args: (None, None, {}))
    monkeypatch.setattr(phase3, "per_camera_mean_reprojection",
                        lambda *args: ({}, np.zeros((1, 2))))
    monkeypatch.setattr(phase3, "solve_quality_gate", lambda *args: {})
    captured = []

    def fake_solve(_cams, _detections, _target, **kwargs):
        captured.append(kwargs["options"])
        record = {"enabled": kwargs["options"]["recover_single_face_telecentric"],
                  "telecentric": True, "recovered": [], "not_recovered": [], "teaching": {}}
        handler = SimpleNamespace(
            get_initial_params=lambda: np.zeros(3), missing_poses=np.zeros(1, dtype=bool),
            record_education=False, single_face_recovery=record, initial_per_im_error=[])
        saved = SimpleNamespace(save=lambda path: path.write_bytes(b"camset"), calibration_report=None)
        result = SimpleNamespace(fun=np.zeros(2), status=1, message="", success=True, nfev=1, x=np.zeros(3))
        return handler, result, saved, {"initial_euclid": 1.0, "final_euclid": 0.5}

    monkeypatch.setattr(phase3, "solve", fake_solve)
    for given, expected in (({}, True), ({"recover_single_face_telecentric": False}, False)):
        run_dir = tmp_path / str(expected)
        run_dir.mkdir()
        params = {"fixed_params": {}, "problem_options": {"outliers": "n"}, "threads": 1, **given}
        _, diagnostics, _, _ = phase3._solve(
            params, run_dir, tmp_path / "seed.camset", tmp_path / "detections.pickle", None, lambda _: None)
        assert captured[-1]["recover_single_face_telecentric"] is expected
        assert diagnostics["D3.14_single_face_recovery"]["enabled"] is expected
        assert params["problem_options"] == {"outliers": "n"}  # the run's params are not edited
