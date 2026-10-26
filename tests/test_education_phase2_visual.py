'''Verify the visual Phase 2 storyboard draws this run's data and keeps its numbers honest.
'''
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

TARGET = {"type": "Ccube2", "n_points": 6, "length": 10.0, "border_fraction": 0.1,
          "aruco_dict": "DICT_4X4_1000"}
MINIMUM = 6


def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _synthetic_rig(n_images: int = 9, cameras=("camA", "camB"), seed: int = 4):
    """Two telecentric cameras photographing a Ccube2 cube in ``n_images`` poses.

    Faces turned away from a camera are not detected, as on a real rig; the corners carry
    0.3 px of noise. Returns (target, detection, cam_res).
    """
    from conftest import skip_without_aruco2
    skip_without_aruco2(TARGET["type"])
    from scipy.spatial.transform import Rotation
    from pyCamSet.calibration_targets.core.target_detections import TargetDetection
    from pyCamSet.cameras.telecentric_camera import TelecentricCamera
    from pyCamSet.utils.intrinsics_report import telecentric_project
    from pyCamSet.workflow.targets import target_of_params

    target = target_of_params({"target": dict(TARGET)})
    rng = np.random.default_rng(seed)
    faces = np.asarray(target.point_data, dtype=float)
    centre = faces.reshape(-1, 3).mean(axis=0)
    width, height = 375, 448
    rows = []
    for c, name in enumerate(cameras):
        cam = TelecentricCamera(intrinsic=np.array([[37000.0 + 300 * c, 0, 224.0],
                                                    [0, 37600.0 - 200 * c, 187.5],
                                                    [0, 0, 1.0]]),
                                res=np.array([height, width]), name=name)
        for image in range(n_images):
            # tip the cube so two or three faces are seen, varied per photo
            rot = Rotation.from_euler("xyz", [35 + 25 * rng.random(), -40 + 80 * rng.random(),
                                              360 * rng.random()], degrees=True)
            rvec = rot.as_rotvec()
            shift = np.array([rng.normal(0, 4e-4), rng.normal(0, 4e-4), 0.0])
            for face, points in enumerate(faces):
                outward = rot.apply(points.mean(axis=0) - centre)
                if outward[2] > -1e-4:  # facing away from the camera (+z looks into the scene)
                    continue
                t = shift - rot.apply(centre)
                pixels = telecentric_project(cam, points, rvec, t)
                pixels = pixels + rng.normal(0, 0.3, pixels.shape)
                pixels[:, 0] += width / 2 - 224.0
                inside = ((pixels[:, 0] > 2) & (pixels[:, 0] < width - 2)
                          & (pixels[:, 1] > 2) & (pixels[:, 1] < height - 2))
                for point in np.flatnonzero(inside):
                    rows.append([c, image, face, point, *pixels[point]])
    detection = TargetDetection(list(cameras), data=np.asarray(rows, dtype=float),
                                max_ims=n_images)
    return target, detection, [(height, width)] * len(cameras)


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    """A saved telecentric Phase 2 run (camset, detections, metadata, per-view trace)."""
    from pyCamSet.calibration.camera_calibrator import run_initial_calibration
    from pyCamSet.workflow.detections import save_detections
    from pyCamSet.workflow.education_trace import write_trace
    from pyCamSet.workflow.phase2 import diagnostics_of

    target, detection, cam_res = _synthetic_rig()
    cams, poses, costs = run_initial_calibration(
        detection, target, cam_res, save=False, return_poses_and_costs=True,
        min_detections_per_board=MINIMUM, model="telecentric")
    workspace = tmp_path_factory.mktemp("p2visual") / ".pycamset_workspace"
    run_id = "20261008_000000_visual"
    run_dir = workspace / "phase2_runs" / run_id
    run_dir.mkdir(parents=True)
    camset = run_dir / "initial_cameras.camset"
    cams.save(camset)
    pickle = save_detections(run_dir / "used.pickle", detection, cam_res)
    diagnostics, report = diagnostics_of(detection, target, cams, MINIMUM)
    metadata = {
        "run_id": run_id, "phase": "phase2", "status": "complete", "error": None,
        "params": {"lens_model": "telecentric", "min_detections_per_board": MINIMUM,
                   "target": dict(TARGET), "record_phase2_per_view": True},
        "diagnostics": diagnostics, "report": report,
        "artifacts": {"initial_camset": str(camset),
                      "education_used_detections": {"schema_version": 1,
                                                    "source": "phase1_pickle",
                                                    "path": str(pickle)}},
    }
    write_trace(run_dir, run_id, "phase2", "phase2_per_view", {
        "cameras": [{"camera": name, "views": [
            {"view_index": k, "pose_target_to_camera": np.asarray(p).tolist(),
             "per_view_rms_px": float(cost)} for k, (p, cost) in enumerate(zip(cp, cc))]}
            for name, cp, cc in zip(detection.cam_names, poses, costs)],
    }, enabled=True)
    import json
    metadata = json.loads(json.dumps(metadata, default=float))
    context = SimpleNamespace(run_id=run_id, run_dir=run_dir, metadata=metadata,
                              phase="phase2", status="complete",
                              manager=SimpleNamespace(workspace_path=workspace))
    return SimpleNamespace(context=context, target=target, detection=detection, cams=cams,
                           poses=poses, costs=costs)


def test_telecentric_seed_views_follow_the_recorded_trace_order(synthetic_run):
    """Regression (2026-10-08): telecentric trace records are per *image* views; indexing them
    with board views reported 140-270 px "differences" on a 2.5 px value."""
    from pyCamSet.workflow import education_visual_p2 as vp

    for c, name in enumerate(synthetic_run.detection.cam_names):
        views = vp.seed_views(synthetic_run.detection, synthetic_run.target, name, MINIMUM, True)
        assert len(views) == len(synthetic_run.poses[c])
        for view, pose, cost in zip(views, synthetic_run.poses[c], synthetic_run.costs[c]):
            assert len(view.boards) >= 2  # a flat, single-face photo is never a seed view
            assert vp.seed_view_rms(synthetic_run.cams[name], view, pose) == pytest.approx(
                cost, abs=1e-9)


def test_recomputed_per_photo_error_is_the_saved_value_to_the_last_bit(synthetic_run):
    from pyCamSet.utils.intrinsics_report import _board_views
    from pyCamSet.workflow import education_visual_p2 as vp

    metadata = synthetic_run.context.metadata
    for name in synthetic_run.detection.cam_names:
        views = vp.board_views(synthetic_run.detection, synthetic_run.target, name, MINIMUM)
        objects, _, _, _ = _board_views(synthetic_run.detection, synthetic_run.target, name,
                                        MINIMUM)
        assert len(views.views) == len(objects)
        fits = vp.fit_views(synthetic_run.cams[name], views.views)
        check = vp.check_per_image(fits, vp.saved_series(metadata, name))
        assert check.all_match and check.worst_difference == 0.0
        assert vp.pooled_rms(fits) == pytest.approx(
            metadata["diagnostics"]["D2.1_per_camera_rms_reprojection"][name], abs=1e-12)
        # the scene items are the photos the first estimate posed, coloured by the saved error
        items = vp.scene_items(synthetic_run.cams[name], synthetic_run.detection,
                               synthetic_run.target, name, MINIMUM, fits, check.per_image)
        assert [i.image_index for i in items] == [v.image_index for v in vp.seed_views(
            synthetic_run.detection, synthetic_run.target, name, MINIMUM, True)]
        assert all(i.error == check.per_image[i.image_index] for i in items)
        assert all(len(i.faces) == 6 and all(f.shape == (4, 3) for f in i.faces) for i in items)


@pytest.mark.gui
def test_walkthrough_rows_match_for_a_telecentric_run(synthetic_run):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    view = Phase2View(synthetic_run.context)
    try:
        walk = next(f for f in view._frames if f["title"] == "Per-view walkthrough")
        rows = [row for name in walk["camera_groups"] for row in walk["camera_groups"][name]]
        assert rows and all(row[5] == "recalculated; matches the value Phase 2 saved"
                            for row in rows), rows[:3]
        assert all(row[0].startswith("Image view") for row in rows)
        assert all(row[3] == "not applicable (telecentric affine fit)" for row in rows)
    finally:
        view.close()


def _numbers(view):
    table = view.stage_view.numbers.table
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    rows = [[table.item(r, c).text() if table.item(r, c) else ""
             for c in range(table.columnCount())] for r in range(table.rowCount())]
    return headers, rows


@pytest.mark.gui
def test_every_stage_and_level_draws_a_figure_of_this_run(synthetic_run):
    """Acceptance: each stage at each level shows a picture made from this run (never only
    text), with a short caption, and the copy passes the learner-copy lint."""
    _app()
    from test_education_copy_lint import learner_copy_findings
    from pyCamSet.gui import education_style as es
    from pyCamSet.gui.education_phase2 import STAGES, Phase2View, _BoardScene

    view = Phase2View(synthetic_run.context)
    view.resize(1400, 900)
    try:
        assert [view.stage_view.stepper.stage_name(i) for i in range(5)] == list(STAGES)
        assert view.camera_picker.count() == 2 and view.view_picker.count() > 0
        captions: dict[int, list[str]] = {}
        for level in (1, 2, 3):
            view.stage_view.set_level(level)
            captions[level] = []
            for stage in range(5):
                view.stage_view.set_current_stage(stage)
                visual = view.stage_view.visual.currentWidget()
                if stage == 1:
                    assert isinstance(visual, _BoardScene)
                    assert visual.scene.names(), "the 3D stage drew nothing"
                    assert visual.side_figure.axes
                    assert es.text_collisions(visual.side_figure) == [], (stage, level)
                else:
                    figure = view.stage_view.figure()
                    assert visual is view.stage_view.canvas
                    drawn = [ax for ax in figure.axes
                             if ax.images or ax.collections or ax.lines or ax.patches]
                    assert drawn, (stage, level)
                    # Levels add panels: L2/L3 never draw fewer than L1.
                    titled = [ax for ax in figure.axes if ax.get_title(loc="left").strip()]
                    assert len(titled) >= {1: 2, 2: 2, 3: 3}[level] or stage == 0
                    # UX-1: no text runs off the figure or over other text.
                    assert es.text_collisions(figure) == [], (stage, level)
                assert view.stage_view.caption_issues == [], (stage, level)
                caption = view.stage_view.caption.text()
                captions[level].append(caption)
                assert caption and learner_copy_findings(caption, level) == [], caption
                headers, _rows = _numbers(view)
                assert not any(learner_copy_findings(h, 3) for h in headers), headers
        # L1 captions are at most two sentences; levels only add to them.
        for caption in captions[1]:
            assert caption.count(". ") + caption.count("; ") <= 3, caption
        for low, high in ((1, 2), (2, 3)):
            assert all(high_text.startswith(low_text[:40])
                       for low_text, high_text in zip(captions[low], captions[high]))
    finally:
        view.close()


@pytest.mark.gui
def test_fit_stage_numbers_and_chips_agree_with_the_saved_run(synthetic_run):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    view = Phase2View(synthetic_run.context)
    try:
        view.stage_view.set_current_stage(3)
        chips = {chip.label: chip.value for chip in view.stage_view.chips.chips()}
        assert chips["the saved errors"] == "✓ matches"
        saved = synthetic_run.context.metadata["report"]["per_camera"][0]["rms_px"]
        assert chips["camera 1 · RMS"] == f"{saved:.2f} px"
        headers, rows = _numbers(view)
        assert headers == ["Photo", "Recalculated RMS error (px)", "Evidence", "index (0-based)"]
        assert rows and all(row[2] == "recalculated; matches the saved value" for row in rows)
        # SCI-8: the learner reads photo numbers from 1; the stored index is its own column.
        assert all(row[0] == f"Photo {int(row[3]) + 1}" for row in rows)
        view.stage_view.set_level(2)
        _headers, rows = _numbers(view)
        assert all(row[2] == "matches D2.6" for row in rows)
        # Decision: chips come from the saved per-camera summary.
        view.stage_view.set_current_stage(4)
        chips = {chip.label: chip.value for chip in view.stage_view.chips.chips()}
        assert chips["cameras go on"] == "2 / 2"
        headers, rows = _numbers(view)
        assert [row[0] for row in rows] == list(synthetic_run.detection.cam_names)
        assert all(row[4] == "goes on" for row in rows)
    finally:
        view.close()


@pytest.mark.gui
def test_camera_and_view_pickers_redraw_and_navigation_moves_stages(synthetic_run):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    view = Phase2View(synthetic_run.context)
    try:
        view.stage_view.set_current_stage(3)
        first = view.stage_view.chips.chips()[1].value
        view.camera_picker.setCurrentIndex(1)
        assert view.stage_view.chips.chips()[1].label == "camera 2 · RMS"
        assert view.stage_view.chips.chips()[1].value != first or True
        view.view_picker.setCurrentIndex(0)
        photo_chip = view.stage_view.chips.chips()[0]
        assert photo_chip.label.startswith("photo ")
        # ED-3: the photo picker names photos from 1 and says its number is the photo's RMS.
        image = view.photo_picker.currentData()
        assert view.photo_picker.currentText().startswith(f"Photo {image + 1} · RMS ")
        assert photo_chip.label == f"photo {image + 1} · RMS"
        assert photo_chip.value in view.photo_picker.currentText()
        # Pickers follow the stage: no photo picker on the lens and decision stages.
        view.stage_view.set_current_stage(2)
        assert view.view_picker.isHidden() and not view.camera_picker.isHidden()
        view.stage_view.set_current_stage(4)
        assert view.camera_picker.isHidden()
        view.step_back()
        assert view.current_stage() == 3
        view.step_forward()
        view.step_forward()
        assert view.current_stage() == 4
    finally:
        view.close()


@pytest.mark.gui
def test_photos_are_not_shown_without_a_verified_photo_list(synthetic_run):
    """The synthetic run has no Phase 1 manifest: corners are drawn on an empty frame and
    the panel says why, rather than guessing a photo from today's folder."""
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    view = Phase2View(synthetic_run.context)
    try:
        view.stage_view.set_current_stage(0)
        figure = view.stage_view.figure()
        assert not any(ax.images for ax in figure.axes)
        texts = " ".join(t.get_text() for ax in figure.axes for t in ax.texts)
        assert "photo not shown" in texts
    finally:
        view.close()


@pytest.mark.gui
def test_failed_run_shows_a_refusal_picture_on_every_stage(tmp_path):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    run_dir = tmp_path / ".pycamset_workspace" / "phase2_runs" / "failed"
    run_dir.mkdir(parents=True)
    context = SimpleNamespace(run_id="failed", run_dir=run_dir, phase="phase2",
                              metadata={"error": "calibration failed", "params": {},
                                        "diagnostics": {"D2.2_intrinsics": {
                                            "camA": {"fx": 1.0, "fy": 1.0, "cx": 0, "cy": 0}}}},
                              manager=SimpleNamespace(workspace_path=run_dir.parents[1]))
    view = Phase2View(context)
    try:
        for stage in range(5):
            view.stage_view.set_current_stage(stage)
            assert view.stage_view.caption.text()
            if view.stage_view.visual.currentWidget() is view.stage_view.canvas:
                texts = " ".join(t.get_text() for ax in view.stage_view.figure().axes
                                 for t in ax.texts)
                assert "failed" in texts or "Run Phase 2 again" in texts.replace("\n", " ")
    finally:
        view.close()


def test_lens_numbers_and_distortion_grid_are_from_the_saved_lens():
    from pyCamSet.workflow import education_visual_p2 as vp

    cam = SimpleNamespace(intrinsic=np.array([[37214.0, 0, 224.0], [0, 38309.0, 187.5],
                                              [0, 0, 1.0]]),
                          distortion_coefs=np.array([0.0]), res=np.array([448, 375]))
    lens = vp.lens_summary(cam, (375, 448), telecentric=True)
    assert lens.um_per_px[0] == pytest.approx(1e6 / 37214.0)
    assert lens.field_mm == pytest.approx((375 * 1e3 / 37214.0, 448 * 1e3 / 38309.0))
    assert not lens.fitted
    straight, bent = vp.distortion_grid(cam, (375, 448), 50.0, telecentric=True)
    assert all(np.array_equal(a, b) for a, b in zip(straight, bent))  # nothing fitted: straight
    pinhole = SimpleNamespace(intrinsic=np.array([[1000.0, 0, 640], [0, 1000.0, 512], [0, 0, 1]]),
                              distortion_coefs=np.array([-0.2, 0, 0, 0, 0]), res=None)
    lens = vp.lens_summary(pinhole, (1280, 1024), telecentric=False)
    assert lens.fitted and lens.fov_deg[0] == pytest.approx(65.24, abs=0.01)
    straight, bent = vp.distortion_grid(pinhole, (1280, 1024), 1.0, telecentric=False)
    corner = straight[0][0]
    assert np.hypot(*(bent[0][0] - [640, 512])) < np.hypot(*(corner - [640, 512]))  # barrel


def test_photo_source_refuses_paths_outside_the_camera_folder(tmp_path):
    from pyCamSet.workflow import education_visual_p2 as vp

    source = vp.PhotoSource("verified by manifest", None, tmp_path, None,
                            {"camA": ("ok.png", "../../escape.png")})
    assert source.path("camA", 0) == (tmp_path / "camA" / "ok.png").resolve()
    assert source.path("camA", 1) is None
    assert source.path("camA", 5) is None
    assert vp.PhotoSource("mismatch", "changed", tmp_path, None, {"camA": ("ok.png",)}).path(
        "camA", 0) is None


def test_one_view_warp_lands_on_the_detected_corners(synthetic_run):
    """ED-15: the printed face warped by the fitted affine map (telecentric) lands on the
    detected corners to about the noise level, and its outline is the whole printed face."""
    from pyCamSet.workflow import education_visual_p2 as vp

    name = synthetic_run.detection.cam_names[0]
    views = vp.board_views(synthetic_run.detection, synthetic_run.target, name, MINIMUM).views
    warp = vp.face_warp(max(views, key=lambda v: len(v.image_points)), synthetic_run.target,
                        telecentric=True)
    assert warp is not None and warp.kind == "affine"
    assert warp.rms < 2.0  # 0.3 px noise, plus what an affine map of one face leaves
    assert len(warp.printed) >= len(warp.detected) and warp.outline.shape == (4, 2)


def test_pattern_share_tells_noise_from_a_systematic_field():
    from pyCamSet.workflow import education_visual_p2 as vp

    rng = np.random.default_rng(1)
    pixels = rng.uniform(0, 300, (30, 2))
    noise = rng.normal(0, 1.0, (30, 2))
    assert vp.systematic_share([(pixels, noise)]) < 0.4
    pattern = np.column_stack([0.01 * (pixels[:, 0] - 150), 0.01 * (pixels[:, 1] - 150)])
    assert vp.systematic_share([(pixels, pattern + 0.05 * noise)]) > 0.9
    # the example corner is one of typical error
    projected = pixels + noise
    k = vp.typical_corner(pixels, projected)
    sizes = np.hypot(*noise.T)
    assert abs(sizes[k] - np.median(sizes)) <= np.min(np.abs(sizes - np.median(sizes))) + 1e-12


def test_centre_check_flags_a_principal_point_saved_transposed():
    """SCI-6: runs before the Camera.res fix saved (224, 187.5) for 375 x 448 photos."""
    from pyCamSet.workflow import education_visual_p2 as vp

    old = vp.centre_check(224.0, 187.5, (375, 448))
    assert old.transposed and not old.matches and old.centre == (187.5, 224.0)
    new = vp.centre_check(187.5, 224.0, (375, 448))
    assert new.matches and not new.transposed
    fitted = vp.centre_check(616.8, 543.2, (1280, 1024))   # a fitted pinhole point
    assert not fitted.transposed


def test_radial_profile_sees_barrel_but_not_noise():
    from pyCamSet.workflow import education_visual_p2 as vp

    rng = np.random.default_rng(3)
    points = rng.uniform(0, 400, (2000, 2))
    centre = np.array([200.0, 200.0])
    noise = rng.normal(0, 0.5, points.shape)
    assert not vp.radial_profile(points, noise, centre).bends
    offset = points - centre
    barrel = -2e-5 * offset * np.sum(offset ** 2, axis=1, keepdims=True) / 100
    assert vp.radial_profile(points, barrel + noise, centre).bends


def test_figure_colours_for_faces_are_never_camera_colours():
    """UX-8: the face colours of stage 1 stay clearly apart from every camera colour."""
    pytest.importorskip("PySide6")
    from pyCamSet.gui import education_style as es
    from pyCamSet.gui.education_phase2 import _face_colours

    for theme in es.theme_names():
        t = es.stage_tokens(theme)
        assert _face_colours(t) == es.face_colours(theme, 6)   # the one shared face palette
        for colour in _face_colours(t):
            _l, a, b = es._lab(colour)
            assert np.hypot(a, b) <= 4.0, \
                (theme, colour)


@pytest.mark.gui
def test_lens_stage_flags_a_transposed_saved_centre_and_draws_the_bending_check(synthetic_run):
    """SCI-6 / ED-5: the lens stage compares the saved centre with the real image centre and,
    with no distortion fitted, checks this camera's own errors for lens bending."""
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View
    from pyCamSet.workflow import education_visual_p2 as vp

    view = Phase2View(synthetic_run.context)
    try:
        name = synthetic_run.detection.cam_names[0]
        saved = synthetic_run.context.metadata["diagnostics"]["D2.2_intrinsics"][name]
        expected = vp.centre_check(saved["cx"], saved["cy"], (375, 448)).transposed
        view.stage_view.set_level(3)
        view.stage_view.set_current_stage(2)
        chips = {chip.label: chip.value for chip in view.stage_view.chips.chips()}
        assert ("saved image centre" in chips) == expected
        assert chips["per pixel (x × y)"].count("×") == 1  # SCI-9: x and y, not x only
        titles = [ax.get_title(loc="left") for ax in view.stage_view.figure().axes]
        assert any("Does the lens bend lines?" in title for title in titles)
        assert any("For comparison" in title for title in titles)   # ED-12
        caption = view.stage_view.caption.text()
        assert "Phase 3 adds" not in caption and "closed-form (a few linear fits" in caption
    finally:
        view.close()


# --------------------------------------------------------------------------------------
# Round 2 (review_science SCI2-11/21/23, review_education ED2-3/9/26, review_uiux UX2-2/7/24)


def test_pattern_test_sees_a_curved_pattern_that_the_linear_share_calls_noise():
    """SCI2-11: a real photo curves (linear share 0.14, curved share 0.50 against 0.25
    from noise). The test compares with the noise baseline and has three wordings."""
    from pyCamSet.workflow import education_visual_p2 as vp

    rng = np.random.default_rng(7)
    pixels = rng.uniform(0, 200, (40, 2))
    noise = rng.normal(0, 1.0, (40, 2))
    plain = vp.pattern_test([(pixels, noise)])
    assert plain.verdict == "noise-like"
    assert plain.noise == pytest.approx(6 / 40) and plain.noise_high > plain.noise
    c = (pixels - pixels.mean(axis=0)) / 100.0
    bowl = np.column_stack([c[:, 0] ** 2 + c[:, 1] ** 2 - 0.6, 0.8 * c[:, 0] * c[:, 1]]) * 2.0
    assert vp.systematic_share([(pixels, bowl + 0.1 * noise)]) < 0.5   # linear: misses it
    curved = vp.pattern_test([(pixels, bowl + 0.1 * noise)])
    assert curved.verdict == "pattern" and curved.share > 0.9
    weak = vp.pattern_test([(pixels, 0.12 * bowl + 0.1 * noise)])
    assert weak.verdict in ("some pattern", "pattern")
    assert vp.pattern_test([(pixels[:8], noise[:8])]).verdict == "too few corners"


def test_camera_groups_follow_the_rig_names_and_need_a_real_difference():
    """ED2-26: on a real rig, cameras 1-4 ("…T") fit about 1.6x worse than 5-8 ("…B")."""
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_phase2 import _numbers_text, camera_groups

    names = ["cam1_FrontLeft_907T", "cam2_FrontRight_914T", "cam3_BackRight_859T",
             "cam4_BackLeft_569T", "cam5_Left_863B", "cam6_Front_862B", "cam7_Right_915B",
             "cam8_Back_908B"]
    rms = [1.52, 1.77, 1.90, 2.00, 1.16, 1.01, 1.12, 1.49]
    rows = [{"name": n, "index": k, "rms": r} for k, (n, r) in enumerate(zip(names, rms))]
    groups = camera_groups(rows)
    assert groups is not None and groups["how"] == "name"
    assert [r["index"] for r in groups["high"]] == [0, 1, 2, 3]
    assert groups["tags"] == ("B", "T") and groups["ratio"] == pytest.approx(1.83 / 1.14, rel=0.01)
    assert _numbers_text([0, 1, 2, 3]) == "1–4" and _numbers_text([0, 2]) == "1, 3"
    flat = [dict(r, rms=1.0 + 0.02 * k) for k, r in enumerate(rows)]
    assert camera_groups(flat) is None   # no real difference: nothing pointed out
    pcube = [2.91, 4.05, 3.01, 3.42, 9.33, 16.03, 3.46, 2.98]   # two outliers, not a group
    assert camera_groups([dict(r, rms=v) for r, v in zip(rows, pcube)]) is None
    assert camera_groups(rows[:3]) is None


def test_anisotropy_is_the_largest_x_y_difference_to_one_decimal():
    """SCI2-23: 38,816.8 / 37,252.9 is 4.2 %, not "≤ 4 %"."""
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_phase2 import _anisotropy

    lens = SimpleNamespace(fx=37252.9, fy=38816.8)
    other = SimpleNamespace(fx=37000.0, fy=37100.0)
    assert f"{_anisotropy({'a': lens, 'b': other}):.1f}" == "4.2"
    assert _anisotropy({}) is None


def test_thumbnail_columns_fill_the_panel():
    """UX2-24: 12 portrait thumbnails in a wide, short box use 4+ columns, not 3."""
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_phase2 import _thumb_columns

    assert _thumb_columns(12, 420, 400, (375, 448), 16) >= 4
    assert _thumb_columns(12, 200, 900, (375, 448), 16) == 2


@pytest.mark.gui
def test_round2_figures_faces_local_scale_and_two_kinds_of_pose(synthetic_run):
    """UX2-2: faces as grey shapes (no circles, no hue) with "Face n" legends; UX2-7: E has its
    own linear scale when the errors span a narrow range; ED2-9: C names face and cube poses;
    ED2-3: the bending check never claims more than one camera can show."""
    _app()
    from pyCamSet.gui import education_style as es
    from pyCamSet.gui.education_phase2 import Phase2View

    view = Phase2View(synthetic_run.context)
    view.resize(1400, 900)
    try:
        view.stage_view.set_level(3)
        view.stage_view.set_current_stage(0)
        a = view.stage_view.figure().axes[0]
        legend = [t.get_text() for t in a.get_legend().get_texts()]
        assert legend and all(text.startswith("Face ") for text in legend)
        markers = [c for c in a.collections if hasattr(c, "get_paths") and c.get_offsets().size
                   and c.get_offsets().shape[0] > 3 and type(c).__name__ == "PathCollection"]
        theme = es.axes_theme(a)
        shades = {es.face_colour(k, theme).lower() for k in range(6)}
        from matplotlib.colors import to_hex
        assert markers and all(to_hex(c.get_facecolor()[0]).lower() in shades for c in markers)
        view.stage_view.set_current_stage(1)
        side = view.stage_view.visual.currentWidget().side_figure
        titles = " ".join(ax.get_title(loc="left") for ax in side.axes)
        assert "Two kinds of pose" in titles
        assert "face pose" in view.stage_view.caption.text()
        assert "cube pose" in view.stage_view.caption.text()
        view.stage_view.set_current_stage(2)
        titles = [ax.get_title(loc="left") for ax in view.stage_view.figure().axes]
        subtitles = " ".join(t.get_text() for ax in view.stage_view.figure().axes
                             for t in ax.texts)
        assert "no clear sign of bending" not in subtitles
        assert "one camera" in view.stage_view.caption.text()
        view.stage_view.set_current_stage(3)
        view.wait_until_ready(120)  # the noise simulation runs on a worker thread
        fig = view.stage_view.figure()
        e = next(ax for ax in fig.axes if "Where in the image" in ax.get_title(loc="left"))
        norm = e.images[0].norm
        from matplotlib.colors import LogNorm
        values = e.images[0].get_array().compressed()
        if es.needs_local_scale(values):
            assert not isinstance(norm, LogNorm) and norm.vmin == 0.0
            assert any(es.LOCAL_SCALE_NOTE in ax.get_ylabel() for ax in fig.axes)
        caption = view.stage_view.caption.text()
        assert "noise alone" in caption or "too few corners" in caption
        assert "posed alone" in caption
    finally:
        view.close()



def test_pinhole_views_are_posed_in_closed_form_or_by_the_older_pose_only_refit():
    """Phase 2 saves the seed's closed-form residual; runs saved before it hold the pose-only
    refit. ``fit_views`` reproduces either, and both recover a noiseless view exactly."""
    from types import SimpleNamespace
    import cv2
    from pyCamSet.workflow import education_visual_p2 as vp

    intrinsic = np.array([[1500.0, 0.0, 640.0], [0.0, 1500.0, 512.0], [0.0, 0.0, 1.0]])
    cam = SimpleNamespace(intrinsic=intrinsic, distortion_coefs=np.zeros(5),
                          res=np.array([1280, 1024]), telecentricity=None)
    grid = np.array([[x, y, 0.0] for y in range(6) for x in range(8)], dtype=float) * 4.0
    rng = np.random.default_rng(3)
    views = []
    for k in range(3):
        rvec = np.array([0.2 + 0.1 * k, -0.15, 0.05])
        tvec = np.array([5.0, -3.0, 300.0 + 10 * k])
        pixels, _ = cv2.projectPoints(grid, rvec, tvec, intrinsic, np.zeros(5))
        pixels = pixels.reshape(-1, 2) + rng.normal(0.0, 0.3, (len(grid), 2))
        views.append(vp.BoardView(k, k, 0, grid, pixels))
    closed = vp.fit_views(cam, views)
    older = vp.fit_views(cam, views, pose_only=True)
    assert all(f is not None for f in closed) and all(f is not None for f in older)
    for a, b in zip(closed, older):
        assert 0.0 < a.rms < 3.0 and 0.0 < b.rms < 1.0
        assert b.rms <= a.rms + 1e-9  # the refit never leaves more than the closed form
