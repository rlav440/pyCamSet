'''Verify the visual Phase 4 education view: every stage at every level draws this
run's data, the 3D scenes/playback/morph show exactly the prepared geometry, and the layout fits.
workflow/education_visual_p4 and a data-backed replay check on a generated run.
'''
from __future__ import annotations

import math
import re
from types import SimpleNamespace

import numpy as np
import pytest


def _app():
    pytest.importorskip("PySide6")
    import os
    os.environ.setdefault("PYCAMSET_EDU_3D", "static")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _look_at(centre, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    c = np.asarray(centre, float)
    z = np.asarray(target, float) - c
    z /= np.linalg.norm(z)
    y = -(np.asarray(up, float) - (np.asarray(up, float) @ z) * z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.vstack([x, y, z])
    return R, -R @ c


def _cube(size=0.01, grid=4):
    s = np.linspace(-size / 2, size / 2, grid)
    faces = []
    for axis in range(3):
        for sign in (-1, 1):
            u, v = np.meshgrid(s, s)
            face = np.zeros((u.size, 3))
            others = [i for i in range(3) if i != axis]
            face[:, others[0]], face[:, others[1]] = u.ravel(), v.ravel()
            face[:, axis] = sign * size / 2
            faces.append(face)
    return np.unique(np.round(np.vstack(faces), 12), axis=0)


def _synthetic_visual(*, telecentric=True, steps=4, excluded=True, blocking=()):
    """A complete, internally consistent Phase4Visual (4 cameras around a 10 mm cube)."""
    from pyCamSet.workflow.education_visual_p4 import (
        CameraPose, ErrorTable, ExcludedImage, Phase4Visual, Playback, SolverState,
        TELECENTRIC_DISPLAY_DISTANCE, gate_checks,
    )
    rng = np.random.default_rng(3)
    printed = _cube()
    n = len(printed)
    one = (n - 4, n - 3)
    unseen = (n - 2, n - 1)
    several = tuple(i for i in range(n) if i not in one + unseen)
    correction = rng.normal(0, 1.5e-4, printed.shape)
    correction[list(unseen)] = 0.0
    size = float(np.linalg.norm(np.ptp(printed, axis=0)))
    n_cams, n_images = 4, 6
    names = tuple(f"cam{k + 1}" for k in range(n_cams))

    def cameras(turn):
        out = []
        for k in range(n_cams):
            a = 2 * math.pi * k / n_cams + turn * (k + 1)
            R, t = _look_at((math.cos(a), math.sin(a), 0.3))
            if telecentric:
                position = -TELECENTRIC_DISPLAY_DISTANCE * size * R[2]
                out.append(CameraPose(names[k], k, R, -R @ position, True, placed=True))
            else:
                out.append(CameraPose(names[k], k, R, t, False))
        return tuple(out)

    # Observations: every camera sees every several-direction corner in images 0..4.
    cams_col, imgs_col, pts_col = [], [], []
    for c in range(n_cams):
        for i in range(n_images - 1):
            for p in several + one:
                cams_col.append(c), imgs_col.append(i), pts_col.append(p)
    cams_col, imgs_col, pts_col = map(np.asarray, (cams_col, imgs_col, pts_col))

    def table(scale):
        norms = scale * (0.5 + rng.random(len(cams_col)))
        return ErrorTable(norms, cams_col, imgs_col, pts_col, n)

    states = []
    for k in range(steps + 1):
        s = k / steps
        errors = table(3.0 * (1 - s) + 0.3)
        states.append(SolverState(
            nit=k, cost=1000.0 * (0.05 ** s), cameras=cameras(0.002 * (1 - s)),
            shape=printed + correction * s, display_shape=printed + correction * s,
            pose_centres=rng.normal(0, 1e-3, (n_images - 1, 3)), norms=errors.norms,
            per_camera=errors.per_camera(), mean_px=errors.mean_px,
            point_error=errors.per_point(), params=np.full(10, float(k))))
    share = np.zeros((len(states), 4))
    share[1:] = rng.random((steps, 4))
    share[1:] /= share[1:].sum(axis=0)
    playback = Playback("available", "", tuple(states), tuple((s.nit, s.cost) for s in states),
                        share, (24, 12, 30, 3 * n))
    coverage = np.zeros((n_images, n_cams), dtype=int)
    coverage[:n_images - 1] = len(several) + len(one)
    excluded_images = ()
    if excluded:
        coverage[n_images - 1, 2] = 16
        excluded_images = (ExcludedImage(n_images - 1, (2,), (16,)),)
    gate = {
        "status": "incomplete" if blocking else "complete", "blocking_flags": list(blocking),
        "notices": ([f"1 image(s) had no usable target pose and were excluded from the solve: "
                     f"[{n_images - 1}]"] if excluded else []),
        "excluded_images": [n_images - 1] if excluded else [],
        "camera_coverage": True, "observed_cameras": list(range(n_cams)),
        "expected_camera_count": n_cams,
        "observed_images": list(range(n_images - 1 if excluded else n_images)),
        "expected_image_count": n_images, "error_reduced": True, "solver_success": True,
        "camera_quality": {"physically_plausible": True},
        "constraint_disposition": {"status": "constrained", "global_nullity": 0,
                                   "point_local_nullity": 2, "fixed_pose_observed": True},
        "jacobian_nullity": {"rank": 98, "free_parameters": 100, "total_nullity": 2,
                             "point_local_nullity": 2, "global_nullity": 0, "tolerance": 1e-5,
                             "smallest_singular_values": [3e-3, 1e-2, 2e-2, 4e-2],
                             "jacobian_shape": [400, 100]},
    }
    metadata = {"diagnostics": {"quality_gate": gate, "D4.3_initial_euclid_px": states[0].mean_px,
                                "D4.3_final_euclid_px": states[-1].mean_px},
                "report": {"solver_message": "ftol reached"}}
    from pyCamSet.workflow.education_visual_p4 import ClosingSummary, CornerRays
    example = CornerRays(photo=2, corner=5, cameras=(0, 1), pixels=((10.0, 20.0), (30.0, 40.0)),
                         origins=(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])),
                         directions=(np.array([0.0, 0.5, -0.866]), np.array([-0.5, 0.0, -0.866])),
                         angle_deg=60.0, gap=3.5e-6, point=np.zeros(3))
    visual = Phase4Visual(
        "available", "", camera_names=names, telecentric=(telecentric,) * n_cams, unit="mm",
        mm_per_unit=1000.0, target_size=size, printed=printed, gauge_indices=several[:3],
        start=states[0], final=states[-1], start_label="Phase 4's recorded start",
        start_errors=table(3.0), final_errors=table(0.3), excluded=excluded_images,
        coverage=coverage, n_images=n_images,
        phase3_camera_mean={c: 3.0 + 0.1 * c for c in range(n_cams)},
        phase3_camera_source="Phase 3 run's saved D3.12",
        phase4_camera_mean={c: 0.3 + 0.01 * c for c in range(n_cams)},
        checks=gate_checks(metadata, "incomplete" if blocking else "complete"),
        notices=tuple(gate["notices"]),
        blocking=tuple(blocking), nullity=gate["jacobian_nullity"],
        refined=printed + correction, correction=correction,
        ray_separation_deg=np.where(np.isin(np.arange(n), several), 40.0, 1.0),
        several=several, one_direction=one, unseen=unseen, exaggeration=10,
        distortion_p3={c: 5.0 + c for c in range(n_cams)},
        distortion_p4={c: 0.3 + 0.1 * c for c in range(n_cams)},
        swap_px={"lens_swap": 3.1, "shape_swap": 9.0},
        closing=ClosingSummary({c: 27.0 for c in range(n_cams)}, 120, 3.5, 60.0, example,
                               "run/self_calibrated_cameras.camset", 1e6),
        telecentricity_p3={c: -2.0 - 0.1 * c for c in range(n_cams)} if telecentric else {},
        telecentricity_p4={c: 0.4 for c in range(n_cams)} if telecentric else {})
    visual._playback = playback
    visual._precision = {"precision_mm": 0.001, "precision_note": "synthetic",
                         "weak_shares": {"lenses": 0.1, "camera turns": 0.05,
                                         "photo poses": 0.7, "corners": 0.15}}
    return visual, metadata


def _derivation(visual, final_px):
    from pyCamSet.workflow.education_derive import Phase4Derivation
    classes = SimpleNamespace(several_directions=visual.several,
                              one_direction=visual.one_direction, unseen=visual.unseen,
                              ray_separation_deg=tuple(visual.ray_separation_deg))
    return Phase4Derivation(
        status="available", reason=None, run_id="run-4", d41_visible_points=len(visual.several),
        d43_final_px=final_px, d45_gauge_scale=1.0, d47_mean_displacement_mm=0.2,
        point_classes=classes, display_refit_points=visual.refined, display_refit_scale=1.0012,
        scale_consistency_ratio=1.0, metric_qualifier_required=False, frames=(),
        several_directions_mean_displacement_mm=0.18, saved_units_unverified=False)


def _state(**overrides):
    values = {
        "path": None, "origin_run_id": "p3", "verification": SimpleNamespace(value="verified"),
        "reason": None, "confirmed": True, "source_check": "verified by manifest",
        "phase3_before_px": None, "stored_mean_px": 3.3, "start_mismatch": None,
        "fixed_pose_matches": True, "effective_fixed_parameter_differences": (),
        "rig_comparison_withheld": False, "photo_names": ("ref.png",),
        "phase3_fixed_pose_ordinal1": 1, "phase4_fixed_pose_ordinal1": 1,
        "loss": "linear", "f_scale": 1.0,
    }
    return SimpleNamespace(**(values | overrides))


def _view(tmp_path, monkeypatch, *, status="complete", **visual_kwargs):
    from pyCamSet.gui import education_phase4
    visual, metadata = _synthetic_visual(**visual_kwargs)
    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_a, **_k: _derivation(visual, visual.final.mean_px))
    monkeypatch.setattr(education_phase4, "prepare_phase4_visual", lambda *_a, **_k: visual)
    run_dir = tmp_path / "phase4_runs" / "run-4"
    run_dir.mkdir(parents=True, exist_ok=True)
    context = SimpleNamespace(
        run_id="run-4", phase="phase4", run_dir=run_dir, status=status,
        metadata={"params": {"record_education": True}, **metadata},
        phase3_state_source=lambda: _state(), artifact=lambda phase: None,
        quality_disposition="checked")
    view = education_phase4.Phase4View(context)
    # Laid out at the dialog's real size, so figure-text checks see the real panels.
    from PySide6.QtWidgets import QApplication
    view.resize(1380, 820)
    view.show()
    for _ in range(3):
        QApplication.processEvents()
    return view, visual


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _axes_with_data(figure):
    return [ax for ax in figure.axes
            if ax.collections or ax.lines or ax.patches or ax.images or ax.artists]


def es_title(ax):
    """The panel title text of ``ax`` ("" for none)."""
    from pyCamSet.gui.education_style import _edu_title
    title = _edu_title(ax)
    return title.get_text() if title is not None else ""


# --------------------------------------------------------------------------------------
# Qt-free preparation


def test_no_free_directions_card_states_what_the_check_means_at_l1():
    """Teaching P1 (A2 / A3): the card must agree with the symmetry account. With no
    single-detection corner the line says every move changes a photo once the fixed numbers
    are removed; a failed check says the rig's position or turn would stay free."""
    from pyCamSet.workflow.education_visual_p4 import gate_checks
    _visual, metadata = _synthetic_visual()
    disposition = metadata["diagnostics"]["quality_gate"]["constraint_disposition"]
    disposition["point_local_nullity"] = 0
    checks = {c.key: c for c in gate_checks(metadata, "complete")}
    assert checks["nullspace"].line == "other moves all change a photo"
    assert "drift" not in checks["nullspace"].guard
    assert checks["nullspace"].guard == "else: the rig's position or turn stays free"


def test_gate_checks_read_the_saved_gate_as_traffic_lights():
    from pyCamSet.workflow.education_visual_p4 import gate_checks
    _visual, metadata = _synthetic_visual()
    checks = {c.key: c for c in gate_checks(metadata, "complete")}
    assert [c for c in checks] == ["cameras", "images", "nullspace", "error", "solver"]
    assert checks["cameras"].tone == "good" and checks["cameras"].value == "4 / 4"
    # An excluded image is a notice (amber), never a pass and never blocking.
    assert checks["images"].tone == "warn" and checks["images"].value == "5 / 6"
    assert "1 left out" in checks["images"].line
    assert checks["nullspace"].tone == "good" and checks["nullspace"].value == "0 free directions"
    assert checks["error"].tone == "good" and "→" in checks["error"].value
    # Plain words for the solver's stop reason (ED-8 / UX-19), and what failing would mean.
    assert checks["solver"].value == "Converged"
    assert checks["solver"].line == "stopped: the error stopped falling"
    assert all(c.guard for c in checks.values())
    # SCI-3: a numerical rank check, not "nothing can slide"; once-seen corners named.
    assert checks["nullspace"].title == "No free directions"
    assert checks["nullspace"].line == "2 one-detection corners may slide"
    assert checks["nullspace"].guard == "else: the rig's position or turn stays free"
    # A failed measurement is red and never reported as zero.
    gate = metadata["diagnostics"]["quality_gate"]
    gate["jacobian_nullity"] = None
    gate["constraint_disposition"] = {"status": "weakly_constrained",
                                      "global_nullity": "not_verified",
                                      "reason": "Jacobian null-space measurement failed: x"}
    checks = {c.key: c for c in gate_checks(metadata, "complete")}
    assert checks["nullspace"].tone == "bad" and checks["nullspace"].value == "not measured"
    # No gate at all: "not recorded", not green.
    only = gate_checks({"diagnostics": {}}, "complete")
    assert len(only) == 1 and only[0].tone == "unknown" and only[0].value == "not recorded"


def test_seeding_only_options_are_not_other_settings():
    """SCI-4: Phase 3's seeding switch (recover_single_face_telecentric) and logging options do
    not make 'other start settings differ'; a real solve setting (a step limit) still does."""
    from pyCamSet.workflow.education_visual_p4 import Phase4Narrative
    narrative = Phase4Narrative.__new__(Phase4Narrative)
    narrative.metadata = {"params": {"problem_options": {
        "fixed_pose": 0, "loss": "linear", "verbosity": 2, "max_nfev": 1000}}}
    narrative.state = _state()
    narrative.phase3_options = lambda: ({"fixed_pose": 0, "verbosity": 0, "max_nfev": 1000,
                                         "recover_single_face_telecentric": True}, {})
    assert narrative.setting_differences() == []
    narrative.phase3_options = lambda: ({"fixed_pose": 0, "max_nfev": 4}, {})
    assert narrative.setting_differences() == ["max_nfev"]


def test_exaggeration_units_and_error_tables():
    from pyCamSet.workflow.education_visual_p4 import (
        EXAGGERATIONS, ErrorTable, length_unit, nice_exaggeration,
    )
    assert nice_exaggeration(17.3, 0.3) == 5
    assert nice_exaggeration(17.3, 0.0) == 1
    assert nice_exaggeration(10.0, math.nan) == 1
    assert nice_exaggeration(10.0, 1e-9) == EXAGGERATIONS[-1]
    assert length_unit(np.array([0.2, 0.4])) == ("µm", 1000.0)
    assert length_unit(np.array([2.0, 4.0])) == ("mm", 1.0)
    table = ErrorTable(np.array([1.0, 3.0, 2.0, 4.0]), np.array([0, 0, 1, 1]),
                       np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), 3)
    assert table.mean_px == 2.5
    assert table.per_camera() == {0: 2.0, 1: 3.0}
    assert table.per_image() == {0: 1.5, 1: 3.5}
    images, means = table.by_camera_image()[1]
    assert list(images) == [0, 1] and list(means) == [2.0, 4.0]
    per_point = table.per_point()
    assert per_point[0] == 1.5 and per_point[1] == 3.5 and math.isnan(per_point[2])


def test_display_helpers_keep_cameras_off_the_target():
    """The display frame puts the rig's up on +z; the view itself comes from the shared
    auto-framing (Scene3D.fit_view), tested with the foundation."""
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_phase4 import box_edges, display_rotation
    visual, _ = _synthetic_visual()
    cams = visual.start.cameras
    Q = display_rotation(cams)
    np.testing.assert_allclose(Q @ Q.T, np.eye(3), atol=1e-12)
    # The rig's image-up direction becomes +z.
    up = -np.mean([c.R[1] for c in cams], axis=0)
    np.testing.assert_allclose(Q @ (up / np.linalg.norm(up)), [0, 0, 1], atol=1e-9)
    origins, ends = box_edges(visual.printed)
    assert origins.shape == ends.shape == (12, 3)
    np.testing.assert_allclose(np.linalg.norm(ends - origins, axis=1), 0.01, atol=1e-12)


# --------------------------------------------------------------------------------------
# The view


@pytest.mark.gui
def test_every_stage_and_level_draws_this_runs_data(tmp_path, monkeypatch):
    """Acceptance: every stage at every level has a figure of this run's data; captions
    keep to 2 / 3 / 4 sentences; no stage is text only."""
    _app()
    from pyCamSet.gui.education_phase4 import SplitVisual
    from pyCamSet.workflow.education_visual_p4 import STAGES
    view, visual = _view(tmp_path, monkeypatch)
    try:
        assert view.stage_view.stepper.count() == 6
        assert [view.stage_view.stepper.stage_name(i) for i in range(6)] == list(STAGES)
        for level in (1, 2, 3):
            for stage in range(6):
                view.stage_text(stage, level=level)
                widget = view.stage_view.visual.currentWidget()
                caption = view.stage_view.caption.text()
                assert len(_sentences(caption)) <= {1: 2, 2: 3, 3: 4}[level], (stage, level)
                assert view.stage_view.chips.chips(), (stage, level)
                assert view.numbers_rows(), (stage, level)
                if isinstance(widget, SplitVisual):
                    assert stage in (0, 1, 2)
                    assert widget.scene.names(), (stage, level)
                    assert widget.title.text().startswith("A  ")
                    if widget.canvas.isVisible() or not widget.canvas.isHidden():
                        assert _axes_with_data(widget.figure), (stage, level)
                else:
                    assert stage in (3, 4, 5), (stage, level)
                    assert _axes_with_data(view.stage_view.figure()), (stage, level)
        # Detail levels change what is drawn, not only the text.
        def drawn(stage, level):
            view.stage_text(stage, level=level)
            widget = view.stage_view.visual.currentWidget()
            if isinstance(widget, SplitVisual):
                side = len(widget.figure.axes) if not widget.canvas.isHidden() else 0
                return sorted(widget.scene.names()), side
            return len(view.stage_view.figure().axes)
        for stage in range(6):
            assert drawn(stage, 1) != drawn(stage, 3), stage
    finally:
        view.close()


@pytest.mark.gui
def test_start_stage_shows_phase3_errors_and_the_left_out_images(tmp_path, monkeypatch):
    _app()
    view, visual = _view(tmp_path, monkeypatch)
    try:
        text = view.stage_text(0, level=2)
        # Photos are numbered from 1 (SCI-8): stored index 5 is "Photo 6".
        assert ("Photo 6 is left out: no target position could be found (each was seen "
                "by one camera only).") in text
        chips = {c.label: c for c in view.stage_view.chips.chips()}
        assert chips["photos placed"].value == "5 / 6" and chips["photos placed"].tone == "warn"
        assert chips["mean error at the start"].value == f"{visual.start.mean_px:.2f} px"
        assert ("Photo 6 left out", "camera 3",
                "16 corners; no usable target pose; index (0-based) 5") in view.numbers_rows()
        scene = view.split.scene
        corners = scene._impl.items["corners"]
        seen = np.isfinite(visual.start_errors.per_point())
        assert len(corners["verts"]) == int(seen.sum())
        # Plain numbered badges in 3D (no labels that could collide); each camera's Phase 3
        # mean error is a bar on its strip-plot column at L2.
        badges = sorted(item["badge"].get_text() for name, item in scene._impl.items.items()
                        if name.startswith("cam"))
        assert badges == ["1", "2", "3", "4"]
        strip = next(ax for ax in view.split.figure.axes
                     if ax.get_title(loc="left").startswith("B  Error per photo"))
        means = sorted(line.get_ydata()[0] for line in strip.lines if len(line.get_ydata()) == 2)
        np.testing.assert_allclose(means, sorted(visual.phase3_camera_mean.values()))
        # The side figure: strip plot by camera, then the left-out photo card.
        titles = [ax.get_title(loc="left") for ax in view.split.figure.axes]
        assert any(t.startswith("B  Error per photo") for t in titles)
        assert any(t.startswith("C  Left out") for t in titles)
        # Corners use the shared log error scale (one colour = one error size, UX-6/ED-19).
        from pyCamSet.gui.education_style import ERROR_SCALE_PX
        assert corners["pts"].norm.vmin == ERROR_SCALE_PX[0]
        assert corners["pts"].norm.vmax == ERROR_SCALE_PX[1]
        from pyCamSet.gui.education_style import text_collisions
        assert text_collisions(view.split.figure) == []
    finally:
        view.close()


@pytest.mark.gui
def test_shape_morph_exaggerates_only_several_direction_corners(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui.education_dialog import TransportBar
    from pyCamSet.gui.education_phase4 import MORPH_POSITIONS, _to_display
    view, visual = _view(tmp_path, monkeypatch)
    try:
        view.stage_text(1, level=2)
        # The one transport bar (UX-3), Exaggerate behind its ⚙ menu.
        assert view.stage_view._controls is view.morph
        assert isinstance(view.morph, TransportBar) and view.morph.count() == MORPH_POSITIONS
        assert view.morph.position() == MORPH_POSITIONS - 1  # opens on the refined shape
        # UX2-21: the counter reads like every other bar's.
        assert view.morph.short_text(10) == f"step 11 of {MORPH_POSITIONS}"
        assert view.morph.short_text(0).startswith(f"step 1 of {MORPH_POSITIONS} · ")
        factor = view.morph.exaggeration
        assert factor == visual.exaggeration == 10
        Q = view._morph_items["Q"]
        items = view.split.scene._impl.items
        several, one = np.array(visual.several), np.array(visual.one_direction)
        np.testing.assert_allclose(
            items["refined"]["verts"],
            _to_display(Q, visual.printed[several] + factor * visual.correction[several]))
        np.testing.assert_allclose(
            items["refined_one"]["verts"],
            _to_display(Q, visual.printed[one] + visual.correction[one]))
        view.morph.go_to((MORPH_POSITIONS - 1) // 2)  # no faces here: a linear blend
        np.testing.assert_allclose(
            items["refined"]["verts"],
            _to_display(Q, visual.printed[several] + 0.5 * factor * visual.correction[several]))
        view.morph.go_to(0)
        np.testing.assert_allclose(items["refined"]["verts"],
                                   _to_display(Q, visual.printed[several]), atol=1e-15)
        # Printed: nothing has moved yet, so the colour (distance moved) is zero (ED-28).
        np.testing.assert_allclose(items["refined"]["pts"].get_array(), 0.0, atol=1e-12)
        # A new factor redraws, and every label names it.
        view.morph.set_exaggeration(1)
        assert view.morph.exaggeration == 1
        assert "lines ×1 longer" in view.split.subtitle.text()
        # The real (not exaggerated) sizes drive the colour scale, in µm.
        assert "(µm)" in view.split.subtitle.text()
        sizes, unit = visual.correction_display()
        assert unit == "µm"
        view.morph.go_to(MORPH_POSITIONS - 1)
        np.testing.assert_allclose(items["refined"]["pts"].get_array(), sizes[several])
        # Play restarts from the printed design and never runs on its own.
        view.morph.toggle_playback()
        assert view.morph.position() == 0 and view.morph.is_playing()
        view.morph.stop_for_context_change()
        assert not view.morph.is_playing()
    finally:
        view.close()


def test_face_decomposition_recovers_known_plate_moves():
    """SCI-1 / ED-7: each face of a cube fitted as a rigid plate recovers a known outward
    offset and tilt; what is left inside the faces is the injected warp; boards give ()."""
    from pyCamSet.workflow.education_visual_p4 import decompose_faces, target_faces
    g = np.linspace(-0.004, 0.004, 5)
    u, w = (a.ravel() for a in np.meshgrid(g, g))
    faces = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            face = np.zeros((u.size, 3))
            others = [i for i in range(3) if i != axis]
            face[:, others[0]], face[:, others[1]] = u, w
            face[:, axis] = sign * 0.005
            faces.append(face)
    point_data = np.stack(faces)                       # (6, 25, 3), metres
    nominal = point_data.reshape(-1, 3)
    groups = target_faces(point_data)
    assert len(groups) == 6 and all(len(g_) == 25 for g_ in groups)
    refined = nominal.copy()
    offsets = [1e-4, 2e-4, 0.5e-4, 3e-4, 1.5e-4, 2.5e-4]
    tilt = math.radians(0.5)
    for k, idx in enumerate(groups):
        axis, sign = divmod(k, 2)
        normal = np.zeros(3)
        normal[axis] = -1.0 if sign == 0 else 1.0
        centre = nominal[idx].mean(axis=0)
        other = (axis + 1) % 3
        c, s = math.cos(tilt), math.sin(tilt)
        R = np.eye(3)
        R[[axis, axis, other, other], [axis, other, axis, other]] = [c, -s, s, c]
        refined[idx] = (nominal[idx] - centre) @ R.T + centre + offsets[k] * normal
    refined[3] += np.array([0.0, 0.0, 2e-5])          # a small warp inside face 1
    moves, plate = decompose_faces(nominal, refined, range(len(nominal)), groups, to_mm=1000.0)
    assert [round(m.offset, 4) for m in moves] == pytest.approx(
        [o * 1000 for o in offsets], abs=2e-3)
    assert all(m.tilt_deg == pytest.approx(0.5, abs=0.05) for m in moves)
    inside = np.linalg.norm(refined - plate, axis=1) * 1000
    assert inside[3] == pytest.approx(0.02, abs=0.005) and np.median(inside) < 0.005
    # A flat board (rows of points) and a single plane are not faces.
    assert target_faces(np.zeros((7, 9, 3)) + np.arange(9)[None, :, None] * [1, 0, 0]) == ()
    assert target_faces(point_data[:1]) == ()
    # SCI2-6 / ED2-5: opposite faces give the alignment-free width change (the two outward
    # moves added) against the design's 10 mm, and how far the pair is from parallel.
    from pyCamSet.workflow.education_visual_p4 import face_pairs, face_squareness
    pairs = face_pairs(moves, to_mm=1000.0)
    assert [p.faces for p in pairs] == [(0, 1), (2, 3), (4, 5)]
    assert [p.change for p in pairs] == pytest.approx([0.3, 0.35, 0.4], abs=2e-3)
    assert all(p.design == pytest.approx(10.0) for p in pairs)
    assert all(p.angle_deg < 1.1 for p in pairs)
    # A rigid turn + shift of the whole refined cube changes no angle between its faces.
    c, s = math.cos(0.3), math.sin(0.3)
    turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    moved, _plate = decompose_faces(nominal, refined @ turn.T + 1e-3, range(len(nominal)),
                                    groups, to_mm=1000.0)
    angles = {(a, b): d for a, b, d in face_squareness(moves)}
    angles_moved = {(a, b): d for a, b, d in face_squareness(moved)}
    shifted, _plate = decompose_faces(nominal, refined + [1e-4, -2e-4, 3e-4],
                                      range(len(nominal)), groups, to_mm=1000.0)
    assert [p.change for p in face_pairs(shifted, 1000.0)] == pytest.approx(
        [p.change for p in pairs], abs=1e-9)
    assert len(angles) == 12
    for key, value in angles.items():
        assert angles_moved[key] == pytest.approx(value, abs=1e-6)


def test_swap_noise_and_freedom_helpers():
    """ED2-1 swap test (one evaluation each, nothing re-solved), SCI2-8 3D noise, SCI2-9
    freedom-alone drop and ED2-6 gate wording."""
    from pyCamSet.workflow.education_visual_p4 import (
        ErrorTable, Phase4Visual, distortion_shifts, gate_checks, swap_errors)

    class FakeReplay:
        blocks = (2, 3, 5, 8)

        def errors(self, params):
            # error grows with the distance of lens (0:2) and corners (5:8) from a "fit"
            lens_off = abs(params[0] - params[5])
            return ErrorTable(np.array([0.3 + lens_off]), np.zeros(1, int), np.zeros(1, int),
                              np.zeros(1, int), 1)

    start = np.array([1.0, 1.0, 0, 0, 0, 1.0, 1, 1])
    final = np.array([2.0, 2.0, 0, 0, 0, 2.0, 2, 2])
    # SCI3-1: the swap is a pose re-fit on the real handler; a replay without one gives {}
    # (never a number from a method it did not run), and mismatched states give {}.
    assert swap_errors(FakeReplay(), start, final) == {}
    assert swap_errors(FakeReplay(), start, final[:4]) == {}
    stored = Phase4Visual("available", "", swap_px={"lens_swap": 2.2})
    assert stored.swap_results() == {"lens_swap": 2.2}
    assert distortion_shifts(None) == {}
    visual = Phase4Visual("available", "", nullity={"jacobian_shape": [30894, 1466],
                                                    "rank": 1463})
    assert visual.freedom_drop() == pytest.approx(1 - math.sqrt((30894 - 1463) / 30894))
    visual._precision = {"precision_mm": 0.0064, "precision_note": "test"}
    sd3, mean3 = visual.noise_3d()
    assert sd3 == pytest.approx(0.0064 * math.sqrt(3))
    assert mean3 == pytest.approx(sd3 * math.sqrt(8 / (3 * math.pi)))
    metadata = {"diagnostics": {"quality_gate": {"error_reduced": True},
                                "D4.3_initial_euclid_px": 3.3, "D4.3_final_euclid_px": 0.3}}
    card = next(c for c in gate_checks(metadata, "complete") if c.key == "error")
    assert card.title == "Mean error fell?" and "sanity" in card.line


@pytest.mark.gui
def test_solver_playback_syncs_scene_cost_and_camera_bars(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_preferences
    monkeypatch.setattr(education_preferences, "preferences_path",
                        lambda: tmp_path / "prefs.json")
    view, visual = _view(tmp_path, monkeypatch)
    playback = visual.playback()
    try:
        assert view.step_controls is None  # nothing until the learner reaches the stage
        view.stage_text(2, level=3)
        controls = view.step_controls
        assert controls is not None and controls.at_start
        assert not controls._timer.isActive()
        Q = view._display_frame()
        items = view.split.scene._impl.items
        for _ in range(len(playback.states) - 1):
            controls.step_forward()
            view.player.stop_for_context_change()  # finish the tween
            k = controls._index
            state = playback.states[k]
            frame = view._solver_frame(state, Q, view.level())
            np.testing.assert_allclose(items["shape"]["verts"], frame["points"]["shape"]["xyz"])
            np.testing.assert_allclose(items["shape"]["pts"].get_array(),
                                       np.nan_to_num(state.point_error[view._solver_indices()]))
            # The marker sits on the error curve (px, ED-19) at the step shown.
            x, y = view._cost_marker.get_data()
            assert (x[0], y[0]) == (state.nit, pytest.approx(state.mean_px))
            # The bars show each camera's error at that step.
            # (UX2-9: on the shared log axis, so each bar runs from the axis floor up to it.)
            bars_ax = view.split.figure.axes[1]
            assert bars_ax.get_yscale() == "log"
            tops = [p.get_y() + p.get_height() for p in bars_ax.patches]
            np.testing.assert_allclose(tops[:len(state.per_camera)],
                                       [state.per_camera[c] for c in sorted(state.per_camera)])
        assert controls._status.text().startswith(f"Step {playback.states[-1].nit} of ")
        # L3 adds the parameter-block heat strip and the RMS curve.
        assert any(ax.get_title(loc="left").startswith("D  Which parameters moved")
                   for ax in view.split.figure.axes)
        curve = view.split.figure.axes[0]
        assert curve.get_ylabel() == "mean error (px)"
        # Playback time follows the error change (UX-11).
        assert view.player._weights is not None
        from pyCamSet.gui.education_style import text_collisions
        assert text_collisions(view.split.figure) == []
        # Leaving and returning keeps the position; a level change too.
        view.set_stage(1)
        view.set_stage(2)
        assert controls._index == len(playback.states) - 1
        view.set_level(1)
        assert view.step_controls is controls and controls._index == len(playback.states) - 1
    finally:
        view.close()


@pytest.mark.gui
def test_trust_stage_draws_gate_coverage_and_null_space(tmp_path, monkeypatch):
    _app()
    view, visual = _view(tmp_path, monkeypatch)
    try:
        from pyCamSet.gui.education_style import text_collisions
        view.stage_text(3, level=1)
        figure = view.stage_view.figure()
        texts = [t.get_text() for ax in figure.axes for t in ax.texts]
        for check in visual.checks:
            assert check.title in texts and check.value in texts and check.guard in texts
        titles = [ax.get_title(loc="left") for ax in figure.axes]
        # L1: real evidence first (ED2-6: the change against its noise), gauge freedom
        # (ED-6) and the overfitting bar (ED-8 / UX2-16: one proportional bar).
        noise_ax = next(ax for ax in figure.axes
                        if ax.get_title(loc="left").startswith("B  Bigger than its noise?"))
        labels = [t.get_text() for t in noise_ax.get_yticklabels()]
        assert labels == ["noise alone (expected)", "mean corner change"]
        unknowns_ax = next(ax for ax in figure.axes
                           if ax.get_title(loc="left").startswith("D  Far more measurements"))
        widths = sorted(p.get_width() for p in unknowns_ax.patches)
        assert widths[-1] == 400  # the recorded measured numbers
        # ED4-10: the look-alike changes panel was retitled for a learner.
        assert any(t.startswith("C  Some changes look identical") for t in titles)
        # SCI2-7: telecentric lenses get 1.3x weaker (magnification / 1.3); the cameras do not move.
        notes = [t.get_text() for ax in figure.axes for t in ax.texts]
        assert sum(t.startswith("same photos") for t in notes) == 2
        assert "lenses 1.3× weaker" in notes and "lenses 30 % weaker" not in notes
        assert "Error went down" not in notes and "Mean error fell?" in notes
        caption = view.stage_view.caption.text()
        assert "The real test is B" in caption and "noise" in caption
        chips = {c.label: c for c in view.stage_view.chips.chips()}
        assert chips["quality gate"].value == "Usable" and chips["quality gate"].tone == "good"
        assert "photos left out" not in chips  # the cards say it (UX-37)
        assert text_collisions(figure) == []
        view.stage_text(level=2)
        figure = view.stage_view.figure()
        coverage_ax = next(ax for ax in figure.axes
                           if ax.get_title(loc="left").startswith("B  Who saw which photo"))
        reds = [p for p in coverage_ax.patches if not p.get_fill()]
        assert len(reds) == len(visual.excluded)
        # Photo numbers from 1 on the raster band (UX-22, SCI-8).
        assert "6" in [t.get_text() for t in coverage_ax.texts]
        assert text_collisions(figure) == []
        view.stage_text(level=3)
        figure = view.stage_view.figure()
        null_ax = next(ax for ax in figure.axes
                       if ax.get_title(loc="left").startswith("E  Numerical rank check"))
        heights = sorted(p.get_height() for p in null_ax.patches)
        assert heights == sorted(visual.nullity["smallest_singular_values"])
        assert any(abs(line.get_ydata()[0] - 1e-5) < 1e-12 for line in null_ax.lines)
        assert null_ax.get_ylabel() == "singular value (mixed units)"
        texts = [t.get_text() for ax in figure.axes for t in ax.texts]
        assert any(t.startswith("100 unknowns (the numbers the solver can change), of which 2 are "
                              "not pinned down (rank 98): 98 determined by the photos + 2 corners "
                              "with one detection (may slide) + 0 weakly fixed lens or pose "
                              "directions")
                   for t in texts)
        assert not any("fixed by the photos" in t or "whole rig" in t or "seen once" in t
                       or "rig-wide" in t for t in texts)
        assert not any("stiffness" in t or "×" in t and "above" in t for t in texts)
        # SCI2-3: the weakest directions are reported on unit-free scales, never as "100 %
        # in the lens numbers".
        caption = view.stage_view.caption.text()
        assert "lens numbers" not in caption and "photo poses (70%)" in caption
        # SCI2-9: what freedom alone could explain, against the observed drop.
        view.stage_text(level=2)
        assert "could lower the RMS error by only about" in view.stage_view.caption.text()
    finally:
        view.close()


@pytest.mark.gui
def test_blocked_run_shows_why_it_is_held_back(tmp_path, monkeypatch):
    _app()
    view, _visual = _view(tmp_path, monkeypatch, status="incomplete",
                          blocking=("solver did not report successful termination",))
    try:
        view.stage_text(3, level=2)
        chips = {c.label: c for c in view.stage_view.chips.chips()}
        assert chips["quality gate"].value == "Held back" and chips["quality gate"].tone == "bad"
        texts = [t.get_text() for ax in view.stage_view.figure().axes for t in ax.texts]
        # UX3-4: the internal flag is said in plain words, with what to do.
        assert any("The solver stopped before it settled." in t for t in texts)
        assert not any("solver did not report successful termination" in t for t in texts)
    finally:
        view.close()


@pytest.mark.gui
def test_result_stage_compares_phase3_and_phase4_per_camera_and_per_photo(tmp_path, monkeypatch):
    _app()
    view, visual = _view(tmp_path, monkeypatch)
    try:
        counts = {}
        for level in (1, 2, 3):
            view.stage_text(4, level=level)
            counts[level] = len(_axes_with_data(view.stage_view.figure()))
        assert counts[1] < counts[2] < counts[3]
        view.stage_text(4, level=1)
        paired = view.stage_view.figure().axes[0]
        heights = [p.get_height() for p in paired.patches if p.get_width() < 0.5]
        expected = []
        for cam in range(4):
            expected += [visual.phase3_camera_mean[cam], visual.phase4_camera_mean[cam]]
        np.testing.assert_allclose(heights, expected)
        chips = {c.label: c for c in view.stage_view.chips.chips()}
        chip = chips["mean error, Phase 3 → 4"]
        assert chip.value == f"3.30 → {visual.final.mean_px:.2f} px" and chip.tone == "good"
        # ED2-1: the corners were freed but the lenses and cameras kept being refined, so the
        # drop is never credited to the shape alone; the lens-vs-shape trade-off is drawn.
        caption = view.stage_view.caption.text()
        assert "only the target's corners were freed" not in caption
        assert "corners, lenses and cameras refined together" in caption
        # ED10-1: the L1 caption carries the swap sentence (C); the lens-bend sentence (B) is at L2+
        assert "Put one part back to Phase 3" in caption and "at most 3.10 px" in caption
        view.set_level(2)
        caption = view.stage_view.caption.text()
        assert "bent photos up to 8.0 px" in caption and "they bent 0.60 px (B)" in caption
        titles = [es_title(ax) for ax in view.stage_view.figure().axes]
        assert any("The lenses changed too" in t for t in titles)
        assert any("Swap one part back" in t for t in titles)
        swap_ax = next(ax for ax in view.stage_view.figure().axes
                       if "Swap one part back" in es_title(ax))
        widths = sorted(round(p.get_x() + p.get_width(), 3) for p in swap_ax.patches)
        assert widths == sorted(round(v_, 3) for v_ in (3.3, 9.0, 3.1, visual.final.mean_px))
        view.stage_text(4, level=2)
        caption = view.stage_view.caption.text()
        assert "lenses at most 3.10 px" in caption and "against Phase 4's 0.30 px" in caption
        assert "3.10 px" in caption and "9.00 px" in caption
        view.stage_text(4, level=3)
        assert "ε flipped sign too (E)" in view.stage_view.caption.text()
        rows = view.numbers_rows()
        assert any(r[0] == "Camera 1 ε" and "% per cm" in r[1] for r in rows)
        assert any(r[0] == "Camera 1 lens distortion" and r[1] == "5 → 0.3 px" for r in rows)
        from pyCamSet.gui.education_style import text_collisions
        for level in (1, 2, 3):
            view.stage_text(4, level=level)
            assert text_collisions(view.stage_view.figure()) == [], level
    finally:
        view.close()


@pytest.mark.gui
def test_theme_switch_redraws_figures_and_scenes_in_the_theme(tmp_path, monkeypatch):
    application = _app()
    from matplotlib.colors import to_hex
    from pyCamSet.gui.education_style import stage_tokens
    from pyCamSet.gui.theme import apply_theme
    view, _visual = _view(tmp_path, monkeypatch)
    old = application.property("pycamsetTheme") or "Light"
    try:
        for theme in ("Dark", "Sepia", "Light"):
            apply_theme(application, theme)
            application.processEvents()
            stage = stage_tokens(theme)["stage"]
            view.set_stage(4)
            assert to_hex(view.stage_view.figure().get_facecolor()) == stage
            view.set_stage(0)
            assert to_hex(view.split.scene.figure.get_facecolor()) == stage
            assert to_hex(view.split.figure.get_facecolor()) == stage
    finally:
        apply_theme(application, old)
        view.close()


@pytest.mark.gui
def test_stage_navigation_seams_and_factory(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase4
    view, _visual = _view(tmp_path, monkeypatch)
    try:
        assert view.current_stage() == 0
        view.step_back()
        assert view.current_stage() == 0
        for expected in (1, 2, 3, 4, 5, 5):
            view.step_forward()
            assert view.current_stage() == expected
        view.step_back()
        assert view.current_stage() == 4
        assert callable(education_phase4.create_education_phase_view)
    finally:
        view.close()


@pytest.mark.gui
def test_failed_run_draws_no_result_but_says_why(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_a, **_k: DerivationUnavailable("the saved result failed"))
    run_dir = tmp_path / "phase4_runs" / "run-4"
    run_dir.mkdir(parents=True)
    context = SimpleNamespace(
        run_id="run-4", phase="phase4", run_dir=run_dir, status="failed",
        metadata={"error": "save/reload check failed", "params": {}, "diagnostics": {}},
        phase3_state_source=lambda: _state(), artifact=lambda phase: None)
    view = education_phase4.Phase4View(context)
    try:
        for stage in (0, 1, 4):
            text = view.stage_text(stage)
            assert text.startswith("This run failed; nothing here is a usable calibration.")
            figure = view.stage_view.figure()
            assert view.stage_view.visual.currentWidget() is view.stage_view.canvas
            messages = [t.get_text() for ax in figure.axes for t in ax.texts]
            assert any("the run failed: save/reload check failed" in m for m in messages), stage
        assert view.split.scene.names() == []
    finally:
        view.close()


@pytest.mark.gui
@pytest.mark.parametrize("size", [(1400, 900), (1280, 800)])
def test_every_stage_fits_the_dialog_without_vertical_scrolling(tmp_path, monkeypatch, size):
    """Layout contract: at 1400x900 and 1280x800 the stage fits with no vertical scroll."""
    application = _app()
    from pyCamSet.gui import education_dialog
    view, _visual = _view(tmp_path, monkeypatch)
    view.close()
    run = SimpleNamespace(run_id="run-4", phase="phase4", status="complete")
    monkeypatch.setattr(education_dialog, "discover_phase_view",
                        lambda _run, parent=None: _view(tmp_path, monkeypatch)[0])
    dialog = education_dialog.EducationDialog([run], ["run-4"])
    try:
        dialog.resize(*size)
        dialog.show()
        phase_view = dialog._view
        for level in (1, 3):
            for stage in range(6):
                phase_view.stage_text(stage, level=level)
                for _ in range(4):
                    application.processEvents()
                bar = dialog._phase_scroll.verticalScrollBar()
                assert bar.maximum() == 0, (size, stage, level)
                # The picture keeps most of the height (this offscreen environment has no
                # font directory, so its fallback font may wrap captions more than a desktop font).
                visual = phase_view.stage_view.visual_panel.height()
                floor = 0.55 if level == 1 else 0.38
                assert visual >= floor * dialog.height(), (size, stage, level, visual)
    finally:
        dialog.close()


# --------------------------------------------------------------------------------------
# Data-backed: the replay reproduces the saved record


@pytest.mark.slow
@pytest.mark.data
def test_replay_reproduces_the_saved_record_on_a_generated_run(tmp_path):
    from education_p4_fixtures import generate_phase4_run
    from pyCamSet.workflow.education_derive import derive_phase4
    from pyCamSet.workflow.education_trace import read_trace
    from pyCamSet.workflow.education_visual_p4 import prepare_phase4_visual

    camset_path, metadata = generate_phase4_run(tmp_path)
    context = SimpleNamespace(
        run_id=metadata["run_id"], phase="phase4", status="complete",
        run_dir=camset_path.parent, metadata=metadata,
        phase3_state_source=lambda: _state(), artifact=lambda phase: camset_path
        if phase == "phase4" else None)
    derivation = derive_phase4(context, include_playback=True)
    assert derivation.status == "available", derivation.reason
    trace = read_trace(context.run_dir, context.run_id, "phase4")
    visual = prepare_phase4_visual(context, metadata, _state(), derivation, trace)
    diagnostics = metadata["diagnostics"]
    assert visual.available and not visual.errors_reason, visual.errors_reason
    assert visual.start.mean_px == pytest.approx(diagnostics["D4.3_initial_euclid_px"], rel=1e-9)
    assert visual.final.mean_px == pytest.approx(diagnostics["D4.3_final_euclid_px"], rel=1e-9)
    per_image = visual.final_errors.per_image()
    for image, value in diagnostics["D4.13_per_image_mean_reprojection"].items():
        assert per_image[int(image)] == pytest.approx(value, rel=1e-9)
    per_camera = visual.final_errors.per_camera()
    for index, name in enumerate(visual.camera_names):
        assert per_camera[index] == pytest.approx(
            diagnostics["D4.12_per_camera_mean_reprojection"][name], rel=1e-9)
    # The display refit of the saved result is derive_phase4's own.
    np.testing.assert_allclose(visual.final.display_shape, derivation.display_refit_points,
                               rtol=0, atol=1e-12)
    playback = visual.playback()
    assert playback.playable
    assert len(playback.states) == len(derivation.frames)
    for state, frame in zip(playback.states, derivation.frames):
        assert state.nit == frame.nit and state.cost == frame.cost
        np.testing.assert_allclose(state.shape, frame.shape_points, rtol=0, atol=1e-12)
    # Every recorded step's replayed objective is its stored cost.
    from pyCamSet.workflow.education_playback import _cost_from_residuals
    replay = visual._replay
    for state in playback.states:
        residuals = np.asarray(replay.loss(state.params.copy()), dtype=np.float64)
        assert _cost_from_residuals(replay.handler, residuals) == pytest.approx(state.cost,
                                                                               rel=1e-8)
    shares = playback.block_share[1:]
    np.testing.assert_allclose(shares.sum(axis=0)[shares.sum(axis=0) > 0], 1.0)


# ED5-1 / ED5-2 / SCI5-1 (round 5): a held-back run states no success. Every text a learner can
# read on it (captions, chips, Numbers rows, panel titles and subtitles) is scanned for the
# success or verdict phrases that round 5 found on the held pcube run.
HELD_FORBIDDEN = (
    r"\bimproved\b", r"\bFit: yes\b", r"\byes, lower\b", r"\blower than Phase 3",
    r"\bis lower than\b", r"\bShape: backed\b", r"\bbacked, all\b", r"\bdid not rise\b",
    r"\bError did not rise\b", r"\balways lowers\b", r"\berror went down\b",
    r"\bMean error fell\b", r"\bdrop comes from\b", r"\bfreed the corners\b",
    r"\bminimises squared errors, so\b", r"\bwent down\b",
    # ED6-1: panel titles of round 6 (stage 5 panels A, B, D and the shape stage's claims)
    r"\bchanged too\b", r"before\s*(?:→|->)\s*after", r"\bWhy the cube is\b",
    r"\bpin a corner down\b", r"\bmoved below the dashes\b",
)
# Phrases that are negated or are a question, so they are not claims: stripped before the scan.
HELD_EXEMPT = (
    "do not show that the calibration improved",
    "are the target-shape differences backed by enough views to be believable?",
)


def _learner_texts(view):
    """(stage, level, text) for every piece of text a learner can read on each stage."""
    out = []
    for stage in range(6):
        for level in (1, 2, 3):
            out.append((stage, level, view.stage_text(stage, level=level)))
            figure = view.stage_view.figure()
            for ax in figure.axes:
                out.append((stage, level, ax.get_title(loc="left")))
                out.append((stage, level, ax.get_title(loc="right")))
                out += [(stage, level, t.get_text()) for t in ax.texts]
                legend = ax.get_legend()
                if legend is not None:
                    out += [(stage, level, t.get_text()) for t in legend.get_texts()]
            out += [(stage, level, t.get_text()) for t in figure.texts]
    return out


@pytest.mark.gui
@pytest.mark.parametrize("flag", ["image observation graph has missing image indices",
                                  "solver did not report successful termination"])
def test_held_run_text_states_no_success_or_verdict(tmp_path, monkeypatch, flag):
    _app()
    view, _visual = _view(tmp_path, monkeypatch, status="incomplete", blocking=(flag,))
    try:
        hits = []
        for stage, level, text in _learner_texts(view):
            cleaned = text or ""
            for exempt in HELD_EXEMPT:
                cleaned = cleaned.replace(exempt, "")
            for pattern in HELD_FORBIDDEN:
                if re.search(pattern, cleaned, flags=re.IGNORECASE):
                    hits.append((stage, level, pattern, cleaned[:160]))
        assert hits == [], f"held-run text states a success or verdict: {hits[:5]}"
    finally:
        view.close()


# SCI7-3 / ED7-2: every measured number on a held run (a px, µm, mm or % value, or a before →
# after pair) carries the held-back tag in the same sentence, line or chip. Plain definitions
# ("1 px", "one square") are not results and are exempt.
HELD_NUMBER = re.compile(r"\d[\d,.]*\s*(?:px|µm|um|mm|%)|\d[\d.]*\s*(?:→|->)\s*\d")
HELD_TAG = re.compile(r"held back|not verified|not judged|not quoted|not a result|for inspection",
                      re.IGNORECASE)
HELD_DEFINITION = re.compile(r"(?<![\d.,])(?:1 px|1 square|1 pixel|one pixel)(?![\d.])")


def _held_untagged_numbers(view) -> list[tuple[int, int, str]]:
    """(stage, level, sentence) for each held-run sentence that shows a number with no tag.

    Round 9 (SCI9-4): the figure's own text is scanned too (panel titles, legends and every
    label drawn on the axes, one label at a time), so a number in a picture needs the tag on
    that label as well."""
    hits = []
    for stage in range(6):
        for level in (1, 2, 3):
            pieces = view.stage_text(stage, level=level).splitlines()
            figure = view.stage_view.figure()
            for ax in figure.axes:
                pieces += [ax.get_title(loc="left"), ax.get_title(loc="right")]
                pieces += [t.get_text() for t in ax.texts]
                legend = ax.get_legend()
                if legend is not None:
                    pieces += [t.get_text() for t in legend.get_texts()]
            pieces += [t.get_text() for t in figure.texts]
            for piece in pieces:
                # a Numbers row is one unit (its tag sits at its end); a caption splits by sentence
                parts = [piece] if " | " in piece else re.split(r"(?<=[.!?])\s+", piece)
                for sentence in parts:
                    if HELD_TAG.search(sentence) or not HELD_NUMBER.search(
                            HELD_DEFINITION.sub("", sentence)):
                        continue
                    hits.append((stage, level, sentence.strip()[:140]))
    return hits


@pytest.mark.gui
def test_held_run_numbers_all_carry_the_held_back_tag(tmp_path, monkeypatch):
    _app()
    view, _visual = _view(tmp_path, monkeypatch, status="incomplete",
                          blocking=("solver did not report successful termination",))
    try:
        hits = _held_untagged_numbers(view)
        assert hits == [], f"held-run numbers without the held-back tag: {hits[:12]}"
    finally:
        view.close()


def test_left_out_label_counts_photos_not_list_tokens():
    """SCI7-2: the left-out list names photos and counts the photos it does not name."""
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_phase4 import _left_out_label
    # stored 0-based indices of the 22 photos left out of the held pcube run (photo = index + 1)
    stored = [9, 12, 13, 14, 17, 18, 24, 28, 36, 42, 57, 63, 64, 69, 70, 71, 72, 75, 88, 89, 94, 95]
    assert _left_out_label(stored) == "Photos 10, 13–15, 18, 19, 25, 29 and 14 other photos"
    assert _left_out_label([4]) == "Photo 5"
    assert _left_out_label([]) == "no photos"


def test_held_tags_go_on_measured_sentences_only():
    """SCI7-3 / ED7-2: a sentence that shows a px, µm, mm, % or × value gets the tag, once; a
    definition ('1 px') and a sentence already saying 'held back' or 'not verified' do not."""
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_phase4 import _held_caption, _held_text, _measures
    assert _measures("the mean error is 8.0 px") and _measures("0.30 px") and _measures("5 µm")
    assert not _measures("1 px is one square of the printed target")
    assert _held_text("The error is 3.3 px.") == "The error is 3.3 px (not verified)."
    assert _held_text("Held back, not verified: 3.3 px.") == "Held back, not verified: 3.3 px."
    assert _held_text("The error is 3.3 px (not verified).") == "The error is 3.3 px (not verified)."
    assert _held_caption("A plain sentence. The cube is 353 µm wider.") == (
        "A plain sentence. The cube is 353 µm wider (not verified).")


@pytest.mark.gui
def test_held_result_panels_carry_not_verified_and_one_band(tmp_path, monkeypatch):
    """ED6-1: on a held run every stage 5 panel title says 'not verified', and the figure
    carries one visible 'held back: not verified' band."""
    _app()
    view, _visual = _view(tmp_path, monkeypatch, status="incomplete",
                          blocking=("solver did not report successful termination",))
    try:
        for level in (1, 2, 3):
            view.stage_text(4, level=level)  # the Result stage (review "stage 5", 0-based 4)
            figure = view.stage_view.figure()
            band = [x.get_text() for ax in figure.axes for x in ax.texts]
            assert any("held back: not verified" in text for text in band), "no held band"
            titles = [ax.get_title(loc="left") for ax in figure.axes if ax.get_title(loc="left")]
            assert titles, "the held result figure has no panel titles"
            untagged = [t for t in titles if "not verified" not in t]
            assert untagged == [], f"held result panel titles without a tag: {untagged}"
    finally:
        view.close()


def test_cost_column_names_the_loss_it_holds():
    """SCI6-1: Σ px² is named only for a linear loss; a robust loss names its own objective."""
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_phase4 import cost_axis_label, cost_column_title

    assert cost_column_title("linear") == "Phase 4 cost (½ Σ px²)"
    assert "Σ px²" not in cost_column_title("soft_l1")
    assert "soft_l1" in cost_column_title("soft_l1")
    assert "squared" not in cost_axis_label("soft_l1")
    assert "soft_l1" in cost_axis_label("soft_l1")


@pytest.mark.gui
def test_complete_run_keeps_its_verdict_wording(tmp_path, monkeypatch):
    """The scan above must not pass by silencing every run: a usable run still gets its answer."""
    _app()
    view, _visual = _view(tmp_path, monkeypatch, status="complete")
    try:
        view.stage_text(5, level=3)
        view.stage_text(4, level=3)
        rows = [" | ".join(r) for r in view.numbers_rows()]
        assert any(row.startswith("Decision") and "Fit: yes" in row for row in rows)
    finally:
        view.close()


# --------------------------------------------------------------------------------------
# Gauge wording: the look-alike changes are symmetries that the solver removes by fixing a
# few numbers (one photo's pose and the printed positions of the run's own gauge corners).
# Only data limits (a corner seen once may slide) are called undetermined.


def test_held_corner_count_comes_from_the_runs_gauge_indices():
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_phase4 import held_corners_copy
    assert held_corners_copy(SimpleNamespace(gauge_indices=(4, 9, 11))) == ("three", 7)
    assert held_corners_copy(SimpleNamespace(gauge_indices=(4, 9))) == ("two", None)


def test_no_recorded_indices_means_no_count_in_the_gauge_copy():
    """G2 / CONTRIBUTING accuracy rule 1: with no saved gauge indices the copy must not state a
    count (the old fallback was a hard-coded "three")."""
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_phase4 import held_corners_copy, look_alike_subtitle
    corners, coordinates = held_corners_copy(SimpleNamespace(gauge_indices=()))
    assert (corners, coordinates) == (None, None)
    # "one photo's pose" is the fixed photo, not a corner count; no corner count may appear
    count_words = {"two", "three", "four", "five", "six", "seven"}
    for tele in (True, False):
        text = look_alike_subtitle(tele, None)
        assert text.endswith("and the corners fixed")
        assert not re.search(r"\d", text) and not (count_words & set(re.findall(r"\w+", text)))
    # the source no longer carries the fallback word either
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "pyCamSet" / "gui"
              / "education_phase4.py").read_text(encoding="utf-8")
    assert 'return "three", None' not in source


def test_look_alike_subtitle_is_a_plain_sentence_with_the_run_count():
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_phase4 import look_alike_subtitle
    assert (look_alike_subtitle(True, 3) == "turned or scaled: no photo changes; one "
            "photo's pose and 3 corners fixed")
    assert look_alike_subtitle(False, 3) == ("moved, turned or scaled: no photo changes; one "
                                             "pose and 3 corners fixed")
    assert len(look_alike_subtitle(False, 3)) <= 73
    assert "a pose and" not in look_alike_subtitle(True, 3)
    # the subtitle must fit one line at 1280 px at L1: keep it to the measured length
    assert len(look_alike_subtitle(True, 3)) <= 73


@pytest.mark.gui
def test_look_alike_panel_says_the_solver_fixes_numbers_not_a_held_photo(tmp_path, monkeypatch):
    _app()
    view, visual = _view(tmp_path, monkeypatch)
    try:
        assert len(visual.gauge_indices) == 3  # the copy's count must come from the run
        def look_alike_subtitle(letter_pattern):
            ax = next(ax for ax in view.stage_view.figure().axes
                      if re.match(letter_pattern, ax.get_title(loc="left")))
            sub = next(t for t in ax.texts if t.get_gid() == "edu:subtitle")
            assert sub.get_text() == sub._edu_full, "the subtitle is elided at 1380 px"
            return sub.get_text()

        view.stage_text(3, level=1)
        subtitle = look_alike_subtitle(r"^C  Some changes look identical")
        assert "turned or scaled: no photo changes" in subtitle  # telecentric: no shift claimed
        assert "one photo's pose and 3 corners fixed" in subtitle
        for old in ("one photo is fixed", "target move", "held", "gauge", "drift", "moved"):
            assert old not in subtitle
        view.stage_text(3, level=3)
        compact = look_alike_subtitle(r"^[A-Z]  Some changes look identical")
        assert "one photo's pose and 3 corners fixed" in compact
        rows = {row[0]: row for row in view.numbers_rows()}
        moving = rows["Moving everything together"]
        assert "one photo's pose (6 numbers)" in moving[2]
        assert "the printed positions of three corners (7 numbers" in moving[2]
        assert "sideways shift: nearly exact" in moving[1]  # the synthetic rig is telecentric
        assert "lenses 1.3× weaker" in moving[1] and "30 %" not in moving[1]
        # the sideways-shift driver is the telecentricity term times depth, not "bending"
        assert "ε times the target's depth" in moving[2] and "bend" not in moving[2]
        nullity = rows["Null directions"]
        assert "weakly fixed lens or pose" in nullity[1] and "one-detection corner" in nullity[1]
        assert "one camera in one photo may slide along its ray" in nullity[2]
        assert "rig-wide" not in nullity[1] and "single-corner" not in nullity[1]
        view.stage_text(1, level=3)  # the solver gauge row is on the shape stage
        solver = {row[0]: row for row in view.numbers_rows()}["Solver gauge (during the solve)"]
        assert "set to the printed values and removed from the unknowns" in solver[2]
        assert "held to stop" not in solver[2]
    finally:
        view.close()


@pytest.mark.gui
def test_pinhole_sideways_shift_is_exact_in_the_numbers_row(tmp_path, monkeypatch):
    _app()
    view, _visual = _view(tmp_path, monkeypatch, telecentric=False)
    try:
        view.stage_text(3, level=1)
        ax = next(ax for ax in view.stage_view.figure().axes
                  if ax.get_title(loc="left").startswith("C  Some changes look identical"))
        sub = next(t for t in ax.texts if t.get_gid() == "edu:subtitle")
        assert sub.get_text() == sub._edu_full == (
            "moved, turned or scaled: no photo changes; one pose and 3 corners fixed")
        view.stage_text(3, level=3)
        moving = {row[0]: row for row in view.numbers_rows()}["Moving everything together"]
        assert moving[1].endswith("sideways shift: exact")
        assert "nearly" not in moving[1] and "an exact move for this lens model" in moving[2]
    finally:
        view.close()


@pytest.mark.gui
def test_stage4_terms_list_reaches_the_gauge_terms_and_does_not_leak(tmp_path, monkeypatch):
    _app()
    view, _visual = _view(tmp_path, monkeypatch)
    try:
        view.stage_text(3, level=2)
        assert {"gauge freedom", "null space"} <= set(view.stage_view.glossary.terms())
        view.stage_text(1, level=2)
        assert "gauge freedom" not in view.stage_view.glossary.terms()
    finally:
        view.close()


def test_gauge_and_null_space_entries_separate_symmetries_from_loose_directions():
    from pyCamSet.gui.education_style import GLOSSARY
    freedom = GLOSSARY["gauge freedom"]
    assert "symmetry" in freedom and "one photo's pose" in freedom
    assert "must hold" not in freedom and "whole rig" in freedom
    assert "a few numbers" not in freedom
    null = GLOSSARY["null space"]
    assert "Symmetries are not counted" in null and "may slide along its ray" in null
    # P1-1: only a corner detected by one camera in one photo may slide (two cameras fix it)
    assert "detected by one camera in one photo, which may slide along its ray" in null
    assert "a corner seen in one photo" not in null
    # P1-2: the lens/pose count is named for what it counts, not as a rig motion
    assert "other weakly fixed lens or pose directions" in null
    assert "loose" not in null and "whole rig" not in null
    assert "cannot pin down" not in null


def test_held_back_reasons_name_the_fixed_numbers_not_a_held_photo():
    from pyCamSet.workflow.education_visual_p4 import plain_blocking
    what, _todo = plain_blocking("target gauge does not fix three non-collinear points")
    assert what == ("The solver could not fix three corners, so the shape's own symmetries "
                    "stayed in the solve.")
    assert "non-collinear" not in what and "hold" not in what and "drift" not in what
    what, _todo = plain_blocking("pose gauge has no fixed reference pose")
    assert "pose was fixed" in what and "drift" not in what and "held" not in what


def test_loose_for_the_whole_rig_and_seen_once_wording_is_gone_from_the_owned_files():
    """P1-1 / P1-2 regression: the retired phrases must not come back in the learner-visible
    gauge copy (the lens/pose count is named for what it counts)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "pyCamSet"
    owned = [root / "gui" / "education_phase4.py", root / "gui" / "education_style.py",
             root / "workflow" / "education_visual_p4.py"]
    text = "\n".join(p.read_text(encoding="utf-8") for p in owned)
    for retired in ("loose for", "whole rig\n", "rig-wide", "corners seen once",
                    "once-seen corner", "seen in one photo only", "a corner seen in one photo",
                    "single-corner", "nothing moves without changing a photo",
                    "could drift unseen", "30 % weaker"):
        assert retired not in text, retired


def test_phase3_and_phase4_describe_the_fixed_pose_the_same_way():
    from pathlib import Path
    gui = Path(__file__).resolve().parents[1] / "pyCamSet" / "gui"
    phase3 = (gui / "education_phase3.py").read_text(encoding="utf-8")
    phase4 = (gui / "education_phase4.py").read_text(encoding="utf-8")
    for old in ("(held fixed)", "held fixed so", "cannot drift"):
        assert old not in phase3
    assert "removing the rig's overall position and turn" in phase3
    assert "known size already fixes" in phase3 and "the scale" in phase3
    assert "Reference photo and position origin" not in phase4
    assert "Which photo sets the position origin" not in phase4
