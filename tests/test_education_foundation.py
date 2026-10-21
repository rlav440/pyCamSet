"""Contract tests for the education visual foundation (style, stage view, 3D scene, player).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["PYCAMSET_EDU_3D"] = "static"


def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _settle(app, rounds: int = 6) -> None:
    for _ in range(rounds):
        app.processEvents()


# --------------------------------------------------------------------------- style


def test_camera_palette_is_fixed_then_tab10():
    from pyCamSet.gui import education_style as es
    assert es.CAMERA_PALETTE == ("#D4B483", "#8B5A3C", "#D98BA6", "#F28C28",
                                 "#A9B8C9", "#3B4252", "#A48BBF", "#9C9283")
    assert [es.camera_colour(i) for i in range(8)] == list(es.CAMERA_PALETTE)
    assert es.camera_colour(8) == "#1F77B4"
    assert es.camera_colour(17) == es.camera_colour(17)  # stable
    assert es.camera_colour(18) == es.camera_colour(8)    # tab10 cycles
    assert [es.camera_badge_label(i) for i in (0, 3, 7)] == ["1", "4", "8"]
    with pytest.raises(ValueError):
        es.camera_colour(-1)
    # Badge numbers stay readable on every camera colour.
    for i in range(18):
        assert es.contrast(es.badge_text_colour(i), es.camera_colour(i)) >= 2.2


def test_stage_tokens_come_from_the_gui_themes():
    from pyCamSet.gui import education_style as es
    from pyCamSet.gui.theme import THEME_TOKENS
    assert set(es.theme_names()) == set(THEME_TOKENS) >= {"Light", "Dark", "Sepia"}
    for name, tok in THEME_TOKENS.items():
        t = es.stage_tokens(name)
        assert t["stage"] == tok["surface"] and t["text"] == tok["text"]
        assert t["ui_accent"] == tok["accent"] and t["ui_warn"] == tok["warning"]
        # The theme's own colour is kept whenever it passes the camera-distance rule.
        for role, token in (("accent", "accent"), ("warn", "warning"), ("bad", "danger"),
                            ("good", "success")):
            passes = (min(es.colour_distance(tok[token], c) for c in es.CAMERA_PALETTE)
                      >= es.MIN_CAMERA_DISTANCE
                      and es.contrast(tok[token], tok["surface"]) >= es.MIN_STAGE_CONTRAST)
            assert (t[role] == tok[token]) == passes, (name, role)
        rc = es.figure_style(name)
        assert rc["figure.facecolor"] == tok["surface"] == rc["axes.facecolor"]
        assert rc["figure.constrained_layout.use"] is True
        assert rc["axes.spines.top"] is False and rc["axes.spines.right"] is False
        assert rc["axes.titlesize"] == 13 and rc["xtick.labelsize"] == 9
    assert es.resolve_theme("sepia") == "Sepia" and es.resolve_theme("nonsense") == "Light"


@pytest.mark.parametrize("theme", ["Light", "Dark", "Sepia"])
def test_figure_helpers_draw_without_overlap(theme):
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_style as es
    rng = np.random.default_rng(0)
    with matplotlib.rc_context(es.figure_style(theme)):
        fig = Figure(figsize=(10, 6), dpi=80)
        FigureCanvasAgg(fig)
        a, b, c = fig.subplots(1, 3)
        image = rng.integers(0, 255, (120, 160), dtype=np.uint8)
        pts = rng.uniform(10, 110, (20, 2))
        es.image_with_points(a, image, pts, rejected=pts[:3], camera=2)
        es.panel_title(a, "A", "Camera image", "camera 3, 20 corners")
        values = {k: rng.gamma(2, 0.2, 30) for k in range(4)}
        values[1] = np.r_[values[1], 9.0]
        info = es.strip_plot_by_camera(b, values, highlight=(0, 1))
        es.panel_title(b, "B", "Error per image", "one dot per image")
        flagged = es.bars_by_camera(c, [50, 20, 60, 45], threshold=30, threshold_label="keep")
        fig.canvas.draw()
    assert es.axes_theme(a) == theme
    assert 1 in info["outliers"] and info["outliers"][1] == [9.0]
    assert info["cap"] < 9.0
    assert flagged == [1]
    # Title and subtitle sit above the axes, not on the data.
    renderer = fig.canvas.get_renderer()
    sub = [t for t in a.texts if t.get_gid() == "edu:subtitle"][0]
    assert sub.get_window_extent(renderer).y0 >= a.get_window_extent(renderer).y1 - 1
    assert a.title.get_window_extent(renderer).y0 > sub.get_window_extent(renderer).y0
    # UX-1: nothing clipped by the figure edge, no text on text.
    es.align_panel_titles(fig)
    assert es.text_collisions(fig) == []


def test_residual_arrows_returns_mappable_for_px_colourbar():
    import matplotlib
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_style as es
    with matplotlib.rc_context(es.figure_style("Dark")):
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot()
        obs = np.array([[10.0, 10.0], [50.0, 40.0]])
        q = es.residual_arrows(ax, None, obs, obs + [[0.5, 0], [0, 1.0]], 20,
                               image_size=(100, 80))
        bar = es.colourbar_px(fig, q, ax)
        fig.canvas.draw()
    assert bar.ax.get_title(loc="left") == "px"
    assert np.allclose(q.get_array(), [0.5, 1.0])


# --------------------------------------------------------------------------- stepper


@pytest.mark.gui
def test_stage_stepper_keyboard_and_clicks():
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from pyCamSet.gui.education_stage import StageStepper, _StagePill
    _app()
    stepper = StageStepper(["Photos", "Detector", "Markers", "Final"])
    seen = []
    stepper.currentChanged.connect(seen.append)
    assert stepper.currentIndex() == 0 and stepper.count() == 4
    QTest.keyClick(stepper, Qt.Key.Key_Right)
    QTest.keyClick(stepper, Qt.Key.Key_Right)
    assert stepper.currentIndex() == 2
    QTest.keyClick(stepper, Qt.Key.Key_Left)
    QTest.keyClick(stepper, Qt.Key.Key_End)
    QTest.keyClick(stepper, Qt.Key.Key_Right)  # clamps at the end
    assert stepper.currentIndex() == 3
    QTest.keyClick(stepper, Qt.Key.Key_Home)
    assert stepper.currentIndex() == 0
    pills = stepper.findChildren(_StagePill)
    pills[2].click()
    assert stepper.currentIndex() == 2
    assert seen == [1, 2, 1, 3, 0, 2]
    assert [p.state for p in pills] == ["done", "done", "current", "upcoming"]
    assert "Stage 3 of 4" in pills[2].accessibleName()


# --------------------------------------------------------------------------- stage view


def _populated_view(level: int, with_player: bool):
    from PySide6.QtWidgets import QComboBox
    from pyCamSet.gui.education_stage import Chip, EducationStageView
    view = EducationStageView(["Photos", "What the detector sees", "Finding markers",
                               "Final", "Decision"])
    combo = QComboBox()
    combo.addItems([f"Camera {i}" for i in range(1, 9)])
    view.add_picker("Camera", combo)
    image = QComboBox()
    image.addItems(["Image 1", "Image 2"])
    view.add_picker("Image", image)
    view.set_level(level)
    view.draw_figure(lambda fig, theme: fig.add_subplot().plot([0, 1], [1, 0]))
    view.set_caption("A caption sentence that explains the picture. " * (level + 1))
    view.set_chips([Chip("63", "corners"), Chip("0.41 px", "RMS", tone="good"),
                    Chip("7 / 8", "cameras", tone="warn"), Chip("12", "images", camera=3)])
    view.set_numbers(["Camera", "Points"], [(str(i), 10 * i) for i in range(1, 30)])
    player = None
    if with_player:
        from pyCamSet.gui.education_scene3d import Scene3D, ScenePlayer
        scene = Scene3D()
        scene.add_camera("c0", 0, np.eye(3), [0, 0, 2])
        frames = [{"cameras": {"c0": {"R": np.eye(3), "t": [0, 0, 2 + k]}}} for k in range(3)]
        player = ScenePlayer(scene, frames, reduce_motion=True)
        view.set_visual(scene)
        view.set_controls(player)
    return view, player


@pytest.mark.gui
@pytest.mark.parametrize("size", [(1400, 900), (1280, 800)])
@pytest.mark.parametrize("level,with_player", [(1, False), (3, False), (3, True)])
def test_stage_view_fits_without_scroll_and_visual_dominates(size, level, with_player):
    from PySide6.QtWidgets import QFrame, QScrollArea
    app = _app()
    view, _player = _populated_view(level, with_player)
    # As the education dialog hosts phase views: inside a resizable scroll area.
    scroll = QScrollArea()
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidgetResizable(True)
    scroll.setWidget(view)
    scroll.resize(*size)
    scroll.show()
    _settle(app)
    assert scroll.verticalScrollBar().maximum() == 0
    assert view.height() <= size[1]
    assert view.visual_panel.height() >= 0.65 * size[1], (
        view.visual_panel.height(), size)
    # Caption, chips and Numbers button do not overlap each other or the visual area.
    caption = view.caption.geometry()
    chips = view.chips.geometry()
    assert caption.top() >= view.visual_panel.geometry().bottom()
    assert caption.right() < chips.left()
    assert view.caption.height() >= view.caption.heightForWidth(view.caption.width())
    assert not view.numbers.isVisible()


@pytest.mark.gui
def test_numbers_drawer_toggles():
    app = _app()
    view, _ = _populated_view(1, False)
    view.resize(1400, 900)
    view.show()
    _settle(app)
    before = view.visual_panel.height()
    assert view.numbers.button.isEnabled() and view.numbers.button.text() == "Numbers (29)"
    view.numbers.button.click()
    _settle(app)
    assert view.numbers.is_open() and view.numbers.table.rowCount() == 29
    assert view.visual_panel.height() < before
    view.numbers.button.click()
    _settle(app)
    assert not view.numbers.is_open()
    assert view.visual_panel.height() == before
    view.set_numbers(["A"], [])
    assert not view.numbers.button.isEnabled()


@pytest.mark.gui
def test_stage_view_redraws_on_theme_switch():
    from pyCamSet.gui.education_style import stage_tokens
    from pyCamSet.gui.theme import THEME_TOKENS
    app = _app()
    calls = []
    from pyCamSet.gui.education_stage import EducationStageView
    view = EducationStageView(["One", "Two"])
    view.draw_figure(lambda fig, theme: calls.append(theme) or fig.add_subplot())
    for theme in ("Dark", "Sepia"):
        app.setProperty("pycamsetTheme", theme)
        _settle(app, 2)
        assert calls[-1] == theme
        face = view.figure().get_facecolor()
        from matplotlib.colors import to_hex
        assert to_hex(face) == stage_tokens(theme)["stage"] == THEME_TOKENS[theme]["surface"]
    levels = []
    view.levelChanged.connect(levels.append)
    view.set_level(3)
    assert levels == [3] and view.level() == 3


# --------------------------------------------------------------------------- 3D


@pytest.mark.gui
def test_scene3d_static_fallback_builds_and_updates_in_place():
    from pyCamSet.gui.education_scene3d import Scene3D, camera_centre, scene_backend
    _app()
    assert scene_backend() == "static"
    scene = Scene3D(theme="Dark")
    assert scene.backend == "static" and scene.figure is not None
    R = np.eye(3)
    scene.add_camera("cam0", 0, R, [0, 0, 2], scale=0.3)
    scene.add_camera("cam1", 1, R, [0.5, 0, 2], scale=0.3, telecentric=True, label="(0.4)")
    board = scene.add_board(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]]),
                            scalars=0.3, clim=(0, 1))
    pts = scene.add_points(np.random.default_rng(0).normal(size=(20, 3)),
                           np.linspace(0, 1, 20), scalar_bar="px")
    rays = scene.add_rays(np.zeros((3, 3)), np.eye(3))
    assert scene.names() == ["cam0", "cam1", board, pts, rays]
    item = scene._impl.items["cam0"]
    surf = item["surf"]
    scene.update_camera("cam0", R, [1.0, 0, 2])
    assert scene._impl.items["cam0"]["surf"] is surf          # moved, not rebuilt
    assert np.allclose(item["verts"][0], camera_centre(R, [1.0, 0, 2]))
    scene.update_board(board, np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1.0]]), 0.9)
    scene.update_points(pts, np.zeros((20, 3)), np.ones(20))
    scene.update_rays(rays, np.zeros((3, 3)), 2 * np.eye(3))
    scene.set_view(azimuth=30, elevation=20)
    scene.set_theme("Light")
    scene.figure.canvas.draw()


@pytest.mark.gui
@pytest.mark.parametrize("telecentric", [False, True])
def test_scene3d_static_camera_limits_come_from_the_glyph_alone(telecentric):
    """A camera glyph's polygons have different vertex counts; matplotlib pads them into an
    uninitialised array and autoscaled from the padding too, so the limits depended on what
    the heap held (a full-suite run failed with "Axis limits cannot be NaN or Inf"). The data
    limits must be the glyph's own, whatever memory was freed just before."""
    from pyCamSet.gui.education_scene3d import Scene3D, camera_glyph
    _app()
    scene = Scene3D(theme="Light")
    R, t, scale = np.eye(3), np.array([0.01, -0.02, 0.03]), 0.005
    verts = camera_glyph(R, t, scale, telecentric)[0]
    for _ in range(3):
        poison = np.full((40, 12, 3), 1e300)  # leave huge values in freshly freed memory
        del poison
        scene.clear()
        scene.add_camera("cam0", 0, R, t, scale=scale, telecentric=telecentric, axis=False)
        ax = scene._impl.ax
        for got, k in ((ax.xy_dataLim.intervalx, 0), (ax.xy_dataLim.intervaly, 1),
                       (ax.zz_dataLim.intervalx, 2)):
            assert np.allclose(got, [verts[:, k].min(), verts[:, k].max()]), (k, got)
        assert np.isfinite([*ax.get_xlim(), *ax.get_ylim(), *ax.get_zlim()]).all()


def test_camera_glyph_shapes_and_pose_conventions():
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_scene3d import (
        as_pose, axis_length_to, camera_axis_segment, camera_centre, camera_glyph,
    )
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    t = np.array([1.0, 2.0, 3.0])
    assert np.allclose(camera_centre(R, t), -R.T @ t)
    E = np.eye(4)
    E[:3, :3], E[:3, 3] = R, t
    assert np.allclose(as_pose(E)[1], t)
    assert np.allclose(as_pose(np.zeros(3))[0], np.eye(3))
    verts, faces, edges = camera_glyph(R, t, 1.0)
    c = camera_centre(R, t)
    assert np.allclose(verts[0], c)
    z = R[2]
    # Pinhole: a frustum whose four far corners sit 1 scale ahead, plus a body behind.
    ahead = (verts - c) @ z
    assert np.isclose(ahead, 1.0).sum() == 4 and (ahead < -0.3).sum() >= 4
    tele, tele_faces, tele_edges = camera_glyph(R, t, 1.0, telecentric=True)
    assert np.allclose(tele[0], c)
    # Telecentric: a slim barrel (narrower than the frustum) and four parallel ray lines.
    radial = tele - c - np.outer((tele - c) @ z, z)
    lens = ((tele - c) @ z) > 0
    assert np.linalg.norm(radial[lens], axis=1).max() < 0.2
    rays = [(i, j) for i, j in tele_edges if (tele[j] - c) @ z > 1.0]
    assert len(rays) == 4
    for i, j in rays:
        d = (tele[j] - tele[i]) / np.linalg.norm(tele[j] - tele[i])
        assert np.allclose(d, z)
    # Neither glyph is a plain box: both have more than 8 vertices.
    assert len(verts) > 8 and len(tele) > 8
    seg = camera_axis_segment(R, t, 2.0)
    assert np.allclose(seg[0], c) and np.allclose(seg[1], c + 2.0 * z)
    assert axis_length_to(R, t, c + 3 * z + R[0]) == pytest.approx(3.0)


def test_interpolate_frames_slerps_cameras_and_lerps_points():
    pytest.importorskip("PySide6")
    from scipy.spatial.transform import Rotation
    from pyCamSet.gui.education_scene3d import camera_centre, interpolate_frames
    Ra = np.eye(3)
    Rb = Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
    a = {"cameras": {"c": {"R": Ra, "t": -Ra @ [0, 0, 0.0]}},
         "points": {"p": {"xyz": np.zeros((2, 3)), "scalars": np.zeros(2)}}}
    b = {"cameras": {"c": {"R": Rb, "t": -Rb @ [2.0, 0, 0]}},
         "points": {"p": {"xyz": np.ones((2, 3)), "scalars": np.ones(2)}}}
    mid = interpolate_frames(a, b, 0.5)
    cam = mid["cameras"]["c"]
    assert np.allclose(camera_centre(cam["R"], cam["t"]), [1.0, 0, 0])
    angle = np.degrees(np.linalg.norm(Rotation.from_matrix(cam["R"]).as_rotvec()))
    assert angle == pytest.approx(45.0)
    assert np.allclose(mid["points"]["p"]["xyz"], 0.5)
    assert np.allclose(mid["points"]["p"]["scalars"], 0.5)
    assert np.allclose(interpolate_frames(a, b, 1.0)["points"]["p"]["xyz"], 1.0)


@pytest.mark.gui
def test_scene_player_interpolates_and_syncs_callbacks():
    from pyCamSet.gui.education_scene3d import Scene3D, ScenePlayer, camera_centre
    app = _app()
    scene = Scene3D()
    scene.add_camera("c", 0, np.eye(3), [0, 0, 0])
    scene.add_points(np.zeros((4, 3)), np.zeros(4), name="pts")
    frames = [{"cameras": {"c": {"R": np.eye(3), "t": [-float(k), 0, 0]}},
               "points": {"pts": {"xyz": np.full((4, 3), float(k)),
                                  "scalars": np.full(4, float(k))}}}
              for k in range(4)]
    seen = []
    steps = []
    player = ScenePlayer(scene, frames, step_numbers=[2, 5, 9], total_steps=9, substeps=4,
                         on_frame=lambda pos, frame: seen.append(pos), reduce_motion=False)
    player.stepReached.connect(steps.append)
    assert player.position == 0.0 and seen == [0.0]
    frame = player.show_position(1.5)
    cam = frame["cameras"]["c"]
    assert np.allclose(camera_centre(cam["R"], cam["t"]), [1.5, 0, 0])
    assert np.allclose(scene._impl.items["pts"]["verts"], 1.5)
    # Stepping forward tweens through fractional positions to the next recorded frame.
    player.go_to(1, animate=False)
    assert player.position == 1.0 and player.controls.current_step.solver_step == 2
    seen.clear()
    player.controls.step_forward()
    from PySide6.QtCore import QElapsedTimer
    timer = QElapsedTimer()
    timer.start()
    while player.position != 2.0 and timer.elapsed() < 5000:
        app.processEvents()
    assert player.position == 2.0
    assert seen[-1] == 2.0 and any(1.0 < p < 2.0 for p in seen)
    assert steps[-1] == 2
    # With reduce motion the scene jumps straight to the step.
    player.controls.set_reduce_motion(True)
    seen.clear()
    player.go_to(3)
    assert seen == [3.0] and player.position == 3.0
    player.stop_for_context_change()
    with pytest.raises(ValueError):
        ScenePlayer(scene, frames[:1])


# --------------------------------------------------------------------------- round-1 shared layer


def test_figure_colours_never_collide_with_camera_colours():
    """UX-8: accent, link, warn, bad and good are far from every camera colour and readable."""
    from pyCamSet.gui import education_style as es
    for theme in ("Light", "Dark", "Sepia"):
        t = es.stage_tokens(theme)
        for role in ("accent", "link", "warn", "bad", "good"):
            distance = min(es.colour_distance(t[role], c) for c in es.CAMERA_PALETTE)
            assert distance >= es.MIN_CAMERA_DISTANCE, (theme, role, t[role], distance)
            assert es.contrast(t[role], t["stage"]) >= es.MIN_STAGE_CONTRAST, (theme, role)
        # The roles stay distinguishable from each other too.
        roles = [t[r] for r in ("accent", "warn", "bad", "good")]
        assert min(es.colour_distance(a, b) for i, a in enumerate(roles)
                   for b in roles[i + 1:]) >= 20, theme
    # The two clashes the round-1 review found are gone.
    assert es.stage_tokens("Sepia")["accent"] != "#8c5a2b"
    assert es.stage_tokens("Light")["warn"] != "#b45309"


def test_camera_outline_rule_is_one_rule_for_every_mark():
    """UX-7 / UX2-1: a low-contrast camera is drawn in ONE substitute fill on that theme for
    every kind of mark (never hollow, never white); only its badge gets the outline ring."""
    import matplotlib
    from matplotlib.colors import to_hex
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_style as es
    assert [i + 1 for i in range(8) if es.camera_needs_outline(i, "Dark")] == [6]
    assert [i + 1 for i in range(8) if es.camera_needs_outline(i, "Light")] == [1, 5]
    assert [i + 1 for i in range(8) if es.camera_needs_outline(i, "Sepia")] == [1, 5]
    for theme in ("Light", "Dark", "Sepia"):
        stage = es.stage_tokens(theme)["stage"]
        fills = [es.camera_fill(i, theme) for i in range(8)]
        for i in range(8):
            needs = (es.contrast(es.camera_colour(i), stage) < es.CAMERA_OUTLINE_CONTRAST)
            assert es.camera_needs_outline(i, theme) == needs
            assert (fills[i] != es.camera_colour(i)) == needs
            assert es.contrast(fills[i], stage) >= es.CAMERA_OUTLINE_CONTRAST, (theme, i)
            assert es.camera_colour(i, theme) == fills[i]
            # A substitute keeps clear of every other camera's colour on that theme.
            others = [fills[j] for j in range(8) if j != i] + \
                [es.camera_colour(j) for j in range(8) if j != i]
            if needs:
                assert min(es.colour_distance(fills[i], c) for c in others) >= 25, (theme, i)
            for kind in ("patch", "dot", "line"):
                kw = es.camera_mark_kwargs(i, theme, kind)
                colour = kw.get("facecolor", kw.get("color"))
                assert colour == fills[i]
                assert kw.get("edgecolor", kw.get("edgecolors", "none")) == "none"
            badge = es.camera_mark_kwargs(i, theme, "badge")
            assert badge["facecolor"] == fills[i]
            assert (badge["edgecolor"] == es.stage_tokens(theme)["outline"]) == needs
            assert es.camera_line_effects(i, theme) == []
            assert es.contrast(es.badge_text_colour(i, theme), fills[i]) >= 3.0
    with matplotlib.rc_context(es.figure_style("Dark")):
        fig = Figure()
        ax = fig.add_subplot()
        es.bars_by_camera(ax, [1, 2, 3, 4, 5, 6, 7, 8])
        bar6 = ax.patches[5]
        assert to_hex(bar6.get_facecolor()) == es.camera_fill(5, "Dark").lower()
        assert bar6.get_linewidth() == 0  # no outline on bars
        badge6 = ax.get_xticklabels()[5].get_bbox_patch()
        assert to_hex(badge6.get_facecolor()) == es.camera_fill(5, "Dark").lower()
        assert to_hex(badge6.get_edgecolor()) == es.stage_tokens("Dark")["outline"].lower()


def test_error_vocabulary_scale_and_dark_colour_map():
    """ED-19 / UX-6: one name, one log colour scale, a visible low end on Dark."""
    import matplotlib
    from matplotlib.colors import to_hex
    from pyCamSet.gui import education_style as es
    assert es.error_label() == "mean reprojection error (px)"
    assert es.error_label("rms", per="photo") == "RMS reprojection error per photo (px)"
    assert es.error_label(None, short=True) == "error (px)"
    scale = es.error_scale("Dark")
    assert scale.clim == es.ERROR_SCALE_PX == (0.1, 100.0)
    assert scale.cmap == es.error_cmap("Dark") != "viridis"
    assert es.error_cmap("Light") == es.error_cmap("Sepia") == "viridis"
    stage = es.stage_tokens("Dark")["stage"]
    low = to_hex(matplotlib.colormaps[scale.cmap](0.0))
    plain = to_hex(matplotlib.colormaps["viridis"](0.0))
    assert es.contrast(low, stage) > es.contrast(plain, stage)
    # Log scale: one decade is the same colour step anywhere; values above the cap are
    # flagged; 1 px sits a third of the way up.
    n = scale.norm()
    assert n(1.0) == pytest.approx(1 / 3) and n(10.0) == pytest.approx(2 / 3)
    assert list(scale.over_cap([50, 150])) == [False, True]
    assert scale.scene_kwargs()["log_scale"] is True
    assert es.value_dot_kwargs("Dark")["edgecolors"] == es.stage_tokens("Dark")["point_edge"]


def test_number_photo_and_camera_wording():
    """SCI-8: photos are numbered from 1 like cameras; one number format."""
    from pyCamSet.gui import education_style as es
    assert es.PHOTO_NUMBER_BASE == 1
    assert es.photo_label(16) == "Photo 17" and es.photo_number(0) == 1
    assert es.photos_label([25, 29, 69, 83, 89]) == "Photos 26, 30, 70, 84 and 90"
    assert es.photos_label([0, 1, 2, 5]) == "Photos 1–3 and 6"
    assert es.photos_label([3]) == "Photo 4"
    assert es.camera_title(0, "cam1_FrontLeft_907T") == "Camera 1 · cam1_FrontLeft_907T"
    assert es.camera_title(1, "2") == es.camera_title(1, "cam2") == "Camera 2"
    assert [es.format_number(v) for v in (0.297, 3.2954, 121.904, 15447)] == [
        "0.30", "3.30", "122", "15,447"]
    assert es.format_px(0.3) == "0.30 px"
    assert es.format_change(3.2954, 0.297) == "3.30 → 0.30 px"
    assert es.format_um(0.1723) == "172 µm" and es.format_percent(0.774) == "77 %"
    assert es.format_count(1, "photo") == "1 photo" and es.format_count(5, "photo") == "5 photos"


def test_glossary_and_caption_limits():
    from pyCamSet.gui import education_style as es
    assert "pixels" in es.glossary("Reprojection Error")
    assert es.glossary("no such term") == ""
    assert es.glossary_terms_in("The Jacobian and the cost; bundle adjustment.") == [
        "Jacobian", "cost", "bundle adjustment"]
    assert es.count_sentences("Error 3.30 px, e.g. on face 1. Two! Three?") == 3
    assert es.caption_problems("One. Two. Three.", 1) == ["3 sentences at L1 (limit 2)"]
    assert es.caption_problems("One. Two. Three.", 2) == []


@pytest.mark.parametrize("theme", ["Light", "Dark"])
def test_panel_titles_share_a_baseline_and_long_ones_are_elided(theme):
    """UX-13 / UX-1: titles of one row on one baseline; overflow ends in an ellipsis."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_style as es
    rng = np.random.default_rng(1)
    with matplotlib.rc_context(es.figure_style(theme)):
        fig = Figure(figsize=(7, 4), dpi=100)
        FigureCanvasAgg(fig)
        es.align_panel_titles(fig)
        a, b, c = fig.subplots(1, 3)
        a.imshow(rng.random((40, 200)))   # wide image: its axes shrink vertically
        es.panel_title(a, "A", "A rather long panel title that cannot fit",
                       "and an even longer subtitle that would run into panel B")
        b.imshow(rng.random((200, 90)))
        es.panel_title(b, "B", "Tall", "sub")
        es.bars_by_camera(c, [1.51, 1.33, 1.31, 2.9, 3.5, 3.7, 3.5, 1.2])
        es.panel_title(c, "C", "Eight cameras", "narrow panel")
        issues = es.text_collisions(fig)
    renderer = fig.canvas.get_renderer()
    tops = [ax._left_title.get_window_extent(renderer).y1 for ax in (a, b, c)]
    assert max(tops) - min(tops) < 1.5, tops
    assert a._left_title.get_text().split("\n")[0].endswith("…")
    sub = [t for t in a.texts if t.get_gid() == "edu:subtitle"][0]
    assert sub.get_text().endswith("…")
    assert sub.get_window_extent(renderer).x1 < b.get_position().x0 * fig.bbox.width
    assert issues == [], issues


def test_text_collision_checker_finds_clipped_and_overlapping_text():
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_style as es
    with matplotlib.rc_context(es.figure_style("Light")):
        fig = Figure(figsize=(4, 3), dpi=100)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ax.plot([0, 1], [0, 1])
        ax.text(0.5, 0.5, "first label", transform=ax.transAxes)
        ax.text(0.52, 0.5, "second label", transform=ax.transAxes)
        ax.text(0.98, 0.9, "runs off the right edge of the figure", transform=ax.transAxes,
                in_layout=False)
        issues = es.text_collisions(fig)
    problems = {(i["problem"], i["text"]) for i in issues}
    assert ("overlap", "first label") in problems
    assert ("outside figure", "runs off the right edge of the figure") in problems


def test_layout_badges_separates_and_keeps_inside():
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_scene3d import layout_badges
    pts = np.array([[100.0, 100.0], [102.0, 101.0], [100.0, 100.0], [-30.0, 50.0],
                    [500.0, 390.0]])
    out = layout_badges(pts, 10.0, (0, 0, 400, 300), label_widths=[0, 0, 0, 0, 40])
    for i in range(len(out)):
        x, y = out[i]
        assert 10 <= x <= 400 - 10 and 10 <= y <= 300 - 10
        for j in range(i + 1, len(out)):
            assert np.hypot(*(out[i] - out[j])) >= 20 - 1e-6
    assert out[4, 0] <= 400 - 10 - 40


def test_choose_view_separates_cameras_and_keeps_selected_unblocked():
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_scene3d import _screen_basis, _view_direction, choose_view
    # Cameras 0 and 1 line up when seen along -y; camera 1 is the selected one.
    centres = np.array([[0, -2.0, 0.3], [0, -4.0, 0.3], [2.0, 0, 0.3], [-2.0, 0.5, 0.3]])
    az, el = choose_view(centres, np.zeros(3), selected=1, glyph_size=0.3)
    d = _view_direction(az, el, (0, 0, 1))
    right, up = _screen_basis(d, (0, 0, 1))
    xy = np.stack([centres @ right, centres @ up], axis=1)
    gaps = [np.hypot(*(xy[i] - xy[j])) for i in range(4) for j in range(i + 1, 4)]
    assert min(gaps) > 0.5
    # Nobody nearer the viewer sits on top of the selected camera.
    near = np.hypot(*(xy - xy[1]).T) < 0.3
    assert not np.any(near & (centres @ d > centres[1] @ d))




@pytest.mark.gui
def test_camera_and_photo_pickers_and_glossary_button():
    from PySide6.QtWidgets import QFrame
    from pyCamSet.gui.education_stage import (
        Chip, EducationStageView, camera_badge_pixmap, camera_picker, photo_picker,
    )
    app = _app()
    combo = camera_picker({0: "cam1_FrontLeft_907T", 5: "6"}, current=5)
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "Camera 1 · cam1_FrontLeft_907T", "Camera 6"]
    assert combo.currentData() == 5 and not combo.itemIcon(0).isNull()
    assert camera_badge_pixmap(5, 18, "Dark").width() > 0
    photos = photo_picker([16, 17], details={16: "1.87 px per photo"})
    assert photos.itemText(0) == "Photo 17 · 1.87 px per photo" and photos.itemData(0) == 16
    view = EducationStageView(["One", "Two"])
    view.add_picker("", combo)
    view.resize(1200, 800)
    view.show()
    view.set_caption("Each dot is a corner; the bundle adjustment lowers the cost.")
    view.set_chips([Chip("0.30 px", "mean reprojection error")])
    _settle(app)
    assert view.glossary.isHidden()          # L1: no glossary button
    view.set_level(2)
    _settle(app)
    assert view.glossary.isVisible()
    assert view.glossary.terms() == ["bundle adjustment", "cost", "reprojection error"]
    chip = [f for f in view.chips.findChildren(QFrame) if f.objectName() == "eduChip"][0]
    assert "pixels" in chip.toolTip()
    view.set_glossary([])
    assert view.glossary.isHidden()
    view.set_caption("One. Two. Three. Four.")
    assert view.caption_issues == ["4 sentences at L2 (limit 3)"]
    view.close()


# --------------------------------------------------------------------------- round-2 shared layer


def test_text_checker_ignores_switched_off_axes_and_needs_no_canvas():
    import matplotlib
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_style as es
    with matplotlib.rc_context(es.figure_style("Light")):
        fig = Figure(figsize=(3, 2), dpi=100)  # no canvas attached: the checker adds Agg
        es.align_panel_titles(fig)
        a, b, c = fig.subplots(1, 3)  # three narrow panels: 0..1 ticks would collide
        es.empty_panel(a, "no data")
        b.axis("off")
        c.set_visible(False)
        issues = es.text_collisions(fig)
    assert not [i for i in issues if "tick" in i["role"]], issues
    # empty_panel also clears the ticks, so turning the axis back on shows none.
    assert list(a.get_xticks()) == [] and list(a.get_yticks()) == []


def test_repelled_camera_badges_do_not_overlap_and_get_leaders():
    import matplotlib
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_style as es
    xy = np.array([(10, 1.0), (10.3, 1.02), (10.1, 0.98), (30, 2.0), (31, 2.05), (50, 0.5)])
    with matplotlib.rc_context(es.figure_style("Light")):
        fig = Figure(figsize=(4, 3), dpi=100)
        es.align_panel_titles(fig)
        ax = fig.add_subplot()
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 3)
        artists = es.draw_camera_badges(ax, xy, range(6), size=10, repel=True)
        es.panel_title(ax, "A", "Error vs photos", "one badge per camera")
        assert es.text_collisions(fig) == []
    badges = [a for a in artists if hasattr(a, "get_bbox_patch") and a.get_bbox_patch()]
    leaders = [a for a in artists if a.get_gid() == "edu:leader"]
    assert len(badges) == 6 and len(leaders) == 6
    final = np.array([b.get_position() for b in badges])
    assert np.allclose(final[5], xy[5])          # a lone badge stays on its point
    assert not leaders[5].get_visible()
    assert any(line.get_visible() for line in leaders[:3])   # a moved one shows its leader


def test_strip_plot_log_axis_and_consistent_outlier_labels():
    import matplotlib
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_style as es
    rng = np.random.default_rng(1)
    values = {k: rng.gamma(2, 0.6, 30) for k in range(8)}
    values[2] = np.r_[values[2], 13.8, 29.0]
    values[3] = np.r_[values[3], 2.1e1, 150.0, 300.0, 410.0]
    values[4] = np.r_[values[4], 12.1]
    with matplotlib.rc_context(es.figure_style("Light")):
        fig = Figure(figsize=(4.5, 3.2), dpi=100)
        es.align_panel_titles(fig)
        a = fig.add_subplot(1, 2, 1)
        b = fig.add_subplot(1, 2, 2)
        info = es.strip_plot_by_camera(a, values, cap=6.0)
        es.panel_title(a, "A", "Linear", "")
        log = es.strip_plot_by_camera(b, values, log=True)
        es.panel_title(b, "B", "Log", "")
        assert es.text_collisions(fig) == []
    marks = [t.get_text() for t in a.texts if t.get_gid() == "edu:outlier"]
    assert len(marks) == 3
    numbers = [n for m in marks for n in m.replace("▲", " ").replace("≤", " ").replace(",", " ")
               .split() if n[0].isdigit()]
    # one decimal rule for the whole plot (all outliers >= 10 -> no decimals)
    assert numbers and all("." not in n for n in numbers), marks
    assert es.outlier_decimals([2.1, 13.8]) == 1 and es.outlier_decimals([14, 150]) == 0
    assert b.get_yscale() == "log"
    assert log["cap"] == 100.0 and log["outliers"] == {3: [150.0, 300.0, 410.0]}
    assert [round(t, 6) for t in b.get_yticks()] == [t for t in es.ERROR_TICKS_PX
                                                     if log["ylim"][0] <= t <= log["ylim"][1]]
    assert es.error_scale().log_range([0.42, 7.0]) == (0.3, 10.0)


def test_panel_title_on_3d_axes_and_y_badge_ticks():
    import matplotlib
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_style as es
    with matplotlib.rc_context(es.figure_style("Sepia")):
        fig = Figure(figsize=(7, 3), dpi=100)
        es.align_panel_titles(fig)
        ax3 = fig.add_subplot(1, 2, 1, projection="3d")
        es.panel_title(ax3, "A", "The rig", "seen from above")
        es.panel_title_3d(ax3, "A", "The rig", "seen from the side")
        m = fig.add_subplot(1, 2, 2)
        m.imshow(np.arange(64.0).reshape(8, 8))
        es.badge_ticks(m, range(8), range(8), xlabel=None, size=9)
        es.badge_ticks_y(m, range(8), range(8), size=9, flagged=[2])
        es.panel_title(m, "B", "Shared photos", "rows and columns are cameras")
        assert es.text_collisions(fig) == []
    subs = [t for t in ax3.texts if t.get_gid() == "edu:subtitle"]
    assert [t.get_text() for t in subs] == ["seen from the side"]
    assert not hasattr(subs[0], "get_position_3d")   # a 2D text, not one placed at z
    labels = m.get_yticklabels()
    assert [lab.get_text() for lab in labels] == [str(k) for k in range(1, 9)]
    assert all(lab.get_bbox_patch() is not None for lab in labels)


@pytest.mark.gui
def test_static_scene_colour_bar_beside_the_scene_and_labels_turn_outward():
    from pyCamSet.gui.education_scene3d import Scene3D
    _app()
    scene = Scene3D(theme="Light", backend="static")
    scene.figure.set_size_inches(9, 5)
    for i in range(6):
        ang = i / 6 * 2 * np.pi
        C = np.array([np.cos(ang) * 5, np.sin(ang) * 5, 1.0])
        z = -C / np.linalg.norm(C)
        x = np.cross([0, 0, 1], z)
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z])
        scene.add_camera(f"cam{i}", i, R, -R @ C, scale=0.8, label=f"({i + 1.5:.2f})")
    pts = np.random.default_rng(0).normal(size=(200, 3))
    scene.add_points(pts, np.abs(pts[:, 0]) * 3 + 0.2, scalar_bar="px", cmap="viridis",
                     clim=(0.1, 100), log_scale=True)
    scene.fit_view()
    fig = scene.figure
    fig.canvas.draw()
    bar = fig.axes[1].get_window_extent()
    width = fig.bbox.width
    extent = scene._impl.badges.extent_x1
    assert extent is not None and bar.x0 > extent         # right of everything drawn
    assert bar.x0 - extent < 0.12 * width                  # ... and next to it
    assert bar.x1 < width                                  # inside the figure
    positions = scene.badge_positions()
    sides = scene._impl.badges.label_sides
    centre_x = np.mean([x for x, _ in positions.values()])
    # labels turn away from the target, never across the camera's own glyph
    for name, (x, _y) in positions.items():
        if abs(x - centre_x) > 40:
            assert sides[name] == (1 if x > centre_x else -1), (name, x, centre_x)
    assert set(sides.values()) == {-1, 1}


# --------------------------------------------------------------------------- round-2 review fixes


def _mark_rgb(fig, ax, xy_data, radius_px=2):
    """Mean RGB of the rendered pixels around a data point (Agg)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[..., :3].astype(float)
    x, y = ax.transData.transform(xy_data)
    row = int(round(img.shape[0] - y))
    col = int(round(x))
    patch = img[row - radius_px:row + radius_px + 1, col - radius_px:col + radius_px + 1]
    return patch.reshape(-1, 3).mean(axis=0) / 255.0


def test_camera_6_on_dark_is_filled_on_every_mark():
    """UX2-1: dot, bar, raster cell, line and badge of camera 6 on Dark all show its one fill
    (the mark's inside is the fill, not a hollow ring or a white line)."""
    import matplotlib
    from matplotlib.colors import to_rgb
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from pyCamSet.gui import education_style as es
    fill = np.array(to_rgb(es.camera_fill(5, "Dark")))
    with matplotlib.rc_context(es.figure_style("Dark")):
        fig = Figure(figsize=(4, 3), dpi=100)
        fig._edu_theme = "Dark"
        ax = fig.add_subplot()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.scatter([1], [5], s=400, **es.camera_mark_kwargs(5, ax, "dot"))
        ax.bar([3], [8], width=1.2, **es.camera_mark_kwargs(5, ax, "patch"))
        ax.add_patch(Rectangle((4.5, 4), 1.4, 1.4, **es.camera_mark_kwargs(5, ax, "patch")))
        ax.plot([7, 9], [5, 5], **{**es.camera_mark_kwargs(5, ax, "line"), "linewidth": 8})
        for point in ((1, 5), (3, 4), (5.2, 4.7), (8, 5)):
            got = _mark_rgb(fig, ax, point)
            assert np.abs(got - fill).max() < 0.06, (point, got, fill)




def test_local_value_range_scale():
    """UX2-7: a linear scale over the panel's own range, round ends, labelled bar."""
    import matplotlib
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_style as es
    values = np.array([1.02, 1.6, 2.4, 3.23])
    assert es.needs_local_scale(values) and not es.needs_local_scale([0.2, 40.0])
    scale = es.local_scale(values, "Dark")
    assert scale.lo == 0.0 and scale.hi >= 3.23 and scale.hi in scale.ticks
    assert scale.label == "px · this map only" and scale.note == "this map only"
    colours = scale.colours([0.0, scale.hi])
    assert not np.allclose(colours[0], colours[1])
    with matplotlib.rc_context(es.figure_style("Dark")):
        fig = Figure(figsize=(4, 3), dpi=100)
        es.align_panel_titles(fig)
        ax = fig.add_subplot()
        im = ax.imshow(values.reshape(2, 2), **scale.imshow_kwargs())
        bar = scale.colourbar(fig, ax, im)
        es.panel_title(ax, "E", "Where in the image", "mean error per area")
        assert es.text_collisions(fig) == []
    assert [t.get_text() for t in bar.ax.get_yticklabels()][-1] == scale.tick_text(scale.hi)


def test_trail_geometry_and_static_trail():
    """UX2-6: fading path, arrowhead at the newest point, ghost start pose, hidden stubs."""
    pytest.importorskip("PySide6")
    from matplotlib.colors import to_hex
    from pyCamSet.gui import education_scene3d as s3
    pts = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0, 0], [1.0, 0.2, 0]], float)
    geom = s3.trail_geometry(pts)
    assert len(geom["points"]) == 3 and geom["segments"].shape == (2, 2, 3)
    assert geom["alphas"][0] < geom["alphas"][-1] == s3.TRAIL_FADE[1]
    tips = geom["arrow"][:12, 0]
    assert np.allclose(tips, pts[-1])                # the cone's tip is the newest point
    assert s3.trail_geometry(pts[:1])["length"] == 0.0
    _app()
    scene = s3.Scene3D(theme="Dark", backend="static")
    R = np.eye(3)
    scene.add_camera("cam5", 5, R, [0, 0, 5], scale=0.4)
    scene.add_points(np.random.default_rng(1).normal(size=(30, 3)), name="target")
    path = np.array([[0, 0, -5.0], [0.3, 0.1, -5.0], [0.6, 0.3, -5.1]])
    name = scene.add_trail(path, camera=5, start_pose=(R, [0, 0, 5]))
    assert scene.trail_visible(name)
    item = scene._impl.items[name]
    assert to_hex(item["arrow"].get_facecolor()[0]) == s3.camera_fill(5, "Dark").lower()
    alphas = [c[3] for c in item["line"].get_colors()]
    assert alphas[0] < alphas[-1]
    scene.update_trail(name, np.array([[0, 0, -5.0], [0.001, 0, -5.0]]))  # a stub: hidden
    assert not scene.trail_visible(name)
    scene.set_theme("Light")
    scene.remove(name)
    assert name not in scene.names()


@pytest.mark.gui
def test_camera_picker_elides_only_the_name_and_never_clips():
    """UX2-3: sized to its longest item; narrower, it paints "Camera 8 · …" with the name
    elided in the middle, keeps the number and shows the full title as tooltip data."""
    from pyCamSet.gui.education_stage import CameraCombo, camera_picker
    _app()
    combo = camera_picker({7: "cam8_Back_908B", 0: "cam1_FrontLeft_907T"}, current=7,
                          max_width=1000)
    assert isinstance(combo, CameraCombo)
    assert combo.full_text() == "Camera 8 · cam8_Back_908B"
    metrics = combo.fontMetrics()
    assert combo.sizeHint().width() >= metrics.horizontalAdvance(combo.itemText(1))
    assert camera_picker({0: "x" * 80}).sizeHint().width() <= 300   # the default cap
    assert combo.minimumSizeHint().width() < combo.sizeHint().width()
    combo.resize(combo.sizeHint())
    assert not combo.is_elided()
    combo.resize(combo.minimumSizeHint().width() + 10, combo.sizeHint().height())
    shown = combo.displayText()
    assert shown.startswith("Camera 8 · ") and "…" in shown and combo.is_elided()
    assert metrics.horizontalAdvance(shown) <= combo._text_room()
    # A phase that asks for a narrow picker gets one that still never clips mid-word.
    from PySide6.QtWidgets import QComboBox
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(4)
    combo.resize(combo.sizeHint())
    shown = combo.displayText()
    assert shown.startswith("Camera 8") and metrics.horizontalAdvance(shown) <= combo._text_room()


@pytest.mark.gui
def test_caption_lines_are_counted_at_1280():
    """UX2-5: caption issues count rendered lines at the narrowest supported width."""
    from pyCamSet.gui.education_stage import EducationStageView
    app = _app()
    view = EducationStageView(["One"])
    view.resize(1400, 860)
    view.show()
    _settle(app)
    long = ("This caption is a single sentence that goes on and on with many clauses joined "
            "by semicolons; it explains the lens model, the image centre, the depth-dependent "
            "magnification, the solver, the gauge and the reference photo; it then repeats "
            "itself to be sure it is far too long for a level-one caption at any width")
    long = long + "; " + long  # more than two lines in any font a test machine has
    view.set_caption(long)
    _settle(app)
    lines = view.caption_line_count()
    assert lines is not None and lines > 2
    assert view.caption_line_count(width=4000) < lines
    from PySide6.QtGui import QFontInfo
    from pyCamSet.gui.education_style import CAPTION_BUDGET_FONT
    if QFontInfo(view.caption.font()).family() == CAPTION_BUDGET_FONT:  # judged only in it
        assert view.caption_lines_at_narrowest == lines
        assert any("lines at 1280 px at L1" in issue for issue in view.caption_issues)
    from pyCamSet.gui.education_style import caption_problems
    assert caption_problems("One.", 1, lines=3) == ["3 lines at 1280 px at L1 (limit 2)"]
    view.set_caption("Short and sweet.")
    _settle(app)
    assert view.caption_issues == [] and view.caption_line_count() == 1
    view.close()


@pytest.mark.gui
def test_panel_heading_matches_panel_title_spacing():
    """UX2-19 / UX-31: Qt 3D panel headings use the letter gap and parse old titles."""
    from pyCamSet.gui.education_stage import PanelHeading
    from pyCamSet.gui.education_style import LETTER_GAP
    _app()
    heading = PanelHeading()
    heading.set_text("A The solver at work", "changes shown ×5")
    assert heading.heading_text() == f"A{LETTER_GAP}The solver at work"
    heading.resize(220, heading.height())
    title, _sub = heading.lines_shown()
    assert title.endswith("…") and title.startswith("A")
    assert "The solver at work" in heading.toolTip()


def test_significant_digits_and_weight_widths():
    """UX2-14 / UX-36: one precision for a row of labels; widths that show "more"."""
    from pyCamSet.gui import education_style as es
    assert [es.format_sig(v) for v in (0.0834, 0.14, 6.81, 52.7, 139)] == [
        "0.083", "0.14", "6.8", "53", "140"]
    assert es.format_sig(float("nan")) == "–" and es.format_sig(0) == "0"
    widths = es.weight_linewidths([0, 1, 4, 16])
    assert widths[0] == es.WEIGHT_LINEWIDTHS[0] and widths[-1] == es.WEIGHT_LINEWIDTHS[1]
    assert np.all(np.diff(widths) > 0.9)          # every step is visibly thicker


# --------------------------------------------------------------------------- round-3 shared layer


def test_amount_and_signed_maps_are_not_the_error_map():
    """UX3-3: viridis only for errors; counts/shares get a one-hue map, signed values a
    diverging one, on every theme; LocalScale picks the map from its unit."""
    import matplotlib
    from matplotlib.colors import to_hex
    from pyCamSet.gui import education_style as es
    for theme in es.theme_names():
        amount, signed = es.amount_cmap(theme), es.signed_cmap(theme)
        assert not es.is_error_cmap(amount) and not es.is_error_cmap(signed)
        assert es.is_error_cmap(es.error_cmap(theme))
        hue = es.amount_hue(theme)
        assert es.viridis_distance(hue) >= es.AMOUNT_MIN_VIRIDIS_DISTANCE
        assert es.contrast(hue, es.stage_tokens(theme)["stage"]) >= es.MIN_STAGE_CONTRAST
        # One hue: luminance moves monotonically from the stage end to the hue end.
        ramp = matplotlib.colormaps[amount](np.linspace(0, 1, 9))
        lum = [es.relative_luminance(to_hex(c)) for c in ramp]
        assert np.all(np.diff(lum) < 0) or np.all(np.diff(lum) > 0), theme
        # Diverging: two distinct ends, a neutral middle close to the stage.
        ends = matplotlib.colormaps[signed]([0.0, 0.5, 1.0])
        assert es.colour_distance(to_hex(ends[0]), to_hex(ends[2])) > 40
        assert es.colour_distance(to_hex(ends[1]), es.stage_tokens(theme)["stage"]) < 25
    assert es.local_scale([1, 2, 3], "Light").cmap == es.error_cmap("Light")
    assert es.local_scale([1, 2, 3], "Light", unit="points").cmap == es.amount_cmap("Light")
    scale = es.amount_scale([0, 4, 17], "Dark", unit="points")
    assert scale.quantity == "amount" and scale.cmap == es.amount_cmap("Dark") and scale.hi >= 17
    signed = es.signed_scale([-3.0, 1.2], "Sepia", unit="µm")
    assert signed.lo == -signed.hi and signed.hi >= 3.0 and signed.quantity == "signed"
    assert es.is_error_unit("px · this map only") and es.is_error_unit("µm")
    assert not es.is_error_unit("points") and not es.is_error_unit("share")


def test_colour_map_misuse_finds_non_error_data_in_viridis():
    """UX3-3 checker: counts / ties / hexbin densities in viridis are reported; real error
    scales (log px, local px) are not."""
    import matplotlib
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_style as es
    with matplotlib.rc_context(es.figure_style("Light")):
        fig = Figure(figsize=(10, 3))
        a, b, c, d = fig.subplots(1, 4)
        counts = a.imshow(np.arange(16).reshape(4, 4), cmap=es.error_cmap(a))
        es.colourbar_px(fig, counts, a, label="points")
        es.panel_title(a, "A", "Coverage", "points per cell")
        b.hexbin(np.random.default_rng(0).normal(size=200),
                 np.random.default_rng(1).normal(size=200), gridsize=8, cmap=es.error_cmap(b))
        es.panel_title(b, "B", "Where the residuals fall", "px")
        c.imshow(np.random.default_rng(2).random((4, 4)), cmap=es.error_cmap(c))
        es.panel_title(c, "C", "Which parameters moved", "share of the change")
        scale = es.error_scale(d)
        errors = d.scatter([0, 1, 2], [0, 1, 2], c=[0.3, 3, 30], **scale.scatter_kwargs())
        scale.colourbar(fig, d, errors)
        es.panel_title(d, "D", "Error per corner", "log px")
    found = es.colour_map_misuse(fig)
    panels = {issue["panel"].split()[1] for issue in found}
    assert panels == {"Coverage", "Where", "Which"}, found
    assert all(issue["problem"] == "error colour map on non-error data" for issue in found)
    with matplotlib.rc_context(es.figure_style("Dark")):
        fig = Figure(figsize=(4, 3))
        ax = fig.subplots()
        scale = es.amount_scale([0, 9], ax, unit="points")
        image = ax.imshow(np.arange(9).reshape(3, 3), **scale.imshow_kwargs())
        scale.colourbar(fig, ax, image)
        es.panel_title(ax, "A", "Coverage", "points per cell")
    assert es.colour_map_misuse(fig) == []


#: Phase functions that still draw a non-error quantity with ``error_cmap`` (round-3 review
#: UX3-3). The phase changers remove them; a
#: function may leave this set, never join it.
PENDING_ERROR_MAP_ADOPTION: set[tuple[str, str]] = set()
#: Direct ``error_cmap`` users that do draw errors / displacements.
ERROR_MAP_USERS = {
    ("education_scene3d.py", "_resolve_cmap"),
    ("education_phase4.py", "draw_coloured_hist"),     # shape change in µm (the 3D key)
}


def test_no_new_direct_error_map_users_in_phase_code():
    """UX3-3, static half of the check: every function that calls ``error_cmap`` directly
    (or passes ``cmap="viridis"``) is a known error/displacement user or still pending
    adoption, so non-error data never newly lands in viridis. The shared scales
    (``error_scale``, ``local_scale``, ``amount_scale``) carry the rule themselves."""
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "pyCamSet"
    users = set()
    for path in list((root / "gui").glob("education_*.py")) + \
            list((root / "workflow").glob("education_*.py")):
        if path.name == "education_style.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.FunctionDef):
                continue
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Call):
                    continue
                fn = sub.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                literal = any(isinstance(k.value, ast.Constant) and k.value.value == "viridis"
                              for k in sub.keywords)
                if name == "error_cmap" or literal:
                    users.add((path.name, node.name))
    unexpected = users - PENDING_ERROR_MAP_ADOPTION - ERROR_MAP_USERS
    assert not unexpected, f"non-error data must use es.amount_cmap / amount_scale: {unexpected}"




def test_card_overflow_checker_and_card_text_fit():
    """UX3-1: a finding written past its card's border is reported; ``card_text`` wraps and
    shrinks it so it stays inside."""
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch
    from pyCamSet.gui import education_style as es
    finding = "0 free directions · 3 once-seen corners may slide (fine)"

    def card(ax):
        ax.set_axis_off()
        patch = FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0",
                               transform=ax.transAxes, facecolor="#eeeeee",
                               edgecolor="#999999")
        ax.add_patch(patch)
        return patch

    with matplotlib.rc_context(es.figure_style("Light")):
        fig = Figure(figsize=(7, 1.6))
        es.align_panel_titles(fig)
        a, b, c = fig.subplots(1, 3)
        card(a)
        a.text(0.08, 0.5, finding, transform=a.transAxes, fontsize=8.4, va="center")
        patch = card(b)
        label = es.card_text(b, 0.08, 0.5, finding, card=patch, transform=b.transAxes,
                             fontsize=8.4, va="center")
        card(c)
        c.text(0.5, 0.5, "fits", transform=c.transAxes, ha="center", va="center")
    found = es.card_overflows(fig)
    assert len(found) == 1 and found[0]["problem"] == "outside its card"
    assert found[0]["text"].startswith("0 free") and found[0]["overflow_px"] > 10
    assert "\n" in label.get_text() and label.get_fontsize() >= es.CARD_TEXT_MIN_SIZE


def test_limit_lines_are_neutral_until_crossed():
    """UX3-11: limits in muted dashes; red only when a value crosses; cursors in accent."""
    import matplotlib
    from matplotlib.colors import to_hex
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_style as es
    with matplotlib.rc_context(es.figure_style("Sepia")):
        ax = Figure().subplots()
        t = es.stage_tokens("Sepia")
        calm = es.limit_line(ax, 20, values=[1, 5, 19], label="20 px limit")
        hot = es.limit_line(ax, 5, values=[1, 7])
        low = es.limit_line(ax, 6, values=[8, 9], above_is_bad=False, axis="x")
    assert to_hex(calm.get_color()) == to_hex(t["muted"]) and not calm._edu_limit_crossed
    assert to_hex(hot.get_color()) == to_hex(t["bad"]) and hot._edu_limit_crossed
    assert not low._edu_limit_crossed
    assert ax.texts[0].get_color() == t["muted"]
    assert es.cursor_style("Dark")["color"] == es.stage_tokens("Dark")["accent"]
    assert es.limit_crossed([np.nan, 3], 2) and not es.limit_crossed([], 2)


@pytest.mark.gui
def test_caption_limits_hold_at_1280_and_1400_without_orphans():
    """UX3-2: lines are measured at both supported widths and a one-word last line is a
    problem."""
    from pyCamSet.gui.education_stage import EducationStageView
    from pyCamSet.gui.education_style import SUPPORTED_DIALOG_WIDTHS, caption_problems
    assert SUPPORTED_DIALOG_WIDTHS == (1280, 1400)
    assert caption_problems("One. Two.", 2, lines_by_width={1280: 3, 1400: 4}) == [
        "4 lines at 1400 px at L2 (limit 3)"]
    assert caption_problems("A b.", 3, lines_by_width={1400: 2},
                            last_lines={1400: "focus)."}) == [
        'one-word last line "focus)." at 1400 px']
    assert caption_problems("A.", 1, lines_by_width={1400: 1}, last_lines={1400: "A."}) == []
    app = _app()
    view = EducationStageView(["One"])
    view.resize(1400, 860)
    view.show()
    _settle(app)
    view.set_caption("A first sentence that is long enough to wrap onto a second line at the "
                     "width of a caption beside the chips in a dialog of this size, and one "
                     "more clause to be sure it wraps.")
    _settle(app)
    lines = view.caption_lines_text(view.caption.width())
    assert len(lines) >= 1 and "".join(lines).replace(" ", "") == \
        view.caption.text().replace(" ", "")
    assert view.caption_width_at(1280) == view.caption.width() - 120
    from PySide6.QtGui import QFontInfo
    from pyCamSet.gui.education_style import CAPTION_BUDGET_FONT
    if QFontInfo(view.caption.font()).family() == CAPTION_BUDGET_FONT:  # judged only in it
        assert set(view.caption_lines_by_width) == {1280, 1400}
        assert view.caption_lines_by_width[1280] >= view.caption_lines_by_width[1400]
        assert view.caption_lines_at_narrowest == view.caption_lines_by_width[1280]
        last = view.caption_last_lines[1400]
        if len(last.split()) == 1 and view.caption_lines_by_width[1400] > 1:
            assert any("one-word last line" in i for i in view.caption_issues)
    view.close()


@pytest.mark.gui
def test_chip_numbers_use_the_text_colour_and_a_tone_mark():
    """UX3-9: warn numbers are no longer drawn in the (camera-2-like) ochre."""
    from PySide6.QtWidgets import QLabel
    from pyCamSet.gui.education_stage import Chip, StatChips, theme_tokens_for
    _app()
    chips = StatChips()
    chips.set_chips([Chip("7", "dropped", "warn"), Chip("8 / 8", "cameras go on", "good"),
                     ("2", "bad border", "bad")])
    marks = [lab.text() for lab in chips.findChildren(QLabel, "eduChipMark")]
    assert sorted(marks) == ["!", "✕"]
    text = theme_tokens_for()["text"]
    for frame in chips._frames:
        assert f"QLabel#eduChipValue {{ color: {text};" in frame.styleSheet()


@pytest.mark.gui
def test_scene_player_counter_names_the_interpolated_frame():
    """UX3-6: half-way between the start and step 1 the counter reads "→ Step 2 of 9", not
    "Start", and a whole position restores the usual counter."""
    from pyCamSet.gui.education_scene3d import Scene3D, ScenePlayer
    _app()
    scene = Scene3D(backend="static")
    scene.add_camera("c", 0, np.eye(3), [0, 0, 0])
    frames = [{"cameras": {"c": {"R": np.eye(3), "t": [-float(k), 0, 0]}}} for k in range(3)]
    player = ScenePlayer(scene, frames, step_numbers=[2, 5], total_steps=9,
                         reduce_motion=True)
    status = player.controls._status
    assert status.displayText() == "Start"
    player.show_position(0.5)
    assert status.displayText() == "→ Step 2 of 9" and "interpolated" in status.text()
    player.show_position(1.0)
    assert status.displayText() == "Step 2 of 9"
    player.show_position(1.5)
    assert status.displayText() == "→ Step 5 of 9" and "step 2" in status.text()
    assert status.width() >= status.fontMetrics().horizontalAdvance("→ Step 5 of 9")
    player.stop_for_context_change()


def test_trail_arrowhead_and_ghost_stay_small():
    """UX3-8: the cone is at most 35 % of the lens length, its tip short of the body; the
    start ghost has no lens rays and hides while the camera has barely moved."""
    pytest.importorskip("PySide6")
    from pyCamSet.gui import education_scene3d as s3
    scale = 0.4
    spec = {"index": 2, "fade": s3.TRAIL_FADE, "min_fraction": 0.0, "scale": scale,
            "stop_short": 0.45 * scale}
    path = np.c_[np.linspace(0, 3.0, 20), np.zeros(20), np.zeros(20)]
    geom, visible, ghost = s3.trail_pieces(path, spec, scene_size=10.0)
    assert visible and ghost
    assert geom["arrow_length"] <= s3.TRAIL_ARROW_BODY_FRACTION * 0.85 * scale + 1e-9
    assert geom["points"][-1][0] <= 3.0 * s3.TRAIL_ARROW_AT + 1e-9
    short = np.c_[np.linspace(0, 0.1, 5), np.zeros(5), np.zeros(5)]
    _geom, visible, ghost = s3.trail_pieces(short, spec, scene_size=10.0)
    assert visible and not ghost
    full = s3.camera_glyph(np.eye(3), [0, 0, 0], 1.0, True)
    bare = s3.camera_glyph(np.eye(3), [0, 0, 0], 1.0, True, rays=False)
    assert len(bare[2]) == len(full[2]) - 4


