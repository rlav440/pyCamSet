'''Check how education mode picks its 3D backend and draws it cheaply: the backend
rules per platform, environment value and probe result; one render per batched primitive
and per playback frame; frame skipping on slow draws; the DPR cap; and the opt-in
diagnostics (watchdog off by default, logging when on).
OpenGL in the offscreen Qt platform): PyVista is exercised through a fake plotter only.
'''
from __future__ import annotations

import sys
import threading

import numpy as np
import pytest

# Every test here drives pyCamSet.gui.education_scene3d, which needs Qt.
pytest.importorskip("PySide6")


def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def clean_env(monkeypatch):
    # QT_QPA_PLATFORM is deliberately left alone: a QApplication created while it is unset
    # would use the native platform and change font metrics for every later test.
    for name in ("PYCAMSET_EDU_3D", "PYCAMSET_EDU_DEBUG"):
        monkeypatch.delenv(name, raising=False)
    from pyCamSet.gui import education_scene3d as es
    monkeypatch.setattr(es, "FORCE_STATIC", False)
    monkeypatch.setattr(es, "_LAST_CHOICE", None)
    yield monkeypatch


def _host(monkeypatch, *, platform="win32", cpus=8, ram=16 * 2**30, probe_ok=True,
          pyvistaqt=True, gui_thread=True):
    """A fake machine: platform, CPU count, RAM, OpenGL probe, pyvistaqt and thread."""
    from pyCamSet.gui import education_scene3d as es
    calls = {"probe": 0}

    def probe(*, refresh=False):
        calls["probe"] += 1
        return {"ok": probe_ok, "vendor": "V", "renderer": "R", "version": "4.1",
                "error": None if probe_ok else "RuntimeError: no context"}

    monkeypatch.setattr(sys, "platform", platform)
    # Not an offscreen platform for the decision (and no application is created here).
    monkeypatch.setenv("QT_QPA_PLATFORM", "")
    monkeypatch.setattr(es.perf.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(es.perf, "physical_ram_bytes", lambda: ram)
    monkeypatch.setattr(es, "gl_probe", probe)
    monkeypatch.setattr(es, "_pyvistaqt_importable", lambda: pyvistaqt)
    monkeypatch.setattr(es, "_on_gui_thread", lambda: gui_thread)
    return calls


# --------------------------------------------------------------------------- backend rules


def test_capable_windows_host_with_working_gl_chooses_pyvista(clean_env):
    from pyCamSet.gui import education_scene3d as es
    calls = _host(clean_env)
    choice = es.scene_backend_choice()
    assert choice.name == "pyvista" and not choice.automatic
    assert calls["probe"] == 1
    assert es.scene_backend() == "pyvista"


def test_macos_with_working_gl_chooses_pyvista(clean_env):
    from pyCamSet.gui import education_scene3d as es
    calls = _host(clean_env, platform="darwin")
    choice = es.scene_backend_choice()
    assert choice.name == "pyvista" and not choice.automatic
    assert calls["probe"] == 1


def test_macos_opt_in_with_env_pyvista(clean_env):
    from pyCamSet.gui import education_scene3d as es
    calls = _host(clean_env, platform="darwin")
    clean_env.setenv("PYCAMSET_EDU_3D", "pyvista")
    assert es.scene_backend() == "pyvista"
    assert calls["probe"] == 0  # forced: no automatic checks


@pytest.mark.parametrize("cpus, ram", [(2, 16 * 2**30), (1, 16 * 2**30), (8, 2 * 2**30)])
def test_weak_hardware_keeps_pyvista_with_cheaper_settings(clean_env, cpus, ram):
    from pyCamSet.gui import education_perf as perf
    from pyCamSet.gui import education_scene3d as es
    _host(clean_env, cpus=cpus, ram=ram)
    assert es.scene_backend_choice().name == "pyvista"
    assert perf.reduced_settings()
    assert perf.msaa_samples() == 0
    assert perf.scaled_resolution(16) < 16


@pytest.mark.parametrize("info, fragment", [
    ({"renderer": "llvmpipe (LLVM 15)", "version": "4.5"}, "software rendering"),
    ({"renderer": "Apple Software Renderer", "version": "4.1"}, "software rendering"),
    ({"renderer": "Apple M1", "version": "2.1 Metal"}, "older than"),
    ({"renderer": "Intel HD 4000", "version": "4.1", "clear_ms": 120.0}, "full-frame clear"),
])
def test_unsuitable_gl_falls_back_to_static(clean_env, info, fragment):
    from pyCamSet.gui import education_scene3d as es
    _host(clean_env, platform="darwin")
    clean_env.setattr(es, "gl_probe", lambda *, refresh=False: {"ok": True, **info})
    choice = es.scene_backend_choice()
    assert choice.name == "static" and choice.automatic
    assert fragment in choice.reason


def test_good_gl_verdicts_pass():
    from pyCamSet.gui import education_perf as perf
    assert perf.gl_verdict({"renderer": "Apple M1", "version": "4.1 Metal - 88",
                            "clear_ms": 1.2}) is None
    assert perf.gl_verdict({"renderer": None, "version": None}) is None  # unknown: not held
    assert perf.parse_gl_version("3.3.0 NVIDIA 535") == (3, 3)
    assert perf.parse_gl_version("OpenGL ES 3.0 Mesa") == (3, 0)
    assert perf.parse_gl_version("garbage") is None


def test_repeated_slow_renders_switch_new_scenes_to_static(clean_env):
    from pyCamSet.gui import education_perf as perf
    from pyCamSet.gui import education_scene3d as es
    _host(clean_env)
    perf.reset_slow_state()
    try:
        assert not perf.note_render_time(2.0) and not perf.note_render_time(2.0)
        assert perf.note_render_time(2.0)  # third in a row
        choice = es.scene_backend_choice()
        assert choice.name == "static" and choice.automatic and "too slow" in choice.reason
        clean_env.setenv("PYCAMSET_EDU_3D", "pyvista")  # forced: the user's choice stands
        assert es.scene_backend() == "pyvista"
    finally:
        perf.reset_slow_state()


def test_a_fast_render_resets_the_slow_count():
    from pyCamSet.gui import education_perf as perf
    perf.reset_slow_state()
    try:
        perf.note_render_time(2.0)
        perf.note_render_time(2.0)
        perf.note_render_time(0.01)
        assert not perf.note_render_time(2.0)
        assert perf.too_slow_reason() is None
    finally:
        perf.reset_slow_state()


def test_surface_format_is_requested_only_on_macos_before_the_app(clean_env):
    from pyCamSet.gui import education_perf as perf
    clean_env.setattr(sys, "platform", "win32")
    assert perf.configure_surface_format() is False


def test_weak_hardware_opt_in_with_env_pyvista(clean_env):
    from pyCamSet.gui import education_scene3d as es
    _host(clean_env, cpus=2)
    clean_env.setenv("PYCAMSET_EDU_3D", "pyvista")
    assert es.scene_backend() == "pyvista"


def test_probe_failure_falls_back_to_static_with_reason(clean_env):
    from pyCamSet.gui import education_scene3d as es
    _host(clean_env, probe_ok=False)
    choice = es.scene_backend_choice()
    assert choice.name == "static" and choice.automatic
    assert "OpenGL probe failed" in choice.reason


@pytest.mark.parametrize("value", ["static", " STATIC "])
def test_env_static_forces_static_without_a_note(clean_env, value):
    from pyCamSet.gui import education_scene3d as es
    calls = _host(clean_env)
    clean_env.setenv("PYCAMSET_EDU_3D", value)
    choice = es.scene_backend_choice()
    assert choice.name == "static" and not choice.automatic
    assert calls["probe"] == 0
    es._remember_choice(choice)
    assert es.backend_note() == ""


@pytest.mark.parametrize("value", ["auto", "", "AUTO"])
def test_env_auto_or_unset_applies_the_automatic_rules(clean_env, value):
    from pyCamSet.gui import education_scene3d as es
    _host(clean_env)
    if value:
        clean_env.setenv("PYCAMSET_EDU_3D", value)
    assert es.scene_backend() == "pyvista"


def test_unknown_env_value_is_treated_as_auto_and_says_so(clean_env):
    from pyCamSet.gui import education_scene3d as es
    _host(clean_env)
    clean_env.setenv("PYCAMSET_EDU_3D", "bogus")
    choice = es.scene_backend_choice()
    assert choice.name == "pyvista"
    assert "unknown PYCAMSET_EDU_3D" in choice.reason


def test_offscreen_platform_is_headless_static_without_a_note(clean_env):
    from pyCamSet.gui import education_scene3d as es
    _host(clean_env)
    clean_env.setenv("QT_QPA_PLATFORM", "offscreen")
    choice = es.scene_backend_choice()
    assert choice.name == "static" and choice.headless and not choice.automatic
    es._remember_choice(choice)
    assert es.backend_note() == ""


def test_forced_pyvista_still_falls_back_when_pyvistaqt_is_missing(clean_env):
    from pyCamSet.gui import education_scene3d as es
    _host(clean_env, pyvistaqt=False)
    clean_env.setenv("PYCAMSET_EDU_3D", "pyvista")
    choice = es.scene_backend_choice()
    assert choice.name == "static" and "pyvistaqt" in choice.reason


def test_not_on_the_gui_thread_is_static(clean_env):
    from pyCamSet.gui import education_scene3d as es
    _host(clean_env, gui_thread=False)
    assert es.scene_backend_choice().name == "static"


def test_backend_note_names_the_opt_in_only_for_automatic_static(clean_env):
    from pyCamSet.gui import education_scene3d as es
    automatic = es.BackendChoice("static", "macOS: the embedded 3D view is off by default",
                                 automatic=True)
    note = es.backend_note_for(automatic)
    assert "chosen automatically" in note and "PYCAMSET_EDU_3D=pyvista" in note
    assert es.backend_note_for(es.BackendChoice("pyvista", "x")) == ""
    assert es.backend_note_for(es.BackendChoice("static", "forced", automatic=False)) == ""


def test_gl_probe_reports_its_fields(clean_env):
    from pyCamSet.gui import education_scene3d as es
    _app()
    info = es.gl_probe(refresh=True)
    assert {"ok", "vendor", "renderer", "version", "error"} <= set(info)
    assert isinstance(info["ok"], bool)


# --------------------------------------------------------------------------- cost policy


def test_msaa_is_reduced_on_macos_and_weak_hardware(clean_env):
    from pyCamSet.gui import education_perf as perf
    _host(clean_env, cpus=8)
    assert perf.msaa_samples() == 8
    _host(clean_env, platform="darwin", cpus=8)
    assert perf.msaa_samples() == perf.MSAA_SAMPLES_LOW == 0  # FXAA instead
    _host(clean_env, cpus=2)
    assert perf.msaa_samples() == perf.MSAA_SAMPLES_LOW


def test_pixel_ratio_cap_rules(clean_env):
    from pyCamSet.gui import education_perf as perf
    _host(clean_env, cpus=8)
    assert perf.pixel_ratio_cap(1000, 600, 1.0) == 1.0
    assert perf.pixel_ratio_cap(1000, 600, 2.0) == 2.0  # normal display, within budget
    _host(clean_env, cpus=2)
    assert perf.pixel_ratio_cap(1000, 600, 2.0) == perf.DPR_CAP_WEAK
    assert perf.pixel_ratio_cap(1000, 600, 1.0) == 1.0
    _host(clean_env, cpus=8)
    capped = perf.pixel_ratio_cap(2000, 1500, 2.0)  # 12 MP physical: over the budget
    assert 1.0 <= capped < 2.0
    assert 2000 * 1500 * capped * capped <= perf.PIXEL_BUDGET + 1
    assert perf.pixel_ratio_cap(0, 0, 2.0) == 2.0  # no size yet: nothing to budget


def test_capped_canvas_is_freed_when_dropped(clean_env):
    """The cap must not keep its canvas alive (the lifetime tests count live widgets)."""
    import gc
    import weakref
    _app()
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_perf as perf
    canvas = FigureCanvasQTAgg(Figure(figsize=(4, 3)))
    perf.cap_canvas_pixel_ratio(canvas)
    ref = weakref.ref(canvas)
    del canvas
    gc.collect()
    assert ref() is None


def test_cap_canvas_routes_the_device_pixel_ratio(clean_env):
    _app()
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_perf as perf
    _host(clean_env, cpus=2)
    weak = FigureCanvasQTAgg(Figure(figsize=(4, 3)))
    perf.cap_canvas_pixel_ratio(weak)
    weak._set_device_pixel_ratio(2.0)
    assert weak.device_pixel_ratio == pytest.approx(perf.DPR_CAP_WEAK)
    perf.cap_canvas_pixel_ratio(weak)  # twice is harmless
    _host(clean_env, cpus=8)
    normal = FigureCanvasQTAgg(Figure(figsize=(4, 3)))
    perf.cap_canvas_pixel_ratio(normal)
    normal._set_device_pixel_ratio(2.0)
    assert normal.device_pixel_ratio == pytest.approx(2.0)


# --------------------------------------------------------------------------- PyVista batching


class _FakeBadges:
    """Stands in for the VTK badge overlay (needs a real renderer otherwise)."""

    def __init__(self, plotter, theme):
        self.p = plotter
        self.theme = theme
        self.bar_actors = []
        self.gutter = 0.0
        self.positions = {}

    def add(self, *args, **kwargs):
        pass

    def move(self, *args, **kwargs):
        pass

    def remove(self, *args, **kwargs):
        pass

    def clear(self):
        pass

    def set_theme(self, theme):
        pass

    def mark(self, *args, **kwargs):
        pass

    def _scale(self):
        return 1.0


class _FakeProperty:
    def __init__(self):
        self.colour = None
        self.opacity = None

    def SetColor(self, *rgb):  # noqa: N802 - VTK API name
        self.colour = rgb

    def SetOpacity(self, value):  # noqa: N802
        self.opacity = value


class _FakeMapper:
    def __init__(self, actor):
        self.actor = actor

    def SetInputData(self, mesh):  # noqa: N802
        self.actor.inputs.append(mesh)


class _FakeActor:
    def __init__(self):
        self.inputs = []
        self.visible = True
        self._mapper = _FakeMapper(self)
        self._property = _FakeProperty()

    def GetMapper(self):  # noqa: N802
        return self._mapper

    def GetProperty(self):  # noqa: N802
        return self._property

    def SetVisibility(self, flag):  # noqa: N802
        self.visible = bool(flag)


class _FakePlotter:
    """Records add_mesh calls and render calls; no VTK, no window."""

    def __init__(self):
        self.renders = 0
        self.meshes: list[dict] = []
        self.actors: dict[str, _FakeActor] = {}
        self.window_size = [800, 600]
        self.scalar_bars: dict = {}

    def set_background(self, *args, **kwargs):
        pass

    def enable_anti_aliasing(self, *args, **kwargs):
        self.msaa = kwargs

    def render(self):
        self.renders += 1

    def add_mesh(self, mesh, **kwargs):
        self.meshes.append(kwargs)
        actor = _FakeActor()
        if kwargs.get("name"):
            self.actors[kwargs["name"]] = actor
        return actor

    def remove_actor(self, name, render=True):
        self.actors.pop(name, None)

    def remove_scalar_bar(self, *args, **kwargs):
        pass

    def clear(self):
        self.actors.clear()


@pytest.fixture
def fake_pyvista(clean_env):
    pytest.importorskip("pyvista")
    from pyCamSet.gui import education_scene3d as es
    clean_env.setattr(es, "_PyVistaBadges", _FakeBadges)
    plotter = _FakePlotter()
    impl = es._PyVistaImpl(plotter, "Light")
    plotter.renders = 0  # the constructor renders once; count from here
    plotter.meshes.clear()
    return impl, plotter


def test_camera_board_points_rays_each_render_once_and_add_without_render(fake_pyvista):
    impl, plotter = fake_pyvista
    impl.add_camera("c0", 0, np.eye(3), [0.0, 0.0, -5.0], 1.0, False, "1", True)
    assert plotter.renders == 1
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    plotter.renders = 0
    impl.add_board("b0", corners, "#336699", None, None, None, 0.5, 4, False)
    assert plotter.renders == 1
    plotter.renders = 0
    impl.add_points("p0", corners, None, None, None, "#aa3300", 5, None, False)
    assert plotter.renders == 1
    plotter.renders = 0
    impl.add_rays("r0", corners, corners + 1.0, "#005500", None, None, None, 1.0, 1.0, False)
    assert plotter.renders == 1
    assert plotter.meshes and all(m.get("render") is False for m in plotter.meshes)


def _trail_spec(scale=0.4):
    from pyCamSet.gui import education_scene3d as es
    return {"index": 0, "colour": None, "ghost": None, "arrow": True, "width": 3.0,
            "fade": es.TRAIL_FADE, "min_fraction": 0.0, "scale": scale, "stop_short": 0.0}


def test_trail_updates_reuse_their_actors_and_render_only_when_asked(fake_pyvista):
    impl, plotter = fake_pyvista
    path = np.c_[np.linspace(0, 3.0, 40), np.zeros(40), np.zeros(40)]
    impl.add_trail("t0", path[:5], _trail_spec())
    added = len(plotter.meshes)
    assert added >= 2  # the line and the arrowhead
    plotter.renders = 0
    for k in range(6, 40):  # a playback: many updates, each without a render
        impl.update_trail("t0", path[:k], render=False)
    assert len(plotter.meshes) == added, "trail pieces were removed and re-added"
    assert plotter.renders == 0
    line_actor = plotter.actors["t0:trail"]
    assert len(line_actor.inputs) == 34 and line_actor.visible
    impl.update_trail("t0", path[:20])  # the default renders once
    assert plotter.renders == 1


def test_trail_hidden_when_too_short_and_shown_again(fake_pyvista):
    impl, plotter = fake_pyvista
    path = np.c_[np.linspace(0, 3.0, 40), np.zeros(40), np.zeros(40)]
    # A reference item sets the scene size (a trail alone would always be "long enough").
    impl.add_points("ref", path, None, None, None, "#888888", 3, None, False)
    spec = dict(_trail_spec(), min_fraction=0.5)
    impl.add_trail("t0", path, spec)
    actor = plotter.actors["t0:trail"]
    impl.update_trail("t0", path[:2], render=False)  # about 0.05 of the scene: hidden
    assert actor.visible is False
    impl.update_trail("t0", path[:30], render=False)  # about 0.7 of the scene: shown again
    assert actor.visible is True


def test_single_point_trail_is_hidden_not_stale(fake_pyvista):
    """Playback position 0 gives a one-point trail: it must hide, not keep the old path."""
    impl, plotter = fake_pyvista
    path = np.c_[np.linspace(0, 3.0, 40), np.zeros(40), np.zeros(40)]
    impl.add_trail("t0", path[:10], _trail_spec())
    actor = plotter.actors["t0:trail"]
    impl.update_trail("t0", path[:1], render=False)
    assert actor.visible is False


def test_inplace_trail_matches_a_fresh_build_with_real_vtk(clean_env):
    """With real VTK (where an offscreen render window can be made): playback updates of an
    existing trail give the same picture as a trail built once from the final points."""
    pv = pytest.importorskip("pyvista")
    from pyCamSet.gui import education_scene3d as es
    try:
        plotters = [pv.Plotter(off_screen=True, window_size=[320, 240]) for _ in range(2)]
    except Exception as exc:  # no render window on this machine: nothing to compare
        pytest.skip(f"no offscreen VTK render window ({type(exc).__name__})")
    path = np.c_[np.linspace(0, 3.0, 40), np.sin(np.linspace(0, 3, 40)), np.zeros(40)]
    spec = _trail_spec()

    def scene(plotter):
        impl = es._PyVistaImpl(plotter, "Light")
        impl.add_camera("c0", 0, np.eye(3), [0.0, 0.0, -5.0], 1.0, False, "1", False)
        return impl

    try:
        incremental = scene(plotters[0])
        incremental.add_trail("t0", path[:3], spec)
        for k in range(4, 40):
            incremental.update_trail("t0", path[:k], render=False)
        incremental.render()
        fresh = scene(plotters[1])
        fresh.add_trail("t0", path[:39], spec)
        fresh.render()
        image_a = plotters[0].screenshot(return_img=True)
        image_b = plotters[1].screenshot(return_img=True)
    except Exception as exc:  # a GL that cannot render offscreen: skip, do not fail
        pytest.skip(f"offscreen VTK could not render ({type(exc).__name__})")
    finally:
        for plotter in plotters:
            plotter.close()
    assert np.array_equal(image_a, image_b)


# --------------------------------------------------------------------------- playback


class _StubScene:
    """Records what the player asks of a scene; apply_frame can cost time on a fake clock."""

    def __init__(self, clock=None, cost_ms=0.0):
        self.clock = clock
        self.cost_ms = cost_ms
        self.applied: list[bool] = []  # the ``render`` flag of each apply_frame call
        self.renders = 0
        self.animating: list[bool] = []

    def apply_frame(self, frame, render=True):
        self.applied.append(render)
        if self.clock is not None:
            self.clock["t"] += self.cost_ms

    def render(self):
        self.renders += 1

    def set_animating(self, on):
        self.animating.append(bool(on))

    def names(self):
        return []


def _frames(n):
    return [{"cameras": {}} for _ in range(n)]


def _player(scene, frames, positions=None):
    _app()
    from pyCamSet.gui.education_scene3d import ScenePlayer
    player = ScenePlayer(scene, frames, reduce_motion=False, substeps=8,
                         on_frame=(lambda p, _f: positions.append(p)) if positions is not None
                         else None)
    return player


def test_each_shown_frame_costs_one_render(clean_env):
    scene = _StubScene()
    player = _player(scene, _frames(3))
    scene.applied.clear()
    scene.renders = 0
    seen = []
    player.frameShown.connect(lambda position, _frame: seen.append(position))  # listeners
    player.show_position(0.5)
    assert seen == [0.5]
    assert scene.applied == [False]  # apply_frame does not render itself ...
    assert scene.renders == 1  # ... the player renders once, after the listeners ran


def test_playback_on_a_fast_machine_passes_through_intermediate_positions(clean_env):
    clock = {"t": 0.0}
    scene = _StubScene(clock=clock, cost_ms=0.0)
    positions: list[float] = []
    player = _player(scene, _frames(2), positions)
    player._now_ms = lambda: clock["t"]
    positions.clear()
    player._target_changed(1.0)
    ticks = 0
    while player.position != 1.0 and ticks < 200:
        clock["t"] += 40  # a 40 ms timer on a machine that draws instantly
        player._tween_tick()
        ticks += 1
    assert player.position == 1.0 and positions[-1] == 1.0
    assert sum(1 for p in positions if 0.0 < p < 1.0) >= 5


def test_slow_draws_skip_positions_and_the_final_frame_is_always_drawn(clean_env):
    clock = {"t": 0.0}
    scene = _StubScene(clock=clock, cost_ms=200.0)  # each draw takes 200 ms
    positions: list[float] = []
    player = _player(scene, _frames(2), positions)
    player._now_ms = lambda: clock["t"]
    positions.clear()
    player._target_changed(1.0)
    ticks = 0
    while player.position != 1.0 and ticks < 200:
        clock["t"] += 78  # the timer interval the player chose for this step
        player._tween_tick()
        ticks += 1
    assert player.position == 1.0
    assert positions[-1] == 1.0
    # A fast machine would draw about 8 tween frames for this step; a slow one draws fewer,
    # and the playback still ends on the target.
    assert len(positions) <= 4


def test_a_tick_cannot_run_inside_another_draw(clean_env):
    clock = {"t": 0.0}
    holder = {}

    class ReentrantScene(_StubScene):
        def apply_frame(self, frame, render=True):
            super().apply_frame(frame, render)
            if "player" in holder and holder.get("nest"):
                holder["player"]._tween_tick()  # a nested tick, as if a draw ran the event loop

    scene = ReentrantScene(clock=clock, cost_ms=10.0)
    player = _player(scene, _frames(2))
    holder["player"] = player
    player._now_ms = lambda: clock["t"]
    player._target_changed(1.0)
    scene.applied.clear()
    holder["nest"] = True
    clock["t"] += 80
    player._tween_tick()
    assert len(scene.applied) == 1  # the nested tick returned at once


def test_static_scene_draws_are_throttled_while_animating_and_full_when_done():
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_perf as perf
    from pyCamSet.gui.education_scene3d import _StaticImpl
    drawn = []
    impl = _StaticImpl(Figure(figsize=(4, 3)), "Light", lambda: drawn.append(1))
    drawn.clear()  # the constructor draws once; count from here
    now = {"t": 100.0}
    impl._clock = lambda: now["t"]
    impl.set_animating(True)
    impl.render()  # first frame of the tween
    now["t"] += 0.5 * perf.STATIC_DRAW_INTERVAL_MS / 1000.0
    impl.render()  # too soon: skipped
    now["t"] += perf.STATIC_DRAW_INTERVAL_MS / 1000.0
    impl.render()
    assert len(drawn) == 2
    impl.set_animating(False)  # the tween has stopped: the final frame is drawn at once
    impl.render()
    assert len(drawn) == 3


# --------------------------------------------------------------------------- diagnostics


@pytest.fixture
def diagnostics(clean_env, tmp_path):
    from pyCamSet.gui import education_diagnostics as dg
    clean_env.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path / "config"))
    dg._reset_for_tests()
    yield dg, tmp_path / "config"
    dg._reset_for_tests()


def test_watchdog_is_off_by_default(diagnostics):
    dg, config = diagnostics
    from pyCamSet.gui.education_scene3d import BackendChoice
    assert dg.install() is None
    dg.note_backend(BackendChoice("static", "test", automatic=True))
    assert not any(t.name == dg.WATCHDOG_THREAD_NAME for t in threading.enumerate())
    assert not (config / "logs").exists()


def test_debug_logging_writes_the_backend_and_a_stall_stack(diagnostics, monkeypatch):
    dg, config = diagnostics
    from pyCamSet.gui.education_scene3d import BackendChoice
    monkeypatch.setenv("PYCAMSET_EDU_DEBUG", "1")
    dg.note_backend(BackendChoice("static", "macOS: the embedded 3D view is off by default",
                                  automatic=True))
    if dg._WATCHDOG is not None:  # no heartbeat in this test: stop the thread, keep the log
        dg._WATCHDOG.stop()
    clock = {"t": 0.0}
    dog = dg.StallWatchdog(threading.get_ident(), clock=lambda: clock["t"],
                           frames=sys._current_frames)
    clock["t"] = 2.5
    assert dog.check() is True
    assert dog.check() is False  # one report per stall
    dog.beat()
    text = (config / "logs" / dg.LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "3D backend: static" in text and "macOS" in text
    assert "GUI thread stalled for 2.5 s" in text
    assert "test_debug_logging_writes_the_backend_and_a_stall_stack" in text  # the stack
    assert "responsive again" in text


def test_watchdog_starts_once_when_enabled(diagnostics, monkeypatch):
    dg, _config = diagnostics
    _app()
    monkeypatch.setenv("PYCAMSET_EDU_DEBUG", "1")
    dog = dg.install()
    try:
        assert dog is not None and dg.install() is dog
        assert any(t.name == dg.WATCHDOG_THREAD_NAME for t in threading.enumerate())
    finally:
        dg._reset_for_tests()


def test_environment_lines_name_the_platform_and_cpus(clean_env):
    from pyCamSet.gui import education_diagnostics as dg
    _app()
    lines = dg.environment_lines()
    assert "logical CPUs" in lines[0] and sys.platform in lines[0]
    assert any("Qt" in line for line in lines[1:])
