"""3D scenes for education mode: cameras, boards, points and rays that can be animated.

* :class:`Scene3D` (a QWidget) draws with an embedded ``pyvistaqt.QtInteractor`` when one
  can be made, else with a static matplotlib 3D canvas, so offscreen tests never break.
  ``PYCAMSET_EDU_3D=static`` (or :data:`FORCE_STATIC` = True) forces the fallback;
  ``PYCAMSET_EDU_3D=pyvista`` forces PyVista even under ``QT_QPA_PLATFORM=offscreen``.
* :class:`OffscreenScene` has the same primitives on ``pyvista.Plotter(off_screen=True)``
  for PNG exports and review screenshots.
* :class:`ScenePlayer` plays a list of frames through a scene with smooth interpolation
  between recorded frames, using the dialog's one transport bar
  (:class:`~pyCamSet.gui.education_dialog.RecordedStepControls`).

One camera depiction everywhere (UX-4, UX-7, UX-9, round-1 review):

* glyph per lens type, never a box that could pass for the cube target: a small camera body
  plus a **frustum** (pinhole) or a slim **lens barrel with parallel ray lines**
  (telecentric); every camera has a thin **viewing-direction line** in its colour
  (``add_camera(axis=True)``; its length reaches the scene's target, see :meth:`fit_view`);
* the badge is the same **filled circle** with the 1-based number as in 2D, drawn as a
  screen overlay: never hidden behind geometry, never outside the frame, pushed apart when
  two badges would overlap (:func:`layout_badges`); a ``label`` such as ``"(0.47)"`` sits
  beside the circle, as on the poster;
* every camera glyph, viewing line, trail and badge uses the camera's one fill for the theme
  (:func:`~pyCamSet.gui.education_style.camera_fill`): low-contrast poster colours (camera 6
  on Dark, 1 and 5 on Light/Sepia) are replaced by their substitute fill, never by a white or
  outline-coloured edge; only their badge gets the thin outline ring (UX2-1, round 2);
* :meth:`_SceneAPI.add_trail` draws motion (a camera or pose travelling during the solve) as a
  fading path in the camera's fill with a ghost of the start pose and an arrowhead (UX2-6).

:meth:`_SceneAPI.fit_view` picks the azimuth/elevation that separates the cameras best,
keeps a ``selected`` camera in front of the others and frames everything with a margin
(and a gutter for the colour bar).

Conventions: camera poses are world-to-camera, ``x_cam = R @ x_world + t`` (as the solver
stores them); ``R`` may be a 3x3 matrix or a rotation vector, or pass a 4x4 extrinsic as
``R`` with ``t=None``. The camera looks along its +z axis, image x right, image y down.
Camera ``index`` is 0-based (badge shows index + 1) and picks :func:`camera_colour`.

Frame dicts (for :meth:`apply_frame`, :func:`interpolate_frames`, :class:`ScenePlayer`)::

    {"cameras": {name: {"R": R, "t": t}},
     "boards":  {name: corners_Nx3 | {"corners": Nx3, "scalars": float | N}},
     "points":  {name: xyz_Nx3 | {"xyz": Nx3, "scalars": N}},
     "rays":    {name: {"origins": Nx3, "ends": Nx3, "scalars": N (optional)}}}

Every name must have been added first (``add_camera`` / ``add_board`` / ...); updates move
the existing geometry in place (no actor rebuild), so playback is cheap.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyCamSet.gui import education_perf as perf
from pyCamSet.gui.education_style import (
    CONTINUOUS_CMAP, badge_text_colour, camera_badge_label, camera_colour, camera_edge,
    camera_fill, camera_needs_outline, error_cmap, face_colour, face_label, face_tag_bbox,
    face_text_colour, resolve_theme, stage_tokens,
)
from pyCamSet.gui.education_style import layout_badges as _shared_layout_badges

#: Set True to force the static matplotlib fallback (tests, no OpenGL).
FORCE_STATIC = False
ENV_SWITCH = "PYCAMSET_EDU_3D"
#: Fraction of the view's width kept free on the right for the colour bar.
SCALAR_BAR_GUTTER = 0.13
#: Static fallback colour bar: width, gap to the scene and room for its tick labels (pixels
#: at 100 dpi).
BAR_WIDTH_PX = 12.0
BAR_GAP_PX = 22.0
BAR_LABEL_ROOM_PX = 46.0
#: Vertical view angle (degrees) that :meth:`_SceneAPI.fit_view` uses.
FIT_VIEW_ANGLE = 18.0
#: Badge radius in screen pixels (PyVista overlay; scaled with the window).
BADGE_RADIUS_PX = 11.0


#: Values of ``PYCAMSET_EDU_3D``: ``static`` forces the matplotlib picture, ``pyvista``
#: forces the embedded OpenGL view (no automatic checks), ``auto`` (or unset) applies them.
ENV_VALUES = ("", "auto", "static", "pyvista")
_HEADLESS_PLATFORMS = frozenset({"offscreen", "minimal"})


@dataclass(frozen=True)
class BackendChoice:
    """Which 3D backend a scene uses, and why.

    ``automatic`` is True when the choice came from the automatic rules (macOS, weak
    hardware, a failed OpenGL probe) rather than from ``PYCAMSET_EDU_3D``; the dialog then
    explains the static picture and how to opt in. ``headless`` marks an offscreen Qt
    platform (tests, screenshots), which gets no note.
    """

    name: str
    reason: str
    automatic: bool = False
    headless: bool = False


#: The most recent choice made by a :class:`Scene3D` (read by :func:`backend_note`).
_LAST_CHOICE: BackendChoice | None = None
#: Result of the OpenGL probe, cached for the process (None until it runs).
_PROBE: dict[str, Any] | None = None


def _pyvistaqt_importable() -> bool:
    try:
        import pyvistaqt  # noqa: F401  (availability probe, as gui/assess_panel.py)
    except Exception:  # noqa: BLE001 - a missing or broken install means "no PyVista"
        return False
    return True


def _on_gui_thread() -> bool:
    """True on the Qt GUI thread (or when no application exists yet)."""
    try:
        from PySide6.QtCore import QCoreApplication, QThread
    except ImportError:
        return False
    app = QCoreApplication.instance()
    return app is None or QThread.currentThread() == app.thread()


def _gl_string(functions: Any, name: int) -> str | None:
    try:
        value = functions.glGetString(name)
    except Exception:  # noqa: BLE001 - the strings are informational only
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    return str(value) if value is not None else None


def _time_clear(functions: Any) -> float | None:
    """Median milliseconds for clearing a 1024 x 768 off-screen buffer (None when it cannot
    be measured). A GPU does this in well under a millisecond; a software or very weak
    renderer takes tens of milliseconds, which is what the verdict looks for."""
    import time
    try:
        from PySide6.QtOpenGL import QOpenGLFramebufferObject
    except ImportError:
        return None
    try:
        fbo = QOpenGLFramebufferObject(1024, 768)
        if not fbo.isValid() or not fbo.bind():
            return None
        try:
            samples = []
            for _ in range(5):
                start = time.perf_counter()
                functions.glClearColor(0.1, 0.2, 0.3, 1.0)
                functions.glClear(0x4000 | 0x100)  # GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT
                functions.glFinish()
                samples.append((time.perf_counter() - start) * 1000.0)
            return float(sorted(samples)[len(samples) // 2])
        finally:
            fbo.release()
    except Exception:  # noqa: BLE001 - an unmeasurable clear is not held against the machine
        return None


def gl_probe(*, refresh: bool = False) -> dict[str, Any]:
    """Create an OpenGL context on an offscreen surface and report whether it works.

    Cheap (one context, no window), cached per process, GUI thread only. ``ok`` is False when
    no context can be made; ``vendor``/``renderer``/``version`` are filled when it can. This
    does not catch a native crash later in VTK, only the common "no GL at all" failure.
    """
    global _PROBE
    if _PROBE is not None and not refresh:
        return dict(_PROBE)
    info: dict[str, Any] = {"ok": False, "vendor": None, "renderer": None,
                            "version": None, "error": None}
    try:
        from PySide6.QtGui import QGuiApplication, QOffscreenSurface, QOpenGLContext
        if QGuiApplication.instance() is None:
            raise RuntimeError("no QGuiApplication")
        context = QOpenGLContext()
        if not context.create():
            raise RuntimeError("QOpenGLContext.create() failed")
        surface = QOffscreenSurface()
        surface.setFormat(context.format())
        surface.create()
        if not surface.isValid():
            raise RuntimeError("offscreen surface is not valid")
        if not context.makeCurrent(surface):
            raise RuntimeError("makeCurrent() failed")
        try:
            functions = context.functions()
            info["vendor"] = _gl_string(functions, 0x1F00)    # GL_VENDOR
            info["renderer"] = _gl_string(functions, 0x1F01)  # GL_RENDERER
            info["version"] = _gl_string(functions, 0x1F02)   # GL_VERSION
            info["clear_ms"] = _time_clear(functions)
        finally:
            context.doneCurrent()
        info["ok"] = True
    except Exception as exc:  # noqa: BLE001 - the probe must never break the stage
        info["error"] = f"{type(exc).__name__}: {exc}"
    _PROBE = info
    return dict(info)


def scene_backend_choice() -> BackendChoice:
    """Decide the 3D backend for a new :class:`Scene3D` (read on every call).

    Order: :data:`FORCE_STATIC` or ``PYCAMSET_EDU_3D=static`` give static. Then an offscreen
    Qt platform gives static. ``PYCAMSET_EDU_3D=pyvista`` gives PyVista (skipping the checks
    below, but still static if pyvistaqt is missing). Otherwise: a view already found too slow
    this session gives static; an OpenGL probe that fails or reports software rendering, a
    version below 3.2 or a slow timed clear gives static; the rest gives PyVista (macOS and
    weak hardware included, with the cheaper drawing settings of ``education_perf``). An unknown value of the variable is ignored, with a note in the reason.
    """
    raw = os.environ.get(ENV_SWITCH, "").strip().casefold()
    note = ""
    if raw not in ENV_VALUES:
        note = f"unknown {ENV_SWITCH}={raw!r} treated as auto; "
        raw = "auto"
    if FORCE_STATIC:
        return BackendChoice("static", "forced in code (FORCE_STATIC)")
    if raw == "static":
        return BackendChoice("static", f"forced by {ENV_SWITCH}=static")
    forced = raw == "pyvista"
    qt_platform = os.environ.get("QT_QPA_PLATFORM", "").casefold()
    if not forced and qt_platform in _HEADLESS_PLATFORMS:
        return BackendChoice("static", f"Qt platform {qt_platform} has no OpenGL surface",
                             headless=True)
    if not forced and perf.too_slow_reason():
        return BackendChoice("static",
                             f"the 3D view was too slow here ({perf.too_slow_reason()})",
                             automatic=True)
    if not _pyvistaqt_importable():
        return BackendChoice("static", "pyvistaqt is not installed")
    if not _on_gui_thread():
        return BackendChoice("static", "not created on the GUI thread")
    if not forced:
        probe = gl_probe()
        if not probe.get("ok"):
            return BackendChoice("static", f"OpenGL probe failed ({probe.get('error')})",
                                 automatic=True)
        unsuitable = perf.gl_verdict(probe)
        if unsuitable:
            return BackendChoice(
                "static", f"this computer's OpenGL is unsuitable ({unsuitable})",
                automatic=True)
        return BackendChoice("pyvista", note + "automatic: OpenGL probe passed")
    return BackendChoice("pyvista", note + f"forced by {ENV_SWITCH}=pyvista")


def scene_backend() -> str:
    """``"pyvista"`` when an embedded PyVista view should be used, else ``"static"``."""
    return scene_backend_choice().name


def backend_note_for(choice: BackendChoice | None) -> str:
    """One line explaining a static picture that was chosen automatically, else ""."""
    if choice is None or choice.name != "static" or not choice.automatic or choice.headless:
        return ""
    return (f"3D view: static picture, chosen automatically ({choice.reason}). To try the "
            f"interactive 3D view, set {ENV_SWITCH}=pyvista before starting pyCamSet.")


def backend_note() -> str:
    """The note for the most recent 3D scene (the dialog shows it), or "" when none applies."""
    return backend_note_for(_LAST_CHOICE)


def forget_backend_choice() -> None:
    """Clear the remembered choice, so a view with no 3D scene shows no note."""
    _remember_choice(None)


def _remember_choice(choice: BackendChoice | None) -> None:
    global _LAST_CHOICE
    _LAST_CHOICE = choice


# --------------------------------------------------------------------------------------
# Geometry (pure numpy)


def as_pose(R: Any, t: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Normalise a pose to (3x3 rotation, translation) from a matrix, a rotation vector or a
    4x4 extrinsic (``t=None``)."""
    R = np.asarray(R, dtype=float)
    if R.shape == (4, 4):
        return R[:3, :3].copy(), R[:3, 3].copy()
    if R.shape in {(3,), (3, 1), (1, 3)}:
        from scipy.spatial.transform import Rotation
        R = Rotation.from_rotvec(R.ravel()).as_matrix()
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3, a rotation vector or a 4x4 extrinsic")
    t = np.zeros(3) if t is None else np.asarray(t, dtype=float).ravel()
    return R, t


def camera_centre(R: Any, t: Any = None) -> np.ndarray:
    """World position of a camera with world-to-camera pose (R, t): ``C = -R^T t``."""
    R, t = as_pose(R, t)
    return -R.T @ t


_BARREL_SIDES = 12


def camera_glyph(R: Any, t: Any = None, scale: float = 1.0,
                 telecentric: bool = False, *, rays: bool = True
                 ) -> tuple[np.ndarray, list[list[int]], list[tuple[int, int]]]:
    """Vertices, faces and edges of a camera glyph in world coordinates.

    Both lens types share a small camera **body** behind the centre (with a tab on its top
    edge, -y, showing which way is up). Pinhole: a **frustum** in front (vertex 0 is its
    apex = the camera centre; the image rectangle sits ``scale`` ahead). Telecentric: a
    slim **lens barrel** (a 12-sided cylinder, vertex 0 = the centre of its back ring)
    with four **parallel ray lines** running on from its front. Neither can be mistaken
    for a box-shaped target. ``rays=False`` leaves out the telecentric ray lines (start
    ghosts, UX3-8).
    """
    R, t = as_pose(R, t)
    c = -R.T @ t
    x, y, z = R[0], R[1], R[2]
    s = float(scale)
    verts: list[np.ndarray] = []
    faces: list[list[int]] = []
    edges: list[tuple[int, int]] = []

    def add(point: np.ndarray) -> int:
        verts.append(np.asarray(point, dtype=float))
        return len(verts) - 1

    if telecentric:
        radius, length, ray_end = 0.13 * s, 0.85 * s, 1.75 * s
        add(c)                                                  # 0: back-ring centre
        back = [add(c + radius * (np.cos(a) * x + np.sin(a) * y))
                for a in np.linspace(0, 2 * np.pi, _BARREL_SIDES, endpoint=False)]
        front = [add(c + length * z + radius * (np.cos(a) * x + np.sin(a) * y))
                 for a in np.linspace(0, 2 * np.pi, _BARREL_SIDES, endpoint=False)]
        n = _BARREL_SIDES
        for k in range(n):
            faces.append([back[k], back[(k + 1) % n], front[(k + 1) % n], front[k]])
            edges += [(back[k], back[(k + 1) % n]), (front[k], front[(k + 1) % n])]
        faces.append(list(front))
        for k in range(0, n, n // 4):                           # four parallel rays
            tip = add(c + ray_end * z + radius * (np.cos(2 * np.pi * k / n) * x
                                                  + np.sin(2 * np.pi * k / n) * y))
            edges += [(back[k], front[k])] + ([(front[k], tip)] if rays else [])
    else:
        hw, hh = 0.42 * s, 0.30 * s
        add(c)                                                  # 0: apex = centre
        far = [add(c + s * z + sx * hw * x + sy * hh * y)
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        faces += [[0, far[0], far[1]], [0, far[1], far[2]], [0, far[2], far[3]],
                  [0, far[3], far[0]]]
        edges += [(0, f) for f in far] + [(far[k], far[(k + 1) % 4]) for k in range(4)]
    # Body: a small box behind the centre, with an "up" tab on its top (-y) edge.
    bw, bh, bd = 0.2 * s, 0.15 * s, 0.32 * s
    body = [add(c - dz * z + sx * bw * x + sy * bh * y)
            for dz in (0.02 * s, bd) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for quad in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6),
                 (3, 0, 4, 7)):
        faces.append([body[i] for i in quad])
    edges += [(body[a], body[b]) for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                                              (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                                              (2, 6), (3, 7))]
    tab_tip = add(c - 0.2 * s * z - (bh + 0.12 * s) * y)
    faces.append([body[0], body[1], tab_tip])
    return np.array(verts), faces, edges


def camera_axis_segment(R: Any, t: Any = None, length: float = 1.0) -> np.ndarray:
    """The viewing-direction line: ``[centre, centre + length * optical axis]`` (2 x 3)."""
    R, t = as_pose(R, t)
    c = -R.T @ t
    return np.vstack([c, c + float(length) * R[2]])


def axis_length_to(R: Any, t: Any, target: Any, minimum: float = 0.0) -> float:
    """Distance along the camera's optical axis to the foot of ``target`` (>= ``minimum``)."""
    R, t = as_pose(R, t)
    c = -R.T @ t
    return max(float(minimum), float((np.asarray(target, dtype=float) - c) @ R[2]))


def _badge_anchor(R: Any, t: Any, scale: float) -> np.ndarray:
    """Where a camera's badge points to: just above its body."""
    R, t = as_pose(R, t)
    c = -R.T @ t
    return c - 0.2 * scale * R[2] - 0.42 * scale * R[1]


_badge_position = _badge_anchor  # older name


def layout_badges(points: Any, radius: float | Sequence[float],
                  bounds: tuple[float, float, float, float], *,
                  centre: Sequence[float] | None = None,
                  label_widths: Sequence[float] | None = None,
                  label_sides: Sequence[int] | None = None, gap: float = 3.0,
                  iterations: int = 60) -> np.ndarray:
    """Screen positions for badges (see :func:`pyCamSet.gui.education_style.layout_badges`,
    the one shared implementation: 2D figures, the PyVista overlay and the static
    fallback use the same spreading)."""
    return _shared_layout_badges(points, radius, bounds, centre=centre,
                                 label_widths=label_widths, label_sides=label_sides, gap=gap,
                                 iterations=iterations)


#: Face tags sit this far outside their face (a share of the target's largest extent), so
#: the tag never covers the corners it names.
FACE_TAG_OFFSET = 0.10


def face_tag_layout(groups: Sequence[tuple[int, Any]]) -> list[tuple[int, np.ndarray]]:
    """Where each face tag goes: ``groups`` is ``[(0-based face, corners Nx3), ...]``; the tag
    of a face sits at its corners' centroid pushed away from the target's centre, so it
    lies just outside that face. Returns ``[(face, xyz), ...]`` in the same order."""
    faces = [int(f) for f, _ in groups]
    corners = [np.asarray(pts, dtype=float).reshape(-1, 3) for _, pts in groups]
    if not corners:
        return []
    everything = np.vstack(corners)
    centre = everything.mean(axis=0)
    size = float(np.max(np.ptp(everything, axis=0))) or 1.0
    out = []
    for face, pts in zip(faces, corners):
        c = pts.mean(axis=0)
        d = c - centre
        n = float(np.linalg.norm(d))
        push = FACE_TAG_OFFSET * size * (d / n if n > 1e-12 else np.zeros(3))
        out.append((face, c + push))
    return out


def _view_direction(azimuth: float, elevation: float, up: Sequence[float]) -> np.ndarray:
    """Unit vector from the focus to the eye, for azimuth/elevation in degrees around ``up``."""
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = ref - up * (ref @ up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    az, el = np.radians(azimuth), np.radians(elevation)
    return np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * up


def _screen_basis(direction: np.ndarray, up: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    up = np.asarray(up, dtype=float)
    right = np.cross(up, direction)
    if np.linalg.norm(right) < 1e-9:
        right = np.cross([1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0], direction)
    right /= np.linalg.norm(right)
    return right, np.cross(direction, right)


def choose_view(centres: Any, focus: Any, *, up: Sequence[float] = (0, 0, 1),
                selected: int | None = None, glyph_size: float = 0.0,
                target_radius: float = 0.0,
                elevations: Sequence[float] = (18, 26, 34, 42),
                azimuths: Sequence[float] | None = None) -> tuple[float, float]:
    """(azimuth, elevation) in degrees that best separates the cameras on screen.

    Score: the smallest screen distance between two camera centres (relative to the
    picture's extent), minus a penalty when the ``selected`` camera (row in ``centres``)
    would sit behind another camera or behind the target; a slight preference for the
    selected camera facing the viewer and for a moderate elevation.
    """
    pts = np.asarray(centres, dtype=float).reshape(-1, 3)
    focus = np.asarray(focus, dtype=float)
    azimuths = np.arange(0.0, 360.0, 12.0) if azimuths is None else azimuths
    best, best_score = (-60.0, 26.0), -np.inf
    rel = pts - focus
    for el in elevations:
        for az in azimuths:
            d = _view_direction(az, el, up)
            right, upv = _screen_basis(d, up)
            xy = np.stack([rel @ right, rel @ upv], axis=1)
            depth = rel @ d
            extent = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-9)
            if len(pts) >= 2:
                diff = xy[:, None, :] - xy[None, :, :]
                dist = np.hypot(diff[..., 0], diff[..., 1])
                dist[np.diag_indices(len(pts))] = np.inf
                score = float(dist.min()) / extent
            else:
                score = 0.0
            if selected is not None and 0 <= selected < len(pts):
                s = int(selected)
                near = np.hypot(*(xy - xy[s]).T) < max(glyph_size, 0.05 * extent)
                blockers = near & (depth > depth[s]) & (np.arange(len(pts)) != s)
                score -= 0.5 * float(blockers.sum())
                if depth[s] < 0 and np.hypot(*xy[s]) < max(target_radius, 1e-9):
                    score -= 0.5
                score += 0.04 * float(depth[s] / (np.abs(depth).max() + 1e-9))
            score -= 0.002 * abs(el - 26)
            if score > best_score + 1e-9:
                best, best_score = (float(az), float(el)), score
    return best


def fit_distance(points: Any, focus: Any, direction: Any, up: Sequence[float], *,
                 view_angle: float = 30.0, aspect: float = 1.5, margin: float = 0.1,
                 gutter: float = 0.0) -> float:
    """Camera distance so every point is inside a perspective view of ``view_angle``
    (vertical, degrees) and ``aspect`` (width/height) with ``margin`` on each side and a
    right-hand ``gutter`` (fraction of the width) left free."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3) - np.asarray(focus, dtype=float)
    d = np.asarray(direction, dtype=float)
    right, upv = _screen_basis(d, up)
    half = np.tan(np.radians(view_angle) / 2)
    half_w = half * aspect * max(1.0 - 2 * gutter, 0.3)
    z, x, y = pts @ d, np.abs(pts @ right), np.abs(pts @ upv)
    k = 1.0 + 2 * margin
    need = np.maximum(z + y * k / half, z + x * k / half_w)
    return float(max(need.max(), 1e-6)) if len(pts) else 1.0


def _hull_order(points: np.ndarray) -> list[int]:
    """Indices of the outline of (nearly) planar points, in order around the plane."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return list(range(len(pts)))
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    uv = centred @ vt[:2].T
    order = sorted(range(len(uv)), key=lambda i: (uv[i, 0], uv[i, 1]))

    def cross(o, a, b):
        return (uv[a, 0] - uv[o, 0]) * (uv[b, 1] - uv[o, 1]) - \
               (uv[a, 1] - uv[o, 1]) * (uv[b, 0] - uv[o, 0])
    lower: list[int] = []
    for i in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)
    upper: list[int] = []
    for i in reversed(order):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


def _lerp(a: Any, b: Any, s: float) -> Any:
    if a is None or b is None:
        return b if s >= 0.5 else a
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        return b if s >= 0.5 else a
    return (1 - s) * a_arr + s * b_arr


def _slerp(Ra: np.ndarray, Rb: np.ndarray, s: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp
    rots = Rotation.from_matrix(np.stack([Ra, Rb]))
    return Slerp([0.0, 1.0], rots)([s]).as_matrix()[0]


def interpolate_pose(a: Mapping, b: Mapping, s: float) -> dict[str, np.ndarray]:
    """Camera pose between ``a`` and ``b`` (dicts with R, t): slerp the rotation, lerp the
    camera *centre* (so cameras fly in straight lines), return world-to-camera R, t."""
    Ra, ta = as_pose(a["R"], a.get("t"))
    Rb, tb = as_pose(b["R"], b.get("t"))
    R = _slerp(Ra, Rb, float(s))
    centre = (1 - s) * (-Ra.T @ ta) + s * (-Rb.T @ tb)
    return {"R": R, "t": -R @ centre}


def _interp_entry(a: Any, b: Any, s: float) -> Any:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return {key: _lerp(a.get(key), b.get(key), s) for key in set(a) | set(b)}
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        return b if s >= 0.5 else a
    return _lerp(a, b, s)


def interpolate_frames(a: Mapping, b: Mapping, s: float) -> dict[str, dict[str, Any]]:
    """The frame a fraction ``s`` (0..1) of the way from frame ``a`` to frame ``b``.

    Cameras: rotation slerp + centre lerp. Boards, points, rays and their scalars: linear.
    Entries present in only one frame are taken from that frame.
    """
    s = float(np.clip(s, 0.0, 1.0))
    out: dict[str, dict[str, Any]] = {}
    for kind in ("cameras", "boards", "points", "rays"):
        fa, fb = a.get(kind, {}) or {}, b.get(kind, {}) or {}
        if not fa and not fb:
            continue
        merged: dict[str, Any] = {}
        for name in list(fa) + [n for n in fb if n not in fa]:
            if name in fa and name in fb:
                merged[name] = (interpolate_pose(fa[name], fb[name], s) if kind == "cameras"
                                else _interp_entry(fa[name], fb[name], s))
            else:
                merged[name] = fa.get(name, fb.get(name))
        out[kind] = merged
    return out


def pace_weights(values: Sequence[float], *, floor: float = 0.2, log: bool = True) -> list[float]:
    """Playback weights from a per-state value (e.g. mean error per recorded state): the
    time spent on each step is proportional to how much the value changes in it (on a
    log scale by default), never below ``floor`` x the average (UX-11, ED-1). Returns one
    weight per step (``len(values) - 1``), mean 1."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return []
    if log:
        v = np.log(np.maximum(v, 1e-12))
    change = np.abs(np.diff(v))
    if not np.isfinite(change).all() or change.sum() <= 0:
        return [1.0] * len(change)
    w = change / change.mean()
    w = np.maximum(w, floor)
    return list(w / w.mean())


def _colour_for_scalar(value: float, cmap: str, clim: tuple[float, float] | None,
                       log_scale: bool = False) -> str:
    import matplotlib
    from matplotlib.colors import to_hex
    lo, hi = clim if clim is not None else (0.0, 1.0)
    if log_scale and lo > 0 and hi > lo and value > 0:
        frac = float(np.clip((np.log(value) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1))
    else:
        frac = 0.5 if hi <= lo else float(np.clip((value - lo) / (hi - lo), 0, 1))
    return to_hex(matplotlib.colormaps[cmap](frac))


def _rgb(colour: str) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb
    return to_rgb(colour)


def _edge_colour(index: int, theme: str) -> str:
    """Colour of camera ``index``'s glyph edges: its fill for ``theme`` (:func:`camera_fill`),
    the same colour as every other mark of that camera (UX2-1). Never white, never hollow."""
    return camera_fill(index, theme)


#: Opacity of a camera glyph's body (every camera alike; edges are opaque).
CAMERA_BODY_OPACITY = 0.6


def _resolve_cmap(cmap: str | None, theme: str) -> str:
    """``None`` or plain viridis -> the theme-aware map (clipped low end on Dark)."""
    if cmap is None or cmap == CONTINUOUS_CMAP:
        return error_cmap(theme)
    return cmap


def _nice_ticks(lo: float, hi: float, log_scale: bool) -> list[float]:
    if log_scale and lo > 0:
        from pyCamSet.gui.education_style import ERROR_TICKS_PX
        ticks = [t for t in ERROR_TICKS_PX if lo * 0.999 <= t <= hi * 1.001]
        if len(ticks) >= 2:
            return ticks
        exps = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
        return [10.0 ** e for e in exps if lo <= 10.0 ** e <= hi] or [lo, hi]
    from matplotlib.ticker import MaxNLocator
    ticks = MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    return [float(t) for t in ticks if lo - 1e-9 <= t <= hi + 1e-9]


def _tick_format(ticks: Sequence[float]) -> str:
    span = max(abs(t) for t in ticks) if ticks else 1.0
    if all(abs(t - round(t)) < 1e-9 for t in ticks) and span >= 1:
        return "%.0f"
    return "%.1f" if span >= 1 else "%.2f"


# --------------------------------------------------------------------------------------
# Trails: motion drawn as a fading path, a ghost start pose and an arrowhead (UX2-6)

#: Alpha at the start and the end of a trail (old -> new).
TRAIL_FADE = (0.25, 1.0)
#: Trails shorter than this fraction of the scene's size are hidden (a stub reads as a
#: rendering glitch, not as motion).
TRAIL_MIN_FRACTION = 0.02
#: Opacity of the ghost of the start pose (body, edges). UX3-8: 15 % body, no lens rays.
TRAIL_GHOST_OPACITY = (0.15, 0.32)
#: Arrowhead size (UX3-8, round 3: PyVista cones were as large as the lens barrel): at most
#: this fraction of the glyph's lens length (``0.85 * scale``), radius this fraction of
#: its length, tip at :data:`TRAIL_ARROW_AT` of the trail (never on the body).
TRAIL_ARROW_BODY_FRACTION = 0.35
TRAIL_ARROW_RADIUS = 0.30
TRAIL_ARROW_AT = 0.85
#: The start ghost is hidden while the start-to-end distance is below this many glyph
#: scales (one body width), so it never overlaps the live body.
TRAIL_GHOST_MIN_TRAVEL = 0.40


def trail_pieces(points: Any, spec: Mapping[str, Any], scene_size: float
                 ) -> tuple[dict[str, Any], bool, bool]:
    """Shared trail geometry for both backends: ``(geometry, visible, ghost_visible)``.

    The path stops short of the camera body (``spec["stop_short"]``) and at
    :data:`TRAIL_ARROW_AT` of its length, so the arrowhead sits on the path, not on the
    body; the cone is at most :data:`TRAIL_ARROW_BODY_FRACTION` of the lens length. The
    ghost shows only once the camera moved at least :data:`TRAIL_GHOST_MIN_TRAVEL` scales."""
    full = trail_geometry(points, fade=spec["fade"])
    visible = full["length"] >= spec["min_fraction"] * scene_size
    scale = float(spec["scale"])
    gap = max(float(spec.get("stop_short", 0.0)),
              (1.0 - TRAIL_ARROW_AT) * full["length"] if spec.get("index") is not None
              else 0.0)
    path = trim_path_end(full["points"], gap) if full["length"] > 3 * gap else full["points"]
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) \
        if len(path) > 1 else 0.0
    arrow = min(0.05 * scene_size, TRAIL_ARROW_BODY_FRACTION * 0.85 * scale, 0.6 * length)
    geom = trail_geometry(path, fade=spec["fade"], arrow_length=arrow,
                          radius_ratio=TRAIL_ARROW_RADIUS)
    pts = full["points"]
    travel = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) > 1 else 0.0
    return geom, bool(visible), bool(visible and travel >= TRAIL_GHOST_MIN_TRAVEL * scale)


def trim_path_end(points: Any, gap: float) -> np.ndarray:
    """``points`` (N x 3) with the last ``gap`` of path length cut off (the path then ends
    ``gap`` before its last point)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if gap <= 0 or len(pts) < 2:
        return pts
    remaining = float(gap)
    out = [p for p in pts]
    while len(out) > 1:
        seg = out[-1] - out[-2]
        length = float(np.linalg.norm(seg))
        if length > remaining:
            out[-1] = out[-1] - seg / length * remaining
            return np.array(out)
        remaining -= length
        out.pop()
    return np.array(out)


def trail_geometry(points: Any, *, fade: tuple[float, float] = TRAIL_FADE,
                   arrow_length: float | None = None,
                   radius_ratio: float = 0.42) -> dict[str, Any]:
    """Pure-numpy pieces of a trail through ``points`` (N x 3, oldest first): ``points``
    (consecutive duplicates removed), ``segments`` (M x 2 x 3), ``alphas`` (M, ``fade[0]`` at
    the oldest segment to ``fade[1]`` at the newest), ``length`` (path length), ``direction``
    (unit vector of the last move) and ``arrow`` (triangles, K x 3 x 3, of a cone whose tip is
    the last point; ``arrow_length`` default 25 % of the path, at most the last
    segment x 3). Empty pieces when the path has fewer than two distinct points."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) > 1:
        keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-12]
        pts = pts[keep]
    out: dict[str, Any] = {"points": pts, "segments": np.zeros((0, 2, 3)),
                           "alphas": np.zeros(0), "length": 0.0, "direction": None,
                           "arrow": np.zeros((0, 3, 3)), "arrow_length": 0.0}
    if len(pts) < 2:
        return out
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    total = float(lengths.sum())
    m = len(segments)
    alphas = np.linspace(fade[0], fade[1], m) if m > 1 else np.array([fade[1]])
    direction = segments[-1, 1] - segments[-1, 0]
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    size = arrow_length if arrow_length is not None else min(0.25 * total, 3.0 * lengths[-1])
    size = float(max(size, 0.0))
    tip = pts[-1]
    base = tip - direction * size
    radius = float(radius_ratio) * size
    helper = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(direction, helper)
    u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(direction, u)
    angles = np.linspace(0, 2 * np.pi, 13)
    rim = base + radius * (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v))
    tris = [np.array([tip, rim[k], rim[k + 1]]) for k in range(12)]
    tris += [np.array([base, rim[k + 1], rim[k]]) for k in range(12)]
    out.update(segments=segments, alphas=alphas, length=total, direction=direction,
               arrow=np.array(tris) if size > 0 else np.zeros((0, 3, 3)), arrow_length=size)
    return out


# --------------------------------------------------------------------------------------
# Backends


class _PyVistaBadges:
    """Camera badges as a screen overlay on a PyVista renderer: a filled circle in the
    camera colour (outline ring when needed, red ring when flagged), the number, and an
    optional label beside it. Repositioned before every render (:func:`layout_badges`)."""

    def __init__(self, plotter: Any, theme: str) -> None:
        from vtkmodules.vtkCommonCore import vtkCommand
        self.p = plotter
        self.theme = theme
        self.entries: dict[str, dict[str, Any]] = {}
        self.gutter = 0.0
        #: Colour bars on the plotter: badges and their labels stay left of each bar's ticks.
        self.bar_actors: list[Any] = []
        self.positions: dict[str, tuple[float, float]] = {}
        #: World point the labels turn away from (the target), set by the scene.
        self.centre_world: np.ndarray | None = None
        renderer = plotter.renderer
        self._observer = renderer.AddObserver(vtkCommand.StartEvent, self._before_render)

    def _scale(self) -> float:
        w, h = self.p.window_size
        return float(np.clip(min(w, h) / 700.0, 0.9, 2.0))

    def _disk(self, radius: float, colour: str) -> Any:
        from vtkmodules.vtkFiltersSources import vtkRegularPolygonSource
        from vtkmodules.vtkRenderingCore import vtkActor2D, vtkPolyDataMapper2D
        src = vtkRegularPolygonSource()
        src.SetNumberOfSides(48)
        src.SetRadius(radius)
        src.SetCenter(0, 0, 0)
        src.GeneratePolygonOn()
        src.GeneratePolylineOff()
        mapper = vtkPolyDataMapper2D()
        mapper.SetInputConnection(src.GetOutputPort())
        actor = vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(_rgb(colour))
        actor.PickableOff()
        return actor, src

    def _text(self, size: int, colour: str, *, centred: bool, bold: bool,
              background: str | None = None) -> Any:
        from vtkmodules.vtkRenderingCore import vtkTextActor
        actor = vtkTextActor()
        prop = actor.GetTextProperty()
        prop.SetFontSize(int(size))
        prop.SetBold(bool(bold))
        prop.SetFontFamilyToArial()
        prop.SetColor(_rgb(colour))
        if centred:
            prop.SetJustificationToCentered()
        else:
            prop.SetJustificationToLeft()
        prop.SetVerticalJustificationToCentered()
        if background is not None:
            prop.SetBackgroundColor(_rgb(background))
            prop.SetBackgroundOpacity(0.72)
        actor.PickableOff()
        return actor

    def add(self, name: str, index: int, anchor: np.ndarray, label: str | None) -> None:
        self.remove(name)
        k = self._scale()
        r = BADGE_RADIUS_PX * k
        t = stage_tokens(self.theme)
        ring, ring_src = self._disk(r + 1.6 * k, t["outline"])
        fill, fill_src = self._disk(r, camera_fill(index, self.theme))
        number = self._text(int(round(13 * k)), badge_text_colour(index, self.theme),
                            centred=True, bold=True)
        number.SetInput(camera_badge_label(index))
        extra = None
        if label:  # no box: the label sits outward, away from the target (UX2-13)
            # UX6-1: a stage-coloured box behind the value, so lines and ticks never run under it
            extra = self._text(int(round(12 * k)), t["text"], centred=False, bold=False,
                               background=t["stage"])
            extra.SetInput(str(label))
        renderer = self.p.renderer
        for actor in (ring, fill, number, extra):
            if actor is not None:
                renderer.AddViewProp(actor)
        ring.SetVisibility(camera_needs_outline(index, self.theme))
        self.entries[name] = {"index": int(index), "anchor": np.asarray(anchor, float),
                              "label": label, "ring": ring, "fill": fill, "number": number,
                              "extra": extra, "radius": r, "ring_src": ring_src,
                              "fill_src": fill_src, "flagged": False, "highlight": False}

    def move(self, name: str, anchor: np.ndarray) -> None:
        if name in self.entries:
            self.entries[name]["anchor"] = np.asarray(anchor, float)

    def remove(self, name: str) -> None:
        entry = self.entries.pop(name, None)
        if entry is None:
            return
        for key in ("ring", "fill", "number", "extra"):
            if entry[key] is not None:
                try:
                    self.p.renderer.RemoveViewProp(entry[key])
                except Exception:  # noqa: BLE001
                    pass

    def clear(self) -> None:
        for name in list(self.entries):
            self.remove(name)

    def set_theme(self, theme: str) -> None:
        self.theme = theme
        t = stage_tokens(theme)
        for entry in self.entries.values():
            entry["ring"].GetProperty().SetColor(_rgb(t["outline"]))
            entry["ring"].SetVisibility(camera_needs_outline(entry["index"], theme)
                                        or entry["highlight"] or entry["flagged"])
            entry["fill"].GetProperty().SetColor(_rgb(camera_fill(entry["index"], theme)))
            entry["number"].GetTextProperty().SetColor(
                _rgb(badge_text_colour(entry["index"], theme)))
            if entry["extra"] is not None:
                prop = entry["extra"].GetTextProperty()
                prop.SetColor(_rgb(t["text"]))

    def mark(self, name: str, *, highlight: bool = False, flagged: bool = False) -> None:
        entry = self.entries.get(name)
        if entry is None:
            return
        t = stage_tokens(self.theme)
        entry["highlight"], entry["flagged"] = bool(highlight), bool(flagged)
        colour = t["bad"] if flagged else (t["badge_edge"] if highlight else t["outline"])
        entry["ring"].GetProperty().SetColor(_rgb(colour))
        grow = 2.6 if (flagged or highlight) else 1.6
        entry["ring_src"].SetRadius(entry["radius"] + grow * self._scale())
        entry["ring"].SetVisibility(flagged or highlight
                                    or camera_needs_outline(entry["index"], self.theme))

    def _before_render(self, *_args: Any) -> None:
        if not self.entries:
            return
        try:
            self.layout()
        except Exception:  # noqa: BLE001 - a badge must never break rendering
            pass

    def layout(self) -> dict[str, tuple[float, float]]:
        from vtkmodules.vtkRenderingCore import vtkCoordinate
        renderer = self.p.renderer
        w, h = renderer.GetSize()
        if w <= 1 or h <= 1:
            return {}
        names = list(self.entries)
        coord = vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        pts = []
        for name in names:
            coord.SetValue(*self.entries[name]["anchor"])
            pts.append(coord.GetComputedDoubleDisplayValue(renderer))
        pts = np.asarray(pts, dtype=float)
        k = self._scale()
        widths = [(len(str(self.entries[n]["label"])) * 7.2 * k + 8) if self.entries[n]["label"]
                  else 0.0 for n in names]
        radii = [self.entries[n]["radius"] + 2.6 * k for n in names]
        centre = (w / 2, h / 2)
        if self.centre_world is not None:
            coord.SetValue(*np.asarray(self.centre_world, dtype=float))
            centre = tuple(coord.GetComputedDoubleDisplayValue(renderer))[:2]
        # A label turns away from the target, so it never lies on the camera's own glyph.
        sides = [1 if x >= centre[0] else -1 for x, _y in pts[:, :2]]
        # UX6-1: the badges and their labels end left of the colour bar's ticks (its left
        # edge less a margin), not at the gutter's edge, which the bar may sit inside
        right = w * (1 - self.gutter)
        for bar in self.bar_actors:
            try:
                right = min(right, float(bar.GetPosition()[0]) * w - 10.0 * k)
            except Exception:  # noqa: BLE001 - a bar without a position: keep the gutter
                pass
        laid = layout_badges(pts[:, :2], radii, (0, 0, right, h),
                             centre=centre, label_widths=widths, label_sides=sides)
        self.positions = {}
        for name, (x, y), side in zip(names, laid, sides):
            entry = self.entries[name]
            for key in ("ring", "fill", "number"):
                entry[key].SetPosition(float(x), float(y))
            if entry["extra"] is not None:
                prop = entry["extra"].GetTextProperty()
                if side > 0:
                    prop.SetJustificationToLeft()
                else:
                    prop.SetJustificationToRight()
                entry["extra"].SetPosition(float(x + side * (entry["radius"] + 4 * k)), float(y))
            self.positions[name] = (float(x), float(y))
        return dict(self.positions)


class _PyVistaImpl:
    """Primitives on a pyvista Plotter (QtInteractor or off-screen Plotter)."""

    kind = "pyvista"

    def __init__(self, plotter: Any, theme: str) -> None:
        self.p = plotter
        self.items: dict[str, dict[str, Any]] = {}
        self.theme = resolve_theme(theme)
        self.axis_target: np.ndarray | None = None
        self.badges = _PyVistaBadges(plotter, self.theme)
        self.set_theme(self.theme)
        samples = perf.msaa_samples()
        try:  # MSAA keeps point sizes in screen pixels (SSAA would shrink them)
            if samples > 0:
                self.p.enable_anti_aliasing("msaa", multi_samples=samples)
            else:  # reduced settings: FXAA is a cheap post-process
                self.p.enable_anti_aliasing("fxaa")
        except Exception:
            pass

    # theme
    def set_theme(self, theme: str) -> None:
        self.theme = resolve_theme(theme)
        t = stage_tokens(self.theme)
        self.p.set_background(t["stage"])
        for item in self.items.values():
            if item["kind"] == "camera":
                rgb = _rgb(camera_fill(item["index"], self.theme))
                for actor in item["actors"]:
                    actor.GetProperty().SetColor(rgb)
            elif item["kind"] == "trail":
                self._style_trail(item)
            if item.get("bar_actor") is not None:
                try:
                    item["bar_actor"].GetTitleTextProperty().SetColor(_rgb(t["text"]))
                    item["bar_actor"].GetLabelTextProperty().SetColor(_rgb(t["text"]))
                except Exception:  # noqa: BLE001
                    pass
        self.badges.set_theme(self.theme)
        self.render()

    def render(self) -> None:
        start = time.perf_counter()
        try:
            self.p.render()
        except Exception:
            pass
        perf.note_render_time(time.perf_counter() - start)

    # cameras (each public primitive adds its meshes with render=False and renders once)
    def add_camera(self, name, index, R, t, scale, telecentric, label, axis):
        import pyvista as pv
        verts, faces, edges = camera_glyph(R, t, scale, telecentric)
        colour = camera_fill(index, self.theme)
        surf = pv.PolyData(verts, faces=np.hstack([[len(f)] + list(f) for f in faces]))
        lines = pv.PolyData(verts, lines=np.hstack([[2, i, j] for i, j in edges]))
        a1 = self.p.add_mesh(surf, color=colour, opacity=CAMERA_BODY_OPACITY,
                             smooth_shading=False, show_edges=False, name=f"{name}:surf",
                             pickable=False, lighting=False, render=False)  # flat colour
        a2 = self.p.add_mesh(lines, color=_edge_colour(index, self.theme), line_width=2.2,
                             name=f"{name}:edges", pickable=False, lighting=False, render=False)
        item = {"kind": "camera", "index": int(index), "surf": surf, "lines": lines,
                "actors": [a1, a2], "scale": scale, "telecentric": telecentric,
                "label": label, "pose": as_pose(R, t), "axis": None,
                "verts": verts}
        if axis:
            segment = camera_axis_segment(R, t, self._axis_length(R, t, scale))
            mesh = pv.PolyData(segment, lines=np.array([2, 0, 1]))
            a3 = self.p.add_mesh(mesh, color=colour, line_width=1.6, opacity=0.8,
                                 name=f"{name}:axis", pickable=False, lighting=False,
                                 render=False)
            item["axis"] = mesh
            item["actors"].append(a3)
        self.items[name] = item
        self.badges.add(name, index, _badge_anchor(R, t, scale), label)
        self.render()

    def _axis_length(self, R, t, scale) -> float:
        if self.axis_target is None:
            return 2.5 * scale
        return axis_length_to(R, t, self.axis_target, minimum=1.2 * scale)

    def update_camera(self, name, R, t):
        item = self.items[name]
        verts, _, _ = camera_glyph(R, t, item["scale"], item["telecentric"])
        item["surf"].points = verts
        item["lines"].points = verts
        item["verts"] = verts
        item["pose"] = as_pose(R, t)
        if item["axis"] is not None:
            item["axis"].points = camera_axis_segment(R, t, self._axis_length(R, t, item["scale"]))
        self.badges.move(name, _badge_anchor(R, t, item["scale"]))

    def refresh_axes(self) -> None:
        for item in self.items.values():
            if item["kind"] == "camera" and item["axis"] is not None:
                R, t = item["pose"]
                item["axis"].points = camera_axis_segment(R, t,
                                                          self._axis_length(R, t, item["scale"]))

    # boards
    def add_board(self, name, corners, colour, scalars, cmap, clim, opacity, point_size,
                  log_scale):
        import pyvista as pv
        corners = np.asarray(corners, dtype=float).reshape(-1, 3)
        cmap = _resolve_cmap(cmap, self.theme)
        hull = _hull_order(corners)
        face = pv.PolyData(corners[hull], faces=np.hstack([[len(hull)], np.arange(len(hull))]))
        fill = colour
        per_corner = scalars is not None and np.ndim(scalars) > 0 and np.size(scalars) > 1
        if scalars is not None and not per_corner:
            fill = _colour_for_scalar(float(np.ravel(scalars)[0]), cmap, clim, log_scale)
        t = stage_tokens(self.theme)
        a1 = self.p.add_mesh(face, color=fill or t["muted"], opacity=opacity,
                             name=f"{name}:face", pickable=False, render=False)
        outline = pv.PolyData(corners[hull], lines=np.hstack(
            [[len(hull) + 1], np.arange(len(hull)), [0]]))
        a2 = self.p.add_mesh(outline, color=fill or t["muted"], line_width=1.5,
                             name=f"{name}:outline", pickable=False, render=False)
        cloud = pv.PolyData(corners)
        kwargs = {}
        if per_corner:
            cloud["scalars"] = np.asarray(scalars, dtype=float).ravel()
            kwargs = dict(scalars="scalars", cmap=cmap, clim=clim, show_scalar_bar=False,
                          log_scale=bool(log_scale))
        else:
            kwargs = dict(color=fill or t["muted"])
        a3 = self.p.add_mesh(cloud, render_points_as_spheres=True, point_size=point_size,
                             name=f"{name}:pts", pickable=False, render=False, **kwargs)
        self.items[name] = {"kind": "board", "face": face, "outline": outline, "cloud": cloud,
                            "hull": hull, "actors": [a1, a2, a3], "per_corner": per_corner,
                            "cmap": cmap, "clim": clim, "log_scale": log_scale,
                            "verts": corners}
        self.render()

    def update_board(self, name, corners, scalars=None):
        item = self.items[name]
        corners = np.asarray(corners, dtype=float).reshape(-1, 3)
        item["face"].points = corners[item["hull"]]
        item["outline"].points = corners[item["hull"]]
        item["cloud"].points = corners
        item["verts"] = corners
        if scalars is not None:
            if item["per_corner"]:
                item["cloud"]["scalars"] = np.asarray(scalars, dtype=float).ravel()
            else:
                rgb = _rgb(_colour_for_scalar(float(np.ravel(scalars)[0]), item["cmap"],
                                              item["clim"], item["log_scale"]))
                for actor in item["actors"]:
                    actor.GetProperty().SetColor(rgb)

    # points
    def add_points(self, name, xyz, scalars, cmap, clim, colour, size, scalar_bar, log_scale):
        import pyvista as pv
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        cloud = pv.PolyData(xyz)
        kwargs: dict[str, Any] = {}
        bar_actor = None
        if scalars is not None:
            cmap = _resolve_cmap(cmap, self.theme)
            values = np.asarray(scalars, dtype=float).ravel()
            cloud["scalars"] = values
            t = stage_tokens(self.theme)
            kwargs = dict(scalars="scalars", cmap=cmap, clim=clim, show_scalar_bar=False,
                          log_scale=bool(log_scale))
            if scalar_bar:
                lo, hi = clim if clim is not None else (float(np.nanmin(values)),
                                                        float(np.nanmax(values)))
                ticks = _nice_ticks(lo, hi, bool(log_scale))
                kwargs["show_scalar_bar"] = True
                kwargs["scalar_bar_args"] = dict(
                    title=str(scalar_bar), color=t["text"], vertical=True,
                    n_labels=len(ticks), fmt=_tick_format(ticks), title_font_size=16,
                    label_font_size=14, bold=False, position_x=1 - SCALAR_BAR_GUTTER + 0.04,
                    position_y=0.22, height=0.56, width=0.045, font_family="arial")
                self.badges.gutter = SCALAR_BAR_GUTTER
        else:
            kwargs = dict(color=colour or stage_tokens(self.theme)["accent"])
        actor = self.p.add_mesh(cloud, render_points_as_spheres=True, point_size=size,
                                name=f"{name}:pts", pickable=False, ambient=0.25, render=False,
                                **kwargs)
        if scalars is not None and scalar_bar:
            try:
                bar_actor = self.p.scalar_bars[str(scalar_bar)]
                self.badges.bar_actors.append(bar_actor)  # badges keep clear of its ticks
                from vtkmodules.vtkCommonCore import vtkDoubleArray
                labels = vtkDoubleArray()
                for tick in ticks:  # round ticks (0.1 0.3 1 3 ... on the log scale)
                    labels.InsertNextValue(float(tick))
                bar_actor.SetCustomLabels(labels)
                bar_actor.SetUseCustomLabels(True)
                bar_actor.SetLabelFormat(_tick_format(ticks) if not log_scale else "%g")
                # Room between the title ("px") and the top tick label (UX2-13).
                bar_actor.SetVerticalTitleSeparation(int(round(14 * self.badges._scale())))
                bar_actor.SetTitleRatio(0.6)
            except Exception:  # noqa: BLE001
                bar_actor = None
        self.items[name] = {"kind": "points", "cloud": cloud, "actors": [actor],
                            "verts": xyz, "bar_actor": bar_actor,
                            "bar_title": str(scalar_bar) if scalar_bar else None}
        self.render()

    def update_points(self, name, xyz, scalars=None):
        item = self.items[name]
        item["cloud"].points = np.asarray(xyz, dtype=float).reshape(-1, 3)
        item["verts"] = item["cloud"].points
        if scalars is not None:
            item["cloud"]["scalars"] = np.asarray(scalars, dtype=float).ravel()

    def add_face_tags(self, name, groups):
        """F1..F6 tags: a rounded box in the face's shade with its label, always on top."""
        w, h = self.p.window_size
        k = float(np.clip(min(w, h) / 700.0, 0.9, 2.0))
        actors = []
        tags = face_tag_layout(groups)
        for i, (face, xyz) in enumerate(tags):
            actor_name = f"{name}:tag{i}"
            self.p.add_point_labels(
                np.asarray(xyz, dtype=float).reshape(1, 3), [face_label(face)],
                name=actor_name, show_points=False, always_visible=True, pickable=False,
                shape="rounded_rect", shape_color=face_colour(face, self.theme),
                shape_opacity=1.0, text_color=face_text_colour(face, self.theme),
                font_size=int(round(12 * k)), bold=True, margin=int(round(4 * k)),
                render=False)
            actors.append(actor_name)
        self.items[name] = {"kind": "tags", "actors": actors,
                            "verts": np.vstack([xyz for _, xyz in tags]) if tags else []}

    # rays
    def add_rays(self, name, origins, ends, colour, scalars, cmap, clim, width, opacity,
                 log_scale):
        import pyvista as pv
        mesh = _ray_mesh(pv, origins, ends)
        kwargs: dict[str, Any]
        if scalars is not None:
            mesh.cell_data["scalars"] = np.asarray(scalars, dtype=float).ravel()
            kwargs = dict(scalars="scalars", cmap=_resolve_cmap(cmap, self.theme), clim=clim,
                          show_scalar_bar=False, log_scale=bool(log_scale))
        else:
            kwargs = dict(color=colour or stage_tokens(self.theme)["accent"])
        actor = self.p.add_mesh(mesh, line_width=width, opacity=opacity, name=f"{name}:rays",
                                pickable=False, lighting=False, render=False, **kwargs)
        self.items[name] = {"kind": "rays", "mesh": mesh, "actors": [actor],
                            "verts": np.asarray(mesh.points)}
        self.render()

    def update_rays(self, name, origins, ends, scalars=None):
        item = self.items[name]
        pts = np.empty((2 * len(origins), 3))
        pts[0::2] = np.asarray(origins, dtype=float).reshape(-1, 3)
        pts[1::2] = np.asarray(ends, dtype=float).reshape(-1, 3)
        if pts.shape == item["mesh"].points.shape:
            item["mesh"].points = pts
            item["verts"] = pts
        if scalars is not None:
            item["mesh"].cell_data["scalars"] = np.asarray(scalars, dtype=float).ravel()

    # trails (shared bookkeeping)
    def _trail_colour(self, spec):
        if spec["index"] is not None:
            return camera_fill(spec["index"], self.theme)
        return spec["colour"] or stage_tokens(self.theme)["accent"]

    def _trail_scene_size(self, extra):
        others = [np.asarray(item["verts"]) for item in self.items.values()
                  if item["kind"] != "trail" and len(item.get("verts", []))]
        pts = np.vstack(others + [extra]) if others else extra
        if not len(pts):
            return 1.0
        return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0

    def _trail_pieces(self, points, spec):
        return trail_pieces(points, spec, self._trail_scene_size(points))

    def add_trail(self, name, points, spec):
        self.items[name] = {"kind": "trail", "index": spec["index"], "spec": spec,
                            "verts": np.asarray(points), "visible": False, "actors": []}
        self._build_trail(name)

    def update_trail(self, name, points, *, render=True):
        self.items[name]["verts"] = np.asarray(points)
        self._build_trail(name, render=render)  # the PyVista impl renders here unless told not to

    def _style_trail(self, item):
        if "name" in item:
            self._build_trail(item["name"], render=False)  # set_theme renders once after

    def _build_trail(self, name, *, render=True):
        """Shape a trail's meshes from its current points.

        The actors are kept between frames: a piece that already exists gets its new geometry
        on its mapper and is shown again, so playback does not remove and re-add five actors
        per frame. New pieces are added with ``render=False``; one render follows unless the
        caller batches several updates (``render=False``) and renders itself.
        """
        import pyvista as pv
        item = self.items[name]
        item["name"] = name
        spec = item["spec"]
        geom, visible, ghost_visible = self._trail_pieces(item["verts"], spec)
        # A single point (playback position 0) has no segment to draw: hide the trail rather
        # than leave the previous frame's geometry on screen.
        visible = bool(visible) and len(geom["points"]) >= 2
        item["visible"] = bool(visible)
        item["geom"] = geom
        if not visible:
            for actor in item.get("trail_actors", {}).values():
                actor.SetVisibility(False)
            if render:
                self.render()
            return
        colour = self._trail_colour(spec)
        rgb = _rgb(colour)
        pts = geom["points"]
        line = pv.PolyData(pts, lines=np.hstack([[len(pts)], np.arange(len(pts))]))
        alphas = np.r_[geom["alphas"][0], geom["alphas"]]
        rgba = np.column_stack([np.tile(np.array(rgb), (len(pts), 1)), alphas])
        line["rgba"] = (rgba * 255).astype(np.uint8)
        self._trail_piece(item, "trail", line, scalars="rgba", rgba=True,
                          line_width=spec["width"], pickable=False, lighting=False,
                          render_lines_as_tubes=False)
        if spec["arrow"] and len(geom["arrow"]):
            tris = geom["arrow"].reshape(-1, 3)
            faces = np.hstack([[3, 3 * k, 3 * k + 1, 3 * k + 2] for k in range(len(geom["arrow"]))])
            self._trail_piece(item, "arrow", pv.PolyData(tris, faces=faces), colour=rgb,
                              color=colour, pickable=False, lighting=False)
        else:
            self._hide_trail_piece(item, "arrow")
        if spec["ghost"] is not None and ghost_visible:
            verts, faces, edges = spec["ghost"]
            surf = pv.PolyData(verts, faces=np.hstack([[len(f)] + list(f) for f in faces]))
            lines = pv.PolyData(verts, lines=np.hstack([[2, i, j] for i, j in edges]))
            self._trail_piece(item, "ghostsurf", surf, colour=rgb, opacity=TRAIL_GHOST_OPACITY[0],
                              color=colour, pickable=False, lighting=False)
            self._trail_piece(item, "ghostedges", lines, colour=rgb,
                              opacity=TRAIL_GHOST_OPACITY[1], color=colour, line_width=1.2,
                              pickable=False, lighting=False)
            ring = pv.Sphere(radius=max(geom["arrow_length"] * 0.28, 1e-6), center=pts[0],
                             theta_resolution=perf.scaled_resolution(16),
                             phi_resolution=perf.scaled_resolution(12))
            self._trail_piece(item, "start", ring, colour=rgb, opacity=0.45, color=colour,
                              style="wireframe", line_width=1.0, pickable=False, lighting=False)
        else:
            for key in ("ghostsurf", "ghostedges", "start"):
                self._hide_trail_piece(item, key)
        if render:
            self.render()

    def _trail_piece(self, item, key, mesh, *, colour=None, opacity=None, **add_kwargs):
        """One trail mesh: reuse its actor when it exists, else add it (render=False)."""
        pieces = item.setdefault("trail_actors", {})
        actor = pieces.get(key)
        if actor is None:
            actor = self.p.add_mesh(mesh, name=f"{item['name']}:{key}", render=False,
                                    **add_kwargs)
            pieces[key] = actor
        else:
            actor.GetMapper().SetInputData(mesh)
            actor.SetVisibility(True)
        if colour is not None:  # the theme may have changed the colour since it was added
            actor.GetProperty().SetColor(*colour)
        if opacity is not None:
            actor.GetProperty().SetOpacity(opacity)
        return actor

    def _hide_trail_piece(self, item, key):
        actor = item.get("trail_actors", {}).get(key)
        if actor is not None:
            actor.SetVisibility(False)

    def mark_camera(self, name, highlight, flagged):
        self.badges.mark(name, highlight=highlight, flagged=flagged)

    def remove(self, name):
        item = self.items.pop(name, None)
        if item is None:
            return
        self.badges.remove(name)
        self.badges.bar_actors = [b for b in self.badges.bar_actors
                                  if b is not item.get("bar_actor")]
        if item.get("bar_title"):
            try:
                self.p.remove_scalar_bar(item["bar_title"], render=False)
            except Exception:  # noqa: BLE001
                pass
        for suffix in ("surf", "edges", "axis", "face", "outline", "pts", "rays", "trail",
                       "arrow", "ghostsurf", "ghostedges", "start"):
            try:
                self.p.remove_actor(f"{name}:{suffix}", render=False)
            except Exception:
                pass
        for actor_name in item.get("actors", []):  # face tags (kind "tags")
            try:
                self.p.remove_actor(actor_name, render=False)
            except Exception:
                pass

    def clear(self):
        for name in list(self.items):
            self.remove(name)
        self.badges.clear()
        self.badges.gutter = 0.0
        self.badges.bar_actors = []
        self.p.clear()
        self.set_theme(self.theme)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        allv = [np.asarray(item["verts"]) for item in self.items.values()
                if len(item.get("verts", []))]
        if not allv:
            b = np.asarray(self.p.bounds, dtype=float)
            return b[0::2], b[1::2]
        pts = np.vstack(allv)
        return pts.min(axis=0), pts.max(axis=0)

    def all_points(self) -> np.ndarray:
        allv = [np.asarray(item["verts"]) for item in self.items.values()
                if len(item.get("verts", []))]
        return np.vstack(allv) if allv else np.zeros((1, 3))

    def aspect(self) -> float:
        w, h = self.p.window_size
        return float(w) / max(float(h), 1.0)

    def set_view(self, azimuth, elevation, up, zoom, focus, distance, view_angle=30.0):
        lo, hi = self.bounds()
        centre = np.asarray(focus, dtype=float) if focus is not None else (lo + hi) / 2
        radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-9)
        direction = _view_direction(azimuth, elevation, up)
        dist = distance if distance is not None else radius * 2.6
        self.p.camera_position = [tuple(centre + dist * direction), tuple(centre),
                                  tuple(np.asarray(up, dtype=float))]
        self.p.camera.view_angle = float(view_angle)
        if zoom and zoom != 1.0:
            self.p.camera.zoom(zoom)
        self.p.renderer.ResetCameraClippingRange()
        self.render()

    def _projected(self) -> np.ndarray:
        """Normalised device coordinates (x, y in -1..1) of the scene's points."""
        pts = self.all_points()
        if len(pts) > 4000:
            pts = pts[np.linspace(0, len(pts) - 1, 4000).astype(int)]
        matrix = self.p.camera.GetCompositeProjectionTransformMatrix(self.aspect(), -1, 1)
        m = np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
        h = np.c_[pts, np.ones(len(pts))] @ m.T
        h = h[h[:, 3] > 1e-12]
        return h[:, :2] / h[:, 3:4]

    def tighten(self, gutter: float = 0.0, fill: float = 0.9) -> None:
        """UX3-14: after fitting, zoom and re-centre so the projected scene fills ``fill`` of
        the limiting dimension (no 20-30 % blank bands), keeping ``gutter`` (fraction of the
        width) free on the right for the colour bar, which is then put just right of the
        drawn scene (UX3-13), not at the far edge."""
        cam = self.p.camera
        for _ in range(2):
            ndc = self._projected()
            if not len(ndc):
                return
            x0, y0 = ndc.min(axis=0)
            x1, y1 = ndc.max(axis=0)
            width, height = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
            zoom = float(np.clip(fill * min(2 * (1 - gutter) / width, 2 / height), 0.5, 3.0))
            dx, dy = (x0 + x1) / 2 + gutter, (y0 + y1) / 2
            position = np.array(cam.GetPosition())
            focal = np.array(cam.GetFocalPoint())
            up = np.array(cam.GetViewUp())
            up /= max(np.linalg.norm(up), 1e-12)
            forward = focal - position
            forward /= max(np.linalg.norm(forward), 1e-12)
            right = np.cross(forward, up)
            half_h = cam.GetDistance() * np.tan(np.radians(cam.GetViewAngle() / 2))
            shift = dx * half_h * self.aspect() * right + dy * half_h * up
            cam.SetPosition(*(position + shift))
            cam.SetFocalPoint(*(focal + shift))
            if abs(zoom - 1.0) > 0.02:
                cam.Zoom(zoom)
        self.p.renderer.ResetCameraClippingRange()
        self.render()  # lays the badges out at the new zoom (their positions are used below)
        ndc = self._projected()
        bars = [item["bar_actor"] for item in self.items.values()
                if item.get("bar_actor") is not None]
        if len(ndc) and bars:
            w = max(float(self.p.window_size[0]), 1.0)
            right_edge = (float(ndc[:, 0].max()) + 1) / 2
            if getattr(self.badges, "positions", None):
                # a badge's "(0.40)" label runs about 80 px right of its centre
                right_edge = max([right_edge] + [(x + 80.0 * self.badges._scale()) / w
                                                 for x, _y in self.badges.positions.values()])
            x = min(right_edge + 24.0 / w, 1 - 0.045 - 0.07)
            for bar in bars:
                try:
                    bar.SetPosition(x, bar.GetPosition()[1])
                except Exception:  # noqa: BLE001
                    pass
        self.render()

    def screenshot(self, path, scale=None):
        self.render()  # the badge overlay is laid out at the start of a render
        # ``scale`` multiplies the window size in pixels (a larger PNG of the same view)
        if scale is None:
            self.p.screenshot(str(path))
        else:
            self.p.screenshot(str(path), scale=int(scale))


def _ray_mesh(pv, origins, ends):
    origins = np.asarray(origins, dtype=float).reshape(-1, 3)
    ends = np.asarray(ends, dtype=float).reshape(-1, 3)
    pts = np.empty((2 * len(origins), 3))
    pts[0::2], pts[1::2] = origins, ends
    lines = np.hstack([[2, 2 * i, 2 * i + 1] for i in range(len(origins))]) \
        if len(origins) else np.empty(0, dtype=int)
    return pv.PolyData(pts, lines=lines)


def _static_badge_layer_class():
    """Built lazily so matplotlib is only imported when a static scene is made."""
    import matplotlib.text as mtext
    from matplotlib.artist import Artist
    from matplotlib.transforms import IdentityTransform
    from mpl_toolkits.mplot3d import proj3d

    class _BadgeLayer(Artist):
        """Badges of a static 3D scene, drawn last in screen space: never hidden by
        geometry, kept inside the axes and pushed apart (:func:`layout_badges`)."""

        zorder = 50

        def __init__(self, theme: str) -> None:
            super().__init__()
            self.theme = theme
            self.entries: dict[str, dict[str, Any]] = {}
            self.positions: dict[str, tuple[float, float]] = {}
            self.gutter = 0.0
            #: Pixels kept free at the figure's right edge (the colour bar and its ticks).
            self.reserve_px = 0.0
            #: Screen point the labels turn away from (the target), set by the scene.
            self.centre_world: np.ndarray | None = None
            #: Right-most pixel of badges and labels as last drawn (colour-bar placement).
            self.extent_x1: float | None = None

        def draw(self, renderer: Any) -> None:
            if not self.get_visible() or not self.entries or self.axes is None:
                return
            ax = self.axes
            names = list(self.entries)
            anchors = np.array([self.entries[n]["anchor"] for n in names])
            xs, ys, _ = proj3d.proj_transform(anchors[:, 0], anchors[:, 1], anchors[:, 2],
                                              ax.get_proj())
            pts = ax.transData.transform(np.column_stack([xs, ys]))
            dpi = self.figure.dpi / 72.0
            size = 10.0
            r = 0.9 * size * dpi
            widths = [len(str(self.entries[n]["label"])) * 0.6 * 9 * dpi + 10
                      if self.entries[n]["label"] else 0.0 for n in names]
            box = ax.bbox
            centre = ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
            if self.centre_world is not None:
                c = np.asarray(self.centre_world, dtype=float)
                cx, cy, _ = proj3d.proj_transform(c[0:1], c[1:2], c[2:3], ax.get_proj())
                centre = tuple(ax.transData.transform([[cx[0], cy[0]]])[0])
            # A label turns away from the target, so it never lies across the camera's own
            # glyph or viewing line (which point at the target).
            sides = [1 if x >= centre[0] else -1 for x, _y in pts]
            right = box.x1 - self.gutter * box.width
            if self.reserve_px:
                right = min(right, self.figure.bbox.x1 - self.reserve_px)
            laid = layout_badges(pts, r + 2, (box.x0, box.y0, right, box.y1),
                                 centre=centre, label_widths=widths, label_sides=sides)
            t = stage_tokens(self.theme)
            from pyCamSet.gui.education_style import _installed_fonts
            families = _installed_fonts()  # drawn outside the rc context: name the font
            self.positions = {}
            self.label_sides: dict[str, int] = {}
            extent = -np.inf
            for name, (x, y), side in zip(names, laid, sides):
                entry = self.entries[name]
                index = entry["index"]
                edge = (t["bad"] if entry["flagged"] else
                        t["badge_edge"] if entry["highlight"] else camera_edge(index, self.theme))
                fill = camera_fill(index, self.theme)
                width = 2.0 if (entry["flagged"] or entry["highlight"]) else (
                    1.2 if edge != "none" else 0.0)
                text = mtext.Text(x, y, camera_badge_label(index), ha="center", va="center",
                                  fontsize=size, fontweight="bold", fontfamily=families,
                                  color=badge_text_colour(index, self.theme),
                                  transform=IdentityTransform(),
                                  bbox=dict(boxstyle="circle,pad=0.28",
                                            facecolor=fill, edgecolor=edge,
                                            linewidth=width))
                text.set_figure(self.figure)
                text.draw(renderer)
                extent = max(extent, x + r)
                if entry["label"]:
                    extra = mtext.Text(x + side * (r + 5), y, str(entry["label"]),
                                       fontfamily=families,
                                       ha="left" if side > 0 else "right", va="center",
                                       fontsize=9, color=t["text"],
                                       transform=IdentityTransform(),
                                       bbox=dict(boxstyle="round,pad=0.15",
                                                 facecolor=t["stage"], edgecolor="none",
                                                 alpha=0.8))
                    extra.set_figure(self.figure)
                    extra.draw(renderer)
                    if side > 0:
                        extent = max(extent, extra.get_window_extent(renderer).x1)
                self.positions[name] = (float(x), float(y))
                self.label_sides[name] = int(side)
            self.extent_x1 = float(extent) if np.isfinite(extent) else None
            self.stale = False

    return _BadgeLayer


class _BadgeHandle:
    """What ``items[name]["badge"]`` holds in static mode (``get_text()`` -> "4")."""

    def __init__(self, layer: Any, name: str) -> None:
        self._layer, self._name = layer, name

    def get_text(self) -> str:
        entry = self._layer.entries.get(self._name)
        return camera_badge_label(entry["index"]) if entry else ""

    def get_label(self) -> str:
        entry = self._layer.entries.get(self._name)
        return str(entry["label"] or "") if entry else ""

    def remove(self) -> None:
        self._layer.entries.pop(self._name, None)


def _add_ragged_polygons(ax: Any, surf: Any, verts: Any) -> None:
    """Add a ``Poly3DCollection`` whose polygons have different vertex counts (a camera glyph:
    quads, a 12-sided lens face, a triangular tab), with the data limits taken from ``verts``.

    matplotlib (seen in 3.11.2) stores ragged polygons in an ``np.empty`` array padded to the
    longest polygon and masks the padding when drawing, but ``Axes3D.add_collection3d``
    autoscales from the whole array, padding included. That padding is uninitialised memory,
    so the limits picked up whatever the heap last held (values up to ~1e308 after other
    figures had been drawn): the view overflowed and ``set_zlim`` raised "Axis limits cannot
    be NaN or Inf", depending only on what ran before. Autoscale from the real vertices.
    """
    had_data = ax.has_data()
    ax.add_collection3d(surf, autolim=False)
    pts = np.asarray(verts, dtype=float).reshape(-1, 3)
    ax.auto_scale_xyz(pts[:, 0], pts[:, 1], pts[:, 2], had_data=had_data)


class _StaticImpl:
    """The same primitives drawn with matplotlib's 3D axes (no OpenGL needed)."""

    kind = "static"

    def __init__(self, figure: Any, theme: str, draw: Callable[[], None]) -> None:
        self.fig = figure
        self._draw = draw
        self.items: dict[str, dict[str, Any]] = {}
        self.theme = resolve_theme(theme)
        self.axis_target: np.ndarray | None = None
        self._view = dict(azimuth=-60.0, elevation=25.0, up=(0, 0, 1), zoom=1.0)
        #: Box proportions and fitted zoom from the last :meth:`set_view` (see :meth:`tighten`).
        self._fit: dict[str, Any] | None = None
        #: Playback throttling (see :meth:`render`); the clock is injectable for tests.
        self.animating = False
        self._last_draw = float("-inf")
        self._clock: Callable[[], float] = time.monotonic
        self._new_axes()

    def _new_axes(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(projection="3d")
        self.ax.set_axis_off()
        self.ax.computed_zorder = False  # badges stay on top of the glyphs
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.badges = _static_badge_layer_class()(self.theme)
        self.ax.add_artist(self.badges)
        self.set_theme(self.theme)

    def set_theme(self, theme):
        self.theme = resolve_theme(theme)
        t = stage_tokens(self.theme)
        self.fig.set_facecolor(t["stage"])
        self.ax.set_facecolor(t["stage"])
        self.badges.theme = self.theme
        for item in self.items.values():
            if item["kind"] == "camera":
                colour = camera_fill(item["index"], self.theme)
                item["lines"].set_color(colour)
                item["surf"].set_facecolor(colour)
                if item["axis"] is not None:
                    item["axis"].set_color(colour)
            elif item["kind"] == "trail":
                self._style_trail(item)
        self.render()

    def render(self):
        if self.animating:
            # Playback: a full matplotlib 3D draw is the expensive part, so at most one per
            # STATIC_DRAW_INTERVAL_MS. The stop of the tween draws the final frame.
            now = self._clock()
            if now - self._last_draw < perf.STATIC_DRAW_INTERVAL_MS / 1000.0:
                return
            self._last_draw = now
        self._draw()

    def set_animating(self, on: bool) -> None:
        """Start or end the throttling of :meth:`render` (no draw here: the caller renders)."""
        self.animating = bool(on)

    def _rc(self):
        """Create artists under the education figure style (fonts, colours)."""
        import matplotlib
        from pyCamSet.gui.education_style import figure_style
        style = figure_style(self.theme)
        style["figure.constrained_layout.use"] = False
        return matplotlib.rc_context(style)

    def add_camera(self, *args):
        with self._rc():
            self._add_camera(*args)

    def add_points(self, *args):
        with self._rc():
            self._add_points(*args)

    def _axis_length(self, R, t, scale) -> float:
        if self.axis_target is None:
            return 2.5 * scale
        return axis_length_to(R, t, self.axis_target, minimum=1.2 * scale)

    def _add_camera(self, name, index, R, t, scale, telecentric, label, axis):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
        verts, faces, edges = camera_glyph(R, t, scale, telecentric)
        if not np.isfinite(verts).all():
            raise ValueError(f"camera {name!r} cannot be drawn: its pose or glyph size is not "
                             f"finite (scale={scale!r})")
        colour = camera_fill(index, self.theme)
        surf = Poly3DCollection([verts[f] for f in faces], facecolors=colour,
                                alpha=CAMERA_BODY_OPACITY, edgecolors="none")
        lines = Line3DCollection([verts[[i, j]] for i, j in edges],
                                 colors=_edge_colour(index, self.theme), linewidths=1.4)
        _add_ragged_polygons(self.ax, surf, verts)
        self.ax.add_collection3d(lines)
        axis_line = None
        if axis:
            segment = camera_axis_segment(R, t, self._axis_length(R, t, scale))
            axis_line = Line3DCollection([segment], colors=colour, linewidths=1.3, alpha=0.85,
                                         linestyles=(0, (4, 2)))
            self.ax.add_collection3d(axis_line)
        self.badges.entries[name] = {"index": int(index), "label": label,
                                     "anchor": _badge_anchor(R, t, scale), "flagged": False,
                                     "highlight": False}
        self.items[name] = {"kind": "camera", "index": int(index), "surf": surf,
                            "lines": lines, "axis": axis_line,
                            "badge": _BadgeHandle(self.badges, name), "faces": faces,
                            "edges": edges, "scale": scale, "telecentric": telecentric,
                            "verts": verts, "pose": as_pose(R, t)}

    def update_camera(self, name, R, t):
        item = self.items[name]
        verts, faces, edges = camera_glyph(R, t, item["scale"], item["telecentric"])
        item["surf"].set_verts([verts[f] for f in faces])
        item["lines"].set_segments([verts[[i, j]] for i, j in edges])
        if item["axis"] is not None:
            item["axis"].set_segments([camera_axis_segment(R, t, self._axis_length(
                R, t, item["scale"]))])
        if name in self.badges.entries:
            self.badges.entries[name]["anchor"] = _badge_anchor(R, t, item["scale"])
        item["verts"] = verts
        item["pose"] = as_pose(R, t)

    def refresh_axes(self) -> None:
        for item in self.items.values():
            if item["kind"] == "camera" and item["axis"] is not None:
                R, t = item["pose"]
                item["axis"].set_segments([camera_axis_segment(R, t, self._axis_length(
                    R, t, item["scale"]))])

    # trails (shared bookkeeping)
    def _trail_colour(self, spec):
        if spec["index"] is not None:
            return camera_fill(spec["index"], self.theme)
        return spec["colour"] or stage_tokens(self.theme)["accent"]

    def _trail_scene_size(self, extra):
        others = [np.asarray(item["verts"]) for item in self.items.values()
                  if item["kind"] != "trail" and len(item.get("verts", []))]
        pts = np.vstack(others + [extra]) if others else extra
        if not len(pts):
            return 1.0
        return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0

    def _trail_pieces(self, points, spec):
        return trail_pieces(points, spec, self._trail_scene_size(points))

    def add_trail(self, name, points, spec):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
        item = {"kind": "trail", "index": spec["index"], "spec": spec,
                "verts": np.asarray(points), "visible": False}
        # Placeholders at the first point (an empty 3D collection cannot be added).
        p0 = np.asarray(points, dtype=float).reshape(-1, 3)
        p0 = p0[0] if len(p0) else np.zeros(3)
        item["line"] = Line3DCollection([np.array([p0, p0])], linewidths=spec["width"],
                                        capstyle="round")
        item["arrow"] = Poly3DCollection([np.array([p0, p0, p0])], edgecolors="none")
        self.ax.add_collection3d(item["line"])
        self.ax.add_collection3d(item["arrow"])
        item["ghost_surf"] = item["ghost_lines"] = item["start"] = None
        if spec["ghost"] is not None:
            verts, faces, edges = spec["ghost"]
            item["ghost_surf"] = Poly3DCollection([verts[f] for f in faces], edgecolors="none")
            item["ghost_lines"] = Line3DCollection([verts[[i, j]] for i, j in edges],
                                                   linewidths=1.0, linestyles=(0, (3, 2)))
            _add_ragged_polygons(self.ax, item["ghost_surf"], verts)
            self.ax.add_collection3d(item["ghost_lines"])
            p0 = np.asarray(points, dtype=float).reshape(-1, 3)[:1]
            item["start"] = self.ax.scatter(p0[:, 0], p0[:, 1], p0[:, 2], s=46,
                                            facecolors="none", linewidths=1.4,
                                            depthshade=False)
        self.items[name] = item
        self._update_trail_artists(item)

    def update_trail(self, name, points, *, render=True):
        item = self.items[name]
        item["verts"] = np.asarray(points)
        self._update_trail_artists(item)
        if render:
            self.render()

    def _update_trail_artists(self, item):
        geom, visible, ghost_visible = self._trail_pieces(item["verts"], item["spec"])
        item["geom"], item["visible"] = geom, bool(visible)
        item["line"].set_visible(bool(visible))
        item["arrow"].set_visible(bool(visible and item["spec"]["arrow"]))
        if visible:
            item["line"].set_segments(list(geom["segments"]))
            if len(geom["arrow"]):
                item["arrow"].set_verts(list(geom["arrow"]))
        if item["start"] is not None and len(geom["points"]):
            p0 = geom["points"][:1]
            item["start"]._offsets3d = (p0[:, 0], p0[:, 1], p0[:, 2])
        for key in ("ghost_surf", "ghost_lines", "start"):
            if item.get(key) is not None:
                item[key].set_visible(bool(ghost_visible))
        self._style_trail(item)

    def _style_trail(self, item):
        from matplotlib.colors import to_rgba
        colour = self._trail_colour(item["spec"])
        geom = item.get("geom") or {"alphas": np.zeros(0)}
        item["line"].set_color([to_rgba(colour, float(a)) for a in geom["alphas"]] or colour)
        item["arrow"].set_facecolor(colour)
        if item.get("ghost_surf") is not None:
            item["ghost_surf"].set_facecolor(to_rgba(colour, TRAIL_GHOST_OPACITY[0]))
            item["ghost_lines"].set_color(to_rgba(colour, TRAIL_GHOST_OPACITY[1]))
        if item.get("start") is not None:
            item["start"].set_edgecolor(to_rgba(colour, 0.85))

    def mark_camera(self, name, highlight, flagged):
        entry = self.badges.entries.get(name)
        if entry is not None:
            entry["highlight"], entry["flagged"] = bool(highlight), bool(flagged)

    def _norm(self, clim, log_scale):
        if log_scale and clim is not None and clim[0] > 0:
            from matplotlib.colors import LogNorm
            return LogNorm(vmin=clim[0], vmax=clim[1], clip=True)
        from matplotlib.colors import Normalize
        return Normalize(vmin=None if clim is None else clim[0],
                         vmax=None if clim is None else clim[1])

    def add_board(self, name, corners, colour, scalars, cmap, clim, opacity, point_size,
                  log_scale):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        corners = np.asarray(corners, dtype=float).reshape(-1, 3)
        cmap = _resolve_cmap(cmap, self.theme)
        hull = _hull_order(corners)
        t = stage_tokens(self.theme)
        per_corner = scalars is not None and np.ndim(scalars) > 0 and np.size(scalars) > 1
        fill = colour
        if scalars is not None and not per_corner:
            fill = _colour_for_scalar(float(np.ravel(scalars)[0]), cmap, clim, log_scale)
        fill = fill or t["muted"]
        face = Poly3DCollection([corners[hull]], facecolors=fill, alpha=opacity,
                                edgecolors=fill, linewidths=1.0)
        self.ax.add_collection3d(face)
        if per_corner:
            pts = self.ax.scatter(*corners.T, c=np.asarray(scalars, float).ravel(), cmap=cmap,
                                  norm=self._norm(clim, log_scale), s=point_size * 1.5,
                                  depthshade=False, edgecolors=t["point_edge"],
                                  linewidths=0.3)
        else:
            pts = self.ax.scatter(*corners.T, color=fill, s=point_size, depthshade=False)
        self.items[name] = {"kind": "board", "face": face, "pts": pts, "hull": hull,
                            "per_corner": per_corner, "cmap": cmap, "clim": clim,
                            "log_scale": log_scale, "verts": corners}

    def update_board(self, name, corners, scalars=None):
        item = self.items[name]
        corners = np.asarray(corners, dtype=float).reshape(-1, 3)
        item["face"].set_verts([corners[item["hull"]]])
        item["pts"]._offsets3d = tuple(corners.T)
        if scalars is not None:
            if item["per_corner"]:
                item["pts"].set_array(np.asarray(scalars, float).ravel())
            else:
                colour = _colour_for_scalar(float(np.ravel(scalars)[0]), item["cmap"],
                                            item["clim"], item["log_scale"])
                item["face"].set_facecolor(colour)
                item["pts"].set_color(colour)
        item["verts"] = corners

    def _add_points(self, name, xyz, scalars, cmap, clim, colour, size, scalar_bar, log_scale):
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        t = stage_tokens(self.theme)
        if scalars is not None:
            cmap = _resolve_cmap(cmap, self.theme)
            pts = self.ax.scatter(*xyz.T, c=np.asarray(scalars, float).ravel(), cmap=cmap,
                                  norm=self._norm(clim, log_scale), s=size * 1.6,
                                  depthshade=False, edgecolors=t["point_edge"],
                                  linewidths=0.3)
            if scalar_bar:
                self.ax.set_position([0.0, 0.0, 1 - SCALAR_BAR_GUTTER, 1.0])
                self.badges.gutter = 0.0
                self.badges.reserve_px = (BAR_WIDTH_PX + BAR_LABEL_ROOM_PX + BAR_GAP_PX)                     * self.fig.dpi / 100.0
                cax = self.fig.add_axes([1 - SCALAR_BAR_GUTTER + 0.035, 0.22, 0.018, 0.56])
                cax.set_axes_locator(self._bar_locator)
                bar = self.fig.colorbar(pts, cax=cax)
                bar.outline.set_visible(False)
                lo, hi = pts.norm.vmin, pts.norm.vmax
                if lo is not None and hi is not None:
                    ticks = _nice_ticks(float(lo), float(hi), bool(log_scale))
                    from matplotlib.ticker import FixedLocator, NullLocator
                    bar.ax.yaxis.set_major_locator(FixedLocator(ticks))
                    bar.ax.yaxis.set_minor_locator(NullLocator())
                    fmt = "%g" if log_scale else _tick_format(ticks)
                    bar.ax.set_yticklabels([fmt % v for v in ticks])
                bar.ax.tick_params(labelsize=9, colors=t["muted"], labelcolor=t["text"])
                bar.ax.set_title(str(scalar_bar), fontsize=10, color=t["text"])
        else:
            pts = self.ax.scatter(*xyz.T, color=colour or t["accent"], s=size * 1.6,
                                  depthshade=False)
        self.items[name] = {"kind": "points", "pts": pts, "verts": xyz}

    def _bar_locator(self, cax, renderer):
        """Colour bar right beside what the scene draws (UX: never floating mid-panel),
        vertically centred on it, kept inside the figure."""
        from matplotlib.transforms import Bbox
        from mpl_toolkits.mplot3d import proj3d
        fig = self.fig
        W, H = fig.bbox.width, fig.bbox.height
        k = fig.dpi / 100.0
        try:
            self.ax.apply_aspect()
            pts = self.all_points()
            xs, ys, _ = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2],
                                              self.ax.get_proj())
            disp = self.ax.transData.transform(np.column_stack([xs, ys]))
            disp = disp[np.all(np.isfinite(disp), axis=1)]
            right = float(disp[:, 0].max())
            low, high = float(disp[:, 1].min()), float(disp[:, 1].max())
            if self.badges.extent_x1 is not None:
                right = max(right, self.badges.extent_x1)
        except Exception:  # noqa: BLE001 - fall back to the right edge
            right, low, high = W * (1 - SCALAR_BAR_GUTTER), 0.22 * H, 0.78 * H
        width = BAR_WIDTH_PX * k
        x = min(max(right + BAR_GAP_PX * k, 0.0), W - width - BAR_LABEL_ROOM_PX * k)
        height = min(max(0.56 * (high - low), 0.30 * H), 0.56 * H)
        mid = min(max((low + high) / 2, height / 2 + 28 * k), H - height / 2 - 28 * k)
        return Bbox.from_bounds(x / W, (mid - height / 2) / H, width / W, height / H)

    def update_points(self, name, xyz, scalars=None):
        item = self.items[name]
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        item["pts"]._offsets3d = tuple(xyz.T)
        if scalars is not None:
            item["pts"].set_array(np.asarray(scalars, float).ravel())
        item["verts"] = xyz

    def add_face_tags(self, name, groups):
        """F1..F6 tags: 3D text in the face's shade (drawn above the geometry, zorder 45)."""
        texts = []
        tags = face_tag_layout(groups)
        with self._rc():
            for face, xyz in tags:
                texts.append(self.ax.text(
                    float(xyz[0]), float(xyz[1]), float(xyz[2]), face_label(face),
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color=face_text_colour(face, self.theme), zorder=45,
                    bbox=face_tag_bbox(face, self.theme, size=9.0)))
        self.items[name] = {"kind": "tags", "texts": texts,
                            "verts": np.vstack([xyz for _, xyz in tags]) if tags else []}

    def add_rays(self, name, origins, ends, colour, scalars, cmap, clim, width, opacity,
                 log_scale):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        segs = np.stack([np.asarray(origins, float).reshape(-1, 3),
                         np.asarray(ends, float).reshape(-1, 3)], axis=1)
        lines = Line3DCollection(list(segs), linewidths=width * 0.6, alpha=opacity)
        if scalars is not None:
            lines.set_array(np.asarray(scalars, float).ravel())
            lines.set_cmap(_resolve_cmap(cmap, self.theme))
            lines.set_norm(self._norm(clim, log_scale))
        else:
            lines.set_color(colour or stage_tokens(self.theme)["accent"])
        self.ax.add_collection3d(lines)
        self.items[name] = {"kind": "rays", "lines": lines, "verts": segs.reshape(-1, 3)}

    def update_rays(self, name, origins, ends, scalars=None):
        item = self.items[name]
        segs = np.stack([np.asarray(origins, float).reshape(-1, 3),
                         np.asarray(ends, float).reshape(-1, 3)], axis=1)
        item["lines"].set_segments(list(segs))
        if scalars is not None:
            item["lines"].set_array(np.asarray(scalars, float).ravel())
        item["verts"] = segs.reshape(-1, 3)

    def remove(self, name):
        item = self.items.pop(name, None)
        if item is None:
            return
        self.badges.entries.pop(name, None)
        for key in ("surf", "lines", "axis", "face", "pts", "line", "arrow", "ghost_surf",
                    "ghost_lines", "start"):
            artist = item.get(key)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
        for text in item.get("texts", []):  # face tags (kind "tags")
            try:
                text.remove()
            except Exception:
                pass

    def clear(self):
        self.items.clear()
        self._new_axes()

    def bounds(self):
        allv = [item["verts"] for item in self.items.values() if len(item.get("verts", []))]
        if not allv:
            return np.zeros(3) - 1, np.zeros(3) + 1
        pts = np.vstack(allv)
        return pts.min(axis=0), pts.max(axis=0)

    def all_points(self) -> np.ndarray:
        allv = [item["verts"] for item in self.items.values() if len(item.get("verts", []))]
        return np.vstack(allv) if allv else np.zeros((1, 3))

    def aspect(self) -> float:
        w, h = self.fig.get_size_inches()
        return float(w) / max(float(h), 1e-6)

    def tighten(self, gutter: float = 0.0, fill: float = 0.9) -> None:
        """The static twin of the PyVista ``tighten``: zoom the 3D box until the projected
        scene covers ``fill`` of the axes' limiting dimension (ED3-1: the Phase 2 scene left
        a 15-20 % blank band because the fixed box zoom ignored the chosen view), keeping
        ``gutter`` (fraction of the width) free on the right for the colour bar. The view
        direction and the box's proportions are unchanged, so the geometry is not distorted."""
        if self._fit is None:
            return
        self._fit["gutter"], self._fit["fill"] = float(gutter), float(fill)
        # A 3D axes keeps its box proportions, so its frame is smaller than the panel and the
        # artists are clipped to it: let the scene use the whole panel (the frame clips nothing)
        for child in self.ax.get_children():
            try:
                child.set_clip_on(False)
            except Exception:  # noqa: BLE001 - a non-artist child keeps its own clipping
                pass
        for _ in range(4):
            pts = self._projected_px()
            if pts is None:
                return
            # The panel is the figure: a 3D axes keeps its box proportions, so its own
            # bbox can be much smaller than the panel (the blank band of ED3-1).
            box = self.fig.bbox
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
            factor = min(fill * box.width * (1 - gutter) / width, fill * box.height / height)
            if abs(factor - 1.0) < 0.03:
                break
            self._fit["zoom"] = float(np.clip(self._fit["zoom"] * factor, 0.3, 3.0))
            self._apply_fit()

    def _projected_px(self) -> np.ndarray | None:
        """Display pixels of every scene vertex as the 3D axes currently project them."""
        from mpl_toolkits.mplot3d import proj3d
        allv = [item["verts"] for item in self.items.values() if len(item.get("verts", []))]
        if not allv:
            return None
        p = np.vstack(allv).astype(float)
        xs, ys, _ = proj3d.proj_transform(p[:, 0], p[:, 1], p[:, 2], self.ax.get_proj())
        return self.ax.transData.transform(np.column_stack([xs, ys]))

    def _apply_fit(self) -> None:
        """Box proportions (``half``) and the fitted zoom, as set by :meth:`set_view`."""
        self.ax.set_box_aspect(tuple(self._fit["half"]), zoom=1.3 * self._fit["zoom"])

    def set_view(self, azimuth, elevation, up, zoom, focus, distance, view_angle=30.0):
        lo, hi = self.bounds()
        centre = np.asarray(focus, dtype=float) if focus is not None else (lo + hi) / 2
        # Equal scale on every axis, but each axis only as long as the scene needs.
        half = np.maximum((hi - lo) / 2 * 1.08, 0.04 * float(np.max(hi - lo)) + 1e-9)
        half = np.maximum(half, np.abs(np.vstack([lo, hi]) - centre).max(axis=0) * 1.04)
        for setter, c, h in zip((self.ax.set_xlim, self.ax.set_ylim, self.ax.set_zlim),
                                centre, half):
            setter(c - h, c + h)
        # ``zoom`` is not a box zoom any more: the view is always filled with the projected
        # scene (see :meth:`tighten`), so a caller's zoom-out (Phase 2's 0.85) cannot crop it
        # or leave a blank band. The last gutter/fill asked of :meth:`tighten` is kept.
        prev = self._fit or {}
        self._fit = {"half": np.asarray(half, dtype=float), "zoom": 1.0,
                     "gutter": prev.get("gutter", 0.0), "fill": prev.get("fill", 0.9)}
        self._apply_fit()
        up = np.asarray(up, dtype=float)
        k = int(np.argmax(np.abs(up)))
        axis = "xyz"[k]
        try:
            self.ax.view_init(elev=elevation, azim=azimuth, vertical_axis=axis)
        except TypeError:
            self.ax.view_init(elev=elevation, azim=azimuth)
        if up[k] < 0:
            # matplotlib's vertical axis always points up; honour a negative ``up``
            # (e.g. a y-down camera frame) by flipping that axis.
            low, high = getattr(self.ax, f"get_{axis}lim")()
            getattr(self.ax, f"set_{axis}lim")(high, low)
        # ED3-1: fill the box with the projected scene for this view (the old fixed 1.3x box
        # zoom left a 15-20 % blank band).
        self.tighten(self._fit["gutter"], fill=self._fit["fill"])
        self.render()

    def screenshot(self, path, scale=None):
        # ``scale`` is a PyVista option; the static picture is saved at its own size.
        self.fig.savefig(str(path), facecolor=self.fig.get_facecolor())


# --------------------------------------------------------------------------------------
# Public API (shared by the widget and the off-screen scene)


class _SceneAPI:
    """The primitives; subclasses set ``self._impl``."""

    _impl: Any

    @property
    def backend(self) -> str:
        """``"pyvista"`` or ``"static"``."""
        return self._impl.kind

    def names(self) -> list[str]:
        """Names of everything added."""
        return list(self._impl.items)

    def add_camera(self, name: str, index: int, R: Any, t: Any = None, scale: float = 1.0,
                   telecentric: bool = False, label: str | None = None, *,
                   axis: bool = True) -> str:
        """Camera glyph in its colour (body + frustum for pinhole, body + slim barrel with
        parallel rays for telecentric), a circular numbered badge and, with ``axis``, a thin
        viewing-direction line reaching the scene's target (:meth:`set_axis_target`,
        :meth:`fit_view`). ``label`` (e.g. ``"(0.47)"``) is written beside the badge."""
        self._impl.add_camera(name, index, R, t, float(scale), bool(telecentric), label,
                              bool(axis))
        return name

    def update_camera(self, name: str, R: Any, t: Any = None) -> None:
        """Move an existing camera (no rebuild)."""
        self._impl.update_camera(name, R, t)

    def mark_camera(self, name: str, *, highlight: bool = False, flagged: bool = False) -> None:
        """Ring a camera's badge: ``highlight`` (theme text, the selected camera) or
        ``flagged`` (red, e.g. would be dropped)."""
        self._impl.mark_camera(name, highlight, flagged)

    def set_axis_target(self, point: Any) -> None:
        """Make every viewing-direction line end level with ``point`` (e.g. the target
        centre); :meth:`fit_view` sets it to the centre of the non-camera items."""
        self._impl.axis_target = None if point is None else np.asarray(point, dtype=float)
        badges = getattr(self._impl, "badges", None)
        if badges is not None and hasattr(badges, "centre_world"):
            badges.centre_world = self._impl.axis_target  # labels turn away from it
        self._impl.refresh_axes()

    def add_board(self, corners_3d: Any, colour: str | None = None, scalars: Any = None, *,
                  name: str | None = None, cmap: str | None = None,
                  clim: tuple[float, float] | None = None, opacity: float = 0.45,
                  point_size: float = 6.0, log_scale: bool = False) -> str:
        """A target/board view: its outline as a translucent plane plus its corner points.

        ``scalars``: one value (whole board coloured by ``cmap``/``clim``, e.g. its RMS) or one
        per corner (corners coloured). ``cmap`` None = the theme-aware error map. Returns the
        name (auto ``board<k>`` when omitted). ``**error_scale().scene_kwargs()`` gives the
        shared log error scale.
        """
        name = name or self._auto_name("board")
        self._impl.add_board(name, corners_3d, colour, scalars, cmap, clim, opacity, point_size,
                             bool(log_scale))
        return name

    def update_board(self, name: str, corners_3d: Any, scalars: Any = None) -> None:
        """Move/reshape an existing board (same number of corners)."""
        self._impl.update_board(name, corners_3d, scalars)

    def add_face_tags(self, groups: Sequence[tuple[int, Any]], *, name: str | None = None) -> str:
        """Square "F1".."F6" tags on the faces of a target, the same face numbers and shades
        as the 2D panels. ``groups``: ``[(0-based face, corner xyz (k x 3)), ...]`` (the
        corners of each face); each tag sits just outside its face. Returns the name."""
        name = name or self._auto_name("tags")
        self._impl.add_face_tags(name, groups)
        return name

    def add_points(self, xyz: Any, scalars: Any = None, cmap: str | None = None, *,
                   name: str | None = None, clim: tuple[float, float] | None = None,
                   colour: str | None = None, size: float = 7.0,
                   scalar_bar: str | None = None, log_scale: bool = False) -> str:
        """Point cloud; with ``scalars`` coloured by ``cmap`` (None = theme-aware viridis),
        else ``colour`` (default theme accent). ``scalar_bar="px"`` adds a vertical colour
        bar with that title on the right (both renderers; nice round ticks)."""
        name = name or self._auto_name("points")
        self._impl.add_points(name, xyz, scalars, cmap, clim, colour, size, scalar_bar,
                              bool(log_scale))
        return name

    def update_points(self, name: str, xyz: Any, scalars: Any = None) -> None:
        """Move existing points (same count) and optionally recolour them."""
        self._impl.update_points(name, xyz, scalars)

    def add_rays(self, origins: Any, ends: Any, *, name: str | None = None,
                 colour: str | None = None, scalars: Any = None, cmap: str | None = None,
                 clim: tuple[float, float] | None = None, width: float = 1.5,
                 opacity: float = 0.7, log_scale: bool = False) -> str:
        """Line segments ``origins[i] -> ends[i]`` (rays, residuals, links)."""
        name = name or self._auto_name("rays")
        self._impl.add_rays(name, origins, ends, colour, scalars, cmap, clim, width, opacity,
                            bool(log_scale))
        return name

    def update_rays(self, name: str, origins: Any, ends: Any, scalars: Any = None) -> None:
        """Move existing rays (same count)."""
        self._impl.update_rays(name, origins, ends, scalars)

    def remove(self, name: str) -> None:
        """Remove one item."""
        self._impl.remove(name)

    def clear(self) -> None:
        """Remove everything."""
        self._impl.clear()

    def apply_frame(self, frame: Mapping[str, Any], render: bool = True) -> None:
        """Move existing items to the poses/positions in ``frame`` (see module docstring)."""
        for name, pose in (frame.get("cameras") or {}).items():
            self._impl.update_camera(name, pose["R"], pose.get("t"))
        for name, entry in (frame.get("boards") or {}).items():
            if isinstance(entry, Mapping):
                self._impl.update_board(name, entry["corners"], entry.get("scalars"))
            else:
                self._impl.update_board(name, entry)
        for name, entry in (frame.get("points") or {}).items():
            if isinstance(entry, Mapping):
                self._impl.update_points(name, entry["xyz"], entry.get("scalars"))
            else:
                self._impl.update_points(name, entry)
        for name, entry in (frame.get("rays") or {}).items():
            self._impl.update_rays(name, entry["origins"], entry["ends"], entry.get("scalars"))
        if render:
            self._impl.render()

    def set_view(self, azimuth: float = -60.0, elevation: float = 25.0, *,
                 up: Sequence[float] = (0, 0, 1), zoom: float = 1.0,
                 focus: Sequence[float] | None = None, distance: float | None = None) -> None:
        """Look at the scene (or ``focus``) from ``azimuth``/``elevation`` degrees around ``up``."""
        self._impl.set_view(azimuth, elevation, up, zoom, focus, distance)

    def camera_centres(self) -> dict[str, np.ndarray]:
        """World centre of every camera, by name."""
        return {name: camera_centre(*item["pose"]) for name, item in self._impl.items.items()
                if item["kind"] == "camera"}

    def fit_view(self, *, selected: str | int | None = None, up: Sequence[float] = (0, 0, 1),
                 margin: float = 0.08, elevations: Sequence[float] = (18, 26, 34, 42),
                 azimuth: float | None = None, elevation: float | None = None
                 ) -> tuple[float, float]:
        """Frame the whole scene (UX-9/10): pick the view that best separates the cameras,
        keep ``selected`` (a camera name or index) in front where possible, aim every
        viewing-direction line at the target and fit everything with ``margin`` (plus a
        colour-bar gutter). Fix ``azimuth``/``elevation`` to skip the search. Returns the
        (azimuth, elevation) used."""
        cams = [(name, item) for name, item in self._impl.items.items()
                if item["kind"] == "camera"]
        others = [np.asarray(item["verts"]) for item in self._impl.items.values()
                  if item["kind"] not in ("camera", "trail") and len(item.get("verts", []))]
        target_pts = np.vstack(others) if others else None
        centres = np.array([camera_centre(*item["pose"]) for _, item in cams]) \
            if cams else np.zeros((0, 3))
        if target_pts is not None:
            lo, hi = target_pts.min(axis=0), target_pts.max(axis=0)
            target_centre = (lo + hi) / 2
            target_radius = float(np.linalg.norm(hi - lo)) / 2
        else:
            target_centre = centres.mean(axis=0) if len(centres) else np.zeros(3)
            target_radius = 0.0
        if cams:
            self.set_axis_target(target_centre)
        sel = None
        if selected is not None:
            for k, (name, item) in enumerate(cams):
                if name == selected or (isinstance(selected, (int, np.integer))
                                        and item["index"] == int(selected)):
                    sel = k
        if azimuth is None or elevation is None:
            glyph = max((item["scale"] for _, item in cams), default=0.0)
            az, el = choose_view(centres if len(centres) else target_centre[None],
                                 target_centre, up=up, selected=sel, glyph_size=glyph,
                                 target_radius=target_radius, elevations=elevations)
            azimuth = az if azimuth is None else azimuth
            elevation = el if elevation is None else elevation
        pts = self._impl.all_points()
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        focus = (lo + hi) / 2
        direction = _view_direction(azimuth, elevation, up)
        has_bar = any(item.get("bar_title") for item in self._impl.items.values()) or \
            getattr(self._impl, "badges", None) is not None and \
            getattr(self._impl.badges, "gutter", 0.0) > 0
        # A narrow lens (18 deg) keeps near and far glyphs about the same size.
        distance = fit_distance(pts, focus, direction, up, view_angle=FIT_VIEW_ANGLE,
                                aspect=self._impl.aspect(), margin=margin,
                                gutter=SCALAR_BAR_GUTTER if has_bar else 0.0)
        self._impl.set_view(azimuth, elevation, up, 1.0, focus, distance, FIT_VIEW_ANGLE)
        tighten = getattr(self._impl, "tighten", None)
        if callable(tighten):  # PyVista: fill the frame, bar beside the scene (UX3-13/14)
            try:
                tighten(SCALAR_BAR_GUTTER if has_bar else 0.0, fill=1.0 - 2 * margin)
            except Exception:  # noqa: BLE001 - framing is a nicety
                pass
        if cams and sel is not None:
            self.mark_camera(cams[sel][0], highlight=True)
        return float(azimuth), float(elevation)

    def add_trail(self, points: Any, *, name: str | None = None, camera: int | None = None,
                  colour: str | None = None, start_pose: Any = None,
                  telecentric: bool = False, scale: float | None = None, arrow: bool = True,
                  width: float = 3.0, fade: tuple[float, float] = TRAIL_FADE,
                  min_fraction: float = TRAIL_MIN_FRACTION) -> str:
        """Motion as a trail (UX2-6): a path through ``points`` (N x 3, oldest first) that
        fades from ``fade[0]`` to ``fade[1]`` opacity, an arrowhead at the newest point and,
        with ``start_pose`` (``(R, t)`` world-to-camera, or a 4x4), a ghost of the camera
        glyph where it started plus a hollow start dot. Colour: camera ``camera``'s fill for
        the theme (:func:`camera_fill`), else ``colour``, else the accent. ``scale`` is the
        ghost glyph size (default: as the cameras already added). Hidden while the path is
        shorter than ``min_fraction`` of the scene size. Grow it with :meth:`update_trail`
        during playback. Put the exaggeration ("×5") in the panel subtitle, not in 3D."""
        name = name or self._auto_name("trail")
        if scale is None:
            scales = [item["scale"] for item in self._impl.items.values()
                      if item["kind"] == "camera"]
            scale = max(scales) if scales else 0.1
        ghost = None
        if start_pose is not None:
            if isinstance(start_pose, (tuple, list)) and len(start_pose) == 2:
                R, t = start_pose
            else:
                R, t = start_pose, None
            ghost = camera_glyph(R, t, float(scale), bool(telecentric), rays=False)
        spec = {"index": None if camera is None else int(camera), "colour": colour,
                "ghost": ghost, "arrow": bool(arrow), "width": float(width),
                "fade": tuple(fade), "min_fraction": float(min_fraction),
                "scale": float(scale),
                "stop_short": 0.45 * float(scale) if camera is not None else 0.0}
        self._impl.add_trail(name, np.asarray(points, dtype=float).reshape(-1, 3), spec)
        return name

    def update_trail(self, name: str, points: Any, *, render: bool = True) -> None:
        """Replace a trail's path (any number of points, e.g. the poses up to the current
        playback position). ``render=False`` when the caller renders once after several
        updates (playback does; see :class:`ScenePlayer`)."""
        self._impl.update_trail(name, np.asarray(points, dtype=float).reshape(-1, 3),
                                render=render)

    def set_animating(self, on: bool) -> None:
        """Tell the scene whether a playback tween is running. The static picture then
        redraws at most every :data:`~pyCamSet.gui.education_perf.STATIC_DRAW_INTERVAL_MS`
        and draws in full again when the tween stops. The OpenGL view ignores it."""
        setter = getattr(self._impl, "set_animating", None)
        if callable(setter):
            setter(bool(on))

    def trail_visible(self, name: str) -> bool:
        """False while a trail is hidden for being shorter than its ``min_fraction``."""
        return bool(self._impl.items[name].get("visible", False))

    def badge_positions(self) -> dict[str, tuple[float, float]]:
        """Screen positions (pixels) of the badges as last drawn, by camera name."""
        return dict(self._impl.badges.positions)

    def set_theme(self, theme_name: str) -> None:
        """Re-colour the background for a GUI theme (camera colours never change)."""
        self._impl.set_theme(theme_name)

    def render(self) -> None:
        """Redraw now."""
        self._impl.render()

    def screenshot(self, path: Any, scale: int | None = None) -> None:
        """Save the current view as a PNG (``scale`` enlarges a PyVista render)."""
        self._impl.render()
        self._impl.screenshot(path, scale=scale)

    def _auto_name(self, prefix: str) -> str:
        k = 0
        while f"{prefix}{k}" in self._impl.items:
            k += 1
        return f"{prefix}{k}"


from PySide6.QtCore import QElapsedTimer, QTimer, Signal  # noqa: E402  (Qt only below)
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget  # noqa: E402


class Scene3D(QWidget, _SceneAPI):
    """A 3D scene widget: embedded PyVista when possible, static matplotlib otherwise.

    ``backend``: None/"auto" (see :func:`scene_backend`), ``"pyvista"`` or ``"static"``.
    ``theme``: GUI theme name (None = active). The stage view calls :meth:`set_theme` on a
    theme switch. ``interactor`` is the QtInteractor (None in static mode).
    """

    def __init__(self, parent: QWidget | None = None, *, theme: str | None = None,
                 backend: str | None = None) -> None:
        super().__init__(parent)
        self.setProperty("education3DView", True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(240, 180)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        theme = resolve_theme(theme)
        if backend in (None, "auto"):
            decision = scene_backend_choice()
        else:
            decision = BackendChoice(backend, "requested by the caller")
        self.interactor = None
        self.canvas = None
        if decision.name == "pyvista":
            try:
                from pyvistaqt import QtInteractor
                from pyCamSet.gui.education_dialog import install_render_window_key_filter
                self.interactor = QtInteractor(self)
                install_render_window_key_filter(self.interactor)
                layout.addWidget(self.interactor)
                self._impl = _PyVistaImpl(self.interactor, theme)
                self.interactor.setAccessibleName("3D scene: drag to rotate, scroll to zoom")
            except Exception as exc:  # no OpenGL context: fall back, never break the stage
                if self.interactor is not None:
                    self.interactor.setParent(None)
                    self.interactor.deleteLater()
                self.interactor = None
                decision = BackendChoice("static", f"the OpenGL view could not be created "
                                                   f"({type(exc).__name__})", automatic=True)
        if decision.name == "static":
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            figure = Figure(figsize=(6, 4), dpi=100)
            self.canvas = FigureCanvasQTAgg(figure)
            perf.cap_canvas_pixel_ratio(self.canvas)
            self.canvas.setMinimumSize(200, 150)
            self.canvas.setAccessibleName("3D scene (static picture)")
            note = backend_note_for(decision)
            if note:
                self.canvas.setToolTip(note)
                self.canvas.setAccessibleDescription(note)
            layout.addWidget(self.canvas)
            self._impl = _StaticImpl(figure, theme, self.canvas.draw_idle)
        self.backend_choice = decision
        _remember_choice(decision)
        try:  # the maintainer's diagnostics log (no-op unless PYCAMSET_EDU_DEBUG=1)
            from pyCamSet.gui import education_diagnostics
            if education_diagnostics.enabled():
                education_diagnostics.note_backend(decision, _PROBE)
        except Exception:  # noqa: BLE001 - diagnostics must never break the stage
            pass

    @property
    def figure(self):
        """The matplotlib figure in static mode (None with PyVista)."""
        return self._impl.fig if self._impl.kind == "static" else None

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.interactor is not None:
            try:
                self.interactor.close()
            except Exception:
                pass
        super().closeEvent(event)


class OffscreenScene(_SceneAPI):
    """The same primitives on ``pyvista.Plotter(off_screen=True)`` for PNG exports."""

    def __init__(self, theme: str | None = None, window_size: tuple[int, int] = (1200, 800)):
        import pyvista as pv
        self.plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
        self._impl = _PyVistaImpl(self.plotter, resolve_theme(theme))

    def close(self) -> None:
        """Release the render window."""
        self.plotter.close()


# --------------------------------------------------------------------------------------
# Player


def _speeds() -> tuple[tuple[str, int], ...]:
    from pyCamSet.gui.education_dialog import SPEEDS as speeds
    return speeds


#: Playback speeds offered by the transport bar (label, milliseconds per recorded step).
SPEEDS = (("×0.25", 1400), ("×0.5", 700), ("×1", 350))


class ScenePlayer(QWidget):
    """Plays ``frames`` through ``scene`` with smooth interpolation between recorded frames.

    ``frames[0]`` is the starting state (shown first, not a step); ``frames[1:]`` are the
    recorded steps, numbered by ``step_numbers`` (default 1..n-1; strictly increasing, >= 1)
    out of ``total_steps``. The controls are the one transport bar
    (:class:`RecordedStepControls`: ⏮ ◀ ▶Play ▶, scrubber, counter, speed). Between two
    recorded frames the scene is tweened in ``substeps`` steps unless reduce motion is on.
    ``step_weights`` (one per step, e.g. :func:`pace_weights` of the error per state) spend
    more playback time on the steps that change most (UX-11).

    Each shown (possibly fractional) position calls ``on_frame(position, frame)`` and emits
    ``frameShown(position, frame)``; ``stepReached(index)`` fires on whole positions, so a
    phase can sync a cost curve (cheap) on every frame and bars (dearer) per step.
    """

    frameShown = Signal(float, object)
    stepReached = Signal(int)

    def __init__(self, scene: _SceneAPI, frames: Sequence[Mapping[str, Any]], *,
                 step_numbers: Sequence[int] | None = None, total_steps: int | None = None,
                 substeps: int = 8, on_frame: Callable[[float, dict], None] | None = None,
                 reduce_motion: bool | None = None, unit_label: str = "Recorded solver steps",
                 step_weights: Sequence[float] | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from pyCamSet.gui.education_dialog import RecordedStep, RecordedStepControls
        if len(frames) < 2:
            raise ValueError("ScenePlayer needs a starting frame and at least one step")
        self.scene = scene
        self.frames = [dict(f) for f in frames]
        numbers = list(step_numbers) if step_numbers is not None else \
            list(range(1, len(frames)))
        if len(numbers) != len(frames) - 1:
            raise ValueError("step_numbers must number frames[1:]")
        self.substeps = max(1, int(substeps))
        self._weights = None
        if step_weights is not None:
            weights = np.asarray(step_weights, dtype=float).ravel()
            if len(weights) != len(frames) - 1:
                raise ValueError("step_weights needs one weight per step")
            weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
            self._weights = weights / weights.mean()
        self._on_frame = on_frame
        self.controls = RecordedStepControls(
            [RecordedStep(int(n), i + 1) for i, n in enumerate(numbers)], total_steps,
            self, reduce_motion=reduce_motion, unit_label=unit_label, start_frame=0,
            speeds=_speeds())
        self.controls.set_playback_interval(_speeds()[1][1])
        self.controls.speed.setCurrentIndex(1)
        self.controls.interval_for = self._interval_for
        self.speed = self.controls.speed
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.controls, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._position = 0.0
        self._target = 0.0
        self._step_ticks = self.substeps
        self._clock = QElapsedTimer()
        self._clock.start()
        self._tween_from = 0.0
        self._tween_started = 0.0
        self._tween_span = 1.0
        self._in_tick = False
        self._tween = QTimer(self)
        self._tween.timeout.connect(self._tween_tick)
        self.controls.frameChanged.connect(self._target_changed)
        self.show_position(0.0)

    def _weight(self, step_index: int) -> float:
        """Weight of the step from position ``step_index`` to ``step_index + 1``."""
        if self._weights is None or not 0 <= step_index < len(self._weights):
            return 1.0
        return float(np.clip(self._weights[step_index], 0.35, 4.0))

    def _interval_for(self, position: int) -> int:
        return int(self.controls.playback_interval_ms * self._weight(position))

    # positions
    @property
    def position(self) -> float:
        """The shown position (0 = start; may be fractional while tweening)."""
        return self._position

    def frame_at(self, position: float) -> dict[str, Any]:
        """The (interpolated) frame at ``position`` in 0..len(frames)-1."""
        position = float(np.clip(position, 0, len(self.frames) - 1))
        lo = int(np.floor(position))
        hi = min(lo + 1, len(self.frames) - 1)
        if hi == lo or position == lo:
            return interpolate_frames(self.frames[lo], self.frames[lo], 0.0)
        return interpolate_frames(self.frames[lo], self.frames[hi], position - lo)

    def show_position(self, position: float) -> dict[str, Any]:
        """Show the scene at ``position`` (no animation); returns the frame shown.

        The scene is drawn once, after every listener has moved its own parts (``frameShown``
        handlers update trails and panels with ``render=False``), so a frame costs one render.
        """
        self._position = float(np.clip(position, 0, len(self.frames) - 1))
        frame = self.frame_at(self._position)
        self.scene.apply_frame(frame, render=False)
        # UX3-6: the counter names the frame actually shown ("→ Step 1 of 11" between two
        # recorded steps), so it never says "Start" while the scene is half-way.
        show_fraction = getattr(self.controls, "show_fraction", None)
        if callable(show_fraction):
            show_fraction(self._position)
        if self._on_frame is not None:
            self._on_frame(self._position, frame)
        self.frameShown.emit(self._position, frame)
        self._render_scene()
        if abs(self._position - round(self._position)) < 1e-9:
            self.stepReached.emit(int(round(self._position)))
        return frame

    # animation: the position follows the wall clock, so a slow frame skips positions rather
    # than stretching the playback; the final position is always shown
    def _now_ms(self) -> float:
        return float(self._clock.elapsed())

    def _set_animating(self, on: bool) -> None:
        setter = getattr(self.scene, "set_animating", None)
        if callable(setter):
            setter(bool(on))

    def _render_scene(self) -> None:
        # Through the implementation: Scene3D is a QWidget, so its own .render is QWidget.render.
        impl = getattr(self.scene, "_impl", None)
        if impl is not None:
            impl.render()
        else:
            self.scene.render()

    def _stop_tween(self) -> None:
        self._tween.stop()
        self._set_animating(False)

    def _target_changed(self, target: Any) -> None:
        self._target = float(target)
        if self.controls.reduce_motion or abs(self._target - self._position) < 1e-9:
            self._stop_tween()
            self.show_position(self._target)
            return
        if abs(self._target - self._position) > 1.0:  # a scrub jump: tween only the last step
            self._position = self._target - np.sign(self._target - self._position)
        step_index = int(min(self._position, self._target))
        weight = self._weight(step_index)
        self._step_ticks = max(2, int(round(self.substeps * weight)))
        interval = self.controls.playback_interval_ms * weight
        self._tween_from = self._position
        self._tween_span = max(1.0, 0.9 * interval)  # wall-clock length of this step
        self._tween_started = self._now_ms()
        self._tween.setInterval(max(perf.tween_floor_ms(),
                                    int(self._tween_span / self._step_ticks)))
        self._set_animating(True)
        self._tween.start()

    def _tween_tick(self) -> None:
        if self._in_tick:  # one draw in flight: a tick never runs inside another
            return
        self._in_tick = True
        try:
            fraction = (self._now_ms() - self._tween_started) / self._tween_span
            if fraction >= 1.0:
                self._stop_tween()
                self.show_position(self._target)
            else:
                self.show_position(self._tween_from
                                   + (self._target - self._tween_from) * max(0.0, fraction))
        finally:
            self._in_tick = False

    def go_to(self, index: int, animate: bool = True) -> None:
        """Move the controls to recorded position ``index`` (0 = start), tweening if allowed."""
        if not animate:
            self._stop_tween()
            self._position = float(index)
        self.controls.go_to(index)
        if not animate:
            self.show_position(float(index))

    def stop_for_context_change(self) -> None:
        """Stop playback and finish any tween (called on stage/camera/theme/run changes)."""
        self.controls.stop_for_context_change()
        if self._tween.isActive():
            self._stop_tween()
            self.show_position(self._target)

    def stop_playback(self) -> None:
        """Alias used by the dialog's teardown."""
        self.stop_for_context_change()


__all__ = [
    "FORCE_STATIC", "ENV_SWITCH", "scene_backend", "scene_backend_choice", "BackendChoice",
    "backend_note", "gl_probe", "as_pose", "camera_centre", "camera_glyph",
    "camera_axis_segment", "axis_length_to", "layout_badges", "choose_view", "fit_distance",
    "pace_weights", "interpolate_pose", "interpolate_frames", "Scene3D", "OffscreenScene",
    "ScenePlayer", "SPEEDS", "SCALAR_BAR_GUTTER",
]
