"""Render the education visual foundation (style, stage view, 3D scene) to PNGs for review.

Usage (from the checkout root)::

    python tools/edu_release/foundation_gallery.py [--out DIR] [--no-pyvista]

Writes, for each GUI theme (Light, Dark, Sepia):
  stage_figure_<theme>.png   a figure stage at 1400x900 (camera image, residuals on the shared log
                             error scale, bars, strip plot, chips, caption, "? Terms" glossary,
                             camera/photo pickers) and stage_figure_<theme>_1280.png at 1280x800
  stage_3d_<theme>.png       a 3D stage (static fallback): fit_view framing, circle badges, lens
                             glyphs, viewing-direction lines and the one-row transport bar
  stage_steps_<theme>.png    a step-through stage with named step pills in the transport bar
  pieces_<theme>.png         the shared rules: figure colours vs camera colours (Delta E), the
                             camera outline rule, badges, the error colour scale, number wording
  pyvista_ring_<theme>.png   the same 8-camera ring rendered by real off-screen PyVista
  pyvista_glyphs_<theme>.png close-up of the pinhole and telecentric glyphs next to a cube target
  faces_<theme>.png          round 2: the one face look (neutral shades, square F tags, marker
                             shapes), a face matrix, and a local-range heat map beside the shared
                             log scale (UX2-2, UX2-7)
  split_<theme>.png          round 2: a 3D panel with the shared PanelHeading beside a figure
                             panel (one title baseline, UX2-19), camera trails with a ghost start
                             pose and arrowheads (UX2-6), at 1280x800 with the camera picker
  pyvista_trails_<theme>.png the same trails rendered by real off-screen PyVista
  round3_<theme>.png         round 3: the colour-map rule (amount map for counts, diverging map
                             for signed values, viridis only for errors; UX3-3), text kept
                             inside its card (UX3-1), limits neutral until crossed (UX3-11)
Every matplotlib figure is also run through ``text_collisions``, ``card_overflows`` and
``colour_map_misuse`` (printed; must be empty).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":  # point Qt at the system fonts on Windows only
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
os.environ.setdefault("PYCAMSET_EDU_3D", "static")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def _below_normal_priority() -> None:
    """Lower the process priority so the GUI stays responsive while rendering."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),
                                                    0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# Synthetic data


def synthetic_photo(seed: int = 1):
    """A ChArUco board photographed at an angle, with its true corner pixels."""
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((9, 7), 0.04, 0.03, d)
    sq = 80
    img = board.generateImage((9 * sq, 7 * sq), marginSize=0)
    src = np.float32([[0, 0], [9 * sq, 0], [9 * sq, 7 * sq], [0, 7 * sq]])
    dst = np.float32([[250, 150], [1020, 240], [975, 770], [190, 700]])
    H = cv2.getPerspectiveTransform(src, dst)
    yy, xx = np.mgrid[0:900, 0:1200]
    canvas = (55 + 45 * xx / 1200 + 25 * yy / 900).astype(np.uint8)
    warped = cv2.warpPerspective(img, H, (1200, 900), borderValue=0)
    mask = cv2.warpPerspective(np.full_like(img, 255), H, (1200, 900))
    out = np.where(mask > 0, (warped * 0.8 + 25).astype(np.uint8), canvas)
    rng = np.random.default_rng(seed)
    out = np.clip(out + rng.normal(0, 6, out.shape), 0, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    corners = board.getChessboardCorners()[:, :2] / 0.04 * sq
    pts = cv2.perspectiveTransform(corners.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    return out, pts


def look_at(centre, target=(0, 0, 0), up=(0, 0, 1)):
    """World-to-camera (R, t) for a camera at ``centre`` looking at ``target`` (image y down)."""
    c = np.asarray(centre, float)
    z = np.asarray(target, float) - c
    z /= np.linalg.norm(z)
    up = np.asarray(up, float)
    y = -(up - (up @ z) * z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.vstack([x, y, z])
    return R, -R @ c


def ring_rig(n: int = 8, radius: float = 1.0, height: float = 0.35):
    poses = []
    for k in range(n):
        a = 2 * np.pi * k / n
        poses.append(look_at((radius * np.cos(a), radius * np.sin(a),
                              height * (1 if k % 2 == 0 else 0.4))))
    return poses


def cube_points(size: float = 0.24, grid: int = 6):
    pts = []
    s = np.linspace(-size / 2, size / 2, grid)
    for axis in range(3):
        for sign in (-1, 1):
            if axis == 2 and sign < 0:
                continue  # the bottom face is not visible
            u, v = np.meshgrid(s, s)
            face = np.zeros((u.size, 3))
            others = [i for i in range(3) if i != axis]
            face[:, others[0]] = u.ravel()
            face[:, others[1]] = v.ravel()
            face[:, axis] = sign * size / 2
            pts.append(face)
    return np.vstack(pts)


def strip_values(seed: int = 3):
    rng = np.random.default_rng(seed)
    vals = {c: np.abs(rng.gamma(2.2, 0.17 + 0.02 * c, 48)) for c in range(8)}
    vals[2] = np.r_[vals[2], 3.1, 4.1]
    vals[5] = np.r_[vals[5], 2.8, 4.4]
    return vals


# --------------------------------------------------------------------------------------
# Scenes


def build_ring(scene, *, with_rays: bool = True, perturb: float = 0.0, seed: int = 0,
               selected: int | None = None):
    """8 cameras in a ring around a cube target, points coloured by error (shared log px
    scale); framed with ``fit_view`` (selected camera in front, badges inside)."""
    from pyCamSet.gui.education_style import error_scale, stage_tokens
    rng = np.random.default_rng(seed)
    poses = ring_rig()
    rms = [0.47, 0.59, 0.85, 0.56, 0.56, 0.58, 0.60, 0.40]
    for k, (R, t) in enumerate(poses):
        scene.add_camera(f"cam{k}", k, R, t, scale=0.24, telecentric=(k == 7),
                         label=f"({rms[k]:.2f})")
    pts = cube_points()
    err = np.abs(rng.gamma(2.0, 0.12, len(pts)))
    theme = getattr(scene._impl, "theme", None)
    scene.add_points(pts, err, name="target", scalar_bar="px", size=8,
                     **error_scale(theme).scene_kwargs())
    corners = cube_points(0.2, 4)[:16] + np.array([0, 0, -0.32])
    scene.add_board(corners, scalars=0.3, name="board0", **error_scale(theme).scene_kwargs())
    if with_rays:
        R, t = poses[3]
        centre = -R.T @ t
        seen = pts[(pts - centre) @ (-centre) > 0][::3]
        scene.add_rays(np.repeat(centre[None], len(seen), 0), seen, name="rays3",
                       colour=stage_tokens(theme)["link"], opacity=0.35, width=1.0)
    scene.fit_view(selected=selected)
    return poses, pts, err


def solver_frames(poses, pts, err, steps: int = 6, seed: int = 5):
    """Seed -> solution frames: cameras start perturbed and settle; errors shrink."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed)
    start = []
    for R, t in poses:
        dR = Rotation.from_rotvec(rng.normal(0, 0.12, 3)).as_matrix()
        c = -R.T @ t + rng.normal(0, 0.08, 3)
        R0 = dR @ R
        start.append((R0, -R0 @ c))
    frames = []
    for k in range(steps + 1):
        s = 1 - 0.55 ** k if k < steps else 1.0
        cams = {}
        for i, ((Ra, ta), (Rb, tb)) in enumerate(zip(start, poses)):
            from pyCamSet.gui.education_scene3d import interpolate_pose
            cams[f"cam{i}"] = interpolate_pose({"R": Ra, "t": ta}, {"R": Rb, "t": tb}, s)
        frames.append({"cameras": cams,
                       "points": {"target": {"xyz": pts, "scalars": err * (3 - 2 * s)}}})
    return frames


# --------------------------------------------------------------------------------------
# Stages


def figure_stage(view, photo, corners, theme):
    from pyCamSet.gui import education_style as es
    from pyCamSet.gui.education_stage import Chip
    rng = np.random.default_rng(11)
    projected = corners + rng.normal(0, 0.5, corners.shape)
    values = strip_values()

    def draw(fig, _theme):
        gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
        a = fig.add_subplot(gs[0, 0])
        es.image_with_points(a, photo, corners[:-5], rejected=corners[-5:], camera=3)
        es.panel_title(a, "A", "Camera image", f"camera 4, {len(corners) - 5} corners")
        b = fig.add_subplot(gs[0, 1])
        scale = es.error_scale(_theme)
        q = es.residual_arrows(b, photo, corners, projected, 40, camera=3, colour_scale=scale)
        scale.colourbar(fig, b, q)
        es.panel_title(b, "B", "How well it fits", "residuals ×40, shared log error scale")
        c = fig.add_subplot(gs[0, 2])
        es.bars_by_camera(c, {i: v for i, v in enumerate([52, 61, 48, 63, 57, 22, 59, 55])},
                          threshold=30, threshold_label="keep \u2265 30", ylabel="Points found")
        es.panel_title(c, "C", "Points per camera", "camera 6 would be dropped")
        d = fig.add_subplot(gs[1, :])
        es.strip_plot_by_camera(d, values, highlight=(3, 5), ylabel=es.error_label("rms", short=True,
                                                                                unit=False))
        es.panel_title(d, "D", "Error per photo", f"one dot per photo; ringed: {es.photo_label(5)}")

    view.draw_figure(draw)
    view.set_caption("Each circle is a corner the detector found on camera 4's photo; grey "
                     "circles were rejected. The strip plot shows each photo's reprojection "
                     "error, so one bad photo stands out at once.")
    view.set_chips([Chip(str(len(corners) - 5), "corners kept", camera=3),
                    Chip(es.format_px(0.42), "RMS reprojection error", tone="good"),
                    Chip("7 / 8", "cameras kept", tone="warn"), Chip("12", "photos")])
    view.set_numbers(["Camera", "Points", "Coverage", "RMS (px)"],
                     [(f"{i + 1}", p, f"{60 + 4 * i} %", f"{0.3 + 0.05 * i:.2f}")
                      for i, p in enumerate([52, 61, 48, 63, 57, 22, 59, 55])])


CAMERA_NAMES = {0: "cam1_FrontLeft_907T", 1: "cam2_Front", 2: "cam3", 3: "cam4_Right",
                4: "cam5_Back", 5: "cam6_TopLeft", 6: "cam7", 7: "cam8_Side"}


def build_view(stages, header_pickers=True):
    from pyCamSet.gui.education_stage import EducationStageView, camera_picker, photo_picker
    view = EducationStageView(stages)
    if header_pickers:
        view.add_picker("", camera_picker(CAMERA_NAMES, current=3))
        view.add_picker("", photo_picker(range(12), current=5,
                                         details={5: "0.42 px RMS per photo"}))
    return view


def grab(widget, path, size):
    from PySide6.QtWidgets import QApplication
    widget.resize(*size)
    widget.show()
    for _ in range(6):
        QApplication.processEvents()
    canvas = getattr(widget, "canvas", None)
    if canvas is not None:
        canvas.draw()
    for _ in range(3):
        QApplication.processEvents()
    widget.grab().save(str(path))
    print("wrote", path)


def pieces_figure(fig, theme):
    """The shared rules on one page: colours, outline rule, badges, error scale, wording."""
    from matplotlib.patches import Circle, Rectangle
    from pyCamSet.gui import education_style as es
    t = es.stage_tokens(theme)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1])

    a = fig.add_subplot(gs[0, 0:2])
    a.set_xlim(0, 10)
    a.set_ylim(0, 4.2)
    a.set_axis_off()
    for k, colour in enumerate(es.CAMERA_PALETTE):
        a.add_patch(Rectangle((0.2 + 1.2 * k, 2.55), 1.0, 0.9,
                              **es.camera_mark_kwargs(k, theme)))
        a.text(0.7 + 1.2 * k, 2.3, f"cam {k + 1}", ha="center", va="top", fontsize=9,
               color=t["text"])
    for k, role in enumerate(("accent", "link", "warn", "bad", "good")):
        colour = t[role]
        distance = min(es.colour_distance(colour, c) for c in es.CAMERA_PALETTE)
        a.add_patch(Rectangle((0.2 + 1.9 * k, 0.55), 1.0, 0.75, facecolor=colour))
        a.text(1.3 + 1.9 * k, 1.05, role, fontsize=10, color=t["text"], va="center")
        a.text(1.3 + 1.9 * k, 0.7, f"ΔE ≥ {distance:.0f}", fontsize=8.5, color=t["muted"],
               va="center")
    es.panel_title(a, "A", "Figure colours never look like a camera",
                   f"rule: ΔE ≥ {es.MIN_CAMERA_DISTANCE:.0f} from every camera colour, "
                   f"contrast ≥ {es.MIN_STAGE_CONTRAST:.0f}")

    b = fig.add_subplot(gs[0, 2])
    b.set_xlim(-0.5, 7.5)
    b.set_ylim(-3.0, 1.4)
    b.set_axis_off()
    es.draw_camera_badges(b, [(k, 0.75) for k in range(8)], range(8), size=11)
    for k in range(8):  # dot, raster cell, bar, thick line: the one fill per theme
        b.scatter([k], [-0.1], s=70, **es.camera_mark_kwargs(k, theme, "dot"))
        b.add_patch(Rectangle((k - 0.32, -0.95), 0.64, 0.5, **es.camera_mark_kwargs(k, theme)))
        b.add_patch(Rectangle((k - 0.18, -2.05), 0.36, 0.9, **es.camera_mark_kwargs(k, theme)))
        b.plot([k - 0.32, k + 0.32], [-2.55, -2.55],
               **{**es.camera_mark_kwargs(k, theme, "line"), "linewidth": 3.0})
    swapped = [k + 1 for k in range(8) if es.camera_needs_outline(k, theme)]
    es.panel_title(b, "B", "One camera look", "badge, dot, cell, bar, line; own fill: "
                   + (", ".join(map(str, swapped)) or "none"))

    c = fig.add_subplot(gs[1, 0])
    scale = es.error_scale(theme)
    values = np.geomspace(0.12, 90, 40)
    c.scatter(np.arange(40), values, c=values, **scale.scatter_kwargs(),
              **es.value_dot_kwargs(theme, 30))
    scale.log_axis(c)
    c.set_ylabel(es.error_label("mean", short=True))
    c.set_xlabel("photo")
    es.integer_axis(c, "x")
    scale.colourbar(fig, c)
    es.panel_title(c, "C", "One error scale", "log 0.1–100 px in every phase")

    d = fig.add_subplot(gs[1, 1])
    d.set_axis_off()
    lines = [es.error_label(), es.error_label("rms", per="photo"),
             es.photos_label([25, 29, 69, 83, 89]), es.camera_title(0, CAMERA_NAMES[0]),
             es.format_change(3.2954, 0.297), es.format_um(0.1723) + " mean correction",
             es.format_count(15447, "detection")]
    for k, line in enumerate(lines):
        d.text(0.0, 0.95 - k * 0.135, line, fontsize=10.5, color=t["text"],
               transform=d.transAxes, va="top")
    es.panel_title(d, "D", "One wording", "error name, photo numbers, numbers")

    e = fig.add_subplot(gs[1, 2])
    bars = es.bars_by_camera(e, [1.51, 1.33, 1.31, 2.9, 3.5, 3.7, 3.5, 1.2], threshold=3.6,
                             fail_below=False, ylabel="RMS", unit="px",
                             threshold_label="limit")
    es.panel_title(e, "E", "Narrow panel", "value labels hide, badges shrink, never collide")
    del bars


def faces_figure(fig, theme, photo, corners):
    """Round 2: the one face look, a face matrix and the two colour scales."""
    from matplotlib.patches import Polygon
    from pyCamSet.gui import education_style as es
    t = es.stage_tokens(theme)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
    a = fig.add_subplot(gs[0, 0])
    a.set_aspect("equal")
    a.set_axis_off()
    plates = {0: [(0, 1), (1, 1.5), (2, 1), (1, 0.5)], 1: [(0, 1), (1, 0.5), (1, -0.7), (0, -0.2)],
              5: [(1, 0.5), (2, 1), (2, -0.2), (1, -0.7)]}
    for face, outline in plates.items():
        xy = np.array(outline)
        a.add_patch(Polygon(xy, closed=True, facecolor=es.face_colour(face, theme),
                            alpha=0.25, edgecolor=es.face_colour(face, theme), linewidth=1.6))
    es.draw_face_tags(a, [np.mean(v, axis=0) for v in plates.values()], list(plates),
                      size=10, selected=[1])
    a.set_xlim(-0.2, 2.2)
    a.set_ylim(-0.9, 1.7)
    es.panel_title(a, "A", "Your target", "faces: grey + square tag; F2 selected")

    b = fig.add_subplot(gs[0, 1])
    b.imshow(photo, cmap="gray")
    b.set_xticks([])
    b.set_yticks([])
    half = len(corners) // 2
    for face, pts in ((1, corners[:half]), (5, corners[half:])):
        b.scatter(pts[:, 0], pts[:, 1], zorder=4, **es.face_point_kwargs(face, theme,
                                                                          on_photo=True))
    es.frame_axes(b, 5, width=3.0)
    b.legend(handles=es.face_legend_handles([1, 5], theme, counts={1: half, 5: half}),
             loc="upper center", bbox_to_anchor=(0.5, -0.01), ncols=2)
    es.panel_title(b, "B", "Seen by camera 6", "corners marked by face (shape + shade)")

    c = fig.add_subplot(gs[0, 2])
    seen = np.random.default_rng(4).integers(0, 30, size=(8, 6)).astype(float)
    seen[4:, 3:] = 0
    top = seen.max()
    for cam in range(8):
        for face in range(6):
            if seen[cam, face] <= 0:
                c.scatter([face], [cam], s=8, color=t["grid"])
                continue
            c.scatter([face], [cam], s=30 + 330 * seen[cam, face] / top, zorder=3,
                      color=t["accent"] if face == 1 else t["muted"])
    c.set_xlim(-0.6, 5.6)
    c.set_ylim(7.6, -0.6)
    es.face_ticks(c, range(6), range(6), selected=[1])
    es.badge_ticks_y(c, range(8), range(8), size=9, highlight=[5])
    for spine in c.spines.values():
        spine.set_visible(False)
    es.panel_title(c, "C", "Which camera sees which face", "dot size = photos; F2 in accent")

    rng = np.random.default_rng(9)
    heat = 1.1 + 1.9 * np.exp(-((np.arange(6)[None, :] - 4.5) ** 2 / 6
                                 + (np.arange(5)[:, None] - 0.5) ** 2 / 4))
    heat += rng.normal(0, 0.08, heat.shape)
    d = fig.add_subplot(gs[1, 0])
    shared = es.error_scale(theme)
    im = d.imshow(heat, cmap=shared.cmap, norm=shared.norm())
    shared.colourbar(fig, d, im)
    d.set_xticks([])
    d.set_yticks([])
    es.panel_title(d, "D", "Shared log scale", "1–3 px: one colour, pattern lost")
    e = fig.add_subplot(gs[1, 1])
    local = es.local_scale(heat, theme)
    im = e.imshow(heat, **local.imshow_kwargs())
    local.colourbar(fig, e, im)
    e.set_xticks([])
    e.set_yticks([])
    es.panel_title(e, "E", "Local range scale", "same data: where the error is largest")
    f = fig.add_subplot(gs[1, 2])
    f.set_axis_off()
    rows = [("camera", "colour + round badge", "camera"), ("face", "grey + square tag", "face"),
            ("error", "viridis + px bar", "error")]
    for k, (what, how, _key) in enumerate(rows):
        f.text(0.0, 0.85 - 0.25 * k, what, fontsize=11, fontweight="bold", color=t["text"],
               transform=f.transAxes, va="center")
        f.text(0.32, 0.85 - 0.25 * k, how, fontsize=10.5, color=t["text"],
               transform=f.transAxes, va="center")
    es.draw_camera_badges(f, [(0.92, 0.85)], [5], transform=f.transAxes, size=10)
    es.draw_face_tags(f, [(0.92, 0.60)], [1], transform=f.transAxes)
    es.panel_title(f, "F", "Three encodings, never mixed", "")


def round3_figure(fig, theme):
    """Round 3: which colour map for which quantity, card text that fits, neutral limits."""
    from matplotlib.patches import FancyBboxPatch
    from pyCamSet.gui import education_style as es
    from pyCamSet.gui.education_stage import theme_tokens_for
    t = es.stage_tokens(theme)
    tok = theme_tokens_for(theme)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    rng = np.random.default_rng(3)

    a = fig.add_subplot(gs[0, 0])
    counts = rng.poisson(6, size=(7, 9)).astype(float) * np.linspace(0.2, 1.4, 9)[None, :]
    counts[0, :3] = 0
    amount = es.amount_scale(counts, theme, unit="points")
    image = a.imshow(np.ma.masked_where(counts <= 0, counts), **amount.imshow_kwargs())
    amount.colourbar(fig, a, image)
    a.set_xticks([])
    a.set_yticks([])
    es.panel_title(a, "A", "Counts: one-hue amount map", "points per cell (never viridis)")

    b = fig.add_subplot(gs[0, 1])
    yy, xx = np.mgrid[-1:1:24j, -1:1:24j]
    change = 380 * xx * np.exp(-(xx ** 2 + yy ** 2))
    signed = es.signed_scale(change, theme, unit="µm")
    image = b.imshow(change, **signed.imshow_kwargs())
    signed.colourbar(fig, b, image)
    b.set_xticks([])
    b.set_yticks([])
    es.panel_title(b, "B", "Signed: diverging map", "blue shrank · magenta grew · 0 neutral")

    c = fig.add_subplot(gs[0, 2])
    scale = es.error_scale(theme)
    values = np.geomspace(0.15, 60, 30)
    dots = c.scatter(np.arange(30), values, c=values, **scale.scatter_kwargs(),
                     **es.value_dot_kwargs(theme, 30))
    scale.log_axis(c)
    scale.colourbar(fig, c, dots)
    c.set_xlabel("photo")
    es.panel_title(c, "C", "Errors: viridis + px bar", "yellow = far off, in every phase")

    d = fig.add_subplot(gs[1, 0:2])
    d.set_axis_off()
    cards = [("Rank check", "good", "0 free directions · 3 once-seen corners may slide (fine)"),
             ("Photos used", "warn", "94 / 99 · 5 photos left out: no target pose"),
             ("Solver", "good", "Converged · stopped: the error stopped falling")]
    sub = gs[1, 0:2].subgridspec(1, len(cards), wspace=0.05)
    d.remove()
    for k, (title, tone, finding) in enumerate(cards):
        ax = fig.add_subplot(sub[0, k])
        ax.set_axis_off()
        patch = FancyBboxPatch((0.01, 0.04), 0.98, 0.9, boxstyle="round,pad=0,rounding_size=0.05",
                               facecolor=tok["surface_alt"], edgecolor=tok["border"],
                               transform=ax.transAxes, clip_on=False)
        ax.add_patch(patch)
        ax.add_patch(FancyBboxPatch((0.01, 0.04), 0.03, 0.9, boxstyle="round,pad=0",
                                    facecolor=t[tone], edgecolor="none", transform=ax.transAxes))
        ax.text(0.09, 0.78, title, fontsize=10, fontweight="bold", color=t["text"],
                transform=ax.transAxes, va="center")
        es.card_text(ax, 0.09, 0.42, finding, card=patch, transform=ax.transAxes,
                     fontsize=9, color=t["text"], va="center")
        if k == 0:
            es.panel_title(ax, "D", "Card text stays in its card",
                           "card_text wraps / shrinks at every layout")

    e = fig.add_subplot(gs[1, 2])
    rms = [1.5, 2.1, 1.3, 3.2, 2.8, 1.9]
    e.bar(range(6), rms, color=t["muted"], width=0.6)
    es.limit_line(e, 5.0, values=rms, label="5 px limit (not crossed)")
    es.limit_line(e, 3.0, values=rms, label="3 px (crossed)")
    e.set_ylim(0, 6)
    e.set_ylabel("RMS error (px)")
    es.panel_title(e, "E", "Limits: grey until crossed", "red only where a value fails")


def split_stage(view, theme):
    """Round 2: a 3D panel under the shared PanelHeading beside a figure panel, with
    camera trails (fading path, ghost start pose, arrowhead)."""
    import matplotlib
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
    from pyCamSet.gui import education_style as es
    from pyCamSet.gui.education_scene3d import Scene3D, camera_centre
    from pyCamSet.gui.education_stage import PanelHeading
    host = QWidget()
    row = QHBoxLayout(host)
    row.setContentsMargins(0, 0, 0, 0)
    left = QWidget(host)
    column = QVBoxLayout(left)
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(0)
    heading = PanelHeading(left)
    heading.set_heading("A", "The solver at work",
                        "changes shown ×5 · line: each camera's path · ghost: where it started")
    scene = Scene3D(left, theme=theme)
    column.addWidget(heading)
    column.addWidget(scene, 1)
    row.addWidget(left, 11)
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasQTAgg(fig)
    row.addWidget(canvas, 10)
    add_trails(scene, theme)
    with matplotlib.rc_context(es.figure_style(theme)):
        fig._edu_theme = theme
        fig.set_facecolor(es.stage_tokens(theme)["stage"])
        es.align_panel_titles(fig)
        ax = fig.add_subplot()
        es.strip_plot_by_camera(ax, strip_values(), log=True, ylabel="mean error")
        es.panel_title(ax, "B", "Each camera's error at this step", "log scale, px")
    view.set_visual(host)
    view.set_caption("Each camera's trail fades from where it started (ghost) to where it is "
                     "now (arrow). Short moves are hidden.")
    return host, scene


def add_trails(scene, theme):
    """8-camera ring with trails for three cameras that moved (start pose ghosted)."""
    from scipy.spatial.transform import Rotation
    from pyCamSet.gui.education_scene3d import camera_centre
    poses, pts, err = build_ring(scene, with_rays=False, selected=5)
    rng = np.random.default_rng(7)
    for k in (1, 3, 5):
        R, t = poses[k]
        end = camera_centre(R, t)
        start = end + rng.normal(0, 0.22, 3) + np.array([0, 0, 0.25])  # shown x5
        mids = [start + (end - start) * s + 0.08 * np.sin(np.pi * s) * np.array([0, 0, 1])
                for s in np.linspace(0, 1, 7)]
        R0 = Rotation.from_rotvec(rng.normal(0, 0.08, 3)).as_matrix() @ R
        scene.add_trail(np.array(mids), camera=k, start_pose=(R0, -R0 @ start))
    # a camera that barely moved: its trail is hidden (a stub would read as a glitch)
    R, t = poses[0]
    c0 = camera_centre(R, t)
    scene.add_trail(np.array([c0 + [0.002, 0, 0], c0]), camera=0)
    scene.fit_view(selected=5)
    return poses


def steps_stage(view, photo, corners):
    """A step-through stage (Phase 1 style): named step pills inside the one transport bar."""
    from pyCamSet.gui import education_style as es
    from pyCamSet.gui.education_dialog import TransportBar
    from pyCamSet.gui.education_stage import Chip
    names = ["Outlines", "Squares", "Read IDs", "Markers", "Corners", "Grid check",
             "Sub-pixel", "Final"]
    bar = TransportBar(len(names), names=names, unit_label="Detector steps",
                       interval_ms=1600, min_interval_ms=400, start=4)
    bar.add_menu_choice("Exaggerate", ["×1", "×5", "×20"], 1, lambda _k: None)

    def draw(fig, _theme):
        a, b = fig.subplots(1, 2, width_ratios=[1.6, 1])
        es.image_with_points(a, photo, corners, camera=3)
        es.panel_title(a, "A", "Corners where markers meet", f"camera 4, {es.photo_label(5)}")
        crop = photo[300:420, 420:560]
        es.image_with_points(b, crop, corners[(corners[:, 0] > 420) & (corners[:, 0] < 560)
                                              & (corners[:, 1] > 300) & (corners[:, 1] < 420)]
                             - [420, 300], radius=4)
        es.panel_title(b, "B", "Close-up", "accent rings: corners found")

    view.draw_figure(draw)
    view.set_controls(bar)
    view.set_caption("Where four markers meet, the detector puts a corner. Step through the "
                     "detector with the bar below; the pills name each step.")
    view.set_chips([Chip("48", "corners"), Chip("19", "markers")])
    return bar


def glyph_closeup(scene, theme):
    """Pinhole and telecentric glyphs beside a cube target (UX-9: never confusable)."""
    from pyCamSet.gui.education_style import error_scale
    pts = cube_points(0.3, 5)
    R1, t1 = look_at((0.9, -0.7, 0.35))
    R2, t2 = look_at((-0.9, -0.7, 0.35))
    scene.add_camera("cam0", 0, R1, t1, scale=0.3, telecentric=False, label="(0.47) pinhole")
    scene.add_camera("cam5", 5, R2, t2, scale=0.3, telecentric=True, label="(0.58) telecentric")
    err = np.abs(np.random.default_rng(2).gamma(2.0, 0.4, len(pts)))
    scene.add_points(pts, err, name="target", scalar_bar="px", size=10,
                     **error_scale(theme).scene_kwargs())
    scene.fit_view(selected=5)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "edu-foundation"),
                        help="output folder (default: edu-foundation under the system temp folder)")
    parser.add_argument("--no-pyvista", action="store_true")
    parser.add_argument("--themes", default="Light,Dark,Sepia")
    args = parser.parse_args(argv)
    _below_normal_priority()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
    from pyCamSet.gui.theme import apply_theme
    app = QApplication.instance() or QApplication([])
    app.setFont(QFont("Segoe UI", 10))

    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from pyCamSet.gui import education_style as es
    from pyCamSet.gui.education_scene3d import OffscreenScene, Scene3D, ScenePlayer, pace_weights
    from pyCamSet.gui.education_stage import Chip

    photo, corners = synthetic_photo()
    stages = ["Photos", "What the detector sees", "Finding markers", "Final", "Decision"]
    stages3 = ["Who saw what", "First guess", "Single-face images", "Watching the solver",
               "Result"]
    problems = 0
    for theme in [t.strip() for t in args.themes.split(",") if t.strip()]:
        apply_theme(app, theme)

        def shell():
            # The dialog surface around the stage, as the education dialog shows it.
            frame = QWidget()
            frame.setObjectName("galleryShell")
            frame.setAutoFillBackground(True)
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(12, 12, 12, 12)
            return frame, layout

        def check(figure, name):
            nonlocal problems
            issues = (es.text_collisions(figure) + es.card_overflows(figure, draw=False)
                      + es.colour_map_misuse(figure))
            problems += len(issues)
            print(f"text check {name}: {len(issues)} issue(s)")
            for issue in issues:
                print("   ", issue)

        frame, layout = shell()
        view = build_view(stages)
        view.set_current_stage(3)
        view.set_level(2)
        layout.addWidget(view)
        figure_stage(view, photo, corners, theme)
        grab(frame, out / f"stage_figure_{theme}.png", (1400, 900))
        check(view.figure(), f"stage_figure_{theme}")
        grab(frame, out / f"stage_figure_{theme}_1280.png", (1280, 800))
        check(view.figure(), f"stage_figure_{theme}_1280")
        frame.hide()
        frame.deleteLater()

        frame, layout = shell()
        view = build_view(stages)
        view.set_current_stage(2)
        layout.addWidget(view)
        bar = steps_stage(view, photo, corners)
        grab(frame, out / f"stage_steps_{theme}.png", (1400, 900))
        check(view.figure(), f"stage_steps_{theme}")
        bar.stop_for_context_change()
        frame.hide()
        frame.deleteLater()

        frame, layout = shell()
        view = build_view(stages3, header_pickers=False)
        view.add_picker("", __import__("pyCamSet.gui.education_stage", fromlist=["x"])
                        .camera_picker(CAMERA_NAMES, current=5))
        view.set_current_stage(3)
        view.set_level(2)
        layout.addWidget(view)
        scene = Scene3D()
        poses, pts, err = build_ring(scene, selected=5)
        frames = solver_frames(poses, pts, err)
        mean_error = [float(np.mean(f["points"]["target"]["scalars"])) for f in frames]
        player = ScenePlayer(scene, frames, total_steps=9, step_numbers=[1, 2, 3, 5, 7, 9],
                             reduce_motion=False, step_weights=pace_weights(mean_error))
        view.set_visual(scene)
        view.set_controls(player)
        player.go_to(3, animate=False)
        player.show_position(2.5)  # mid-flight between recorded steps 3 and 5
        view.set_caption("The solver moves every camera and target pose a little at each "
                         "accepted step; cameras settle into the ring and the corners' "
                         "reprojection error shrinks.")
        view.set_chips([Chip("6 / 9", "steps recorded"),
                        Chip(es.format_change(1.92, 0.41, ""), es.error_label(), tone="good"),
                        Chip("8", "cameras")])
        view.set_numbers(["Step", "Cost"], [(n, f"{2.0 / (n + 1):.3f}") for n in range(6)])
        grab(frame, out / f"stage_3d_{theme}.png", (1400, 900))
        player.stop_for_context_change()
        frame.hide()
        frame.deleteLater()

        frame, layout = shell()
        view = build_view(stages3, header_pickers=False)
        view.add_picker("", __import__("pyCamSet.gui.education_stage", fromlist=["x"])
                        .camera_picker(CAMERA_NAMES, current=7))
        view.set_current_stage(3)
        view.set_level(2)
        layout.addWidget(view)
        split_stage(view, theme)
        grab(frame, out / f"split_{theme}.png", (1280, 800))
        frame.hide()
        frame.deleteLater()

        with matplotlib.rc_context(es.figure_style(theme)):
            fig = Figure(figsize=(14, 7.6), dpi=100)
            FigureCanvasAgg(fig)
            fig._edu_theme = theme
            es.align_panel_titles(fig)
            faces_figure(fig, theme, photo, corners)
            fig.savefig(out / f"faces_{theme}.png", facecolor=fig.get_facecolor())
            print("wrote", out / f"faces_{theme}.png")
            check(fig, f"faces_{theme}")

        with matplotlib.rc_context(es.figure_style(theme)):
            fig = Figure(figsize=(14, 7.2), dpi=100)
            FigureCanvasAgg(fig)
            fig._edu_theme = theme
            es.align_panel_titles(fig)
            pieces_figure(fig, theme)
            fig.savefig(out / f"pieces_{theme}.png", facecolor=fig.get_facecolor())
            print("wrote", out / f"pieces_{theme}.png")
            check(fig, f"pieces_{theme}")

        with matplotlib.rc_context(es.figure_style(theme)):
            fig = Figure(figsize=(14, 7.2), dpi=100)
            FigureCanvasAgg(fig)
            fig._edu_theme = theme
            es.align_panel_titles(fig)
            round3_figure(fig, theme)
            fig.savefig(out / f"round3_{theme}.png", facecolor=fig.get_facecolor())
            print("wrote", out / f"round3_{theme}.png")
            check(fig, f"round3_{theme}")

        if not args.no_pyvista:
            off = OffscreenScene(theme=theme, window_size=(1400, 900))
            build_ring(off, selected=5)
            off.screenshot(out / f"pyvista_ring_{theme}.png")
            print("wrote", out / f"pyvista_ring_{theme}.png")
            off.close()
            off = OffscreenScene(theme=theme, window_size=(1400, 900))
            add_trails(off, theme)
            off.screenshot(out / f"pyvista_trails_{theme}.png")
            print("wrote", out / f"pyvista_trails_{theme}.png")
            off.close()
            off = OffscreenScene(theme=theme, window_size=(1100, 700))
            glyph_closeup(off, theme)
            off.screenshot(out / f"pyvista_glyphs_{theme}.png")
            print("wrote", out / f"pyvista_glyphs_{theme}.png")
            off.close()
    print(f"text problems in gallery figures: {problems}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
