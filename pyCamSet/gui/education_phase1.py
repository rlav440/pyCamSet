"""Teach Phase 1 ("find the target in every photo") visually, with this run's own photos.

Every stage is a figure at each detail level, built on the shared figure style
(``education_style``) and transport bar (``education_dialog.TransportBar``). Stages 3-4 draw a
live, checked re-run of Phase 1's own detection on the chosen photo
(``workflow/education_detect_explain``); the other stages draw the run's saved data and target.

Stages:

1. Target - what a calibration target is, with this run's own target (its faces and design
   corners in 3D), one of its photos with the corners coloured by face, and which camera saw
   which face (why a cube: cameras all around share faces).
2. Photos - every camera's photo of the chosen photo number, badge + point count.
3. What the detector sees - photo -> one global cut-off vs the detector's local threshold, with
   one pixel's neighbourhood drawn and its test written out.
4. Finding markers - step through the detector (transport bar): outlines, squares, IDs (bits ->
   code -> dictionary lookup), markers, corners, sub-pixel (edges crossing), grid check, result.
5. Final - saved circles on the photo + where this camera's points fall over all photos.
6. Decision - per-camera bars (photos with the target, coverage, ...) in camera colours.

Photos are numbered from 1 (``photo_label``) like cameras; the stored 0-based index appears only
in the Numbers drawer.
"""
from __future__ import annotations

import textwrap
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QSize, QTimer
from PySide6.QtWidgets import QLabel, QWidget

from pyCamSet.gui.education_levels import orientation, phase_name
from pyCamSet.gui.education_photo_cache import ByteLRU, PhotoPrefetcher
from pyCamSet.gui.education_stage import Chip, EducationStageView, camera_picker, photo_picker
from pyCamSet.gui.education_style import (
    REJECTED_ALPHA, TICK_SIZE, badge_ticks_y, bars_by_camera, camera_colour,
    amount_hue, amount_scale, camera_mark_kwargs, colourbar_px, draw_camera_badges, empty_panel,
    face_label, face_marker, format_count, format_number, frame_axes, integer_axis, limit_line,
    limit_style, panel_title, photo_label, relative_luminance, stage_tokens, tokens_for,
)
from pyCamSet.workflow import education_visual_p1 as vis

STAGES = ("Target", "Photos", "What the detector sees", "Finding markers", "Final", "Decision")
#: Stage index by key, so code and tests never count positions.
STAGE = {key: k for k, key in enumerate(("target", "photos", "prep", "find", "final", "decision"))}

# A1-approved captions are composed by reason, not by a guessed cache story.
_PHOTOLESS_COPY = {
    "reuse": (
        "No photo is shown because its detections may have come from an earlier photo set; "
        "the saved record does not say which photo each detection came from. In Phase 1, "
        "turn off ‘Cache detections’ and run detection again. Before a later run with caching "
        "on, remove the matching `detected_datapoints*.npz` cache file from the image folder; "
        "do not delete the run's saved results. If the folder's photos change again, repeat "
        "these steps; the cache does not record image contents."
    ),
    "layout": (
        "No photo is shown because the photos in its folder cannot be compared with its "
        "detection run's list. In Phase 1, turn off ‘Cache detections’ and run detection "
        "again, then continue from that new run."
    ),
    "count": (
        "No photo is shown because the number of image files in its folder has changed since "
        "its detection run. In Phase 1, turn off ‘Cache detections’ and run detection again, "
        "then continue from that new run."
    ),
    "changed": (
        "No photo is shown because the photo list of its detection run no longer matches the "
        "photos in the folder. In Phase 1, turn off ‘Cache detections’ and run detection again, "
        "then continue from that new run."
    ),
}
_PHOTOLESS_COPY_KEYS = frozenset(_PHOTOLESS_COPY)

#: One-line versions for inside a figure (the caption carries the full copy above).
_PHOTOLESS_SHORT = {
    "reuse": "Photo not shown: these detections may come from an earlier photo set.",
    "layout": "Photo not shown: the folder cannot be compared with the run's photo list.",
    "count": "Photo not shown: the folder's number of images has changed since the run.",
    "changed": "Photo not shown: the folder's photos differ from the run's photo list.",
}


# ---------------------------------------------------------------------------------------------
# Run data: photo identity, run-private detections (kept from the static-first view)
# ---------------------------------------------------------------------------------------------

def _plain(value: Any) -> Any:
    """Convert the immutable JSON view's nested mappings for workflow helpers."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


#: Photo side length the Phase 1 tiles are drawn at (the display copy, not the decode).
_PHOTO_TILE_SIDE = 720


def _decode_display_photo(path: Path) -> tuple[Optional[np.ndarray], Optional[str], Optional[tuple]]:
    """Decode one photo for the tiles: (display image, decode problem or None, header size).

    Safe on any thread: it touches no widget and no view state.
    """
    image = vis.read_display_photo(path, max_side=_PHOTO_TILE_SIDE)
    reason = None
    if image is None:
        reason = f"Photo unavailable: {path.name} could not be decoded."
    size = vis.photo_size(path)
    return image, reason, size


def _has_tif_below(folder: Path) -> bool:
    """True when any .tif file sits anywhere under ``folder``; stops at the first one found."""
    import os
    for _root, _dirs, files in os.walk(folder):
        if any(Path(name).suffix.lower() == ".tif" for name in files):
            return True
    return False


def _source_state(context) -> tuple[Any, dict[str, tuple[str, ...]], str]:
    """Return the manifest verdict and writer-owned per-camera image names."""
    from pyCamSet.workflow.education_manifest import ManifestResult

    metadata = _plain(context.metadata)
    diagnostics = metadata.get("diagnostics") or {}
    params = metadata.get("params") or {}
    manifest = diagnostics.get("education_manifest")
    cache = diagnostics.get("education_cache") or {}
    if callable(getattr(context, "phase1_source_check", None)):
        result = context.phase1_source_check()
    else:
        from pyCamSet.workflow.education_trace import RunContext
        run_context = RunContext.from_run(
            context.manager, {**metadata, "_run_dir": context.run_dir}, "phase1")
        result = run_context.phase1_source_check()
    names: dict[str, tuple[str, ...]] = {}
    if isinstance(manifest, Mapping) and manifest.get("recordable") is True:
        stored = manifest.get("cameras") or {}
        names = {str(camera): tuple(
            str(item["name"]) for item in entries
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        ) for camera, entries in stored.items() if isinstance(entries, list)}
        return result, names, "manifest"

    # Legacy folders are mapped only under the conservative flat/count/cache rule.
    folder_value = params.get("f_loc")
    if not folder_value:
        return ManifestResult("could not be checked", {}, {}, "image folder unavailable"), {}, "layout"
    root = Path(str(folder_value))
    camera_names = diagnostics.get("cam_names") or []
    if not isinstance(camera_names, list) or not camera_names:
        return ManifestResult("could not be checked", {}, {}, "camera list unavailable"), {}, "layout"
    from natsort import natsorted
    from pyCamSet.utils.general_utils import glob_ims
    from pyCamSet.workflow.education_trace import classify_legacy_pose_mapping

    report = metadata.get("report") or {}
    report_rows = report.get("per_camera") or []
    expected = {str(row.get("name")): row.get("n_images_expected")
                for row in report_rows if isinstance(row, Mapping)}
    caching = params.get("caching")
    capped = params.get("n_lim") is not None
    reason = None
    for camera in camera_names:
        image_folder = root / str(camera)
        subfolder = params.get("subfolder_string")
        if subfolder:
            image_folder /= str(subfolder)
        try:
            paths = natsorted(glob_ims(image_folder))
            # Only a folder whose photos all sit directly inside it can be flat, so the
            # subtree walk for .tif files (slow on big or external drives) runs only then.
            flat = (image_folder.is_dir() and all(path.parent == image_folder for path in paths)
                    and not _has_tif_below(image_folder))
        except OSError:
            paths, flat = [], False
        if capped and isinstance(params.get("n_lim"), int) and not isinstance(params.get("n_lim"), bool):
            paths = paths[:max(0, params["n_lim"])]
        mapping = classify_legacy_pose_mapping(
            caching=caching, promoted=bool(diagnostics.get("promoted_from")),
            max_ims=_max_ims_from_run(context),
            current_photo_names=tuple(path.name for path in paths),
            flat_without_tif=flat,
            report={"n_images_expected": expected.get(str(camera)),
                    "n_images": report.get("n_images")},
            listing_capped=capped,
        )
        if mapping.reason is not None:
            reason = mapping.reason
        else:
            names[str(camera)] = mapping.photo_names
    if reason is not None:
        from pyCamSet.workflow.education_trace import PoseMappingReason
        text = str(reason.value if hasattr(reason, "value") else reason)
        if reason is PoseMappingReason.POSSIBLE_STALE_HIT or cache.get("reused") is True:
            copy_key = "reuse"
        elif reason is PoseMappingReason.PHOTO_LIST_CHANGED:
            copy_key = "changed"
        elif reason is PoseMappingReason.IMAGE_COUNT_CHANGED:
            copy_key = "count"
        else:
            copy_key = "layout"
        result = ManifestResult("mismatch" if copy_key == "changed" else "could not be checked",
                                {}, {}, text)
        return result, {}, copy_key
    return ManifestResult("could not be checked", names, {}, "legacy source identity not recorded"), names, "legacy"


def _max_ims_from_run(context) -> int:
    """Read only the run-private detection artifact; shared image-folder caches are forbidden."""
    detections, _reason = _load_run_detections(context)
    try:
        return int(detections.max_ims) if detections is not None else 0
    except (AttributeError, TypeError, ValueError):
        return 0


def _load_run_detections(context):
    """Load and type-check this run's own pickle; never fall back to a cache slot."""
    from pyCamSet.workflow.education_trace import load_run_pickle
    from pyCamSet.workflow.workspace import path_exists
    from pyCamSet.calibration_targets.core.target_detections import TargetDetection

    candidates = [context.run_dir / "detected_datapoints.pickle"]
    artifacts = _plain(context.metadata).get("artifacts") or {}
    recorded = artifacts.get("detected_datapoints_pickle")
    if recorded:
        candidate = Path(str(recorded))
        try:
            candidate.resolve(strict=False).relative_to(context.run_dir.resolve(strict=False))
            candidates.insert(0, candidate)
        except (OSError, ValueError):
            pass
    for path in candidates:
        if path_exists(path):
            payload, error = load_run_pickle(path, (tuple, TargetDetection),
                                            expected_run_dir=context.run_dir)
            if error:
                return None, error
            if isinstance(payload, tuple) and len(payload) == 2:
                payload = payload[0]
            if isinstance(payload, TargetDetection):
                return payload, None
            return None, "pickle payload has no supported detections"
    return None, "this run's detections were not saved"


def _rows_for(detections, camera: str, image_index: int) -> np.ndarray:
    """Return points by camera name and pose index without changing saved rows."""
    return vis.camera_points(detections, camera, image_index)


def _photo_path(root: Path, camera: str, name: str, subfolder: str | None = None) -> Path:
    """Resolve a manifest-relative camera photo and reject path escapes."""
    base = (root / camera / subfolder).resolve() if subfolder else (root / camera).resolve()
    candidate = (base / name).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return base / "__unavailable__"
    return candidate


def _reason_caption(key: str) -> str:
    return _PHOTOLESS_COPY.get(key, "No photo is shown because this run's photo mapping cannot be checked. "
                               "In Phase 1, turn off ‘Cache detections’ and run detection again, "
                               "then continue from that new run.")


def _target_family_of_params(params: Mapping[str, Any]) -> Optional[str]:
    """The preparation family of the run's recorded target, or None when it is not recorded."""
    try:
        from pyCamSet.calibration_targets.core.preparation import family_of_target_type
        from pyCamSet.workflow.targets import target_spec_of
        return family_of_target_type(str(target_spec_of(dict(params)).get("type")))
    except (KeyError, TypeError, ValueError, ImportError):
        return None


def _rescale_and_gamma_state(params: Mapping[str, Any]) -> str:
    """Whether rescale-and-gamma really ran, the way Phase 1 detection decides it.

    Since 2026-10-13 Phase 1 applies a ticked box to any target and records
    ``rescale_and_gamma_any_target`` in the params; runs saved before that
    honoured it for PuzzleBoard targets only and ignored it for every other
    target, while the saved params keep the ticked value. Returns "off" (not
    ticked), "applied", "ignored" (ticked, an older run, not a PuzzleBoard
    target) or "unknown" (ticked, an older run, target not recorded).
    """
    from pyCamSet.calibration_targets.core.preparation import ANY_TARGET_KEY, PUZZLEBOARD
    if params.get("rescale_and_gamma") is not True:
        return "off"
    if params.get(ANY_TARGET_KEY) is True:
        return "applied"
    family = _target_family_of_params(params)
    if family is None:
        return "unknown"
    return "applied" if family == PUZZLEBOARD else "ignored"


def _resolved_preparation(params: Mapping[str, Any]):
    """The preparation this run's saved params resolve to, or None when off or not resolvable."""
    from pyCamSet.calibration_targets.core.preparation import resolve_for_family
    family = _target_family_of_params(params)
    if family is None:
        return None
    try:
        return resolve_for_family(params, family)
    except (TypeError, ValueError):
        return None


def _detector_input_copy(params: Mapping[str, Any]) -> str:
    """Say, from this run's saved settings, what the detector searched (A1 section 10).

    Mirrors the detection worker: rescale-and-gamma (any target; PuzzleBoard
    only for runs saved before 2026-10-13) replaces the upscale, an upscale
    factor above 1 enlarges the photo, otherwise the photo is used as is.
    """
    if "rescale_and_gamma" not in params and "upscale_factor" not in params:
        return "This run's detector settings were not recorded, so what the detector searched is unknown."
    state = _rescale_and_gamma_state(params)
    if state == "unknown":
        return ("Rescale and gamma was ticked for this run, but runs saved before 2026-10-13 "
                "applied it only to PuzzleBoard targets and the target type was not recorded, "
                "so what the detector searched is unknown.")
    prefix = ""
    if state == "ignored":
        prefix = ("Rescale and gamma was ticked, but runs saved before 2026-10-13 applied it "
                  "only to PuzzleBoard targets, so it was not used for this run. ")
    if state == "applied":
        prep = _resolved_preparation(params)
        if prep is not None:
            # Blank fields took the target's defaults; show the values that ran.
            scale, gamma = prep.scale, prep.gamma
        else:
            scale, gamma = params.get("preprocessing_scale"), params.get("preprocessing_gamma")
        size = f"resized each photo ×{scale:g}" if isinstance(scale, (int, float)) else "resized each photo"
        tone = (f"adjusted its brightness and contrast (gamma {gamma:g})"
                if isinstance(gamma, (int, float)) else "adjusted its brightness and contrast")
        return (f"For this run, pyCamSet {size} and {tone} before the detector searched it; "
                "the points were then mapped back to the original photo's pixels.")
    try:
        factor = float(params.get("upscale_factor", 1) or 1)
    except (TypeError, ValueError):
        return prefix + "This run's enlargement setting is unreadable, so what the detector searched is unknown."
    if factor > 1:
        return (prefix + f"For this run, pyCamSet enlarged each photo ×{factor:g} before the detector "
                "searched it, so the detected points are in the enlarged image's pixels.")
    return prefix + "For this run, the detector searched each photo at its original size, with no enlargement."


def _frame_to_photo(params: Mapping[str, Any]) -> float:
    """Photo pixels per saved-detection pixel (1 / upscale, unless rescale-and-gamma ran)."""
    if _rescale_and_gamma_state(params) in {"applied", "unknown"}:
        return 1.0
    try:
        return 1.0 / max(float(params.get("upscale_factor", 1) or 1), 1.0)
    except (TypeError, ValueError):
        return 1.0


# ---------------------------------------------------------------------------------------------
# Drawing: shared pieces
# ---------------------------------------------------------------------------------------------

_ACCENT_STATES = {"kept", "accepted", "moved", "candidate", "bit_ok", "after"}
_MUTED_STATES = {"rejected", "dropped", "unchanged", "predicted", "before"}
_BAD_STATES = {"removed", "bit_wrong"}


def _tok(ax) -> dict[str, str]:
    return tokens_for(ax)


def _halo(t: dict[str, str], width: float) -> list:
    """A stage-coloured stroke under a mark drawn on a chart."""
    import matplotlib.patheffects as pe
    return [pe.Stroke(linewidth=width + 2.2, foreground=t["stage"], alpha=0.85), pe.Normal()]


def _photo_halo(colour: str, width: float) -> list:
    """A stroke under a mark drawn on a *photo*: black under light marks, white under dark
    ones, so a ring stays visible on bright and on dark photos in every theme (UX-34)."""
    import matplotlib.patheffects as pe
    under = "#000000" if relative_luminance(colour) > 0.35 else "#FFFFFF"
    return [pe.Stroke(linewidth=width + 1.8, foreground=under, alpha=0.75), pe.Normal()]


def _state_colour(state: Optional[str], t: dict[str, str]) -> str:
    if state in _BAD_STATES:
        return t["bad"]
    if state in _MUTED_STATES:
        return t["rejected"]
    return t["accent"]


def _stretch(image: np.ndarray) -> np.ndarray:
    """Contrast-stretch a small grey crop (2nd-98th percentile) so its pattern is readable."""
    img = np.asarray(image, dtype=float)
    lo, hi = np.percentile(img, [2, 98])
    if hi - lo < 1:
        return np.asarray(image)
    return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _show(ax, image: Optional[np.ndarray], *, nearest: bool = False, cmap: Optional[str] = None,
          stretch: bool = False):
    """An image (gray or RGB, uint8) with no ticks, the theme border as a thin frame."""
    if image is None:
        return None
    image = np.asarray(image)
    if stretch and image.ndim == 2:
        image = _stretch(image)
    shown = ax.imshow(image, cmap=(cmap or "gray") if image.ndim == 2 else None, vmin=0, vmax=255,
                      interpolation="nearest" if nearest else "antialiased", zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    t = _tok(ax)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(t["grid"])
        spine.set_linewidth(0.8)
    return shown


def face_colours(theme: Optional[str], count: int) -> list[str]:
    """``count`` face shades for ``theme``: the shared face palette
    (``education_style.face_colours``, UX2-2 / ED2-8; moved to the shared layer in round 2)."""
    from pyCamSet.gui.education_style import face_colours as _shared_face_colours
    return _shared_face_colours(theme, count)


def _draw_overlays(ax, step, *, zoom: bool = False, labels: bool = False,
                   skip: tuple[str, ...] = (), only: Optional[tuple[str, ...]] = None,
                   fill: Optional[bool] = None, face_palette: Optional[list[str]] = None,
                   label_box=None, px_per_unit: Optional[float] = None, avoid_box=None) -> None:
    """Draw an explanation step's typed overlays on a photo (image pixels = frame x scale)."""
    from matplotlib.collections import LineCollection, PolyCollection
    t = _tok(ax)
    scale = float(step.image_scale or 1.0)
    if fill is None:  # tint the markers only where they are the subject
        fill = getattr(step, "key", "") == "markers"
    for overlay in step.overlays:
        if overlay.name in skip or (only is not None and overlay.name not in only):
            continue
        if not len(overlay):
            continue
        states = list(overlay.states) if overlay.states else ["kept"] * len(overlay)
        if overlay.kind == "polylines":
            lines = [np.asarray(p, dtype=float).reshape(-1, 2) * scale for p in overlay.data]
            lw = 1.4 if zoom else 0.8
            coll = LineCollection(lines, colors=t["accent"], linewidths=lw, alpha=0.95, zorder=3)
            coll.set_path_effects(_photo_halo(t["accent"], lw))
            ax.add_collection(coll)
        elif overlay.kind == "polygons":
            polys = [np.asarray(p, dtype=float).reshape(-1, 2) * scale for p in overlay.data]
            groups: dict[str, list] = {}
            for poly, state in zip(polys, states):
                groups.setdefault(state, []).append(poly)
            for state, group in groups.items():
                colour = _state_colour(state, t)
                muted = state in _MUTED_STATES
                lw = (1.2 if muted else 2.0) if zoom else (0.9 if muted else 1.5)
                if state == "accepted" and fill:
                    ax.add_collection(PolyCollection(group, closed=True, edgecolors="none",
                                                     facecolors=_with_alpha(colour, 0.16),
                                                     zorder=2))
                coll = PolyCollection(
                    group, closed=True, edgecolors=colour, facecolors="none",
                    linewidths=lw, linestyles="--" if state == "dropped" else "-",
                    alpha=(REJECTED_ALPHA + 0.25) if muted else None, zorder=3)
                if not muted:
                    coll.set_path_effects(_photo_halo(colour, lw))
                ax.add_collection(coll)
            if labels and overlay.labels:
                _label_items(ax, [p.mean(axis=0) for p in polys], overlay.labels, t, centre=True,
                             box=label_box, px_per_unit=px_per_unit, avoid_box=avoid_box)
        elif overlay.kind == "points":
            xy = np.asarray(overlay.data, dtype=float).reshape(-1, 2) * scale
            colours = [_state_colour(s, t) for s in states]
            face_ids = None
            if overlay.name == "faces" and overlay.labels:
                face_ids = np.asarray(
                    [int(lab.split(":")[0]) if lab.split(":")[0].lstrip("-").isdigit() else 0
                     for lab in overlay.labels], dtype=int)
            states_arr = np.asarray(states)
            size = 95 if zoom else 26
            crosses = np.isin(states_arr, ["removed", "dropped"])
            plus = states_arr == "predicted"
            rings = ~(crosses | plus)
            if face_ids is not None and rings.any():
                # Faces: the shared face look (shade + shape, never a circle; UX2-2 / ED2-8).
                from pyCamSet.gui.education_style import face_point_kwargs
                muted = np.isin(states_arr, list(_MUTED_STATES))
                for face in np.unique(face_ids[rings]):
                    pick = rings & (face_ids == face)
                    kwargs = face_point_kwargs(int(face), ax, size=size * 0.8, on_photo=True)
                    for sub, alpha in ((pick & ~muted, 1.0), (pick & muted, REJECTED_ALPHA)):
                        if sub.any():
                            ax.scatter(xy[sub, 0], xy[sub, 1], zorder=4, alpha=alpha, **kwargs)
                rings = np.zeros_like(rings)
            if rings.any():
                ring_colours = [c for c, r in zip(colours, rings) if r]
                coll = ax.scatter(xy[rings, 0], xy[rings, 1], s=size, facecolors="none",
                                  edgecolors=ring_colours, linewidths=1.8 if zoom else 1.3,
                                  zorder=4)
                coll.set_path_effects(_photo_halo(ring_colours[0], 1.3))
            if plus.any():
                ax.scatter(xy[plus, 0], xy[plus, 1], s=size * 0.7, marker="+",
                           color=t["rejected"], linewidths=1.2, zorder=4,
                           alpha=REJECTED_ALPHA + 0.3)
            if crosses.any():
                coll = ax.scatter(xy[crosses, 0], xy[crosses, 1], s=size * 1.1, marker="x",
                                  color=t["bad"], linewidths=2.0, zorder=5)
                coll.set_path_effects(_photo_halo(t["bad"], 2.0))
            if labels and overlay.labels:
                _label_items(ax, xy, overlay.labels, t, centre=False, box=label_box,
                             px_per_unit=px_per_unit)
        elif overlay.kind == "segments":
            segs = np.asarray(overlay.data, dtype=float).reshape(-1, 2, 2) * scale
            colours = [_state_colour(s, t) for s in states]
            coll = LineCollection(segs, colors=colours, linewidths=1.6 if zoom else 1.0,
                                  zorder=3, alpha=0.95)
            ax.add_collection(coll)


def _with_alpha(colour: str, alpha: float):
    from matplotlib.colors import to_rgba
    return to_rgba(colour, alpha)


def _shift_title(ax, points: float) -> None:
    """Move a panel's title and subtitle right by ``points`` (room for a badge before it), so
    the badge never sits on the title text (UX-25)."""
    from matplotlib.transforms import ScaledTranslation
    offset = ScaledTranslation(points / 72.0, 0, ax.figure.dpi_scale_trans)
    for text in [ax.title, getattr(ax, "_left_title", None)] + list(ax.texts):
        if text is None:
            continue
        if text is ax.title or text is getattr(ax, "_left_title", None) or \
                (text.get_gid() or "") == "edu:subtitle":
            text.set_transform(text.get_transform() + offset)


def _shown_key(label: Any) -> str:
    """A point/marker key as the learner reads it: ``"5:9"`` (face 5 counted from 0) becomes
    ``"6:9"`` (faces counted from 1, like cameras and photos); other labels unchanged."""
    text = str(label)
    parts = text.split(":")
    if len(parts) == 2 and all(p.strip().lstrip("-").isdigit() for p in parts):
        return f"{int(parts[0]) + 1}:{int(parts[1])}"
    return text


#: Pixels per point at the figure dpi (100) used for label-spacing estimates.
_PX_PER_PT = 100 / 72.0


def _clear_of_border(rect, border, gap: float = 3.0) -> tuple[float, float]:
    """(dx, dy) in px that moves a label box ``rect`` (x0, y0, x1, y1, px, y down) off the
    edges of ``border`` (same units), to the side its centre is on, ``gap`` px clear: an ID
    tag never sits on the dashed close-up square (round 3, P1-4)."""
    x0, y0, x1, y1 = rect
    bx0, by0, bx1, by1 = border
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx = dy = 0.0
    if y1 > by0 and y0 < by1:                       # beside a vertical edge
        for edge in (bx0, bx1):
            if x0 - gap < edge < x1 + gap:
                dx = (edge + gap - x0) if cx >= edge else (edge - gap - x1)
    if x1 + dx > bx0 and x0 + dx < bx1:             # above / below a horizontal edge
        for edge in (by0, by1):
            if y0 - gap < edge < y1 + gap:
                dy = (edge + gap - y0) if cy >= edge else (edge - gap - y1)
    return dx, dy


def _label_items(ax, anchors, labels, t, *, centre: bool, box=None,
                 px_per_unit: Optional[float] = None, fontsize: float = 9.0,
                 colour: Optional[str] = None, avoid_box=None) -> None:
    """Small labels with a stage-coloured box. With ``box`` (x0, y0, x1, y1 in the axes'
    data units) only items inside it are labelled; with ``px_per_unit`` labels that would
    overlap an earlier one are skipped, so a close-up never shows a pile of text. A label
    that would sit on the edge of ``avoid_box`` (the dashed close-up square) is nudged to the
    side its item is on."""
    placed: list[tuple[float, float, float, float]] = []
    height = fontsize * _PX_PER_PT * 1.6
    for (x, y), text in zip(anchors, labels):
        if not text:
            continue
        text = _shown_key(text)
        if box is not None:
            x0, y0, x1, y1 = box
            mx, my = 0.06 * (x1 - x0), 0.06 * (y1 - y0)
            if not (x0 + mx <= x <= x1 - 3 * mx and y0 + my <= y <= y1 - 2 * my):
                continue
        shift = (0.0, 0.0)
        if px_per_unit:
            width = len(str(text)) * fontsize * _PX_PER_PT * 0.7 + 4
            px, py = x * px_per_unit, y * px_per_unit
            rect = ((px - width / 2, py - height / 2, px + width / 2, py + height / 2) if centre
                    else (px + 6 * _PX_PER_PT, py + 6 * _PX_PER_PT,
                          px + 6 * _PX_PER_PT + width, py + 6 * _PX_PER_PT + height))
            if centre and avoid_box is not None:
                # The drawn box: the text plus its 0.18 em pad, and a generous estimate.
                pad = 0.18 * fontsize * _PX_PER_PT + 2
                drawn = (rect[0] - pad, py - fontsize * _PX_PER_PT * 0.75 - pad,
                         rect[2] + pad, py + fontsize * _PX_PER_PT * 0.75 + pad)
                shift = _clear_of_border(drawn, tuple(v * px_per_unit for v in avoid_box))
                rect = (rect[0] + shift[0], rect[1] + shift[1], rect[2] + shift[0],
                        rect[3] + shift[1])
            # 6 px of clear space between tags (the drawn boxes are wider than the estimate)
            if any(rect[0] - 6 < r[2] and r[0] < rect[2] + 6 and rect[1] - 6 < r[3]
                   and r[1] < rect[3] + 6 for r in placed):
                continue
            placed.append(rect)
        offset = (shift[0] / _PX_PER_PT, -shift[1] / _PX_PER_PT)
        ax.annotate(str(text), (x, y), xytext=offset if centre else (6, -6),
                    textcoords="offset points", ha="center" if centre else "left",
                    va="center" if centre else "top", fontsize=fontsize,
                    color=colour or t["text"],
                    annotation_clip=True, clip_on=True, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=t["stage"],
                              edgecolor="none", alpha=0.85))


def _zoom_to(ax, box, scale: float) -> None:
    x0, y0, x1, y1 = box
    ax.set_xlim(x0 * scale - 0.5, x1 * scale - 0.5)
    ax.set_ylim(y1 * scale - 0.5, y0 * scale - 0.5)


def _mark_box(ax, box, scale: float) -> None:
    """The close-up square on the full picture (no letter on the photo, UX-25: the panel's
    subtitle says which panel enlarges it)."""
    from matplotlib.patches import Rectangle
    t = _tok(ax)
    x0, y0, x1, y1 = (v * scale for v in box)
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=t["text"],
                     linewidth=1.4, linestyle=(0, (4, 2)), zorder=7)
    rect.set_path_effects(_photo_halo(t["text"], 1.4))
    ax.add_patch(rect)


def _wrap(text: str, width: int = 26) -> str:
    return "\n".join(textwrap.wrap(str(text), width)) or str(text)


def _values_panel(ax, values, *, limit: Optional[float] = None, xlabel: str, title: str,
                  subtitle: str, letter: str, limit_label: str = "", noun: str = "corners") -> None:
    """A histogram of per-corner values; with fewer than 10 values a dot strip instead (UX-21)."""
    t = _tok(ax)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        empty_panel(ax, "Nothing to show for this photo.")
        panel_title(ax, letter, title, subtitle)
        return
    top = max(float(values.max()), (limit or 0) * 1.1, 1e-6)
    if values.size < 10:
        rng = np.random.default_rng(3)
        ax.scatter(values, rng.uniform(-0.25, 0.25, values.size), s=60, color=t["accent"],
                   zorder=3, edgecolors=t["stage"], linewidths=0.8)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(0, top * 1.08)
    else:
        bins = np.linspace(0, top * 1.02, 18)
        ax.hist(values, bins=bins, color=t["accent"], alpha=0.9, zorder=3,
                edgecolor=t["stage"], linewidth=0.6)
        ax.set_ylabel(noun)
        integer_axis(ax, "y")
        ax.yaxis.grid(True)
    if limit is not None:
        # UX3-11: neutral dashes while every value passes; red only when one is beyond it.
        limit_line(ax, limit, values=values, axis="x", label=limit_label or None)
    ax.set_xlabel(xlabel)
    panel_title(ax, letter, title, subtitle)


def _photo_points(ax, xy: np.ndarray, colours, *, size: float = 30) -> None:
    """Filled dots on a photo, each with a contrasting edge (target / face pictures)."""
    if not len(xy):
        return
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=colours, edgecolors="#FFFFFF", linewidths=0.8,
               zorder=4)


# ---------------------------------------------------------------------------------------------
# Stage 1: the target
# ---------------------------------------------------------------------------------------------

def _face_plates(summary) -> list[dict]:
    """Per face: its design corners (mm), a plate outline around them and its outward normal."""
    every = np.concatenate([f for f in summary.faces if len(f)])
    centroid = every.mean(axis=0)
    plates = []
    for k, pts in enumerate(summary.faces):
        if len(pts) < 3:
            continue
        centre = pts.mean(axis=0)
        _u, _s, vt = np.linalg.svd(pts - centre)
        basis = vt[:2]
        normal = vt[2]
        outward = centre - centroid
        if np.linalg.norm(outward) < 1e-9:
            outward = np.array([0.0, 0.0, 1.0])
        if np.dot(normal, outward) < 0:
            normal = -normal
        flat = (pts - centre) @ basis.T
        try:
            from scipy.spatial import ConvexHull
            ring = flat[ConvexHull(flat).vertices]
        except Exception:  # noqa: BLE001 - degenerate face: a bounding box
            lo, hi = flat.min(axis=0), flat.max(axis=0)
            ring = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        inradius = _inradius(ring)
        if len(summary.faces) > 1 and np.isfinite(summary.size_mm) and inradius > 0:
            # A solid: each plate is the whole face, so neighbouring plates meet at the edges
            # (the corners may sit in the middle of a larger printed face).
            ring = ring * (0.5 * summary.size_mm / inradius)
        else:
            # A flat board reaches about half a square beyond its outer corners.
            pad = 0.5 * (summary.spacing_mm if np.isfinite(summary.spacing_mm) else 0.0)
            radius = np.linalg.norm(ring, axis=1, keepdims=True)
            ring = ring * (1.0 + pad * np.sqrt(2) / np.maximum(radius, 1e-9))
        plates.append(dict(face=k, points=pts, centre=centre, normal=normal,
                           outline=centre + ring @ basis))
    return plates


def _inradius(ring: np.ndarray) -> float:
    """Distance from the origin to the nearest edge of a convex polygon around it."""
    best = np.inf
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        edge = b - a
        length = float(np.hypot(*edge))
        if length > 1e-12:
            best = min(best, abs(float(edge[0] * a[1] - edge[1] * a[0])) / length)
    return float(best) if np.isfinite(best) else 0.0


def _view_vector(elev: float, azim: float) -> np.ndarray:
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _target_view(plates: list[dict], seen: set[int]) -> tuple[float, float]:
    """A viewing direction that shows the faces in the photo next to the 3D target (and, on a
    cube, one more face for depth): the sum of their outward normals, tipped toward a third."""
    if not plates:
        return 24.0, -56.0
    normals = {p["face"]: p["normal"] / max(np.linalg.norm(p["normal"]), 1e-9) for p in plates}
    chosen = [normals[f] for f in sorted(seen) if f in normals] or [normals[plates[0]["face"]]]
    eye = np.sum(chosen, axis=0)
    eye = eye / max(np.linalg.norm(eye), 1e-9)
    others = [n for f, n in normals.items() if f not in seen and abs(np.dot(n, eye)) < 0.5]
    if others:
        extra = max(others, key=lambda n: n[2])
        eye = eye + (0.6 if len(chosen) == 1 else 0.5) * extra
        if len(chosen) == 1:
            rest = [n for n in others if abs(np.dot(n, extra)) < 0.5]
            if rest:
                eye = eye + 0.35 * rest[0]
    else:
        eye = eye + np.array([0.25, -0.2, 0.35])
    eye = eye / max(np.linalg.norm(eye), 1e-9)
    elev = float(np.degrees(np.arcsin(np.clip(eye[2], -1, 1))))
    azim = float(np.degrees(np.arctan2(eye[1], eye[0])))
    return elev, azim


#: Cells per face side the 3-D target view draws a print with: at least enough for a coarse
#: board, at most what matplotlib's 3-D surfaces draw quickly.
_TEXTURE_CELLS = (72, 144)


def _texture_cells(summary, corners) -> int:
    """Cells per face side for the 3-D target view: about eight per corner spacing across the
    printed face, so a fine print (a dense ChArUco, say) shows its pattern
    instead of a grey blur."""
    low, high = _TEXTURE_CELLS
    spacing = float(getattr(summary, "spacing_mm", float("nan")))
    side = float(np.linalg.norm(np.asarray(corners[1], float) - np.asarray(corners[0], float)))
    if not (np.isfinite(spacing) and spacing > 0 and np.isfinite(side)):
        return low
    return int(np.clip(round(8 * side / spacing), low, high))


def _texture_grid(corners: np.ndarray, texture: np.ndarray, n: int, dark: str, light: str):
    """X, Y, Z (n+1 x n+1) and face colours (n x n x 4) of a printed face texture laid on a
    plate (``corners``: the texture's top-left, top-right, bottom-right, bottom-left in mm),
    the print drawn faintly between ``dark`` and ``light`` (ED2-25)."""
    import cv2
    from matplotlib.colors import to_rgb
    u = np.linspace(0.0, 1.0, n + 1)
    uu, vv = np.meshgrid(u, u)          # rows follow the texture's rows (v down)
    tl, tr, _br, bl = (np.asarray(c, dtype=float) for c in corners)
    grid = (tl[None, None] + uu[..., None] * (tr - tl)[None, None]
            + vv[..., None] * (bl - tl)[None, None])
    tex = np.asarray(texture)
    if tex.ndim == 3:
        tex = tex.mean(axis=2)
    small = cv2.resize(tex.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA) / 255.0
    d, lt = np.asarray(to_rgb(dark)), np.asarray(to_rgb(light))
    small = np.clip(small, 0.0, 1.0)
    rgb = np.clip(d[None, None] + small[..., None] * (lt - d)[None, None], 0.0, 1.0)
    colours = np.concatenate([rgb, np.ones((n, n, 1))], axis=2)
    return grid[..., 0], grid[..., 1], grid[..., 2], colours


def _mix(a: str, b: str, share: float) -> str:
    """``share`` of colour ``a`` over ``b`` (hex)."""
    from matplotlib.colors import to_hex, to_rgb
    ca, cb = np.asarray(to_rgb(a)), np.asarray(to_rgb(b))
    return to_hex(ca * share + cb * (1.0 - share))


def _flat_board_view(plates: list[dict]) -> Optional[tuple[float, float]]:
    """A flat board lying in z: seen nearly square-on (tilted 28 degrees) with an edge across
    the picture (UX2-17), so it reads as a board, not a diamond."""
    if len(plates) != 1:
        return None
    plate = plates[0]
    normal = plate["normal"] / max(np.linalg.norm(plate["normal"]), 1e-9)
    if abs(normal[2]) < 0.95:
        return None
    ring = plate["outline"] - plate["centre"]
    edge = ring[1] - ring[0] if len(ring) >= 2 else np.array([1.0, 0.0, 0.0])
    angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
    angle = (angle + 45.0) % 90.0 - 45.0  # the edge closest to horizontal
    elev = 62.0 if normal[2] > 0 else -62.0
    return elev, angle - 90.0


def _draw_target_3d(host, summary, colours: list[str], highlight: set[int],
                    textures: bool = True) -> None:
    """The target as plates (one per face) with its printed pattern (when the target says
    where it sits) and its design corners; faces facing away are hidden. The faces of the
    photo beside it face the viewer, outlined in the accent with a ringed square tag; the
    others are fainter (ED2-25)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from pyCamSet.gui.education_style import draw_face_tags, face_point_kwargs
    t = _tok(host)
    host.set_axis_off()
    ax = host.inset_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    ax.set_facecolor((0, 0, 0, 0))
    try:
        ax.computed_zorder = False  # drawn far-to-near by hand: plates, corners, tags
    except AttributeError:
        pass
    plates = _face_plates(summary)
    elev, azim = _flat_board_view(plates) or _target_view(plates, highlight)
    if elev < 0 and len(plates) > 1:
        # Seen from below, an orthographic cube reads as a hollow corner (Necker); turning
        # the picture upside down (a rotation, not a mirror) shows the same faces from "above".
        try:
            ax.view_init(elev=elev, azim=azim, roll=180)
        except TypeError:  # matplotlib < 3.6: no roll
            ax.view_init(elev=elev, azim=azim)
    else:
        ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    eye = _view_vector(elev, azim)
    order = sorted(plates, key=lambda p: float(np.dot(p["centre"], eye)))  # far faces first
    placed = summary.texture_corners if textures else ()
    tags_xyz, tags_faces = [], []
    z = 1.0
    for plate in order:
        k = plate["face"]
        shade = colours[k % len(colours)]
        facing = float(np.dot(plate["normal"], eye)) > 0.05 or len(plates) == 1
        if not facing:  # the far side is hidden, as on the real target
            continue
        seen = k in highlight or not highlight
        edge = t["accent"] if (k in highlight and len(plates) > 1) else shade
        corners = placed[k] if k < len(placed) else None
        texture = summary.textures[k] if k < len(summary.textures) else None
        width = 2.6 if (k in highlight and len(plates) > 1) else 1.2
        if corners is not None and texture is not None:
            # The print keeps its polarity on every theme (black squares darker than white).
            if relative_luminance(t["stage"]) > 0.3:
                dark, light = _mix(t["text"], t["stage"], 0.55 if seen else 0.26), t["stage"]
            else:
                dark = _mix("#000000", t["stage"], 0.55)
                light = _mix("#FFFFFF", t["stage"], 0.62 if seen else 0.32)
            X, Y, Z, fc = _texture_grid(corners, texture, _texture_cells(summary, corners),
                                        dark, light)
            ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1, linewidth=0,
                            antialiased=False, shade=False, zorder=z)
            poly = Poly3DCollection([plate["outline"]], facecolors=[(0, 0, 0, 0)],
                                    edgecolors=[edge], linewidths=width, zorder=z + 0.1)
        else:
            poly = Poly3DCollection([plate["outline"]],
                                    facecolors=[_with_alpha(shade, 0.30 if seen else 0.12)],
                                    edgecolors=[edge], linewidths=width, zorder=z)
        ax.add_collection3d(poly)
        pts = plate["points"]
        kwargs = face_point_kwargs(k, host, size=10 if seen else 6, on_photo=True)
        if not seen:
            kwargs["alpha"] = 0.55
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], depthshade=False, zorder=z + 0.2, **kwargs)
        tags_xyz.append(plate["centre"] + plate["normal"] * 0.16 * max(summary.size_mm, 1e-6))
        tags_faces.append(k)
        z += 1.0
    if tags_faces and len(plates) > 1:
        texts = draw_face_tags(ax, np.asarray(tags_xyz), tags_faces, size=10,
                               selected=sorted(highlight))
        for text in texts:
            text.set_zorder(z + 5)
    every = np.concatenate([p["outline"] for p in plates] + [p["points"] for p in plates])
    centre = (every.max(axis=0) + every.min(axis=0)) / 2
    half = float(np.ptp(every, axis=0).max()) * 0.56
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.12 if len(plates) > 1 else 1.55)
    except TypeError:  # older matplotlib: no zoom
        ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def _face_matrix(ax, summary, selected: int, photo_faces: set[int], letter: str) -> None:
    """Rows = cameras (badges), columns = faces (square F-tags): dot area = photos in which
    that camera saw that face. One neutral colour; the selected camera's faces in this photo
    in the accent (UX2-2: colour + circle = camera, grey + square tag = face)."""
    from pyCamSet.gui.education_style import badge_ticks_y, face_ticks
    t = _tok(ax)
    seen = np.asarray(summary.seen, dtype=float)
    n_cams, n_faces = seen.shape
    top = max(float(seen.max()), 1.0)
    for cam in range(n_cams):
        for face in range(n_faces):
            if seen[cam, face] <= 0:
                ax.scatter([face], [cam], s=10, color=t["grid"], zorder=2)
                continue
            area = 40 + 520 * seen[cam, face] / top
            hot = cam == selected and face in photo_faces
            ax.scatter([face], [cam], s=area, color=t["accent"] if hot else t["muted"],
                       alpha=1.0 if hot else 0.7, zorder=3, edgecolors=t["stage"],
                       linewidths=0.8)
    ax.set_xlim(-0.6, n_faces - 0.4)
    ax.set_ylim(n_cams - 0.5, -0.6)
    face_ticks(ax, range(n_faces), list(range(n_faces)), size=9.5, selected=sorted(photo_faces))
    badge_ticks_y(ax, range(n_cams), list(range(n_cams)), highlight=[selected], size=9.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    panel_title(ax, letter, "Which camera sees which face",
                "cameras × faces · dot size = photos")


def _board_bars(ax, summary, selected: int, letter: str) -> None:
    """A flat board: per camera, the photos in which it saw the board (UX2-17)."""
    seen = np.asarray(summary.seen, dtype=float)
    values = {cam: float(seen[cam, 0]) for cam in range(seen.shape[0])} if seen.size else {}
    if not values:
        empty_panel(ax, "No saved detections to count.")
        panel_title(ax, letter, "Photos that saw the board", "per camera")
        return
    bars_by_camera(ax, values, None, value_format="{:.0f}", highlight=selected,
                   ylabel="photos", xlabel="Camera")
    integer_axis(ax, "y")
    panel_title(ax, letter, "Photos that saw the board", "per camera, all photos of this run")


def _faces_text(faces: list[int]) -> str:
    """``"F2"`` / ``"F2 and F6"`` / ``"F1, F2 and F6"`` (square-tag names, ED2-8)."""
    names = [face_label(f) for f in faces]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


#: marker shape -> plain word for the target view's face legend (ED5-5); see FACE_MARKERS
_SHAPE_WORDS = {"s": "squares", "^": "triangles", "D": "diamonds", "v": "down-tri",
                "p": "pentagons", "h": "hexagons"}


def draw_target(fig, data: dict, level: int) -> None:
    """What a calibration target is, with this run's: A the target in 3D (faces, printed
    pattern, design corners), B one photo with its corners marked per face; L2 + C who saw
    which face (a flat board: photos per camera); L3 + D one face as printed."""
    from pyCamSet.gui.education_style import face_colour, face_point_kwargs
    summary = data.get("summary")
    if summary is None:
        empty_panel(fig.add_subplot(), "This run's target could not be rebuilt, so its design "
                                       "cannot be drawn.")
        return
    theme = getattr(fig, "_edu_theme", None)
    colours = face_colours(theme, len(summary.faces))
    faces_in_photo = sorted({int(f) for f in data.get("point_faces", [])})
    n_faces = len(summary.faces)
    texture = None
    if level >= 3:
        pick = faces_in_photo[0] if faces_in_photo else 0
        texture = (pick, summary.textures[pick]) if pick < len(summary.textures) \
            and summary.textures[pick] is not None else None
    columns = ([1.15, 1.0] if n_faces > 1 else [0.95, 1.25]) + ([1.0] if level >= 2 else [])         + ([0.8] if texture else [])
    grid = fig.add_gridspec(1, len(columns), width_ratios=columns)
    host = fig.add_subplot(grid[0, 0])
    corners = sum(len(f) for f in summary.faces)
    size = format_number(summary.size_mm, 0 if summary.size_mm >= 10 else 1)
    if n_faces > 1:
        pattern = summary.pattern_mm
        decimals = 0 if pattern >= 10 or abs(pattern - round(pattern)) < 0.05 else 1
        across = (f"{format_number(pattern, decimals)} mm pattern on a {size} mm cube"
                  if np.isfinite(pattern) and pattern < summary.size_mm - 1e-6
                  else f"{size} mm across")
        sub = f"{n_faces} × {len(summary.faces[0])} corners · {across}"
    else:
        sub = f"{corners} corners · {size} mm across"
    panel_title(host, "A", "Your target", sub)
    _draw_target_3d(host, summary, colours, set(faces_in_photo))
    # B: one photo, corners marked per face (shade + shape)
    ax = fig.add_subplot(grid[0, 1])
    cam = data["camera"]
    photo = data.get("photo")
    pts = np.asarray(data.get("points", np.zeros((0, 2))), dtype=float).reshape(-1, 2)
    faces = np.asarray(data.get("point_faces", np.zeros(0)), dtype=int)
    if photo is not None:
        _show(ax, photo)
        if len(pts):
            for face in sorted(set(faces.tolist())):
                pick = faces == face
                ax.scatter(pts[pick, 0], pts[pick, 1], zorder=4,
                           **face_point_kwargs(face, ax, size=48, on_photo=True))
        frame_axes(ax, cam, width=3.0)
    else:
        empty_panel(ax, _wrap(data.get("photo_reason") or "Photo unavailable.", 34))
    if n_faces > 1 and faces_in_photo:
        # ED5-5: each shape is one face (as in A); every marker drawn is a found corner
        shapes = ", ".join(f"{face_label(f)} {_SHAPE_WORDS.get(face_marker(f), 'marks')}"
                           for f in faces_in_photo)
        # round 8: "found" dropped so the subtitle fits the panel at 1280 px (was elided)
        sub = f"{data['photo_label']} · {len(pts)} corners: {shapes}"
        # ED5-5 (round 10): the one-line key the shapes and the accent outline in A lacked
        # L1 only: at L2/L3 the narrower B puts the line into C's badges (text_collisions)
        if level == 1:
            ax.set_xlabel("Shape = face, as in A · outline in A = faces shown here",
                          fontsize=TICK_SIZE, color=_tok(ax)["muted"], labelpad=4)
    else:
        sub = f"{data['photo_label']} · {len(pts)} corners found"
    panel_title(ax, "B", f"Seen by camera {cam + 1}", sub)
    if level >= 2:
        axc = fig.add_subplot(grid[0, 2])
        if n_faces > 1:
            _face_matrix(axc, summary, cam, set(faces_in_photo), "C")
        else:
            _board_bars(axc, summary, cam, "C")
    if texture is not None:
        axd = fig.add_subplot(grid[0, len(columns) - 1])
        face, image = texture
        _show(axd, image)
        for spine in axd.spines.values():
            spine.set_color(face_colour(face, axd))
            spine.set_linewidth(3.0)
        panel_title(axd, "D", f"{face_label(face, short=False)} as printed" if n_faces > 1
                    else "The board as printed", "each marker has its own ID")


# ---------------------------------------------------------------------------------------------
# Stage 2: photos
# ---------------------------------------------------------------------------------------------

def _grid_shape(n: int, aspect: float, width: float, height: float,
                row_room: Optional[float] = None) -> tuple[int, int]:
    """Rows and columns that make n photo tiles (height/width = aspect) as large as possible.
    With ``row_room`` (inches of title space above each row) that space is taken off exactly;
    without it each row keeps 18 % of its height for its title."""
    best, best_side = (1, max(n, 1)), 0.0
    for cols in range(1, max(n, 1) + 1):
        rows = int(np.ceil(n / cols))
        if row_room is None:
            tile_h = (height / rows) * 0.82
        else:
            tile_h = max(height / rows - row_room, 0.05)
        tile_w = min(width / cols, tile_h / max(aspect, 1e-3))
        if tile_w > best_side:
            best, best_side = (rows, cols), tile_w
    return best


def draw_photos(fig, tiles: list[dict], selected: int, level: int,
                axes_out: Optional[dict] = None) -> None:
    """Every camera's photo for one photo number: badge, name, point count; selected ringed."""
    n = max(len(tiles), 1)
    aspects = [t["image"].shape[0] / t["image"].shape[1] for t in tiles if t.get("image") is not None]
    aspect = float(np.median(aspects)) if aspects else 0.75
    width, height = fig.get_size_inches()
    # Fixed margins with room above every row for the badge, name and count (about 0.6 in):
    # constrained layout does not count the badge line, so on a short stage (a long L3
    # caption) the names were pushed off the top of the figure.
    room = 0.6
    rows, cols = _grid_shape(n, aspect, width, height - 0.08, row_room=room)
    _use_fixed_layout(fig)
    cell_h = max((height - 0.08 - room * rows) / rows, 0.3)
    grid = fig.add_gridspec(rows, cols, left=0.008, right=0.992, bottom=0.08 / height,
                            top=1.0 - room / height, wspace=0.04,
                            hspace=room / cell_h if rows > 1 else 0.0)
    for k, tile in enumerate(tiles):
        ax = fig.add_subplot(grid[k // cols, k % cols])
        t = _tok(ax)
        cam = tile["index"]
        image = tile.get("image")
        count = tile.get("count", 0)
        if image is not None:
            _show(ax, image)
            if level >= 2 and len(tile.get("points", ())):
                pts = np.asarray(tile["points"], dtype=float) * tile.get("scale", 1.0)
                coll = ax.scatter(pts[:, 0], pts[:, 1], s=14, facecolors="none",
                                  edgecolors=t["accent"], linewidths=1.3, zorder=3)
                coll.set_path_effects(_photo_halo(t["accent"], 1.3))
            frame_axes(ax, cam, width=4.5 if k == selected else 1.6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(t["stage"])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(camera_colour(cam, ax))
                spine.set_linestyle((0, (3, 3)))
                spine.set_linewidth(1.2)
            ax.set_aspect(aspect)
            ax.set_xlim(0, 1)
            ax.set_ylim(aspect, 0)
            ax.text(0.5, aspect / 2, _wrap(tile.get("reason") or "Photo unavailable.", 22),
                    ha="center", va="center", fontsize=9, color=t["muted"])
        name = tile["name"]
        short = name if len(name) <= 22 else name[:20] + "…"
        sub = f"{count} point{'s' if count != 1 else ''}" if count else "not found"
        if level >= 3 and tile.get("file"):
            file = tile["file"]
            sub += " · " + (file if len(file) <= 14 else "…" + file[-13:])
        panel_title(ax, None, short, sub)
        _shift_title(ax, 21)
        if not count:
            for text in ax.texts:
                if (text.get_gid() or "") == "edu:subtitle":
                    text.set_color(t["bad"])
        draw_camera_badges(ax, [(0.0, 1.0)], [cam], transform=_title_line_transform(ax),
                           size=10, highlight=[cam] if k == selected else ())
        if axes_out is not None:
            axes_out[ax] = k


def _title_line_transform(ax, two_lines: bool = True):
    """Axes-fraction x, with y on the panel title's first line (above any subtitle)."""
    from matplotlib.transforms import ScaledTranslation
    return ax.transAxes + ScaledTranslation(9 / 72, (27.5 if two_lines else 13.0) / 72,
                                            ax.figure.dpi_scale_trans)


# ---------------------------------------------------------------------------------------------
# Stage 3: what the detector sees
# ---------------------------------------------------------------------------------------------

def _traced_black(image: np.ndarray) -> np.ndarray:
    """A detector binary image shown with black = the pixels the outline finder traces.

    The detectors store "darker than the neighbourhood" as 255 (white); inverting it for display
    makes the picture read the way the captions say it ("darker turns black").
    """
    return 255 - np.asarray(image, dtype=np.uint8)


def _as_seen(step) -> Optional[np.ndarray]:
    """A step's picture for display (black-and-white images inverted, see _traced_black)."""
    if step.image is None:
        return None
    return _traced_black(step.image) if step.key == "threshold" else step.image


def _is_grey_photo(explanation) -> bool:
    first = explanation.step("input")
    if first is None:
        return False
    if first.extra.get("file_channels") == 1 or first.extra.get("grey_content") is True:
        return True
    return int(first.counts.get("channels", 3) or 3) == 1


def photo_file_text(explanation) -> str:
    """``"16-bit grey file · shown as 8-bit"`` from the photo's own file (SCI-12)."""
    first = explanation.step("input")
    if first is None:
        return ""
    depth = first.counts.get("bit depth")
    depth_text = f"{depth}-bit" if isinstance(depth, (int, np.integer)) else str(depth)
    kind = "grey" if _is_grey_photo(explanation) else "colour"
    text = f"{depth_text} {kind} file"
    if isinstance(depth, (int, np.integer)) and depth > 8:
        text += ", read as 8-bit"
    return text


def _strip_subtitle(step, explanation) -> str:
    c = step.counts
    if step.key == "input":
        return f"{c.get('width px')} × {c.get('height px')} px · {photo_file_text(explanation)}"
    if step.key == "grayscale":
        return f"brightness only · mean {vis._fmt(c.get('mean brightness'))}"
    if step.key == "rescale_gamma":
        return f"{vis._fmt(c.get('scale'))}× size · gamma {vis._fmt(c.get('gamma'))}"
    if step.key == "detector_input":
        return f"enlarged {c.get('upscaled')} · {c.get('width px')} × {c.get('height px')} px"
    if step.key == "puzzle_smooth":
        return f"blurred, sigma {vis._fmt(c.get('blur sigma px'))} px"
    return step.caption


_STRIP_TITLES = {
    "input": "Your photo", "grayscale": "Grey", "rescale_gamma": "Shrink + brighten",
    "detector_input": "Enlarged", "threshold": "Black and white", "puzzle_smooth": "Smoothed",
}


def _arrow_axes(ax, word: str = "") -> None:
    """A narrow column holding one 'then' arrow (or a word such as 'vs') between pictures."""
    t = _tok(ax)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if word:
        ax.text(0.5, 0.5, word, ha="center", va="center", fontsize=12, color=t["muted"],
                fontweight="bold")
        return
    ax.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5),
                arrowprops=dict(arrowstyle="-|>,head_length=0.7,head_width=0.4",
                                color=t["muted"], linewidth=2.0, shrinkA=0, shrinkB=0))


def _window_square(ax, centre, side: float, colour: str, *, pixel: bool = True) -> None:
    """The neighbourhood square (dashed) and its centre pixel (solid) on a picture."""
    from matplotlib.patches import Rectangle
    x, y = centre
    square = Rectangle((x - side / 2, y - side / 2), side, side, fill=False, edgecolor=colour,
                       linewidth=1.8, linestyle=(0, (4, 2)), zorder=6)
    square.set_path_effects(_photo_halo(colour, 1.8))
    ax.add_patch(square)
    if pixel:
        cell = Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor=colour, linewidth=2.0,
                         zorder=7)
        cell.set_path_effects(_photo_halo(colour, 2.0))
        ax.add_patch(cell)


def _verdict(black: bool) -> str:
    return "black" if black else "white"


def draw_preprocessing(fig, explanation, level: int) -> None:
    """Stage 3: what the marker finder works on. ArUco-family targets: the photo, a single
    global cut-off (for comparison) and the detector's local threshold, with one pixel's
    neighbourhood drawn (L1), that pixel's test enlarged (L2), the two rules as histograms and
    the other black-and-white images the detector also tries (L3). PuzzleBoard: photo ->
    shrink + brighten -> smoothed."""
    demo = vis.threshold_demo(explanation)
    if demo is None:
        _draw_prep_strip(fig, explanation, level)
        return
    t = stage_tokens(getattr(fig, "_edu_theme", None))
    first = explanation.step("input")
    grey_step = explanation.step("grayscale")
    threshold = explanation.step("threshold")
    grey_photo = _is_grey_photo(explanation)
    columns: list[tuple[str, Any]] = [("photo", first)]
    if not grey_photo and grey_step is not None:
        columns.append(("grey", grey_step))
    columns += [("global", None), ("local", threshold)]
    ratios: list[float] = []
    for k, (kind, _s) in enumerate(columns):
        ratios.append(1.0)
        if k < len(columns) - 1:
            ratios.append(0.12)
    rows = 1 if level == 1 else 2
    # Fixed margins (constrained layout can collapse this mix of pictures, arrow columns and
    # charts); the shared title alignment still runs.
    _use_fixed_layout(fig)
    heights = [1.0] if rows == 1 else ([1.45, 1.0] if level == 2 else [2.9, 1.0])
    grid = fig.add_gridspec(rows, len(ratios), width_ratios=ratios, height_ratios=heights,
                            left=0.012, right=0.99, top=0.9 if rows == 1 else 0.91,
                            bottom=0.03 if level < 3 else 0.1, wspace=0.04,
                            hspace=0.3 if level == 2 else 0.4)
    thr_scale = float(threshold.image_scale or 1.0)
    probe_frame = np.asarray(demo.probe, dtype=float) / thr_scale
    letters = iter("ABCDEFGH")
    tops = {}
    for k, (kind, step) in enumerate(columns):
        if k:
            word = "vs" if kind == "local" else ""
            _arrow_axes(fig.add_subplot(grid[0, 2 * k - 1]), word)
        ax = fig.add_subplot(grid[0, 2 * k])
        # Pictures lean towards the arrow / "vs" between them, so a height-limited row leaves
        # its spare width at the outer edges, not in wide gutters (UX3-16).
        ax.set_anchor("E" if k == 0 else ("W" if k == len(columns) - 1 else "C"))
        letter = next(letters)
        if kind in ("photo", "grey"):
            _show(ax, step.image)
            scale = float(step.image_scale or 1.0)
            side = demo.window * scale / thr_scale
            _window_square(ax, probe_frame * scale, side, t["accent"])
            title = "Your photo" if kind == "photo" else "Grey"
            subtitle = _strip_subtitle(step, explanation)
            if kind == "photo" and grey_photo:
                subtitle = (f"{step.counts.get('width px')} × {step.counts.get('height px')} px"
                            f" · {photo_file_text(explanation)}")
            panel_title(ax, letter, title, subtitle)
        elif kind == "global":
            shown = np.where(demo.global_black, 0, 255).astype(np.uint8)
            _show(ax, shown)
            for spine in ax.spines.values():
                spine.set_color(t["muted"])
                spine.set_linestyle((0, (3, 3)))
                spine.set_linewidth(1.4)
            panel_title(ax, letter, "One cut-off for all",
                        f"grey ≤ {demo.cutoff:.0f} → black · for comparison only")
        else:
            _show(ax, _as_seen(step))
            for spine in ax.spines.values():
                spine.set_color(t["accent"])
                spine.set_linewidth(2.4)
            rule = "≥" if threshold.extra.get("rule") == "opencv" else ">"
            panel_title(ax, letter, "What the detector uses",
                        f"{demo.window_prepared} px square: {rule} {demo.constant:g} darker → "
                        f"black{_thinned_note(threshold) if level >= 3 else ''}")
            _window_square(ax, demo.probe, demo.window, t["accent"], pixel=False)
        tops[kind] = (ax, letter)
    if level == 2:
        _probe_closeups(fig, grid, columns, demo, explanation, letters, t)
    elif level >= 3:
        _threshold_evidence(fig, grid[1, :], demo, threshold, explanation, letters, t)


def _thinned_note(threshold) -> str:
    """``", then thinned once"`` when the shown black-and-white image is the eroded one
    (aruco2's black-marker pass, SCI2-14)."""
    rep = int(threshold.extra.get("representative_index", 0) or 0)
    images = list(getattr(threshold, "images", ()) or ())
    name = str(images[rep][0]) if rep < len(images) else ""
    if name.startswith("black markers") and "eroded" in name:
        passes = "".join(ch for ch in name.split("eroded")[-1] if ch.isdigit()) or "1"
        return ", thinned" if passes == "1" else f", thinned {passes}×"
    return ""


def _other_is_inverted(threshold) -> bool:
    """True when the one other black-and-white image shown (F) is the white-marker pass."""
    rep = int(threshold.extra.get("representative_index", 0) or 0)
    others = [str(name) for k, (name, _i) in enumerate(getattr(threshold, "images", ()) or ())
              if k != rep]
    return bool(others) and others[0].startswith("white markers")


def _inverted_copy_sentence(threshold, demo) -> str:
    """L3 clause for the inverted copy (F). The white-marker pass has its own square when the
    detector uses one (aruco2: 15 px against C's 25 px), so the clause says so (round 9)."""
    windows = list(threshold.extra.get("window_sizes") or [])
    text = "; F is the inverted copy in which light-on-dark markers are found"
    if threshold.extra.get("rule") != "opencv" and len(windows) >= 2             and int(windows[1]) != int(demo.window_prepared):
        return text + (f", with its own {int(windows[1])} px square (C uses "
                       f"{int(demo.window_prepared)} px).")
    return text + "."


def _layer_title(name: str, threshold) -> tuple[str, str]:
    """Title and subtitle of one of the other black-and-white images, with its own window
    when the detector uses a different one for it (aruco2's white pass, SCI2-14)."""
    title, subtitle = vis.threshold_layer_name(name)
    windows = list(threshold.extra.get("window_sizes") or [])
    if str(name).startswith("white markers") and len(windows) >= 2 \
            and threshold.extra.get("rule") != "opencv":
        subtitle = f"inverted copy · {windows[1]} px square"
    elif str(name).startswith("white markers"):
        subtitle = "inverted copy"
    return title, subtitle


class _ShiftedSpec:
    """A gridspec seen without its first ``skip`` columns (``spec[0, k]`` -> ``k + skip``)."""

    def __init__(self, spec, skip: int) -> None:
        self._spec, self._skip = spec, int(skip)

    def __getitem__(self, key):
        row, col = key
        return self._spec[row, col + self._skip]


def _probe_box(demo, factor: float = 2.6) -> tuple[int, int, int, int]:
    h, w = demo.grey.shape
    half = int(np.ceil(demo.window * factor / 2))
    x, y = demo.probe
    x0, y0 = max(0, x - half), max(0, y - half)
    x1, y1 = min(w, x + half + 1), min(h, y + half + 1)
    return x0, y0, x1, y1


def _probe_closeups(fig, grid, columns, demo, explanation, letters, t) -> None:
    """L2: the probe pixel's neighbourhood enlarged under each picture, with its test."""
    x0, y0, x1, y1 = _probe_box(demo)
    gap = demo.mean_value - demo.centre_value
    rule = "≥" if explanation.step("threshold").extra.get("rule") == "opencv" else ">"
    for k, (kind, _step) in enumerate(columns):
        ax = fig.add_subplot(grid[1, 2 * k])
        if kind in ("photo", "grey"):
            crop = demo.grey[y0:y1, x0:x1]
            sub = (f"pixel {demo.centre_value} · its square's mean {demo.mean_value:.0f}")
            title = "One pixel, enlarged"
        elif kind == "global":
            crop = np.where(demo.global_black[y0:y1, x0:x1], 0, 255).astype(np.uint8)
            sign = "≤" if demo.global_says_black else ">"
            right = demo.global_says_black == demo.local_black
            sub = (f"{demo.centre_value} {sign} {demo.cutoff:.0f} → {_verdict(demo.global_says_black)}"
                   + ("" if right else " (wrong here)"))
            title = "One cut-off"
        else:
            crop = np.where(demo.local[y0:y1, x0:x1], 0, 255).astype(np.uint8)
            if demo.local_black:
                sub = f"darker than its mean by {gap:.0f} → black"
            elif gap <= 0:
                sub = f"brighter than its mean by {-gap:.0f} → white"
            else:
                sub = f"darker by only {gap:.0f} (not {rule} {demo.constant:g}) → white"
            title = "Its own neighbourhood"
        _show(ax, crop, nearest=True)
        centre = (demo.probe[0] - x0, demo.probe[1] - y0)
        _window_square(ax, centre, demo.window, t["accent"])
        panel_title(ax, next(letters), title, sub)


def _threshold_evidence(fig, spec, demo, threshold, explanation, letters, t) -> None:
    """L3: the two rules as histograms, then the other black-and-white images it tries."""
    others = [(name, image) for k, (name, image) in enumerate(threshold.images)
              if k != int(threshold.extra.get("representative_index", 0) or 0)][:2]
    # A narrow first column keeps room for the y label under the fixed margins.
    ratios = [0.05, 1.0, 1.0] + [0.75] * len(others)
    sub = spec.subgridspec(1, len(ratios), width_ratios=ratios, wspace=0.32)
    sub = _ShiftedSpec(sub, 1)
    ax = fig.add_subplot(sub[0, 0])
    hist = np.bincount(demo.grey.reshape(-1), minlength=256).astype(float)
    share = hist / max(hist.sum(), 1) * 100
    x = np.arange(256)
    ax.fill_between(x, share, step="mid", color=t["muted"], alpha=0.55, zorder=2)
    # The y axis fits the bulk of the levels; a spike (e.g. saturated white at 255) is marked
    # with its value instead of flattening everything else (UX2-15).
    body = float(np.percentile(share, 99))
    top = max(body * 1.25, 1e-3)
    spikes = np.flatnonzero(share > top)
    ax.set_ylim(0, top)
    if len(spikes):
        # One mark for the highest level off the scale (neighbouring levels share it).
        peak = int(spikes[np.argmax(share[spikes])])
        ax.annotate(f"▲ {share[peak]:.0f} % at {peak}", (peak, top),
                    xytext=(-3 if peak > 170 else (5 if peak < 85 else 0), -2),
                    textcoords="offset points",
                    ha="right" if peak > 170 else ("left" if peak < 85 else "center"),
                    va="top", fontsize=TICK_SIZE, color=t["text"], annotation_clip=False)
    ax.axvline(demo.cutoff, color=t["text"], linestyle=(0, (5, 3)), linewidth=1.3, zorder=3,
               alpha=0.8)
    ax.set_xlim(0, 255)
    ax.set_xlabel("grey level (0 black … 255 white)")
    ax.set_ylabel("pixels (%)")
    ax.yaxis.grid(True)
    panel_title(ax, next(letters), "One cut-off",
                f"left of {demo.cutoff:.0f} turns black, wherever it is")
    ax = fig.add_subplot(sub[0, 1])
    diff = demo.difference.reshape(-1)
    counts, edges = np.histogram(diff, bins=60, range=(-30, 30))
    centres = (edges[:-1] + edges[1:]) / 2
    black = centres > demo.constant
    share = counts / max(counts.sum(), 1) * 100
    ax.bar(centres[~black], share[~black], width=1.0, color=t["muted"], alpha=0.55, zorder=2)
    ax.bar(centres[black], share[black], width=1.0, color=t["accent"], zorder=2)
    ax.axvline(demo.constant, color=t["accent"], linestyle=(0, (5, 3)), linewidth=1.4, zorder=3)
    ax.set_xlim(-30, 30)
    ax.set_xlabel("square's mean − pixel (grey levels)")
    ax.set_ylabel("pixels (%)")
    ax.yaxis.grid(True)
    panel_title(ax, next(letters), "Each pixel's own test",
                f"right of {demo.constant:g} turns black (accent)")
    box = vis.focus_box(explanation, vis.frame_size(explanation))
    h, w = threshold.image.shape[:2]
    frame_w = w / float(threshold.image_scale or 1.0)
    for k, (name, image) in enumerate(others):
        axo = fig.add_subplot(sub[0, 2 + k])
        _show(axo, _traced_black(image), nearest=True)
        _zoom_to(axo, box, image.shape[1] / frame_w)
        title, subtitle = _layer_title(name, threshold)
        panel_title(axo, next(letters), title, subtitle or "close-up")


def _draw_prep_strip(fig, explanation, level: int) -> None:
    """PuzzleBoard: photo -> shrink + brighten -> smoothed (+ close-ups at L2, histogram L3)."""
    split = vis.split_steps(explanation)
    strip = vis.prep_strip(split.prep)
    if _is_grey_photo(explanation):
        strip = [s for s in strip if s.key != "grayscale"]
    if not strip:
        empty_panel(fig.add_subplot(), "The detector reported no preprocessing for this photo.")
        return
    box = vis.focus_box(explanation, vis.frame_size(explanation))
    rescale = explanation.step("rescale_gamma")
    grey = explanation.step("grayscale")
    rows = 1 if level == 1 else 2
    n = len(strip)
    ratios = []
    for k in range(n):
        ratios += [1.0] + ([0.12] if k < n - 1 else [])
    grid = fig.add_gridspec(rows, len(ratios), width_ratios=ratios,
                            height_ratios=[1.4, 1.0] if rows == 2 else [1.0])
    for k, step in enumerate(strip):
        if k:
            _arrow_axes(fig.add_subplot(grid[0, 2 * k - 1]))
        ax = fig.add_subplot(grid[0, 2 * k])
        small = step.image is not None and max(step.image.shape[:2]) < 400
        _show(ax, _as_seen(step), nearest=small)
        subtitle = _strip_subtitle(step, explanation)
        if level >= 2 and step.source in ("reproduced", "mixed"):
            subtitle += " · reproduced"
        panel_title(ax, chr(ord("A") + k), _STRIP_TITLES.get(step.key, step.title), subtitle)
        if rows == 2:
            _mark_box(ax, box, step.image_scale)
    if level == 2:
        for k, step in enumerate(strip):
            ax = fig.add_subplot(grid[1, 2 * k])
            _show(ax, _as_seen(step), nearest=True)
            _zoom_to(ax, box, step.image_scale)
            panel_title(ax, chr(ord("A") + n + k), "Close-up", "the dashed square, enlarged")
        return
    if level >= 3:
        sub = grid[1, :].subgridspec(1, 2, width_ratios=[1.0, 1.25])
        _brightness_histogram(fig.add_subplot(sub[0, 0]), grey, rescale, chr(ord("A") + n))
        last = strip[-1]
        ax_z = fig.add_subplot(sub[0, 1])
        _show(ax_z, last.image, nearest=True)
        _zoom_to(ax_z, box, last.image_scale)
        panel_title(ax_z, chr(ord("A") + n + 1), "Close-up of the finder's image",
                    "the dashed square")


def _brightness_histogram(ax, grey, rescale, letter: str = "D") -> None:
    t = _tok(ax)
    if rescale is not None and "histogram_before" in rescale.extra:
        before = np.asarray(rescale.extra["histogram_before"], dtype=float)
        after = np.asarray(rescale.extra["histogram_after"], dtype=float)
        x = np.arange(256)
        ax.fill_between(x, before / max(before.sum(), 1) * 100, step="mid", color=t["rejected"],
                        alpha=REJECTED_ALPHA, label="grey photo", zorder=2)
        ax.plot(x, after / max(after.sum(), 1) * 100, drawstyle="steps-mid", color=t["accent"],
                linewidth=1.6, label=f"after gamma {vis._fmt(rescale.counts.get('gamma'))}",
                zorder=3)
        ax.legend(loc="upper right")
        panel_title(ax, letter, "Brightness before and after",
                    "gamma lifts the dark tones the pattern sits in")
    elif grey is not None and "histogram" in grey.extra:
        hist = np.asarray(grey.extra["histogram"], dtype=float)
        ax.fill_between(np.arange(256), hist / max(hist.sum(), 1) * 100, step="mid",
                        color=t["accent"], alpha=0.8, zorder=2)
        panel_title(ax, letter, "Brightness of the grey image", "share of pixels at each level")
    else:
        empty_panel(ax, "No brightness histogram for this photo.")
        return
    ax.set_xlim(0, 255)
    ax.set_xlabel("grey level (0 black … 255 white)")
    ax.set_ylabel("pixels (%)")
    ax.yaxis.grid(True)


# ---------------------------------------------------------------------------------------------
# Stage 4: finding markers, one step at a time
# ---------------------------------------------------------------------------------------------

def _step_subtitle(step, level: int = 2) -> str:
    c = step.counts
    key = step.key
    if key == "contours":
        images = int(c.get("images traced", 1) or 1)
        return (f"{vis._fmt(c.get('outlines', 0))} outlines"
                + (f" · one of {images} traced images" if images > 1 and level >= 2 else ""))
    if key == "candidates":
        return (f"{vis._fmt(c.get('candidates', 0))} squares · "
                f"{vis._fmt(c.get('became markers', 0))} became markers")
    if key == "markers":
        return f"{vis._fmt(c.get('markers', 0))} markers"
    if key == "board_corners":
        return f"{vis._fmt(c.get('corners', 0))} corners"
    if key == "subpixel":
        return (f"{vis._fmt(c.get('moved', 0))} of {vis._fmt(c.get('corners', 0))} moved · "
                f"median {vis._fmt(c.get('median move px', 0))} px")
    if key == "grid_check":
        return f"{vis._fmt(c.get('kept', 0))} kept · {vis._fmt(c.get('removed', 0))} removed"
    if key == "final":
        return f"{vis._fmt(c.get('points found', 0))} points"
    if key == "puzzle_corners":
        return f"{vis._fmt(c.get('candidates', 0))} corner candidates"
    if key == "puzzle_grid":
        patches = int(c.get("patches", 0) or 0)
        return (f"{vis._fmt(c.get('grid points', 0))} grid points in {patches} "
                f"patch{'es' if patches != 1 else ''}")
    if key == "puzzle_bits":
        return f"{vis._fmt(c.get('bits read', 0))} bits · {vis._fmt(c.get('disagree', 0))} disagree"
    if key == "puzzle_decoded":
        return f"{vis._fmt(c.get('positions decoded', 0))} positions"
    if key == "puzzle_faces":
        return f"{vis._fmt(c.get('assigned', 0))} assigned · {vis._fmt(c.get('dropped', 0))} dropped"
    if key == "puzzle_duplicates":
        return f"{vis._fmt(c.get('dropped', 0))} dropped · {vis._fmt(c.get('kept', 0))} kept"
    first = next(iter(c.items()), None)
    return f"{vis._fmt(first[1])} {first[0]}" if first else ""


def draw_find_step(fig, explanation, step, level: int, context: Optional[dict] = None) -> None:
    """One detector step: the photo with its overlays (A), a close-up (B), per-level extras.

    Levels add evidence *drawn on or beside the picture* (reason bars, residual circles,
    labels), never a chart that repeats the chips (UX-14)."""
    context = context or {}
    if step.key == "decoding":
        _draw_decoding(fig, explanation, step, level)
        return
    box = vis.focus_box(explanation, vis.frame_size(explanation))
    reproduced = step.source in ("reproduced", "mixed")
    extras: list[str] = []
    if step.key == "subpixel" and step.insets:
        extras.append("insets")
        if level >= 3:
            extras.append("shift")
    else:
        extras.append("zoom")
        if level >= 2:
            if step.key == "candidates":
                extras.append("reasons")
            elif step.key == "contours":
                extras.append("binary")
            elif step.key == "grid_check" and "residual_px" in step.extra:
                extras.append("residual")
    ratios = [1.0]
    for item in extras:
        ratios.append({"zoom": 1.0, "binary": 1.0, "insets": 1.45, "reasons": 0.9,
                       "residual": 0.85, "shift": 0.75}[item])
    grid = fig.add_gridspec(1, len(ratios), width_ratios=ratios)
    fig_w, fig_h = (v * fig.dpi for v in fig.get_size_inches())
    ax = fig.add_subplot(grid[0, 0])
    main_image = step.image
    if step.key == "puzzle_corners":
        smooth = explanation.step("puzzle_smooth")
        if smooth is not None and smooth.image is not None and step.image is not None \
                and smooth.image.shape[:2] == step.image.shape[:2]:
            main_image = smooth.image
    small = main_image is not None and max(main_image.shape[:2]) < 400
    _show(ax, main_image, nearest=small)
    if step.image is None:
        empty_panel(ax, "No picture for this step.")
    if step.key == "puzzle_bits":
        _draw_bits(ax, step, main_image)
    else:
        main_ppu = None
        if main_image is not None:
            ih, iw = main_image.shape[:2]
            main_ppu = min(fig_w * ratios[0] / sum(ratios) * 0.9 / iw, fig_h * 0.82 / ih)
        _draw_overlays(ax, step, labels=(level >= 2 and step.key == "markers"
                                         and len(step.overlays[0]) <= 40),
                       face_palette=context.get("face_colours"), px_per_unit=main_ppu,
                       avoid_box=(tuple(v * float(step.image_scale or 1.0) for v in box)
                                  if "zoom" in extras else None))
    points = _context_points(explanation, step)
    _draw_context(ax, points)
    if step.key == "subpixel" and step.insets:
        _ring_insets(ax, _chosen_insets(step, level), float(step.image_scale or 1.0))
    # "reproduced" goes on the close-up's short subtitle (UX2-23: A's stays short).
    subtitle = _step_subtitle(step, level)
    first_extra = extras[0] if extras else ""
    if first_extra == "zoom":
        subtitle += " · dashed square = B"
    elif first_extra == "insets" and step.insets:
        subtitle += " · rings = close-ups"
    panel_title(ax, "A", step.title, subtitle)
    letters = iter("BCDEFG")
    for k, item in enumerate(extras, start=1):
        if item == "zoom":
            _mark_box(ax, box, step.image_scale)
            axz = fig.add_subplot(grid[0, k])
            letter = next(letters)
            if step.key == "puzzle_corners":
                _score_closeup(fig, axz, explanation, step, box, letter)
                continue
            _show(axz, main_image if step.key == "puzzle_bits" else step.image, nearest=True)
            if step.key == "puzzle_bits":
                _draw_bits(axz, step, main_image, zoom=True)
            elif step.key == "grid_check":
                exaggerate = _grid_exaggeration(step, box)
                _draw_grid_check(axz, step, box, level, exaggerate)
            else:
                side_px = min(fig_w * ratios[k] / sum(ratios) * 0.9, fig_h * 0.82)
                zbox = tuple(v * float(step.image_scale or 1.0) for v in box)
                _draw_overlays(
                    axz, step, zoom=True, face_palette=context.get("face_colours"),
                    label_box=zbox, px_per_unit=side_px / max(zbox[2] - zbox[0], 1e-6),
                    labels=step.key == "markers" or (level >= 2 and step.key in {
                        "board_corners", "final", "puzzle_decoded", "puzzle_faces"})
                    and not (level >= 3 and step.key == "candidates"))
                if level >= 3 and step.key == "candidates":
                    _reason_tags(axz, step, zbox, side_px / max(zbox[2] - zbox[0], 1e-6))
                _draw_context(axz, points, zoom=True)
            _zoom_to(axz, box, step.image_scale)
            title, sub = "Close-up", "the dashed square, enlarged"
            if level >= 2 and reproduced:
                sub += " · " + ("reproduced" if step.source == "reproduced" else
                                "partly reproduced")
            if step.key == "grid_check":
                title, sub = "Predicted vs found", _grid_check_subtitle(step, exaggerate)
            panel_title(axz, letter, title, sub)
        elif item == "binary":
            threshold = explanation.step("threshold")
            axb = fig.add_subplot(grid[0, k])
            letter = next(letters)
            if threshold is not None and threshold.image is not None:
                _show(axb, _as_seen(threshold), nearest=True)
                _draw_overlays(axb, _Rescaled(step, threshold.image_scale), zoom=True)
                _zoom_to(axb, box, threshold.image_scale)
                panel_title(axb, letter, "Traced in", "the black-and-white image, close-up")
            else:
                empty_panel(axb, "No black-and-white image for this photo.")
        elif item == "reasons":
            _reasons_panel(fig.add_subplot(grid[0, k]), step, next(letters))
        elif item == "residual":
            residual = np.asarray(step.extra.get("residual_px", []), dtype=float)
            tolerance = np.asarray(step.extra.get("tolerance_px", []), dtype=float)
            ratio = residual / np.where(tolerance > 0, tolerance, np.nan) \
                if residual.size == tolerance.size else residual
            _values_panel(fig.add_subplot(grid[0, k]), ratio, limit=1.0,
                          xlabel="disagreement ÷ allowed", title="How far off?",
                          subtitle="right of the line = removed", letter=next(letters),
                          limit_label="limit")
        elif item == "insets":
            used = _insets_panel(fig, grid[0, k], step, level, next(letters), context)
            for _ in range(used - 1):
                next(letters)
        elif item == "shift":
            shifts = np.asarray(step.extra.get("shift_px", []), dtype=float)
            _values_panel(fig.add_subplot(grid[0, k]), shifts, xlabel="move (px)",
                          title="How far each moved", subtitle="every corner of this photo",
                          letter=next(letters))


def _context_points(explanation, step) -> Optional[np.ndarray]:
    """When a step has nothing to draw (e.g. no duplicates), the points kept so far, in image px."""
    if any(len(o) for o in step.overlays) or step.image is None:
        return None
    points = np.asarray(getattr(explanation, "image_points", np.zeros((0, 2))), dtype=float)
    if not points.size:
        return None
    return points.reshape(-1, 2) * float(step.image_scale or 1.0)


def _draw_context(ax, points: Optional[np.ndarray], zoom: bool = False) -> None:
    if points is None:
        return
    t = _tok(ax)
    coll = ax.scatter(points[:, 0], points[:, 1], s=95 if zoom else 26, facecolors="none",
                      edgecolors=t["accent"], linewidths=1.8 if zoom else 1.2, zorder=4)
    coll.set_path_effects(_photo_halo(t["accent"], 1.2))


class _Rescaled:
    """A step seen through another image's scale (outlines on the threshold image)."""

    def __init__(self, step, scale: float) -> None:
        self.overlays = step.overlays
        self.image_scale = scale


def _reasons_panel(ax, step, letter: str) -> None:
    """What happened to every four-cornered outline: became a marker, a square that was not a
    marker (per reason), or not square-shaped (per reason). The numbers add up to the chips
    (UX-12): markers + dropped = candidates + not square."""
    t = _tok(ax)
    kept = 0
    candidate_reasons: dict[str, int] = {}
    shape_reasons: dict[str, int] = {}
    for overlay in step.overlays:
        states = overlay.states or []
        reasons = overlay.reasons or [""] * len(states)
        target = shape_reasons if overlay.name == "not square" else candidate_reasons
        for state, reason in zip(states, reasons):
            if state == "kept":
                kept += 1
            else:
                key = reason or "not read as a marker"
                target[key] = target.get(key, 0) + 1
    plain = {"its border is not solid black": "border not solid",
             "its pattern is not one of this target's markers": "unknown code",
             "too little contrast to read": "too faint to read",
             "not read as a marker": "read, not kept",
             "a side is too short": "a side too short",
             "corners too close together": "corners too close",
             "too close to the image edge": "at the image edge"}
    rows = [("became a marker", kept, t["accent"], 1.0)]
    rows += [("square: " + plain.get(k, k), v, t["text"], 0.55) for k, v in
             sorted(candidate_reasons.items(), key=lambda kv: -kv[1])[:4]]
    rows += [("shape: " + plain.get(k, k), v, t["rejected"], REJECTED_ALPHA) for k, v in
             sorted(shape_reasons.items(), key=lambda kv: -kv[1])[:3]]
    y = np.arange(len(rows))[::-1]
    for yy, (_label, value, colour, alpha) in zip(y, rows):
        ax.barh([yy], [value], color=colour, alpha=alpha, height=0.62, zorder=3)
        ax.annotate(str(value), (value, yy), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=TICK_SIZE, color=t["text"])
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(label, 22) for label, *_ in rows], fontsize=TICK_SIZE)
    ax.tick_params(axis="y", length=0)
    top = max([v for _l, v, *_ in rows] + [1])
    ax.set_xlim(0, top * 1.3)
    integer_axis(ax, "x")
    ax.xaxis.grid(True)
    ax.set_xlabel("outlines")
    total = kept + sum(candidate_reasons.values()) + sum(shape_reasons.values())
    panel_title(ax, letter, f"What happened to the {total} outlines",
                "every outline with four corners")


def _reason_tags(ax, step, box=None, px_per_unit=None) -> None:
    """L3: a short tag on each rejected candidate in the close-up (only those inside it,
    never overlapping)."""
    t = _tok(ax)
    scale = float(step.image_scale or 1.0)
    short = {"its border is not solid black": "border", "too little contrast to read": "faint",
             "its pattern is not one of this target's markers": "unknown",
             "not convex": "not convex", "corners too close together": "too small",
             "too close to the image edge": "at edge", "a side is too short": "too small"}
    anchors, labels = [], []
    for overlay in step.overlays:
        for poly, state, reason in zip(overlay.data, overlay.states or [],
                                       overlay.reasons or []):
            if state == "kept" or not reason:
                continue
            anchors.append(np.asarray(poly, dtype=float).reshape(-1, 2).mean(axis=0) * scale)
            labels.append("✕ " + short.get(reason, "dropped"))
    _place_tags(ax, anchors, labels, t, box=box, px_per_unit=px_per_unit, limit=4)


#: Most reason tags drawn in one close-up (UX2-12): the bars beside it count the rest.
MAX_REASON_TAGS = 4


def _place_tags(ax, anchors, labels, t, *, box=None, px_per_unit=None,
                limit: int = MAX_REASON_TAGS, fontsize: float = 9.0) -> int:
    """Up to ``limit`` short red-✕ tags, each at its item or nudged above / below / beside
    it so no two tags touch (estimated text boxes, 4 px apart). Returns how many were drawn."""
    import matplotlib.patheffects as pe
    if box is None or not px_per_unit:
        return 0
    x0, y0, x1, y1 = box
    # Generous estimated boxes: the final layout may still scale the close-up a little.
    height = fontsize * _PX_PER_PT * 1.9
    placed: list[tuple[float, float, float, float]] = []
    drawn = 0
    order = np.argsort([np.hypot(a[0] - (x0 + x1) / 2, a[1] - (y0 + y1) / 2) for a in anchors]) \
        if len(anchors) else []
    for k in order:
        if drawn >= limit:
            break
        (x, y), text = anchors[k], labels[k]
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            continue
        width = len(text) * fontsize * _PX_PER_PT * 0.68 + 14
        px, py = (x - x0) * px_per_unit, (y - y0) * px_per_unit
        w_box, h_box = (x1 - x0) * px_per_unit, (y1 - y0) * px_per_unit
        for dx, dy in ((0, 0), (0, -1.2), (0, 1.2), (-0.7, 0), (0.7, 0), (0, -2.4), (0, 2.4)):
            cx, cy = px + dx * width, py + dy * height
            rect = (cx - width / 2 - 2, cy - height / 2 - 2, cx + width / 2 + 2, cy + height / 2 + 2)
            if rect[0] < 2 or rect[2] > w_box - 2 or rect[1] < 2 or rect[3] > h_box - 2:
                continue
            # 6 px of clear space between tags (the drawn boxes are wider than the estimate)
            if any(rect[0] - 6 < r[2] and r[0] < rect[2] + 6 and rect[1] - 6 < r[3]
                   and r[1] < rect[3] + 6 for r in placed):
                continue
            placed.append(rect)
            # A stage-coloured stroke, not a box: the text keeps its own (wide, low) extent.
            ax.annotate(text, (x0 + cx / px_per_unit, y0 + cy / px_per_unit), ha="center",
                        va="center", fontsize=fontsize, color=t["bad"], zorder=6,
                        annotation_clip=False, fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=3.2, foreground=t["stage"])])
            drawn += 1
            break
    return drawn


def _grid_check_subtitle(step, exaggerate: float = 20.0) -> str:
    return f"+ predicted · ring = found · dashed = allowed (magnified ×{exaggerate:g})"


def _nice_factor(value: float) -> float:
    for nice in (2, 5, 10, 20, 50, 100, 200, 500):
        if value <= nice * 1.4:
            return float(nice)
    return 500.0


def _grid_exaggeration(step, box) -> float:
    """A magnification that makes the largest allowed circle in the close-up about 40 % of
    the corner spacing, rounded to 2, 5, 10, 20, 50, ..."""
    scale = float(step.image_scale or 1.0)
    predicted = next((o for o in step.overlays if o.name == "predicted"), None)
    if predicted is None or not len(predicted):
        return 20.0
    pred = np.asarray(predicted.data, dtype=float).reshape(-1, 2) * scale
    tolerance = np.asarray(step.extra.get("tolerance_px", np.ones(len(pred))), dtype=float) * scale
    x0, y0, x1, y1 = (v * scale for v in box)
    inside = (pred[:, 0] >= x0) & (pred[:, 0] <= x1) & (pred[:, 1] >= y0) & (pred[:, 1] <= y1)
    pts = pred[inside] if inside.sum() >= 2 else pred
    if len(pts) < 2 or tolerance.size != len(pred):
        return 20.0
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    d[d == 0] = np.inf
    spacing = float(np.median(d.min(axis=1)))
    top = float(np.max(tolerance[inside] if inside.any() else tolerance))
    if not np.isfinite(spacing) or top <= 0:
        return 20.0
    return _nice_factor(0.4 * spacing / top)


def _draw_grid_check(ax, step, box, level: int, exaggerate: float = 20.0) -> None:
    """Each corner's predicted place (+), its allowed circle and the disagreement arrow,
    both magnified; when nothing is removed, one dashed example of a corner that would be."""
    from matplotlib.patches import Circle
    t = _tok(ax)
    scale = float(step.image_scale or 1.0)
    corners = next((o for o in step.overlays if o.name == "corners"), None)
    predicted = next((o for o in step.overlays if o.name == "predicted"), None)
    if corners is None or predicted is None or not len(corners):
        _draw_overlays(ax, step, zoom=True)
        return
    actual = np.asarray(corners.data, dtype=float).reshape(-1, 2) * scale
    pred = np.asarray(predicted.data, dtype=float).reshape(-1, 2) * scale
    tolerance = np.asarray(step.extra.get("tolerance_px", np.ones(len(actual))), dtype=float) * scale
    if tolerance.size != len(actual):
        tolerance = np.full(len(actual), float(np.median(tolerance)) if tolerance.size else scale)
    states = list(corners.states or ["kept"] * len(actual))
    x0, y0, x1, y1 = (v * scale for v in box)
    inside = (pred[:, 0] >= x0) & (pred[:, 0] <= x1) & (pred[:, 1] >= y0) & (pred[:, 1] <= y1)
    for p, a, tol, state, show in zip(pred, actual, tolerance, states, inside):
        if not show:
            continue
        colour = t["bad"] if state != "kept" else t["accent"]
        circle = Circle(p, max(tol, 0.05) * exaggerate, fill=False, edgecolor=t["muted"],
                        linewidth=1.2, linestyle=(0, (3, 2)), zorder=4)
        circle.set_path_effects(_photo_halo(t["muted"], 1.2))
        ax.add_patch(circle)
        tip = p + (a - p) * exaggerate
        ax.annotate("", xy=tip, xytext=p, arrowprops=dict(
            arrowstyle="-|>,head_length=0.45,head_width=0.22", color=colour, linewidth=1.8,
            shrinkA=0, shrinkB=0, path_effects=_photo_halo(colour, 1.8)), zorder=6)
        plus = ax.scatter([p[0]], [p[1]], marker="+", s=220, color=t["text"], linewidths=2.2,
                          zorder=7)
        plus.set_path_effects(_photo_halo(t["text"], 2.2))
        # The corner as found (ED3-9), at the magnified offset like the arrow it ends.
        ring = ax.scatter([tip[0]], [tip[1]], s=150, facecolors="none", edgecolors=colour,
                          linewidths=2.0, zorder=8)
        ring.set_path_effects(_photo_halo(colour, 2.0))
    removed = [s for s in states if s != "kept"]
    shown = np.flatnonzero(inside)
    if not removed and len(shown):
        # One illustration (dashed, labelled "example"), never a real corner.
        middle = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        k = shown[np.argmin(np.linalg.norm(pred[shown] - middle, axis=1))]
        p, tol = pred[k], max(tolerance[k], 0.05)
        tip = p + np.array([0.0, 1.0]) * tol * exaggerate * 1.6
        ax.annotate("", xy=tip, xytext=p, arrowprops=dict(
            arrowstyle="-|>,head_length=0.45,head_width=0.22", color=t["muted"], linewidth=1.6,
            linestyle=(0, (3, 2)), shrinkA=0, shrinkB=0), zorder=6)
        # The note sits in the margin under the photo, never on it (UX2-12, ED3-9).
        ax.annotate("grey dashed arrow = illustration only:\n"
                    "a corner found that far out would be removed", (0.5, 0.0),
                    xycoords="axes fraction", xytext=(0, -6),
                    textcoords="offset points", ha="center", va="top", fontsize=9,
                    color=t["muted"], annotation_clip=False, wrap=False)


def _draw_bits(ax, step, image, zoom: bool = False) -> None:
    """PuzzleBoard edge bits as filled squares, dark or light as read (UX-24); a red edge on a
    bit that disagrees with the printed code."""
    t = _tok(ax)
    overlay = next((o for o in step.overlays if o.name == "edge bits"), None)
    if overlay is None or not len(overlay):
        return
    scale = float(step.image_scale or 1.0)
    xy = np.asarray(overlay.data, dtype=float).reshape(-1, 2) * scale
    values = np.asarray(step.extra.get("bit_values", np.zeros(len(xy))), dtype=int)
    if values.size != len(xy):
        values = np.zeros(len(xy), dtype=int)
    dark_value = _dark_bit_value(image, xy, values)
    fill = ["#111111" if v == dark_value else ("#F5F5F5" if v != 0 else "none")
            for v in values]
    edges = [t["bad"] if s == "bit_wrong" else t["accent"] for s in overlay.states or []]
    size = 80 if zoom else 18
    coll = ax.scatter(xy[:, 0], xy[:, 1], marker="s", s=size, c=fill, edgecolors=edges,
                      linewidths=2.2 if zoom else 1.1, zorder=5)
    coll.set_path_effects(_photo_halo(t["accent"], 1.0))


def _dark_bit_value(image, xy: np.ndarray, values: np.ndarray) -> int:
    """Which stored bit value (the detector stores -1 / +1) prints dark: the one whose dots
    are darker in the picture itself."""
    if image is None or not len(xy):
        return -1
    grey = np.asarray(image, dtype=float)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    xi = np.clip(np.rint(xy[:, 0]).astype(int), 0, grey.shape[1] - 1)
    yi = np.clip(np.rint(xy[:, 1]).astype(int), 0, grey.shape[0] - 1)
    sampled = grey[yi, xi]
    means = {v: float(sampled[values == v].mean()) for v in (-1, 1) if np.any(values == v)}
    if len(means) < 2:
        return -1
    return min(means, key=means.get)


def _score_closeup(fig, ax, explanation, step, box, letter: str) -> None:
    """PuzzleBoard crossing score laid over the grey close-up (UX-15): transparent where the
    score is low, the accent where high, candidate rings on top, a score bar."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgba
    smooth = explanation.step("puzzle_smooth")
    base = smooth.image if smooth is not None and smooth.image is not None else step.image
    _show(ax, base, nearest=True)
    # One hue (the amount hue, never the error map's yellow, UX3-3), opacity rising with the
    # score from a low floor, so low-score regions read as well as the peaks (ED3-16).
    hue = to_rgba(amount_hue(ax))
    cmap = LinearSegmentedColormap.from_list(
        "edu_score", [(0.0, (hue[0], hue[1], hue[2], 0.14)),
                      (0.15, (hue[0], hue[1], hue[2], 0.32)),
                      (0.5, (hue[0], hue[1], hue[2], 0.62)),
                      (1.0, (hue[0], hue[1], hue[2], 0.92))])
    continuous = step.extra.get("score_map") if hasattr(step, "extra") else None
    if continuous is not None and np.shape(continuous)[:2] == np.shape(step.image)[:2]:
        score = np.asarray(continuous, dtype=float) / 255.0   # already stretched (sqrt)
        sub = "tint = crossing score · rings = its peaks"
    else:
        score = np.asarray(step.image, dtype=float) / 255.0
        positive = score[score > 0]
        top = float(np.percentile(positive, 99)) if positive.size else 1.0
        score = np.clip(score / max(top, 1e-6), 0.0, 1.0) ** 0.5  # weak spots show too
        sub = "tint = score at its peaks · rings = candidates"
    heat = ax.imshow(score, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest", zorder=1)
    _draw_overlays(ax, step, zoom=True)
    _zoom_to(ax, box, step.image_scale)
    panel_title(ax, letter, "Crossing score, close-up", sub)
    heat._edu_quantity = "amount"
    bar = colourbar_px(fig, heat, ax, label="score", shrink=0.6)
    bar.set_ticks([0, 1])
    bar.set_ticklabels(["low", "high"])


def _insets_panel(fig, spec, step, level: int, letter: str, context: dict) -> int:
    """Corner close-ups, contrast-stretched: the two edges that cross (dashed), the first
    position (ring), the refined one (dot) and an arrow between them (ED-11)."""
    insets = _chosen_insets(step, level)
    cols = 2 if len(insets) >= 2 else 1
    rows = int(np.ceil(len(insets) / cols))
    sub = spec.subgridspec(rows, cols)
    um = context.get("um_per_px", float("nan"))
    n_faces = int(context.get("n_faces", 1) or 1)
    for k, inset in enumerate(insets):
        ax = fig.add_subplot(sub[k // cols, k % cols])
        t = _tok(ax)
        _show(ax, inset.image, nearest=True, stretch=True)
        factor = inset.zoom * inset.frame_to_crop
        origin = np.asarray(inset.origin, dtype=float)
        before = (np.asarray(inset.before) - origin) * factor - 0.5 if inset.before is not None else None
        after = (np.asarray(inset.after) - origin) * factor - 0.5 if inset.after is not None else None
        if after is not None and len(after):
            edges = vis.corner_edges(inset.image, after[0], radius=max(3.0, 1.2 * inset.zoom))
            if edges is not None:
                reach = max(inset.image.shape[:2])
                for direction in edges:
                    a, b = after[0] - direction * reach, after[0] + direction * reach
                    ax.plot([a[0], b[0]], [a[1], b[1]], color=t["accent"], linewidth=1.3,
                            linestyle=(0, (5, 3)), alpha=0.9, zorder=3,
                            path_effects=_photo_halo(t["accent"], 1.3))
        if before is not None and len(before):
            coll = ax.scatter(before[:, 0], before[:, 1], s=190, facecolors="none",
                              edgecolors=t["warn"], linewidths=2.2, zorder=4)
            coll.set_path_effects(_photo_halo(t["warn"], 2.2))
        if before is not None and after is not None and len(before) and len(after):
            ax.annotate("", xy=after[0], xytext=before[0], arrowprops=dict(
                arrowstyle="-|>,head_length=0.6,head_width=0.3", color=t["text"], linewidth=1.8,
                shrinkA=0, shrinkB=3, path_effects=_photo_halo(t["text"], 1.8)), zorder=5)
        if after is not None and len(after):
            coll = ax.scatter(after[:, 0], after[:, 1], s=60, color=t["accent"], zorder=6)
            coll.set_path_effects(_photo_halo(t["accent"], 1.0))
        h, w = inset.image.shape[:2]
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        moved = inset.label or ""
        try:
            px = float(moved.split()[0])
        except (ValueError, IndexError):
            px = float("nan")
        typical = k == len(insets) - 1 and _is_typical(step, inset)
        title = ("Typical: " if typical else ("Largest: " if k == 0 else "Moved ")) + \
            (moved if moved else "corner")
        if np.isfinite(px) and np.isfinite(um):
            title += f" ≈ {format_number(px * um, 0)} µm"
        name = vis.corner_name(inset.title, n_faces)
        # SCI3-12: the µm figure is the target's scale (its untilted direction), not a
        # distance along a tilted face; the sub-pixel chip says "at the target's scale".
        first_sub = f"{name} · one block = one pixel"
        panel_title(ax, chr(ord(letter) + k), title[:1].upper() + title[1:],
                    first_sub if k == 0 else name)
    return max(len(insets), 1)


def _is_typical(step, inset) -> bool:
    """True for the close-up of the median move (the detector adds it after the largest)."""
    shifts = np.asarray(step.extra.get("shift_px", []), dtype=float)
    shifts = shifts[np.isfinite(shifts)]
    try:
        moved = float(str(inset.label).split()[0])
    except (ValueError, IndexError):
        return False
    if not shifts.size:
        return False
    return abs(moved - float(np.median(shifts))) <= max(0.02, 0.25 * float(np.median(shifts)))


def _chosen_insets(step, level: int) -> list:
    """The close-ups drawn: the largest moves (1 at L1, 3 from L2) and the typical (median)
    corner last, so a learner sees both the extreme and the ordinary case (ED2-16)."""
    insets = list(step.insets)
    if not insets:
        return []
    typical = [i for i in insets if _is_typical(step, i)]
    largest = [i for i in insets if not any(i is j for j in typical)]
    want = 1 if level == 1 else 3
    chosen = largest[:want] + typical[:1]
    return chosen if chosen else insets[: (2 if level == 1 else 4)]


def _ring_insets(ax, insets, scale: float) -> None:
    """Ring the corners shown in the close-ups on the full photo (dashed, the text colour)."""
    t = _tok(ax)
    xy = [np.asarray(i.after, dtype=float).reshape(-1, 2)[0] for i in insets
          if i.after is not None and len(np.asarray(i.after).reshape(-1, 2))]
    if not xy:
        return
    xy = np.asarray(xy) * scale
    coll = ax.scatter(xy[:, 0], xy[:, 1], s=420, facecolors="none", edgecolors=t["text"],
                      linewidths=1.6, linestyles=(0, (3, 2)), zorder=6)
    coll.set_path_effects(_photo_halo(t["text"], 1.6))


def _draw_decoding(fig, explanation, step, level: int) -> None:
    """Reading IDs (ED-10): A one square worked through (cells, ring, 0/1), B its code looked up
    in the target's list, C the gallery of every square with the detector's verdict."""
    crops = list(step.crops)
    worked = next((c for c in crops if c.detector_id is not None and c.code is not None
                   and c.lookup), None)
    if worked is None:
        worked = next((c for c in crops if c.detector_id is not None), None)
    show_lookup = worked is not None and worked.code is not None and worked.lookup
    ratios = ([1.05] if worked is not None else []) + ([0.75] if show_lookup else []) + [2.0]
    # Constrained layout shrinks a gallery of many small titled tiles to stamps; this figure
    # uses a fixed grid and keeps only the shared title alignment.
    _use_fixed_layout(fig)
    grid = fig.add_gridspec(1, len(ratios), width_ratios=ratios, left=0.008, right=0.995,
                            top=0.9, bottom=0.03, wspace=0.07)
    col = 0
    letters = iter("ABCD")
    if worked is not None:
        ax = fig.add_subplot(grid[0, col])
        ax.set_anchor("N")
        _draw_worked_crop(ax, worked)
        side = worked.marker_bits + 2
        ring = "white" if worked.inverted else "black"
        panel_title(ax, next(letters), f"Reading ID {_shown_key(worked.detector_id)}",
                    f"{side} × {side} cells · the {ring} ring must be solid")
        # UX6-5 (round 10): the square is narrower than its cell, so the space under A held
        # nothing; a two-line key to the digits takes it (text outside the axes, below the crop)
        key_colour = _tok(ax)["muted"]
        ax.text(0.0, -0.03, "Each inner cell is read as a 1 or a 0 (the digit shown).\n"
                "The inner cells form the code, looked up in B.",
                transform=ax.transAxes, ha="left", va="top", fontsize=TICK_SIZE,
                color=key_colour, linespacing=1.35, clip_on=False)
        col += 1
    if show_lookup:
        _draw_lookup(fig, grid[0, col], worked, next(letters))
        col += 1
    _draw_gallery(fig, grid[0, col], explanation, step, level, next(letters),
                  exclude=None)


def _use_fixed_layout(fig) -> None:
    """Swap the figure's constrained layout for fixed gridspec margins plus the shared title
    alignment and eliding (``education_style.finish_panel_layout``)."""
    from matplotlib.layout_engine import LayoutEngine
    from pyCamSet.gui.education_style import finish_panel_layout

    class _TitlesOnly(LayoutEngine):
        _adjust_compatible = True
        _colorbar_gridspec = True

        def execute(self, figure):
            try:
                finish_panel_layout(figure)
            except Exception:  # noqa: BLE001 - the picture must still draw
                pass

    fig.set_layout_engine(_TitlesOnly())


def _draw_worked_crop(ax, crop) -> None:
    """One flattened square, large: its cell grid, the ring cells hatched, 0/1 in every
    inner cell as read (1 = black)."""
    from matplotlib.patches import Rectangle
    t = _tok(ax)
    image = np.asarray(crop.image)
    side = image.shape[0]
    _show(ax, image)
    n = crop.bits.shape[0] if crop.bits is not None else crop.marker_bits + 2
    cell = side / float(n)
    for k in range(1, n):
        ax.axhline(k * cell - 0.5, color=t["muted"], linewidth=0.7, alpha=0.8)
        ax.axvline(k * cell - 0.5, color=t["muted"], linewidth=0.7, alpha=0.8)
    for r in range(n):
        for c in range(n):
            if 0 < r < n - 1 and 0 < c < n - 1:
                continue
            ax.add_patch(Rectangle((c * cell - 0.5, r * cell - 0.5), cell, cell, fill=False,
                                   hatch="///", edgecolor=_with_alpha(t["accent"], 0.55),
                                   linewidth=0.0, zorder=2))
    inner = Rectangle((cell - 0.5, cell - 0.5), side - 2 * cell, side - 2 * cell, fill=False,
                      edgecolor=t["accent"], linewidth=2.2, zorder=3)
    ax.add_patch(inner)
    if crop.bits is not None:
        bits = np.asarray(crop.bits)
        for r in range(1, n - 1):
            for c in range(1, n - 1):
                black = bits[r, c] == 1
                ax.text((c + 0.5) * cell - 0.5, (r + 0.5) * cell - 0.5, "1" if black else "0",
                        ha="center", va="center", fontsize=11, fontweight="bold",
                        color="#FFFFFF" if black else "#111111", zorder=4)
    for spine in ax.spines.values():
        spine.set_color(t["accent"])
        spine.set_linewidth(2.4)


def _code_square(ax, bits: np.ndarray, *, frame: str, width: float) -> None:
    """A dictionary-style code: n x n cells, 1 = white, inside a black ring."""
    n = bits.shape[0]
    full = np.zeros((n + 2, n + 2))
    full[1:-1, 1:-1] = bits
    ax.imshow(full, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(frame)
        spine.set_linewidth(width)


def _draw_lookup(fig, spec, crop, letter: str) -> None:
    """The read code (turned to the dictionary's orientation) above the dictionary entries
    around the match: the one that is equal gets the accent frame (ED-10)."""
    entries = list(crop.lookup)[:2]
    n = max(len(entries), 1)
    sub = spec.subgridspec(4, n, height_ratios=[1.0, 0.32, 0.55, 0.42], hspace=0.04,
                           wspace=0.22)
    ax = fig.add_subplot(sub[0, :])
    ax.set_anchor("N")
    t = _tok(ax)
    _code_square(ax, np.asarray(crop.code), frame=t["accent"], width=2.4)
    turned = {0: "", 1: " · turned 90°", 2: " · turned 180°", 3: " · turned 270°"}[crop.turns % 4]
    swapped = " · light/dark swapped" if crop.inverted else ""
    total = f"{crop.dictionary_size:,}" if crop.dictionary_size else "its"
    panel_title(ax, letter, "Code → ID", f"the inner cells{turned}{swapped}")
    gap = fig.add_subplot(sub[1, :])
    gap.set_axis_off()
    gap.set_xlim(0, 1)
    gap.set_ylim(0, 1)
    gap.annotate("", xy=(0.3, 0.08), xytext=(0.3, 0.92),
                 arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.35",
                                 color=t["muted"], linewidth=1.8))
    gap.text(0.36, 0.5, f"look up among\n{total} markers", ha="left", va="center",
             fontsize=9, color=t["muted"])
    names = fig.add_subplot(sub[3, :])
    names.set_axis_off()
    names.set_xlim(0, n)
    names.set_ylim(0, 1)
    for k, (marker_id, bits) in enumerate(entries[:2]):
        axk = fig.add_subplot(sub[2, k])
        bits = np.asarray(bits)
        differ = int(np.sum(bits != crop.code)) if bits.shape == np.asarray(crop.code).shape else -1
        match = marker_id == crop.marker_id
        _code_square(axk, bits, frame=t["accent"] if match else t["grid"],
                     width=2.6 if match else 1.0)
        ties = int(getattr(crop, "closest_ties", 0) or 0)
        verdict = "✓ same" if match and differ <= 0 else (
            f"✓ {differ} fixed" if match else
            (f"closest other\n{differ} cells differ" if ties <= 1 else
             f"1 of {ties} closest\n{differ} cells differ"))   # SCI3-14: a tie is said
        names.text(k + 0.5, 0.97, f"ID {marker_id}\n{verdict}", ha="center", va="top",
                   fontsize=9, color=t["accent"] if match else t["muted"],
                   fontweight="bold" if match else "normal", linespacing=1.15)
    rule = _misread_rule(crop)
    if rule:
        names.text(n / 2, 0.0, rule, ha="center", va="bottom", fontsize=8.5,
                   color=t["text"], linespacing=1.15)


def _misread_rule(crop) -> str:
    """How safe the lookup is (ED3-10): with the closest other code ``d`` cells away and
    ``c`` cells corrected, fewer than ``d - c`` misread cells can never give a wrong ID."""
    d = int(getattr(crop, "closest_distance", -1) or -1)
    c = int(getattr(crop, "correctable_cells", 0) or 0)
    if d <= 0:
        return ""
    if c <= 0:
        cells = "1 misread cell" if d == 2 else f"up to {d - 1} misread cells"
        return f"{cells} → no ID,\nnever a wrong one"
    return f"up to {c} misread cells fixed;\na wrong ID needs {d - c}"


#: Squares that were not markers shown per reason (UX2-22: no wall of rejects); the rest are
#: counted on one summary tile. L1 shows none of them.
REJECT_EXAMPLES = {2: 2, 3: 4}


def _gallery_rejects(crops: list, per_reason: int) -> list:
    """Up to ``per_reason`` rejected squares of each verdict, grouped by verdict."""
    by_kind: dict[str, list] = {}
    for crop in crops:
        if crop.detector_id is None:
            by_kind.setdefault(vis.decode_category(crop), []).append(crop)
    order = [key for key, _l, _t in vis.DECODE_CATEGORIES if key in by_kind]
    return [c for key in order for c in by_kind[key][:per_reason]]


def _draw_gallery(fig, spec, explanation, step, level: int, letter: str, exclude=None) -> None:
    """Every marker, flattened and framed in the accent with its ID; at L2/L3 a few squares
    of each kind that were not markers, muted with a grey ✕ reason, and one tile counting the
    rest (UX-16, UX2-22). L1 shows the markers plus one summary tile."""
    width, height = fig.get_size_inches()
    crops_all = list(step.crops)
    counts = vis.decode_counts(step)
    markers_only = [c for c in crops_all if c.detector_id is not None]
    if level == 1:
        shown = markers_only
    else:
        shown = markers_only + _gallery_rejects(crops_all, REJECT_EXAMPLES.get(level, 4))
    rejected = len(crops_all) - len(shown)
    share = 0.55
    gallery_w = width * share - 0.2
    gallery_h = height - 0.9
    capacity = max(6, int(gallery_w / 0.9) * int(gallery_h / 1.1))
    crops = vis.select_crops(shown, capacity - (1 if rejected else 0))
    n_tiles = len(crops) + (1 if rejected else 0)
    rows, cols = _grid_shape(max(n_tiles, 1), 1.25, gallery_w, gallery_h / 0.82)
    sub = spec.subgridspec(rows + 1, cols, height_ratios=[0.001] + [1.0] * rows,
                           wspace=0.12, hspace=0.32)
    head = fig.add_subplot(sub[0, :])
    head.set_axis_off()
    for k in range(rows * cols):
        ax = fig.add_subplot(sub[1 + k // cols, k % cols])
        ax.set_axis_off()
        if k < len(crops):
            _draw_crop(ax, crops[k], level)
        elif k == len(crops) and rejected:
            _summary_tile(ax, rejected, counts, more=level > 1)
    if not crops and not rejected:
        empty_panel(fig.add_subplot(sub[1, :]), "No square candidates in this photo.")
    markers = counts["marker"]
    total = len(crops_all)
    sub_text = f"{total} squares · {markers} markers"
    if level >= 2 and total > len(crops):
        sub_text += f" · {len(crops)} shown"
    if level >= 3:
        read = [c for c in crops_all if c.detector_id is not None and c.code is not None]
        fixed = sum(1 for c in read if c.bit_errors)
        sub_text += (f" · cells fixed by error correction: {fixed} of {len(read)} (✎)"
                     if fixed else f" · every code read exactly ({len(read)})")
    panel_title(head, letter, "Every square, read", sub_text)


def _summary_tile(ax, rejected: int, counts: dict, more: bool = False) -> None:
    t = _tok(ax)
    ax.set_axis_on()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(t["rejected"])
        spine.set_linestyle((0, (3, 3)))
        spine.set_alpha(REJECTED_ALPHA + 0.2)
    ax.text(0.5, 0.58, f"+{rejected}", ha="center", va="center", fontsize=16,
            fontweight="bold", color=t["muted"])
    ax.text(0.5, 0.28, "not markers", ha="center",
            va="center", fontsize=9, color=t["muted"])
    ax.set_title("rest in chips" if more else "L2 shows some", fontsize=8.5,
                 color=t["muted"], pad=2, loc="center")


def _draw_crop(ax, crop, level: int) -> None:
    from matplotlib.patches import Rectangle
    t = _tok(ax)
    image = np.asarray(crop.image)
    side = image.shape[0]
    ax.set_axis_on()
    category = vis.decode_category(crop)
    marker = category == "marker"
    shown = _show(ax, image, nearest=False)
    if not marker and shown is not None:
        shown.set_alpha(REJECTED_ALPHA + 0.15)
    n = crop.bits.shape[0] if crop.bits is not None else crop.marker_bits + 2
    cell = side / float(n)
    for k in range(1, n):
        ax.axhline(k * cell - 0.5, color=t["muted"], linewidth=0.4, alpha=0.6)
        ax.axvline(k * cell - 0.5, color=t["muted"], linewidth=0.4, alpha=0.6)
    if marker:
        # The code area only; the cells as read are written out once, large, in A (UX2-22:
        # a dot per cell on every small tile read as noise).
        inner = Rectangle((cell - 0.5, cell - 0.5), side - 2 * cell, side - 2 * cell, fill=False,
                          edgecolor=t["accent"], linewidth=1.2, zorder=3)
        ax.add_patch(inner)
    for spine in ax.spines.values():
        spine.set_color(t["accent"] if marker else t["rejected"])
        spine.set_linewidth(2.0 if marker else 1.0)
        spine.set_alpha(1.0 if marker else REJECTED_ALPHA + 0.2)
    if marker:
        label = f"ID {_shown_key(crop.detector_id)}"
        if level >= 3 and crop.code is not None and crop.bit_errors:
            label += f" ·{crop.bit_errors}✎"
        ax.set_title(label, fontsize=9, color=t["accent"], pad=2, loc="center",
                     fontweight="bold")
    else:
        tag = next((tg for key, _l, tg in vis.DECODE_CATEGORIES if key == category), "")
        # Not a failure of the run (most squares on a photo are not markers): muted, so the
        # gallery is not a wall of red (UX2-22).
        ax.set_title("✕ " + tag, fontsize=9, color=t["muted"], pad=2, loc="center")


# ---------------------------------------------------------------------------------------------
# Stage 5: final
# ---------------------------------------------------------------------------------------------

def _coverage_axes(ax, points_frame, frame_wh, *, title, subtitle, camera, letter=None,
                   small=False, highlight=False, vmax=None):
    t = _tok(ax)
    w, h = frame_wh
    counts, extent = vis.coverage_grid(points_frame, w, h, 14 if small else 22)
    masked = np.ma.masked_where(counts <= 0, counts)
    # UX3-3: a count, so the one-hue amount map (viridis is kept for errors in px).
    scale = amount_scale(counts[counts > 0], ax, unit="points",
                         hi=vmax if vmax else max(float(counts.max()), 1.0))
    image = ax.imshow(masked, extent=extent, aspect="equal", zorder=1, **scale.imshow_kwargs())
    image._edu_quantity = "amount"
    image._edu_scale = scale
    ax.set_facecolor(t["stage"])
    hull = vis.hull_polygon(points_frame)
    if hull is not None:
        from matplotlib.patches import Polygon
        ring = Polygon(hull, closed=True, fill=False, edgecolor=t["accent"],
                       linewidth=1.4 if small else 2.0, zorder=3)
        ax.add_patch(ring)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    frame_axes(ax, camera, width=3.0 if highlight else (1.2 if small else 2.0))
    if small:
        panel_title(ax, None, title, subtitle)
        _shift_title(ax, 19)
        draw_camera_badges(ax, [(0.0, 1.0)], [camera],
                           transform=_title_line_transform(ax, bool(subtitle)),
                           size=9, highlight=[camera] if highlight else ())
    else:
        panel_title(ax, letter, title, subtitle)
    return image


def draw_final(fig, data: dict, level: int) -> None:
    """Saved circles on the photo + where this camera's points fall over all its photos."""
    from pyCamSet.gui.education_style import image_with_points
    cams = data["cameras"]
    sel = data["selected"]
    if level == 1:
        grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    elif level == 2:
        grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5])
    else:
        grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.5], height_ratios=[2.4, 1.0])
    ax = fig.add_subplot(grid[:, 0] if level < 3 else grid[0, 0])
    photo = data.get("photo")
    pts = np.asarray(data["points"], dtype=float).reshape(-1, 2) * data["to_photo"]
    label = data["photo_label"]
    if photo is not None:
        scale = photo.shape[1] / float(data["photo_wh"][0]) if data.get("photo_wh") else 1.0
        image_with_points(ax, photo, pts * scale, camera=sel["index"], linewidth=1.6)
        for coll in ax.collections:
            coll.set_path_effects(_photo_halo(_tok(ax)["accent"], 1.6))
        panel_title(ax, "A", "Points found", f"{sel['name']} · {label} · {len(pts)} points")
    else:
        w, h = data["photo_wh"] or (1, 1)
        image_with_points(ax, None, pts, image_size=(int(w), int(h)), camera=sel["index"])
        panel_title(ax, "A", "Points found (photo not shown)",
                    f"{sel['name']} · {label} · {len(pts)} points")
        ax.text(0.5, -0.04, _wrap(data.get("photo_reason") or "", 60), transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color=_tok(ax)["muted"])
    if level == 1:
        axc = fig.add_subplot(grid[0, 1])
        info = cams[sel["index"]]
        cov = info["coverage"]
        cov_text = f"outline = {cov * 100:.0f} % of the image" if np.isfinite(cov) \
            else "coverage not recorded"
        image = _coverage_axes(axc, info["all_points"], info["frame_wh"],
                               title="Where its points fall",
                               subtitle=f"its {info['n_images']} photos with points · {cov_text}",
                               camera=sel["index"], letter="B")
        image._edu_scale.colourbar(fig, axc, image, shrink=0.7)
        return
    spec = grid[:, 1] if level == 2 else grid[0, 1]
    n = len(cams)
    width, height = fig.get_size_inches()
    sub_w = width * 1.5 / 2.5
    sub_h = height * (1.0 if level == 2 else 0.7) - 0.5
    aspect = np.median([c["frame_wh"][1] / max(c["frame_wh"][0], 1) for c in cams]) if cams else 0.75
    rows, cols = _grid_shape(n, aspect, sub_w * 0.9, sub_h)
    # The header row is tall enough (> 3 % of the figure) that the title aligner never puts
    # the tiles' titles on the header's baseline.
    outer = spec.subgridspec(rows + 1, cols, height_ratios=[0.16] + [1.0] * rows)
    head = fig.add_subplot(outer[0, :])
    head.set_axis_off()
    panel_title(head, "B", "Where each camera's points fall",
                "% = image area its points outline · colour = points per cell")
    vmax = max([float(vis.coverage_grid(c["all_points"], *c["frame_wh"], 14)[0].max())
                for c in cams] + [1.0])
    axes = []
    image = None
    for k, info in enumerate(cams):
        axk = fig.add_subplot(outer[1 + k // cols, k % cols])
        cov = info["coverage"]
        image = _coverage_axes(axk, info["all_points"], info["frame_wh"],
                               title=f"{cov * 100:.0f} %" if np.isfinite(cov) else "–",
                               subtitle=None, camera=info["index"], small=True,
                               highlight=info["index"] == sel["index"], vmax=vmax)
        axes.append(axk)
    if image is not None:
        _shared_colourbar(fig, image, axes, "points")
    if level >= 3:
        _points_per_image(fig.add_subplot(grid[1, :]), data, sel)


def _shared_colourbar(fig, mappable, axes: list, label: str):
    """One slim colour bar for a grid of small maps (the colourbar_px look, UX-35)."""
    from matplotlib.ticker import MaxNLocator
    t = _tok(axes[0])
    mappable._edu_quantity = "amount"
    bar = fig.colorbar(mappable, ax=axes, shrink=0.75, aspect=28, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=TICK_SIZE, colors=t["muted"], labelcolor=t["text"], length=3)
    bar.locator = MaxNLocator(4, integer=True)
    bar.update_ticks()
    # Label on the side, not above: a short grid of maps leaves no room over the bar.
    bar.ax.set_ylabel(label, fontsize=TICK_SIZE, color=t["text"], labelpad=4)
    return bar


def _points_per_image(ax, data, sel) -> None:
    t = _tok(ax)
    counts = np.asarray(data["per_image"], dtype=float)
    if not counts.size:
        empty_panel(ax, "No saved points per photo.")
        return
    column = counts[:, sel["index"]] if counts.ndim == 2 else counts
    x = np.arange(1, len(column) + 1)
    ax.bar(x, column, width=0.85, zorder=3, **_bar_kwargs(sel["index"], ax))
    idx = data["image_index"]
    if 0 <= idx < len(column):
        ax.bar([idx + 1], [column[idx]], color=t["accent"], width=0.85, zorder=4)
    ax.set_xlim(0, len(column) + 1)
    ax.set_ylim(0, max(column.max(), 1) * 1.15)
    ax.set_xlabel("photo")
    ax.set_ylabel("points")
    integer_axis(ax, "y")
    ax.yaxis.grid(True)
    panel_title(ax, "C", "Points in each photo",
                f"{sel['name']} · {data['photo_label']} in the accent colour")


def _bar_kwargs(cam: int, ax) -> dict:
    """One camera's bar: its theme fill, no outline (UX2-1)."""
    kwargs = camera_mark_kwargs(cam, ax, "patch")
    return {"color": kwargs["facecolor"], "edgecolor": kwargs["edgecolor"],
            "linewidth": kwargs["linewidth"]}


# ---------------------------------------------------------------------------------------------
# Stage 6: decision
# ---------------------------------------------------------------------------------------------

def _pct(values) -> list[float]:
    return [v * 100 if np.isfinite(v) else 0.0 for v in values]


def draw_decision(fig, data: dict, level: int) -> None:
    """Per-camera bars in camera colours; a camera Phase 1 stops is ringed in red."""
    cams = data["cameras"]
    flagged = data["flagged"]
    rate_sub = ("share of its photos · dashed = 90 % aim" if level == 1 else
                "share of its photos · dashed = 90 % and 60 %")
    panels = [("rate", "Photos with the target", rate_sub, "%")]
    if data["has_coverage"]:
        panels.append(("coverage", "Image area covered",
                       "outline of all its points · more is better", "%"))
    if level >= 2:
        big = max([v for v in data["points"] if np.isfinite(v)] + [0]) >= 2000
        panels.append(("points", "Points found",
                       "all photos together" + (", in thousands" if big else ""),
                       "k" if big else ""))
        if data["has_completeness"]:
            # D1.3 (setup_reports): per photo, corners found / (corners per face x faces
            # seen), averaged over the photos; so 100 % = every seen face complete.
            sub = ("corners found, of the faces it saw" if data.get("n_faces", 1) > 1
                   else "share of the board, when seen")
            panels.append(("completeness", "Board completeness", sub, "%"))
    if level >= 3:
        grid = fig.add_gridspec(2, len(panels), height_ratios=[1.25, 1.0])
    else:
        grid = fig.add_gridspec(1, len(panels))
    for k, (key, title, subtitle, unit) in enumerate(panels):
        ax = fig.add_subplot(grid[0, k])
        values = data[key]
        if unit == "%":
            shown = _pct(values)
        elif unit == "k":
            shown = [v / 1000.0 if np.isfinite(v) else 0.0 for v in values]
        else:
            shown = [v if np.isfinite(v) else 0.0 for v in values]
        bars_by_camera(ax, {c: v for c, v in zip(range(len(cams)), shown)}, None,
                       flagged=flagged, unit="" if unit == "k" else unit,
                       ylabel="thousand points" if unit == "k" else (
                           "points" if key == "points" else ""),
                       value_format="{:.0f} %" if unit == "%" else (
                           "{:.1f}" if unit == "k" else "{:,.0f}"),
                       highlight=data["selected"], xlabel="Camera")
        if unit == "%":
            ax.set_ylim(0, 118)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
        if unit == "" and key == "points":
            integer_axis(ax, "y")
        if key == "rate":
            # The aim is drawn at every level (ED2-17); L2 adds the summary's lower band.
            # Grade boundaries, not failures: neutral dashes (UX3-11), and the caption names no
            # colour that is not drawn (UX3-12).
            bands = ((90, "aim"),) if level == 1 else ((90, "90 %"), (60, "60 %"))
            style = limit_style(ax, False)
            for limit, label in bands:
                ax.axhline(limit, **style)
                ax.annotate(label, (1.0, limit), xycoords=("axes fraction", "data"),
                            xytext=(3, 0), textcoords="offset points", va="center",
                            fontsize=TICK_SIZE, color=style["color"], annotation_clip=False)
            _lift_value_labels(ax, [limit for limit, _l in bands])
        panel_title(ax, chr(ord("A") + k), title, subtitle)
    if level >= 3:
        _features_matrix(fig.add_subplot(grid[1, :]), data, chr(ord("A") + len(panels)))


def _lift_value_labels(ax, levels, below: float = 9.0, above: float = 3.0) -> None:
    """Bar value labels that would sit on a reference line (bar top from ``below`` under it
    to ``above`` over it, in data units) are written just above the line instead, 6 pt clear
    (UX3-12: camera 6's "61" sat on the 60 % line)."""
    for text in ax.texts:
        if (text.get_gid() or "") != "edu:value" or not hasattr(text, "xy"):
            continue
        x, v = text.xy
        for level in levels:
            if level - below <= v <= level + above:
                text.xy = (x, max(v, level))
                text.xyann = (0, 6)
                break


def _features_matrix(ax, data, letter: str) -> None:
    """Who saw the target in which photo: one row per camera, its colour where it found the
    target, stronger for more points (the encoding Phase 4 uses; UX-22)."""
    from matplotlib.colors import to_rgba
    t = _tok(ax)
    matrix = np.asarray(data["per_image"], dtype=float)
    if not matrix.size:
        empty_panel(ax, "The per-photo counts were not saved for this run.")
        return
    n_ims, n_cams = matrix.shape
    top = max(float(matrix.max()), 1.0)
    rgba = np.zeros((n_cams, n_ims, 4))
    for cam in range(n_cams):
        colour = np.asarray(to_rgba(camera_colour(cam, ax)))
        alpha = np.where(matrix[:, cam] > 0, 0.3 + 0.7 * matrix[:, cam] / top, 0.0)
        rgba[cam, :, :3] = colour[:3]
        rgba[cam, :, 3] = alpha
    ax.imshow(rgba, aspect="auto", interpolation="nearest",
              extent=(0.5, n_ims + 0.5, n_cams - 0.5, -0.5))
    ax.set_facecolor(t["stage"])
    # Badges sized to the row height they will get (the map is ~38 % of the figure).
    row_px = ax.figure.get_size_inches()[1] * ax.figure.dpi * 0.36 / max(n_cams, 1)
    badge_ticks_y(ax, range(n_cams), list(range(n_cams)), highlight=[data["selected"]],
                  flagged=data["flagged"], size=float(np.clip(row_px / 3.1, 6.5, 9.5)))
    idx = data["image_index"]
    if 0 <= idx < n_ims:
        ax.axvline(idx + 1, color=t["text"], linewidth=1.4, linestyle=(0, (3, 2)))
    ax.set_xlabel("photo")
    integer_axis(ax, "x")
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_title(ax, letter, "Who saw the target in which photo",
                f"stronger colour = more points; blank = none · dashed line = "
                f"{photo_label(idx)}")


# ---------------------------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------------------------

def _sentences_within(parts: list[str], level: int) -> str:
    """Join caption parts, dropping trailing optional parts past the level's sentence limit."""
    from pyCamSet.gui.education_style import CAPTION_MAX_SENTENCES, count_sentences
    limit = CAPTION_MAX_SENTENCES.get(int(level), 4)
    text = ""
    for part in parts:
        if not part:
            continue
        candidate = (text + " " + part).strip()
        if count_sentences(candidate) > limit and text:
            continue
        text = candidate
    return text


class Phase1EducationView(EducationStageView):
    """Phase 1 storyboard: six stages, each a figure of this run's own target, photos and
    detections."""

    def __init__(self, context, parent: QWidget | None = None) -> None:
        super().__init__(STAGES, parent, phase_label="Phase 1")
        self.context = context
        self.setAccessibleName(f"{phase_name('phase1')} explanation, run {context.run_id}")
        self.setAccessibleDescription(orientation("phase1", 1))
        metadata = _plain(context.metadata)
        self._params = metadata.get("params") or {}
        self._diagnostics = metadata.get("diagnostics") or {}
        self._detections, self._load_error = _load_run_detections(context)
        self._manifest, self._photo_names, self._mapping_kind = _source_state(context)
        self._trace = self._read_trace()
        camera_names = self._diagnostics.get("cam_names") or (
            list(self._detections.cam_names) if self._detections is not None else [])
        self._cameras = tuple(str(name) for name in camera_names)
        try:
            self._image_count = max(1, int(self._detections.max_ims)) if self._detections else 1
        except (TypeError, ValueError, AttributeError):
            self._image_count = 1
        if self._mapping_kind == "manifest" and self._manifest.status == "verified by manifest":
            # The writer enumerated even photos with no detected target; saved
            # detection max_ims alone can hide a trailing blank photo slot.
            self._image_count = max(self._image_count,
                                    *(len(names) for names in self._photo_names.values()), 1)
        self._per_image = (vis.points_per_image(self._detections)
                           if self._detections is not None else np.zeros((0, 0)))
        # Byte-capped (PYCAMSET_EDU_CACHE_MB): browsing many photos evicts the oldest ones.
        self._photo_cache: ByteLRU = ByteLRU()
        self._size_cache: dict[str, tuple[float, float]] = {}
        self._explain_cache: ByteLRU = ByteLRU()
        # Neighbouring photos are decoded here, off the GUI thread, so a photo change is a hit.
        self._prefetch = PhotoPrefetcher(_decode_display_photo, self, max_workers=2,
                                         thread_name_prefix="edu-p1-photo")
        self._prefetch.ready.connect(self._on_prefetched, Qt.ConnectionType.QueuedConnection)
        # Views are deleted (not closed) when the learner switches runs: stop decodes then too.
        prefetch = self._prefetch
        self.destroyed.connect(lambda *_: prefetch.shutdown())
        self._future: Optional[Future] = None
        self._future_key: Optional[tuple[str, int]] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._target = None
        self._target_error: Optional[str] = None
        self._summary_cache: Optional[tuple] = None
        self._tile_axes: dict = {}
        self._step_index = 0
        #: The detector-step transport bar (the shared ``TransportBar``), built per photo.
        self.transport = None
        self._transport_names: tuple[str, ...] = ()

        # Header pickers: the shared camera and photo pickers (photos numbered from 1).
        self._camera = camera_picker({k: name for k, name in enumerate(self._cameras)})
        self._camera.setAccessibleName("Camera for Phase 1 view")
        self.add_picker("", self._camera)
        busiest = (vis.busiest_image(self._detections, camera=0)
                   if self._detections is not None else 0)
        self._image = photo_picker(range(self._image_count),
                                   current=min(busiest, self._image_count - 1))
        self._image.setAccessibleName("Photo for Phase 1 view")
        self._image.setMaxVisibleItems(16)
        # Older callers (integration tests, tools) drove a spin box of stored indices.
        self._image.setValue = self.set_image_index
        self._image.value = self.image_index
        self.add_picker("", self._image)

        self._poll = QTimer(self)
        self._poll.setInterval(40)
        self._poll.timeout.connect(self._check_future)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(160)
        self._resize_timer.timeout.connect(self._refresh)
        self._last_visual_size = QSize()
        self.visual.installEventFilter(self)

        self._camera.currentIndexChanged.connect(self._selection_changed)
        self._image.currentIndexChanged.connect(self._selection_changed)
        self.stageChanged.connect(self._stage_chosen)
        self.levelChanged.connect(self._level_chosen)
        self.figure()
        self.canvas.mpl_connect("button_press_event", self._figure_clicked)
        self._refresh()
        self._start_explanation()

    # -- compatibility helpers (tests, the dialog) ------------------------------------------

    @property
    def _stage_index(self) -> int:
        return self.current_stage()

    @property
    def _caption(self) -> QLabel:
        return self.caption

    @property
    def _stepbar(self):
        """Older name of :attr:`transport` (review tools)."""
        return self.transport

    def _set_stage(self, index: int) -> None:
        self.set_current_stage(index)

    def _reserve_caption(self) -> None:
        """Measure the caption free of its previous fixed height (QLabel.heightForWidth never
        reports less than the label's minimum height, so the shared reserve only ever grew)."""
        self.caption.setMinimumHeight(0)
        self.caption.setMaximumHeight(16777215)
        super()._reserve_caption()

    def set_chips(self, chips) -> None:
        """Stat chips that never shrink below their text (the shared row may squeeze them)."""
        for frame in list(getattr(self.chips, "_frames", ())):
            frame.hide()  # gone at once, not when the deferred delete runs
        chips = list(chips)
        super().set_chips(chips)
        spacing = self.chips.layout().spacing()

        def needed() -> int:
            frames = list(getattr(self.chips, "_frames", ()))
            return (sum(f.sizeHint().width() for f in frames)
                    + spacing * max(len(frames) - 1, 0)) if frames else 0
        # Keep at least half the band for the caption: drop the least important chips.
        budget = 0.48 * max(self.width(), 1000)
        while len(chips) > 2 and needed() > budget:
            last = chips[-1]
            is_check = isinstance(last, Chip) and ("saved" in last.label)
            chips = (chips[:-2] + [last]) if is_check else chips[:-1]  # the check stays
            for frame in list(getattr(self.chips, "_frames", ())):
                frame.hide()
            super().set_chips(chips)
        self.chips.setMinimumWidth(needed())

    def step_back(self) -> None:
        """Previous stage (the dialog's shared navigation)."""
        self.set_current_stage(self.current_stage() - 1)

    def step_forward(self) -> None:
        """Next stage."""
        self.set_current_stage(self.current_stage() + 1)

    def provenance_text(self) -> str:
        """Where the pictures come from and how far the photos could be checked."""
        status = getattr(self._manifest, "status", "not checkable")
        if getattr(self._manifest, "reason", None) == "Phase 1 manifest missing":
            status = "not checkable (no manifest)"
        return (f"photos: your source images as they are now ({status}) · points: this run's "
                "saved results · steps: a live re-run of Phase 1's detector on the shown photo")

    def _read_trace(self) -> dict[str, Any]:
        from pyCamSet.workflow.education_trace import read_trace
        return read_trace(self.context.run_dir, self.context.run_id, "phase1")

    def _trace_stages_for_selection(self, camera: str, image_index: int):
        """Return only the exact camera/image record; never borrow a neighbour."""
        records = self._trace.get("records", [])
        matches = [record.get("payload", {}) for record in records
                   if isinstance(record, Mapping)
                   and record.get("kind") == "stage"
                   and isinstance(record.get("payload"), Mapping)
                   and record["payload"].get("camera") == camera
                   and type(record["payload"].get("image_index")) is int
                   and record["payload"].get("image_index") == image_index]
        if self._trace.get("status") != "available":
            return [], str(self._trace.get("reason") or "This run recorded no detector stages.")
        if not matches:
            return [], "No recorded index exists for this camera and image; another record was not substituted."
        stages = matches[0].get("stages")
        if not isinstance(stages, list):
            return [], "The matching trace record has no usable stage index."
        usable = [item for item in stages if isinstance(item, Mapping) and isinstance(item.get("name"), str)]
        if not usable:
            return [], "This camera and image has no recorded detector stages."
        return usable, None

    # -- selection ------------------------------------------------------------------------

    def camera_index(self) -> int:
        return max(self._camera.currentIndex(), 0)

    def camera_name(self) -> str:
        return self._cameras[self.camera_index()] if self._cameras else ""

    def image_index(self) -> int:
        """The selected photo's stored (0-based) index."""
        data = self._image.currentData()
        return int(data) if data is not None else 0

    def set_image_index(self, index: int) -> None:
        """Select the photo with stored (0-based) ``index``."""
        found = self._image.findData(int(index))
        if found >= 0:
            self._image.setCurrentIndex(found)

    def photo_text(self) -> str:
        """The selected photo as the learner reads it, e.g. "Photo 11"."""
        return photo_label(self.image_index())

    def _selection_changed(self, *_args) -> None:
        if self.transport is not None:
            self.transport.stop_playback()
        self._start_explanation()
        self._refresh()

    def _stage_chosen(self, _index: int) -> None:
        self._refresh()

    def _level_chosen(self, level: int) -> None:
        self.setAccessibleDescription(orientation("phase1", level))
        self._refresh()

    def _step_chosen(self, index: int) -> None:
        self._step_index = int(index)
        if self.current_stage() == STAGE["find"]:
            self._refresh()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API
        if watched is self.visual and event.type() == QEvent.Type.Resize:
            size = event.size()
            old = self._last_visual_size
            if (not old.isValid() or abs(size.width() - old.width()) > 40
                    or abs(size.height() - old.height()) > 40):
                self._last_visual_size = QSize(size)
                self._resize_timer.start()
        return super().eventFilter(watched, event)

    def _figure_clicked(self, event) -> None:
        if self.current_stage() != STAGE["photos"] or event.inaxes is None:
            return
        k = self._tile_axes.get(event.inaxes)
        if k is not None and k != self.camera_index():
            self._camera.setCurrentIndex(k)

    # -- photos ---------------------------------------------------------------------------

    def _photo_path_for(self, camera: str, image_index: int) -> tuple[Optional[Path], Optional[str]]:
        """The checked photo of one camera + image, or None and a short reason."""
        if self._mapping_kind in _PHOTOLESS_COPY_KEYS:
            return None, _PHOTOLESS_SHORT[self._mapping_kind]
        names = self._photo_names.get(camera, ())
        root = self._params.get("f_loc")
        if not root:
            return None, "Photo unavailable: the run does not record its image folder."
        if image_index >= len(names):
            return None, "Photo unavailable: no photo is recorded for this photo number."
        photo = _photo_path(Path(str(root)), camera, names[image_index],
                            self._params.get("subfolder_string"))
        if self._mapping_kind == "manifest" and self._manifest.status != "verified by manifest":
            if photo.is_file():
                return None, f"Photo not shown: {photo.name} does not match the saved image manifest."
            return None, f"Photo unavailable: {photo.name} is no longer in the folder."
        if not photo.is_file():
            return None, f"Photo unavailable: {photo.name} is no longer in the folder."
        return photo, None

    def _photo(self, camera: str, image_index: int) -> tuple[Optional[np.ndarray], Optional[str]]:
        key = (camera, image_index)
        if key in self._photo_cache:
            return self._photo_cache[key]
        path, reason = self._photo_path_for(camera, image_index)
        if path is None:
            entry: tuple[Optional[np.ndarray], Optional[str]] = (None, reason)
        else:
            # A neighbour's background decode is usually done by now; otherwise this decodes.
            decoded = self._prefetch.take(key)
            if not isinstance(decoded, tuple):
                decoded = _decode_display_photo(path)
            image, decode_reason, size = decoded
            entry = (image, decode_reason)
            self._remember_size(camera, size)
        self._photo_cache[key] = entry
        return entry

    def _remember_size(self, camera: str, size: Optional[tuple]) -> None:
        if size and camera not in self._size_cache:
            self._size_cache[camera] = (float(size[0]), float(size[1]))

    def _on_prefetched(self, key: tuple[str, int], value: Any) -> None:
        """A background decode finished (GUI thread, queued). Keep it unless already cached."""
        self._prefetch.forget(key)
        if key in self._photo_cache or not isinstance(value, tuple):
            return
        image, reason, size = value
        self._photo_cache[key] = (image, reason)
        self._remember_size(key[0], size)

    def _prefetch_neighbours(self, image_index: int) -> None:
        """Start decoding the next and previous photo of every camera off the GUI thread."""
        for neighbour in (image_index + 1, image_index - 1):
            if not 0 <= neighbour < self._image_count:
                continue
            for name in self._cameras:
                key = (name, neighbour)
                if key in self._photo_cache:
                    continue
                path, _reason = self._photo_path_for(name, neighbour)
                if path is not None:
                    self._prefetch.request(key, path)

    def _photo_wh(self, camera: str) -> Optional[tuple[float, float]]:
        if camera not in self._size_cache:
            path, _ = self._photo_path_for(camera, self.image_index())
            size = vis.photo_size(path) if path is not None else None
            if size:
                self._size_cache[camera] = (float(size[0]), float(size[1]))
        return self._size_cache.get(camera)

    def _frame_wh(self, camera: str, points: np.ndarray) -> tuple[float, float]:
        """Detection-frame size of a camera's photos (photo size / photo-per-frame)."""
        wh = self._photo_wh(camera)
        if wh is not None:
            factor = _frame_to_photo(self._params)
            return wh[0] / factor, wh[1] / factor
        if len(points):
            return float(np.ceil(points[:, 0].max() * 1.05)), float(np.ceil(points[:, 1].max() * 1.05))
        return 1.0, 1.0

    # -- live explanation -------------------------------------------------------------------

    def _explain_target(self):
        if self._target is None and self._target_error is None:
            try:
                from pyCamSet.workflow.targets import target_of_params
                self._target = target_of_params(self._params)
            except Exception as exc:  # noqa: BLE001 - shown to the learner as a reason
                self._target_error = f"the run's target could not be rebuilt ({exc})"
        return self._target

    def _summary(self):
        """The target's faces and who saw which face (cached; None when not rebuildable)."""
        if self._summary_cache is None:
            target = self._explain_target() if self._detections is not None else None
            try:
                summary = vis.target_summary(target, self._detections) if target is not None \
                    else None
            except Exception:  # noqa: BLE001 - the stage then explains it cannot draw
                summary = None
            self._summary_cache = (summary,)
        return self._summary_cache[0]

    def _start_explanation(self) -> None:
        """Re-run the detector on the selected photo in the background (cached per photo)."""
        if self._detections is None or not self._cameras:
            return
        key = (self.camera_name(), self.image_index())
        if key in self._explain_cache or (self._future_key == key and self._future is not None):
            return
        path, reason = self._photo_path_for(*key)
        if path is None:
            self._explain_cache[key] = RuntimeError(reason or "Photo unavailable.")
            return
        target = self._explain_target()
        if target is None:
            self._explain_cache[key] = RuntimeError(self._target_error or "target unavailable")
            return
        from pyCamSet.calibration_targets.core.preparation import resolve_preparation
        from pyCamSet.workflow.education_detect_explain import (
            explain_detection, saved_image_detection,
        )
        params = self._params
        saved = saved_image_detection(self._detections, key[0], key[1])
        # Phase 1's own rule for this target: None leaves explain_detection to take the
        # target's defaults for anything the run left blank.
        try:
            prep = resolve_preparation(params, target)
        except (TypeError, ValueError):
            prep = None
        kwargs = dict(
            rescale_and_gamma=prep is not None,
            preprocessing_scale=None if prep is None else prep.scale,
            preprocessing_gamma=None if prep is None else prep.gamma,
            preprocessing_loader=None if prep is None else prep.loader,
            saved_detection=saved, upscale_factor=int(params.get("upscale_factor") or 1))

        def job():
            _below_normal_thread()
            return explain_detection(path, target, **kwargs)

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="edu-p1-explain")
        if self._future is not None and not self._future.done():
            self._future.cancel()
        self._future_key = key
        self._future = self._executor.submit(job)
        self._poll.start()

    def _check_future(self) -> None:
        future, key = self._future, self._future_key
        if future is None or not future.done():
            return
        self._poll.stop()
        self._future, self._future_key = None, None
        try:
            self._explain_cache[key] = future.result()
        except Exception as exc:  # noqa: BLE001 - shown to the learner
            self._explain_cache[key] = exc
        if key == (self.camera_name(), self.image_index()) and \
                self.current_stage() in (STAGE["prep"], STAGE["find"], STAGE["final"]):
            self._refresh()
        self._start_explanation()

    def wait_for_explanation(self, timeout: float = 30.0) -> Any:
        """Block until the selected photo's explanation is ready (tests, screenshots)."""
        self._start_explanation()
        if self._future is not None:
            try:
                self._future.result(timeout=timeout)
            except Exception:  # noqa: BLE001 - kept in the cache by _check_future
                pass
            self._check_future()
        return self._explain_cache.get((self.camera_name(), self.image_index()))

    def explanation(self) -> Any:
        """The cached explanation of the selected photo (None while it runs)."""
        return self._explain_cache.get((self.camera_name(), self.image_index()))

    def stop_for_context_change(self) -> None:
        if self.transport is not None:
            self.transport.stop_playback()
        super().stop_for_context_change()

    def close_background(self) -> None:
        """Stop the re-detection poll, its executor and the photo prefetch (the dialog calls
        this when it closes; it never closes its child views)."""
        if self.transport is not None:
            self.transport.stop_playback()
        self._poll.stop()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._prefetch.shutdown()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.close_background()
        super().closeEvent(event)

    # -- the detector-step transport bar ------------------------------------------------------

    def _ensure_transport(self, steps) -> Any:
        """The shared TransportBar for these detector steps (rebuilt when the steps change)."""
        from pyCamSet.gui.education_dialog import TransportBar
        names = tuple(vis.short_name(s.key, s.title) for s in steps)
        titles = tuple(s.title for s in steps)
        if self.transport is None or names != self._transport_names:
            old = self.transport
            bar = TransportBar(
                len(names), self, names=names, unit_label="Detector steps", interval_ms=1600,
                min_interval_ms=400, start=min(self._step_index, len(names) - 1),
                describe=lambda i: f"Detector step {i + 1} of {len(titles)}: {titles[i]}.",
                short=lambda i: f"{i + 1} / {len(titles)}", play_name="Play the detector steps")
            bar.positionChanged.connect(self._step_chosen)
            self.transport = bar
            self._transport_names = names
            if old is not None:
                old.stop_playback()
                if self._controls is old:
                    self.set_controls(bar)
                old.hide()
                old.deleteLater()
        return self.transport

    # -- drawing ----------------------------------------------------------------------------

    def _refresh(self, *_args) -> None:
        self.set_busy(None)
        stage = self.current_stage()
        if stage != STAGE["find"] or self._detections is None:
            self.set_controls(None)
        if self._detections is None:
            self._show_no_detections()
            return
        [self._show_target, self._show_photos, self._show_prep, self._show_find,
         self._show_final, self._show_decision][stage]()
        self.visual_panel.setToolTip(self.provenance_text())
        if self.canvas is not None:
            self.canvas.setAccessibleName(
                f"Phase 1 {STAGES[stage]} view for camera {self.camera_name()}, "
                f"{self.photo_text()}")
            self.canvas.setAccessibleDescription(self.caption.text())

    def _show_no_detections(self) -> None:
        reason = self._load_error or "this run's detections were not saved"
        self.draw_figure(lambda fig, _theme: empty_panel(
            fig.add_subplot(), "No run-private detection output is available.\n" + reason))
        self.set_caption(f"Final unavailable: {reason}. Phase 1 explains a run from its own saved "
                         "detections only; no cache or other run is substituted.")
        self.set_chips([])
        self.set_numbers([], [])

    def _busy(self, what: str) -> None:
        camera = self.camera_name()
        photo = self.photo_text()

        def drawer(fig, _theme):
            empty_panel(fig.add_subplot(),
                        f"Re-running Phase 1's detector on {camera}, {photo} …")
        self.draw_figure(drawer)
        self.set_caption(what)
        self.set_chips([])
        self.set_busy(f"Re-running the detector on {camera}, {photo}")

    def _explanation_or_reason(self):
        """(explanation, None) when ready, (None, reason) when unavailable, (None, None) busy."""
        result = self.explanation()
        if result is None:
            return None, None
        if isinstance(result, Exception):
            return None, str(result)
        return result, None

    def _unavailable(self, reason: str, stage_word: str) -> None:
        mapping_copy = (_reason_caption(self._mapping_kind)
                        if self._mapping_kind in _PHOTOLESS_COPY_KEYS else "")

        def drawer(fig, _theme):
            empty_panel(fig.add_subplot(), _wrap(reason, 70))
        self.draw_figure(drawer)
        self.set_caption(mapping_copy or f"{stage_word} needs this camera's photo. {reason}",
                         lines=None)
        self.set_chips([])
        self.set_numbers([], [])

    def _camera_rows(self, camera: str, image_index: int) -> np.ndarray:
        """Saved detection rows (camera, photo, key..., x, y) of one camera and photo."""
        det = self._detections
        if det is None or camera not in list(getattr(det, "cam_names", [])):
            return np.zeros((0, 4))
        data = det.get_data()
        if data is None or not len(data):
            return np.zeros((0, 4))
        cam = list(det.cam_names).index(camera)
        return data[(data[:, 0].astype(int) == cam) & (data[:, 1].astype(int) == int(image_index))]

    # Stage 1: target ---------------------------------------------------------------------

    def _target_data(self) -> dict:
        summary = self._summary()
        cam, name, idx = self.camera_index(), self.camera_name(), self.image_index()
        photo, reason = self._photo(name, idx) if self._cameras else (None, "no camera")
        rows = self._camera_rows(name, idx)
        pts = np.asarray(rows[:, -2:], dtype=float).reshape(-1, 2) if len(rows) else np.zeros((0, 2))
        n_faces = len(summary.faces) if summary is not None else 1
        faces = vis.point_faces(rows[:, 2:-2], n_faces) if len(rows) else np.zeros(0, dtype=int)
        if photo is not None:
            wh = self._photo_wh(name)
            scale = photo.shape[1] / wh[0] if wh else 1.0
            pts = pts * _frame_to_photo(self._params) * scale
        return dict(summary=summary, camera=cam, camera_name=name, photo=photo,
                    photo_reason=reason, points=pts, point_faces=faces,
                    photo_label=self.photo_text())

    def _show_target(self) -> None:
        level = self.level()
        data = self._target_data()
        self.draw_figure(lambda fig, _theme: draw_target(fig, data, level))
        summary = data["summary"]
        if summary is None:
            reason = self._target_error or "this run's target could not be rebuilt"
            self.set_caption(f"{reason[:1].upper() + reason[1:]}.")
            self.set_chips([])
            self.set_numbers([], [])
            return
        n_faces = len(summary.faces)
        cam = data["camera"] + 1
        # Line-limited captions (UX2-5): L1 <= 2 rendered lines at 1280 px, L2 3, L3 4.
        parts = ["A calibration target's pattern is known exactly, so every corner found has "
                 "a known place on it (A)."]
        if n_faces > 1:
            shape = "cube" if n_faces == 6 else "target"
            parts.append(f"This {n_faces}-faced {shape} lets cameras all around see shared "
                         f"faces at once (B, camera {cam}).")
        else:
            parts.append(f"This flat board must face every camera (B, camera {cam}).")
        if level >= 2:
            parts.append("C counts the photos in which each camera saw each face; cameras "
                         "sharing a face are tied together later." if n_faces > 1 else
                         "C counts the photos in which each camera saw the board.")
        if level >= 3:
            parts.append("Each marker carries its own ID, so the detector knows which corner "
                         "is which" + (" (D)." if summary.textures and any(
                             x is not None for x in summary.textures) else "."))
        self.set_caption(_sentences_within(parts, level))
        corners = sum(len(f) for f in summary.faces)
        chips = [Chip(str(n_faces), "faces" if n_faces > 1 else "flat board", tone="accent"),
                 Chip(format_count(corners), "corners on the target"),
                 Chip(f"{format_number(summary.size_mm, 0 if summary.size_mm >= 10 else 1)} mm",
                      "cube edge" if n_faces > 1 else "across"),
                 Chip(str(len(data["points"])), f"found by camera {cam}, {data['photo_label']}",
                      camera=data["camera"])]
        if np.isfinite(summary.spacing_mm):
            chips.insert(3, Chip(f"{format_number(summary.spacing_mm)} mm", "corner spacing"))
        self.set_chips(chips[:4])
        rows = []
        for face, pts in enumerate(summary.faces):
            seen = summary.seen[:, face] if summary.seen.size else []
            rows.append([face_label(face), len(pts), ", ".join(
                f"{k + 1}: {int(n)}" for k, n in enumerate(seen) if n) or "–"])
        self.set_numbers(["Face", "Design corners", "Photos per camera (camera: photos)"], rows)

    # Stage 2: photos -----------------------------------------------------------------------

    def _tiles(self) -> list[dict]:
        idx = self.image_index()
        to_photo = _frame_to_photo(self._params)
        tiles = []
        for k, name in enumerate(self._cameras):
            image, reason = self._photo(name, idx)
            pts = vis.camera_points(self._detections, name, idx)
            scale = 1.0
            if image is not None:
                wh = self._photo_wh(name)
                scale = image.shape[1] / wh[0] if wh else 1.0
            names = self._photo_names.get(name, ())
            tiles.append(dict(index=k, name=name, image=image, reason=reason, count=len(pts),
                              points=pts * to_photo, scale=scale,
                              file=names[idx] if idx < len(names) else ""))
        self._prefetch_neighbours(idx)
        return tiles

    def _show_photos(self) -> None:
        level = self.level()
        tiles = self._tiles()
        selected = self.camera_index()
        self._tile_axes = {}

        def drawer(fig, _theme):
            self._tile_axes.clear()
            draw_photos(fig, tiles, selected, level, self._tile_axes)
        self.draw_figure(drawer)
        idx = self.image_index()
        found = sum(1 for t in tiles if t["count"])
        caption = (f"Each tile is one camera's photo number {photo_label(idx, short=True)}. Under "
                   "each name is how many target points Phase 1 found in it; click a photo to "
                   "follow that camera.")
        if level >= 2:
            caption += " Circles mark the saved points."
        if level >= 3:
            caption += " Photos are shown as Phase 1 read them (8-bit), with the run's file names."
        if self._mapping_kind in _PHOTOLESS_COPY_KEYS:
            caption = _reason_caption(self._mapping_kind)
        elif self._mapping_kind == "manifest" and self._manifest.status != "verified by manifest":
            caption += " Photos that do not match the run's saved photo list are not shown."
        self.set_caption(caption)
        total = sum(t["count"] for t in tiles)
        self.set_chips([
            Chip(f"{found} / {len(tiles)}", "cameras found it", tone="good" if found == len(tiles) else "warn"),
            Chip(f"{total:,}", "points, this photo", tone="accent"),
            Chip(str(tiles[selected]["count"]) if tiles else "0", f"camera {selected + 1}",
                 camera=selected),
            Chip(f"{photo_label(idx, short=True)} / {self._image_count}", "photo"),
        ])
        self.set_numbers(["Camera", "Photo file", "index (0-based)", "Points (this photo)",
                          "Points (all photos)"],
                         [[t["name"], t["file"] or "–", idx, t["count"],
                           int(self._per_image[:, t["index"]].sum()) if self._per_image.size else 0]
                          for t in tiles])

    # Stage 3: what the detector sees ---------------------------------------------------------

    def _show_prep(self) -> None:
        explanation, reason = self._explanation_or_reason()
        if explanation is None and reason is None:
            self._busy("Re-running the detector on this photo to show what it sees.")
            return
        if explanation is None:
            self._unavailable(reason, "This stage")
            return
        level = self.level()
        self.draw_figure(lambda fig, _theme: draw_preprocessing(fig, explanation, level))
        demo = vis.threshold_demo(explanation)
        threshold = explanation.step("threshold")
        rescale = explanation.step("rescale_gamma")
        if demo is None:
            parts = ["The photo is shrunk, brightened (gamma) and lightly blurred. The corner "
                     "finder works on this smooth copy."]
            if level >= 2:
                parts.append("The close-ups follow one spot of the board through each step.")
        else:
            # SCI2-13: count only the dim half's *light* pixels (the light squares); dark
            # pixels turn black under both rules.
            lost = demo.dim_light_lost
            first = (f"One cut-off for the whole photo (B) blackens {lost * 100:.0f} % of the "
                     "light squares in its dim half." if lost >= 0.5 else
                     "One cut-off for the whole photo (B) treats bright and dim parts alike.")
            parts = [first, "The detector compares each pixel with its own square (A), so every "
                            "face keeps its pattern (C)."]
            if level == 2:
                parts.append("The close-ups follow one pixel through both rules.")
            if level >= 3:
                n_other = min(max(len(getattr(threshold, "images", ()) or ()) - 1, 0), 2)
                parts.append("D and E apply both rules to every pixel" + (
                    _inverted_copy_sentence(threshold, demo)
                    if n_other == 1 and _other_is_inverted(threshold) else
                    "; F is a second image the detector also traces." if n_other == 1 else
                    "; F and G are other images the detector also traces." if n_other == 2
                    else "."))
        self.set_caption(_sentences_within(parts, level))
        chips = []
        first = explanation.step("input")
        if demo is not None:
            # The photo size and bit depth are in A's subtitle; the chip says what B shows
            # (UX3-16).
            chips.append(Chip(f"{demo.dim_light_lost * 100:.0f} %",
                              "dim-half light squares one cut-off blackens",
                              tooltip="Of the light squares in the photo's dim half, the share "
                                      "a single cut-off for the whole photo (B) turns black."))
        elif first is not None:
            chips.append(Chip(f"{first.counts.get('width px')}×{first.counts.get('height px')}",
                              "photo px"))
            depth = first.counts.get("bit depth")
            chips.append(Chip(f"{depth}-bit" if isinstance(depth, (int, np.integer)) else str(depth),
                              "grey file" if _is_grey_photo(explanation) else "colour file"))
        if rescale is not None:
            chips.append(Chip(f"×{vis._fmt(rescale.counts.get('scale'))}", "shrunk to", tone="accent"))
            chips.append(Chip(vis._fmt(rescale.counts.get("gamma")), "gamma", tone="accent"))
        elif demo is not None:
            chips.append(Chip(f"{demo.window_prepared} px", "neighbourhood square", tone="accent"))
            rule = "≥" if threshold is not None and threshold.extra.get("rule") == "opencv" \
                else ">"
            chips.append(Chip(f"{rule} {demo.constant:g}", "darker by, to turn black"))
        self.set_chips(chips[:4])
        rows = []
        for step in vis.split_steps(explanation).prep:
            rows.append([step.title, ", ".join(f"{k}: {vis._fmt(v)}" for k, v in step.counts.items()),
                         step.source])
        if demo is not None:
            rows.append(["Global cut-off (comparison, Otsu)", f"grey {demo.cutoff:.0f}", "drawn here"])
            rows.append(["Example pixel", f"x {demo.probe[0]}, y {demo.probe[1]}: grey "
                         f"{demo.centre_value}, square mean {demo.mean_value:.1f}", "drawn here"])
        if threshold is not None and threshold.extra.get("window_sizes"):
            rows.append(["Neighbourhoods tried", ", ".join(
                f"{w} px" for w in threshold.extra["window_sizes"]), threshold.source])
        if demo is not None:
            rows.append(["Dim half: light squares a single cut-off blackens",
                         f"{demo.dim_light_lost * 100:.0f} %", "drawn here"])
        rows.append(["What the detector searched", _detector_input_copy(self._params),
                     "saved settings"])
        self.set_numbers(["Step", "Values", "Source"], rows)

    # Stage 4: finding markers ------------------------------------------------------------------

    def _show_find(self) -> None:
        explanation, reason = self._explanation_or_reason()
        if explanation is None and reason is None:
            self.set_controls(None)
            self._busy("Re-running the detector on this photo, step by step.")
            return
        if explanation is None:
            self.set_controls(None)
            self._unavailable(reason, "Finding markers")
            return
        steps = vis.split_steps(explanation).find
        if not steps:
            self.set_controls(None)
            self._unavailable(explanation.error or "The detector reported no steps.", "This stage")
            return
        self._step_index = min(max(self._step_index, 0), len(steps) - 1)
        bar = self._ensure_transport(steps)
        if bar.position() != self._step_index:
            bar.blockSignals(True)
            bar.go_to(self._step_index)
            bar.blockSignals(False)
        self.set_controls(bar)
        step = steps[self._step_index]
        level = self.level()
        summary = self._summary()
        n_faces = len(summary.faces) if summary is not None else 1
        # Foreshortening-free scale from this photo's faces (SCI2-12); NaN = no µm figure.
        um = vis.um_per_px(explanation.image_points, keys=explanation.keys,
                           design=summary.design) if summary is not None else float("nan")
        context = dict(um_per_px=um, n_faces=n_faces,
                       face_colours=face_colours(self.theme(), max(n_faces, 1)))
        self.draw_figure(lambda fig, _theme: draw_find_step(
            fig, explanation, step, level, dict(context, face_colours=face_colours(
                _theme, max(n_faces, 1)))))
        caption = vis.step_caption(step, level, consistency=explanation.consistency,
                                   backend=explanation.backend)
        if explanation.error:
            caption = (f"The detector stopped on this photo ({explanation.error}); Phase 1 keeps no "
                       "points from it. " + caption)
        self.set_caption(caption)
        tips = {
            "read, not kept": "The reproduction reads an ID here, but Phase 1's board finder did "
                              "not keep a marker at this place (for example a lone marker it "
                              "could not place on a face).",
            "bad border": "The outer ring of cells is not one solid colour, so this is not a "
                          "marker.",
            "unknown code": "The inner cells match none of this target's markers.",
            "too faint": "Too little contrast inside the square to read its cells.",
            "dropped": "Square candidates not read as markers, plus four-cornered outlines that "
                       "are not square-shaped.",
            "black-and-white images": "The detector traces every one of these images (dark "
                                      "and light markers, or several neighbourhood sizes); "
                                      "the picture shows one of them.",
        }
        chips = [Chip(v, lab, tone=tone, tooltip=tips.get(lab, ""))
                 for v, lab, tone in vis.step_chips(step, limit=3)]
        if step.key == "subpixel":
            median = float(step.counts.get("median move px", 0) or 0)
            chips = [Chip(f"{median:.2f} px", "median move", tone="accent")]
            if np.isfinite(um):
                chips.append(Chip(
                    f"≈ {format_number(median * um, 0)} µm", "at the target's scale",
                    tooltip=f"One pixel ≈ {format_number(um, 1)} µm on the target: the printed "
                            "corner grid fitted to this photo's corners along the face's "
                            "untilted direction (a tilted face is shortened only across it). "
                            "Phase 2 measures the lens scale itself."))
            chips.append(Chip(f"{step.counts.get('moved', 0)} / {step.counts.get('corners', 0)}",
                              "corners moved"))
        check = vis.consistency_chip(explanation.consistency)
        if check is not None:
            value, label, tone, tip = check
            chips.append(Chip(value, label, tone=tone, tooltip=tip))
        self.set_chips(chips)
        rows = [[k, vis._fmt(v)] for k, v in step.counts.items()]
        if step.key == "decoding":
            for key, label, _tag in vis.DECODE_CATEGORIES:
                rows.append([f"squares: {label}", vis.decode_counts(step)[key]])
        rows.append(["source", step.source])
        if step.source in ("reproduced", "mixed") and str(explanation.backend).startswith("aruco2"):
            rows.append(["picture", "redrawn with the OpenCV operations aruco2 uses, without "
                                    "its white pass's diagonal cuts"])
        rows.append(["check", explanation.consistency.message])
        recorded, _why = self._trace_stages_for_selection(self.camera_name(), self.image_index())
        for item in recorded:
            ids = item.get("marker_ids") or item.get("corner_ids")
            if isinstance(ids, list):
                rows.append([f"recorded: {item.get('name')}", f"{len(ids)} IDs"])
        self.set_numbers(["What", "Value"], rows)

    # Stage 5: final ------------------------------------------------------------------------

    def _final_data(self) -> dict:
        idx = self.image_index()
        sel = self.camera_index()
        name = self.camera_name()
        to_photo = _frame_to_photo(self._params)
        coverage = vis.finite_mapping(self._diagnostics.get("D1.6_spatial_coverage"), self._cameras)
        cams = []
        for k, cam in enumerate(self._cameras):
            all_pts = vis.camera_points(self._detections, cam)
            seen = int(np.count_nonzero(self._per_image[:, k])) if self._per_image.size else 0
            cams.append(dict(index=k, name=cam, all_points=all_pts,
                             frame_wh=self._frame_wh(cam, all_pts), coverage=coverage[k],
                             n_images=seen))
        photo, reason = self._photo(name, idx)
        wh = self._photo_wh(name)
        pts = vis.camera_points(self._detections, name, idx)
        if wh is None:
            fw, fh = cams[sel]["frame_wh"] if cams else (1.0, 1.0)
            wh = (fw * to_photo, fh * to_photo)
        return dict(cameras=cams, selected=dict(index=sel, name=name), image_index=idx,
                    photo_label=photo_label(idx), photo=photo, photo_reason=reason, photo_wh=wh,
                    points=pts, to_photo=to_photo, per_image=self._per_image)

    def _show_final(self) -> None:
        level = self.level()
        data = self._final_data()
        self.draw_figure(lambda fig, _theme: draw_final(fig, data, level))
        caption = ("Circles are the points Phase 1 saved for this photo (A). B adds up all of "
                   "this camera's photos; its outline is the image area the points cover.")
        if level >= 2:
            caption = ("Circles are the points Phase 1 saved for this photo. Small maps add up "
                       "one camera's photos; % is the area its points cover.")
        if level >= 3:
            caption += " Bars: points per photo."
        if data["photo"] is None and self._mapping_kind in _PHOTOLESS_COPY_KEYS:
            caption = _reason_caption(self._mapping_kind)
        elif data["photo"] is None and data.get("photo_reason"):
            caption = (f"{data['photo_reason']} The saved points are drawn without it, paired with "
                       "the photo name saved when detection ran; no replacement photo is substituted.")
        self.set_caption(caption)
        sel = data["selected"]["index"]
        cov = data["cameras"][sel]["coverage"] if data["cameras"] else float("nan")
        chips = [Chip(str(len(data["points"])), "points, this photo", camera=sel),
                 Chip(f"{cov * 100:.0f} %" if np.isfinite(cov) else "–", "area covered", camera=sel)]
        if self._cameras:
            seen = data["cameras"][sel]["n_images"]
            share = seen / max(self._image_count, 1)
            # SCI5-7: name the camera, since the count is that camera's alone
            chips.append(Chip(f"{seen} of {self._image_count}",
                              f"photos with target, cam {sel + 1} ({share * 100:.0f} %)",
                              tone=vis.rate_band(share)))
        explanation = self.explanation()
        if explanation is not None and not isinstance(explanation, Exception):
            check = vis.consistency_chip(explanation.consistency)
            if check is not None:
                chips.append(Chip(check[0], check[1], tone=check[2], tooltip=check[3]))
        self.set_chips(chips)
        self.set_numbers(["Camera", "Points (all photos)", "Photos with points", "Area covered"],
                         [[c["name"], len(c["all_points"]), c["n_images"],
                           f"{c['coverage'] * 100:.1f} %" if np.isfinite(c["coverage"]) else "–"]
                          for c in data["cameras"]])

    # Stage 6: decision ---------------------------------------------------------------------

    def _decision_data(self) -> dict:
        d = self._diagnostics
        cams = list(self._cameras)
        totals = vis.finite_mapping(d.get("D1.1_total_detections"), cams)
        if not any(np.isfinite(totals)) and self._per_image.size:
            totals = [float(v) for v in self._per_image.sum(axis=0)]
        rate = vis.finite_mapping(d.get("D1.2_detection_rate"), cams)
        completeness = vis.finite_mapping(d.get("D1.3_board_completeness"), cams)
        coverage = vis.finite_mapping(d.get("D1.6_spatial_coverage"), cams)
        flagged = [k for k, v in enumerate(totals) if np.isfinite(v) and v == 0]
        summary = self._summary()
        return dict(cameras=cams, rate=rate, coverage=coverage, points=totals,
                    completeness=completeness, flagged=flagged,
                    has_coverage=any(np.isfinite(coverage)),
                    has_completeness=any(np.isfinite(completeness)),
                    per_image=self._per_image, image_index=self.image_index(),
                    selected=self.camera_index(),
                    n_faces=len(summary.faces) if summary is not None else 1)

    def _show_decision(self) -> None:
        level = self.level()
        data = self._decision_data()
        self.draw_figure(lambda fig, _theme: draw_decision(fig, data, level))
        # ED7-3: no "never found" line; a stopped camera (no points) is named, the rest is the
        # aim and the point count, in the same units as the chart and the chips.
        parts = (["A camera with no points is stopped (ringed in red)."] if data["flagged"]
                 else [])
        parts.append(_aim_sentence(data))
        if level >= 2:
            parts.append("Dashed lines: the report's grading boundaries, 90 % and 60 % of "
                         "photos.")
        if level >= 3:
            parts.append(f"The map below: who found the target in each photo (dashed: "
                         f"{self.photo_text()}).")
            if not (self._diagnostics.get("education_manifest") or {}).get("recordable"):
                parts.append("This older run did not record its photo list.")
        self.set_caption(_sentences_within(parts, level))
        cams = data["cameras"]
        kept = len(cams) - len(data["flagged"])
        rates = [(r, c) for r, c in zip(data["rate"], cams) if np.isfinite(r)]
        # ED5-4: green only when every camera meets the 90 % aim; a kept camera below it is amber
        rule = _aim_rule(data)
        n_below, least, n_rated = rule[:3] if rule is not None else (0, 0, 0)
        if data["flagged"]:
            keep_tone, keep_label = "bad", "cameras kept"
        elif n_below and least < KEEP_MIN_POINTS:
            # ED6-3: the same rule as the caption; below aim with too few points is a re-shoot
            keep_tone, keep_label = "bad", "re-shoot: few points"
        elif n_below:
            # short labels: the chip row shares its width with the caption
            keep_tone = "warn"
            keep_label = "kept, none at aim" if n_below == len(rates) else "kept, below aim"
        else:
            keep_tone, keep_label = "good", "cameras kept"
        chips = [Chip(f"{kept} / {len(cams)}", keep_label, tone=keep_tone)]
        if keep_label == "kept, below aim":
            # round 9: the chip counts the below-aim cameras, as its caption does (6 / 8, not 8 / 8)
            chips[0] = Chip(f"{n_below} / {n_rated}", "below aim, kept", tone=keep_tone)
        if keep_label == "re-shoot: few points":
            # ED7-3 / SCI7-8: the number counts the cameras below aim (the reason for the
            # re-shoot), so it matches the label instead of "8 / 8" kept
            chips[0] = Chip(f"{n_below} / {n_rated}", "below aim: re-shoot", tone=keep_tone)
        if rates:
            low, who = min(rates)
            k = cams.index(who)
            # ED8-1: the tile is a share of photos (%); the caption's count is corners found
            chips.append(Chip(f"{low * 100:.0f} %", f"fewest photos: camera {k + 1}",
                              camera=k))
        covs = [c for c in data["coverage"] if np.isfinite(c)]
        if covs:
            chips.append(Chip(f"{np.median(covs) * 100:.0f} %", "median area covered"))
        total = sum(v for v in data["points"] if np.isfinite(v))
        chips.append(Chip(f"{int(total):,}", "points in total", tone="accent"))
        self.set_chips(chips)

        def pct(v):
            return f"{v * 100:.0f} %" if np.isfinite(v) else "not recorded"
        rows = [[name, int(p) if np.isfinite(p) else "–", pct(r), pct(c), pct(v),
                 "stopped: no points" if k in data["flagged"] else "kept"]
                for k, (name, p, r, c, v) in enumerate(zip(
                    cams, data["points"], data["rate"], data["completeness"],
                    data["coverage"]))]
        # ED7-3 / SCI7-4: where the 500 comes from, in the drawer that holds the numbers
        rows.append(["Keep rule", f"{KEEP_MIN_POINTS:,} corners per camera is a rule of thumb of "
                     "this lesson, not a library limit (the library's default minimum is 12 "
                     "corners per board observation)", "", "", "", ""])
        self.set_numbers(
            ["Camera", "Points (D1.1)", "Photos with target (D1.2)", "Board completeness (D1.3)",
             "Area covered (D1.6)", "Phase 1"], rows)


# ED6-3: the one keep/re-shoot rule, used by both the caption and the keep chip. Below the 90 %
# aim a camera is kept while it has at least this many points; fewer means re-shoot.
KEEP_MIN_POINTS = 500


def _aim_rule(data: dict) -> tuple[int, int, int, int] | None:
    """(kept cameras below aim, lowest point count among the kept cameras below aim, kept cameras
    with a rate, index of the below-aim camera with that lowest count), or None when no kept
    camera has a detection rate. Stopped cameras (no points) are left out, so the caption and the
    chip count the same cameras (SCI7-4). Only below-aim cameras are counted for the points rule,
    so a camera that meets the aim is never named as the reason to re-shoot (SCI8-4).
    Shared by the caption and the keep chip (ED6-3)."""
    kept = [k for k in range(len(data["rate"])) if k not in data["flagged"]]
    rates = [data["rate"][k] for k in kept if np.isfinite(data["rate"][k])]
    if not rates:
        return None
    below_cams = [k for k in kept if np.isfinite(data["rate"][k]) and data["rate"][k] < 0.9]
    # SCI9-6 (round 10): a below-aim camera with no point count (NaN) does not meet the
    # points rule, so it is reported as the reason (count 0), never left out of the minimum.
    unknown = [k for k in below_cams if not np.isfinite(data["points"][k])]
    if unknown:
        return len(below_cams), 0, len(rates), unknown[0]
    counted = [(int(data["points"][k]), k) for k in below_cams]
    least, least_cam = min(counted) if counted else (0, 0)
    return len(below_cams), least, len(rates), least_cam


def _aim_sentence(data: dict) -> str:
    """What the 90 % aim means for this run, with the rule that decides keep or re-shoot and
    what to change (ED2-17, ED6-3): below the aim a camera is kept while it has at least
    KEEP_MIN_POINTS points; otherwise the photos need retaking."""
    rule = _aim_rule(data)
    if rule is None:
        return "A short bar gives the calibration less to work with."
    below, least, n_cams, least_cam = rule
    if not below:
        return "Every camera meets the 90 % aim (share of its photos with the target)."
    who = (f"All {n_cams} cameras are" if below == n_cams else
           f"{below} of {n_cams} cameras {'are' if below != 1 else 'is'}")
    if least >= KEEP_MIN_POINTS:
        return (f"{who} below the 90 % aim (share of photos with the target), but the "
                f"fewest corners found among them is {least:,} (a rule of thumb: at least "
                f"{KEEP_MIN_POINTS} corners per camera): kept.")
    # one sentence, short enough for two caption lines at L1 (ED7-3, length check); the points
    # are corners found, the 90 % is a share of photos, so each number carries its unit
    # "among them" only when some cameras meet the aim, so the all-below sentence stays short
    if not np.isfinite(data["points"][least_cam]):
        # SCI9-6: no count recorded, so the rule is not met, and the sentence says so
        return (f"{who} below the 90 % aim; camera {least_cam + 1} has no corner count "
                f"recorded, so it does not meet the {KEEP_MIN_POINTS}-corner rule of thumb: "
                "re-shoot.")
    among = "" if below == n_cams else " among them"
    return (f"{who} below the 90 % aim; the fewest corners found{among} is {least:,}, "
            f"in camera {least_cam + 1}: re-shoot, as keeping needs {KEEP_MIN_POINTS} "
            "corners (a rule of thumb).")


def _below_normal_thread() -> None:
    """Run the background re-detection at reduced priority so the GUI stays responsive."""
    import sys
    if sys.platform != "win32":
        return
    try:
        import ctypes
        handle = ctypes.windll.kernel32.GetCurrentThread()
        ctypes.windll.kernel32.SetThreadPriority(handle, -1)  # THREAD_PRIORITY_BELOW_NORMAL
    except Exception:  # noqa: BLE001 - priority is a courtesy only
        pass


def create_education_phase_view(context, parent: QWidget | None = None) -> QWidget:
    """Fixed factory consumed by the shared education dialog."""
    return Phase1EducationView(context, parent)
