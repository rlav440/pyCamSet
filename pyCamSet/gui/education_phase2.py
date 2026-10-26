'''Teach Phase 2 (each camera calibrated on its own) with pictures of this run's data.

Five stages, each a figure at every
detail level; levels add overlays and panels, not paragraphs. A camera picker and a photo
picker (photos numbered from 1; the detail says the number is the photo's RMS error) drive
every stage:

1. Board views  - the chosen photo with each face it shows as its own grey shape (one face seen in
                  one photo = one "board view"), the other photos as thumbnails;
                  L2 adds where all corners fell; L3 the corners-per-face histogram.
2. Board in 3D  - the 3D scene (the camera, the posed target, the real field of view) beside
                  "one photo -> one pose": the printed face warped onto the photo (affine for
                  a telecentric lens, a homography for a pinhole lens) and, telecentric, which
                  photos can be posed at all (two faces) / pinhole, the error of each view.
                  Telecentric poses are all drawn at one distance, with a "depth unknown"
                  marker (the lens cannot see depth). L2 rays to the chosen photo's corners;
                  L3 its corners coloured by their own error.
3. Lens model   - this camera's projection diagram next to the other kind of lens (an
                  illustration), L2 every camera's lens and the distortion panel (fitted: the
                  bent grid; not fitted: "does the lens bend lines?" from this camera's
                  errors), L3 the model equations with this camera's numbers and the image-
                  centre check (older runs saved it transposed, SCI-6).
4. Fit per view - what "reprojection error" is (one corner at true size: ring = found,
                  dot = model), the photo's residual arrows (magnified, shared log colour
                  scale) and the per-photo RMS strip plot of every camera; L2 the error
                  cloud; L3 where in the image the errors are largest.
5. Decision     - RMS per camera against the warning level (µm on the target too for
                  telecentric lenses), photos per camera and what Phase 3 changes (all cameras
                  must share one target pose, so the error usually rises); L2 corners;
                  L3 error vs photos.

Honesty: figures use the saved camset and the detections the run record names (never a current
glob); the per-image errors are recomputed and checked against the saved values; photos are
shown only when Phase 1's manifest verifies them. The evidence copy of the earlier storyboard
(``_frames``) is kept for the Numbers drawers and the refusals.
'''
from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from pyCamSet.gui import education_style as es
from pyCamSet.gui.education_levels import phase_name
from pyCamSet.gui.education_photo_cache import ByteLRU
from pyCamSet.gui.education_worker import EducationWorker
from pyCamSet.gui.education_stage import Chip, EducationStageView, camera_picker
from pyCamSet.workflow import education_visual_p2 as vp
from pyCamSet.workflow.education_trace import read_trace
from pyCamSet.workflow.run_quality import blocking_reasons
from pyCamSet.utils.saving import RES_ORDER
from pyCamSet.workflow.workspace import resolve_artifact

#: Short stage names for the stepper pills.
STAGES = ("Board views", "Board in 3D", "Lens model", "Fit per view", "Decision")

#: Round 1's linear-only cut, kept for callers; round 2 (SCI2-11) words B by
#: ``vp.pattern_test`` (linear + curved field against the detection-noise baseline).
PATTERN_SHARE = 0.5


class _InputsUnavailable(Exception):
    """A plain, learner-facing reason why saved Phase 2 inputs cannot be used."""


class _LazyFrame(dict):
    """An evidence frame whose costly values (callables) are computed on first read."""

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        if callable(value):
            value = value()
            super().__setitem__(key, value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self else default


class _LazyGroups(Mapping):
    """Camera name -> walkthrough rows, each camera computed on first read."""

    def __init__(self, cameras: Mapping[str, Any], build) -> None:
        self._cameras, self._build, self._rows = dict(cameras), build, {}

    def __getitem__(self, name: str) -> list:
        if name not in self._rows:
            self._rows[name] = self._build(self._cameras[name])
        return self._rows[name]

    def __iter__(self):
        return iter(self._cameras)

    def __len__(self) -> int:
        return len(self._cameras)

    def __repr__(self) -> str:
        return f"camera groups: {', '.join(self._cameras)}"


class _CameraData:
    """One camera's views, fits and checks (computed once, on first use)."""

    def __init__(self, name: str, index: int, cam: Any, views: vp.CameraViews,
                 fits: list, check: vp.PerImageCheck) -> None:
        self.name, self.index, self.cam = name, index, cam
        self.views, self.fits, self.check = views, fits, check
        self.target: Any = None
        self.items: list | None = None
        good = [f for f in fits if f is not None]
        self.rms = vp.pooled_rms(fits)
        self.points = np.vstack([f.view.image_points for f in good]) if good else np.empty((0, 2))
        self.residuals = np.vstack([f.residuals for f in good]) if good else np.empty((0, 2))

    def fits_of(self, image: int | None) -> list:
        """The solved views (faces) of one photo, in face order."""
        if image is None:
            return []
        return sorted((f for f in self.fits if f is not None and f.view.image_index == image),
                      key=lambda f: f.view.board)

    @property
    def several_faces(self) -> bool:
        return len({f.view.board for f in self.fits if f is not None}) > 1


class Phase2View(QWidget):
    """Phase 2 education view: five visual stages over this run's saved data."""

    def __init__(self, context: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.context = context
        self.metadata = context.metadata
        self._trace = read_trace(context.run_dir, context.run_id, "phase2")
        self._unmarked_promotion = self._is_unmarked_promotion()
        self._frames_cache: list[dict[str, Any]] | None = None
        self._cameras: dict[str, _CameraData] = {}
        self._camera_errors: dict[str, str] = {}
        # SCI8-1: the pattern test's simulated noise baseline, one per (camera, photo); a pose
        # fit per draw is too slow to repeat on every redraw
        self._pattern_cache: dict[tuple[str, int | None], vp.PatternTest] = {}
        # The simulation (hundreds of pose fits) runs on a worker thread; the stage shows a
        # loading bar and fills in when it is ready.
        self._pattern_worker = EducationWorker(self)
        self._pattern_worker.ready.connect(self._pattern_ready)
        self._pattern_worker.failed.connect(self._pattern_failed)
        self._pattern_pending: tuple[str, int | None] | None = None
        # Byte-capped: a full-resolution photo per camera-photo must not accumulate.
        self._photos: ByteLRU = ByteLRU()
        self._photo_source: vp.PhotoSource | None = None
        self._scene_widget: _BoardScene | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.stage_view = EducationStageView(STAGES, self, phase_label=phase_name("phase2"))
        self.stage_view.setAccessibleName(f"{phase_name('phase2')} — run {context.run_id}")
        layout.addWidget(self.stage_view)

        # UX-4: the one camera picker ("Camera n · name", badge icon); item data = index.
        self.camera_picker = camera_picker(dict(enumerate(self._camera_names())))
        self.camera_picker.setToolTip("Choose which camera's own calibration to look at.")
        self.camera_picker.setMaximumWidth(270)
        self.stage_view.add_picker("", self.camera_picker)
        # Photos numbered from 1 (SCI-8); the detail names the number (ED-3: per photo).
        self.photo_picker = QComboBox()
        self.photo_picker.setAccessibleName("Photo")
        self.photo_picker.setToolTip(
            "Choose one photo of this camera. Photos are numbered from 1; the number after it "
            "is that photo's RMS reprojection error over every face it shows.")
        self.photo_picker.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.photo_picker.setMaximumWidth(230)
        self.stage_view.add_picker("", self.photo_picker)
        #: Old name kept for callers of the round-0 view picker.
        self.view_picker = self.photo_picker

        self.camera_picker.currentIndexChanged.connect(self._camera_changed)
        self.photo_picker.currentIndexChanged.connect(lambda _i: self._show_stage())
        self.stage_view.stageChanged.connect(lambda _i: self._show_stage())
        self.stage_view.levelChanged.connect(lambda _l: self._show_stage())
        self._fill_photos()
        self._show_stage()

    # ==================================================================================
    # Navigation API used by the dialog

    def step_back(self) -> None:
        self.stage_view.stepper.previous()

    def step_forward(self) -> None:
        self.stage_view.stepper.next()

    def stop_for_context_change(self) -> None:
        self.stage_view.stop_for_context_change()

    def current_stage(self) -> int:
        return self.stage_view.current_stage()

    def level(self) -> int:
        return self.stage_view.level()

    # ==================================================================================
    # Data

    def _camera_names(self) -> list[str]:
        """Cameras in the run's own order (report, then intrinsics, then the trace)."""
        cached = self.__dict__.get("_names_cache")
        if cached is not None:
            return list(cached)
        report = self.metadata.get("report") or {}
        names = report.get("camera_names") if isinstance(report, Mapping) else None
        if not names:
            diagnostics = self.metadata.get("diagnostics") or {}
            intrinsics = diagnostics.get("D2.2_intrinsics") if isinstance(diagnostics, Mapping) else None
            names = list(intrinsics) if isinstance(intrinsics, Mapping) else []
        if not names:
            names = [str(c.get("camera")) for c in self._trace_cameras()]
        self._names_cache = [str(n) for n in names]
        return list(self._names_cache)

    def _blocked_reason(self) -> str | None:
        """Why nothing from the saved inputs may be drawn for this run (None = allowed)."""
        if self.metadata.get("error"):
            return (f"This Phase 2 run failed before producing lens estimates "
                    f"({str(self.metadata.get('error')).rstrip('.')}). Run Phase 2 again.")
        if self._unmarked_promotion:
            return ("This run came from an unmarked Optimisation-tab trial: its camera file may "
                    "hold later-stage cameras, so Phase 2's own estimates are not shown. "
                    "Run Phase 2 again to create a checkable result.")
        return None

    def _camera_data(self, name: str) -> _CameraData | None:
        """Views and fits of one camera; None with a reason in ``_camera_errors``."""
        if name in self._cameras:
            return self._cameras[name]
        if name in self._camera_errors:
            return None
        blocked = self._blocked_reason()
        if blocked:
            self._camera_errors[name] = blocked
            return None
        try:
            cams, detection, target = self._used_inputs()
            minimum = self._minimum()
            views = vp.board_views(detection, target, name, minimum)
            fits = vp.fit_views(cams[name], views.views)
            saved = vp.saved_series(self.metadata, name)
            check = vp.check_per_image(fits, saved)
            if saved and not check.all_match and not vp.is_telecentric(cams[name]):
                # a run saved before the closed-form seed holds the older pose-only refit values
                older = vp.fit_views(cams[name], views.views, pose_only=True)
                older_check = vp.check_per_image(older, saved)
                if older_check.all_match:
                    fits, check = older, older_check
        except _InputsUnavailable as exc:
            self._camera_errors[name] = _sentence(str(exc))
            return None
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Phase 2 education: rebuilding %s failed", name)
            self._camera_errors[name] = ("This camera's board views could not be rebuilt from "
                                         "the saved detections.")
            return None
        index = self._camera_names().index(name) if name in self._camera_names() else 0
        data = _CameraData(name, index, cams[name], views, fits, check)
        data.target = target
        self._cameras[name] = data
        return data

    def _minimum(self) -> int:
        params = self.metadata.get("params") or {}
        try:
            return int(params.get("min_detections_per_board", 12))
        except (TypeError, ValueError):
            return 12

    def _lens_model(self) -> str:
        params = self.metadata.get("params") or {}
        return str(params.get("lens_model", "pinhole") if isinstance(params, Mapping) else "pinhole")

    def _photo(self, camera: str, image_index: int) -> np.ndarray | None:
        key = (camera, int(image_index))
        if key not in self._photos:
            if self._photo_source is None:
                self._photo_source = vp.photo_source(self.context)
            self._photos[key] = vp.load_photo(self._photo_source.path(camera, image_index))
        return self._photos[key]

    def _photo_note(self) -> str:
        source = self._photo_source or vp.PhotoSource("could not be checked", None)
        if source.available:
            return ""
        return "photo not shown: Phase 1's photo list is not verified for this run"

    def _selected_camera(self) -> str | None:
        data = self.camera_picker.currentData()
        names = self._camera_names()
        if data is None:
            return None
        index = int(data)
        return names[index] if 0 <= index < len(names) else None

    def _selected_image(self) -> int | None:
        data = self.photo_picker.currentData()
        return int(data) if data is not None else None

    def _selected_fit(self, data: _CameraData | None):
        """The chosen photo's largest face (its main board view), or None."""
        if data is None:
            return None
        fits = data.fits_of(self._selected_image())
        return max(fits, key=lambda f: len(f.view.image_points)) if fits else None

    def _camera_changed(self, _index: int) -> None:
        self.stop_for_context_change()
        self._fill_photos()
        self._show_stage()

    def _fill_photos(self) -> None:
        """List the camera's photos (1-based, with their RMS), defaulting to the photo of
        typical (median) error that the 3D stage can pose."""
        name = self._selected_camera()
        data = self._camera_data(name) if name else None
        combo = self.photo_picker
        combo.blockSignals(True)
        combo.clear()
        if data is not None and data.check.per_image:
            for image, value in sorted(data.check.per_image.items()):
                combo.addItem(f"{es.photo_label(image)} · RMS {es.format_px(value)}", image)
                combo.setItemData(combo.count() - 1,
                                  f"{es.photo_label(image)}: RMS reprojection error "
                                  f"{es.format_px(value)} over every face it shows "
                                  f"(stored index {image})", Qt.ItemDataRole.ToolTipRole)
            posed = set(data.check.per_image)
            if vp.is_telecentric(data.cam):
                posed &= {i.image_index for i in self._scene_items(data)} or posed
            ranked = sorted((data.check.per_image[i], i) for i in posed)
            combo.setCurrentIndex(max(0, combo.findData(ranked[len(ranked) // 2][1])))
        combo.setEnabled(combo.count() > 0)
        combo.blockSignals(False)

    # ==================================================================================
    # Stages

    def _pattern_ready(self, key: Any, test: Any) -> None:
        self._pattern_cache[key] = test
        self._pattern_pending = None
        self._show_stage()

    def _pattern_failed(self, key: Any, _message: str) -> None:
        # No baseline to show: the stage reads as 'too few corners to tell', never a crash.
        self._pattern_cache[key] = None
        self._pattern_pending = None
        self._show_stage()

    def close_background(self) -> None:
        """Cancel any simulation still running (called when the dialog closes)."""
        self._pattern_worker.close()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Block until a pending simulation has finished (tests and screenshot tools only)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            left = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not self._pattern_worker.wait_until_idle(left):
                return False
            if self._pattern_pending is None or self._pattern_pending in self._pattern_cache:
                return True
            # The job for the shown photo was superseded and dropped before it reported, so
            # nothing would ever arrive: ask the stage for it again.
            self._pattern_pending = None
            self._show_stage()
            if deadline is not None and time.monotonic() >= deadline:
                return False

    def _show_stage(self) -> None:
        stage = self.stage_view.current_stage()
        level = self.stage_view.level()
        self.stage_view.set_busy(None)
        self.camera_picker.setVisible(stage in (0, 1, 2, 3))
        self.photo_picker.setVisible(stage in (0, 1, 3))
        if stage != 1:
            self.stage_view.set_controls(None)
        [self._stage_views, self._stage_scene, self._stage_lens, self._stage_fit,
         self._stage_decision][stage](level)

    # -- 1 Board views -----------------------------------------------------------------

    def _stage_views(self, level: int) -> None:
        name = self._selected_camera()
        data = self._camera_data(name) if name else None
        sv = self.stage_view
        if data is None:
            reason = self._camera_errors.get(name or "", self._input_refusal())
            sv.draw_figure(lambda fig, _t: _refusal(fig, "Board views", reason))
            sv.set_caption(_level_text(level, [
                "Phase 2 calibrates each camera on its own, from the photos where it saw the target.",
                "The views of this run cannot be drawn: " + _lower_first(reason),
            ]))
            self._set_chips([])
            sv.set_numbers(["Item", "Value", "Evidence"], self._input_numbers())
            return
        image = self._selected_image()
        images = sorted({v.image_index for v in data.views.views})
        photo = self._photo(data.name, image) if image is not None else None
        note = self._photo_note()
        minimum = self._minimum()
        thumbs = _sample(images, 12)
        if image is not None and image not in thumbs and thumbs:
            nearest = int(np.argmin([abs(i - image) for i in thumbs]))
            thumbs[nearest] = image
            thumbs = sorted(set(thumbs))
        thumb_photos = {i: self._photo(data.name, i) for i in thumbs}

        def draw(fig, _theme):
            _draw_board_views(fig, data, image, photo, thumbs, thumb_photos, images, level,
                              minimum, note)

        sv.draw_figure(draw)
        n = data.index + 1
        dropped = len(data.views.dropped)
        faces = data.fits_of(image)
        what = ("Each face of the target seen in one photo is one flat “board view”"
                if data.several_faces else "The board seen in one photo is one “board view”")
        sv.set_caption(_level_text(level, [
            f"Phase 2 calibrates each camera on its own, from the photos where it saw the "
            f"target. {what}: camera {n} learned from {len(data.views.views)} board views in "
            f"{len(images)} photos" + (f" ({es.photo_label(image)} shows {len(faces)})."
                                       if image is not None and len(faces) > 1 else "."),
            "Map C stacks every corner of every view: corners spread over the whole picture pin "
            "the lens down best.",
            (f"Faces with fewer than {minimum} corners are left out (grey): "
             f"{dropped} for this camera." if dropped else
             f"Faces with fewer than {minimum} corners would be left out; none were for this camera."),
        ]))
        self._set_chips([
            Chip(f"{len(data.views.views):,}", "board views", camera=data.index),
            Chip(f"{len(images):,}", "photos"),
            Chip(f"{len(data.points):,}", "corners used"),
            *([Chip(str(dropped), "faces left out", tone="warn" if dropped else None)]
              if level >= 2 else []),
        ])
        sv.set_numbers(["Item", "Value", "Evidence"], self._input_numbers())

    # -- 2 Board in 3D ------------------------------------------------------------------

    def _stage_scene(self, level: int) -> None:
        name = self._selected_camera()
        data = self._camera_data(name) if name else None
        sv = self.stage_view
        if data is None or not any(f is not None for f in data.fits):
            reason = self._camera_errors.get(name or "", "No board view has a pose.")
            sv.draw_figure(lambda fig, _t: _refusal(fig, "Board in 3D", reason))
            sv.set_caption(_level_text(level, [
                "Phase 2 works out where the board was in front of each camera.",
                "This picture cannot be drawn: " + _lower_first(reason)]))
            self._set_chips([])
            sv.set_numbers(["View", "Recalculated fit error (px)", "Fit error saved by Phase 2 (px)",
                            "Status"], self._walkthrough_rows(name or "", level))
            return
        fit = self._selected_fit(data)
        image = self._selected_image()
        telecentric = vp.is_telecentric(data.cam)
        items = self._scene_items(data)
        if image is None or not items:
            selected = None
        elif telecentric:
            selected = next((k for k, i in enumerate(items) if i.image_index == image), None)
        else:
            selected = next((k for k, i in enumerate(items) if i.fit_index is not None
                             and data.fits[i.fit_index] is fit), None)
        warp = vp.face_warp(fit.view, data.target, telecentric) if fit is not None else None
        photo = self._photo(data.name, image) if image is not None else None
        photos = sorted(data.check.per_image)
        if self._scene_widget is None:
            self._scene_widget = _BoardScene(self)
        self._scene_widget.show_camera(data, items, selected, level, telecentric,
                                       self._lens_size(data), warp=warp, photo=photo,
                                       image=image, photos=photos, face_mm=self._face_mm())
        sv.set_visual(self._scene_widget)
        n = data.index + 1
        posed = len(items)
        face_poses = sum(1 for f in data.fits if f is not None)
        photo_error = self._image_error(data, image) if image is not None else float("nan")
        if telecentric:
            # ED2-9: two kinds of pose. Every face gets a face pose (it fits the lens and gives
            # each photo its error); only a photo with 2+ faces gets a cube pose (drawn in A).
            first = (f"Every face seen gets its own face pose from its warped print (B): camera "
                     f"{n}'s {face_poses} fit the lens and give each photo its error. A photo "
                     f"showing two faces or more also gets a cube pose: {posed} of "
                     f"{len(photos)} (C), drawn in A.")
            second = ("The lens sees no depth: A draws the cube at one distance (grey ↔), "
                      "with parallel lines to its corners.")
        else:
            first = (f"One photo fixes where the board was: the printed board, warped in "
                     f"perspective onto its corners (B), gives tilt, turn and distance. Each "
                     f"plane is one board view of camera {n}, coloured by its error (C).")
            second = "Lines from the lens meet in one point: far boards look smaller."
        third = "Dots in A: each corner's own error."
        if warp is not None and np.isfinite(photo_error):
            # SCI3-9: compare like with like. The warp and the pose are of the same face (a
            # pose is a warp without shear, so it can only miss by more); the photo's error
            # pools all its faces and is named as such.
            faces_in_photo = sum(1 for f in data.fits if f is not None
                                 and f.view.image_index == fit.view.image_index)
            # "face 5" in words: a caption must not carry "F5" (the copy lint's code pattern)
            face = (es.face_label(fit.view.board, short=False).lower()
                    if telecentric else "the board")
            # One sentence (the L3 caption allows four): the semicolon keeps the clause count.
            # Pinhole: no single face to name, so the sentence starts at the numbers (shorter
            # keeps L3 within four rendered lines at 1280 px).
            head = f"for {face}, the warp in B" if telecentric else "the warp in B"
            # round 10: the pinhole variant drops "(no shear)" to stay within four lines at 1280 px
            pose = "its pose (no shear)" if telecentric else "its pose"
            third = (f"Dots in A are each corner's error: {head} misses by "
                     f"{es.format_px(warp.rms)}; {pose} by {es.format_px(fit.rms)}"
                     + (f"; all {faces_in_photo} faces together, {es.format_px(photo_error)}."
                        if faces_in_photo > 1 else "."))
        if selected is None and image is not None and telecentric:
            third = (f"{es.photo_label(image)} shows one face: it has a face pose and an "
                     "error, but no cube pose to draw.")
        sv.set_caption(_level_text(level, [first, second, third]))
        errors = np.array([i.error for i in items if np.isfinite(i.error)])
        chips = [Chip(f"{posed} of {len(photos)}" if telecentric else f"{posed}",
                      "photos with a cube pose" if telecentric else "board views",
                      camera=data.index),
                 Chip(es.format_px(float(np.median(errors))) if errors.size else "—",
                      "median RMS error")]
        if selected is not None and level >= 2:
            chips.append(Chip(es.format_px(items[selected].error),
                              es.photo_label(image).lower(), tone="accent"))
        if fit is not None and level >= 3 and not telecentric:
            chips.append(Chip(f"{vp.pose_angles(fit.R)[0]:.0f}°", "tilt to the camera"))
            chips.append(Chip(f"{fit.t[2] * 1000:.0f} mm", "distance"))
        self._set_chips(chips)
        if level >= 3:
            sv.set_numbers(["View", "Recalculated fit error (px)",
                            "Fit error saved by Phase 2 (px)", "Status"],
                           self._walkthrough_rows(data.name, level))
        elif telecentric:
            sv.set_numbers(["Photo", "Corners", "RMS error of the photo (px)", "index (0-based)"],
                           [(es.photo_label(i.image_index), len(i.pixels), f"{i.error:.3g}",
                             i.image_index) for i in items])
        else:
            sv.set_numbers(["View", "Photo", "Corners", "RMS error (px)", "Tilt (°)",
                            "Distance (mm)"],
                           [(k + 1, es.photo_label(f.view.image_index), len(f.view.object_points),
                             f"{f.rms:.3g}", f"{vp.pose_angles(f.R)[0]:.0f}",
                             f"{f.t[2] * 1000:.0f}")
                            for k, f in enumerate(data.fits) if f is not None])

    def _face_mm(self) -> float | None:
        """A cube target's face size (mm, its ``length`` setting), else None: the printed
        corner grid is smaller than the face it sits on (SCI2-21)."""
        params = self.metadata.get("params") or {}
        spec = params.get("target") if isinstance(params, Mapping) else None
        if not isinstance(spec, Mapping) or "cube" not in str(spec.get("type", "")).lower():
            return None
        try:
            length = float(spec.get("length"))
        except (TypeError, ValueError):
            return None
        return length if math.isfinite(length) and length > 0 else None

    def _scene_items(self, data: _CameraData) -> list:
        if data.items is None:
            try:
                _, detection, target = self._used_inputs()
                data.items = vp.scene_items(data.cam, detection, target, data.name,
                                            self._minimum(), data.fits, data.check.per_image)
            except Exception:
                data.items = []
        return data.items

    def _set_chips(self, chips: Sequence[Chip]) -> None:
        """Hide the old chips first: StatChips only deleteLater()s them, so they linger
        (drawn under the new row) until the event loop runs the deferred deletes."""
        for frame in list(getattr(self.stage_view.chips, "_frames", ())):
            frame.hide()
        self.stage_view.set_chips(chips)

    def _lens_size(self, data: _CameraData) -> tuple[int, int]:
        first = next((f for f in data.fits if f is not None), None)
        photo = self._photo(data.name, first.view.image_index) if first is not None else None
        return vp.image_size_of(data.cam, photo, data.points)

    # -- 3 Lens model -------------------------------------------------------------------

    def _stage_lens(self, level: int) -> None:
        sv = self.stage_view
        lenses = self._saved_lenses()
        name = self._selected_camera()
        blocked = self._blocked_reason()
        if blocked or not lenses or name not in lenses:
            reason = blocked or "This run's record has no saved lens values."
            sv.draw_figure(lambda fig, _t: _refusal(fig, "Lens model", reason))
            sv.set_caption(_level_text(level, ["Phase 2 fits one lens model per camera.",
                                               "It cannot be drawn: " + _lower_first(reason)]))
            self._set_chips([])
            sv.set_numbers(["Camera", "Saved lens estimate", "Saved lens distortion"],
                           self._camera_rows(plain=level < 2))
            return
        index = self._camera_names().index(name)
        lens = lenses[name]
        not_fitted = not lens.fitted
        cam = self._lens_camera(name)
        exaggerate = 1.0 if not_fitted else _exaggeration(lens, cam)
        photo, target_mm, distance_mm = self._lens_context(name)
        check = vp.centre_check(lens.cx, lens.cy, (lens.width, lens.height))
        data = self._camera_data(name)

        def draw(fig, _theme):
            _draw_lens(fig, name, index, lens, lenses, self._camera_names(), level, cam,
                       exaggerate, photo, target_mm, distance_mm, data, check)

        sv.draw_figure(draw)
        n = index + 1
        if lens.model == "telecentric":
            first = (f"Camera {n}'s lens is telecentric: parallel rays, so the target looks "
                     "the same size near or far (B). "
                     f"1 px = {_um_pair(lens)}"
                     + ("; no distortion fitted." if not_fitted else "."))
        else:
            first = (f"Camera {n} is a pinhole camera: rays meet in the lens, so far objects look "
                     "smaller (B: a telecentric lens). Its focal length is "
                     f"{lens.fx:,.0f} px, a {lens.fov_deg[0]:.0f}° wide view"
                     + ("; Phase 2 fits no lens distortion." if not_fitted else "."))
        # ED2-3: one camera's photos cannot tell bending from other errors (target shape,
        # poses); Phase 3 fits it with all cameras, and Phase 4 takes out what is target shape.
        second = ("D: one camera cannot tell bending from other errors; Phase 3 fits it, "
                  "Phase 4 drops target shape."
                  if not_fitted else
                  f"D shows the saved distortion, exaggerated ×{exaggerate:g} to be seen.")
        aniso = _anisotropy(lenses)
        third = ("The estimate is closed-form (a few linear fits, no iterations); E is the model."
                 if aniso is None or aniso < 2.0 else
                 f"C: x and y scales differ by up to {aniso:.1f} % in the fit; the data cannot "
                 "tell whether the sensor or the model causes it.")
        if check.transposed:
            # the chip's meaning, in the L1 clause (one line more would overflow L3; round 9)
            first = first[:-1] + "; saved centre x and y swapped."
        sv.set_caption(_level_text(level, [first, second, third]))
        chips = []
        if lens.model == "telecentric":
            chips.append(Chip(_um_pair(lens, short=True), "per pixel (x × y)", camera=index))
            chips.append(Chip(f"{lens.field_mm[0]:.1f} × {lens.field_mm[1]:.1f} mm", "field of view"))
        else:
            chips.append(Chip(f"{lens.fx:,.0f} px", "focal length", camera=index))
            chips.append(Chip(f"{lens.fov_deg[0]:.0f}° × {lens.fov_deg[1]:.0f}°", "field of view"))
        if not check.transposed:  # (room for the caption: the warning chip comes first)
            chips.append(Chip("not fitted" if not_fitted else "saved", "lens distortion"))
        if check.transposed:
            chips.append(Chip("transposed", "saved image centre", tone="warn",
                              tooltip=_transposed_text(check)))
        self._set_chips(chips)
        number_rows = list(self._camera_rows(plain=level < 2))
        if check.transposed:
            # round 10 (item 8): the 'transposed' chip's meaning, in the drawer at every level
            # (L2 and L3 captions have no room for a fourth sentence)
            number_rows.append((f"Camera {n} image centre", _transposed_text(check), "–"))
        sv.set_numbers(["Camera", "Saved lens estimate", "Saved lens distortion"]
                       if level < 2 else
                       ["Camera", "Saved intrinsic parameters", "Saved distortion coefficients"],
                       number_rows)

    def _lens_context(self, name: str) -> tuple[Any, float | None, float | None]:
        """The chosen photo, the target's size (mm) and the typical board distance (mm,
        pinhole) for the lens diagram; each None when this run cannot supply it."""
        data = self._camera_data(name)
        image = self._selected_image()
        if data is not None and image not in data.check.per_image:
            image = next(iter(sorted(data.check.per_image)), None)
        photo = self._photo(name, image) if (data is not None and image is not None) else None
        target_mm = distance_mm = None
        try:
            _, _, target = self._used_inputs()
            points = np.asarray(target.point_data, dtype=float).reshape(-1, 3)
            target_mm = float(np.ptp(points, axis=0).max()) * 1000.0
        except Exception:
            target_mm = None
        if data is not None and not vp.is_telecentric(data.cam):
            good = [f for f in data.fits if f is not None]
            if good:
                distance_mm = float(np.median([f.t[2] for f in good])) * 1000.0
                # the part of the board a view typically saw, not the whole printed board
                target_mm = float(np.median([np.ptp(f.view.object_points, axis=0).max()
                                             for f in good])) * 1000.0
        return photo, target_mm, distance_mm

    def _saved_lenses(self) -> dict[str, vp.LensSummary]:
        """Every camera's saved lens (D2.2/D2.3) as a LensSummary."""
        out = {}
        for name in self._camera_names():
            cam = self._lens_camera(name)
            if cam is None:
                continue
            size = self._image_size_for(name, cam)
            out[name] = vp.lens_summary(cam, size, telecentric=self._lens_model() == "telecentric")
        return out

    def _lens_camera(self, name: str):
        """A lens rebuilt from the saved run record (D2.2 intrinsics, D2.3 distortion)."""
        diagnostics = self.metadata.get("diagnostics") or {}
        intrinsics = (diagnostics.get("D2.2_intrinsics") or {}).get(name) \
            if isinstance(diagnostics, Mapping) else None
        if not isinstance(intrinsics, Mapping):
            return None
        try:
            fx, fy, cx, cy = (float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy"))
        except (KeyError, TypeError, ValueError):
            return None
        distortion = (diagnostics.get("D2.3_distortion") or {}).get(name) or {}
        coeffs = distortion.get("coeffs") if isinstance(distortion, Mapping) else distortion
        try:
            coeffs = [float(v) for v in (coeffs or [0.0])]
        except (TypeError, ValueError):
            coeffs = [0.0]
        res = [float(v) for v in (intrinsics.get("res") or (0, 0))][:2]
        if len(res) == 2 and intrinsics.get("res_order") != RES_ORDER:
            # Runs before the 2026-10-08 Camera.res fix saved (height, width).
            res = [res[1], res[0]]
        from types import SimpleNamespace
        return SimpleNamespace(intrinsic=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
                               distortion_coefs=np.array(coeffs),
                               res=np.array(res or (0, 0), dtype=float))

    def _image_size_for(self, name: str, cam: Any) -> tuple[int, int]:
        """(width, height) of the camera's photos: from the photos/detections when they can be
        read, else the saved size (older runs' (height, width) already swapped)."""
        data = self._cameras.get(name)
        if data is None and name not in self._camera_errors and not self._blocked_reason():
            data = self._camera_data(name)
        if data is not None:
            return self._lens_size(data)
        res = [int(v) for v in np.asarray(cam.res).reshape(-1)[:2]] if cam is not None else []
        if len(res) == 2 and min(res) > 0:
            return res[0], res[1]
        return 640, 480

    # -- 4 Fit per view -----------------------------------------------------------------

    def _stage_fit(self, level: int) -> None:
        sv = self.stage_view
        name = self._selected_camera()
        data = self._camera_data(name) if name else None
        image = self._selected_image()
        faces = data.fits_of(image) if data is not None else []
        series = self._saved_series_all()
        index = self._camera_names().index(name) if name in self._camera_names() else 0
        photo = self._photo(name, image) if faces else None
        note = self._photo_note()
        reason = self._camera_errors.get(name or "")
        if self._blocked_reason():
            reason = self._blocked_reason()
            series = {}
        highlight: Any = index
        if faces:
            keys = sorted(vp.saved_series(self.metadata, name))
            if image in keys:
                highlight = (index, keys.index(image))
        lens = self._saved_lenses().get(name or "")
        # SCI2-11 / SCI8-1: a smooth (linear + curved) field against the noise baseline that the
        # page's own pose fit gives (simulated once per photo, then cached)
        test = None
        pending = False
        if faces and data is not None:
            key = (name or "", image)
            if key not in self._pattern_cache:
                pending = True
                if self._pattern_pending != key:
                    self._pattern_pending = key
                    cam = data.cam
                    self._pattern_worker.submit(
                        key, lambda _report, cam=cam, faces=faces: vp.simulated_pattern_test(
                            cam, faces))
                sv.set_busy("Measuring what detection noise alone would give")
            else:
                test = self._pattern_cache[key]

        def draw(fig, theme):
            _draw_fit(fig, data, faces, photo, series, highlight, index, level, note, reason,
                      lens, theme, test)

        sv.draw_figure(draw)
        n = index + 1
        if data is not None and faces:
            observed = np.vstack([f.view.image_points for f in faces])
            projected = np.vstack([f.projected for f in faces])
            k = vp.typical_corner(observed, projected, photo)
            example = float(np.hypot(*(projected[k] - observed[k])))
            pattern = ("the noise comparison is still being measured" if pending
                       else _pattern_words(test))
            # round 8: one entry per level, so the ring/dot key and the B verdict share entry 1
            # (L1 two lines at 1280 px); C/D and E follow as entries 2 and 3
            first = (f"Ring = detected corner, dot = the model, "
                     f"{es.format_px(example)} in A. In B {pattern}.")
        elif data is not None:
            first = ("The dots are the RMS error of every photo of every camera; this photo has "
                     "no solved view.")
        else:
            first = "The dots are the RMS error of every photo of every camera, as this run saved them."
        sv.set_caption(_level_text(level, [
            first,
            # ED2-9: why single-face photos have a dot in C too
            ("C: every photo (one-face photos too, each face posed alone); D: a round cloud at "
             "zero is noise."
             if data is not None and data.several_faces else
             "C: every photo of every camera; D: a round cloud at zero is noise."),
            "E shows the mean error in each part of the image, in this camera's colours.",
        ]))
        chips = []
        if data is not None:
            if faces:
                chips.append(Chip(es.format_px(self._image_error(data, image)),
                                  f"{es.photo_label(image).lower()} · RMS", tone="accent"))
            chips.append(Chip(es.format_px(data.rms), f"camera {n} · RMS", camera=data.index))
            check = data.check
            chips.append(Chip("✓ matches" if check.all_match else "differs",
                              "the saved errors", tone="good" if check.all_match else "bad",
                              tooltip=(f"{check.matched} of {check.compared} photos recalculated "
                                       "from the saved camera file and detections agree with the "
                                       "values Phase 2 saved.")))
        self._set_chips(chips)
        number_rows = self._fit_rows(data, level)
        # SCI9-7 (round 10): a silent fallback in the pattern test is named here, not hidden
        if test is not None and test.fallback:
            number_rows.append(("Pattern test", "–", test.fallback, "–"))
        sv.set_numbers(["Photo", "Recalculated RMS error (px)", "Evidence", "index (0-based)"],
                       number_rows)

    def _image_error(self, data: _CameraData, image: int) -> float:
        return float(data.check.per_image.get(image, float("nan")))

    def _saved_series_all(self) -> dict[int, list[float]]:
        """Saved per-photo errors by camera index (finite values, photo order)."""
        out = {}
        for index, name in enumerate(self._camera_names()):
            series = vp.saved_series(self.metadata, name)
            out[index] = [series[k] for k in sorted(series)]
        return out

    def _fit_rows(self, data: _CameraData | None, level: int) -> list[tuple]:
        if data is None:
            return []
        saved = vp.saved_series(self.metadata, data.name)
        rows = []
        for image, value in data.check.per_image.items():
            expected = saved.get(image)
            agrees = expected is not None and abs(value - expected) <= max(1e-6, 1e-6 * abs(expected))
            evidence = ("matches D2.6" if agrees else "does not match D2.6") if level >= 2 else \
                ("recalculated; matches the saved value" if agrees else
                 "recalculated; differs from the saved value")
            rows.append((es.photo_label(image), _format_number(value), evidence, image))
        return rows

    # -- 5 Decision ---------------------------------------------------------------------

    def _stage_decision(self, level: int) -> None:
        from pyCamSet.utils.intrinsics_report import HIGH_INITIAL_RMS_PX
        sv = self.stage_view
        warning = float(HIGH_INITIAL_RMS_PX)
        rows = self._report_rows()
        blocked = self._blocked_reason()
        if blocked or not rows:
            reason = blocked or ("This run's record has no per-camera fit summary; run Phase 2 "
                                 "again to record one.")
            sv.draw_figure(lambda fig, _t: _refusal(fig, "Decision", reason))
            sv.set_caption(_level_text(level, [
                "Did each camera see the board in enough photos, with a fit error below "
                "pyCamSet's warning level?", "Not checkable here: " + _lower_first(reason)]))
            self._set_chips([])
            sv.set_numbers(["Camera", "Board seen in", "Fit error (px)", "Verdict"], [])
            return
        um = None
        if self._lens_model() == "telecentric":
            sizes = [l.um_per_px[0] for l in self._saved_lenses().values()
                     if l.um_per_px and np.isfinite(l.um_per_px[0])]
            um = float(np.mean(sizes)) if sizes else None

        groups = camera_groups(rows)
        later = self._later_group_check(groups) if groups is not None and level >= 3 else None

        def draw(fig, _theme):
            _draw_decision(fig, rows, warning, level, um, later)

        sv.draw_figure(draw)
        above = [r for r in rows if r["views"] and r["rms"] is not None and r["rms"] > warning]
        none = [r for r in rows if not r["views"] or r["rms"] is None]
        going = len(rows) - len(none)
        first = (f"{going} of {len(rows)} cameras have a first lens estimate and go on to "
                 "Phase 3")
        if above:
            badges = ", ".join(str(r["index"] + 1) for r in above)
            first += (f" (Camera{'s' if len(above) > 1 else ''} {badges} "
                      f"{'are' if len(above) > 1 else 'is'} above the {warning:g} px warning "
                      "level: check in Phase 1).")
        elif none:
            first += "; cameras with no usable view are left out."
        else:
            first += f", all below the {warning:g} px warning level."
        # SCI3-2: the rise comes from sharing one pose on the *drawn* target; Phase 4 refines it
        next_step = (" Next, one shared pose per photo (C): the error usually rises in "
                     "Phase 3, then falls in Phase 4.")
        improve = "To improve a camera, add photos with the board at new angles across its view."
        third = improve
        if groups is not None:
            # ED2-26: point out the groups rather than a misleading "more photos, more error"
            third = _group_sentence(groups, later)
        sv.set_caption(_level_text(level, [
            first + next_step,
            "Warnings advise; no-view cameras are dropped.",
            third,
        ]))
        errors = [r["rms"] for r in rows if r["rms"] is not None]
        self._set_chips([
            Chip(f"{going} / {len(rows)}", "cameras go on", tone="good" if not none else "warn",
                 tooltip=improve),
            Chip(_px_range_text(errors) if errors else "—", "RMS error range",
                 tone="good" if errors and max(errors) <= warning else "warn"),
            Chip(f"{sum(r['views'] or 0 for r in rows):,}", "photos, all cameras"),
            *([Chip(f"{sum(r['points'] or 0 for r in rows):,}", "corners")] if level >= 2 else []),
        ])
        after = (later or {}).get("rms") or {}
        sv.set_numbers(["Camera", "Board seen in", "Corners", "Fit error (px)", "Verdict"]
                       + ([f"RMS error after Phase 4, run {later['run']} (px)"] if after else []), [
            (r["name"], f"{r['views'] or 0} photos", f"{r['points'] or 0:,}",
             _format_number(r["rms"]) if r["rms"] is not None else "no usable view",
             ("dropped: no usable view" if not r["views"] or r["rms"] is None else
              f"above the {warning:g} px warning level" if r["rms"] > warning else "goes on"))
            + ((_format_number(after[r["name"]]) if r["name"] in after else "—",)
               if after else ())
            for r in rows])

    def _later_phase4(self) -> Mapping | None:
        """The latest complete Phase 4 run that continued, through a complete Phase 3 run,
        from this Phase 2 run (or None). Never for a blocked run."""
        if "_phase4_cache" not in self.__dict__:
            self._phase4_cache = None
            manager = getattr(self.context, "manager", None)
            if manager is not None and self._blocked_reason() is None:
                try:
                    def usable(run: Mapping) -> bool:
                        return run.get("status") == "complete" and not run.get("error")
                    phase3 = {str(run.get("run_id")) for run in manager.load_runs("phase3")
                              if usable(run) and (run.get("inputs") or {}).get("phase2_run_id")
                              == self.context.run_id}
                    runs = [run for run in manager.load_runs("phase4") if usable(run)
                            and str((run.get("inputs") or {}).get("phase3_run_id")) in phase3]
                    # load_runs is oldest first by recency (run ids do not sort by date)
                    self._phase4_cache = runs[-1] if runs else None
                except Exception:  # noqa: BLE001 - no later run is shown, never guessed
                    self._phase4_cache = None
        return self._phase4_cache

    def _later_group_check(self, groups: Mapping[str, Any]) -> dict[str, Any] | None:
        """SCI3-8: the same two camera groups after Phase 4 (a run that continued from this
        one): each camera's RMS error replayed from Phase 4's saved result (the statistic
        Phase 2 saves, so both sit on one axis), the groups' medians and their ratio. None
        when no such run exists or its replay does not reproduce its saved error."""
        if "_later_groups_cache" in self.__dict__:
            return self._later_groups_cache
        self._later_groups_cache = None
        run = self._later_phase4()
        if run is not None:
            try:
                from pyCamSet.workflow.education_visual_p3 import phase4_camera_rms
                rms = phase4_camera_rms(run)
            except Exception:  # noqa: BLE001
                rms = {}
            high = [rms[r["name"]] for r in groups["high"] if r["name"] in rms]
            low = [rms[r["name"]] for r in groups["low"] if r["name"] in rms]
            if high and low and len(high) == len(groups["high"]) and \
                    len(low) == len(groups["low"]):
                m_high, m_low = float(np.median(high)), float(np.median(low))
                self._later_groups_cache = {
                    "run": str(run.get("run_id") or ""), "rms": dict(rms),
                    "median_high": m_high, "median_low": m_low,
                    "ratio": m_high / max(m_low, 1e-12),
                    "gone": m_high < GROUP_RATIO * m_low}
        return self._later_groups_cache

    def _report_rows(self) -> list[dict[str, Any]]:
        """The saved per-camera summary (report.per_camera) in camera order."""
        report = self.metadata.get("report") or {}
        rows = report.get("per_camera") if isinstance(report, Mapping) else None
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            return []
        names = self._camera_names()
        out = []
        for k, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("name", f"camera {k + 1}"))
            rms = row.get("rms_px")
            rms = None if _is_missing(rms) else float(rms)
            out.append({"name": name, "index": names.index(name) if name in names else k,
                        "views": int(row.get("n_views") or 0), "rms": rms,
                        "points": int(row.get("n_points") or 0)})
        return out

    # ==================================================================================
    # Evidence (the earlier storyboard's checked copy, kept for Numbers and refusals)

    @property
    def _frames(self) -> list[dict[str, Any]]:
        if self._frames_cache is None:
            self._frames_cache = self._make_frames()
        return self._frames_cache

    def _input_refusal(self) -> str:
        frame = next(f for f in self._frames if f["title"] == "Input selection")
        return frame["copy"].split(". ", 1)[-1]

    def _input_numbers(self) -> list[tuple[str, str, str]]:
        """Stage 1 Numbers: what was used, the run status and per-camera contributions."""
        frames = {f["title"]: f for f in self._frames}
        rows = [("Detections used", self._used_label(), frames["Input selection"]["copy"]),
                ("Run status", str(self.metadata.get("status") or "not recorded"),
                 frames["Run status"]["copy"])]
        for title in ("High-distortion second pass", "Pruned rerun"):
            if title in frames:
                rows.append((title, "see note", frames[title]["copy"]))
        rows += [tuple(str(c) for c in row) for row in frames["Input selection"]["rows"]]
        return rows

    def _used_label(self) -> str:
        artifacts = self.metadata.get("artifacts") or {}
        used = artifacts.get("education_used_detections") if isinstance(artifacts, Mapping) else None
        source = used.get("source") if isinstance(used, Mapping) else None
        return {"phase1_pickle": "Phase 1's saved detections",
                "override_pickle": "an override file",
                "filtered_pickle": "the filtered detections",
                "fresh_pass_not_saved": "detected during this run (not saved)"}.get(
                    source, "not recorded")

    def _walkthrough_rows(self, camera: str, level: int) -> list[tuple]:
        """Recorded first-estimate views of one camera, re-checked (L3 Numbers of stage 2)."""
        walk = next((f for f in self._frames if f["title"] == "Per-view walkthrough"), None)
        if walk is None or not walk.get("camera_groups"):
            return [("No recorded views", "", "", walk["copy"] if walk else "")]
        rows = walk["camera_groups"].get(camera, [])
        return [(r[0], r[1], r[2], r[5]) for r in rows]

    def _make_frames(self) -> list[dict[str, Any]]:
        """Project evidence conservatively; never infer detections from a parent ID."""
        artifacts = self.metadata.get("artifacts", {})
        artifacts = artifacts if isinstance(artifacts, Mapping) else {}
        used = artifacts.get("education_used_detections", {})
        used = used if isinstance(used, Mapping) else {}
        source = used.get("source", "legacy_unknown")
        reasons = used.get("fresh_pass_reasons", ())
        reason_text = ", ".join(str(item).replace("_", " ") for item in reasons) or "not recorded"
        cache_hit = used.get("cache_hit")
        if source == "fresh_pass_not_saved":
            origin_text = ("Origin: detections from a reused cache; the originating run is unknown."
                           if cache_hit is True else
                           "Origin: detected during this run (not saved).")
            input_text = f"{origin_text} Fresh-pass reason: {reason_text}."
            if cache_hit is True:
                input_text += " This pass reused cached detections. No photo is shown."
                raw_text = "Raw unavailable (detections made during Phase 2, not saved); this pass reused cached detections."
            elif cache_hit is False:
                input_text += " The pass did not report a cache hit."
                raw_text = ("Raw unavailable (detections made during Phase 2, not saved); "
                            "the current photo listing is not checkable against this pass.")
            else:
                input_text += " Cache use was not recorded."
                raw_text = "Raw unavailable (detections made during Phase 2, not saved); cache use was not recorded."
            # A1 section 9: a refusal ends with its recovery step.
            raw_text += ((" Untick ‘Use detection cache when fallback-detecting’ in Phase 2 and run "
                          "Phase 2 again, then continue from its new result.")
                         if cache_hit is True else
                         " The fresh detections were not saved; re-run Phase 2 to create a new result.")
        elif source in {"phase1_pickle", "override_pickle", "filtered_pickle"}:
            path = used.get("path")
            label = {"phase1_pickle": "Phase 1's saved detections",
                     "override_pickle": "an override pickle",
                     "filtered_pickle": "the filtered detections"}[source]
            input_text = f"The run record identifies {label} as the input: {path or 'path unavailable'}."
            raw_text = "Input detections are identified by the run record; no source photo is implied."
        else:
            input_text = "Input used not recorded for this run; Phase 1 detections are not substituted."
            raw_text = ("Raw unavailable: the detections used by this run cannot be established. "
                        "Re-run Phase 2 to create a new result that records its input, then continue from that run.")

        frames = [
            {"title": "Input selection", "copy": f"Run {self.context.run_id}. {input_text} {raw_text}", "rows": self._input_rows(used)},
            {"title": "Run status", "copy": self._status_copy(), "rows": []},
        ]
        fit_frame = {"title": "Phase 2 input" if self._unmarked_promotion else "Initial camera fit",
                     "copy": self._seed_copy() + "\n\n" + self._distortion_effect_copy() + "\n\n" + self._later_distortion_copy(),
                     "detail": self._distortion_values_copy(), "rows": self._camera_rows()}
        decision_frame = {"title": "Fit error and decision", "copy": self._decision_copy(),
                          "detail": "Recalculated with the same closed-form board poses Phase 2 used, from the detections embedded in this run's saved camset. Values are checked against D2.6; Phase 2 has no optimisation-iteration series. "
                                    "D2.6 gives one value per camera and image, pooled over every board seen in that image; the per-view walkthrough instead numbers each accepted board view, so its numbers are not image numbers.",
                          "rows": self._residual_rows}
        aside = []
        params = self.metadata.get("params", {})
        params = params if isinstance(params, Mapping) else {}
        pruning = self.metadata.get("threshold_pruning")
        is_pruned = bool(pruning or artifacts.get("filtered_detection_pickle"))
        used_reasons = used.get("fresh_pass_reasons", ())
        second_pass_recorded = (isinstance(used_reasons, Sequence)
                                and "high_distortion_second_pass" in used_reasons)
        if params.get("high_distortion"):
            if is_pruned and self.metadata.get("error"):
                high_copy = ("This requested pruned rerun failed before calibration; no pruned detection copy was produced. "
                             "No second search ran. High-distortion: on in settings, not applied. "
                             f"Failure: {str(self.metadata.get('error')).rstrip().rstrip('.')}.")
            elif is_pruned:
                count = _removed_count(pruning)
                count_text = ("none removed this time" if count == 0 else
                              f"{count} observations removed" if count is not None else
                              "removal count unavailable")
                high_copy = (f"This rerun used Phase 1's saved detections, with {count_text}; "
                             "no second search ran. High-distortion: on in settings, not applied.")
            elif second_pass_recorded:
                limit = params.get("n_lim")
                limit_text = "all images" if limit is None else f"up to {limit} images per camera"
                high_copy = ("Phase 2 ran detection a second time "
                             f"({limit_text}), using first-pass camera estimates to guide detection, "
                             "then recalibrated from those detections. A picture of how the detections "
                             "moved between the two passes is not available for this pass.")
            else:
                high_copy = ("High-distortion: on in settings, not applied. "
                             "This run's record does not identify a high-distortion second pass.")
            try:
                upscale = float(params.get("upscale_factor", 1))
            except (TypeError, ValueError, OverflowError):
                upscale = 1.0
            if upscale > 1 and second_pass_recorded and not is_pruned:
                high_copy += (" Upscaled high-distortion overlay unavailable (reason): second-pass detections "
                              "use a different pixel frame; no Phase 1 points or photo are substituted.")
            aside.append({"title": "High-distortion second pass", "copy": high_copy, "rows": []})
        elif is_pruned:
            count = _removed_count(pruning)
            count_text = ("none removed this time" if count == 0 else
                          f"{count} observations removed" if count is not None else
                          "removal count unavailable")
            aside.append({"title": "Pruned rerun", "copy": f"This rerun used Phase 1's saved detections, with {count_text}; no second search ran.", "rows": []})

        if self._trace.get("status") == "available":
            camera_groups = _LazyGroups(
                {str(camera.get("camera", "camera")): camera for camera in self._trace_cameras()},
                self._walkthrough_camera_rows)
            frames.append({"title": "Per-view walkthrough", "copy": "Per-view contributions recorded for this run, recalculated from the detections Phase 2 used. Choose a camera, then read its usable views in the table, one row per view; these are views, not optimisation steps. "
                           "Each row is one accepted board view: one image can hold several board views, and images with too few detections are skipped.",
                           "detail": "The board-to-image homography and the recorded board pose of each view are shown at detail level L3.",
                           "rows": [], "camera_groups": camera_groups})
        else:
            setting = params.get("record_phase2_per_view")
            message = ("Per-view contributions not recorded for this run. To see per-view contributions next "
                       "time, tick ‘Record per-view contributions for this run’ in Phase 2 settings before "
                       "you run it. This run is not deficient."
                       if setting is not True else
                       f"Per-view contribution trace unavailable ({self._trace.get('reason', 'no buildable records')}). "
                       "Run Phase 2 again with ‘Record per-view contributions for this run’ ticked to create a new record.")
            frames.append({"title": "Per-view walkthrough", "copy": message, "rows": []})
        return [_LazyFrame(frame) for frame in (*frames, fit_frame, *aside, decision_frame)]

    def _walkthrough_camera_rows(self, camera: Mapping[str, Any]) -> list[tuple]:
        """One camera's recorded first-estimate views, each re-checked (computed on demand)."""
        rows = []
        for i, item in enumerate(camera.get("views", [])):
            if not isinstance(item, Mapping):
                continue
            view_index = item.get("view_index", i)
            camera_name = str(camera.get("camera", ""))
            recalculated, cost_status = self._pose_only_error(
                camera_name, view_index, item.get("pose_target_to_camera"),
                item.get("per_view_rms_px"))
            rows.append((self._board_view_label(camera_name, view_index), recalculated,
                         _format_number(item.get("per_view_rms_px")),
                         self._homography_status(camera_name, view_index),
                         _pose_summary(item.get("pose_target_to_camera")),
                         cost_status))
        return rows

    def _input_rows(self, used: Mapping[str, Any]) -> list[tuple[str, str, str]]:
        """Summarise camera/image/board evidence only from the saved used detection set."""
        removed = _removed_count(self.metadata.get("threshold_pruning"))
        prune_text = ("none removed this time" if removed == 0 else
                      f"{removed} points removed" if removed is not None else
                      "threshold-pruned points not recorded")
        if self._unmarked_promotion or used.get("source") == "fresh_pass_not_saved":
            return [("Detection contributions", "unavailable", prune_text)]
        try:
            from pyCamSet.utils.intrinsics_report import _board_views

            _, detection, target = self._used_inputs()
            params = self.metadata.get("params", {})
            minimum = int(params.get("min_detections_per_board", 12)) if isinstance(params, Mapping) else 12
            rows = []
            for name in detection.cam_names:
                objects, _, image_indices, _ = _board_views(detection, target, name, minimum)
                point_count = sum(int(obj.shape[1]) for obj in objects)
                rows.append((str(name), f"{len(objects)} usable board views / {len(set(image_indices))} images",
                             f"{point_count} used points; {prune_text}"))
            return rows or [("Detection contributions", "none", prune_text)]
        except _InputsUnavailable as exc:
            return [("Detection contributions", "unavailable", f"{exc}; {prune_text}")]
        except Exception:
            return [("Detection contributions", "unavailable",
                     f"the used detections could not be counted; {prune_text}")]

    def _later_distortion_copy(self) -> str:
        """Show a later solution only when its linked artifact exists on disk."""
        if self._unmarked_promotion or self.metadata.get("error"):
            return "Later solved Phase 3/4 distortion suppressed for this unverified/failed Phase 2 input."
        artifacts = self.metadata.get("artifacts", {})
        linked = artifacts.get("phase3_solved_distortion") if isinstance(artifacts, Mapping) else None
        if isinstance(linked, Mapping) and linked.get("run_id") and linked.get("path") and linked.get("coefficients") is not None:
            artifact_path = Path(str(linked["path"]))
            if not artifact_path.is_absolute():
                artifact_path = Path(self.context.manager.workspace_path) / artifact_path
            if artifact_path.is_file():
                return (f"Later solved Phase 3/4 distortion from run {linked['run_id']}; "
                        f"artifact {artifact_path}; coefficients {self._coefficient_text(linked['coefficients'])}.")
        return "Later solved Phase 3/4 distortion unavailable: no separately linked artifact exists on disk."

    def _distortion_effect_copy(self) -> str:
        """Describe first-estimate distortion while refusing an unobserved counterfactual."""
        if self._unmarked_promotion or self.metadata.get("error"):
            return "Distortion effect not shown for this unverified or failed Phase 2 input."
        distortions = self._saved_distortions()
        if not distortions:
            return "Distortion effect unavailable: Phase 2 distortion coefficients were not recorded."
        if self._distortion_not_fitted():
            return ("Phase 2 does not fit lens distortion: its first estimate assumes none, so these "
                    "coefficients are zero placeholders, not a finding that the lens is distortion-free. "
                    "Distortion is first solved in Phase 3, unless you lock it there. "
                    "No with/without-distortion effect is inferred from a single run.")
        return ("The distortion coefficients are Phase 2's saved first estimates, not a later solution. "
                "No with/without-distortion effect is inferred from a single run.")

    @staticmethod
    def _all_zero(coeffs: Any) -> bool:
        """True when a saved coefficient list is present and every value is exactly zero."""
        if isinstance(coeffs, Mapping):
            coeffs = list(coeffs.values())
        if not isinstance(coeffs, Sequence) or isinstance(coeffs, (str, bytes)) or not coeffs:
            return False
        try:
            return all(float(value) == 0.0 for value in coeffs)
        except (TypeError, ValueError, OverflowError):
            return False

    def _distortion_not_fitted(self) -> bool:
        """Zhang/affine seeds hand on zero distortion; a nonzero value means an older fitted seed."""
        distortions = self._saved_distortions()
        if not distortions:
            return False
        return all(self._all_zero(values.get("coeffs") if isinstance(values, Mapping) else values)
                   for values in distortions.values())

    def _saved_distortions(self) -> Mapping[str, Any]:
        diagnostics = self.metadata.get("diagnostics", {})
        distortions = diagnostics.get("D2.3_distortion", {}) if isinstance(diagnostics, Mapping) else {}
        return distortions if isinstance(distortions, Mapping) else {}

    def _distortion_values_copy(self) -> str:
        """L2: one readable line of saved coefficients per camera (D2.3)."""
        if self._unmarked_promotion or self.metadata.get("error"):
            return ""
        lines = []
        for name, values in self._saved_distortions().items():
            coeffs = values.get("coeffs") if isinstance(values, Mapping) else values
            suffix = " (saved as zero, not fitted)" if self._all_zero(coeffs) else ""
            lines.append(f"{name}: {self._coefficient_text(coeffs)}{suffix}")
        return ("Saved distortion coefficients (D2.3): " + "; ".join(lines) + ".") if lines else ""

    def _coefficient_text(self, coeffs: Any) -> str:
        """Name coefficients in OpenCV order for pinhole lenses, never as a raw container."""
        if isinstance(coeffs, Mapping):
            return ", ".join(f"{key} = {_format_number(value)}" for key, value in coeffs.items()) or "none"
        if not isinstance(coeffs, Sequence) or isinstance(coeffs, (str, bytes)):
            return "unavailable"
        model = self._lens_model()
        names = (_OPENCV_DISTORTION_NAMES if model != "telecentric" and len(coeffs) <= len(_OPENCV_DISTORTION_NAMES)
                 else tuple(f"coefficient {index + 1}" for index in range(len(coeffs))))
        return ", ".join(f"{name} = {_format_number(value)}" for name, value in zip(names, coeffs)) or "none"

    def _status_copy(self) -> str:
        """Use the shared quality rule when older Phase 2 runs lack status."""
        status = self.metadata.get("status")
        if status:
            return f"Run {self.context.run_id} status: {status}."
        reasons = blocking_reasons(dict(self.metadata))
        if reasons:
            return (f"Run {self.context.run_id} status was not recorded. "
                    "The shared run-quality rule reports blocking reasons: "
                    + "; ".join(reasons))
        return (f"Run {self.context.run_id} status was not recorded. "
                "The shared run-quality rule found no blocking reasons; this is not a recorded status.")

    def _seed_copy(self) -> str:
        if self.metadata.get("error"):
            return (f"Phase 2 run {self.context.run_id} failed before producing first lens estimates: "
                    f"{self.metadata.get('error')}")
        if self._unmarked_promotion:
            return (f"Run {self.context.run_id} came from an unmarked Optimisation-tab trial. "
                    "Its Phase 2 camset may contain later-stage cameras, so Phase 2's first lens estimates are not shown.")
        params = self.metadata.get("params", {})
        params = params if isinstance(params, Mapping) else {}
        notes = self.metadata.get("calibration_notes", ())
        note_text = " " + " ".join(str(note) for note in notes) if isinstance(notes, Sequence) and not isinstance(notes, str) else ""
        return (f"First lens estimates for run {self.context.run_id}. Values below are the saved Phase 2 camera estimates, "
                "not a later Phase 3/4 solution. Phase 2 builds no multi-camera rig. "
                f"Lens model: {params.get('lens_model', 'not recorded')}. Distortion is shown as saved; "
                "it is not assumed to be the later solved lens." + note_text)

    def _camera_rows(self, plain: bool = False) -> list[tuple[str, str, str]]:
        """Saved lens values; ``plain`` (L1) uses plain names instead of fx/fy/cx/cy and k1…."""
        if self._unmarked_promotion or self.metadata.get("error"):
            return []
        diagnostics = self.metadata.get("diagnostics", {})
        diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
        intrinsics = diagnostics.get("D2.2_intrinsics", {})
        distortions = diagnostics.get("D2.3_distortion", {})
        if not isinstance(intrinsics, Mapping):
            return []
        rows = []
        for name, values in intrinsics.items():
            if not isinstance(values, Mapping):
                continue
            distortion = distortions.get(name, {}) if isinstance(distortions, Mapping) else {}
            coeffs = distortion.get("coeffs") if isinstance(distortion, Mapping) else None
            if plain:
                number = {key: _format_number(values.get(key)) for key in ("fx", "fy", "cx", "cy")}
                k = (f"focal length (x, y): {number['fx']}, {number['fy']} px; "
                     f"image centre (x, y): {number['cx']}, {number['cy']} px")
                rows.append((str(name), k, _plain_distortion_text(coeffs)))
                continue
            k = ", ".join(f"{key}={_format_number(values.get(key))}" for key in ("fx", "fy", "cx", "cy"))
            rows.append((str(name), k, self._coefficient_text(coeffs) if coeffs is not None else "unavailable"))
        return rows

    def _residual_rows(self) -> list[tuple[str, str, str]]:
        """Recompute D2.6 from the run's saved camset; never use a current glob."""
        if self.metadata.get("error"):
            return [("All cameras", "unavailable", "Phase 2 run failed")]
        if self._unmarked_promotion:
            return [("All cameras", "unavailable", "unmarked promoted camset may contain later-stage state")]
        diagnostics = self.metadata.get("diagnostics", {})
        expected = diagnostics.get("D2.6_per_view_reprojection", {}) if isinstance(diagnostics, Mapping) else {}
        try:
            from pyCamSet.utils.intrinsics_report import report_initial_calibration

            cams, detections, target = self._used_inputs()
            report = report_initial_calibration(
                cams, detections, target,
                int((self.metadata.get("params") or {}).get("min_detections_per_board", 12)),
            )
            values = report.per_view
        except _InputsUnavailable as exc:
            return [("All cameras", "unavailable", str(exc))]
        except Exception:
            return [("All cameras", "unavailable",
                     "the fit error could not be recalculated from the saved inputs")]
        rows = []
        if isinstance(values, Mapping):
            for camera, series in values.items():
                if isinstance(series, Mapping):
                    errors = series.get("rms_px", ())
                else:
                    errors = series
                if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
                    expected_series = expected.get(camera, {}) if isinstance(expected, Mapping) else {}
                    expected_errors = (expected_series.get("rms_px", ())
                                       if isinstance(expected_series, Mapping) else expected_series)
                    image_indices = (series.get("image_indices") if isinstance(series, Mapping) else None)
                    for index, error in enumerate(errors):
                        image = (image_indices[index]
                                 if isinstance(image_indices, Sequence) and index < len(image_indices)
                                 else index)
                        label = f"{camera} · {es.photo_label(image)} (index {image})"
                        saved = expected_errors[index] if index < len(expected_errors) else None
                        if index < len(expected_errors) and _is_missing(error) and _is_missing(saved):
                            rows.append((label, "no usable board in this image",
                                         "matches D2.6: none recorded"))
                            continue
                        match = (index < len(expected_errors)
                                 and _numbers_match(error, saved))
                        rows.append((label, _format_number(error),
                                     "matches D2.6" if match else "does not match D2.6"))
        return rows

    def _trace_cameras(self) -> list[Mapping[str, Any]]:
        records = self._trace.get("records", ())
        for record in records:
            if record.get("kind") == "phase2_per_view":
                payload = record.get("payload", {})
                cameras = payload.get("cameras", ()) if isinstance(payload, Mapping) else ()
                return [camera for camera in cameras if isinstance(camera, Mapping)]
        return []

    def _is_unmarked_promotion(self) -> bool:
        """Do not attribute contaminated trial camsets to the Phase 2 stage."""
        diagnostics = self.metadata.get("diagnostics", {})
        diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
        education = self.metadata.get("education", {})
        education = education if isinstance(education, Mapping) else {}
        marker = education.get("trial_camset_snapshot_version")
        marked = type(marker) is int and marker == 1
        return bool(diagnostics.get("promoted_from") or self.metadata.get("promoted_from")) and not marked

    def _homography_status(self, camera_name: str, view_index: Any) -> str:
        """Fit a homography from the selected camset's embedded detections only."""
        params = self.metadata.get("params", {})
        model = params.get("lens_model", "pinhole") if isinstance(params, Mapping) else "pinhole"
        if model == "telecentric":
            return "not applicable (telecentric affine fit)"
        try:
            from pyCamSet.calibration.zhang import homography_of
            from pyCamSet.utils.intrinsics_report import _board_views

            try:
                _, detection, target = self._used_inputs()
            except _InputsUnavailable as exc:
                return f"unavailable ({exc})"
            minimum = int((params or {}).get("min_detections_per_board", 12))
            objects, images, image_indices, _ = _board_views(
                detection, target, camera_name, minimum)
            index = int(view_index)
            if index < 0 or index >= len(objects):
                return "unavailable (view has no saved correspondences)"
            matrix = homography_of(objects[index], images[index])
            if matrix.shape != (3, 3):
                return "unavailable (invalid fit)"
            matrix_text = "; ".join(
                ",".join(_format_number(v) for v in row) for row in matrix)
            return f"recalculated: board view {index}: {matrix_text}"
        except Exception:
            return "unavailable (the homography could not be fitted to this view's points)"

    def _board_view_label(self, camera_name: str, view_index: Any) -> str:
        """Name a walkthrough row by its view, with the image it came from.

        Telecentric first estimates solve one view per *image* (every kept face together),
        so their rows are image views; pinhole ones solve one per board.
        """
        cache = self.__dict__.setdefault("_board_view_images", {})
        telecentric = self._lens_model() == "telecentric"
        if camera_name not in cache:
            cache[camera_name] = None
            try:
                _, detection, target = self._used_inputs()
                if telecentric:
                    cache[camera_name] = [view.image_index for view in vp.seed_views(
                        detection, target, camera_name, self._minimum(), True)]
                else:
                    from pyCamSet.utils.intrinsics_report import _board_views
                    _, _, image_indices, _ = _board_views(
                        detection, target, camera_name, self._minimum())
                    cache[camera_name] = list(image_indices)
            except Exception:
                cache[camera_name] = None
        images = cache[camera_name]
        kind = "Image view" if telecentric else "Board view"
        try:
            index = int(view_index)
        except (TypeError, ValueError):
            return f"{kind} {view_index}"
        if images is not None and 0 <= index < len(images):
            return f"{kind} {index} · {es.photo_label(images[index])}"
        return f"{kind} {index}"

    def _pose_only_error(self, camera_name: str, view_index: Any,
                         pose_value: Any, recorded_cost: Any) -> tuple[str, str]:
        """Reproject this saved first-estimate pose and compare its RMS with the saved value."""
        if self._lens_model() == "telecentric":
            return self._telecentric_pose_only_error(camera_name, view_index, pose_value, recorded_cost)
        try:
            import cv2
            from pyCamSet.utils.intrinsics_report import _board_views

            cams, detection, target = self._used_inputs()
            objects, images, _, _ = _board_views(detection, target, camera_name, self._minimum())
            index = int(view_index)
            if index < 0 or index >= len(objects):
                raise _InputsUnavailable("this view has no saved point matches")
            pose = np.asarray(pose_value, dtype=float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise _InputsUnavailable("the recorded board pose for this view is not valid")
            rotation, _ = cv2.Rodrigues(pose[:3, :3])
            camera = cams[camera_name]
            projected, _ = cv2.projectPoints(
                np.asarray(objects[index]).reshape(-1, 3), rotation, pose[:3, 3],
                np.asarray(camera.intrinsic, dtype=float),
                np.asarray(camera.distortion_coefs, dtype=float).reshape(-1))
            residual = projected.reshape(-1, 2) - np.asarray(images[index]).reshape(-1, 2)
            error = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
            return _format_number(error), _rms_comparison(error, recorded_cost)
        except _InputsUnavailable as exc:
            return "unavailable", f"not recalculated: {exc}"
        except Exception:
            return "unavailable", "not recalculated: the saved inputs could not be reprojected"

    def _telecentric_pose_only_error(self, camera_name: str, view_index: Any,
                                     pose_value: Any, recorded_cost: Any) -> tuple[str, str]:
        """Reproject a recorded telecentric first-estimate pose through the telecentric model.

        The telecentric first estimate solves one view per *image* (all kept faces together,
        in the target frame; flat images skipped), so the recorded ``view_index`` numbers those
        image views (``education_visual_p2.seed_views``), not board views. (On a real telecentric
        rig, indexing board views reported 140-270 px "differences" on 2.5 px values.)
        """
        try:
            cams, detection, target = self._used_inputs()
            views = vp.seed_views(detection, target, camera_name, self._minimum(), True)
            index = int(view_index)
            if index < 0 or index >= len(views):
                raise _InputsUnavailable("this view has no saved point matches")
            pose = np.asarray(pose_value, dtype=float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise _InputsUnavailable("the recorded board pose for this view is not valid")
            error = vp.seed_view_rms(cams[camera_name], views[index], pose)
        except _InputsUnavailable as exc:
            return "unavailable", f"not recalculated: {exc}"
        except Exception:
            return "unavailable", "not recalculated: the saved inputs could not be reprojected"
        return _format_number(error), _rms_comparison(error, recorded_cost)

    def _used_inputs(self) -> tuple[Any, Any, Any]:
        """Return (saved camset, used detections, target) or raise a plain reason.

        Phase 2 saves its camset without a calibration handler, so the used
        detections come from the file the run record names and the target
        from the run's own settings; a handler embedded in the camset wins.
        Only a successful load is kept, so a later retry can still succeed.
        """
        cached = self.__dict__.get("_used_inputs_cache")
        if cached is not None:
            return cached
        from pyCamSet.utils.saving import load_CameraSet

        try:
            artifact = resolve_artifact(dict(self.metadata), "phase2", self.context.manager.workspace_path)
            cams = load_CameraSet(artifact)
        except Exception:
            raise _InputsUnavailable("this run's saved camera file could not be read") from None
        handler = getattr(cams, "calibration_handler", None)
        detection, target = getattr(handler, "detection", None), getattr(handler, "target", None)
        if detection is None or target is None:
            detection, target = self._recorded_detections_and_target()
        self._used_inputs_cache = (cams, detection, target)
        return self._used_inputs_cache

    def _recorded_detections_and_target(self) -> tuple[Any, Any]:
        """Read the used detections from the file the run record names."""
        artifacts = self.metadata.get("artifacts", {})
        used = artifacts.get("education_used_detections", {}) if isinstance(artifacts, Mapping) else {}
        used = used if isinstance(used, Mapping) else {}
        source, path = used.get("source"), used.get("path")
        if source == "fresh_pass_not_saved":
            raise _InputsUnavailable("Phase 2 detected these points during the run and did not save them")
        if source not in {"phase1_pickle", "override_pickle", "filtered_pickle"} or not path:
            raise _InputsUnavailable("this run did not record which detections it used")
        detections_path = Path(str(path))
        if not detections_path.is_absolute():
            detections_path = Path(self.context.manager.workspace_path) / detections_path
        try:
            from pyCamSet.utils.paths import long_path
            from pyCamSet.utils.saving import load_pickle
            from pyCamSet.workflow.detections import extract_detection_and_cam_res

            detection, _ = extract_detection_and_cam_res(load_pickle(long_path(detections_path)))
        except Exception:
            raise _InputsUnavailable(
                f"the detections this run used could not be read from {detections_path.name}") from None
        try:
            from pyCamSet.workflow.targets import target_of_params

            params = self.metadata.get("params", {})
            target = target_of_params(dict(params) if isinstance(params, Mapping) else {})
        except Exception:
            raise _InputsUnavailable("the target in this run's settings could not be rebuilt") from None
        return detection, target

    def _decision_copy(self) -> str:
        """Answer A1's Phase 2 question from the run's saved per-camera summary."""
        question = ("Did each camera see the board in enough photos, with a fit error below "
                    "pyCamSet's warning level?")
        if self.metadata.get("error") or self._unmarked_promotion:
            return (f"{question} Not checkable for this run: it failed or came from an unmarked "
                    "Optimisation-tab trial. Run Phase 2 again to create a checkable result.")
        report = self.metadata.get("report", {})
        rows = report.get("per_camera") if isinstance(report, Mapping) else None
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
            return (f"{question} Not checkable here: this run's record has no per-camera fit summary. "
                    "Run Phase 2 again to record one; the table below still lists the recalculated "
                    "per-image values.")
        from pyCamSet.utils.intrinsics_report import HIGH_INITIAL_RMS_PX

        level = float(HIGH_INITIAL_RMS_PX)
        lines = [question]
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            name = row.get("name", "camera")
            views = row.get("n_views")
            error = row.get("rms_px")
            if not views or _is_missing(error):
                lines.append(f"{name}: no usable board view, so this camera has no lens estimate; "
                             "pyCamSet does not carry it into Phase 3.")
                continue
            side = ("above" if float(error) > level else "at or below")
            lines.append(f"{name}: board seen in {views} photo(s); fit error {_format_number(error)} px, "
                         f"{side} pyCamSet's {level:g} px warning level.")
        lines.append(f"The {level:g} px warning level comes from pyCamSet's own check of a first lens "
                     "estimate; a camera above it gets a warning, not a block. pyCamSet blocks only a "
                     "camera with no usable view: it sets no other minimum number of photos and does not "
                     "check how varied the board angles were.")
        if not self._saved_distortions() or self._distortion_not_fitted():
            lines.append("Part of a camera's fit error can come from lens distortion, which Phase 2 does "
                         "not fit: a wide-angle lens can be above the warning level even with good "
                         "detections. Phase 3 solves the distortion, unless you lock it there.")
        lines.append("What to do next: if a camera is above the warning level or saw the board in few "
                     "photos, check its detections in Phase 1, or add photos with the board at other "
                     "angles and positions in that camera's view, then run Phases 1 and 2 again.")
        return "\n".join(lines)


# ======================================================================================
# The 3D stage widget

#: Camera axes (x right, y down, z forward) -> scene axes (x right, y forward, z up), so both
#: the PyVista and the static backend can use their natural z-up view.
_TO_WORLD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
#: Telecentric: every pose is drawn with its centre this many target sizes from the lens.
_TELECENTRIC_DEPTH = 1.9


class _BoardScene(QWidget):
    """Stage 2: the 3D scene (PyVista or static) under a titled header, beside a side figure
    with "one photo -> one pose" (B) and which photos were posed / each view's error (C)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from pyCamSet.gui.education_scene3d import Scene3D
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 6, 4, 0)
        outer.setSpacing(8)
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(0)
        self.title = QLabel(self)
        self.title.setObjectName("eduSceneTitle")
        self.title.setTextFormat(Qt.TextFormat.RichText)
        self.title.setWordWrap(True)
        self.title.setStyleSheet("background: transparent;")
        left.addWidget(self.title)
        self.scene = Scene3D(self)
        left.addWidget(self.scene, 1)
        outer.addLayout(left, 3)
        self.side_figure = Figure(figsize=(4.2, 5), dpi=100, layout="constrained")
        self.side = FigureCanvasQTAgg(self.side_figure)
        from pyCamSet.gui.education_perf import cap_canvas_pixel_ratio
        cap_canvas_pixel_ratio(self.side)
        self.side.setParent(self)
        self.side.setMinimumWidth(330)
        self.side.setMaximumWidth(560)
        self.side.setAccessibleName("One photo gives one pose")
        outer.addWidget(self.side, 2)
        self._args: tuple | None = None
        self._extra: dict[str, Any] = {}
        self._theme = es.active_theme_name()

    def set_theme(self, name: str) -> None:
        self._theme = es.resolve_theme(name)
        if self._args is not None:
            self._render()
        else:
            self.scene.set_theme(self._theme)

    def show_camera(self, data: _CameraData, items: list, selected: int | None, level: int,
                    telecentric: bool, image_size: tuple[int, int], *, warp: Any = None,
                    photo: Any = None, image: int | None = None,
                    photos: Sequence[int] = (), face_mm: float | None = None) -> None:
        self._args = (data, items, selected, level, telecentric, image_size)
        self._extra = {"warp": warp, "photo": photo, "image": image, "photos": list(photos),
                       "face_mm": face_mm}
        self._theme = es.active_theme_name()
        self._render()

    def _render(self) -> None:
        data, items, selected, level, telecentric, image_size = self._args
        t = es.stage_tokens(self._theme)
        shown = build_board_scene(self.scene, data, items, selected, level, telecentric,
                                  image_size, self._theme)
        image = self._extra.get("image")
        hint = " · drag to turn, scroll to zoom" if self.scene.backend == "pyvista" else ""
        chosen = (f"{es.photo_label(image)} solid" if selected is not None and image is not None
                  else "")
        if telecentric and len(shown) == 1 and chosen:
            what, chosen = f"{es.photo_label(image)} (1 of {len(items)} cube poses)", ""
            rays = "at one distance: no depth seen (grey ↔)"
        elif telecentric:
            what = f"{len(shown)} of {len(items)} cube poses"
            rays = "at one distance: no depth seen (grey ↔)"
        else:
            what = f"{len(shown)} of {len(items)} board views" if len(shown) < len(items) \
                else f"{len(items)} board views"
            rays = "each at the distance Phase 2 found"
        parts = [f"camera {data.index + 1}", what] + ([chosen] if chosen else []) + [rays]
        self.title.setText(
            f"<span style='font-size:13pt; font-weight:600; color:{t['text']}'>"
            f"A{es.LETTER_GAP.replace(' ', '&nbsp;')}Where the target was</span><br>"
            f"<span style='font-size:10pt; color:{t['muted']}'>{' · '.join(parts)}{hint}</span>")
        self._draw_side()

    def _draw_side(self) -> None:
        import matplotlib
        data, items, selected, level, telecentric, image_size = self._args
        figure = self.side_figure
        figure.clear()
        figure._edu_theme = self._theme
        with matplotlib.rc_context(es.figure_style(self._theme)):
            figure.set_facecolor(es.stage_tokens(self._theme)["stage"])
            es.align_panel_titles(figure)
            gs = figure.add_gridspec(2, 1, height_ratios=[1.45, 1.0])
            _one_view_pose(figure, gs[0], self._extra.get("warp"), self._extra.get("photo"),
                           self._extra.get("image"), image_size, telecentric,
                           self._extra.get("face_mm"))
            ax = figure.add_subplot(gs[1])
            if telecentric:
                _posed_funnel(ax, len(self._extra.get("photos") or ()), len(items),
                              sum(1 for f in data.fits if f is not None))
            else:
                _error_histogram(ax, items, selected)
        self.side.draw_idle()


def _shown_items(items: Sequence[Any], selected: int | None, level: int,
                 telecentric: bool) -> list[int]:
    if telecentric and selected is not None:
        # A telecentric pose has no depth, so every pose would sit at one distance, piled
        # into one blob: the chosen photo's pose is drawn alone (C counts the others).
        return [selected]
    limit = ({1: 8, 2: 10, 3: 12} if telecentric else {1: 40, 2: 40, 3: 60})[level]
    keep = _sample(list(range(len(items))), limit)
    if selected is not None and selected not in keep:
        keep[int(np.argmin([abs(k - selected) for k in keep]))] = selected
        keep = sorted(set(keep))
    return keep


def build_board_scene(scene: Any, data: _CameraData, items: Sequence[Any], selected: int | None,
                      level: int, telecentric: bool, image_size: tuple[int, int],
                      theme: str) -> list[int]:
    """Fill a Scene3D/OffscreenScene; returns the indices of the items drawn.

    The camera sits at the left looking right (scene y = camera z), with its real field of
    view (a parallel tube or a cone from the lens). Each item is a board view (pinhole, at its
    own distance) or the whole target in one photo (telecentric), filled by its error on the
    shared log scale. A telecentric lens cannot see depth, so every pose is drawn with its
    centre at one distance (ED-14), the chosen photo solid and the others faint, beside a grey
    double arrow (depth unknown). L2 adds rays to the chosen photo's corners (parallel for a
    telecentric lens); L3 colours those corners by their own error.
    """
    scene.clear()
    scene.set_theme(theme)
    t = es.stage_tokens(theme)
    scale = es.error_scale(theme)
    kw = scale.scene_kwargs()
    shown = _shown_items(items, selected, level, telecentric)
    matrix = np.asarray(data.cam.intrinsic, dtype=float)
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    width, height = image_size
    size = max((float(np.ptp(np.vstack(items[k].faces), axis=0).max()) for k in shown),
               default=0.01)
    offsets = {}
    if telecentric:  # no depth: every pose's centre at one display distance
        for k in shown:
            centre = float(np.vstack(items[k].faces)[:, 2].mean())
            offsets[k] = _TELECENTRIC_DEPTH * size - centre

    def world(points: np.ndarray, k: int | None = None) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3).copy()
        if k is not None and k in offsets:
            pts[:, 2] += offsets[k]
        return pts @ _TO_WORLD.T

    far = 0.0
    centres, errors = [], []
    for k in shown:
        item = items[k]
        chosen = k == selected
        for f, face in enumerate(item.faces):
            corners = world(face, k)
            far = max(far, float(corners[:, 1].max()))
            if telecentric:
                opacity = 0.85 if chosen else (0.1 if selected is not None else 0.4)
            else:
                opacity = 0.9 if chosen else 0.3
            scene.add_board(corners, scalars=item.error, name=f"item{k}_{f}", opacity=opacity,
                            point_size=2.0 if chosen else 1.0, **kw)
        if np.isfinite(item.error):
            centres.append(world(np.vstack(item.faces), k).mean(axis=0))
            errors.append(float(item.error))
    far = max(far, 1e-6) * 1.12
    sensor = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    rays = np.column_stack([(sensor[:, 0] - cx) / fx, (sensor[:, 1] - cy) / fy, np.ones(4)])
    if telecentric:
        glyph = float(np.ptp(rays[:, 0])) / 0.84  # prism cross-section = the field
        starts = np.column_stack([rays[:, :2], np.zeros(4)])
        ends = np.column_stack([rays[:, :2], np.full(4, far)])
    else:
        glyph = 0.16 * far
        starts = np.zeros((4, 3))
        ends = rays * far
    # x_cam = R @ x_world with R = _TO_WORLD.T: the camera looks along scene +y.
    scene.add_camera("camera", data.index, _TO_WORLD.T, np.zeros(3), scale=glyph,
                     telecentric=telecentric)
    scene.add_rays(world(starts), world(ends), name="field", colour=t["muted"], width=2.0,
                   opacity=0.75)
    if centres:
        # the colour key (shared log scale, px): a dot at each pose's centre
        scene.add_points(np.asarray(centres), np.asarray(errors), name="pose_errors", size=3.0,
                         scalar_bar="px", **kw)
    if telecentric and shown:
        pts = np.vstack([world(np.vstack(items[k].faces), k) for k in shown])
        x = float(pts[:, 0].max()) + 0.45 * size
        z = float(pts[:, 2].mean())
        y0, y1 = _TELECENTRIC_DEPTH * size - 1.2 * size, _TELECENTRIC_DEPTH * size + 1.2 * size
        head = 0.22 * size
        origins = [[x, y0, z], [x, y0, z], [x, y0, z], [x, y1, z], [x, y1, z]]
        ends_ = [[x, y1, z], [x + head * 0.6, y0 + head, z], [x - head * 0.6, y0 + head, z],
                 [x + head * 0.6, y1 - head, z], [x - head * 0.6, y1 - head, z]]
        scene.add_rays(np.asarray(origins), np.asarray(ends_), name="depth_unknown",
                       colour=t["muted"], width=3.0, opacity=0.95)
    if selected is not None and level >= 2 and selected in shown:
        item = items[selected]
        corners = world(item.points, selected)
        if telecentric:
            origins = world(np.column_stack([item.points[:, :2], np.zeros(len(item.points))]))
        else:
            origins = np.zeros_like(corners)
        step = max(1, len(corners) // 36)
        scene.add_rays(origins[::step], corners[::step], name="chosen_rays", colour=t["link"],
                       width=1.4, opacity=0.6)
        if level >= 3:
            scene.add_points(corners, _corner_errors(data, item.pixels), name="chosen_errors",
                             size=10, **kw)
    # Round 2 (UX-10, "half-empty Board in 3D"): look across the camera-to-target line, so
    # the long scene runs left to right and fills the wide panel, with a tight margin.
    view = scene.fit_view(azimuth=-26 if telecentric else -30,
                          elevation=42 if telecentric else 34, margin=0.06)
    if getattr(scene, "backend", "pyvista") != "pyvista" and view is not None:
        # the static fallback zooms a fixed 1.3x into its box, which crops this long
        # side view at the camera end: zoom out to keep the camera in the picture
        scene.set_view(view[0], view[1], zoom=0.85)
    return shown


def _corner_errors(data: _CameraData, pixels: np.ndarray) -> np.ndarray:
    """Each corner's own reprojection error (px), matched to the fits by its detected pixel."""
    lookup = {}
    for f in data.fits:
        if f is None:
            continue
        sizes = np.hypot(*f.residuals.T)
        for (u, v), e in zip(np.round(f.view.image_points, 3), sizes):
            lookup[(u, v)] = e
    return np.array([lookup.get((u, v), np.nan) for u, v in np.round(pixels, 3)])


def _one_view_pose(fig: Any, spec: Any, warp: Any, photo: Any, image: int | None,
                   image_size: tuple[int, int], telecentric: bool,
                   face_mm: float | None = None) -> None:
    """Panel B (ED-15): the printed face and the same face warped onto the chosen photo.

    The warp is fitted from the printed corners to the detected ones; its shape is what holds
    the pose (affine: tilt and turn; homography: tilt, turn and distance)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import ConnectionPatch, Polygon
    sub = spec.subgridspec(2, 2, height_ratios=[0.001, 1.0], width_ratios=[0.8, 1.0],
                           wspace=0.25)
    head = fig.add_subplot(sub[0, :])
    head.set_axis_off()
    kind = "warp: tilt and turn (no distance)" if telecentric else \
        "perspective warp: tilt, turn and distance"
    es.panel_title(head, "B", "One photo → one pose", kind)
    left, right = fig.add_subplot(sub[1, 0]), fig.add_subplot(sub[1, 1])
    t = es._tokens_for(left)
    if warp is None:
        left.set_axis_off()
        es.empty_panel(right, _wrap("The chosen photo has no face with enough corners.", 26))
        return
    printed = warp.printed * 1000.0
    outline = warp.outline * 1000.0
    for ax in (left, right):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    left.add_patch(Polygon(outline, closed=True, facecolor=t["grid"], edgecolor=t["text"],
                           lw=1.4, zorder=1))
    left.scatter(printed[:, 0], printed[:, 1], s=7, color=t["link"], linewidths=0, zorder=2)
    pad = 0.08 * float(np.ptp(outline, axis=0).max())
    left.set_xlim(outline[:, 0].min() - pad, outline[:, 0].max() + pad)
    left.set_ylim(outline[:, 1].max() + pad, outline[:, 1].min() - pad)
    for spine in left.spines.values():
        spine.set_visible(False)
    # SCI2-21 / ED2-22: the corners span a grid smaller than the face they are printed on
    grid_mm = float(np.ptp(outline[:, 0]))
    face_text = (f"\n(face {face_mm:g} mm)" if face_mm and grid_mm < face_mm <= 2.0 * grid_mm
                 else "")
    left.set_xlabel(f"corner grid {grid_mm:.0f} mm{face_text}", fontsize=es.TICK_SIZE)
    landed = warp.warp(warp.printed)
    edge = warp.warp(warp.outline)
    box = np.vstack([edge, warp.pixels])
    lo, hi = box.min(axis=0), box.max(axis=0)
    margin = 0.18 * float(max(hi - lo))
    x0, x1 = max(-0.5, lo[0] - margin), min(image_size[0] - 0.5, hi[0] + margin)
    y0, y1 = max(-0.5, lo[1] - margin), min(image_size[1] - 0.5, hi[1] + margin)
    if photo is not None:
        right.imshow(photo, cmap="gray", vmin=0, vmax=255, interpolation="antialiased",
                     extent=(-0.5, photo.shape[1] - 0.5, photo.shape[0] - 0.5, -0.5), zorder=0)
    else:
        right.set_facecolor(t["grid"])
    right.add_patch(Polygon(edge, closed=True, facecolor="none", edgecolor=t["link"], lw=1.6,
                            zorder=3))
    right.scatter(landed[:, 0], landed[:, 1], s=7, color=t["link"], linewidths=0, zorder=4)
    radius = 0.022 * float(max(x1 - x0, y1 - y0))
    from matplotlib.collections import EllipseCollection
    rings = EllipseCollection(np.full(len(warp.pixels), 2 * radius),
                              np.full(len(warp.pixels), 2 * radius), np.zeros(len(warp.pixels)),
                              units="xy", offsets=warp.pixels, offset_transform=right.transData,
                              facecolors="none", edgecolors=t["accent"], linewidths=1.2,
                              zorder=5)
    right.add_collection(rings)
    right.set_xlim(x0, x1)
    right.set_ylim(y1, y0)
    for spine in right.spines.values():
        spine.set_visible(False)
    label = es.photo_label(image) if image is not None else "the photo"
    right.set_xlabel(f"on {label} · warp off by {es.format_px(warp.rms)}",
                     fontsize=es.TICK_SIZE)
    arrow = ConnectionPatch(xyA=(1.03, 0.5), coordsA=left.transAxes, xyB=(-0.03, 0.5),
                            coordsB=right.transAxes, arrowstyle="-|>", color=t["muted"],
                            lw=1.6, mutation_scale=14)
    fig.add_artist(arrow)
    right.legend(handles=[
        Line2D([], [], marker="o", lw=0, markersize=6, markerfacecolor="none",
               markeredgecolor=t["accent"], label="found"),
        Line2D([], [], marker="o", lw=0, markersize=4, color=t["link"], label="printed, warped")],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2, borderaxespad=0.0,
        handletextpad=0.3, columnspacing=0.8, fontsize=es.TICK_SIZE)


def _posed_funnel(ax: Any, photos: int, posed: int, face_poses: int | None = None) -> None:
    """Panel C (telecentric, ED-14 / ED2-9): the two kinds of pose. Top: every board view
    gets a face pose (these fit the lens and give every photo its error). Below: the photos,
    split into those with a cube pose (2+ faces, drawn in A) and those with one face only
    (face pose only: one flat face can tilt two ways, so no cube pose)."""
    t = es._tokens_for(ax)
    one = max(photos - posed, 0)
    face_poses = photos if face_poses is None else int(face_poses)
    top = max(face_poses, photos, 1)
    # round 9: a lighter bar, so the dark label inside it reads in Sepia as in Light
    ax.barh(1, face_poses, height=0.56, color=t["muted"], alpha=0.45, zorder=3)
    ax.annotate(f"{face_poses:,} faces → the lens fit and every photo's error",
                (0, 1), xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                fontsize=es.TICK_SIZE, color=es.WHITE if es.is_dark_axes(ax) else "#000000",
                zorder=5)
    ax.barh(0, posed, height=0.56, color=t["accent"], zorder=3)
    ax.barh(0, one, left=posed, height=0.56, color=t["rejected"], alpha=es.REJECTED_ALPHA,
            zorder=3)
    for left, value, words, colour in ((0, posed, f"{posed} cube pose", t["text"]),
                                       (posed, one, f"{one} one face", t["text"])):
        if value:
            ax.annotate(words, (left + value / 2, -0.28), xytext=(0, -3),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=es.TICK_SIZE, color=colour)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["face poses", "photos"])
    ax.set_ylim(-0.95, 1.5)
    ax.set_xlim(0, top * 1.04)
    ax.set_xlabel("Count")
    es.integer_axis(ax, "x")
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    es.panel_title(ax, "C", "Two kinds of pose",
                   "every face: face pose · 2+ faces: cube pose")


def _error_histogram(ax: Any, items: Sequence[Any], selected: int | None) -> None:
    """Panel C (pinhole): every view's RMS error on the shared log scale (the scene's key)."""
    t = es._tokens_for(ax)
    scale = es.error_scale(es.axes_theme(ax))
    errors = np.array([i.error for i in items if np.isfinite(i.error)])
    if not errors.size:
        es.empty_panel(ax, "No view has an error.")
        es.panel_title(ax, "C", "Error of each view", "")
        return
    lo, hi = scale.log_range(errors)
    edges = np.geomspace(lo, hi, 15)
    counts, _ = np.histogram(np.clip(errors, lo, hi), bins=edges)
    centres = np.sqrt(edges[:-1] * edges[1:])
    ax.barh(centres, counts, height=np.diff(edges) * 0.88, color=scale.colours(centres),
            zorder=3)
    scale.log_axis(ax, "y")
    ax.set_ylim(lo, hi)
    ax.set_xlim(0, max(int(counts.max()), 1) * 1.15)
    ax.set_ylabel(es.error_label("rms", per="view", short=True))
    ax.set_xlabel("Board views")
    es.integer_axis(ax, "x")
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    if selected is not None and np.isfinite(items[selected].error):
        # UX3-11 / P2-1: a selection, not a failure: the accent cursor
        ax.axhline(min(max(items[selected].error, lo), hi), **{**es.cursor_style(ax),
                                                               "linewidth": 2.2})
    es.panel_title(ax, "C", "Error of each view",
                   "colours as in A · accent line: the chosen photo")


# ======================================================================================
# Figure drawers (matplotlib, inside the stage's figure_style)


def _refusal(fig: Any, title: str, reason: str) -> None:
    ax = fig.add_subplot()
    es.empty_panel(ax, _wrap(reason, 70))
    es.panel_title(ax, None, title, "nothing from this run can be drawn here")


def _face_colours(t: Mapping[str, str]) -> list[str]:
    """Face shades, indexed by the 0-based face number: the shared face palette
    (``es.face_colours``, UX2-2 / ED2-8; moved to the shared layer in round 2)."""
    return es.face_colours(t.get("theme", "Light"), 6)


def _draw_board_views(fig, data: _CameraData, image, photo, thumbs, thumb_photos, images, level,
                      minimum, note) -> None:
    from matplotlib.lines import Line2D
    n = data.index + 1
    if level == 1:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
        a, b_spec, c, d = fig.add_subplot(gs[0, 0]), gs[0, 1], None, None
    elif level == 2 and _landscape(data, photo):
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0])
        a, b_spec, c, d = fig.add_subplot(gs[0, 0]), gs[:, 1], fig.add_subplot(gs[1, 0]), None
    elif level == 2:
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.0])
        a, b_spec, c, d = fig.add_subplot(gs[0, 0]), gs[0, 1], fig.add_subplot(gs[0, 2]), None
    else:
        gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.05, 1.0], height_ratios=[2.8, 1.0])
        a, b_spec, c = fig.add_subplot(gs[0, 0]), gs[0, 1], fig.add_subplot(gs[0, 2])
        d = fig.add_subplot(gs[1, :])
    size = vp.image_size_of(data.cam, photo, data.points)
    t = es._tokens_for(a)
    a.set_anchor("N")
    if c is not None:
        c.set_anchor("N")
    faces = data.fits_of(image)
    if faces:
        handles = []
        es.image_with_points(a, photo, np.empty((0, 2)), image_size=size, camera=data.index)
        dropped = [d_.image_points for d_ in data.views.dropped if d_.image_index == image]
        if dropped and level >= 3:
            es.image_with_points(a, None, np.empty((0, 2)), rejected=np.vstack(dropped),
                                 image_size=size, radius=0.010 * max(size), camera=data.index)
        if data.several_faces:
            # UX2-2 / ED2-8: faces as in every phase: one grey shade and one marker shape per
            # face (never a hue, never a circle), named "Face 2" like Phase 1's F-tags.
            for fit in faces:
                pts = fit.view.image_points
                a.scatter(pts[:, 0], pts[:, 1], zorder=4,
                          **es.face_point_kwargs(fit.view.board, a, size=52, on_photo=True))
            handles = es.face_legend_handles(
                [f.view.board for f in faces], a,
                counts={f.view.board: len(f.view.image_points) for f in faces})
        else:
            fit = faces[0]
            es.image_with_points(a, None, fit.view.image_points, image_size=size,
                                 radius=0.012 * max(size), linewidth=1.7, camera=data.index)
            handles.append(Line2D([], [], marker="o", lw=0, markersize=8, markerfacecolor="none",
                                  markeredgecolor=t["accent"], markeredgewidth=1.8,
                                  label=f"the board · {len(fit.view.image_points)} corners"))
        if dropped and level >= 3:
            handles.append(Line2D([], [], marker="o", lw=0, markersize=8, markerfacecolor="none",
                                  markeredgecolor=t["rejected"], label="face left out"))
        a.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                 ncols=1 if len(handles) > 2 else len(handles), borderaxespad=0.2,
                 handletextpad=0.4, columnspacing=1.0)
        count = len(faces)
        what = f"{count} board views" if count > 1 else "1 board view"
        total = sum(len(f.view.image_points) for f in faces)
        sub = (f"camera {n} · {total} corners · one shape per face" if count > 1 else
               f"camera {n} · {total} corners") + (f" · {note}" if note else "")
        es.panel_title(a, "A", f"{es.photo_label(image)}: {what}", sub)
    else:
        es.empty_panel(a, "No board view was solved for this camera.")
        es.panel_title(a, "A", "The chosen photo", f"camera {n}")
    # B thumbnails (UX2-24): as many columns as make each photo largest in B's box
    numbers = level >= 2
    fig_w, fig_h = (float(v) * fig.dpi for v in fig.get_size_inches())
    grid = b_spec.get_gridspec()
    widths = grid.get_width_ratios() or [1.0] * grid.ncols
    heights = grid.get_height_ratios() or [1.0] * grid.nrows
    cols_span = range(b_spec.colspan.start, b_spec.colspan.stop)
    rows_span = range(b_spec.rowspan.start, b_spec.rowspan.stop)
    box_w = fig_w * sum(widths[k] for k in cols_span) / sum(widths)
    box_h = fig_h * sum(heights[k] for k in rows_span) / sum(heights) - 56.0
    cols = _thumb_columns(len(thumbs), box_w, box_h, size, 16.0 if numbers else 2.0)
    rows = int(math.ceil(len(thumbs) / cols)) or 1
    # a thin empty last row keeps the bottom row's photo numbers inside the figure
    sub = b_spec.subgridspec(rows + 2, cols,
                             height_ratios=[0.001] + [1] * rows + [0.12 if numbers else 0.001],
                             hspace=0.16 if numbers else 0.04, wspace=0.04)
    head = fig.add_subplot(sub[0, :])
    head.set_axis_off()
    es.panel_title(head, "B", "The photos it learned from",
                   f"{len(thumbs)} of {len(images)} photos"
                   + (" · photo number below each" if numbers else " · corners in colour"))
    for k, shot in enumerate(thumbs):
        ax = fig.add_subplot(sub[1 + k // cols, k % cols])
        pts = np.vstack([f.view.image_points for f in data.fits
                         if f is not None and f.view.image_index == shot] or [np.empty((0, 2))])
        thumb = thumb_photos.get(shot)
        small, _stride = vp.shrink(thumb, 220) if thumb is not None else (None, 1)
        if small is not None:
            ax.imshow(small, cmap="gray", vmin=0, vmax=255, interpolation="antialiased",
                      extent=(-0.5, size[0] - 0.5, size[1] - 0.5, -0.5))
        ax.scatter(pts[:, 0], pts[:, 1], s=6 if level == 1 else 4, color=t["accent"],
                   linewidths=0)
        ax.set_xlim(-0.5, size[0] - 0.5)
        ax.set_ylim(size[1] - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        chosen = image is not None and shot == image
        if chosen:
            es.frame_axes(ax, data.index, width=3.0)
        else:
            es.frame_axes(ax, None, width=0.8)
        if small is None:
            ax.set_facecolor(t["grid"])
        if numbers:
            # under the thumbnail, outside the layout so the pictures keep their size
            ax.text(0.5, -0.03, es.photo_label(shot, short=True), transform=ax.transAxes,
                    ha="center", va="top", fontsize=es.TICK_SIZE - 1, in_layout=False,
                    clip_on=False, color=t["text"] if chosen else t["muted"],
                    fontweight="bold" if chosen else "normal")
    # C coverage
    if c is not None:
        es.image_with_points(c, None, np.empty((0, 2)), image_size=size, camera=data.index)
        c.set_facecolor(t["grid"] if es.is_dark_axes(c) else _mix(t["stage"], t["grid"], 0.6))
        c.scatter(data.points[:, 0], data.points[:, 1], s=6, color=t["accent"], alpha=0.45,
                  linewidths=0)
        es.panel_title(c, "C", "Where the corners fell",
                       f"all {len(data.points):,} corners, every view")
    # D corners per face
    if d is not None:
        counts = [len(v.image_points) for v in data.views.views]
        dropped = [len(x.image_points) for x in data.views.dropped]
        top = max(counts + dropped + [minimum + 1])
        bins = np.arange(0, top + 2) - 0.5 if top < 80 else np.linspace(0, top + 1, 40)
        # UX2-1: the camera's one fill on this theme, no outline
        # (a thin gap in the stage colour between bins, not an outline)
        d.hist(counts, bins=bins, color=es.camera_colour(data.index, d), zorder=3,
               edgecolor=t["stage"], linewidth=0.8, label="used")
        if dropped:
            d.hist(dropped, bins=bins, color=t["rejected"], alpha=es.REJECTED_ALPHA + 0.2,
                   zorder=3, label="left out")
        # UX3-11: neutral dashes unless a used face is below the minimum (none can be: the
        # left-out faces are drawn grey beside it)
        es.limit_line(d, minimum - 0.5, values=counts, above_is_bad=False, axis="x",
                      label=f"minimum {minimum}")
        d.set_xlabel("Corners found on one face")
        d.set_ylabel("Faces")
        es.integer_axis(d, "y")
        d.yaxis.grid(True, color=t["grid"], linewidth=0.8)
        es.panel_title(d, "D", "Corners per face",
                       f"{len(counts)} faces used, {len(dropped)} left out")


def _thumb_columns(count: int, box_w: float, box_h: float, image_size: tuple[int, int],
                   label_px: float) -> int:
    """Columns (2-6) that make ``count`` thumbnails of ``image_size`` largest inside a
    ``box_w`` x ``box_h`` pixel box, each with ``label_px`` under it (UX2-24)."""
    width, height = (max(float(v), 1.0) for v in image_size)
    best, best_scale = 3, -1.0
    for cols in range(2, 7):
        rows = int(math.ceil(max(count, 1) / cols))
        cell_w = 0.94 * box_w / cols
        cell_h = 0.94 * box_h / rows - label_px
        scale = min(cell_w / width, cell_h / height)
        if scale > best_scale + 1e-9:
            best, best_scale = cols, scale
    return best


def _landscape(data: _CameraData, photo: Any) -> bool:
    width, height = vp.image_size_of(data.cam, photo, data.points)
    return width > 1.1 * height


def _mix(a: str, b: str, s: float) -> str:
    from matplotlib.colors import to_hex, to_rgb
    ca, cb = np.array(to_rgb(a)), np.array(to_rgb(b))
    return to_hex((1 - s) * ca + s * cb)


def _draw_lens(fig, name, index, lens: vp.LensSummary, lenses, names, level, cam, exaggerate,
               photo=None, target_mm=None, distance_mm=None, data=None, check=None):
    if level == 1:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0])
        a, b, c, d, e = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), None, None, None
    else:
        rows = 2 if level == 2 else 3
        ratios = [1.0, 1.05] if level == 2 else [1.0, 1.05, 0.44]
        gs = fig.add_gridspec(rows, 2, width_ratios=[1.3, 1.0], height_ratios=ratios)
        a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        c, d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        e = fig.add_subplot(gs[2, :]) if level == 3 else None
    _projection_diagram(a, lens, index, target_mm, distance_mm)
    _projection_diagram(b, lens, index, target_mm, distance_mm, illustrate=True)
    if c is not None:
        _lens_bars(c, lenses, names, index)
    if d is not None:
        if lens.fitted:
            _distortion_panel(d, cam, lens, index, exaggerate, photo, check)
        elif data is not None and len(data.residuals):
            _bending_check(d, data, lens, check)
        else:
            es.empty_panel(d, "No errors to check the lens with.")
            es.panel_title(d, "D", "Does the lens bend lines?", "")
    if e is not None:
        _lens_equations(e, lens, check)


def _projection_diagram(ax, lens: vp.LensSummary, index: int, target_mm: float | None,
                        distance_mm: float | None, *, illustrate: bool = False) -> None:
    """Side view at true vertical scale: sensor | lens | the target near and twice as far.

    ``illustrate`` draws the *other* kind of lens with the same target (ED-12, an
    illustration in neutral colours, never this camera's colour): a telecentric camera is
    compared with an ordinary (pinhole) lens and a pinhole camera with a telecentric one.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle
    t = es._tokens_for(ax)
    own_tele = lens.model == "telecentric"
    tele = (not own_tele) if illustrate else own_tele
    colour = t["muted"] if illustrate else es.camera_colour(index, ax)
    edge = t["muted"] if illustrate else es.camera_label_colour(ax, index)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    x_sensor, x_lens, near, far = -0.9, 0.55, 4.2, 8.1
    if not tele:
        x_sensor, x_lens = -1.0, 1.3
    # the same field and target in both drawings, so they compare like for like
    if own_tele:
        field = lens.field_mm[1]
        size = target_mm if target_mm else field * 0.8
        distance = None
    else:
        distance = distance_mm if distance_mm else 500.0
        field = distance * lens.height / lens.fy if lens.fy else distance
        size = target_mm if target_mm else field * 0.5
    near_c, far_c = t["bad"], t["link"]
    if tele:
        x_end = 9.0
        top = max(field, size) / 2
        ax.set_ylim(-top * 1.95, top * 1.5)
        label_y = -top * 1.3
        ax.set_xlim(-1.5, 10.2)
        ax.add_patch(Rectangle((x_sensor - 0.28, -field / 2), 0.28, field, facecolor=colour,
                               edgecolor=edge, lw=1.2, zorder=4))
        ax.add_patch(FancyBboxPatch((x_lens - 0.3, -field * 0.62), 0.6, field * 1.24,
                                    boxstyle="round,pad=0.02,rounding_size=0.25",
                                    facecolor=colour, alpha=0.3, edgecolor=edge, lw=1.2,
                                    zorder=3))
        for y in np.linspace(-field / 2, field / 2, 7):
            rim = abs(y) >= field / 2 - 1e-9
            ax.plot([x_sensor, x_end], [y, y], color=t["accent"], lw=1.6 if rim else 1.1,
                    alpha=0.95 if rim else 0.35, zorder=2)
        for x, c_, alpha in ((near, near_c, 1.0), (far, far_c, 0.85)):
            ax.plot([x, x], [-size / 2, size / 2], color=t["text"], lw=4, alpha=alpha,
                    solid_capstyle="butt", zorder=5)
        ax.plot([x_sensor + 0.02] * 2, [-size / 2, size / 2], color=near_c, lw=4, zorder=6)
        if not illustrate:
            ax.text(x_end + 0.35, 0, f"field {field:.1f} mm", rotation=90, ha="left",
                    va="center", fontsize=es.TICK_SIZE, color=t["muted"])
        ax.text(near, top * 1.14, f"target {size:.0f} mm" if not illustrate else "near",
                ha="center", va="bottom", fontsize=es.TICK_SIZE, color=t["text"])
        ax.text(far, top * 1.14, "twice as far" if not illustrate else "far", ha="center",
                va="bottom", fontsize=es.TICK_SIZE, color=t["muted"])
        ax.text((near + far) / 2, -top * 1.72, "near or far: the same size on the sensor",
                ha="center", va="center", fontsize=es.LABEL_SIZE, color=t["text"])
        if illustrate:
            title, subtitle = "For comparison: a telecentric lens", \
                "illustration · same size at any distance"
        else:
            title = "Your lens: telecentric (parallel rays)"
            subtitle = f"camera {index + 1} · 1 px = {_um_pair(lens)} · drawn to scale"
    else:
        x_end = far + 0.35
        top = field / 2 * (far - x_lens) / (near - x_lens)
        reach = field / 2 * (x_end - x_lens) / (near - x_lens)
        ax.set_ylim(-reach * 1.32, reach * 1.3)
        label_y = -reach * 0.5
        ax.set_xlim(-1.5, 10.2)
        sensor_half = field / 2 * (x_lens - x_sensor) / (near - x_lens)
        ax.add_patch(Rectangle((x_sensor - 0.28, -sensor_half), 0.28, 2 * sensor_half,
                               facecolor=colour, edgecolor=edge, lw=1.2, zorder=4))
        ax.add_patch(FancyBboxPatch((x_lens - 0.12, -top * 0.22), 0.24, top * 0.44,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=colour, alpha=0.4, edgecolor=edge, lw=1.2,
                                    zorder=3))
        for sign in (-1, 1):
            ax.plot([x_sensor, x_lens, x_end],
                    [-sign * sensor_half, 0, sign * reach], color=t["accent"], lw=1.6, zorder=2)
        for x, c_, alpha in ((near, near_c, 1.0), (far, far_c, 0.85)):
            ax.plot([x, x], [-size / 2, size / 2], color=t["text"], lw=4, alpha=alpha,
                    solid_capstyle="butt", zorder=5)
            image_half = size / 2 * (x_lens - x_sensor) / (x - x_lens)
            for sign in (-1, 1):
                ax.plot([x, x_lens, x_sensor + 0.02], [sign * size / 2, 0, -sign * image_half],
                        color=c_, lw=0.9, alpha=0.6, zorder=2)
            ax.plot([x_sensor + 0.02] * 2, [-image_half, image_half], color=c_, lw=4, zorder=6)
        near_label = ("near" if illustrate else
                      f"board {size:.0f} mm at {distance:.0f} mm" if distance else "near")
        ax.text(near, reach * 1.1, near_label, ha="center", va="bottom", fontsize=es.TICK_SIZE,
                color=t["text"])
        ax.text(far, reach * 1.1, "twice as far" if not illustrate else "far", ha="center",
                va="bottom", fontsize=es.TICK_SIZE, color=t["muted"])
        ax.text((near + far) / 2, -reach * 1.17, "twice as far: half the size on the sensor",
                ha="center", va="center", fontsize=es.LABEL_SIZE, color=t["text"])
        if illustrate:
            title, subtitle = "For comparison: an ordinary lens", \
                "illustration · far things look smaller"
        else:
            title = "Your lens: pinhole (rays through one point)"
            subtitle = (f"camera {index + 1} · focal length {lens.fx:,.0f} px · "
                        f"{lens.fov_deg[0]:.0f}° × {lens.fov_deg[1]:.0f}° view")
    ax.text(x_sensor - 0.14, label_y, "sensor", ha="center", va="top",
            fontsize=es.TICK_SIZE, color=t["muted"])
    ax.text(x_lens, label_y, "lens", ha="center", va="top", fontsize=es.TICK_SIZE,
            color=t["muted"])
    es.panel_title(ax, "B" if illustrate else "A", title, subtitle)


def _distortion_panel(ax, cam, lens: vp.LensSummary, index: int, exaggerate: float,
                      photo=None, check: vp.CentreCheck | None = None) -> None:
    """D (fitted distortion): the image frame with a grid bent by the saved distortion."""
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe
    t = es._tokens_for(ax)
    size = (lens.width, lens.height)
    straight, bent = vp.distortion_grid(cam, size, exaggerate, lines=9,
                                        telecentric=lens.model == "telecentric")
    halo = None
    if photo is not None and photo.shape[1] == size[0] and photo.shape[0] == size[1]:
        ax.imshow(photo, cmap="gray", vmin=0, vmax=255, alpha=0.75, zorder=0,
                  extent=(-0.5, size[0] - 0.5, size[1] - 0.5, -0.5))
        line, halo = es.WHITE, "#000000"
    else:
        line = t["text"]
    ax.set_xlim(-0.04 * size[0], 1.04 * size[0])
    ax.set_ylim(1.04 * size[1], -0.04 * size[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    effects = [pe.withStroke(linewidth=3.0, foreground=halo, alpha=0.6)] if halo else None
    for points in straight:
        ax.plot(points[:, 0], points[:, 1], color=t["muted"], lw=0.7, alpha=0.6)
    for points in bent:
        ax.plot(points[:, 0], points[:, 1], color=line, lw=1.4, path_effects=effects)
    handles = [Line2D([], [], color=t["muted"], lw=0.8, label="straight"),
               Line2D([], [], color=t["text"], lw=1.4, label=f"this lens ×{exaggerate:g}")]
    ax.plot([lens.cx], [lens.cy], marker="+", markersize=14, mew=2.4, color=t["bad"], zorder=5)
    handles.append(Line2D([], [], marker="+", markersize=10, mew=2, color=t["bad"], lw=0,
                          label="principal point (as saved)"))
    if check is not None and not check.matches:
        ax.plot([check.centre[0]], [check.centre[1]], marker="x", markersize=9, mew=2,
                color=t["text"], zorder=5)
        handles.append(Line2D([], [], marker="x", markersize=8, mew=2, color=t["text"], lw=0,
                              label="image centre"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=2,
              borderaxespad=0.3, handlelength=1.6, columnspacing=0.9)
    es.panel_title(ax, "D", "Lens distortion", f"saved coefficients, exaggerated ×{exaggerate:g}")


def _bending_check(ax, data: _CameraData, lens: vp.LensSummary,
                   check: vp.CentreCheck | None) -> None:
    """D (no distortion fitted, ED-5): does this camera's error grow towards the edges?

    The outward part of every corner's error against its distance from the image centre
    (W/2, H/2: no distortion is fitted, so the image centre is the honest reference, SCI-6).
    Barrel or pincushion distortion that the model lacks makes the line rise or fall."""
    t = es._tokens_for(ax)
    centre = check.centre if check is not None else (lens.width / 2, lens.height / 2)
    profile = vp.radial_profile(data.points, data.residuals, centre)
    ax.scatter(profile.radius, profile.outward, s=4, alpha=0.4, zorder=3,
               **es.camera_mark_kwargs(data.index, ax, "dot"))
    if len(profile.bin_mid):
        ax.plot(profile.bin_mid, profile.bin_median, color=t["text"], lw=2.2, zorder=4,
                marker="o", markersize=4)
    inside = None
    if len(profile.bin_se) == len(profile.bin_mid) and len(profile.bin_mid) >= 2:
        # ED3-12: the band noise alone gives the line (+/- 2 standard errors of each bin's
        # median), so "flat" is read as "inside the band"
        band = 2.0 * profile.bin_se
        ax.fill_between(profile.bin_mid, -band, band, color=t["muted"], alpha=0.22, lw=0,
                        zorder=2)
        inside = bool(np.all(np.abs(profile.bin_median) <= band))
    lim = es._nice_ceiling(float(np.percentile(np.abs(profile.outward), 98)))
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color=t["muted"], lw=0.9, zorder=2)
    ax.set_xlabel("distance from the image centre (px)")
    ax.set_ylabel("outward error (px)")
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    # ED2-3: say only what one camera can show (Phase 3 later fits bending for camera 1
    # of a real rig that this check cannot see; part of it is the target's shape error)
    verdict = ("rises or falls: bending, or another error" if profile.bends else
               "flat: inside the noise band (grey)" if inside else
               "flat: no bending large enough to see from one camera")
    es.panel_title(ax, "D", "Does the lens bend lines?", verdict)


def _lens_bars(ax, lenses: Mapping[str, vp.LensSummary], names: Sequence[str], index: int) -> None:
    """Every camera's lens scale, across (x, circle) and down (y, square), as filled dots with
    badge ticks. Dots, not bars: the values differ by a few percent, and bars from a cut axis
    would exaggerate that."""
    present = [(k, lenses[n]) for k, n in enumerate(names) if n in lenses]
    if not present:
        es.empty_panel(ax, "No saved lens values.")
        return
    t = es._tokens_for(ax)
    tele = present[0][1].model == "telecentric"
    xs = np.arange(len(present))
    across = np.array([l.um_per_px[0] if tele else l.fx for _, l in present])
    down = np.array([l.um_per_px[1] if tele else l.fy for _, l in present])
    for x, (k, _), a_, d_ in zip(xs, present, across, down):
        ax.plot([x, x], [a_, d_], color=t["grid"], lw=2.0, zorder=2)
        kw = es.camera_mark_kwargs(k, ax, "dot")
        ax.scatter([x], [a_], s=64, marker="o", zorder=3, **kw)
        ax.scatter([x], [d_], s=58, marker="s", zorder=3, **kw)
    values = np.concatenate([across, down])
    lo, hi = float(values.min()), float(values.max())
    pad = max((hi - lo) * 0.25, abs(hi) * 0.01)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.6, len(present) - 0.4)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_ylabel("1 pixel on the target (µm)" if tele else "Focal length (px)")
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _p: f"{v:.1f}" if tele else f"{v:,.0f}"))
    es._badge_ticks(ax, xs, [k for k, _ in present], highlight=[index])
    # SCI2-23: the largest x/y difference, to one decimal (4.2 %, not "≤ 4 %")
    aniso = _anisotropy(lenses) or 0.0
    spread = 100 * (hi - lo) / max(abs(np.mean(values)), 1e-12)
    es.panel_title(ax, "C", "Every camera's lens",
                   f"● across (x)  ■ down (y) · x and y differ by up to {aniso:.1f} % · "
                   f"all {len(values)} values span {spread:.1f} % of their mean")


def _anisotropy(lenses: Mapping[str, vp.LensSummary]) -> float | None:
    """Largest difference between a camera's x and y scale, in % of the larger (SCI2-23:
    camera 4 of a real telecentric rig, 38,816.8 / 37,252.9 px/m = 4.2 %); None without lenses."""
    values = [(l.fx, l.fy) for l in lenses.values()
              if np.isfinite(l.fx) and np.isfinite(l.fy) and max(abs(l.fx), abs(l.fy)) > 0]
    if not values:
        return None
    return 100.0 * max(abs(fx - fy) / min(abs(fx), abs(fy)) for fx, fy in values)


def _math_number(value: float, digits: int = 0) -> str:
    """A number for mathtext: thousands separated by a thin space, never a spaced comma."""
    text = f"{value:,.{digits}f}"
    return text.replace(",", r"\,")


def _lens_equations(ax, lens: vp.LensSummary, check: vp.CentreCheck | None = None) -> None:
    t = es._tokens_for(ax)
    ax.set_axis_off()
    if lens.model == "telecentric":
        model = r"$u = m_x\,x + c_x \qquad v = m_y\,y + c_y$"
        values = (rf"$m_x = {_math_number(lens.fx)}$ px/m,   $m_y = {_math_number(lens.fy)}$ px/m,   "
                  rf"$c = ({_math_number(lens.cx, 1)},\ {_math_number(lens.cy, 1)})$ px")
        note = "the depth z does not appear: near and far look the same"
    else:
        model = r"$u = f_x\,X/Z + c_x \qquad v = f_y\,Y/Z + c_y$"
        values = (rf"$f_x = {_math_number(lens.fx)}$ px,   $f_y = {_math_number(lens.fy)}$ px,   "
                  rf"$c = ({_math_number(lens.cx, 1)},\ {_math_number(lens.cy, 1)})$ px")
        note = "dividing by the depth Z makes far things smaller"
    ax.text(0.0, 0.84, model, transform=ax.transAxes, ha="left", va="center",
            fontsize=es.TITLE_SIZE + 1, color=t["text"])
    ax.text(0.0, 0.46, values, transform=ax.transAxes, ha="left", va="center",
            fontsize=es.LABEL_SIZE + 1, color=t["text"])
    centre_note, colour = note, t["muted"]
    if check is not None:
        cx, cy = check.centre
        if check.transposed:
            centre_note = (f"c saved at ({check.saved[0]:g}, {check.saved[1]:g}), but the image "
                           f"centre is ({cx:g}, {cy:g}): an older run read the size swapped")
            colour = t["bad"]
        elif lens.model == "telecentric" and check.matches:
            centre_note = note + f" · c is the image centre ({cx:g}, {cy:g})"
    ax.text(0.0, 0.1, centre_note, transform=ax.transAxes, ha="left", va="center",
            fontsize=es.LABEL_SIZE, color=colour)
    es.panel_title(ax, "E", "The model Phase 2 fitted",
                   "a target point (x, y, z) in camera axes lands on pixel (u, v)")


def _arrow_scale(data: _CameraData | None, faces: Sequence[Any], photo: Any) -> float:
    if data is None or not faces:
        return 1.0
    size = vp.image_size_of(data.cam, photo, data.points)
    median = float(np.median(np.hypot(*data.residuals.T))) if len(data.residuals) else 1.0
    raw = 0.035 * max(size) / max(median, 1e-9)
    for step in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        if step >= raw * 0.7:
            return float(step)
    return 1000.0


def _draw_fit(fig, data: _CameraData | None, faces, photo, series, highlight, index, level,
              note, reason, lens: vp.LensSummary | None = None, theme: str | None = None,
              test: vp.PatternTest | None = None):
    scale = es.error_scale(theme)
    widths = [0.62, 1.0, 1.2]
    if level == 1:
        # Rebalanced: B's photo (aspect ~4:3) is width-limited, so a wider B fills the height
        # that A and the photo left blank below them (the L1 blank-area check).
        gs = fig.add_gridspec(1, 3, width_ratios=[0.62, 1.5, 1.0])
        a, b, c = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                   fig.add_subplot(gs[0, 2]))
        d = e = None
    elif level == 2:
        gs = fig.add_gridspec(2, 3, width_ratios=widths, height_ratios=[1.0, 1.0])
        a, d = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
        b, c = fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])
        e = None
    else:
        # UX3-5: E (the image's own, often portrait, shape) left a 400 x 340 px hole beside it
        # in a wide C column; B (no colour bar now) takes more width, E's row more height
        gs = fig.add_gridspec(2, 3, width_ratios=[0.62, 1.12, 0.92], height_ratios=[0.85, 1.15])
        a, d = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
        b = fig.add_subplot(gs[:, 1])
        c, e = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
        e.set_anchor("C")
    a.set_anchor("N")
    b.set_anchor("N")
    n = index + 1
    if data is not None and faces:
        image = faces[0].view.image_index
        observed = np.vstack([f.view.image_points for f in faces])
        projected = np.vstack([f.projected for f in faces])
        size = vp.image_size_of(data.cam, photo, data.points)
        _corner_example(a, data, faces, observed, projected, photo, lens, scale)
        arrows = _arrow_scale(data, faces, photo)
        quiver = es.residual_arrows(b, photo, observed, projected, arrows, image_size=size,
                                    camera=data.index, colour_scale=scale)
        # UX3-13 / P2-6: one photo's errors span a sliver of the 0.1-100 px scale, so the
        # arrows were all one colour under a colour bar that said nothing. The length (and
        # the scale-bar chip) carries the size; the colour is the camera's, as in C.
        quiver.set_array(None)
        quiver.set_facecolor(es.camera_colour(data.index, b))
        quiver.set_edgecolor("#FFFFFF")
        quiver.set_linewidth(0.4)
        _scale_bar_chip(b)
        error = data.check.per_image.get(image, float("nan"))
        sub = f"camera {n} · every corner, arrows ×{arrows:g}"
        if level == 2 and len(faces) > 1:
            # UX-12: the photo's RMS pools its faces; each face's own RMS, named as such
            sub = f"arrows ×{arrows:g} · " + " · ".join(
                f"{es.face_label(f.view.board)} {f.rms:.2f}" for f in faces) + " px"
        if level >= 3 and test is not None and test.testable:
            # SCI3-10 / UX3-10: post-fit residuals, so the noise share is an upper bound. At
            # L3 the per-face values are not repeated here: three of them ran into C's axis.
            # (multi-face L3: the caption carries the pattern share; the subtitle would run
            # into C's axis label)
            if not (level >= 3 and len(faces) > 1):
                sub += (f" · pattern {100 * test.share:.0f} % "
                        f"(noise ≤ {100 * test.noise:.0f} %)")
        es.panel_title(b, "B", f"{es.photo_label(image)} · RMS {es.format_px(error)}", sub)
        if note:  # a long note under the photo, not in the subtitle (it would run into C)
            b.set_xlabel(_wrap(note, 34), fontsize=es.TICK_SIZE,
                         color=es._tokens_for(b)["muted"])
    else:
        es.empty_panel(a, "")
        es.panel_title(a, "A", "One corner", "")
        es.empty_panel(b, _wrap(reason or "No board view to draw.", 44))
        es.panel_title(b, "B", "How well one photo fits", f"camera {n}")
    if series and any(len(v) for v in series.values()):
        values = np.concatenate([np.asarray(v, float) for v in series.values() if len(v)])
        values = values[np.isfinite(values) & (values > 0)]
        # ED-19: the shared log error axis, so a 70 px photo and a 0.5 px one both read
        es.strip_plot_by_camera(c, dict(series), highlight=highlight,
                                ylabel=es.error_label("rms", per="photo", unit=False,
                                                      short=True),
                                unit="px", log=scale if values.size else False)
        # UX4-7: the ring marks which photo B shows, not a failure. The shared helper draws
        # it in the 'bad' red, so recolour that ring here to the neutral text colour (no limit
        # is drawn on this panel, so nothing is crossed).
        neutral = es._tokens_for(c)["text"]
        for ring in c.collections:
            if ring.get_zorder() == 4 and np.all(np.asarray(ring.get_sizes()) == 150):
                ring.set_edgecolor(neutral)
        total = sum(len(v) for v in series.values())
        es.panel_title(c, "C", "Every photo, every camera",
                       ("one dot per photo (its faces' poses)"
                        if data is not None and data.several_faces else "one dot per photo")
                       + f" · {total:,} · ring = B")
    else:
        es.empty_panel(c, "This run saved no per-photo errors.")
        es.panel_title(c, "C", "Every photo, every camera", "")
    if d is not None:
        if data is not None and len(data.residuals):
            _error_cloud(d, data, scale)
        else:
            es.empty_panel(d, "No errors to collect.")
            es.panel_title(d, "D", "All of this camera's errors", "")
    if e is not None:
        if data is not None and len(data.residuals):
            _error_map(e, data, photo, scale, fig)
        else:
            es.empty_panel(e, "No errors to map.")
            es.panel_title(e, "E", "Where in the image", "")


def _scale_bar_chip(ax) -> None:
    """UX2-12: put the arrow scale bar ("2 px error (shown ×20)") on a solid chip in the
    theme's surface colour, so its text never sits straight on the photo. (Local: the shared
    ``es.residual_arrows`` draws it on a 55 % black patch.)"""
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    t = es._tokens_for(ax)
    for artist in ax.artists:
        if not isinstance(artist, AnchoredSizeBar):
            continue
        artist.patch.set_facecolor(t["stage"])
        artist.patch.set_alpha(0.92)
        for child in artist.size_bar.get_children():
            if hasattr(child, "set_color"):
                child.set_color(t["text"])
        label = getattr(artist.txt_label, "_text", None)
        if label is not None:
            label.set_color(t["text"])


def _corner_example(ax, data: _CameraData, faces, observed, projected, photo,
                    lens: vp.LensSummary | None, scale: Any) -> None:
    """A (ED-3): what reprojection error *is*, at true size: one corner of this photo (the one
    of typical error), the ring where the detector found it, the dot where the fitted lens
    puts it and the arrow between them, over the photo's own pixels."""
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe
    from matplotlib.patches import Circle, FancyArrowPatch
    t = es._tokens_for(ax)
    k = vp.typical_corner(observed, projected, photo)
    found, model = observed[k], projected[k]
    delta = model - found
    error = float(np.hypot(*delta))
    half = max(3.0, 1.5 * error + 1.6)
    centre = (found + model) / 2
    x0, x1 = centre[0] - half, centre[0] + half
    y0, y1 = centre[1] - half, centre[1] + half
    if photo is not None:
        ax.imshow(photo, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                  extent=(-0.5, photo.shape[1] - 0.5, photo.shape[0] - 0.5, -0.5), zorder=0)
        grid = "#FFFFFF"
    else:
        ax.set_facecolor(t["grid"])
        grid = t["muted"]
    for x in np.arange(math.floor(x0) - 0.5, x1 + 1, 1.0):
        ax.axvline(x, color=grid, lw=0.4, alpha=0.35, zorder=1)
    for y in np.arange(math.floor(y0) - 0.5, y1 + 1, 1.0):
        ax.axhline(y, color=grid, lw=0.4, alpha=0.35, zorder=1)
    ring = 0.28
    ax.add_patch(Circle(found, ring, fill=False, edgecolor=t["accent"], lw=2.6, zorder=4))
    ax.scatter([model[0]], [model[1]], s=46, color=t["text"], edgecolors="#FFFFFF",
               linewidths=1.2, zorder=6)
    ax.add_patch(FancyArrowPatch(found, model, arrowstyle="-|>", mutation_scale=16, lw=2.4,
                                 color=es.camera_colour(data.index, ax), zorder=5,
                                 shrinkA=0, shrinkB=5,
                                 path_effects=[pe.withStroke(linewidth=4.2,
                                                             foreground="#FFFFFF",
                                                             alpha=0.8)]))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    es.frame_axes(ax, data.index, width=2.0)
    # SCI6-2: one pixel's size and the size of the error are different numbers. The subtitle
    # states the pixel (telecentric: one constant size, x × y); the error's µm on target goes
    # in the legend, labelled as the error (0.80 px ≈ 21 µm is the error, not one pixel).
    pixel_text = ""
    error_um = None
    if lens is not None and lens.model == "telecentric" and lens.um_per_px:
        error_um = math.hypot(delta[0] * lens.um_per_px[0], delta[1] * lens.um_per_px[1])
        pixel_text = f" · {_um_pair(lens)}"
    else:
        face = next((f for f in faces if len(f.view.image_points) and np.any(
            np.all(np.isclose(f.view.image_points, found), axis=1))), None)
        if face is not None and lens is not None and lens.fx:
            row = int(np.flatnonzero(np.all(np.isclose(face.view.image_points, found),
                                            axis=1))[0])
            depth = float((face.R @ face.view.object_points[row] + face.t)[2])
            error_um = error * depth / lens.fx * 1e6
    # ED7-6: one µm value per pixel on this screen (the subtitle); this corner's own error is
    # labelled as the error, in µm on the target, so the two are not read as one scale
    # the legend names the error in px and µm; the subtitle names the scale ("scale: 1 px = …")
    error_label = (f"this corner's error: {es.format_px(error)} = {error_um:.0f} µm"
                   if error_um is not None else "this corner's error (drawn longer in B)")
    ax.legend(handles=[
        Line2D([], [], marker="o", lw=0, markersize=9, markerfacecolor="none",
               markeredgecolor=t["accent"], markeredgewidth=2, label="found by the detector"),
        Line2D([], [], marker="o", lw=0, markersize=7, color=t["text"],
               label="where the lens model puts it"),
        Line2D([], [], color=es.camera_colour(data.index, ax), lw=2.2, marker=">", markersize=6,
               markevery=[1], label=error_label)],
        loc="upper center", bbox_to_anchor=(0.5, -0.01), ncols=1, borderaxespad=0.2,
        handletextpad=0.4)
    es.panel_title(ax, "A", f"One corner: {es.format_px(error)}",
                   f"scale: 1 px = {_um_pair(lens)}" if lens is not None and pixel_text
                   else "1 square = 1 px")


def _error_cloud(ax, data: _CameraData, scale: Any) -> None:
    t = es._tokens_for(ax)
    r = data.residuals
    size = np.hypot(*r.T)
    lim = es._nice_ceiling(float(np.percentile(size, 99)) * 1.1)
    ax.scatter(r[:, 0], r[:, 1], s=4, c=size, alpha=0.7, linewidths=0, zorder=3,
               **scale.scatter_kwargs())
    from matplotlib.patches import Circle
    # UX3-11: the RMS ring is a reference, not a failure: neutral dashes, never red
    ax.add_patch(Circle((0, 0), data.rms, fill=False, color=t["text"], lw=1.6, zorder=4,
                        linestyle=(0, (4, 2))))
    ax.axhline(0, color=t["grid"], lw=0.8, zorder=1)
    ax.axvline(0, color=t["grid"], lw=0.8, zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)
    ax.set_aspect("equal")
    ax.set_xlabel("→ across the image (px)")
    ax.set_ylabel("↓ down the image (px)")
    es.panel_title(ax, "D", f"All of camera {data.index + 1}'s errors",
                   f"{len(r):,} corners · dashed ring: RMS {es.format_px(data.rms)}")


def _error_map(ax, data: _CameraData, photo, scale: Any, fig: Any = None) -> None:
    """E: mean error per area of the image, every photo of this camera.

    UX2-7: the means span a narrow range (on a real rig: about 1-3 px), one band of the shared
    0.1-100 px log scale, so the map would be flat. When they do (``es.needs_local_scale``)
    E gets its own linear scale from 0 with its own colour bar, marked "this map only"."""
    import matplotlib
    t = es._tokens_for(ax)
    size = vp.image_size_of(data.cam, photo, data.points)
    errors = np.hypot(*data.residuals.T)
    mean, xe, ye = vp.error_map(data.points, errors, size, cells=8)
    finite = mean[np.isfinite(mean)]
    local = es.needs_local_scale(finite)
    use = es.local_scale(finite, ax) if local else scale
    cmap = matplotlib.colormaps[use.cmap].copy()
    cmap.set_bad(t["grid"])
    image = ax.imshow(np.ma.masked_invalid(mean), cmap=cmap, norm=use.norm(),
                      interpolation="nearest",
                      extent=(xe[0] - 0.5, xe[-1] - 0.5, ye[-1] - 0.5, ye[0] - 0.5), zorder=1)
    if local and fig is not None:
        use.colourbar(fig, ax, image)
    ax.set_xlim(-0.5, size[0] - 0.5)
    ax.set_ylim(size[1] - 0.5, -0.5)
    # Fill the cell (UX3-5): an equal aspect left a blank block beside a portrait photo's map.
    # The cells are then not square; the map is a mean per area, not a picture of the photo.
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    es.frame_axes(ax, data.index, width=2.0)
    worst = f"max {es.format_sig(float(finite.max()))} px" if finite.size else ""
    es.panel_title(ax, "E", "Where in the image",
                   "mean per area" + ("" if local else " · colours as in D")
                   + (f" · {worst}" if worst else ""))


def _draw_decision(fig, rows: list[dict[str, Any]], warning: float, level: int,
                   um_per_px: float | None = None,
                   later: Mapping[str, Any] | None = None) -> None:
    """Stage 5. UX3-5: the "Next" illustration (C) is drawn in a cell shaped like it (a
    strip under A/B at L1, a quarter at L2, the top-right cell at L3), never a tall column
    that leaves it floating in blank space."""
    groups = camera_groups(rows) if level >= 3 else None
    if level == 1:
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 0.36])
        a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        c, wide = fig.add_subplot(gs[1, :]), True
        d = e = None
    elif level == 2:
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0])
        a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        d, c, wide = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), False
        e = None
    else:
        gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 0.95, 1.0])
        a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
        d, e, wide = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1:]), False
    dropped = [r["index"] for r in rows if not r["views"] or r["rms"] is None]
    rms = {r["index"]: (r["rms"] if r["rms"] is not None else 0.0) for r in rows}
    above = [r["index"] for r in rows if r["rms"] is not None and r["rms"] > warning]
    top = max([v for v in rms.values() if np.isfinite(v)] + [0.0])
    show_line = top >= 0.6 * warning   # UX-26: use the axis for the data, not an empty band
    es.bars_by_camera(a, rms, threshold=warning if show_line else None, fail_below=False,
                      threshold_label=None, unit="px",
                      ylabel=es.error_label("rms", unit=False, short=True),
                      flagged=dropped + above, value_format="{:.2f}")
    t = es._tokens_for(a)
    if show_line:
        _knock_out_labels_on_line(a, warning)
        # UX3-11: the warning band is red only when a camera is above it
        crossed = bool(above)
        a.axhspan(warning, a.get_ylim()[1], color=t["bad"] if crossed else t["muted"],
                  alpha=0.07, zorder=0, lw=0)
        # round 9: named in a legend, not beside the line where it runs into the bar labels
        from matplotlib.lines import Line2D
        warn_style = es.limit_style(a, crossed)
        a.legend(handles=[Line2D([0], [0], color=warn_style["color"], linestyle=(0, (5, 3)),
                                 linewidth=1.4, label=f"warning {warning:g} px")],
                 loc="upper left", fontsize=es.TICK_SIZE, frameon=False, borderaxespad=0.3)
    if um_per_px:
        second = a.secondary_yaxis("right", functions=(lambda p: p * um_per_px,
                                                       lambda u: u / um_per_px))
        second.set_ylabel("≈ µm on the target")
        second.tick_params(labelsize=es.TICK_SIZE, colors=t["muted"], labelcolor=t["text"])
    if above:
        sub = f"{len(above)} above the {warning:g} px warning level, where a first estimate counts as poor"
    elif show_line:
        sub = f"all below the {warning:g} px warning level (dashed line)"
    else:
        sub = (f"all well below the {warning:g} px warning level, where a first estimate "
               "counts as poor")
    es.panel_title(a, "A", "RMS error per camera", sub)
    views = {r["index"]: r["views"] for r in rows}
    es.bars_by_camera(b, views, flagged=dropped, ylabel="Photos with the board", value_format="{:.0f}")
    es.integer_axis(b, "y")
    es.panel_title(b, "B", "Photos per camera",
                   "a camera with none is dropped" if not dropped else f"{len(dropped)} with none")
    _next_phase_sketch(c, [r["index"] for r in rows if r["rms"] is not None][:2] or [0],
                       wide=wide)
    if d is not None:
        points = {r["index"]: r["points"] for r in rows}
        es.bars_by_camera(d, points, flagged=dropped, ylabel="Corners used", value_format="{:,.0f}")
        es.integer_axis(d, "y")
        es.panel_title(d, "D", "Corners per camera", "more corners, steadier lens")
    if e is not None and groups is not None:
        _camera_group_panel(e, rows, groups, later)
    elif e is not None:
        t = es._tokens_for(e)
        xy = np.array([(r["views"], r["rms"]) for r in rows if r["rms"] is not None], dtype=float)
        if len(xy):
            idx = [r["index"] for r in rows if r["rms"] is not None]
            pad_x = max(np.ptp(xy[:, 0]) * 0.15, 2)
            pad_y = max(np.ptp(xy[:, 1]) * 0.2, 0.05)
            e.set_xlim(xy[:, 0].min() - pad_x, xy[:, 0].max() + pad_x)
            e.set_ylim(max(0, xy[:, 1].min() - pad_y), xy[:, 1].max() + pad_y)
            for (x0, y0), cam in zip(xy, idx):
                e.scatter([x0], [y0], s=14, zorder=3, **es.camera_mark_kwargs(cam, e, "dot"))
            # Shared spread: badges pushed apart, a leader to each moved badge's point.
            es.draw_camera_badges(e, xy, idx, size=10, repel=True)
        e.set_xlabel("Photos with the board")
        e.set_ylabel(es.error_label("rms", short=True))
        e.yaxis.grid(True, color=t["grid"], linewidth=0.8)
        es.panel_title(e, "E", "Error vs photos", "one badge per camera")


#: Two camera groups are pointed out only when their medians differ by at least this ratio.
GROUP_RATIO = 1.25


def _name_tag(name: str) -> str | None:
    """The letter a camera name ends with after its number ("cam8_Back_908B" -> "B"), the
    rig's own label (e.g. top / bottom row), or None."""
    name = str(name)
    if len(name) >= 2 and name[-1].isalpha() and name[-2].isdigit():
        return name[-1]
    return None


def camera_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """ED2-26: two groups of cameras whose fit errors differ, or None.

    By the letter the names end with when it splits the rig in two (on a real rig: "…T" cameras 1-4
    and "…B" cameras 5-8), else at the largest gap between sorted errors. Shown only when each
    group has 2+ cameras, the groups do not overlap and the higher median is at least
    :data:`GROUP_RATIO` x the lower (pcube: two outlier cameras in one name group are not a
    group difference)."""
    good = [r for r in rows if r.get("rms") is not None and np.isfinite(r["rms"])]
    if len(good) < 4:
        return None
    tags = [_name_tag(r["name"]) for r in good]
    split: list[list[Mapping[str, Any]]] | None = None
    how = "error"
    if None not in tags and len(set(tags)) == 2:
        a, b = sorted(set(tags))
        split = [[r for r, g in zip(good, tags) if g == a], [r for r, g in zip(good, tags) if g == b]]
        how = "name"
    else:
        ordered = sorted(good, key=lambda r: r["rms"])
        values = np.log([r["rms"] for r in ordered])
        k = int(np.argmax(np.diff(values))) + 1
        split = [ordered[:k], ordered[k:]]
    if min(len(s) for s in split) < 2:
        return None
    medians = [float(np.median([r["rms"] for r in s])) for s in split]
    low, high = (0, 1) if medians[0] <= medians[1] else (1, 0)
    if medians[high] < GROUP_RATIO * medians[low]:
        return None
    if max(r["rms"] for r in split[low]) >= min(r["rms"] for r in split[high]):
        return None   # the groups overlap (e.g. two outliers): not a group difference
    return {"low": split[low], "high": split[high], "median_low": medians[low],
            "median_high": medians[high], "ratio": medians[high] / medians[low], "how": how,
            "tags": ((_name_tag(split[low][0]["name"]), _name_tag(split[high][0]["name"]))
                     if how == "name" else (None, None))}


def _px_range_text(values: Sequence[float]) -> str:
    """'a–b px' for a range; one value when the ends print the same (UX4-2: never '9.56–9.56')."""
    low, high = f"{min(values):.2f}", f"{max(values):.2f}"
    return f"{low} px" if low == high else f"{low}–{high} px"


def _group_sentence(groups: Mapping[str, Any], later: Mapping[str, Any] | None) -> str:
    """The L3 caption's camera-group sentence (ED2-26, SCI3-8). One camera's fit cannot
    tell a lens-model or target-shape error from image quality, so the advice depends on
    whether a Phase 4 run that continued from this one (target refined) removed the gap."""
    high = _numbers_text([r["index"] for r in groups["high"]])
    low = _numbers_text([r["index"] for r in groups["low"]])
    # The name tags are on E's axis; the caption stays short (UX3-2: at most 4 lines)
    head = f"E: cameras {high} fit {groups['ratio']:.1f}× worse than {low}"
    if later is None:
        return head + (", maybe the model or target: Phase 4 will show if the gap stays.")
    if later["gone"]:
        # the medians are printed in E's subtitle and tick labels, so the caption stays short
        # SCI4-6: the per-camera lens fit can absorb photo-dependent error too, so the cause
        # is not settled (not "the model or target, not the photos").
        return head + (", alike after Phase 4: the per-camera lens fit absorbs most of it, "
                       "so the cause is not yet settled.")
    return (head + f", still {later['ratio']:.1f}× after Phase 4: check what they see "
            "differently (angle, light, focus).")


def _numbers_text(indices: Sequence[int]) -> str:
    """Camera numbers as "1–4" when consecutive, else "1, 3, 5"."""
    numbers = sorted(int(i) + 1 for i in indices)
    if len(numbers) > 2 and numbers == list(range(numbers[0], numbers[-1] + 1)):
        return f"{numbers[0]}–{numbers[-1]}"
    return ", ".join(str(n) for n in numbers)


def _camera_group_panel(ax: Any, rows: Sequence[Mapping[str, Any]],
                        groups: Mapping[str, Any],
                        later: Mapping[str, Any] | None = None) -> None:
    """E (ED2-26): the cameras in their two groups, each camera a badge at its RMS error, with
    the group's median as a line: the pattern a learner can act on, instead of a trend line.
    SCI3-8: with a Phase 4 run that continued from this one, the same cameras' RMS error
    after Phase 4 (same statistic, same axis) as hollow rings with a dashed median, so the
    learner sees whether the gap stays."""
    t = es._tokens_for(ax)
    columns = [groups["high"], groups["low"]]
    tags = (groups["tags"][1], groups["tags"][0])
    after = (later or {}).get("rms") or {}
    labels = []
    xy, indices = [], []
    values_all = [float(r["rms"]) for r in rows if r.get("rms") is not None]
    for x, (members, tag) in enumerate(zip(columns, tags)):
        cams = [int(r["index"]) for r in members]
        values = [float(r["rms"]) for r in members]
        median = float(np.median(values))
        left = x - 0.18 if after else x
        ax.plot([left - 0.3, left + 0.3], [median, median], color=t["muted"], lw=2.0, zorder=2)
        offsets = np.linspace(-0.22, 0.22, len(cams)) if len(cams) > 1 else [0.0]
        order = np.argsort(values)
        for off, k in zip(offsets, order):
            ax.scatter([left + off], [values[k]], s=14, zorder=3,
                       **es.camera_mark_kwargs(cams[k], ax, "dot"))
            xy.append((left + off, values[k]))
            indices.append(cams[k])
        line = f"median {es.format_px(median)}"
        if after:
            later_values = [after[r["name"]] for r in members if r["name"] in after]
            later_median = float(np.median(later_values))
            right = x + 0.3
            ax.plot([right - 0.14, right + 0.14], [later_median, later_median],
                    color=t["muted"], lw=1.6, linestyle=(0, (3, 2)), zorder=2)
            spread = np.linspace(-0.1, 0.1, len(later_values)) if len(later_values) > 1 else [0.0]
            for off, r in zip(spread, sorted(members, key=lambda r: after.get(r["name"], 0))):
                if r["name"] in after:
                    ax.scatter([right + off], [after[r["name"]]], s=34, zorder=3,
                               facecolors="none", linewidths=1.6,
                               edgecolors=es.camera_colour(int(r["index"]), ax))
            values_all += later_values
            line += f" → {es.format_px(later_median)}"
        labels.append(f"cameras {_numbers_text(cams)}"
                      + (f" · names end “{tag}”" if tag else "") + "\n" + line)
    es.draw_camera_badges(ax, np.asarray(xy), indices, size=10, repel=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=es.TICK_SIZE)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.65, 1.65)
    pad = max(np.ptp(values_all) * 0.12, 0.05)
    ax.set_ylim(max(0.0, min(values_all) - pad), max(values_all) + pad)
    ax.set_ylabel(es.error_label("rms", short=True))
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    sub = f"one group fits {groups['ratio']:.1f}× worse · line: median"
    if after:
        from matplotlib.lines import Line2D
        sub = (f"Phase 2: one group {groups['ratio']:.1f}× worse · after Phase 4 (rings): "
               + ("alike" if later["gone"] else f"still {later['ratio']:.1f}×"))
        ax.legend(handles=[
            Line2D([], [], marker="o", lw=0, markersize=6, color=t["muted"],
                   label="Phase 2 (this run)"),
            Line2D([], [], marker="o", lw=0, markersize=7, markerfacecolor="none",
                   markeredgecolor=t["muted"], markeredgewidth=1.6,
                   label="after Phase 4")],
            loc="upper right", fontsize=es.TICK_SIZE, borderaxespad=0.3, handletextpad=0.3)
    es.panel_title(ax, "E", "Two groups of cameras", sub)


def _next_phase_sketch(ax, cameras: Sequence[int], wide: bool = False) -> None:
    """C (ED-4): why the error rises in Phase 3. Left: in Phase 2 each camera places the target
    where it suits that camera alone (two slightly different poses, in the cameras' colours).
    Right: in Phase 3 every camera must explain the photo with one shared pose. Illustration.
    ``wide``: one row (a strip under A and B), the words beside each sketch (UX3-5)."""
    from matplotlib.patches import FancyArrowPatch, Polygon
    t = es._tokens_for(ax)
    ax.set_axis_off()

    def square(cx: float, cy: float, side: float, angle: float) -> np.ndarray:
        a = math.radians(angle)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * side / 2
        return base @ rot.T + [cx, cy]

    if wide:
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 4.0)
        left, right, mid = (2.45, 2.05), (13.6, 2.05), 2.05
        arrow = ((9.8, mid), (11.4, mid))
        texts = ((5.0, "Phase 2", "each camera on its own"),
                 (15.5, "Phase 3", "one shared pose"))
    else:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6.4)
        left, right, mid = (2.55, 3.25), (7.75, 3.25), 3.25
        arrow = ((4.55, mid), (5.75, mid))
        texts = ((2.55, "Phase 2", "each camera\non its own"),
                 (7.75, "Phase 3", "one shared\npose"))
    ax.set_aspect("equal")
    poses = [(left[0] - 0.3, left[1] + 0.22, -9.0), (left[0] + 0.3, left[1] - 0.23, 8.0)]
    for cam, (cx, cy, angle) in zip(list(cameras) * 2, poses):
        outline = square(cx, cy, 2.2, angle)
        ax.add_patch(Polygon(outline, closed=True, facecolor="none", lw=2.4, zorder=3,
                             edgecolor=es.camera_colour(cam, ax)))
    for k, (cam, (cx, cy, angle)) in enumerate(zip(list(cameras) * 2, poses)):
        corner = square(cx, cy, 2.2, angle)[3 if k == 0 else 1]
        es.draw_camera_badges(ax, [corner], [cam], size=9)
    ax.add_patch(FancyArrowPatch(*arrow, arrowstyle="-|>", mutation_scale=18,
                                 lw=2.0, color=t["muted"], zorder=2))
    ax.add_patch(Polygon(square(right[0], right[1], 2.2, 0.0), closed=True, facecolor=t["grid"],
                         edgecolor=t["text"], lw=2.6, zorder=3))
    # The words sit a fixed number of points above/below the drawn squares, so the corner
    # badges (sized in points) never touch them however small the panel gets.
    drawn = np.vstack([square(cx, cy, 2.2, angle) for cx, cy, angle in poses])
    top, bottom = float(drawn[:, 1].max()), float(drawn[:, 1].min())
    gap = 16  # points: a size-9 badge's radius plus a margin
    for x, phase, words in texts:
        if wide:
            ax.text(x, mid + 0.35, phase, ha="left", va="bottom", fontsize=es.LABEL_SIZE + 1,
                    fontweight="bold", color=t["text"])
            ax.text(x, mid - 0.15, words, ha="left", va="top", fontsize=es.LABEL_SIZE,
                    color=t["muted"])
        else:
            ax.annotate(phase, (x, top), xytext=(0, gap), textcoords="offset points",
                        ha="center", va="bottom", fontsize=es.LABEL_SIZE + 1,
                        fontweight="bold", color=t["text"])
            ax.annotate(words, (x, bottom), xytext=(0, -gap), textcoords="offset points",
                        ha="center", va="top", fontsize=es.LABEL_SIZE, color=t["muted"],
                        linespacing=1.3)
    # SCI3-2: the rise is the shared pose on the drawn target; Phase 4 refines the target
    es.panel_title(ax, "C", "Next: one target for all cameras",
                   "illustration · rises, then falls")


# ======================================================================================
# Small helpers


def _knock_out_labels_on_line(ax, line: float) -> None:
    """A bar's value label that a dashed reference line would strike through keeps its place
    on the bar; it gets the panel's background behind it and is drawn over the line, so the
    line passes behind the number (round 9: pcube stage 5 A, the 4.05 label). A label may not
    move onto the line: a moved label collides with the line's own annotation."""
    fig = ax.figure
    height_pt = ax.bbox.height * 72.0 / float(fig.dpi)
    low, high = ax.get_ylim()
    if height_pt <= 0 or high <= low:
        return
    per_pt = (high - low) / height_pt        # data units per point
    band_lo = 3.0 * per_pt                   # the label's offset above its bar (3 pt)
    band_hi = (3.0 + 1.5 * es.TICK_SIZE + 1.0) * per_pt
    face = ax.get_facecolor()
    for text in ax.texts:
        if (text.get_gid() or "") != "edu:value" or not hasattr(text, "xy"):
            continue
        _x, v = text.xy
        if v + band_lo <= line <= v + band_hi:
            text.set_bbox(dict(boxstyle="square,pad=0.1", facecolor=face, edgecolor="none"))
            text.set_zorder(6)


def _pattern_words(test: vp.PatternTest | None) -> str:
    """B's reading rule from :func:`vp.pattern_test` (SCI2-11): three wordings measured
    against what detection noise alone gives, never a fixed 0.5 cut on a linear-only fit."""
    if test is None or not test.testable:
        return "the faces have too few corners to tell a pattern from noise"
    # SCI8-1: the noise mean and the 99.9 % bar are simulated through this photo's own pose fit
    # (vp.simulated_pattern_test), so they are the values this page's fit actually gives
    # SCI7-6 / ED6-5: "noise alone gives about X %" is the average under noise (an expected
    # value, not a cap); the 99.9 % bar is the level noise alone passes in 1 case in 1,000
    share = f"{100 * test.share:.0f} %"
    noise = f"{100 * test.noise:.0f} %"
    # SCI9-1: the bar moves by a few points with the draws, so it is printed as an approximate
    # value; the 1-in-1,000 meaning is kept: noise alone exceeds this bar in about 1 case in 1,000
    bar = f"about {100 * test.noise_high:.0f} %"
    # ED6-5: say what the share is of: the squared error of this photo's corners (not arrows)
    # one clause each (no full stop): the L1 caption is two sentences at most
    if test.verdict == "borderline":
        return (f"a smooth pattern takes {share} of this photo's squared error: borderline, "
                f"within three standard deviations of the 1-in-1,000 bar of {bar} "
                f"(noise averages {noise})")
    if test.verdict == "pattern":
        return (f"a smooth pattern takes {share} of this photo's squared error; noise alone "
                f"averages {noise} and exceeds {bar} in about 1 case in 1,000: the model "
                "misses something")
    if test.verdict == "some pattern":
        return (f"a weak smooth pattern takes {share} of this photo's squared error, above the "
                f"bar of {bar} that noise alone exceeds in about 1 case in 1,000 "
                f"(noise averages {noise})")
    return (f"the errors look like detection noise: a smooth pattern takes {share} of this "
            f"photo's squared error, and noise alone gives about {noise} on average")


def _level_text(level: int, sentences: Sequence[str]) -> str:
    """Caption at a detail level: L1 the first entry, each level adds the next one."""
    return " ".join(s for s in list(sentences)[:max(1, level)] if s)


def _sentence(text: str) -> str:
    text = text.strip()
    return (text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")) if text else text


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text and not text[:2].isupper() else text


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


def _sample(items: Sequence[int], count: int) -> list[int]:
    items = list(items)
    if len(items) <= count:
        return items
    picks = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in sorted(set(picks))]


def _short_camera(name: str) -> str:
    return name if len(name) <= 22 else name[:20] + "…"




def _exaggeration(lens: vp.LensSummary, cam: Any) -> float:
    """A round factor that makes the saved distortion move the grid corner by ~4 % of the width."""
    if cam is None:
        return 1.0
    corner = np.array([[0.0, 0.0]])
    moved = vp._distort_pixels(cam, corner, 1.0, lens.model == "telecentric")
    shift = float(np.hypot(*(moved - corner)[0]))
    if shift <= 0:
        return 1.0
    raw = 0.04 * lens.width / shift
    for step in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        if step >= raw:
            return float(step)
    return 1000.0


def _um_pair(lens: vp.LensSummary, short: bool = False) -> str:
    """Object size of one pixel, across x down (SCI-9): "26.9 × 26.1 µm" (``short`` drops
    nothing today; kept for chip call sites)."""
    x, y = lens.um_per_px or (float("nan"), float("nan"))
    return f"{x:.1f} × {y:.1f} µm"


def _transposed_text(check: vp.CentreCheck) -> str:
    """Why a saved image centre is flagged (SCI-6)."""
    return (f"Saved principal point ({check.saved[0]:g}, {check.saved[1]:g}) px is half the "
            f"image size read the wrong way round; the image centre is ({check.centre[0]:g}, "
            f"{check.centre[1]:g}) px. Runs before the 2026-10-08 fix did this; for a telecentric "
            "lens the fit is unaffected, but run Phase 2 again for a correct record.")


#: L1 wording of the fit-error Evidence column; D2.6 is named from L2.
_L1_RESIDUAL_EVIDENCE = {
    "matches D2.6": "recalculated; matches the saved value",
    "does not match D2.6": "recalculated; differs from the saved value",
    "matches D2.6: none recorded": "recalculated; no value, as saved",
}


def _plain_distortion_text(coeffs: Any) -> str:
    """L1 distortion cell: whether Phase 2 saved any distortion, without coefficient names."""
    if isinstance(coeffs, Mapping):
        coeffs = list(coeffs.values())
    if not isinstance(coeffs, Sequence) or isinstance(coeffs, (str, bytes)):
        return "unavailable"
    try:
        values = [float(value) for value in coeffs]
    except (TypeError, ValueError):
        return "unavailable"
    if all(value == 0.0 for value in values):
        return "none fitted (zero)"
    return "saved (coefficient values from L2 — details)"


# OpenCV's distortion-vector order, which the saved pinhole cameras use.
_OPENCV_DISTORTION_NAMES = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
                            "s1", "s2", "s3", "s4", "tx", "ty")


def _pose_summary(value: Any) -> str:
    """Render the recorded target-to-camera transform compactly and honestly."""
    try:
        pose = np.asarray(value, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            return "unavailable (invalid recorded pose)"
        matrix_text = "; ".join(",".join(_format_number(v) for v in row) for row in pose)
        return f"recorded seed target→camera pose (4×4): {matrix_text}"
    except Exception:
        return "unavailable (invalid recorded pose)"


def _removed_count(pruning: Any) -> int | None:
    """Return only an explicit removal count from the saved pruning record."""
    if isinstance(pruning, Mapping):
        for key in ("removed_count", "n_removed", "n_removed_observations", "removed"):
            value = pruning.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        removed_pairs = pruning.get("removed_pairs")
        if isinstance(removed_pairs, Mapping):
            counts = [len(items) for items in removed_pairs.values()
                      if isinstance(items, Sequence) and not isinstance(items, (str, bytes))]
            if counts:
                return sum(counts)
        if "removed_pairs" in pruning and not removed_pairs:
            return 0
    return None


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "unavailable"
    return f"{number:.4g}" if number == number and abs(number) != float("inf") else "unavailable"


def _is_missing(value: Any) -> bool:
    """True for a value D2.6 records as absent (None, or NaN before JSON)."""
    if value is None:
        return True
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return number != number


def _numbers_match(actual: Any, expected: Any) -> bool:
    """Use a narrow float tolerance for recomputed report values."""
    try:
        left, right = float(actual), float(expected)
    except (TypeError, ValueError, OverflowError):
        return False
    return left == right or abs(left - right) <= max(1e-9 * abs(left), 1e-9)


def _rms_comparison(recalculated: float, saved: Any) -> str:
    """Compare a recomputed per-view RMS with the saved one at pixel precision.

    The saved value comes from the seed calibration and the recalculation from
    cv2.projectPoints, so they agree only to about 1e-6 relative; a narrower
    tolerance would report identical displayed values as different.
    """
    try:
        saved_value = float(saved)
    except (TypeError, ValueError, OverflowError):
        return "recalculated; Phase 2 saved no value to compare"
    if not math.isfinite(saved_value):
        return "recalculated; Phase 2 saved no value to compare"
    difference = abs(recalculated - saved_value)
    if not math.isfinite(difference):
        return "recalculated; differs from the value Phase 2 saved"
    if difference <= max(1e-6, 1e-5 * abs(saved_value)):
        return "recalculated; matches the value Phase 2 saved"
    return f"recalculated; differs from the value Phase 2 saved by {difference:.2g} px"


def create_education_phase_view(context: Any, parent: QWidget | None = None) -> QWidget:
    """Factory consumed by the shared education dialog's lazy phase discovery."""
    return Phase2View(context, parent)
