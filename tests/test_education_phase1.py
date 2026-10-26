'''Test the visual Phase 1 education storyboard and its fail-closed photo identity.
the step-by-step stages are fed a synthetic explanation, so no detector runs here.
'''
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


#: Stage index by key (the view's own table; a sixth "Target" stage opens Phase 1).
STAGE = {key: k for k, key in enumerate(("target", "photos", "prep", "find", "final",
                                          "decision"))}


def text_collisions(figure):
    from pyCamSet.gui.education_style import text_collisions as check
    return check(figure)


def _photo(path: Path, value: int = 90) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((60, 80, 3), value, dtype=np.uint8)
    image[20:40, 30:50] = 20
    assert cv2.imwrite(str(path), image)


def _manifest(names, status="verified by manifest"):
    from pyCamSet.workflow.education_manifest import ManifestResult
    return ManifestResult(status, dict(names), {})


def _context(root: Path, cams=("cam0", "cam1"), diagnostics=None, params=None, run_id="p1-run"):
    diag = {"cam_names": list(cams)}
    diag.update(diagnostics or {})
    par = {"f_loc": str(root), "caching": False}
    par.update(params or {})
    return SimpleNamespace(run_id=run_id, run_dir=root, status="complete",
                           metadata={"params": par, "diagnostics": diag})


def _view(monkeypatch, root: Path, *, cams=("cam0", "cam1"), rows=None, max_ims=2,
          names=None, kind="manifest", status="verified by manifest", diagnostics=None,
          params=None, detections="default"):
    _app()
    from pyCamSet.calibration_targets.core.target_detections import TargetDetection
    from pyCamSet.gui import education_phase1

    if detections == "default":
        rows = rows if rows is not None else [[0, 0, 7, 10, 20], [1, 0, 8, 30, 40],
                                               [1, 1, 9, 31, 41], [0, 1, 7, 12, 22]]
        detections = TargetDetection(list(cams), np.asarray(rows, dtype=float), max_ims=max_ims)
    names = names if names is not None else {c: ("a.png", "b.png") for c in cams}
    monkeypatch.setattr(education_phase1, "_load_run_detections",
                        lambda _context: (detections, None if detections is not None
                                          else "this run's detections were not saved"))
    monkeypatch.setattr(education_phase1, "_source_state",
                        lambda _context: (_manifest(names, status), names, kind))
    monkeypatch.setattr(education_phase1.Phase1EducationView, "_read_trace",
                        lambda _view: {"status": "unavailable", "reason": "missing"})
    context = _context(root, cams, diagnostics, params)
    return education_phase1.Phase1EducationView(context)


def _close(view):
    view.close()
    view.deleteLater()
    _app().processEvents()


def _fake_explanation(consistent=True):
    """Every ArUco step kind the backend produces, on a small synthetic image."""
    from pyCamSet.workflow.education_detect_explain import (
        ConsistencyCheck, DetectionExplanation, ExplainStep, Inset, MarkerCrop, Overlay,
    )
    grey = np.full((120, 160), 128, np.uint8)
    rgb = np.dstack([grey] * 3)
    poly = np.array([[40, 40], [60, 40], [60, 60], [40, 60]], float)
    crop = np.zeros((64, 64), np.uint8)
    crop[8:56, 8:56] = 255
    after = np.array([[50.5, 50.2]])
    steps = [
        ExplainStep(key="input", title="Your photo", caption="", image=rgb,
                    counts={"width px": 160, "height px": 120, "bit depth": 16, "channels": 3}),
        ExplainStep(key="grayscale", title="Grey image", caption="", image=grey,
                    counts={"mean brightness": 128.0},
                    extra={"histogram": np.bincount(grey.ravel(), minlength=256)}),
        ExplainStep(key="detector_input", title="What the finder gets", caption="", image=grey,
                    counts={"width px": 160, "height px": 120, "upscaled": "no"}),
        ExplainStep(key="threshold", title="Black-and-white image", caption="", image=grey,
                    images=[("window 3 px", grey), ("window 13 px", grey)],
                    counts={"window sizes px": "3, 13", "constant": 7.0},
                    extra={"window_sizes": [3, 13], "representative": "window 13 px"},
                    source="reproduced"),
        ExplainStep(key="contours", title="Outlines", caption="", image=grey,
                    overlays=[Overlay("polylines", "outlines", [poly])],
                    counts={"outlines": 1, "four-cornered": 1}, source="reproduced"),
        ExplainStep(key="candidates", title="Square candidates", caption="", image=grey,
                    overlays=[Overlay("polygons", "not square", [poly + 30], states=["dropped"],
                                      reasons=["not convex"]),
                              Overlay("polygons", "candidates", [poly, poly + 50],
                                      states=["kept", "rejected"],
                                      reasons=["", "too little contrast to read"])],
                    counts={"candidates": 2, "became markers": 1, "rejected": 1, "not square": 1},
                    source="mixed"),
        ExplainStep(key="decoding", title="Reading each square", caption="", image=None,
                    crops=[MarkerCrop(crop, np.ones((6, 6), np.uint8), 4, "accepted", 5, 0, poly,
                                      detector_id="1:5"),
                           MarkerCrop(crop, None, 4, "too_uniform", -1, 0, poly + 50)],
                    counts={"read as an ID": 1, "bad border": 0, "unknown pattern": 0,
                            "too faint": 1}, source="reproduced"),
        ExplainStep(key="markers", title="Markers found", caption="", image=grey,
                    overlays=[Overlay("polygons", "markers", [poly], states=["accepted"],
                                      labels=["1:5"])],
                    counts={"markers": 1, "faces seen": 1}),
        ExplainStep(key="board_corners", title="Corners between markers", caption="", image=grey,
                    overlays=[Overlay("polygons", "markers", [poly], states=["accepted"]),
                              Overlay("points", "corners", np.array([[50, 50], [2, 2]], float),
                                      states=["kept", "dropped"], labels=["1:0", "1:1"],
                                      reasons=["", "too close to the image edge"])],
                    counts={"corners": 2, "too close to edge": 1}),
        ExplainStep(key="subpixel", title="Sub-pixel refinement", caption="", image=grey,
                    overlays=[Overlay("segments", "refinement",
                                      np.array([[[50, 50], [50.5, 50.2]]]), states=["moved"]),
                              Overlay("points", "after", after, states=["moved"])],
                    insets=[Inset("corner 1:0", np.zeros((96, 96), np.uint8), (44, 44), 8.0, 1.0,
                                  np.array([[50, 50]]), after, "0.54 px", "moved")],
                    counts={"corners": 1, "moved": 1, "median move px": 0.54,
                            "largest move px": 0.54},
                    extra={"shift_px": np.array([0.54])}),
        ExplainStep(key="grid_check", title="Do the neighbours agree?", caption="", image=grey,
                    overlays=[Overlay("points", "corners", after, states=["kept"])],
                    counts={"checked": 1, "kept": 1, "removed": 0},
                    extra={"residual_px": np.array([0.1]), "tolerance_px": np.array([1.0])}),
        ExplainStep(key="final", title="Result", caption="", image=rgb,
                    overlays=[Overlay("points", "detections", after, states=["accepted"],
                                      labels=["1:0"])],
                    counts={"points found": 1, "match saved run": "yes", "max shift px": 0.0}),
    ]
    check = ConsistencyCheck(True, consistent, 1, 1, matched=1, max_shift_px=0.0,
                             message="Live recompute matches the saved run: 1 points, "
                                     "largest difference 0 px.")
    return DetectionExplanation(steps, "aruco1", "Ccube", np.array([[1, 0]]), after, check, 0.01,
                                (120, 160, 3), "uint8")


# ---------------------------------------------------------------------------------------------
# The view: stages, pickers, captions
# ---------------------------------------------------------------------------------------------

@pytest.mark.gui
def test_phase1_view_uses_stage_pills_pickers_and_shared_navigation(tmp_path, monkeypatch):
    from pyCamSet.gui.education_phase1 import STAGES
    _photo(tmp_path / "cam0" / "a.png")
    view = _view(monkeypatch, tmp_path)
    try:
        assert [view.stepper.stage_name(i) for i in range(view.stepper.count())] == list(STAGES)
        from pyCamSet.gui.education_phase1 import STAGE as VIEW_STAGE
        assert VIEW_STAGE == STAGE and STAGES[0] == "Target"  # the opener comes first
        assert view.current_stage() == 0 and view._stage_index == 0
        # The shared pickers: "Camera n · name" with its badge, photos numbered from 1.
        assert view._camera.property("eduCameraPicker") is True
        assert view._camera.count() == 2 and view._camera.itemText(1) == "Camera 2 · cam1"
        assert not view._camera.itemIcon(0).isNull()  # camera badge in its colour
        assert [view._image.itemText(i) for i in range(view._image.count())] == \
            ["Photo 1", "Photo 2"]
        view.set_image_index(1)
        assert view.image_index() == 1 and view.photo_text() == "Photo 2"
        view.step_forward()
        assert view.current_stage() == 1
        view.step_back()
        view.step_back()  # no-op at the start
        assert view.current_stage() == 0
        last = len(STAGES) - 1
        view._set_stage(last)
        assert view.current_stage() == last
        view.step_forward()
        assert view.current_stage() == last
        assert "p1-run" in view.accessibleName()
    finally:
        _close(view)


@pytest.mark.gui
def test_photos_stage_grid_has_one_tile_per_camera_and_click_selects(tmp_path, monkeypatch):
    for cam in ("cam0", "cam1"):
        _photo(tmp_path / cam / "a.png")
        _photo(tmp_path / cam / "b.png")
    view = _view(monkeypatch, tmp_path)
    try:
        view.set_image_index(0)
        view._set_stage(STAGE["photos"])
        figure = view.figure()
        tiles = [ax for ax in figure.axes if ax.images]
        assert len(tiles) == 2
        assert "Each tile is one camera's photo number 1." in view.caption.text()
        assert [c.label for c in view.chips.chips()][:2] == ["cameras found it", "points, this photo"]
        assert text_collisions(figure) == []
        # Clicking a tile follows that camera.
        target = next(ax for ax, k in view._tile_axes.items() if k == 1)
        view._figure_clicked(SimpleNamespace(inaxes=target))
        assert view._camera.currentIndex() == 1
        # L2 draws the saved points on every tile.
        view.set_level(2)
        assert any(ax.collections for ax in view.figure().axes if ax.images)
        assert "Circles mark the saved points." in view.caption.text()
    finally:
        _close(view)


@pytest.mark.gui
def test_missing_run_pickle_never_substitutes_image_folder_cache(tmp_path, monkeypatch):
    view = _view(monkeypatch, tmp_path, cams=(), detections=None, names={}, kind="layout")
    try:
        assert "this run's detections were not saved" in view.caption.text()
        texts = [t.get_text() for ax in view.figure().axes for t in ax.texts]
        assert any("No run-private detection output" in t for t in texts)
        for stage in range(6):
            view._set_stage(stage)
            assert "no cache or other run is substituted" in view.caption.text()
    finally:
        _close(view)


@pytest.mark.gui
def test_photo_path_rejects_manifest_path_escape(tmp_path: Path):
    from pyCamSet.gui.education_phase1 import _photo_path

    escaped = _photo_path(tmp_path, "cam0", "../outside.png")
    assert escaped.name == "__unavailable__"


@pytest.mark.gui
def test_manifest_deletion_keeps_writer_name_and_never_shifts_to_next_photo(tmp_path, monkeypatch):
    _photo(tmp_path / "cam0" / "next.png")  # a later photo that must not be substituted
    view = _view(monkeypatch, tmp_path, cams=("cam0",), rows=[[0, 0, 1, 5, 6]], max_ims=1,
                 names={"cam0": ("deleted.png",)})
    try:
        view._set_stage(STAGE["final"])
        assert "deleted.png is no longer in the folder" in view.caption.text()
        assert "the photo name saved when detection ran" in view.caption.text()
        assert "next.png" not in view.caption.text()
        assert not any(ax.images for ax in view.figure().axes[:1])
        # The step-by-step stages cannot re-run on a photo that is gone.
        view._set_stage(STAGE["find"])
        assert "deleted.png is no longer in the folder" in view.caption.text()
    finally:
        _close(view)


@pytest.mark.gui
def test_manifest_mismatch_never_displays_or_re_detects_a_photo(tmp_path, monkeypatch):
    from pyCamSet.workflow import education_visual_p1 as vis
    _photo(tmp_path / "cam0" / "same-name.png")
    monkeypatch.setattr(vis, "read_display_photo",
                        lambda *_a, **_k: pytest.fail("mismatched source was displayed"))
    view = _view(monkeypatch, tmp_path, cams=("cam0",), rows=[[0, 0, 1, 5, 6]], max_ims=1,
                 names={"cam0": ("same-name.png",)}, status="mismatch")
    try:
        view._set_stage(STAGE["photos"])
        assert not any(ax.images for ax in view.figure().axes)
        view._set_stage(STAGE["final"])
        assert not view.figure().axes[0].images  # the heat map beside it is not a photo
        view._set_stage(STAGE["final"])
        assert "does not match the saved image manifest" in view.caption.text()
        assert isinstance(view.explanation(), RuntimeError)  # never re-run on that file
        view._set_stage(STAGE["find"])
        assert "does not match the saved image manifest" in view.caption.text()
        assert "mismatch" in view.provenance_text()
    finally:
        _close(view)


@pytest.mark.gui
def test_final_stage_reports_verified_image_decode_failure(tmp_path, monkeypatch):
    camera = tmp_path / "cam0"
    camera.mkdir()
    (camera / "corrupt.png").write_bytes(b"not decodable")
    view = _view(monkeypatch, tmp_path, cams=("cam0",), rows=[[0, 0, 1, 5, 6]], max_ims=1,
                 names={"cam0": ("corrupt.png",)})
    try:
        view._set_stage(STAGE["final"])
        assert "could not be decoded" in view.caption.text()
        assert not view.figure().axes[0].images
    finally:
        _close(view)


@pytest.mark.parametrize("kind,expected", [
    ("reuse", "earlier photo set"), ("count", "number of images has changed"),
    ("layout", "cannot be compared"), ("changed", "differ from the run's photo list"),
])
@pytest.mark.gui
def test_photoless_mapping_kinds_show_no_photo_and_explain_why(tmp_path, monkeypatch, kind, expected):
    from pyCamSet.gui.education_phase1 import _reason_caption
    _photo(tmp_path / "cam0" / "a.png")
    view = _view(monkeypatch, tmp_path, cams=("cam0",), rows=[[0, 0, 1, 5, 6]], max_ims=1,
                 names={}, kind=kind, status="could not be checked")
    try:
        view._set_stage(STAGE["photos"])
        assert not any(ax.images for ax in view.figure().axes)
        texts = " ".join(" ".join(t.get_text() for ax in view.figure().axes for t in ax.texts).split())
        assert expected in texts
        assert view.caption.text() == _reason_caption(kind)
        view._set_stage(STAGE["final"])
        assert view.caption.text() == _reason_caption(kind)
    finally:
        _close(view)


# ---------------------------------------------------------------------------------------------
# Stages 2-3: the live explanation
# ---------------------------------------------------------------------------------------------

@pytest.mark.gui
@pytest.mark.parametrize("level", [1, 2, 3])
def test_detector_stages_draw_every_step_with_a_check_chip(tmp_path, monkeypatch, level):
    from pyCamSet.gui.education_dialog import TransportBar
    _photo(tmp_path / "cam0" / "a.png")
    view = _view(monkeypatch, tmp_path, cams=("cam0",), rows=[[0, 0, 1, 50.5, 50.2]], max_ims=1,
                 names={"cam0": ("a.png",)})
    try:
        view._explain_cache[(view.camera_name(), view.image_index())] = _fake_explanation()
        view.set_level(level)
        view._set_stage(STAGE["prep"])
        strip = [ax for ax in view.figure().axes if ax.images]
        # photo -> (grey) -> one global cut-off vs the detector's local threshold
        assert len(strip) >= 4
        titles = " ".join(ax.get_title(loc="left") for ax in view.figure().axes)
        assert "One cut-off for all" in titles and "What the detector uses" in titles
        if level == 2:
            assert len(strip) >= 7  # + the example pixel's neighbourhood under each picture
        if level == 3:
            assert len(strip) >= 5  # + the other black-and-white image the detector tries
        assert view.caption_issues == []
        view._set_stage(STAGE["find"])
        bar = view.transport
        assert isinstance(bar, TransportBar) and view._controls is bar
        assert bar.isVisibleTo(view)
        assert bar.count() == 8  # contours .. final
        for index in range(bar.count()):
            bar.go_to(index)
            figure = view.figure()
            assert figure.axes, index
            labels = [c.label for c in view.chips.chips()]
            assert "points match saved run" in labels, labels
            assert view.caption.text() and view.caption_issues == [], (index, view.caption.text())
            # No chart that only repeats the chips (UX-14 / ED-29).
            assert not any("Counts" in ax.get_title(loc="left") for ax in figure.axes)
        # The reproduced pictures say so from L2, unobtrusively, in the subtitle.
        bar.go_to(1)
        subtitles = [t.get_text() for ax in view.figure().axes for t in ax.texts
                     if (t.get_gid() or "") == "edu:subtitle"]
        assert any("reproduced" in s for s in subtitles) is (level >= 2)
        # The gallery labels crops with the detector's verdict; faces count from 1.
        bar.go_to(2)
        titles = [ax.get_title() for ax in view.figure().axes]
        assert "ID 2:5" in titles
        assert ("✕ faint" in titles) is (level >= 2)  # L1 shows one "+1 not markers" tile
        bar.go_to(bar.count() - 1)
        assert "matches the saved result" in view.caption.text()
    finally:
        _close(view)


@pytest.mark.gui
def test_levels_add_drawn_panels_not_just_text(tmp_path, monkeypatch):
    for cam in ("cam0", "cam1"):
        _photo(tmp_path / cam / "a.png")
    view = _view(monkeypatch, tmp_path, rows=[[0, 0, 1, 50.5, 50.2], [1, 0, 1, 40, 40]],
                 max_ims=1, names={"cam0": ("a.png",), "cam1": ("a.png",)})
    try:
        view._explain_cache[(view.camera_name(), view.image_index())] = _fake_explanation()
        counts = {}
        for stage, step in ((STAGE["prep"], None), (STAGE["find"], 1), (STAGE["final"], None),
                           (STAGE["decision"], None)):
            per_level = []
            for level in (1, 2, 3):
                view.set_level(level)
                view._set_stage(stage)
                if step is not None:
                    view.transport.go_to(step)
                per_level.append(len(view.figure().axes))
            counts[stage] = per_level
            assert per_level[0] < per_level[2], (stage, per_level)
    finally:
        _close(view)


@pytest.mark.gui
def test_step_bar_plays_only_when_asked_and_stops_on_context_change(tmp_path, monkeypatch):
    _photo(tmp_path / "cam0" / "a.png")
    view = _view(monkeypatch, tmp_path, cams=("cam0",), rows=[[0, 0, 1, 50.5, 50.2]], max_ims=1,
                 names={"cam0": ("a.png",)})
    try:
        view._explain_cache[(view.camera_name(), view.image_index())] = _fake_explanation()
        view._set_stage(STAGE["find"])
        bar = view.transport
        assert not bar.is_playing() and bar.position() == 0
        assert not bar._previous.isEnabled() and bar._next.isEnabled()
        bar.step_forward()
        assert bar.position() == 1 and view._step_index == 1
        bar.toggle_playback()
        assert bar.is_playing()
        view.stop_for_context_change()
        assert not bar.is_playing()
        bar.toggle_playback()
        view._set_stage(STAGE["final"])  # leaving the stage hides and stops the controls
        assert not bar.is_playing() and not bar.isVisibleTo(view)
    finally:
        _close(view)


@pytest.mark.gui
def test_inconsistent_recompute_is_flagged_not_hidden(tmp_path, monkeypatch):
    _photo(tmp_path / "cam0" / "a.png")
    view = _view(monkeypatch, tmp_path, cams=("cam0",), rows=[[0, 0, 1, 50.5, 50.2]], max_ims=1,
                 names={"cam0": ("a.png",)})
    try:
        view._explain_cache[(view.camera_name(), view.image_index())] = _fake_explanation(False)
        view._set_stage(STAGE["find"])
        view.transport.go_to(view.transport.count() - 1)
        chips = {c.label: c for c in view.chips.chips()}
        check = next(c for c in chips.values() if c.label.startswith("saved points re-found"))
        assert check.tone == "warn" and "/" in check.value  # what differs, on the chip (ED2-18)
        assert "the saved points are the ones Phase 1 used" in view.caption.text()
    finally:
        _close(view)


# ---------------------------------------------------------------------------------------------
# Stages 4-5: final and decision
# ---------------------------------------------------------------------------------------------

@pytest.mark.gui
@pytest.mark.parametrize(("target_type", "divisor"), [("ChArUco", 2.0), ("PuzzleBoard", 1.0)])
def test_final_overlay_scale_follows_the_effective_rescale_setting(
        tmp_path, monkeypatch, target_type, divisor):
    """In a run saved before 2026-10-13 (no rescale_and_gamma_any_target key) a ticked
    rescale-and-gamma box was ignored for non-PuzzleBoard targets, so the detector worked on
    the x2 enlarged photo and the overlay must divide by 2."""
    from pyCamSet.gui.education_phase1 import _frame_to_photo
    params = {"upscale_factor": 2, "rescale_and_gamma": True, "target": {"type": target_type}}
    assert _frame_to_photo(params) == pytest.approx(1.0 / divisor)
    _photo(tmp_path / "cam0" / "a.png")
    view = _view(monkeypatch, tmp_path, cams=("cam0",), rows=[[0, 0, 1, 20.0, 30.0]], max_ims=1,
                 names={"cam0": ("a.png",)}, params=params)
    try:
        view._set_stage(STAGE["final"])
        ax = view.figure().axes[0]
        assert ax.images
        offsets = ax.collections[0].get_offsets()
        assert np.allclose(offsets, [[20.0 / divisor, 30.0 / divisor]])
    finally:
        _close(view)


@pytest.mark.gui
def test_final_stage_shows_photo_and_this_cameras_coverage(tmp_path, monkeypatch):
    for name in ("a.png", "b.png"):
        _photo(tmp_path / "cam0" / name)
        _photo(tmp_path / "cam1" / name)
    view = _view(monkeypatch, tmp_path, diagnostics={
        "D1.6_spatial_coverage": {"cam0": 0.4, "cam1": 0.2},
        "D1.2_detection_rate": {"cam0": 1.0, "cam1": 1.0}})
    try:
        view._set_stage(STAGE["final"])
        titles = [ax.get_title(loc="left") for ax in view.figure().axes]
        assert any("Points found" in t for t in titles)
        assert any("Where its points fall" in t for t in titles)
        labels = {c.label: c.value for c in view.chips.chips()}
        assert labels.get("area covered", "40 %") == "40 %"
        rows = {view.numbers.table.item(r, 0).text(): view.numbers.table.item(r, 3).text()
                for r in range(view.numbers.table.rowCount())}
        assert rows == {"cam0": "40.0 %", "cam1": "20.0 %"}
        view.set_level(2)  # every camera's map
        assert len([ax for ax in view.figure().axes if ax.images]) >= 3
    finally:
        _close(view)


@pytest.mark.gui
def test_decision_bars_are_the_saved_rates_and_a_blind_camera_is_flagged(tmp_path, monkeypatch):
    view = _view(monkeypatch, tmp_path, cams=("cam0", "cam1", "cam2"),
                 rows=[[0, 0, 1, 5, 6], [1, 0, 2, 5, 6]], max_ims=1,
                 names={"cam0": ("a.png",), "cam1": ("a.png",), "cam2": ("a.png",)},
                 diagnostics={"D1.1_total_detections": {"cam0": 1, "cam1": 1, "cam2": 0},
                              "D1.2_detection_rate": {"cam0": 0.75, "cam1": 0.5, "cam2": 0.0},
                              "D1.3_board_completeness": {"cam0": 0.4, "cam1": 0.3, "cam2": 0.0},
                              "D1.6_spatial_coverage": {"cam0": 0.4, "cam1": 0.2}})
    try:
        view._set_stage(STAGE["decision"])
        rate_axes = view.figure().axes[0]
        heights = [p.get_height() for p in rate_axes.patches]
        assert heights[:3] == pytest.approx([75.0, 50.0, 0.0])
        assert "A camera with no points is stopped" in view.caption.text()
        assert "never found" not in view.caption.text()
        chips = {c.label: c for c in view.chips.chips()}
        assert chips["cameras kept"].value == "2 / 3" and chips["cameras kept"].tone == "bad"
        assert len(view.figure().axes) == 2
        view.set_level(2)
        assert len(view.figure().axes) == 4
        assert "grading" in view.caption.text()
        view.set_level(3)
        assert len(view.figure().axes) >= 5  # + who saw the target in which photo
        # three cameras plus the rule-of-thumb row (SCI7-4: where the 500 comes from)
        assert view.numbers.table.rowCount() == 4
        statuses = [view.numbers.table.item(r, 5).text() for r in range(3)]
        assert statuses == ["kept", "kept", "stopped: no points"]
        assert "rule of thumb" in view.numbers.table.item(3, 1).text()
    finally:
        _close(view)


@pytest.mark.gui
def test_theme_switch_redraws_on_the_theme_surface(tmp_path, monkeypatch):
    from matplotlib.colors import to_hex
    from pyCamSet.gui.theme import THEME_TOKENS
    app = _app()
    _photo(tmp_path / "cam0" / "a.png")
    view = _view(monkeypatch, tmp_path, cams=("cam0",), rows=[[0, 0, 1, 5, 6]], max_ims=1,
                 names={"cam0": ("a.png",)})
    try:
        for theme in ("Dark", "Sepia", "Light"):
            app.setProperty("pycamsetTheme", theme)
            view._on_theme_changed(theme)
            assert to_hex(view.figure().get_facecolor()) == THEME_TOKENS[theme]["surface"]
    finally:
        app.setProperty("pycamsetTheme", "Light")
        _close(view)


@pytest.mark.gui
@pytest.mark.parametrize("size", [(1400, 900), (1280, 800)])
def test_every_stage_fits_without_vertical_scrolling(tmp_path, monkeypatch, size):
    from PySide6.QtWidgets import QScrollArea
    app = _app()
    _photo(tmp_path / "cam0" / "a.png")
    view = _view(monkeypatch, tmp_path, cams=("cam0",), rows=[[0, 0, 1, 50.5, 50.2]], max_ims=1,
                 names={"cam0": ("a.png",)},
                 diagnostics={"D1.2_detection_rate": {"cam0": 1.0}})
    view._explain_cache[(view.camera_name(), view.image_index())] = _fake_explanation()
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(view)
    scroll.resize(size[0] - 24, size[1] - 70)  # the dialog's header row and margins
    scroll.show()
    try:
        for level in (1, 2, 3):
            view.set_level(level)
            for stage in range(6):
                view._set_stage(stage)
                app.processEvents()
                assert scroll.verticalScrollBar().maximum() == 0, (level, stage)
                # Font-less offscreen test fonts are wide; with Segoe UI the review
                # screenshots give the visual >= 70 % of the dialog's content height.
                share = view.visual.height() / scroll.viewport().height()
                assert share >= 0.5, (level, stage, share, view.caption.height())
    finally:
        scroll.close()
        _close(view)


# ---------------------------------------------------------------------------------------------
# Kept data-correctness contracts (photo identity, trace records, copy)
# ---------------------------------------------------------------------------------------------

@pytest.mark.gui
def test_trace_records_are_matched_exactly_never_borrowed(tmp_path, monkeypatch):
    view = _view(monkeypatch, tmp_path, cams=("camA", "camB"), rows=[[0, 0, 1, 3, 4]],
                 names={"camA": ("a.png", "b.png"), "camB": ("a.png", "b.png")})
    try:
        view._trace = {"status": "available", "records": [
            {"kind": "stage", "payload": {"camera": "camA", "image_index": 0,
             "stages": [{"name": "aruco_markers", "marker_ids": [4, 9]}]}},
            {"kind": "stage", "payload": {"camera": "camB", "image_index": 1,
             "stages": [{"name": "wrong-record", "marker_ids": [99]}]}}]}
        stages, reason = view._trace_stages_for_selection("camA", 0)
        assert [s["name"] for s in stages] == ["aruco_markers"] and reason is None
        stages, reason = view._trace_stages_for_selection("camB", 0)
        assert stages == [] and "another record was not substituted" in reason
        stages, _ = view._trace_stages_for_selection("camB", 1)
        assert stages[0]["name"] == "wrong-record"
    finally:
        _close(view)


@pytest.mark.gui
def test_copy_branches_and_stage_budget_are_explicit():
    from pyCamSet.gui.education_phase1 import STAGES, _PHOTOLESS_COPY, _reason_caption

    assert len(STAGES) <= 6
    assert {"reuse", "layout", "count", "changed"} <= set(_PHOTOLESS_COPY)
    assert "remove the matching `detected_datapoints*.npz`" in _reason_caption("reuse")
    assert "folder count changed" not in _reason_caption("count")
    assert "photo list of its detection run" in _reason_caption("changed")


def _legacy_context(root: Path, *, caching: bool, expected: int, observed: int,
                    max_ims: int, n_lim: int | None = None, reused: bool = False):
    """Build frozen-shaped legacy metadata from explicit detector-time counts."""
    from pyCamSet.workflow.education_manifest import ManifestResult

    context = SimpleNamespace(
        run_id="legacy-run", run_dir=root,
        metadata={
            "params": {"f_loc": str(root), "caching": caching, "n_lim": n_lim},
            "report": {"n_images": observed,
                       "per_camera": [{"name": "cam0", "n_images_expected": expected}]},
            "diagnostics": {"cam_names": ["cam0"],
                            "education_cache": {"reused": reused}},
        },
        phase1_source_check=lambda: ManifestResult(
            "source identity not recorded", {}, {}, "Phase 1 manifest missing"),
    )
    return context, max_ims


@pytest.mark.parametrize(
    "case,caching,expected,observed,max_ims,n_lim,reused,subdirs,expected_key",
    [
        ("legacy-nested-cache-off", False, 2, 2, 2, None, False,
         ("nested",), "layout"),
        ("flat-post-delete-count", False, 3, 3, 2, None, False,
         (), "count"),
        ("case-r-delete-hit-restore", True, 2, 3, 3, None, True,
         (), "reuse"),
        ("legacy-capped-n-lim", False, 2, 2, 2, 2, False,
         (), "layout"),
    ],
)
@pytest.mark.gui
def test_source_state_legacy_provenance_matrix(
        tmp_path, monkeypatch, case, caching, expected, observed, max_ims,
        n_lim, reused, subdirs, expected_key):
    from pyCamSet.gui import education_phase1

    camera = tmp_path / "cam0"
    image_folder = camera.joinpath(*subdirs)
    image_folder.mkdir(parents=True)
    if subdirs:
        nested_folder = image_folder / "capture"
        nested_folder.mkdir()
    else:
        nested_folder = image_folder
    file_count = 2 if case == "flat-post-delete-count" else 3
    for index in range(file_count):
        (nested_folder / f"frame{index}.png").write_bytes(b"frozen source photo")
    context, recorded_max_ims = _legacy_context(
        tmp_path, caching=caching, expected=expected, observed=observed,
        max_ims=max_ims, n_lim=n_lim, reused=reused)
    if subdirs:
        context.metadata["params"]["subfolder_string"] = subdirs[0]
    monkeypatch.setattr(education_phase1, "_load_run_detections",
                        lambda _context: (SimpleNamespace(max_ims=recorded_max_ims), None))

    result, photo_names, mapping_kind = education_phase1._source_state(context)

    assert mapping_kind == expected_key, case
    assert not photo_names, case
    if case == "flat-post-delete-count":
        assert "folder" in (result.reason or "")
    if case == "case-r-delete-hit-restore":
        assert "earlier photo set" in education_phase1._reason_caption(mapping_kind)


@pytest.mark.gui
def test_verified_manifest_keeps_trailing_undetected_blank_photo_slot(tmp_path, monkeypatch):
    from pyCamSet.gui import education_phase1
    from pyCamSet.workflow.education_manifest import (
        ManifestResult, describe_enumeration, verify,
    )
    from pyCamSet.calibration_targets.core.abstract_target import enumerate_detection_images

    camera = tmp_path / "cam0"
    camera.mkdir()
    first = camera / "frame0.png"
    trailing_blank = camera / "frame1.png"
    first.write_bytes(b"detector input")
    trailing_blank.write_bytes(b"")
    enumeration = tuple(enumerate_detection_images(camera))
    manifest = describe_enumeration({"cam0": enumeration}, tmp_path)
    assert tuple(item.name for item in enumeration) == ("frame0.png", "frame1.png")
    context = SimpleNamespace(
        run_id="manifest-run", run_dir=tmp_path, status="complete",
        metadata={"params": {"f_loc": str(tmp_path)},
                  "diagnostics": {"cam_names": ["cam0"],
                                  "education_manifest": manifest}},
        phase1_source_check=lambda: verify(manifest, tmp_path, ("cam0",)),
    )
    detections = SimpleNamespace(max_ims=1)
    original_loader = education_phase1._load_run_detections
    try:
        education_phase1._load_run_detections = lambda _context: (detections, None)
        result, names, kind = education_phase1._source_state(context)
    finally:
        education_phase1._load_run_detections = original_loader

    assert isinstance(result, ManifestResult)
    assert result.status == "verified by manifest"
    assert kind == "manifest"
    assert names["cam0"] == ("frame0.png", "frame1.png")
    assert detections.max_ims < len(names["cam0"])
    _app()
    from pyCamSet.calibration_targets.core.target_detections import TargetDetection
    saved = TargetDetection(["cam0"], np.asarray([[0, 0, 1, 3, 4]], dtype=float), max_ims=1)
    monkeypatch.setattr(education_phase1, "_load_run_detections", lambda _context: (saved, None))
    monkeypatch.setattr(education_phase1.Phase1EducationView, "_read_trace",
                        lambda _view: {"status": "unavailable", "reason": "missing"})
    view = education_phase1.Phase1EducationView(context)
    try:
        assert view._image.count() == 2  # Photos must reach the blank last photo.
        view.set_image_index(1)
        assert view.image_index() == 1 and view._image.currentText() == "Photo 2"
    finally:
        _close(view)


@pytest.mark.gui
def test_manifest_mismatch_and_not_checkable_are_distinct(tmp_path):
    from pyCamSet.gui import education_phase1
    from pyCamSet.workflow.education_manifest import describe_enumeration, verify
    from pyCamSet.calibration_targets.core.abstract_target import enumerate_detection_images

    camera = tmp_path / "cam0"
    camera.mkdir()
    photo = camera / "same.png"
    photo.write_bytes(b"writer-time bytes")
    manifest = describe_enumeration(
        {"cam0": tuple(enumerate_detection_images(camera))}, tmp_path)
    metadata = {"params": {"f_loc": str(tmp_path)},
                "diagnostics": {"cam_names": ["cam0"],
                                "education_manifest": manifest}}
    photo.write_bytes(b"different current bytes")
    mismatch = SimpleNamespace(metadata=metadata,
                               phase1_source_check=lambda: verify(manifest, tmp_path, ("cam0",)))
    result, names, kind = education_phase1._source_state(mismatch)
    assert result.status == "mismatch" and kind == "manifest"
    assert names["cam0"] == ("same.png",)

    photo.unlink()
    camera.rmdir()
    uncheckable = SimpleNamespace(
        metadata=metadata,
        phase1_source_check=lambda: verify(manifest, tmp_path, ("cam0",)))
    result, _names, kind = education_phase1._source_state(uncheckable)
    assert result.status == "could not be checked" and kind == "manifest"


@pytest.mark.gui
def test_f1a_phase_factory_opens_phase1_view_without_starting_detection(monkeypatch):
    _app()
    from pyCamSet.gui import education_phase1
    from pyCamSet.gui.education_dialog import EducationRunView, discover_phase_view
    from pyCamSet.workflow.education_manifest import ManifestResult

    detections = SimpleNamespace(cam_names=["cam0"], max_ims=1, get_data=lambda: None)
    calls = []
    monkeypatch.setattr(education_phase1, "_load_run_detections",
                        lambda context: (calls.append(context.run_id) or detections, None))
    monkeypatch.setattr(education_phase1, "_source_state", lambda _context: (
        ManifestResult("could not be checked", {}, {}, "fixture"), {}, "layout"))
    monkeypatch.setattr(education_phase1.Phase1EducationView, "_read_trace",
                        lambda _view: {"status": "unavailable", "reason": "fixture"})
    run = EducationRunView(
        "phase1", "factory-run", Path("."), "complete",
        {"diagnostics": {"cam_names": ["cam0"]}, "params": {}}, object())
    view = discover_phase_view(run)
    try:
        assert isinstance(view, education_phase1.Phase1EducationView)
        assert "factory-run" in view.accessibleName()
        assert calls == ["factory-run"]
        assert view._future is None  # no photo to explain: nothing was started
    finally:
        _close(view)


@pytest.mark.parametrize(("params", "expected"), [
    ({"upscale_factor": 2, "rescale_and_gamma": False},
     "For this run, pyCamSet enlarged each photo ×2 before the detector searched it"),
    ({"upscale_factor": 1, "rescale_and_gamma": False},
     "the detector searched each photo at its original size"),
    ({"upscale_factor": 3, "rescale_and_gamma": True, "preprocessing_scale": 0.25,
      "preprocessing_gamma": 0.5, "target": {"type": "PuzzleBoard"}},
     "resized each photo ×0.25 and adjusted its brightness and contrast (gamma 0.5)"),
    ({"upscale_factor": 2, "rescale_and_gamma": True, "preprocessing_scale": 0.25,
      "preprocessing_gamma": 0.5, "target": {"type": "ChArUco"}},
     "Rescale and gamma was ticked, but runs saved before 2026-10-13 applied it only to "
     "PuzzleBoard targets, so it was not used for this run. For this run, pyCamSet enlarged "
     "each photo ×2"),
    ({"upscale_factor": 2, "rescale_and_gamma": True},
     "the target type was not recorded, so what the detector searched is unknown"),
    ({}, "detector settings were not recorded"),
])
def test_detector_input_copy_states_this_runs_settings(params, expected):
    """A1 section 10: say what the detector searched for this run (shown at L3, stage 2)."""
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_phase1 import _detector_input_copy
    assert expected in _detector_input_copy(params)


# ---------------------------------------------------------------------------------------------
# Qt-free helpers
# ---------------------------------------------------------------------------------------------

def test_visual_helpers_choose_images_close_ups_and_crops():
    from pyCamSet.calibration_targets.core.target_detections import TargetDetection
    from pyCamSet.workflow import education_visual_p1 as vis

    rows = [[0, 0, k, 1, 1] for k in range(3)] + [[1, 1, k, 1, 1] for k in range(5)] \
        + [[0, 1, 9, 2, 2]]
    det = TargetDetection(["a", "b"], np.asarray(rows, dtype=float), max_ims=2)
    counts = vis.points_per_image(det)
    assert counts.tolist() == [[3, 0], [1, 5]]
    assert vis.busiest_image(det) == 1
    assert vis.busiest_image(det, camera=0) in (0, 1)
    explanation = _fake_explanation()
    box = vis.focus_box(explanation, vis.frame_size(explanation))
    assert box[0] <= 50 <= box[2] and box[1] <= 50 <= box[3]  # the marker is in the close-up
    assert box[2] - box[0] == pytest.approx(box[3] - box[1])
    split = vis.split_steps(explanation)
    assert [s.key for s in vis.prep_strip(split.prep)] == ["input", "grayscale", "threshold"]
    assert split.find[0].key == "contours" and split.find[-1].key == "final"
    crops = explanation.step("decoding").crops * 10
    chosen = vis.select_crops(crops, 6)
    assert len(chosen) == 6 and any(c.detector_id is None for c in chosen)
    hull = vis.hull_polygon(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], float))
    assert len(hull) == 4
    grid, extent = vis.coverage_grid(np.array([[1.0, 1.0]]), 100, 50, 10)
    assert grid.sum() == 1 and grid.shape == (5, 10) and extent == (0.0, 100, 50, 0.0)
    assert vis.rate_band(0.95) == "good" and vis.rate_band(0.7) == "warn" and vis.rate_band(0.2) == "bad"


def test_l1_step_captions_are_short_and_plain():
    from pyCamSet.workflow import education_visual_p1 as vis
    explanation = _fake_explanation()
    for step in vis.split_steps(explanation).find:
        text = vis.step_caption(step, 1, consistency=explanation.consistency)
        sentences = [s for s in text.replace("?", ".").split(". ") if s.strip()]
        assert len(sentences) <= 3, (step.key, text)  # 2 + the 'none here' / check note
        assert "OpenCV" not in text and "D1." not in text
        assert len(vis.step_caption(step, 3)) > len(vis.step_caption(step, 1))
        from pyCamSet.gui.education_style import caption_problems
        for level in (1, 2, 3):  # the layout contract: <= 2 / 3 / 4 sentences
            assert caption_problems(vis.step_caption(
                step, level, consistency=explanation.consistency), level) == [], (step.key, level)


# ---------------------------------------------------------------------------------------------
# Round-1 review additions: the target opener, thresholding, IDs, counts that add up
# ---------------------------------------------------------------------------------------------

def _cube_target():
    """A two-face design (metres, like Ccube): faces 0 and 1 meet at a right angle."""
    grid = np.array([[x, y] for y in (0.0, 0.002, 0.004) for x in (0.0, 0.002, 0.004)])
    face0 = np.column_stack([grid, np.zeros(len(grid))])                       # z = 0 plane
    face1 = np.column_stack([np.zeros(len(grid)), grid[:, 0], grid[:, 1]])      # x = 0 plane
    return SimpleNamespace(point_data=np.stack([face0, face1]), textures=None)


@pytest.mark.gui
@pytest.mark.parametrize("level", [1, 2, 3])
def test_target_stage_opens_with_this_runs_target_and_faces(tmp_path, monkeypatch, level):
    from pyCamSet.gui import education_phase1
    _photo(tmp_path / "cam0" / "a.png")
    _photo(tmp_path / "cam1" / "a.png")
    rows = [[0, 0, 0, 1, 20, 20], [0, 0, 1, 4, 30, 25], [1, 0, 1, 2, 40, 30]]
    view = _view(monkeypatch, tmp_path, rows=rows, max_ims=1,
                 names={"cam0": ("a.png",), "cam1": ("a.png",)})
    try:
        monkeypatch.setattr(view, "_explain_target", lambda: _cube_target())
        view._summary_cache = None
        view.set_level(level)
        view._set_stage(STAGE["photos"])
        view._set_stage(STAGE["target"])
        figure = view.figure()
        assert any(child.name == "3d" for ax in figure.axes                # the target itself
                   for child in ax.child_axes)
        photo = [ax for ax in figure.axes if ax.images]
        assert photo and photo[0].collections                         # its corners, by face
        text = view.caption.text()
        assert "calibration target" in text and "2-faced" in text
        assert view.caption_issues == []
        chips = {c.label: c.value for c in view.chips.chips()}
        assert chips["faces"] == "2" and chips["corners on the target"] == "18"
        titles = " ".join(ax.get_title(loc="left") for ax in figure.axes)
        assert ("Which camera sees which face" in titles) is (level >= 2)
        assert text_collisions(figure) == []
        summary = view._summary()
        assert summary.seen.tolist() == [[1, 1], [0, 1]]               # photos per camera, face
        assert summary.size_mm == pytest.approx(4.0)
        assert education_phase1.face_colours("Light", 2)[0] != education_phase1.face_colours(
            "Light", 2)[1]
    finally:
        _close(view)


def test_face_colours_are_never_camera_or_meaning_colours():
    """UX2-2 / ED2-8: Phase 1 uses the one shared face palette (neutral shades, no hue, so a
    face never reads as a camera; identity comes from the square "F" tags)."""
    pytest.importorskip("PySide6")
    _app()
    from pyCamSet.gui.education_phase1 import face_colours
    from pyCamSet.gui import education_style as es
    for theme in ("Light", "Dark", "Sepia"):
        colours = face_colours(theme, 6)
        assert colours == es.face_colours(theme, 6)
        assert len(set(colours)) == 3, (theme, colours)
        for c in colours:
            _l, a, b = es._lab(c)
            assert np.hypot(a, b) <= 4.0, (theme, c)          # no hue: never a camera colour
            t = es.stage_tokens(theme)
            assert min(es.colour_distance(c, t[r]) for r in ("accent", "good", "warn", "bad")) >= 20


def _threshold_explanation():
    """A photo whose right half is dim: one cut-off loses it, the local threshold does not."""
    import cv2
    from pyCamSet.workflow.education_detect_explain import ExplainStep
    rng = np.random.default_rng(1)
    grey = np.zeros((120, 160), np.uint8)
    for y in range(0, 120, 20):
        for x in range(0, 160, 20):
            bright = ((x // 20) + (y // 20)) % 2 == 0
            level = (210 if bright else 120) if x < 80 else (70 if bright else 25)
            grey[y:y + 20, x:x + 20] = level
    grey = np.clip(grey + rng.normal(0, 2, grey.shape), 0, 255).astype(np.uint8)
    local = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                  13, 7)
    steps = [
        ExplainStep(key="input", title="Your photo", caption="", image=grey,
                    counts={"width px": 160, "height px": 120, "bit depth": 8, "channels": 1}),
        ExplainStep(key="detector_input", title="", caption="", image=grey,
                    counts={"upscaled": "no"}),
        ExplainStep(key="threshold", title="", caption="", image=local, source="reproduced",
                    images=[("window 13 px", local)],
                    extra={"window_sizes": [13], "constant": 7.0, "rule": "opencv",
                           "representative_index": 0, "display_per_prepared": 1.0}),
    ]
    return SimpleNamespace(steps=steps, image_points=np.zeros((0, 2)),
                           step=lambda key: next((s for s in steps if s.key == key), None))


def test_threshold_demo_shows_why_the_threshold_is_local():
    from pyCamSet.workflow import education_visual_p1 as vis
    demo = vis.threshold_demo(_threshold_explanation())
    assert demo is not None and demo.window == 13 and demo.constant == 7.0
    # One global cut-off blackens almost all of the dim half ...
    assert demo.dim_share_lost > 0.9
    # ... and the example pixel is one where the two rules disagree, with the local verdict
    # the detector's own image shows, and the numbers that decide it.
    x, y = demo.probe
    assert demo.local_black == bool(demo.local[y, x])
    assert demo.local_black != demo.global_says_black
    assert demo.local_black == (demo.mean_value - demo.centre_value >= demo.constant)
    assert vis.threshold_layer_name("black markers, eroded 1x")[0] == "Dark markers"
    assert "swapped" in vis.threshold_layer_name("white markers (inverted)")[1]


@pytest.mark.gui
def test_preprocessing_figure_draws_window_global_and_local(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    import matplotlib
    from pyCamSet.gui.education_phase1 import draw_preprocessing
    from pyCamSet.gui.education_style import align_panel_titles, figure_style
    _app()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    for level in (1, 2, 3):
        fig = Figure(figsize=(13.6, 6.5), dpi=100)
        FigureCanvasAgg(fig)  # a real renderer, so the checker measures the drawn layout
        with matplotlib.rc_context(figure_style("Light")):
            align_panel_titles(fig)
            draw_preprocessing(fig, _threshold_explanation(), level)
        titles = [ax.get_title(loc="left") for ax in fig.axes]
        joined = " ".join(titles)
        assert "Your photo" in joined and "One cut-off for all" in joined
        assert "Grey" not in joined  # a grey file is not shown twice (ED-9)
        subtitles = " ".join(t.get_text() for ax in fig.axes for t in ax.texts
                             if (t.get_gid() or "") == "edu:subtitle")
        assert "8-bit grey file" in subtitles and "colour" not in subtitles  # SCI-12
        if level == 2:
            assert "One pixel, enlarged" in joined and "its square's mean" in subtitles
        if level == 3:
            assert "Each pixel's own test" in joined
        assert text_collisions(fig) == [], level


def test_read_ids_lookup_matches_the_dictionary_in_either_polarity():
    import cv2
    from pyCamSet.workflow.education_detect_explain import MarkerCrop, _attach_lookup
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    entry = np.asarray(dictionary.bytesList)[7]
    code = np.asarray(cv2.aruco.Dictionary.getBitsFromByteList(entry.reshape(1, *entry.shape), 4))
    for inverted in (False, True):
        raw = np.ones((6, 6), np.uint8)               # black ring, 1 = black
        raw[1:-1, 1:-1] = 1 - np.rot90(code, 1)       # the marker seen turned by 90 degrees
        seen = (1 - raw) if inverted else raw          # a white marker looks swapped
        crop = MarkerCrop(np.zeros((64, 64), np.uint8), seen.astype(np.uint8), 4, "accepted",
                          7, 1, np.zeros((4, 2)), inverted=inverted)
        _attach_lookup(crop, dictionary)
        assert np.array_equal(crop.code, code) and crop.bit_errors == 0
        assert crop.turns in (1, 3) and crop.dictionary_size == 50
        # The match, then the closest *other* code over its four turns (SCI2-15).
        assert crop.lookup[0][0] == 7 and len(crop.lookup) == 2
        other, other_bits = crop.lookup[1]
        assert other != 7
        all_bits = [np.asarray(cv2.aruco.Dictionary.getBitsFromByteList(
            np.asarray(dictionary.bytesList)[k].reshape(1, 2, 4), 4)) for k in range(50)]
        nearest = min(min(int(np.sum(np.rot90(code, r) != b)) for r in range(4))
                      for k, b in enumerate(all_bits) if k != 7)
        assert int(np.sum(np.asarray(other_bits) != crop.code)) == nearest


def test_square_counts_add_up_across_chips_and_bars():
    from pyCamSet.workflow import education_visual_p1 as vis
    explanation = _fake_explanation()
    candidates = explanation.step("candidates")
    chips = {label: int(value) for value, label, _t in vis.step_chips(candidates)}
    counts = candidates.counts
    assert chips["became markers"] + chips["dropped"] == \
        counts["candidates"] + counts["not square"]
    decoding = explanation.step("decoding")
    verdicts = vis.decode_counts(decoding)
    assert sum(verdicts.values()) == len(decoding.crops)
    assert verdicts["marker"] == 1 and verdicts["too_uniform"] == 1
    assert [label for _v, label, _t in vis.step_chips(decoding)] == ["markers read", "too faint"]
    check = vis.consistency_chip(explanation.consistency)
    assert check[1] == "points match saved run"  # plain words, no "= saved" (ED-18)
    assert vis.corner_name("1:25", 6) == "F2 · corner 25"


@pytest.mark.gui
def test_every_stage_level_and_step_has_clean_text(tmp_path, monkeypatch):
    """Adoption checklist: no figure text collisions and captions within their limits."""
    for cam in ("cam0", "cam1"):
        _photo(tmp_path / cam / "a.png")
    view = _view(monkeypatch, tmp_path, rows=[[0, 0, 1, 50.5, 50.2], [1, 0, 1, 40, 40]],
                 max_ims=1, names={"cam0": ("a.png",), "cam1": ("a.png",)},
                 diagnostics={"D1.2_detection_rate": {"cam0": 1.0, "cam1": 1.0},
                              "D1.6_spatial_coverage": {"cam0": 0.4, "cam1": 0.2}})
    try:
        view.resize(1360, 820)
        view.show()  # the figure takes the canvas's real size, as in the dialog
        _app().processEvents()
        view._explain_cache[(view.camera_name(), view.image_index())] = _fake_explanation()
        for level in (1, 2, 3):
            view.set_level(level)
            for stage in range(6):
                view._set_stage(stage)
                steps = range(view.transport.count()) if stage == STAGE["find"] else [None]
                for step in steps:
                    if step is not None:
                        view.transport.go_to(step)
                    assert view.caption_issues == [], (level, stage, step, view.caption.text())
                    assert text_collisions(view.figure()) == [], (level, stage, step)
    finally:
        _close(view)


@pytest.mark.gui
@pytest.mark.data
@pytest.mark.slow
def test_real_saved_run_explains_live_and_refuses_a_changed_photo(data_dir, tmp_path, monkeypatch):
    import shutil
    from pyCamSet.workflow import phase1
    from pyCamSet.workflow.education_trace import RunContext
    from pyCamSet.workflow.workspace import WorkspaceManager, workspace_path_for
    from pyCamSet.gui.education_phase1 import Phase1EducationView

    images = tmp_path / "calibration_charuco"
    shutil.copytree(data_dir / "calibration_charuco", images)
    manager = WorkspaceManager(workspace_path_for(images))
    params = {"f_loc": str(images), "target": {
        "type": "ChArUco", "num_squares_x": 20, "num_squares_y": 20,
        "square_size": 4, "legacy": False}, "selected_cameras": [],
        "n_lim": None, "caching": False, "upscale_factor": 1,
        "rescale_and_gamma": False, "manifest_digest": True,
        "record_phase1_trace": True, "threads": 2}
    saved = phase1.run(params, manager, log=lambda _line: None)
    assert saved["error"] is None
    context = RunContext.from_run(manager, {
        **saved, "_run_dir": manager.run_dir("phase1", saved["run_id"])}, "phase1")
    assert context.phase1_source_check().status == "verified by manifest"
    monkeypatch.setattr(phase1, "detect_datapoints_in_imfile",
                        lambda **_kwargs: pytest.fail("opening view re-ran folder detection"))
    _app()
    view = Phase1EducationView(context)
    try:
        assert view._detections is not None and view._mapping_kind == "manifest"
        explanation = view.wait_for_explanation(timeout=120)
        assert explanation is not None and not isinstance(explanation, Exception)
        assert explanation.consistency.available and explanation.consistency.ok
        view._set_stage(STAGE["find"])
        assert view.transport.count() >= 5
        assert any(c.label == "points match saved run" for c in view.chips.chips())
        view._set_stage(STAGE["final"])
        assert view.figure().axes[0].images  # final overlay uses the checked photo
    finally:
        _close(view)

    shutil.copyfile(images / "1" / "cal1.jpg", images / "1" / "cal2.jpg")
    assert context.phase1_source_check().status == "mismatch"
    stale = Phase1EducationView(context)
    try:
        stale._set_stage(STAGE["final"])
        assert not stale.figure().axes[0].images
        assert "mismatch" in stale.provenance_text()
    finally:
        _close(stale)


# ---------------------------------------------------------------------------------------------
# Round-2 review additions
# ---------------------------------------------------------------------------------------------

def test_um_per_px_is_free_of_foreshortening():
    """SCI2-12: a face tilted by 60 degrees about one grid axis keeps its scale along the
    other axis; the estimate is the untilted scale, not the shortened one."""
    from pyCamSet.workflow import education_visual_p1 as vis
    grid = np.array([[x, y, 0.0] for y in range(7) for x in range(7)], dtype=float) * 1.5  # mm
    design = grid[None]
    scale = 1000.0 / 26.9                                       # px per mm (26.9 um / px)
    tilt = np.cos(np.radians(60.0))
    turn = np.radians(30.0)
    rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    pixels = (np.column_stack([grid[:, 0], grid[:, 1] * tilt]) * scale) @ rot.T + [200, 150]
    keys = np.arange(len(grid))[:, None]
    um = vis.um_per_px(pixels, keys=keys, design=design)
    assert um == pytest.approx(26.9, rel=1e-6)
    assert np.isnan(vis.um_per_px(pixels, 1.5))                  # no keys: no µm figure


def test_threshold_demo_counts_only_the_light_squares_it_loses():
    """SCI2-13: the share is over the dim half's light pixels (those the local rule keeps
    white); dark pixels are black under both rules."""
    from pyCamSet.workflow import education_visual_p1 as vis
    demo = vis.threshold_demo(_threshold_explanation())
    mean = vis._box_mean(demo.grey, demo.window)
    dim = mean < np.median(mean)
    light = dim & ~demo.local
    assert demo.dim_light_lost == pytest.approx(float(np.mean(demo.global_black[light])))
    assert demo.dim_light_lost > 0.9


@pytest.mark.gui
def test_reason_tags_are_few_and_never_touch():
    """UX2-12: at most four reason tags per close-up, none overlapping."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from pyCamSet.gui.education_phase1 import MAX_REASON_TAGS, _place_tags
    from pyCamSet.gui.education_style import stage_tokens
    _app()
    fig = Figure(figsize=(4, 4), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)
    anchors = [(50 + k, 50 + k) for k in range(8)]               # a pile of rejects
    drawn = _place_tags(ax, anchors, ["✕ not convex"] * 8, stage_tokens("Light"),
                        box=(0, 0, 100, 100), px_per_unit=3.2)
    assert 1 <= drawn <= MAX_REASON_TAGS
    fig.canvas.draw()
    boxes = [t.get_window_extent() for t in ax.texts]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not a.overlaps(b)


def test_subpixel_close_ups_show_the_largest_and_a_typical_corner():
    """ED2-16: the largest moves plus the median corner, last."""
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_phase1 import _chosen_insets, _is_typical
    shifts = np.array([2.8, 2.1, 1.9, 1.3, 0.9, 0.35, 0.3, 0.4, 0.2])
    insets = [SimpleNamespace(label=f"{v:.2f} px") for v in (2.8, 2.1, 1.9, 1.3, 0.9, 0.35)]
    step = SimpleNamespace(insets=insets, extra={"shift_px": shifts})
    for level, largest in ((1, 1), (2, 3), (3, 3)):
        chosen = _chosen_insets(step, level)
        assert len(chosen) == largest + 1
        assert _is_typical(step, chosen[-1]) and chosen[0].label == "2.80 px"


@pytest.mark.gui
def test_decision_draws_the_aim_with_its_consequence_at_l1(tmp_path, monkeypatch):
    """ED2-17: the 90 % aim is a line at L1 and the caption says what being below it means."""
    for cam in ("cam0", "cam1"):
        _photo(tmp_path / cam / "a.png")
    view = _view(monkeypatch, tmp_path, rows=[[0, 0, 1, 50.5, 50.2], [1, 0, 1, 40, 40]],
                 max_ims=1, names={"cam0": ("a.png",), "cam1": ("a.png",)},
                 diagnostics={"D1.2_detection_rate": {"cam0": 0.7, "cam1": 0.95},
                              "D1.1_total_detections": {"cam0": 1200, "cam1": 1500}})
    try:
        view.set_level(1)
        view._set_stage(STAGE["decision"])
        ax = view.figure().axes[0]
        assert any(abs(line.get_ydata()[0] - 90) < 1e-9 for line in ax.lines)
        text = view.caption.text()
        assert "below the 90 % aim" in text and "1,200" in text
        assert view.caption_issues == []
    finally:
        _close(view)


@pytest.mark.gui
def test_camera_raster_has_no_outline_cells(tmp_path, monkeypatch):
    """UX2-1: every camera's raster cells are filled in its theme fill, never outlined."""
    from matplotlib.collections import PatchCollection
    for cam in ("cam0", "cam1"):
        _photo(tmp_path / cam / "a.png")
    view = _view(monkeypatch, tmp_path, rows=[[0, 0, 1, 50.5, 50.2], [1, 0, 1, 40, 40]],
                 max_ims=1, names={"cam0": ("a.png",), "cam1": ("a.png",)})
    try:
        view.set_level(3)
        view._set_stage(STAGE["decision"])
        raster = [ax for ax in view.figure().axes if ax.images]
        assert raster and not any(isinstance(c, PatchCollection) for c in raster[0].collections)
    finally:
        _close(view)
