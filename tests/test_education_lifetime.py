"""Test that closing the education dialog releases it, that theme switching never reaches
a dead widget, and that the photo and explanation caches stay bounded.
"""
from __future__ import annotations

import gc
import os
import sys
import time
import types
import weakref
from pathlib import Path

import numpy as np
import pytest


def _application():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _flush_deletions(app) -> None:
    """Run queued deleteLater() work: process events, then the deferred-delete queue, twice."""
    from PySide6.QtCore import QCoreApplication, QEvent

    for _ in range(2):
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def _install_fake_phase2(monkeypatch):
    """A phase-2 factory that returns a real stage view, so the dialog has live widgets."""
    from pyCamSet.gui.education_stage import EducationStageView

    fake = types.ModuleType("pyCamSet.gui.education_phase2")

    def make_view(context, parent=None):
        view = EducationStageView(["Stage one", "Stage two"], parent)
        view.setObjectName(f"fakeView-{context.run_id}")
        return view

    fake.create_education_phase_view = make_view
    monkeypatch.setitem(sys.modules, fake.__name__, fake)


def _run_dir(tmp_path: Path, run_id: str) -> Path:
    run_dir = tmp_path / ".pycamset_workspace" / "phase2_runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        f'{{"run_id":"{run_id}","status":"complete"}}', encoding="utf-8")
    return run_dir


@pytest.fixture
def theme_errors(monkeypatch):
    """Record anything that reaches sys.excepthook (PySide reports slot errors there)."""
    errors: list[BaseException] = []

    def hook(kind, value, tb):
        errors.append(value)

    monkeypatch.setattr(sys, "excepthook", hook)
    return errors


# -- theme watching ------------------------------------------------------------------------------

@pytest.mark.gui
def test_theme_watcher_detaches_and_switching_theme_after_close_raises_nothing(theme_errors):
    app = _application()
    from PySide6.QtWidgets import QWidget
    from pyCamSet.gui.education_stage import ThemeWatcher

    owner = QWidget()
    watcher = ThemeWatcher(owner)
    seen: list[str] = []
    watcher.themeChanged.connect(seen.append)
    app.setProperty("pycamsetTheme", "Light")
    app.setProperty("pycamsetTheme", "Dark")
    assert seen == ["Dark"]

    watcher.detach()
    watcher.detach()  # idempotent
    app.setProperty("pycamsetTheme", "Sepia")
    assert seen == ["Dark"]

    # A watcher whose owner is deleted must not be reached by a later theme switch.
    other_owner = QWidget()
    ThemeWatcher(other_owner)
    other_owner.deleteLater()
    _flush_deletions(app)
    app.setProperty("pycamsetTheme", "Light")
    app.processEvents()
    assert theme_errors == []
    app.setProperty("pycamsetTheme", "Light")
    owner.deleteLater()
    _flush_deletions(app)


@pytest.mark.gui
def test_dialog_closed_by_escape_releases_theme_subscription(tmp_path, monkeypatch, theme_errors):
    """done() (Escape, reject) never sends closeEvent: the dialog must still let go of the theme."""
    app = _application()
    from pyCamSet.gui.education_dialog import EducationDialog, EducationRunView

    _install_fake_phase2(monkeypatch)
    run = EducationRunView.from_run_dir(_run_dir(tmp_path, "run-1"))
    dialog = EducationDialog([run], [run.run_id])
    dialog.show()
    dialog.done(0)
    dialog.deleteLater()
    ref = weakref.ref(dialog)
    del dialog
    _flush_deletions(app)
    gc.collect()
    app.setProperty("pycamsetTheme", "Dark")
    app.processEvents()
    app.setProperty("pycamsetTheme", "Light")
    app.processEvents()
    assert theme_errors == []
    assert ref() is None, "the dismissed dialog is still referenced after deleteLater"


# -- dialog lifetime -----------------------------------------------------------------------------

def _count_live(cls) -> int:
    gc.collect()
    count = 0
    for obj in gc.get_objects():
        try:
            count += isinstance(obj, cls)
        except ReferenceError:  # a dead weak proxy: not a live instance
            continue
    return count


@pytest.mark.gui
def test_open_close_cycles_do_not_accumulate_dialogs_or_stage_views(tmp_path, monkeypatch):
    app = _application()
    from pyCamSet.gui import education_dialog, run_results
    from pyCamSet.gui.education_stage import EducationStageView

    _install_fake_phase2(monkeypatch)
    _run_dir(tmp_path, "run-1")
    workspace = tmp_path / ".pycamset_workspace"
    run_dir = workspace / "phase2_runs" / "run-1"
    monkeypatch.setattr(run_results, "education_mode_enabled", lambda: True)

    def open_and_dismiss(self):  # stands in for the learner closing the dialog with Escape
        self.show()
        self.done(0)
        return 0

    monkeypatch.setattr(education_dialog.EducationDialog, "exec", open_and_dismiss)
    # Views left alive by earlier tests in the same process are not this test's to count.
    start_dialogs = _count_live(education_dialog.EducationDialog)
    start_stages = _count_live(EducationStageView)
    for cycle in range(5):
        assert run_results.open_education_run({"run_id": "run-1", "_run_dir": str(run_dir)},
                                              "phase2", workspace)
        _flush_deletions(app)
        dialogs = _count_live(education_dialog.EducationDialog)
        stages = _count_live(EducationStageView)
        assert dialogs <= start_dialogs, f"cycle {cycle}: {dialogs} dialog(s) still alive"
        assert stages <= start_stages, f"cycle {cycle}: {stages} stage view(s) still alive"


@pytest.mark.gui
def test_repeated_open_close_keeps_rss_flat(tmp_path, monkeypatch):
    psutil = pytest.importorskip("psutil")
    app = _application()
    from pyCamSet.gui import education_dialog, run_results

    _install_fake_phase2(monkeypatch)
    _run_dir(tmp_path, "run-1")
    workspace = tmp_path / ".pycamset_workspace"
    run_dir = workspace / "phase2_runs" / "run-1"
    monkeypatch.setattr(run_results, "education_mode_enabled", lambda: True)

    def open_and_dismiss(self):
        self.show()
        self.done(0)
        return 0

    monkeypatch.setattr(education_dialog.EducationDialog, "exec", open_and_dismiss)
    process = psutil.Process(os.getpid())

    def cycle():
        run_results.open_education_run({"run_id": "run-1", "_run_dir": str(run_dir)},
                                       "phase2", workspace)
        _flush_deletions(app)
        gc.collect()

    cycle()  # warm-up: imports, style sheets and first allocations are not a leak
    before = process.memory_info().rss
    for _ in range(5):
        cycle()
    growth_mb = (process.memory_info().rss - before) / 1e6
    # Five re-opens of a small stage view must not keep their dialogs alive.
    assert growth_mb < 30, f"RSS grew {growth_mb:.1f} MB over 5 open-close cycles"


# -- bounded caches ------------------------------------------------------------------------------

def test_byte_lru_hit_miss_and_eviction_in_least_recently_used_order():
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_photo_cache import ByteLRU

    cache = ByteLRU(max_bytes=3 * 100 * 1024)
    for name in ("a", "b", "c"):
        cache[name] = np.zeros(100 * 1024, dtype=np.uint8)
    assert cache.get("a") is not None           # hit: "a" becomes most recent
    assert cache.get("zzz") is None             # miss counted
    assert (cache.hits, cache.misses) == (1, 1)
    cache["d"] = np.zeros(100 * 1024, dtype=np.uint8)   # over the cap: "b" is the oldest
    assert "b" not in cache and "a" in cache and "d" in cache
    assert cache.evictions == 1
    assert cache.bytes_used() <= cache.max_bytes


def test_byte_lru_cap_bounds_total_bytes_and_keeps_the_newest_entry():
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_photo_cache import ByteLRU

    cache = ByteLRU(max_bytes=1024 * 1024)
    for index in range(50):
        cache[index] = np.zeros(300 * 1024, dtype=np.uint8)
        assert cache.bytes_used() <= cache.max_bytes or len(cache) == 1
    assert len(cache) <= 3
    assert cache.evictions >= 47
    # An entry bigger than the whole cap is still kept, otherwise it would be recomputed forever.
    cache["huge"] = np.zeros(4 * 1024 * 1024, dtype=np.uint8)
    assert "huge" in cache and len(cache) == 1


def test_cache_limit_comes_from_environment_and_is_clamped(monkeypatch):
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_photo_cache import (
        DEFAULT_CACHE_MB, MAX_CACHE_MB, MIN_CACHE_MB, cache_limit_bytes)

    monkeypatch.delenv("PYCAMSET_EDU_CACHE_MB", raising=False)
    assert cache_limit_bytes() == DEFAULT_CACHE_MB * 1024 * 1024
    monkeypatch.setenv("PYCAMSET_EDU_CACHE_MB", "96")
    assert cache_limit_bytes() == 96 * 1024 * 1024
    monkeypatch.setenv("PYCAMSET_EDU_CACHE_MB", "not a number")
    assert cache_limit_bytes() == DEFAULT_CACHE_MB * 1024 * 1024
    monkeypatch.setenv("PYCAMSET_EDU_CACHE_MB", "1")
    assert cache_limit_bytes() == MIN_CACHE_MB * 1024 * 1024
    monkeypatch.setenv("PYCAMSET_EDU_CACHE_MB", "999999")
    assert cache_limit_bytes() == MAX_CACHE_MB * 1024 * 1024


def test_estimate_counts_arrays_inside_dataclasses_and_containers():
    pytest.importorskip("PySide6")
    from dataclasses import dataclass

    from pyCamSet.gui.education_photo_cache import estimate_nbytes

    @dataclass
    class Step:
        image: np.ndarray
        label: str

    step = Step(np.zeros((10, 10, 3), dtype=np.uint8), "x")
    assert estimate_nbytes(step) == 300
    # Two different steps add up; the same object counted twice (aliasing) is counted once.
    assert estimate_nbytes({"steps": [step, Step(np.zeros((10, 10, 3), dtype=np.uint8), "y")]}) == 600
    assert estimate_nbytes([step, step]) == 300
    assert estimate_nbytes(RuntimeError("no photo")) == 256


def _write_photo(path: Path, width: int, height: int) -> Path:
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def test_display_read_of_a_large_photo_is_reduced_and_capped(tmp_path):
    from pyCamSet.workflow import education_visual_p1 as vis

    photo = _write_photo(tmp_path / "large.jpg", 2000, 1500)
    assert vis._reduction_for((2000, 1500), 720) == 2
    assert vis._reduction_for((800, 600), 720) == 1
    assert vis._reduction_for(None, 720) == 1
    image = vis.read_display_photo(photo, max_side=720)
    assert image is not None and image.ndim == 3 and image.shape[2] == 3
    assert max(image.shape[:2]) == 720
    assert image.dtype == np.uint8 and image.flags["C_CONTIGUOUS"]


def test_load_photo_honours_an_explicit_max_side_and_keeps_full_size_by_default(tmp_path):
    from pyCamSet.workflow import education_visual_p2 as vp

    photo = _write_photo(tmp_path / "wide.png", 1600, 1200)
    full = vp.load_photo(photo)
    assert full.shape == (1200, 1600)                      # Phase 2/3 geometry is unchanged
    small = vp.load_photo(photo, max_side=400)
    assert max(small.shape) <= 400 and small.dtype == np.uint8
    assert vp.load_photo(None) is None
    assert vp.load_photo(tmp_path / "missing.png") is None


def test_prefetcher_decodes_once_delivers_on_the_gui_thread_and_stops_on_shutdown(tmp_path):
    app = _application()
    from PySide6.QtCore import QObject, Qt, Slot
    from pyCamSet.gui.education_photo_cache import PhotoPrefetcher

    calls: list[str] = []

    def decode(argument: str) -> str:
        calls.append(argument)
        return f"decoded:{argument}"

    received: list[tuple] = []

    class Collector(QObject):
        @Slot(object, object)
        def on_ready(self, key, value):
            received.append((key, value))

    owner = Collector()
    prefetch = PhotoPrefetcher(decode, owner, max_workers=2)
    prefetch.ready.connect(owner.on_ready, Qt.ConnectionType.QueuedConnection)
    assert prefetch.request(("cam", 1), "one")
    assert not prefetch.request(("cam", 1), "one")         # already pending: no second decode
    assert prefetch.take(("cam", 2)) is None                # never requested
    value = prefetch.take(("cam", 1))                       # waits for the running decode
    assert value == "decoded:one"
    assert calls == ["one"]
    assert prefetch.take(("cam", 1)) is None                # taken once only

    assert prefetch.request(("cam", 3), "three")
    deadline = time.monotonic() + 5
    while not any(key == ("cam", 3) for key, _ in received) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    # The take() above also emitted a signal for cam 1; the view ignores it when already cached.
    assert (("cam", 3), "decoded:three") in received

    prefetch.shutdown()
    assert not prefetch.request(("cam", 4), "four")
    assert "four" not in calls


def test_prefetch_delivers_a_decode_error_as_a_value_not_an_exception():
    app = _application()
    from PySide6.QtCore import QObject, Qt, Slot
    from pyCamSet.gui.education_photo_cache import PhotoPrefetcher

    def decode(_argument):
        raise OSError("unreadable photo")

    received: list[object] = []

    class Collector(QObject):
        @Slot(object, object)
        def on_ready(self, key, value):
            received.append(value)

    owner = Collector()
    prefetch = PhotoPrefetcher(decode, owner)
    prefetch.ready.connect(owner.on_ready, Qt.ConnectionType.QueuedConnection)
    prefetch.request("k", "p")
    deadline = time.monotonic() + 5
    while not received and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert len(received) == 1 and isinstance(received[0], OSError)
    prefetch.shutdown()


def test_folder_tif_search_stops_at_the_first_tif_and_skips_flat_folders(tmp_path):
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_phase1 import _has_tif_below

    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "a.jpg").write_bytes(b"x")
    assert _has_tif_below(flat) is False
    nested = flat / "sub" / "deeper"
    nested.mkdir(parents=True)
    (nested / "scan.TIF").write_bytes(b"x")
    assert _has_tif_below(flat) is True
