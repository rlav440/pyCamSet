"""
Unit tests for pyCamSet.gui.shared_functions.

These tests exercise all logic that does not require a running Qt display
server (i.e. WorkspaceManager, build_predecessor_chain, extract_detection_and_cam_res,
resolve_phase1_pickle_artifact, suppress_matplotlib_gui, and related helpers).

Run with::

    pytest tests/test_gui_shared.py -v
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Qt stub — lets us import shared_functions without a display server
# ---------------------------------------------------------------------------

def _make_qt_stub():
    """Return a minimal stub for PySide6 so shared_functions can be imported."""
    pyside6_stub = types.ModuleType("PySide6")

    for submod in ("QtCore", "QtGui", "QtWidgets"):
        mod = types.ModuleType(f"PySide6.{submod}")
        sys.modules[f"PySide6.{submod}"] = mod
        setattr(pyside6_stub, submod, mod)

    # Create stub classes that shared_functions imports at module level
    class _Stub:
        def __init__(self, *a, **kw):
            pass

    class Signal(_Stub):
        def connect(self, *a):
            pass
        def emit(self, *a):
            pass
        def __call__(self, *a):
            return self

    class _Qt:
        class Orientation:
            Horizontal = 1
            Vertical = 2
        class AlignmentFlag:
            AlignTop = 0
            AlignCenter = 0
        class ShortcutContext:
            WidgetWithChildrenShortcut = 0
        Key_Left = 0
        Key_Right = 0

    names_core    = ["QThread", "Signal", "QTextCursor", "Qt"]
    names_gui     = ["QTextCursor", "QKeySequence", "QShortcut"]
    names_widgets = [
        "QCheckBox", "QFileDialog", "QFrame", "QLabel", "QListWidget", "QListWidgetItem",
        "QPushButton", "QTextEdit", "QVBoxLayout", "QWidget", "QFormLayout",
        "QHBoxLayout", "QScrollArea", "QSplitter", "QDialog", "QMessageBox",
    ]

    for n in names_core:
        if n == "Qt":
            setattr(sys.modules["PySide6.QtCore"], n, _Qt)
        elif n == "Signal":
            setattr(sys.modules["PySide6.QtCore"], n, Signal)
        else:
            setattr(sys.modules["PySide6.QtCore"], n, _Stub)
    for n in names_widgets:
        setattr(sys.modules["PySide6.QtWidgets"], n, _Stub)
    for n in names_gui:
        setattr(sys.modules["PySide6.QtGui"], n, _Stub)

    sys.modules["PySide6"] = pyside6_stub
    return pyside6_stub


_make_qt_stub()

# Now we can import the module under test
from pyCamSet.gui.shared_functions import (  # noqa: E402
    CHARUCO_DETECTION_OPTION_METADATA,
    WorkspaceManager,
    build_charuco_option_tooltip,
    collect_charuco_detection_options,
    extract_detection_and_cam_res,
    resolve_phase1_pickle_artifact,
    suppress_matplotlib_gui,
)


# ===========================================================================
# WorkspaceManager tests
# ===========================================================================

class TestCanonicalPhaseName:
    def test_known_phases_returned_unchanged(self):
        for phase in ("phase0", "phase1", "phase2", "phase3", "phase4"):
            assert WorkspaceManager.canonical_phase_name(phase) == phase

    def test_normalises_case_and_whitespace(self):
        assert WorkspaceManager.canonical_phase_name("  Phase1  ") == "phase1"
        assert WorkspaceManager.canonical_phase_name("PHASE3") == "phase3"

    def test_legacy_phase5_raises(self):
        with pytest.raises(ValueError, match="phase5"):
            WorkspaceManager.canonical_phase_name("phase5")

    def test_legacy_phase_5_raises(self):
        with pytest.raises(ValueError, match="Legacy"):
            WorkspaceManager.canonical_phase_name("phase_5")

    def test_legacy_visualise_target_raises(self):
        with pytest.raises(ValueError, match="Legacy"):
            WorkspaceManager.canonical_phase_name("visualise_target")

    def test_legacy_assess_calibration_raises(self):
        with pytest.raises(ValueError, match="Legacy"):
            WorkspaceManager.canonical_phase_name("assess_calibration")


class TestEnsureDirs:
    def test_creates_phase0_through_phase4_only(self, tmp_path: Path):
        ws = WorkspaceManager(tmp_path / "ws")
        created = {p.name for p in (tmp_path / "ws").iterdir()}
        assert created == {
            "phase0_runs", "phase1_runs", "phase2_runs",
            "phase3_runs", "phase4_runs",
        }
        # Legacy folders must NOT be created
        assert "phase5_runs" not in created
        assert "visualise_target_runs" not in created
        assert "assess_calibration_runs" not in created


class TestSaveAndLoadRuns:
    def test_round_trip(self, tmp_path: Path):
        ws = WorkspaceManager(tmp_path / "ws")
        meta = {"run_id": "run1", "phase": "phase1", "params": {}, "diagnostics": {}}
        saved_path = ws.save_run("phase1", "run1", meta)
        assert saved_path.exists()
        loaded = ws.load_runs("phase1")
        assert len(loaded) == 1
        assert loaded[0]["run_id"] == "run1"

    def test_load_returns_empty_for_nonexistent_phase(self, tmp_path: Path):
        ws = WorkspaceManager(tmp_path / "ws")
        assert ws.load_runs("phase1") == []

    def test_load_rejects_legacy_phase(self, tmp_path: Path):
        ws = WorkspaceManager(tmp_path / "ws")
        with pytest.raises(ValueError, match="Legacy"):
            ws.load_runs("phase5")


class TestBuildPredecessorChain:
    def _make_ws(self, tmp_path: Path) -> WorkspaceManager:
        return WorkspaceManager(tmp_path / "ws")

    def _save(self, ws: WorkspaceManager, phase: str, run_id: str, inputs: dict) -> dict:
        meta = {"run_id": run_id, "phase": phase, "params": {}, "diagnostics": {}, "inputs": inputs}
        ws.save_run(phase, run_id, meta)
        return meta

    def test_empty_chain_for_phase1_run_with_no_inputs(self, tmp_path: Path):
        ws = self._make_ws(tmp_path)
        run = self._save(ws, "phase1", "r1", {})
        assert ws.build_predecessor_chain(run) == []

    def test_single_predecessor(self, tmp_path: Path):
        ws = self._make_ws(tmp_path)
        p1 = self._save(ws, "phase1", "p1_run", {})
        p2 = self._save(ws, "phase2", "p2_run", {"phase1_run_id": "p1_run"})
        chain = ws.build_predecessor_chain(p2)
        assert len(chain) == 1
        assert chain[0]["run_id"] == "p1_run"
        assert chain[0]["phase"] == "phase1"

    def test_two_hop_chain(self, tmp_path: Path):
        ws = self._make_ws(tmp_path)
        self._save(ws, "phase1", "r1", {})
        self._save(ws, "phase2", "r2", {"phase1_run_id": "r1"})
        r3 = self._save(ws, "phase3", "r3", {"phase2_run_id": "r2", "phase1_run_id": "r1"})
        chain = ws.build_predecessor_chain(r3)
        # Phase 3 → Phase 2 → Phase 1; chain is oldest-first
        assert [r["run_id"] for r in chain] == ["r1", "r2"]

    def test_full_four_hop_chain(self, tmp_path: Path):
        ws = self._make_ws(tmp_path)
        self._save(ws, "phase1", "r1", {})
        self._save(ws, "phase2", "r2", {"phase1_run_id": "r1"})
        self._save(ws, "phase3", "r3", {"phase2_run_id": "r2", "phase1_run_id": "r1"})
        r4 = self._save(ws, "phase4", "r4", {"phase3_run_id": "r3"})
        chain = ws.build_predecessor_chain(r4)
        # chain should be r3 → r2 → r1, reversed = [r1, r2, r3]
        assert [r["run_id"] for r in chain] == ["r1", "r2", "r3"]

    def test_chain_returns_deep_copies(self, tmp_path: Path):
        ws = self._make_ws(tmp_path)
        self._save(ws, "phase1", "r1", {})
        r2 = self._save(ws, "phase2", "r2", {"phase1_run_id": "r1"})
        chain = ws.build_predecessor_chain(r2)
        # Modifying the copy must not affect what load_runs returns
        chain[0]["phase"] = "MUTATED"
        reloaded = ws.load_runs("phase1")
        assert reloaded[0]["phase"] == "phase1"

    def test_missing_parent_terminates_gracefully(self, tmp_path: Path):
        ws = self._make_ws(tmp_path)
        # phase2 run references a phase1 run that was never saved
        r2 = {"run_id": "r2", "phase": "phase2", "inputs": {"phase1_run_id": "nonexistent"}}
        chain = ws.build_predecessor_chain(r2)
        assert chain == []

    def test_cycle_guard(self, tmp_path: Path):
        """Artificially self-referential inputs must not cause infinite loops."""
        ws = self._make_ws(tmp_path)
        # phase2 run pointing at itself
        r = self._save(ws, "phase2", "r2", {"phase1_run_id": "r2"})
        # No phase1 run saved — terminates immediately
        chain = ws.build_predecessor_chain(r)
        assert len(chain) <= 1


# ===========================================================================
# extract_detection_and_cam_res tests
# ===========================================================================

class TestExtractDetectionAndCamRes:
    def _make_detection(self):
        """Return a minimal object that looks like a TargetDetection."""
        det = MagicMock()
        det.get_cam_list = MagicMock(return_value=[])
        return det

    def test_bare_target_detection(self):
        det = self._make_detection()
        result_det, result_cam = extract_detection_and_cam_res(det)
        assert result_det is det
        assert result_cam is None

    def test_tuple_format(self):
        det = self._make_detection()
        cam_res = {"cam0": (1920, 1080)}
        result_det, result_cam = extract_detection_and_cam_res((det, cam_res))
        assert result_det is det
        assert result_cam is cam_res

    def test_list_format(self):
        det = self._make_detection()
        result_det, result_cam = extract_detection_and_cam_res([det])
        assert result_det is det
        assert result_cam is None

    def test_legacy_dict_raises_valueerror(self):
        det = self._make_detection()
        legacy_payload = {"detections": det, "cam_res": None}
        with pytest.raises(ValueError, match="Legacy dict-format"):
            extract_detection_and_cam_res(legacy_payload)

    def test_unknown_type_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unrecognised detection payload"):
            extract_detection_and_cam_res("not_a_detection")

    def test_tuple_without_get_cam_list_raises(self):
        with pytest.raises(ValueError):
            extract_detection_and_cam_res((42, None))


# ===========================================================================
# ChArUco detection options metadata/parsing tests
# ===========================================================================

class TestCharucoDetectionOptions:
    def test_tooltip_contains_required_fields(self):
        assert CHARUCO_DETECTION_OPTION_METADATA
        for meta in CHARUCO_DETECTION_OPTION_METADATA:
            tip = build_charuco_option_tooltip(meta)
            assert "Concept:" in tip
            assert "Default:" in tip
            assert "Range:" in tip
            assert "Range source:" in tip
            assert "Suggested value(s):" in tip

    def test_collect_defaults_and_groups(self):
        opts = collect_charuco_detection_options({})
        assert "DetectorParameters" in opts
        assert "CharucoParameters" in opts
        assert "RefineParameters" in opts
        assert opts["DetectorParameters"]["minMarkerPerimeterRate"] == 0.03
        assert opts["CharucoParameters"]["minMarkers"] == 2
        assert opts["RefineParameters"]["minRepDistance"] == 10.0
        assert "cameraMatrix" not in opts["CharucoParameters"]
        assert "distCoeffs" not in opts["CharucoParameters"]

    def test_collect_rejects_bad_matrix_shape(self):
        with pytest.raises(ValueError, match="3x3"):
            collect_charuco_detection_options({"CharucoParameters.cameraMatrix": "[1,2,3]"})

    def test_collect_rejects_adaptive_thresh_window_order(self):
        with pytest.raises(ValueError, match="adaptiveThreshWinSizeMax"):
            collect_charuco_detection_options(
                {
                    "DetectorParameters.adaptiveThreshWinSizeMin": "31",
                    "DetectorParameters.adaptiveThreshWinSizeMax": "3",
                }
            )

    def test_collect_accepts_enum_and_optional_json_inputs(self):
        opts = collect_charuco_detection_options(
            {
                "DetectorParameters.cornerRefinementMethod": "CORNER_REFINE_SUBPIX",
                "CharucoParameters.cameraMatrix": "[[1000,0,500],[0,1000,400],[0,0,1]]",
                "CharucoParameters.distCoeffs": "[0.1,-0.2,0.0,0.0,0.0]",
            }
        )
        assert opts["DetectorParameters"]["cornerRefinementMethod"] == "CORNER_REFINE_SUBPIX"
        assert opts["CharucoParameters"]["cameraMatrix"][0][0] == 1000
        assert len(opts["CharucoParameters"]["distCoeffs"]) == 5


# ===========================================================================
# resolve_phase1_pickle_artifact tests
# ===========================================================================

class TestResolvePhase1PickleArtifact:
    def test_returns_artifact_path_when_exists(self, tmp_path: Path):
        pkl = tmp_path / "detected_datapoints.pickle"
        pkl.write_bytes(b"")
        run = {"run_id": "r1", "artifacts": {"detected_datapoints_pickle": str(pkl)}}
        result = resolve_phase1_pickle_artifact(run, tmp_path)
        assert result == pkl

    def test_returns_workspace_path_when_artifact_missing(self, tmp_path: Path):
        run_dir = tmp_path / "phase1_runs" / "r1"
        run_dir.mkdir(parents=True)
        pkl = run_dir / "detected_datapoints.pickle"
        pkl.write_bytes(b"")
        run = {"run_id": "r1", "artifacts": {}}
        result = resolve_phase1_pickle_artifact(run, tmp_path)
        assert result == pkl

    def test_returns_none_when_neither_location_exists(self, tmp_path: Path):
        run = {"run_id": "r1", "artifacts": {}}
        result = resolve_phase1_pickle_artifact(run, tmp_path)
        assert result is None

    def test_does_not_fall_back_to_f_loc(self, tmp_path: Path):
        """Legacy f_loc fallback must be absent (no silent stale-data reads)."""
        # Place a pickle at f_loc — should NOT be found
        f_loc = tmp_path / "images"
        f_loc.mkdir()
        (f_loc / "detected_datapoints.pickle").write_bytes(b"")
        run = {"run_id": "r1", "artifacts": {}, "params": {"f_loc": str(f_loc)}}
        result = resolve_phase1_pickle_artifact(run, tmp_path)
        assert result is None


# ===========================================================================
# suppress_matplotlib_gui tests
# ===========================================================================

class TestSuppressMatplotlibGui:
    def test_context_manager_runs_without_matplotlib(self):
        """Should not raise even when matplotlib is unavailable."""
        with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
            with suppress_matplotlib_gui():
                pass  # No exception expected

    def test_context_manager_runs_normally(self):
        """Normal code inside the block executes and returns."""
        result = []
        with suppress_matplotlib_gui():
            result.append(1)
        assert result == [1]

    def test_exception_propagates(self):
        """Exceptions inside the block must propagate out."""
        with pytest.raises(RuntimeError, match="test error"):
            with suppress_matplotlib_gui():
                raise RuntimeError("test error")


# ===========================================================================
# resolve_run_camset_artifact tests (from assess_calibration)
# ---------------------------------------------------------------------------
# assess_calibration.py imports Qt at the module level.  Import it here after
# the stub is set up (the stub was populated at test-module import time).
# We wrap the import in a function so that if it fails for any reason the
# remaining tests are unaffected.
# ===========================================================================

def _try_import_assess():
    try:
        # Stub additional symbols that assess_calibration.py needs
        import pyCamSet.gui.assess_calibration as _ac
        return _ac
    except Exception:
        return None


_ASSESS = _try_import_assess()


@pytest.mark.skipif(_ASSESS is None, reason="assess_calibration not importable in headless env")
class TestResolveRunCamsetArtifact:
    def test_self_calibrated_camset_found(self, tmp_path: Path):
        f = tmp_path / "out.camset"
        f.write_bytes(b"")
        run = {"artifacts": {"self_calibrated_camset": str(f)}}
        assert _ASSESS.resolve_run_camset_artifact(run) == f

    def test_optimised_camset_fallback(self, tmp_path: Path):
        f = tmp_path / "opt.camset"
        f.write_bytes(b"")
        run = {"artifacts": {"optimised_camset": str(f)}}
        assert _ASSESS.resolve_run_camset_artifact(run) == f

    def test_legacy_phase5_key_not_found(self, tmp_path: Path):
        """phase5_camset key must no longer be supported."""
        f = tmp_path / "legacy.camset"
        f.write_bytes(b"")
        run = {"artifacts": {"phase5_camset": str(f)}}
        assert _ASSESS.resolve_run_camset_artifact(run) is None

    def test_returns_none_when_no_artifact(self):
        assert _ASSESS.resolve_run_camset_artifact({"artifacts": {}}) is None


@pytest.mark.skipif(_ASSESS is None, reason="assess_calibration not importable in headless env")
class TestCanonicalPhaseTag:
    def test_normalises_phase3(self):
        assert _ASSESS.canonical_phase_tag("phase3") == "phase3"
        assert _ASSESS.canonical_phase_tag("Phase3") == "phase3"
        assert _ASSESS.canonical_phase_tag(" PHASE3 ") == "phase3"

    def test_empty_returns_unknown(self):
        assert _ASSESS.canonical_phase_tag(None) == "unknown"
        assert _ASSESS.canonical_phase_tag("") == "unknown"

    def test_legacy_names_no_longer_mapped(self):
        """phase5 / visualise_target must no longer be silently aliased."""
        assert _ASSESS.canonical_phase_tag("phase5") == "phase5"
        assert _ASSESS.canonical_phase_tag("visualise_target") == "visualise_target"
        assert _ASSESS.canonical_phase_tag("assess_calibration") == "assess_calibration"
