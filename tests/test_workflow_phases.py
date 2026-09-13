"""The phases as callables, with no interface attached.

What a phase does used to be a closure inside a Qt tab method: reachable only
by clicking Run, and testable only by driving a widget.  These tests are the
reason for moving it out -- they call the phases directly, assert on the run
record they save, and import no GUI toolkit at all.

The parameter rules get the same treatment.  Every value a phase takes used
to be validated inline next to the widget it came from, in five nearly
identical copies; the rules now have one home, so they can be stated once
here.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from pyCamSet.workflow import phase1, phase2, phase3
from pyCamSet.workflow.detections import DetectionFilter
from pyCamSet.workflow.params import (
    ParamError,
    as_int,
    as_json_object,
    as_optional_positive_int,
    as_outlier_mode,
    as_positive_float,
    as_positive_int,
    require_image_folder,
    require_target_match,
)
from pyCamSet.workflow.targets import CHARUCO_DETECTION_OPTION_METADATA
from pyCamSet.workflow.workspace import (
    WorkspaceManager,
    make_run_id,
    resolve_artifact,
    workspace_path_for,
)

CHARUCO_PARAMS = dict(
    target_type="ChArUco",
    n_points=20,
    length=4.0,
    marker_fraction=0.8,
    marker_backend="aruco1",
)


# ---------------------------------------------------------------------------
# The workflow is reachable without a GUI toolkit
# ---------------------------------------------------------------------------


# Run in a process of its own, with PySide6 made unimportable.  Asserting on
# this process's sys.modules would prove nothing: the suite imports the GUI
# elsewhere, so Qt is already loaded by the time this runs.
_NO_QT_PROBE = """
import sys

class RefuseQt:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("PySide6", "PySide2", "PyQt5", "PyQt6",
                                  "shiboken6"):
            raise ImportError(f"no GUI toolkit on this install: {name}")
        return None

sys.meta_path.insert(0, RefuseQt())

from pyCamSet.workflow import phase1, phase2, phase3, phase4  # noqa: F401
from pyCamSet.workflow import detections, diagnostics, logs, params  # noqa: F401
from pyCamSet.workflow import lockbox_geometry, targets, workspace  # noqa: F401

assert all(p.BACKEND_OK for p in (phase1, phase2, phase3, phase4)), (
    "a phase lost its backend when the GUI toolkit was taken away")
print("ok")
"""


def test_no_phase_reaches_for_qt(repo_root):
    """The point of the package: a phase runs where there is no display.

    The lean install documented in the README leaves PySide6 out, so a phase
    that imports it -- directly, or through a module it borrows a helper from
    -- is unusable there and would take the headless path down with it.
    """
    finished = subprocess.run(
        [sys.executable, "-c", _NO_QT_PROBE],
        capture_output=True, text=True, cwd=repo_root,
    )

    assert finished.returncode == 0, (
        "pyCamSet.workflow does not import on an install without a GUI "
        f"toolkit:\n{finished.stderr}"
    )
    assert "ok" in finished.stdout


# ---------------------------------------------------------------------------
# The parameter rules
# ---------------------------------------------------------------------------


def test_a_blank_image_folder_is_refused():
    with pytest.raises(ParamError, match="Image folder is required"):
        require_image_folder("   ")


@pytest.mark.parametrize("text", ["0", "-3"])
def test_a_non_positive_count_is_refused(text):
    with pytest.raises(ParamError, match="positive integer"):
        as_positive_int(text, "n_lim")


def test_a_blank_optional_count_is_none_rather_than_an_error():
    """Blank means "no limit", and only a blank one may skip the rule."""
    assert as_optional_positive_int("", "n_lim") is None
    assert as_optional_positive_int("  ", "n_lim") is None
    assert as_optional_positive_int("12", "n_lim") == 12
    with pytest.raises(ParamError):
        as_optional_positive_int("nope", "n_lim")


def test_a_field_names_itself_in_its_own_error():
    """The message is read next to a form, so it has to say which field."""
    with pytest.raises(ParamError, match="PuzzleBoard paper_width"):
        as_positive_float("wide", "PuzzleBoard paper_width")
    with pytest.raises(ParamError, match="ref_cam"):
        as_int("first", "ref_cam")


@pytest.mark.parametrize("text", ["0", "-1", "inf", "nan"])
def test_a_length_must_be_finite_and_positive(text):
    """A target with no size indexes nothing; infinity is not a size either."""
    with pytest.raises(ParamError):
        as_positive_float(text, "Length")


def test_json_params_must_be_an_object():
    assert as_json_object("", "Fixed params JSON") is None
    assert as_json_object('{"cam0": "ext"}', "Fixed params JSON") == {"cam0": "ext"}
    with pytest.raises(ParamError, match="must be a JSON object"):
        as_json_object("[1, 2]", "Fixed params JSON")
    with pytest.raises(ParamError, match="Fixed params JSON"):
        as_json_object("{not json", "Fixed params JSON")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("y", "y"), ("yes", "y"), ("Enabled", "y"), ("1", "y"),
     ("ask", "y"),
     ("n", "n"), ("no", "n"), ("off", "n"), ("", "n"), (None, "n"),
     ("perhaps", "n")],
)
def test_the_outlier_mode_falls_back_to_off(text, expected):
    """Rejecting observations is the surprising outcome, so it is never the
    reading of a value nobody recognised."""
    assert as_outlier_mode(text) == expected


def test_a_target_that_cannot_read_its_detections_is_refused():
    """The check that used to be three copies of itself, one per phase.

    A detection stores its keys as indices into the target's points, so a
    target of a different size addresses points that do not exist.
    """
    run = {"run_id": "r1", "params": dict(CHARUCO_PARAMS)}

    require_target_match(run, dict(CHARUCO_PARAMS))
    require_target_match(None, {"target_type": "Ccube", "n_points": 6})

    with pytest.raises(ParamError) as caught:
        require_target_match(run, {**CHARUCO_PARAMS, "n_points": 6})

    message = str(caught.value)
    assert "r1" in message
    assert "n_points" in message
    assert "20" in message and "6" in message


# ---------------------------------------------------------------------------
# The workspace
# ---------------------------------------------------------------------------


def test_a_run_round_trips_through_the_workspace(tmp_path):
    workspace = WorkspaceManager(workspace_path_for(tmp_path))
    run_id = make_run_id()

    workspace.save_run("phase1", run_id, {"run_id": run_id, "phase": "phase1",
                                          "params": {"f_loc": str(tmp_path)}})

    runs = workspace.load_runs("phase1")
    assert [run["run_id"] for run in runs] == [run_id]
    assert runs[0]["params"]["f_loc"] == str(tmp_path)
    assert workspace.find_run("phase1", run_id) is not None
    assert workspace.find_run("phase1", "nothing-like-it") is None


def test_runs_are_ordered_by_when_they_were_made_not_by_name(tmp_path):
    """Producers mint run ids in incompatible formats, and digits sort before
    letters, so a letter-prefixed id used to land last whatever its date."""
    workspace = WorkspaceManager(workspace_path_for(tmp_path))

    workspace.save_run("phase3", "phase3_20260101_000000",
                       {"created_at": "2026-01-01T00:00:00"})
    workspace.save_run("phase3", "20260601_120000_abc123",
                       {"created_at": "2026-06-01T12:00:00"})
    workspace.save_run("phase3", "phase3_20261231_235959",
                       {"created_at": "2026-12-31T23:59:59"})

    assert [run["run_id"] for run in workspace.load_runs("phase3")] == [
        "phase3_20260101_000000",
        "20260601_120000_abc123",
        "phase3_20261231_235959",
    ]


def test_a_run_written_without_a_timestamp_still_sorts(tmp_path):
    """A run from a script that set no ``created_at`` falls back to the
    metadata file's mtime, so no migration step is needed."""
    workspace = WorkspaceManager(workspace_path_for(tmp_path))
    directory = workspace.run_dir("phase2", "written_by_hand")
    (directory / "metadata.json").write_text(json.dumps({"phase": "phase2"}))

    runs = workspace.load_runs("phase2")
    assert [run["run_id"] for run in runs] == ["written_by_hand"]
    assert runs[0]["_recency_ts"] > 0


def test_a_phase_name_from_an_earlier_build_says_so(tmp_path):
    workspace = WorkspaceManager(workspace_path_for(tmp_path))
    with pytest.raises(ValueError, match="no longer supported"):
        workspace.load_runs("phase5")


def test_the_predecessor_chain_walks_back_through_the_phases(tmp_path):
    workspace = WorkspaceManager(workspace_path_for(tmp_path))
    workspace.save_run("phase1", "p1", {"phase": "phase1"})
    workspace.save_run("phase2", "p2",
                       {"phase": "phase2", "inputs": {"phase1_run_id": "p1"}})
    phase3 = {"phase": "phase3",
              "inputs": {"phase1_run_id": "p1", "phase2_run_id": "p2"}}
    workspace.save_run("phase3", "p3", phase3)

    chain = workspace.build_predecessor_chain(workspace.find_run("phase3", "p3"))

    # Oldest first, and through the highest phase in each inputs dict: phase 3
    # names phase 1 too, and following that instead would skip phase 2.
    assert [run["run_id"] for run in chain] == ["p1", "p2"]


# ---------------------------------------------------------------------------
# A phase, run for real
# ---------------------------------------------------------------------------


@pytest.fixture
def charuco_image_folder(session_data_dir, tmp_path):
    """The ChArUco corpus, somewhere a run may write to.

    Phase 1 caches its detections beside the images, so it needs a folder of
    its own; symlinked per camera rather than copied, because the corpus is
    20 MB and nothing here writes to the images themselves.
    """
    folder = tmp_path / "images"
    folder.mkdir()
    for camera in sorted((session_data_dir / "calibration_charuco").iterdir()):
        if camera.is_dir():
            (folder / camera.name).symlink_to(camera, target_is_directory=True)
    return folder


@pytest.mark.data
@pytest.mark.slow
def test_phase_1_detects_and_records_a_run(charuco_image_folder, tmp_path):
    """Phase 1 end to end, with nothing but a folder and a dict.

    This is what the extraction bought: the detection pass, its diagnostics,
    the saved artifact and the run record, asserted without a Qt event loop
    anywhere in the process.
    """
    image_folder = charuco_image_folder
    workspace = WorkspaceManager(workspace_path_for(tmp_path))

    lines: list[str] = []
    metadata = phase1.run(
        {
            **CHARUCO_PARAMS,
            "f_loc": str(image_folder),
            "caching": True,
            "high_distortion": False,
            "n_lim": 3,
            "threads": 1,
            "upscale_factor": 1,
            "fixed_params": None,
            "problem_options": None,
            "charuco_detection_options": None,
            "border_fraction": 0.1,
            "selected_cameras": [],
        },
        workspace,
        lines.append,
    )

    assert metadata["error"] is None, metadata["error"]
    assert metadata["phase"] == "phase1"

    diagnostics = metadata["diagnostics"]
    assert diagnostics["cam_names"]
    assert diagnostics["D1.7_min_features"] > 0
    assert set(diagnostics["D1.2_detection_rate"]) <= set(diagnostics["cam_names"])
    assert all(0.0 <= rate <= 1.0
               for rate in diagnostics["D1.2_detection_rate"].values())

    # The run is on disk, and the next phase can find its detections.
    saved = workspace.find_run("phase1", metadata["run_id"])
    assert saved is not None
    pickle_path = saved["artifacts"]["detected_datapoints_pickle"]
    assert pickle_path.endswith("detected_datapoints.pickle")

    # Every line of the phase's own output reached the callable it was given,
    # rather than a terminal widget or the process's stdout.
    assert any(line.startswith("1b") for line in lines)
    assert any("Run saved" in line for line in lines)


@pytest.mark.data
def test_a_phase_records_its_failure_rather_than_raising(tmp_path):
    """A phase that cannot run still saves a run saying why.

    The interface shows the error from the record it gets back, so a phase
    that raised instead would leave nothing on disk to look at afterwards.
    """
    empty = tmp_path / "no_cameras_here"
    empty.mkdir()
    workspace = WorkspaceManager(workspace_path_for(tmp_path))

    metadata = phase1.run(
        {
            **CHARUCO_PARAMS,
            "f_loc": str(empty),
            "caching": False,
            "high_distortion": False,
            "n_lim": None,
            "threads": 1,
            "upscale_factor": 1,
            "fixed_params": None,
            "problem_options": None,
            "charuco_detection_options": None,
            "border_fraction": 0.1,
            "selected_cameras": [],
        },
        workspace,
        lambda _line: None,
    )

    assert "No selected camera sub-folders found." in metadata["error"]
    assert workspace.find_run("phase1", metadata["run_id"])["error"] == metadata["error"]


def test_phase_2_refuses_a_workspace_it_has_not_been_given():
    with pytest.raises(RuntimeError, match="Workspace path is not set"):
        phase2.run({"f_loc": "/nowhere"}, WorkspaceManager(), lambda _line: None)


# ---------------------------------------------------------------------------
# Re-running a phase without some of its observations
# ---------------------------------------------------------------------------


class _FakeDetections:
    """Records which filter it was asked for, the way a detection would be."""

    def __init__(self):
        self.deleted = None

    def delete_row(self, **kwargs):
        self.deleted = kwargs
        return self


def test_a_filter_drops_images_across_every_camera():
    prune = DetectionFilter(images=[3, 7, 11])
    detections = _FakeDetections()

    prune.apply(detections)

    assert detections.deleted == {"global_im_num": [3, 7, 11]}
    assert prune.count == 3


def test_a_filter_can_drop_an_image_from_one_camera_only():
    """Phase 2's threshold is per camera: one camera's bad view of an image
    should not cost every other camera that image."""
    prune = DetectionFilter(camera_images={"cam0": [1, 2], "cam1": [5]})
    detections = _FakeDetections()

    prune.apply(detections)

    assert detections.deleted == {"cam_im_num": {"cam0": [1, 2], "cam1": [5]}}
    assert prune.count == 3


def test_a_filter_says_what_it_did_while_it_does_it():
    lines: list[str] = []
    DetectionFilter(images=[4]).apply(_FakeDetections(), lines.append)

    assert any("global_im_num" in line and "1 image(s)" in line for line in lines)


@pytest.mark.parametrize("phase", [phase2, phase3])
def test_a_rerun_repeats_the_source_run_s_settings(phase, tmp_path, monkeypatch):
    """A re-run differs from its source only by what was pruned.

    Both re-runs used to be their own copy of the phase; this asserts the
    thing that replaced them -- that ``rerun`` hands the runner the source's
    own parameters and its own inputs, and links the new run back to it.
    """
    workspace = WorkspaceManager(workspace_path_for(tmp_path))
    workspace.save_run("phase1", "p1", {"phase": "phase1"})
    workspace.save_run("phase2", "p2",
                       {"phase": "phase2", "inputs": {"phase1_run_id": "p1"}})
    source = {
        "run_id": "src",
        "phase": phase.__name__.rsplit(".", 1)[-1],
        "params": {"f_loc": str(tmp_path), "n_points": 11, "length": 7.5},
        "inputs": {"phase1_run_id": "p1", "phase2_run_id": "p2"},
    }

    seen = {}

    def fake_run(params, ws, log=None, **kwargs):
        seen.update(params=params, kwargs=kwargs)
        return {"run_id": "new"}

    monkeypatch.setattr(phase, "run", fake_run)
    prune = DetectionFilter(images=[2], record={"n_removed": 1})

    phase.rerun(source, workspace, lambda _line: None, prune)

    assert seen["params"] == source["params"]
    assert seen["kwargs"]["prune"] is prune
    assert seen["kwargs"]["source_run"] is source
    assert seen["kwargs"]["phase1_run"]["run_id"] == "p1"
    if phase is phase3:
        assert seen["kwargs"]["phase2_run"]["run_id"] == "p2"


def test_a_rerun_falls_back_to_the_latest_input_run(tmp_path):
    """An old run whose linked input is gone should still find inputs."""
    workspace = WorkspaceManager(workspace_path_for(tmp_path))
    workspace.save_run("phase1", "older", {"created_at": "2026-01-01T00:00:00"})
    workspace.save_run("phase1", "newer", {"created_at": "2026-06-01T00:00:00"})

    linked = workspace.linked_run("phase1", {"inputs": {"phase1_run_id": "gone"}})
    assert linked["run_id"] == "newer"

    exact = workspace.linked_run("phase1", {"inputs": {"phase1_run_id": "older"}})
    assert exact["run_id"] == "older"

    assert workspace.linked_run("phase4", {"inputs": {}}) is None


# ---------------------------------------------------------------------------
# Resolving a run's output
# ---------------------------------------------------------------------------


def test_an_artifact_resolves_from_the_run_s_own_record(tmp_path):
    workspace = WorkspaceManager(workspace_path_for(tmp_path))
    ws_path = workspace.workspace_path
    directory = workspace.run_dir("phase3", "r1")
    recorded = directory / "somewhere_else.camset"
    recorded.write_text("")

    run = {"run_id": "r1", "artifacts": {"optimised_camset": str(recorded)}}
    assert resolve_artifact(run, "phase3", ws_path) == recorded


def test_an_artifact_resolves_by_convention_when_the_record_is_stale(tmp_path):
    """A run whose recorded path moved with the folder still resolves, which
    is why the conventional filename is tried after the recorded one."""
    workspace = WorkspaceManager(workspace_path_for(tmp_path))
    ws_path = workspace.workspace_path
    directory = workspace.run_dir("phase2", "r2")
    (directory / "initial_cameras.camset").write_text("")

    run = {"run_id": "r2", "artifacts": {"initial_camset": "/gone/away.camset"}}
    assert resolve_artifact(run, "phase2", ws_path).name == "initial_cameras.camset"

    assert resolve_artifact({"run_id": "nothing"}, "phase2", ws_path) is None


def test_a_high_distortion_camset_wins_over_the_plain_one(tmp_path):
    """Phase 2 writes one name or the other, never both -- but if both are
    there the refined one is the run's output."""
    workspace = WorkspaceManager(workspace_path_for(tmp_path))
    directory = workspace.run_dir("phase2", "r3")
    (directory / "initial_cameras.camset").write_text("")
    (directory / "initial_cameras_high_distortion.camset").write_text("")

    resolved = resolve_artifact({"run_id": "r3"}, "phase2", workspace.workspace_path)
    assert resolved.name == "initial_cameras_high_distortion.camset"


# ---------------------------------------------------------------------------
# The detector option table
# ---------------------------------------------------------------------------


def test_the_option_table_loads_from_the_package(repo_root):
    """It is JSON beside the module now, so it has to be in the wheel."""
    assert len(CHARUCO_DETECTION_OPTION_METADATA) == 17
    for option in CHARUCO_DETECTION_OPTION_METADATA:
        assert set(option) >= {"key", "label", "default", "parser_type",
                               "value_type", "concept"}
        assert "." in option["key"], option["key"]
