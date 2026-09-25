"""
Purpose: Tests for the Detection Cost measurement -- the engine and the target
         form that adapts to whichever target is selected.
Status:  Working
Future:  Add a live-folder (watcher) case when that mode exists.

The two things worth locking down here are the ones that were previously wrong
in the control software's copy of this tool: a target form that offered every
target's fields at once, and a measurement that could run against a target the
acquisition never declared.
"""

import json
import pathlib

import numpy as np
import pytest

from pyCamSet.calibration_targets.markers.backend_registry import marker_backend_available
from pyCamSet.workflow import detection_cost as dc


# --------------------------------------------------------------------------- #
# The target describes itself                                                #
# --------------------------------------------------------------------------- #

class TestTargetSchemaDrivesTheForm:
    """A target is asked only for the settings it declares."""

    def test_ccube_has_no_square_count(self):
        """The defect this replaces: a cube offered ChArUco board fields."""
        cube_keys = dc.construction_keys("Ccube2")
        assert "num_squares_x" not in cube_keys
        assert "num_squares_y" not in cube_keys
        assert "n_points" in cube_keys

    def test_charuco_has_no_cube_n_points(self):
        board_keys = dc.construction_keys("ChArUco2")
        assert "n_points" not in board_keys
        assert "num_squares_x" in board_keys

    def test_every_target_declares_something(self):
        for name in dc.target_names():
            assert dc.construction_keys(name), f"{name} declares no settings"

    def test_defaults_come_from_the_target(self):
        defaults = dc.default_construction_values("Ccube2")
        assert set(defaults) == set(dc.construction_keys("Ccube2"))

    def test_unknown_target_is_refused(self):
        with pytest.raises(ValueError):
            dc.construction_keys("NoSuchBoard")


class TestTargetSpecRefusesStraySettings:
    """A value for a setting the target lacks is an error, not dropped.

    Passing num_squares_x to a cube means the caller believes it is measuring a
    board; measuring a cube instead would produce a confident number for the
    wrong geometry.
    """

    def test_ccube_refuses_a_square_count(self):
        with pytest.raises(ValueError) as excinfo:
            dc.target_spec_from_values("Ccube2", {"num_squares_x": 5})
        assert "num_squares_x" in str(excinfo.value)

    def test_charuco_refuses_n_points(self):
        with pytest.raises(ValueError):
            dc.target_spec_from_values("ChArUco2", {"n_points": 6})

    def test_declared_values_build_the_spec(self):
        spec = dc.target_spec_from_values(
            "Ccube2", {"length": 10.0, "n_points": 6, "border_fraction": 0.1})
        assert spec["type"] == "Ccube2"
        assert spec["length"] == 10.0
        assert spec["n_points"] == 6

    def test_omitted_values_take_the_target_default(self):
        spec = dc.target_spec_from_values("Ccube2", {})
        assert spec["type"] == "Ccube2"
        assert spec["n_points"] == dc.default_construction_values("Ccube2")["n_points"]


class TestTargetSpecComparesLayoutNotReading:
    """Only fields that decide what a detected key means are compared."""

    def test_dictionary_difference_is_not_a_mismatch(self):
        left = {"type": "Ccube2", "n_points": 6, "length": 10.0,
                "border_fraction": 0.1, "aruco_dict": "DICT_4X4_50"}
        right = {"type": "Ccube2", "n_points": 6, "length": 10.0,
                 "border_fraction": 0.1, "aruco_dict": "DICT_4X4_1000"}
        assert dc.compare_target_specs(left, right) == []

    def test_length_difference_is_a_mismatch(self):
        left = {"type": "Ccube2", "n_points": 6, "length": 10.0, "border_fraction": 0.1}
        right = {"type": "Ccube2", "n_points": 6, "length": 20.0, "border_fraction": 0.1}
        differences = dc.compare_target_specs(left, right)
        assert differences and "length" in differences[0]

    def test_different_target_type_is_a_mismatch(self):
        left = {"type": "ChArUco2", "num_squares_x": 5}
        right = {"type": "Ccube2", "n_points": 5}
        assert dc.compare_target_specs(left, right)


# --------------------------------------------------------------------------- #
# The frame listing                                                          #
# --------------------------------------------------------------------------- #

def _write_frame(path: pathlib.Path, shape=(720, 540), seed: int = 0):
    import cv2
    image = (np.random.default_rng(seed).random(shape) * 4000).astype(np.uint16)
    cv2.imwrite(str(path), image)


def _make_folder(root: pathlib.Path, cameras=("cam1", "cam2"), frames=3,
                 stamp="20260919_134530"):
    for camera in cameras:
        folder = root / camera
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(frames):
            _write_frame(folder / f"{camera}_{stamp}_{index:06d}.tiff", seed=index)
    return root


class TestListFrames:

    def test_reads_the_producer_convention(self, tmp_path):
        _make_folder(tmp_path)
        listing = dc.list_frames(tmp_path)
        assert listing["camera_folders"] == ["cam1", "cam2"]
        assert listing["total_frames"] == 6
        assert listing["frame_count_uniform"] is True
        assert listing["acquisition_timestamps"] == ["20260919_134530"]

    def test_a_workspace_directory_is_not_a_camera(self, tmp_path):
        """A real folder holds a workspace dir; treating it as a camera gave an
        all-NaN timing row and a false unequal-counts warning."""
        _make_folder(tmp_path)
        (tmp_path / ".pycamset_workspace").mkdir()
        listing = dc.list_frames(tmp_path)
        assert listing["camera_folders"] == ["cam1", "cam2"]
        assert ".pycamset_workspace" in listing["non_frame_dirs"]

    def test_two_acquisition_timestamps_are_reported(self, tmp_path):
        _make_folder(tmp_path, cameras=("cam1",), stamp="20260919_134530")
        _make_folder(tmp_path, cameras=("cam2",), stamp="20260919_140000")
        listing = dc.list_frames(tmp_path)
        assert len(listing["acquisition_timestamps"]) == 2
        assert any("timestamp" in e for e in listing["errors"])

    def test_missing_folder_is_reported_not_raised(self, tmp_path):
        listing = dc.list_frames(tmp_path / "nope")
        assert listing["exists"] is False
        assert listing["errors"]


class TestParseFrameName:

    def test_round_trips_the_declared_pattern(self):
        parsed = dc.parse_frame_name("cam3_20260919_134530_000042.tiff")
        assert parsed["camera"] == "cam3"
        assert parsed["timestamp"] == "20260919_134530"
        assert parsed["index"] == 42

    def test_rejects_a_foreign_name(self):
        assert dc.parse_frame_name("IMG_0042.png") is None


# --------------------------------------------------------------------------- #
# The stretch                                                                #
# --------------------------------------------------------------------------- #

class TestContrastStretch:

    def test_uses_the_frames_own_range_not_the_dtype_range(self):
        """Mono12 data in a uint16 container must reach the full uint8 range."""
        frame = np.array([[0, 2048], [3000, 4095]], dtype=np.uint16)
        stretched = dc.to_uint8_contrast_stretch(frame)
        assert stretched.dtype == np.uint8
        assert stretched.max() == 255
        assert stretched.min() == 0

    def test_uint8_passes_through(self):
        frame = np.array([[0, 255]], dtype=np.uint8)
        assert np.array_equal(dc.to_uint8_contrast_stretch(frame), frame)

    def test_flat_frame_does_not_divide_by_zero(self):
        flat = np.full((4, 4), 1000, dtype=np.uint16)
        stretched = dc.to_uint8_contrast_stretch(flat)
        assert stretched.dtype == np.uint8
        assert stretched.max() == 0


# --------------------------------------------------------------------------- #
# The run                                                                    #
# --------------------------------------------------------------------------- #

class TestPreflight:

    def test_refuses_a_folder_with_no_frames(self, tmp_path):
        with pytest.raises(dc.DetectionCostError):
            dc.preflight(dc.MeasurementOptions(folder=tmp_path, out_dir=tmp_path))

    def test_accepts_a_well_formed_folder(self, tmp_path):
        _make_folder(tmp_path)
        options = dc.MeasurementOptions(folder=tmp_path, out_dir=tmp_path / "out")
        pre = dc.preflight(options)
        assert pre["listing"]["total_frames"] == 6

    def test_refuses_a_target_the_acquisition_contradicts(self, tmp_path):
        """A declared target that disagrees must stop the run, not be measured
        through: wrong geometry gives plausible-looking garbage."""
        _make_folder(tmp_path)
        declared = {"type": "Ccube2", "n_points": 6, "length": 20.0,
                    "border_fraction": 0.1}
        (tmp_path / "metadata.json").write_text(
            json.dumps({"Calibration_Targets": {"Targets": [declared]}}),
            encoding="utf-8")
        options = dc.MeasurementOptions(
            folder=tmp_path, out_dir=tmp_path / "out", target_type="Ccube2",
            target_values={"length": 10.0, "n_points": 6, "border_fraction": 0.1})
        with pytest.raises(dc.DetectionCostError) as excinfo:
            dc.preflight(options)
        assert "length" in str(excinfo.value)

    def test_accepts_a_target_the_acquisition_agrees_with(self, tmp_path):
        _make_folder(tmp_path)
        declared = {"type": "Ccube2", "n_points": 6, "length": 10.0,
                    "border_fraction": 0.1}
        (tmp_path / "metadata.json").write_text(
            json.dumps({"Calibration_Targets": {"Targets": [declared]}}),
            encoding="utf-8")
        options = dc.MeasurementOptions(
            folder=tmp_path, out_dir=tmp_path / "out", target_type="Ccube2",
            target_values={"length": 10.0, "n_points": 6, "border_fraction": 0.1})
        assert dc.preflight(options)["matched_declared_target"] == declared


@pytest.mark.skipif(
    not marker_backend_available("aruco2"),
    reason="measures a ChArUco2 target, whose detector (aruco2) is not installed")
class TestMeasureFolder:

    def test_produces_a_report_and_writes_both_files(self, tmp_path):
        _make_folder(tmp_path, cameras=("cam1",), frames=2)
        options = dc.MeasurementOptions(
            folder=tmp_path, out_dir=tmp_path / "out", target_type="ChArUco2",
            target_values={"num_squares_x": 5, "num_squares_y": 7, "square_size": 30.0},
            max_frames_per_camera=2, measure_single_thread_too=False)
        report = dc.measure_folder(options)
        written = report["written"]
        assert pathlib.Path(written["json"]).is_file()
        assert pathlib.Path(written["summary"]).is_file()
        # The JSON must be JSON, not a repr of the report.
        json.loads(pathlib.Path(written["json"]).read_text(encoding="utf-8"))

    def test_the_report_is_json_safe(self, tmp_path):
        """The in-process frame list must not leak into the report.

        Checked structurally, not by substring: ``unparsed_frames`` is a
        legitimate report key, so searching the serialised text for ``_frames``
        would flag the report for a key that belongs there.
        """
        _make_folder(tmp_path, cameras=("cam1",), frames=1)
        options = dc.MeasurementOptions(
            folder=tmp_path, out_dir=tmp_path / "out", target_type="ChArUco2",
            target_values={"num_squares_x": 5, "num_squares_y": 7, "square_size": 30.0},
            max_frames_per_camera=1, measure_single_thread_too=False)
        report = dc.measure_folder(options)
        json.dumps(report, default=str)
        listing = report["folder_listing"]
        assert "_frames" not in listing
        for name, info in listing["cameras"].items():
            assert "_frames" not in info, f"{name} leaked its in-process frames"

    def test_timings_are_collected(self, tmp_path):
        _make_folder(tmp_path, cameras=("cam1",), frames=2)
        options = dc.MeasurementOptions(
            folder=tmp_path, out_dir=tmp_path / "out", target_type="ChArUco2",
            target_values={"num_squares_x": 5, "num_squares_y": 7, "square_size": 30.0},
            max_frames_per_camera=2, measure_single_thread_too=False)
        report = dc.measure_folder(options)
        assert report["overall"]["frames_measured"] == 2
        assert report["overall"]["decode"]["n"] == 2
        assert report["overall"]["detect"]["n"] == 2

    def test_raw_path_refusal_is_recorded_not_crashed(self, tmp_path):
        """aruco2 requires uint8, so a Mono16 frame is refused. That is a
        finding, not a failure, and it must not stop the run."""
        _make_folder(tmp_path, cameras=("cam1",), frames=1)
        options = dc.MeasurementOptions(
            folder=tmp_path, out_dir=tmp_path / "out", target_type="ChArUco2",
            target_values={"num_squares_x": 5, "num_squares_y": 7, "square_size": 30.0},
            max_frames_per_camera=1, measure_single_thread_too=False)
        report = dc.measure_folder(options)
        assert report["paths"]["raw_frame_dtype"] == "uint16"
        assert report["paths"]["raw_path_accepted"] is False
        assert report["paths"]["raw_path_message"]

    def test_summary_names_the_comparability_hash(self, tmp_path):
        _make_folder(tmp_path, cameras=("cam1",), frames=1)
        options = dc.MeasurementOptions(
            folder=tmp_path, out_dir=tmp_path / "out", target_type="ChArUco2",
            target_values={"num_squares_x": 5, "num_squares_y": 7, "square_size": 30.0},
            max_frames_per_camera=1, measure_single_thread_too=False)
        report = dc.measure_folder(options)
        assert report["module_sha256"]
        assert report["module_sha256"] in dc.summarise(report)


class TestModuleHash:

    def test_is_stable_across_calls(self):
        assert dc.module_hash() == dc.module_hash()

    def test_is_a_sha256(self):
        value = dc.module_hash()
        assert value and len(value) == 64


# --------------------------------------------------------------------------- #
# GUI                                                                        #
# --------------------------------------------------------------------------- #

class TestDetectionCostTab:
    """The tab renders, and its form follows the chosen target."""

    @pytest.fixture(scope="class")
    def qapp(self):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication
        return QApplication.instance() or QApplication([])

    def test_tab_builds(self, qapp):
        from pyCamSet.gui.detection_cost_tab import DetectionCostTab
        tab = DetectionCostTab()
        assert tab._target_form is not None

    def test_form_rows_follow_the_target(self, qapp):
        """The requirement: no field for a setting the target does not have."""
        from pyCamSet.gui.detection_cost_tab import DetectionCostTab
        tab = DetectionCostTab()
        form = tab._target_form

        form.set_target_type("Ccube2")
        cube_keys = {k for k in form.spec() if k != "type"}
        assert "num_squares_x" not in cube_keys
        assert "num_squares_y" not in cube_keys

        form.set_target_type("ChArUco2")
        board_keys = {k for k in form.spec() if k != "type"}
        assert "num_squares_x" in board_keys
        assert "n_points" not in board_keys

    def test_the_tab_is_registered_in_the_main_window(self, qapp):
        from pyCamSet.gui.detection_cost_tab import TAB_DETECTION_COST
        from pyCamSet.gui.main_window import PyCamSetApp
        window = PyCamSetApp()
        titles = [window._notebook.tabText(i)
                  for i in range(window._notebook.count())]
        assert TAB_DETECTION_COST in titles

    def test_options_refuse_an_empty_folder(self, qapp):
        from pyCamSet.gui.detection_cost_tab import DetectionCostTab
        tab = DetectionCostTab()
        with pytest.raises(ValueError):
            tab._options()
