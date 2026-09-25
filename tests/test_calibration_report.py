"""Tests for the calibration report and the logging setup.

The report exists so that a run's result is legible and comparable, so the
tests pin the two things that would quietly ruin that: residuals being
attributed to the wrong camera, and the summary growing past the width of a
terminal.
"""

from __future__ import annotations

import io
import logging
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from pyCamSet.calibration_targets.core.target_detections import (
    ImageDetection, TargetDetection)
from pyCamSet.utils import logs
from pyCamSet.utils import report_format as fmt
from pyCamSet.utils.calibration_report import (
    HIGH_FINAL_ERROR_PX,
    CalibrationReport,
    CameraErrorStats,
)
from pyCamSet.utils.intrinsics_report import (
    HIGH_INITIAL_RMS_PX, IntrinsicsReport)
from pyCamSet.utils.progress import OptimisationProgress
from pyCamSet.utils.setup_reports import DetectionReport, RigConsistencyReport

MAX_WIDTH = 80
REPO_ROOT = Path(__file__).resolve().parent.parent


def _over_width(text: str) -> list[str]:
    """Lines that reach MAX_WIDTH, measured with the escape codes stripped:
    colour costs characters but no columns."""
    return [line for line in text.split("\n")
            if len(fmt.strip_ansi(line)) >= MAX_WIDTH]


def _colours(text: str, row_contains: str) -> list[str]:
    """The colour names used on the first line containing row_contains."""
    names = {v: k for k, v in fmt.SOLARIZED.items()}
    line = next(l for l in text.split("\n") if row_contains in fmt.strip_ansi(l))
    return [names[int(c)] for c in re.findall(r"\x1b\[38;5;(\d+)m", line)]


class _StubHandler:
    """The three attributes the report reads off a parameter handler."""

    def __init__(self, detection, missing_poses=None):
        self.detection = detection
        self.missing_poses = missing_poses


def _detection(rows) -> TargetDetection:
    """A detection built straight from | cam | im | key | x | y | rows."""
    return TargetDetection(
        cam_names=["left", "right"], data=np.array(rows, dtype=float)
    )


def _residuals(pairs) -> np.ndarray:
    return np.array(pairs, dtype=float).reshape(-1)


def _report(rows, pairs, **kwargs) -> CalibrationReport:
    detection = _detection(rows)
    optimisation = OptimizeResult(
        x=np.zeros(7), fun=_residuals(pairs), status=2,
        message="ftol reached", nit=11, nfev=13,
    )
    return CalibrationReport.from_optimisation(
        optimisation, _StubHandler(detection, kwargs.pop("missing_poses", None)),
        initial_error_px=kwargs.pop("initial_error_px", 4.0),
        duration_s=kwargs.pop("duration_s", 1.5),
        solver=kwargs.pop("solver", "schur"),
    )


# Two points on camera 0 with error 3 and 4 px, two on camera 1 with 30 and 40.
_ROWS = [
    [0, 0, 0, 1.0, 1.0],
    [0, 1, 1, 1.0, 1.0],
    [1, 0, 0, 1.0, 1.0],
    [1, 1, 1, 1.0, 1.0],
]
_PAIRS = [[3.0, 0.0], [0.0, 4.0], [30.0, 0.0], [0.0, 40.0]]


def test_errors_are_attributed_to_the_right_camera():
    """The whole point of the per camera table is that it is per camera.

    The residuals arrive in the row order of the detection they were built
    from, so this is the correspondence a reordering would silently break.
    """
    report = _report(_ROWS, _PAIRS)

    by_name = {c.name: c for c in report.per_camera}
    assert set(by_name) == {"left", "right"}
    assert by_name["left"].mean_px == pytest.approx(3.5)
    assert by_name["right"].mean_px == pytest.approx(35.0)
    assert by_name["left"].n_points == 2
    assert by_name["left"].max_px == pytest.approx(4.0)


def test_overall_statistics_describe_the_whole_run():
    report = _report(_ROWS, _PAIRS)

    assert report.n_control_points == 4
    assert report.n_cameras == 2
    assert report.mean_px == pytest.approx(np.mean([3, 4, 30, 40]))
    assert report.median_px == pytest.approx(17.0)
    assert report.rms_px == pytest.approx(np.sqrt(np.mean([9, 16, 900, 1600])))
    assert report.max_px == pytest.approx(40.0)
    assert report.n_parameters == 7
    assert report.solver == "schur"
    assert report.n_iterations == 11


def test_worst_images_are_ranked_worst_first():
    report = _report(_ROWS, _PAIRS)

    # image 1 holds the 4 px and 40 px points, image 0 the 3 px and 30 px
    assert [i.index for i in report.worst_images] == [1, 0]
    assert report.worst_images[0].mean_px == pytest.approx(22.0)


def test_a_mismatched_detection_degrades_instead_of_lying():
    """A handler whose loss is built from something else must not produce a
    confidently wrong per camera table."""
    report = _report(_ROWS, _PAIRS[:2])  # two residual pairs, four rows

    assert report.per_camera == []
    assert report.worst_images == []
    # the overall numbers still describe what the solver returned
    assert report.n_control_points == 2


def test_high_final_error_is_flagged_with_its_threshold():
    report = _report(_ROWS, _PAIRS)

    assert report.mean_px > HIGH_FINAL_ERROR_PX
    assert any(str(int(HIGH_FINAL_ERROR_PX)) in f for f in report.flags)


def test_a_good_calibration_carries_no_flags():
    quiet = [[0.1, 0.0], [0.0, 0.1], [0.1, 0.0], [0.0, 0.1]]
    report = _report(_ROWS, quiet, initial_error_px=2.0)

    assert report.flags == []


def test_missing_poses_are_flagged():
    report = _report(_ROWS, _PAIRS,
                     missing_poses=np.array([False, True, True]))

    assert report.n_missing_poses == 2
    assert any("no usable target pose" in f for f in report.flags)


def test_nan_errors_are_flagged_rather_than_formatted():
    report = _report(_ROWS, [[np.nan, np.nan]] * 4)

    assert any("NaN" in f for f in report.flags)
    assert "nan" in report.summary()


def test_a_full_worst_image_list_cannot_widen_the_table():
    """The regression that motivated the wrapping.

    With five images named, each as "index (n.nn px)", this line ran to 83
    columns on a real three camera run.
    """
    rows, pairs = [], []
    for im in range(12, 18):           # two digit indices, as in a real run
        for cam in (0, 1):
            rows.append([cam, im, 0, 1.0, 1.0])
            pairs.append([float(im), 0.0])
    report = _report(rows, pairs)

    assert len(report.worst_images) == 5
    over = _over_width(report.summary(colour=True))
    assert not over, f"lines at or over {MAX_WIDTH} columns: {over}"


def test_the_summary_names_every_camera():
    report = _report(_ROWS, _PAIRS)
    summary = report.summary()

    for name in ("left", "right"):
        assert name in summary
    assert "Calibration summary" in summary


def test_a_long_camera_name_cannot_widen_the_table():
    report = _report(_ROWS, _PAIRS)
    report.per_camera = [
        CameraErrorStats(name="a" * 60, index=0, n_points=1, mean_px=1.0,
                         median_px=1.0, rms_px=1.0, p95_px=1.0, max_px=1.0)
    ]

    assert not _over_width(report.summary(colour=True))


def test_the_report_round_trips_through_a_dict():
    report = _report(_ROWS, _PAIRS)

    rebuilt = CalibrationReport.from_dict(report.to_dict())

    assert rebuilt.to_dict() == report.to_dict()
    assert isinstance(rebuilt.per_camera[0], CameraErrorStats)
    assert rebuilt.per_camera[0].name == "left"


def test_from_dict_ignores_fields_it_does_not_know():
    """An older or newer file must not crash the load."""
    data = _report(_ROWS, _PAIRS).to_dict()
    data["something_from_the_future"] = 12

    assert CalibrationReport.from_dict(data).n_cameras == 2


class TestLoggingSetup:
    """pyCamSet must not hijack the root logger of an importing application."""

    def setup_method(self):
        self.logger = logging.getLogger(logs.LOGGER_NAME)
        self.root = logging.getLogger()
        self._saved = list(self.logger.handlers)
        self._saved_level = self.logger.level
        self._saved_root = list(self.root.handlers)

    def teardown_method(self):
        self.logger.handlers = self._saved
        self.logger.setLevel(self._saved_level)
        self.root.handlers = self._saved_root

    def test_importing_pycamset_installs_no_real_handler(self):
        """Checked in a clean interpreter: pytest puts its own capture
        handler on the root logger, and an earlier test in the suite may
        legitimately have run a calibration and configured logging."""
        code = (
            "import logging, pyCamSet;"
            "real = lambda lg: [h for h in lg.handlers"
            "                   if not isinstance(h, logging.NullHandler)];"
            "print(bool(real(logging.getLogger()))"
            "      or bool(real(logging.getLogger('pyCamSet'))))"
        )
        done = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "False", done.stdout

    def test_setup_installs_on_the_pycamset_logger_only(self):
        self.root.handlers = []
        self.logger.handlers = []

        logs.setup_logging()

        assert logs._has_real_handler(self.logger)
        assert not logs._has_real_handler(self.root)

    def test_setup_defers_to_a_configured_root(self):
        """An application that configured logging keeps its configuration,
        and does not get pyCamSet's records twice."""
        self.root.handlers = [logging.StreamHandler()]
        self.logger.handlers = []

        logs.setup_logging()

        assert not logs._has_real_handler(self.logger)

    def test_setup_defers_to_an_existing_configuration(self):
        self.root.handlers = []
        self.logger.handlers = [logging.StreamHandler()]

        logs.setup_logging()

        assert len(self.logger.handlers) == 1

    def test_records_still_propagate_for_capture(self):
        """caplog and any application root handler rely on propagation."""
        self.root.handlers = []
        self.logger.handlers = []
        logs.setup_logging()

        assert self.logger.propagate

    def test_verbosity_maps_onto_levels(self):
        assert logs.VERBOSITY_LEVELS[0] == logging.WARNING
        assert logs.VERBOSITY_LEVELS[2] == logging.INFO
        assert logs.VERBOSITY_LEVELS[3] == logging.DEBUG


# --------------------------------------------------------------------------
# Setup stage reports
# --------------------------------------------------------------------------

def _consistency_stack(n_cams=3, n_poses=5, drift=0.0, twist=0.0):
    """Target poses per camera per image, optionally drifting per image."""
    stack = np.tile(np.eye(4), (n_cams, n_poses, 1, 1))
    for pose in range(n_poses):
        stack[1, pose, :3, -1] = [pose * drift, 0, 0]
        angle = pose * twist
        c, s = np.cos(angle), np.sin(angle)
        stack[1, pose, :2, :2] = [[c, -s], [s, c]]
    return stack


def test_rig_consistency_reports_scatter_in_mm_and_degrees():
    report = RigConsistencyReport.from_transforms(
        _consistency_stack(drift=0.001), ref_cam=0, cam_names=["a", "b", "c"])

    moving = [c for c in report.per_camera if c.name == "b"][0]
    # 0, 1, 2, 3, 4 mm has a population stdev of ~1.41 mm
    assert moving.translation_stdev_mm == pytest.approx(1.414, abs=0.01)
    assert moving.angle_stdev_deg == pytest.approx(0.0, abs=1e-6)
    assert report.flags == []


def test_rig_consistency_flags_a_rotating_camera():
    report = RigConsistencyReport.from_transforms(
        _consistency_stack(twist=np.radians(20)), ref_cam=0)

    assert any("rotates" in f and "degrees" in f for f in report.flags)


def test_both_setup_summaries_fit_a_terminal():
    """The setup blocks share the layout with the final summary, so they
    share its one hard constraint. Measured with the escape sequences
    stripped, since those cost characters but no columns."""
    rig = RigConsistencyReport.from_transforms(
        _consistency_stack(drift=0.02, twist=np.radians(20)), ref_cam=0,
        cam_names=["a_very_long_camera_name_indeed" * 2, "b", "c"])

    over = _over_width(rig.summary(colour=True))
    assert not over, f"rig consistency over {MAX_WIDTH}: {over}"


def _detection_over(n_images: int, seen: int) -> TargetDetection:
    """A one camera detection that saw *seen* of *n_images* images."""
    detection = TargetDetection(cam_names=["cam"])
    detection.max_ims = n_images
    for im_num in range(seen):
        detection.add_detection(
            "cam", im_num,
            ImageDetection(keys=np.arange(4),
                           image_points=np.tile([1.0, 2.0], (4, 1))))
    return detection


def test_a_detection_rate_is_measured_against_the_images_asked_for():
    """A run limited to 5 of 50 images saw everything it was asked to see,
    and reporting that as 10% would send someone looking for a fault that is
    not there."""
    report = DetectionReport.from_detection(
        _detection_over(5, 5), _Corners(4),
        image_counts={"cam": 50}, n_lim=5)

    assert report.per_camera[0].n_images_expected == 5
    assert report.per_camera[0].detection_rate == 1.0


def test_a_detection_rate_counts_the_images_a_camera_was_given():
    """With no cap it is the folder's own image count, not the images that
    happened to produce a detection."""
    report = DetectionReport.from_detection(
        _detection_over(5, 5), _Corners(4), image_counts={"cam": 10})

    assert report.per_camera[0].n_images_expected == 10
    assert report.per_camera[0].detection_rate == 0.5


def test_a_wholly_blind_run_is_said_once_rather_than_per_camera():
    detection = TargetDetection(cam_names=["a", "b", "c"])
    detection.max_ims = 4

    report = DetectionReport.from_detection(detection, _Corners(4))

    assert len(report.flags) == 1
    assert "no camera detected the target" in report.flags[0]


# --------------------------------------------------------------------------
# Initial intrinsics
# --------------------------------------------------------------------------

class _StubCamera:
    """The three attributes the intrinsics report reads off a camera."""

    def __init__(self, focal: float):
        self.intrinsic = np.array(
            [[focal, 0.0, 640.0], [0.0, focal, 480.0], [0.0, 0.0, 1.0]])
        self.distortion_coefs = np.array([0.1, -0.05, 0.0, 0.0])
        self.res = (1280, 960)


class _StubCamSet:
    def __init__(self, names, focal=1000.0):
        self._cams = {name: _StubCamera(focal) for name in names}

    def get_names(self):
        return list(self._cams)

    def __getitem__(self, name):
        return self._cams[name]


def _intrinsics(monkeypatch, rms: dict[str, float], focal=1000.0,
                n_views=4) -> IntrinsicsReport:
    """A report over cameras with the given pooled RMS, a NaN meaning the
    camera was never posed."""
    from pyCamSet.utils import intrinsics_report as module

    per_view = {
        name: {"valid_pose": [not np.isnan(value)] * n_views,
               "n_points": [10] * n_views}
        for name, value in rms.items()
    }
    monkeypatch.setattr(
        module, "per_view_reprojection", lambda *_: (per_view, rms))
    return IntrinsicsReport.from_calibration(
        _StubCamSet(list(rms), focal), SimpleNamespace(max_ims=8), object())


def test_the_intrinsics_summary_reports_each_camera(monkeypatch):
    report = _intrinsics(monkeypatch, {"left": 0.4, "right": 0.6})

    assert [c.name for c in report.per_camera] == ["left", "right"]
    assert report.per_camera[0].n_views == 4
    assert report.per_camera[0].n_points == 40
    assert report.per_camera[0].fx == 1000.0
    assert report.flags == []


def test_a_camera_that_was_never_posed_is_flagged(monkeypatch):
    report = _intrinsics(monkeypatch, {"left": 0.4, "right": float("nan")})

    assert len(report.flags) == 1
    assert 'camera "right"' in report.flags[0]
    assert report.per_camera[1].n_views == 0


def test_a_high_initial_rms_is_flagged_with_its_threshold(monkeypatch):
    report = _intrinsics(monkeypatch, {"left": HIGH_INITIAL_RMS_PX + 1.0})

    assert len(report.flags) == 1
    assert f"{HIGH_INITIAL_RMS_PX:.0f} px" in report.flags[0]


def test_the_intrinsics_summary_fits_a_terminal(monkeypatch):
    """The fourth block shares the layout, so it shares its one hard
    constraint: a long camera name and a wide focal length cannot push it
    past the width."""
    report = _intrinsics(
        monkeypatch, {"a_very_long_camera_name_indeed" * 2: 123.456},
        focal=123456.789)

    over = _over_width(report.summary(colour=True))
    assert not over, f"initial intrinsics over {MAX_WIDTH}: {over}"


def test_the_intrinsics_report_keeps_its_working_out_of_the_record(monkeypatch):
    """The per view series is one entry per image per camera: the report is
    computed from it, and the phase records it as a diagnostic of its own."""
    report = _intrinsics(monkeypatch, {"left": 0.4})

    assert report.per_view
    assert "per_view" not in report.to_dict()
    assert report.to_dict()["per_camera"][0]["rms_px"] == 0.4


class TestColourBands:
    """90%+ green, 60-89% orange, under 60% red."""


    def test_colour_costs_no_columns(self):
        """The whole reason cells are padded before they are coloured: a
        coloured table must line up with an uncoloured one."""
        rows = [["cam", fmt.quality_cell(0.5), fmt.quality_cell(1.0)]]
        headers, widths = ["camera", "a", "b"], [10, 8, 8]

        plain = fmt.table(headers, rows, widths, colour=False)
        shaded = fmt.table(headers, rows, widths, colour=True)

        assert [fmt.strip_ansi(line) for line in shaded] == plain
        assert "\x1b[" in shaded[-1]
        assert "\x1b[" not in plain[-1]

    def test_no_colour_env_var_is_honoured(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert fmt.colour_enabled() is False


    def test_it_stays_plain_when_nothing_is_watching(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        buffer = io.StringIO()          # a StringIO is not a terminal
        monkeypatch.setattr(sys, "stderr", buffer)

        assert fmt.colour_enabled() is False

    def test_a_front_end_can_say_it_shows_colour(self, monkeypatch):
        """The GUI paints these blocks into a console of its own, and reads
        them off a pipe to do it -- which is exactly what the detection above
        calls "nothing is watching"."""
        monkeypatch.setattr(fmt, "_FORCED", None)
        monkeypatch.setattr(sys, "stderr", io.StringIO())

        fmt.force_colour(True)
        assert fmt.colour_enabled() is True
        assert "\x1b[" in fmt.table(
            ["camera", "rate"], [["cam", fmt.quality_cell(0.5)]], [10, 8])[-1]

        fmt.force_colour(None)
        assert fmt.colour_enabled() is False

    def test_a_coloured_detection_summary_still_fits(self):
        """Colour must not be a back door around the width limit."""
        detection = TargetDetection(cam_names=["a" * 40, "b"])
        detection.max_ims = 3
        for im_num in range(3):
            detection.add_detection(
                "a" * 40, im_num,
                ImageDetection(keys=np.arange(4),
                               image_points=np.tile([1.0, 2.0], (4, 1))))
        report = DetectionReport.from_detection(detection, _Corners(16))

        summary = report.summary(colour=True)
        over = [line for line in summary.split("\n")
                if len(fmt.strip_ansi(line)) >= MAX_WIDTH]
        assert not over, f"over {MAX_WIDTH} columns: {over}"
        assert "\x1b[" in summary


class _Corners:
    """A target with a known number of corners per face."""

    def __init__(self, corners: int):
        self.point_data = np.zeros((1, corners, 3))


def test_setup_reports_round_trip_to_plain_types():
    rig = RigConsistencyReport.from_transforms(_consistency_stack(), ref_cam=0)

    assert isinstance(rig.to_dict()["per_camera"], list)
    assert isinstance(rig.to_dict()["per_camera"][0]["angle_stdev_deg"], float)


class TestProgress:
    """The counter must be invisible unless someone is watching."""

    def test_it_writes_nothing_when_not_interactive(self):
        buffer = io.StringIO()
        real, sys.stderr = sys.stderr, buffer
        try:
            with OptimisationProgress() as progress:
                progress.update(1, 5.0, np.zeros(10))
        finally:
            sys.stderr = real

        assert buffer.getvalue() == ""

    def test_it_reports_cost_and_error_when_interactive(self):
        # tqdm binds its stream when the bar is built, so the swap has to
        # happen before the context is entered
        buffer = io.StringIO()
        real, sys.stderr = sys.stderr, buffer
        try:
            with OptimisationProgress(enabled=True) as progress:
                progress.update(1, 1234.0, np.array([3.0, 4.0, 3.0, 4.0]))
        finally:
            sys.stderr = real

        written = buffer.getvalue()
        assert "cost 1234" in written
        assert "error 5.000 px" in written

    def test_it_survives_empty_and_missing_residuals(self):
        with OptimisationProgress(enabled=False) as progress:
            progress.update(1, 5.0, None)
            progress.update(2, float("nan"), np.array([]))


class TestDeviationBands:
    """Rig deviations: green under 1, orange under 5, red at or above."""


    def test_both_rig_columns_are_graded(self):
        """Shift in mm and rotation in degrees share one scale."""
        stack = np.tile(np.eye(4), (2, 5, 1, 1))
        for pose in range(5):                 # ~8.5 mm and ~3.2 deg of drift
            stack[1, pose, :3, -1] = [pose * 0.006, 0, 0]
            angle = pose * 0.04
            c, s = np.cos(angle), np.sin(angle)
            stack[1, pose, :2, :2] = [[c, -s], [s, c]]

        report = RigConsistencyReport.from_transforms(
            stack, ref_cam=0, cam_names=["ref", "drifter"])

        assert _colours(report.summary(colour=True), "drifter") == ["red", "orange"]


class TestErrorBands:
    """Reprojection error: blue under 0.1, green under 1, orange under 5,
    red at or above 5 -- the same 5 px the high error flag uses."""


    def test_the_red_band_matches_the_high_error_flag(self):
        assert fmt.ERROR_FAIR_PX == HIGH_FINAL_ERROR_PX


    def test_the_per_camera_columns_span_all_four_bands(self):
        rows, pairs = [], []
        for cam, error in enumerate([0.05, 0.4, 3.1, 7.6]):
            rows.append([cam, 0, 0, 1.0, 1.0])
            pairs.append([error, 0.0])
        detection = TargetDetection(
            cam_names=["exceptional", "good", "workable", "bad"],
            data=np.array(rows, dtype=float))
        optimisation = OptimizeResult(
            x=np.zeros(4), fun=np.array(pairs).reshape(-1),
            status=2, message="ftol reached", nit=1, nfev=1)
        report = CalibrationReport.from_optimisation(
            optimisation, _StubHandler(detection), initial_error_px=9.0,
            duration_s=1.0, solver="schur")

        summary = report.summary(colour=True)
        # four graded columns per row: mean, median, p95, max
        assert _colours(summary, "exceptional") == ["blue"] * 4
        assert _colours(summary, "good") == ["green"] * 4
        assert _colours(summary, "workable") == ["orange"] * 4
        assert _colours(summary, "bad") == ["red"] * 4

    def test_the_points_column_is_never_graded(self):
        """It is a count, not a quality."""
        report = _report(_ROWS, _PAIRS)

        # two cameras, four graded cells each, and nothing more
        assert len(_colours(report.summary(colour=True), "left")) == 4
