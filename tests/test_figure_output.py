"""Tests for saving the calibration diagnostics instead of blocking on them.

A bare ``plt.show()`` is fine at a desk and useless anywhere else: from a
script it waits on a window nobody will close, and the figure it meant to draw
is the diagnostic the run most needed to keep. These pin that every plot on
the calibration path can be written to a file and that none of them opens a
window unless asked.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyCamSet.utils.visualisation import finalise_figure, finalise_plotter


@pytest.fixture
def figure():
    """A throwaway figure, closed however the test leaves it."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    yield fig
    plt.close(fig)


@pytest.fixture
def no_new_figures():
    """Asserts the block under it leaves no figure of its own open.

    Measured as a difference rather than against an empty registry: pyplot's
    figure list is process wide, and other tests in the suite leave figures
    behind, so an absolute check here would fail for someone else's reason.
    """
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    assert not leaked, f"left figures open: {sorted(leaked)}"


class TestFinaliseFigure:

    def test_it_writes_a_png_and_says_where(self, figure, tmp_path):
        written = finalise_figure(figure, "coverage", show=False,
                                  save_dir=tmp_path)

        assert written == tmp_path / "coverage.png"
        assert written.stat().st_size > 0

    def test_it_creates_the_directory(self, figure, tmp_path):
        target = tmp_path / "does" / "not" / "exist"

        written = finalise_figure(figure, "coverage", show=False,
                                  save_dir=target)

        assert written.is_file()

    def test_it_closes_the_figure_when_nothing_is_shown(self, figure, tmp_path):
        """Left open, a few hundred figures from a batch run exhaust memory
        and matplotlib starts warning about it."""
        finalise_figure(figure, "coverage", show=False, save_dir=tmp_path)

        assert figure.number not in plt.get_fignums()

    def test_it_writes_nothing_without_a_directory(self, figure):
        assert finalise_figure(figure, "coverage", show=False,
                               save_dir=None) is None

    def test_it_can_both_save_and_show(self, figure, tmp_path):
        # the Agg backend makes show() a no-op, so this checks the save half
        # still happens when show is asked for
        written = finalise_figure(figure, "coverage", show=True,
                                  save_dir=tmp_path)

        assert written.is_file()


class _Plotter:
    """A stand-in for a pyvista plotter, recording what was asked of it."""

    def __init__(self, fail=False):
        self.fail = fail
        self.off_screen = False
        self.shown_with = None
        self.closed = False
        self.key_events = {}
        self.screenshots = []

    def show(self, screenshot=None):
        if self.fail:
            raise RuntimeError("no render window available")
        self.shown_with = screenshot
        if screenshot is not None:
            with open(screenshot, "wb") as handle:
                handle.write(b"png")

    def close(self):
        self.closed = True

    def add_key_event(self, key, callback):
        self.key_events[key] = callback

    def screenshot(self, path):
        self.screenshots.append(path)
        with open(path, "wb") as handle:
            handle.write(b"png")


class TestFinalisePlotter:

    def test_it_screenshots_to_the_directory(self, tmp_path):
        plotter = _Plotter()

        written = finalise_plotter(plotter, "reconstruction", show=False,
                                   save_dir=tmp_path)

        assert written == tmp_path / "reconstruction.png"
        assert plotter.shown_with == str(written)

    def test_it_goes_off_screen_to_avoid_blocking(self, tmp_path):
        """pyvista holds the screenshot until the window is closed, so a
        save with nobody watching has to render off screen."""
        plotter = _Plotter()

        finalise_plotter(plotter, "reconstruction", show=False,
                         save_dir=tmp_path)

        assert plotter.off_screen is True

    def test_showing_leaves_the_window_on_screen(self, tmp_path):
        plotter = _Plotter()

        finalise_plotter(plotter, "reconstruction", show=True,
                         save_dir=tmp_path)

        assert plotter.off_screen is False

    def test_showing_binds_the_screenshot_key(self, tmp_path):
        """The reason to open a window is to find a view worth keeping."""
        plotter = _Plotter()

        finalise_plotter(plotter, "reconstruction", show=True, save_dir=tmp_path)

        assert "s" in plotter.key_events
        plotter.key_events["s"]()
        assert len(plotter.screenshots) == 1
        assert Path(plotter.screenshots[0]).parent == tmp_path

    def test_off_screen_does_not_bind_the_screenshot_key(self):
        """Nobody is at the keyboard, so there is nothing to bind."""
        plotter = _Plotter()

        finalise_plotter(plotter, "x", show=False, save_dir=None)

        assert plotter.key_events == {}

    def test_it_closes_the_plotter_when_neither_showing_nor_saving(self):
        plotter = _Plotter()

        assert finalise_plotter(plotter, "x", show=False, save_dir=None) is None
        assert plotter.closed

    def test_a_render_failure_warns_rather_than_raising(self, tmp_path, caplog):
        """A diagnostic that cannot be drawn must not take a finished
        calibration down with it."""
        plotter = _Plotter(fail=True)

        with caplog.at_level(logging.WARNING):
            written = finalise_plotter(plotter, "reconstruction", show=False,
                                       save_dir=tmp_path)

        assert written is None
        assert "Could not save the reconstruction view" in caplog.text
        assert plotter.closed


class TestCameraSetPlots:

    def test_distortions_can_be_saved_without_a_window(
            self, synthetic_camset, tmp_path, no_new_figures):
        written = synthetic_camset.draw_camera_distortions(
            show=False, save_dir=tmp_path)

        assert written == tmp_path / "camera_distortions.png"
        assert written.stat().st_size > 0

    def test_visualise_calibration_needs_a_calibration(self, synthetic_camset):
        with pytest.raises(ValueError, match="no calibration data"):
            synthetic_camset.visualise_calibration(show=False)


@pytest.mark.slow
@pytest.mark.data
# The only two tests that drive a real pyvista plotter rather than a stub:
# visualise_calibration renders the reconstruction view through VTK.
@pytest.mark.needs_opengl
def test_a_whole_diagnostic_set_can_be_saved(charuco_problem, tmp_path,
                                             no_new_figures):
    """The end of the exercise: every figure from a real calibration, written
    to disk by a run with nobody watching it."""
    from pyCamSet.calibration.camera_calibrator import run_stereo_calibration

    target, detections, cams = charuco_problem
    final = run_stereo_calibration(
        cams, detections, target, save=False, threads=1,
        problem_options={"outliers": "n", "max_nfev": 20},
    )

    written = final.visualise_calibration(show=False, save_dir=tmp_path)

    assert [path.name for path in written] == [
        "error_distribution.png",
        "per_camera_coverage.png",
        "accuracy_precision.png",
        "reconstruction.png",
        "target_coordinates.png",
    ]
    for path in written:
        assert path.stat().st_size > 1000, f"{path.name} looks blank"


@pytest.mark.slow
@pytest.mark.data
# The only two tests that drive a real pyvista plotter rather than a stub:
# visualise_calibration renders the reconstruction view through VTK.
@pytest.mark.needs_opengl
def test_special_plots_are_skipped_when_nobody_is_watching(
        charuco_problem, tmp_path):
    """special_plots drives its own window and its signature belongs to the
    parameter handler API outside this repository, so it is not asked to
    save -- it is skipped instead."""
    from pyCamSet.calibration.camera_calibrator import run_stereo_calibration

    target, detections, cams = charuco_problem
    final = run_stereo_calibration(
        cams, detections, target, save=False, threads=1,
        problem_options={"outliers": "n", "max_nfev": 20},
    )
    calls = []
    final.calibration_handler.special_plots = lambda x: calls.append(x)

    final.visualise_calibration(show=False, save_dir=tmp_path)

    assert calls == []


# The docs draw every target by running `target.plot()` while the site is
# built, so a target that cannot draw itself headlessly breaks the docs build
# rather than anything a user would see.  These say so here instead.
PLANAR_TARGETS = ["ChArUco", "PuzzleBoard"]
CUBE_TARGETS = ["Ccube", "PuzzleBoardCube"]


def _skip_without_cairo():
    """Skip when the native cairo library is missing, as it is on test CI.

    cairosvg raises OSError, not ImportError, when the library it binds to is
    absent, so importorskip does not catch it.
    """
    try:
        import cairosvg  # noqa: F401
    except (ImportError, OSError) as err:
        pytest.skip(f"native cairo is unavailable: {err}")


def test_the_two_groups_cover_every_target():
    """A fifth target has to decide which of these it is."""
    from pyCamSet.calibration_targets import TARGET_NAMES

    assert sorted(PLANAR_TARGETS + CUBE_TARGETS) == sorted(TARGET_NAMES)


@pytest.mark.parametrize("name", PLANAR_TARGETS)
def test_a_flat_target_draws_itself_into_a_figure(name, no_new_figures):
    """What the docs capture from a flat board: the printed page drawn into the
    current figure and left open for the build to write out as an SVG.

    The *current* figure, because that is what lets a docs block set the size
    before it draws.  So this opens one and checks plot() used it, rather than
    checking plot() created one -- which it does not when a figure is open,
    including one another test in the suite left behind.
    """
    _skip_without_cairo()
    from pyCamSet.calibration_targets import target_class

    figure = plt.figure()
    try:
        target_class(name)().plot()

        assert plt.gcf() is figure, f"{name}.plot() drew into its own figure"
        assert figure.axes and figure.axes[0].images, f"{name}.plot() drew nothing"
    finally:
        plt.close(figure)


@pytest.mark.needs_opengl
@pytest.mark.parametrize("name", CUBE_TARGETS)
def test_a_cube_target_renders_a_scene(name, no_new_figures):
    """What the docs capture from a cube: a plotter that renders, which is
    what pyvista serialises into the page's turnable frame."""
    _skip_without_cairo()
    from pyCamSet.calibration_targets import target_class

    scene = target_class(name)().plot(return_scene=True)
    scene.off_screen = True
    try:
        image = np.asarray(scene.screenshot(return_img=True))
    finally:
        scene.close()

    assert image.ndim == 3 and image.size, f"{name} rendered no image"
    assert len(np.unique(image.reshape(-1, image.shape[-1]), axis=0)) > 1, (
        f"{name} rendered a blank image")
