from __future__ import annotations
import datetime
import json
import logging
from dataclasses import dataclass
from math import copysign
from copy import copy
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import numpy as np
try:
    import pyvista as pv
    _PYVISTA_OK = True
except ImportError:  # pragma: no cover
    pv = None
    _PYVISTA_OK = False
from matplotlib.colors import LogNorm, LinearSegmentedColormap

from pathlib import Path

from pyCamSet.utils.general_utils import h_tform, get_close_square_tuple
from pyCamSet.utils.gui_safety import refuse_window_inside_qt
from pyCamSet.utils.calibration_report import reprojection_residuals
from pyCamSet.optimisation.compiled_helpers import n_htform_prealloc, n_inv_pose

logger = logging.getLogger(__name__)

# Open3D is optional (the "viz" extra) but its import is heavy -- ~1.4s,
# mostly open3d.visualization.draw_plotly pulling in dash. This module is
# imported by cameras.camera_set, which every target build imports, so an
# eager `import open3d` here used to tax every caller with Open3D installed
# regardless of whether they ever touched the Open3D renderer. _open3d()
# resolves and caches the module on first actual use instead; every use
# below goes through it rather than the module being imported at parse time.
_o3d = None
_OPEN3D_OK: bool | None = None  # None = not yet resolved


def _open3d():
    """Import Open3D on first call and cache it; return the module, or None if unavailable.

    Caught as ``Exception`` rather than just ``ImportError``: a broken or
    partial Open3D install (mismatched native libs, missing CUDA, etc.) can
    raise other exception types during import, and this path must stay
    tolerant so callers fall through to the "not installed" message instead
    of an unrelated traceback.
    """
    global _o3d, _OPEN3D_OK
    if _OPEN3D_OK is None:
        try:
            import open3d as _module
            _o3d = _module
            _OPEN3D_OK = True
        except Exception:  # pragma: no cover
            _o3d = None
            _OPEN3D_OK = False
    return _o3d


def _target_mean_distance(target) -> float:
    """Compute the mean Euclidean distance of target points from the origin.

    For flat PuzzleBoard, point_data spans the entire 501x501 virtual code-lookup
    field (251,001 positions), not the physically printed window. Computing the
    mean over all positions gives an inflated value (~765mm vs ~194mm for the
    printed window), which degrades the 3*mean_dist outlier filter and coordinate
    frame sizing. This helper dispatches on target type: PuzzleBoard uses only
    the printed-window points; all other targets use point_data directly.
    """
    if target.__class__.__name__ == "PuzzleBoard":
        nsx = int(getattr(target, "num_squares_x", 105))
        nsy = int(getattr(target, "num_squares_y", 148))
        start_x = int(getattr(target, "start_x", 0))
        start_y = int(getattr(target, "start_y", 0))
        from pyCamSet.calibration_targets.puzzleboard.target import _CODE_SIZE
        # Gather only the printed-window point coordinates from point_data.
        pts = []
        for row in range(start_y, start_y + nsy):
            for col in range(start_x, start_x + nsx):
                pid = row * _CODE_SIZE + col
                pts.append(target.point_data[0, pid])
        pts = np.array(pts, dtype=float)
        return float(np.mean(np.linalg.norm(pts, axis=-1)))
    return float(np.mean(np.linalg.norm(target.point_data, axis=-1)))


def _publication_render_size(width_mm: float, dpi: int, aspect_ratio: float = 8 / 3) -> tuple[int, int]:
    """Convert a publication width/DPI preset to deterministic renderer pixels."""
    if width_mm <= 0 or dpi <= 0 or aspect_ratio <= 0:
        raise ValueError("Width, DPI and aspect ratio must all be positive.")
    width_px = round(float(width_mm) * int(dpi) / 25.4)
    return width_px, round(width_px / aspect_ratio)


def save_pyvista_screenshot(plotter, output_path: str | Path,
                            width_mm: float = 160.0, dpi: int = 150) -> tuple[int, int]:
    """Save a PyVista render at the requested publication pixel dimensions."""
    size = _publication_render_size(width_mm, dpi)
    plotter.screenshot(str(output_path), window_size=size)
    return size

blues_with_white = LinearSegmentedColormap.from_list('Blues_with_white', [(1, 1, 1), *plt.cm.Blues(np.linspace(0, 1, 1024)[:900])])


def finalise_figure(figure, name: str, show: bool = True,
                    save_dir: Path | str | None = None) -> Path | None:
    """
    Save a matplotlib figure, show it, or both, and then let it go.

    A bare ``plt.show()`` is fine at a desk and useless anywhere else: from a
    batch run it blocks on a window nobody will close, and the figure it was
    going to draw is the diagnostic the run most needed to keep. This writes
    the figure where it can be looked at later instead.

    :param figure: the figure to dispose of
    :param name: the file stem to save it under
    :param show: open a window
    :param save_dir: a directory to write ``<name>.png`` into
    :return: where it was written, if it was
    """
    written = save_figure(figure, name, save_dir)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return written


def save_figure(figure, name: str,
                save_dir: Path | str | None, *, width_mm: float | None = None,
                dpi: int = 150, formats: tuple[str, ...] = ("png",)) -> Path | None:
    """
    Write a figure into a directory and leave it open.

    What :func:`finalise_figure` does without disposing of the figure, for a
    caller drawing several: ``plt.show()`` is global, so closing the earlier
    ones is what leaves a window with only the last figure in it.

    :param figure: the figure to write
    :param name: the file stem to write it under
    :param save_dir: a directory to write the requested formats into, or None
    :return: where it was written, if it was
    """
    if save_dir is None:
        return None
    written = Path(save_dir) / f"{name}.png"
    written.parent.mkdir(parents=True, exist_ok=True)
    targets = [Path(save_dir) / f"{name}.{output_format}" for output_format in formats]
    existing = [str(target) for target in targets if target.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing figure export(s): " + ", ".join(existing))
    original_size = figure.get_size_inches().copy()
    if width_mm is None:
        output_size = original_size
    else:
        width_inches = float(width_mm) / 25.4
        height_inches = width_inches * float(original_size[1]) / float(original_size[0])
        output_size = (width_inches, height_inches)
    if width_mm is not None and "png" in formats:
        pixel_size = (round(width_inches * dpi) / dpi,
                      round(height_inches * dpi) / dpi)
        output_size = pixel_size
    if width_mm is not None:
        figure.set_size_inches(*output_size, forward=False)
    try:
        for output_format, target in zip(formats, targets):
            figure.savefig(target, dpi=int(dpi), format=output_format,
                           bbox_inches=(None if width_mm is not None else "tight"))
    finally:
        figure.set_size_inches(original_size, forward=False)
    return written


def _bind_screenshot_key(plotter, save_dir: Path | str | None = None) -> None:
    """
    Bind "s" to write a timestamped screenshot of the open window.

    A saved figure is whatever view the code chose.  The reason to open a
    window at all is to orbit until something is visible, and that view is
    otherwise unrecoverable -- hence a key that keeps it.  This takes over
    pyvista's own "s" (surface representation), which is the lesser loss.

    :param plotter: the plotter about to be shown
    :param save_dir: where to write, defaulting to the working directory
    """
    def _screenshot() -> None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(save_dir) if save_dir is not None else Path.cwd()
        out = out / f"calibration_3d_{stamp}.png"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            plotter.screenshot(str(out))
            logger.info(f"Wrote a screenshot to {out}")
        except Exception as e:
            # the window stays usable: a failed screenshot is not a reason to
            # lose the view the person was trying to keep
            logger.warning(f"Could not write the screenshot: {e}")

    plotter.add_key_event("s", _screenshot)


def finalise_plotter(plotter, name: str, show: bool = True,
                     save_dir: Path | str | None = None) -> Path | None:
    """
    The same, for a pyvista plotter.

    :param plotter: the plotter to dispose of
    :param name: the file stem to save it under
    :param show: open a window
    :param save_dir: a directory to write ``<name>.png`` into
    :return: where it was written, if it was
    """
    if show:
        refuse_window_inside_qt("Showing a pyvista plotter")
    if save_dir is None:
        if show:
            _bind_screenshot_key(plotter)
            plotter.show()
        else:
            plotter.close()
        return None

    written = Path(save_dir) / f"{name}.png"
    written.parent.mkdir(parents=True, exist_ok=True)
    if not show:
        # otherwise this waits for a window to be closed before it will
        # hand over the screenshot
        plotter.off_screen = True
    else:
        _bind_screenshot_key(plotter, save_dir)
    try:
        plotter.show(screenshot=str(written))
        return written
    except Exception as e:
        # a diagnostic that cannot be rendered must not take the calibration
        # down with it, but it should say so rather than vanish
        logger.warning(f"Could not save the {name} view: {e}")
        plotter.close()
        return None


def cluster_plot(data_list, ranges = None, titles=None, alphas=None,
                 s_per=None, save=None):
    """
    Takes an input list of data, and plots it as a cluster plot.
    for clarity, it also plots the 1, 2, and 3 sigma contours of the data.

    :param data_list: the input data (can be a list of arrays, which will plot both methods)
    :param ranges: the ranges to plot the data over (can be a list of ranges)
    :param titles: the titles for each plot (can be a list of titles)
    :param alphas: the alpha values for each plot (can be a list of alphas)
    :param s_per: the percentage of points to plot (can be a list of percentages)
    :param save: the file to save the plot to.
    """

    n = len(data_list)
    if ranges is None:
        ranges = [None] * n
    if titles is None:
        titles = [None] * n
    if alphas is None:
        alphas = [None] * n
    if s_per is None:
        s_per = [1] * n

    fig, axs = plt.subplots(1, n, layout="constrained")

    r_ax = axs.ravel() if n > 1 else [axs]

    for datum, ax, rang, title, alp, s in zip(data_list, r_ax, ranges, titles,
                                           alphas, s_per
                                           ):

        # split into x,y based on ordering

        # d = datum.reshape((-1, 2))

        x, y = datum[::2], datum[1::2]
        # breakpoint()
        m_1 = np.mean((x**2 + y**2)**(1/2))
        if alp is None:
            pass
        alp = 0.01

        cov = np.cov(x,y)
        eigenvalues, _ = np.linalg.eigh(cov)
        width, height = np.sqrt(eigenvalues)
        # print(np.sqrt(eigenvalues))
        # raise ValueError
        sd = max(width, height)

        ranges = list(ax.get_ylim()) + list(ax.get_xlim())
        # ax.scatter(x, y, s=0.1, alpha=alp)
        _, _, _, img = ax.hist2d(x=x, y=y, bins=np.linspace(-3*sd, 3*sd, 100), norm=LogNorm(vmin=0.0001, vmax=1), cmap=blues_with_white, density=True, rasterized=True)
        clm = plt.colorbar(img, label="Density")
        sd = fancy_confidence_contours(x, y, ax=ax, ranges=ranges)
        ax.set_aspect('equal')

        if rang is not None:
            ax.set_xlim([-rang, rang])
            ax.set_ylim([-rang, rang])
        else:
            sf = 3
            # sd = 2
            ax.set_ylim([-sf*sd, sf*sd])
            ax.set_xlim([-sf*sd, sf*sd])

        if title is None:
            ax.set_title(f'Mean euclidean error = {m_1:.2f} '
                            f'px',
                            )
        else:
            ax.set_title(title + f'\nMean euclidean error = {m_1:.2f} '
                            f'px',
                            )
        ax.set_ylabel(r'$\it{y}$ error (px)')
        ax.set_xlabel(r'$\it{x}$ error (px)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.locator_params(nbins=5)

        # plt.show()
        # raise ValueError()

    if save is not None:
        plt.savefig(save)

    return fig


def fancy_confidence_contours(x,y, ax, ranges):
    """
    Plots the 1, 2, and 3 sigma contours of the data.

    :param x: x locations
    :param y: y locations
    :param ax: the axis object to plot too
    :param ranges: the ranges to use.
    """
    cov = np.cov(x,y)
    # var = multivariate_normal(cov=np.cov(x,y))
    xx, yy = np.meshgrid(
        np.linspace(ranges[0], ranges[1], 100),
        np.linspace(ranges[2], ranges[3], 100)
    )
    # pos = np.dstack((xx,yy))
    # res = var.pdf(pos)

    lbs = [r'$3\sigma$', r'$2\sigma$', r'$1\sigma$']

    # dist = np.sqrt(var.cov[1, 1])

    #so we have the covariance matrix of the data.

    #if we see little covariance, we need to address this in the plot


    # levels = [var.pdf([0,3*dist]), var.pdf([0,2*dist]), var.pdf([0,dist])]
    # cset = ax.contour(xx,
    #             yy,
    #             res,
    #             levels = levels,
    #             colors='firebrick')
    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Calculate the angle of the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigenvalues)

    # Create a figure and axis
    # Plot the covariance ellipse
    ellipse = Ellipse((0,0), width, height, angle=angle, edgecolor='firebrick', facecolor='none', lw=1)
    ax.add_patch(ellipse)
    ellipse = Ellipse((0,0), 2*width, 2*height, angle=angle, edgecolor='firebrick', facecolor='none', lw=1)
    ax.add_patch(ellipse)
    ellipse = Ellipse((0,0), 3*width, 3*height, angle=angle, edgecolor='firebrick', facecolor='none', lw=1)
    ax.add_patch(ellipse)

    phi = np.radians(angle)
    x_text = 0 + width/2 * np.cos(0) * np.cos(phi) - height/2 * np.sin(0) * np.sin(phi)
    y_text = 0 + width/2 * np.cos(0) * np.sin(phi) + height/2 * np.sin(0) * np.cos(phi) 

    ax.text(1.3*x_text, 1.3*y_text, r'$\sigma$', fontsize=12, color='firebrick', rotation=0, ha='center', va = 'center') 
    ax.text(2.3*x_text, 2.3*y_text, r'$2\sigma$', fontsize=12, color='firebrick', rotation=0, ha='center', va = 'center') 
    ax.text(3.3*x_text, 3.3*y_text, r'$3\sigma$', fontsize=12, color='firebrick', rotation=0, ha='center', va = 'center') 

    # locations = [(0,-1*height),(0,-2*height),(0,-3*height)]
    # fmt = {}
    # for l,s in zip(cset.levels, lbs):
    #     fmt[l] = s
    # plt.clabel(cset, inline=True, fmt=fmt, fontsize=12, manual=locations)
    return max(height, width)/2

 
#from pyCamera.optimisers.base_optimiser import AbstractParamHandler
@dataclass
class CalibrationDiagnostics:
    """
    What a calibration's diagnostic plots are drawn from, computed once.

    Every view below needs the detections triangulated and carried back into
    the target's own frame, which is the expensive part; building this once
    means a page or a report can draw the views it wants individually rather
    than paying for the triangulation per plot.
    """

    cams: 'CameraSet'
    detection: 'TargetDetection'
    residuals: np.ndarray       #: the solver's reprojection residuals, in pixels
    euclidean_err: np.ndarray   #: one Euclidean error per observation, in pixels
    e_lim: float                #: the colour limit every view shares
    scene_points: np.ndarray    #: triangulated features, in scene coordinates
    object_points: np.ndarray   #: the same features, in the target's own frame
    point_error: np.ndarray     #: the mean reprojection error of each of them
    rejected: int               #: features too far from the target to be real
    accuracy: np.ndarray        #: per feature distance from where it was drawn, mm
    precision: np.ndarray       #: per feature scatter about its own mean, mm
    feature_error: np.ndarray   #: the mean reprojection error of each feature

    @classmethod
    def from_results(cls, o_results: dict, param_handler) -> 'CalibrationDiagnostics':
        """
        :param o_results: the optimisation results, as ``x`` and ``err``
        :param param_handler: the parameter handler that organised the solve
        :return: everything the views below draw
        """
        residuals, _ = reprojection_residuals(o_results['err'], param_handler)
        euclidean_err = np.linalg.norm(residuals.reshape(-1, 2), axis=1)

        detection = param_handler.get_detection()
        cams, poses = param_handler.get_camset(o_results['x'], return_pose=True)

        to_reconstruct = detection.sort(['key', 'global_im_num']).get_data()
        reconstructed, subset, where_mask, _ = cams.multi_cam_triangulate(
            to_reconstruct, return_used=True)
        point_error = np.array(
            [np.mean(euclidean_err[datum]) for datum in where_mask])

        # one row per reconstructed point, saying which image it was seen in
        # and which target feature it is
        inv = np.sort(np.unique(subset[:, 1:-2], axis=0, return_index=True)[1])
        im_nums = subset[inv, 1]
        keys = subset[inv, 2:-2]

        mean_dist = _target_mean_distance(param_handler.target)
        kept, object_points, feature_locs, feature_errs = [], [], {}, {}
        for point, im, key, err in zip(reconstructed, im_nums, keys, point_error):
            inv_pose = np.empty(12)
            n_inv_pose(poses[int(im)], inv_pose)
            obj_point = np.empty(3)
            n_htform_prealloc(point, inv_pose, obj_point)
            # a point that lands further from the target's centre than the
            # target extends is a failed triangulation, not a measurement
            good = bool(np.linalg.norm(obj_point) < 3 * mean_dist)
            kept.append(good)
            if good:
                object_points.append(obj_point)
                feature_locs.setdefault(tuple(key.astype(int)), []).append(obj_point)
                feature_errs.setdefault(tuple(key.astype(int)), []).append(err)

        kept = np.array(kept, dtype=bool)
        accuracy, precision, feature_error = _feature_consistency(
            param_handler.target, feature_locs, feature_errs)

        return cls(
            cams=cams,
            detection=detection,
            residuals=residuals,
            euclidean_err=euclidean_err,
            e_lim=float(np.median(euclidean_err) * 3),
            scene_points=reconstructed[:len(kept)][kept],
            object_points=np.array(object_points),
            point_error=point_error[:len(kept)][kept],
            rejected=int(np.sum(~kept)),
            accuracy=accuracy,
            precision=precision,
            feature_error=feature_error,
        )


def _reject_outliers(data, m=2.):
    """The data within m median absolute deviations of its median."""
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def _feature_consistency(target, feature_locs: dict, feature_errs: dict):
    """
    How close each target feature landed to where it was drawn, and to itself.

    Accuracy is the mean distance from the drawn location and precision the
    scatter about wherever the feature actually landed, so the two separate a
    target that is misprinted -- precise but inaccurate -- from one that is
    badly observed, which is neither.  A feature seen in two images or fewer
    has no scatter worth the name and is left out.

    :param target: the calibration target, as drawn
    :param feature_locs: target frame locations, keyed by feature
    :param feature_errs: reprojection errors, keyed the same way
    :return: accuracy and precision in millimetres, and the mean error, per feature
    """
    accuracy, precision, errors = [], [], []
    for (key, locations), errs in zip(feature_locs.items(), feature_errs.values()):
        if len(locations) <= 2:
            continue
        drawn = target.original_points[(0, key[0]) if len(key) == 1 else key]
        dif = np.array(locations) - drawn
        accuracy.append(np.mean(np.linalg.norm(dif, axis=1)))
        precision.append(np.mean(_reject_outliers(
            np.linalg.norm(dif - np.mean(dif, axis=0), axis=1))))
        errors.append(np.mean(errs))
    # the library works in metres and a printed target is discussed in mm
    return (np.array(accuracy) * 1000, np.array(precision) * 1000,
            np.array(errors))


def per_camera_coverage(diagnostics: CalibrationDiagnostics) -> plt.Figure:
    """
    Where on each sensor the error falls, and which way it points.

    A feature is coloured by its Euclidean error, signed by whether the
    residual points away from the principal point or back towards it: a lens
    model that has not absorbed the distortion leaves a sensor red at one
    radius and blue at another, which unsigned error hides.

    :param diagnostics: the calibration to draw
    :return: the figure
    """
    cams, detection = diagnostics.cams, diagnostics.detection
    n_cams = cams.get_n_cams()
    windows = get_close_square_tuple(n_cams)
    fig, axes = plt.subplots(*windows[::-1], layout="constrained")
    ax = np.atleast_1d(axes).ravel()

    err_buff = copy(diagnostics.euclidean_err)
    full_err = diagnostics.residuals.reshape(-1, 2)

    im = None
    for cam_detection in detection.get_cam_list():
        datum = cam_detection.get_data()
        if datum is None:
            continue
        cam_n = int(datum[0, 0])
        p_x = cams[cam_n].intrinsic[0, 2]
        p_y = cams[cam_n].intrinsic[1, 2]
        loc_x, loc_y = datum[:, -2], datum[:, -1]
        error, err_buff = err_buff[:len(datum)], err_buff[len(datum):]
        err, full_err = full_err[:len(datum)], full_err[len(datum):]
        away = np.copysign(np.ones(datum.shape[0]),
                           (loc_x - p_x) * err[:, 0] + (loc_y - p_y) * err[:, 1])

        im = ax[cam_n].scatter(loc_x, loc_y, c=error * away, s=2, alpha=0.4,
                               vmin=-diagnostics.e_lim, vmax=diagnostics.e_lim,
                               cmap="coolwarm")
        # two lines: long rig camera names otherwise run into their neighbours
        ax[cam_n].set_title(
            f"{detection.cam_names[cam_n]}\nmean error {np.mean(error):.2f} px",
            fontsize=8)
        ax[cam_n].set_xlim([0, cams[cam_n].res[0]])
        ax[cam_n].set_ylim([0, cams[cam_n].res[1]])
        ax[cam_n].set_aspect('equal')

    if n_cams > 15:
        for axs in ax:
            axs.set_xticks([])
            axs.set_yticks([])

    for i in range(n_cams, windows[0] * windows[1]):
        fig.delaxes(ax[i])

    if im is not None:
        cbar = fig.colorbar(im, ax=list(ax[:n_cams]))
        cbar.set_label("Polarised Reprojection Error (px)")
    fig.suptitle("Per Camera Coverage")
    return fig


FRUSTUM_ACTOR_PREFIX = "camera-frustum"


def is_dark(colour) -> bool:
    """Whether *colour* (hex or 0-1 RGB) is a dark background."""
    from matplotlib.colors import to_rgb

    red, green, blue = to_rgb(colour)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5


def contrast_colours(background) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Frustum line and text colours that read against *background*.

    Black frustums vanish on a dark background, so on one they are drawn in a
    light grey, a step softer than the text so the data points stay dominant.

    :param background: the scene background, hex or 0-1 RGB
    :return: the frustum colour and the text colour, as 0-1 RGB
    """
    if is_dark(background):
        return (0.76, 0.79, 0.85), (0.91, 0.92, 0.95)
    return (0.0, 0.0, 0.0), (0.1, 0.1, 0.1)


def _apply_3d_cosmetics(plotter, theme_name: str = "Light", background: str = "theme",
                        point_size: float = 3.0, view: str = "isometric",
                        axes: bool = True) -> None:
    """Apply presentation-only PyVista options without touching scene arrays."""
    from pyCamSet.gui.theme import THEME_TOKENS

    if theme_name not in THEME_TOKENS:
        raise ValueError(f"Unknown 3D theme: {theme_name}")
    if background not in {"theme", "white", "charcoal"}:
        raise ValueError("3D background must be theme, white or charcoal")
    if not 1.0 <= float(point_size) <= 20.0:
        raise ValueError("3D point size must be between 1 and 20")
    if view not in {"isometric", "top", "front", "side"}:
        raise ValueError("3D view must be isometric, top, front or side")
    colour = (THEME_TOKENS[theme_name]["background"] if background == "theme"
              else {"white": "#ffffff", "charcoal": "#242a32"}[background])
    plotter.set_background(colour, all_renderers=True)
    frustum_colour, text_colour = contrast_colours(colour)
    for bar in getattr(plotter, "scalar_bars", {}).values():
        bar.GetTitleTextProperty().SetColor(*text_colour)
        bar.GetLabelTextProperty().SetColor(*text_colour)
    renderers = list(plotter.renderers)
    columns = max(1, int(plotter.shape[1]))
    for index in range(len(renderers)):
        plotter.subplot(index // columns, index % columns)
        if axes:
            plotter.add_axes()
        for name, actor in plotter.renderer.actors.items():
            if str(name).startswith(FRUSTUM_ACTOR_PREFIX):
                actor.GetProperty().SetColor(*frustum_colour)
                continue
            text_property = (actor.GetTextProperty() if hasattr(actor, "GetTextProperty")
                             else None)
            if text_property is not None:
                text_property.SetColor(*text_colour)
                continue
            mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else None
            dataset = mapper.GetInput() if mapper is not None else None
            if (dataset is not None and dataset.GetNumberOfCells() == 0
                    and dataset.GetNumberOfPoints() > 0):
                actor.GetProperty().SetPointSize(float(point_size))
        if view == "top":
            plotter.view_xy()
        elif view == "front":
            plotter.view_xz()
        elif view == "side":
            plotter.view_yz()
        else:
            plotter.view_isometric()
        plotter.reset_camera()


def reconstruction_scene(diagnostics: CalibrationDiagnostics, point_size: float = 3.0,
                         show_legend: bool = True, plotter=None) -> 'pv.Plotter':
    """
    The triangulated features where the cameras put them, with the cameras.

    :param diagnostics: the calibration to draw
    :param plotter: draw into this plotter (an embedded Qt view, say)
        rather than a new window
    :return: the plotter, to show or screenshot
    """
    pv.set_plot_theme('document')
    if plotter is None:
        plotter = pv.Plotter()
        plotter.title = "Reconstructed Points in Scene Coordinates"
    plotter.add_text("Reconstructed Points in Scene Coordinates",
                     position='upper_edge', font_size=10, font="times")
    diagnostics.cams.get_scene(scene=plotter, labels=False)
    if len(diagnostics.scene_points):
        points = pv.PolyData(diagnostics.scene_points)
        points['Reprojection error (px)'] = diagnostics.point_error
        plotter.add_mesh(points, render_points_as_spheres=True, point_size=point_size,
                         clim=[0, diagnostics.e_lim], show_scalar_bar=show_legend,
                         scalar_bar_args={"title": "Reprojection error (px)"})
    else:
        plotter.add_text("No points within outlier threshold",
                         position='lower_left', font_size=10, font='times')
    return plotter


def target_space_scene(diagnostics: CalibrationDiagnostics,
                       title: str = "Reconstructed Points in Target Coordinates",
                       point_size: float = 3.0, show_legend: bool = True,
                       plotter=None) -> 'pv.Plotter':
    """
    The same features carried back into the target's own frame.

    Every image's view of a feature lands on top of every other image's, so
    the size of each cluster is how consistently the target was measured, and
    the shape of the cloud is the target as the cameras believe it to be.

    :param diagnostics: the calibration to draw
    :param title: what to write across the top of it
    :param plotter: draw into this plotter rather than a new window
    :return: the plotter, to show or screenshot
    """
    pv.set_plot_theme('document')
    if plotter is None:
        plotter = pv.Plotter()
        plotter.title = title
    plotter.add_text(title, position="upper_edge", font_size=10, font='times')
    plotter.add_text(f"{diagnostics.rejected} erroneous Points",
                     position='lower_left', font_size=10, font='times')
    points = pv.PolyData(diagnostics.object_points)
    points['Reprojection Error (px)'] = diagnostics.point_error
    plotter.add_mesh(points, render_points_as_spheres=True, point_size=point_size,
                     clim=[0, diagnostics.e_lim], show_scalar_bar=show_legend,
                     scalar_bar_args={"title": "Reprojection error (px)"})
    return plotter


def accuracy_precision_plot(diagnostics: CalibrationDiagnostics,
                            title: str = "Accuracy vs precision of target features",
                            ) -> plt.Figure:
    """
    How consistently each target feature was recovered, against how far from
    its drawn location it was recovered.

    One point per feature seen in more than two images, coloured by the
    reprojection error it carries.  The diagonal is where the two are equal:
    a feature below it is reproduced more tightly than it is placed, which is
    a target printed wrong rather than one observed badly.

    :param diagnostics: the calibration to draw
    :param title: what to write above it
    :return: the figure
    """
    fig, ax = plt.subplots(layout="constrained")
    ax.set_title(title)
    ax.set_xlabel("Accuracy, mean distance from expected location (mm)")
    ax.set_ylabel("Precision, mean distance from mean location (mm)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if len(diagnostics.accuracy) == 0:
        # a feature needs to be seen more than twice to have scattered at all
        ax.text(0.5, 0.5, "n/a for single timestep images", ha='center',
                transform=ax.transAxes)
        return fig

    marks = ax.scatter(diagnostics.accuracy, diagnostics.precision, s=8,
                       c=np.clip(diagnostics.feature_error, 0, diagnostics.e_lim),
                       cmap='viridis', vmin=0, vmax=diagnostics.e_lim)
    line = np.linspace(0, np.amax(diagnostics.accuracy), 100)
    ax.plot(line, line, color='r', lw=1)
    fig.colorbar(marks, ax=ax, label="Reprojection error (px)")
    return fig


def _assessment_csv_payloads(diagnostics: CalibrationDiagnostics) -> dict[str, tuple[list[str], list[tuple]]]:
    """Return source-array rows for each assessment figure; never digitise artists."""
    residuals = np.asarray(diagnostics.residuals).reshape(-1, 2)
    euclidean = np.asarray(diagnostics.euclidean_err).reshape(-1)
    if len(residuals) != len(euclidean):
        raise ValueError("Residual and Euclidean-error arrays must have matching observation counts")
    error_rows = [(index, float(vector[0]), float(vector[1]), float(euclidean[index]))
                  for index, vector in enumerate(residuals)]

    coverage_rows = []
    residual_cursor = 0
    for cam_detection in diagnostics.detection.get_cam_list():
        datum = cam_detection.get_data()
        if datum is None:
            continue
        cam_n = int(datum[0, 0])
        camera_name = diagnostics.detection.cam_names[cam_n]
        residual_slice = residuals[residual_cursor:residual_cursor + len(datum)]
        if len(residual_slice) != len(datum):
            raise ValueError("Detection and residual arrays must have matching observation counts")
        for local_index, (source_row, vector) in enumerate(zip(datum, residual_slice)):
            magnitude = float(np.linalg.norm(vector))
            principal = diagnostics.cams[cam_n].intrinsic[:2, 2]
            direction = np.sign(np.dot(source_row[-2:] - principal, vector))
            coverage_rows.append((cam_n, camera_name, local_index, float(source_row[-2]),
                                  float(source_row[-1]), magnitude * (direction or 1)))
        residual_cursor += len(datum)
    if residual_cursor != len(residuals):
        raise ValueError("Residual arrays contain observations not represented by detection rows")

    feature_lengths = (len(diagnostics.accuracy), len(diagnostics.precision),
                       len(diagnostics.feature_error))
    if len(set(feature_lengths)) != 1:
        raise ValueError("Accuracy, precision and feature-error arrays must have matching feature counts")
    feature_count = feature_lengths[0]
    accuracy_rows = [(index, float(diagnostics.accuracy[index]),
                      float(diagnostics.precision[index]),
                      float(diagnostics.feature_error[index]))
                     for index in range(feature_count)]
    return {
        "error_distribution": (["residual_index", "x_error_px", "y_error_px", "euclidean_error_px"], error_rows),
        "per_camera_coverage": (["camera_index", "camera_name", "camera_observation_index", "x_px", "y_px", "signed_euclidean_error_px"], coverage_rows),
        "accuracy_precision": (["feature_array_index", "accuracy_mm", "precision_mm", "mean_reprojection_error_px"], accuracy_rows),
    }


def _write_assessment_csvs(
    diagnostics: CalibrationDiagnostics, save_dir: Path | str,
    provenance: str | None = None,
) -> list[Path]:
    """Write explicit source-backed CSVs beside the assessment figures."""
    import csv
    import json

    output_dir = Path(save_dir)
    payloads = _assessment_csv_payloads(diagnostics)
    units = {
        "error_distribution": {"x_error_px": "px", "y_error_px": "px", "euclidean_error_px": "px"},
        "per_camera_coverage": {"x_px": "px", "y_px": "px", "signed_euclidean_error_px": "px"},
        "accuracy_precision": {"accuracy_mm": "mm", "precision_mm": "mm", "mean_reprojection_error_px": "px"},
    }
    empty_reasons = {
        "error_distribution": "No reprojection residual observations",
        "per_camera_coverage": "No camera detection observations",
        "accuracy_precision": "No feature was observed in more than two images",
    }
    targets = [output_dir / f"{name}.csv" for name in payloads]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing assessment CSV(s): " + ", ".join(existing))
    written = []
    for (name, (columns, rows)), path in zip(payloads.items(), targets):
        with path.open("x", newline="", encoding="utf-8") as stream:
            stream.write("# " + json.dumps({
                "source": "CalibrationDiagnostics.from_results source arrays",
                "diagnostic": name, "units": units[name],
                "camset_source": provenance,
                "row_count": len(rows),
                "empty_reason": empty_reasons[name] if not rows else None,
                "index_semantics": "array order only; no feature identity inferred",
            }, ensure_ascii=False) + "\n")
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        written.append(path)
    return written


def visualise_calibration(
        o_results:dict,
        param_handler,#: AbstractParamHandler
        show: bool = True,
        save_dir: Path | str | None = None,
        theme_name: str = "Light",
        figure_width_mm: float = 160.0,
        figure_dpi: int = 150,
        figure_formats: tuple[str, ...] = ("png",),
        matplotlib_only: bool = False,
        figure_themes: tuple[str, ...] | None = None,
        provenance: str | None = None,
        export_csv: bool = False,
        three_d_background: str = "theme",
        three_d_point_size: float = 3.0,
        three_d_view: str = "isometric",
        three_d_axes: bool = True,
        three_d_legend: bool = True,
        three_d_only: bool = False,
    ) -> list[Path]:
    """
    A function to draw and plot the errors in a calibration given the results.

    With ``save_dir`` set the figures are written there as PNGs, which is what
    makes this usable from a script: with ``show=False`` as well, nothing
    blocks and the whole diagnostic set survives the run.

    :param o_results: The optimisation results
    :param param_handler: The parameter handler that organised the optimisation.
    :param show: open the figures in windows
    :param save_dir: a directory to write the figures into
    :param theme_name: application theme for Matplotlib chrome only
    :param figure_width_mm: output width for saved Matplotlib figures
    :param figure_dpi: raster DPI for PNG output
    :param figure_formats: Matplotlib output formats, e.g. PNG/SVG/PDF
    :param matplotlib_only: skip 3D scene generation for a 2D-only export request
    :param three_d_only: skip the 2D figures, for a viewer opened beside a GUI
        that already shows them
    :return: the files written, if any
    """
    if not _PYVISTA_OK and not matplotlib_only:
        raise ImportError("pyvista is required for visualisation. Install with: pip install pyvista")
    if not matplotlib_only:
        import os as _os
        if _os.name != "nt" and not _os.environ.get("DISPLAY") and not _os.environ.get("PYVISTA_OFF_SCREEN"):
            _os.environ["PYVISTA_OFF_SCREEN"] = "true"

    diagnostics = CalibrationDiagnostics.from_results(o_results, param_handler)

    if save_dir is not None:
        output_dir = Path(save_dir)
        output_names = ("error_distribution", "per_camera_coverage", "accuracy_precision")
        targets = [output_dir / f"{name}.{output_format}"
                   for name in output_names for output_format in figure_formats]
        if export_csv:
            targets.extend(output_dir / f"{name}.csv" for name in output_names)
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing assessment export(s): "
                                  + ", ".join(existing))

    written: list[Path | None] = []
    figures = [] if three_d_only else [
        (cluster_plot([diagnostics.residuals], alphas=[0.1]), "error_distribution"),
        (per_camera_coverage(diagnostics), "per_camera_coverage"),
        (accuracy_precision_plot(diagnostics), "accuracy_precision"),
    ]

    # Apply GUI chrome in this isolated process without recolouring data series.
    from pyCamSet.gui.theme import apply_matplotlib_theme
    per_figure_themes = figure_themes or (theme_name,) * len(figures)
    if three_d_only:
        per_figure_themes = ()
    if len(per_figure_themes) != len(figures):
        raise ValueError("figure_themes must contain one theme per assessment figure")
    for (figure, _), figure_theme in zip(figures, per_figure_themes):
        apply_matplotlib_theme(figure, figure_theme)
    # plt.show() is global -- one call draws every figure that is still open --
    # so they are written first and then disposed of together
    written += [save_figure(figure, name, save_dir, width_mm=figure_width_mm,
                            dpi=figure_dpi, formats=figure_formats)
                for figure, name in figures]
    if save_dir is not None and export_csv:
        written.extend(_write_assessment_csvs(diagnostics, save_dir, provenance))
    if show and figures:
        plt.show()
    else:
        for figure, _ in figures:
            plt.close(figure)

    if not matplotlib_only:
        scenes = ((lambda: reconstruction_scene(diagnostics, three_d_point_size, three_d_legend),
                   "reconstruction"),
                  (lambda: target_space_scene(diagnostics, point_size=three_d_point_size,
                                               show_legend=three_d_legend), "target_coordinates"))
        for build, name in scenes:
            plotter = build()
            _apply_3d_cosmetics(plotter, theme_name, three_d_background,
                                three_d_point_size, three_d_view, three_d_axes)
            written.append(finalise_plotter(plotter, name, show, save_dir))

    # special_plots opens and drives its own window, and its signature is part
    # of the parameter handler API that lives outside this repository, so it
    # is not asked to save anything -- it is simply skipped when nobody is
    # there to look.
    if show:
        param_handler.special_plots(o_results['x'])

    written = [path for path in written if path is not None]
    if written:
        logger.info(f"Wrote {len(written)} calibration figures to "
                    f"{Path(written[0]).parent}")
    return written



def _pv_polydata_to_o3d_lineset(pv_mesh):
    """Convert a PyVista mesh of any face shape into an Open3D wireframe LineSet.

    PyVista's flat face array prefixes each face with its vertex count, so a
    mesh of quads reads ``[4, a, b, c, d, 4, ...]`` where one of triangles
    reads ``[3, a, b, c, 3, ...]``.  Reading it as triangles works only for a
    camera drawn as a frustum: a telecentric lens images a rectangular prism,
    whose box is six quads, and the fixed stride turned that into a reshape
    error that left the whole 3-D view blank for a telecentric rig.
    """
    o3d = _open3d()
    verts = np.asarray(pv_mesh.points, dtype=np.float64)
    faces_arr = np.asarray(pv_mesh.faces)

    faces = []
    if faces_arr.ndim == 1:
        i = 0
        while i < len(faces_arr):
            n = int(faces_arr[i])
            faces.append([int(v) for v in faces_arr[i + 1:i + 1 + n]])
            i += n + 1
    else:
        faces = [[int(v) for v in row[1:]] for row in faces_arr]

    edges = set()
    for face in faces:
        for i in range(len(face)):
            edge = tuple(sorted((face[i], face[(i + 1) % len(face)])))
            edges.add(edge)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(verts)
    line_set.lines = o3d.utility.Vector2iVector(np.array(list(edges), dtype=np.int32))
    return line_set


def _build_o3d_axes_lines(scale: float):
    """Return a coloured Open3D LineSet for the world axes."""
    o3d = _open3d()
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, scale],
        ],
        dtype=np.float64,
    )
    lines = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    colours = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.4, 1.0],
        ],
        dtype=np.float64,
    )
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector(points)
    axes.lines = o3d.utility.Vector2iVector(lines)
    axes.colors = o3d.utility.Vector3dVector(colours)
    return axes


def _merge_o3d_bounds(geoms: list[object]):
    """Return one bounding box that contains every geometry in *geoms*."""
    bounds = None
    for geom in geoms:
        geom_bounds = geom.get_axis_aligned_bounding_box()
        bounds = geom_bounds if bounds is None else bounds + geom_bounds
    return bounds


def _render_open3d_geometries_offscreen(
    geoms: list[object],
    width: int,
    height: int,
    point_size: float,
):
    """Render Open3D geometries and return image plus camera projection data."""
    o3d = _open3d()
    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([0.08, 0.10, 0.16, 1.0])

        point_mat = o3d.visualization.rendering.MaterialRecord()
        point_mat.shader = "defaultUnlit"
        point_mat.point_size = point_size

        line_mat = o3d.visualization.rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 2.0

        mesh_mat = o3d.visualization.rendering.MaterialRecord()
        mesh_mat.shader = "defaultLit"

        for idx, geom in enumerate(geoms):
            if isinstance(geom, o3d.geometry.PointCloud):
                material = point_mat
            elif isinstance(geom, o3d.geometry.LineSet):
                material = line_mat
            else:
                material = mesh_mat
            renderer.scene.add_geometry(f"geom_{idx}", geom, material)

        bounds = _merge_o3d_bounds(geoms)
        renderer.setup_camera(60.0, bounds, bounds.get_center())

        image = np.asarray(renderer.render_to_image())
        camera = renderer.scene.camera
        return image, {
            'mode': 'clip',
            'view_matrix': np.asarray(camera.get_view_matrix(), dtype=np.float64),
            'projection_matrix': np.asarray(camera.get_projection_matrix(), dtype=np.float64),
            'width': width,
            'height': height,
        }
    except Exception as offscreen_exc:
        logger.warning('Open3D OffscreenRenderer unavailable; using hidden Visualizer fallback: %s', offscreen_exc)

    try:
        vis = o3d.visualization.Visualizer()
        created = vis.create_window(window_name='Open3DHiddenRender', width=width, height=height, visible=False)
        if not created:
            raise RuntimeError('Open3D hidden Visualizer window could not be created')
        try:
            render_option = vis.get_render_option()
            render_option.background_color = np.array([0.08, 0.10, 0.16], dtype=np.float64)
            render_option.point_size = float(point_size)
            render_option.line_width = 2.0
            for geom in geoms:
                vis.add_geometry(geom)
            vis.poll_events()
            vis.update_renderer()
            image = (np.clip(np.asarray(vis.capture_screen_float_buffer(do_render=True)), 0.0, 1.0) * 255).astype(np.uint8)
            params = vis.get_view_control().convert_to_pinhole_camera_parameters()
            return image, {
                'mode': 'pinhole',
                'intrinsic': np.asarray(params.intrinsic.intrinsic_matrix, dtype=np.float64),
                'extrinsic': np.asarray(params.extrinsic, dtype=np.float64),
                'width': width,
                'height': height,
            }
        finally:
            vis.destroy_window()
    except Exception as viz_exc:
        raise RuntimeError(
            "Open3D offscreen rendering failed via both OffscreenRenderer and "
            "hidden Visualizer. This typically indicates missing EGL/GPU support "
            "on headless systems. Consider installing Open3D with EGL support or "
            f"running with a display. Last error: {viz_exc}"
        ) from viz_exc


def _project_points_to_pixels(
    points: np.ndarray,
    camera_projection: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points into pixel coordinates for 2D tag overlay."""
    width = int(camera_projection['width'])
    height = int(camera_projection['height'])
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=bool)

    pts_h = np.hstack((points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)))
    if camera_projection['mode'] == 'clip':
        clip = pts_h @ camera_projection['view_matrix'].T @ camera_projection['projection_matrix'].T
        w = clip[:, 3]
        valid = np.abs(w) > 1e-9
        ndc = np.zeros((len(points), 3), dtype=np.float64)
        ndc[valid] = clip[valid, :3] / w[valid, None]
        valid &= np.isfinite(ndc).all(axis=1)
        valid &= ndc[:, 2] >= -1.05
        valid &= ndc[:, 2] <= 1.05

        pixels = np.empty((len(points), 2), dtype=np.float64)
        pixels[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * width
        pixels[:, 1] = (1.0 - ndc[:, 1]) * 0.5 * height
        return pixels, valid

    cam_xyz = pts_h @ camera_projection['extrinsic'].T
    valid = cam_xyz[:, 2] > 1e-9
    proj = cam_xyz[:, :3] @ camera_projection['intrinsic'].T
    pixels = np.empty((len(points), 2), dtype=np.float64)
    pixels[:, 0] = proj[:, 0] / np.clip(proj[:, 2], 1e-9, None)
    pixels[:, 1] = proj[:, 1] / np.clip(proj[:, 2], 1e-9, None)
    valid &= np.isfinite(pixels).all(axis=1)
    return pixels, valid


def _boxes_overlap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> bool:
    """Return True when two 2D label boxes overlap."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _overlay_camera_tags(
    image_rgb: np.ndarray,
    camera_positions: np.ndarray,
    camera_names: list[str],
    camera_projection: dict,
) -> np.ndarray:
    """Draw human-readable camera tags with leader lines over an RGB render."""
    height, width = image_rgb.shape[:2]
    pixels, valid = _project_points_to_pixels(
        camera_positions,
        camera_projection,
    )

    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(0.95, width / 1200.0))
    text_thickness = max(1, int(round(font_scale * 2.2)))
    pad = max(5, int(round(font_scale * 8)))
    used_boxes: list[tuple[int, int, int, int]] = []

    visible = [
        (name, int(round(px)), int(round(py)))
        for name, (px, py), ok in zip(camera_names, pixels, valid)
        if ok and -40 <= px <= width + 40 and -40 <= py <= height + 40
    ]
    visible.sort(key=lambda item: (item[2], item[1]))

    for idx, (name, px, py) in enumerate(visible):
        text_size, baseline = cv2.getTextSize(name, font, font_scale, text_thickness)
        text_w, text_h = text_size
        direction = 1 if px < width * 0.5 else -1
        offset_x = int((70 + 12 * (idx % 3)) * direction)
        offset_y = int(34 + 18 * (idx % 4))

        anchor_y = max(16, min(height - 16, py))
        box_y0 = max(6, min(height - text_h - (2 * pad) - 6, anchor_y - offset_y - text_h // 2 - pad))
        if direction > 0:
            box_x0 = max(6, min(width - text_w - (2 * pad) - 6, px + offset_x))
        else:
            box_x0 = max(6, min(width - text_w - (2 * pad) - 6, px + offset_x - text_w - (2 * pad)))

        candidate = (box_x0, box_y0, box_x0 + text_w + (2 * pad), box_y0 + text_h + (2 * pad))
        for _ in range(12):
            if not any(_boxes_overlap(candidate, prev) for prev in used_boxes):
                break
            shift = text_h + pad + 6
            new_y0 = candidate[1] + shift
            if new_y0 + text_h + (2 * pad) > height - 6:
                new_y0 = max(6, candidate[1] - shift)
            candidate = (candidate[0], new_y0, candidate[2], new_y0 + text_h + (2 * pad))

        used_boxes.append(candidate)
        box_x0, box_y0, box_x1, box_y1 = candidate
        text_org = (box_x0 + pad, box_y1 - pad - baseline)
        line_end_x = box_x0 if direction > 0 else box_x1
        line_end_y = box_y0 + (text_h // 2) + pad

        cv2.line(canvas, (px, anchor_y), (line_end_x, line_end_y), (0, 215, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (px, anchor_y), 4, (0, 215, 255), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (box_x0, box_y0), (box_x1, box_y1), (20, 20, 20), -1)
        cv2.rectangle(canvas, (box_x0, box_y0), (box_x1, box_y1), (245, 245, 245), 2)
        cv2.putText(canvas, name, text_org, font, font_scale, (0, 0, 0), text_thickness + 2, cv2.LINE_AA)
        cv2.putText(canvas, name, text_org, font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _overlay_panel_title(image_rgb: np.ndarray, title: str) -> np.ndarray:
    """Add a high-contrast title strip to an RGB panel image."""
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (18, 18, 24), -1)
    cv2.putText(canvas, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _build_open3d_scene_view(
    cams,
    reconstructed_points: np.ndarray,
    error_subset: np.ndarray,
    e_lim: float,
) -> tuple[list[object], np.ndarray, list[str]]:
    """Build the world-space geometry set with cameras for Open3D diagnostics."""
    o3d = _open3d()
    geoms: list[object] = []
    if reconstructed_points.size:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(reconstructed_points.astype(np.float64))
        norm = plt.Normalize(vmin=0, vmax=max(e_lim, 1e-9))
        point_cloud.colors = o3d.utility.Vector3dVector(
            plt.cm.viridis(norm(np.clip(error_subset, 0, e_lim)))[:, :3]
        )
        geoms.append(point_cloud)

    cam_positions = np.asarray([cam.position for cam in cams], dtype=np.float64)
    cam_scale = max(np.max(np.linalg.norm(cam_positions, axis=1)) * 0.1, 0.03)
    cam_meshes = cams.get_camera_meshes(viewcone=None, scale=cam_scale)
    for mesh in cam_meshes:
        line_set = _pv_polydata_to_o3d_lineset(mesh)
        line_set.paint_uniform_color([0.92, 0.92, 0.92])
        geoms.append(line_set)

    geoms.append(_build_o3d_axes_lines(max(cam_scale * 1.4, 0.05)))
    return geoms, cam_positions, list(cams._cam_dict.keys())


def _build_open3d_target_view(
    raw_obj_points: list[np.ndarray],
    errors: list[float],
    e_lim: float,
    mean_dist: float,
) -> list[object]:
    """Build the target/object-space geometry set for Open3D diagnostics."""
    o3d = _open3d()
    geoms: list[object] = []
    if raw_obj_points:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.asarray(raw_obj_points, dtype=np.float64))
        norm = plt.Normalize(vmin=0, vmax=max(e_lim, 1e-9))
        point_cloud.colors = o3d.utility.Vector3dVector(
            plt.cm.viridis(norm(np.clip(errors, 0, e_lim)))[:, :3]
        )
        geoms.append(point_cloud)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=max(mean_dist * 0.5, 0.03), origin=[0.0, 0.0, 0.0]
    )
    geoms.append(coord_frame)
    return geoms


def visualise_calibration_open3d(
    o_results: dict,
    param_handler,
    output_widget=None,
) -> tuple[bool, str]:
    """Open3D equivalent of :func:`visualise_calibration`.

    Renders a two-panel Open3D assessment: world-space reconstructed points with
    camera meshes and readable camera-name tags, plus target-space reconstructed
    points for object-shape assessment.
    """
    o3d = _open3d()
    if not _OPEN3D_OK or o3d is None:
        return False, (
            "Open3D is not installed. Install it with:\n"
            "    pip install open3d\n"
            "and restart the application."
        )

    try:
        euclidean_err = np.linalg.norm(np.reshape(o_results['err'], (-1, 2)), axis=1)
        e_lim = np.median(euclidean_err) * 3

        detection = param_handler.get_detection()
        cams, poses = param_handler.get_camset(o_results['x'], return_pose=True)

        to_reconstruct = detection.sort(['key', 'global_im_num']).get_data()
        reconstructed, reconstructed_subset, where_mask, _ = cams.multi_cam_triangulate(
            to_reconstruct, return_used=True
        )
        error_subset = np.array([np.mean(euclidean_err[datum]) for datum in where_mask])

        raw_obj_points: list[np.ndarray] = []
        errors: list[float] = []
        mean_dist = _target_mean_distance(param_handler.target)

        if reconstructed.size == 0 or reconstructed_subset.size == 0:
            # No reconstructed points — render an empty scene rather than
            # crashing on the indexing below.
            bad_points = 0
            im_nums = np.array([], dtype=int)
        else:
            inv = np.sort(np.unique(reconstructed_subset[:, 1:-2], axis=0, return_index=True)[1])
            im_nums = reconstructed_subset[inv, 1]

            bad_points = 0
            for point, im, c in zip(reconstructed, im_nums, error_subset):
                inv_pose = np.empty(12)
                n_inv_pose(poses[int(im)], inv_pose)
                obj_point = np.empty(3)
                n_htform_prealloc(point, inv_pose, obj_point)
                if np.linalg.norm(obj_point) < 3 * mean_dist:
                    raw_obj_points.append(obj_point.copy())
                    errors.append(float(c))
                else:
                    bad_points += 1

        scene_geoms, camera_positions, camera_names = _build_open3d_scene_view(
            cams,
            reconstructed,
            error_subset,
            e_lim,
        )
        target_geoms = _build_open3d_target_view(raw_obj_points, errors, e_lim, mean_dist)

        if output_widget is not None:
            try:
                panel_width = 760
                panel_height = 640
                scene_img, scene_projection = _render_open3d_geometries_offscreen(
                    scene_geoms,
                    panel_width,
                    panel_height,
                    point_size=5.0,
                )
                scene_img = _overlay_panel_title(scene_img, "Scene coordinates — cameras + labels")
                scene_img = _overlay_camera_tags(
                    scene_img,
                    camera_positions,
                    camera_names,
                    scene_projection,
                )

                target_img, _ = _render_open3d_geometries_offscreen(
                    target_geoms,
                    panel_width,
                    panel_height,
                    point_size=5.0,
                )
                target_img = _overlay_panel_title(target_img, "Target coordinates")

                img_np = np.concatenate((scene_img, target_img), axis=1)
                from PySide6.QtCore import Qt
                from PySide6.QtGui import QImage, QPixmap
                h, w, ch = img_np.shape
                qt_img = QImage(img_np.data, w, h, ch * w, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_img)
                output_widget.setPixmap(
                    pixmap.scaled(
                        output_widget.width() or w,
                        output_widget.height() or h,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
                return True, (
                    f"Open3D visualisation rendered (offscreen) with camera tags. "
                    f"{bad_points} erroneous points filtered."
                )
            except Exception as egl_exc:
                egl_msg = str(egl_exc)
                logger.warning(
                    "Open3D offscreen/EGL render failed (%s); retrying with windowed fallback.",
                    egl_msg,
                )
                output_widget.setText(
                    f"Open3D offscreen render failed:\n{egl_msg}\n\n"
                    "Falling back to interactive Open3D window.\n"
                    "Embedded camera tags are only available in the offscreen render path."
                )
                try:
                    o3d.visualization.draw_geometries(
                        scene_geoms,
                        window_name="Calibration Assessment — Scene Coordinates (Open3D)",
                        width=1100,
                        height=760,
                    )
                    # Also show the target-coordinates view, matching both the
                    # offscreen-embed path above (which combines both views into one
                    # image) and the output_widget=None path below (which also opens
                    # both windows) -- this branch previously returned right after the
                    # scene view alone, silently dropping the target view whenever it
                    # was reached (latent bug; unreachable today since every current
                    # caller passes output_widget=None, but fixed here to remove the
                    # inconsistency for any future caller that does pass a widget).
                    if target_geoms:
                        o3d.visualization.draw_geometries(
                            target_geoms,
                            window_name="Calibration Assessment — Target Coordinates (Open3D)",
                            width=800,
                            height=640,
                        )
                    return True, (
                        f"Open3D visualisation shown (windowed fallback; embedded tags unavailable). "
                        f"{bad_points} erroneous points filtered."
                    )
                except Exception as win_exc:
                    msg = f"Open3D windowed render also failed: {win_exc}"
                    output_widget.setText(msg)
                    return False, msg

        # Build 3D text labels for each camera at its position.
        # Open3D TriangleMesh.create_text produces 3D extruded text — scale it
        # relative to the camera distance so labels are legible.
        _cam_dist = (
            np.max(np.linalg.norm(camera_positions, axis=1)) * 0.1
            if len(camera_positions) else 0.03
        )
        label_scale = max(_cam_dist * 0.25, 0.008)
        all_geoms = list(scene_geoms)
        for pos, name in zip(camera_positions, camera_names):
            try:
                text_mesh = o3d.geometry.TriangleMesh.create_text(name, depth=label_scale, extra=None)
                # create_text returns text centred at origin; translate to camera position
                # and offset slightly above so it does not overlap the wireframe.
                text_mesh.translate(pos + np.array([0, label_scale * 2, 0], dtype=np.float64))
                text_mesh.paint_uniform_color([1.0, 0.85, 0.2])  # warm yellow for readability
                all_geoms.append(text_mesh)
            except Exception:
                # If create_text is unavailable or fails for any camera, skip silently
                # rather than blocking the entire visualisation.
                pass

        # Launch the interactive Open3D window with scene + camera labels.
        o3d.visualization.draw_geometries(
            all_geoms,
            window_name="Calibration Assessment — Scene Coordinates (Open3D)",
            width=1100,
            height=760,
        )

        # Also show the target-coordinates view in a second window.
        if target_geoms:
            o3d.visualization.draw_geometries(
                target_geoms,
                window_name="Calibration Assessment — Target Coordinates (Open3D)",
                width=800,
                height=640,
            )

        return True, f"Open3D visualisation shown with camera tags. {bad_points} erroneous points filtered."

    except Exception as exc:
        return False, f"Open3D visualisation error: {exc}"


def render_calibration_pyvista_png(
    o_results: dict,
    param_handler,
    output_path: str,
    width_mm: float = 160.0,
    dpi: int = 150,
    theme_name: str = "Light",
    background: str = "theme",
    point_size: float = 3.0,
    view: str = "isometric",
    axes: bool = True,
    show_legend: bool = True,
) -> tuple[bool, str]:
    """Render calibration assessment offscreen with PyVista and save to *output_path*.

    :param o_results: The optimisation results dict with keys ``err`` and ``x``.
    :param param_handler: The parameter handler used in the optimisation.
    :param output_path: Destination file path for the PNG screenshot.
    :param width_mm: Generic publication preset width; preserves the 8:3 view ratio.
    :param dpi: Render density used to calculate the PNG pixel dimensions.
    :returns: ``(success, message)`` tuple.
    """
    if not _PYVISTA_OK:
        return False, (
            "pyvista is required for offscreen PNG rendering. "
            "Install with: pip install pyvista"
        )
    try:
        euclidean_err = np.linalg.norm(np.reshape(o_results['err'], (-1, 2)), axis=1)
        e_lim = np.median(euclidean_err) * 3

        detection = param_handler.get_detection()
        cams, poses = param_handler.get_camset(o_results['x'], return_pose=True)

        to_reconstruct = detection.sort(['key', 'global_im_num']).get_data()
        reconstructed, reconstructed_subset, where_mask, _ = cams.multi_cam_triangulate(
            to_reconstruct, return_used=True
        )
        error_subset = np.array([np.mean(euclidean_err[datum]) for datum in where_mask])

        mean_dist = _target_mean_distance(param_handler.target)
        inv = np.sort(np.unique(reconstructed_subset[:, 1:-2], axis=0, return_index=True)[1])
        im_nums = reconstructed_subset[inv, 1]
        keys = reconstructed_subset[inv, 2:-2]

        raw_obj_points: list[np.ndarray] = []
        errors_scene: list[float] = []
        mask = []
        for point, im, c in zip(reconstructed, im_nums, error_subset):
            inv_pose = np.empty(12)
            n_inv_pose(poses[int(im)], inv_pose)
            obj_point = np.empty(3)
            n_htform_prealloc(point, inv_pose, obj_point)
            good = bool(np.linalg.norm(obj_point) < 3 * mean_dist)
            mask.append(good)
            if good:
                raw_obj_points.append(obj_point)
                errors_scene.append(float(c))

        m = np.array(mask)

        pv.set_plot_theme('document')
        pv.global_theme.multi_rendering_splitting_position = 0.50
        pixel_width, pixel_height = _publication_render_size(width_mm, dpi)
        try:
            plotter = pv.Plotter(shape='1|2', off_screen=True,
                                 window_size=(pixel_width, pixel_height))
        except Exception as plotter_exc:
            return False, (
                "PyVista offscreen plotter could not be created. "
                "On headless Linux this usually means EGL/GPU support is missing. "
                f"Original error: {plotter_exc}"
            )
        plotter.title = "Calibration Evaluation (Offscreen)"

        # Subplot 0: scene coordinates
        plotter.subplot(0)
        plotter.add_text("Reconstructed Points in Scene Coordinates",
                         position='upper_edge', font_size=10, font="times")
        cams.get_scene(scene=plotter, labels=False)
        if np.any(m):
            seen_pts = pv.PolyData(reconstructed[m])
            seen_pts['Reprojection error (px)'] = error_subset[m]
            plotter.add_mesh(seen_pts, render_points_as_spheres=True, point_size=point_size,
                             clim=[0, e_lim], show_scalar_bar=show_legend,
                             scalar_bar_args={"title": "Reprojection error (px)"})

        # Subplot 1: target coordinates
        plotter.subplot(1)
        plotter.add_text("Reconstructed Points in Target Coordinates",
                         position="upper_edge", font_size=10, font='times')
        bad_points = int(np.sum(~m))
        plotter.add_text(f"{bad_points} erroneous Points",
                         position='lower_left', font_size=10, font='times')
        if raw_obj_points:
            cube_locs = pv.PolyData(np.array(raw_obj_points))
            cube_locs['Reprojection Error (px)'] = errors_scene
            plotter.add_mesh(cube_locs, render_points_as_spheres=True, point_size=point_size,
                             clim=[0, e_lim], show_scalar_bar=show_legend,
                             scalar_bar_args={"title": "Reprojection error (px)"})

        _apply_3d_cosmetics(plotter, theme_name, background, point_size, view, axes)
        save_pyvista_screenshot(plotter, output_path, width_mm, dpi)
        plotter.close()
        return True, f"PyVista screenshot saved to {output_path}"

    except Exception as exc:
        return False, f"PyVista offscreen render failed: {exc}"


def export_calibration_3d(
    o_results: dict,
    param_handler,
    output_path: str | Path,
    provenance: str | None = None,
) -> tuple[bool, str]:
    """Export a reusable 3D scene (GLTF/OBJ) or target-frame point cloud (PLY).

    GLTF retains scene geometry and material colours but not the interactive
    camera/view state, scalar arrays, physical units, or text annotations.
    OBJ retains geometry/material colours through its MTL companion, but loses
    scalar arrays, units, view state, and annotation. PLY contains target-frame
    point coordinates; this VTK writer drops point arrays, which are copied
    into the metadata sidecar. PLY contains no camera meshes, view, or
    annotations.
    A JSON sidecar records the source/frame and these known losses.
    """
    if not _PYVISTA_OK:
        return False, "PyVista is required for 3D export; select the PyVista backend."
    path = Path(output_path)
    fmt = path.suffix.lower()
    if fmt not in {".gltf", ".obj", ".ply"}:
        return False, "Choose GLTF, OBJ, or PLY; GLB is not emitted by this PyVista exporter."
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    companion_mtl = path.with_suffix(".mtl") if fmt == ".obj" else None
    existing = next((candidate for candidate in (path, metadata_path, companion_mtl)
                     if candidate is not None and candidate.exists()), None)
    if existing is not None:
        return False, f"Refusing to overwrite an existing export: {existing}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics = CalibrationDiagnostics.from_results(o_results, param_handler)
        if fmt == ".ply":
            if len(diagnostics.object_points) == 0:
                return False, "No target-frame points are available to export."
            cloud = pv.PolyData(np.asarray(diagnostics.object_points, dtype=float))
            cloud["Reprojection error (px)"] = np.asarray(diagnostics.point_error, dtype=float)
            cloud.save(path)
            frame = "target coordinates"
            losses = ["camera geometry", "view state", "annotations",
                      "PLY writer drops point arrays; reprojection errors are copied to this sidecar"]
        else:
            scene = reconstruction_scene(diagnostics)
            try:
                if fmt == ".gltf":
                    scene.export_gltf(str(path), rotate_scene=False)
                else:
                    scene.export_obj(str(path))
            finally:
                scene.close()
            frame = "scene coordinates"
            losses = (["interactive camera/view state", "scalar arrays", "physical units", "text annotations"]
                      if fmt == ".gltf" else
                      ["scalar arrays", "physical units", "view state", "text annotations"])
        metadata = {
            "source": "CalibrationDiagnostics.from_results",
            "source_camset": provenance,
            "coordinate_frame": frame,
            "units": "not declared by the calibration artifact",
            "format": fmt.lstrip("."),
            "format_limitations": losses,
            "view_state_included": False,
        }
        if fmt == ".ply":
            metadata["point_data_sidecar"] = {
                "Reprojection error (px)": np.asarray(diagnostics.point_error, dtype=float).tolist(),
                "units": {"Reprojection error (px)": "px"},
                "row_count": len(diagnostics.point_error),
                "note": "PLY writer round-trip drops this point array; values correspond by point row order.",
            }
        with metadata_path.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return True, f"Exported {path} and metadata sidecar {metadata_path}."
    except FileExistsError as exc:
        return False, f"Refusing to overwrite an existing export: {exc}"
    except Exception as exc:
        return False, f"3D export failed: {exc}"
