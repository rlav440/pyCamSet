from __future__ import annotations
import datetime
import logging
from math import copysign
from copy import copy
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import numpy as np
import pyvista as pv
from matplotlib.colors import LogNorm, LinearSegmentedColormap

from pyCamSet.utils.general_utils import h_tform, get_close_square_tuple
from pyCamSet.optimisation.compiled_helpers import n_htform_prealloc, n_inv_pose

# Optional Open3D import — deferred to avoid import-time failures when unavailable.
try:
    import open3d as _o3d
    _OPEN3D_OK = True
except ImportError:  # pragma: no cover
    _o3d = None
    _OPEN3D_OK = False

_LOGGER = logging.getLogger(__name__)

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
        from pyCamSet.calibration_targets.target_puzzleboard import _CODE_SIZE
        # Gather only the printed-window point coordinates from point_data.
        pts = []
        for row in range(start_y, start_y + nsy):
            for col in range(start_x, start_x + nsx):
                pid = row * _CODE_SIZE + col
                pts.append(target.point_data[0, pid])
        pts = np.array(pts, dtype=float)
        return float(np.mean(np.linalg.norm(pts, axis=-1)))
    return float(np.mean(np.linalg.norm(target.point_data, axis=-1)))

blues_with_white = LinearSegmentedColormap.from_list('Blues_with_white', [(1, 1, 1), *plt.cm.Blues(np.linspace(0, 1, 1024)[:900])])


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

    fig, axs = plt.subplots(1,n,)

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
def visualise_calibration(
        o_results:dict,
        param_handler#: AbstractParamHandler
    ):
    """
    A function to draw and plot the errors in a calibration given the results.

    :param o_results: The optimisation results
    :param param_handler: The parameter handler that organised the optimisation.
    :return:
    """
    euclidean_err = np.linalg.norm(np.reshape(o_results['err'], (-1,2)), axis=1)
    e_lim = np.median(euclidean_err) * 3
    # print("Calibration Standard Deviation of euclidean Error:", np.std(euclidean_err))
    # raise ValueError

    detection = param_handler.get_detection()
    cams, poses = param_handler.get_camset(o_results['x'], return_pose=True)

    cluster_plot([o_results['err']], alphas=[0.1])

    # the coverage for each camera
    n_cams = cams.get_n_cams()
    windows = get_close_square_tuple(n_cams)
    fig, axes = plt.subplots(*windows[::-1])
    ax = axes.ravel()
    err_buff = copy(euclidean_err)
    full_err = copy(o_results['err'].reshape((-1,2)))

    if param_handler.missing_poses is not None:
        icam_n = np.cumsum(~param_handler.missing_poses) - 1


    for idc_cam, cam_detection in enumerate(detection.get_cam_list()):
        datum = cam_detection.get_data()
        if datum is not None:
            cam_n = int(datum[0,0])

            p_x = cams[cam_n].intrinsic[0,2]
            p_y = cams[cam_n].intrinsic[1,2]

            loc_x, loc_y = datum[:,-2], datum[:, -1]
            error, err_buff = err_buff[:len(datum)], err_buff[len(datum):]
            m_error = np.mean(error)
            err, full_err = full_err[:len(datum)], full_err[len(datum):]
            #what we can do is calculate if the error is going away or towards the principle axis
            away_vec = np.copysign(np.ones(datum.shape[0]), (loc_x - p_x) * err[:, 0] + (loc_y - p_y) * err[:, 1])

            
            im = ax[cam_n].scatter(loc_x, loc_y, c=error*away_vec, vmin=-e_lim, vmax=e_lim, s=2, alpha=0.4, cmap="coolwarm")
            ax[cam_n].set_title(detection.cam_names[cam_n] + f" mean error {m_error:.2f}", fontsize=8)
            ax[cam_n].set_xlim([0, cams[cam_n].res[0]])
            ax[cam_n].set_ylim([0, cams[cam_n].res[1]])
            ax[cam_n].set_aspect('equal')

    if n_cams > 15:
        for axs in ax:
            axs.set_xticks([])
            axs.set_yticks([])

    for i in range(n_cams, windows[0]*windows[1]):
        fig.delaxes(ax[i])

    cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    cbar.set_label("Polarised Reprojection Error (px)")
    fig.suptitle("Per Camera Coverage")
    plt.show()

    #err_buff = copy.copy(euclidean_err)
    to_reconstruct = detection.sort(['key', 'global_im_num']).get_data()
    ## Triangulation of points in world space
    reconstructed, reconstructed_subset,  where_mask, _ = cams.multi_cam_triangulate(to_reconstruct, return_used=True)
    error_subset = np.array([np.mean(euclidean_err[datum]) for datum in where_mask])
    # at the same time
    pv.set_plot_theme('document')
    pv.global_theme.multi_rendering_splitting_position = 0.50
    plotter = pv.Plotter(shape='1|2')
    plotter.title = "Calibration Evaluation"
    plotter.subplot(0)
    plotter.add_text("Reconstructed Points in Scene Coordinates", position='upper_edge', font_size=10, font="times")
    cams.get_scene(scene=plotter, labels=False)
    ## Triangulation of points in target space
    inv = np.sort(np.unique(reconstructed_subset[:, 1:-2], axis=0, return_index=True,)[1])
    im_nums = reconstructed_subset[inv, 1]
    keys = reconstructed_subset[inv, 2:-2]
    #point_errors = error_subset[inv]
    mask = []
    point_locs = {}
    col_locs = {}
    raw_obj_points  = []
    errors = []
    mean_dist = _target_mean_distance(param_handler.target)
    bad_points = 0
    for point, im, key, c in zip(reconstructed, im_nums, keys, error_subset):
        inv_pose = np.empty(12)
        n_inv_pose(poses[int(im)], inv_pose)
        obj_point = np.empty(3)
        n_htform_prealloc(point, inv_pose, obj_point)
        mask.append(np.linalg.norm(obj_point) < 3 * mean_dist)
        if np.linalg.norm(obj_point) > 3 * mean_dist:
            bad_points = bad_points + 1
        else:
            # # get the error of the point?
            raw_obj_points.append(obj_point)
            point_locs.setdefault(tuple(key.astype(int)), []).append(obj_point)
            col_locs.setdefault(tuple(key.astype(int)), []).append(c)
            errors.append(c)

    m = np.array(mask, dtype=bool)
    if np.any(m):
        seen_pts = pv.PolyData(reconstructed[m])
        seen_pts['Reprojection error (px)'] = error_subset[m]
        plotter.add_mesh(seen_pts, render_points_as_spheres=True, point_size=2, clim=[0, e_lim])
    else:
        plotter.add_text(
            "No points within outlier threshold",
            position='upper_edge', font_size=10, font='times'
        )

    plotter.subplot(1)
    plotter.add_text("Reconstructed Points in Target Coordinates", position="upper_edge", font_size=10, font='times')
    plotter.add_text(f"{bad_points} erroneous Points", position='lower_left', font_size=10, font='times')

    cube_locs = pv.PolyData(np.array(raw_obj_points))
    cube_locs['Reprojection Error (px)'] = errors
    plotter.add_mesh(cube_locs, render_points_as_spheres=True, point_size=4, clim=[0, e_lim])

    def reject_outliers(data, m=2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / mdev if mdev else 0.
        return data[s < m]

    # precision v. accuracy in the recovered object shape
    plotter.subplot(2)
    raw_data = []
    err_buff = []
    for (key, point_loc), err in zip(point_locs.items(), col_locs.values()):
        if len(point_loc) > 2:
            if len(key) == 1:
                key = (0, key[0])
            obj_point = param_handler.target.original_points[key]
            data_array = np.array(point_loc)
            dif = data_array - obj_point
            mean_err = np.mean(np.linalg.norm(dif, axis=1))
            obj_scatter = np.mean(reject_outliers(np.linalg.norm(dif - np.mean(dif, axis=0), axis=1)))
            raw_data.append([mean_err, obj_scatter])
            err_buff.append(np.mean(err))
    raw_data = np.array(raw_data)
    err_buff = np.array(err_buff)

    if len(raw_data) > 0:
        norm = plt.Normalize()
        colours = (plt.cm.viridis(norm(np.clip(err_buff, 0, e_lim)))[:,:3] * 255).astype(np.uint8)

        chart = pv.Chart2D()
        chart.title = 'Accuracy vs Precision of target feature locations'
        chart.y_label = 'Precision, mean distance from mean feature location (mm)'
        chart.x_label = 'Accuracy, mean distance from expected location (mm)'
        for r0, r1, c in zip(raw_data[:,0], raw_data[:,1], colours):
            _ = chart.scatter([r0 * 1000], [r1 * 1000], color=c, size=4)
        line = np.linspace(0, np.amax(raw_data[:,0]) * 1000, 100)
        _ = chart.line(line, line, color='r')
        plotter.add_chart(chart)

    else:

        plotter.add_text("n/a for single timestep images", position='upper_edge', font='times')

    def _screenshot_callback() -> None:
        fname = datetime.datetime.now().strftime("calibration_3d_%Y%m%d_%H%M%S.png")
        plotter.screenshot(fname)
        print(f"Screenshot saved: {fname}")

    plotter.add_key_event("s", _screenshot_callback)
    plotter.show()
    param_handler.special_plots(o_results['x'])


def _pv_polydata_to_o3d_lineset(pv_mesh):
    """Convert a PyVista triangle mesh into an Open3D wireframe LineSet."""
    verts = np.asarray(pv_mesh.points, dtype=np.float64)
    faces_arr = np.asarray(pv_mesh.faces)
    if faces_arr.ndim == 1:
        n_tri = len(faces_arr) // 4
        triangles = faces_arr.reshape(n_tri, 4)[:, 1:]
    else:
        triangles = faces_arr[:, 1:]

    edges = set()
    for tri in triangles:
        for i in range(3):
            edge = tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
            edges.add(edge)

    line_set = _o3d.geometry.LineSet()
    line_set.points = _o3d.utility.Vector3dVector(verts)
    line_set.lines = _o3d.utility.Vector2iVector(np.array(list(edges), dtype=np.int32))
    return line_set


def _build_o3d_axes_lines(scale: float):
    """Return a coloured Open3D LineSet for the world axes."""
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
    axes = _o3d.geometry.LineSet()
    axes.points = _o3d.utility.Vector3dVector(points)
    axes.lines = _o3d.utility.Vector2iVector(lines)
    axes.colors = _o3d.utility.Vector3dVector(colours)
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
    try:
        renderer = _o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([0.08, 0.10, 0.16, 1.0])

        point_mat = _o3d.visualization.rendering.MaterialRecord()
        point_mat.shader = "defaultUnlit"
        point_mat.point_size = point_size

        line_mat = _o3d.visualization.rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 2.0

        mesh_mat = _o3d.visualization.rendering.MaterialRecord()
        mesh_mat.shader = "defaultLit"

        for idx, geom in enumerate(geoms):
            if isinstance(geom, _o3d.geometry.PointCloud):
                material = point_mat
            elif isinstance(geom, _o3d.geometry.LineSet):
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
        _LOGGER.warning('Open3D OffscreenRenderer unavailable; using hidden Visualizer fallback: %s', offscreen_exc)

    vis = _o3d.visualization.Visualizer()
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
    geoms: list[object] = []
    if reconstructed_points.size:
        point_cloud = _o3d.geometry.PointCloud()
        point_cloud.points = _o3d.utility.Vector3dVector(reconstructed_points.astype(np.float64))
        norm = plt.Normalize(vmin=0, vmax=max(e_lim, 1e-9))
        point_cloud.colors = _o3d.utility.Vector3dVector(
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
    geoms: list[object] = []
    if raw_obj_points:
        point_cloud = _o3d.geometry.PointCloud()
        point_cloud.points = _o3d.utility.Vector3dVector(np.asarray(raw_obj_points, dtype=np.float64))
        norm = plt.Normalize(vmin=0, vmax=max(e_lim, 1e-9))
        point_cloud.colors = _o3d.utility.Vector3dVector(
            plt.cm.viridis(norm(np.clip(errors, 0, e_lim)))[:, :3]
        )
        geoms.append(point_cloud)

    coord_frame = _o3d.geometry.TriangleMesh.create_coordinate_frame(
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
    if not _OPEN3D_OK or _o3d is None:
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
                _LOGGER.warning(
                    "Open3D offscreen/EGL render failed (%s); retrying with windowed fallback.",
                    egl_msg,
                )
                output_widget.setText(
                    f"Open3D offscreen render failed:\n{egl_msg}\n\n"
                    "Falling back to interactive Open3D window.\n"
                    "Embedded camera tags are only available in the offscreen render path."
                )
                try:
                    _o3d.visualization.draw_geometries(
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
                        _o3d.visualization.draw_geometries(
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
                text_mesh = _o3d.geometry.TriangleMesh.create_text(name, depth=label_scale, extra=None)
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
        _o3d.visualization.draw_geometries(
            all_geoms,
            window_name="Calibration Assessment — Scene Coordinates (Open3D)",
            width=1100,
            height=760,
        )

        # Also show the target-coordinates view in a second window.
        if target_geoms:
            _o3d.visualization.draw_geometries(
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
) -> tuple[bool, str]:
    """Render calibration assessment offscreen with PyVista and save to *output_path*.

    :param o_results: The optimisation results dict with keys ``err`` and ``x``.
    :param param_handler: The parameter handler used in the optimisation.
    :param output_path: Destination file path for the PNG screenshot.
    :returns: ``(success, message)`` tuple.
    """
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
        plotter = pv.Plotter(shape='1|2', off_screen=True, window_size=(1600, 600))
        plotter.title = "Calibration Evaluation (Offscreen)"

        # Subplot 0: scene coordinates
        plotter.subplot(0)
        plotter.add_text("Reconstructed Points in Scene Coordinates",
                         position='upper_edge', font_size=10, font="times")
        cams.get_scene(scene=plotter, labels=False)
        if np.any(m):
            seen_pts = pv.PolyData(reconstructed[m])
            seen_pts['Reprojection error (px)'] = error_subset[m]
            plotter.add_mesh(seen_pts, render_points_as_spheres=True, point_size=2, clim=[0, e_lim])

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
            plotter.add_mesh(cube_locs, render_points_as_spheres=True, point_size=4, clim=[0, e_lim])

        plotter.screenshot(output_path)
        plotter.close()
        return True, f"PyVista screenshot saved to {output_path}"

    except Exception as exc:
        return False, f"PyVista offscreen render failed: {exc}"

