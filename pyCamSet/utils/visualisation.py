from __future__ import annotations
from math import copysign
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import numpy as np
import pyvista as pv
from matplotlib.colors import LogNorm, LinearSegmentedColormap

from pyCamSet.utils.general_utils import h_tform, get_close_square_tuple
from pyCamSet.optimisation.compiled_helpers import n_htform_prealloc, n_inv_pose

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
        ax.set_ylabel('$\it{y}$ error (px)')
        ax.set_xlabel('$\it{x}$ error (px)')
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
    print(np.std(euclidean_err))
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
    to_reconstruct = detection.sort(['key', 'im_num']).get_data()
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
    mean_dist = np.mean(np.linalg.norm(param_handler.target.point_data, axis=-1))
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

    m = np.array(mask)
    seen_pts = pv.PolyData(reconstructed[m])
    seen_pts['Reprojection error (px)'] = error_subset[m]
    plotter.add_mesh(seen_pts, render_points_as_spheres=True, point_size=2, clim=[0, e_lim])

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
        _ = chart.line(line, line, 'r')
        plotter.add_chart(chart)

    else:

        plotter.add_text("n/a for single timestep images", position='upper_edge', font='times')
    plotter.show()
    param_handler.special_plots(o_results['x'])



