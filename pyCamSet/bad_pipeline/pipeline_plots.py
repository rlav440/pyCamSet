"""
Purpose: Plot-saving helpers for the pyCamSet phased calibration bad_pipeline.
         Re-exports the headless plotting functions from pyCamSet.utils.visualisation
         and exposes title_to_filename / save_csv from pipeline_cache so that
         callers can use a single import for all bad_pipeline plot utilities.
Status:  Working
Future:  Add multi-page PDF export and resolution (DPI) configuration.
"""

# File-naming and CSV helpers from the bad_pipeline cache module
from pyCamSet.bad_pipeline.pipeline_cache import title_to_filename, save_csv  # re-exported for callers

# All plotting functions live in pyCamSet.utils.visualisation (implemented there).
# Re-export them here so that bad_pipeline code can do:
#   from pyCamSet.bad_pipeline.pipeline_plots import plot_error_histogram, ...
from pyCamSet.utils.visualisation import (                    # headless plot helpers
    save_figure,                                              # save a matplotlib Figure to .png
    save_numeric_summary,                                     # save a numeric dict to .csv
    plot_error_histogram,                                     # histogram of reprojection errors
    plot_per_camera_errors,                                   # bar chart of per-camera mean errors
    plot_residual_clusters,                                   # 2-D cluster plot with covariance ellipses
    plot_coverage_scatter,                                    # per-camera image-plane coverage scatter
    plot_camera_arrangement,                                  # pyvista screenshot of camera 3-D layout
)

__all__ = [                                                   # explicit public API surface
    "title_to_filename",
    "save_csv",
    "save_figure",
    "save_numeric_summary",
    "plot_error_histogram",
    "plot_per_camera_errors",
    "plot_residual_clusters",
    "plot_coverage_scatter",
    "plot_camera_arrangement",
]
