"""
Purpose: Public API for the pyCamSet phased calibration bad_pipeline.
         Re-exports the six phase functions, the run_pipeline() orchestrator,
         file-caching helpers, and plot helpers so that callers can do:
             from pyCamSet.bad_pipeline import run_pipeline
         Plot helpers live in pyCamSet.bad_pipeline.pipeline_plots and are
         re-exported here for convenience.
Status:  Working
Future:  Add CLI entry-point and progress-callback support.
"""

# Re-export the six phase functions and the top-level orchestrator.
from pyCamSet.bad_pipeline.phased_pipeline import (   # all six phases plus orchestrator
    run_phase1_detection,                          # Phase 1 — target construction and detection
    run_phase2_culling,                            # Phase 2 — image culling by detection count
    run_phase3_calibration,                        # Phase 3 — two-stage multi-camera calibration
    run_phase4_analysis,                           # Phase 4 — reprojection error analysis
    run_phase5_visualisation,                      # Phase 5 — calibration visualisation
    run_phase6_self_calibration,                   # Phase 6 — self (feature-free) bundle adjustment
    run_pipeline,                                  # Full bad_pipeline orchestrator (phases 1–6)
)

# Re-export caching helpers for users who want direct access.
from pyCamSet.bad_pipeline.pipeline_cache import (    # file I/O helpers
    save_json,                                     # write a dict to .json
    load_json,                                     # read a dict from .json
    save_csv,                                      # write rows to .csv
    load_csv,                                      # read rows from .csv
    save_pickle,                                   # write an object to .pickle
    load_pickle,                                   # read an object from .pickle
    save_camset,                                   # write a CameraSet to .camset
    load_camset,                                   # read a CameraSet from .camset
    phase_cache_path,                              # build a canonical cache file path
    cache_exists,                                  # check whether a cache file is non-empty
)

# Plot helpers live in pyCamSet.bad_pipeline.pipeline_plots — re-exported here for convenience.
from pyCamSet.bad_pipeline.pipeline_plots import (    # headless plot helpers
    save_figure,                                   # save a matplotlib Figure to .png
    save_numeric_summary,                          # save a numeric dict to .csv
    plot_error_histogram,                          # histogram of reprojection errors
    plot_per_camera_errors,                        # bar chart of per-camera mean errors
    plot_residual_clusters,                        # 2-D cluster plot with covariance ellipses
    plot_coverage_scatter,                         # per-camera image-plane coverage scatter
    plot_camera_arrangement,                       # pyvista screenshot of camera 3-D layout
)

__all__ = [                                        # explicit public API surface
    # Pipeline orchestrator
    "run_pipeline",
    # Individual phases
    "run_phase1_detection",
    "run_phase2_culling",
    "run_phase3_calibration",
    "run_phase4_analysis",
    "run_phase5_visualisation",
    "run_phase6_self_calibration",
    # Cache helpers
    "save_json",
    "load_json",
    "save_csv",
    "load_csv",
    "save_pickle",
    "load_pickle",
    "save_camset",
    "load_camset",
    "phase_cache_path",
    "cache_exists",
    # Plot helpers
    "save_figure",
    "save_numeric_summary",
    "plot_error_histogram",
    "plot_per_camera_errors",
    "plot_residual_clusters",
    "plot_coverage_scatter",
    "plot_camera_arrangement",
]
