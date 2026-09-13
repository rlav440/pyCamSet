"""
Searching for detector settings that calibrate well.

A study runs the calibration phases over and over with different ChArUco
detector parameters, scores each run, and keeps the ones worth keeping.  It
sits beside the phases rather than inside :mod:`pyCamSet.optimisation`,
which is the bundle adjustment itself: a study calls a solve, and the solver
knows nothing about studies.

Nothing is re-exported here on purpose -- optuna is an optional dependency,
and importing :mod:`pyCamSet.workflow` should not pull it in.
"""
