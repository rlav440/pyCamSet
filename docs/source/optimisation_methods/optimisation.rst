=================================
Optimisation
=================================

Like every camera library, the core of pyCamSet is the bundle adjustment problem.
The bundle adjustment is accellerated and parallelised with numba, but currently solved with the scipy least squares solver.
For users familiar with bundle adjustment: this will indicate that some options for efficient computation are missed.
However:

#. Minimising the bundle adjustment is not the main time sink of performing a calibration, which is typically feature detection.
#. pyCamSet's goal is to provide an extensible framework for calibration, and some calibration formulations will not have this Schur decomposable structure.
