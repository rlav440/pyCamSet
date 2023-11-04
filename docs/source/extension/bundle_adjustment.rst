========================
Bundle Adjustment
========================

Introduction
========================

The bundle adjustment loss is a standard loss used in the computer vision literature.
The implementation of the bundle adjustment eables fast calibrations.

This function is implemented in python, but compiled with numba.
As a result, the function takes only numpy arrays of a specific format.
The code and format are detailed below.

Code example
========================



.. literalinclude:: ../../../pyCamSet/optimisation/compiled_helpers.py
   :pyobject: bundle_adjustment_costfn


The first is that all input data can only be numpy arrays, as this achieves best performance with numba acceleration.

Additionally, the bundle adjustment cost function operates over an unrolled representaiton of the object geometry.
This unrolled representation has the form (i, x, 3) where i is the number of images, x is the number of unrolled keys and 3 is the world coordinate.


