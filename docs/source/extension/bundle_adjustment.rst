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


The numba implementation has a few limitations.
The first to note is that as the shape of the object can be complex, there is currently limited support for the dimensions that can be used.
A future update my fix this by unrolling the image points to a flat array and doing a direct lookup.

Secondly, the code will  
