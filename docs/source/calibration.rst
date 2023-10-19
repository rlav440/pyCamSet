================================
Calibration
================================

Calibration is the process of mathematically optimising a pinhole model of a camera to represent some real camera.
There are two models of calibration: a base method, and manual set up of an optimisation.


Base Calibration
================

The base calibration provides a single function call for implementing a target pose based bundle adjustment.

.. literalInclude:: ../../examples/calibrate_cameras.py

The function is passed the path to the header folder containing the tree of calibration data, and the desired calibration target.
By default, this function will cache detections, intermediate cameras, and the optimised camera set in the passed file location.

Initial estimates of the camera parameters are obtained using openCV.

Manual Calibration
==================

Manual calibration requires initialisation and detection of data points.

.. literalInclude:: ../../examples/manual_calibration.py

If a multi-step optimisation process is required (e.g. stepping through different formulations) this interface is much easier to use.

