================================
Ccube Targets
================================

Introduction
============

The Ccube (ChArUco Cube) target is an easily manufacturable, 3D calibration target designed for use in inwards facing camera systems.
When a calbiration target is being made, laminating to a flat surface is generally pretty forgiving.
When faces are being applied to a target, there is a lot more scope for error.

The Ccube tries to address this by generating the calibation pattern as a foldable net, ensuring that the order and rough orientation of the faces is constrained.
Functionally, the Ccube behaves like 6 rigidly connected ChArUco targets.

Examples
========

Calibrating with a Ccube is functionally the same as calibrating with any other target.
The length of a Ccube is defined in millimeters.

::

   from pathlib import Path
   from pyCamSet import CameraCalibrator, Ccube

   calibration_data = Path('my/calibration/path')
   calibration_target = Ccube(length=40, n_points=10):

   calibrator = CameraCalibrator()
   cams = calibrator(f_loc=calibration_data, calibration_target=calibration_target, draw=True)


However, there are a couple of functions to help you get to the calibration.
The first is plotting the cube, and its associated textures.

::
   
   calibration_target.plot()

This is useful for checking that everything lines up virtually and in reality!

::

   calibration_target.save_to_pdf()

This will save the calibration target to a pdf file. 
Saving to the pdf ensures that the target will have the correct dimensions (print at 100% scale).


