================================================================================
CameraSet
================================================================================



Introduction
============

The CameraSet contains multiple cameras.

A CameraSet is typically the output of a camera calibration.
As such, it represents the pinhole model based description of the system calibrated.
Each CameraSet records the calibration that was used to create it.

A CameraSet is saveable. 
A saved CameraSet is written to a json formatted file for portability.
By convention, these files use the ".camset" extension.
Most of the data is directly parsable from the json structure.
However, the data and results used to generate the calibration are compressed before being written to the file.
All human readable content is written to the first section of the file

A CameraSet is iterable, and indexable by either camera name or index number.


Examples 
========


Constructing a CameraSet
------------------------

A camera set can be assmebled from lists of camera components:
The following code assembles a synthetic ring of cameras.




Calibrating a CameraSet
-----------------------

However, a CameraSet is most powerful when used to represent real camera system.
To achieve this, it must first be calibrated.
To calibrate a camera set, created a file structure of the following form:




This file structure can then be used in the following fashion:

::

   from pathlib import Path
   from pyCamSet import CameraCalibrator, ChArUco

   calibration_data = Path('my/calibration/path')
   calibration_target = ChArUco(num_squares_x=10, num_squares_y=10, square_size=4):

   calibrator = CameraCalibrator()
   cams = calibrator(f_loc=calibration_data, calibration_target=calibration_target, draw=True)

This code will detect the given calibration target in the image data, and calibrate all cameras first independantly, then jointly.
More detail is given in the optimisation section.

Loading a saved CameraSet
-------------------------

Calibrating a camera set will, by default, write a saved version of the calibration to the input directory.
This saved version can be read by other scripts, and used to perform tasks with the reconstructed cameras.

::
   
   from pathlib import Path
   from pyCamSet import load_CameraSet

   cam_loc = Path('my/calibration/path/optimised_cameras.camset)
   my_cams = load_CameraSet(cam_loc)


Visualising the results of a calibration
----------------------------------------

The calibration of a CameraSet is an important mark of quality.
Calibration results are compressed and saved with each CameraSet.
For any CameraSet that was the result of a calibration, that calibration result can be evaluated with the following code.

::

   my_cams.visualise_calibration()

Advice on how to interpret this visualisation detail is available in the documentation for calibration targets.


