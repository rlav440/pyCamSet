===============================
ChArUco Board
===============================

The ChArUco board is a robust calibration target for handling multi camera calibration.
It is a wrapper around the openCV ChArUco board that allows it to interface with the multi-camera calibration code.
The implementation of this target is a simple example of how to implement a custom calibration target.



Examples
==============================

Calibrating with the ChArUco target is the simplest method of calibration with this library.

::

   from pathlib import Path
   from pyCamSet import CameraCalibrator, ChArUco

   calibration_data = Path('my/calibration/path')
   calibration_target = ChArUco(num_squares_x=10, num_squares_y=10, square_size=4):

   calibrator = CameraCalibrator()
   cams = calibrator(f_loc=calibration_data, calibration_target=calibration_target, draw=True)
