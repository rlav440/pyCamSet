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

The following examples indicate how you can work with a CameraSet to visualise and create the camera data.

Constructing a CameraSet
------------------------

A camera set can be assmebled from lists of camera components:
The following code assembles a synthetic ring of cameras, where each camera is defined by a rotation matrix.

.. literalinclude:: ../../examples/make_camera_ring.py
   :pyobject: make_cams


Calibrating a CameraSet
-----------------------

However, a CameraSet is most powerful when used to represent real camera system.
To achieve this, it must first be calibrated.
To calibrate a camera set, created a file structure of the following form:
This file structure can then be used in the following fashion:

.. literalinclude:: ../../examples/calibrate_cameras.py

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

Cameras and point data
-------------------------------------
The primary purpose of a Camera is to define the projective mapping from 3D to 2D space.

Projection
^^^^^^^^^^
A CameraSet object provides a convinience function for projecting some point in 3D space to all cameras in the camera set.
This function is used in the below code snippet to map a point in 3D space to a camera set, and print the output.

.. literalinclude:: ../../examples/make_camera_ring.py
   :pyobject: project_point

When used with a 5 camera ring and a point at [0.01, 0.03, -0.05] this produces the following output;

:: 

{'cam_0': array([566.66666667, 700. ]), 'cam_1': array([245.98368788, 671.39078209]), 'cam_2': array([340.22273234, 627.8919584 ]), 'cam_3': array([586.4661425, 621.7884872]), 'cam_4': array([760.96604152, 654.59159018])}


Triangulation
^^^^^^^^^^^^^

The inverse of projection is triangulation.
If a point is seen by multiple cameras, its location is constrained.
Triangulation is also provided by pyCamSet as a method of CameraSets, but pyCamSet performs a least squares minimisation (DLT) which is limited in accuracy.

.. literalinclude:: ../../examples/make_camera_ring.py
   :pyobject: triangulate_points

The input proj data, for a single feature, is a dictionary, where each key:value pair is the name of the identifying camera, and the location at which that feature was seen.
Multiple features can be passed as a list of such dictionaries.

For rapid triangulation of larger detection sets, the numpy array produced by the TargetDetection.get_data() method is also accepted.

Writing a CameraSet to an output file
-------------------------------------

Many external tools work with specific camera formats.
Currently, the code provides the capability to write a camera set to a set of MVSnet formatted files.
These files can then be used as input to neural network powered or classical photogrammetry algorithms.

