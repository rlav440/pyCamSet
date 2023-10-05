===========================
AbstractTarget
===========================


Introduction
============

The AbstractTarget is an abstract base class that all calibration targets extend from.
All specific implementations define both the location of the feature points of the target, and a implement the "find_in_image" method.
Other methods left to implement are defining a printable net, and optionally visualisation of the target.


Examples
========

Implementing ChArUco 
--------------------

The ChArUco target is a target defined by the openCV library.
It defines methods for detecting and calibrating a camera, and is robust and reliable.
It is also well suited for calibration of multiple cameras, as partial views of the target still result in valid detections.
However, within the openCV framework it can't be used to jointly calibrate 3+ cameras, so we can wrap the base target with the AbstractTarget to enable this.
This starts with the input:

::

   class ChArUco(AbstractTarget):


There are 2 tasks that need to be completed to implement the wrapper.
#. An initial set up that creates the point data array representing the locations of the points.
#. Implementing the find_in_image function, ensuring that the code returns an ImageDetection object.


The first task takes the form of the __init__ method of the class.

.. literalinclude:: ../../../pyCamSet/calibration_targets/target_charuco.py
   :pyobject: ChArUco.__init__

There are three sections of this code. 
The first is the call to the super initialisation. 
This call passes the input arguments to the super class, where they are stored as a reference for regeneration of the target.
The middle section creates a ChArUco board, adding it as a class member (where it can later be used for detection), and defines the point_data array.
The final call to _process_data() generates a local coordinate system for each face of the target (although there is one face for this target), which is used for initial calibrations.

With this defined, we can write the find_in_image class method.


.. literalinclude:: ../../../pyCamSet/calibration_targets/target_charuco.py
   :pyobject: ChArUco.find_in_image

This is a standard implementation of target detection using the ChArUco board, with an optional draw flag.
With these methods defined, the new ChArUco target can be used as an input to the calibration library, allowing the use of ChArUco targets for calibrating n-camera systems.
