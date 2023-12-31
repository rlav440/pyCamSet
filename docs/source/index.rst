..  cv_tools documentation master file, created by
   sphinx-quickstart on Wed Apr 14 08:35:44 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

====================================
PyCamSet Documentation
====================================


Welcome to the documentation for the pyCamSet library.

The library aims to provide tools that make it easy to work with, and calibrate cameras.
Specifically, it aims to enable the use of arbitrary calibration target geometries for calibrating "n" cameras.

While openCV provides methods for calibrating and working with cameras (many of which are used here), the pyCamSet library adds useful abstractions and visualisation tools for sets of calibrated cameras, calibration targets, and an easy to use implementation of "n" camera calibration.


Standard Components 
===================
At the heart of the library are the Camera and CameraSet objects, and the calibrate_cameras function, which constructs Cameras and CameraSet objects from calibration images.
CameraSets are save-able to a .json formatted file.

The calibration process is defined by two extensible classes: an abstract model of the calibration target, and an extensible representation of the standard bundle adjustment.

Getting Started
===============

This library can either be installed via pypi:

.. ::
   pip install pyCamSet

or can be cloned from github and then installed with:

.. ::
   git clone https://github.com/rlav440/pyCamSet
   pip install -e . 

Which enables development and extension.

Library extension
=================

pyCamSet's goal of enabling arbitrary calibration targets for "n" cameras is based on a base target, a standard calibration procedure, and a formulation of the bundle adjustment loss.
These are extendable, and examples demonstrate how to extend these components to implement a new calibration target and alter the calibration formulation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   camera
   camera_set
   calibration
   calibration_targets/index
   extension/index


