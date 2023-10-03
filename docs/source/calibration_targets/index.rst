
.. cv_tools documentation master file, created by
   sphinx-quickstart on Wed Apr 14 08:35:44 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

====================
Calibration Targets
====================




Calibration targets define the location of object features and how they are detected, acting as digital twins of a real calibration target.
In this library, they always provide a method for detecting the object features, and 3D descriptors of the locations of the features in object coordinates.

Optionally, calibration targets provide methods for reprenting themselves in a 3D plot and providing a printable surface with controlled dimensions.
As this library provides methods for working with 3D targets, the printable surfaces can be returned as a "net" which can be folded over some underlying geometry.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   charuco_target
   ccube_target
