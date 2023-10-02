====================
Calibration Targets
====================




Calibration targets define the location of object features and how they are detected, acting as digital twins of a real calibration target.
In this library, they always provide a method for detecting the object features, and 3D descriptors of the locations of the features in object coordinates.

Optionally, calibration targets provide methods for reprenting themselves in a 3D plot and providing a printable surface with controlled dimensions.
As this library provides methods for working with 3D targets, the printable surfaces can be returned as a "net" which can be folded over some underlying geometry.

AbstractTarget
==============

All calibration targets inheret from the AbstractTarget class. 
To create a functional target, the implementer must define the "find_in_image" function, and define the self.point_data array.
A complete calibration target also defines a method to plot the target and a method to save the target to a printable representation. 


Specific Targets
================

ChArUco
-------

The ChArUco target is a wrapper around the opencv ChArUco board target.
As a result, this target can be used for multi-camera calibration where the views of all cameras align.

Ccube 
-----

The Ccube, short for ChArUco cube, implements a higher order target defined by 6 aligned ChArUco boards.
This target is useful for camera arrays where a single 2D target would not be fully visible in all cameras.
As the target is 3D, the printable target defines a net, which folds over a base shape of a given geometry.

