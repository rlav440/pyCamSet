
.. cv_tools documentation master file, created by
   sphinx-quickstart on Wed Apr 14 08:35:44 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
========================================
Calibration
========================================

The library provides a single main function for calibrating a set of cameras.
Easy calibration is a main goal.

This function takes a calbiration target, and returns a CameraSet object.

To use this function, the data for the calibration should be in the following tree structure.



Calibration Example
===================

To calibrate a camera system with this library, the first step is to create a calibration target, and take images of the target with the camera system.
This should be placed into the example tree structure.

This can then be passed to the calibrator class, along with the class representing the calibration target that was used to make the images.


Understanding Calibration Metrics
=================================


If a calibration is visualised, with the CameraSet().visualise_calibration() method, there are several plots created.
The first plots are 2D plots, which look at error in image space.
This is the domain over which the bundle adjustment loss is calculated.

The first image looks at the coverage of calibration data over the sensors of the cameras.
This will highlight if the calibration was too localised.
If a region of a camera sensor does not have data, it is not calibrated.

The second plot displays the Euclidean reprojection error achieved, and the distribution of the errors as the relative distance from the true detection.


The second set of plots are 3D plots.
This plots display 3D information that is recovered from the detections with the calbration.
The first plot displays the least squares triangulation of all triangulatable information in world coordinates.
This can give a feel for if the calibration will perform well with later triangulation tasks.

The second plot shows the triangulations displayed in object coordinates.
Ideally, this looks like the internal point_data structure.
However, if there are uncertainties in the location of the object, or other systematic defects this will result in noticable noise in the reconstructed surface.
Some deductions about the cause of the systematic error can be made from this plot.
As an example, if there is low spread in the local x,y coordinates, but a lot of spread in the z, this indicates that the camera system was insensitive to the z direction.
This could be because the effective baseline of the calibrated system is too small.

The final plot aims to detect defects in the calibration target.
If the distance from the mean of a cluster of key point detections to the expected location is greater than the standard deviation of the cluster, this likely indicates a manufacturing defect in the calibration target.




Optimisation Methods
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   detections
   optimisation
