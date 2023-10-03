================================================================================
CameraSet
================================================================================

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

