================
Detections
================

An optimisation requires a collection of detected data.
This data is detected per image, and collated to a reference that contains the detections from all images.

TargetDetection
---------------

The TargetDetection class contains a central collection of data, and methods that operate on that data.
As the library uses numba to accelerate the calculation of the bundle adjustment loss function, the detection format must fundamentally be a numpy array.
It provides methods to split the model into iterable chunks by detection key, image and camera.

:meth:`pyCamSet.calibration_targets.targetDetections.TargetDetection`


ImageDetection
--------------

The ImageDetection method contains the keys and values seen in a single image.
It is used as a return type from an implemented "find_in_image" function to ensure that the data can be later parsed into the target detection
