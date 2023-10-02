================
Detections
================

An optimisation requires a collection of detected data.
This data is detected per image, and collated to a reference that contains the detections from all images.

TargetDetection
---------------

The TargetDetection class contains a central collection of data, and methods that operate on that data.
As the library uses numba to accelerate the calculation of the bundle adjustment loss function, the detection format must fundamentally be a numpy array.
It provides methods to split the model into iterable chunks by 


ImageDetection
--------------
