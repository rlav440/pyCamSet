

===================================
Camera
===================================

The camera is the main block of the functionality provided by the CV tools
library.

It is an object specific implementation of a physical camera. In practise, it is
intended to serve as a virtual instance representing a real camera.
The camera provides methods for locating in space, representing the calibration
of an input camera and performing useful operations on a per camera basis.

Cameras are initialised with a opencv style calibration matrix, and an extrinsic
matrix.

The main functionality provided by the cameras is the ability to project to and
from the camera image plane.