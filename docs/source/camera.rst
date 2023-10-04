

===================================
Camera
===================================

The Camera object is a base component of this library.

It is an object intended to serve as a virtual instance representing a real camera.
Cameras currently implement the pinhole model of the camera with a 5 component Brown-Conrady model of distortion.

Camera Examples
===============

The Camera object, like a pinhole camera, is represented with an intrinsic and extrinsic matrix and a vector of distortion coefficients.
A Camera can also contain useful non-mathematical information, like a name and the resolution of the mathematical camera it represents.

Camera Initialisation
---------------------

The following code represents a camera at 0,0,0, with a focal length of 1000 pixels, a resolution of [640, 480] and a principle axis offset of [320, 240].

::

   import numpy as np
   from pyCamSet import Camera

   

   extr = np.eye(4)
   intr = np.array([1000, 0, 320], [0, 1000, 240], [0, 0, 1])

   my_camera = Camera(extrinsic=extr, intrinsic=intr, res=(640, 320))


Note that the extrinsic transform uses 4x4 homogenous coordinates.


Camera Visualisation
--------------------


A Camera can be drawn in a 3D plot, given the coordinates that define it.
The get_mesh method returns a pyVista based mesh that can then be plotted.

::

   camera_mesh = my_camera.get_mesh()
   camera_mesh.plot()





