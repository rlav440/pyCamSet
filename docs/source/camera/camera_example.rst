================================================================================
Camera Example
================================================================================

This documentation provides extra commentary on the runnable camera_example.py
file found in examples.


Defining a Default Camera
================================================================================

The camera provides a default camera that can be initialised with an intrinsic
matrix, and no transformation.
The camera can be visualised by requesting a mesh representation of the camera,
which gives the camera in world coordinates.
This designed to be viewed as a wireframe.

::

    example_cam = Camera()

    #  get_mesh to get a camera model orientated to the reference frame
    cam_mesh = example_cam.get_mesh()
    cam_mesh.plot(style='wireframe') #pyvista plotting syntax

Gives a representation of the camera in it's default position.

.. image:: cam_mesh.png
    :width: 400
    :alt: Default camera mesh

Camera Functions
================================================================================
The camera provides two main functions:

1. A method of projecting world to camera coordinates.

2. A method of projecting image points to world rays.

The first method is projection of points to camera coordinates, which is done
with the project_points method

::

    # project points to camera coordinates

    world_cord = np.array([0.1, 0, -2]) #m for world coordinates
    im_cord = example_cam.project_points(world_cord)


The camera also provides a sensor map.
This is a representative location of each pixel of the camera in 3D space.
The line created by each pixel and the camera location defines the rays which
project image onto the camera sensor.
For a depth image, multiplication of this sensor map creates a 3D point cloud in
world coordinates

::

    # it also provides the location of the virtual imaging plane as a "sensor map"
    # world_sensor_map is in world coordinates, sensor_map is in camera coordinates
    sensor_map = example_cam.world_sensor_map

    #add the camera center location at 0,0,0 for reference, then make a mesh
    sensor_mesh = pv.PolyData(np.block([[sensor_map.reshape(-1, 3)],[0,0,0]]))
    sensor_mesh.plot()


.. image:: sensor_map.png
    :width: 400
    :alt: Sensor Map representation

