=======================
The Pinhole Camera
=======================

Underlying model
================

The pinhole camera is a simplification used to represent most cameras.
It represents the cameras as an infinitely small aperture, projecting to an
image plane.
The camera is defined by the focal distance (the distance of the imaging plane
to the aperture) and the principle point (the location of the image plane).

.. math::

    c_{int} = \begin{bmatrix}
        f_x & 0 & p_x\\
        0 & f_y & p_y\\
        0 & 0 & 1
    \end{bmatrix}

The pose of the camera is represented as a combination of a rotation and a
translation.
This repository uses the 4x4 homogenous representation of the transformation.
In combination, the cameras used in this repository are fundamentally defined by
these two matrices.

.. math::

    c_{int} = \begin{bmatrix}
                            f_x & 0 & p_x\\
                            0 & f_y & p_y\\
                            0 & 0 & 1
                         \end{bmatrix}
       \:, \:
    c_{ext} = \begin{bmatrix}
                   R & -t \\
                   0 & 1
              \end{bmatrix}

Implementation
==============

In practise, the transformation from world to image coordinates can be expressed
by the equation:

.. math::

    \begin{bmatrix}
        u\\
        v\\
        1
    \end{bmatrix}
    =
    \begin{bmatrix}
        f_x & 0 & p_x\\
        0 & f_y & p_y\\
        0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
        R, & -t \\
    \end{bmatrix}
        \begin{bmatrix}
        x\\
        y\\
        z\\
        1
    \end{bmatrix}

This projects x,y,z into u,v coordinates.
However, this causes issues, as images are traditionally stored as v,u arrays.
Internally, the camera works in u,v coordinates, but reports v,u coordinates.

A similar fudge is applied to the creation of a sensor map, in order to ensure
alignment with input images.
