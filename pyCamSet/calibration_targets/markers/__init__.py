"""
Reading fiducial markers out of an image.

One module per detector: OpenCV's ArUco/ChArUco, the ``aruco2`` package's,
and PuzzleBoard's.  Each is a :class:`DetectorParameterisation` -- the
settings it takes and the detector holding them -- so a target picks a
detector without knowing what library is behind it.

Deliberately bare: importing one backend must not import the others, two of
which rest on optional dependencies.
"""
