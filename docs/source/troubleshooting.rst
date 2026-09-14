================================================================================
Troubleshooting
================================================================================

A target class is ``None``
==========================

``ChArUco``, ``Ccube``, ``PuzzleBoard`` and ``PuzzleBoardCube`` are imported
behind a guard, because they need graphics libraries that a reconstruction-only
install does not. When one of those imports fails, the name is bound to
``None`` rather than raising, so the failure surfaces later and somewhere else:

.. code-block:: text

   TypeError: 'NoneType' object is not callable

To see the real cause, import the target's module directly:

.. code-block:: python

   import pyCamSet.calibration_targets.charuco.target

That raises the underlying :class:`ImportError` -- most often a missing
``svgwrite`` or ``cairosvg``, or the native Cairo library those depend on.

The native Cairo library
========================

``cairosvg`` depends on ``cairocffi``, which needs the native ``cairo``
library installed separately; ``pip`` alone cannot provide this reliably on
Windows. Without it, importing any target module fails with an
:class:`OSError` about a missing ``cairo-2`` library.

pyCamSet already tries to fix the common case for you: importing ``pyCamSet``
calls :func:`pyCamSet.utils.cairo_dll_helper.ensure_cairo_dll_available`, which
locates the conda environment's library directory and registers it before
anything imports ``cairosvg``. It is a no-op where Cairo already loads.

When that is not enough, install the native library for your platform:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Platform
     - Command
   * - conda (any OS)
     - ``conda install -c conda-forge cairo``
   * - macOS (Homebrew)
     - ``brew install cairo``
   * - Debian / Ubuntu
     - ``apt install libcairo2``

.. note::

   The Debian package name is a best guess and has not been verified by the
   pyCamSet team; it may differ on other distributions.

Afterwards ``pip install cairosvg`` (or
``conda install -c conda-forge cairosvg``) should complete without errors, and
``import pyCamSet.calibration_targets.charuco.target`` should succeed.

``pycamset`` will not start
===========================

The GUI needs PySide6, which the lean install leaves out. The command says so
rather than failing obscurely:

.. code-block:: text

   The pyCamSet GUI could not start: <the import error>.
   This is the lean install, which leaves the GUI toolkit out.
   Add it with:
       pip install PySide6

Everything except the GUI works from a lean install, plotting included --- and
the whole calibration is reachable from a script through
:doc:`the workflow package <workflow>`, with no GUI toolkit involved.

PuzzleBoard is not installed
============================

The PuzzleBoard targets depend on the upstream `PuzzleBoard repository
<https://github.com/PStelldinger/PuzzleBoard>`_ by Peer Stelldinger and the HAW
Hamburg authors. It is a research codebase that is not published on PyPI, so it
is optional, and constructing the target without it raises an
:class:`ImportError` naming the fix:

.. code-block:: bash

   pip install "pyCamSet[puzzle]"

The PuzzleBoard source is released under CC0 upstream. Retain the upstream
attribution, and cite the original PuzzleBoard work when publishing results
that use this target.

PuzzleBoardCube rejects ``n_points``
====================================

``PuzzleBoardCube`` assigns six disjoint, deterministic windows of one periodic
PuzzleBoard code to the cube faces, in the fixed order front, right, back,
left, top, bottom. Its ``puzzle-cube-v1`` layout is not random: identical
``n_points`` and ``length`` produce identical face patterns and cube geometry.

The maximum face size is **167 squares per side**, set by packing six faces
into the 501-position code field three across and two down. Larger values are
rejected rather than being allowed to overlap or exceed the code period:

.. code-block:: text

   ValueError: n_points must not exceed 167 for puzzle-cube-v1

OpenCV version differences
==========================

pyCamSet supports OpenCV 4.8 and above, including 5.x, and handles the API
differences at the call sites. Be aware that **OpenCV 5 shifts detected ChArUco
corners by about half a pixel**, so calibrations produced under OpenCV 4 and
OpenCV 5 are not numerically comparable. Do not compare reprojection errors
across the major versions.
