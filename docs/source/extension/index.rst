================================================
Extension of the pyCamSet Library
================================================

This library was written to enable exploration of calibration targets and methodologies.
The calibration of the library relies on the interaction of the AbstractTarget class, the ParameterHandler, and the implementation of the bundle adjustment cost function.

Structure
==============

The following images detail the structure of the library, with functions responsible for extension higlighted.

.. image:: ../source/figs/pyCamSet.png
  :width: 600
  :alt: Relationships between extendable components of the pyCamSet library.

Derivates of these classes can be substituted into this framework, which allows customisation of the calibration target, calibration method, and the type of optimisation.



Extension examples
==================

The following code gives examples of extending the components of the library to implement new calibration targets and methodologies.

.. toctree::
   :maxdepth: 2
   :caption: Extensible Components

   AbstractTarget
   param_handler
   bundle_adjustment
