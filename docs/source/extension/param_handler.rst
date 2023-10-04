===============================
Parameter Handler
===============================

Introduction
============

The ParameterHandler is a class that handles the transformation from a set of input parameters to a set of matricies ready to be used in the calculation of the bundle adjustment loss.
The standard ParameterHandler class internally handles the appropriate transformation for a standard object pose based bundle adjustment.


Extension of the Parameter Handler
==================================

The parent ParameterHandler class includes two hook functions for easy extension.
