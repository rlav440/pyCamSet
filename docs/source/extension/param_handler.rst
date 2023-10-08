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
These are 1. add_extra_params and 2. parse_extra_params_and_setup
These functions allow additional params to reflect and change the set up of the target.

Here, the use of these functions is demonstrated by implementing a calibration with two targets, fixed rigidly by some unknown transform.

The code extends the base target by adding an additional parameter that rpepresents the rigid transformation between the targets. 


__init__
--------
.. literalinclude:: ../../../examples/extend_param_handler.py
   :pyobject: TwoTargetCalibrator.__init__

The init file performs a standard init for one of the targets.
The data related to the second target is also stored.

To populate the single point_data array, both arrays are flattened and concatenated.


add_extra_params
----------------

.. literalinclude:: ../../../examples/extend_param_handler.py
   :pyobject: TwoTargetCalibrator.add_extra_params

This function adds one extra set of parameters to the optimistaion, the relative pose of the two targets.
A quick first guess is taken at the relative transformation of the two targets is taken.



parse_extra_params_and_setup
----------------------------

.. literalinclude:: ../../../examples/extend_param_handler.py
   :pyobject: TwoTargetCalibrator.parse_extra_params_and_setup

This method mutates the internal point_data structure to reflect the estimated constant transformation between the targets.
This function is called every time the cost function is evaluated, so expensive operations here can really slow down an optimisation.


get_detection_data
------------------


.. literalinclude:: ../../../examples/extend_param_handler.py
   :pyobject: TwoTargetCalibrator.get_detection_data

One more override is required for this particular class.
Because this target has multiple detections, the data from both detections needs to be returned.
Here, we leverage and duplicate the super class, then return the concatenated detection data structures.
A constant index offset is applied to the second detection.
