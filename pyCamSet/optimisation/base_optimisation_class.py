from abc import ABC, abstractmethod
import numpy 

#maybe define a class of bound variables.

#bind to image
#bind to feature
#bind to pose number



# what are the responsibilities of the optimisation class:


class paramblock(ABC):


class bundle_optimiser(ABC):
    def __init__():
    # give it a specification of:

    # the bundles of parameters that serve as inputs to the function
    # the input function to optimise over
    # the data to use.

    # or a custom number of poses.

    # how do I define full reprojection with a param block - this doesn't work because you can't think of everything as individual parameters with this step
    # but I loose no generality in what I want to solve this way.
    # hmmm f







































