# this is a sketch for what would be required for a direct optimisation.


# for a template based optimisation: we would need to consider:
# the derivative of every effected parameter:
# so we would need to assess: the value of every parameter and it's derivative at a certain point.

# and then invert it.

# so the question is: 

# how do we calculate the derivative of every point from these parameteres.

# so we basically have then a set of deritvatives in space with respect to the derivatives in 
#then we take the projection of the vector to the cameras in all of the spaces

# and since I have done it myself.I can even do something like implement schurr decompisition.

# and think about the solver that is implemented.



# basically we are looking at a cost block .

# but we do find for a single obseervation,

# we calculate the derivative of: 
# all of the parameters that effect it.

# which means:
# camera extrinsics
# camera intrinsics
# position.

# really the only complex thing is the extrinsics.

#so now a parameter needs to implement the gauss newton method/ LM relaxation over the base image.

# what am I thinking here. 

# so I would say that we have 
