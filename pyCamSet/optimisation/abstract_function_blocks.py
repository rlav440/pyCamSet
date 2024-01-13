from __future__ import annotations
from abc import ABC, abstractmethod 
from dataclasses import dataclass
from enum import IntEnum    
import numpy as np
from numba import njit, prange
from numba.typed import List 
import numba
from typing import Callable, Optional, TYPE_CHECKING, Type


def alloc_wrap(alloc_size, memory):
    """
    If the input function has a zero for memory, returns a function d
    """
    if isinstance(memory, np.ndarray):
        return memory
    else:
        return np.zeros(alloc_size)


class key_type(IntEnum):
    PER_CAM = 0
    PER_IMG = 1
    PER_KEY = 2
    SINGLE  = 3

#then we need some kind of functional lookup: think about the rotating calibration

@dataclass
class param_type:
    def __init__(self, link_type:key_type, n_params:int, mod_function=None) -> None:
        self.link_type = link_type
        self.n_params = n_params
        self.sparse_param = False # used to implement ordering for schops decomp

        if mod_function is None:
            self.mod_function = njit(lambda x: x)
        else:
            self.mod_function = mod_function

    def __hash__(self):
        return hash((int(self.link_type), self.n_params, self.sparse_param))

    


class optimisation_function:
    def __init__(self, function_blocks: list[Type[abstract_function_block]]) -> None:
    
        self.function_blocks = function_blocks
        self.n_blocks = len(function_blocks)


        #the jac functions take either a list '=

        ftemplate = numba.types.FunctionType(
            numba.void(
                numba.types.Array(numba.float64, 1, "C"),
                numba.types.Array(numba.float64, 1, "C"),
                numba.types.Array(numba.float64, 1, "C"),
                numba.types.Array(numba.float64, 1, "C"),
            )
        )
        self.loss_jac = List.empty_list(ftemplate)
        self.loss_fun = List.empty_list(ftemplate)

        for block in self.function_blocks:
            self.loss_jac.append(block.compute_jac)# a bunch of function objects to call
            self.loss_fun.append(block.compute_fun)# a bunch of function objects to call

        #required working memory:
        self.working_memories = [b.array_memory for b in function_blocks]
        self.can_determine_working_memory = all(
            [m is not None for m in self.working_memories]
        )
        if self.can_determine_working_memory:
            self.wrk_mem_req = max(self.working_memories)
        else:
            self.wrk_mem_req = None

        #required output_memory:
        self.grad_outputsize = [b.num_inp * b.num_out for b in function_blocks]
        self.out_mem_req = max(
            max(self.grad_outputsize),
            max([b.num_out for b in function_blocks]) #uses the same memory to store outpute 
            )
        self.inp_mem_req = max([b.num_inp for b in function_blocks])

        # point locations within the jacobean matrix that will be used

        self.unique_link_inds = list(set([fb.params for fb in function_blocks]))
        self.block_param_inds = [self.unique_link_inds.index(fb.params) for fb in function_blocks]
        self.param_starts = []
        self.param_offset = []
        self.associated = []
        starting_point = 0
        for link_ind in self.unique_link_inds:
            self.param_starts.append(starting_point)
            self.param_offset.append(link_ind.n_params)
            self.associated.append(int(link_ind.n_params))

        self.normal_params = np.sum(self.param_offset)
        self.linking_params = np.sum([b.num_out for b in self.function_blocks[1:]]) 
        self.param_len = self.normal_params + self.linking_params
        self.n_outs = np.array([b.num_out for b in self.function_blocks])
        self.n_inps = np.array([b.num_inp for b in self.function_blocks])
        self.starting_ind = np.cumsum([0] + self.n_outs) + self.param_len

        self.param_slices = []

        self.templated = function_blocks[-1].template



class abstract_function_block(ABC):
    """
    The abstract function block defines a chain of methods.
    An optimisation function is defined as chains of abstract function blocks.
    This is similar to ceres.

    The class needs to define how to calculate it's index from a line datum

    How does the class optimise for cached variables?
    """

    array_memory: int | None = None #amount of array memory required to compute
    template = False #if the class pulls from a calibration template for some params
    # best example of this is calibration targets, where the location of a feature is 
    # treated as a given and isn't optimised.

    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def num_inp(self):
        pass

    num_inp: int #number of differentiable inputs

    @property
    @abstractmethod
    def num_out(self):
        pass

    num_out: int #number of function return values

    @property
    @abstractmethod
    def params(self): 
        pass

    params: param_type #number of parameters to use.

    @abstractmethod
    def compute_fun(param, input, output, memory=0):
        pass

    @abstractmethod
    def compute_jac(param, input, output, memory=0):
        pass

    def __add__(self, other: Type[abstract_function_block]|optimisation_function) -> optimisation_function:
        if issubclass(type(other), abstract_function_block):
            return optimisation_function([self, other])
        if isinstance(other, optimisation_function):
            return optimisation_function([self] + other.function_blocks)
        raise ValueError(f"could not combine function block with {other}")
       

    def __radd__(self, other: [abstract_function_block]|optimisation_function) -> optimisation_function:
        if issubclass(type(other), abstract_function_block):
            return optimisation_function([other, self])
        if isinstance(other, optimisation_function):
            return optimisation_function(other.function_blocks + [self])
        raise ValueError(f"could not combine function block with {other}")

def make_param_struct(
        function_blocks : list[abstract_function_block], detection_data
    ) -> tuple[np.ndarray,np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes an input function block definition and detection data, then defines a structure
    for the parameter matrix associated with the input sequence of function blocks.
    Coalesces data into numpy arrays for use in compiled numba code

    :param function_blocks: a list of function blocks that define a loss function
    :param detection_data: the detection data to optimise.
    :returns param_starts: the starting point of each unique set of parameters
    :returns param_offset: the length of each individual parameter string
    :returns param_ids: the id mapping from each function block to the unique params associated.
    :returns param_assoc: the id of the associated parameter in the detection data structure.
    """
    max_imgs = np.max(detection_data[:, 1])
    max_keys = np.max(detection_data[:, 2])
    max_cams = np.max(detection_data[:, 0])

    param_numbers = {
        key_type.PER_CAM:max_cams,
        key_type.PER_IMG:max_imgs,
        key_type.PER_KEY:max_keys,
    }

    #TODO account for the mod function here, modifying the numbers of params

    # find all of the unique instances of link indicators
    unique_link_inds = list(set([fb.params for fb in function_blocks]))
    block_param_inds = [unique_link_inds.index(fb.params) for fb in function_blocks]
    param_starts = []
    param_offset = []
    associated = []
    starting_point = 0
    for link_ind in unique_link_inds:
        param_starts.append(starting_point)
        param_offset.append(link_ind.n_params)
        associated.append(int(link_ind.n_params))
        starting_point += link_ind.n_params * param_numbers[link_ind.link_type] 

    return np.array(param_starts), np.array(param_offset), np.array(block_param_inds), np.array(associated)
    

def make_lookup_fn(function_blocks : list[abstract_function_block], detection_data):
    """
    builds a lookup function that takes the cam, im and feature number and returns the correct params for each feature.
    """

    #takes in the input function blocks.
    #finds all of the unique params used within the optimisation
    
    starts, offset, param_inds, key_type = make_param_struct(function_blocks, detection_data) 
    param_len = np.sum(offset)

    @njit
    def lookups(param_line):
        param_data = np.empty(param_len)
        nblocks = len(param_inds)
        k = 0
        for idb in range(nblocks):
            s_num = param_line[key_type[idb]] #optionally we have some numba function on this value.
            p_ind = param_inds[idb]
            start = starts[p_ind] + s_num * offset[p_ind] #and the associated change in the start location
            new_k = k + s_num * offset[p_ind]
            param_data[k:new_k] = np.arange(start, start + offset[p_ind])
            k += offset[p_ind]
        return param_data
    return lookups

def block_string_to_compiled_jacobian_line(function_def: optimisation_function):

    """
    This function takes a list of abstract function blocks and converts them 
    to function that calculates the jacobean for a line of the output function.

    For every parameter in the final jacobian, it's values are calculated

    """
    
    if function_def.can_determine_working_memory:
        #make an array called memslices, and slice up the input arrays
        @njit
        def make_jacobean(dense_param_arr,
                          base, jac, per_block, #memory for storing jacobean calcs
                          input_memory, output_memory, #inputs and outputs for the functions
                          working_memory, #space filled template memory 
                          ):
            jac[:] = base[:] 

            for i in range(function_def.n_blocks):
                per_block[:] = base[:]  
                # param, inp, output, memory
                grads = function_def.loss_jac[i](
                    dense_param_arr[function_def.param_slices[i]:function_def.param_slices[i+1]], 
                    input_memory,
                    output_memory[:function_def.grad_outputsize[i]],
                    working_memory[:function_def.working_memories[i]],
                )

                output_ind = function_def.starting_ind[i] 
                for ii in range(function_def.n_outs[i]):
                    per_block[output_ind + ii, function_def.param_slices[i]:function_def.param_slices[i+1]] = grads[ii]
                jac = per_block @ jac
                
                function_def.loss_func[i](
                    dense_param_arr[function_def.param_slices[i]:function_def.param_slices[i+1]], 
                    input_memory,
                    output_memory[:function_def.n_outs[i]],
                    working_memory[:function_def.working_memories[i]],
                )
                # compute the value for the current points 
                input_memory[:function_def.n_outs[i]] = output_memory[:function_def.n_outs[i]]
            return jac
    else:
        @njit
        def make_jacobean(dense_param_arr,
                          base, jac, per_block, #memory for storing jacobean calcs
                          input_memory, output_memory, #inputs and outputs for the functions
                          working_memory, #space filled template memory 
                          ):
            jac[:] = base[:]
            for i in range(function_def.n_blocks):
                per_block[:] = base[:]  
                # param, inp, output, memory
                grads = function_def.loss_jac[i](
                    dense_param_arr[function_def.param_slices[i]:function_def.param_slices[i+1]], 
                    input_memory[:function_def.n_inps[i]],
                    output_memory[:function_def.grad_outputsize[i]],
                    -1,
                )

                output_ind = function_def.starting_ind[i] 
                for ii in range(function_def.n_outs[i]):
                    per_block[output_ind + ii, function_def.param_slices[i]:function_def.param_slices[i+1]] = grads[ii]
                jac = per_block @ jac
                
                function_def.loss_func[i](
                    dense_param_arr[function_def.param_slices[i]:function_def.param_slices[i+1]], 
                    input_memory[:function_def.n_inps[i]],
                    output_memory[:function_def.n_outs[i]],
                    -1,
                )
                # compute the value for the current points 
                input_memory[:function_def.n_outs[i]] = output_memory[:function_def.n_outs[i]]
            return jac
    return make_jacobean

def make_jacobean_evaluator(function_def: optimisation_function, detections: np.ndarray, threads, template: np.ndarray| None) -> Callable:

    #take and reshape the detections?
    d_shape = detections.shape
    p_shape = (threads, int(np.ceil(d_shape[0]/threads)), *d_shape)
    p_detect = np.resize(detections, p_shape)

    # get the shape of the jacobean from the data
    j_shape = 2 * d_shape[0]

    j_line = block_string_to_compiled_jacobian_line(function_def)

    lookups = make_lookup_fn(function_def.function_blocks, detections)

    templated = True if template is not None else False
    template_point = np.zeros(3)

    @njit
    def compute_jacobean(params):
        output_jacobean = np.zeros((j_shape,len(params)))
        for idp in prange(threads):
            d_data = p_detect[idp]
            l = len(d_data)
            base_array = np.eye(function_def.param_len) 
            jac_array = np.eye(function_def.param_len) 
            out_array = np.eye(function_def.param_len) 
            #allocate the right amount of memory
            output_memory = np.empty(function_def.out_mem_req)
            input_memory = np.empty(function_def.inp_mem_req)


            if function_def.can_determine_working_memory:
                working_memory = np.empty(function_def.wrk_mem_req)
            else:
                working_memory = 0

            for idl in range(l):
                d = d_data[idl]
                #get the associated indexes for each detection

                if templated:
                    input_memory[:3] = template[d[2]] # use unrolled keys 

                lookup = lookups(d)
                param_string = params[lookup]
                jacs = j_line(
                    param_string,
                    base_array, 
                    jac_array,
                    out_array,
                    input_memory,
                    output_memory,
                    working_memory,
                )
                output_jacobean[idp, lookup] = jacs
        
        output_jacobean = output_jacobean.reshape()#and then drop the latter parts of the jacobean.
        return output_jacobean

    return compute_jacobean





