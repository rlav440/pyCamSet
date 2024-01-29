from __future__ import annotations

from pathlib import Path
import inspect
from abc import ABC, abstractmethod 
from dataclasses import dataclass
from enum import IntEnum    

import numpy as np
from numba import njit, prange
from numba.typed import List
from numba.types import UniTuple
import numba
from typing import Callable, Optional, TYPE_CHECKING, Type
import ast
from collections import namedtuple

Import = namedtuple("Import", ["module", "name", "alias"])

def get_imports(path):
    with open(path) as fh:        
       root = ast.parse(fh.read(), path)

    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import):
            module = ()
        elif isinstance(node, ast.ImportFrom):  
            module = (node.module,)
        else:
            continue
        for n in node.names:
            yield Import(module, (n.name,), n.asname)

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

    def __key(self):
        return (int(self.link_type), self.n_params, self.sparse_param) 
    def __hash__(self):
        key = self.__key 
        return hash((key))

    def __eq__(self, other):
        return self.__key == other.__key

    


class optimisation_function:
    def __init__(self, function_blocks: list[Type[abstract_function_block]]) -> None:
    
        self.function_blocks = function_blocks
        self.n_blocks = len(function_blocks)

        self.loss_jac = None
        self.loss_fun = None

        #required working memory:
        self.working_memories = [b.array_memory for b in function_blocks]
        self.wrk_mem_req = max(self.working_memories)

        #required output_memory:
        self.grad_outputsize = [(b.num_inp + b.params.n_params) * b.num_out for b in function_blocks]

        self.out_mem_req = max(
            max(self.grad_outputsize),
            max([b.num_out for b in function_blocks]) #uses the same memory to store outpute 
            )
        self.inp_mem_req = max([b.num_inp for b in function_blocks])

        # this defines the locations of the 

        self.param_slices: np.ndarray
        self.param_line_length = None

        self.templated = function_blocks[-1].template

        self.ready_to_compute = False

    def _prep_for_computation(self):
        if self.ready_to_compute:
            return
        # create numba compatible containers of functions

        self.ftemplate = numba.types.FunctionType(
            numba.void(
                numba.types.Array(numba.float64, 1, "C"),
                numba.types.Array(numba.float64, 1, "C"),
                numba.types.Array(numba.float64, 1, "C"),
                numba.types.Array(numba.float64, 1, "C"),
            )
        )

        # point locations within the jacobean matrix that will be used
        self.unique_link_inds = []
        for fb in self.function_blocks:
            if not fb.params in self.unique_link_inds:
                self.unique_link_inds.append(fb.params)
        self.block_param_inds = [self.unique_link_inds.index(fb.params) for fb in self.function_blocks]

        self.param_starts = np.cumsum([0] + [b.n_params for b in self.unique_link_inds])
        self.n_outs = np.array([b.num_out for b in self.function_blocks])
        self.n_inps = np.array([b.num_inp for b in self.function_blocks])
        self.n_params = np.array([b.params.n_params for b in self.function_blocks])

        slice = []                       
        for idb, block in enumerate(self.function_blocks):
            b_ind = self.block_param_inds[idb]
            slice.append(self.param_starts[b_ind]) 
            slice.append(self.param_starts[b_ind + 1]) 


        self.param_line_length = self.param_starts[-1] #the length of the params passed in is the last of the param starts vector.

        #so we get the number of params: with a buffer for the first function
        inp_buffer = np.cumsum([0] + [b.num_out for b in  self.function_blocks] + [0])      

        #input and output are a sliding window along an array 
        #define the shape of the jacobean, as the shape of the param structure + a little more for the 
        

        for i in range(len(self.function_blocks) + 1):
            slice.append(inp_buffer[i]  + self.param_line_length) 
            slice.append(inp_buffer[i + 1] + self.param_line_length) 
        
        self.param_slices = np.array(slice)
        self.ready_to_compute = True


    def block_string_to_compiled_jacobian_line(self) -> Callable:

        """
        This function takes a list of abstract function blocks and converts them 
        to function that calculates the jacobean for a line of the output function.

        For every parameter in the final jacobian, it's values are calculated

        """
        self._prep_for_computation()
        

        loss_jac = List.empty_list(self.ftemplate)
        loss_fun = List.empty_list(self.ftemplate)

        for block in self.function_blocks:
            loss_jac.append(block.compute_jac)# a bunch of function objects to call
            loss_fun.append(block.compute_fun)# a bunch of function objects to call

        n_blocks = self.n_blocks
        param_slices = self.param_slices
        grad_outputsize = np.array(self.grad_outputsize)
        working_memories = np.array(self.working_memories)
        n_params  = np.array(self.n_params)
        n_outs = np.array(self.n_outs)


        @njit(cache=True)
        def make_jacobean(dense_param_arr,
                          base, jac, per_block, #memory for storing jacobean calcs
                          input_memory, output_memory, #inputs and outputs for the functions
                          working_memory, #space filled template memory 
                          loss_jac0,
                          loss_fun0,
                          ):
            jac[:] = base[:] 

            for i in range(n_blocks-1, -1, -1):
                per_block[:] = base[:]  
                # param, inp, output, memory
                i0 = dense_param_arr[param_slices[2*i]:param_slices[2*i+1]]

                # numba.gdb_init()
                loss_jac0[i](
                    i0,
                    input_memory,
                    output_memory[:grad_outputsize[i]],
                    working_memory[:working_memories[i]],
                )

                out_ind_start = param_slices[2*n_blocks + 2*(i)] 
                out_ind_end = param_slices[2*n_blocks + 2*(i) + 1]
                n_outputs = out_ind_end - out_ind_start
                
                inp_ind_start = param_slices[2*n_blocks + 2*i + 2] 
                inp_ind_end = param_slices[2*n_blocks + 2*i + 1 + 2]
                n_inputs = inp_ind_end - inp_ind_start
                #write the derivatives of the parameters
                n_param  = n_params[i]


                param_start = param_slices[2*i]
                param_end = param_slices[2*i + 1]
                ll = n_inputs + n_param

                for idc, output_var in enumerate(range(out_ind_start, out_ind_end)):

                    #write the derivative with respect to the controlling params
                    per_block[output_var, param_start:param_end] = output_memory[idc*ll:idc*ll + n_param] #envisions this as a dense array
                    #write the derivative with respect to the inputs
                    if n_inputs != 0:
                        per_block[output_var, inp_ind_start:inp_ind_end] = output_memory[idc*ll + n_param:(idc + 1)*ll] #envisions this as a dense array


                jac = per_block @ jac


                loss_fun0[i](
                    dense_param_arr[param_slices[2*i]:param_slices[2*i+1]], 
                    input_memory,
                    output_memory,
                    working_memory,
                )
                input_memory[:n_outs[i]] = output_memory[:n_outs[i]]
            return jac
        return make_jacobean

    def _make_lookup_fn(self, detection_data):
        """
        builds a lookup function that takes the cam, im and feature number and returns the correct params for each feature.
        """

        #takes in the input function blocks.
        #finds all of the unique params used within the optimisation
        
        starts, block_n_params, param_inds, key_type = make_param_struct(self.function_blocks, detection_data) 

        # point locations within the jacobean matrix that will be used
        param_len = np.sum(block_n_params)
        param_slices = self.param_slices

        @njit
        def lookups(param_line):
            """
            Constructs an array that indexes the input param structure for the param elements relevant to this computation
            """
            param_data = np.empty(param_len)
            nblocks = len(param_inds)
            for idb in range(nblocks):
                s_num = param_line[key_type[idb]] #the index value of the associated parameter
                p_ind = param_inds[idb] # maps the param to it's index in the unique params
                start = starts[p_ind] + s_num * block_n_params[p_ind] #and the associated change in the start location
                param_data[param_slices[2*idb]:param_slices[2*idb + 1]] = np.arange(start=int(start), stop=int(start + block_n_params[p_ind]))
            return param_data
        return lookups

    def _get_loss_fn_strings(self) -> tuple[list[str],list[str],list[str]]:
        self._prep_for_computation()

        n_blocks = self.n_blocks

        import_set = set()

        for block in self.function_blocks[::-1]:
            base =  inspect.unwrap(block.compute_fun)
            base_code = inspect.getsource(base)
            base_code = inspect.getsourcefile(base)
            for imp in get_imports(base_code):
                import_set.add(imp)

        import_lines = []
        for s in import_set:
            # print(s)
            if len(s.module) == 0:
                st = f"import {s.name[0]}" + (f" as {s.alias}" if s.alias is not None else "")
            else:
                st = f"from {s.module[0]} import {s.name[0]}" + (f" as {s.alias}" if s.alias is not None else "")
            import_lines.append(st)
        import_lines.append("\n")
        
        needed_preamble = [
            # "def make_loss_fn(op_fun):"
            "param_slices = op_fun.param_slices",
            "n_outs = op_fun.n_outs",
            "working_memories = np.array(op_fun.working_memories)",
            "n_blocks = op_fun.n_blocks",
        ]

        fn_content = []
        for i in range(n_blocks-1, -1, -1):
            prep = [ 
            f"\tparams = dense_param_arr[param_slices[2*{i}]:param_slices[2*{i}+1]]",
            ]
            base =  inspect.unwrap(self.function_blocks[i].compute_fun)
            base_code = inspect.getsource(base)
            section = base_code.split('\n')[3:-2] #TODO replace this with AST parsing
            section = [s.replace("        ", "    ") for s in section]
            end = f"\tinp[:n_outs[{i}]] = output[:n_outs[{i}]]"
            fn_content.extend(prep) 
            fn_content.extend(section) 
            fn_content.append(end)

        return import_lines, needed_preamble, fn_content

    def make_full_loss_template(self, detections, template, threads) -> Callable:

        def t(l):
            return ["\t" + li for li in l]

        needed_imports, addittional_preamble, content = self._get_loss_fn_strings()
        

        start = ["from numba import prange",
                "from pyCamSet.optimisation.abstract_function_blocks import make_param_struct",
                 " ",
                "def make_full_loss(op_fun, detections, template, threads):"
        ]
        preamble = [
            f"n_threads = threads",
            f"op_fun._prep_for_computation()",
            f"d_shape = detections.shape",
            f"p_shape = (threads, int(np.ceil(d_shape[0]/threads)), *d_shape[1:])",
            f"detection_data = np.resize(detections, p_shape)",

            f"n_lines = detection_data.shape[1]",
            f"inp_mem = op_fun.inp_mem_req",
            f"out_mem = op_fun.out_mem_req",
            f"wrk_mem = op_fun.wrk_mem_req",

            "starts, block_n_params, param_inds, key_type = make_param_struct(op_fun.function_blocks, detections)",
            "param_len = np.sum(block_n_params)",

            f"line_n_params = op_fun.param_line_length",
            f"param_slices = op_fun.param_slices",

            f"use_template = op_fun.templated and (template is not None)",
            f"if op_fun.templated and not use_template:",
            f'\traise ValueError("A templated optimisation was defined, but no template data was given to create the loss function")',
            f"t_data: np.ndarray = template if use_template else np.zeros(3)",
            f"@njit",
            f"def full_loss(inp_params):",
            f"\tlosses = np.empty((n_threads, n_lines, 2))",
            f"\tfor i in prange(n_threads):",
            f"\t\t#make the memory components required",
            f"\t\tinp = np.empty(inp_mem)",
            f"\t\toutput = np.empty(out_mem)",
            f"\t\tmemory = np.empty(wrk_mem)",
            f"\t\tdense_param_arr = np.empty(line_n_params)",

            f"\t\tfor ii in range(n_lines):",
            f"\t\t\tdatum = detection_data[i, ii]",
        ]
        # write the script that populates the array here
        param_slicing = [
            f"\t\t\tfor idb in range(n_blocks):",
            f"\t\t\t\ts_num = datum[key_type[idb]] #the index value of the associated parameter",
            f"\t\t\t\tp_ind = param_inds[idb] # maps the param to it's index in the unique params",
            f"\t\t\t\tstart = starts[p_ind] + s_num * block_n_params[p_ind] #and the associated change in the start location",

            f"\t\t\t\tdense_param_arr[param_slices[2*idb]:param_slices[2*idb + 1]] = inp_params[int(start):int(start + block_n_params[p_ind])]",
        ]


        mid_amble = [
            f"\t\t\tif use_template:",
            f"\t\t\t\tinp[:3] = t_data[int(datum[2])] ",
        ]

        loss_calc = [ '\t\t' + c for c in content] #this really needs to be AST

        postamble = [
            f"\t\t\tlosses[i, ii] = [output[0] - datum[3], output[1] - datum[4]]",
            f"\treturn losses",
            f"return full_loss"
        ]

        fn = needed_imports + start + t(addittional_preamble) + t(preamble) + t(param_slicing) + t(mid_amble) + t(loss_calc) + t(postamble)
        str_fn = "\n".join(fn).replace("\t", "    ")

        loc = Path(__file__)
        # with open(loc.parent/"template_functions/test.py", 'w') as f:
            # f.write(str_fn)

        from . import template_functions
        loss_fn: Callable = template_functions.test.make_full_loss(self, detections, template, threads)
        print(loss_fn)
        return loss_fn

    def _make_loss_per_line_function(self) -> Callable:
        self._prep_for_computation()

        param_slices = self.param_slices
        n_outs = self.n_outs
        working_memories = np.array(self.working_memories)
        n_blocks = self.n_blocks

        import_set = set()

        for block in self.function_blocks[::-1]:
            base =  inspect.unwrap(block.compute_fun)
            base_code = inspect.getsource(base)
            base_code = inspect.getsourcefile(base)
            # section = '\n'.join(base_code.split('\n')[3:-2])
            for imp in get_imports(base_code):
                import_set.add(imp)

        import_lines = []
        for s in import_set:
            # print(s)
            if len(s.module) == 0:
                st = f"import {s.name[0]}" + (f" as {s.alias}" if s.alias is not None else "")
            else:
                st = f"from {s.module[0]} import {s.name[0]}" + (f" as {s.alias}" if s.alias is not None else "")
            import_lines.append(st)
        import_lines.append("\n")
            #find all lines tat are either from or import

        def_function = "\n".join([
            # "def make_loss_fn(op_fun):"
            "param_slices = op_fun.param_slices",
            "n_outs = op_fun.n_outs",
            "working_memories = np.array(op_fun.working_memories)",
            "n_blocks = op_fun.n_blocks",
            "@njit",
            "def loss_fn(dense_param_arr, input_memory, output_memory, working_memory):" 
        ])
        for i in range(n_blocks-1, -1, -1):
            prep = [ 
            f"\tparams = dense_param_arr[param_slices[2*{i}]:param_slices[2*{i}+1]]",
            f"\tinp = input_memory",
            f"\toutput = output_memory[:n_outs[{i}]]",
            f"\tmemory = working_memory[:working_memories[{i}]]",
            ]
            base =  inspect.unwrap(self.function_blocks[i].compute_fun)
            base_code = inspect.getsource(base)
            section = base_code.split('\n')[3:-2]
            end = f"\tinput_memory[:n_outs[{i}]] = output[:n_outs[{i}]]"
            def_function = "\n".join([def_function] + prep + section + [end])
        end = "\treturn output"
        def_function = "\n".join([def_function, end]).replace("        ", "    ").replace("\t", "    ")
        def_function = ("def make_loss_fn(op_fun):\n" + def_function).replace("\n", "\n    ")
        end = "\n    return loss_fn" 
        def_function += end

        def_function = "\n".join(import_lines) + def_function
        print(def_function)

        loc = Path(__file__)
        with open(loc.parent/"template_functions/test.py", 'w') as f:
            f.write(def_function)

        from . import template_functions
        loss_fn = template_functions.test.make_loss_fn(self)
        # @njit
        # def loss_fn(dense_param_arr, input_memory, output_memory, working_memory, loss_fn):
        #     for i in range(n_blocks -1, -1, -1):
        #         # param, inp, output, memory 
        #         loss_fn[i](
        #             dense_param_arr[param_slices[2*i]:param_slices[2*i+1]], 
        #             input_memory,
        #             output_memory[:n_outs[i]],
        #             working_memory[:working_memories[i]],
        #         )
        #         # compute the value for the current points 
        #         # print(f"block {i}")
        #         # print(f"params {dense_param_arr[param_slices[2*i]:param_slices[2*i+1]]}" )
        #         # print(f"input memory: {input_memory}")
        #         # print(f"ourput mems {output_memory[:n_outs[i]]}")
        #         # print(f"working mems {working_memory[:working_memories[i]]}")
        #         input_memory[:n_outs[i]] = output_memory[:n_outs[i]]
        #     return output_memory
        return loss_fn

    
    def make_full_loss_fn(self, detections, threads, template: np.ndarray = None):
        n_threads = threads

        self._prep_for_computation()

        return self.make_full_loss_template(detections, template, threads)

#         d_shape = detections.shape
#         p_shape = (threads, int(np.ceil(d_shape[0]/threads)), *d_shape[1:])
#         detection_data = np.resize(detections, p_shape)
#
#         # get the shape of the jacobean from the data
#         n_lines = detection_data.shape[1]
#         inp_mem = self.inp_mem_req
#         out_mem = self.out_mem_req
#         wrk_mem = self.wrk_mem_req
#
#         line_n_params = self.param_line_length
#
#         use_template = self.templated and (template is not None)
#         if self.templated and not use_template:
#             raise ValueError("A templated optimisation was defined, but no template data was given to create the loss function")
#         t_data: np.ndarray = template if use_template else np.zeros(3)
# #
#         @njit
#         def full_loss(params, per_line_loss, loss_fns, lookup_make):
#             losses = np.empty((n_threads, n_lines, 2))
#             for i in prange(n_threads):
#                 #make the memory components required
#                 inp_memory = np.empty(inp_mem)
#                 out_memory = np.empty(out_mem)
#                 wrk_memory = np.empty(wrk_mem)
#                 param_lst = np.empty(line_n_params)
#
#                 for ii in range(n_lines):
#                     datum = detection_data[i, ii]
#                     lookup = lookup_make(datum)
#                     for idl, l in enumerate(lookup):
#                         param_lst[idl] = params[int(l)]
#                     # param_lst = params[lookup.astype(int)]
#                     if use_template:
#                         inp_memory[:3] = t_data[int(datum[2])] 
#                     line_loss = per_line_loss(param_lst, inp_memory,
#                                               out_memory, wrk_memory, 
#                                               # loss_fns,
#                                               )
#                     losses[i, ii] = line_loss[:2]
#             return losses
#
#         #have to return a python function that wraps the code in order to get the right result.
#         line_loss_fn = self._make_loss_per_line_function()
#         lookup_fn = self._make_lookup_fn(detection_data=detections)
#         loss_fun = List.empty_list(self.ftemplate)
#
#         for block in self.function_blocks:
#             loss_fun.append(block.compute_fun)# a bunch of function objects to call
#
#         def loss_evaluator(params):
#             return full_loss(params, line_loss_fn, loss_fun, lookup_fn).flatten() 
#
#         return loss_evaluator
    
    def make_jacobean(self, threads):
        jac_maker = lambda x:x
        return jac_maker

    def build_param_list(self, *args: list[np.ndarray])->np.ndarray:
        """
        Takes a list of matrices representing the params to use, which is then turned into a 1D param string as expected by the loss function
        The params have the same order as the functional list
        """
        param_list = []

        for param_chunk in args:
            #check it matches what is expected for the param in terms of dimension.
            #flatten then concatenate to one ginormous array
            param_list.append(param_chunk.flatten())
            # print(param_chunk[0,:])
        return np.concatenate(param_list, axis=0)



class abstract_function_block(ABC):
    """
    The abstract function block defines a chain of methods.
    An optimisation function is defined as chains of abstract function blocks.
    This is similar to ceres.

    The class needs to define how to calculate it's index from a line datum

    How does the class optimise for cached variables?
    """

    array_memory: int = 0 #amount of array memory required to compute - default 0, meaning the computation doesn't "need" array mem
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
       

    def __radd__(self, other: Type[abstract_function_block]|optimisation_function) -> optimisation_function:
        if issubclass(type(other), abstract_function_block):
            return optimisation_function([other, self])
        if isinstance(other, optimisation_function):
            return optimisation_function(other.function_blocks + [self])
        raise ValueError(f"could not combine function block with {other}")


def make_param_struct(
        function_blocks : list[Type[abstract_function_block]], detection_data
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
    max_imgs = np.max(detection_data[:, 1]) + 1
    max_keys = np.max(detection_data[:, 2]) + 1
    max_cams = np.max(detection_data[:, 0]) + 1
       
 
    param_numbers = {
        key_type.PER_CAM:max_cams,
        key_type.PER_IMG:max_imgs,
        key_type.PER_KEY:max_keys,
    }

    print(param_numbers)

    #TODO account for the mod function here, modifying the numbers of params

    # find all of the unique instances of link indicators
    unique_link_inds = []
    for fb in function_blocks:
        if not fb.params in unique_link_inds:
            unique_link_inds.append(fb.params)

    block_param_inds = [unique_link_inds.index(fb.params) for fb in function_blocks]
    param_starts = []
    param_offset = []
    associated = []
    starting_point = 0 
    for link_ind in unique_link_inds:
        param_starts.append(starting_point)
        param_offset.append(link_ind.n_params)
        associated.append(int(link_ind.link_type))
        starting_point += link_ind.n_params * param_numbers[link_ind.link_type] 
    print("testing")
    print(param_offset)
    print(param_starts) 
    print(associated)

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

def make_jacobean_evaluator(function_def: optimisation_function, detections: np.ndarray, threads, template: np.ndarray=None) -> Callable:

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





