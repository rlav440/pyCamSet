from __future__ import annotations

import importlib
from pathlib import Path
import inspect
from abc import ABC, abstractmethod 
from dataclasses import dataclass
from enum import IntEnum    
from copy import copy
from scipy.optimize import approx_fprime
from datetime import datetime

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

def find_first_non(stri, char):
    return next((i for i, c in enumerate(stri) if c != char), -1)

def clean_fn_data(fn):
    lbl = fn.split("\n")
    if lbl[0][0] == " ":
        #find the first non " " object
        i_level = find_first_non(lbl[0], " ")
        token_type = " "
    elif lbl[0][0] == "\t":
        #find the first non "\t" object
        i_level = find_first_non(lbl[0], "\t")
        token_type = "\t"
    else:
        raise ValueError
    #gives the indent level
    # print(f"indent level = {i_level}")
    output = []
    text_indent = -1
    for l in lbl: 
        l_text = l[i_level:]
        if len(l_text) == 0:
            continue
        if l_text[0] != token_type:
            continue
        if text_indent == -1:
            poss_score = find_first_non(l_text, token_type)
            if poss_score == -1:
                continue
            text_indent = poss_score
        clean_string = l[(i_level + text_indent):]
        if token_type == " ":
            clean_string.replace(" "*text_indent, "\t")
        if not clean_string[:6] == "return":
            output.append(clean_string)
    return output



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


    def _get_loss_fn_strings(self) -> tuple[set[Import],list[str],list[str]]:
        self._prep_for_computation()

        n_blocks = self.n_blocks

        import_set = set()

        for block in self.function_blocks[::-1]:
            base =  inspect.unwrap(block.compute_fun)
            base_code = inspect.getsource(base)
            base_code = inspect.getsourcefile(base)
            for imp in get_imports(base_code):
                import_set.add(imp)

        
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
            f"params = dense_param_arr[param_slices[2*{i}]:param_slices[2*{i}+1]]",
            ]
            base =  inspect.unwrap(self.function_blocks[i].compute_fun)
            base_code = inspect.getsource(base)
            section = clean_fn_data(base_code)
            end = f"inp[:n_outs[{i}]] = output[:n_outs[{i}]]"
            fn_content.append([prep, section, [end]])
        return import_set, needed_preamble, fn_content

    def make_full_loss_template(self, detections, template, threads) -> Callable:

        def t(l):
            return ["\t" + li for li in l]

        needed_import_set, additional_preamble, content = self._get_loss_fn_strings()
        
        needed_import = []
        for s in needed_import_set:
            # print(s)
            if len(s.module) == 0:
                st = f"import {s.name[0]}" + (f" as {s.alias}" if s.alias is not None else "")
            else:
                st = f"from {s.module[0]} import {s.name[0]}" + (f" as {s.alias}" if s.alias is not None else "")
            needed_import.append(st)
        needed_import.append("\n")


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

            f"param_slices = op_fun.param_slices",

            f"use_template = op_fun.templated and (template is not None)",
            f"if op_fun.templated and not use_template:",
            f'\traise ValueError("A templated optimisation was defined, but no template data was given to create the loss function")',
            f"t_data: np.ndarray = template if use_template else np.zeros(3)",
            f"@njit(parallel=True, fastmath=True)",
            f"def full_loss(inp_params):",
            f"\tlosses = np.empty((n_threads, n_lines, 2))",
            f"\tfor i in prange(n_threads):",
            f"\t\t#make the memory components required",
            f"\t\tinp = np.empty(inp_mem)",
            f"\t\toutput = np.empty(out_mem)",
            f"\t\tmemory = np.empty(wrk_mem)",
            f"\t\tdense_param_arr = np.empty(param_len)",

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

        #first thing to do is to unwrap the individual components:
        flat_content = []
        [[flat_content.extend(item) for item in c] for c in content]
        loss_calc = [ '\t\t\t' + c for c in flat_content] #this really needs to be AST

        postamble = [
            f"\t\t\tlosses[i, ii] = [output[0] - datum[3], output[1] - datum[4]]",
            f"\treturn np.resize(losses, (d_shape[0], 2))",
            f"return full_loss"
        ]

        fn = needed_import + start + t(additional_preamble) + t(preamble) + t(param_slicing) + t(mid_amble) + t(loss_calc) + t(postamble)
        str_fn = "\n".join(fn).replace("\t", "    ")

        loc = Path(__file__)
        with open(loc.parent/"template_functions/loss_fun.py", 'w') as f:
            f.write(str_fn)

        from .template_functions.loss_fun import make_full_loss
        loss_fn = make_full_loss(self, detections, template, threads)
        return loss_fn

    def _get_loss_jac_strings(self):
        self._prep_for_computation()

        n_blocks = self.n_blocks

        import_set = set()

        for block in self.function_blocks[::-1]:
            base =  inspect.unwrap(block.compute_jac)
            base_code = inspect.getsource(base)
            base_code = inspect.getsourcefile(base)
            for imp in get_imports(base_code):
                import_set.add(imp)

        
        needed_preamble = [
            "param_slices = op_fun.param_slices",
            "n_outs = op_fun.n_outs",
            "working_memories = np.array(op_fun.working_memories)",
            "n_blocks = op_fun.n_blocks",


            f"n_params  = np.array(op_fun.n_params)",
            f"n_outs = np.array(op_fun.n_outs)",

        ]

        fn_content = []
        for i in range(n_blocks-1, -1, -1):
            prep = [ 
                f"################# BLOCK {i} #################",
                f"params = dense_param_arr[param_slices[2*{i}]:param_slices[2*{i}+1]]",
                f"per_block[:] = base[:]",
            ]
            base =  inspect.unwrap(self.function_blocks[i].compute_jac)
            base_code = inspect.getsource(base)
            section = clean_fn_data(base_code)
            end = [ #replace this with an inlined function?
                f"############### POPULATING THE JACOBEAN ###########",
                f"out_ind_start = param_slices[2*n_blocks + 2*({i})] ",
                f"out_ind_end = param_slices[2*n_blocks + 2*({i}) + 1]",
                f"n_outputs = out_ind_end - out_ind_start",
                f"\n",
                f"inp_ind_start = param_slices[2*n_blocks + 2*{i} + 2] ",
                f"inp_ind_end = param_slices[2*n_blocks + 2*{i} + 1 + 2]",
                f"n_inputs = inp_ind_end - inp_ind_start",
                f"#write the derivatives of the parameters",
                f"n_param  = n_params[{i}]",
                f"param_start = param_slices[2*{i}]",
                f"param_end = param_slices[2*{i} + 1]",
                f"ll = n_inputs + n_param",
                f"for idc, output_var in enumerate(range(out_ind_start, out_ind_end)):",
                f"\t#write the derivative with respect to the controlling params",
                f"\tper_block[output_var, param_start:param_end] = output[idc*ll:idc*ll + n_param] #envisions this as a dense array",
                f"\t#write the derivative with respect to the inputs",
                f"\tif n_inputs != 0:",
                f"\t\tper_block[output_var, inp_ind_start:inp_ind_end] = output[idc*ll + n_param:(idc + 1)*ll] #envisions this as a dense array",
                f"jac = per_block @ jac",
            ]
            fn_content.append([prep, section, end])


        return import_set, needed_preamble, fn_content

    def make_full_jac_template(self, detections, template, threads) -> Callable:

        def t(l):
            return ["\t" + li for li in l]

        needed_import_set, additional_preamble, content = self._get_loss_fn_strings()
        jac_needed_imports_set, jac_additional_preamble, jac_content = self._get_loss_jac_strings()
        
        needed_import = []
        imports_set = needed_import_set | jac_needed_imports_set
        for s in imports_set:
            # print(s)
            if len(s.module) == 0:
                st = f"import {s.name[0]}" + (f" as {s.alias}" if s.alias is not None else "")
            else:
                st = f"from {s.module[0]} import {s.name[0]}" + (f" as {s.alias}" if s.alias is not None else "")
            needed_import.append(st)
        needed_import.append("\n")


        start = ["from numba import prange",
                "from pyCamSet.optimisation.abstract_function_blocks import make_param_struct",
                 " ",
                "def make_full_jac(op_fun, detections, template, threads):"
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

            f"grad_outputsize = np.max(op_fun.grad_outputsize)",

            "starts, block_n_params, param_inds, key_type = make_param_struct(op_fun.function_blocks, detections)",
            "param_len = np.sum(block_n_params)",
            f"param_slices = op_fun.param_slices",
            f"out_param_start = param_len",

            f"jac_size = param_slices[-1]", 
            f"f_outs = op_fun.n_outs[0]", 
            f"use_template = op_fun.templated and (template is not None)",
            f"if op_fun.templated and not use_template:",
            f'\traise ValueError("A templated optimisation was defined, but no template data was given to create the loss function")',
            f"t_data: np.ndarray = template if use_template else np.zeros(3)",
            f"#workingtag {datetime.now().strftime('%H:%M:%S')}",
            f"@njit(parallel=True,fastmath=True)",
            # f"@njit(parallel=False,fastmath=True)",
            f"def full_jac(inp_params):",
            f"\tp_size = len(inp_params)",
            f"\tout_jac = np.zeros((n_threads, n_lines * f_outs, p_size))", 
            f"\tfor i in prange(n_threads):", #below here is parallel scoped
            f"\t\t#make the memory components required",
            f"\t\tinp = np.empty(inp_mem)",
            f"\t\toutput = np.empty(grad_outputsize)",
            f"\t\tmemory = np.empty(wrk_mem)",
            f"\t\tdense_param_arr = np.empty(param_len)",
            f"\t\tbase = np.eye(jac_size)",
            f"\t\tjac = np.eye(jac_size)",
            f"\t\tper_block = np.eye(jac_size)",
            f"\t\tfor ii in range(n_lines):",
            f"\t\t\tdatum = detection_data[i, ii]",
            f"\t\t\tjac[:] = base[:]",

        ]
        # write the script that populates the array here
        param_slicing = [
            f"\t\t\tfor idb in range(n_blocks):",
            f"\t\t\t\ts_num = datum[key_type[idb]]",
            f"\t\t\t\tp_ind = param_inds[idb]",
            f"\t\t\t\tstart = starts[p_ind] + s_num * block_n_params[p_ind]",
            f"\t\t\t\tdense_param_arr[param_slices[2*idb]:param_slices[2*idb + 1]] = inp_params[int(start):int(start + block_n_params[p_ind])]",
        ]


        mid_amble = [
            f"\t\t\tif use_template:",
            f"\t\t\t\tinp[:3] = t_data[int(datum[2])] ",
        ]

        pre_calcs = [val for pair in zip(jac_content, content) for val in pair]
        total_content = []
        [[total_content.extend(item) for item in c] for c in pre_calcs[:-1]]

        calcs = [ '\t\t\t' + c for c in total_content] #this really needs to be AST

        param_slicing_out = [
            f"\t\t\tfor idb in range(n_blocks):",
            f"\t\t\t\ts_num = datum[key_type[idb]] #the index value of the associated parameter",
            f"\t\t\t\tp_ind = param_inds[idb] # maps the param to it's index in the unique params",
            f"\t\t\t\tstart = starts[p_ind] + s_num * block_n_params[p_ind]",
            f"\t\t\t\tfor i_out in range(f_outs):",
            f"\t\t\t\t\tout_jac[i, ii*f_outs + i_out, int(start):int(start + block_n_params[p_ind])] = jac["
            f"\t\t\t\t\t\tout_param_start + i_out, param_slices[2*idb]:param_slices[2*idb + 1]"
            f"\t\t\t\t\t]"
        ]

        postamble = [
            f"\treturn np.resize(out_jac, (2*d_shape[0], p_size))",
            f"return full_jac"
        ] 

        fn = (
            needed_import + start + t(jac_additional_preamble) +  t(additional_preamble)
                + t(preamble) + t(param_slicing) + t(mid_amble) 
                + t(calcs) + t(param_slicing_out) + t(postamble)
        )
        str_fn = "\n".join(fn).replace("\t", "    ")

        loc = Path(__file__)
        with open(loc.parent/"template_functions/loss_jac.py", 'w') as f:
            f.write(str_fn)

        from .template_functions.loss_jac import make_full_jac
        jac_fn: Callable = make_full_jac(self, detections, template, threads)
        return jac_fn

    
    def make_full_loss_fn(self, detections, threads, template: np.ndarray = None):
        self._prep_for_computation()
        return self.make_full_loss_template(detections, template, threads)

    
    def make_jacobean(self, detections, threads, template: np.ndarray = None):
        self._prep_for_computation()
        func =  self.make_full_jac_template(detections, template, threads)
        return func

    def build_param_list(self, *args: list[np.ndarray])->np.ndarray:
        """
        Takes a list of matrices representing the params to use, which is then turned into a 1D param string as expected by the loss function
        The params have the same order as the functional list
        """
        param_list = []
        # print("param_chunks")
        for param_chunk in args:
            #check it matches what is expected for the param in terms of dimension.
            #flatten then concatenate to one ginormous array
            param_list.append(param_chunk.flatten())
            # print(param_chunk[0,:])
        return np.concatenate(param_list, axis=0)

    def can_make_jac(self):
        return all([hasattr(f, "compute_jac") for f in self.function_blocks])




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

    @staticmethod
    @abstractmethod
    def compute_fun(param, input, output, memory: int | np.ndarray = 0):
        pass

    @staticmethod
    @abstractmethod
    def compute_jac(param, input, output, memory: int | np.ndarray = 0):
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

    def test_self(self):
        params = np.ones(self.params.n_params)

        outsize = (self.params.n_params + self.num_inp) * self.num_out
        jac_output = np.empty(outsize)

        def fn(params):
            output = np.empty(self.num_out)
            self.compute_fun(
                params[:self.params.n_params],
                params[-self.num_inp:],
                output, 
                memory=np.empty(self.array_memory),
            )
            return output

        self.compute_jac(
            params, 
            np.ones(self.num_inp), 
            jac_output, 
            np.empty((self.array_memory))
        )
        given_jac = jac_output.reshape((self.num_out, -1)) 
        numeric_jac = approx_fprime(np.ones(self.num_inp + self.params.n_params), fn)
        error = given_jac - numeric_jac
        assert np.all(error < 1e-4)

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
    #TODO account for the mod function here, modifying the numbers of params

    # find all of the unique instances of link indicators
    #TODO use memory location of link indicators to compare (is) I think
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
    return np.array(param_starts), np.array(param_offset), np.array(block_param_inds), np.array(associated)
    
