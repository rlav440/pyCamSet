from dataclasses import dataclass
import numpy as np
from pathlib import Path
import importlib

# Sparsity is inferred by evaluating each block numerically and testing
# entries against 0 and 1, stochastic, so 100x samples witha constant seed
SPARSITY_PROBE_SAMPLES = 100
SPARSITY_PROBE_SEED = 0

# The chain of block jacobians is left-collapsed into a tree: a list is a
# sum, a tuple a product, a dict one matrix, and Indx where a value came from.


def convert_matrix(mat_samples, size, matnum):
    """
    given jacobian samples of one function block, converts them to the form
    used for encapsulation

    :param mat_samples: an (n_samples, size, size) stack of evaluated jacobians
    :param size: the side length of the jacobian
    :param matnum: the index of the function block the samples came from
    """
    samples = np.asarray(mat_samples)
    if samples.ndim == 2:  # tolerate a single un-stacked sample
        samples = samples[None, ...]
    always_zero = np.all(samples == 0, axis=0)
    always_one = np.all(samples == 1, axis=0)

    output = {}
    for i in range(size):
        for j in range(size):
            if always_zero[i,j]:
                output[(i,j)] = 0
            elif always_one[i,j]:
                output[(i,j)] = 1
            else:
                output[(i,j)] = indx(i,j, matnum)
    return output

def unconvert_matrix(mat_trace, size, og_mat):
    """ 
    A test function to check if the conversion is working
    """

    output = np.empty((size, size))
    for key, value in mat_trace.items():
        if isinstance(value, indx):
            output[key[0], key[1]] = og_mat[value.x, value.y]
        else:
            output[key[0], key[1]] = value
    return output

@dataclass
class indx:
    def __init__(self,x,y,matnum):
        self.x = x
        self.y = y
        self.matnum = matnum
    def __repr__(self):
        return f"idx x={self.x}, y={self.y}, mat={self.matnum}"

def encapsulate_multiplication(mat_0, mat_1, size):
    """
    Given a representation of a, probably sparse, matrix structure, returns the encoding of the matrx that gives:
    0,1 as representations of this number (assumes that a 0 or 1 is constant for the jac terms)
    an idx representation if a float value was found.

    """
    # the m=n.
    output = {}
    for i in range(size):
        for j in range(size):
            elem_rep = []
            #matrix multiplication is the sum of the products of row and column
            for k in range(size):
                d_0 = mat_0[(i,k)] #along the row
                d_1 = mat_1[(k,j)] #down the column
                if d_0 == 0 or d_1 == 0:
                    continue #never write the zero/add nothing to the sum so it's not worth doing.
                elif d_0 == 1:
                    elem = d_1 #we add the other value to the sum, as 1 is multiplicative identity
                elif d_1 == 1: 
                    elem = d_0 #
                else: #now just must be a product of the two elements
                    elem = (d_0, d_1) 
                elem_rep.append(elem)
            #then we need to reduce the elemental representation.
            if len(elem_rep) == 0:
                elem_rep = 0
            elif len(elem_rep) == 1:
                elem_rep = elem_rep[0]

            output[(i,j)] = elem_rep
    return output

def parse_m(elem, m2i) -> str:
    """
    A recursive parser to convert a matrix element into a string representing sums
    """

    if isinstance(elem, indx): #base_case
        return f"d{m2i[elem.matnum][elem.x, elem.y]}"

    if isinstance(elem, tuple):
        return f"{parse_m(elem[0], m2i)}*{parse_m(elem[1], m2i)}"
    if isinstance(elem, list):
        sum_string = "("
        first = True
        for sub_elem in elem:
            if not first:
                sum_string += "+"
            sum_string += f"{parse_m(sub_elem, m2i)}"
            first = False
        sum_string += ")"
        return sum_string
    if isinstance(elem, int):
        return f"{elem}"
    raise TypeError("input was of the wrong type")

def map_outputs_to_jac(output, jac, param_slices, n_blocks, ide, m2i, output_offsets):
    """
    places parameters inside a jacobean given a representation
    """
    #work out where the points go for the matrix
    out_ind_start = param_slices[2*n_blocks + 2*ide] 
    out_ind_end = param_slices[2*n_blocks + 2*ide + 1] 

    inp_ind_start = param_slices[2*n_blocks + 2*ide + 2] 
    inp_ind_end = param_slices[2*n_blocks + 2*ide + 1 + 2]
    n_inputs = inp_ind_end - inp_ind_start
    
    #write the derivatives of the parameters
    param_start = param_slices[2*ide]
    param_end = param_slices[2*ide + 1]
    n_param = param_end - param_start

    ll = n_inputs + n_param
    for idc, output_var in enumerate(range(out_ind_start, out_ind_end)):
        jac[output_var, param_start:param_end] = output[idc*ll:idc*ll + n_param] 

        #also write the mat to ind arr
        for ids, idv in enumerate(range(param_start, param_end)):
            m2i[ide][(output_var, idv)] = idc * ll + ids + output_offsets[ide]

        if n_inputs != 0:
            jac[output_var, inp_ind_start:inp_ind_end] = output[idc*ll + n_param:(idc + 1)*ll]
            for ids, idv in enumerate(range(inp_ind_start, inp_ind_end)):
                m2i[ide][(output_var, idv)] = idc * ll + + n_param + ids + output_offsets[ide]


def input_buffer_width(blocks) -> int:
    """
    Return the number of input-buffer entries the widest block reads.

    :param blocks: the function blocks making up the optimisation
    """
    return max(
        (block.n_inp_read if block.n_inp_read is not None else block.num_inp)
        for block in blocks
    )


def check_block_stays_in_bounds(element, probe_inp_len: int):
    """
    Raises when a block's kernel reads past the input buffer it is given.

    :param element: the function block to check
    :param probe_inp_len: the width of the buffer the probe will pass
    """
    py_jac = getattr(element.compute_jac, "py_func", None)
    if py_jac is None:  # not an njit dispatcher; nothing to cross-check
        return

    outsize = (element.params.n_params + element.num_inp) * element.num_out
    try:
        py_jac(
            params=np.zeros(element.params.n_params),
            inp=np.zeros(probe_inp_len),
            output=np.zeros(outsize),
            memory=np.zeros(element.array_memory),
        )
    except IndexError as err:
        # function_blocks holds instances, but accept a class too.
        name = getattr(element, "__name__", None) or type(element).__name__
        raise RuntimeError(
            f"{name}.compute_jac reads outside the "
            f"{probe_inp_len}-entry input buffer it is given ({err}). Declare "
            f"how many entries the kernel reads with n_inp_read on the block; "
            f"num_inp counts only the differentiable inputs. Left unfixed, "
            f"numba would not report this and the generated jacobian would "
            f"differentiate unrelated memory."
        ) from err


def check_all_params_reach_the_output(out_mat, param_len, n_outputs=2):
    """
    Raises when a parameter has no surviving derivative path to the residuals.

    :param out_mat: the encoded jacobian, keyed by (row, column)
    :param param_len: the number of parameter columns
    :param n_outputs: the number of residual rows the jacobian writes
    """
    dead = [
        col for col in range(param_len)
        if all(out_mat.get((row, col), 0) == 0 for row in range(param_len, param_len + n_outputs))
    ]
    if dead:
        raise RuntimeError(
            f"Generated jacobian is degenerate: {len(dead)} of {param_len} "
            f"function-block parameters have no derivative path to the "
            f"residuals (block-space columns {dead}; these index the combined "
            f"function-block parameter vector, not the full optimisation "
            f"parameter array). The sparsity probe has encoded live derivative "
            f"terms as structural zeros, so those parameters could not be "
            f"optimised. This is a code generation fault, not a data problem "
            f"-- please report it with your platform and numpy/numba versions."
        )


def create_optimisable_compute_flow(opfun, out_name:str, in_name:str):
    """
    multivariable calculus expresses a product of the jacobians of each individual function block (expanded to operate on each function block).
    The below code creates a mapping from a vector containing the outputs of all function blocks to partial derivatives of the overall function.
    
    :param opfun: an optimisation function defined by combinign abstract function block derived classes with defined jacobians.
    :param out_name: the name of the output array to write into.
    
    :returns a list of strings that are lines that can be evaluated to map input arrays to an output.
    """

    opfun._prep_for_computation()


    param_slices, _, _, _ = opfun._get_function_constants()
    mat_size = param_slices[-1] 
    n_blocks = len(opfun.function_blocks)
    param_len = param_slices[2*(n_blocks-1) + 1] 

    out_sizes = [0]
    for element in opfun.function_blocks:
        elem_outsize = (element.params.n_params + element.num_inp) * element.num_out
        out_sizes.append(elem_outsize)
    locs = np.cumsum(out_sizes) #this is actually quite a large array, but still kind of small

    mat_ind_2_derivout_ind = [{} for _ in range(n_blocks)]
    rng = np.random.default_rng(SPARSITY_PROBE_SEED)

    probe_inp_len = input_buffer_width(opfun.function_blocks)
    for element in opfun.function_blocks:
        check_block_stays_in_bounds(element, probe_inp_len)

    matricies = []
    for ide, element in enumerate(opfun.function_blocks):
        outsize = (element.params.n_params + element.num_inp) * element.num_out
        jac_samples = np.empty((SPARSITY_PROBE_SAMPLES, mat_size, mat_size))

        for ids in range(SPARSITY_PROBE_SAMPLES):
            jac = np.eye(mat_size)
            output = np.full(outsize, np.nan)
            element.compute_jac(
                inp=rng.random(probe_inp_len),
                params=rng.random(element.params.n_params),
                output=output,
                memory=np.zeros(element.array_memory),
            )
            #write the permutation into the array.
            map_outputs_to_jac(output, jac, param_slices, n_blocks, ide, mat_ind_2_derivout_ind, locs)
            jac_samples[ids] = jac

        matricies.append(convert_matrix(jac_samples, mat_size, matnum=ide))

    for i in range(len(matricies) -1):
        matricies[i+1] = encapsulate_multiplication(matricies[i], matricies[i+1], mat_size)
        

    out_mat = matricies[i+1]            
    check_all_params_reach_the_output(out_mat, param_len)

    #with this structure in place, what we then need to do is to write the code that converts this to a multiplication
    def jac2ret(ind):
        return f"{ind[0] - param_len},{ind[1]}"
    
    lines_to_write = []
    lhs = ""
    rhs = ""
    first = True
    for i in range(locs[-1]): #this is pulling from the written output array
        if not first:
            lhs += ", "
            rhs += ", "
        lhs += f"d{i}"
        rhs += f"{in_name}[{i}]"
        first = False
    first_string = lhs + " = " + rhs
    lines_to_write.append(first_string)
    lines_to_write.append(f"{out_name}[:] = 0")
    #write some code that produces the string d0, d1 ... dn = output[0], output[1], ... output[n]
    #we know that f_outs will be 2 for this code
    elements_we_care_about = [(param_len, n) for n in range(param_len)] 
    elements_we_care_about += [(param_len + 1, n) for n in range(param_len)] 

    for element in elements_we_care_about:
        val = parse_m(out_mat[element], mat_ind_2_derivout_ind)
        if val == "0":
            continue
        l = f"{out_name}[{jac2ret(element)}] = " + val
        lines_to_write.append(l)
    return lines_to_write

def matmul_get_name(opfun):
    return "matflow_jac_" +  "_".join([str(name.__class__.__name__) for name in opfun.function_blocks])

def write_fun(opfun, lines, input_name, output_name):
    strings = "matflow_jac_" +  "_".join([str(name.__class__.__name__) for name in opfun.function_blocks])
    file_name = "template_functions/" + strings + ".py"
    write_file = (Path(__file__).parent)/file_name
    start_lines = [
        "from numba import njit",
        " ",
        "@njit",
        f"def matflow({input_name}, {output_name}):"
    ]
    lines = ["\t" + l for l in lines]
    fn = start_lines + lines
    un_string = [f.replace("\t", "    ") for f in fn]

    with open(write_file, 'w', encoding="utf-8", newline="\n") as f:
        f.writelines((un + "\n" for un in un_string))

def import_fn(opfun):
    strings = "matflow_jac_" +  "_".join([str(name.__class__.__name__) for name in opfun.function_blocks])
    file_string = 'pyCamSet.optimisation.template_functions.'  + strings
    importlib.invalidate_caches()
    top_module = importlib.import_module(file_string)
    return top_module.matflow
