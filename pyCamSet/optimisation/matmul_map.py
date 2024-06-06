from dataclasses import dataclass
import numpy as np
from pathlib import Path
import importlib
from matplotlib import pyplot as plt

from abstract_function_blocks import optimisation_function
#

# we take a function and have some parameters of interest.
# we take the matricies and left collapse them.
# then we can represent the multiplication in some form of hierachical way.
# list represents a sum, tuple represents a product.
# matrix is represented by a dictionairy.
# indx class represents the origin of data

@dataclass
class indx:
    def __init__(self,x,y,matnum):
        self.x = x
        self.y = y
        self.matnum = matnum
    def __repr__(self):
        return f"idx x={self.x}, y={self.y}, mat={self.matnum}"

def convert_matrix(mat, size, matnum):
    """
    given a matrix, converts it to the form used for encapsulation
    """
    output = {}
    for i in range(size):
        for j in range(size):
            if mat[i,j] == 0:
                output[(i,j)] = 0
            elif mat[i,j] == 1:
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
    param_end = param_slices[2**ide + 1]
    n_param = param_end - param_start
    ll = n_inputs + n_param
    for idc, output_var in enumerate(range(out_ind_start, out_ind_end)):
        jac[output_var, param_start:param_end] = output[idc*ll:idc*ll + n_param] 
        #also write the mat to ind arr
        for ids, idv in enumerate(range(param_start, param_end)):
            m2i[ide][(output_var, idv)] = idc * ll + ids + output_offsets[ide]

        if n_inputs != 0:
            jac[output_var, inp_ind_start:inp_ind_end] = output[idc*ll + n_param:(idc + 1)*ll] #envisions this as a dense array
            for ids, idv in enumerate(range(inp_ind_start, inp_ind_end)):
                m2i[ide][(output_var, idv)] = idc * ll + + n_param + ids + output_offsets[ide]


def create_optimisable_compute_flow(opfun: optimisation_function, out_name:str, in_name:str):
    """
    We use multivariate calculus to perpetuate the derivatives of each function block.
    This can be expressed as a product of the jacobians of each individual function block (expanded to operate on each function block).
    While this works, the individual construction of each jacobian involves a lot of busy work + copies.
    The below code creates a mapping from a vector containing the outputs of all function blocks to partial derivatives of the overall function.

    There are some issues: the implementation is lazy and checks for values ==1 or == 0, which will break with complex derivatives.
    It is also recursive, so will hit the recursion limit in python for very long function chains!
    
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
    
    #create and build the matricies
    matricies = []
    for ide, element in enumerate(opfun.function_blocks):
        jac = np.eye(mat_size)
        outsize = (element.params.n_params + element.num_inp) * element.num_out
        output = np.empty(outsize)
        inps = np.random.random(element.num_inp)
        params = np.random.random(element.params.n_params)

        element.compute_jac(
            inp=inps,
            params=params,
            output=output,
            memory=np.empty(element.array_memory),
        )
        #write the permutation into the array.
        map_outputs_to_jac(output, jac, param_slices, n_blocks, ide, mat_ind_2_derivout_ind, locs)
        # og_jac = jac.copy()
        jac = convert_matrix(jac, mat_size, matnum=ide)
        # re_jac = unconvert_matrix(jac, mat_size, og_jac)
        # plt.imshow(np.abs(og_jac - re_jac))
        # plt.show()
        matricies.append(jac)

    for i in range(len(matricies) -1):
        matricies[i+1] = encapsulate_multiplication(matricies[i], matricies[i+1], mat_size)
        
        # print(f"\n\n\n###################### multiplaction {i} ##################")
        # for key, value in matricies[i+1].items():
        #     if (key[0] == param_len) or (key[0] == param_len + 1):
        #     # if not (value ==0 or value==1):
        #         print(f"{key} = {value}")

    out_mat = matricies[i+1]            
    
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

    with open(write_file, 'w') as f:
        f.writelines((un + "\n" for un in un_string))

def import_fn(opfun):
    strings = "matflow_jac_" +  "_".join([str(name.__class__.__name__) for name in opfun.function_blocks])
    file_string = 'pyCamSet.optimisation.template_functions.'  + strings
    importlib.invalidate_caches()
    top_module = importlib.import_module(file_string)
    return top_module.matflow


def test_compute_flow(opfun: optimisation_function, out_name:str, in_name:str):
    """
    We use multivariate calculus to perpetuate the derivatives of each function block.
    This can be expressed as a product of the jacobians of each individual function block (expanded to operate on each function block).
    While this works, the individual construction of each jacobian involves a lot of busy work + copies.
    The below code creates a mapping from a vector containing the outputs of all function blocks to partial derivatives of the overall function.

    There are some issues: the implementation is lazy and checks for values ==1 or == 0, which will break with complex derivatives.
    It is also recursive, so will hit the recursion limit in python for very long function chains!
    
    :param opfun: an optimisation function defined by combinign abstract function block derived classes with defined jacobians.
    :param out_name: the name of the output array to write into.
    
    :returns a list of strings that are lines that can be evaluated to map input arrays to an output.
    """

    opfun._prep_for_computation()

    param_slices, _, _, _ = opfun._get_function_constants()
    mat_size = param_slices[-1] 
    n_blocks = len(opfun.function_blocks)
    param_len = param_slices[2*(n_blocks-1) + 1] 

    mat_ind_2_derivout_ind = [{} for _ in range(n_blocks)]
    
    out_sizes = [0]
    for element in opfun.function_blocks:
        elem_outsize = (element.params.n_params + element.num_inp) * element.num_out
        out_sizes.append(elem_outsize)
    locs = np.cumsum(out_sizes)
    output_arr = np.empty(locs[-1])

    #create and build the matricies
    matricies = []
    for ide, element in enumerate(opfun.function_blocks):
        jac = np.eye(mat_size)
        outsize = (element.params.n_params + element.num_inp) * element.num_out
        output = np.empty(outsize)
        inps = np.ones(element.num_inp)
        params = np.ones(element.params.n_params)
        element.compute_jac(
            inp=inps,
            params=params,
            output=output,
            memory=np.empty(element.array_memory),
        )
        #write the permutation into the array.
        output_arr[locs[ide]:locs[ide+1]] = output

        map_outputs_to_jac(output, jac, param_slices, n_blocks, ide, mat_ind_2_derivout_ind, locs)
        matricies.append(jac)

    for i in range(len(matricies) -1):
        matricies[i+1] = matricies[i] @ matricies[i+1]
    out_mat = matricies[i+1]            

    view_elems = out_mat[param_len:param_len+2, :param_len]
    data = np.empty_like(view_elems)
    mapper = import_fn(opfun) 
    mapper(output_arr, data)
   
    # fig, ax = plt.subplots(3,1)
    # ax[0].imshow(view_elems)
    # ax[1].imshow(data)
    # ax[2].imshow(np.abs(view_elems - data))
    # plt.show()
    assert np.all(np.isclose(data, view_elems))


if __name__ == "__main__":
    from numba import njit
    from pyCamSet import load_CameraSet
    from pyCamSet.utils.general_utils import benchmark
    cams = load_CameraSet('tests/test_data/calibration_ccube/self_calib_test.camset')
    opfun = cams.calibration_handler.op_fun

    lines = create_optimisable_compute_flow(opfun, "output", "input")
    write_fun(opfun, lines, 'input', 'output')
    test_compute_flow(opfun, "output", "input")
    temp = import_fn(opfun)
    data = np.empty((2,50))
    inps = np.random.random(69)
    temp(inps,data)
    
    @njit
    def array_mult(n0, n1, n2):
        return n0@n1@n2

    n0 = np.eye(33)
    n1 = np.eye(33)
    n2 = np.eye(33)
    array_mult(n0,n1,n2)
    print("Compiled calculation")
    benchmark(lambda :temp(inps, data), repeats=1000, mode='us')
    print("Matrix multiplication")
    benchmark(lambda :array_mult(n0,n1,n2), repeats=1000, mode='us')


    






