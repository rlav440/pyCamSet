from __future__ import annotations
import base64
import logging
import json
import numpy as np
import blosc
import dill

from pathlib import Path

import importlib
from copy import copy


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pyCamSet.cameras import CameraSet

def save_pickle(dic, filename):
    """
    Saves an object to a pickle file

    :param dic: object to save
    :param filename: filename to save to
    """
    with open(filename, 'wb') as f:
        dill.dump(dic, f)
    return

def load_pickle(filename):
    """
    Loads an object from a pickle file

    :param filename: filename to load from
    :return: object
    """

    with open(filename, 'rb') as f:
        object_n = dill.load(f)
    return object_n


def instance_obj(class_module, class_name, **kwargs):
    """
    A function to instantiate an object from a module and class name

    :param class_module: The module name
    :param class_name: The class name
    :param kwargs: The keyword arguments to pass to the class
    :return:
    """
    class_var = getattr(importlib.import_module(class_module), class_name)
    return class_var(**kwargs)


def numpy_dict_to_list(d):
    """
    A function to convert numpy arrays in a dictionary to lists

    :param d: the dict to operate over
    :return: a reshaped dict.
    """
    if not isinstance(d, dict):
        return d
    for key, value in d.items():
        if isinstance(value, dict):
            numpy_dict_to_list(value)
        elif isinstance(value, np.ndarray):
            d[key] = value.tolist()
        else:
            pass
    return d


def save_camset(
        cams: CameraSet, f_name: Path = Path('cams.camset')
):
    """
    A function to save a CameraSet to a .json formatted file.
    Some useful data, like the optimisation results are also saved.
    These are however compressed, and placed at the base of the file.

    :param cams: The camera set to save
    :param f_name: The file to write too.
    :return:
    """
    save_dict = {}
    cam_dict = save_dict.setdefault('cams', {})
    cam_config = save_dict.setdefault('cam_config', {})
    cam_config['camset_module'] = cams.__class__.__module__
    cam_config['camset_name'] = cams.__class__.__name__
    cam_config['cam_name'] = cams[0].__class__.__name__

    for cam in cams:
        temp_dict = {
            'int': cam.intrinsic.tolist(),
            'ext': cam.extrinsic.tolist(),
            'dst': cam.distortion_coefs.tolist(),
            'res': np.array(cam.res).tolist(),
        }
        cam_dict[cam.name] = temp_dict

    optim_dict = save_dict.setdefault('optim', {})

    try:
        optim_dict['params'] = cams.calibration_params.tolist()

    except AttributeError:
        pass

    handler = cams.calibration_handler
    handler_config = optim_dict.setdefault('handler_config', {})

    if handler is not None:
        handler_config['handler_module'] = handler.__class__.__module__
        handler_config['handler_name'] = handler.__class__.__name__
        handler_config['fixed_params'] = numpy_dict_to_list(copy(handler.fixed_params))
        handler_config['options'] = handler.problem_opts
        handler_config['co_optimise'] = False #handler.cooptimise
        if handler.missing_poses is not None:
            handler_config['missing_poses'] = handler.missing_poses.astype(int).tolist()

        target_config = optim_dict.setdefault('target_config', {})
        target = handler.target
        target_config['target_name'] = target.__class__.__name__
        target_config['target_module'] = target.__class__.__module__
        try:
            target_config['input'] = target.input_args
        except AttributeError:
            pass

        dtct_config = optim_dict.setdefault('dtct_config', {})
        dtct = handler.detection
        dtct_config['dtct_name'] = dtct.__class__.__name__
        dtct_config['dtct_module'] = dtct.__class__.__module__
        dtct_config['cam_names'] = dtct.cam_names
        dtct_config['max_ims'] = dtct.max_ims
        dtct_config['compressed_data'] = compress(dtct.get_data())

    try:
        optim_dict['results'] = compress(cams.calibration_result.copy())
        optim_dict['jac'] = compress(cams.calibration_jac.copy())
    except AttributeError:
        pass

    with open(f_name, 'w') as f:
        json.dump(save_dict, fp=f, indent=4)

    return


def load_CameraSet(f_loc: Path|str) -> CameraSet:
    """
    A function to load a CameraSet from a .json formatted file.

    :param f_loc: The file to load
    :return: A camera set object.
    """

    with open(f_loc) as f:
        saved_structure = json.load(fp=f)

    # make the camerasets
    cam_dict = {}
    cam_config = saved_structure['cam_config']

    cam_module = 'pyCamSet.cameras.camera'
    camset_module = 'pyCamSet.cameras.camera_set'

    for cam_name, data in saved_structure['cams'].items():
        

        cam_dict[cam_name] = instance_obj(
            cam_module,
            'Camera',
            extrinsic=np.array(data['ext']), intrinsic=np.array(data['int']),
            distortion_coefs=np.array(data['dst']), res=np.array(data['res']),
            name=cam_name)
    camset = instance_obj(
        camset_module,
        'CameraSet',
        camera_dict=cam_dict)

    try:
        optim = saved_structure['optim']
        dtct = optim['dtct_config']
        input_args = {
            'data':decompress(dtct['compressed_data']),
            'cam_names':dtct['cam_names'],
            'max_ims':dtct['max_ims']
        }

        detection = instance_obj(
            dtct['dtct_module'], dtct['dtct_name'], **input_args
        )
    except:
        logging.warning("Failed to load detections, returning just the CameraSet")
        return camset

    try:
        target_config = optim['target_config']
        target = instance_obj(
            target_config['target_module'], target_config['target_name'],
            **target_config['input']
        )
    except:
        logging.warning("Failed to load calibration target, returning just the CameraSet")
        return camset

    # try:
    handler_config = optim['handler_config']

    input_args = dict(
        camset=camset, target=target, detection=detection,
        fixed_params=handler_config['fixed_params'], target_cooptimise=handler_config['co_optimise'],
        options=handler_config['options']
    )
    if "missing_poses" in handler_config:
        input_args["missing_poses"] = np.array(handler_config["missing_poses"]).astype(bool)

    handler = instance_obj(
        handler_config['handler_module'], handler_config['handler_name'], **input_args
    )
    # except:
    #     logging.warning("Failed to intialise the Parameterhandler, returning just the CameraSet")
    #     return camset

    try:
        camset.calibration_result = decompress(optim['results'])
        # camset.calibration_jac = decompress(optim['jac'])
        camset.calibration_params = np.array(optim['params'])
    except:
        logging.warning("Failed to load calibration data, returning just the CameraSet")
        return camset

    camset.calibration_handler = handler
    return camset


def compress(arr, clevel=3, cname='lz4', shuffle=1):
    """
    from https://stackoverflow.com/questions/56708673/python-decompression-relative-performance
    compresses the input array for writing to file

    :param file:  path to file
    :param arr:      numpy nd-array
    :param clevel:   0..9
    :param cname:    blosclz,lz4,lz4hc,snappy,zlib
    :param shuffle:  0-> no shuffle, 1->shuffle,2->bitshuffle
    """
    max_blk_size=100_000_000 #100 MB

    shape=arr.shape
    #dtype np.object is not implemented
    if arr.dtype==object:
        raise(TypeError("dtype object is not implemented"))

    #Handling of fortran ordered arrays (avoid copy)
    is_f_contiguous=False
    if arr.flags['F_CONTIGUOUS']==True:
        is_f_contiguous=True
        arr=arr.T.reshape(-1)
    else:
        arr=np.ascontiguousarray(arr.reshape(-1))

    #Writing
    max_num=max_blk_size//arr.dtype.itemsize
    num_chunks=arr.size//max_num

    if arr.size%max_num!=0:
        num_chunks+=1
    num_write=max_num
    c_arr = []
    sizes = []
    for i in range(num_chunks):

        if max_num*(i+1)>arr.size: #check if the final size is correct
            num_write = arr.size-max_num*i

        c = blosc.compress_ptr(arr[max_num*i:].__array_interface__['data'][0], num_write,
                               arr.dtype.itemsize, clevel=clevel, cname=cname, shuffle=shuffle)
        sizes.append(len(c))
        c_arr.append(str(base64.b64encode(c).decode()))
    save_dict = {
        'shape':shape,
        'size':arr.size,
        'dtype':str(arr.dtype),
        'f':is_f_contiguous,
        'num_chunk':num_chunks,
        'max_num':max_num,
        'data':c_arr,
        'sizes':sizes
    }
    return save_dict


def decompress(save_dict, prealloc_arr=None):
    """
    from https://stackoverflow.com/questions/56708673/python-decompression-relative-performance
    Decompresses the data from a saved dictionary

    :param save_dict: The raw data to decopress.
    :param prealloc_arr: A preallocated array to store the data.
    """
    shape = save_dict['shape']
    arr_size = save_dict['size']
    dtype = save_dict['dtype']
    is_f_contiguous = save_dict['f']
    num_chunks = save_dict['num_chunk']
    max_num = save_dict['max_num']

    if prealloc_arr is None:
        arr=np.empty(arr_size,dtype)
    else:
        if prealloc_arr.flags['F_CONTIGUOUS']==True:
            prealloc_arr=prealloc_arr.T
        if prealloc_arr.flags['C_CONTIGUOUS']!=True:
            raise(TypeError("Contiguous array is needed"))
        arr=np.frombuffer(prealloc_arr.data, dtype=dtype, count=arr_size)

    for i in range(num_chunks):
        size=save_dict['sizes'][i]
        c=save_dict['data'][i]
        blosc.decompress_ptr(base64.b64decode(c),
                             arr[max_num*i:].__array_interface__['data'][0])

    #reshape
    if is_f_contiguous:
        arr=arr.reshape(shape[::-1]).T
    else:
        arr=arr.reshape(shape)
    return arr
