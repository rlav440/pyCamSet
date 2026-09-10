from __future__ import annotations
import base64
import logging
import json
import os
import ntpath
import numpy as np
from scipy.spatial.transform import Rotation as R
import blosc
import dill

from pathlib import Path

import importlib
from copy import copy

from pyCamSet.utils.calibration_report import CalibrationReport

logger = logging.getLogger(__name__)


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pyCamSet.cameras import CameraSet


def _normalise_windows_open_path(path: Path | str) -> str:
    """
    Return a path string suitable for open() on Windows long paths.
    """
    p_str = os.fspath(path)
    if os.name != "nt" or not ntpath.isabs(p_str) or p_str.startswith("\\\\?\\"):
        return p_str

    # Python can open >260 char paths when prefixed with \\?\ on Windows.
    if len(p_str) >= 248:
        if p_str.startswith("\\\\"):
            return "\\\\?\\UNC\\" + p_str[2:]
        return "\\\\?\\" + p_str
    return p_str

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

    # plain json rather than a compressed blob: the point of the report is
    # that a person can read it, including straight out of the file.
    if getattr(cams, 'calibration_report', None) is not None:
        optim_dict['report'] = cams.calibration_report.to_dict()

    save_path = Path(f_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_normalise_windows_open_path(save_path), 'w', encoding="utf-8", newline="\n") as f:
        json.dump(save_dict, fp=f, indent=4)

    return


def load_CameraSet(f_loc: Path|str) -> CameraSet:
    """
    A function to load a CameraSet from a .json formatted file.

    :param f_loc: The file to load
    :return: A camera set object.
    """

    with open(_normalise_windows_open_path(f_loc), encoding="utf-8") as f:
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
    except Exception as e:
        logger.warning(f"Failed to load detections with reason {e} \n returning just the CameraSet")
        return camset

    try:
        target_config = optim['target_config']
        target = instance_obj(
            target_config['target_module'], target_config['target_name'],
            **target_config['input']
        )
    except Exception as e:
        logger.warning(f"Failed to load calibration target with reason {e}, returning just the CameraSet")
        return camset

    try:
        handler_config = optim['handler_config']

        input_args = dict(
            camset=camset, target=target, detection=detection,
            fixed_params=handler_config['fixed_params'], 
            options=handler_config['options']
        )
        if "missing_poses" in handler_config:
            input_args["missing_poses"] = np.array(handler_config["missing_poses"]).astype(bool)

        handler = instance_obj(
            handler_config['handler_module'], handler_config['handler_name'], **input_args
        )
    except Exception as e:
        logger.warning(f"Failed to intialise the Parameterhandler with reason {e}, returning just the CameraSet")
        return camset

    try:
        camset.calibration_result = decompress(optim['results'])
        # camset.calibration_jac = decompress(optim['jac'])
        camset.calibration_params = np.array(optim['params'])
    except:
        logger.warning("Failed to load calibration data, returning just the CameraSet")
        return camset

    # a file written before the report existed simply has no report
    if 'report' in optim:
        try:
            camset.calibration_report = CalibrationReport.from_dict(
                optim['report'])
        except Exception as e:
            logger.warning(f"Failed to load the calibration report: {e}")

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


# ---------------------------------------------------------------------------
# Convert camset to colmap-readable format
# ---------------------------------------------------------------------------

def rotation_matrix_to_quaternion_wxyz(rot_mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to COLMAP quaternion (w, x, y, z)."""
    r = R.from_matrix(rot_mat)                        # construct scipy Rotation
    quat_xyzw = r.as_quat()                          # scipy returns (x, y, z, w)
    # reorder to COLMAP convention: (w, x, y, z)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def export_cameras_txt(cams, output_folder: Path):
    """
    Write cameras.txt containing one entry per physical camera.

    Camera names, intrinsics, distortion, and resolution are all read
    directly from the CameraSet's internal dictionary — no user input
    beyond the CameraSet object itself is required.

    :param cams: pyCamSet CameraSet object
    :param output_folder: directory to write cameras.txt into
    """
    output_folder = Path(output_folder)               # normalise to Path
    output_folder.mkdir(parents=True, exist_ok=True)  # ensure dir exists

    cam_names = cams.get_names()                      # keys of _cam_dict

    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cam_names)}",
    ]

    for idx, cam_name in enumerate(cam_names):
        cam_id = idx + 1                              # COLMAP uses 1-indexed IDs
        cam = cams[cam_name]                          # retrieve Camera object

        # --- intrinsics from the 3x3 K matrix ---
        K = cam.intrinsic                             # pyCamSet pinhole K
        fx, fy = K[0, 0], K[1, 1]                    # focal lengths in pixels
        cx, cy = K[0, 2], K[1, 2]                    # principal point

        # --- resolution (pyCamSet stores [width, height]) ---
        width, height = int(cam.res[0]), int(cam.res[1])

        # --- distortion: pyCamSet [k1, k2, p1, p2, k3] ---
        dist = cam.distortion_coefs                   # 5-param Brown-Conrady
        if len(dist) >= 5:
            k1, k2, p1, p2, k3 = dist[:5]
        else:
            padded = list(dist) + [0.0] * (5 - len(dist))
            k1, k2, p1, p2, k3 = padded

        # FULL_OPENCV params: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        # k4, k5, k6 are not modelled by pyCamSet — set to zero
        k4, k5, k6 = 0.0, 0.0, 0.0
        model = "FULL_OPENCV"
        params = (
            f"{fx} {fy} {cx} {cy} "
            f"{k1} {k2} {p1} {p2} {k3} {k4} {k5} {k6}"
        )
        lines.append(f"{cam_id} {model} {width} {height} {params}")

    cameras_path = output_folder / "cameras.txt"      # target file path
    with open(cameras_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")              # write all lines

    print(f"Wrote cameras.txt with {len(cam_names)} cameras to {output_folder}")


def export_rig_config(
    cams,
    output_path: Path,
    ref_cam_name: str | None = None,
):
    """
    Generate a COLMAP rig configuration JSON from pyCamSet relative extrinsics.

    Camera names are read from the CameraSet dictionary keys.  COLMAP's
    ``image_prefix`` for each camera is derived automatically as
    ``"{cam_name}/"`` — matching the pyCamSet convention where images are
    organised into per-camera subfolders named after the camera.

    The reference camera is assigned identity pose in the rig frame.
    All other cameras' poses are expressed relative to it.

    This function avoids deepcopy of the CameraSet, which would fail
    when the calibration handler contains unpicklable objects (e.g.
    cv2.aruco.Dictionary from a ChArUco calibration).

    :param cams: pyCamSet CameraSet object
    :param output_path: path to write the JSON file (e.g. "rig_config.json")
    :param ref_cam_name: name of the camera to use as the rig reference.
                         If None, defaults to the first camera in the set.
    """
    cam_names = cams.get_names()                      # ordered camera names

    # --- default to first camera if no reference specified ---
    if ref_cam_name is None:
        ref_cam_name = cam_names[0]                   # first key in _cam_dict

    # --- validate the reference camera exists ---
    if ref_cam_name not in cam_names:
        raise ValueError(
            f"Reference camera '{ref_cam_name}' not found in CameraSet. "
            f"Available names: {cam_names}"
        )

    # --- compute sensor_from_rig transforms WITHOUT deepcopy ---
    # The rig frame is defined as the reference camera's coordinate system.
    # For any camera C with extrinsic E_c (cam-from-world) and reference
    # camera R with extrinsic E_r (ref-from-world), the sensor_from_rig
    # transform is:
    #
    #   sensor_from_rig = E_c @ inv(E_r)
    #
    # For the reference camera itself, this yields identity.
    ref_ext = cams[ref_cam_name].extrinsic            # 4x4 ref cam-from-world
    ref_ext_inv = np.linalg.inv(ref_ext)              # 4x4 world-from-ref

    rig_cameras = []                                  # list of rig camera entries

    for cam_name in cam_names:
        # --- image_prefix derived from the camera name ---
        # COLMAP matches images via StringStartsWith(image.Name(), prefix).
        # With images in subfolders like "cam_name/step_000.png", the image
        # name stored in the database is "cam_name/step_000.png", so the
        # prefix "cam_name/" selects all images from that camera.
        prefix = f"{cam_name}/"                       # automatic derivation

        entry = {"image_prefix": prefix}              # required for every camera

        if cam_name == ref_cam_name:
            # --- reference sensor: identity pose, no cam_from_rig needed ---
            entry["ref_sensor"] = True
        else:
            # --- non-reference sensor: compute cam_from_rig directly ---
            cam_ext = cams[cam_name].extrinsic        # 4x4 cam-from-world
            sensor_from_rig = cam_ext @ ref_ext_inv   # 4x4 cam-from-ref

            R_mat = sensor_from_rig[:3, :3]           # 3x3 rotation
            t_vec = sensor_from_rig[:3, 3]            # 3x1 translation

            # convert rotation to COLMAP quaternion [w, x, y, z]
            quat = rotation_matrix_to_quaternion_wxyz(R_mat)

            entry["cam_from_rig_rotation"] = quat.tolist()
            entry["cam_from_rig_translation"] = t_vec.tolist()

        rig_cameras.append(entry)

    # --- COLMAP expects a JSON array of rigs, each with a "cameras" key ---
    rig_config = [{"cameras": rig_cameras}]

    output_path = Path(output_path)                   # normalise to Path
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rig_config, f, indent=2)            # write formatted JSON

    # --- summary ---
    non_ref = [n for n in cam_names if n != ref_cam_name]
    print(f"Wrote rig config to {output_path}")
    print(f"  Reference camera: {ref_cam_name}")
    print(f"  Non-reference cameras: {non_ref}")
    print(f"  Image prefixes (auto-derived): "
          f"{[f'{n}/' for n in cam_names]}")


# ---------------------------------------------------------------------------
# Convenience wrapper: export everything needed in one call
# ---------------------------------------------------------------------------
def camset_to_colmap(
    cams,
    output_folder: Path,
    ref_cam_name: str | None = None,
):
    """
    Export a pyCamSet CameraSet to COLMAP format in a single call.

    Produces:
      - cameras.txt       (intrinsics for each physical camera)
      - rig_config.json   (inter-camera geometry for rig constraint)

    :param cams: pyCamSet CameraSet object
    :param output_folder: directory to write output files into
    :param ref_cam_name: optional reference camera name; defaults to the
                         first camera in the set if not provided.
    """
    output_folder = Path(output_folder)               # normalise to Path

    export_cameras_txt(cams, output_folder)           # write cameras.txt
    export_rig_config(                                # write rig_config.json
        cams,
        output_folder / "rig_config.json",
        ref_cam_name=ref_cam_name,
    )
