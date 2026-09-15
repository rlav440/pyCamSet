from __future__ import annotations
import base64
import logging
import json
import os
import ntpath
import re
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


def check_names_match_keys(cams: CameraSet) -> None:
    """
    Refuses a camera set whose cameras are not named as the set stores them.

    The file is keyed on ``Camera.name``, while a ``CameraSet`` indexes on the
    key it holds a camera under, and nothing forces the two to agree.  A set
    built as ``CameraSet(camera_dict={"cam_0": Camera(extrinsic=e)})`` has
    cameras named ``None``, so every one of them wrote to the same key and the
    file held one camera called ``null`` -- five cameras in, one out, with no
    error raised.  A name that merely disagrees with its key is the same fault
    quietly renaming a camera instead of dropping it.

    :param cams: the camera set about to be written
    :raises ValueError: if any camera's name is not the key it is stored under
    """
    mismatched = [
        (key, cam.name) for key, cam in cams.get_cam_dict().items()
        if cam.name != key
    ]
    if mismatched:
        listing = "\n".join(f"  stored as {key!r}, named {name!r}"
                            for key, name in mismatched)
        raise ValueError(
            "A saved camera is keyed on its own name, so every camera's name "
            "has to be the name its set stores it under. These are not:\n"
            f"{listing}\n"
            "Pass name= when constructing each Camera, or build the set from "
            "parameter lists with CameraSet(camera_names=..., ...), which "
            "names them for you."
        )


def save_camset(
        cams: CameraSet, f_name: Path = Path('cams.camset')
):
    """
    A function to save a CameraSet to a .json formatted file.
    Some useful data, like the optimisation results are also saved.
    These are however compressed, and placed at the base of the file.

    :param cams: The camera set to save
    :param f_name: The file to write too.
    :raises ValueError: if a camera is not named as the set stores it, which
        would silently drop or rename cameras in the file.
    :return:
    """
    check_names_match_keys(cams)

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
    # COLMAP has no telecentric model, and FULL_OPENCV would silently reinterpret
    # a magnification as a focal length and a division coefficient as k1.
    telecentric = [cam.name for cam in cams
                   if type(cam).__name__ == "TelecentricCamera"]
    if telecentric:
        raise ValueError(
            "COLMAP has no telecentric camera model, so "
            f"{', '.join(map(str, telecentric))} cannot be exported to it. "
            "Every COLMAP model is perspective."
        )

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


# ---------------------------------------------------------------------------
# Convert camset to APDe-MVS / ACMMP ("cams/" + pair.txt) format
# ---------------------------------------------------------------------------

def _apde_view_geometry(cams) -> tuple[np.ndarray, np.ndarray]:
    """
    Camera centres and unit viewing directions, in ``cams.get_names()`` order.

    Both come straight off ``Camera.position`` and ``Camera.view``, which
    ``Camera._update_state`` already derives from
    ``cam_to_world = np.linalg.inv(cam.extrinsic)`` -- so no extra extrinsic
    inversion happens here.

    :param cams: pyCamSet CameraSet object
    :return: (centres, directions), each an (N, 3) array ordered like cams.get_names()
    """
    cam_names = cams.get_names()
    centres = np.array([cams[name].position for name in cam_names])       # world-frame camera centres
    directions = np.array([cams[name].view for name in cam_names])        # world-frame viewing directions
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)  # guard against non-unit input
    return centres, directions


def export_apde_cams(
    cams,
    output_folder: Path,
    depth_min: float = 0.1,
    depth_max: float = 0.8,
    depth_num: int = 192,
) -> None:
    """
    Write one ``cams/%08d_cam.txt`` file per camera, in the ACMMP/MVSNet
    layout that APD-MVS and APDe-MVS read for their ``cams/`` folder.

    File index ``i`` corresponds to ``cams.get_names()[i]``; a
    ``cam_index_map.txt`` is written alongside the ``cams`` folder recording
    that mapping, so the numbering can be matched back up to whichever
    images the caller pairs this export with.

    Each file's extrinsic block is ``cam.extrinsic`` written out unchanged --
    no inversion is applied. pyCamSet stores ``cam.extrinsic`` world-to-camera:
    ``Camera._update_state`` sets ``cam.cam_to_world = np.linalg.inv(cam.extrinsic)``,
    and ``Camera._calc_projection_matrix`` forms the projection matrix as
    ``cam.intrinsic @ cam.extrinsic[:3, :4]`` -- the standard
    ``x = K [R|t] X_world`` form, which only holds if ``[R|t]`` is
    world-to-camera. This is the same convention COLMAP's images.txt uses,
    and the one this module's own :func:`export_rig_config` already relies
    on when it takes ``cam.extrinsic`` directly as "cam-from-world" without
    inverting it. The actual writing is delegated to ``Camera.to_MVSnet_txt``,
    which already produces exactly this file's block structure (including
    the blank lines between sections).

    Distortion is NOT written: APDe-MVS consumes a pinhole model only, and
    pyCamSet cameras may carry a 5-parameter Brown-Conrady model
    (``cam.distortion_coefs``). If any exported camera has non-negligible
    distortion, the resulting cams files are only valid against images that
    have already been undistorted with that camera's model (see
    ``Camera.undistort``). A warning is logged for every such camera rather
    than silently dropping the distortion.

    A CameraSet is a calibration and carries no scene points, so the depth
    range cannot be derived from it. ``depth_min``, ``depth_max`` and
    ``depth_num`` are therefore placeholders -- the defaults chosen here
    (0.1 to 0.8, 192 steps) match this codebase's existing
    ``pyCamSet.reconstruction.acmmp_utils.ReconParams`` defaults for the
    same file format, but are still just a starting point: the caller is
    expected to set values that match their actual scene, in whatever
    world units the CameraSet's extrinsics use.
    ``DEPTH_INTERVAL`` is computed as ``(depth_max - depth_min) / depth_num``,
    inherited unchanged from ``Camera.to_MVSnet_txt``, which this function
    delegates to. Whether the ACMMP/MVSNet convention divides by
    ``depth_num`` or by ``depth_num - 1`` is not settled by the evidence
    found: some derivative repos compute the interval as
    ``(max - min) / num_depth`` while others document the relationship as
    ``DEPTH_MAX = DEPTH_MIN + DEPTH_INTERVAL * (num_depth - 1)``, i.e.
    ``/ (num_depth - 1)`` -- both conventions are in active use across the
    MVSNet family. Left unchanged here: it matches this codebase's existing
    ``Camera.to_MVSnet_txt`` (outside this module), which other code already
    depends on.

    :param cams: pyCamSet CameraSet object
    :param output_folder: directory that will contain the ``cams`` subfolder
    :param depth_min: nearest depth plane, in the CameraSet's world units (placeholder default)
    :param depth_max: furthest depth plane, in the CameraSet's world units (placeholder default)
    :param depth_num: number of depth planes / DEPTH_NUM (192 is APDe-MVS's usual default)
    """
    output_folder = Path(output_folder)                # normalise to Path
    cams_dir = output_folder / "cams"

    # A re-export into the same output dir (the GUI's out_dir is deterministic
    # per run) must not leave stale *_cam.txt files behind from a previous,
    # larger export -- pair.txt and cam_index_map.txt get overwritten below to
    # describe only the new view count, so an orphaned 00000004_cam.txt from a
    # 5-camera export would silently survive a later 3-camera one. Only remove
    # files matching exactly the pattern this function itself writes -- an
    # 8-digit index followed by "_cam.txt" -- and only directly inside cams/,
    # never recursing and never touching the directory itself or anything
    # outside it, so any unrelated file a user placed in cams/ is left alone.
    stale_cam_pattern = re.compile(r"^\d{8}_cam\.txt$")
    if cams_dir.exists():
        for stale_path in cams_dir.iterdir():
            if stale_path.is_file() and stale_cam_pattern.match(stale_path.name):
                stale_path.unlink()
                logger.info("export_apde_cams: removed stale cams file %s", stale_path)

    cams_dir.mkdir(parents=True, exist_ok=True)         # ensure cams/ exists

    cam_names = cams.get_names()                        # deterministic, dict-insertion order

    # --- flag any camera whose distortion would silently invalidate the pinhole export ---
    distorted = [
        name for name in cam_names
        if np.any(np.abs(np.asarray(cams[name].distortion_coefs)) > 1e-9)
    ]
    if distorted:
        logger.warning(
            "camset_to_apde: %d camera(s) carry non-negligible distortion (%s); "
            "the exported pinhole cams files are only valid for images already "
            "undistorted with those cameras' models.",
            len(distorted), ", ".join(distorted),
        )

    index_lines = []
    for idx, name in enumerate(cam_names):
        cam = cams[name]
        cam_path = cams_dir / f"{idx:08d}_cam.txt"       # ACMMP/MVSNet numbering
        cam.to_MVSnet_txt(cam_path, (depth_min, depth_max), depth_num)  # reuse the existing per-camera writer
        index_lines.append(f"{idx:08d} {name}")

    map_path = output_folder / "cam_index_map.txt"
    with open(map_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(index_lines) + "\n")

    logger.info(
        "camset_to_apde: wrote %d cams file(s) to %s; index-to-name map in %s",
        len(cam_names), cams_dir, map_path,
    )
    print(f"Wrote {len(cam_names)} cams file(s) to {cams_dir}")


def _apde_convergence_point(centres: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    The point minimising the sum of squared perpendicular distances to every
    camera's optical axis -- the standard least-squares closest-point-to-N-lines
    solution, used as a stand-in "scene point" for a calibration rig that has
    no reconstructed geometry of its own.

    For camera ``i`` with centre ``c_i`` and unit view direction ``d_i``, the
    axis is the line through ``c_i`` along ``d_i``. The minimising point
    solves the normal-equations system ``A @ p = b`` with::

        A = sum_i (I - d_i d_i^T)
        b = sum_i (I - d_i d_i^T) @ c_i

    ("I - d d^T" projects a vector onto the plane perpendicular to d, so this
    is exactly the least-squares closest point to N lines.)

    For an ordinary inward-facing calibration rig, this is (close to) the
    rig's physical convergence point -- the target volume every camera is
    pointed at -- solved with ``np.linalg.lstsq``. For near-parallel axes
    (e.g. a forward-facing stereo array) the lines barely converge at all,
    and ``A`` becomes singular or numerically unstable; this is detected via
    its singular values (near-zero smallest singular value, or a very large
    condition number) rather than trusted blindly, and a documented fallback
    is used instead: a point in front of the rig's centroid, along the mean
    viewing direction, at the rig's own characteristic scale (the mean
    pairwise camera-centre distance). That is a reasonable stand-in scene
    point for a forward-facing array -- "somewhere out in front of all the
    cameras, roughly where they'd all agree the scene starts" -- and it is
    always finite, never NaN, unlike extrapolating from an ill-conditioned
    solve.

    :param centres: (N, 3) camera centres
    :param directions: (N, 3) unit viewing directions
    :return: (p, well_conditioned) -- the 3-vector point estimate, and
        whether it came from the well-conditioned lstsq solve (True) or the
        degenerate-axes fallback (False).
    """
    eye = np.eye(3)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(centres, directions):
        proj = eye - np.outer(d, d)   # projector onto the plane perpendicular to this camera's axis
        A += proj
        b += proj @ c

    # A well-conditioned 3x3 normal-equations matrix has all three singular
    # values comfortably away from zero. Near-parallel axes make the
    # smallest singular value collapse towards zero (or exactly zero, for
    # exactly-parallel axes) -- np.linalg.lstsq would still return *some*
    # point for a singular/ill-conditioned system, but that point can be
    # arbitrarily unstable, so this is checked explicitly rather than trusted.
    singular_values = np.linalg.svd(A, compute_uv=False)
    well_conditioned = (
        singular_values[-1] > 1e-8
        and (singular_values[0] / singular_values[-1]) < 1e8
    )

    if well_conditioned:
        p, *_ = np.linalg.lstsq(A, b, rcond=None)
        return p, True

    # Degenerate fallback: axes do not usefully converge.
    centroid = centres.mean(axis=0)
    mean_dir = directions.mean(axis=0)
    mean_dir_norm = np.linalg.norm(mean_dir)
    # opposing view directions could average to (near) zero; fall back again
    # to a fixed axis rather than divide by ~0.
    mean_dir = mean_dir / mean_dir_norm if mean_dir_norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    if len(centres) > 1:
        pairwise = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
        scale = pairwise[np.triu_indices(len(centres), k=1)].mean()
    else:
        scale = 0.0
    scale = scale if scale > 1e-9 else 1.0   # single camera / coincident centres: an arbitrary unit standoff
    p = centroid + mean_dir * scale
    return p, False


def _mvsnet_pair_score(theta_deg: np.ndarray, theta0: float = 5.0, sigma1: float = 1.0, sigma2: float = 10.0) -> np.ndarray:
    """
    MVSNet-style (Yao et al., *MVSNet: Depth Inference for Unstructured
    Multi-view Stereo*, ECCV 2018) piecewise-Gaussian score of the angle
    ``theta`` (degrees), subtended at a 3D point, between the rays to two
    cameras. The score peaks at ``theta0`` -- a small but non-degenerate
    triangulation angle -- and falls off on both sides: tightly (``sigma1``)
    towards ``theta = 0``, where triangulation degenerates, and more broadly
    (``sigma2``) towards large ``theta``, where the two views no longer see
    overlapping scene content.

    :param theta_deg: subtended angle(s) in degrees
    :param theta0: peak angle in degrees (MVSNet's usual default: 5)
    :param sigma1: std. dev. for theta <= theta0 (MVSNet's usual default: 1)
    :param sigma2: std. dev. for theta > theta0 (MVSNet's usual default: 10)
    :return: score(s) in (0, 1], same shape as theta_deg
    """
    theta = np.asarray(theta_deg, dtype=float)
    sigma = np.where(theta <= theta0, sigma1, sigma2)
    return np.exp(-0.5 * ((theta - theta0) / sigma) ** 2)


def export_apde_pairs(cams, output_folder: Path) -> None:
    """
    Write ``pair.txt``, scoring every other view as a source-view candidate
    for each view.

    A CameraSet is a calibration, not a reconstruction: there are no scene
    points to compute a real co-visibility or photo-consistency score from.
    The score used here is a rig-geometry heuristic standing in for one --
    it is deliberately not dressed up as a measured MVS co-visibility score
    computed from an actual reconstruction.

    A single stand-in scene point ``p`` is found -- see
    :func:`_apde_convergence_point` -- as the point minimising summed squared
    distance to every camera's optical axis (well-defined and close to the
    physical convergence point for an ordinary inward-facing rig; a
    documented fallback is used when the axes are near-parallel). For each
    pair of views ``(i, j)``, ``theta_ij`` is the angle subtended at ``p``
    between the rays ``p - c_i`` and ``p - c_j``, and the score is the
    MVSNet-style piecewise-Gaussian of that angle (see
    :func:`_mvsnet_pair_score`): peaked at a small baseline angle (a few
    degrees) that triangulates well, and falling off for both too-small
    angles (near-degenerate triangulation) and too-large ones (no shared
    view of ``p``). This replaces an earlier
    ``baseline_ij * max(0, cos(angle between view_i, view_j)))`` score, which
    scored every pair as exactly 0 for an ordinary inward-facing rig: cameras
    that converge on a common point necessarily have *opposing* view
    directions, so that cosine term was always <= 0.

    Every other view is written as a candidate for each view, ranked by that
    score, descending: the caller's MVS pipeline is left to apply its own
    top-k cutoff rather than one being guessed here.

    READ THE ORDER, NOT THE MAGNITUDE. The scores are peaked at a few
    degrees because that is what MVSNet's heuristic is for: image sets
    where neighbouring views are a short step apart. A calibration rig is
    wide-baseline by construction -- cameras tens of degrees apart, often
    ninety -- so every pair lands far out in the tail and the scores come
    out vanishingly small. For four cameras on a ring the neighbours score
    about ``2e-16`` and the opposite view about ``3e-67``. That ranks
    correctly, which is what a top-k cutoff needs, but the numbers are not
    meaningful as weights, and a reader that parses them into a 32-bit
    float will flush the smallest of them to zero (the smallest normal
    float32 is about ``1.2e-38``). They are written in scientific notation
    for that reason: fixed-point would print ``0.000000`` for every pair
    and silently restore the very defect this scoring replaced.

    NOTE ON DUPLICATION: ``pyCamSet.reconstruction.acmmp_utils`` already has a
    pair-file writer for ``CameraSet.write_to_txt`` (``calc_pairs`` +
    ``write_pair_file``), targeting this same ``pair.txt`` format. It is
    a genuinely different implementation, not just a different call site:
    it filters candidates to an angle window (``minangle``/``maxangle``) and
    caps the list at ``max_n_view``, and writes a constant dummy score of 1
    for every surviving candidate rather than a real score. This function
    is unbounded (every other view is written) and scores with the
    MVSNet-style angle heuristic above. Folding them into one shared
    implementation is plausible in principle, but ``calc_pairs``/
    ``write_pair_file`` currently have no test coverage and no other
    production caller, so changing their behaviour is not a free move --
    left as two implementations deliberately, so a future reader who
    notices the overlap is not left wondering whether it was accidental.

    :param cams: pyCamSet CameraSet object
    :param output_folder: directory to write pair.txt into
    """
    output_folder = Path(output_folder)                 # normalise to Path
    output_folder.mkdir(parents=True, exist_ok=True)    # ensure dir exists

    centres, directions = _apde_view_geometry(cams)      # (N,3) each, cams.get_names() order
    n_views = len(centres)

    p, well_conditioned = _apde_convergence_point(centres, directions)
    if not well_conditioned:
        logger.warning(
            "export_apde_pairs: camera axes are near-parallel (the rig does "
            "not converge to a well-defined point); falling back to a point "
            "in front of the rig's centroid, along its mean viewing "
            "direction, at the rig's own mean baseline scale, for pair "
            "scoring.",
        )

    rays = p[np.newaxis, :] - centres                    # (N,3): ray from each camera centre to p
    ray_norms = np.linalg.norm(rays, axis=1, keepdims=True)
    ray_norms = np.where(ray_norms > 1e-12, ray_norms, 1.0)  # guard a camera centre coinciding with p
    unit_rays = rays / ray_norms

    cos_theta = np.clip(unit_rays @ unit_rays.T, -1.0, 1.0)   # (N,N) cosine of the angle at p between ray i, ray j
    theta_deg = np.degrees(np.arccos(cos_theta))
    scores = _mvsnet_pair_score(theta_deg)                # (N,N), same convention as the loop below

    lines = [str(n_views)]
    for i in range(n_views):
        others = [j for j in range(n_views) if j != i]
        others.sort(key=lambda j: scores[i, j], reverse=True)      # best candidate first

        lines.append(str(i))
        # Scientific notation, not fixed-point: a wide-baseline calibration
        # rig routinely has inter-view convergence angles of tens of
        # degrees, well past sigma2 -- the piecewise-Gaussian score for
        # those pairs is a genuinely tiny (but still informative, still
        # correctly ordered) positive number. Fixed-point ".6f" would round
        # anything below 1e-6 to "0.000000", silently recreating the exact
        # all-zeroes illusion this function exists to fix.
        neighbour_terms = " ".join(f"{j} {scores[i, j]:.6e}" for j in others)
        line = f"{len(others)}" if not neighbour_terms else f"{len(others)} {neighbour_terms}"
        lines.append(line)

    pair_path = output_folder / "pair.txt"
    with open(pair_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote pair.txt with {n_views} views to {output_folder}")


def camset_to_apde(
    cams,
    output_folder: Path,
    depth_min: float = 0.1,
    depth_max: float = 0.8,
    depth_num: int = 192,
) -> None:
    """
    Export a pyCamSet CameraSet to APDe-MVS format in a single call.

    Produces:
      - cams/%08d_cam.txt   (per-view extrinsic, intrinsic, depth range)
      - cam_index_map.txt   (index -> camera name, to match against images)
      - pair.txt            (per-view ranked list of candidate source views)

    See :func:`export_apde_cams` and :func:`export_apde_pairs` for the
    conventions each file follows -- in particular, the depth-range
    placeholders and the rig-geometry pair score both need the caller's
    judgement about their own scene; they are not derived from anything a
    calibration alone can provide.

    :param cams: pyCamSet CameraSet object
    :param output_folder: directory to write output files into
    :param depth_min: nearest depth plane (placeholder default; tune per scene)
    :param depth_max: furthest depth plane (placeholder default; tune per scene)
    :param depth_num: number of depth planes / DEPTH_NUM (192 is the usual default)
    """
    output_folder = Path(output_folder)                # normalise to Path

    export_apde_cams(                                  # write cams/*_cam.txt + cam_index_map.txt
        cams, output_folder,
        depth_min=depth_min, depth_max=depth_max, depth_num=depth_num,
    )
    export_apde_pairs(cams, output_folder)              # write pair.txt
