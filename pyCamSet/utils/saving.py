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
from pyCamSet.reconstruction.acmmp_utils import ReconParams, calc_convergence_pair_scores

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

    Serialises to bytes in memory first, then writes those bytes, rather than
    streaming straight from ``dill.dump`` -- a caller that needs to know
    exactly what was written (e.g. to hash it for a cache identity sidecar,
    without a second, racy read of the file back off disk) gets those same
    bytes back as the return value.

    :param dic: object to save
    :param filename: filename to save to
    :return: the bytes written
    """
    data = dill.dumps(dic)
    with open(_normalise_windows_open_path(filename), 'wb') as f:
        f.write(data)
    return data

def load_pickle(filename):
    """
    Loads an object from a pickle file

    :param filename: filename to load from
    :return: object
    """

    with open(_normalise_windows_open_path(filename), 'rb') as f:
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

# APD-MVS and APDe-MVS (github.com/whoiszzj/APD-MVS, github.com/whoiszzj/APDe-MVS)
# both hard-code `#define MAX_IMAGES 32` and load one reference image plus every
# candidate source image from pair.txt whose score is > 0 -- their reader has no
# top-k cutoff of its own, and neither tool degrades gracefully past MAX_IMAGES:
# main.cpp's per-problem image loader does `if (images.size() > MAX_IMAGES) {
# std::cout << "Can't process so much images: " ...; exit(EXIT_FAILURE); }`.
# camset_to_apde otherwise hands write_to_txt every other camera as a
# candidate, unbounded, so a CameraSet with more than 32 cameras would
# crash both tools on every single reference view. 31 leaves room for the
# one reference image itself.
_APDE_MVS_MAX_SRC_VIEWS = 31


def camset_to_apde(
    cams,
    output_folder: Path,
    depth_min: float = 0.1,
    depth_max: float = 0.8,
    depth_num: int = 192,
    max_src_views: int = _APDE_MVS_MAX_SRC_VIEWS,
) -> None:
    """
    Export a pyCamSet CameraSet to APDe-MVS format in a single call.

    Produces:
      - cams/%08d_cam.txt   (per-view extrinsic, intrinsic, depth range)
      - cam_index_map.txt   (index -> camera name, to match against images)
      - pair.txt            (per-view ranked list of candidate source views)

    The ``cams/`` files and ``pair.txt`` are both written through
    :meth:`~pyCamSet.cameras.camera_set.CameraSet.write_to_txt` -- the same
    writer other MVSNet/ACMMP-format exports use -- rather than a second,
    separate per-camera loop and pair.txt writer. Two things stay specific
    to APDe-MVS and are handled here, around that call, rather than inside
    ``write_to_txt`` itself:

    - the pair score. ``write_to_txt`` would score this rig at its
      convergence point anyway (its ``pair_scoring="auto"`` default picks
      that for a rig whose cameras look at a shared point), but it would
      also cap the list at ``r.max_n_view``, which is a reconstruction
      quality knob rather than the downstream reader's hard limit. So this
      computes
      ``pyCamSet.reconstruction.acmmp_utils.calc_convergence_pair_scores``
      explicitly and passes it as ``write_to_txt``'s ``pair_scores``
      argument, which writes every other view, ranked by that score, capped
      at ``max_src_views`` -- see that parameter below. The scores reaching
      pair.txt are row-normalised, so each reference view's best candidate
      is written as 1.
    - ``cam_index_map.txt``, and removing a previous, larger export's stale
      ``cams/*_cam.txt`` files before writing -- neither has a COLMAP/MVSNet
      analogue, so both stay specific to this exporter.

    Each ``cams/%08d_cam.txt`` file's extrinsic block is ``cam.extrinsic``
    written out unchanged -- no inversion is applied. pyCamSet stores
    ``cam.extrinsic`` world-to-camera: ``Camera._update_state`` sets
    ``cam.cam_to_world = np.linalg.inv(cam.extrinsic)``, and
    ``Camera._calc_projection_matrix`` forms the projection matrix as
    ``cam.intrinsic @ cam.extrinsic[:3, :4]`` -- the standard
    ``x = K [R|t] X_world`` form, which only holds if ``[R|t]`` is
    world-to-camera. This is the same convention COLMAP's images.txt uses,
    and the one this module's own :func:`export_rig_config` already relies
    on when it takes ``cam.extrinsic`` directly as "cam-from-world" without
    inverting it.

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
    inherited unchanged from ``Camera.to_MVSnet_txt``, which ``write_to_txt``
    (and so this function) delegates to per camera. Whether the ACMMP/MVSNet
    convention divides by ``depth_num`` or by ``depth_num - 1`` is not
    settled by the evidence found: some derivative repos compute the
    interval as ``(max - min) / num_depth`` while others document the
    relationship as ``DEPTH_MAX = DEPTH_MIN + DEPTH_INTERVAL * (num_depth - 1)``,
    i.e. ``/ (num_depth - 1)`` -- both conventions are in active use across the
    MVSNet family. Left unchanged here: it matches this codebase's existing
    ``Camera.to_MVSnet_txt`` (outside this module), which other code already
    depends on.

    :param cams: pyCamSet CameraSet object
    :param output_folder: directory to write output files into
    :param depth_min: nearest depth plane (placeholder default; tune per scene)
    :param depth_max: furthest depth plane (placeholder default; tune per scene)
    :param depth_num: number of depth planes / DEPTH_NUM (192 is the usual default)
    :param max_src_views: cap on the number of top-scoring candidates written
        per reference view in pair.txt (default 31, matching
        ``_APDE_MVS_MAX_SRC_VIEWS`` above). Should not be raised without also
        raising ``MAX_IMAGES`` in a matching build of APD-MVS/APDe-MVS.
    :raises ValueError: if ``max_src_views`` is not a non-negative integer --
        this parameter exists specifically to keep the export from crashing
        the downstream tool, so a caller's own bug computing it (e.g. a
        negative value, which ``write_to_txt`` would otherwise take via
        Python's "drop the last few" slice semantics instead of raising)
        should not be able to silently reintroduce that failure mode.
    """
    if not isinstance(max_src_views, int) or isinstance(max_src_views, bool) or max_src_views < 0:
        raise ValueError(f"max_src_views must be a non-negative int, got {max_src_views!r}")

    output_folder = Path(output_folder)                 # normalise to Path
    cams_dir = output_folder / "cams"

    # A re-export into the same output dir (the GUI's out_dir is deterministic
    # per run) must not leave stale *_cam.txt files behind from a previous,
    # larger export -- pair.txt and cam_index_map.txt get overwritten below to
    # describe only the new view count, so an orphaned 00000004_cam.txt from a
    # 5-camera export would silently survive a later 3-camera one. Only remove
    # files matching exactly the pattern write_to_txt itself writes -- an
    # 8-digit index followed by "_cam.txt" -- and only directly inside cams/,
    # never recursing and never touching the directory itself or anything
    # outside it, so any unrelated file a user placed in cams/ is left alone.
    stale_cam_pattern = re.compile(r"^\d{8}_cam\.txt$")
    if cams_dir.exists():
        for stale_path in cams_dir.iterdir():
            if stale_path.is_file() and stale_cam_pattern.match(stale_path.name):
                stale_path.unlink()
                logger.info("camset_to_apde: removed stale cams file %s", stale_path)

    cams_dir.mkdir(parents=True, exist_ok=True)          # ensure cams/ exists; write_to_txt does not create it

    cam_names = cams.get_names()                         # deterministic, dict-insertion order

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

    scores, well_conditioned = calc_convergence_pair_scores(cams)
    if not well_conditioned:
        logger.warning(
            "camset_to_apde: camera axes are near-parallel (the rig does not "
            "converge to a well-defined point); falling back to a point in "
            "front of the rig's centroid, along its mean viewing direction, "
            "at the rig's own mean baseline scale, for pair scoring.",
        )

    n_other_views = len(cam_names) - 1
    if n_other_views > max_src_views:
        logger.warning(
            "camset_to_apde: %d camera(s) exceeds the %d-source-view cap "
            "(max_src_views); pair.txt keeps only the %d best-scoring "
            "candidates per reference view, dropping %d.",
            len(cam_names), max_src_views, max_src_views, n_other_views - max_src_views,
        )

    r = ReconParams(mindist=depth_min, maxdist=depth_max, steps=depth_num)
    cams.write_to_txt(                                   # writes cams/*_cam.txt and pair.txt
        cams_dir, r, pair_scores=scores, max_pair_candidates=max_src_views,
    )

    index_lines = [f"{idx:08d} {name}" for idx, name in enumerate(cam_names)]
    map_path = output_folder / "cam_index_map.txt"
    with open(map_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(index_lines) + "\n")

    logger.info(
        "camset_to_apde: wrote %d cams file(s) to %s; index-to-name map in %s",
        len(cam_names), cams_dir, map_path,
    )
    print(f"Wrote {len(cam_names)} cams file(s) to {cams_dir}")
