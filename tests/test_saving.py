"""The ``.camset`` persistence layer.

``load_CameraSet`` is one of the five names pyCamSet exports at top level, and
``.camset`` is the format users' calibrations live in, but the module was at 11%
coverage: nothing asserted that a saved camera set loads back as itself.  The
first round-trip written against it failed, because ``Camera.__eq__`` raised
``ValueError`` instead of returning a bool -- so equality, the obvious thing to
assert, was unusable.

These tests need no image corpus: the cameras come from ``synthetic_camset``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyCamSet import Camera, CameraSet, load_CameraSet
from pyCamSet.utils.saving import (
    compress,
    decompress,
    load_pickle,
    numpy_dict_to_list,
    save_camset,
    save_pickle,
)

from conftest import REF_INTRINSIC, make_camera

# --------------------------------------------------------------------------
# Camera and CameraSet equality
#
# The round-trip tests below are only meaningful if equality works, so it is
# pinned first.
# --------------------------------------------------------------------------


def test_camera_equality_is_a_bool_not_an_array():
    """``__eq__`` must return a bool for real (matrix-valued) cameras.

    Regression: the implementation was ``all([np.isclose(a, b), ...])``, which
    raises "truth value of an array with more than one element is ambiguous"
    for any camera whose intrinsic is a 3x3.  Every comparison was an error.
    """
    cam = make_camera("a")
    same = make_camera("b")  # __eq__ ignores the name by design

    result = cam == same
    assert isinstance(result, bool)
    assert result is True


def test_cameras_differing_in_any_parameter_are_unequal():
    """Each of the three compared parameters must be able to break equality."""
    base = make_camera("base")

    moved = make_camera("moved", translation=(0.1, 0.0, 0.0))
    assert not base == moved

    distorted = make_camera("distorted", distortion=[0.1, 0.0, 0.0, 0.0, 0.0])
    assert not base == distorted

    zoomed = make_camera("zoomed")
    zoomed.intrinsic = REF_INTRINSIC * 2.0
    assert not base == zoomed


def test_camera_is_never_equal_to_a_non_camera():
    assert not make_camera("a") == "not a camera"
    assert not make_camera("a") == None  # noqa: E711 -- __eq__ is what is under test


def test_distortion_models_of_different_length_are_unequal():
    """Shape is checked before values, so broadcasting cannot fake equality.

    ``np.allclose`` broadcasts, so a 5 coefficient model and a 1 coefficient
    model of the same value would otherwise compare equal.
    """
    five = make_camera("five", distortion=np.zeros(5))
    one = make_camera("one", distortion=np.zeros(1))
    assert not five == one


def test_camera_set_equality_ignores_camera_order(synthetic_camset):
    """Sets are compared by name, so insertion order must not matter."""
    reversed_dict = dict(reversed(list(synthetic_camset.get_cam_dict().items())))
    assert synthetic_camset == CameraSet(camera_dict=reversed_dict)


def test_camera_set_inequality(synthetic_camset):
    subset = synthetic_camset.make_subset(slice(0, 2))
    assert not synthetic_camset == subset
    assert not synthetic_camset == "not a camera set"


# --------------------------------------------------------------------------
# The round trip
# --------------------------------------------------------------------------


def test_camset_round_trips_through_save_and_load(synthetic_camset, tmp_path):
    """A saved camera set must load back equal to what was saved."""
    target = tmp_path / "rig.camset"
    synthetic_camset.save(target)

    assert target.is_file()
    loaded = load_CameraSet(target)

    assert loaded == synthetic_camset
    assert loaded.get_names() == synthetic_camset.get_names()
    assert len(loaded) == len(synthetic_camset)


def test_round_trip_preserves_every_camera_parameter(synthetic_camset, tmp_path):
    """Equality ignores name and resolution, so check those explicitly.

    ``__eq__`` compares intrinsic, extrinsic and distortion only.  Resolution
    and name round-trip through separate fields in the file and would survive
    an equality assertion untested.
    """
    target = tmp_path / "rig.camset"
    synthetic_camset.save(target)
    loaded = load_CameraSet(target)

    for name in synthetic_camset.get_names():
        before, after = synthetic_camset[name], loaded[name]
        assert after.name == before.name == name
        assert np.allclose(after.intrinsic, before.intrinsic)
        assert np.allclose(after.extrinsic, before.extrinsic)
        assert np.allclose(after.distortion_coefs, before.distortion_coefs)
        # res is written as a list and read back as an array; the values must
        # match even though the container type does not.
        assert np.array_equal(np.asarray(after.res), np.asarray(before.res))


def test_round_trip_preserves_distortion(tmp_path):
    """Non-zero distortion must survive; zeros would hide a dropped field."""
    cam = make_camera("wide", distortion=[-0.31, 0.12, 1e-3, -2e-3, 0.04])
    camset = CameraSet(camera_dict={"wide": cam})

    target = tmp_path / "wide.camset"
    camset.save(target)
    loaded = load_CameraSet(target)

    assert np.allclose(loaded["wide"].distortion_coefs, cam.distortion_coefs)
    assert loaded == camset


def test_round_tripped_cameras_project_identically(synthetic_camset, world_points, tmp_path):
    """The real requirement: the loaded rig must do the same geometry.

    Parameter equality is a proxy; this asserts the thing users depend on.
    """
    target = tmp_path / "rig.camset"
    synthetic_camset.save(target)
    loaded = load_CameraSet(target)

    for name in synthetic_camset.get_names():
        expected = synthetic_camset[name].project_points(world_points)
        assert np.allclose(loaded[name].project_points(world_points), expected)


def test_a_camset_with_no_calibration_still_loads(synthetic_camset, tmp_path, caplog):
    """A rig built by hand has no optimisation data, and must load anyway.

    ``load_CameraSet`` walks the optimisation payload in stages and degrades to
    "just the CameraSet" on the first failure.  A hand-built set has no
    detections at all, so it takes that path on the very first stage -- which
    is the common case for a set assembled in code, and must stay non-fatal.
    """
    target = tmp_path / "bare.camset"
    synthetic_camset.save(target)

    loaded = load_CameraSet(target)

    assert loaded == synthetic_camset
    assert loaded.calibration_handler is None


def test_save_writes_readable_json(synthetic_camset, tmp_path):
    """The format is documented as JSON, so it must be loadable as JSON."""
    target = tmp_path / "rig.camset"
    synthetic_camset.save(target)

    with open(target) as f:
        raw = json.load(f)

    assert set(raw["cams"]) == set(synthetic_camset.get_names())
    assert raw["cam_config"]["cam_name"] == "Camera"
    assert raw["cam_config"]["camset_name"] == "CameraSet"
    for entry in raw["cams"].values():
        assert set(entry) == {"int", "ext", "dst", "res"}


def test_save_camset_default_argument(synthetic_camset, tmp_path, monkeypatch):
    """``save_camset`` writes cams.camset when given no path."""
    monkeypatch.chdir(tmp_path)
    save_camset(synthetic_camset)
    assert (tmp_path / "cams.camset").is_file()


def test_camera_set_save_accepts_str_and_path(synthetic_camset, tmp_path):
    """Both spellings of a path are in the signature, so both must work."""
    as_path = tmp_path / "from_path.camset"
    as_str = tmp_path / "from_str.camset"

    synthetic_camset.save(as_path)
    synthetic_camset.save(str(as_str))

    assert load_CameraSet(as_path) == synthetic_camset
    assert load_CameraSet(str(as_str)) == synthetic_camset


# --------------------------------------------------------------------------
# The blosc compression used for the bulky optimisation arrays
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(10, dtype=np.float64),
        np.arange(24, dtype=np.int32).reshape(2, 3, 4),
        np.zeros((5, 5)),
        np.array([1.5]),
        np.linspace(0, 1, 1000).reshape(10, 10, 10),
    ],
    ids=["1d_float", "3d_int32", "zeros", "single", "large"],
)
def test_compress_decompress_round_trip(arr):
    """Shape, dtype and values must all survive the round trip."""
    restored = decompress(compress(arr))
    assert restored.shape == arr.shape
    assert restored.dtype == arr.dtype
    assert np.array_equal(restored, arr)


def test_compress_preserves_fortran_ordered_arrays():
    """F-ordered input takes a separate branch that transposes twice."""
    arr = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4))
    assert arr.flags["F_CONTIGUOUS"]

    restored = decompress(compress(arr))
    assert restored.shape == arr.shape
    assert np.array_equal(restored, arr)


def test_compress_rejects_object_arrays():
    """dtype=object cannot be compressed and must say so."""
    with pytest.raises(TypeError):
        compress(np.array([{"a": 1}, None], dtype=object))


def test_decompress_into_a_preallocated_array():
    """The prealloc path writes through an existing buffer."""
    arr = np.arange(20, dtype=np.float64).reshape(4, 5)
    save_dict = compress(arr)

    prealloc = np.empty(20, dtype=np.float64)
    restored = decompress(save_dict, prealloc_arr=prealloc)

    assert np.array_equal(restored, arr)
    assert np.array_equal(prealloc.reshape(4, 5), arr)


def test_compress_records_the_metadata_decompress_needs():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    save_dict = compress(arr)

    assert save_dict["shape"] == (2, 3)
    assert save_dict["size"] == 6
    assert save_dict["dtype"] == "float32"
    assert save_dict["f"] is False
    assert save_dict["num_chunk"] == len(save_dict["data"]) == len(save_dict["sizes"])
    # base64 text, so the whole payload stays JSON serialisable
    assert all(isinstance(chunk, str) for chunk in save_dict["data"])
    json.dumps(save_dict)


# --------------------------------------------------------------------------
# The small helpers
# --------------------------------------------------------------------------


def test_pickle_round_trip(tmp_path):
    payload = {"a": np.arange(3), "b": [1, 2, 3], "c": "text"}
    target = tmp_path / "obj.pkl"

    save_pickle(payload, target)
    restored = load_pickle(target)

    assert np.array_equal(restored["a"], payload["a"])
    assert restored["b"] == payload["b"]
    assert restored["c"] == payload["c"]


def test_numpy_dict_to_list_converts_nested_arrays():
    """Used to make handler config JSON serialisable, so nesting must work."""
    payload = {
        "flat": np.arange(3),
        "nested": {"inner": np.eye(2)},
        "untouched": 5,
        "text": "keep",
    }
    result = numpy_dict_to_list(payload)

    assert result["flat"] == [0, 1, 2]
    assert result["nested"]["inner"] == [[1.0, 0.0], [0.0, 1.0]]
    assert result["untouched"] == 5
    assert result["text"] == "keep"
    json.dumps(result)


def test_numpy_dict_to_list_passes_through_non_dicts():
    assert numpy_dict_to_list(5) == 5
    assert numpy_dict_to_list(None) is None
