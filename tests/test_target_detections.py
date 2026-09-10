"""``ImageDetection`` and ``TargetDetection``: the detection bookkeeping.

Every calibration flows through this array, whose rows are

    | cam | im_num | key ... | data_x | data_y |

so a wrong mask or a wrong axis here silently calibrates against the wrong
correspondences.  At 56% the indexing was largely unexercised, and the
uncovered ``_glomp_buffer`` branch crashed outright on any add onto a
detection that already held data.

The detections are built by hand, so none of this needs the image corpus.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyCamSet.calibration_targets import ImageDetection, TargetDetection

CAMS = ["cam_0", "cam_1", "cam_2"]

# Column layout for a 1-d key: cam, im_num, key, x, y
COL_CAM, COL_IM, COL_KEY = 0, 1, 2


def make_rows(entries):
    """Build a data block from (cam_index, im_num, key, x, y) tuples."""
    return np.array(entries, dtype=float)


@pytest.fixture
def detection():
    """Three cameras, two images, three keys -- not a full cross product.

    cam_2 sees nothing in image 1, and key 2 is only seen by cam_0, so masks
    that accidentally return everything are visible as a wrong count.
    """
    rows = make_rows(
        [
            (0, 0, 0, 10.0, 20.0),
            (0, 0, 1, 11.0, 21.0),
            (0, 0, 2, 12.0, 22.0),
            (0, 1, 0, 13.0, 23.0),
            (1, 0, 0, 14.0, 24.0),
            (1, 1, 1, 15.0, 25.0),
            (2, 0, 1, 16.0, 26.0),
        ]
    )
    return TargetDetection(cam_names=list(CAMS), data=rows)


# --------------------------------------------------------------------------
# ImageDetection
# --------------------------------------------------------------------------


def test_image_detection_with_data():
    det = ImageDetection(keys=[0, 1], image_points=[[1.0, 2.0], [3.0, 4.0]])

    assert det.has_data is True
    assert det.data_len == 2
    assert isinstance(det.keys, np.ndarray)
    assert isinstance(det.image_points, np.ndarray)


def test_an_empty_image_detection_has_no_data():
    """A camera that saw nothing is normal, not an error."""
    assert ImageDetection().has_data is False
    assert ImageDetection(keys=[], image_points=[]).has_data is False


def test_image_detection_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        ImageDetection(keys=[0, 1, 2], image_points=[[1.0, 2.0]])


def test_image_detection_rejects_half_a_detection():
    """Keys without points, or points without keys, is a caller bug."""
    with pytest.raises(ValueError, match="requires both"):
        ImageDetection(keys=[0, 1], image_points=[])
    with pytest.raises(ValueError, match="requires both"):
        ImageDetection(keys=[], image_points=[[1.0, 2.0]])


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_duplicate_camera_names_are_rejected():
    """Names index into the data, so duplicates would be ambiguous."""
    with pytest.raises(ValueError, match="must be unique"):
        TargetDetection(cam_names=["a", "a"])


def test_an_empty_detection_has_no_data():
    assert TargetDetection(cam_names=list(CAMS)).has_data() is False


def test_a_populated_detection_has_data(detection):
    assert detection.has_data() is True
    assert detection.get_data().shape == (7, 5)


def test_construction_copies_the_input_data():
    """The detection must not alias an array the caller still holds."""
    rows = make_rows([(0, 0, 0, 1.0, 2.0)])
    det = TargetDetection(cam_names=list(CAMS), data=rows)
    rows[0, 3] = 999.0
    assert det.get_data()[0, 3] == 1.0


def test_max_ims_counts_from_the_data(detection):
    """Images are zero indexed, so two images means max_ims == 2."""
    assert detection.max_ims == 2


def test_max_ims_can_be_raised_but_not_lowered(detection):
    """The setter is a floor: the data always wins if it holds more images."""
    detection.max_ims = 10
    assert detection.max_ims == 10

    detection.max_ims = 1
    assert detection.max_ims == 2  # the data still has image 1


# --------------------------------------------------------------------------
# get
# --------------------------------------------------------------------------


def test_get_by_cam(detection):
    subset = detection.get(cam="cam_0")
    data = subset.get_data()

    assert data.shape[0] == 4
    assert np.all(data[:, COL_CAM] == 0)


def test_get_by_im_num(detection):
    data = detection.get(global_im_num=1).get_data()

    assert data.shape[0] == 2
    assert np.all(data[:, COL_IM] == 1)


def test_get_by_key(detection):
    data = detection.get(key=[1]).get_data()

    assert data.shape[0] == 3
    assert np.all(data[:, COL_KEY] == 1)


def test_get_preserves_the_camera_names(detection):
    assert detection.get(cam="cam_0").cam_names == CAMS


def test_get_with_no_matches_returns_an_empty_detection(detection):
    """A camera that saw nothing must come back empty, not raise."""
    empty = detection.get(cam="cam_2").get(global_im_num=1)
    assert empty.get_data() is None
    assert empty.has_data() is False


def test_get_rejects_more_than_one_direction(detection):
    with pytest.raises(ValueError, match="only get one item at a time"):
        detection.get(cam="cam_0", global_im_num=1)


def test_get_rejects_an_unknown_direction(detection):
    with pytest.raises(ValueError, match="not a gettable item"):
        detection.get(nonsense=1)


def test_get_by_index_is_not_gettable(detection):
    """``index`` is a delete_row direction only, by design."""
    with pytest.raises(ValueError, match="not a gettable item"):
        detection.get(index=0)


# --------------------------------------------------------------------------
# The list accessors
# --------------------------------------------------------------------------


def test_get_cam_list_returns_one_entry_per_camera(detection):
    per_cam = detection.get_cam_list()

    assert len(per_cam) == len(CAMS)
    assert [d.get_data().shape[0] for d in per_cam] == [4, 2, 1]


def test_get_image_list_returns_one_entry_per_image(detection):
    per_image = detection.get_image_list()

    assert len(per_image) == 2
    assert [d.get_data().shape[0] for d in per_image] == [5, 2]


def test_get_key_list_returns_one_entry_per_unique_key(detection):
    per_key = detection.get_key_list()

    assert len(per_key) == 3
    # every row must land in exactly one bucket
    assert sum(d.get_data().shape[0] for d in per_key) == 7


def test_the_lists_partition_the_data(detection):
    """No row may be dropped or double counted by a split."""
    total = detection.get_data().shape[0]
    assert sum(d.get_data().shape[0] for d in detection.get_cam_list()) == total
    assert sum(d.get_data().shape[0] for d in detection.get_image_list()) == total


# --------------------------------------------------------------------------
# Deletion
# --------------------------------------------------------------------------


def test_delete_row_by_cam(detection):
    remaining = detection.delete_row(cam="cam_0")
    data = remaining.get_data()

    assert data.shape[0] == 3
    assert 0 not in data[:, COL_CAM]


def test_delete_row_by_a_list_of_cams(detection):
    remaining = detection.delete_row(cam=["cam_0", "cam_1"])
    assert remaining.get_data().shape[0] == 1


def test_delete_row_by_im_num(detection):
    remaining = detection.delete_row(global_im_num=0)
    assert remaining.get_data().shape[0] == 2
    assert np.all(remaining.get_data()[:, COL_IM] == 1)


def test_delete_row_by_index(detection):
    remaining = detection.delete_row(index=[0, 1])
    assert remaining.get_data().shape[0] == 5


def test_delete_row_does_not_mutate_the_original(detection):
    before = detection.get_data().shape[0]
    detection.delete_row(cam="cam_0")
    assert detection.get_data().shape[0] == before


def test_delete_row_rejects_an_unknown_direction(detection):
    with pytest.raises(ValueError, match="not a gettable item"):
        detection.delete_row(nonsense=1)


def test_delete_row_rejects_more_than_one_direction(detection):
    with pytest.raises(ValueError, match="only get one item at a time"):
        detection.delete_row(cam="cam_0", global_im_num=1)


def test_delete_col_removes_a_column(detection):
    narrower = detection.delete_col(COL_KEY)
    assert narrower.get_data().shape == (7, 4)


# --------------------------------------------------------------------------
# Adding
# --------------------------------------------------------------------------


def test_add_detection_populates_an_empty_detection():
    det = TargetDetection(cam_names=list(CAMS))
    det.add_detection("cam_1", 0, ImageDetection(keys=[0, 1], image_points=[[1.0, 2.0], [3.0, 4.0]]))

    data = det.get_data()
    assert data.shape == (2, 5)
    assert np.all(data[:, COL_CAM] == 1)
    assert np.all(data[:, COL_IM] == 0)
    assert np.allclose(data[:, 3:], [[1.0, 2.0], [3.0, 4.0]])


def test_add_detection_onto_existing_data_keeps_the_row_layout(detection):
    """Regression: this raised IndexError.

    ``_glomp_buffer`` merged with ``np.append`` and no axis, which ravels both
    operands.  ``_data`` became 1-D and the ``max_ims`` update on the very next
    line raised "too many indices for array".  Only the empty branch worked,
    which is why building a detection from scratch was unaffected.
    """
    before = detection.get_data().shape[0]
    detection.add_detection("cam_2", 1, ImageDetection(keys=[2], image_points=[[99.0, 98.0]]))

    data = detection.get_data()
    assert data.ndim == 2
    assert data.shape == (before + 1, 5)
    assert np.allclose(data[-1], [2, 1, 2, 99.0, 98.0])


def test_add_detection_of_nothing_is_a_no_op(detection):
    before = detection.get_data().shape[0]
    detection.add_detection("cam_0", 0, ImageDetection())
    assert detection.get_data().shape[0] == before


def test_add_detection_updates_max_ims():
    det = TargetDetection(cam_names=list(CAMS))
    det.add_detection("cam_0", 4, ImageDetection(keys=[0], image_points=[[1.0, 2.0]]))
    assert det.max_ims == 5


def test_add_detection_accepts_multidimensional_keys():
    """A Ccube key is (face, row, col), so keys may be 2-d."""
    det = TargetDetection(cam_names=list(CAMS))
    det.add_detection(
        "cam_0",
        0,
        ImageDetection(keys=np.array([[1, 2, 3], [4, 5, 6]]), image_points=[[1.0, 2.0], [3.0, 4.0]]),
    )
    data = det.get_data()
    assert data.shape == (2, 7)  # cam, im, 3 key columns, x, y
    assert np.allclose(data[0, 2:5], [1, 2, 3])


def test_adding_two_detections_concatenates_them(detection):
    other = TargetDetection(cam_names=list(CAMS), data=make_rows([(2, 1, 2, 30.0, 40.0)]))
    combined = detection + other

    assert combined.get_data().shape[0] == 8
    assert combined.cam_names == CAMS


def test_adding_an_empty_detection_changes_nothing(detection):
    empty = TargetDetection(cam_names=list(CAMS))

    assert (detection + empty).get_data().shape[0] == 7
    assert (empty + detection).get_data().shape[0] == 7


def test_adding_two_empty_detections_is_empty():
    a, b = TargetDetection(cam_names=list(CAMS)), TargetDetection(cam_names=list(CAMS))
    assert (a + b).has_data() is False


def test_adding_takes_the_larger_max_ims(detection):
    other = TargetDetection(cam_names=list(CAMS), data=make_rows([(0, 5, 0, 1.0, 2.0)]))
    assert (detection + other).max_ims == 6


def test_adding_detections_with_different_cameras_is_refused(detection):
    """Regression: the error message referenced a non-existent attribute.

    It interpolated ``self.names``, so raising the ValueError raised
    AttributeError instead and the real cause was hidden.
    """
    other = TargetDetection(cam_names=["different"])

    with pytest.raises(ValueError, match="consistent camera names"):
        detection + other


# --------------------------------------------------------------------------
# Sorting
# --------------------------------------------------------------------------


def test_sort_by_cam(detection):
    data = detection.sort("cam").get_data()
    assert np.all(np.diff(data[:, COL_CAM]) >= 0)


def test_sort_by_im_num(detection):
    data = detection.sort("global_im_num").get_data()
    assert np.all(np.diff(data[:, COL_IM]) >= 0)


def test_sort_by_key(detection):
    data = detection.sort("key").get_data()
    assert np.all(np.diff(data[:, COL_KEY]) >= 0)


def test_sort_by_several_keys_respects_the_order(detection):
    """List order is sort priority: cam first, then image within a camera."""
    data = detection.sort(["cam", "global_im_num"]).get_data()

    assert np.all(np.diff(data[:, COL_CAM]) >= 0)
    for cam_index in range(len(CAMS)):
        rows = data[data[:, COL_CAM] == cam_index]
        assert np.all(np.diff(rows[:, COL_IM]) >= 0)


def test_sort_preserves_every_row(detection):
    """Sorting must be a permutation, not a filter."""
    before = detection.get_data()
    after = detection.sort("cam").get_data()

    assert after.shape == before.shape
    assert np.allclose(np.sort(after, axis=0), np.sort(before, axis=0))


def test_sort_in_place_returns_nothing_and_mutates(detection):
    assert detection.sort("cam", inplace=True) is None
    assert np.all(np.diff(detection.get_data()[:, COL_CAM]) >= 0)


def test_sort_not_in_place_leaves_the_original_alone(detection):
    before = detection.get_data().copy()
    detection.sort("cam")
    assert np.allclose(detection.get_data(), before)


def test_sort_rejects_an_unknown_key(detection):
    with pytest.raises(ValueError, match="not an accepted sort key"):
        detection.sort("nonsense")


def test_sort_by_a_multidimensional_key():
    """The multi-column key path scores keys positionally, so it needs a check."""
    det = TargetDetection(cam_names=list(CAMS))
    det.add_detection(
        "cam_0",
        0,
        ImageDetection(
            keys=np.array([[1, 1, 1], [0, 0, 1], [0, 0, 0]]),
            image_points=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ),
    )
    sorted_keys = det.sort("key").get_data()[:, 2:5]
    assert np.allclose(sorted_keys, [[0, 0, 0], [0, 0, 1], [1, 1, 1]])


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def test_features_per_im_per_cam_counts_the_grid(detection):
    block = detection.features_per_im_per_cam()

    assert block.shape == (2, len(CAMS))  # (images, cameras)
    # cam_0 sees three features in image 0 and one in image 1
    assert block[0, 0] == 3
    assert block[1, 0] == 1
    # cam_2 sees one in image 0 and nothing in image 1
    assert block[0, 2] == 1
    assert block[1, 2] == 0
    assert block.sum() == detection.get_data().shape[0]
