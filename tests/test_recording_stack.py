"""The acquisition software's BigTIFF recording stacks read as camera folders.

The fixtures in ``tests/test_data/recording_stacks`` were written by the
acquisition software's own writer, so these tests check the reader against
the real format rather than against a copy of the reader's assumptions.
Every edge case is built from a copy in a temporary folder; the fixtures
themselves are never modified.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import UndrawableTarget
from pyCamSet.calibration_targets import AbstractTarget, ImageDetection
from pyCamSet.utils import recording_stack as rs
from pyCamSet.utils.general_utils import _is_candidate_camera_folder, glob_ims

pytestmark = pytest.mark.data

STEM = "{cam}_20261009_143015"


@pytest.fixture
def stacks(data_dir) -> Path:
    """The fixture folder holding ``cam1``, ``cam2`` and ``cam3``."""
    return data_dir / "recording_stacks"


def _stack(folder: Path, cam: str) -> Path:
    return folder / cam / f"{STEM.format(cam=cam)}.tif"


def _expected(folder: Path, cam: str) -> np.ndarray:
    return np.load(folder / cam / f"{STEM.format(cam=cam)}_expected.npy")


def _copy_camera(stacks: Path, cam: str, destination: Path) -> Path:
    """Copy one fixture camera folder to *destination*, returning its stack."""
    shutil.copytree(stacks / cam, destination)
    return destination / f"{STEM.format(cam=cam)}.tif"


# --------------------------------------------------------------------------
# The reader against the real writer's output
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cam, dtype, pages", [
    ("cam1", np.uint16, [0, 1, 2, 3]),
    ("cam2", np.uint16, [1, 2]),       # page 0 flagged as foreign
    ("cam3", np.uint8, [0, 1, 2]),
])
def test_pages_hold_exactly_the_frames_written(stacks, cam, dtype, pages):
    stack = rs.RecordingStack(_stack(stacks, cam))
    expected = _expected(stacks, cam)
    assert rs.is_recording_stack(stack.path)
    assert stack.finished and stack.closed_cleanly
    assert stack.frames_indexed == len(expected)
    assert stack.usable_pages() == pages
    for page in range(stack.frames_indexed):
        frame = stack.read_frame(page)
        assert frame.dtype == dtype
        np.testing.assert_array_equal(frame, expected[page])


def test_mono12_is_msb_aligned(stacks):
    frame = rs.RecordingStack(_stack(stacks, "cam1")).read_frame(0)
    assert np.all(frame % 16 == 0)


def test_an_unfinished_stack_reads_up_to_its_last_complete_frame(stacks, tmp_path):
    path = _copy_camera(stacks, "cam1", tmp_path / "cam1")
    index = path.with_suffix(".idx")
    index.write_bytes(index.read_bytes()[:-rs.INDEX_RECORD_SIZE])  # drop the END record
    stack = rs.RecordingStack(path)
    assert not stack.finished
    assert stack.closed_cleanly is None
    assert stack.frames_indexed == 4
    with pytest.warns(RuntimeWarning, match="no END record"):
        assert stack.usable_pages() == [0, 1, 2, 3]


def test_a_corrupt_record_ends_the_stack_before_it(stacks, tmp_path):
    path = _copy_camera(stacks, "cam1", tmp_path / "cam1")
    index = path.with_suffix(".idx")
    raw = bytearray(index.read_bytes())
    raw[rs.INDEX_RECORD_SIZE + 9] ^= 0xFF  # record 1's data_offset
    index.write_bytes(bytes(raw))
    stack = rs.RecordingStack(path)
    assert stack.corrupt_at == 1
    assert stack.frames_indexed == 1
    np.testing.assert_array_equal(stack.read_frame(0), _expected(stacks, "cam1")[0])
    with pytest.raises(IndexError):
        stack.read_frame(1)


def test_a_tif_without_a_header_is_not_a_stack(stacks, tmp_path):
    path = _copy_camera(stacks, "cam1", tmp_path / "cam1")
    path.with_suffix(".json").unlink()
    assert not rs.is_recording_stack(path)
    assert glob_ims(tmp_path / "cam1") == []


def test_page_paths_round_trip(stacks):
    stack = _stack(stacks, "cam1")
    page = rs.page_path(stack, 17)
    assert page.name == f"{stack.name}#page000017"
    assert rs.split_page_path(page) == (stack, 17)
    assert rs.split_page_path(stack) is None
    assert rs.is_page_path(page)
    assert rs.frame_file_name(page) == f"{stack.stem}_000017.tiff"


# --------------------------------------------------------------------------
# Listing a camera folder
# --------------------------------------------------------------------------


def test_a_stack_lists_as_its_usable_pages(stacks):
    found = glob_ims(stacks / "cam2")
    stack = _stack(stacks, "cam2").resolve()
    assert found == [rs.page_path(stack, 1), rs.page_path(stack, 2)]


def test_a_folder_of_stacks_and_frames_sorts_by_name_then_page(stacks, tmp_path):
    """A reused folder: two stacks and the legacy per-frame files of a third run."""
    from natsort import natsorted

    folder = tmp_path / "cam3"
    first = _copy_camera(stacks, "cam3", folder)
    second_stem = f"{first.stem}_1"
    for suffix in (".tif", ".idx"):
        shutil.copy(first.with_suffix(suffix), folder / f"{second_stem}{suffix}")
    header = first.with_suffix(".json").read_text(encoding="utf-8")
    header = header.replace(first.stem, second_stem)
    (folder / f"{second_stem}.json").write_text(header, encoding="utf-8")
    frame = _expected(stacks, "cam3")[0]
    for i in (0, 1):
        assert cv2.imwrite(str(folder / f"cam3_20261009_120000_{i:06d}.tiff"), frame)

    names = [p.name for p in natsorted(glob_ims(folder))]
    assert names == [
        "cam3_20261009_120000_000000.tiff",
        "cam3_20261009_120000_000001.tiff",
        f"{first.name}#page000000", f"{first.name}#page000001", f"{first.name}#page000002",
        f"{second_stem}.tif#page000000", f"{second_stem}.tif#page000001",
        f"{second_stem}.tif#page000002",
    ]
    assert names == [p.name for p in sorted(glob_ims(folder))]


def test_a_folder_holding_only_stacks_is_a_camera(stacks):
    assert _is_candidate_camera_folder(stacks / "cam1")


def test_images_are_counted_as_pages(stacks):
    from pyCamSet.workflow.workspace import count_images_in_folder

    assert [count_images_in_folder(stacks / c) for c in ("cam1", "cam2", "cam3")] == [4, 2, 3]


def test_a_newer_format_is_refused_rather_than_skipped(stacks, tmp_path):
    path = _copy_camera(stacks, "cam1", tmp_path / "cam1")
    header = path.with_suffix(".json")
    header.write_text(header.read_text(encoding="utf-8").replace(
        '"format_version": 1', '"format_version": 99'), encoding="utf-8")
    with pytest.raises(rs.StackFormatError, match="newer"):
        glob_ims(tmp_path / "cam1")


# --------------------------------------------------------------------------
# Reading a page is reading the per-frame file it replaces
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cam", ["cam1", "cam3"])
@pytest.mark.parametrize("flags", [cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE])
def test_read_image_matches_the_legacy_frame_file(stacks, tmp_path, cam, flags):
    page = rs.page_path(_stack(stacks, cam), 1)
    legacy = tmp_path / rs.frame_file_name(page)
    assert cv2.imwrite(str(legacy), _expected(stacks, cam)[1])
    from_page = rs.read_image(page, flags)
    from_file = cv2.imread(str(legacy), flags)
    assert from_page.dtype == from_file.dtype
    np.testing.assert_array_equal(from_page, from_file)


@pytest.mark.parametrize("cam", ["cam1", "cam3"])
def test_the_viewers_raw_frame_matches_matplotlib_on_the_legacy_file(stacks, tmp_path, cam):
    import matplotlib.image as mpimg

    page = rs.page_path(_stack(stacks, cam), 2)
    legacy = tmp_path / rs.frame_file_name(page)
    assert cv2.imwrite(str(legacy), _expected(stacks, cam)[2])
    from_page = rs.read_page(page)
    from_file = mpimg.imread(legacy)
    assert from_page.dtype == from_file.dtype
    np.testing.assert_array_equal(from_page, from_file)


def test_an_unindexed_page_reads_as_none(stacks):
    assert rs.read_image(rs.page_path(_stack(stacks, "cam1"), 99)) is None


def test_image_sizes_come_from_the_stack_header(stacks):
    from pyCamSet.utils.saving import image_sizes_from_folder

    # 6 rows x 4 columns, reported (width, height) like every other image
    assert image_sizes_from_folder(stacks) == {"cam1": (4, 6), "cam2": (4, 6), "cam3": (4, 6)}


def test_camera_resolutions_from_a_folder_of_stacks(stacks):
    from conftest import make_camera
    from pyCamSet import CameraSet

    cams = CameraSet(camera_dict={c: make_camera(c) for c in ("cam1", "cam2", "cam3")})
    cams.set_resolutions_from_file(floc=stacks)
    for name in ("cam1", "cam2", "cam3"):
        np.testing.assert_array_equal(cams[name].res, (4, 6))


# --------------------------------------------------------------------------
# Detection through the real entry point
# --------------------------------------------------------------------------


class _PixelEchoTarget(UndrawableTarget, AbstractTarget):
    """Reports the first row of whatever image it is given as its four points.

    What reached the detector is then visible in the detections, even from a
    worker process.
    """

    def __init__(self):
        super().__init__(inputs=locals())
        self.point_data = np.zeros((4, 3))
        self._process_data()

    def find_in_image(self, image, draw=False, camera=None, wait_len=1) -> ImageDetection:
        row = np.asarray(image)[0, :4].reshape(4, -1)[:, 0].astype(float)
        return ImageDetection(keys=np.arange(4), image_points=np.stack([row, row], axis=1))


@pytest.mark.parametrize("threads", [1, 2])
def test_detection_reads_every_usable_page(stacks, tmp_path, threads):
    from pyCamSet.calibration.camera_calibrator import detect_datapoints_in_imfile

    root = tmp_path / "rig"
    for name in ("left", "right"):
        _copy_camera(stacks, "cam1", root / name)

    detections, cam_res = detect_datapoints_in_imfile(
        f_loc=root, calibration_target=_PixelEchoTarget(), caching=False, threads=threads)

    assert cam_res == [(6, 4), (6, 4)]
    expected = _expected(stacks, "cam1")
    data = detections.get_data()
    assert data.shape[0] == 2 * 4 * 4  # two cameras, four pages, four points
    for cam_index in range(2):
        for page in range(4):
            # IMREAD_COLOR keeps the top 8 bits of each MSB-aligned sample
            seen = cv2.imread(str(_legacy_copy(tmp_path, expected[page])), cv2.IMREAD_COLOR)
            rows = data[(data[:, 0] == cam_index) & (data[:, 1] == page)]
            np.testing.assert_array_equal(rows[:, -2], seen[0, :4, 0])


def _legacy_copy(tmp_path: Path, frame: np.ndarray) -> Path:
    path = tmp_path / "legacy_frame.tiff"
    assert cv2.imwrite(str(path), frame)
    return path


@pytest.mark.gui
def test_the_run_results_viewer_shows_a_page(stacks):
    from pyCamSet.gui.run_results import _camera_images, detection_figure

    images = _camera_images(stacks, "cam2")
    assert [rs.split_page_path(p)[1] for p in images] == [1, 2]
    figure = detection_figure(images[0], np.zeros((1, 2)))
    shown = figure.axes[0].images
    assert len(shown) == 1, "the page was reported unreadable"
    np.testing.assert_array_equal(shown[0].get_array(), _expected(stacks, "cam2")[1])
