"""One bad frame must not lose a whole folder's detections, or a Pool.

Stage A (``wj182014z.output``, ``result.dll``) found a rare, clustered
``cv2.error: Unknown C++ exception from OpenCV code`` that is not caused by
aruco2 and can hit any cv2 aruco call in an affected process.  Before the
fix in ``abstract_target.py``, any exception out of a target's
``find_in_image`` aborted the whole ``find_in_imfolder`` call and, with
``threads > 1``, the whole ``multiprocessing.Pool.starmap``.

These tests use a stub target rather than a real detector, so that the
failure is under test control rather than depending on ever actually
reproducing the rare OpenCV exception. Which image fails is read from the
image's own pixel content (a marker value baked into the file each test
writes), not from call order or in-process state: the Pool path
re-instantiates the target in each worker from ``input_args`` alone, so any
state that lived only on the original instance would not be there to
consult, and the same stub must behave identically whichever process
happens to detect the failing image.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

from conftest import UndrawableTarget
from pyCamSet.calibration_targets import AbstractTarget, ImageDetection

#: A marker value no ordinary image index in these tests collides with.
FAIL_MARKER = 250
N_IMAGES = 4
FAIL_IDX = 2  # the third image (0-indexed) is the one made to fail


class _FlakyStubTarget(UndrawableTarget, AbstractTarget):
    """A minimal target whose own detection can be made to fail on command."""

    def __init__(self, fail_marker: int | None = None, raise_type: str = "cv2"):
        super().__init__(inputs=locals())
        self.fail_marker = fail_marker
        self.raise_type = raise_type
        # A trivial single-face point cloud: just enough structure for
        # AbstractTarget's bookkeeping (_process_data -> make_local) to run.
        self.point_data = np.zeros((4, 3))
        self._process_data()

    def find_in_image(self, image, draw=False, camera=None, wait_len=1) -> ImageDetection:
        marker = int(np.asarray(image).reshape(-1)[0])
        if self.fail_marker is not None and marker == self.fail_marker:
            if self.raise_type == "cv2":
                raise cv2.error("synthetic per-image OpenCV failure")
            if self.raise_type == "opencv_valueerror":
                # aruco2's own grid-board detector hands the SAME class of
                # rare, OpenCV-internal failure back as a ValueError rather
                # than a cv2.error, for any native OpenCV assertion beyond
                # the one cornerSubPix message it already recognises (round-3
                # review, P1) -- this is that message's shape, not its
                # content: "OpenCV(<version>) <file>:<line>: error:
                # (<code>:<name>) <what> in function '<func>'".
                raise ValueError(
                    "OpenCV(4.11.0) some/other/file.cpp:123: error: "
                    "(-215:Assertion failed) some other invariant in "
                    "function cv::someOtherFn"
                )
            if self.raise_type == "plain_valueerror":
                # A hand-written ValueError -- a malformed board, a bad
                # argument -- has none of OpenCV's own message shape and
                # must not be isolated: it is a programming/configuration
                # error, not a transient detection failure.
                raise ValueError("synthetic plain programming error")
            raise RuntimeError("synthetic programming error")
        return ImageDetection(keys=np.arange(4), image_points=np.ones((4, 2)))


def _write_images(folder, n=N_IMAGES, fail_idx=FAIL_IDX, fail_marker=FAIL_MARKER):
    """Writes *n* tiny uniform-colour images.

    Image *fail_idx* carries *fail_marker*; every other image carries its
    own index as its marker (never colliding with *fail_marker* for these
    small values of *n*).
    """
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        marker = fail_marker if i == fail_idx else i
        im = np.full((8, 8, 3), marker, dtype=np.uint8)
        cv2.imwrite(str(folder / f"im_{i:02d}.png"), im)
    return folder


def _write_all_failing_images(folder, n=N_IMAGES, fail_marker=FAIL_MARKER):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        im = np.full((8, 8, 3), fail_marker, dtype=np.uint8)
        cv2.imwrite(str(folder / f"im_{i:02d}.png"), im)
    return folder


def _assert_only_the_chosen_image_is_empty(detections, fail_idx=FAIL_IDX, n=N_IMAGES):
    for idx in range(n):
        data = detections.get(global_im_num=idx).get_data()
        if idx == fail_idx:
            assert data is None, f"image {idx} should have no detections"
        else:
            assert data is not None and data.shape[0] == 4, (
                f"image {idx} should be unaffected by image {fail_idx}'s failure"
            )


# --------------------------------------------------------------------------
# threads == 1 -- the single-process path
# --------------------------------------------------------------------------


def test_single_process_isolates_one_bad_image(tmp_path, caplog):
    folder = _write_images(tmp_path / "cam0")
    target = _FlakyStubTarget(fail_marker=FAIL_MARKER, raise_type="cv2")

    with caplog.at_level(logging.WARNING):
        detections = target.find_in_imfolder(folder, cam_names=["cam0"], threads=1)

    _assert_only_the_chosen_image_is_empty(detections)
    assert "cam0" in caplog.text
    assert f"im_{FAIL_IDX:02d}" in caplog.text


# --------------------------------------------------------------------------
# threads > 1 -- the real multiprocessing.Pool path
# --------------------------------------------------------------------------


def test_pool_path_isolates_one_bad_image(tmp_path, caplog):
    folder = _write_images(tmp_path / "cam0")
    target = _FlakyStubTarget(fail_marker=FAIL_MARKER, raise_type="cv2")

    with caplog.at_level(logging.WARNING):
        detections = target.find_in_imfolder(folder, cam_names=["cam0"], threads=2)

    _assert_only_the_chosen_image_is_empty(detections)
    assert "cam0" in caplog.text
    assert f"im_{FAIL_IDX:02d}" in caplog.text


# --------------------------------------------------------------------------
# aruco2's grid-board detector can hand the same rare, OpenCV-internal
# failure back as a ValueError instead of a cv2.error (round-3 review, P1):
# that must be isolated the same way, but only when it has OpenCV's own
# native message shape -- a plain ValueError must still abort the folder/Pool
# --------------------------------------------------------------------------


def test_single_process_isolates_one_bad_image_raised_as_an_opencv_shaped_valueerror(tmp_path, caplog):
    folder = _write_images(tmp_path / "cam0")
    target = _FlakyStubTarget(fail_marker=FAIL_MARKER, raise_type="opencv_valueerror")

    with caplog.at_level(logging.WARNING):
        detections = target.find_in_imfolder(folder, cam_names=["cam0"], threads=1)

    _assert_only_the_chosen_image_is_empty(detections)
    assert "cam0" in caplog.text
    assert f"im_{FAIL_IDX:02d}" in caplog.text


def test_pool_path_isolates_one_bad_image_raised_as_an_opencv_shaped_valueerror(tmp_path, caplog):
    folder = _write_images(tmp_path / "cam0")
    target = _FlakyStubTarget(fail_marker=FAIL_MARKER, raise_type="opencv_valueerror")

    with caplog.at_level(logging.WARNING):
        detections = target.find_in_imfolder(folder, cam_names=["cam0"], threads=2)

    _assert_only_the_chosen_image_is_empty(detections)
    assert "cam0" in caplog.text
    assert f"im_{FAIL_IDX:02d}" in caplog.text


def test_a_plain_valueerror_still_propagates_single_process(tmp_path):
    folder = _write_images(tmp_path / "cam0")
    target = _FlakyStubTarget(fail_marker=FAIL_MARKER, raise_type="plain_valueerror")

    with pytest.raises(ValueError, match="synthetic plain programming error"):
        target.find_in_imfolder(folder, cam_names=["cam0"], threads=1)


def test_a_plain_valueerror_still_propagates_through_the_pool(tmp_path):
    folder = _write_images(tmp_path / "cam0")
    target = _FlakyStubTarget(fail_marker=FAIL_MARKER, raise_type="plain_valueerror")

    with pytest.raises(ValueError, match="synthetic plain programming error"):
        target.find_in_imfolder(folder, cam_names=["cam0"], threads=2)


# --------------------------------------------------------------------------
# a programming error is not a detector hiccup, and must still surface
# --------------------------------------------------------------------------


def test_a_programming_error_still_propagates_single_process(tmp_path):
    folder = _write_images(tmp_path / "cam0")
    target = _FlakyStubTarget(fail_marker=FAIL_MARKER, raise_type="runtime")

    with pytest.raises(RuntimeError, match="synthetic programming error"):
        target.find_in_imfolder(folder, cam_names=["cam0"], threads=1)


def test_a_programming_error_still_propagates_through_the_pool(tmp_path):
    folder = _write_images(tmp_path / "cam0")
    target = _FlakyStubTarget(fail_marker=FAIL_MARKER, raise_type="runtime")

    with pytest.raises(RuntimeError, match="synthetic programming error"):
        target.find_in_imfolder(folder, cam_names=["cam0"], threads=2)


# --------------------------------------------------------------------------
# every image failing must raise, not return a silently empty detection
# --------------------------------------------------------------------------


def test_every_image_failing_raises_single_process(tmp_path):
    folder = _write_all_failing_images(tmp_path / "cam0")
    target = _FlakyStubTarget(fail_marker=FAIL_MARKER, raise_type="cv2")

    with pytest.raises(RuntimeError, match="every one of"):
        target.find_in_imfolder(folder, cam_names=["cam0"], threads=1)


def test_every_image_failing_raises_through_the_pool(tmp_path):
    folder = _write_all_failing_images(tmp_path / "cam0")
    target = _FlakyStubTarget(fail_marker=FAIL_MARKER, raise_type="cv2")

    with pytest.raises(RuntimeError, match="every one of"):
        target.find_in_imfolder(folder, cam_names=["cam0"], threads=2)


# --------------------------------------------------------------------------
# an unreadable/undecodable image file must be isolated too -- cv2.imread
# does not raise, it returns None, and that None must not reach resize or
# find_in_image
# --------------------------------------------------------------------------

#: which image in a folder is replaced by an unreadable file
BAD_IDX = 1


def _write_images_with_unreadable(tmp_path, kind: str, n=N_IMAGES, bad_idx=BAD_IDX):
    """Writes *n* real images, all readable, except *bad_idx* which is
    replaced by a file that ``cv2.imread`` cannot decode and returns
    ``None`` for (verified empirically: both kinds decode to ``None``, not
    an exception, on this OpenCV build).

    :param kind: ``"text_as_image"`` writes plain text under a ``.png``
        name; ``"truncated_jpeg"`` writes a real JPEG's first bytes only.
    """
    folder = tmp_path / "cam0"
    folder.mkdir(parents=True, exist_ok=True)
    good_marker = 7  # distinct from FAIL_MARKER; every readable image carries it
    for i in range(n):
        if i == bad_idx:
            continue
        im = np.full((8, 8, 3), good_marker, dtype=np.uint8)
        cv2.imwrite(str(folder / f"im_{i:02d}.png"), im)

    if kind == "text_as_image":
        bad_path = folder / f"im_{bad_idx:02d}.png"
        bad_path.write_text("this is not an image, just text\n" * 5, encoding="utf-8")
    elif kind == "truncated_jpeg":
        im = np.full((32, 32, 3), good_marker, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", im)
        assert ok
        bad_path = folder / f"im_{bad_idx:02d}.jpg"
        bad_path.write_bytes(buf.tobytes()[:20])
    else:
        raise ValueError(kind)

    return folder, bad_path, good_marker


def _assert_only_the_bad_image_is_empty(detections, bad_idx, n, good_marker):
    for idx in range(n):
        data = detections.get(global_im_num=idx).get_data()
        if idx == bad_idx:
            assert data is None, f"unreadable image {idx} should have no detections"
        else:
            assert data is not None and data.shape[0] == 4, (
                f"image {idx} should be unaffected by image {bad_idx} being unreadable"
            )


@pytest.mark.parametrize("kind", ["text_as_image", "truncated_jpeg"])
@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("upscale_factor", [1, 2])
def test_unreadable_image_is_isolated(tmp_path, caplog, kind, threads, upscale_factor):
    """A corrupt file must not abort the folder (or the Pool), whichever
    upscale/threading path handles it.

    Before the fix, ``cv2.imread`` returning ``None`` for this file was fed
    straight into ``cv2.resize`` (when ``upscale_factor > 1``, raising
    ``cv2.error`` *outside* the per-image ``try``/``except``) or into
    ``find_in_image`` (when ``upscale_factor == 1``, raising ``TypeError``
    from the stub's own pixel-content lookup, which is not a ``cv2.error``
    either) -- either way, aborting the whole call instead of being isolated
    to this one image.
    """
    folder, bad_path, good_marker = _write_images_with_unreadable(tmp_path, kind)
    # fail_marker=None: the stub itself never fails: any failure recorded
    # here must come from the unreadable file, not from the stub.
    target = _FlakyStubTarget(fail_marker=None)

    with caplog.at_level(logging.WARNING):
        detections = target.find_in_imfolder(
            folder, cam_names=["cam0"], threads=threads, upscale_factor=upscale_factor,
        )

    _assert_only_the_bad_image_is_empty(detections, BAD_IDX, N_IMAGES, good_marker)
    assert "cam0" in caplog.text
    assert bad_path.name in caplog.text


def test_unreadable_image_pool_message_is_not_doubled(tmp_path, caplog):
    """The Pool-path aggregator must report an unreadable file the same way
    the single-process path does, not by nesting the single-process
    message's own "could not read image ..." sentence inside a second,
    differently-worded "detection failed (...)" wrapper.
    """
    folder, bad_path, good_marker = _write_images_with_unreadable(tmp_path, "text_as_image")
    target = _FlakyStubTarget(fail_marker=None)

    with caplog.at_level(logging.WARNING):
        target.find_in_imfolder(folder, cam_names=["cam0"], threads=2)

    messages = [r.message for r in caplog.records if bad_path.name in r.message]
    assert len(messages) == 1, f"expected exactly one warning about {bad_path.name}, got {messages}"
    message = messages[0]
    # The reason must appear once, not doubled inside a second wrapper, and
    # an unreadable file is a read failure, not a "detection failed" one.
    assert message.count("unreadable or undecodable") == 1
    assert message.count(bad_path.name) == 1
    assert "detection failed" not in message
    assert "could not read image" in message


def test_unreadable_image_does_not_mask_a_real_programming_error(tmp_path):
    """The None-check for an unreadable file must not turn into a catch-all
    that also swallows an unrelated programming error elsewhere in the
    folder.
    """
    folder, bad_path, good_marker = _write_images_with_unreadable(tmp_path, "text_as_image")
    target = _FlakyStubTarget(fail_marker=good_marker, raise_type="runtime")

    with pytest.raises(RuntimeError, match="synthetic programming error"):
        target.find_in_imfolder(folder, cam_names=["cam0"], threads=1)


# --------------------------------------------------------------------------
# a folder of unreadable/corrupt files must be diagnosed as a read problem,
# not misreported as "an OpenCV error" (round-2 review, P1): no cv2.error is
# ever raised for an unreadable file, so the folder-level summary must not
# say it was
# --------------------------------------------------------------------------


def _write_all_unreadable_images(folder, n=N_IMAGES):
    """Writes *n* files that ``cv2.imread`` cannot decode (returns ``None``
    for), with no readable image in the folder at all."""
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"im_{i:02d}.png").write_text("not an image\n", encoding="utf-8")
    return folder


@pytest.mark.parametrize("threads", [1, 2])
def test_all_unreadable_images_raise_a_read_error_not_an_opencv_error(tmp_path, caplog, threads):
    """If every image in the folder is unreadable/corrupt (no cv2.error is
    ever raised for this -- cv2.imread just returns None), the folder-level
    warning and the RuntimeError it raises must say the files could not be
    read, not misattribute the failure to "an OpenCV error"/the detector.
    """
    folder = _write_all_unreadable_images(tmp_path / "cam0")
    # fail_marker=None: the stub itself never raises; every failure recorded
    # here must come from the files being unreadable, not from detection.
    target = _FlakyStubTarget(fail_marker=None)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="every one of") as excinfo:
            target.find_in_imfolder(folder, cam_names=["cam0"], threads=threads)

    assert "could not be read" in str(excinfo.value)
    assert "OpenCV error" not in str(excinfo.value)
    # The pre-raise folder-level summary warning must carry the same
    # distinction, not just the final exception message.
    summary_messages = [
        r.message for r in caplog.records
        if r.message.startswith("cam0:") and "of 4 images" in r.message
    ]
    assert len(summary_messages) == 1, summary_messages
    assert "could not be read" in summary_messages[0]
    assert "OpenCV error" not in summary_messages[0]


@pytest.mark.parametrize("threads", [1, 2])
def test_mixed_unreadable_and_detect_error_folder_reports_both_kinds(tmp_path, caplog, threads):
    """A folder with both an unreadable file and a genuine cv2.error during
    detection (and at least one image that succeeds, so the folder does not
    raise) must report both kinds in the summary, not silently attribute
    the unreadable one to "an OpenCV error" too.
    """
    folder = tmp_path / "cam0"
    folder.mkdir(parents=True, exist_ok=True)
    # idx 0: readable, detects fine. idx 1: unreadable (cv2.imread -> None).
    # idx 2: readable, but the stub is made to raise cv2.error on it. idx 3:
    # readable, detects fine. So 1 of each failure kind, 2 successes -- not
    # all 4 fail, so this must warn without raising.
    cv2.imwrite(str(folder / "im_00.png"), np.full((8, 8, 3), 5, dtype=np.uint8))
    (folder / "im_01.png").write_text("not an image\n", encoding="utf-8")
    cv2.imwrite(str(folder / "im_02.png"), np.full((8, 8, 3), FAIL_MARKER, dtype=np.uint8))
    cv2.imwrite(str(folder / "im_03.png"), np.full((8, 8, 3), 6, dtype=np.uint8))
    target = _FlakyStubTarget(fail_marker=FAIL_MARKER, raise_type="cv2")

    with caplog.at_level(logging.WARNING):
        detections = target.find_in_imfolder(folder, cam_names=["cam0"], threads=threads)

    for idx in (0, 3):
        assert detections.get(global_im_num=idx).get_data() is not None
    for idx in (1, 2):
        assert detections.get(global_im_num=idx).get_data() is None

    summary_messages = [
        r.message for r in caplog.records
        if r.message.startswith("cam0:") and "of 4 images" in r.message
    ]
    assert len(summary_messages) == 1, summary_messages
    message = summary_messages[0]
    assert "could not be read" in message
    assert "failed detection with an OpenCV error" in message
