"""Read the panoramic control software's per-camera BigTIFF recording stacks.

The acquisition software (software-panoramic-refactored, ``utils/recording_stack.py``
and ``docs/RECORDING_STACK_FORMAT.md``) records each camera as one uncompressed
little-endian BigTIFF with one page per frame, plus two sibling files that share
its stem::

    cam1_20261009_143015.tif    the stack: page i is frame i
    cam1_20261009_143015.idx    index: one 64-byte record per complete frame
    cam1_20261009_143015.json   header: dtype, shape, layout, camera

The index, not the TIFF chain, says which frames exist: a frame's record is
written only after all of its pixels, so a stack that is still being written,
or whose writer crashed, is read up to its last complete frame. Pixels are read
at the indexed offset with numpy; neither tifffile nor OpenCV is needed. The
pixels are exactly what the legacy one-TIFF-per-frame writer stored (rotated 90
degrees clockwise, Mono12 MSB-aligned in uint16), so a page can stand in for a
legacy frame file unchanged.

Only reading is supported. The writer, live tailing and cross-camera trigger
pairing stay in the acquisition software's reference module.

The format is specified by that software's ``docs/RECORDING_STACK_FORMAT.md``
(commit ``15fff89d9f91c0f0770d5de5c1d5e109f8c9b203``); this reader is kept
identical across the projects that vendor it, so keep a change here in step
with them.
"""

import json
import struct
import warnings
import zlib
from pathlib import Path

import numpy as np

FORMAT_NAME = "panoramic-recording-stack"
#: The newest header ``format_version`` this reader understands.
FORMAT_VERSION = 1
STACK_SUFFIX = ".tif"
INDEX_SUFFIX = ".idx"
HEADER_SUFFIX = ".json"

#: One 64-byte little-endian index record (``struct`` format ``"<QQQQQqQII"``).
INDEX_DTYPE = np.dtype([
    ("page", "<u8"), ("data_offset", "<u8"), ("data_nbytes", "<u8"),
    ("block_id", "<u8"), ("timestamp", "<u8"), ("host_time_ns", "<i8"),
    ("ifd_offset", "<u8"), ("magic", "<u4"), ("crc32", "<u4"),
])
INDEX_RECORD_SIZE = 64
INDEX_MAGIC = 0x31465352  # the bytes b"RSF1"
#: The record a clean or failed close appends: page == END_PAGE, data_nbytes == status.
END_PAGE = 0xFFFFFFFFFFFFFFFF
END_FSYNC_SKIPPED = 4  # closed without the final fsync: not a failure on its own
END_NOT_DURABLE = 2  # an fsync failed: only frames below the durable watermark are on disk

#: The two durable-watermark slots in the stack's prefix (header ``durable_offset``).
_DURABLE_SLOT = struct.Struct("<4sIQQqI12x")
_DURABLE_MAGIC = b"RSDW"
_DURABLE_SLOTS = 2


class StackFormatError(ValueError):
    """The files are not a recording stack this reader can read."""


def header_path(path):
    """The ``.json`` header beside a stack file (``.tif``, ``.idx`` or ``.json``),
    or ``None`` when there is none or it is not a recording-stack header."""
    path = Path(path)
    if path.suffix.lower() not in (STACK_SUFFIX, ".tiff", INDEX_SUFFIX, HEADER_SUFFIX):
        return None
    candidate = path.with_suffix(HEADER_SUFFIX)
    try:
        if not candidate.is_file() or candidate.stat().st_size == 0:
            return None
        with candidate.open("r", encoding="utf-8") as f:
            header = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(header, dict) or header.get("format") != FORMAT_NAME:
        return None
    return candidate


def is_recording_stack(path):
    """True when ``path`` is the ``.tif`` of a recording stack (its ``.json``
    header names the format and its ``.idx`` exists)."""
    path = Path(path)
    if path.suffix.lower() != STACK_SUFFIX or not path.is_file():
        return False
    header = header_path(path)
    return header is not None and header.with_suffix(INDEX_SUFFIX).is_file()


def find_stacks(folder):
    """The recording stacks directly in ``folder``, sorted by name (the stem holds
    the start time, so this is oldest first for one camera)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if is_recording_stack(p))


class RecordingStack:
    """A snapshot of one recording stack: the frames indexed when it was opened.

    :param path: the stack's ``.tif``, ``.idx`` or ``.json``.
    :raises StackFormatError: when the header is missing, of another format, or
        newer than :data:`FORMAT_VERSION`.
    """

    def __init__(self, path):
        path = Path(path)
        found = header_path(path)
        if found is None:
            raise StackFormatError(f"{path}: no {FORMAT_NAME} header ({path.stem}{HEADER_SUFFIX})")
        with found.open("r", encoding="utf-8") as f:
            self.header = json.load(f)
        version = int(self.header.get("format_version", 0))
        if version > FORMAT_VERSION:
            raise StackFormatError(f"{found}: format version {version} is newer than this "
                                   f"reader ({FORMAT_VERSION})")
        directory = found.parent
        self.header_path = found
        self.path = directory / self.header["stack_file"]
        self.index_path = directory / self.header["index_file"]
        self.camera_name = self.header.get("camera_name")
        self.shape = tuple(int(v) for v in self.header["shape"])
        self.dtype = np.dtype(self.header["dtype"]).newbyteorder("<")
        self.frame_nbytes = int(self.header["frame_nbytes"])
        self.first_data_offset = int(self.header["first_data_offset"])
        self.page_stride = int(self.header["page_stride"])
        if (len(self.shape) != 2
                or self.frame_nbytes != self.shape[0] * self.shape[1] * self.dtype.itemsize):
            raise StackFormatError(f"{found}: shape {self.shape} and dtype {self.dtype} do not "
                                   f"match frame_nbytes {self.frame_nbytes}")
        self.end_status = None
        #: the first index record that is corrupt (records after it do not count), else None
        self.corrupt_at = None
        self.records = self._read_index()

    def _read_index(self):
        """The valid records, counted from the start while each is intact, has the
        right page number and points at a whole frame inside the stack file."""
        raw = self.index_path.read_bytes()
        stack_size = self.path.stat().st_size
        total = len(raw) // INDEX_RECORD_SIZE  # a torn last record is not counted
        good = 0
        for k in range(total):
            chunk = raw[k * INDEX_RECORD_SIZE:(k + 1) * INDEX_RECORD_SIZE]
            magic, crc = struct.unpack_from("<II", chunk, 56)
            if magic != INDEX_MAGIC or crc != (zlib.crc32(chunk[:60]) & 0xFFFFFFFF):
                if k + 1 < total:  # the last record may only be unfinished
                    self.corrupt_at = k
                break
            page, offset, nbytes = struct.unpack_from("<QQQ", chunk, 0)
            if page == END_PAGE:
                self.end_status = int(nbytes)
                break
            if (page != k or offset != self.first_data_offset + page * self.page_stride
                    or nbytes != self.frame_nbytes or offset + nbytes > stack_size):
                self.corrupt_at = k
                break
            good = k + 1
        return np.frombuffer(raw[:good * INDEX_RECORD_SIZE], dtype=INDEX_DTYPE).copy()

    @property
    def frames_indexed(self):
        """Number of complete frames in the stack, flagged ones included."""
        return len(self.records)

    @property
    def finished(self):
        """True once the writer closed the stack (its END record was read)."""
        return self.end_status is not None

    @property
    def closed_cleanly(self):
        """True: closed without a write or durability failure; False: closed after
        one; None: not closed (still being written, or the writer crashed)."""
        if self.end_status is None:
            return None
        return (self.end_status & ~END_FSYNC_SKIPPED) == 0

    @property
    def frames_durable(self):
        """Frames known to be on disk (the durable watermark), or None when the
        header has no watermark."""
        offset = self.header.get("durable_offset")
        if offset is None:
            return None
        with self.path.open("rb") as f:
            f.seek(int(offset))
            raw = f.read(_DURABLE_SLOTS * _DURABLE_SLOT.size)
        best = None
        for k in range(_DURABLE_SLOTS):
            slot = raw[k * _DURABLE_SLOT.size:(k + 1) * _DURABLE_SLOT.size]
            if len(slot) != _DURABLE_SLOT.size:
                continue
            magic, _version, frames, seq, _t, crc = _DURABLE_SLOT.unpack(slot)
            if magic != _DURABLE_MAGIC or seq == 0 or crc != (zlib.crc32(slot[:32]) & 0xFFFFFFFF):
                continue
            if best is None or seq > best[0]:
                best = (seq, frames)
        return 0 if best is None else min(int(best[1]), self.frames_indexed)

    @property
    def first_valid_page(self):
        """1 when the acquisition flagged page 0 as possibly not this run's (a
        'foreign' frame from an earlier pulse train), else 0."""
        first = (self.header.get("extra") or {}).get("first_frame") or {}
        return 1 if first.get("flagged") else 0

    def usable_pages(self):
        """The pages to use as images: from :attr:`first_valid_page` to the last
        indexed frame, or only up to the durable watermark when the END record
        says an fsync failed. Warns when the stack is not closed cleanly."""
        stop = self.frames_indexed
        if self.end_status is not None and self.end_status & END_NOT_DURABLE:
            stop = min(stop, self.frames_durable or 0)
            warnings.warn(f"{self.path.name}: the recording was not durably written; using "
                          f"only the {stop} frame(s) known to be on disk", RuntimeWarning,
                          stacklevel=2)
        elif self.closed_cleanly is None:
            warnings.warn(f"{self.path.name}: the recording has no END record (still being "
                          f"written, or the writer stopped); using the {stop} frame(s) "
                          f"complete now", RuntimeWarning, stacklevel=2)
        elif self.closed_cleanly is False:
            warnings.warn(f"{self.path.name}: the recording ended after a write failure "
                          f"(status {self.end_status}); using the {stop} complete frame(s)",
                          RuntimeWarning, stacklevel=2)
        return list(range(min(self.first_valid_page, stop), stop))

    def read_frame(self, page):
        """Page ``page`` as a ``(height, width)`` array of the stack's dtype.

        :raises IndexError: when the page is not indexed.
        """
        page = int(page)
        if not 0 <= page < self.frames_indexed:
            raise IndexError(f"{self.path.name}: page {page} is not indexed "
                             f"({self.frames_indexed} frames)")
        offset = int(self.records["data_offset"][page])
        count = self.shape[0] * self.shape[1]
        frame = np.fromfile(self.path, dtype=self.dtype, count=count, offset=offset)
        if frame.size != count:
            raise OSError(f"{self.path.name}: page {page} is indexed but its pixels are short")
        return frame.reshape(self.shape).astype(self.dtype.newbyteorder("="), copy=False)

    def __len__(self):
        return self.frames_indexed

    def __repr__(self):
        return (f"RecordingStack({self.path.name!r}, frames={self.frames_indexed}, "
                f"shape={self.shape}, dtype={self.dtype.name})")


# ---------------------------------------------------------------------------
# Addressing single pages as image "paths"
# ---------------------------------------------------------------------------

#: A page of a stack is addressed as ``<stack .tif path>#page<NNNNNN>``. No such
#: file exists; :func:`split_page_path` recognises it and :func:`read_image`
#: reads it. ``#`` is legal in file names on every supported OS, so the path
#: survives ``pathlib`` (``resolve``, ``relative_to``, ``as_posix``) unchanged.
PAGE_MARKER = "#page"
_PAGE_DIGITS = 6


def page_path(stack_path, page):
    """The image path that addresses ``page`` of the stack at ``stack_path``."""
    stack_path = Path(stack_path)
    return stack_path.with_name(f"{stack_path.name}{PAGE_MARKER}{int(page):0{_PAGE_DIGITS}d}")


def split_page_path(path):
    """``(stack .tif path, page)`` for a path made by :func:`page_path`, else None."""
    name = Path(path).name
    stack_name, marker, digits = name.rpartition(PAGE_MARKER)
    if not marker or not digits.isdigit() or not stack_name.lower().endswith(STACK_SUFFIX):
        return None
    return Path(path).with_name(stack_name), int(digits)


def is_page_path(path):
    """True when ``path`` addresses a page of a recording stack that exists."""
    split = split_page_path(path)
    return split is not None and is_recording_stack(split[0])


def frame_file_name(path, suffix=".tiff"):
    """The legacy per-frame file name of a page path, ``<stem>_<NNNNNN><suffix>``
    (what the acquisition's ``to-tiffs`` conversion writes); a plain path's own
    name otherwise. Use it wherever a page's image is written out under a name."""
    split = split_page_path(path)
    if split is None:
        return Path(path).name
    return f"{split[0].stem}_{split[1]:0{_PAGE_DIGITS}d}{suffix}"


def stack_page_paths(stack_path):
    """The page paths of a stack's usable pages (see :meth:`RecordingStack.usable_pages`)."""
    stack = open_stack(stack_path)
    return [page_path(stack.path, page) for page in stack.usable_pages()]


_OPEN_STACKS = {}
_OPEN_STACKS_MAX = 64


def open_stack(stack_path):
    """A :class:`RecordingStack` for ``stack_path``, reused while its index file is
    unchanged (a growing stack is re-read when its index grows)."""
    stack_path = Path(stack_path)
    found = header_path(stack_path)
    if found is None:
        raise StackFormatError(f"{stack_path}: not a {FORMAT_NAME} stack")
    index = found.with_suffix(INDEX_SUFFIX)
    stat = index.stat()
    key = str(found.resolve())
    cached = _OPEN_STACKS.get(key)
    if cached is not None and cached[0] == (stat.st_size, stat.st_mtime_ns):
        return cached[1]
    stack = RecordingStack(found)
    if len(_OPEN_STACKS) >= _OPEN_STACKS_MAX:
        _OPEN_STACKS.pop(next(iter(_OPEN_STACKS)))
    _OPEN_STACKS[key] = ((stat.st_size, stat.st_mtime_ns), stack)
    return stack


def read_page(path):
    """The raw frame (stack dtype, single channel) a page path addresses.

    :raises StackFormatError: when ``path`` is not a page path of a stack.
    """
    split = split_page_path(path)
    if split is None:
        raise StackFormatError(f"{path}: not a recording-stack page path")
    return open_stack(split[0]).read_frame(split[1])


def read_image(path, flags=None):
    """``cv2.imread(path, flags)`` that also reads page paths.

    A page is decoded through an in-memory lossless TIFF with ``cv2.imdecode``,
    so every ``cv2.IMREAD_*`` flag (colour conversion, bit-depth reduction,
    ``IMREAD_REDUCED_*``) behaves exactly as it does for the legacy per-frame
    file holding the same pixels. Returns None when the image cannot be read,
    as ``cv2.imread`` does.
    """
    import cv2

    if flags is None:
        flags = cv2.IMREAD_COLOR
    if split_page_path(path) is None:
        return cv2.imread(str(path), flags)
    try:
        frame = read_page(path)
    except (OSError, IndexError, ValueError):
        return None
    ok, encoded = cv2.imencode(".tiff", frame, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    if not ok:
        return None
    return cv2.imdecode(encoded, flags)
