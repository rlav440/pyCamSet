'''
Purpose: Measure the real per-frame cost of calibration-target detection on a
         folder of finished acquisition frames, so pyCamSet's streaming
         calibration cadence constants can be set from a measured number rather
         than a guess.
Status:  In Development
Future:  Extend with a live-folder (watcher) mode once the producer writes
         frames atomically; feed the measured per-frame cost into the streaming
         plan's Tier 1 thread-pool sizing.

WHAT THIS ANSWERS
-----------------
The streaming-calibration plan needs one number nobody has: how long one
detection actually takes on this rig's frames (720x540 Mono16). It also needs
to know whether the Mono16 -> uint8 contrast stretch costs anything meaningful,
how much of the per-frame time is TIFF decode rather than detection, and how
many worker threads the target frame rate needs.

The engine here is deliberately Qt-free: it runs from a script, a test, or a GUI
tab, and nothing about it needs a QApplication. ``pyCamSet.gui`` renders it.

REUSE, NOT REINVENTION
----------------------
Detection goes through this package's own target classes -- the same
``find_in_image`` the phases and the GUI call -- so the measured path is the
path that ships. A faster measurement of a path we do not use would be
worthless.

THE TARGET IS DESCRIBED BY THE TARGET
-------------------------------------
A measurement names a target and that target's own constructor arguments, taken
from its ``construction_parameters()``: a Ccube has no ``num_squares_x``, so
there is no way to ask it for one. ``target_spec_from_values`` is the single
place values become a spec, and it refuses a value the chosen target does not
declare rather than silently ignoring it.
'''
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import platform
import re
import socket
import typing
from pathlib import Path

import numpy as np

from pyCamSet.calibration_targets.core.target_registry import (
    TARGET_NAMES,
    build_target,
    target_class,
)
from pyCamSet.utils.general_utils import _SUPPORTED_IMAGE_SUFFIXES

# OpenCV is used here only to read frames and to report/limit its internal
# thread count. Detection itself goes through the target classes, never cv2.aruco.
import cv2


# --------------------------------------------------------------------------- #
# The frame layout the producer writes                                     #
# --------------------------------------------------------------------------- #

#: The producer's name for one acquired frame:
#: ``{camera}_{YYYYmmdd_HHMMSS}_{frame_index:06d}.tiff``. The timestamp is the
#: acquisition's FIRST frame, so every frame of one acquisition shares it.
FRAME_NAME_PATTERN = "{camera}_{YYYYmmdd_HHMMSS}_{frame_index:06d}.tiff"

_NAME_RE = re.compile(
    r"^(?P<camera>.+)_(?P<stamp>\d{8}_\d{6})_(?P<index>\d{6})\.(?P<suffix>[A-Za-z]+)$"
)

#: ``(rows, cols)`` a calibration frame is expected to have. The sensor frame is
#: 720x540 and the producer rotates every frame 90 degrees clockwise before
#: saving, so a saved frame is 720 rows x 540 cols -- portrait.
#:
#: RECORDED, not enforced: a folder of another geometry is measured and reported
#: as-is, with its actual shape, rather than refused.
EXPECTED_FRAME_SHAPE = (720, 540)


@dataclasses.dataclass(frozen=True)
class FrameRef:
    """One frame on disk, and where its name says it belongs."""

    path: Path
    camera: str
    timestamp: str
    index: int


def parse_frame_name(name: str) -> typing.Optional[dict]:
    """Read ``camera/timestamp/index`` out of a produced frame name, or None."""
    match = _NAME_RE.match(name)
    if match is None:
        return None
    return {
        "camera": match.group("camera"),
        "timestamp": match.group("stamp"),
        "index": int(match.group("index")),
        "suffix": match.group("suffix").lower(),
    }


def list_frames(folder) -> dict:
    """List a finished acquisition folder: cameras, frames, and anything odd.

    Returns a dict rather than raising for the ordinary pathologies -- no
    subfolders, a stray file, two acquisition timestamps in one folder --
    because those are exactly the facts worth reporting. ``errors`` carries what
    could not be read at all.

    A directory here is NOT assumed to be a camera: a finished acquisition
    legitimately also holds a pyCamSet workspace directory, and treating that as
    a camera produces an all-NaN timing row and a false "unequal frame counts"
    warning. Only a directory holding at least one parsable frame is a camera.
    """
    folder = Path(folder)
    listing: dict = {
        "folder": str(folder),
        "exists": folder.is_dir(),
        "camera_folders": [],
        "cameras": {},
        "non_frame_dirs": {},
        "stray_files": [],
        "acquisition_timestamps": [],
        "errors": [],
        "expected_frame_shape": list(EXPECTED_FRAME_SHAPE),
    }
    if not folder.is_dir():
        listing["errors"].append(f"no such folder: {folder}")
        return listing

    directory_names: list[str] = []
    for child in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir():
            directory_names.append(child.name)
        elif child.name != "metadata.json":
            listing["stray_files"].append(child.name)

    stamps: set[str] = set()
    for camera_name in directory_names:
        camera_dir = folder / camera_name
        frames: list[FrameRef] = []
        other: list[str] = []
        for path in sorted(camera_dir.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file():
                continue
            parsed = parse_frame_name(path.name)
            if parsed is None:
                other.append(path.name)
                continue
            frames.append(FrameRef(path=path, camera=parsed["camera"],
                                   timestamp=parsed["timestamp"],
                                   index=parsed["index"]))
            stamps.add(parsed["timestamp"])
        frames.sort(key=lambda f: f.index)
        mtimes = [f.path.stat().st_mtime for f in frames]
        listing["cameras"][camera_name] = {
            "count": len(frames),
            # The names the run actually saw, capped so one long acquisition
            # cannot bloat the JSON. The cap is stated in the report.
            "filenames": [f.path.name for f in frames[:200]],
            "filenames_truncated": len(frames) > 200,
            "first_name": frames[0].path.name if frames else None,
            "last_name": frames[-1].path.name if frames else None,
            "indices_contiguous": [f.index for f in frames] == list(range(len(frames))),
            "first_mtime": min(mtimes) if mtimes else None,
            "last_mtime": max(mtimes) if mtimes else None,
            "unparsed_frames": other[:50],
            "_frames": frames,   # consumed in-process; stripped before writing
        }

    listing["camera_folders"] = [
        name for name, info in listing["cameras"].items() if info["count"]]
    listing["non_frame_dirs"] = {
        name: info["unparsed_frames"]
        for name, info in listing["cameras"].items() if not info["count"]}
    for name in listing["non_frame_dirs"]:
        del listing["cameras"][name]

    listing["acquisition_timestamps"] = sorted(stamps)
    counts = [c["count"] for c in listing["cameras"].values()]
    listing["camera_frame_counts"] = {
        name: c["count"] for name, c in listing["cameras"].items()}
    listing["frame_count_uniform"] = len(set(counts)) <= 1 if counts else None
    listing["total_frames"] = sum(counts)
    if len(stamps) > 1:
        listing["errors"].append(
            f"more than one acquisition timestamp in this folder: {sorted(stamps)} "
            "-- the streaming plan assumes one acquisition per session")
    return listing


def read_producer_metadata(folder) -> typing.Optional[dict]:
    """Read the acquisition's own ``metadata.json``, or None when absent."""
    path = Path(folder) / "metadata.json"
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        return {"_unreadable": f"{type(exc).__name__}: {exc}"}


def declared_targets(metadata: typing.Optional[dict]) -> typing.Optional[list]:
    """The target definitions the acquisition declared, when it declared any.

    The producer writes these under ``Calibration_Targets``; the exact nesting
    is probed rather than assumed, because it is the producer's shape and not
    this package's.
    """
    if not metadata:
        return None
    block = metadata.get("Calibration_Targets")
    if isinstance(block, list):
        return block
    if isinstance(block, dict):
        targets = block.get("Targets")
        if isinstance(targets, list):
            return targets
    return None


# --------------------------------------------------------------------------- #
# The target is whatever the target says it is                             #
# --------------------------------------------------------------------------- #

def target_names() -> tuple[str, ...]:
    """Every target this package can build, in registry order."""
    return tuple(TARGET_NAMES)


def construction_keys(target_name: str) -> list[str]:
    """The constructor arguments *target_name* declares, settable ones first.

    This is what makes a form adapt to the target: a Ccube declares no
    ``num_squares_x``, so nothing can ask it for one. Raises for an unknown
    name rather than returning an empty list, so a typo is not read as a target
    with no settings.
    """
    return [p.key for p in target_class(target_name).construction_parameters().settable()]


def default_construction_values(target_name: str) -> dict:
    """The defaults *target_name* declares for its own construction settings."""
    return {
        p.key: p.default
        for p in target_class(target_name).construction_parameters().settable()
    }


def target_spec_from_values(target_name: str, values: dict) -> dict:
    """Build the spec for *target_name* from *values*, refusing stray keys.

    Only the keys that target declares are accepted. A value for a setting the
    target does not have is an error, not something quietly dropped: passing
    ``num_squares_x`` to a Ccube means the caller believes it is measuring a
    board, and measuring a cube instead would produce a confident number for the
    wrong geometry.
    """
    allowed = set(construction_keys(target_name))
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            f"{target_name} declares no setting called {', '.join(unknown)}; "
            f"it takes: {', '.join(sorted(allowed)) or '(none)'}."
        )
    spec = {p.key: p.coerce(values.get(p.key, p.default))
            for p in target_class(target_name).construction_parameters().settable()}
    return {"type": target_name, **spec}


def build_measured_target(target_name: str, values: dict):
    """The target class instance a measurement detects with."""
    spec = target_spec_from_values(target_name, values)
    arguments = dict(spec)
    arguments.pop("type")
    return build_target({**spec, "type": target_name})


def describe_target(spec: dict) -> str:
    """A one-line description of a spec, for a report or a log line."""
    name = spec.get("type", "?")
    fields = ", ".join(f"{k}={v}" for k, v in spec.items() if k != "type")
    return f"{name}({fields})"


# --------------------------------------------------------------------------- #
# Timing                                                                     #
# --------------------------------------------------------------------------- #

def percentiles(samples: list) -> dict:
    """Median and p95 of a sample list, in milliseconds."""
    if not samples:
        return {"n": 0}
    array = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(array.size),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "mean_ms": float(array.mean()),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def to_uint8_contrast_stretch(frame) -> np.ndarray:
    """Stretch a frame's actual value range to ``uint8``, for detection.

    Uses the frame's own per-frame min/max, not the dtype's range: Mono12 data
    is 12-bit values (0..4095) in a uint16 container, and dividing by 65535
    would compress a real frame into about 6% of the uint8 range and degrade
    detection. A flat frame (lens cap, saturated sensor) becomes black rather
    than dividing by zero.

    This mirrors what the GUI already does before detection. It lives here so
    the measurement, the GUI and any future streaming path all stretch the same
    way; the measurement times this call specifically.
    """
    if frame.dtype == np.uint8:
        return frame
    low = int(frame.min())
    high = int(frame.max())
    if high > low:
        return ((frame.astype(np.float32) - low) / (high - low) * 255).astype(np.uint8)
    return np.zeros_like(frame, dtype=np.uint8)


class DetectionCostError(RuntimeError):
    """Raised when a measurement must not proceed.

    Used for the blocking cases only: an unusable folder, or a target that
    disagrees with what the acquisition itself declared. A mismatch means
    detection would run against the wrong geometry, which produces
    plausible-looking garbage -- so it is reported and the run stops rather than
    being measured through.
    """


@dataclasses.dataclass
class MeasurementOptions:
    """Everything a run needs, so the engine has no hidden defaults."""

    folder: Path
    out_dir: Path
    target_type: str = "ChArUco2"
    #: The chosen target's own construction values, keyed by its declared
    #: settings. Empty means "use the target's defaults".
    target_values: dict = dataclasses.field(default_factory=dict)
    #: How many frames per camera to time. None = every frame found.
    max_frames_per_camera: typing.Optional[int] = None
    #: Also time one pass with OpenCV's internal pool pinned to a single thread,
    #: which separates "expensive" from "parallelises well": a detection spread
    #: over several cores looks cheap per frame while costing the same CPU.
    measure_single_thread_too: bool = True
    #: Frames per second the thread answer is computed for, as the whole-rig
    #: total. The acquisition rate is a setting, not a constant.
    expected_fps: float = 16.0
    #: Camera folders to measure; None means every camera folder found.
    cameras: typing.Optional[list] = None

    def spec(self) -> dict:
        """The target spec this run detects with."""
        return target_spec_from_values(self.target_type, self.target_values)

    def target(self):
        """The target instance this run detects with."""
        return build_measured_target(self.target_type, self.target_values)


def _time_now() -> float:
    import time
    return time.perf_counter()


def _since_ms(started: float) -> float:
    return (_time_now() - started) * 1000.0


@dataclasses.dataclass
class CameraTiming:
    """The timings collected for one camera."""

    camera: str
    decode_ms: list = dataclasses.field(default_factory=list)
    stretch_ms: list = dataclasses.field(default_factory=list)
    detect_ms: list = dataclasses.field(default_factory=list)
    detect_raw_ms: list = dataclasses.field(default_factory=list)
    detect_single_thread_ms: list = dataclasses.field(default_factory=list)
    end_to_end_ms: list = dataclasses.field(default_factory=list)
    points_found: list = dataclasses.field(default_factory=list)
    raw_dtype: typing.Optional[str] = None
    failures: list = dataclasses.field(default_factory=list)


def read_frame(path: Path):
    """Read one frame the way the folder reader does, or None if undecodable."""
    frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        return None
    if frame.ndim == 3:
        # A colour read means the file is not the mono frame the plan assumes.
        # Converted so the run continues, and the dtype is recorded.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def warm_up(target) -> None:
    """Make the first timed call representative.

    The first detection in a process builds the target's detector and touches
    lazily-imported code. Counting that as detection cost would inflate the
    first frame of every run, so one throwaway call is made before timing.
    """
    probe = np.zeros((64, 64), dtype=np.uint8)
    try:
        target.find_in_image(probe)
    except Exception:
        # A refusal on a black probe is fine; the point was to load the code.
        pass


# --------------------------------------------------------------------------- #
# Preflight: everything cheap, before the timing run                        #
# --------------------------------------------------------------------------- #

def compare_target_specs(declared: dict, configured: dict) -> list[str]:
    """Where an acquisition's declared target and the chosen spec differ.

    Compared over the LAYOUT fields only -- the ones that decide what a detected
    key means. A difference there means detection ran against the wrong
    geometry, which is the failure this check exists to stop. Fields that only
    change how markers are read (the dictionary, the detector tuning) are
    deliberately not compared: they do not change what a key addresses.
    """
    layout_fields = {
        "ChArUco": ("num_squares_x", "num_squares_y", "square_size", "marker_fraction"),
        "ChArUco2": ("num_squares_x", "num_squares_y", "square_size"),
        "Ccube": ("n_points", "length", "border_fraction"),
        "Ccube2": ("n_points", "length", "border_fraction"),
        "PuzzleBoard": ("num_squares_x", "num_squares_y", "square_size",
                        "start_x", "start_y"),
        "PuzzleBoardCube": ("n_points", "length"),
    }

    def canonical(entry: dict) -> dict:
        # An acquisition may write its own vocabulary; map the names this
        # package knows onto one spelling so a rename cannot read as a
        # geometry mismatch.
        out = dict(entry)
        for key in ("target_type", "type", "target"):
            if key in out:
                out["_type"] = str(out[key])
                break
        return out

    left, right = canonical(declared), canonical(configured)
    declared_type = left.get("_type")
    configured_type = right.get("_type")
    if declared_type != configured_type:
        return [
            f"target type: the acquisition declared {declared_type}, "
            f"this measurement would detect with {configured_type}"
        ]

    differences = []
    for field in layout_fields.get(str(configured_type), ()):
        a, b = left.get(field), right.get(field)
        if a is None or b is None:
            continue
        try:
            same = abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
        except (TypeError, ValueError):
            same = str(a) == str(b)
        if not same:
            differences.append(
                f"{field}: the acquisition declared {a}, "
                f"this measurement would use {b}")
    return differences


def preflight(options: MeasurementOptions) -> dict:
    """Everything cheap before timing: listing, metadata, target check.

    :raises DetectionCostError: for a folder that cannot be measured, or a
        declared-vs-configured target mismatch
    """
    folder = Path(options.folder)
    listing = list_frames(folder)
    if not listing["exists"]:
        raise DetectionCostError(f"No such folder: {folder}")
    if not listing["camera_folders"]:
        raise DetectionCostError(
            f"No camera subfolders in {folder} -- the layout expected is "
            f"<folder>/<camera_name>/<frame>.tiff")
    if listing["total_frames"] == 0:
        raise DetectionCostError(
            f"No frames matching {FRAME_NAME_PATTERN} in {folder}")

    metadata = read_producer_metadata(folder)
    declared = declared_targets(metadata)
    configured = options.spec()

    mismatch: list = []
    matched: typing.Optional[dict] = None
    for entry in (declared or []):
        differences = compare_target_specs(entry, configured)
        if not differences:
            matched = entry
            mismatch = []
            break
        if not mismatch:
            mismatch = differences

    if declared and matched is None:
        listed = "\n".join(f"  - {line}" for line in mismatch)
        raise DetectionCostError(
            "The acquisition's declared calibration target does not match the "
            "target this measurement would detect with:\n"
            f"{listed}\n\n"
            "Detecting against the wrong target geometry produces "
            "plausible-looking garbage, so this run is refused rather than "
            "measured through. Choose the target the acquisition declares and "
            "run again."
        )

    return {
        "listing": listing,
        "metadata": metadata,
        "declared_targets": declared,
        "matched_declared_target": matched,
        "configured_target": configured,
        "configured_description": describe_target(configured),
        "target_mismatch": mismatch,
    }


# --------------------------------------------------------------------------- #
# Timing one camera                                                          #
# --------------------------------------------------------------------------- #

def measure_camera(camera: str, frames: list, target, options: MeasurementOptions,
                   progress=None, raw_path_state=None) -> CameraTiming:
    """Time one camera's frames end to end.

    Per frame, in order:
      1. decode alone -- ``cv2.imread(..., IMREAD_UNCHANGED)``, the read the
         folder reader does;
      2. the Mono16 -> uint8 contrast stretch;
      3. detection on the stretched frame -- the path the GUI runs;
      4. detection on the raw frame as read, once it is known the detector
         accepts it.

    The raw path is timed only if the detector ACCEPTS it. aruco2's detectors
    require uint8, so on a Mono16 acquisition the raw path is refused; timing it
    every frame would produce a number for a call that does no detection work.
    """
    timing = CameraTiming(camera=camera)
    state = raw_path_state if raw_path_state is not None else {"accepted": None,
                                                               "message": None}
    total = len(frames)
    for position, frame in enumerate(frames, start=1):
        try:
            started = _time_now()
            raw = read_frame(frame.path)
            timing.decode_ms.append(_since_ms(started))
            if raw is None:
                timing.failures.append(
                    f"{frame.path.name}: cv2.imread could not decode it")
                continue
            if timing.raw_dtype is None:
                timing.raw_dtype = str(raw.dtype)

            started = _time_now()
            stretched = to_uint8_contrast_stretch(raw)
            timing.stretch_ms.append(_since_ms(started))

            started = _time_now()
            points = _detect_points(target, stretched)
            detect_ms = _since_ms(started)
            timing.detect_ms.append(detect_ms)

            if state["accepted"] is None:
                try:
                    target.find_in_image(raw)
                    state["accepted"] = True
                    state["message"] = None
                except Exception as exc:
                    # A refusal is a finding, not a failure: recorded with the
                    # detector's own message, then the path is skipped.
                    state["accepted"] = False
                    state["message"] = f"{type(exc).__name__}: {exc}"
            elif state["accepted"]:
                started = _time_now()
                target.find_in_image(raw)
                timing.detect_raw_ms.append(_since_ms(started))

            # End to end for the path that ships: read, stretch, detect.
            timing.end_to_end_ms.append(
                timing.decode_ms[-1] + timing.stretch_ms[-1] + detect_ms)
            timing.points_found.append(0 if points is None else int(len(points)))
        except Exception as exc:
            # One bad frame must not lose the whole run.
            timing.failures.append(f"{frame.path.name}: {type(exc).__name__}: {exc}")

        if progress is not None:
            progress(camera, position, total)
    return timing


def _detect_points(target, image) -> typing.Optional[np.ndarray]:
    """Detected Nx2 points from one image, or None when nothing was found."""
    detection = target.find_in_image(image)
    if not getattr(detection, "has_data", False):
        return None
    points = getattr(detection, "image_points", None)
    if points is None:
        return None
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(-1, 2)
    return points


# --------------------------------------------------------------------------- #
# The run                                                                    #
# --------------------------------------------------------------------------- #

def _host_block() -> dict:
    """What this measurement ran on, so two machines can be compared."""
    thread_info = {}
    try:
        import os
        thread_info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "opencv": getattr(cv2, "__version__", "unknown"),
        "numpy": np.__version__,
        **thread_info,
    }


def module_hash() -> typing.Optional[str]:
    """SHA-256 of this module's own source bytes.

    The comparability anchor: two runs are measuring the same code only if this
    matches. A commit id cannot serve -- the checkout differs per machine -- and
    the source hash is the thing that actually decides whether the numbers mean
    the same. Read from disk with newline normalisation, so a CRLF checkout and
    an LF checkout of identical source agree.
    """
    try:
        source = Path(__file__).read_bytes()
    except OSError:
        return None
    return hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest()


def _thread_answer(per_frame_ms: float, options: MeasurementOptions,
                   single_thread_ms: typing.Optional[float]) -> dict:
    """How many worker threads the target rate needs, and whether it parallelises.

    ``per_frame_ms`` is the measured detection cost on the path that ships.
    The answer is a count of workers for the requested rate, plus the speedup
    between the multi-thread and single-thread runs -- the latter says whether
    adding workers helps or merely spreads the same work.
    """
    if not per_frame_ms or per_frame_ms <= 0:
        return {"workers_needed": None, "note": "no detection timings collected"}
    per_worker = 1000.0 / per_frame_ms          # frames one worker handles a second
    workers = int(np.ceil(options.expected_fps / per_worker)) if per_worker else None
    speedup = None
    if single_thread_ms and single_thread_ms > 0:
        speedup = float(per_frame_ms / single_thread_ms)
    return {
        "expected_fps": options.expected_fps,
        "frames_per_second_per_worker": float(per_worker),
        "workers_needed": workers,
        "internal_parallel_speedup": speedup,
        "note": (
            "workers_needed assumes one detection at a time per worker and no "
            "other work; treat it as a floor."
        ),
    }


def measure_folder(options: MeasurementOptions, progress=None) -> dict:
    """Run the whole measurement and write the JSON + summary. Returns the report.

    :raises DetectionCostError: when :func:`preflight` refuses the run
    """
    pre = preflight(options)
    listing = pre["listing"]
    spec = options.spec()
    target = options.target()
    started_iso = _iso_now()

    threads_default = int(cv2.getNumThreads())
    timings: list[CameraTiming] = []

    # Carried across cameras: whether the detector accepts the raw as-read frame
    # is a property of the detector, not of a camera, so it is decided once and
    # a refusal recorded once rather than per frame.
    raw_path_state: dict = {"accepted": None, "message": None}

    selected = listing["cameras"]
    if options.cameras:
        wanted = set(options.cameras)
        selected = {k: v for k, v in selected.items() if k in wanted}

    for camera_name, camera_info in selected.items():
        frames = camera_info["_frames"]
        if options.max_frames_per_camera is not None:
            frames = frames[:options.max_frames_per_camera]
        # Warm up on this camera's first frame: the first call in a process
        # builds the target and touches lazily-loaded code, and counting that as
        # detection cost would inflate every number.
        if frames:
            warm_up(target)
        timings.append(measure_camera(camera_name, frames, target, options,
                                      progress, raw_path_state))

    # The single-thread pass, on the first camera that actually produced
    # timings -- not timings[0], which on a real folder can be a frame-less
    # directory and would silently reduce the thread answer to a null.
    if options.measure_single_thread_too and any(t.detect_ms for t in timings):
        timed = [t for t in timings if t.detect_ms]
        first_camera = timed[0].camera
        frames = selected[first_camera]["_frames"]
        if options.max_frames_per_camera is not None:
            frames = frames[:options.max_frames_per_camera]
        cv2.setNumThreads(1)
        try:
            single = measure_camera(first_camera, frames, target, options,
                                    progress=None, raw_path_state={"accepted": False,
                                                                   "message": None})
        finally:
            cv2.setNumThreads(threads_default)
        timings_by_camera = {t.camera: t for t in timings}
        timings_by_camera[first_camera].detect_single_thread_ms = single.detect_ms

    report = _build_report(options, pre, spec, timings, threads_default,
                           raw_path_state, started_iso)
    written = _write_report(report, Path(options.out_dir))
    report["written"] = written
    return report


def _build_report(options, pre, spec, timings, threads_default,
                  raw_path_state, started_iso) -> dict:
    """Assemble the JSON report from the collected timings."""
    listing = pre["listing"]

    per_camera = {}
    all_detect: list = []
    all_decode: list = []
    all_stretch: list = []
    all_end_to_end: list = []
    all_points: list = []
    failures: list = []
    for timing in timings:
        per_camera[timing.camera] = {
            "frames": len(timing.decode_ms),
            "decode": percentiles(timing.decode_ms),
            "stretch": percentiles(timing.stretch_ms),
            "detect": percentiles(timing.detect_ms),
            "detect_single_thread": percentiles(timing.detect_single_thread_ms),
            "detect_raw": percentiles(timing.detect_raw_ms),
            "end_to_end": percentiles(timing.end_to_end_ms),
            "points_found": {
                "median": float(np.median(timing.points_found)) if timing.points_found else 0,
                "min": min(timing.points_found) if timing.points_found else 0,
                "max": max(timing.points_found) if timing.points_found else 0,
            },
            "raw_dtype": timing.raw_dtype,
            "failures": timing.failures,
        }
        all_detect += timing.detect_ms
        all_decode += timing.decode_ms
        all_stretch += timing.stretch_ms
        all_end_to_end += timing.end_to_end_ms
        all_points += timing.points_found
        failures += [f"{timing.camera}: {f}" for f in timing.failures]

    overall_detect = percentiles(all_detect)
    single_thread = None
    for timing in timings:
        if timing.detect_single_thread_ms:
            single_thread = percentiles(timing.detect_single_thread_ms)
            break

    # Detection rate: what fraction of the frames the target was actually found
    # in. A timing number from frames with no detection in them measures the
    # detector failing, not detecting.
    found = sum(1 for p in all_points if p)
    detect_rate = (found / len(all_points)) if all_points else None

    return {
        "tool": "pyCamSet detection cost measurement",
        "started": started_iso,
        "finished": _iso_now(),
        "module_sha256": module_hash(),
        "host": _host_block(),
        "folder": str(options.folder),
        "out_dir": str(options.out_dir),
        "folder_listing": _strip_private(listing),
        "producer_metadata_present": pre["metadata"] is not None,
        "declared_targets": pre["declared_targets"],
        "matched_declared_target": pre["matched_declared_target"],
        "target": {
            "spec": spec,
            "description": pre["configured_description"],
            "construction_keys": construction_keys(options.target_type),
        },
        # The measured path. Both were timed so the stretch's cost is separable.
        "paths": {
            "detects_with": "contrast_stretched_uint8 (the path the GUI runs)",
            "raw_frame_dtype": timings[0].raw_dtype if timings else None,
            "raw_path_accepted": raw_path_state["accepted"],
            "raw_path_message": raw_path_state["message"],
            "raw_path_note": (
                "The raw as-read frame was offered to the detector. It requires "
                "uint8, so a Mono16 acquisition reports it as refused rather "
                "than timing a call that does no detection work."
            ),
        },
        "settings": {
            "max_frames_per_camera": options.max_frames_per_camera,
            "measure_single_thread_too": options.measure_single_thread_too,
            "expected_fps": options.expected_fps,
            "opencv_threads": threads_default,
        },
        "per_camera": per_camera,
        "overall": {
            "decode": percentiles(all_decode),
            "stretch": percentiles(all_stretch),
            "detect": overall_detect,
            "detect_single_thread": single_thread,
            "end_to_end": percentiles(all_end_to_end),
            "detection_rate": detect_rate,
            "frames_measured": len(all_points),
        },
        "thread_answer": _thread_answer(
            overall_detect.get("median_ms"),
            options,
            (single_thread or {}).get("median_ms"),
        ),
        "failures": failures,
    }


def _strip_private(listing: dict) -> dict:
    """The listing with in-process-only keys removed, so it is JSON-safe.

    Strips nested keys too: each camera's entry carries the in-process
    ``_frames`` list of FrameRef objects, which holds Path objects and would
    otherwise either bloat the report or fail to serialise. Stripping only the
    top level left it in, so the check is on every value the report carries.
    """
    def cleaned(mapping: dict) -> dict:
        return {k: v for k, v in mapping.items() if not k.startswith("_")}

    clean = cleaned(listing)
    clean["cameras"] = {
        name: cleaned(info) for name, info in listing.get("cameras", {}).items()
    }
    return clean


def _iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _write_report(report: dict, out_dir: Path) -> dict:
    """Write the JSON report and the human summary. Returns their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "detection_cost_report.json"
    summary_path = out_dir / "detection_cost_summary.txt"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(summarise(report))
    return {"json": str(json_path), "summary": str(summary_path)}


def summarise(report: dict) -> str:
    """The report as a short human-readable summary."""
    lines: list[str] = []
    overall = report.get("overall", {})
    detect = overall.get("detect", {})
    single = overall.get("detect_single_thread") or {}
    thread = report.get("thread_answer", {})
    target = report.get("target", {})

    lines.append("Detection cost measurement")
    lines.append("=" * 60)
    lines.append(f"folder          : {report.get('folder')}")
    lines.append(f"target          : {target.get('description')}")
    lines.append(f"frames measured : {overall.get('frames_measured')}")
    rate = overall.get("detection_rate")
    lines.append(
        f"detection rate  : {'n/a' if rate is None else f'{rate:.1%}'} of frames")
    lines.append(f"host            : {(report.get('host') or {}).get('hostname')}")
    lines.append(f"module sha256   : {report.get('module_sha256')}")
    lines.append("")
    lines.append("Per frame, on the path that ships (contrast-stretched uint8):")
    for label, block in (("decode", overall.get("decode")),
                         ("stretch", overall.get("stretch")),
                         ("detect", detect),
                         ("end to end", overall.get("end_to_end"))):
        if block and block.get("n"):
            lines.append(
                f"  {label:<12}: median {block['median_ms']:8.3f} ms   "
                f"p95 {block['p95_ms']:8.3f} ms   (n={block['n']})")
    if single.get("n"):
        lines.append(
            f"  detect, 1 thread: median {single['median_ms']:8.3f} ms   "
            f"p95 {single['p95_ms']:8.3f} ms")
    raw = (report.get("overall") or {}).get("detect")
    if report.get("paths", {}).get("raw_path_accepted") is False:
        lines.append("")
        lines.append(
            "Raw as-read frame: REFUSED by the detector "
            f"({report['paths'].get('raw_path_message')})")
    lines.append("")
    lines.append("Thread answer:")
    lines.append(f"  rate asked about     : {thread.get('expected_fps')} frames/s (whole rig)")
    if thread.get("frames_per_second_per_worker"):
        lines.append(
            f"  one worker handles   : {thread['frames_per_second_per_worker']:.1f} frames/s")
    lines.append(f"  workers needed       : {thread.get('workers_needed')}")
    if thread.get("internal_parallel_speedup"):
        lines.append(
            f"  internal speedup     : {thread['internal_parallel_speedup']:.2f}x "
            "(multi-thread vs one thread)")
    lines.append("")
    lines.append("Per camera, detect median / p95 (ms):")
    for name, block in (report.get("per_camera") or {}).items():
        d = block.get("detect") or {}
        if d.get("n"):
            lines.append(f"  {name:<16} {d['median_ms']:8.3f} / {d['p95_ms']:8.3f}"
                         f"   frames={block.get('frames')}")
    failures = report.get("failures") or []
    if failures:
        lines.append("")
        lines.append(f"Failures ({len(failures)}):")
        for line in failures[:20]:
            lines.append(f"  {line}")
    lines.append("")
    lines.append(
        "Comparability: match module sha256 above. A locally built checkout "
        "differs per machine, so the hash -- not a commit id -- is what says "
        "two runs measured the same code.")
    return "\n".join(lines) + "\n"


