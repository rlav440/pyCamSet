"""
Purpose: File-naming and save/load helpers for the pyCamSet phased calibration pipeline.
         Handles .json, .csv, .txt, .pickle, and .camset formats used by the pipeline
         phases.  Wraps pyCamSet's own save/load utilities where they already exist
         (save_camset, load_CameraSet) and provides lightweight stdlib-only helpers
         for the remaining formats.
Status:  Working
Future:  Add atomic write (write-then-rename) for crash-safe caching.
         Add a compress= option to save_pickle for large detection arrays.
"""

import csv                                                    # for .csv read/write
import json                                                   # for .json read/write
import pickle                                                 # for .pickle read/write
import re                                                     # for title-to-filename conversion
from pathlib import Path                                      # for filesystem path handling
from typing import Any, Dict, List, Optional, Sequence, Tuple  # type annotations


# ══════════════════════════════════════════════════════════════════════════════
#  1. Filename utilities
# ══════════════════════════════════════════════════════════════════════════════

def title_to_filename(title: str) -> str:
    """
    Convert a human-readable plot title to a lower-case, underscore-separated
    filename stem suitable for use in cache file paths.

    Example: ``'Mean Euclidean Error'`` → ``'mean_euclidean_error'``.

    :param title: A human-readable string (e.g. a plot title or phase name).
    :return:      A snake_case string containing only lower-case letters, digits,
                  and underscores, with no leading or trailing underscores.
    """
    lowered = title.lower()                                   # fold to lower case
    cleaned = re.sub(r'[^a-z0-9]+', '_', lowered)            # collapse non-alphanumeric runs to '_'
    cleaned = cleaned.strip('_')                              # remove leading/trailing underscores
    return cleaned                                            # return sanitised stem


def phase_cache_path(out_dir: Path, phase_name: str, ext: str) -> Path:
    """
    Return a standardised cache file path for a pipeline phase.

    The stem is derived from *phase_name* via title_to_filename() so that
    arbitrary phase names are safely converted to valid filenames.

    Example::

        phase_cache_path(Path('/tmp/run'), 'Phase 1 Detections', '.pickle')
        # → Path('/tmp/run/phase_1_detections.pickle')

    :param out_dir:    Output directory (must exist, or be created by the caller).
    :param phase_name: Human-readable phase name used to derive the filename stem.
    :param ext:        File extension including leading dot (e.g. ``'.pickle'``).
    :return:           Full Path object for the cache file.
    """
    stem = title_to_filename(phase_name)                      # normalise to snake_case
    return out_dir / f"{stem}{ext}"                           # combine into a full path


def cache_exists(path: Path) -> bool:
    """
    Return True if a cache file exists on disk and is non-empty.

    A zero-byte file is treated as absent so that failed partial writes are
    transparently re-generated on the next run.

    :param path: Path to the candidate cache file.
    :return:     True if the file exists and has at least one byte of content.
    """
    return path.is_file() and path.stat().st_size > 0        # file must exist and have content


# ══════════════════════════════════════════════════════════════════════════════
#  2. JSON helpers  (.json)
# ══════════════════════════════════════════════════════════════════════════════

def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """
    Serialise *data* to a UTF-8 .json file with pretty indentation.

    Creates parent directories automatically.  Overwrites any existing file.

    :param data:   Any JSON-serialisable object (dict, list, scalar, …).
    :param path:   Destination file path.  The .json extension is expected but
                   not enforced.
    :param indent: Number of spaces for indentation (default 2).
    """
    path = Path(path)                                         # ensure Path object
    path.parent.mkdir(parents=True, exist_ok=True)            # create parent dirs if needed
    with path.open('w', encoding='utf-8') as fh:              # open for writing
        json.dump(data, fh, indent=indent)                    # serialise with indentation


def load_json(path: Path) -> Any:
    """
    Load and return the object stored in a UTF-8 .json file.

    :param path: Source file path.
    :return:     The deserialised Python object.
    :raises FileNotFoundError: If *path* does not exist.
    :raises json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)                                         # ensure Path object
    with path.open('r', encoding='utf-8') as fh:              # open for reading
        return json.load(fh)                                  # deserialise and return


# ══════════════════════════════════════════════════════════════════════════════
#  3. CSV helpers  (.csv)
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(
    rows: Sequence[Sequence[Any]],
    headers: Sequence[str],
    path: Path,
) -> None:
    """
    Write *rows* with *headers* to a UTF-8 .csv file.

    Creates parent directories automatically.  Overwrites any existing file.
    Values are written as-is; no quoting style is enforced beyond the stdlib csv
    module defaults.

    :param rows:    Iterable of row sequences; each row is one line of data.
    :param headers: Column header names; written as the first row.
    :param path:    Destination file path.
    """
    path = Path(path)                                         # ensure Path object
    path.parent.mkdir(parents=True, exist_ok=True)            # create parent dirs if needed
    with path.open('w', newline='', encoding='utf-8') as fh:  # open for writing (no extra newline)
        writer = csv.writer(fh)                               # create CSV writer
        writer.writerow(headers)                              # write column header row
        writer.writerows(rows)                                # write all data rows


def load_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    """
    Load a UTF-8 .csv file and return ``(headers, rows)``.

    :param path:   Source file path.
    :return:       ``(headers, rows)`` where *headers* is the list of column names
                   from the first row and *rows* is a list of lists of string values.
                   Returns ``([], [])`` if the file is empty.
    :raises FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)                                         # ensure Path object
    with path.open('r', newline='', encoding='utf-8') as fh:  # open for reading
        reader = csv.reader(fh)                               # create CSV reader
        headers = next(reader, [])                            # read header row (empty if file blank)
        rows = list(reader)                                   # read all remaining rows
    return headers, rows                                      # return unpacked result


# ══════════════════════════════════════════════════════════════════════════════
#  4. Plain-text helpers  (.txt)
# ══════════════════════════════════════════════════════════════════════════════

def save_txt(lines: Sequence[str], path: Path) -> None:
    """
    Write *lines* to a UTF-8 .txt file, one element per line.

    Creates parent directories automatically.  Overwrites any existing file.

    :param lines: Sequence of strings to write; newlines are added between elements.
    :param path:  Destination file path.
    """
    path = Path(path)                                         # ensure Path object
    path.parent.mkdir(parents=True, exist_ok=True)            # create parent dirs if needed
    with path.open('w', encoding='utf-8') as fh:              # open for writing
        fh.write('\n'.join(lines))                            # join and write all lines


def load_txt(path: Path) -> List[str]:
    """
    Load a UTF-8 .txt file and return a list of non-empty stripped lines.

    :param path: Source file path.
    :return:     List of line strings; blank lines are omitted; trailing newlines stripped.
    :raises FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)                                         # ensure Path object
    with path.open('r', encoding='utf-8') as fh:              # open for reading
        return [ln.rstrip('\n') for ln in fh if ln.strip()]   # strip newlines, skip blanks


# ══════════════════════════════════════════════════════════════════════════════
#  5. Pickle helpers  (.pickle) — used for TargetDetection caching
# ══════════════════════════════════════════════════════════════════════════════

def save_pickle(obj: Any, path: Path) -> None:
    """
    Serialise *obj* to a binary .pickle file using the highest available protocol.

    Note: pyCamSet already provides save_pickle / load_pickle in
    pyCamSet.utils.saving using dill (which handles lambdas and closures).
    This helper uses the stdlib pickle module and is suitable for plain Python
    objects such as dicts, lists, and numpy arrays.  For pyCamSet internal
    objects that require dill, call pyCamSet.utils.saving.save_pickle directly.

    Creates parent directories automatically.  Overwrites any existing file.

    :param obj:  Any picklable Python object.
    :param path: Destination file path.
    """
    path = Path(path)                                         # ensure Path object
    path.parent.mkdir(parents=True, exist_ok=True)            # create parent dirs if needed
    with path.open('wb') as fh:                               # open in binary write mode
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)  # use highest pickle protocol


def load_pickle(path: Path) -> Any:
    """
    Deserialise and return the object stored in a binary .pickle file.

    Uses the stdlib pickle module (see save_pickle docstring for dill note).

    :param path: Source file path.
    :return:     The deserialised Python object.
    :raises FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)                                         # ensure Path object
    with path.open('rb') as fh:                               # open in binary read mode
        return pickle.load(fh)                                # deserialise and return


# ══════════════════════════════════════════════════════════════════════════════
#  6. CamSet helpers  (.camset) — delegated to pyCamSet's own I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_camset(cam_set: Any, path: Path) -> None:
    """
    Save a pyCamSet CameraSet to a .camset file using the CameraSet's own save
    method (which delegates to pyCamSet.utils.saving.save_camset).

    The .camset format is a JSON file containing camera intrinsics, extrinsics,
    distortion coefficients, resolution, and optionally the calibration handler
    and compressed detection data.  It is the standard persistence format used
    throughout pyCamSet.

    Creates parent directories automatically.

    :param cam_set: A pyCamSet CameraSet instance.
    :param path:    Destination file path.  The .camset extension is expected but
                    not enforced.
    """
    path = Path(path)                                         # ensure Path object
    path.parent.mkdir(parents=True, exist_ok=True)            # create parent dirs if needed
    cam_set.save(str(path))                                   # delegate to pyCamSet's save method


def load_camset(path: Path) -> Any:
    """
    Load a pyCamSet CameraSet from a .camset file using
    pyCamSet.utils.saving.load_CameraSet().

    The returned CameraSet will have calibration_handler, calibration_params,
    and calibration_result populated if they were present when the file was saved
    (see MERGE_INVESTIGATION.md §3c for round-trip caveats).

    :param path: Source .camset file path.
    :return:     A pyCamSet CameraSet instance.
    :raises FileNotFoundError: If *path* does not exist.
    """
    from pyCamSet.utils.saving import load_CameraSet          # import here to avoid circular dep
    path = Path(path)                                         # ensure Path object
    return load_CameraSet(path)                               # delegate to pyCamSet's load helper
