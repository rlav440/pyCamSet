"""Qt-free run provenance and optional education-trace contracts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import numpy as np

from pyCamSet.utils.paths import long_path
from pyCamSet.workflow.workspace import (
    WorkspaceManager,
    path_exists,
    resolve_artifact,
)

TRACE_SCHEMA_VERSION = 1
TRACE_FILENAME = "education_trace.json"
MAX_TRACE_BYTES = 8 * 1024 * 1024
MAX_TRACE_RECORDS = 256
USED_DETECTIONS_KEY = "education_used_detections"


class Verification(str, Enum):
    """The three possible results when comparing a recorded input edge."""

    VERIFIED = "verified"
    DIFFERS = "differs"
    UNCHECKABLE = "could not be checked"


class FrameKind(str, Enum):
    OBSERVED = "observed"
    INPUT_SNAPSHOT = "input_snapshot"
    COMPUTED_SNAPSHOT = "computed_snapshot"
    RECORDED_STEP = "recorded_step"
    ALGORITHM_DIAGRAM = "algorithm_diagram"
    REPLAY = "replay"
    COUNTERFACTUAL = "counterfactual"
    UNAVAILABLE = "unavailable"


class Origin(str, Enum):
    TRIAL = "an Optimisation-tab trial"
    CACHED_PASS = "a cached detection pass reused by this run (creator unknown)"
    IMAGE_FOLDER_CACHE = "an image-folder cache not linked to this run"
    PRUNED_SOURCE = "the source run of this pruned rerun"
    LATER_RUN = "a later run linked to this one"
    VERIFIED_UPSTREAM = "a verified upstream run"
    UNVERIFIED_UPSTREAM = "an unverified upstream run"
    OUTSIDE_RUN = "a file outside any run"
    REPLAY = "a separate replay"
    FRESH_PASS = "detected during this run (not saved)"
    SOURCE_IMAGES = "your source images as they are now"
    THIS_RUN = "this run's saved results"
    NONE = "none"


class Evidence(str, Enum):
    IMAGE_NOW = "image as stored now"
    FILE_NOW = "file as it is now (not saved by the run)"
    SAVED_AS_IS = "saved value as-is"
    RECALCULATED = "recalculated from saved results"
    DISPLAY = "display view computed from saved results"
    REDERIVED = "re-derived from current images"
    TRACE = "recorded during the run (trace)"
    DIAGRAM = "explanatory diagram"
    NOT_RECORDED = "not recorded"


class SourceCheck(str, Enum):
    MANIFEST = "verified by manifest"
    NOT_CHECKABLE = "not checkable"
    MISMATCH = "mismatch"
    NOT_APPLICABLE = "not applicable"


class PoseMappingReason(str, Enum):
    """Single reason a legacy detection's image-index mapping is unavailable."""

    PHOTO_LIST_CHANGED = "the photo list of its detection run no longer matches the photos in the folder"
    POSSIBLE_STALE_HIT = "its detections may have come from an earlier photo set"
    IMAGE_COUNT_CHANGED = "the number of image files in its folder has changed since its detection run"
    CACHE_REUSE = "its detection run may have reused saved detections, so which photo each came from was not recorded"
    LAYOUT_UNCOMPARABLE = "the photos in its folder cannot be compared with its detection run's list"


class AncestryReason(str, Enum):
    ANOTHER_IMAGE_FOLDER = "this run started from a camset made from another image folder"
    CANNOT_TRACE = "this run started from a file whose photos cannot be traced"


@dataclass(frozen=True)
class PoseMappingResult:
    """Legacy Phase 1 index-to-file result; names are withheld on uncertainty."""

    photo_names: tuple[str, ...]
    reason: PoseMappingReason | AncestryReason | None
    source_check: SourceCheck = SourceCheck.NOT_CHECKABLE


def classify_legacy_pose_mapping(*, caching: Any, promoted: bool,
                                 max_ims: Any, current_photo_names: tuple[str, ...],
                                 flat_without_tif: bool,
                                 report: Mapping[str, Any] | None,
                                 listing_capped: bool = False) -> PoseMappingResult:
    """Apply the legacy pose-to-file trigger and reason precedence.

    `current_photo_names` is the detector-style listing after applying `n_lim`;
    the caller must establish that the folder is flat and contains no `.tif`
    before setting `flat_without_tif` true.
    """
    detected_count = _finite_nonnegative_int(max_ims)
    if detected_count is None:
        return PoseMappingResult((), PoseMappingReason.LAYOUT_UNCOMPARABLE)
    if any(not isinstance(name, str) for name in current_photo_names):
        raise TypeError("current_photo_names must contain only strings")
    if not isinstance(listing_capped, bool):
        raise TypeError("listing_capped must be a bool")
    current_photo_count = len(current_photo_names)

    # A2m-era manifest/slot records have their own enumeration contract.
    # Legacy max_ims can prove a current-list mismatch only when it exceeds
    # that listing; equality is not required because trailing photos may fail.
    if detected_count > current_photo_count:
        return PoseMappingResult((), PoseMappingReason.PHOTO_LIST_CHANGED)

    values = report if isinstance(report, Mapping) else {}
    expected = _finite_nonnegative_int(values.get("n_images_expected"))
    observed = _finite_nonnegative_int(values.get("n_images"))
    if flat_without_tif and caching is True and expected is not None and observed is not None:
        if observed > expected:
            return PoseMappingResult((), PoseMappingReason.POSSIBLE_STALE_HIT)
    if flat_without_tif and expected is not None and expected != current_photo_count:
        return PoseMappingResult((), PoseMappingReason.IMAGE_COUNT_CHANGED)

    effective_caching = False if promoted else caching
    if effective_caching is True or effective_caching is None and not promoted:
        return PoseMappingResult((), PoseMappingReason.CACHE_REUSE)
    # A capped re-glob cannot prove the full ordering: deletion before the
    # cap can shift every pose while both capped counts remain unchanged.
    if not flat_without_tif or listing_capped:
        return PoseMappingResult((), PoseMappingReason.LAYOUT_UNCOMPARABLE)
    # Legacy mappings are never fully source-verified, but the re-glob can be
    # used for names when the conservative count checks passed.
    return PoseMappingResult(tuple(current_photo_names), None)


def _finite_nonnegative_int(value: Any) -> int | None:
    """Parse count-like report fields without accepting bools or infinities."""
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        return None
    return int(numeric)


def _legacy_phase1_pose_mapping(manager: WorkspaceManager,
                                ancestry: tuple[dict[str, Any], ...],
                                detection: Any,
                                photo_index: Any = 0) -> PoseMappingResult:
    """Conservatively map a legacy detection index to its Phase 1 file list."""
    if not ancestry or any(
            item.get("verification") != Verification.VERIFIED.value
            for item in ancestry):
        return PoseMappingResult((), AncestryReason.CANNOT_TRACE)
    phase1_ids = {str(item.get("parent_run_id")) for item in ancestry
                  if item.get("parent_phase") == "phase1" and item.get("parent_run_id")}
    phase2_ids = {str(item.get("parent_run_id")) for item in ancestry
                  if item.get("parent_phase") == "phase2" and item.get("parent_run_id")}
    if len(phase1_ids) != 1:
        return PoseMappingResult((), AncestryReason.CANNOT_TRACE)
    if len(phase2_ids) != 1:
        return PoseMappingResult((), AncestryReason.CANNOT_TRACE)
    try:
        phase2 = manager.find_run("phase2", next(iter(phase2_ids)))
        phase1 = manager.find_run("phase1", next(iter(phase1_ids)))
    except (KeyError, TypeError, ValueError):
        phase2, phase1 = None, None
    if not isinstance(phase2, Mapping) or not isinstance(phase1, Mapping):
        return PoseMappingResult((), AncestryReason.CANNOT_TRACE)
    phase2_artifacts = phase2.get("artifacts")
    used_detections = (phase2_artifacts.get(USED_DETECTIONS_KEY)
                       if isinstance(phase2_artifacts, Mapping) else None)
    if isinstance(used_detections, Mapping):
        if used_detections.get("schema_version") != 1:
            return PoseMappingResult((), AncestryReason.CANNOT_TRACE)
        source = used_detections.get("source")
        if source in {"fresh_pass_not_saved", "override_pickle", "filtered_pickle"}:
            return PoseMappingResult((), None, SourceCheck.NOT_APPLICABLE)
        if source != "phase1_pickle":
            return PoseMappingResult((), AncestryReason.CANNOT_TRACE)
    params = phase1.get("params")
    params = params if isinstance(params, Mapping) else {}
    try:
        camera_names = tuple(str(name) for name in detection.cam_names)
        max_ims = detection.max_ims
    except (AttributeError, TypeError):
        return PoseMappingResult((), PoseMappingReason.LAYOUT_UNCOMPARABLE)
    folder_value = params.get("f_loc")
    try:
        image_root = Path(str(folder_value)) if folder_value else None
    except (TypeError, ValueError):
        image_root = None
    if image_root is not None and not image_root.is_absolute():
        return PoseMappingResult((), AncestryReason.CANNOT_TRACE)
    education = phase1.get("education")
    education = education if isinstance(education, Mapping) else {}
    snapshot_version = education.get("trial_camset_snapshot_version")
    promoted = (bool(phase1.get("promoted_from"))
                or type(snapshot_version) is int and snapshot_version == 1)
    if image_root is None or not image_root.is_dir() or not camera_names:
        # An absent folder is not evidence that the current list changed.
        count = _finite_nonnegative_int(max_ims) or 0
        return classify_legacy_pose_mapping(
            caching=params.get("caching"), promoted=promoted, max_ims=max_ims,
            current_photo_names=tuple(f"<unavailable-{i}>" for i in range(count)),
            flat_without_tif=False, report=None,
        )

    from natsort import natsorted
    from pyCamSet.utils.general_utils import glob_ims
    from pyCamSet.workflow.detections import selected_camera_folders

    try:
        folders = selected_camera_folders(image_root, list(camera_names))
    except (OSError, RuntimeError, ValueError):
        folders = []
    folder_by_name = {path.name: path for path in folders}
    report = phase1.get("report")
    report_rows = report.get("per_camera", []) if isinstance(report, Mapping) else []
    report_by_camera = {
        str(row.get("name")): {
            "n_images_expected": row.get("n_images_expected"),
            "n_images": report.get("n_images"),
        }
        for row in report_rows
        if isinstance(row, Mapping) and row.get("name") is not None
    } if isinstance(report_rows, list) and isinstance(report, Mapping) else {}
    results: list[PoseMappingResult] = []
    for camera_name in camera_names:
        camera_folder = folder_by_name.get(camera_name)
        if camera_folder is None:
            count = _finite_nonnegative_int(max_ims) or 0
            names = tuple(f"<unavailable-{i}>" for i in range(count))
            flat = False
        else:
            subfolder = params.get("subfolder_string")
            image_folder = camera_folder / str(subfolder) if subfolder else camera_folder
            try:
                paths = natsorted(glob_ims(image_folder))
                has_tif = any(path.is_file() and path.suffix.lower() == ".tif"
                              for path in image_folder.rglob("*"))
                flat = (image_folder.is_dir()
                        and all(path.parent == image_folder for path in paths)
                        and not has_tif)
            except OSError:
                paths, flat = [], False
            n_lim = params.get("n_lim")
            listing_capped = False
            if n_lim is not None:
                limit = _finite_nonnegative_int(n_lim)
                if limit is None:
                    flat = False
                else:
                    listing_capped = len(paths) >= limit
                    paths = paths[:limit]
            names = tuple(path.name for path in paths)
        if camera_folder is None:
            listing_capped = False
        results.append(classify_legacy_pose_mapping(
            caching=params.get("caching"), promoted=promoted, max_ims=max_ims,
            current_photo_names=names, flat_without_tif=flat,
            report=report_by_camera.get(camera_name), listing_capped=listing_capped,
        ))

    priority = (
        PoseMappingReason.PHOTO_LIST_CHANGED,
        PoseMappingReason.POSSIBLE_STALE_HIT,
        PoseMappingReason.IMAGE_COUNT_CHANGED,
        PoseMappingReason.CACHE_REUSE,
        PoseMappingReason.LAYOUT_UNCOMPARABLE,
    )
    reason = next((candidate for candidate in priority
                   if any(result.reason is candidate for result in results)), None)
    if reason is not None:
        source_check = (SourceCheck.MISMATCH
                        if reason is PoseMappingReason.PHOTO_LIST_CHANGED
                        else SourceCheck.NOT_CHECKABLE)
        return PoseMappingResult((), reason, source_check)
    detected_count = _finite_nonnegative_int(max_ims) or 0
    selected_index = _finite_nonnegative_int(photo_index)
    if selected_index is None or selected_index >= detected_count:
        return PoseMappingResult((), PoseMappingReason.LAYOUT_UNCOMPARABLE)
    photo_names = tuple(
        f"{camera_name}/{result.photo_names[selected_index]}"
        for camera_name, result in zip(camera_names, results)
    )
    return PoseMappingResult(photo_names, None)


class FreshPassReason(str, Enum):
    """Ordered reasons Phase 2 can perform an unsaved fresh detection pass."""

    CAMERA_SUBSET = "camera_subset"
    MISSING_CAMERA_RESOLUTION = "no_cam_res"
    MISSING_PHASE1_PICKLE = "pickle_missing"
    HIGH_DISTORTION_SECOND_PASS = "high_distortion_second_pass"


_FRESH_PASS_REASON_ORDER = {reason.value: index for index, reason in enumerate(FreshPassReason)}


def make_used_detections_record(*, source: str, path: str | None,
                                fresh_pass_reasons: list[str] | tuple[str, ...] = (),
                                cache_hit: bool | None = None) -> dict[str, Any]:
    """Build C1's stable key payload; callers retain producer ownership."""
    reasons = [FreshPassReason(reason).value for reason in fresh_pass_reasons]
    if reasons != sorted(reasons, key=_FRESH_PASS_REASON_ORDER.__getitem__):
        raise ValueError("fresh-pass reasons must be supplied in contract order")
    if len(set(reasons)) != len(reasons):
        raise ValueError("fresh-pass reasons must be unique")
    if source not in {"phase1_pickle", "override_pickle", "filtered_pickle", "fresh_pass_not_saved", "legacy_unknown"}:
        raise ValueError(f"Unknown used-detections source: {source}")
    if source == "fresh_pass_not_saved" and not reasons:
        raise ValueError("a fresh pass must name at least one reason")
    if source != "fresh_pass_not_saved" and reasons:
        raise ValueError("fresh-pass reasons only apply to an unsaved fresh pass")
    if cache_hit is not None and not isinstance(cache_hit, bool):
        raise ValueError("cache_hit must be true, false or None")
    return {
        "schema_version": 1,
        "source": source,
        "path": path,
        "fresh_pass_reasons": reasons,
        "cache_hit": cache_hit,
    }


_KIND_EVIDENCE: dict[FrameKind, Evidence] = {
    FrameKind.OBSERVED: Evidence.SAVED_AS_IS,
    FrameKind.INPUT_SNAPSHOT: Evidence.SAVED_AS_IS,
    FrameKind.COMPUTED_SNAPSHOT: Evidence.RECALCULATED,
    FrameKind.RECORDED_STEP: Evidence.TRACE,
    FrameKind.ALGORITHM_DIAGRAM: Evidence.DIAGRAM,
    FrameKind.REPLAY: Evidence.RECALCULATED,
    FrameKind.COUNTERFACTUAL: Evidence.DISPLAY,
    FrameKind.UNAVAILABLE: Evidence.NOT_RECORDED,
}

_ORIGIN_PRECEDENCE = tuple(Origin)
_ORIGIN_RANK = {origin: rank for rank, origin in enumerate(_ORIGIN_PRECEDENCE)}


@dataclass(frozen=True)
class Frame:
    """A Qt-free story frame; provenance is attached to every rendered layer."""

    stage_id: str
    phase: str
    title: str
    kind: FrameKind
    source_run_id: str | None = None
    camera: str | None = None
    image_index: int | None = None
    sample_space: str = "none"
    transformation: str | None = None
    units: str | None = None
    source_artifact: str | None = None
    digest: str | None = None
    code_version: str | None = None
    library_versions: Mapping[str, str] = field(default_factory=dict)
    solver: str | None = None
    iteration_semantics: str | None = None
    predecessor_ids: tuple[str, ...] = ()
    availability_reason: str | None = None
    caption: str = ""
    source_ref: str | None = None
    quality_flags: tuple[str, ...] = ()
    origin: Origin | None = None
    evidence: Evidence | None = None
    source_check: str = "not checkable"

    def __post_init__(self) -> None:
        kind = FrameKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if self.evidence is None:
            object.__setattr__(self, "evidence", _KIND_EVIDENCE[kind])
        if not self.stage_id or not self.phase:
            raise ValueError("Frame stage_id and phase must be non-empty")
        if self.image_index is not None and self.image_index < 0:
            raise ValueError("image_index must be non-negative")

    def layer_provenance(self) -> tuple[str, str, str]:
        """Return the required origin/evidence/source-check triple."""
        return (
            (self.origin or Origin.NONE).value,
            (self.evidence or _KIND_EVIDENCE[self.kind]).value,
            self.source_check,
        )


def choose_origin(origins: list[Origin] | tuple[Origin, ...]) -> Origin:
    """Choose one origin by the contract's total precedence order."""
    if not origins:
        return Origin.NONE
    return min((Origin(value) for value in origins), key=_ORIGIN_RANK.__getitem__)


def sha256_file(path: Path | str) -> str:
    """Hash a file in bounded chunks without loading it all into memory."""
    digest = hashlib.sha256()
    with open(long_path(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_recorded_path(value: Any) -> Path | PureWindowsPath | None:
    """Parse absolute local/Windows paths; never resolve a relative record."""
    if not isinstance(value, (str, os.PathLike)):
        return None
    text = os.fspath(value)
    if not text or not Path(text).is_absolute() and not PureWindowsPath(text).is_absolute():
        return None
    if PureWindowsPath(text).is_absolute() or "\\" in text:
        return PureWindowsPath(text)
    return Path(text)


def normalise_recorded_path(value: Any) -> tuple[str, ...] | None:
    """Normalise a path relative to its nearest workspace, platform-aware.

    A relative path intentionally returns None. A Windows path is compared
    case-insensitively even when this reader runs on POSIX.
    """
    parsed = _absolute_recorded_path(value)
    if parsed is None:
        return None
    if isinstance(parsed, PureWindowsPath):
        parts = tuple(part.casefold() for part in parsed.parts)
        lower = tuple(part.casefold() for part in parsed.parts)
        for index in range(len(lower) - 1, -1, -1):
            if lower[index] == ".pycamset_workspace":
                return parts[index + 1:]
        return None
    resolved = parsed.resolve(strict=False)
    for parent in (resolved, *resolved.parents):
        if parent.name.casefold() == ".pycamset_workspace":
            return tuple(part.casefold() for part in resolved.relative_to(parent).parts)
    return None


def _workspace_root_key(value: Any) -> tuple[str, ...] | None:
    """Return the path to the nearest workspace ancestor, case-folded."""
    parsed = _absolute_recorded_path(value)
    if parsed is None:
        return None
    if isinstance(parsed, PureWindowsPath):
        parts = tuple(part.casefold() for part in parsed.parts)
        for index in range(len(parts) - 1, -1, -1):
            if parts[index] == ".pycamset_workspace":
                return parts[:index + 1]
        return None
    resolved = parsed.resolve(strict=False)
    for parent in (resolved, *resolved.parents):
        if parent.name.casefold() == ".pycamset_workspace":
            return tuple(part.casefold() for part in parent.parts)
    return None


def phase4_ancestry_reason(used_path: Any, current_workspace: Path | str,
                           recorded_phase4_workspace: Path | str | None,
                           *, local_run_copy_exists: bool) -> AncestryReason:
    """Classify a Phase 4 state path without mistaking moved workspaces."""
    relative = normalise_recorded_path(used_path)
    used_workspace = _workspace_root_key(used_path)
    current_root = _workspace_root_key(current_workspace)
    recorded_root = _workspace_root_key(recorded_phase4_workspace)
    if (relative is not None and len(relative) >= 3
            and relative[0] == "phase3_runs"
            and used_workspace is not None
            and used_workspace != current_root
            and used_workspace != recorded_root
            and not local_run_copy_exists):
        return AncestryReason.ANOTHER_IMAGE_FOLDER
    return AncestryReason.CANNOT_TRACE


def _recorded_phase4_workspace(metadata: Mapping[str, Any]) -> Path | PureWindowsPath | None:
    """Find Phase 4's recorded workspace from its own output, then `f_loc`."""
    artifacts = metadata.get("artifacts")
    if isinstance(artifacts, Mapping):
        for key in ("self_calibrated_camset", "optimised_camset"):
            artifact = artifacts.get(key)
            if _workspace_root_key(artifact) is not None:
                return _absolute_recorded_path(artifact)
    params = metadata.get("params")
    folder = params.get("f_loc") if isinstance(params, Mapping) else None
    parsed = _absolute_recorded_path(folder)
    if isinstance(parsed, PureWindowsPath):
        return parsed / ".pycamset_workspace"
    if isinstance(parsed, Path):
        return parsed / ".pycamset_workspace"
    return None


def _run_output_path(run: Mapping[str, Any], phase: str, workspace: Path) -> Path | None:
    """Resolve output using the workspace's recorded-first/copy-fallback rule."""
    try:
        return resolve_artifact(dict(run), phase, workspace)
    except (KeyError, TypeError, ValueError, OSError):
        return None


def resolve_exact_parent(manager: WorkspaceManager, child: Mapping[str, Any],
                         phase: str) -> dict[str, Any] | None:
    """Resolve the explicitly named parent only; never use latest-run fallback."""
    inputs = child.get("inputs")
    run_id = inputs.get(f"{phase}_run_id") if isinstance(inputs, Mapping) else None
    return manager.find_run(phase, str(run_id)) if run_id else None


def resolve_all_parents(manager: WorkspaceManager, run: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Resolve each explicitly recorded parent, ordered by phase then ID."""
    inputs = run.get("inputs") or {}
    if not isinstance(inputs, Mapping):
        return []
    parents: list[dict[str, Any]] = []
    for key, run_id in sorted(inputs.items()):
        if not key.endswith("_run_id") or not run_id:
            continue
        phase = key[:-len("_run_id")]
        try:
            parent = manager.find_run(phase, str(run_id))
        except ValueError:
            parent = None
        if parent is not None:
            parents.append(parent)
    return parents


def _edge_paths_equal(expected: Any, used: Any) -> Verification:
    """Compare edge paths, degrading missing/relative paths to uncheckable."""
    left = normalise_recorded_path(expected)
    right = normalise_recorded_path(used)
    if left is None or right is None:
        return Verification.UNCHECKABLE
    if left == right:
        return Verification.VERIFIED
    # Absolute paths in different copies can still identify the same bytes.
    try:
        expected_path = Path(os.fspath(expected))
        used_path = Path(os.fspath(used))
        if expected_path.is_file() and used_path.is_file():
            if sha256_file(expected_path) == sha256_file(used_path):
                return Verification.VERIFIED
    except (OSError, TypeError, ValueError):
        pass
    return Verification.DIFFERS


def _used_artifact_path(run: Mapping[str, Any], phase: str,
                        parent_phase: str | None = None) -> Any:
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None
    keys = {
        "phase2": ("phase1_detection_pickle_used", "used_detections",
                   "phase1_detection_pickle"),
        "phase3": {
            "phase2": ("phase2_initial_camset_used",),
            "phase1": ("phase1_detection_pickle_used",),
        },
        "phase4": ("phase3_camset_used",),
    }.get(phase, ())
    if phase == "phase3":
        keys = keys.get(parent_phase or "phase2", ())
    if phase == "phase2":
        # C1 stores this producer-owned record under artifacts. Accept the
        # early A2 prototype's top-level location for already-written runs.
        used_detections = artifacts.get(USED_DETECTIONS_KEY)
        if not isinstance(used_detections, Mapping):
            used_detections = run.get(USED_DETECTIONS_KEY)
        if isinstance(used_detections, Mapping):
            # An explicit fresh-pass record with path=None overrides the
            # historical Phase 1-pickle field; that pickle was not used.
            return used_detections.get("path")
    return next((artifacts[key] for key in keys if artifacts.get(key)), None)


def _used_artifact_paths(run: Mapping[str, Any], phase: str) -> tuple[Any, ...]:
    """Return every run input whose identity contributes to a rerun comparison."""
    artifacts = run.get("artifacts")
    if phase == "phase2":
        path = _used_artifact_path(run, phase, "phase1")
        return (path,) if path else ()
    keys = {
        "phase3": ("phase2_initial_camset_used", "phase1_detection_pickle_used"),
        "phase4": ("phase3_camset_used",),
    }.get(phase, ())
    if not isinstance(artifacts, Mapping):
        return ()
    return tuple(artifacts[key] for key in keys if artifacts.get(key))


def _phase2_detection_edge(context: "RunContext", parent: Mapping[str, Any],
                           expected: Path) -> Verification:
    """Apply C1's used-detections rule, including conservative legacy checks."""
    artifacts = context.metadata.get("artifacts") or {}
    record = artifacts.get(USED_DETECTIONS_KEY)
    if not isinstance(record, Mapping):
        # Compatibility for the initial, unintegrated A2 prototype.
        record = context.metadata.get(USED_DETECTIONS_KEY)
    if isinstance(record, Mapping):
        source = record.get("source")
        if source == "fresh_pass_not_saved":
            # A fresh Phase 2 pass is not the Phase 1 pickle, even on a cache hit.
            return Verification.UNCHECKABLE
        used = record.get("path")
        if source != "phase1_pickle" or not used:
            return Verification.UNCHECKABLE
        return _edge_paths_equal(expected, used)

    # Legacy records can only verify when all evidence rules out re-detection,
    # an override, pruning, and a cache/file replacement after the run.
    params = context.metadata.get("params") or {}
    artifacts = context.metadata.get("artifacts") or {}
    used = artifacts.get("phase1_detection_pickle")
    parent_artifacts = parent.get("artifacts") or {}
    parent_pickle = parent_artifacts.get("detected_datapoints_pickle")
    if not used or not parent_pickle or artifacts.get("detection_source_override"):
        return Verification.UNCHECKABLE
    if artifacts.get("filtered_detection_pickle") or context.metadata.get("threshold_pruning"):
        return Verification.UNCHECKABLE
    if params.get("high_distortion"):
        return Verification.UNCHECKABLE
    selected = params.get("selected_cameras") or []
    if selected and list(selected) != list(params.get("cam_names") or []):
        return Verification.UNCHECKABLE
    if not path_exists(used):
        return Verification.UNCHECKABLE
    path_result = _edge_paths_equal(expected, used)
    if path_result is not Verification.VERIFIED:
        return path_result
    try:
        run_timestamp = _run_timestamp(context.metadata)
        used_path = long_path(used)
        if run_timestamp is None or used_path.stat().st_mtime >= run_timestamp:
            return Verification.UNCHECKABLE
        from pyCamSet.utils.saving import load_pickle
        # Load the resolved parent artifact, not an arbitrary metadata path.
        payload = load_pickle(long_path(expected))
        detections, cam_res = payload if isinstance(payload, tuple) and len(payload) == 2 else (payload, None)
        if cam_res is None:
            return Verification.UNCHECKABLE
        cam_names = getattr(detections, "cam_names", None)
        if cam_names is None:
            return Verification.UNCHECKABLE
        if selected and list(selected) != list(cam_names):
            return Verification.UNCHECKABLE
    except Exception:
        # Legacy pickle compatibility is best-effort; a failed read is not proof.
        return Verification.UNCHECKABLE
    return _edge_paths_equal(expected, used)


def _run_timestamp(metadata: Mapping[str, Any]) -> float | None:
    """Parse the run's recorded timestamp for the legacy pickle-age check."""
    value = metadata.get("created_at") or metadata.get("timestamp")
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (ValueError, TypeError, OverflowError):
        return None


@dataclass(frozen=True)
class EdgeResult:
    phase: str
    verification: Verification
    parent_run_id: str | None = None
    used_path: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class StateSource:
    """Summary of the Phase 3 camset used to initialise a Phase 4 run."""

    path: Path | None
    origin_run_id: str | None
    verification: Verification
    reason: str | None = None
    residual_mean_px: float | None = None
    stored_mean_px: float | None = None
    confirmed: bool = False
    start_mismatch: str | None = None
    ancestry: tuple[dict[str, Any], ...] = ()
    photo_names: tuple[str, ...] = ()
    source_check: str = "not checkable"
    phase3_before_px: float | None = None
    start_metric_px: float | None = None
    fixed_pose_matches: bool | None = None
    effective_fixed_parameter_differences: tuple[str, ...] = ()
    rig_comparison_withheld: bool = False
    detections_match: bool | None = None
    ancestry_verified: bool = False
    loss: str | None = None
    f_scale: float | None = None
    phase3_fixed_pose_index0: int | None = None
    phase3_fixed_pose_ordinal1: int | None = None
    phase4_fixed_pose_index0: int | None = None
    phase4_fixed_pose_ordinal1: int | None = None


def saved_fixed_pose_index(options: Any, detection: Any) -> int | None:
    """Return a saved fixed-pose index only when it is valid for its detections.

    The result is the zero-based index stored by pyCamSet. Consumers presenting
    a novice-facing pose number should use ``index + 1``; absent or malformed
    saved evidence remains unavailable.
    """
    if not isinstance(options, Mapping):
        return None
    value = options.get("fixed_pose")
    if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        return None
    try:
        numeric = float(value)
        maximum = _finite_nonnegative_int(getattr(detection, "max_ims", None))
    except (TypeError, ValueError, OverflowError):
        return None
    if (not math.isfinite(numeric) or not numeric.is_integer()
            or maximum is None):
        return None
    index = int(numeric)
    return index if 0 <= index < maximum else None


@dataclass(frozen=True)
class LossSettings:
    """Robust-loss settings read from a saved handler, not inferred from UI defaults."""

    loss: str | None
    f_scale: float | None


def loss_settings_from_options(options: Any) -> LossSettings:
    """Project solver loss options, using the solver's linear/scale defaults."""
    if not isinstance(options, Mapping):
        return LossSettings(None, None)
    loss = options.get("loss", "linear")
    if not isinstance(loss, str) or not loss:
        return LossSettings(None, None)
    scale = options.get("f_scale", 1.0)
    try:
        scale_value = float(scale)
    except (TypeError, ValueError, OverflowError):
        scale_value = math.nan
    if not math.isfinite(scale_value) or scale_value <= 0:
        scale_value = None
    return LossSettings(loss, scale_value)


def detections_equal(left: Any, right: Any) -> bool:
    """Compare embedded detection identity, including camera and image order."""
    if left is None or right is None:
        return False
    try:
        return (list(left.cam_names) == list(right.cam_names)
                and int(left.max_ims) == int(right.max_ims)
                and np.array_equal(np.asarray(left._data), np.asarray(right._data)))
    except (AttributeError, TypeError, ValueError):
        return False


def normalise_fixed_params(fixed_params: Any) -> dict[str, dict[str, Any]]:
    """Drop empty per-camera entries and stabilise camera/block ordering."""
    if not isinstance(fixed_params, Mapping):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for camera, blocks in sorted(fixed_params.items(), key=lambda item: str(item[0])):
        if not isinstance(blocks, Mapping):
            continue
        kept = {str(block): value for block, value in sorted(
            blocks.items(), key=lambda item: str(item[0])) if value is not None}
        if kept:
            result[str(camera)] = kept
    return result


def compare_start_settings(phase4_options: Any, phase4_fixed: Any,
                           used_options: Any, used_fixed: Any,
                           used_cameras: Any, *, start_error_px: Any,
                           stored_mean_px: Any) -> tuple[str | None, tuple[str, ...], bool]:
    """Compare Phase 4 start controls with the camset actually used.

    Fixed-parameter differences count only when the actual start residual also
    disagrees with the used camset's persisted mean.
    """
    p4_options = phase4_options if isinstance(phase4_options, Mapping) else {}
    source_options = used_options if isinstance(used_options, Mapping) else {}
    fixed_pose_equal = p4_options.get("fixed_pose", 0) == source_options.get("fixed_pose", 0)
    if not fixed_pose_equal:
        return "fixed_pose", (), False

    differences: list[str] = []
    p4_fixed = normalise_fixed_params(phase4_fixed)
    camera_map = used_cameras if isinstance(used_cameras, Mapping) else {}
    for camera_name, blocks in p4_fixed.items():
        camera = camera_map.get(camera_name)
        if camera is None:
            continue
        for block, fixed_value in blocks.items():
            solved_value = {
                "int": getattr(camera, "intrinsic", None),
                "dst": getattr(camera, "distortion_coefs", None),
                "ext": getattr(camera, "extrinsic", None),
            }.get(block)
            if solved_value is None:
                continue
            try:
                equal = np.array_equal(np.asarray(fixed_value), np.asarray(solved_value))
            except (TypeError, ValueError):
                equal = False
            if not equal:
                differences.append(f"{camera_name}.{block}")
    if differences and not scalar_matches(start_error_px, stored_mean_px):
        return "fixed_parameters", tuple(differences), True
    return None, tuple(differences), True


def phase3_before_metric(state: "StateSource", manager: WorkspaceManager,
                         *, start_mismatch: str | None,
                         used_stored_mean_px: Any) -> float | None:
    """Use D3.6 as the Phase 4 comparison baseline for a changed start."""
    if start_mismatch is None or state.origin_run_id is None:
        return None
    try:
        if not math.isfinite(float(used_stored_mean_px)):
            return None
    except (TypeError, ValueError):
        return None
    phase3_run = manager.find_run("phase3", state.origin_run_id)
    diagnostics = phase3_run.get("diagnostics") if phase3_run else None
    if not isinstance(diagnostics, Mapping):
        return None
    try:
        metric = float(diagnostics.get("D3.6_final_euclid_px"))
    except (TypeError, ValueError):
        return None
    return metric if math.isfinite(metric) else None


@dataclass
class RunContext:
    """A run-bound, Qt-free projection of saved metadata and artifact paths."""

    phase: str
    run_id: str
    run_dir: Path
    workspace_path: Path
    metadata: dict[str, Any]
    manager: WorkspaceManager = field(repr=False)

    def __post_init__(self) -> None:
        """Keep the re-entrancy cut strictly per thread and per instance.

        The Phase 3 -> Phase 1 chain can be walked concurrently by two views
        that share one context.  A single per-instance flag lets one thread
        observe the other's cut and take the fail-closed ``could not be
        checked`` sentinel for its own verdict, so the marker must not be
        shared across threads.  An instance attribute holding a
        ``threading.local`` keeps the per-instance lifetime while making the
        cut re-entrant per thread, and is created here rather than at class
        level so two contexts cannot share one marker through the descriptor's
        per-class fallback.  The marker is process-local and thread-local, so it
        is excluded from the serialized form (``__getstate__``) and recreated
        idle on load (``__setstate__``); see those methods for the contract.
        """
        self._state_source_in_progress = threading.local()

    #: Instance attribute that is deliberately not part of the serialized state.
    _TRANSIENT_STATE = ("_state_source_in_progress",)

    def __getstate__(self) -> dict[str, Any]:
        """Serialized state without the process-local re-entrancy marker.

        A ``threading.local`` cannot be pickled, so carrying it in the default
        dataclass state would make every ``RunContext`` un-serializable and
        break ``copy.deepcopy``.  The marker is transient by construction - a
        cut in progress on one thread has no meaning for another process - so
        it is dropped here and recreated idle by ``__setstate__``.
        """
        state = dict(self.__dict__)
        for name in self._TRANSIENT_STATE:
            state.pop(name, None)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore a deserialized context with a fresh, idle marker.

        The marker is not restored from the payload; it is rebuilt here so the
        instance invariant ``self._state_source_in_progress`` exists holds for
        a context produced by ``pickle`` or ``copy.deepcopy``.
        """
        self.__dict__.update(state)
        self._state_source_in_progress = threading.local()

    @classmethod
    def from_run(cls, manager: WorkspaceManager, run: Mapping[str, Any],
                 phase: str | None = None) -> "RunContext":
        """Build from the read location, not a tab label or metadata phase."""
        workspace = manager.workspace_path
        if workspace is None:
            raise ValueError("WorkspaceManager has no workspace path")
        run_id = str(run.get("run_id", ""))
        if not run_id:
            raise ValueError("Run record has no run_id")
        actual_phase = _phase_from_run_dir(run, workspace, run_id)
        canonical = WorkspaceManager.canonical_phase_name(actual_phase)
        if phase is not None and WorkspaceManager.canonical_phase_name(phase) != canonical:
            raise ValueError("Requested phase does not match the run directory")
        expected_dir = Path(str(run.get("_run_dir", workspace / f"{canonical}_runs" / run_id)))
        if expected_dir.parent.name != f"{canonical}_runs":
            raise ValueError("Run directory phase does not match its location")
        return cls(canonical, run_id, expected_dir, workspace,
                   dict(run), manager)

    @property
    def status(self) -> str:
        """Return the run disposition, with failure taking precedence."""
        if self.metadata.get("error"):
            return "failed"
        status = self.metadata.get("status")
        if status:
            return str(status)
        diagnostics = self.metadata.get("diagnostics") or {}
        blocking = (self.metadata.get("report") or {}).get("blocking_flags")
        if diagnostics.get("quality_gate"):
            return str(diagnostics["quality_gate"].get("status", "incomplete"))
        if blocking is not None:
            return "incomplete" if blocking else "complete"
        return "unknown"

    @property
    def quality_check(self) -> str:
        """Distinguish unchecked legacy/promoted records from failed runs."""
        if self.status == "failed":
            return "failed"
        diagnostics = self.metadata.get("diagnostics") or {}
        if diagnostics.get("promoted_from") or self.metadata.get("promoted_from"):
            return "not checked"
        report = self.metadata.get("report")
        gate = diagnostics.get("quality_gate")
        if not report and not gate:
            return "not checked"
        return "checked"

    @property
    def quality_disposition(self) -> str:
        """Return the header disposition, including Phase 4 input attribution."""
        if self.phase == "phase4":
            edge = self.verify_input("phase3")
            if edge.verification is not Verification.VERIFIED:
                return "quality check not attributable to the camset used"
        return self.quality_check

    def promoted_camset_is_marked(self) -> bool:
        """Read the literal marker; truthy lookalikes are not accepted."""
        education = self.metadata.get("education") or {}
        marker = education.get("trial_camset_snapshot_version")
        return type(marker) is int and marker == 1

    def artifact(self, phase: str | None = None) -> Path | None:
        """Resolve this run's phase output, preferring its recorded path."""
        return _run_output_path(self.metadata, phase or self.phase, self.workspace_path)

    def loss_settings(self) -> LossSettings:
        """Read loss settings from this run's embedded saved handler."""
        output = self.artifact()
        if output is None:
            return LossSettings(None, None)
        try:
            from pyCamSet.utils.saving import load_CameraSet

            camera_set = load_CameraSet(long_path(output))
            handler = getattr(camera_set, "calibration_handler", None)
            return loss_settings_from_options(
                getattr(handler, "problem_opts", None))
        except Exception:
            return LossSettings(None, None)

    def verify_input(self, parent_phase: str) -> EdgeResult:
        """Verify recorded parent identity against the artifact this run used."""
        parent = resolve_exact_parent(self.manager, self.metadata, parent_phase)
        used = _used_artifact_path(self.metadata, self.phase, parent_phase)
        if parent is None or used is None:
            return EdgeResult(parent_phase, Verification.UNCHECKABLE,
                              str((self.metadata.get("inputs") or {}).get(f"{parent_phase}_run_id") or "") or None,
                              str(used) if used is not None else None,
                              "missing parent or used artifact")
        expected = _run_output_path(parent, parent_phase, self.workspace_path)
        if expected is None:
            return EdgeResult(parent_phase, Verification.UNCHECKABLE,
                              str(parent.get("run_id")), str(used), "parent artifact missing")
        if self.phase == "phase2" and parent_phase == "phase1":
            result = _phase2_detection_edge(self, parent, expected)
        else:
            result = _edge_paths_equal(expected, used)
        if parent.get("diagnostics", {}).get("promoted_from") and not parent.get("education", {}).get("trial_camset_snapshot_version"):
            # The edge remains path-verifiable; contamination is a separate flag.
            pass
        reason = None
        if (result is Verification.VERIFIED
                and _workspace_root_key(expected) != _workspace_root_key(used)):
            reason = "verified (read from another copy of this workspace)"
        return EdgeResult(parent_phase, result, str(parent.get("run_id")), str(used), reason)

    def used_input_matches(self, source: "RunContext") -> Verification:
        """Compare the actual used input, not merely the recorded parent IDs."""
        used_paths = _used_artifact_paths(self.metadata, self.phase)
        source_paths = _used_artifact_paths(source.metadata, source.phase)
        if not used_paths or not source_paths:
            return Verification.UNCHECKABLE
        if len(used_paths) != len(source_paths):
            return Verification.DIFFERS
        results = tuple(_edge_paths_equal(left, right)
                        for left, right in zip(used_paths, source_paths))
        if Verification.DIFFERS in results:
            return Verification.DIFFERS
        if Verification.UNCHECKABLE in results:
            return Verification.UNCHECKABLE
        return Verification.VERIFIED

    def phase1_source_check(self):
        """Verify Phase 1 image identity only along the artifacts actually used.

        Phase 2's producer record is authoritative: a fresh pass, even one
        served by cache, is not attributed to the recorded Phase 1 pickle.
        Phase 4 follows the Phase 3 state-source chain rather than its input ID.
        """
        from pyCamSet.workflow.education_manifest import ManifestResult, verify

        def check(run: Mapping[str, Any]) -> ManifestResult:
            diagnostics = run.get("diagnostics") or {}
            params = run.get("params") or {}
            manifest = diagnostics.get("education_manifest")
            if not isinstance(manifest, Mapping):
                cache = diagnostics.get("education_cache") or {}
                status = ("could not be checked" if cache.get("reused") is True
                          else "source identity not recorded")
                return ManifestResult(status, {}, {},
                                      "Phase 1 manifest missing")
            if not params.get("f_loc"):
                return ManifestResult("could not be checked", {}, {},
                                      "Phase 1 image folder is unavailable")
            camera_names = tuple((manifest.get("cameras") or {}).keys())
            return verify(
                manifest, params.get("f_loc", ""), camera_names,
                params.get("subfolder_string"), n_lim=params.get("n_lim"),
                include_digest=any(
                    isinstance(item, Mapping) and "sha256" in item
                    for entries in (manifest.get("cameras") or {}).values()
                    for item in entries if isinstance(entries, list)),
            )

        phase1_run: Mapping[str, Any] | None = None
        if self.phase == "phase1":
            phase1_run = self.metadata
        elif self.phase == "phase2":
            edge = self.verify_input("phase1")
            record = (self.metadata.get("artifacts") or {}).get(USED_DETECTIONS_KEY)
            if edge.verification is not Verification.VERIFIED:
                return ManifestResult("could not be checked", {}, {},
                                      "Phase 1 detections were not verified as the used input")
            if isinstance(record, Mapping) and record.get("source") != "phase1_pickle":
                return ManifestResult("could not be checked", {}, {},
                                      "Phase 2 detections came from a fresh pass")
            phase1_run = resolve_exact_parent(self.manager, self.metadata, "phase1")
        elif self.phase == "phase3":
            phase2_edge = self.verify_input("phase2")
            phase2 = resolve_exact_parent(self.manager, self.metadata, "phase2")
            if (phase2 is not None
                    and phase2_edge.verification is Verification.VERIFIED):
                phase2_context = RunContext.from_run(self.manager, phase2, "phase2")
                if phase2_context.verify_input("phase1").verification is Verification.VERIFIED:
                    phase1_run = resolve_exact_parent(self.manager, phase2, "phase1")
        elif self.phase == "phase4":
            state = self.phase3_state_source()
            if (state.verification is Verification.VERIFIED and state.origin_run_id
                    and state.ancestry and all(
                        item.get("verification") == Verification.VERIFIED.value
                        for item in state.ancestry)):
                phase3 = self.manager.find_run("phase3", state.origin_run_id)
                phase2 = resolve_exact_parent(self.manager, phase3, "phase2") if phase3 else None
                if phase2 is not None:
                    phase2_context = RunContext.from_run(self.manager, phase2, "phase2")
                    if phase2_context.verify_input("phase1").verification is Verification.VERIFIED:
                        phase1_run = resolve_exact_parent(self.manager, phase2, "phase1")
        if phase1_run is None:
            return ManifestResult("could not be checked", {}, {},
                                  "the used-input chain to Phase 1 was not verified")
        return check(phase1_run)

    def phase3_state_source(self) -> StateSource:
        """Resolve and conservatively validate Phase 4's Phase 3 starting camset.

        The return object deliberately carries a single reason and leaves
        detailed saved-camset comparisons to consumers that have the solver
        dependencies available.
        """
        if self.phase != "phase4":
            return StateSource(None, None, Verification.UNCHECKABLE,
                              "not a Phase 4 record")
        artifacts = self.metadata.get("artifacts") or {}
        recorded = artifacts.get("phase3_camset_used")
        if not recorded or _absolute_recorded_path(recorded) is None:
            return StateSource(None, None, Verification.UNCHECKABLE,
                              "cannot be traced",
                              source_check="not checkable (the photos behind the starting camset cannot be traced)")
        normalised = normalise_recorded_path(recorded)
        # A Phase 4 output is not a Phase 3 state source. Detect it before
        # probing the path or following the metadata's unrelated phase3 ID.
        if normalised and normalised[0] == "phase4_runs":
            return StateSource(None, None, Verification.UNCHECKABLE,
                              AncestryReason.CANNOT_TRACE.value,
                              source_check="not checkable (the photos behind the starting camset cannot be traced)")
        # A local run-directory copy is preferred over a stale absolute path.
        used_path = Path(str(recorded))
        local_copy = None
        run_id = None
        if normalised and len(normalised) >= 3 and normalised[0] == "phase3_runs":
            run_id = normalised[1]
            candidate = self.workspace_path / "phase3_runs" / run_id / Path(*normalised[2:])
            if path_exists(candidate):
                local_copy = candidate
        path = local_copy or (used_path if path_exists(used_path) else None)
        recorded_workspace = _recorded_phase4_workspace(self.metadata)
        ancestry_reason = phase4_ancestry_reason(
            recorded, self.workspace_path, recorded_workspace,
            local_run_copy_exists=local_copy is not None,
        )
        if path is None:
            return StateSource(None, run_id, Verification.UNCHECKABLE,
                              ancestry_reason.value)
        parent = self.manager.find_run("phase3", run_id) if run_id else None
        parent_output = _run_output_path(parent, "phase3", self.workspace_path) if parent else None
        verification = (_edge_paths_equal(parent_output, path)
                        if parent_output is not None else Verification.UNCHECKABLE)
        ancestry = tuple(walk_input_ancestry(self.manager, parent)) if parent else ()
        ancestry_verified = bool(ancestry) and all(
            item["verification"] == Verification.VERIFIED.value for item in ancestry)
        state = StateSource(
            path, run_id, verification,
            None if verification == Verification.VERIFIED and ancestry_verified
            else (ancestry_reason.value if not ancestry_verified
                  else AncestryReason.CANNOT_TRACE.value),
            ancestry=ancestry,
            source_check=("not checkable" if verification != Verification.VERIFIED
                          else "not checkable (the photos behind the starting camset cannot be traced)"),
        )
        try:
            from pyCamSet.utils.saving import load_CameraSet

            # Both objects are read-only evidence. Never rebuild or solve a
            # handler here: reload the actual starting and Phase 4 outputs.
            used_cams = load_CameraSet(long_path(path))
            phase4_output = _run_output_path(self.metadata, "phase4", self.workspace_path)
            if phase4_output is None:
                return state
            phase4_cams = load_CameraSet(long_path(phase4_output))
            used_handler = getattr(used_cams, "calibration_handler", None)
            phase4_handler = getattr(phase4_cams, "calibration_handler", None)
            if used_handler is None or phase4_handler is None:
                return state
            used_loss = loss_settings_from_options(
                getattr(used_handler, "problem_opts", None))

            # The reloaded residual vector starts with 2 values per optimised
            # observation; lockbox terms after that segment are excluded.
            used_rows = used_handler._flat_detections()
            row_count = len(used_rows)
            stored_mean = reprojection_segment_mean(
                used_cams.calibration_result, row_count)
            diagnostics = self.metadata.get("diagnostics") or {}
            start_metric = diagnostics.get("D4.3_initial_euclid_px")
            used_options = getattr(used_handler, "problem_opts", {})
            p4_options = getattr(phase4_handler, "problem_opts", {})
            same_detections = detections_equal(
                getattr(used_handler, "detection", None),
                getattr(phase4_handler, "detection", None),
            )
            mismatch, fixed_differences, fixed_pose_matches = compare_start_settings(
                p4_options, getattr(phase4_handler, "fixed_params", {}),
                used_options, getattr(used_handler, "fixed_params", {}),
                used_cams.get_cam_dict(), start_error_px=start_metric,
                stored_mean_px=stored_mean,
            )
            start_confirmed = ((mismatch is not None
                                or scalar_matches(start_metric, stored_mean))
                               and same_detections)
            source_confirmed = (
                verification == Verification.VERIFIED
                and ancestry_verified
                and start_confirmed
            )
            state_reason = state.reason
            if state_reason is None and not start_confirmed:
                state_reason = "cannot be confirmed"
            if source_confirmed:
                pose_mapping = _legacy_phase1_pose_mapping(
                    self.manager, ancestry,
                    getattr(used_handler, "detection", None),
                    p4_options.get("fixed_pose", 0),
                )
                # Surface the manifest for the Phase 1 run reached through
                # the verified, actually-used Phase 3 chain. The recorded
                # Phase 4 parent ID is not a source of image provenance.
                #
                # phase1_source_check resolves the same Phase 3 chain by
                # calling back into this method, so an unguarded call recurses
                # without bound (the RecursionError is swallowed by the
                # enclosing handler) and re-enumerates the image manifest at
                # every level. Break the cycle on the inner call: it still
                # resolves the chain, and the outer call supplies the
                # manifest status from the non-reentrant Phase 1 chain.
                #
                # The marker is thread-local: a shared RunContext may be walked
                # by two threads at once, and a plain per-instance flag would
                # let one thread's cut decide the other's verdict. The reset is
                # in a finally so an aborted manifest check still releases it.
                guard = self._state_source_in_progress
                if getattr(guard, "active", False):
                    from pyCamSet.workflow.education_manifest import ManifestResult
                    manifest_check = ManifestResult(
                        "could not be checked", {}, {},
                        "Phase 1 manifest check is already in progress")
                else:
                    guard.active = True
                    try:
                        manifest_check = self.phase1_source_check()
                    finally:
                        guard.active = False
                # Ordinals come from the saved solver settings paired with
                # each saved camset, not from a current image-folder listing.
                phase3_pose_index = saved_fixed_pose_index(
                    used_options, getattr(used_handler, "detection", None))
                phase4_pose_index = saved_fixed_pose_index(
                    p4_options, getattr(phase4_handler, "detection", None))
                if manifest_check.status in {"verified by manifest", "mismatch"}:
                    # The manifest checks current enumeration against saved
                    # Phase 1 records; it does not establish legacy cached
                    # detection-index-to-photo attribution.
                    source_check = manifest_check.status
                    if pose_mapping.reason is not None:
                        source_check = f"{source_check} ({pose_mapping.reason.value})"
                else:
                    source_check = pose_mapping.source_check.value
                if (manifest_check.status not in {"verified by manifest", "mismatch"}
                        and pose_mapping.reason is not None):
                    source_check = f"{source_check} ({pose_mapping.reason.value})"
            else:
                pose_mapping = PoseMappingResult((), None, SourceCheck.NOT_CHECKABLE)
                phase3_pose_index = None
                phase4_pose_index = None
                source_check = state.source_check
            # A Phase 4 fixed-pose mismatch is the N2 branch. Fixed parameter
            # differences are explanatory only when they changed the start.
            state = StateSource(
                path=path, origin_run_id=run_id, verification=verification,
                reason=state_reason, residual_mean_px=stored_mean,
                stored_mean_px=stored_mean,
                confirmed=source_confirmed,
                start_mismatch=mismatch, ancestry=ancestry,
                photo_names=pose_mapping.photo_names,
                source_check=source_check,
                phase3_before_px=phase3_before_metric(
                    state, self.manager,
                    start_mismatch=mismatch if source_confirmed else None,
                    used_stored_mean_px=stored_mean),
                start_metric_px=(float(start_metric) if start_metric is not None else None),
                fixed_pose_matches=fixed_pose_matches,
                effective_fixed_parameter_differences=fixed_differences,
                rig_comparison_withheld=(mismatch == "fixed_pose"),
                detections_match=same_detections,
                ancestry_verified=ancestry_verified,
                loss=used_loss.loss,
                f_scale=used_loss.f_scale,
                phase3_fixed_pose_index0=phase3_pose_index,
                phase3_fixed_pose_ordinal1=(
                    phase3_pose_index + 1 if phase3_pose_index is not None else None),
                phase4_fixed_pose_index0=phase4_pose_index,
                phase4_fixed_pose_ordinal1=(
                    phase4_pose_index + 1 if phase4_pose_index is not None else None),
            )
        except Exception:
            # Legacy camsets can load only as a bare CameraSet. Keep the path
            # provenance but do not turn a failed derivation into confirmation.
            return state
        return state


def walk_input_ancestry(manager: WorkspaceManager,
                       root: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Walk exact multi-parent ancestry and record every edge outcome.

    The manager's linear predecessor chain is retained as a deterministic
    first pass; all other explicitly named parents are then visited as well.
    Missing parents are represented as uncheckable edges, never substituted
    with a latest run.
    """
    if manager.workspace_path is None:
        return []
    linear = manager.build_predecessor_chain(dict(root))
    queue = [dict(root), *linear]
    visited_runs: set[tuple[str, str]] = set()
    visited_edges: set[tuple[str, str, str]] = set()
    results: list[dict[str, Any]] = []
    while queue:
        child = queue.pop(0)
        child_id = str(child.get("run_id", ""))
        try:
            child_phase = _phase_from_run_dir(
                child, manager.workspace_path, child_id)
        except ValueError:
            continue
        identity = (child_phase, child_id)
        if identity in visited_runs:
            continue
        visited_runs.add(identity)
        child_context = RunContext.from_run(manager, child, phase=child_phase)
        inputs = child.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        for key, parent_id in sorted(inputs.items()):
            if not key.endswith("_run_id") or not parent_id:
                continue
            parent_phase = key[:-len("_run_id")]
            edge_id = (child_phase, child_id, parent_phase)
            if edge_id in visited_edges:
                continue
            visited_edges.add(edge_id)
            try:
                parent = manager.find_run(parent_phase, str(parent_id))
            except ValueError:
                parent = None
            if parent is None:
                edge = EdgeResult(parent_phase, Verification.UNCHECKABLE,
                                  str(parent_id), reason="missing explicit parent")
            else:
                edge = child_context.verify_input(parent_phase)
                queue.append(parent)
            results.append({
                "child_phase": child_phase,
                "child_run_id": child_id,
                "parent_phase": parent_phase,
                "parent_run_id": str(parent_id),
                "verification": edge.verification.value,
                "used_path": edge.used_path,
                "reason": edge.reason,
            })
    return results


def _phase_from_run_dir(run: Mapping[str, Any], workspace: Path, run_id: str) -> str:
    """Infer phase strictly from the saved run directory location."""
    run_dir = Path(str(run.get("_run_dir", workspace / "unknown_runs" / run_id)))
    parent = run_dir.parent.name
    if parent.endswith("_runs"):
        return parent[:-5]
    raise ValueError("Cannot determine phase from the run directory")


def trace_path(run_dir: Path | str) -> Path:
    """Return the run-private sidecar path."""
    return Path(run_dir) / TRACE_FILENAME


def atomic_write_bytes(path: Path | str, payload: bytes) -> None:
    """Durably replace one file using a same-directory temporary and long_path."""
    destination = Path(path)
    ensure_parent = destination.parent
    long_path(ensure_parent).mkdir(parents=True, exist_ok=True)
    destination_io = long_path(destination)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", dir=long_path(ensure_parent), prefix=f".{destination.name}.",
                suffix=".tmp", delete=False) as stream:
            temp_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(long_path(temp_name), destination_io)
    finally:
        if temp_name is not None:
            try:
                long_path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass


def atomic_write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically in UTF-8 with stable separators."""
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    atomic_write_bytes(path, data)


def _envelope(run_id: str, phase: str, records: list[dict[str, Any]],
              evicted_kinds: list[str] | None = None) -> dict[str, Any]:
    """Build and checksum the sidecar envelope, not producer payload fields."""
    body = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "run_id": run_id,
        "phase": phase,
        "records": records,
        "evicted_kinds": list(evicted_kinds or []),
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    body["sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


def write_trace(run_dir: Path | str, run_id: str, phase: str, kind: str,
                payload: Mapping[str, Any], *, enabled: bool,
                max_records: int = MAX_TRACE_RECORDS,
                max_bytes: int = MAX_TRACE_BYTES) -> Path | None:
    """Append one opt-in trace record; required join records are not evicted."""
    if not enabled:
        return None
    if kind not in {"stage", "phase2_per_view", "accepted_step", "seed_join_order"}:
        raise ValueError(f"Unknown education trace kind: {kind}")
    if max_records < 1 or max_bytes < 1:
        raise ValueError("trace retention limits must be positive")
    directory = Path(run_dir)
    sidecar = trace_path(directory)
    records: list[dict[str, Any]] = []
    evicted: list[str] = []
    if path_exists(sidecar):
        current = read_trace(directory, run_id, phase)
        if current.get("status") != "available" and current.get("reason") != "trace records were evicted":
            raise ValueError("Existing education trace is unreadable; refusing overwrite")
        records = list(current["records"])
        evicted = list(current.get("evicted_kinds", []))
    records.append({"run_id": run_id, "phase": phase, "kind": kind,
                    "payload": dict(payload)})
    protected = {"phase2_per_view", "accepted_step", "seed_join_order"}
    while len(records) > max_records:
        # Protected join records may be dropped only with an explicit marker;
        # readers then refuse the trace rather than imply complete evidence.
        removable = next((index for index, record in enumerate(records)
                          if record["kind"] not in protected), 0)
        evicted.append(records.pop(removable)["kind"])
    envelope = _envelope(run_id, phase, records, evicted)
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    while len(encoded) > max_bytes:
        if not records:
            raise ValueError("Education trace envelope exceeds the sidecar byte cap")
        removable = next((index for index, record in enumerate(records)
                          if record["kind"] not in protected), 0)
        evicted.append(records.pop(removable)["kind"])
        envelope = _envelope(run_id, phase, records, evicted)
        encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    atomic_write_bytes(sidecar, encoded)
    return sidecar


def read_trace(run_dir: Path | str, run_id: str, phase: str,
               *, max_bytes: int = MAX_TRACE_BYTES) -> dict[str, Any]:
    """Load a run-private trace and validate schema, checksum and identity."""
    path = trace_path(run_dir)
    try:
        if not path_exists(path) or long_path(path).stat().st_size > max_bytes:
            return {"status": "unavailable", "reason": "missing or oversized sidecar"}
        with open(long_path(path), "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("sidecar root is not an object")
        digest = payload.pop("sha256", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
        if not isinstance(digest, str) or not hashlib.sha256(canonical).hexdigest() == digest:
            raise ValueError("sidecar checksum mismatch")
        if payload.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported sidecar schema version")
        if payload.get("run_id") != run_id or payload.get("phase") != phase:
            return {"status": "unavailable", "reason": "trace belongs to another run"}
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError("records is not a list")
        for record in records:
            if (not isinstance(record, dict) or record.get("run_id") != run_id
                    or record.get("phase") != phase or not isinstance(record.get("payload"), dict)):
                raise ValueError("invalid trace record envelope")
        if payload.get("evicted_kinds"):
            return {"status": "unavailable", "reason": "trace records were evicted",
                    "evicted_kinds": payload["evicted_kinds"], "records": records}
        return {"status": "available", "records": records,
                "evicted_kinds": payload.get("evicted_kinds", [])}
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        return {"status": "unavailable", "reason": "corrupt or invalid sidecar"}


def load_run_pickle(path: Path | str, expected_type: type | tuple[type, ...],
                    *, expected_run_dir: Path | str) -> tuple[Any | None, str | None]:
    """Load only a run-private pickle, then validate the resulting type.

    Pickle remains trusted-workspace-only: post-load type checks cannot make
    unpickling untrusted bytes safe.
    """
    candidate = Path(path)
    try:
        run_dir = long_path(expected_run_dir).resolve(strict=False)
        candidate_resolved = long_path(candidate).resolve(strict=True)
        if not candidate_resolved.is_relative_to(run_dir):
            return None, "pickle is outside the expected run directory"
        from pyCamSet.utils.saving import load_pickle
        value = load_pickle(candidate_resolved)
    except Exception as exc:  # A corrupt/legacy pickle must degrade gracefully.
        return None, f"pickle could not be read: {type(exc).__name__}"
    if not isinstance(value, expected_type):
        return None, "pickle payload has an unexpected type"
    return value, None


def reprojection_segment_mean(residuals: Any, optimised_row_count: int) -> float:
    """Mean Euclidean residual over the first 2×optimised-row values only."""
    values = np.asarray(residuals, dtype=float).reshape(-1)
    if optimised_row_count < 0 or len(values) < 2 * optimised_row_count:
        raise ValueError("residual vector is shorter than the reprojection segment")
    if optimised_row_count == 0:
        raise ValueError("cannot calculate a mean with no optimised detections")
    pairs = values[:2 * optimised_row_count].reshape(-1, 2)
    return float(np.linalg.norm(pairs, axis=1).mean())


def scalar_matches(actual: Any, expected: Any) -> bool:
    """Compare finite scalar metrics using the contract's absolute floor."""
    try:
        a, b = float(actual), float(expected)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= max(1e-9 * abs(a), 1e-9)
