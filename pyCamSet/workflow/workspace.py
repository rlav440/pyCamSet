"""
The calibration workspace: where a run's settings, artifacts and results live.

A workspace is ``<image folder>/.pycamset_workspace``.  Inside it, each phase
owns a ``phaseN_runs`` directory, and each run is a directory named by its run
id holding ``metadata.json`` plus whatever the phase produced.

Nothing here knows about a user interface.  The GUI reads a workspace to fill
its run lists; a script can read the same workspace with the same calls.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from pyCamSet.utils.general_utils import get_subfolder_names
from pyCamSet.utils.paths import long_path

WORKSPACE_DIR_NAME = ".pycamset_workspace"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

KNOWN_PHASES = ("phase0", "phase1", "phase2", "phase3", "phase4")

# Phase names from earlier builds.  Named rather than ignored, so a workspace
# written by one of them reports itself instead of reading as empty.
_LEGACY_PHASES = {
    "phase5", "phase_5", "phase-5", "phase 5",
    "visualise_target", "assess_calibration",
}


# ---------------------------------------------------------------------------
# Filesystem access
#
# A workspace path is nested by construction -- image folder, workspace,
# phase, run id -- so every filesystem call here goes through long_path().
# ---------------------------------------------------------------------------


def path_exists(path: Path | str) -> bool:
    """Whether *path* exists, long Windows paths included."""
    return long_path(path).exists()


def copy_file(src: Path | str, dst: Path | str) -> None:
    """Copy *src* to *dst* with its metadata, long Windows paths included."""
    shutil.copy2(long_path(src), long_path(dst))


def delete_file(path: Path | str) -> None:
    """Remove *path* if it is there, long Windows paths included.

    A no-op, not an error, when *path* is already gone -- callers use this to
    make sure a stale file (e.g. a cache's identity sidecar that must never
    end up paired with a different cache) is absent, not to report whether
    one was found.
    """
    try:
        os.remove(long_path(path))
    except FileNotFoundError:
        pass


def ensure_directory(path: Path | str) -> None:
    """Create *path* and any missing parents."""
    os.makedirs(long_path(path), exist_ok=True)


def get_camera_subfolders(root: Path) -> list[Path]:
    """Return the camera folders inside an image folder."""
    return list(get_subfolder_names(Path(root), return_full_path=True))


def count_images_in_folder(folder: Path) -> int:
    """Count the image files directly inside *folder*."""
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir()
               if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def workspace_path_for(image_folder: Path | str) -> Path:
    """Return the workspace belonging to an image folder."""
    return Path(image_folder) / WORKSPACE_DIR_NAME


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------


def make_run_id() -> str:
    """Return a unique, timestamp-ordered run identifier.

    Format: ``YYYYMMDD_HHMMSS_<6-char hex>``
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


def _run_recency(data: dict, meta_path: str) -> float:
    """Return a sortable recency value (epoch seconds) for a loaded run.

    Prefers the explicit ``created_at`` field written by
    :meth:`WorkspaceManager.save_run`, and falls back to the metadata file's
    own mtime for runs written without one -- by an older build, or by a
    script.  That fallback is why no migration step is needed to sort a
    workspace that already exists.
    """
    created_at = data.get("created_at")
    if created_at:
        try:
            return datetime.fromisoformat(str(created_at)).timestamp()
        except (ValueError, TypeError):
            pass
    try:
        return os.path.getmtime(meta_path)
    except OSError:
        return 0.0


# ---------------------------------------------------------------------------
# The workspace
# ---------------------------------------------------------------------------


class WorkspaceManager:
    """Reads and writes the runs of one workspace.

    Supported phases are ``phase0`` through ``phase4``.  A legacy phase name
    raises :class:`ValueError` rather than silently reading nothing.
    """

    _PHASE_ORDER = list(KNOWN_PHASES)

    def __init__(self, workspace_path: Optional[Path] = None) -> None:
        self.workspace_path: Optional[Path] = (
            Path(workspace_path) if workspace_path else None)
        if self.workspace_path is not None:
            self.ensure_dirs()

    @classmethod
    def canonical_phase_name(cls, phase: str) -> str:
        """
        Normalise *phase* to its canonical lower-case name.

        :param phase: the phase name to normalise
        :raises ValueError: if *phase* is a name from an earlier build
        """
        name = (phase or "").strip().lower()
        if name in _LEGACY_PHASES:
            raise ValueError(
                f"Legacy phase name {phase!r} is no longer supported.  "
                "Re-run calibration to produce phase4 output in the current "
                "workspace layout."
            )
        return name

    @classmethod
    def _runs_dir_name(cls, phase: str) -> str:
        return f"{cls.canonical_phase_name(phase)}_runs"

    def set_workspace_path(self, workspace_path: Path, ensure: bool = True) -> None:
        """Point at *workspace_path*, creating its phase directories."""
        self.workspace_path = Path(workspace_path)
        if ensure:
            self.ensure_dirs()

    def ensure_dirs(self) -> None:
        """Create the per-phase run directories, if a workspace is set."""
        if self.workspace_path is None:
            return
        for phase in KNOWN_PHASES:
            ensure_directory(self.workspace_path / f"{phase}_runs")

    def run_dir(self, phase: str, run_id: str) -> Path:
        """Return the directory a run's artifacts belong in, creating it."""
        if self.workspace_path is None:
            raise RuntimeError("Workspace path is not set.")
        directory = self.workspace_path / self._runs_dir_name(phase) / run_id
        ensure_directory(directory)
        return directory

    def save_run(self, phase: str, run_id: str, metadata: dict) -> Path:
        """
        Write *metadata* to ``<workspace>/<phase>_runs/<run_id>/metadata.json``.

        :param phase: the phase the run belongs to
        :param run_id: the run's identifier
        :param metadata: the run record
        :return: the path written
        """
        directory = self.run_dir(phase, run_id)
        meta_path = directory / "metadata.json"

        # An explicit creation timestamp, because run ids from different
        # producers are not comparable as strings -- see _run_recency.
        # setdefault, so re-saving a run keeps the time it was first made.
        metadata = dict(metadata)
        metadata.setdefault("created_at", datetime.now().isoformat())
        with open(long_path(meta_path), "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)

        # Deliberately not remembered here.  A run is written by tests, by a
        # parameter search and by any script that calls a phase, and most of
        # those workspaces are temporary directories that no person will ever
        # go back to.  Where someone works is something only the interface
        # knows, so the GUI records it: see pyCamSet.workflow.recent_folders.
        return meta_path

    def load_runs(self, phase: str) -> list[dict]:
        """
        Return every saved run of *phase*, oldest first.

        Ordered by :func:`_run_recency` rather than by run id: producers mint
        ids in different formats -- ``YYYYMMDD_HHMMSS_hex`` here, but
        ``phase3_YYYYMMDD_HHMMSS`` from at least one script -- and digits sort
        before letters, so a plain sort put every letter-prefixed run last
        whatever its date.  Everything built on "the most recent run" read the
        wrong one.

        :param phase: the phase whose runs to read
        :return: the runs, each with ``run_id`` and ``_recency_ts`` filled in
        """
        if self.workspace_path is None:
            return []

        runs_dir = self.workspace_path / self._runs_dir_name(phase)
        runs_dir_io = long_path(runs_dir)
        if not os.path.exists(runs_dir_io):
            return []

        entries: list[tuple[float, dict]] = []
        for run_name in sorted(os.listdir(runs_dir_io)):
            meta_path = long_path(runs_dir / run_name / "metadata.json")
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            data.setdefault("run_id", run_name)
            # Cached on the in-memory dict only -- save_run always builds a
            # fresh record rather than round-tripping one of these -- so a
            # caller merging several load_runs results can sort across them
            # without re-reading mtimes.
            data["_recency_ts"] = _run_recency(data, meta_path)
            # Where the record was read from: a workspace copied to another
            # drive keeps its old absolute artifact paths, and the run's own
            # directory is how its files are still found.
            data["_run_dir"] = str(runs_dir / run_name)
            entries.append((data["_recency_ts"], data))

        entries.sort(key=lambda pair: pair[0])
        return [data for _, data in entries]

    def find_run(self, phase: str, run_id: str | None) -> Optional[dict]:
        """Return the run with *run_id*, or None if there is no such run."""
        if not run_id:
            return None
        return next((run for run in self.load_runs(phase)
                     if run.get("run_id") == run_id), None)

    def linked_run(self, phase: str, source_run: dict) -> Optional[dict]:
        """
        The run of *phase* that *source_run* was built from.

        Falls back to the most recent run of that phase when the link is
        absent or points at something no longer there, because a re-run of an
        old run should still find inputs rather than refuse.

        :param phase: the phase to look in
        :param source_run: the run whose ``inputs`` name the one wanted
        :return: the run, or None when that phase has none at all
        """
        runs = self.load_runs(phase)
        if not runs:
            return None
        wanted = (source_run.get("inputs") or {}).get(f"{phase}_run_id")
        return next((run for run in runs if run.get("run_id") == wanted),
                    runs[-1])

    def build_predecessor_chain(self, run: dict) -> list[dict]:
        """
        Return copies of every run *run* descends from, oldest first.

        Follows ``inputs.phaseN_run_id`` links through the workspace, taking
        the highest-numbered phase in each ``inputs`` dict as the direct
        parent.  Stops when no further parent resolves.
        """
        chain: list[dict] = []
        visited: set[str] = set()
        current = run

        for _ in range(len(self._PHASE_ORDER)):
            inputs = current.get("inputs") or {}

            best_index = -1
            parent_phase: Optional[str] = None
            parent_run_id: Optional[str] = None

            for key, value in inputs.items():
                if not (key.endswith("_run_id") and value):
                    continue
                phase = key[: -len("_run_id")]
                try:
                    index = self._PHASE_ORDER.index(phase)
                except ValueError:
                    continue
                if index > best_index:
                    best_index = index
                    parent_phase = phase
                    parent_run_id = str(value)

            if parent_run_id is None or parent_run_id in visited:
                break
            visited.add(parent_run_id)

            try:
                parent = self.find_run(parent_phase, parent_run_id)  # type: ignore[arg-type]
            except ValueError:
                break
            if parent is None:
                break

            chain.append(copy.deepcopy(parent))
            current = parent

        chain.reverse()
        return chain

    def write_handoff(self, payload: dict) -> None:
        """Write *payload* to ``<workspace>/handoff.json``."""
        if self.workspace_path is None:
            raise RuntimeError("Workspace path is not set.")
        handoff_path = self.workspace_path / "handoff.json"
        with open(long_path(handoff_path), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

    def read_handoff(self) -> dict:
        """Return ``<workspace>/handoff.json``, empty if there is none."""
        if self.workspace_path is None:
            return {}
        handoff_path = self.workspace_path / "handoff.json"
        if not path_exists(handoff_path):
            return {}
        try:
            with open(long_path(handoff_path), encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {}


# ---------------------------------------------------------------------------
# Resolving one phase's artifacts for the next phase to read
# ---------------------------------------------------------------------------

#: Per phase: the artifact keys a run may record its output under, and the
#: filenames that phase writes by convention.  Both are tried, recorded keys
#: first, so a run whose metadata predates the artifacts block -- or whose
#: recorded paths moved with the folder -- still resolves.
_ARTIFACTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "phase1": (
        ("detected_datapoints_pickle",),
        ("detected_datapoints.pickle",),
    ),
    "phase2": (
        ("initial_camset",),
        ("initial_cameras_high_distortion.camset", "initial_cameras.camset"),
    ),
    "phase3": (
        ("optimised_camset", "self_calibrated_camset"),
        ("optimised_cameras.camset",),
    ),
    "phase4": (
        ("self_calibrated_camset", "optimised_camset"),
        ("self_calibrated_cameras.camset",),
    ),
}


def resolve_artifact(run: dict, phase: str, ws_path: Path) -> Optional[Path]:
    """
    Return the output a run of *phase* left behind, or None if it is gone.

    :param run: the run's metadata
    :param phase: which phase the run belongs to
    :param ws_path: the workspace holding it
    """
    recorded_keys, filenames = _ARTIFACTS[phase]
    artifacts = run.get("artifacts") or {}
    candidates = [artifacts.get(key) for key in recorded_keys]

    run_id = run.get("run_id")
    if run_id:
        run_dir = ws_path / f"{phase}_runs" / str(run_id)
        candidates += [run_dir / name for name in filenames]

    for candidate in candidates:
        if candidate and path_exists(candidate):
            return Path(candidate)
    return None
