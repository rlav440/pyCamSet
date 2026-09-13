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
from pyCamSet.workflow.recent_folders import remember_folder

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
# Windows caps a path at 260 characters unless it is given in extended-length
# form, and a workspace path is a nested one by construction: an image folder,
# then the workspace, then a phase, then a run id.  Every filesystem call here
# goes through these rather than through Path directly.
# ---------------------------------------------------------------------------


def _extended(path: Path | str) -> str:
    """Return *path* in Windows extended-length form; unchanged elsewhere."""
    raw = str(path)
    if os.name != "nt":
        return raw
    absolute = os.path.abspath(raw)
    if absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def path_exists(path: Path | str) -> bool:
    """Whether *path* exists, long Windows paths included."""
    if os.name != "nt":
        return Path(path).exists()
    return os.path.exists(_extended(path))


def copy_file(src: Path | str, dst: Path | str) -> None:
    """Copy *src* to *dst* with its metadata, long Windows paths included."""
    if os.name != "nt":
        shutil.copy2(Path(src), Path(dst))
        return
    shutil.copy2(_extended(src), _extended(dst))


def ensure_directory(path: Path | str) -> None:
    """Create *path* and any missing parents."""
    if os.name != "nt":
        os.makedirs(Path(path), exist_ok=True)
        return
    os.makedirs(_extended(path), exist_ok=True)


def as_io_path(path: Path | str) -> Path:
    """Return *path* in the form to hand to a library that opens it itself."""
    if os.name != "nt":
        return Path(path)
    return Path(_extended(path))


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
        with open(_extended(meta_path), "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)

        # A run has just been written here, so this workspace is real and
        # worth offering again.  Recorded from the save rather than from a
        # folder field, which changes on every keystroke.
        remember_folder(self.workspace_path.parent)
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
        runs_dir_io = _extended(runs_dir)
        if not os.path.exists(runs_dir_io):
            return []

        entries: list[tuple[float, dict]] = []
        for run_name in sorted(os.listdir(runs_dir_io)):
            meta_path = _extended(runs_dir / run_name / "metadata.json")
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
            entries.append((data["_recency_ts"], data))

        entries.sort(key=lambda pair: pair[0])
        return [data for _, data in entries]

    def find_run(self, phase: str, run_id: str | None) -> Optional[dict]:
        """Return the run with *run_id*, or None if there is no such run."""
        if not run_id:
            return None
        return next((run for run in self.load_runs(phase)
                     if run.get("run_id") == run_id), None)

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
        with open(_extended(handoff_path), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

    def read_handoff(self) -> dict:
        """Return ``<workspace>/handoff.json``, empty if there is none."""
        if self.workspace_path is None:
            return {}
        handoff_path = self.workspace_path / "handoff.json"
        if not path_exists(handoff_path):
            return {}
        try:
            with open(_extended(handoff_path), encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {}


# ---------------------------------------------------------------------------
# Resolving one phase's artifacts for the next phase to read
#
# Each looks first at what the run recorded, then at where that phase writes
# by convention -- so a run whose metadata predates the artifacts block, or
# whose artifact paths moved with the folder, still resolves.
# ---------------------------------------------------------------------------


def _first_existing(*candidates: Path | str | None) -> Optional[Path]:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path_exists(path):
            return path
    return None


def resolve_phase1_pickle_artifact(
        phase1_run: dict, ws_path: Path) -> Optional[Path]:
    """Return the ``detected_datapoints.pickle`` of a phase 1 run."""
    artifacts = phase1_run.get("artifacts") or {}
    run_id = phase1_run.get("run_id")
    return _first_existing(
        artifacts.get("detected_datapoints_pickle"),
        (ws_path / "phase1_runs" / str(run_id) / "detected_datapoints.pickle"
         if run_id else None),
    )


def resolve_phase2_camset_artifact(
        phase2_run: dict, ws_path: Path) -> Optional[Path]:
    """Return the initial camset of a phase 2 run."""
    artifacts = phase2_run.get("artifacts") or {}
    run_id = phase2_run.get("run_id")
    run_dir = ws_path / "phase2_runs" / str(run_id) if run_id else None
    return _first_existing(
        artifacts.get("initial_camset"),
        run_dir / "initial_cameras_high_distortion.camset" if run_dir else None,
        run_dir / "initial_cameras.camset" if run_dir else None,
    )


def resolve_phase3_camset_artifact(
        phase3_run: dict, ws_path: Path) -> Optional[Path]:
    """Return the optimised camset of a phase 3 run."""
    artifacts = phase3_run.get("artifacts") or {}
    run_id = phase3_run.get("run_id")
    run_dir = ws_path / "phase3_runs" / str(run_id) if run_id else None
    return _first_existing(
        artifacts.get("optimised_camset"),
        artifacts.get("self_calibrated_camset"),
        run_dir / "optimised_cameras.camset" if run_dir else None,
    )
