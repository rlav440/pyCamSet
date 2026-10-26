"""Exercise per-run image enumeration and cache provenance contracts.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pytest
from pyCamSet.calibration_targets.core.abstract_target import enumerate_detection_images
from pyCamSet.workflow.education_manifest import describe_enumeration, verify

# The real ChArUco miss/hit paired run reports no changed semantic leaves;
# In this card-owned paired snapshot, new cache/manifest evidence is carried
# under excluded education metadata rather than a changed semantic leaf.
EQUIVALENCE_DECLARATIONS = ()


def _image_tree(root: Path) -> Path:
    """Create a nested, naturally ordered two-camera image tree."""
    for camera in ("cam0", "cam1"):
        folder = root / camera / "nested"
        folder.mkdir(parents=True)
        for name in ("frame10.png", "frame2.png", "frame1.png"):
            (folder / name).write_bytes((camera + name).encode("utf-8"))
        (folder / "stray.tif").write_bytes(b"not in the detector suffix set")
    return root


def _manifest(root: Path, n_lim: int | None = None, digest: bool = False) -> dict:
    """Build manifest records from the detector's shared enumerator."""
    paths = {
        camera: tuple(enumerate_detection_images(root / camera / "nested", n_lim))
        for camera in ("cam0", "cam1")
    }
    return describe_enumeration(paths, root, "nested", include_digest=digest)


def test_shared_enumerator_is_recursive_natural_and_capped(tmp_path):
    root = _image_tree(tmp_path)
    names = [p.name for p in enumerate_detection_images(root / "cam0" / "nested", 2)]
    assert names == ["frame1.png", "frame2.png"]


def test_manifest_detects_edits_renames_and_reordering(tmp_path):
    root = _image_tree(tmp_path)
    manifest = _manifest(root)
    assert verify(manifest, root, ("cam0", "cam1"), "nested").status == "verified by manifest"

    image = root / "cam0" / "nested" / "frame2.png"
    image.write_bytes(b"edited with a different size")
    assert verify(manifest, root, ("cam0", "cam1"), "nested").status == "mismatch"

    image.rename(image.with_name("frame02.png"))
    result = verify(manifest, root, ("cam0", "cam1"), "nested")
    assert result.status == "mismatch"
    assert "frame2.png" in result.unavailable["cam0"]
    assert result.photo_names["cam0"] == ("frame1.png", "frame2.png", "frame10.png")


def test_optional_digest_detects_same_size_same_mtime_edit(tmp_path):
    root = _image_tree(tmp_path)
    manifest = _manifest(root, digest=True)
    image = root / "cam0" / "nested" / "frame1.png"
    before = image.stat()
    image.write_bytes(b"X" * before.st_size)
    os.utime(image, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert verify(manifest, root, ("cam0", "cam1"), "nested",
                  include_digest=True).status == "mismatch"


def test_manifest_is_fail_closed_when_missing_or_unrecordable(tmp_path):
    root = _image_tree(tmp_path)
    assert verify(None, root).status == "source identity not recorded"
    unrecordable = {"schema_version": 1, "recordable": False, "cameras": {}}
    assert verify(unrecordable, root).status == "could not be checked"


def test_writer_enumeration_carries_metadata_and_optional_sha256(tmp_path):
    root = _image_tree(tmp_path)
    manifest = _manifest(root, n_lim=2, digest=True)
    assert manifest["recordable"] is True
    assert [item["name"] for item in manifest["cameras"]["cam0"]] == [
        "frame1.png", "frame2.png"]
    item = manifest["cameras"]["cam0"][0]
    expected = hashlib.sha256((root / "cam0" / "nested" / item["name"]).read_bytes()).hexdigest()
    assert item["sha256"] == expected
    assert item["size"] > 0 and isinstance(item["mtime_ns"], int)


def test_copy_and_symlink_staged_camera_subsets_resolve_to_source_names(tmp_path):
    source = _image_tree(tmp_path / "source")
    staged = tmp_path / "staged"
    staged.mkdir()
    shutil.copytree(source / "cam1", staged / "cam1")
    try:
        (staged / "cam0").symlink_to(source / "cam0", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink privilege unavailable: {exc}")
    staged_paths = {
        camera: tuple(enumerate_detection_images(staged / camera / "nested"))
        for camera in ("cam0", "cam1")
    }
    manifest = describe_enumeration(staged_paths, staged, "nested")
    result = verify(manifest, source, ("cam0", "cam1"), "nested")
    assert result.status == "verified by manifest"
    assert result.photo_names["cam0"] == ("frame1.png", "frame2.png", "frame10.png")
    assert "stray.tif" not in result.photo_names["cam0"]


def test_copy_staged_camera_subset_resolves_to_source_names(tmp_path):
    source = _image_tree(tmp_path / "source")
    staged = tmp_path / "staged"
    staged.mkdir()
    shutil.copytree(source / "cam1", staged / "cam1")
    shutil.copytree(source / "cam0", staged / "cam0")
    staged_paths = {
        camera: tuple(enumerate_detection_images(staged / camera / "nested"))
        for camera in ("cam0", "cam1")
    }
    result = verify(describe_enumeration(staged_paths, staged, "nested"),
                    source, ("cam0", "cam1"), "nested")
    assert result.status == "verified by manifest"
    assert result.photo_names["cam0"] == ("frame1.png", "frame2.png", "frame10.png")


def test_deleted_enumerated_photo_fails_closed_with_unavailable_name(tmp_path):
    root = _image_tree(tmp_path)
    manifest = _manifest(root)
    (root / "cam1" / "nested" / "frame2.png").unlink()
    result = verify(manifest, root, ("cam0", "cam1"), "nested")
    assert result.status == "mismatch"
    assert result.unavailable["cam1"] == ("frame2.png",)
    assert result.photo_names["cam1"] == ("frame1.png", "frame2.png", "frame10.png")


def test_run_context_inherits_manifest_verdict_and_withholds_fresh_phase2(tmp_path):
    from pyCamSet.workflow.education_trace import RunContext
    from pyCamSet.workflow.workspace import WorkspaceManager

    source = _image_tree(tmp_path / "images")
    workspace = tmp_path / "workspace"
    manager = WorkspaceManager(workspace)
    manifest = _manifest(source)
    phase1 = {
        "run_id": "p1",
        "params": {"f_loc": str(source), "n_lim": None,
                   "subfolder_string": "nested",
                   "selected_cameras": ["cam0", "cam1"]},
        "diagnostics": {"education_manifest": manifest,
                        "education_cache": {"reused": False}},
        "artifacts": {"detected_datapoints_pickle": str(
            workspace / "phase1_runs" / "p1" / "detected_datapoints.pickle")},
    }
    (workspace / "phase1_runs" / "p1").mkdir(parents=True)
    (workspace / "phase1_runs" / "p1" / "detected_datapoints.pickle").write_bytes(b"detections")
    manager.save_run("phase1", "p1", phase1)
    context = RunContext.from_run(manager, {
        "run_id": "p1", "_run_dir": workspace / "phase1_runs" / "p1",
        **phase1,
    })
    assert context.phase1_source_check().status == "verified by manifest"
    legacy_hit = RunContext.from_run(manager, {
        "run_id": "legacy-hit", "_run_dir": workspace / "phase1_runs" / "legacy-hit",
        "params": {"f_loc": str(source)},
        "diagnostics": {"education_manifest": None,
                        "education_cache": {"reused": True}},
    })
    assert legacy_hit.phase1_source_check().status == "could not be checked"

    edited = source / "cam0" / "nested" / "frame1.png"
    edited.write_bytes(b"changed source photo")
    assert context.phase1_source_check().status == "mismatch"

    phase2 = {
        "run_id": "p2", "inputs": {"phase1_run_id": "p1"},
        "params": {},
        "artifacts": {"education_used_detections": {
            "source": "fresh_pass_not_saved", "path": None}},
    }
    manager.save_run("phase2", "p2", phase2)
    phase2_context = RunContext.from_run(manager, {
        "run_id": "p2", "_run_dir": workspace / "phase2_runs" / "p2",
        **phase2,
    })
    assert phase2_context.phase1_source_check().status == "could not be checked"
