"""Exercise the Qt-free education provenance and trace contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyCamSet.workflow.education_trace import (
    AncestryReason,
    Evidence,
    Frame,
    FrameKind,
    FreshPassReason,
    LossSettings,
    Origin,
    PoseMappingReason,
    RunContext,
    SourceCheck,
    Verification,
    atomic_write_json,
    choose_origin,
    classify_legacy_pose_mapping,
    _legacy_phase1_pose_mapping,
    compare_start_settings,
    detections_equal,
    normalise_recorded_path,
    phase4_ancestry_reason,
    phase3_before_metric,
    saved_fixed_pose_index,
    make_used_detections_record,
    load_run_pickle,
    loss_settings_from_options,
    read_trace,
    reprojection_segment_mean,
    scalar_matches,
    trace_path,
    walk_input_ancestry,
    write_trace,
)
from pyCamSet.workflow.workspace import WorkspaceManager, delete_file

# A2 hardening intentionally withholds legacy photo names when a capped
# re-glob could have shifted poses; these are the only changed semantic leaves
# measured by the paired n_lim probe. The duplicate delete_file removal is
# behaviour-neutral, declared here as an implementation change, not an output.
EQUIVALENCE_DECLARATIONS = (
    "workspace.delete_file",
    "diagnostics.photo_names",
    "diagnostics.pose_mapping_reason",
)


def test_frame_has_a_total_kind_to_evidence_mapping_and_default_source_check():
    for kind in FrameKind:
        frame = Frame("stage", "phase1", "Title", kind)
        assert frame.evidence is not None
        assert frame.source_check == "not checkable"
        assert len(frame.layer_provenance()) == 3
    assert Frame("diagram", "phase1", "Diagram", FrameKind.ALGORITHM_DIAGRAM).evidence == Evidence.DIAGRAM
    assert Frame("none", "phase1", "None", FrameKind.UNAVAILABLE).evidence == Evidence.NOT_RECORDED


def test_origin_precedence_is_total_and_uses_the_highest_priority():
    assert set(Origin) == set(Origin(value) for value in Origin)
    assert choose_origin([Origin.THIS_RUN, Origin.VERIFIED_UPSTREAM]) == Origin.VERIFIED_UPSTREAM
    assert choose_origin([Origin.NONE, Origin.TRIAL]) == Origin.TRIAL
    assert choose_origin([]) is Origin.NONE


def test_path_normalisation_refuses_relative_paths_and_folds_windows_case():
    assert normalise_recorded_path("phase3_runs/run-a/file.camset") is None
    assert normalise_recorded_path(r"C:\Rig\.pycamset_workspace\phase3_runs\R1\x.camset") == (
        "phase3_runs", "r1", "x.camset")
    assert normalise_recorded_path(r"c:\rig\.PYCAMSET_WORKSPACE\PHASE3_RUNS\r1\x.camset") == (
        "phase3_runs", "r1", "x.camset")


def test_phase4_ancestry_reason_distinguishes_another_folder_from_moved_workspace(tmp_path: Path):
    current = tmp_path / "new" / ".pycamset_workspace"
    old = tmp_path / "old" / ".pycamset_workspace"
    other = tmp_path / "other" / ".pycamset_workspace"
    used = other / "phase3_runs" / "p3" / "optimised_cameras.camset"
    assert phase4_ancestry_reason(
        used, current, old, local_run_copy_exists=False,
    ) is AncestryReason.ANOTHER_IMAGE_FOLDER
    assert phase4_ancestry_reason(
        used, current, other, local_run_copy_exists=False,
    ) is AncestryReason.CANNOT_TRACE
    assert phase4_ancestry_reason(
        used, current, old, local_run_copy_exists=True,
    ) is AncestryReason.CANNOT_TRACE
    assert phase4_ancestry_reason(
        "relative/.pycamset_workspace/phase3_runs/p3/file.camset",
        current, old, local_run_copy_exists=False,
    ) is AncestryReason.CANNOT_TRACE


def test_context_uses_run_directory_phase_not_metadata_phase(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    phase2_run = {"run_id": "p2", "phase": "phase4", "_run_dir": str(workspace / "phase2_runs" / "p2")}
    context = RunContext.from_run(manager, phase2_run)
    assert context.phase == "phase2"
    assert context.run_dir == workspace / "phase2_runs" / "p2"
    with pytest.raises(ValueError, match="does not match"):
        RunContext.from_run(manager, phase2_run, phase="phase1")


def test_failure_takes_precedence_over_not_checked_and_legacy_is_not_checked(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    failed = RunContext.from_run(manager, {
        "run_id": "bad", "_run_dir": str(workspace / "phase1_runs" / "bad"),
        "error": "boom", "report": None, "diagnostics": {},
    })
    legacy = RunContext.from_run(manager, {
        "run_id": "old", "_run_dir": str(workspace / "phase1_runs" / "old"),
        "report": None, "diagnostics": {},
    })
    assert failed.status == "failed"
    assert failed.quality_check == "failed"
    assert legacy.quality_check == "not checked"


def test_trace_opt_in_atomic_roundtrip_and_run_identity(tmp_path: Path):
    run_dir = tmp_path / "phase3_runs" / "run-1"
    run_dir.mkdir(parents=True)
    assert write_trace(run_dir, "run-1", "phase3", "accepted_step", {"nit": 1}, enabled=False) is None
    path = write_trace(run_dir, "run-1", "phase3", "accepted_step", {"nit": 1}, enabled=True)
    assert path == trace_path(run_dir)
    loaded = read_trace(run_dir, "run-1", "phase3")
    assert loaded["status"] == "available"
    assert loaded["records"][0]["payload"] == {"nit": 1}
    assert read_trace(run_dir, "run-2", "phase3")["reason"] == "trace belongs to another run"


def test_trace_sidecar_is_run_relative_after_workspace_copy(tmp_path: Path):
    original = tmp_path / "original" / "phase3_runs" / "run-1"
    original.mkdir(parents=True)
    write_trace(original, "run-1", "phase3", "stage", {"step": 2}, enabled=True)
    copied = tmp_path / "copied" / "phase3_runs" / "run-1"
    copied.parent.mkdir(parents=True)
    shutil.copytree(original, copied)
    result = read_trace(copied, "run-1", "phase3")
    assert result["status"] == "available"
    assert result["records"][0]["payload"] == {"step": 2}


def test_trace_schema_v1_migrates_missing_eviction_field_and_rejects_v0(tmp_path: Path):
    run_dir = tmp_path / "phase1_runs" / "run-1"
    run_dir.mkdir(parents=True)
    record = {"run_id": "run-1", "phase": "phase1", "kind": "stage",
              "payload": {"name": "detect"}}
    body = {"schema_version": 1, "run_id": "run-1", "phase": "phase1",
            "records": [record]}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    body["sha256"] = hashlib.sha256(canonical).hexdigest()
    atomic_write_json(trace_path(run_dir), body)
    migrated = read_trace(run_dir, "run-1", "phase1")
    assert migrated["status"] == "available"
    assert migrated["evicted_kinds"] == []

    body["schema_version"] = 0
    body.pop("sha256")
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    body["sha256"] = hashlib.sha256(canonical).hexdigest()
    atomic_write_json(trace_path(run_dir), body)
    assert read_trace(run_dir, "run-1", "phase1")["status"] == "unavailable"


def test_trace_corruption_and_oversize_are_unavailable(tmp_path: Path):
    run_dir = tmp_path / "phase1_runs" / "run-1"
    run_dir.mkdir(parents=True)
    write_trace(run_dir, "run-1", "phase1", "stage", {"x": 1}, enabled=True)
    path = trace_path(run_dir)
    content = json.loads(path.read_text(encoding="utf-8"))
    content["records"][0]["payload"]["x"] = 9
    path.write_text(json.dumps(content), encoding="utf-8")
    assert read_trace(run_dir, "run-1", "phase1")["status"] == "unavailable"
    assert read_trace(run_dir, "run-1", "phase1", max_bytes=1)["status"] == "unavailable"


def test_trace_rejects_unknown_kind_and_never_silently_drops_join_records(tmp_path: Path):
    run_dir = tmp_path / "phase2_runs" / "run-1"
    run_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="Unknown"):
        write_trace(run_dir, "run-1", "phase2", "mystery", {}, enabled=True)
    for i in range(3):
        write_trace(run_dir, "run-1", "phase2", "phase2_per_view", {"i": i},
                    enabled=True, max_records=1)
    loaded = read_trace(run_dir, "run-1", "phase2")
    assert loaded["status"] == "unavailable"
    assert loaded["reason"] == "trace records were evicted"


def test_reprojection_mean_uses_only_two_values_per_optimised_row():
    # The trailing pair stands for a prior and must not enter the statistic.
    assert reprojection_segment_mean([3, 4, 0, 5, 100, 100], 2) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="shorter"):
        reprojection_segment_mean([1, 2], 2)
    with pytest.raises(ValueError, match="no optimised"):
        reprojection_segment_mean([], 0)


def test_metric_comparison_has_an_absolute_near_zero_floor():
    assert scalar_matches(0.0, 5e-10)
    assert not scalar_matches(1e-3, 1e-3 + 2e-9)
    assert scalar_matches(2.0, 2.0 + 1e-9)
    assert not scalar_matches(float("nan"), 0.0)


@pytest.mark.parametrize("value", [True, False, -1, 3, 1.5, float("inf"), "1"])
def test_saved_fixed_pose_index_rejects_invalid_values(value):
    assert saved_fixed_pose_index(
        {"fixed_pose": value}, SimpleNamespace(max_ims=3)) is None


def test_saved_fixed_pose_index_preserves_zero_based_index_only_with_evidence():
    detections = SimpleNamespace(max_ims=3)
    assert saved_fixed_pose_index({"fixed_pose": 0}, detections) == 0
    assert saved_fixed_pose_index({"fixed_pose": 2.0}, detections) == 2
    assert saved_fixed_pose_index({}, detections) is None
    assert saved_fixed_pose_index({"fixed_pose": 1}, SimpleNamespace()) is None


def test_loss_settings_preserve_f_scale_and_default_legacy_loss():
    assert loss_settings_from_options({}) == LossSettings("linear", 1.0)
    small_scale = loss_settings_from_options({"loss": "huber", "f_scale": 0.5})
    large_scale = loss_settings_from_options({"loss": "huber", "f_scale": 2.0})
    assert small_scale.loss == large_scale.loss == "huber"
    assert small_scale.f_scale == 0.5
    assert large_scale.f_scale == 2.0


def test_legacy_pose_mapping_reasons_are_total_and_follow_precedence():
    names = ("a.jpg", "b.jpg", "c.jpg")
    list_mismatch = classify_legacy_pose_mapping(
        caching=True, promoted=False, max_ims=4, current_photo_names=names,
        flat_without_tif=True, report={"n_images_expected": 2, "n_images": 3},
    )
    assert list_mismatch.reason is PoseMappingReason.PHOTO_LIST_CHANGED
    stale_hit = classify_legacy_pose_mapping(
        caching=True, promoted=False, max_ims=3, current_photo_names=names,
        flat_without_tif=True, report={"n_images_expected": 2, "n_images": 3},
    )
    assert stale_hit.reason is PoseMappingReason.POSSIBLE_STALE_HIT
    count_change = classify_legacy_pose_mapping(
        caching=False, promoted=False, max_ims=3, current_photo_names=names,
        flat_without_tif=True, report={"n_images_expected": 2, "n_images": 1},
    )
    assert count_change.reason is PoseMappingReason.IMAGE_COUNT_CHANGED
    cache_uncertain = classify_legacy_pose_mapping(
        caching=True, promoted=False, max_ims=3, current_photo_names=names,
        flat_without_tif=True, report=None,
    )
    assert cache_uncertain.reason is PoseMappingReason.CACHE_REUSE
    layout_uncertain = classify_legacy_pose_mapping(
        caching=False, promoted=False, max_ims=3, current_photo_names=names,
        flat_without_tif=False, report=None,
    )
    assert layout_uncertain.reason is PoseMappingReason.LAYOUT_UNCOMPARABLE
    good_legacy = classify_legacy_pose_mapping(
        caching=False, promoted=False, max_ims=2, current_photo_names=names,
        flat_without_tif=True, report={"n_images_expected": 3, "n_images": 2},
    )
    assert good_legacy.reason is None
    assert good_legacy.photo_names == names
    promoted = classify_legacy_pose_mapping(
        caching=None, promoted=True, max_ims=2, current_photo_names=names,
        flat_without_tif=True, report=None,
    )
    assert promoted.reason is None
    assert promoted.photo_names == names

    malformed_count = classify_legacy_pose_mapping(
        caching=False, promoted=False, max_ims=2.5, current_photo_names=names,
        flat_without_tif=True, report=None,
    )
    assert malformed_count.reason is PoseMappingReason.LAYOUT_UNCOMPARABLE


def test_legacy_phase1_mapping_uses_natural_camera_file_order(tmp_path: Path):
    image_root = tmp_path / "images"
    camera = image_root / "cam0"
    camera.mkdir(parents=True)
    for name in ("frame10.png", "frame2.png"):
        (camera / name).write_bytes(b"")
    phase1 = {
        "params": {"f_loc": str(image_root), "caching": False, "n_lim": None},
        "report": {
            "n_images": 2,
            "per_camera": [{"name": "cam0", "n_images_expected": 2}],
        },
    }
    manager = SimpleNamespace(
        find_run=lambda phase, *_: phase1 if phase == "phase1" else {})
    ancestry = (
        {"parent_phase": "phase1", "parent_run_id": "p1",
         "verification": Verification.VERIFIED.value},
        {"parent_phase": "phase2", "parent_run_id": "p2",
         "verification": Verification.VERIFIED.value},
    )
    result = _legacy_phase1_pose_mapping(
        manager, ancestry, SimpleNamespace(cam_names=["cam0"], max_ims=2))
    assert result.reason is None
    assert result.photo_names == ("cam0/frame2.png",)
    assert result.source_check is SourceCheck.NOT_CHECKABLE


def test_legacy_phase1_mapping_never_resolves_relative_image_root():
    phase1 = {"params": {"f_loc": ".", "caching": False}}
    manager = SimpleNamespace(
        find_run=lambda phase, *_: phase1 if phase == "phase1" else {})
    ancestry = (
        {"parent_phase": "phase1", "parent_run_id": "p1",
         "verification": Verification.VERIFIED.value},
        {"parent_phase": "phase2", "parent_run_id": "p2",
         "verification": Verification.VERIFIED.value},
    )
    result = _legacy_phase1_pose_mapping(
        manager, ancestry, SimpleNamespace(cam_names=["cam0"], max_ims=1))
    assert result.reason is AncestryReason.CANNOT_TRACE
    assert result.photo_names == ()


def test_legacy_phase1_mapping_reports_photo_change_before_cache_reuse(tmp_path: Path):
    image_root = tmp_path / "images"
    camera = image_root / "cam0"
    camera.mkdir(parents=True)
    (camera / "frame0.png").write_bytes(b"")
    phase1 = {
        "params": {"f_loc": str(image_root), "caching": True, "n_lim": None},
        "report": {
            "n_images": 2,
            "per_camera": [{"name": "cam0", "n_images_expected": 2}],
        },
    }
    manager = SimpleNamespace(
        find_run=lambda phase, *_: phase1 if phase == "phase1" else {})
    ancestry = (
        {"parent_phase": "phase1", "parent_run_id": "p1",
         "verification": Verification.VERIFIED.value},
        {"parent_phase": "phase2", "parent_run_id": "p2",
         "verification": Verification.VERIFIED.value},
    )
    result = _legacy_phase1_pose_mapping(
        manager, ancestry, SimpleNamespace(cam_names=["cam0"], max_ims=2))
    assert result.reason is PoseMappingReason.PHOTO_LIST_CHANGED
    assert result.source_check is SourceCheck.MISMATCH
    assert result.photo_names == ()


def test_legacy_cache_hit_delete_and_same_bytes_mtime_restore_never_names_photo(tmp_path: Path):
    image_root = tmp_path / "images"
    camera = image_root / "cam0"
    camera.mkdir(parents=True)
    first = camera / "frame0.png"
    second = camera / "frame1.png"
    first.write_bytes(b"same source bytes")
    second.write_bytes(b"second image")
    original_stat = first.stat()
    phase1 = {
        "params": {"f_loc": str(image_root), "caching": True, "n_lim": None},
        "report": {"n_images": 2,
                   "per_camera": [{"name": "cam0", "n_images_expected": 2}]},
    }
    manager = SimpleNamespace(
        find_run=lambda phase, *_: phase1 if phase == "phase1" else {})
    ancestry = (
        {"parent_phase": "phase1", "parent_run_id": "p1",
         "verification": Verification.VERIFIED.value},
        {"parent_phase": "phase2", "parent_run_id": "p2",
         "verification": Verification.VERIFIED.value},
    )
    cached_detections = SimpleNamespace(cam_names=["cam0"], max_ims=2)

    # A cache hit carries the old detections but no writer-side image list.
    first.unlink()
    during_deletion = _legacy_phase1_pose_mapping(manager, ancestry, cached_detections)
    assert during_deletion.reason is PoseMappingReason.PHOTO_LIST_CHANGED
    assert during_deletion.photo_names == ()

    first.write_bytes(b"same source bytes")
    os.utime(first, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    restored = _legacy_phase1_pose_mapping(manager, ancestry, cached_detections)
    assert first.read_bytes() == b"same source bytes"
    assert first.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert restored.reason is PoseMappingReason.CACHE_REUSE
    assert restored.photo_names == ()


def test_legacy_case_r_stale_hit_after_same_bytes_mtime_restore_is_not_a_mismatch(tmp_path: Path):
    image_root = tmp_path / "images"
    camera = image_root / "cam0"
    camera.mkdir(parents=True)
    first = camera / "frame0.png"
    second = camera / "frame1.png"
    first.write_bytes(b"same source bytes")
    second.write_bytes(b"second image")
    original_stat = first.stat()
    phase1 = {
        "params": {"f_loc": str(image_root), "caching": True, "n_lim": None},
        # The cache report's larger source count is the stale-hit evidence.
        "report": {
            "n_images": 2,
            "per_camera": [{"name": "cam0", "n_images_expected": 1}],
        },
    }
    manager = SimpleNamespace(
        find_run=lambda phase, *_: phase1 if phase == "phase1" else {})
    ancestry = (
        {"parent_phase": "phase1", "parent_run_id": "p1",
         "verification": Verification.VERIFIED.value},
        {"parent_phase": "phase2", "parent_run_id": "p2",
         "verification": Verification.VERIFIED.value},
    )
    cached_detections = SimpleNamespace(cam_names=["cam0"], max_ims=2)

    first.unlink()
    while_deleted = _legacy_phase1_pose_mapping(
        manager, ancestry, cached_detections)
    assert while_deleted.reason is PoseMappingReason.PHOTO_LIST_CHANGED
    assert while_deleted.photo_names == ()

    # Restoring identical bytes/mtime still cannot erase the run's stale-hit evidence.
    first.write_bytes(b"same source bytes")
    os.utime(first, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    result = _legacy_phase1_pose_mapping(manager, ancestry, cached_detections)
    assert first.read_bytes() == b"same source bytes"
    assert first.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert result.reason is PoseMappingReason.POSSIBLE_STALE_HIT
    assert result.photo_names == ()
    assert result.source_check is SourceCheck.NOT_CHECKABLE


def test_capped_n_lim_cannot_claim_current_pose_to_file_order(tmp_path: Path):
    image_root = tmp_path / "images"
    camera = image_root / "cam0"
    camera.mkdir(parents=True)
    for name in ("frame0.png", "frame1.png", "frame2.png"):
        (camera / name).write_bytes(name.encode("utf-8"))
    phase1 = {
        "params": {"f_loc": str(image_root), "caching": False, "n_lim": 2},
        "report": {"n_images": 2,
                   "per_camera": [{"name": "cam0", "n_images_expected": 2}]},
    }
    manager = SimpleNamespace(
        find_run=lambda phase, *_: phase1 if phase == "phase1" else {})
    ancestry = (
        {"parent_phase": "phase1", "parent_run_id": "p1",
         "verification": Verification.VERIFIED.value},
        {"parent_phase": "phase2", "parent_run_id": "p2",
         "verification": Verification.VERIFIED.value},
    )
    detections = SimpleNamespace(cam_names=["cam0"], max_ims=2)

    # The full listing exceeds the cap, so equal capped counts cannot prove
    # that an earlier deletion did not shift the detection indices.
    (camera / "frame0.png").unlink()
    result = _legacy_phase1_pose_mapping(manager, ancestry, detections)
    assert result.reason is PoseMappingReason.LAYOUT_UNCOMPARABLE
    assert result.photo_names == ()


@pytest.mark.parametrize(("arguments", "expected"), [
    ({"max_ims": 4, "current_photo_names": ("a", "b", "c"),
      "caching": True, "report": {"n_images_expected": 2, "n_images": 3}},
     PoseMappingReason.PHOTO_LIST_CHANGED),
    ({"max_ims": 3, "current_photo_names": ("a", "b", "c"),
      "caching": True, "report": {"n_images_expected": 2, "n_images": 3}},
     PoseMappingReason.POSSIBLE_STALE_HIT),
    ({"max_ims": 3, "current_photo_names": ("a", "b", "c"),
      "caching": False, "report": {"n_images_expected": 2, "n_images": 1}},
     PoseMappingReason.IMAGE_COUNT_CHANGED),
    ({"max_ims": 3, "current_photo_names": ("a", "b", "c"),
      "caching": True, "report": {"n_images_expected": 3, "n_images": 3},
      "listing_capped": True}, PoseMappingReason.CACHE_REUSE),
    ({"max_ims": 3, "current_photo_names": ("a", "b", "c"),
      "caching": False, "report": None, "listing_capped": True},
     PoseMappingReason.LAYOUT_UNCOMPARABLE),
])
def test_legacy_reason_precedence_has_falsifying_trigger_pairs(arguments, expected):
    result = classify_legacy_pose_mapping(
        promoted=False, flat_without_tif=True,
        **{"max_ims": 3, "current_photo_names": ("a", "b", "c"),
           "caching": False, "report": None, **arguments},
    )
    assert result.reason is expected


def test_legacy_pose_mapping_is_not_applied_to_a_fresh_phase2_detection():
    phase1 = {"params": {"f_loc": "unused", "caching": False}}
    phase2 = {"artifacts": {"education_used_detections": {
        "schema_version": 1, "source": "fresh_pass_not_saved",
    }}}
    manager = SimpleNamespace(
        find_run=lambda phase, *_: phase1 if phase == "phase1" else phase2)
    ancestry = (
        {"parent_phase": "phase1", "parent_run_id": "p1",
         "verification": Verification.VERIFIED.value},
        {"parent_phase": "phase2", "parent_run_id": "p2",
         "verification": Verification.VERIFIED.value},
    )
    result = _legacy_phase1_pose_mapping(
        manager, ancestry, SimpleNamespace(cam_names=["cam0"], max_ims=2))
    assert result.reason is None
    assert result.photo_names == ()
    assert result.source_check is SourceCheck.NOT_APPLICABLE


def test_used_detections_schema_records_fresh_reasons_in_fixed_order():
    record = make_used_detections_record(
        source="fresh_pass_not_saved", path=None,
        fresh_pass_reasons=[FreshPassReason.CAMERA_SUBSET.value,
                            FreshPassReason.MISSING_CAMERA_RESOLUTION.value],
        cache_hit=False,
    )
    assert record == {
        "schema_version": 1,
        "source": "fresh_pass_not_saved",
        "path": None,
        "fresh_pass_reasons": ["camera_subset", "no_cam_res"],
        "cache_hit": False,
    }
    with pytest.raises(ValueError, match="contract order"):
        make_used_detections_record(
            source="fresh_pass_not_saved", path=None,
            fresh_pass_reasons=["no_cam_res", "camera_subset"],
        )


def test_atomic_json_write_and_duplicate_delete_file_fix(tmp_path: Path):
    output = tmp_path / "nested" / "record.json"
    atomic_write_json(output, {"label": "café"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"label": "café"}
    delete_file(output)
    delete_file(output)
    assert not output.exists()


def test_run_pickle_loader_checks_run_boundary_and_loaded_type(tmp_path: Path):
    run_dir = tmp_path / "phase1_runs" / "run-1"
    run_dir.mkdir(parents=True)
    private_pickle = run_dir / "payload.pickle"
    outside_pickle = tmp_path / "outside.pickle"
    from pyCamSet.utils.saving import save_pickle
    save_pickle({"ok": True}, private_pickle)
    save_pickle({"ok": True}, outside_pickle)
    value, reason = load_run_pickle(private_pickle, dict, expected_run_dir=run_dir)
    assert value == {"ok": True}
    assert reason is None
    value, reason = load_run_pickle(outside_pickle, dict, expected_run_dir=run_dir)
    assert value is None
    assert reason == "pickle is outside the expected run directory"
    value, reason = load_run_pickle(private_pickle, list, expected_run_dir=run_dir)
    assert value is None
    assert reason == "pickle payload has an unexpected type"


def test_historical_preflatten_camset_fixture_loads_as_legacy_camera_set():
    fixture = (Path(__file__).parent / "test_data" / "education" / "legacy"
               / "phase4_325fae4_preflatten.camset.json")
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    assert payload["optim"]["target_config"]["target_module"] == (
        "pyCamSet.calibration_targets.target_Ccube")
    from pyCamSet.utils.saving import load_CameraSet
    camera_set = load_CameraSet(fixture)
    # The old target module is absent in the flattened package. pycamsetDev's loader finds a moved
    # class by its name, so the target (and the solver handler) are rebuilt now; the education
    # line used to stop at the bare camera set here. The cameras are preserved either way.
    assert camera_set.n_cams == 3
    assert camera_set.calibration_handler is not None


def test_historical_phase4_camset_fixture_loads_with_legacy_solver_state():
    fixture = (Path(__file__).parent / "test_data" / "education" / "legacy"
               / "phase4_325fae4.camset.json")
    from pyCamSet.utils.saving import load_CameraSet
    camera_set = load_CameraSet(fixture)
    handler = camera_set.calibration_handler
    assert camera_set.get_names() == ["1", "2", "3"]
    assert handler is not None
    assert len(camera_set.calibration_params) == 1077
    assert handler.detection.max_ims == 22


def test_historical_phase1_pickle_fixture_loads_as_detection_and_resolution_pair():
    fixture = (Path(__file__).parent / "test_data" / "education" / "legacy"
               / "phase1_325fae4.pickle")
    from pyCamSet.calibration_targets.core.target_detections import TargetDetection
    from pyCamSet.utils.saving import load_pickle
    payload = load_pickle(fixture)
    assert isinstance(payload, tuple) and len(payload) == 2
    detections, resolution = payload
    assert isinstance(detections, TargetDetection)
    assert detections.cam_names == ["1", "2", "3"]
    assert detections.max_ims == 22
    assert len(resolution) == 3
    assert all(tuple(size) == (1024, 1280) for size in resolution)


def test_resolve_parents_keeps_multiple_explicit_edges_and_never_latest(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    manager.save_run("phase1", "p1-a", {"run_id": "p1-a"})
    manager.save_run("phase1", "p1-newest", {"run_id": "p1-newest"})
    manager.save_run("phase2", "p2-a", {"run_id": "p2-a"})
    run = {"inputs": {"phase1_run_id": "p1-a", "phase2_run_id": "p2-a"}}
    parents = manager.resolve_parents(run)
    assert [(p["phase"], p["run_id"]) for p in parents] == [
        ("phase1", "p1-a"), ("phase2", "p2-a")]
    # The important identity invariant is that an absent explicit parent is
    # not replaced by the newest Phase 1 record.
    run["inputs"]["phase1_run_id"] = "missing"
    assert [(p["phase"], p["run_id"]) for p in manager.resolve_parents(run)] == [
        ("phase2", "p2-a")]
    assert manager.find_run("phase1", "missing") is None


def test_ancestry_walk_records_missing_direct_parent_beside_valid_linear_chain(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    p2_dir = workspace / "phase2_runs" / "p2"
    p2_dir.mkdir(parents=True)
    p2_output = p2_dir / "initial_cameras.camset"
    p2_output.write_bytes(b"p2")
    manager.save_run("phase2", "p2", {
        "run_id": "p2", "artifacts": {"initial_camset": str(p2_output)},
    })
    p3_dir = workspace / "phase3_runs" / "p3"
    p3_dir.mkdir(parents=True)
    p3_output = p3_dir / "optimised_cameras.camset"
    p3_output.write_bytes(b"p3")
    p1_used = workspace / "phase1_runs" / "absent" / "detected_datapoints.pickle"
    manager.save_run("phase3", "p3", {
        "run_id": "p3",
        "inputs": {"phase1_run_id": "missing", "phase2_run_id": "p2"},
        "artifacts": {
            "optimised_camset": str(p3_output),
            "phase1_detection_pickle_used": str(p1_used),
            "phase2_initial_camset_used": str(p2_output),
        },
    })
    loaded = manager.find_run("phase3", "p3")
    ancestry = walk_input_ancestry(manager, loaded)
    edges = {(item["child_phase"], item["parent_phase"]): item for item in ancestry}
    assert edges[("phase3", "phase2")]["verification"] == Verification.VERIFIED.value
    assert edges[("phase3", "phase1")]["verification"] == Verification.UNCHECKABLE.value
    assert edges[("phase3", "phase1")]["reason"] == "missing explicit parent"


def test_input_edge_returns_uncheckable_when_ancestor_or_used_file_is_missing(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    run_dir = workspace / "phase3_runs" / "p3"
    run_dir.mkdir(parents=True)
    metadata = {"run_id": "p3", "_run_dir": str(run_dir),
                "inputs": {"phase2_run_id": "missing"},
                "artifacts": {"phase2_initial_camset_used": "relative.camset"}}
    ctx = RunContext.from_run(manager, metadata)
    edge = ctx.verify_input("phase2")
    assert edge.verification is Verification.UNCHECKABLE
    assert edge.reason == "missing parent or used artifact"


def test_input_edge_names_verified_artifact_read_from_another_workspace_copy(tmp_path: Path):
    original_workspace = tmp_path / "original" / ".pycamset_workspace"
    copied_workspace = tmp_path / "copy" / ".pycamset_workspace"
    original_run = original_workspace / "phase2_runs" / "p2"
    copied_run = copied_workspace / "phase2_runs" / "p2"
    original_run.mkdir(parents=True)
    copied_run.mkdir(parents=True)
    original_artifact = original_run / "initial_cameras.camset"
    copied_artifact = copied_run / "initial_cameras.camset"
    original_artifact.write_bytes(b"same camset")
    copied_artifact.write_bytes(b"same camset")
    manager = WorkspaceManager(copied_workspace)
    manager.save_run("phase2", "p2", {
        "run_id": "p2", "artifacts": {"initial_camset": str(original_artifact)},
    })
    p3_dir = copied_workspace / "phase3_runs" / "p3"
    p3_dir.mkdir(parents=True)
    context = RunContext.from_run(manager, {
        "run_id": "p3", "_run_dir": str(p3_dir),
        "inputs": {"phase2_run_id": "p2"},
        "artifacts": {"phase2_initial_camset_used": str(copied_artifact)},
    })
    edge = context.verify_input("phase2")
    assert edge.verification is Verification.VERIFIED
    assert edge.reason == "verified (read from another copy of this workspace)"


def test_rerun_comparison_checks_every_phase3_used_input(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    source = RunContext.from_run(manager, {
        "run_id": "source", "_run_dir": str(workspace / "phase3_runs" / "source"),
        "artifacts": {
            "phase2_initial_camset_used": str(workspace / "phase2_runs" / "p2" / "initial.camset"),
            "phase1_detection_pickle_used": str(workspace / "phase1_runs" / "p1" / "detections.pickle"),
        },
    })
    rerun = RunContext.from_run(manager, {
        "run_id": "rerun", "_run_dir": str(workspace / "phase3_runs" / "rerun"),
        "artifacts": {
            "phase2_initial_camset_used": str(workspace / "phase2_runs" / "p2" / "initial.camset"),
            "phase1_detection_pickle_used": str(workspace / "phase1_runs" / "other" / "detections.pickle"),
        },
    })
    assert rerun.used_input_matches(source) is Verification.DIFFERS


def test_phase2_fresh_detection_never_verifies_against_phase1(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    parent_dir = workspace / "phase1_runs" / "p1"
    parent_dir.mkdir(parents=True)
    phase1_pickle = parent_dir / "detected_datapoints.pickle"
    phase1_pickle.write_bytes(b"parent")
    manager.save_run("phase1", "p1", {
        "run_id": "p1", "artifacts": {"detected_datapoints_pickle": str(phase1_pickle)},
    })
    p2_dir = workspace / "phase2_runs" / "p2"
    p2_dir.mkdir(parents=True)
    context = RunContext.from_run(manager, {
        "run_id": "p2", "_run_dir": str(p2_dir),
        "inputs": {"phase1_run_id": "p1"},
        "education_used_detections": {
            "schema_version": 1, "source": "fresh_pass_not_saved",
            "path": None, "fresh_pass_reasons": ["camera_subset"], "cache_hit": True,
        },
    })
    assert context.verify_input("phase1").verification is Verification.UNCHECKABLE


def test_phase2_used_detections_record_under_artifacts_is_authoritative(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    parent_dir = workspace / "phase1_runs" / "p1"
    parent_dir.mkdir(parents=True)
    phase1_pickle = parent_dir / "detected_datapoints.pickle"
    phase1_pickle.write_bytes(b"phase1")
    manager.save_run("phase1", "p1", {
        "run_id": "p1", "artifacts": {"detected_datapoints_pickle": str(phase1_pickle)},
    })
    p2_dir = workspace / "phase2_runs" / "p2"
    p2_dir.mkdir(parents=True)
    context = RunContext.from_run(manager, {
        "run_id": "p2", "_run_dir": str(p2_dir),
        "inputs": {"phase1_run_id": "p1"},
        "artifacts": {
            "education_used_detections": {
                "schema_version": 1, "source": "fresh_pass_not_saved",
                "path": None, "fresh_pass_reasons": ["camera_subset"], "cache_hit": False,
            },
            "phase1_detection_pickle": str(phase1_pickle),
        },
    })
    edge = context.verify_input("phase1")
    assert edge.verification is Verification.UNCHECKABLE
    assert edge.used_path is None


def test_fresh_pass_reason_names_match_phase2_producer_contract(tmp_path: Path):
    record = make_used_detections_record(
        source="fresh_pass_not_saved", path=None,
        fresh_pass_reasons=["camera_subset", "no_cam_res", "pickle_missing",
                            "high_distortion_second_pass"],
    )
    assert record["fresh_pass_reasons"] == [
        "camera_subset", "no_cam_res", "pickle_missing", "high_distortion_second_pass"]
    override = make_used_detections_record(
        source="override_pickle", path=str(tmp_path / "override.pickle"))
    assert override["source"] == "override_pickle"


def test_phase2_legacy_edge_verifies_genuine_325fae4_pickle_when_all_guards_pass(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    p1_dir = workspace / "phase1_runs" / "p1"
    p1_dir.mkdir(parents=True)
    p1_pickle = p1_dir / "detected_datapoints.pickle"
    source_pickle = (Path(__file__).parent / "test_data" / "education" / "legacy"
                     / "phase1_325fae4.pickle")
    shutil.copyfile(source_pickle, p1_pickle)
    manager.save_run("phase1", "p1", {
        "run_id": "p1", "artifacts": {"detected_datapoints_pickle": str(p1_pickle)},
    })
    p2_dir = workspace / "phase2_runs" / "p2"
    p2_dir.mkdir(parents=True)
    context = RunContext.from_run(manager, {
        "run_id": "p2", "_run_dir": str(p2_dir),
        "created_at": "2999-01-01T00:00:00",
        "inputs": {"phase1_run_id": "p1"},
        "params": {"high_distortion": False, "selected_cameras": []},
        "artifacts": {"phase1_detection_pickle": str(p1_pickle)},
    })
    assert context.verify_input("phase1").verification is Verification.VERIFIED


def test_phase2_legacy_historical_pickle_is_not_verified_when_it_postdates_run(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    p1_dir = workspace / "phase1_runs" / "p1"
    p1_dir.mkdir(parents=True)
    p1_pickle = p1_dir / "detected_datapoints.pickle"
    shutil.copyfile(
        Path(__file__).parent / "test_data" / "education" / "legacy"
        / "phase1_325fae4.pickle", p1_pickle)
    manager.save_run("phase1", "p1", {
        "run_id": "p1", "artifacts": {"detected_datapoints_pickle": str(p1_pickle)},
    })
    p2_dir = workspace / "phase2_runs" / "p2"
    p2_dir.mkdir(parents=True)
    context = RunContext.from_run(manager, {
        "run_id": "p2", "_run_dir": str(p2_dir),
        "created_at": "1999-01-01T00:00:00",
        "inputs": {"phase1_run_id": "p1"},
        "params": {"high_distortion": False, "selected_cameras": []},
        "artifacts": {"phase1_detection_pickle": str(p1_pickle)},
    })
    assert context.verify_input("phase1").verification is Verification.UNCHECKABLE


def test_phase2_legacy_edge_without_cam_resolution_is_not_provably_equal(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    parent_dir = workspace / "phase1_runs" / "p1"
    parent_dir.mkdir(parents=True)
    phase1_pickle = parent_dir / "detected_datapoints.pickle"
    from pyCamSet.utils.saving import save_pickle
    save_pickle({"cam_names": ["cam1"]}, phase1_pickle)
    manager.save_run("phase1", "p1", {
        "run_id": "p1", "artifacts": {"detected_datapoints_pickle": str(phase1_pickle)},
    })
    p2_dir = workspace / "phase2_runs" / "p2"
    p2_dir.mkdir(parents=True)
    context = RunContext.from_run(manager, {
        "run_id": "p2", "_run_dir": str(p2_dir),
        "created_at": "2999-01-01T00:00:00",
        "inputs": {"phase1_run_id": "p1"},
        "params": {"high_distortion": False, "selected_cameras": []},
        "artifacts": {"phase1_detection_pickle": str(phase1_pickle)},
    })
    assert context.verify_input("phase1").verification is Verification.UNCHECKABLE


def test_promoted_marker_is_literal_not_truthy(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    context = RunContext.from_run(manager, {
        "run_id": "p1", "_run_dir": str(workspace / "phase1_runs" / "p1"),
        "education": {"trial_camset_snapshot_version": True},
    })
    assert context.promoted_camset_is_marked() is False
    context.metadata["education"]["trial_camset_snapshot_version"] = 1
    assert context.promoted_camset_is_marked() is True


def test_phase3_state_source_rejects_relative_path_before_normalisation(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    run_dir = workspace / "phase4_runs" / "p4"
    run_dir.mkdir(parents=True)
    context = RunContext.from_run(manager, {
        "run_id": "p4", "_run_dir": str(run_dir),
        "artifacts": {
            "phase3_camset_used": "old/.pycamset_workspace/phase3_runs/p3/optimised_cameras.camset",
        },
    })
    state = context.phase3_state_source()
    assert state.path is None
    assert state.verification is Verification.UNCHECKABLE
    assert state.reason == "cannot be traced"
    assert state.ancestry == ()


def test_phase4_output_used_as_input_is_not_walked_as_phase3_ancestry(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    phase3_dir = workspace / "phase3_runs" / "p3"
    phase3_dir.mkdir(parents=True)
    phase3_camset = phase3_dir / "optimised_cameras.camset"
    phase3_camset.write_bytes(b"phase3")
    manager.save_run("phase3", "p3", {
        "run_id": "p3", "artifacts": {"optimised_camset": str(phase3_camset)},
    })
    phase4_dir = workspace / "phase4_runs" / "p4"
    phase4_dir.mkdir(parents=True)
    phase4_output = phase4_dir / "self_calibrated_cameras.camset"
    phase4_output.write_bytes(b"phase4")
    context = RunContext.from_run(manager, {
        "run_id": "p4", "_run_dir": str(phase4_dir),
        "inputs": {"phase3_run_id": "p3"},
        "artifacts": {"phase3_camset_used": str(phase4_output)},
    })
    state = context.phase3_state_source()
    assert state.path is None
    assert state.origin_run_id is None
    assert state.verification is Verification.UNCHECKABLE
    assert state.ancestry == ()


def test_deleted_phase3_parent_does_not_fall_back_to_newest_run(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    latest_dir = workspace / "phase3_runs" / "newest"
    latest_dir.mkdir(parents=True)
    latest_output = latest_dir / "optimised_cameras.camset"
    latest_output.write_bytes(b"newer unrelated run")
    manager.save_run("phase3", "newest", {
        "run_id": "newest", "artifacts": {"optimised_camset": str(latest_output)},
    })

    deleted_parent = workspace / "phase3_runs" / "deleted"
    used_output = deleted_parent / "optimised_cameras.camset"
    phase4_dir = workspace / "phase4_runs" / "p4"
    phase4_dir.mkdir(parents=True)
    context = RunContext.from_run(manager, {
        "run_id": "p4", "_run_dir": str(phase4_dir),
        "inputs": {"phase3_run_id": "deleted"},
        "artifacts": {"phase3_camset_used": str(used_output)},
    })
    state = context.phase3_state_source()
    edge = context.verify_input("phase3")
    assert manager.find_run("phase3", "deleted") is None
    assert state.path is None
    assert state.origin_run_id == "deleted"
    assert state.verification is Verification.UNCHECKABLE
    assert edge.verification is Verification.UNCHECKABLE
    assert "newest" not in str(state.path)


def test_moved_workspace_with_deleted_parent_is_not_mislabelled_another_folder(tmp_path: Path):
    old_workspace = tmp_path / "original" / ".pycamset_workspace"
    moved_workspace = tmp_path / "moved" / ".pycamset_workspace"
    manager = WorkspaceManager(moved_workspace)
    latest_dir = moved_workspace / "phase3_runs" / "newest"
    latest_dir.mkdir(parents=True)
    latest = latest_dir / "optimised_cameras.camset"
    latest.write_bytes(b"unrelated newer source")
    manager.save_run("phase3", "newest", {
        "run_id": "newest", "artifacts": {"optimised_camset": str(latest)},
    })

    phase4_dir = moved_workspace / "phase4_runs" / "p4"
    phase4_dir.mkdir(parents=True)
    context = RunContext.from_run(manager, {
        "run_id": "p4", "_run_dir": str(phase4_dir),
        "params": {"f_loc": str(old_workspace.parent)},
        "artifacts": {
            "self_calibrated_camset": str(old_workspace / "phase4_runs" / "p4"
                                           / "self_calibrated_cameras.camset"),
            "phase3_camset_used": str(old_workspace / "phase3_runs" / "deleted"
                                      / "optimised_cameras.camset"),
        },
    })
    state = context.phase3_state_source()
    assert state.path is None
    assert state.origin_run_id == "deleted"
    assert state.reason == AncestryReason.CANNOT_TRACE.value
    assert state.reason != AncestryReason.ANOTHER_IMAGE_FOLDER.value


@pytest.mark.parametrize("caching", [False, True])
def test_phase4_state_source_reads_saved_pose_ordinals(caching, tmp_path: Path,
                                                     monkeypatch):
    from pyCamSet.utils.saving import load_CameraSet, save_camset

    legacy_dir = Path(__file__).parent / "test_data" / "education" / "legacy"
    phase1_pickle = legacy_dir / "phase1_325fae4.pickle"
    phase4_fixture = legacy_dir / "phase4_325fae4.camset.json"
    camera_set = load_CameraSet(phase4_fixture)
    detection = camera_set.calibration_handler.detection
    image_root = tmp_path / "images"
    for camera_name in detection.cam_names:
        camera = image_root / camera_name
        camera.mkdir(parents=True)
        for index in range(detection.max_ims):
            (camera / f"frame{index}.png").write_bytes(b"")

    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    p1_dir = workspace / "phase1_runs" / "p1"
    p1_dir.mkdir(parents=True)
    p1_pickle_copy = p1_dir / "detected_datapoints.pickle"
    shutil.copyfile(phase1_pickle, p1_pickle_copy)
    manager.save_run("phase1", "p1", {
        "run_id": "p1",
        "params": {"f_loc": str(image_root), "caching": caching},
        "artifacts": {"detected_datapoints_pickle": str(p1_pickle_copy)},
    })

    p2_dir = workspace / "phase2_runs" / "p2"
    p2_dir.mkdir(parents=True)
    p2_camset = p2_dir / "initial_cameras.camset"
    shutil.copyfile(phase4_fixture, p2_camset)
    manager.save_run("phase2", "p2", {
        "run_id": "p2", "created_at": "2999-01-01T00:00:00",
        "inputs": {"phase1_run_id": "p1"},
        "artifacts": {
            "initial_camset": str(p2_camset),
            "education_used_detections": {
                "schema_version": 1, "source": "phase1_pickle",
                "path": str(p1_pickle_copy), "fresh_pass_reasons": [],
                "cache_hit": False,
            },
        },
    })

    p3_dir = workspace / "phase3_runs" / "p3"
    p3_dir.mkdir(parents=True)
    p3_camset = p3_dir / "optimised_cameras.camset"
    shutil.copyfile(phase4_fixture, p3_camset)
    manager.save_run("phase3", "p3", {
        "run_id": "p3",
        "inputs": {"phase1_run_id": "p1", "phase2_run_id": "p2"},
        "artifacts": {
            "optimised_camset": str(p3_camset),
            "phase1_detection_pickle_used": str(p1_pickle_copy),
            "phase2_initial_camset_used": str(p2_camset),
        },
    })

    p4_dir = workspace / "phase4_runs" / "p4"
    p4_dir.mkdir(parents=True)
    p4_camset = p4_dir / "self_calibrated_cameras.camset"
    shutil.copyfile(phase4_fixture, p4_camset)
    row_count = len(camera_set.calibration_handler._flat_detections())
    start_mean = reprojection_segment_mean(camera_set.calibration_result, row_count)
    manager.save_run("phase4", "p4", {
        "run_id": "p4", "inputs": {"phase3_run_id": "p3"},
        "params": {"f_loc": str(image_root)},
        "diagnostics": {"D4.3_initial_euclid_px": start_mean},
        "artifacts": {
            "self_calibrated_camset": str(p4_camset),
            "phase3_camset_used": str(p3_camset),
        },
    })

    phase4_run = manager.find_run("phase4", "p4")
    context = RunContext.from_run(manager, phase4_run)
    state = context.phase3_state_source()
    assert state.origin_run_id == "p3"
    assert state.verification is Verification.VERIFIED
    assert state.ancestry_verified, [
        (edge["child_phase"], edge["parent_phase"],
         edge["verification"], edge["reason"])
        for edge in state.ancestry
    ]
    assert state.confirmed
    if caching:
        assert state.photo_names == ()
        assert "reused saved detections" in state.source_check
    else:
        assert state.photo_names == ("1/frame0.png", "2/frame0.png", "3/frame0.png")
    assert state.phase3_fixed_pose_index0 == 0
    assert state.phase3_fixed_pose_ordinal1 == 1
    assert state.phase4_fixed_pose_index0 == 0
    assert state.phase4_fixed_pose_ordinal1 == 1
    expected_loss = loss_settings_from_options(
        camera_set.calibration_handler.problem_opts)
    assert (state.loss, state.f_scale) == (expected_loss.loss, expected_loss.f_scale)
    assert context.loss_settings() == expected_loss

    # A reloaded camset with different embedded camera order cannot prove that
    # its photos still correspond to the Phase 4 starting state.
    changed_phase4 = load_CameraSet(p4_camset)
    changed_detection = changed_phase4.calibration_handler.detection
    changed_detection.cam_names = list(reversed(changed_detection.cam_names))
    save_camset(changed_phase4, p4_camset)
    state_after_detection_change = RunContext.from_run(
        manager, manager.find_run("phase4", "p4")).phase3_state_source()
    assert not state_after_detection_change.confirmed
    assert state_after_detection_change.photo_names == ()
    assert "cannot be traced" in state_after_detection_change.source_check

    # Removing the option from both saved inputs leaves no ordinal to report;
    # the implicit comparison default is not evidence of a saved pose.
    original_load_camset = load_CameraSet

    def load_without_saved_pose(path):
        saved = original_load_camset(path)
        if Path(path) in (p3_camset, p4_camset):
            saved.calibration_handler.problem_opts.pop("fixed_pose", None)
        return saved

    monkeypatch.setattr("pyCamSet.utils.saving.load_CameraSet", load_without_saved_pose)
    without_saved_pose = RunContext.from_run(
        manager, manager.find_run("phase4", "p4")).phase3_state_source()
    assert without_saved_pose.phase3_fixed_pose_index0 is None
    assert without_saved_pose.phase3_fixed_pose_ordinal1 is None
    assert without_saved_pose.phase4_fixed_pose_index0 is None
    assert without_saved_pose.phase4_fixed_pose_ordinal1 is None


@pytest.mark.parametrize("mismatch", ["fixed_pose", "fixed_parameters"])
def test_phase4_state_source_integrates_saved_camset_start_mismatch(mismatch, tmp_path: Path):
    from pyCamSet.utils.saving import load_CameraSet, save_camset

    legacy_dir = Path(__file__).parent / "test_data" / "education" / "legacy"
    phase1_pickle = legacy_dir / "phase1_325fae4.pickle"
    phase4_fixture = legacy_dir / "phase4_325fae4.camset.json"
    source_camset = load_CameraSet(phase4_fixture)
    image_root = tmp_path / "images"
    for camera_name in source_camset.calibration_handler.detection.cam_names:
        camera = image_root / camera_name
        camera.mkdir(parents=True)
        for index in range(source_camset.calibration_handler.detection.max_ims):
            (camera / f"frame{index}.png").write_bytes(b"")

    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    p1_dir = workspace / "phase1_runs" / "p1"
    p1_dir.mkdir(parents=True)
    p1_pickle = p1_dir / "detected_datapoints.pickle"
    shutil.copyfile(phase1_pickle, p1_pickle)
    manager.save_run("phase1", "p1", {
        "run_id": "p1", "params": {"f_loc": str(image_root), "caching": False},
        "artifacts": {"detected_datapoints_pickle": str(p1_pickle)},
    })

    p2_dir = workspace / "phase2_runs" / "p2"
    p2_dir.mkdir(parents=True)
    p2_camset = p2_dir / "initial_cameras.camset"
    shutil.copyfile(phase4_fixture, p2_camset)
    manager.save_run("phase2", "p2", {
        "run_id": "p2", "inputs": {"phase1_run_id": "p1"},
        "artifacts": {
            "initial_camset": str(p2_camset),
            "education_used_detections": {
                "schema_version": 1, "source": "phase1_pickle",
                "path": str(p1_pickle), "fresh_pass_reasons": [],
                "cache_hit": False,
            },
        },
    })

    p3_dir = workspace / "phase3_runs" / "p3"
    p3_dir.mkdir(parents=True)
    p3_camset = p3_dir / "optimised_cameras.camset"
    shutil.copyfile(phase4_fixture, p3_camset)
    manager.save_run("phase3", "p3", {
        "run_id": "p3", "inputs": {"phase1_run_id": "p1", "phase2_run_id": "p2"},
        "diagnostics": {"D3.6_final_euclid_px": 12.345},
        "artifacts": {
            "optimised_camset": str(p3_camset),
            "phase1_detection_pickle_used": str(p1_pickle),
            "phase2_initial_camset_used": str(p2_camset),
        },
    })

    p4_dir = workspace / "phase4_runs" / "p4"
    p4_dir.mkdir(parents=True)
    p4_camset = p4_dir / "self_calibrated_cameras.camset"
    shutil.copyfile(phase4_fixture, p4_camset)
    edited_phase4 = load_CameraSet(p4_camset)
    handler = edited_phase4.calibration_handler
    source_mean = reprojection_segment_mean(
        source_camset.calibration_result,
        len(source_camset.calibration_handler._flat_detections()),
    )
    start_metric = source_mean
    if mismatch == "fixed_pose":
        # N2: a different fixed reference pose invalidates rig comparison.
        handler.problem_opts["fixed_pose"] = 5
        start_metric = source_mean + 1.0
    else:
        # A changed fixed intrinsic is effective only when D4.3 disagrees with the source mean.
        intrinsic = source_camset.get_cam_dict()["1"].intrinsic.reshape(-1).copy()
        intrinsic[0] += 1.0
        handler.fixed_params = {"1": {"int": intrinsic}}
        start_metric = source_mean + 1.0
    save_camset(edited_phase4, p4_camset)
    manager.save_run("phase4", "p4", {
        "run_id": "p4", "inputs": {"phase3_run_id": "p3"},
        "params": {"f_loc": str(image_root)},
        "diagnostics": {"D4.3_initial_euclid_px": start_metric},
        "artifacts": {
            "self_calibrated_camset": str(p4_camset),
            "phase3_camset_used": str(p3_camset),
        },
    })

    state = RunContext.from_run(
        manager, manager.find_run("phase4", "p4")).phase3_state_source()
    assert state.confirmed
    assert state.reason is None
    assert state.start_mismatch == mismatch
    assert state.phase3_before_px == pytest.approx(12.345)
    assert state.source_check == "not checkable"
    if mismatch == "fixed_pose":
        assert state.rig_comparison_withheld
        assert state.fixed_pose_matches is False
        assert state.phase3_fixed_pose_index0 == 0
        assert state.phase3_fixed_pose_ordinal1 == 1
        assert state.phase4_fixed_pose_index0 == 5
        assert state.phase4_fixed_pose_ordinal1 == 6
    else:
        assert not state.rig_comparison_withheld
        assert state.effective_fixed_parameter_differences == ("1.int",)


def test_phase4_state_source_labels_only_a_distinct_workspace_as_another_folder(tmp_path: Path):
    workspace = tmp_path / "current" / ".pycamset_workspace"
    other_workspace = tmp_path / "other" / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    p4_dir = workspace / "phase4_runs" / "p4"
    p4_dir.mkdir(parents=True)
    used = other_workspace / "phase3_runs" / "p3" / "optimised_cameras.camset"
    context = RunContext.from_run(manager, {
        "run_id": "p4", "_run_dir": str(p4_dir),
        "params": {"f_loc": str(tmp_path / "current")},
        "artifacts": {"phase3_camset_used": str(used)},
    })
    state = context.phase3_state_source()
    assert state.reason == AncestryReason.ANOTHER_IMAGE_FOLDER.value
    context.metadata["params"]["f_loc"] = str(tmp_path / "other")
    moved = context.phase3_state_source()
    assert moved.reason == AncestryReason.CANNOT_TRACE.value


def test_phase4_quality_disposition_is_not_attributable_when_used_camset_differs(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    p3_dir = workspace / "phase3_runs" / "p3"
    p3_dir.mkdir(parents=True)
    expected = p3_dir / "optimised_cameras.camset"
    expected.write_bytes(b"phase3")
    manager.save_run("phase3", "p3", {
        "run_id": "p3", "artifacts": {"optimised_camset": str(expected)},
    })
    other_dir = workspace / "phase3_runs" / "other"
    other_dir.mkdir()
    used = other_dir / "optimised_cameras.camset"
    used.write_bytes(b"different phase3")
    p4_dir = workspace / "phase4_runs" / "p4"
    p4_dir.mkdir(parents=True)
    context = RunContext.from_run(manager, {
        "run_id": "p4", "_run_dir": str(p4_dir), "status": "complete",
        "inputs": {"phase3_run_id": "p3"},
        "artifacts": {"phase3_camset_used": str(used)},
        "report": {"blocking_flags": []},
    })
    assert context.status == "complete"
    assert context.quality_disposition == "quality check not attributable to the camset used"


def test_phase4_start_settings_distinguish_n2_and_effective_fixed_parameters():
    camera = SimpleNamespace(
        intrinsic=[[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
        distortion_coefs=[0.0, 0.0, 0.0, 0.0, 0.0],
        extrinsic=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    )
    cameras = {"cam1": camera}
    phase3_options = {"fixed_pose": 0}

    # Empty per-camera dictionaries are equivalent to the empty Phase 4 form.
    result = compare_start_settings(
        {"fixed_pose": 0}, {}, phase3_options, {"cam1": {}}, cameras,
        start_error_px=1.3053232568515647,
        stored_mean_px=1.3053232568515647,
    )
    assert result == (None, (), True)

    # A different fixed reference pose is the N2 branch, not a generic failure.
    result = compare_start_settings(
        {"fixed_pose": 5}, {}, phase3_options, {}, cameras,
        start_error_px=23.166, stored_mean_px=1.305,
    )
    assert result == ("fixed_pose", (), False)

    # A fixed parameter at the solved value has no effective start change.
    result = compare_start_settings(
        {"fixed_pose": 0}, {"cam1": {"int": camera.intrinsic}},
        phase3_options, {"cam1": {}}, cameras,
        start_error_px=1.305, stored_mean_px=1.305,
    )
    assert result == (None, (), True)

    # A value difference matters only when the measured start also differs.
    result = compare_start_settings(
        {"fixed_pose": 0}, {"cam1": {"int": [[101.0, 0, 50], [0, 100, 40], [0, 0, 1]]}},
        phase3_options, {}, cameras, start_error_px=2.7, stored_mean_px=1.3,
    )
    assert result == ("fixed_parameters", ("cam1.int",), True)
    result = compare_start_settings(
        {"fixed_pose": 0}, {"cam1": {"int": [[101.0, 0, 50], [0, 100, 40], [0, 0, 1]]}},
        phase3_options, {}, cameras, start_error_px=1.3, stored_mean_px=1.3,
    )
    assert result == (None, ("cam1.int",), True)

    # Every fixed block is compared against the expanded solved camera from
    # the chosen used camset, not a default or absent block in its metadata.
    for block, solved_attribute in (
            ("dst", camera.distortion_coefs), ("ext", camera.extrinsic)):
        changed = [list(row) if isinstance(row, list) else row
                   for row in solved_attribute]
        if block == "dst":
            changed[0] += 0.1
        else:
            changed[0][3] = 0.1
        result = compare_start_settings(
            {"fixed_pose": 0}, {"cam1": {block: changed}},
            phase3_options, {}, cameras,
            start_error_px=2.0, stored_mean_px=1.0,
        )
        assert result == ("fixed_parameters", (f"cam1.{block}",), True)


def test_detections_identity_includes_array_order_camera_names_and_image_count():
    left = SimpleNamespace(cam_names=["cam1", "cam2"], max_ims=2,
                           _data=[[0, 0, 1, 10, 20], [1, 1, 2, 30, 40]])
    same = SimpleNamespace(cam_names=["cam1", "cam2"], max_ims=2,
                           _data=[[0, 0, 1, 10, 20], [1, 1, 2, 30, 40]])
    reversed_rows = SimpleNamespace(cam_names=["cam1", "cam2"], max_ims=2,
                                    _data=[[1, 1, 2, 30, 40], [0, 0, 1, 10, 20]])
    wrong_camera_order = SimpleNamespace(cam_names=["cam2", "cam1"], max_ims=2,
                                         _data=[[0, 0, 1, 10, 20], [1, 1, 2, 30, 40]])
    wrong_image_count = SimpleNamespace(cam_names=["cam1", "cam2"], max_ims=3,
                                        _data=[[0, 0, 1, 10, 20], [1, 1, 2, 30, 40]])
    assert detections_equal(left, same)
    assert not detections_equal(left, reversed_rows)
    assert not detections_equal(left, wrong_camera_order)
    assert not detections_equal(left, wrong_image_count)


def test_phase3_fallback_baseline_is_only_supplied_for_start_mismatches(tmp_path: Path):
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    manager.save_run("phase3", "p3", {
        "run_id": "p3", "diagnostics": {"D3.6_final_euclid_px": 1.3053},
    })
    source = SimpleNamespace(origin_run_id="p3")
    assert phase3_before_metric(source, manager, start_mismatch="fixed_pose",
                                used_stored_mean_px=1.3) == pytest.approx(1.3053)
    assert phase3_before_metric(source, manager, start_mismatch=None,
                                used_stored_mean_px=1.3) is None
    assert phase3_before_metric(source, manager, start_mismatch="fixed_pose",
                                used_stored_mean_px=float("nan")) is None
