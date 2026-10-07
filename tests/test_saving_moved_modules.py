"""A saved file finds its classes by name when the module it records has moved."""

from __future__ import annotations

import json

import dill
import numpy as np
import pytest

from pyCamSet.utils import saving

#: Paths that files written by earlier layouts record, and that no longer import.
OLD_PATHS = {
    "dtct_module": "pyCamSet.calibration_targets.target_detections",
    "target_module": "pyCamSet.calibration_targets.target_charuco",
    "handler_module": "pyCamSet.optimisation.base_optimiser",
}


def test_a_class_is_found_by_name_when_its_module_is_gone():
    cls = saving.resolve_class("pyCamSet.calibration_targets.target_Ccube", "Ccube")

    assert cls.__module__ == "pyCamSet.calibration_targets.ccube"


def test_a_class_no_pyCamSet_module_defines_is_refused_by_name():
    with pytest.raises(ImportError, match="NoSuchTarget"):
        saving.resolve_class("pyCamSet.calibration_targets.gone", "NoSuchTarget")


def test_every_class_a_camset_saves_has_a_unique_name():
    from pyCamSet.calibration_targets.core.abstract_target import AbstractTarget
    from pyCamSet.calibration_targets.core.target_detections import TargetDetection
    from pyCamSet.cameras.camera import Camera
    from pyCamSet.cameras.camera_set import CameraSet
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    bases = (AbstractTarget, TargetDetection, TemplateBundleHandler, Camera, CameraSet)
    shared = {
        name: [f"{c.__module__}.{c.__name__}" for c in classes]
        for name, classes in saving._classes_by_name().items()
        if len(classes) > 1 and any(issubclass(c, bases) for c in classes)
    }

    assert shared == {}


def test_a_pickle_naming_a_moved_module_loads(tmp_path):
    from pyCamSet.calibration_targets.core.target_detections import TargetDetection

    detection = TargetDetection(
        cam_names=["cam0"], data=np.array([[0, 0, 0, 1.0, 2.0]]), max_ims=1)
    # protocol 0 names classes on newline-terminated lines, so the module
    # can be rewritten without disturbing any length prefix
    raw = dill.dumps(detection, protocol=0)
    old = b"pyCamSet.calibration_targets.target_detections\n"
    path = tmp_path / "detected_datapoints.pickle"
    path.write_bytes(raw.replace(b"pyCamSet.calibration_targets.core.target_detections\n", old))
    assert old in path.read_bytes()

    loaded = saving.load_pickle(path)

    assert isinstance(loaded, TargetDetection)
    assert np.array_equal(loaded.get_data(), detection.get_data())


@pytest.mark.data
def test_a_camset_recording_old_module_paths_loads_whole(charuco_problem, tmp_path):
    from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, cams = charuco_problem
    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detections,
        options={"outliers": "n", "max_nfev": 3, "verbosity": 0})
    _, solved = run_bundle_adjustment(handler, threads=1)
    path = tmp_path / "old.camset"
    solved.save(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    optim = saved["optim"]
    optim["dtct_config"]["dtct_module"] = OLD_PATHS["dtct_module"]
    optim["target_config"]["target_module"] = OLD_PATHS["target_module"]
    optim["handler_config"]["handler_module"] = OLD_PATHS["handler_module"]
    del saved["cam_config"]["cam_module"]
    path.write_text(json.dumps(saved), encoding="utf-8")

    reloaded = saving.load_CameraSet(path)

    assert type(reloaded.calibration_handler) is TemplateBundleHandler
    assert type(reloaded.calibration_handler.target) is type(target)
    assert np.array_equal(
        reloaded.calibration_handler.detection.get_data(), detections.get_data())
    np.testing.assert_allclose(reloaded.calibration_params, solved.calibration_params)


def test_a_camset_whose_class_cannot_be_found_is_refused(tmp_path):
    from pyCamSet.cameras import Camera, CameraSet

    path = tmp_path / "cams.camset"
    saving.save_camset(CameraSet(camera_dict={"cam": Camera(name="cam")}), path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["cam_config"]["cam_name"] = "NoSuchCamera"
    saved["cam_config"].pop("cam_module", None)
    path.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(ImportError, match="NoSuchCamera"):
        saving.load_CameraSet(path)
