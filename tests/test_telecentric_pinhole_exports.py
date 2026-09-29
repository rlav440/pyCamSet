"""A telecentric rig exports to APDe-MVS and COLMAP through its exact pinhole equivalent."""

from __future__ import annotations

import numpy as np
import pytest

from pyCamSet.cameras import CameraSet
from test_telecentric_model import make_telecentric_camera, telecentric_problem  # noqa: F401


def _read_cam_file(path):
    lines = path.read_text(encoding="utf-8").split("\n")
    extrinsic = np.array([[float(v) for v in lines[k].split()] for k in range(1, 5)])
    intrinsic = np.array([[float(v) for v in lines[k].split()] for k in range(7, 10)])
    depth = [float(v) for v in lines[11].split()]
    return extrinsic, intrinsic, depth


@pytest.mark.parametrize("eps", [0.05, 0.3, 2.0])
def test_the_pinhole_equivalent_projects_exactly_as_the_lens(eps):
    cam = make_telecentric_camera("cam", rotation=(0.2, -0.3, 0.1), eps=eps)
    pinhole = cam.pinhole_equivalent()
    points = np.random.default_rng(3).uniform(-0.02, 0.02, (200, 3))
    np.testing.assert_allclose(pinhole.project_points(points, distort=False),
                               cam.project_points(points, distort=False), atol=1e-9)
    # its centre sits 1/eps behind the telecentric frame, along the view axis
    np.testing.assert_allclose(np.linalg.norm(pinhole.position - cam.position), 1.0 / eps)


@pytest.mark.parametrize("eps,why", [(0.0, "infinity"), (-0.2, "in front")])
def test_a_lens_without_a_finite_centre_behind_it_is_refused(eps, why):
    with pytest.raises(ValueError, match=why):
        make_telecentric_camera("cam", eps=eps).pinhole_equivalent()


def test_apde_export_needs_the_calibration_to_size_the_depth(tmp_path):
    from pyCamSet.utils.saving import camset_to_apde

    cams = CameraSet(camera_dict={"cam_0": make_telecentric_camera("cam_0")})
    with pytest.raises(ValueError, match="triangulated points"):
        camset_to_apde(cams, tmp_path)


def test_a_calibrated_telecentric_rig_exports_to_apde(telecentric_problem, tmp_path):
    """The written cams files reproduce the lens, each with a depth range of its own."""
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler
    from pyCamSet.utils.saving import camset_to_apde
    from test_telecentric_model import ground_truth_params

    cams, target, detection, poses = telecentric_problem
    # The exact calibration of this synthetic rig, attached as a solve would.
    handler = TemplateBundleHandler(camset=cams, target=target, detection=detection,
                                    options={"outliers": "n", "verbosity": 0})
    params = ground_truth_params(handler, cams, poses)
    solved = cams
    solved.calibration_handler = handler
    solved.calibration_params = params
    solved.calibration_result = np.zeros(2 * len(handler.get_detection_data()))

    ranges = camset_to_apde(solved, tmp_path, depth_min=0.1, depth_max=0.8)

    names = [line.split()[1] for line in
             (tmp_path / "cam_index_map.txt").read_text(encoding="utf-8").splitlines()]
    assert names == solved.get_names()
    points = np.asarray(target.point_data, dtype=float).reshape(-1, 3)
    for index, name in enumerate(names):
        extrinsic, intrinsic, depth = _read_cam_file(tmp_path / "cams" / f"{index:08d}_cam.txt")
        in_camera = (extrinsic @ np.c_[points, np.ones(len(points))].T)[:3]
        uv = (intrinsic @ in_camera)[:2] / (intrinsic @ in_camera)[2]
        np.testing.assert_allclose(uv.T, solved[name].project_points(points, distort=False),
                                   atol=1e-6)
        # the typed 0.1-0.8 range is replaced by one around this camera's scene
        assert depth[0] == pytest.approx(ranges[name][0]) and depth[3] == pytest.approx(ranges[name][1])
        assert depth[0] < in_camera[2].min() and in_camera[2].max() < depth[3]


def test_a_telecentric_rig_exports_to_colmap_as_pinholes(tmp_path):
    """cameras.txt and rig_config.json together reproduce each lens exactly."""
    import json

    from scipy.spatial.transform import Rotation

    from pyCamSet.utils.saving import camset_to_colmap

    cams = CameraSet(camera_dict={
        name: make_telecentric_camera(name, rotation=rot, eps=eps)
        for name, rot, eps in (("cam_0", (0.0, 0.0, 0.0), 0.3),
                               ("cam_1", (0.0, 0.5, 0.0), 0.45),
                               ("cam_2", (-0.4, 0.0, 0.2), 0.2))
    })
    assert camset_to_colmap(cams, tmp_path) == "PINHOLE"

    rows = [line.split() for line in (tmp_path / "cameras.txt").read_text(encoding="utf-8").splitlines()
            if not line.startswith("#")]
    rig = json.loads((tmp_path / "rig_config.json").read_text(encoding="utf-8"))[0]["cameras"]
    names = cams.get_names()
    reference = cams[names[0]].pinhole_equivalent().extrinsic
    points = np.random.default_rng(4).uniform(-0.02, 0.02, (100, 3))
    in_reference = reference @ np.c_[points, np.ones(len(points))].T
    for name, row, entry in zip(names, rows, rig):
        assert row[1] == "PINHOLE" and len(row) == 8
        fx, fy, cx, cy = map(float, row[4:8])
        magnification = cams[name].magnification
        assert fx == pytest.approx(magnification[0] / cams[name].telecentricity)
        cam_from_rig = np.eye(4)
        if not entry.get("ref_sensor"):
            w, x, y, z = entry["cam_from_rig_rotation"]
            cam_from_rig[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
            cam_from_rig[:3, 3] = entry["cam_from_rig_translation"]
        in_camera = (cam_from_rig @ in_reference)[:3]
        intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        uv = (intrinsic @ in_camera)[:2] / (intrinsic @ in_camera)[2]
        np.testing.assert_allclose(uv.T, cams[name].project_points(points, distort=False), atol=1e-6)


def test_colmap_refuses_a_lens_without_a_pinhole_equivalent(tmp_path):
    from pyCamSet.utils.saving import camset_to_colmap

    cams = CameraSet(camera_dict={"cam_0": make_telecentric_camera("cam_0", eps=0.0)})
    with pytest.raises(ValueError, match="infinity"):
        camset_to_colmap(cams, tmp_path)
