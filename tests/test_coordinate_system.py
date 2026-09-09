import numpy as np
from pyCamSet import Camera, CameraSet
from pyCamSet.utils import make_4x4h_tform

def test_sensor_reprojection_0_offset():
    """
    Notes: This tests for the sensor location reprojecting accurately
    """
    int_m = np.array([[400, 0, 0], [0, 200, 0], [0, 0, 1]])
    test_cam = Camera(intrinsic=int_m)
    test_cam._make_sensormap()
    true_points = np.stack(([0., 0.], np.array(test_cam.res) - 1))

    cam_corners = test_cam.position + \
                  test_cam.world_sensor_map[[0, -1], [0, -1]]

    reprojected = test_cam.project_points(cam_corners)

    assert np.all(np.isclose(true_points, reprojected))


def test_sensor_reprojection_some_offset():
    """
    Notes: This tests for the sensor location reprojecting accurately
    """
    int_m = np.array([[200, 0, 66], [0, 200, 136], [0, 0, 1]])
    test_cam = Camera(intrinsic=int_m)
    test_cam._make_sensormap()
    true_points = np.stack(([0., 0.], np.array(test_cam.res) - 1))

    cam_corners = test_cam.position + \
                  test_cam.world_sensor_map[[0, -1], [0, -1]]

    reprojected = test_cam.project_points(cam_corners)

    assert np.all(np.isclose(true_points, reprojected))


def test_im_to_world_ray():
    """
    Tests that a point taken from the camera sensor to W coords is the same point when back projected.
    """
    # load some cams, with extrinsics

    n_cam = 1

    names = [str(n) for n in range(n_cam)]
    # generate random extrinsic positions, and use the default camera arrangement
    extrinsic = [make_4x4h_tform(np.random.rand(3) * np.pi,
                       np.random.rand(3) * 0.2)
                 for _ in range(n_cam)]
    # A list of a standard calibration
    intrinsic = [np.array([[950, 0, 480],
                           [0, 1700, 320],
                           [0, 0, 1]]
                          )
                 ] * n_cam
    test_cameras = CameraSet(
        camera_names=names,
        extrinsic_matrices=extrinsic,
        intrinsic_matrices=intrinsic,
        distortion_coefs=[np.zeros(5)]*n_cam,
        res=[[640, 980]],
    )

    point = [320, 475]
    test_cam = test_cameras[0]
    world_point = test_cam.im_to_world_ray(point)
    proj_point = test_cam.project_points(world_point)
    assert (np.all(np.isclose(point, proj_point)))


def test_set_reprojection():
    """
    This function tests if points can be projected to the camera space, then reconstructed accurately.
    """
    ext_0 = np.array([[1, 0, 0, -0.05],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0.4],
                      [0, 0, 0, 1]]
                     )
    ext_1 = np.array([[1, 0, 0, 0.05],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0.4],
                      [0, 0, 0, 1]]
                     )
    # define two cameras


    cam_dict = {
        "0":Camera(extrinsic=ext_0, res=[1024,768]),
        "1":Camera(extrinsic=ext_1, res=[1024,768])
    }
    cam_set = CameraSet(camera_dict=cam_dict)

    # define a few points in worldspace
    pt = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, .01],
                   [.01, 0.0, 0.0],
                   [0.0, .01, 0.0],
                   [0.1, 0.0, 0.0],
                   [0.0, 0.1, 0.0],
                   [0.0, 0.0, 0.1]])

    # project that point to the cameras
    cam_cords = cam_set.project_points_to_all_cams(pt)

    for c in cam_cords:
        print(c)
    # reproject that point from cameras to worldspace.
    reconstructed_point = cam_set.multi_cam_triangulate(cam_cords)

    # cam_set.draw_point_cloud_in_cameraset(pt)
    # cam_set.draw_point_cloud_in_cameraset(np.array(reconstructed_point))
    assert np.all(np.isclose(pt, reconstructed_point))

if __name__ == "__main__":
    test_im_to_world_ray()
    test_set_reprojection()
    test_sensor_reprojection_0_offset()
    test_sensor_reprojection_some_offset()
