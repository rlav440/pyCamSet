import numpy as np

from pyCamSet import Camera, CameraSet
from pyCamSet.utils.general_utils import make_4x4h_tform


def make_cams(nc, plot=False):
# the make_4x4h_tform uses the opencv definition of the rotation vector
    tforms = [
       make_4x4h_tform((0, b/nc*2*np.pi, 0), (0, 0, 0.2)) for b in range(nc)
    ]
    cams = {f"cam_{i}":Camera(extrinsic=t) for i, t in enumerate(tforms)}
    ring_cameras = CameraSet(camera_dict=cams)
    if plot:
        ring_cameras.plot()
    return ring_cameras

def project_point(point, cameras: CameraSet):
    
    # the point should be some n x 3 array of information
    
    projection = cameras.project_points_to_all_cams(point)
    print(projection)
    return projection
    
def triangulate_points(proj_data, cameras: CameraSet):

    points_3d = cameras.multi_cam_triangulate(proj_data)

    return points_3d


if __name__ == "__main__":
    cams = make_cams(5)
    project_point(np.array([0.01,0.03,-0.05]), cams)
