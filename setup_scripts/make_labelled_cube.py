import numpy as np
import pyvista as pv
from pyCamSet.calibration_targets.shape_by_faces import make_tforms
from pyCamSet.utils.general_utils import h_tform, make_4x4h_tform

faces = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
]).astype("double")




mp  = np.mgrid[2:9,2:9]
points = np.stack([mp[0].flatten(), mp[1].flatten(), np.zeros_like(mp[0].flatten())]).T/10

tforms = make_tforms(faces, "cube")
htforms = [make_4x4h_tform(*t) for t in tforms]
extra_points = np.concatenate([h_tform(points, h) for h in htforms], axis=0)
centr = np.array([0.5, 0.5, 0])
centres = np.array([h_tform(centr, h) for h in htforms])
scene: pv.Plotter = pv.Plotter()

cube =pv.Cube()
corners = cube.points
labels = [f"{i}" for i in range(len(corners))]

points = pv.PolyData(extra_points)
scene.add_mesh(cube, style='wireframe')
scene.add_mesh(points, color='r')
scene.add_point_labels(corners, labels)
edges = [str(i) for i in range(6)]
scene.add_point_labels(pv.PolyData(centres), edges)
scene.show()

