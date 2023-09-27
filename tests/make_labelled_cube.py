import numpy as np
import pyvista as pv
from pyCamSet.calibration_targets.shape_by_faces import make_tforms
from pyCamSet.utils.general_utils import homogenous_transform, e_4x4

faces = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
]).astype("double")

centre = np.array([0.5, 0.5, 0])

tforms = make_tforms(faces, "cube")
htforms = [e_4x4(*t) for t in tforms]
extra_points = np.array([homogenous_transform(centre, h) for h in htforms])

scene: pv.Plotter = pv.Plotter()

cube =pv.Cube()

corners = cube.points

labels = [f"{i}" for i in range(len(corners))]

print(np.array2string(extra_points, precision=2, suppress_small=True))


points = pv.PolyData(extra_points)
scene.add_mesh(cube, style='wireframe')
scene.add_mesh(points, color='r')
scene.add_point_labels(corners, labels)
scene.show()

