from pyCamSet import calibrate_cameras, Ccube, load_CameraSet
from cv2 import aruco

target = Ccube(n_points=8, length=400, aruco_dict=aruco.DICT_4X4_250, border_fraction=0.1)
target.save_to_pdf()
