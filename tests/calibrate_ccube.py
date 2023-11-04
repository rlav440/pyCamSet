from pyCamSet import calibrate_cameras, Ccube, load_CameraSet
from cv2 import aruco

target = Ccube(n_points=10, length=40,  aruco_dict=aruco.DICT_6X6_1000, border_fraction=0.2)
loc="tests/test_data/calibration_ccube"
calibrate_cameras(loc, target, 
                  # draw=True,
                  # save=False
                  )


