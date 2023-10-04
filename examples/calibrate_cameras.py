from pathlib import Path
from pyCamSet import CameraCalibrator, ChArUco

calibration_data = Path('my/calibration/path')
calibration_target = ChArUco(num_squares_x=10, num_squares_y=10, square_size=4):

calibrator = CameraCalibrator()
cams = calibrator(f_loc=calibration_data, calibration_target=calibration_target, draw=True)

