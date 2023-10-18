from pyCamSet import calibrate_cameras, ChArUco

calibration_data = 'my/calibration/path'
calibration_target = ChArUco(num_squares_x=10, num_squares_y=10, square_size=4)

cams = calibrate_cameras(f_loc=calibration_data, calibration_target=calibration_target, draw=True)

