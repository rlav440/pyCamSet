from pathlib import Path

from pyCamSet import load_CameraSet, CameraCalibrator, ChArUco

# we want to load some n camera calibration model

def test_calibration():
    data_loc = Path("test_data/calibration")
    ref_loc = Path("")
    calibrator = CameraCalibrator()
    target = ChArUco(20, 20, 4)

    camera_model = calibrator(f_loc=data_loc, calibration_target=target, draw=True)
    reference_cams = None #load_CameraSet(ref_loc)
    assert camera_model == reference_cams


if __name__ == '__main__':
    test_calibration()
    