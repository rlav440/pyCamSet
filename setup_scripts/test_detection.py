from pyCamSet import ChArUco
import cv2

target = ChArUco(10, 10, 4, a_dict=cv2.aruco.DICT_6X6_1000)
im=cv2.imread("setup_scripts/test.png") [:,:,0]
im = cv2.GaussianBlur(im, (0,0), 2,2)

d = target.find_in_image(im, draw=True, wait_len=-1)
print(d.has_data)
