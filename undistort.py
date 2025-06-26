import cv2
import numpy as np
import sys

# You should replace these 3 lines with the output in calibration step
DIM=(640, 480)
K=np.array([[531.9386133085125, 0.0, 342.64830526866007], [0.0, 532.337505864686, 233.54133225204058], [0.0, 0.0, 1.0]])
D=np.array([[0.10572614184442579], [-0.9221378032637212], [4.419649235248695], [-7.300333736608778]])
def undistort(img_path):
    img = cv2.imread(img_path)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)