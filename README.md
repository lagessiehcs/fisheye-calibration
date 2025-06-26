Overview
Setup:

Defines the size of a checkerboard pattern used for calibration.

Sets criteria for refining corner accuracy.

Prepares object points (3D points in real-world space) corresponding to the checkerboard corners.

Image Loop:

Reads .jpg images from the current directory.

Finds and refines the checkerboard corners in each image.

Accumulates object and image points.

Calibration:

Uses OpenCV's cv2.fisheye.calibrate() to estimate:

Intrinsic matrix K

Distortion coefficients D

Rotation and translation vectors for each image

Output:

Prints how many valid calibration images were used.

Prints the intrinsic parameters in a copy-pasteable format.