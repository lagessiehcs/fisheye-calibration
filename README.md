# fisheye-calibration

## calibrate.py
### Setting the termination criteria and calibration flags for the fisheye model
````
# Termination criteria for corner sub-pixel refinement
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# Calibration flags for fisheye model
calibration_flags = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
    cv2.fisheye.CALIB_CHECK_COND +
    cv2.fisheye.CALIB_FIX_SKEW
)
````

### Creating the 3D points of the real chessboard
````
# Prepare object points (0,0,0), (1,0,0), ..., (6,9,0)
objp = np.zeros((1, chess board[0] * chess board[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:chess board[0], 0:chess board[1]].T.reshape(-1, 2)

_img_shape = None
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane
````

### Reading the chess board images
````
# Get all .jpg images in the current directory
images = glob.glob('*.jpg')
````

For each image, detect chess board corners (square corners in the chess board) and display them
````
for fname in images:
    img = cv2.imread(fname)
    
    # Ensure all images are the same size
    if _img_shape is None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + 
        cv2.CALIB_CB_FAST_CHECK + 
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # If corners found, refine and save them
    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        if DISPLAY_CORNER:
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(1000)
````

### Performing camera calibration
````
# Initialize calibration parameters
N_OK = len(objpoints)
K = np.zeros((3, 3))       # Intrinsic camera matrix
D = np.zeros((4, 1))       # Distortion coefficients
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

# Perform fisheye camera calibration
cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)
````

### Output of calibration results (matrices K and D)
````
# Output calibration results
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")
````
## undistort.py
To test the calication matrices:
````
python undistort.py file_to_undistort.jpg
````

## Source

Original code: [Calibrate fisheye lens using OpenCV — part 1](https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0)

More information on camera calibration with OpenCV: [Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

Images: [opencv/samples/data](https://github.com/opencv/opencv/tree/master/samples/data)
