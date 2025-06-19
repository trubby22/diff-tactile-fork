import cv2
import numpy as np
import glob

# Checkerboard dimensions (inner corners)
CHECKERBOARD = (4-1, 5-1)  # Adjust to your board

# Prepare 3D object points (0,0,0), (1,0,0), ... (7,5,0)
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Store object/image points
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Load calibration images
images = glob.glob('/home/psb120/camera-calibration-2/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, 
        CHECKERBOARD, 
        None
    )
    
    if ret:
        objpoints.append(objp.copy())
        # Refine corner locations (pixel accuracy)
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        )
        imgpoints.append(corners_refined)
        print(f"Found {len(corners_refined)} corners in {fname}")
        assert len(corners_refined) == CHECKERBOARD[0] * CHECKERBOARD[1]

        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)
    else:
       print(f"Checkerboard not found in {fname}")

cv2.destroyAllWindows()

# Fisheye calibration (requires ≥10 images)
assert len(objpoints) >= 10, "Insufficient images for calibration"

K = np.zeros((3, 3))  # Intrinsic matrix
D = np.zeros((4, 1))  # Distortion coefficients (k1, k2, k3, k4)
rvecs, tvecs = [], []  # Rotation/translation vectors

# Calibrate
ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],  # Image resolution (width, height)
    K,
    D,
    rvecs,
    tvecs,
    None,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

# Results
print(f"Focal Length (fx, fy): {K[0,0]:.2f}, {K[1,1]:.2f}")
print(f"Principal Point (cx, cy): {K[0,2]:.2f}, {K[1,2]:.2f}")
print(f"Distortion Coefficients (k1, k2, k3, k4): {D.flatten()}")

# Save parameters
np.savez("fisheye_params.npz", K=K, D=D)
