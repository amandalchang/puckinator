import numpy as np
import cv2 as cv
import json

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

grid_height = 9
grid_width = 6
objp = np.zeros((grid_height * grid_width, 3), np.float32)
objp[:, :2] = np.mgrid[0:grid_height, 0:grid_width].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Start video capture
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (grid_height, grid_width), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        objpoints.append(objp)

        # Draw and display the corners
        cv.drawChessboardCorners(frame, (grid_height, grid_width), corners2, ret)

    # Display the resulting frame
    cv.imshow("Frame", frame)

    # Press 'q' to exit the loop
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

print("running calibration... this may take a while")

# Camera calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv.CALIB_USE_LU
)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)


calibration_data = {
    "camera_matrix": mtx.tolist(),
    "distortion_coefficients": dist.tolist(),
    "rotation_vectors": [rvec.tolist() for rvec in rvecs],
    "translation_vectors": [tvec.tolist() for tvec in tvecs],
}

with open("camera_calibration.json", "w") as f:
    json.dump(calibration_data, f)
