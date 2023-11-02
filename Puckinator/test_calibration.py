import cv2 as cv
import numpy as np
import json

# Load previously saved calibration data
with open("camera_calibration.json", "r") as f:
    loaded_data = json.load(f)

camera_matrix = np.array(loaded_data["camera_matrix"])
distortion_coefficients = np.array(loaded_data["distortion_coefficients"])

# Initialize the webcam
cap = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
        camera_matrix, distortion_coefficients, (w, h), 1, (w, h)
    )

    # Method 1: Using cv.undistort()
    undistorted_frame = cv.undistort(
        frame, camera_matrix, distortion_coefficients, None, new_camera_matrix
    )
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y : y + h, x : x + w]

    # Resize undistorted frame to match original frame size
    undistorted_frame = cv.resize(undistorted_frame, (frame.shape[1], frame.shape[0]))

    # Display the resulting frame
    combined_frame = np.hstack((frame, undistorted_frame))
    cv.imshow("Original (left) vs Undistorted (right)", combined_frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
