import itertools
import cv2 as cv
import numpy as np

# Constants
CALIB_FRAME = 0 # Number of frames grabbed
table_width = 2200
table_height = 800


def order_points(pts):
    # Initialize an array to hold the ordered coordinates of the points
    # The order is determined as follows: [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")

    # Sum the coordinates of each point; the smallest sum will be the top-left point,
    # and the largest sum will be the bottom-right point.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Compute the difference between the coordinates of each point;
    # the smallest difference will be the top-right point,
    # and the largest difference will be the bottom-left point.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect


def main():
    # Load a predefined dictionary for ArUco marker detection
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    # Create an instance of DetectorParameters for configuring ArUco detection
    parameters = cv.aruco.DetectorParameters()
    # Create an ArucoDetector object with the predefined dictionary and custom parameters
    detector = cv.aruco.ArucoDetector(dictionary, parameters)

    # Initialize the video capture object to capture video from the default camera (camera 0)
    cap = cv.VideoCapture(0)

    # Initialize the number of frames
    num_frames = 0

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        # Check if the frame was successfully captured
        if not ret:
            print("Failed to grab frame")
            break  # Exit the loop if frame capture failed

        # Detect ArUco markers in the frame
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

        # Check if any ArUco markers were detected
        if markerIds is not None:
            detectedMarkers = list(zip(markerCorners, markerIds))
            # Draw the boundaries of the detected ArUco markers on the frame
            cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

            # Proceed if exactly four ArUco markers are detected
            if num_frames <= CALIB_FRAME:
                if len(markerCorners) == 4:
                
                    sorted_markers = list(
                    zip(*sorted(detectedMarkers, key=lambda marker: marker[1])))[0]

                    print(f"Sorted markers:\n{sorted_markers}")

                    desired_corners = np.array(
                    [marker[0][0] for marker in sorted_markers])  # Extracting the first corner of each marker

                    print(
                    f"Desired corners (has shape {desired_corners.shape}):\n{desired_corners}")

                    # Define the coordinates of the corners of the paper in the output image
                    output_pts = np.array(
                        [
                            [0, 0],
                            [table_width - 1, 0],
                            [table_width - 1, table_height - 1],
                            [0, table_height - 1],
                        ],
                        dtype="float32",
                    )

                    # Compute the perspective transform matrix to transform the perspective
                    # of the captured frame to match the dimensions of the paper
                    M = cv.getPerspectiveTransform(desired_corners, output_pts)
                    if num_frames == CALIB_FRAME:
                        calibrated_m = M
                        print("frame 10")
                    # Apply the perspective transformation to the captured frame
                    warped = cv.warpPerspective(frame, M, (table_width, table_height))

                    # Display the result of the perspective transformation
                    cv.imshow("Perspective Transform", warped)

                    num_frames = num_frames + 1
        if num_frames > CALIB_FRAME:
            print("I found my calibration")
            # Apply the perspective transformation to the captured frame
            warped = cv.warpPerspective(frame, calibrated_m, (table_width, table_height))

            # Display the result of the perspective transformation
            cv.imshow("Perspective Transform", warped)

        # Display the original frame with the detected ArUco markers
        cv.imshow("Frame", frame)

        # Wait for a key press for 1 millisecond; exit if 'Esc' is pressed
        key = cv.waitKey(1)
        if key == 27:
            break

    # Release the video capture object to free resources
    cap.release()
    # Destroy all OpenCV-created windows to free resources
    cv.destroyAllWindows()


# Execute the main function when this script is run
if __name__ == "__main__":
    main()
