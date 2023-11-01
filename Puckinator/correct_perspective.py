import cv2 as cv
import numpy as np


def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect


def main():
    # Get the predefined dictionary
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    # Create detector parameters
    parameters = cv.aruco.DetectorParameters()
    # Instantiate the ArucoDetector object
    detector = cv.aruco.ArucoDetector(dictionary, parameters)

    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect markers
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

        if markerIds is not None:
            # Draw marker boundaries
            cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

            if len(markerCorners) == 4:  # Ensure there are exactly 4 markers detected
                # Calculate the center of each ArUco marker
                centers = np.array([np.mean(crn[0], axis=0) for crn in markerCorners])

                # Sort and order the center points
                sorted_corners = order_points(centers)

                # Define the dimensions of your paper in pixels
                # You might want to adjust the width and height values according to your needs
                paper_width = 1500
                paper_height = 800
                output_pts = np.array(
                    [
                        [0, 0],
                        [paper_width - 1, 0],
                        [paper_width - 1, paper_height - 1],
                        [0, paper_height - 1],
                    ],
                    dtype="float32",
                )

                # Compute the perspective transform matrix
                M = cv.getPerspectiveTransform(sorted_corners, output_pts)

                # Apply the perspective transformation
                warped = cv.warpPerspective(frame, M, (paper_width, paper_height))

                # Show the result
                cv.imshow("Perspective Transform", warped)

        # Show the frame
        cv.imshow("Frame", frame)

        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
