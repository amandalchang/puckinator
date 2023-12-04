import cv2 as cv
import numpy as np
import serial
import time
import math
from dataclasses import dataclass

# Constants
CALIB_FRAME = 10  # Number of frames grabbed
TABLE_WIDTH = 3925 + 150
TABLE_HEIGHT = 1875 + 150
ARM_LENGTH = 8  # arm legnth in inches
DISPLACEMENT = 5  # distance between motors in inches
SERIAL_DELAY = 0.01
SPEED_THRESHOLD = 10  # inches per second

WAITING_POSITION = 4.0
HITTING_POSITION = 9.0

ARDUINO_ENABLED = True  # disable arduino comms for debugging


def y_int_predict(prev_pos, latest_pos, intersect_x=HITTING_POSITION):
    """
    Args:
        prev_pos: an instance of TimedstampedPos
        latest_pos: an instance of TimedstampedPos
        intersect_x: the x position represented by a float of the predicted
            spot for the puck
    Returns:
        y_int: a float representing the y-value of the predicted position
            of the puck at a given x
        x3: a float representing the x value of the endpoint of the velocity
            vector
        y3: a float representing the y value of the endpoint of the velocity
            vector

    This function draws a vector to show the predicted direction of the puck
    and outputs the y-intersect. If the y-intersect is predicted to be beyond
    the edges of the table, the function sets it to the closest limit of the
    table. If the vector is vertical or the speed is lower than the speed
    threshold, y_int matches the y-position of the puck. It also outputs x3 and
    y3, which represent the endpoint of a predictive velocity vector drawn
    based upon the direction and speed of the puck.
    """
    time_elapsed = latest_pos.timestamp - prev_pos.timestamp
    x3 = None
    y3 = None
    # in inches? check exact conversion
    speed = (
        math.dist(
            prev_pos.pos_as_tuple(),
            latest_pos.pos_as_tuple(),
        )
        / 100
    ) / time_elapsed
    if latest_pos.x != prev_pos.x:
        m = (latest_pos.y - prev_pos.y) / (latest_pos.x - prev_pos.x)
        # Calculate the y-intercept
        b = prev_pos.y - m * prev_pos.x
        # Calculate the end point of the line
        x3 = latest_pos.x + (latest_pos.x - prev_pos.x)
        y3 = m * x3 + b
        y_int = m * (intersect_x * 100) + b
    else:
        # This is a special case where the line is vertical
        print("the line is vertical")
        y_int = latest_pos.y

    y_int = min(1750, max(y_int, 135))
    print(f"speed: {speed}")
    if speed < SPEED_THRESHOLD:
        y_int = latest_pos.y

    return y_int, x3, y3


def coordinateconverter(cY, cX, arm_length, displacement):
    """
    Note:
        The origin is defined to be at the axis of movement of the left motor.
        The x direction here is the y direction outside of this function
        This function is designed to return the desired angles of the two
        motors on a five-bar parallel robot relative to the horizontal given
        the length of the arms (assumed to be of equal length) and the distance
        between the motors. They must be in the same length units.
    Args:
        cX: The x coordinate of the center of the striker
        cY: The y coordinate of the center of the striker
        length: The length of each of the four arms (inches)
        displacement: The distance between the motors (inches)
    Returns:
        q1: the radian CCW angle of the left motor from the horizontal
        q2: the radian CCW angle of the right motor from the horizontal
    """
    # Length of the diagonal between the origin and (cX, cY)
    diag1 = np.sqrt(cX**2 + cY**2)
    # Calculating left motor angle
    if cX == 0.0:
        cX = 0.001
    theta = np.arctan2(cY, cX) + np.arccos(diag1 / (2 * arm_length))

    # Length of the diagonal between the center of the right motor and (cX, cY)
    diag2 = np.sqrt((displacement - cX) ** 2 + cY**2)
    # Calculating right motor angle
    phi = (
        np.pi - np.arctan2(cY, displacement - cX) - np.arccos(diag2 / (2 * arm_length))
    )

    return (theta, phi)


@dataclass
class TimestampedPos:
    """
    This is in terms of OpenCV coordinates in pixels and timestamp is in seconds
    """

    x: float
    y: float
    timestamp: float

    def pos_as_tuple(self):
        return (self.x, self.y)


@dataclass
class PuckVector:
    """
    This is in terms of OpenCV coordinates in pixels and speed is in inches/second
    """

    end_x: int
    end_y: int
    m: float
    b: float
    speed: float


class PerspectiveCorrector:
    def __init__(self, width, height) -> None:
        # Load a predefined dictionary for ArUco marker detection
        self.dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        # Create an instance of DetectorParameters for configuring ArUco detection
        self.parameters = cv.aruco.DetectorParameters()
        # Create an ArucoDetector object with the predefined dictionary and custom parameters
        self.detector = cv.aruco.ArucoDetector(self.dictionary, self.parameters)

        self.calibrated_transform = None
        self.width = width
        self.height = height

    def calibrate(self, frame):
        # Detect ArUco markers in the frame
        markerCorners, markerIds, _ = self.detector.detectMarkers(frame)

        # Check if any ArUco markers were detected
        if markerIds is not None:
            print(f"There are {len(markerCorners)}")
            detectedMarkers = list(zip(markerCorners, markerIds))
            # Draw the boundaries of the detected ArUco markers on the frame
            cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            # Proceed if exactly four ArUco markers are detected
            if len(markerCorners) == 4:
                sorted_markers = list(
                    zip(*sorted(detectedMarkers, key=lambda marker: marker[1]))
                )[0]

                # print(f"Sorted markers:\n{sorted_markers}")

                desired_corners = np.array(
                    [marker[0][0] for marker in sorted_markers]
                )  # Extracting the first corner of each marker

                # print(
                #     f"Desired corners (has shape {desired_corners.shape}):\n{desired_corners}"
                # )
                # Define the coordinates of the corners of the table in the output image
                output_pts = np.array(
                    [
                        [0, 0],
                        [TABLE_WIDTH - 1, 0],
                        [TABLE_WIDTH - 1, TABLE_HEIGHT - 1],
                        [0, TABLE_HEIGHT - 1],
                    ],
                    dtype="float32",
                )

                # Compute the perspective transform matrix to transform the perspective
                # of the captured frame to match the dimensions of the paper
                self.calibrated_transform = cv.getPerspectiveTransform(
                    desired_corners, output_pts
                )

    def correct_frame(self, frame):
        if self.calibrated_transform is not None:
            return cv.warpPerspective(
                frame, self.calibrated_transform, (self.width, self.height)
            )
        else:
            return None


class PuckDetector:
    def __init__(self) -> None:
        # Load a predefined dictionary for ArUco marker detection
        self.dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        # Create an instance of DetectorParameters for configuring ArUco detection
        self.parameters = cv.aruco.DetectorParameters()
        # Create an ArucoDetector object with the predefined dictionary and custom parameters
        self.detector = cv.aruco.ArucoDetector(self.dictionary, self.parameters)

    def detect_puck(self, frame):
        markerCorners, markerIds, _ = self.detector.detectMarkers(frame)
        # print("detect puck called")
        # Check if any ArUco markers were detected

        if markerIds is not None:
            # print("marker IDs present")
            detectedMarkers = list(zip(markerCorners, markerIds))
            # Draw the boundaries of the detected ArUco markers on the frame
            cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            # print(detectedMarkers)
            # Search for the target marker
            for corners, marker_id in detectedMarkers:
                if marker_id == 4:
                    # print(f"Corners list for id 4:\n{corners}")
                    x_avg = float(np.mean([corner[0] for corner in corners[0]]))
                    y_avg = float(np.mean([corner[1] for corner in corners[0]]))
                    timestamp = time.perf_counter()
                    return (frame, TimestampedPos(x_avg, y_avg, timestamp))
        return (frame, None)


def main():
    # Initialize the video capture object to capture video from the default camera (camera 0)
    cap = cv.VideoCapture(4)
    # cap.set(12, 2)
    corrector = PerspectiveCorrector(TABLE_WIDTH, TABLE_HEIGHT)
    detector = PuckDetector()
    if ARDUINO_ENABLED:
        arduino = serial.Serial(port="/dev/ttyACM0", baudrate=115200, write_timeout=0.1)
    # Initialize the number of frames
    num_frames = 0
    previous_position = None

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        # Converting the image to grayscale and then to binary
        # frame = cv.threshold(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), 127, 255, 0)

        # Check if the frame was successfully captured
        if not ret:
            print("Failed to grab frame")
            break  # Exit the loop if frame capture failed
        # Apply the perspective transformation to the captured frame
        corrected_frame = corrector.correct_frame(frame)
        if corrected_frame is not None:
            # Display the result of the perspective transformation
            detect_result = detector.detect_puck(corrected_frame)
            if detect_result is not None:
                detected_frame, latest_position = detect_result
                if detected_frame is not None:
                    resize = cv.resize(
                        detected_frame,
                        (int(TABLE_WIDTH / 4), int(TABLE_HEIGHT / 4)),
                    )

                    if latest_position is not None:
                        if previous_position is not None:
                            y_int, x3, y3 = y_int_predict(
                                previous_position,
                                latest_position,
                            )
                            if x3 is not None and y3 is not None:
                                resize = cv.arrowedLine(
                                    resize,
                                    (
                                        int(latest_position.x / 4),
                                        int(latest_position.y / 4),
                                    ),
                                    (int(x3 / 4), int(y3 / 4)),
                                    (255, 0, 0),
                                    10,
                                )
                            resize = cv.circle(
                                resize,
                                (int(HITTING_POSITION * 100 / 4), int((y_int) / 4)),
                                25,
                                (0, 0, 255),
                                3,
                            )
                            (theta, phi) = coordinateconverter(
                                # round((float(center[1]) / 100) - 6, 2),
                                HITTING_POSITION,
                                round(y_int / 100 - 6, 2),
                                ARM_LENGTH,
                                DISPLACEMENT,
                            )
                            print(
                                f"go to position {HITTING_POSITION, round(y_int / 100 - 6, 2)}"
                            )

                            if ARDUINO_ENABLED:
                                arduino.write(
                                    f"{theta - (3.14 / 2)},{phi - (3.14 / 2)}\n".encode(
                                        "utf-8"
                                    )
                                )
                            # print(
                            #     f"raw values: ({theta}, {phi}) written to serial: ({theta - (3.14 / 2)},{phi - (3.14 / 2)}) radians "
                            # )

                        previous_position = latest_position

                    cv.imshow("Perspective Transform", resize)

        num_frames += 1

        # Display the original frame with the detected ArUco markers
        cv.imshow("Frame", frame)

        # Wait for a key press for 1 millisecond; exit if 'Esc' is pressed
        key = cv.waitKey(1)
        if key == 27:
            break
        if key == ord("c"):
            corrector.calibrate(frame)

    # Release the video capture object to free resources
    cap.release()
    # Destroy all OpenCV-created windows to free resources
    cv.destroyAllWindows()


# Execute the main function when this script is run
if __name__ == "__main__":
    main()
