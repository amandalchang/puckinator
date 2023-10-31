import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])


# Initialize variables for FPS calculation
prev_time = time.time()
curr_time = time.time()


def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
        pixel = np.array(frame[y, x], dtype="uint8").reshape(1, 1, 3)

        # Convert BGR to HSV
        hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]

        # Update HSV values on the "Controls" window
        hsv_text = f"H: {hsv_pixel[0]}, S: {hsv_pixel[1]}, V: {hsv_pixel[2]}"
        black_background = np.zeros((50, 300, 3), dtype="uint8")
        cv2.putText(
            black_background,
            hsv_text,
            (5, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Controls", black_background)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Set slider positions based on clicked color
            cv2.setTrackbarPos("Lower Hue", "Controls", max(hsv_pixel[0] - 10, 0))
            cv2.setTrackbarPos("Upper Hue", "Controls", min(hsv_pixel[0] + 10, 179))
            cv2.setTrackbarPos(
                "Lower Saturation", "Controls", max(hsv_pixel[1] - 20, 0)
            )
            cv2.setTrackbarPos(
                "Upper Saturation", "Controls", min(hsv_pixel[1] + 20, 255)
            )
            cv2.setTrackbarPos("Lower Value", "Controls", max(hsv_pixel[2] - 20, 0))
            cv2.setTrackbarPos("Upper Value", "Controls", min(hsv_pixel[2] + 20, 255))


cv2.namedWindow("Frame")
cv2.namedWindow("Controls")
cv2.createTrackbar("Lower Hue", "Controls", 0, 179, lambda x: None)  # type: ignore
cv2.createTrackbar("Upper Hue", "Controls", 179, 179, lambda x: None)  # type: ignore
cv2.createTrackbar("Lower Saturation", "Controls", 0, 255, lambda x: None)  # type: ignore
cv2.createTrackbar("Upper Saturation", "Controls", 255, 255, lambda x: None)  # type: ignore
cv2.createTrackbar("Lower Value", "Controls", 0, 255, lambda x: None)  # type: ignore
cv2.createTrackbar("Upper Value", "Controls", 255, 255, lambda x: None)  # type: ignore

cv2.setMouseCallback("Frame", pick_color)  # type: ignore

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of the hue trackbars
    # Get current positions of the hue, saturation, and value trackbars
    lower_hue = cv2.getTrackbarPos("Lower Hue", "Controls")
    upper_hue = cv2.getTrackbarPos("Upper Hue", "Controls")
    lower_saturation = cv2.getTrackbarPos("Lower Saturation", "Controls")
    upper_saturation = cv2.getTrackbarPos("Upper Saturation", "Controls")
    lower_value = cv2.getTrackbarPos("Lower Value", "Controls")
    upper_value = cv2.getTrackbarPos("Upper Value", "Controls")

    # Define HSV Range for thresholding
    lower_red = np.array([lower_hue, lower_saturation, lower_value])
    upper_red = np.array([upper_hue, upper_saturation, upper_value])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour, assuming it is the puck
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(50)
    if key == 27:  # ESC key to break
        break

cap.release()
cv2.destroyAllWindows()
