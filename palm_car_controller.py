import cv2
import mediapipe as mp
import numpy as np
import pyfirmata
import time

# Initialize Mediapipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not found or unable to open.")
    exit()

# Initialize Arduino
port = "/dev/ttyACM0"  # Adjust this port based on your setup
try:
    board = pyfirmata.Arduino(port)
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    exit()

# Initialize Servo Pins
servo_pin_9 = board.get_pin('d:9:s')
servo_pin_10 = board.get_pin('d:10:s')

# Center servos at 90 degrees
servo_pin_9.write(90)
servo_pin_10.write(90)
time.sleep(1)  # Wait a moment for the servos to reach the position

# Function to move the servo smoothly
def smooth_servo_move(servo, start_angle, end_angle, step=1):
    if start_angle < end_angle:
        for angle in range(int(start_angle), int(end_angle) + 1, step):
            servo.write(angle)
            time.sleep(0.01)  # 10ms delay per degree
    else:
        for angle in range(int(start_angle), int(end_angle) - 1, -step):
            servo.write(angle)
            time.sleep(0.01)  # 10ms delay per degree

# Function to get the angle between two points
def get_angle(p1, p2):
    angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    return angle

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    img = cv2.flip(img, 1)  # Flip image horizontally
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    results = hands.process(img_rgb)  # Process the frame for hand detection

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of key landmarks (wrist and palm base)
            wrist = hand_landmarks.landmark[0]
            palm_base = hand_landmarks.landmark[9]

            img_height, img_width, _ = img.shape
            cx_wrist = int(wrist.x * img_width)
            cy_wrist = int(wrist.y * img_height)
            cx_palm_base = int(palm_base.x * img_width)
            cy_palm_base = int(palm_base.y * img_height)

            # Calculate the palm tilt angles
            angle_x = get_angle((cx_wrist, cy_wrist), (cx_palm_base, cy_palm_base))
            angle_y = get_angle((cy_wrist, cx_wrist), (cy_palm_base, cx_palm_base))

            # Map the tilt angles to servo angles
            servo_angle_9 = np.clip(np.interp(angle_x, [-90, 90], [0, 180]), 0, 180)
            servo_angle_10 = np.clip(np.interp(angle_y, [-90, 90], [0, 180]), 0, 180)

            # Move the servos smoothly
            current_angle_9 = servo_pin_9.read()
            current_angle_10 = servo_pin_10.read()

            smooth_servo_move(servo_pin_9, current_angle_9, servo_angle_9)
            smooth_servo_move(servo_pin_10, current_angle_10, servo_angle_10)

            # Draw the wrist and palm base points
            cv2.circle(img, (cx_wrist, cy_wrist), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (cx_palm_base, cy_palm_base), 10, (0, 0, 255), cv2.FILLED)

            # Draw a line connecting the wrist to the palm base
            cv2.line(img, (cx_wrist, cy_wrist), (cx_palm_base, cy_palm_base), (0, 255, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
