import cv2
import mediapipe as mp
import numpy as np
import pyfirmata
import time

# Initialize Mediapipe Face Mesh Detector
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Video Capture
cap = cv2.VideoCapture(0)  # 0 is usually the default camera. Change to 1 or 2 if needed.
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
def smooth_servo_move(servo, start_angle, end_angle):
    step = 1 if end_angle > start_angle else -1
    for angle in range(int(start_angle), int(end_angle), step):
        servo.write(angle)
        time.sleep(0.01)  # 10ms delay per degree

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    img = cv2.flip(img, 1)  # Flip image horizontally
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    results = face_mesh.process(img_rgb)  # Process the frame for face mesh detection
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # Convert back to BGR

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1))

            # Get the coordinates of key landmarks (e.g., nose tip, eyes, and mouth)
            nose_tip = face_landmarks.landmark[1]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            mouth_left = face_landmarks.landmark[61]
            mouth_right = face_landmarks.landmark[291]

            img_height, img_width, _ = img.shape
            cx_nose = int(nose_tip.x * img_width)
            cy_nose = int(nose_tip.y * img_height)
            cx_left_eye = int(left_eye.x * img_width)
            cy_left_eye = int(left_eye.y * img_height)
            cx_right_eye = int(right_eye.x * img_width)
            cy_right_eye = int(right_eye.y * img_height)
            cx_mouth_left = int(mouth_left.x * img_width)
            cy_mouth_left = int(mouth_left.y * img_height)
            cx_mouth_right = int(mouth_right.x * img_width)
            cy_mouth_right = int(mouth_right.y * img_height)

            # Calculate the center of the face
            face_center_x = (cx_left_eye + cx_right_eye) // 2
            face_center_y = (cy_left_eye + cy_right_eye) // 2

            # Determine head tilt direction based on nose position relative to the center
            if cx_nose < face_center_x - 20:
                # Head tilt to the left
                target_angle_9 = 100
            elif cx_nose > face_center_x + 20:
                # Head tilt to the right
                target_angle_9 = 80
            else:
                # Center
                target_angle_9 = 90

            if cy_nose < face_center_y - 20:
                # Head tilt forward
                target_angle_10 = 100
            elif cy_nose > face_center_y + 20:
                # Head tilt backward
                target_angle_10 = 80
            else:
                # Center
                target_angle_10 = 90

            # Move the servos smoothly
            current_angle_9 = servo_pin_9.read()
            current_angle_10 = servo_pin_10.read()

            smooth_servo_move(servo_pin_9, current_angle_9, target_angle_9)
            smooth_servo_move(servo_pin_10, current_angle_10, target_angle_10)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
