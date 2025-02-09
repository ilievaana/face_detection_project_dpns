import cv2
import dlib
import numpy as np
from modules.utils import calculate_ear, calculate_mar, draw_landmarks, overlay_text


# Constants
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
BLINK_FRAME_THRESHOLD = 3 # number of frames for blinking
DROWSINESS_FRAME_THRESHOLD = 30
CLOSED_EYE_TIME_THRESHOLD = 2  # seconds for closed eyes
YAWN_FRAME_THRESHOLD = 15  # number of frames for continuous yawning
FPS = 30  # number of frames per second

def process_video():
    model_path = "resources/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    cap = cv2.VideoCapture(0)
    blink_counter = 0
    drowsy_counter = 0
    closed_eye_frames = 0
    yawn_counter = 0
    avg_ear_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            draw_landmarks(frame, landmarks)

            # Calculate EAR & MAR
            ear_left, ear_right = calculate_ear(landmarks)
            avg_ear = (ear_left + ear_right) / 2.0
            mar = calculate_mar(landmarks)
            avg_ear_history.append(avg_ear)

            # Maintain history length
            if len(avg_ear_history) > 10:
                avg_ear_history.pop(0)

            # Compute dynamic EAR threshold
            dynamic_threshold = np.mean(avg_ear_history) * 0.85
            current_threshold = max(dynamic_threshold, EAR_THRESHOLD)

            # Detect blinks
            if avg_ear < current_threshold:
                blink_counter += 1
                closed_eye_frames += 1
            else:
                if 1 <= blink_counter <= BLINK_FRAME_THRESHOLD:
                    overlay_text(frame, "Blink Detected")
                blink_counter = 0
                closed_eye_frames = 0

            # Detect yawning
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
            else:
                yawn_counter = 0

            # Detect prolonged drowsiness
            if closed_eye_frames / FPS >= CLOSED_EYE_TIME_THRESHOLD or yawn_counter >= YAWN_FRAME_THRESHOLD:
                status = "High Sleepiness Risk!"
            elif drowsy_counter >= DROWSINESS_FRAME_THRESHOLD:
                status = "Drowsiness Detected"
            else:
                status = "Well Rested"

            # Add text overlay
            overlay_text(frame, status)

        cv2.imshow("Live Video", frame)

        # Check for quit shortcuts
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 17:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
