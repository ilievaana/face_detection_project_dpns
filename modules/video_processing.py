import cv2
import dlib
from modules.utils import calculate_ear, draw_landmarks, overlay_text

def process_video():
    model_path = "resources/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            draw_landmarks(frame, landmarks)

            # Calculate EAR
            ear_left, ear_right = calculate_ear(landmarks)
            avg_ear = (ear_left + ear_right) / 2.0
            status = "Well Rested" if avg_ear > 0.25 else "Drowsiness Detected"

            # Add text overlay
            overlay_text(frame, status)

        cv2.imshow("Live Video", frame)

        # Check for quit shortcuts
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 17:  # ASCII for Ctrl+Q is 17
            break

    cap.release()
    cv2.destroyAllWindows()
