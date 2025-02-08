import numpy as np
import cv2

def calculate_ear(landmarks):
    """
    Calculates the Eye Aspect Ratio (EAR) for both eyes.
    """
    # EAR for the left eye
    left_eye = [landmarks.part(i) for i in range(36, 42)]
    # EAR for the right eye
    right_eye = [landmarks.part(i) for i in range(42, 48)]

    def eye_aspect_ratio(eye):
        a = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
        b = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
        c = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
        return (a + b) / (2.0 * c)

    return eye_aspect_ratio(left_eye), eye_aspect_ratio(right_eye)

def calculate_mar(landmarks):
    """
    Calculates the Mouth Aspect Ratio (MAR) for detecting yawning.
    """
    top_lip = np.linalg.norm(np.array([landmarks.part(51).x, landmarks.part(51).y]) -
                             np.array([landmarks.part(57).x, landmarks.part(57).y]))
    bottom_lip = np.linalg.norm(np.array([landmarks.part(62).x, landmarks.part(62).y]) -
                                np.array([landmarks.part(66).x, landmarks.part(66).y]))
    mouth_width = np.linalg.norm(np.array([landmarks.part(48).x, landmarks.part(48).y]) -
                                 np.array([landmarks.part(54).x, landmarks.part(54).y]))

    mar = (top_lip + bottom_lip) / (2.0 * mouth_width)
    return mar

def draw_landmarks(image, landmarks):
    """
    Draws facial landmarks on the given image.
    """
    for i in range(68):
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

def overlay_text(image, status):
    """
    Displays status on the image.
    """
    text = f"Drowsiness Status: {status}"
    color = (0, 255, 0) if status == "Well Rested" else (0, 0, 255)

    # Font and background for text
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_width, text_height = text_size
    overlay_x, overlay_y = 10, 40
    cv2.rectangle(
        image,
        (overlay_x - 5, overlay_y - text_height - 5),
        (overlay_x + text_width + 5, overlay_y + 5),
        (0, 0, 0),
        -1
    )
    # Add text
    cv2.putText(image, text, (overlay_x, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
