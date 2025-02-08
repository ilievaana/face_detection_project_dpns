import cv2
import dlib
from modules.utils import draw_landmarks, calculate_ear, overlay_text


def process_image(filepath):
    model_path = "resources/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    # Вчитување на сликата
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Детекција на лица
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        draw_landmarks(image, landmarks)

        # Пресметај EAR
        ear_left, ear_right = calculate_ear(landmarks)
        avg_ear = (ear_left + ear_right) / 2.0
        status = "Well Rested" if avg_ear > 0.25 else "Drowsiness Detected"

        # Додај текст
        overlay_text(image, status)

    # Прикажи ја резултатната слика
    cv2.imshow("Detected Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_image("test_image.jpg")

