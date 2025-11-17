import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

capture = cv2.VideoCapture(0)
while capture.isOpened():
    ret, frame = capture.read()
    cv2.imshow("Holistic Model Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
