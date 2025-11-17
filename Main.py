import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

monkeyimages = {"neutral": cv2.imread("Monkey Images/neutral monkey.jpg"),
                "happy": cv2.imread("Monkey Images/happymonkey.jpg"),
                "sad": cv2.imread("Monkey Images/sadmonkey.jpg"),
                "finger": cv2.imread("Monkey Images/fuckumonkey.jpg"),
                "handonhead": cv2.imread("Monkey Images/handhead.jpg"),
                "thinking": cv2.imread("Monkey Images/neutral monkey.jpg"),
                "angrymonkey": cv2.imread("Monkey Images/angrymonkey.jpg"),
                "ideamonkey": cv2.imread("Monkey Images/ideamonkey.jpg")}

newsize = (200, 150)
scaledimages = {}
for name, img in monkeyimages.items():
    scaledimages[name] = img.resize(newsize)


cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
        mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(image)
        hand_results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow("Fase Mesh Model Detection", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
