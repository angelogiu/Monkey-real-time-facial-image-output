import mediapipe as mp
import cv2
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

basedirectory = os.path.dirname(os.path.abspath(__file__))
imagedirectory = os.path.join(basedirectory, "Monkey Images")


def load_img(filename):
    path = os.path.join(imagedirectory, filename)
    img = cv2.imread(path)
    if img is None:
        print("failed to open the image")
    return img


monkeyimages = {"neutral": load_img("neutral monkey.jpg"),
                "happy": load_img("happymonkey.jpg"),
                "sad": load_img("sadmonkey.jpg"),
                "finger": load_img("fuckumonkey.jpg"),
                "handonhead": load_img("handhead.jpg"),
                "thinking": load_img("thinkingmonkey.jpg"),
                "angrymonkey": load_img("angrymonkey.jpg"),
                "ideamonkey": load_img("ideamonkey.jpg")}


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

        h, w = image.shape[:2]
        newsize = (w, h)

        open_mouth = None
        tip_to_mouth = None
        index_ratio = None
        hand_to_head = None
        middle_length = None
        mouth_smile = None

        if face_results.multi_face_landmarks:
            face = face_results.multi_face_landmarks[0]
            upper_lip = face.landmark[13]
            lower_lip = face.landmark[14]
            open_mouth = abs(upper_lip.y - lower_lip.y)
            left_mouth_corner = face.landmark[61]
            corner_nose = face.landmark[49]

            top_head = face.landmark[10]
            # print("Mouth Distance is", open_mouth)

            mouth_smile = np.sqrt(
                np.square(corner_nose.x - left_mouth_corner.x) + np.square(corner_nose.y - left_mouth_corner.y))
            smile_ratio = mouth_smile / abs(corner_nose.z)
            # print("Mouth Courner L is", smile_ratio)

        if hand_results.multi_hand_landmarks:
            hand = hand_results.multi_hand_landmarks[0]
            index_tip = hand.landmark[8]
            tip_to_mouth = np.sqrt(
                np.square(index_tip.x - lower_lip.x) + np.square(index_tip.y - lower_lip.y))

            index_base = hand.landmark[5]
            index_length = np.sqrt(
                np.square(index_tip.x - index_base.x) + np.square(index_tip.y - index_base.y))
            index_ratio = index_length / abs(index_tip.z)

            wrist = hand.landmark[0]
            hand_to_head = np.sqrt(
                np.square(wrist.x - top_head.x) + np.square(wrist.y - top_head.y))

            middle_tip = hand.landmark[12]
            middle_base = hand.landmark[9]
            middle_length = np.sqrt(
                np.square(middle_tip.x - middle_base.x) + np.square(middle_tip.y - middle_base.y))

            # print("Middle finger length is", middle_length)
            # print("hand to head length is", hand_to_head)
            print("Index length is", index_ratio)
            #
            # print("Finger to mouth distance is", tip_to_mouth)

        def generatemonkey(open_mouth, tip_to_mouth, index_ratio, hand_to_head, middle_length, mouth_smile):
            if open_mouth is not None and open_mouth >= 0.05:
                return "angrymonkey"
            elif tip_to_mouth is not None and tip_to_mouth < 0.025:
                return "thinking"
            elif index_ratio is not None and 2.0 < index_ratio < 3.0:
                return "ideamonkey"
            elif hand_to_head is not None and hand_to_head < 0.17:
                return "handonhead"
            elif middle_length is not None and middle_length > 0.27:
                return "finger"
            elif mouth_smile is not None and smile_ratio < 2:
                return "happy"
            elif mouth_smile is not None and smile_ratio > 3.0:
                return "sad"

            else:
                return "neutral"

        monkeyface = generatemonkey(
            open_mouth, tip_to_mouth, index_ratio, hand_to_head, middle_length, mouth_smile)

        monkeyimg = monkeyimages.get(monkeyface)

        if monkeyimg is not None:
            monkey_resize = cv2.resize(monkeyimg, newsize)
            combined = np.hstack((image, monkey_resize))
        else:
            combined = image

        cv2.imshow("Face Mesh Model Detection", combined)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
