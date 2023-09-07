import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(circle_radius=0, thickness=0)

cap = cv2.VideoCapture(0)

mp_face = mp.solutions.face_mesh

with mp_face.FaceMesh(max_num_faces=1,
                           refine_landmarks=True,
                           min_tracking_confidence=0.5,
                           min_detection_confidence=0.5
                           ) as face:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        scale_percent = 220
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        results = face.process(image)

        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_FACE_OVAL,
                landmark_drawing_spec=mp_drawing_spec,
                connection_drawing_spec=mp_drawing_styles.get_default_face_contours_style()
            )
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing_spec,
                connection_drawing_spec=mp_drawing_styles.get_default_face_tesselation_style()
            )

        cv2.imshow("Face", image)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()