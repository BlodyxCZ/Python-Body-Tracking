import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(circle_radius=0, thickness=0)

cap = cv2.VideoCapture(0)


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue


    image = cv2.flip(image, 1)

    with mp_face.FaceMesh(max_num_faces=1,
                          refine_landmarks=False,
                          min_tracking_confidence=0.5,
                          min_detection_confidence=0.5
                          ) as face:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scale_percent = 220
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        results = face.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face.FACEMESH_FACE_OVAL,
                    landmark_drawing_spec=mp_drawing_spec,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing_spec,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

    with mp_hands.Hands(model_complexity=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                        ) as hands:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

    with mp_pose.Pose(model_complexity=1,
                      smooth_landmarks=True,
                      enable_segmentation=False,
                      smooth_segmentation=True,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5
                      ) as pose:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

    cv2.imshow('FULL BODY TRACKING', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
