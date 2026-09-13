import cv2
import mediapipe as mp

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION
            )

    cv2.imshow("ExamGuard Face Mesh", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()