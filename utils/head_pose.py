import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================
# Part 6 - Scale-normalized direction thresholds
# =====================================
# These are fractions of the candidate's OWN measured face
# width/height, recomputed live from the actual landmarks every
# frame -- not fixed pixel offsets. A fixed pixel threshold (the
# previous approach) only works at one specific distance from the
# camera: the same real head-turn angle produces a small pixel
# offset when the candidate sits farther away and a large one when
# they sit closer, so a fixed number either misses real turns or
# false-triggers on tiny ones depending on distance. Dividing by the
# live face_width/face_height cancels out that distance-dependent
# scale, so the same real angle is detected consistently regardless
# of how close/far the candidate is or the camera's resolution.
HORIZONTAL_RATIO_THRESHOLD = 0.15
VERTICAL_UP_RATIO_THRESHOLD = 0.12
VERTICAL_DOWN_RATIO_THRESHOLD = 0.18


def get_head_direction(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    direction = "No Face"

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        nose = face.landmark[1]
        left = face.landmark[234]
        right = face.landmark[454]
        top = face.landmark[10]
        bottom = face.landmark[152]

        nose_x = nose.x * w
        nose_y = nose.y * h

        left_x = left.x * w
        right_x = right.x * w

        top_y = top.y * h
        bottom_y = bottom.y * h

        # Live, per-frame face size -- this is what makes the
        # thresholds below scale-invariant instead of static.
        face_width = right_x - left_x
        face_height = bottom_y - top_y

        # A degenerate/unreliable landmark read (e.g. a sliver of face
        # at the very edge of frame) would otherwise divide by ~0 and
        # produce a huge, meaningless ratio -- treat it as no
        # reliable face rather than risk a false violation.
        if face_width <= 1 or face_height <= 1:
            return "No Face"

        horizontal_ratio = (nose_x - (left_x + right_x) / 2) / face_width
        vertical_ratio = (nose_y - (top_y + bottom_y) / 2) / face_height

        if horizontal_ratio < -HORIZONTAL_RATIO_THRESHOLD:
            direction = "Looking Left"

        elif horizontal_ratio > HORIZONTAL_RATIO_THRESHOLD:
            direction = "Looking Right"

        elif vertical_ratio < -VERTICAL_UP_RATIO_THRESHOLD:
            direction = "Looking Up"

        elif vertical_ratio > VERTICAL_DOWN_RATIO_THRESHOLD:
            direction = "Looking Down"

        else:
            direction = "Looking Straight"

    return direction