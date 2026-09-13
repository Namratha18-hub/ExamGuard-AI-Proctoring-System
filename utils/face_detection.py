import cv2
import os
import time
import atexit
import signal
import sys
import threading
import numpy as np
from ultralytics import YOLO
from utils.head_pose import get_head_direction
from utils.event_logger import log_event, end_session as end_db_session
from utils.detection_rules import (
    evaluate_session,
    FACE_ABSENT_DURATION_LIMIT_SECONDS,
    FACE_ABSENT_CONFIRM_SECONDS,
    FACE_ABSENT_COOLDOWN_SECONDS,
    HEAD_TURN_CONFIRM_SECONDS,
    HEAD_TURN_COOLDOWN_SECONDS,
    MULTIPLE_PERSONS_CONFIRM_SECONDS,
    MULTIPLE_PERSONS_COOLDOWN_SECONDS,
    MIN_CONSECUTIVE_CONFIRM_FRAMES,
    PERSON_BOX_IOU_MERGE_THRESHOLD,
    PERSON_BOX_CONTAINMENT_MERGE_THRESHOLD,
    YOLO_MIN_CONFIDENCE,
    PERSON_DETECTION_MIN_CONFIDENCE,
    FACE_HAAR_SCALE_FACTOR,
    FACE_HAAR_MIN_NEIGHBORS,
    FACE_HAAR_MIN_SIZE,
)

# =====================================
# Load YOLO Model
# =====================================

model = YOLO("yolov8m.pt")

# =====================================
# Load Haar Cascade
# =====================================

cascade_path = os.path.join(
    "models",
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise Exception("Cannot load Haar Cascade XML")

# =====================================
# Open Camera
# =====================================
camera = None
latest_frame = None

# threaded=True on app.run() (see app.py) means multiple requests can
# now genuinely execute concurrently -- previously the shared `camera`
# object was implicitly safe only because Werkzeug's single-threaded
# dev server could never have two threads touch it at once. This lock
# protects the actual camera-lifecycle operations (open/reopen/
# release) so two overlapping requests (e.g. a page refresh landing
# while the previous tab's stream hasn't fully torn down yet) can
# never both try to open/replace the same cv2.VideoCapture at the
# same time. It intentionally does NOT wrap per-frame processing
# (Haar cascade / YOLO / MediaPipe) -- only the camera I/O itself --
# so unrelated CPU work in one request is never serialized behind
# another request's frame processing.
_camera_lock = threading.Lock()

# =====================================
# Exam Variables
# =====================================

exam_terminated = False

# =====================================
# Milestone 2 - Session Tracking State
# =====================================
# current_session_id ties camera-side events to the same `sessions`
# row / session_id used by browser-side events in app.py. It stays
# None outside of an active exam (e.g. during registration's face
# capture, which reuses this same video stream), so no events are
# logged to the database unless an exam session is actually active.

current_session_id = None

# Tracks the last known state of each event type so we log to the
# database only on a transition (False -> True), not on every frame.
# phone/laptop/book stay simple edge-triggers on purpose -- a real
# phone/laptop/book appearing in frame is not the kind of ambiguous,
# flickery signal face-presence or person-count are, so no
# confirm/cooldown gating is needed here.
_event_states = {
    "phone_detected": False,
    "laptop_detected": False,
    "book_detected": False,
}

_termination_persisted = False     # guards against repeated end_session() calls


# =====================================
# Debounced "confirmed violation" gate
# =====================================
# Shared by every camera-side signal that is inherently noisy
# frame-to-frame (face presence, head direction, person count) and
# therefore must NOT log a database event on a single flickery frame.
#
# THE BUG THIS FIXES: previously, "face_absent" had NO debounce at
# all -- it logged to event_logs on the very first frame the Haar
# cascade happened to miss a face (a blink, a slight tilt, a moment of
# harsh backlight, one dropped camera frame -- all completely normal
# and momentary), then immediately reset as soon as the next frame
# detected a face again. A candidate sitting normally in frame for a
# whole exam could rack up dozens of these one-frame "face_absent"
# events even though a face was visible essentially the entire time.
# "head_turn" and "multiple_persons" were partially better (they did
# require CONFIRM_SECONDS of continuous time before logging) but had
# no COOLDOWN either, so flicker right at the boundary of a real
# incident could re-confirm a fresh "incident" almost immediately
# after the previous one cleared.
#
# A gate requires BOTH of the following before it will report a fresh
# confirmed violation:
#   1. The raw condition (no face / person_count > 1 / turned away)
#      has held continuously for `confirm_seconds` of wall-clock time.
#   2. AND for at least `min_consecutive_frames` consecutive processed
#      camera frames (guards the time-based check on very slow/lagging
#      hardware where a single slow frame could otherwise span the
#      whole confirm window).
#
# Once confirmed, it will not report again for the SAME incident. A
# recovered (condition-false) frame does not immediately let a brand
# new incident start confirming -- the condition must stay false
# continuously for `cooldown_seconds` first. This is what stops
# boundary flicker (e.g. a face that's juuust barely out of frame,
# alternating detected/not-detected every other frame) from
# re-triggering a fresh confirmation moments after the last one ended.
# An incident that never reached confirmation, on the other hand,
# cancels immediately on recovery -- there's nothing to "cool down"
# from if nothing was ever confirmed.
class _ConfirmGate:

    def __init__(self, confirm_seconds, min_consecutive_frames, cooldown_seconds):
        self.confirm_seconds = confirm_seconds
        self.min_consecutive_frames = min_consecutive_frames
        self.cooldown_seconds = cooldown_seconds
        self.reset()

    def reset(self):
        self.since = None              # when the CURRENT incident started
        self.confirmed = False         # has this incident already been logged?
        self.consistent_frames = 0
        self._cleared_since = None     # when the condition most recently went false

    def ongoing_seconds(self):
        """
        How long (seconds) the current incident has been running --
        0.0 if none is in progress. Keeps counting continuously across
        brief within-cooldown recoveries once confirmed, so a caller
        that needs the TOTAL duration of an ongoing confirmed
        violation (e.g. the 2-minute "prolonged absence" check) gets
        an accurate, flicker-immune number.
        """
        if self.since is None:
            return 0.0
        return time.time() - self.since

    def update(self, condition_now):
        """
        Feed this gate the current frame's raw (undebounced) condition.
        Returns True on exactly the one frame that newly confirms a
        fresh incident (the caller should log_event() now); False on
        every other frame, including all frames of an
        already-confirmed ongoing incident.
        """
        now = time.time()

        if condition_now:

            self._cleared_since = None

            if self.since is None:
                self.since = now

            self.consistent_frames += 1

            duration = now - self.since

            if (
                duration >= self.confirm_seconds
                and self.consistent_frames >= self.min_consecutive_frames
                and not self.confirmed
            ):
                self.confirmed = True
                return True

            return False

        else:

            if self.confirmed:
                # Only clear a CONFIRMED incident after the condition
                # has been continuously false for the full cooldown --
                # a single recovered frame amid ongoing flicker must
                # not immediately re-arm a fresh confirm cycle.
                if self._cleared_since is None:
                    self._cleared_since = now

                if now - self._cleared_since >= self.cooldown_seconds:
                    self.reset()

            else:
                # Never confirmed this incident -- nothing to cool
                # down from, so a recovered frame simply cancels the
                # in-progress (unconfirmed) attempt right away.
                self.reset()

            return False


_face_absent_gate = _ConfirmGate(
    FACE_ABSENT_CONFIRM_SECONDS, MIN_CONSECUTIVE_CONFIRM_FRAMES, FACE_ABSENT_COOLDOWN_SECONDS
)
_face_absent_prolonged_logged = False

_head_turn_gate = _ConfirmGate(
    HEAD_TURN_CONFIRM_SECONDS, MIN_CONSECUTIVE_CONFIRM_FRAMES, HEAD_TURN_COOLDOWN_SECONDS
)

_multiple_persons_gate = _ConfirmGate(
    MULTIPLE_PERSONS_CONFIRM_SECONDS, MIN_CONSECUTIVE_CONFIRM_FRAMES, MULTIPLE_PERSONS_COOLDOWN_SECONDS
)

# =====================================
# YOLO Frame Throttling
# =====================================
# Running the YOLO model on every single frame is the dominant cost
# in this loop -- on CPU-only hardware it can drop the effective
# stream rate low enough that the video looks frozen/black and badly
# behind real time (your "not live" / laggy feed). Haar-cascade face
# detection and the head-pose check are comparatively cheap, so they
# still run every frame; YOLO instead runs every YOLO_FRAME_INTERVAL
# frames, reusing its last real result on the frames in between. This
# does not remove person/phone/laptop/book detection -- it just
# avoids re-running the expensive model on every single frame, so
# violations are still caught within a fraction of a second.
YOLO_FRAME_INTERVAL = 3

_yolo_frame_counter = 0
_last_person_count = 0
_last_phone_detected = False
_last_laptop_detected = False
_last_book_detected = False

# =====================================
# Registration Face-Capture Status
# =====================================
# Plain "is a face currently visible" signal, updated every frame
# regardless of whether an exam session is active. Used by the
# /register page's clean face-capture status text -- NOT used for any
# exam warning/scoring/termination logic.

latest_face_detected = False


# =====================================
# Duplicate person-box merging (fixes false
# "Multiple Persons" with only one real person)
# =====================================

def _compute_iou(box_a, box_b):
    """
    Intersection-over-Union of two (x1, y1, x2, y2) boxes.
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)

    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def _compute_containment_ratio(box_a, box_b):
    """
    What fraction of the SMALLER box's area overlaps with the larger
    box. Complements IoU for duplicate-person detection: YOLO
    sometimes emits a tight box (e.g. head/shoulders) and a much
    larger loose box (full body) for the SAME real person. Such a
    pair can have low IoU (the union is dominated by the big box) even
    though the small box is almost entirely INSIDE the big one --
    IoU alone would fail to merge them, letting one real person read
    as two. This ratio catches that case regardless of size mismatch.
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)

    smaller_area = min(area_a, area_b)

    if smaller_area <= 0:
        return 0.0

    return inter_area / smaller_area


def _merge_person_boxes(boxes, iou_threshold, containment_threshold):
    """
    YOLO occasionally emits two boxes for one real person -- either
    two overlapping boxes of similar size, or a tight box + a loose
    box of very different size (e.g. head/shoulders vs full body).
    Two "person" boxes are treated as ONE physical person if EITHER:
      - their IoU is >= iou_threshold (handles similar-size overlap), OR
      - their containment ratio is >= containment_threshold (handles
        very different-size nested boxes IoU alone would miss).
    This collapses duplicates before person_count is computed, which
    is the direct fix for "Multiple Persons" firing with only one
    person actually on camera.

    boxes: list of (x1, y1, x2, y2, confidence) for label == "person".
    Returns the deduplicated list (highest-confidence box per group
    survives).
    """
    boxes_sorted = sorted(boxes, key=lambda b: b[4], reverse=True)

    kept = []

    for box in boxes_sorted:

        is_duplicate = False

        for kept_box in kept:
            if _compute_iou(box[:4], kept_box[:4]) >= iou_threshold:
                is_duplicate = True
                break
            if _compute_containment_ratio(box[:4], kept_box[:4]) >= containment_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(box)

    return kept


# =====================================
# Session Start / End (Milestone 2)
# =====================================

def start_camera_session(session_id):
    """
    Associates the camera/detection loop with a specific exam
    session_id so face-monitoring violations are logged against the
    correct row in event_logs. Must be called once, right when an
    exam starts (see app.py /exam/<course>).
    """
    global current_session_id
    global _face_absent_prolonged_logged, _termination_persisted
    global _yolo_frame_counter, _last_person_count
    global _last_phone_detected, _last_laptop_detected, _last_book_detected

    current_session_id = session_id
    _face_absent_prolonged_logged = False
    _termination_persisted = False

    # Fresh detection state per exam -- a stale "phone detected" flag
    # (or an in-progress/confirmed gate) left over from a previous
    # candidate's session should never carry into a new one.
    _yolo_frame_counter = 0
    _last_person_count = 0
    _last_phone_detected = False
    _last_laptop_detected = False
    _last_book_detected = False

    _face_absent_gate.reset()
    _head_turn_gate.reset()
    _multiple_persons_gate.reset()

    for key in _event_states:
        _event_states[key] = False

    reset_exam()


def end_camera_session():
    """
    Clears the active session association (called after exam
    submission/logout) so stray frames from later camera use (e.g.
    another candidate's registration photo capture) are never logged
    against a finished exam session.
    """
    global current_session_id

    current_session_id = None


def get_face_detected():
    """
    Returns whether a face is visible in the most recently processed
    frame. Independent of exam sessions -- safe to poll from the
    registration page, which has no active session.
    """
    global latest_face_detected

    return latest_face_detected


# =====================================
# Generate Camera Frames
# =====================================
def generate_frames():

    global camera
    global latest_frame
    global exam_terminated
    global latest_face_detected

def _open_camera():
    """
    Opens the webcam, trying the DirectShow backend first (matches
    the original behaviour) and falling back to OpenCV's default
    backend if that fails -- CAP_DSHOW is occasionally unable to
    re-acquire a device that was released (or left open) by a
    previous process, while the default backend can still succeed.
    Returns an opened cv2.VideoCapture, or None if neither works.
    """
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if capture.isOpened():
        return capture

    capture.release()

    capture = cv2.VideoCapture(0)

    if capture.isOpened():
        return capture

    capture.release()
    return None


def _unavailable_frame_bytes(message="Camera unavailable"):
    """
    A plain placeholder JPEG frame shown in place of real video when
    the webcam can't be opened, so the candidate/registration page
    shows a clear, visible message instead of a silently broken
    <img> tag -- a camera failure should be obvious on screen, not
    just a `print()` in a terminal nobody but a developer sees.
    """
    placeholder = np.zeros((480, 640, 3), dtype="uint8")

    cv2.putText(
        placeholder,
        message,
        (60, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        placeholder,
        "Check the camera is connected and not in use by another app.",
        (30, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )

    ret, buffer = cv2.imencode(".jpg", placeholder)
    return buffer.tobytes() if ret else b""


# Ensures the OS camera handle is released whenever this process
# exits for ANY reason -- including Flask's debug-mode auto-reloader
# restarting the worker process on every code change, which happens
# constantly during development. Without this, a process that never
# explicitly calls stop_camera() (as was the case here -- app.py
# imports it but never calls it) leaves the webcam device held open
# after the process is gone, and the NEXT process's
# cv2.VideoCapture(0) can then fail to (re)acquire it -- exactly the
# kind of failure that shows up as "the camera just stopped working"
# with no obvious cause, and affects every page that uses the camera
# (registration AND exam) since they share this same module-level
# camera object.
atexit.register(lambda: stop_camera())


# atexit alone does not cover every way this process can be stopped.
# It only runs on NORMAL Python interpreter shutdown (reaching the
# end of the program, or an unhandled KeyboardInterrupt/SystemExit
# unwinding cleanly) -- it does NOT run for SIGTERM by default, since
# SIGTERM's default action is immediate termination without invoking
# any Python-level cleanup at all. That gap is exactly what could
# leave the physical webcam indicator light on after "stopping Flask"
# depending on how it was stopped. These explicit handlers cover
# SIGINT (Ctrl+C in a terminal) and SIGTERM (a normal `kill`/stop
# request) by releasing the camera before exiting.
#
# Known, unavoidable limitation: SIGKILL, and Windows' TerminateProcess
# (which is what many IDE "Stop" buttons and Task Manager "End Task"
# use) cannot be intercepted by ANY process -- the OS ends the process
# immediately with zero opportunity for any application code, in any
# language, to run first. No amount of Python-level cleanup can catch
# that case; only the OS/driver reclaiming the hardware handle after
# the fact can release it then. Stopping the server with Ctrl+C in the
# terminal it's actually running in (rather than a forceful IDE stop)
# is what these handlers can guarantee.
def _handle_termination_signal(signum, frame):
    stop_camera()
    sys.exit(0)


try:
    # signal.signal() only works when called from the main thread of
    # the main interpreter -- guarded so importing this module can
    # never raise/crash if it's ever loaded in a different context
    # (e.g. a test runner or WSGI worker importing it on a background
    # thread). In that case, the atexit hook above still applies.
    signal.signal(signal.SIGINT, _handle_termination_signal)
    signal.signal(signal.SIGTERM, _handle_termination_signal)

    if hasattr(signal, "SIGBREAK"):
        # Windows-specific: Ctrl+Break, distinct from Ctrl+C/SIGINT.
        signal.signal(signal.SIGBREAK, _handle_termination_signal)

except ValueError:
    pass


# =====================================
# Generate Camera Frames
# =====================================
def generate_frames():

    global camera
    global latest_frame
    global exam_terminated
    global latest_face_detected

    if camera is None:
        with _camera_lock:
            if camera is None:  # re-check inside the lock (double-checked locking)
                camera = _open_camera()

    if camera is None or not camera.isOpened():

        # Self-healing retry: a camera object that exists but is no
        # longer actually open (e.g. the device was unplugged, or a
        # previous process left it in a bad state) used to be a
        # PERMANENT dead end here -- every future request would see
        # `camera is not None`, skip straight past the `if camera is
        # None` check above, immediately fail the isOpened() check
        # again, and give up -- for the rest of the process's life,
        # with no way to recover short of a full server restart.
        #
        # A FEW retries with a short pause are given here (not just
        # one immediate reopen) because right after opening, some
        # backends report isOpened() before the device has actually
        # finished initializing -- an instant reopen attempt can fail
        # for the same reason the first one did.
        with _camera_lock:
            # Re-check inside the lock -- another concurrent request
            # may have already fixed this between the check above and
            # acquiring the lock, in which case there's nothing to do.
            if camera is None or not camera.isOpened():
                if camera is not None:
                    camera.release()

                camera = None
                for _ in range(3):
                    camera = _open_camera()
                    if camera is not None:
                        break
                    time.sleep(0.3)

        if camera is None:
            print("Cannot open webcam -- yielding an on-screen error frame instead of nothing.")
            error_frame = _unavailable_frame_bytes()
            for _ in range(5):
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + error_frame +
                    b'\r\n'
                )
                time.sleep(1)
            return

    # Consecutive-failure counter for the read loop below: a SINGLE
    # failed camera.read() is common and normal (a dropped USB frame,
    # brief driver hiccup) and must NOT be treated as a disconnect --
    # doing so was the actual bug introduced by an earlier version of
    # this fix. Releasing and reopening the camera on every isolated
    # blip forces the hardware to "warm up" again (auto-exposure/
    # sensor renegotiation), which itself often produces MORE
    # transient failures or solid-black frames during that window --
    # triggering another release+reopen before it ever stabilizes,
    # which is exactly what produced a permanently black video feed
    # with no overlay text at all. Only a SUSTAINED run of failures
    # (genuinely no new frame for well over a second) now triggers an
    # actual reopen.
    consecutive_read_failures = 0
    MAX_CONSECUTIVE_READ_FAILURES = 30  # ~1-1.5s at a typical 20-30fps read rate

    while True:

        success, frame = camera.read()

        if not success:

            consecutive_read_failures += 1

            if consecutive_read_failures < MAX_CONSECUTIVE_READ_FAILURES:
                # Tolerate the blip -- do NOT touch the camera object,
                # just try again next loop iteration exactly like a
                # healthy capture naturally recovers on its own.
                continue

            # Only now, after a genuinely sustained run of failures,
            # treat this as an actual disconnect worth recovering from.
            with _camera_lock:
                camera.release()
                camera = _open_camera()
            consecutive_read_failures = 0

            if camera is None:
                error_frame = _unavailable_frame_bytes("Camera disconnected")
                for _ in range(5):
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + error_frame +
                        b'\r\n'
                    )
                    time.sleep(1)
                return

            continue

        consecutive_read_failures = 0

        frame = cv2.flip(frame, 1)

        latest_frame = frame.copy()
        # Per-frame proctoring processing (Haar cascade, YOLO,
        # mediapipe head-pose, warning/event logging, overlay
        # drawing) is wrapped in try/except: an unexpected exception
        # anywhere in here (e.g. an incompatible dependency version)
        # used to kill the ENTIRE generator -- Flask would drop the
        # connection, and since the camera itself opened/read fine,
        # none of the camera-open/read error handling above would
        # ever trigger either, leaving the browser's <img> retrying
        # forever against the exact same failure with nothing visible
        # on screen -- a silent, undiagnosable black panel. Now the
        # camera stream itself survives a processing error: it shows
        # a clear on-screen message and the real traceback is printed
        # to the server console so the actual cause is diagnosable,
        # instead of a dead stream with no explanation anywhere.
        try:

            # ============================
            # FACE DETECTION
            # ============================

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=FACE_HAAR_SCALE_FACTOR,
                minNeighbors=FACE_HAAR_MIN_NEIGHBORS,
                minSize=FACE_HAAR_MIN_SIZE
            )

            latest_face_detected = len(faces) > 0

            for (x, y, w, h) in faces:

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Face",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            # ============================
            # HEAD POSE
            # ============================

            direction = get_head_direction(frame)

            direction_color = (0, 255, 0)

            if direction != "Looking Straight":
                direction_color = (0, 0, 255)

            cv2.putText(
                frame,
                direction,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                direction_color,
                2
            )

            # ============================
            # YOLO OBJECT DETECTION (throttled)
            # ============================

            global _yolo_frame_counter, _last_person_count
            global _last_phone_detected, _last_laptop_detected, _last_book_detected

            run_yolo_this_frame = (_yolo_frame_counter % YOLO_FRAME_INTERVAL == 0)
            _yolo_frame_counter += 1

            if run_yolo_this_frame:

                results = model(frame, verbose=False)

                raw_person_boxes = []
                phone_detected = False
                laptop_detected = False
                book_detected = False

                for result in results:

                    for box in result.boxes:

                        confidence = float(box.conf[0])

                        if confidence < YOLO_MIN_CONFIDENCE:
                            continue

                        cls = int(box.cls[0])

                        label = model.names[cls]

                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (255, 0, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0),
                            2
                        )

                        if label == "person" and confidence > PERSON_DETECTION_MIN_CONFIDENCE:
                            raw_person_boxes.append((x1, y1, x2, y2, confidence))

                        elif label == "cell phone":
                            phone_detected = True

                        elif label == "laptop":
                            laptop_detected = True

                        elif label == "book":
                            book_detected = True

                # Collapse duplicate boxes for the same physical person
                # (e.g. YOLO emitting a tight box + a loose box around one
                # candidate) BEFORE counting -- this is what stops one
                # real person from ever reading as person_count == 2.
                merged_person_boxes = _merge_person_boxes(
                    raw_person_boxes,
                    PERSON_BOX_IOU_MERGE_THRESHOLD,
                    PERSON_BOX_CONTAINMENT_MERGE_THRESHOLD
                )
                person_count = len(merged_person_boxes)

                # Cache this real result so the next (skipped) frames can
                # reuse it instead of leaving warnings logic with nothing.
                _last_person_count = person_count
                _last_phone_detected = phone_detected
                _last_laptop_detected = laptop_detected
                _last_book_detected = book_detected

            else:

                # Skip the expensive model call this frame -- reuse the
                # most recent real (already-merged) detection so the
                # warnings/marks logic below stays continuous, just
                # without redrawing boxes this frame.
                person_count = _last_person_count
                phone_detected = _last_phone_detected
                laptop_detected = _last_laptop_detected
                book_detected = _last_book_detected

            # =====================================
            # WARNINGS (Milestone 2: transition-based
            # DB logging instead of per-frame counting)
            # =====================================

            global _face_absent_prolonged_logged
            global _termination_persisted

            y = 70
            new_event_logged = False

            # Only draw exam warning/termination overlays when there is an
            # active exam session bound to the camera (see
            # start_camera_session / end_camera_session). During
            # registration's face capture, current_session_id is None, so
            # none of this is drawn on the video -- the registration page
            # shows its own clean status text instead (see /face_status).
            exam_overlay_active = current_session_id is not None

            # ------------------------------
            # No Face (debounced: FACE_ABSENT_CONFIRM_SECONDS continuous +
            # MIN_CONSECUTIVE_CONFIRM_FRAMES, with a FACE_ABSENT_COOLDOWN_SECONDS
            # cooldown before a new incident can re-confirm -- see
            # _ConfirmGate above)
            # ------------------------------
            # THE FIX: previously this block logged "face_absent" on the
            # very first frame the Haar cascade missed a face, with zero
            # debounce -- a single blink or lighting flicker was enough.
            # It now goes through the same confirm-gate every other camera
            # signal uses, so only a genuine, sustained absence is ever
            # written to event_logs. The on-screen "No Face Detected"
            # overlay still shows immediately (so the candidate gets
            # instant visual feedback to reposition), but the DATABASE
            # event only logs once the absence is actually confirmed.

            face_currently_absent = (len(faces) == 0)

            if exam_overlay_active and face_currently_absent:
                cv2.putText(
                    frame,
                    "WARNING : No Face Detected",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            if face_currently_absent:
                y += 35

            if _face_absent_gate.update(face_currently_absent):
                if log_event(current_session_id, "face_absent"):
                    new_event_logged = True

            # "Prolonged" absence uses the SAME gate's ongoing_seconds() --
            # once confirmed, that timer keeps counting continuously across
            # any brief within-cooldown flicker, so this reflects the true
            # total duration of the ongoing confirmed absence, not just
            # time since the last single missed frame.
            if _face_absent_gate.confirmed and not _face_absent_prolonged_logged \
                    and _face_absent_gate.ongoing_seconds() >= FACE_ABSENT_DURATION_LIMIT_SECONDS:

                if log_event(current_session_id, "face_absent_prolonged"):
                    new_event_logged = True

                _face_absent_prolonged_logged = True

            if _face_absent_gate.since is None:
                # Gate fully reset (incident over, cooldown elapsed) --
                # a genuinely NEW absence starting later should be able to
                # reach "prolonged" again too.
                _face_absent_prolonged_logged = False

            # ------------------------------
            # Multiple Persons (IoU + containment deduplicated,
            # confirm-duration + consecutive-frame + cooldown gated)
            # ------------------------------
            # person_count here has ALREADY had duplicate boxes for the
            # same physical person merged away above (both IoU-overlap and
            # containment, so a tight box + a loose box around ONE real
            # person no longer counts as two). On top of that, it only
            # becomes a CONFIRMED violation -- logged to the DB, counted
            # for marks and the 3-strike termination rule -- once it has
            # held continuously for BOTH MULTIPLE_PERSONS_CONFIRM_SECONDS
            # of wall-clock time AND at least MIN_CONSECUTIVE_CONFIRM_FRAMES
            # consecutive processed frames. It is only logged ONCE per
            # continuous incident, and (via the gate's cooldown) won't
            # re-confirm from boundary flicker moments after the extra
            # person leaves -- person_count has to genuinely stay at 1 for
            # MULTIPLE_PERSONS_COOLDOWN_SECONDS before a fresh incident can
            # start confirming again.

            multiple_persons_now = person_count > 1

            if exam_overlay_active and multiple_persons_now:
                cv2.putText(
                    frame,
                    "WARNING : Multiple Persons",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            if multiple_persons_now:
                y += 35

            if _multiple_persons_gate.update(multiple_persons_now):
                if log_event(current_session_id, "multiple_persons"):
                    new_event_logged = True

            # ------------------------------
            # Mobile Phone
            # ------------------------------

            if phone_detected:

                if exam_overlay_active:
                    cv2.putText(
                        frame,
                        "WARNING : Mobile Phone",
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                y += 35

                if not _event_states["phone_detected"]:
                    if log_event(current_session_id, "phone_detected"):
                        new_event_logged = True

            _event_states["phone_detected"] = phone_detected

            # ------------------------------
            # Laptop
            # ------------------------------

            if laptop_detected:

                if exam_overlay_active:
                    cv2.putText(
                        frame,
                        "WARNING : Laptop Detected",
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                y += 35

                if not _event_states["laptop_detected"]:
                    if log_event(current_session_id, "laptop_detected"):
                        new_event_logged = True

            _event_states["laptop_detected"] = laptop_detected

            # ------------------------------
            # Book
            # ------------------------------

            if book_detected:

                if exam_overlay_active:
                    cv2.putText(
                        frame,
                        "WARNING : Book Detected",
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                y += 35

                if not _event_states["book_detected"]:
                    if log_event(current_session_id, "book_detected"):
                        new_event_logged = True

            _event_states["book_detected"] = book_detected

            # ------------------------------
            # Head Direction (confirm-duration + consecutive-frame +
            # cooldown gated)
            # ------------------------------
            # A quick glance to read the question (or a single noisy
            # frame) must NOT log a warning. Only once direction has been
            # continuously a genuine turned-away direction for BOTH
            # HEAD_TURN_CONFIRM_SECONDS of wall-clock time AND at least
            # MIN_CONSECUTIVE_CONFIRM_FRAMES consecutive frames is this
            # treated as a confirmed, intentional look-away -- logged
            # ONCE per continuous look-away, and (via the gate's cooldown)
            # won't re-confirm from boundary flicker moments after the
            # candidate looks back.
            #
            # Two extra guards on top of the confirm-gate itself:
            #   - "No Face" (mediapipe's own failure to find a face this
            #     frame -- a DIFFERENT detector than the Haar cascade used
            #     for face_absent above) is explicitly NOT treated as a
            #     head turn. Previously it was ("No Face" != "Looking
            #     Straight" was True), so a momentary mediapipe miss could
            #     confirm a false head_turn independently of, and even
            #     while, the face was clearly visible to the Haar cascade.
            #   - If the Haar cascade says no face is present THIS frame
            #     (face_currently_absent), we don't have a reliable
            #     direction reading to act on at all -- that frame is left
            #     to the face_absent gate above, not double-counted here.

            head_turn_now = (
                not face_currently_absent
                and direction not in ("Looking Straight", "No Face")
            )

            if exam_overlay_active and head_turn_now:
                cv2.putText(
                    frame,
                    f"WARNING : {direction}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            if head_turn_now:
                y += 35

            if _head_turn_gate.update(head_turn_now):
                if log_event(current_session_id, "head_turn"):
                    new_event_logged = True

            # =====================================
            # Unified Warning Count (camera + browser
            # events combined, read back from event_logs)
            # =====================================
            # Warnings/Marks Deducted are no longer tracked or drawn
            # inside the camera stream -- they're shown exclusively in the
            # exam.html sidebar (outside the camera), which already stays
            # dynamic via the existing /session_status poll and
            # /update_warning responses, both reading from this same
            # evaluate_session() call. Here we only need the "terminated"
            # flag, to know whether to persist the terminated session
            # status and show the in-frame "EXAM TERMINATED" overlay.

            if new_event_logged and current_session_id is not None:

                result = evaluate_session(current_session_id)

                if result["terminated"]:

                    exam_terminated = True

                    if not _termination_persisted:
                        end_db_session(current_session_id, status="terminated")
                        _termination_persisted = True

            # =====================================
            # Terminate Exam
            # =====================================

            if exam_overlay_active and exam_terminated:

                cv2.putText(
                    frame,
                    "EXAM TERMINATED",
                    (100, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

        except Exception:
            import traceback
            traceback.print_exc()
            error_frame = _unavailable_frame_bytes("Proctoring error - check server console")
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + error_frame +
                b'\r\n'
            )
            continue

        # =====================================
        # STREAM TO FLASK
        # =====================================

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame_bytes +
            b'\r\n'
        )


# =====================================
# Capture Candidate Photo
# =====================================

def capture_photo():

    global latest_frame

    if latest_frame is None:
        print("No frame available")
        return False

    os.makedirs("static/photos", exist_ok=True)

    cv2.imwrite(
        "static/photos/candidate.jpg",
        latest_frame
    )

    print("Photo Saved Successfully")

    return True


# =====================================
# Stop Camera
# =====================================

def stop_camera():

    global camera

    with _camera_lock:
        if camera is not None:

            camera.release()

            cv2.destroyAllWindows()

            camera = None


# =====================================
# Reset Exam Data
# =====================================

def reset_exam():

    global exam_terminated

    exam_terminated = False