"""
Milestone 2 - Configurable, rule-based suspicious event detection.

All thresholds live as plain constants at the top of this file so they
can be tuned without touching any detection or logging logic.

Rule evaluation reads from the existing `event_logs` table (via
utils.event_logger) so that camera-triggered events (face absence,
multiple persons, phone/laptop/book, head-turn) and browser-triggered
events (tab switches, fullscreen exits) are judged against ONE shared
set of rules and ONE unified warning count.

IMPORTANT: only CONFIRMED events ever reach event_logs for
multiple_persons and head_turn -- the confirm-duration gating (so a
quick glance to read the question, or a one-frame flicker, never
counts) happens upstream in utils/face_detection.py, not here. This
file only ever sees events that have already been confirmed.
"""

from utils.event_logger import get_event_counts, get_total_event_count


# =====================================
# Camera / vision confidence thresholds
# =====================================
# General minimum confidence for ANY YOLO detection (person, phone,
# laptop, book) to be considered at all.
YOLO_MIN_CONFIDENCE = 0.50

# Person detections specifically need a higher bar than the general
# cutoff above -- a low-confidence "person" box (e.g. a partial/ghost
# detection from motion blur or a reflection) is a common cause of
# false "Multiple Persons" warnings when only one real person is
# present. Raised from 0.75 -> 0.80.
PERSON_DETECTION_MIN_CONFIDENCE = 0.80

# Haar cascade face-detection parameters (unchanged values, just
# named here per this file's own stated design: "thresholds live as
# plain constants ... so they can be tuned without touching detection
# logic"). scaleFactor/minNeighbors/minSize control how aggressively
# the cascade searches -- see cv2.CascadeClassifier.detectMultiScale.
FACE_HAAR_SCALE_FACTOR = 1.1
FACE_HAAR_MIN_NEIGHBORS = 5
FACE_HAAR_MIN_SIZE = (100, 100)

# =====================================
# CONFIGURABLE THRESHOLDS
# =====================================

# How many times a given event type may occur before that specific
# rule is considered "breached" (informational/reporting only -- does
# NOT drive termination; see evaluate_session()).
TAB_SWITCH_LIMIT = 3
FULLSCREEN_EXIT_LIMIT = 3
FACE_ABSENT_LIMIT = 5
MULTIPLE_PERSONS_LIMIT = 2
PHONE_DETECTED_LIMIT = 1
LAPTOP_DETECTED_LIMIT = 1
BOOK_DETECTED_LIMIT = 1
HEAD_TURN_LIMIT = 5

# How long (seconds) the face may be continuously absent before a
# separate, higher-severity "face_absent_prolonged" event is logged.
FACE_ABSENT_DURATION_LIMIT_SECONDS = 120

# =====================================
# Head-turn (looking away) confirm timing
# =====================================
# A single noisy frame, or a normal glance to read the question, must
# NOT log a warning. utils/face_detection.py only logs "head_turn"
# once `direction != "Looking Straight"` has held continuously for at
# least this long.
HEAD_TURN_CONFIRM_SECONDS = 1.5
HEAD_TURN_COOLDOWN_SECONDS = 5.0

# =====================================
# Multiple-persons confirm timing
# =====================================
# Same idea: a momentary double-detection or someone briefly passing
# in the background must not count. utils/face_detection.py only logs
# "multiple_persons" once person_count > 1 has held continuously for
# at least this long, and logs it exactly ONCE per continuous
# incident (not again until the extra person leaves and a NEW
# incident starts).
#
# NOTE: this was previously 30.0 seconds -- far longer than every
# other confirm window in this file (face_absent and head_turn are
# both 1.5s). A second person sitting in frame had to stay for a full
# 30 continuous seconds before ever being logged, which is why a
# clearly-visible, correctly-detected "Multiple Persons" warning could
# show live on screen for a long time without ever reaching
# event_logs / the integrity score. 3.0s is long enough to still
# filter out someone briefly walking through the background, while
# actually catching a second person who sits down and stays.
MULTIPLE_PERSONS_CONFIRM_SECONDS = 3.0
MULTIPLE_PERSONS_COOLDOWN_SECONDS = 5.0

# =====================================
# No-face (face-absent) confirm timing
# =====================================
FACE_ABSENT_CONFIRM_SECONDS = 1.5
FACE_ABSENT_COOLDOWN_SECONDS = 5.0

# =====================================
# Detection stability
# =====================================
MIN_CONSECUTIVE_CONFIRM_FRAMES = 3

# IoU-based duplicate-box merge: two "person" boxes with IoU >= this
# are treated as one physical person.
PERSON_BOX_IOU_MERGE_THRESHOLD = 0.35

# Containment-based duplicate-box merge (NEW): YOLO sometimes emits a
# tight box (e.g. head/shoulders) and a loose box (full body) for the
# SAME real person. Two boxes of very different size/aspect can have
# low IoU despite representing one person -- IoU alone under-merges
# this case. If the smaller box's area is at least this fraction
# contained within the larger box, they're treated as one person too,
# on top of (not instead of) the IoU check above. This is what fixes
# "Multiple Persons" firing from one real person casting two
# differently-sized boxes.
PERSON_BOX_CONTAINMENT_MERGE_THRESHOLD = 0.80


# Per-event-type thresholds used only to compute `breached_rules`
# below, for reporting/summary purposes. NOT a termination trigger.
RULES = {
    "tab_switch": TAB_SWITCH_LIMIT,
    "fullscreen_exit": FULLSCREEN_EXIT_LIMIT,
    "face_absent": FACE_ABSENT_LIMIT,
    "multiple_persons": MULTIPLE_PERSONS_LIMIT,
    "phone_detected": PHONE_DETECTED_LIMIT,
    "laptop_detected": LAPTOP_DETECTED_LIMIT,
    "book_detected": BOOK_DETECTED_LIMIT,
    "head_turn": HEAD_TURN_LIMIT,
    "face_absent_prolonged": 1,
}

# =====================================
# Marks model
# =====================================
# Rule: "Multiple persons" -- deduct this many marks for EVERY
# confirmed multiple_persons violation, no free pass.
MULTIPLE_PERSONS_MARK_PER_VIOLATION = 1

# Rule: "Multiple persons" -- more than this many confirmed
# violations (i.e. on the 4th) terminates the exam immediately.
MULTIPLE_PERSONS_TERMINATE_AFTER = 3

# Rule: "Warnings" -- every OTHER confirmed warning type (face_absent,
# face_absent_prolonged, phone/laptop/book_detected, head_turn,
# fullscreen_exit, tab_switch) is pooled together. The first this-many
# of them deduct no marks at all.
CONFIRMED_WARNING_FREE_COUNT = 3

# Rule: "Warnings" -- every pooled warning AFTER the free allowance
# above deducts this many marks each.
MARKS_PER_WARNING_AFTER_FREE = 1


# =====================================
# RULE EVALUATION
# =====================================

def evaluate_session(session_id):
    """
    Evaluates all suspicious-event rules for a session using the
    unified event_logs table.

    Returns:
        {
            "total_warnings": int,          # unified camera+browser count
            "deducted_marks": int,
            "terminated": bool,
            "termination_reason": str | None,
            "breached_rules": [event_type, ...],
            "event_counts": {event_type: count, ...}
        }

    NOTE on termination: this function only decides termination for
    the multiple_persons rule. Tab-switch termination (3rd switch, or
    >60s continuously away) is decided separately by
    utils/tab_switch_policy.py and layered on top of this result in
    app.py's /update_warning -- exactly as it already was before this
    change, so that logic is untouched here.
    """
    event_counts = get_event_counts(session_id)
    total_warnings = get_total_event_count(session_id)

    breached_rules = [
        event_type
        for event_type, limit in RULES.items()
        if event_counts.get(event_type, 0) >= limit
    ]

    multiple_persons_count = event_counts.get("multiple_persons", 0)

    # Every confirmed warning that ISN'T multiple_persons shares one
    # pooled "first N are free" allowance.
    other_warning_count = total_warnings - multiple_persons_count

    deducted_marks = (
        max(0, other_warning_count - CONFIRMED_WARNING_FREE_COUNT)
        * MARKS_PER_WARNING_AFTER_FREE
    )

    # multiple_persons always deducts its own mark, on top of the
    # pooled amount above, regardless of the free allowance.
    deducted_marks += multiple_persons_count * MULTIPLE_PERSONS_MARK_PER_VIOLATION

    terminated = False
    termination_reason = None

    # "If multiple-person violations occur more than 3 confirmed
    # times, immediately terminate the exam."
    if multiple_persons_count > MULTIPLE_PERSONS_TERMINATE_AFTER:
        terminated = True
        termination_reason = "multiple_persons_repeated"

    # Reaching the pooled free-warning count on its own is explicitly
    # NOT a termination trigger ("Do not terminate the exam merely
    # because 3 warnings occurred unless another termination rule is
    # reached") -- so there is intentionally no check against
    # CONFIRMED_WARNING_FREE_COUNT here.

    return {
        "total_warnings": total_warnings,
        "deducted_marks": deducted_marks,
        "terminated": terminated,
        "termination_reason": termination_reason,
        "breached_rules": breached_rules,
        "event_counts": event_counts,
    }