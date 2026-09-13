"""
Unit tests for the false-positive proctoring fix in
utils/face_detection.py -- specifically:
  - _ConfirmGate (the shared debounce/cooldown state machine)
  - _merge_person_boxes / _compute_containment_ratio (duplicate-person
    box merging)

These test the actual logic classes/functions imported straight from
utils/face_detection.py -- not reimplementations -- so a pass here
means the real code behaves correctly. No camera, YOLO weights, or
mediapipe model files are needed; ultralytics.YOLO and
utils.head_pose.get_head_direction are stubbed purely so the module
can be imported (their real bodies are never called).

Run:
    python test_proctoring_fix.py
"""

import sys
import types


def _install_stubs():
    """
    utils/face_detection.py imports `from ultralytics import YOLO` and
    `from utils.head_pose import get_head_direction` at module level.
    Neither is actually invoked by anything this test calls (we only
    test _ConfirmGate / _merge_person_boxes / _compute_containment_ratio,
    none of which touch the camera, YOLO, or mediapipe), so stub both
    just enough to satisfy the import.
    """
    if "ultralytics" not in sys.modules:
        fake_ultralytics = types.ModuleType("ultralytics")

        class _FakeYOLO:
            def __init__(self, *a, **kw):
                self.names = {}

        fake_ultralytics.YOLO = _FakeYOLO
        sys.modules["ultralytics"] = fake_ultralytics

    if "utils.head_pose" not in sys.modules:
        fake_head_pose = types.ModuleType("utils.head_pose")
        fake_head_pose.get_head_direction = lambda *a, **kw: "Looking Straight"
        sys.modules["utils.head_pose"] = fake_head_pose


_install_stubs()

from utils.face_detection import (  # noqa: E402
    _ConfirmGate,
    _merge_person_boxes,
    _compute_containment_ratio,
)


passed = 0
failed = 0


def check(label, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"[OK  ] {label}")
    else:
        failed += 1
        print(f"[FAIL] {label}")


# =====================================
# 1. face_absent-style gate: THE reported bug
# =====================================
# Simulates the exact real-world scenario from the bug report: a face
# that is detected almost every frame, with occasional single-frame
# misses (a blink, a flicker) -- but is NEVER actually gone for a
# sustained period. The OLD code logged an event on every single
# miss. The fix must log ZERO events for this pattern.

print("\n=== face_absent-style gate: flickery-but-present face ===")

gate = _ConfirmGate(confirm_seconds=1.5, min_consecutive_frames=3, cooldown_seconds=5.0)

# Simulate 40 frames, ~0.05s apart (~20fps), where the face is
# "absent" (Haar miss) on isolated single frames only -- never for
# more than one frame in a row, and each miss is followed by several
# confirmed-present frames (well under both the 1.5s confirm window
# and comfortably recovering before any cooldown could even start).
import time as _time

logged_events = 0
real_time_start = _time.time()

# We can't literally sleep 40*0.05s honestly in a fast test and still
# keep this snappy, so drive the gate directly against synthetic
# timestamps by monkeypatching time.time() for this block only.
_original_time = _time.time
_fake_now = [1000.0]
_time.time = lambda: _fake_now[0]

try:
    face_present_pattern = (
        [True] * 5 + [False] + [True] * 5 + [False] + [True] * 5
        + [False] + [True] * 5 + [False] + [True] * 5 + [False] + [True] * 5
    )
    for face_detected in face_present_pattern:
        _fake_now[0] += 0.05  # ~20fps
        face_absent_now = not face_detected
        if gate.update(face_absent_now):
            logged_events += 1

    check(
        "zero face_absent events logged for a flickery-but-present face "
        f"(old code would have logged one per miss; got {logged_events})",
        logged_events == 0,
    )

    # Now simulate a REAL, sustained absence: face genuinely gone for
    # 2 full seconds (well past the 1.5s confirm window).
    gate.reset()
    logged_events = 0
    for _ in range(60):  # 60 frames * 0.05s = 3.0s of continuous absence
        _fake_now[0] += 0.05
        if gate.update(True):
            logged_events += 1

    check(
        f"exactly one face_absent event logged for a genuine 3s absence (got {logged_events})",
        logged_events == 1,
    )

    # And it must NOT log again while the absence continues (only once
    # per incident).
    for _ in range(20):
        _fake_now[0] += 0.05
        gate.update(True)

    check("gate.confirmed stays True through the rest of the same incident", gate.confirmed)

    # Face comes back and STAYS back past the cooldown -> a fresh,
    # later sustained absence must be able to log again.
    for _ in range(200):  # 200 * 0.05s = 10s, past the 5.0s cooldown
        _fake_now[0] += 0.05
        gate.update(False)

    check("gate fully resets after cooldown elapses", gate.since is None and not gate.confirmed)

    logged_events = 0
    for _ in range(60):
        _fake_now[0] += 0.05
        if gate.update(True):
            logged_events += 1

    check(f"a SECOND genuine sustained absence logs again (got {logged_events})", logged_events == 1)

finally:
    _time.time = _original_time


# =====================================
# 1b. multiple_persons-style gate: real-world confirm timing
# =====================================
# Regression test for the SECOND bug found from the live-exam
# screenshot: MULTIPLE_PERSONS_CONFIRM_SECONDS was 30.0s -- so a
# second person who sat in frame the whole time still wouldn't be
# logged for a full 30 seconds, while the on-screen warning already
# showed. Uses the ACTUAL constant from detection_rules.py (not a
# hand-picked test value) so this test fails again if the timing ever
# regresses back to something unreasonably long.
print("\n=== multiple_persons gate: confirms within a realistic timeframe ===")

from utils.detection_rules import MULTIPLE_PERSONS_CONFIRM_SECONDS  # noqa: E402

check(
    f"MULTIPLE_PERSONS_CONFIRM_SECONDS is reasonable, not 30s (got {MULTIPLE_PERSONS_CONFIRM_SECONDS}s)",
    MULTIPLE_PERSONS_CONFIRM_SECONDS <= 10.0,
)

_fake_now = [3000.0]
_time.time = lambda: _fake_now[0]

try:
    mp_gate = _ConfirmGate(
        confirm_seconds=MULTIPLE_PERSONS_CONFIRM_SECONDS,
        min_consecutive_frames=3,
        cooldown_seconds=5.0,
    )

    # A second person sits down and stays continuously in frame for
    # 5 real seconds (comfortably longer than any reasonable confirm
    # window, but far short of the old 30s bug).
    logged_events = 0
    for _ in range(100):  # 100 * 0.05s = 5.0s of continuous "2 people"
        _fake_now[0] += 0.05
        if mp_gate.update(True):
            logged_events += 1

    check(
        f"a second person present for 5 continuous seconds DOES get logged (got {logged_events} event(s))",
        logged_events == 1,
    )

    # And a momentary background walker (well under the confirm
    # window) must still NOT be logged.
    mp_gate.reset()
    logged_events = 0
    for _ in range(10):  # 10 * 0.05s = 0.5s -- a quick pass-through
        _fake_now[0] += 0.05
        if mp_gate.update(True):
            logged_events += 1
    for _ in range(20):
        _fake_now[0] += 0.05
        mp_gate.update(False)

    check(
        f"a momentary 0.5s background pass-through is still NOT logged (got {logged_events} event(s))",
        logged_events == 0,
    )

finally:
    _time.time = _original_time


# =====================================
# 2. Cooldown prevents boundary-flicker re-trigger
# =====================================
print("\n=== cooldown prevents immediate re-confirmation from boundary flicker ===")

_fake_now = [2000.0]
_time.time = lambda: _fake_now[0]

try:
    gate2 = _ConfirmGate(confirm_seconds=1.0, min_consecutive_frames=2, cooldown_seconds=3.0)

    logged_events = 0

    # Confirm an incident (2s of continuous True).
    for _ in range(40):
        _fake_now[0] += 0.05
        if gate2.update(True):
            logged_events += 1

    check("first incident confirms exactly once", logged_events == 1)

    # Condition clears for less than a second (well under the 3.0s
    # cooldown), then goes True again -- WITHOUT the cooldown fix,
    # this would immediately start a new confirm cycle from a
    # coincidentally-already-satisfied consistent_frames count. With
    # the fix, the incident must not even be allowed to RESET yet, so
    # the still-ongoing incident stays "confirmed" and does not log a
    # second time here.
    for _ in range(10):  # 0.5s of recovery
        _fake_now[0] += 0.05
        gate2.update(False)

    for _ in range(10):
        _fake_now[0] += 0.05
        if gate2.update(True):
            logged_events += 1

    check(
        f"brief recovery under the cooldown does NOT let a second event log (total so far: {logged_events})",
        logged_events == 1,
    )

finally:
    _time.time = _original_time


# =====================================
# 3. Duplicate person-box merging (tight box + loose box, one real person)
# =====================================
print("\n=== duplicate person-box merging ===")

# Two boxes for the SAME person: a tight head/shoulders box and a
# much larger loose full-body box. Their IoU is low (union dominated
# by the big box) but the small box is almost entirely INSIDE the big
# one -- this is exactly the case IoU-only merging misses.
tight_box = (100, 50, 200, 150, 0.91)   # small, high-confidence
loose_box = (80, 40, 260, 400, 0.83)    # large, contains the tight box

boxes_one_person = [tight_box, loose_box]

merged = _merge_person_boxes(boxes_one_person, iou_threshold=0.35, containment_threshold=0.80)

check(
    f"tight+loose box pair for ONE real person merges down to 1 (got {len(merged)})",
    len(merged) == 1,
)

containment = _compute_containment_ratio(tight_box[:4], loose_box[:4])
check(
    f"containment ratio for the nested pair is high (got {containment:.2f})",
    containment >= 0.80,
)

# Two genuinely SEPARATE people, side by side, non-overlapping -- must
# NOT be merged.
person_a = (50, 50, 150, 300, 0.90)
person_b = (400, 50, 500, 300, 0.88)

boxes_two_people = [person_a, person_b]
merged_two = _merge_person_boxes(boxes_two_people, iou_threshold=0.35, containment_threshold=0.80)

check(
    f"two genuinely separate, non-overlapping people stay as 2 (got {len(merged_two)})",
    len(merged_two) == 2,
)


# =====================================
# Summary
# =====================================
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)