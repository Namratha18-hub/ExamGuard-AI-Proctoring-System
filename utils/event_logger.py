"""
Milestone 2 - Central SQLite integration point for exam sessions and
proctoring events.

Both the camera/AI proctoring loop (utils/face_detection.py) and the
browser-side violation handler (app.py -> /update_warning) call into
this module so that ALL violations -- camera-triggered or browser-
triggered -- end up as rows in the SAME `event_logs` table, tied to the
SAME `session_id`. This is what makes the unified warning count
possible.

No schema changes are required: this module only reads/writes the
existing `sessions` and `event_logs` tables created in database.py.
"""

import sqlite3

from config import DATABASE_PATH


def _get_connection():
    return sqlite3.connect(DATABASE_PATH)


# =====================================
# SESSION LIFECYCLE
# =====================================

def create_session(candidate_id, course=None):
    """
    Creates a new row in `sessions` for the given candidate and returns
    the new session_id. Called once, when a candidate opens an exam.

    course (Part 13): which exam course this session is for -- purely
    descriptive, used by utils/analytics.py to break cohort analytics
    down per course. Optional/defaults to None so any other caller of
    create_session() doesn't need updating.
    """
    connection = _get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO sessions (candidate_id, course, start_time, status)
        VALUES (?, ?, CURRENT_TIMESTAMP, ?)
    """, (candidate_id, course, "in_progress"))

    session_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return session_id


def end_session(session_id, status="completed"):
    """
    Marks a session as finished (status = "completed" or "terminated")
    and stamps end_time. Safe to call more than once; if the session is
    already ended this simply overwrites status/end_time again.
    """
    if session_id is None:
        return

    connection = _get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        UPDATE sessions
        SET status=?, end_time=CURRENT_TIMESTAMP
        WHERE session_id=?
    """, (status, session_id))

    connection.commit()
    connection.close()


def get_session_status(session_id):
    """
    Returns the current status string for a session, or None if the
    session_id is missing / not found.
    """
    if session_id is None:
        return None

    connection = _get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT status FROM sessions WHERE session_id=?
    """, (session_id,))

    row = cursor.fetchone()
    connection.close()

    return row[0] if row else None


# =====================================
# EVENT LOGGING
# =====================================

def log_event(session_id, event_type):
    """
    Inserts one row into `event_logs` for the given session.

    Returns True if the event was logged, False if it was skipped
    (session_id is None -- e.g. the camera stream is running outside of
    an active exam session, such as during registration's face
    capture, which reuses the same /video_feed route).
    """
    if session_id is None:
        return False

    connection = _get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO event_logs (session_id, event_type, timestamp)
        VALUES (?, ?, CURRENT_TIMESTAMP)
    """, (session_id, event_type))

    connection.commit()
    connection.close()

    return True


def get_event_counts(session_id):
    """
    Returns {event_type: count} for every distinct event type logged
    against this session.
    """
    if session_id is None:
        return {}

    connection = _get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT event_type, COUNT(*)
        FROM event_logs
        WHERE session_id=?
        GROUP BY event_type
    """, (session_id,))

    rows = cursor.fetchall()
    connection.close()

    return {event_type: count for event_type, count in rows}


def get_total_event_count(session_id):
    """
    Returns the total number of logged events (all types combined) for
    a session. This is the single, unified warning count used by both
    camera-side and browser-side violations.
    """
    if session_id is None:
        return 0

    connection = _get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*) FROM event_logs WHERE session_id=?
    """, (session_id,))

    count = cursor.fetchone()[0]
    connection.close()

    return count