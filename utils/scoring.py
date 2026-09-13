"""
Milestone 3 / Part 11 - Integrity Scoring Module (Pandas-based).

Computes a normalized (0-100) integrity score and a Low/Medium/High
risk label for a completed exam session, based on the SAME event_logs
and sessions data already written by utils/event_logger.py.

This module is intentionally separate from utils/detection_rules.py:
- detection_rules.py drives LIVE behaviour during the exam (unified
  warning count, marks deduction, auto-termination threshold).
- scoring.py produces a POST-HOC integrity score/risk label for
  invigilator review once the session has ended. It does not change
  live warning counts, marks deduction, or termination logic.

No existing table, route, function, or Part 1-10 behaviour is
modified by this file. This Part 11 revision changes HOW the score is
computed (Pandas instead of a plain Python loop) and ADDS a new
`face_presence_ratio` signal; it does not remove or rename any
existing key in the returned dict, and does not change
SEVERITY_WEIGHTS / RISK_THRESHOLDS / the event-based penalty formula
that Milestone 3 originally shipped with.

====================================================================
IMPORTANT: face_presence_ratio IS AN ESTIMATE, NOT A MEASUREMENT
====================================================================
`event_logs` (see utils/event_logger.py / database.py) only stores
ONE timestamp per row -- the moment an event was first confirmed. For
face absence specifically, that means we know WHEN the candidate's
face was first confirmed missing (`face_absent`) and, separately,
whether that same absence later crossed the prolonged threshold
(`face_absent_prolonged`) -- but there is no "face returned" event and
no stored duration for any individual absence episode. It is
therefore impossible to compute an exact "seconds present / seconds
total" ratio from the existing logs alone.

What this module does instead is ESTIMATE total absent time per
session from the two signals we do have:
  - how many confirmed-absence episodes were logged (`face_absent`
    count), each assumed to last about as long as the existing
    confirm/cooldown debounce window already used to log it
    (utils.detection_rules.FACE_ABSENT_COOLDOWN_SECONDS), and
  - how many of those episodes are known to have crossed the
    prolonged threshold (`face_absent_prolonged` count), each of
    which is known to have lasted AT LEAST
    utils.detection_rules.FACE_ABSENT_DURATION_LIMIT_SECONDS, so that
    known-minimum duration is used for those episodes instead of the
    short-episode assumption.

This is a reasonable lower-effort estimate given the current schema,
not a precise measurement. If exact face-presence tracking is needed
later, `event_logs` would need a matching "face_returned" event (or a
duration column) so real intervals could be computed -- that is a
schema change intentionally left out of Part 11's scope.
"""

import sqlite3
from datetime import datetime

import pandas as pd

from config import DATABASE_PATH
from utils.detection_rules import (
    FACE_ABSENT_COOLDOWN_SECONDS,
    FACE_ABSENT_DURATION_LIMIT_SECONDS,
)


# =====================================
# CONFIGURABLE SEVERITY WEIGHTS
# =====================================
# Points deducted from a starting score of 100 for each occurrence of
# an event type. Higher weight = more severe integrity concern.
# Unchanged from the original Milestone 3 implementation.

SEVERITY_WEIGHTS = {
    "tab_switch": 3,
    "fullscreen_exit": 3,
    "face_absent": 2,
    "face_absent_prolonged": 8,
    "multiple_persons": 6,
    "phone_detected": 10,
    "laptop_detected": 8,
    "book_detected": 8,
    "head_turn": 1,
    # Fallback for any event type not listed above (e.g. the generic
    # "browser_violation" label used when exam.html sends no
    # event_type).
    "_default": 2,
}

# Risk label thresholds, based on the final normalized score.
# Unchanged from the original Milestone 3 implementation.
RISK_THRESHOLDS = {
    "Low": 75,      # score >= 75
    "Medium": 50,   # 50 <= score < 75
    # anything below 50 -> "High"
}

# =====================================
# PART 11: FACE PRESENCE RATIO ESTIMATE
# =====================================
# Assumed duration (seconds) of a confirmed face-absent episode that
# did NOT cross the prolonged threshold. Reuses the existing
# FACE_ABSENT_COOLDOWN_SECONDS from detection_rules.py (the window
# during which a further face_absent warning is suppressed) as the
# stand-in for "about how long one such episode lasts", instead of
# inventing an unrelated new constant.
ASSUMED_SHORT_ABSENCE_SECONDS = FACE_ABSENT_COOLDOWN_SECONDS

# Maximum points deducted from the integrity score for a session where
# the face was estimated absent 100% of the time. Scales linearly down
# to 0 points at a face_presence_ratio of 1.0 (always present).
# Kept as a separate, smaller weight than the event-based penalties
# above so that ONE noisy/estimated signal cannot by itself dominate
# the score -- it adjusts the existing event-based score rather than
# replacing it.
FACE_PRESENCE_PENALTY_WEIGHT = 20


def _get_connection():
    return sqlite3.connect(DATABASE_PATH)


# =====================================
# PANDAS DATA LOADING
# =====================================

def _load_event_log_dataframe(connection, session_id):
    """
    Returns a pandas DataFrame of every event_logs row for this
    session, one row per logged event (columns: event_type,
    timestamp). Empty DataFrame (not an error) if the session has no
    events.
    """
    return pd.read_sql_query(
        """
        SELECT event_type, timestamp
        FROM event_logs
        WHERE session_id = ?
        """,
        connection,
        params=(session_id,),
    )


def _load_session_row(connection, session_id):
    """
    Returns a pandas DataFrame (0 or 1 rows) with this session's
    start_time/end_time/status.
    """
    return pd.read_sql_query(
        """
        SELECT start_time, end_time, status
        FROM sessions
        WHERE session_id = ?
        """,
        connection,
        params=(session_id,),
    )


# =====================================
# WEIGHTED EVENT SCORING (Pandas)
# =====================================

def _weighted_event_penalty(events_df):
    """
    Row-level weighted penalty using Pandas: every logged event is
    mapped to its severity weight (falling back to "_default" for any
    unrecognized event_type), then summed. This is mathematically
    identical to the original "weight * count, summed per event type"
    approach -- mapping the weight onto every row and summing achieves
    the same total as summing (weight * count) per distinct type --
    just expressed as a Pandas column operation instead of a manual
    Python loop.
    """
    if events_df.empty:
        return 0.0

    weights = events_df["event_type"].map(SEVERITY_WEIGHTS)
    weights = weights.fillna(SEVERITY_WEIGHTS["_default"])

    return float(weights.sum())


def _event_counts(events_df):
    """
    {event_type: count} for every distinct event type, via Pandas
    value_counts(). Same shape/values as the original
    event_logger.get_event_counts() helper.
    """
    if events_df.empty:
        return {}

    return events_df["event_type"].value_counts().to_dict()


# =====================================
# FACE PRESENCE RATIO (Pandas) -- ESTIMATE, see module docstring
# =====================================

def _face_presence_ratio(events_df, session_row_df):
    """
    Returns an ESTIMATED face_presence_ratio in [0.0, 1.0], or None if
    it cannot be estimated (session row missing, or session duration
    is zero/unknown).

    See the module docstring for exactly why this is an estimate
    rather than a measurement.
    """
    if session_row_df.empty:
        return None

    start_raw = session_row_df.at[0, "start_time"]
    end_raw = session_row_df.at[0, "end_time"]

    if not start_raw:
        return None

    start_time = pd.to_datetime(start_raw, errors="coerce")

    # A session that hasn't ended yet (end_time is NULL) has no fixed
    # duration to divide by -- fall back to "now" so an in-progress
    # session can still get a live estimate (e.g. for a future
    # dashboard), documented here rather than silently guessed.
    if end_raw:
        end_time = pd.to_datetime(end_raw, errors="coerce")
    else:
        end_time = pd.Timestamp(datetime.now())

    if pd.isna(start_time) or pd.isna(end_time):
        return None

    session_duration_seconds = (end_time - start_time).total_seconds()

    if session_duration_seconds <= 0:
        return None

    if events_df.empty:
        prolonged_count = 0
        short_count = 0
    else:
        is_prolonged = events_df["event_type"] == "face_absent_prolonged"
        is_short_absence = events_df["event_type"] == "face_absent"

        prolonged_count = int(is_prolonged.sum())
        # Every face_absent_prolonged event corresponds to an episode
        # that ALSO logged an initial face_absent event -- don't count
        # that episode's duration twice. Treat only the remaining
        # face_absent episodes as "short".
        short_count = max(0, int(is_short_absence.sum()) - prolonged_count)

    estimated_absent_seconds = (
        short_count * ASSUMED_SHORT_ABSENCE_SECONDS
        + prolonged_count * FACE_ABSENT_DURATION_LIMIT_SECONDS
    )

    ratio = 1 - (estimated_absent_seconds / session_duration_seconds)

    # Clamp to [0, 1] -- the estimate can exceed the session length
    # for a very short session with several logged episodes.
    ratio = max(0.0, min(1.0, ratio))

    return round(float(ratio), 4)


def _risk_label_for(score):
    if score >= RISK_THRESHOLDS["Low"]:
        return "Low"
    if score >= RISK_THRESHOLDS["Medium"]:
        return "Medium"
    return "High"


# =====================================
# SCORE COMPUTATION
# =====================================

def calculate_integrity_score(session_id):
    """
    Reads event counts + session duration for a session and computes:
        - integrity_score: float, clamped to [0, 100]
        - risk_label: "Low" | "Medium" | "High"
        - total_events: int
        - event_counts: {event_type: count}
        - face_presence_ratio: float in [0, 1], or None if it could
          not be estimated (see module docstring -- this is an
          ESTIMATE, not a measurement)

    All keys present in the original Milestone 3 return value
    (integrity_score, risk_label, total_events, event_counts) are
    still present with the same meaning -- face_presence_ratio is an
    ADDITIONAL key, not a replacement for any of them.

    Pure calculation only -- does not touch the database beyond a
    read-only connection used for the Pandas queries.
    """
    if session_id is None:
        return {
            "integrity_score": 100.0,
            "risk_label": "Low",
            "total_events": 0,
            "event_counts": {},
            "face_presence_ratio": None,
        }

    connection = _get_connection()

    try:
        events_df = _load_event_log_dataframe(connection, session_id)
        session_row_df = _load_session_row(connection, session_id)
    finally:
        connection.close()

    event_penalty = _weighted_event_penalty(events_df)
    face_presence_ratio = _face_presence_ratio(events_df, session_row_df)

    # Part 11: face presence factored into the score, per the project
    # requirements ("...computes a per-session integrity score based
    # on event frequency, severity weights, and face presence ratio").
    # Kept as a bounded, separate penalty on top of the unchanged
    # event-based penalty above, rather than folded into
    # SEVERITY_WEIGHTS, so the event-only scoring behaviour Milestone
    # 3 originally shipped with stays intact and easy to reason about.
    # If the ratio couldn't be estimated, no penalty is applied for
    # what wasn't measured.
    if face_presence_ratio is None:
        face_presence_penalty = 0.0
    else:
        face_presence_penalty = (1 - face_presence_ratio) * FACE_PRESENCE_PENALTY_WEIGHT

    total_penalty = event_penalty + face_presence_penalty

    integrity_score = max(0.0, min(100.0, 100.0 - total_penalty))
    risk_label = _risk_label_for(integrity_score)

    return {
        "integrity_score": round(integrity_score, 2),
        "risk_label": risk_label,
        "total_events": int(len(events_df)),
        "event_counts": _event_counts(events_df),
        "face_presence_ratio": face_presence_ratio,
    }


# =====================================
# PERSISTENCE
# =====================================

def save_integrity_score(session_id, candidate_id):
    """
    Computes and stores (or updates) the integrity score row for a
    session. Safe to call more than once for the same session_id --
    e.g. if submit_exam runs after the rule engine already terminated
    the session -- since session_id is UNIQUE in integrity_scores.

    Returns the computed result dict (see calculate_integrity_score).

    Requires the Part 11 `face_presence_ratio` column on
    integrity_scores (run update_database_part11.py once). If that
    migration hasn't been run yet, falls back to the pre-Part-11
    INSERT (without face_presence_ratio) so this function doesn't hard
    -crash an unmigrated database -- the computed
    face_presence_ratio is still returned to the caller either way,
    just not persisted until the migration runs.
    """
    result = calculate_integrity_score(session_id)

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    try:
        cursor.execute("""
            INSERT INTO integrity_scores
                (session_id, candidate_id, integrity_score, risk_label,
                 total_events, face_presence_ratio)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                integrity_score=excluded.integrity_score,
                risk_label=excluded.risk_label,
                total_events=excluded.total_events,
                face_presence_ratio=excluded.face_presence_ratio,
                computed_at=CURRENT_TIMESTAMP
        """, (
            session_id,
            candidate_id,
            result["integrity_score"],
            result["risk_label"],
            result["total_events"],
            result["face_presence_ratio"],
        ))

    except sqlite3.OperationalError as error:
        if "face_presence_ratio" not in str(error):
            raise

        print(
            "WARNING: integrity_scores.face_presence_ratio column is "
            "missing -- run `python update_database_part11.py` once. "
            "Saving without it for now."
        )

        cursor.execute("""
            INSERT INTO integrity_scores
                (session_id, candidate_id, integrity_score, risk_label,
                 total_events)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                integrity_score=excluded.integrity_score,
                risk_label=excluded.risk_label,
                total_events=excluded.total_events,
                computed_at=CURRENT_TIMESTAMP
        """, (
            session_id,
            candidate_id,
            result["integrity_score"],
            result["risk_label"],
            result["total_events"],
        ))

    connection.commit()
    connection.close()

    return result


def get_integrity_score(session_id):
    """
    Returns the stored integrity score row for a session as a dict, or
    None if no score has been computed yet.

    Same keys as before (integrity_score, risk_label, total_events,
    computed_at) plus face_presence_ratio. If the database hasn't been
    migrated yet (Part 11 column missing), face_presence_ratio comes
    back as None instead of raising.
    """
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    try:
        cursor.execute("""
            SELECT integrity_score, risk_label, total_events, computed_at,
                   face_presence_ratio
            FROM integrity_scores
            WHERE session_id=?
        """, (session_id,))
        row = cursor.fetchone()
        has_face_presence_column = True

    except sqlite3.OperationalError as error:
        if "face_presence_ratio" not in str(error):
            connection.close()
            raise

        cursor.execute("""
            SELECT integrity_score, risk_label, total_events, computed_at
            FROM integrity_scores
            WHERE session_id=?
        """, (session_id,))
        row = cursor.fetchone()
        has_face_presence_column = False

    connection.close()

    if not row:
        return None

    return {
        "integrity_score": row[0],
        "risk_label": row[1],
        "total_events": row[2],
        "computed_at": row[3],
        "face_presence_ratio": row[4] if has_face_presence_column else None,
    }