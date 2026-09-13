"""
Part 13 - one-off migration script for the Data Science & Analytics
Module (analytics.py) upgrade -- PCA + K-Means cohort clustering,
integrity-score distribution charting, and cohort risk profiling.

Run this ONCE after pulling Part 13, the same way
update_database_part11.py / update_database_part12.py were run once
for their respective parts:

    python update_database_part13.py

What it does, in order:

1. Ensures the `sessions` table has a `course` column. analytics.py
   breaks down cohort analytics per exam course, but `sessions` (see
   database.py) was never given a `course` column -- sessions created
   by Parts 1-12 only have session_id/candidate_id/start_time/
   end_time/status. SQLite's `CREATE TABLE IF NOT EXISTS` cannot add a
   column to a table that already exists, so this uses
   `ALTER TABLE ... ADD COLUMN` instead. Existing rows get `course =
   NULL` (analytics.py already treats that as course = "Unknown" in
   cohort_risk_profile()'s per-course breakdown), and does not raise.

2. Ensures the `session_clusters` table exists, so
   analytics.save_session_clusters() has somewhere to persist a
   K-Means run without recomputing it on every dashboard page view.

Both steps are idempotent -- re-running this script after it has
already succeeded is a safe no-op.

Does not touch candidates, event_logs, questions, exam_results,
integrity_scores, or integrity_reports -- Parts 1-12 schema and data,
and utils/scoring.py / utils/detection_rules.py (live proctoring +
integrity scoring), are left exactly as they are.
"""

import sqlite3

from config import DATABASE_PATH


CREATE_SESSION_CLUSTERS_SQL = """
CREATE TABLE IF NOT EXISTS session_clusters(

    cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER UNIQUE,
    candidate_id INTEGER,
    cluster_label INTEGER,
    cluster_name TEXT,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY(session_id)
    REFERENCES sessions(session_id),

    FOREIGN KEY(candidate_id)
    REFERENCES candidates(candidate_id)

)
"""


def _table_exists(cursor, table_name):
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name=?
    """, (table_name,))
    return cursor.fetchone() is not None


def _column_exists(cursor, table_name, column_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def migrate():

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    # ---------------- Step 1: sessions.course column ----------------
    if _column_exists(cursor, "sessions", "course"):
        print("sessions.course column already present -- nothing to do.")
    else:
        cursor.execute("""
            ALTER TABLE sessions
            ADD COLUMN course TEXT
        """)
        connection.commit()
        print("sessions.course column added successfully! Existing "
              "sessions have course = NULL (shown as 'Unknown' in "
              "cohort_risk_profile()'s per-course breakdown).")

    # ---------------- Step 2: session_clusters table ----------------
    table_existed_before = _table_exists(cursor, "session_clusters")

    cursor.execute(CREATE_SESSION_CLUSTERS_SQL)
    connection.commit()

    if table_existed_before:
        print("session_clusters table already existed -- left as is.")
    else:
        print("session_clusters table was missing -- created it.")

    connection.close()
    print("Part 13 migration complete.")


if __name__ == "__main__":
    migrate()