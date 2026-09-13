"""
Part 12 - one-off migration script for the AI Integrity Report Agent
(utils/report_agent.py).

Run this ONCE after pulling Part 12, the same way
update_database_part11.py was run once for integrity_scores /
face_presence_ratio:

    python update_database_part12.py

What it does:

1. Ensures the `integrity_reports` table exists. A database created
   before Part 12 (including one that already has Part 11's
   `integrity_scores` table) will not have this table yet.

That's the only step needed for Part 12 -- unlike Part 11, this is a
brand-new table with no prior version to add a column to, so there is
no ALTER TABLE step here. The script still checks first and reports
what it did/skipped, and is safe to re-run.

Does not touch candidates, sessions, event_logs, questions,
exam_results, or integrity_scores -- Parts 1-11 schema and data are
left exactly as they are.
"""

import sqlite3

from config import DATABASE_PATH


CREATE_INTEGRITY_REPORTS_SQL = """
CREATE TABLE IF NOT EXISTS integrity_reports(

    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER UNIQUE,
    candidate_id INTEGER,
    integrity_score REAL,
    risk_label TEXT,
    summary_text TEXT NOT NULL,
    generation_method TEXT NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

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


def migrate():

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    table_existed_before = _table_exists(cursor, "integrity_reports")

    cursor.execute(CREATE_INTEGRITY_REPORTS_SQL)
    connection.commit()

    if table_existed_before:
        print("integrity_reports table already existed -- left as is.")
    else:
        print("integrity_reports table was missing -- created it.")

    connection.close()
    print("Part 12 migration complete.")


if __name__ == "__main__":
    migrate()