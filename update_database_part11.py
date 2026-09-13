"""
Part 11 - one-off migration script for the Integrity Scoring Module
(utils/scoring.py) upgrade.

Run this ONCE after pulling Part 11, the same way update_database.py
(Part 1) was run once for the photo_path column:

    python update_database_part11.py

What it does, in order:

1. Ensures the `integrity_scores` table exists at all. Some databases
   (e.g. a fresh clone that never had Part 3's database.py run against
   it, or a Milestone 2-only database) may not have this table yet --
   confirmed by inspecting the actual project database, where
   `integrity_scores` was missing entirely, which is what caused
   `sqlite3.OperationalError: no such table: integrity_scores`.

2. Ensures the `face_presence_ratio` column exists on
   `integrity_scores`. SQLite's `CREATE TABLE IF NOT EXISTS` in
   database.py only helps a brand-new database -- it does NOT add a
   column to a table that already exists from before Part 11. This
   script adds it with `ALTER TABLE ... ADD COLUMN` instead, which
   IS how existing SQLite tables gain a new column.

Both steps are idempotent -- re-running this script after it has
already succeeded is a safe no-op, so it can be run again without
checking state by hand first.

Does not touch candidates, sessions, event_logs, questions, or
exam_results -- Parts 1-10 schema and data are left exactly as they
are.
"""

import sqlite3

from config import DATABASE_PATH


# =====================================
# STEP 1: ENSURE THE TABLE EXISTS
# =====================================
# Identical definition to database.py's create_database() so a
# database that never had this table gets the exact same schema a
# fresh install would get -- including the Part 11 column, so step 2
# below becomes a no-op for a table just created here.

CREATE_INTEGRITY_SCORES_SQL = """
CREATE TABLE IF NOT EXISTS integrity_scores(

    score_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER UNIQUE,
    candidate_id INTEGER,
    integrity_score REAL NOT NULL,
    risk_label TEXT NOT NULL,
    total_events INTEGER DEFAULT 0,
    face_presence_ratio REAL,
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

    # ---------------- Step 1: table ----------------
    table_existed_before = _table_exists(cursor, "integrity_scores")

    cursor.execute(CREATE_INTEGRITY_SCORES_SQL)
    connection.commit()

    if table_existed_before:
        print("integrity_scores table already existed -- left as is.")
    else:
        print("integrity_scores table was missing -- created it "
              "(including face_presence_ratio).")

    # ---------------- Step 2: column ----------------
    # Only needed if the table already existed from before Part 11 --
    # a table created fresh by Step 1 above already has the column.
    if table_existed_before:

        if _column_exists(cursor, "integrity_scores", "face_presence_ratio"):
            print("face_presence_ratio column already present -- "
                  "nothing to do.")
        else:
            cursor.execute("""
                ALTER TABLE integrity_scores
                ADD COLUMN face_presence_ratio REAL
            """)
            connection.commit()
            print("face_presence_ratio column added successfully!")

    connection.close()
    print("Part 11 migration complete.")


if __name__ == "__main__":
    migrate()