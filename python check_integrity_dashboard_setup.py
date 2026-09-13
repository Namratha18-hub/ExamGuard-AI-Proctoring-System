"""
Milestone 3 dashboard - setup checker.

If /dashboard/integrity "isn't displaying anywhere", run this from the
project folder to find out exactly which piece is missing:

    python check_integrity_dashboard_setup.py

Checks, in order:
  1. sessions.course column + session_clusters table exist (i.e.
     update_database_part13.py has actually been run against THIS
     database file).
  2. scikit-learn is importable (needed by analytics.cluster_sessions()).
  3. analytics.py has session_detail() defined.
  4. app.py registers the /dashboard/integrity route.
  5. templates/dashboard_base.html links to /dashboard/integrity (the
     sidebar nav item) -- if this is missing, the route works fine but
     there's simply nothing on any page to click to reach it.
  6. templates/integrity_dashboard.html exists.
  7. static/css/integrity_dashboard.css exists.

Prints a clear PASS/FAIL per check plus what to do about any FAIL,
then exits non-zero if anything failed.
"""

import os
import sqlite3
import sys

CHECK_MARK = "OK  "
CROSS_MARK = "FAIL"

results = []


def check(label, passed, fix_hint=""):
    results.append(passed)
    status = CHECK_MARK if passed else CROSS_MARK
    print(f"[{status}] {label}")
    if not passed and fix_hint:
        print(f"       -> {fix_hint}")


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))

    # ---- 1. database migration ----
    try:
        from config import DATABASE_PATH
        connection = sqlite3.connect(DATABASE_PATH)
        cursor = connection.cursor()

        cursor.execute("PRAGMA table_info(sessions)")
        columns = [row[1] for row in cursor.fetchall()]
        check(
            "sessions.course column exists",
            "course" in columns,
            "Run: python update_database_part13.py",
        )

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='session_clusters'
        """)
        check(
            "session_clusters table exists",
            cursor.fetchone() is not None,
            "Run: python update_database_part13.py",
        )

        connection.close()

    except Exception as error:
        check("database reachable / migrated", False, f"Error: {error}")

    # ---- 2. scikit-learn ----
    try:
        import sklearn  # noqa: F401
        check("scikit-learn is installed", True)
    except ImportError:
        check(
            "scikit-learn is installed", False,
            "Run: pip install scikit-learn --break-system-packages "
            "(or without that flag if not using system Python)",
        )

    # ---- 3. analytics.session_detail exists ----
    try:
        import analytics
        check(
            "analytics.session_detail() is defined",
            hasattr(analytics, "session_detail"),
            "analytics.py is missing session_detail() -- make sure you "
            "copied the latest analytics.py into the project root.",
        )
    except Exception as error:
        check("analytics.py imports cleanly", False, f"Error: {error}")

    # ---- 4. route registered in app.py ----
    app_py_path = os.path.join(project_root, "app.py")
    app_py_has_route = False
    if os.path.exists(app_py_path):
        with open(app_py_path, "r", encoding="utf-8", errors="ignore") as f:
            app_py_source = f.read()
        app_py_has_route = '"/dashboard/integrity"' in app_py_source
    check(
        '/dashboard/integrity route is registered in app.py',
        app_py_has_route,
        "app.py doesn't contain the integrity_dashboard() route -- make "
        "sure you replaced app.py with the version that adds it.",
    )

    # ---- 5. sidebar link in dashboard_base.html ----
    base_path = os.path.join(project_root, "templates", "dashboard_base.html")
    base_has_link = False
    if os.path.exists(base_path):
        with open(base_path, "r", encoding="utf-8", errors="ignore") as f:
            base_has_link = "/dashboard/integrity" in f.read()
    check(
        "templates/dashboard_base.html links to /dashboard/integrity",
        base_has_link,
        "The route works but there's no sidebar link to reach it -- "
        "replace templates/dashboard_base.html with the updated version "
        "(adds an 'Integrity Analytics' nav item).",
    )

    # ---- 6. template file ----
    check(
        "templates/integrity_dashboard.html exists",
        os.path.exists(os.path.join(project_root, "templates", "integrity_dashboard.html")),
        "Copy integrity_dashboard.html into the templates/ folder.",
    )

    # ---- 7. CSS file ----
    check(
        "static/css/integrity_dashboard.css exists",
        os.path.exists(os.path.join(project_root, "static", "css", "integrity_dashboard.css")),
        "Copy integrity_dashboard.css into static/css/.",
    )

    print()
    if all(results):
        print("All checks passed. If it still doesn't show up:")
        print("  1. Make sure the Flask process was fully RESTARTED after")
        print("     copying in these files (not just the browser refreshed).")
        print("  2. Log in, then visit /dashboard/integrity directly in the")
        print("     browser to rule out a sidebar caching issue.")
        print("  3. Check the Flask terminal output for a traceback when")
        print("     you load the page.")
        sys.exit(0)
    else:
        print(f"{results.count(False)} check(s) failed -- fix those first, "
              f"then re-run this script.")
        sys.exit(1)


if __name__ == "__main__":
    main()