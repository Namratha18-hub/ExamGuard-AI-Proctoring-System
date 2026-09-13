"""
Milestone 3 dashboard fix - manual smoke test for
analytics.session_detail() and the /dashboard/integrity route.

Run:
    python test_session_dashboard.py [session_id]

If no session_id is given, tests against the first session found in
`integrity_scores`. Prints every field session_detail() returns and
saves its three charts to ./session_dashboard_preview/ so they can be
opened directly. All values/images come straight from the live
database via analytics.py -- nothing here is mocked, cached, or
hardcoded.
"""

import base64
import os
import sqlite3
import sys

from config import DATABASE_PATH
import analytics

OUTPUT_DIR = "session_dashboard_preview"


def _save_image(image_base64, filename):
    if not image_base64:
        print(f"  (no image for {filename})")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(base64.b64decode(image_base64))
    print(f"  saved {path}")


def _pick_session_id():
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()
    cursor.execute("SELECT session_id FROM integrity_scores ORDER BY session_id LIMIT 1")
    row = cursor.fetchone()
    connection.close()
    return row[0] if row else None


def main():
    if len(sys.argv) > 1:
        session_id = int(sys.argv[1])
    else:
        session_id = _pick_session_id()

    if session_id is None:
        print("No scored sessions found in the database yet -- nothing to test.")
        return

    print(f"=== session_detail({session_id}) ===")
    detail = analytics.session_detail(session_id)

    if not detail["found"]:
        print(f"Session {session_id} does not exist.")
        return

    print(f"candidate: {detail['candidate_name']} ({detail['application_id']})")
    print(f"course: {detail['course']}  status: {detail['status']}")
    print(f"integrity_score: {detail['integrity_score']}  risk_label: {detail['risk_label']}")
    print(f"face_presence_ratio: {detail['face_presence_ratio']}")
    print(f"total_events: {detail['total_events']}  deducted_marks: {detail['deducted_marks']}")
    print(f"breached_rules: {detail['breached_rules']}")
    print("event_category_counts:")
    for label, count in detail["event_category_counts"].items():
        print(f"    {label}: {count}")
    print(f"cluster_name: {detail['cluster_name']}  cluster_label: {detail['cluster_label']}")
    print(f"cluster_message: {detail['cluster_message']}")
    print(f"pca_explained_variance_ratio: {detail['pca_explained_variance_ratio']}")

    _save_image(detail["score_gauge_image_base64"], f"session_{session_id}_gauge.png")
    _save_image(detail["event_bar_image_base64"], f"session_{session_id}_event_bar.png")
    _save_image(detail["pca_cluster_image_base64"], f"session_{session_id}_pca.png")

    print(f"\nDone. Chart PNGs (if any) are under ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
    