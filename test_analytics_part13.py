"""
Part 13 - manual smoke test for the PCA + K-Means analytics upgrade.

Run after `python update_database_part13.py` has been run once:

    python test_analytics_part13.py

Does not modify the database except via analytics.save_session_clusters()
(an upsert into session_clusters, same as the dashboard would trigger).
Prints a summary of every analytics.py function to the console and
writes each generated chart to disk under ./analytics_preview/ so they
can be opened and eyeballed directly.
"""

import base64
import os

import analytics

OUTPUT_DIR = "analytics_preview"


def _save_image(image_base64, filename):
    if not image_base64:
        print(f"  (no image for {filename} -- not enough data yet)")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(base64.b64decode(image_base64))
    print(f"  saved {path}")


def main():
    print("=== 1. Integrity score distribution ===")
    dist = analytics.score_distribution()
    print(f"  session_count={dist['session_count']} mean={dist['mean']} "
          f"median={dist['median']} std_dev={dist['std_dev']} "
          f"min={dist['min']} max={dist['max']}")
    for bucket in dist["histogram"]:
        print(f"    {bucket['range']}: {bucket['count']}")
    _save_image(dist["image_base64"], "score_distribution.png")

    print("\n=== 2. Event frequency heatmap ===")
    heatmap = analytics.event_frequency_heatmap()
    print(f"  event_count={heatmap['event_count']} "
          f"bucket_minutes={heatmap['bucket_minutes']}")
    _save_image(heatmap["image_base64"], "event_heatmap.png")

    print("\n=== 3. PCA + K-Means session clustering ===")
    clusters = analytics.cluster_sessions(k=3)
    if clusters["message"]:
        print(f"  {clusters['message']}")
    else:
        print(f"  session_count={clusters['session_count']} "
              f"pca_explained_variance_ratio={clusters['pca_explained_variance_ratio']}")
        for summary in clusters["cluster_summary"]:
            print(f"    {summary['cluster_name']}: "
                  f"{summary['session_count']} session(s), "
                  f"mean_integrity_score={summary['mean_integrity_score']}")
    _save_image(clusters["image_base64"], "pca_clusters.png")

    print("\n=== 4. Persist cluster assignments (session_clusters table) ===")
    saved = analytics.save_session_clusters(k=3)
    print(f"  persisted {len(saved['clusters'])} session cluster "
          f"assignment(s)." if saved["clusters"] else "  nothing to persist.")

    print("\n=== 5. Cohort risk profile ===")
    profile = analytics.cohort_risk_profile()
    print(f"  session_count={profile['session_count']}")
    print(f"  overall={profile['overall']}")
    for course, breakdown in profile["by_course"].items():
        print(f"    course={course}: {breakdown}")

    print(f"\nDone. Chart PNGs (if any) are under ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()