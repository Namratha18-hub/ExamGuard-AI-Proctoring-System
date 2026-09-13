"""
Part 13 - Data Science & Analytics Module.

Corpus-level analysis across ALL completed exam sessions -- unlike
utils/scoring.py (Part 11) and utils/report_agent.py (Part 12), which
both operate on ONE session at a time, everything here reads across
every row in `integrity_scores` / `event_logs` / `sessions` to answer
questions about the examination cohort as a whole:

    - score_distribution()   -- how integrity scores are spread out
                                 across every completed session
    - event_frequency_heatmap() -- WHEN (minutes into a session)
                                 which event types happen most, summed
                                 across every session
    - cluster_sessions()     -- K-Means grouping of sessions into
                                 behaviour clusters (e.g. "clean",
                                 "borderline", "high concern")
    - cohort_risk_profile()  -- Low/Medium/High risk breakdown, overall
                                 and per exam course

Like Part 12, this module is a pure, read-only, post-hoc consumer of
data Parts 1-11 already produce -- it does not change live proctoring
behaviour, scoring, or report generation. The one schema addition it
needs (`sessions.course`, so analytics can be broken down per exam
course) and the one new table it owns (`session_clusters`, so a
K-Means run doesn't need to be recomputed on every page view) are both
handled by update_database_part13.py, the same one-off migration
pattern as Parts 11 and 12.

Every public function returns plain Python dicts/lists (no numpy /
pandas objects) so callers -- a Flask jsonify() route today, a
Streamlit dashboard in Part 14+ -- can use the result directly without
a separate serialization step.
"""

import base64
import io
import sqlite3

import matplotlib
matplotlib.use("Agg")  # no display available on a Flask server
import matplotlib.pyplot as plt
import pandas as pd

from config import DATABASE_PATH
from utils import detection_rules, event_logger, scoring


def _get_connection():
    return sqlite3.connect(DATABASE_PATH)


# =====================================
# DATA LOADING
# =====================================

def _load_scored_sessions_dataframe(connection):
    """
    One row per session that has a completed Part 11 integrity score
    (i.e. the session has actually finished -- submitted or
    terminated). Sessions still in progress have no integrity_scores
    row yet and are intentionally excluded from corpus-level
    analytics, the same way Part 12's reports only cover finished
    sessions.
    """
    return pd.read_sql_query(
        """
        SELECT
            s.session_id,
            s.candidate_id,
            s.course,
            sc.integrity_score,
            sc.risk_label,
            sc.total_events,
            sc.face_presence_ratio
        FROM integrity_scores sc
        JOIN sessions s ON s.session_id = sc.session_id
        """,
        connection,
    )


def _load_event_log_dataframe(connection):
    """
    One row per logged event across every session, joined with that
    session's start_time so a "minutes into session" offset can be
    computed. Sessions with no start_time (shouldn't normally happen)
    are dropped since an offset can't be computed for them.
    """
    return pd.read_sql_query(
        """
        SELECT
            e.session_id,
            e.event_type,
            e.timestamp,
            s.start_time
        FROM event_logs e
        JOIN sessions s ON s.session_id = e.session_id
        WHERE s.start_time IS NOT NULL
        """,
        connection,
    )


# =====================================
# 1. INTEGRITY SCORE DISTRIBUTION
# =====================================

def score_distribution(bin_size=10):
    """
    Corpus-level distribution of integrity_score across every
    completed session.

    Returns:
        {
            "session_count": int,
            "mean": float | None,
            "median": float | None,
            "std_dev": float | None,
            "min": float | None,
            "max": float | None,
            "histogram": [
                {"range": "0-10", "count": int},
                ...
            ],
            "image_base64": "<PNG bytes, base64-encoded>" | None,
        }
    A session_count of 0 means no session has finished yet -- every
    other field is None / empty in that case rather than raising.
    image_base64 is None only when session_count is 0 (nothing to
    plot).
    """
    connection = _get_connection()
    try:
        df = _load_scored_sessions_dataframe(connection)
    finally:
        connection.close()

    if df.empty:
        return {
            "session_count": 0,
            "mean": None,
            "median": None,
            "std_dev": None,
            "min": None,
            "max": None,
            "histogram": [],
            "image_base64": None,
        }

    scores = df["integrity_score"]

    bins = list(range(0, 101, bin_size))
    if bins[-1] != 100:
        bins.append(100)

    bucket_counts = pd.cut(
        scores, bins=bins, right=True, include_lowest=True
    ).value_counts().sort_index()

    histogram = [
        {
            "range": f"{int(interval.left)}-{int(interval.right)}",
            "count": int(count),
        }
        for interval, count in bucket_counts.items()
    ]

    mean_score = float(scores.mean())
    median_score = float(scores.median())

    # ---------------- render distribution image ----------------
    # A bar chart over the same buckets `histogram` reports, with the
    # mean/median marked -- gives an invigilator a single glance at
    # how integrity scores are spread across the whole cohort.
    labels = [bucket["range"] for bucket in histogram]
    counts = [bucket["count"] for bucket in histogram]

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(labels)), 4))

    ax.bar(labels, counts, color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.axvline(
        x=_bucket_position_for_score(mean_score, bins),
        color="darkorange", linestyle="--", linewidth=1.5,
        label=f"Mean ({mean_score:.1f})",
    )
    ax.axvline(
        x=_bucket_position_for_score(median_score, bins),
        color="green", linestyle=":", linewidth=1.5,
        label=f"Median ({median_score:.1f})",
    )

    ax.set_xlabel("Integrity score range")
    ax.set_ylabel("Number of sessions")
    ax.set_title("Integrity score distribution across all sessions")
    ax.legend()
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.read()).decode("ascii")

    return {
        "session_count": int(len(df)),
        "mean": round(mean_score, 2),
        "median": round(median_score, 2),
        "std_dev": round(float(scores.std()), 2) if len(df) > 1 else 0.0,
        "min": round(float(scores.min()), 2),
        "max": round(float(scores.max()), 2),
        "histogram": histogram,
        "image_base64": image_base64,
    }


def _bucket_position_for_score(score, bins):
    """
    Maps a raw score onto the same categorical x-axis `ax.bar()` uses
    for the histogram buckets above, so the mean/median lines land in
    the correct bucket instead of using a mismatched numeric x-scale.
    """
    bucket_size = bins[1] - bins[0]
    index = min(int(score // bucket_size), len(bins) - 2)
    # Fractional offset within the bucket for a slightly more precise
    # marker position (purely cosmetic).
    bucket_start = bins[index]
    fraction_within_bucket = (
        (score - bucket_start) / bucket_size if bucket_size else 0.0
    )
    return index - 0.5 + min(max(fraction_within_bucket, 0.0), 1.0)


# =====================================
# 2. EVENT FREQUENCY HEATMAP
# =====================================

def event_frequency_heatmap(bucket_minutes=5, max_minutes=60):
    """
    Corpus-level heatmap: event_type x time-bucket (minutes into the
    session, summed across every session), so an invigilator can see
    e.g. "tab switches spike in the last 10 minutes across the
    cohort," not just for one candidate.

    Returns:
        {
            "event_count": int,
            "bucket_minutes": int,
            "pivot_table": {event_type: {bucket_label: count, ...}, ...},
            "image_base64": "<PNG bytes, base64-encoded>" | None,
        }
    image_base64 is None only when there are zero events to plot.
    """
    connection = _get_connection()
    try:
        df = _load_event_log_dataframe(connection)
    finally:
        connection.close()

    if df.empty:
        return {
            "event_count": 0,
            "bucket_minutes": bucket_minutes,
            "pivot_table": {},
            "image_base64": None,
        }

    event_time = pd.to_datetime(df["timestamp"], errors="coerce")
    start_time = pd.to_datetime(df["start_time"], errors="coerce")

    minutes_in = (event_time - start_time).dt.total_seconds() / 60.0
    minutes_in = minutes_in.clip(lower=0, upper=max_minutes)

    bucket_start = (minutes_in // bucket_minutes) * bucket_minutes
    df = df.assign(bucket_start=bucket_start)
    df = df.dropna(subset=["bucket_start"])

    if df.empty:
        return {
            "event_count": 0,
            "bucket_minutes": bucket_minutes,
            "pivot_table": {},
            "image_base64": None,
        }

    df["bucket_label"] = df["bucket_start"].apply(
        lambda m: f"{int(m)}-{int(m) + bucket_minutes}m"
    )

    # Keep bucket columns in chronological order rather than whatever
    # order pandas' pivot_table happens to produce them in.
    ordered_labels = [
        f"{m}-{m + bucket_minutes}m"
        for m in range(0, max_minutes + bucket_minutes, bucket_minutes)
    ]

    pivot = pd.pivot_table(
        df,
        index="event_type",
        columns="bucket_label",
        values="session_id",
        aggfunc="count",
        fill_value=0,
    )

    pivot = pivot.reindex(
        columns=[label for label in ordered_labels if label in pivot.columns]
    )

    # ---------------- render heatmap image ----------------
    fig, ax = plt.subplots(
        figsize=(max(6, 0.6 * len(pivot.columns)), max(3, 0.5 * len(pivot.index)))
    )

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(
        [str(label).replace("_", " ") for label in pivot.index], fontsize=8
    )
    ax.set_xlabel("Minutes into session")
    ax.set_title("Event frequency across all sessions")

    fig.colorbar(im, ax=ax, label="Event count")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.read()).decode("ascii")

    return {
        "event_count": int(len(df)),
        "bucket_minutes": bucket_minutes,
        "pivot_table": {
            str(event_type): {
                str(col): int(pivot.loc[event_type, col]) for col in pivot.columns
            }
            for event_type in pivot.index
        },
        "image_base64": image_base64,
    }


# =====================================
# 3. K-MEANS SESSION CLUSTERING
# =====================================

def _build_cluster_feature_dataframe(connection):
    """
    One row per completed session, with:
      - integrity_score, face_presence_ratio, total_events
        (from Part 11's integrity_scores -- unchanged)
      - one column per event_type, counting how many times that
        event_type occurred in that session (0 if it never did)

    This is the feature matrix K-Means clusters on: sessions that
    behaved similarly (similar event mix + similar score) should end
    up in the same cluster.
    """
    sessions_df = _load_scored_sessions_dataframe(connection)

    if sessions_df.empty:
        return sessions_df

    events_df = pd.read_sql_query(
        "SELECT session_id, event_type FROM event_logs", connection
    )

    if not events_df.empty:
        event_counts = (
            events_df.groupby(["session_id", "event_type"])
            .size()
            .unstack(fill_value=0)
        )
        sessions_df = sessions_df.merge(
            event_counts, how="left", left_on="session_id", right_index=True
        )

    event_type_columns = [
        col for col in sessions_df.columns
        if col not in (
            "session_id", "candidate_id", "course", "integrity_score",
            "risk_label", "total_events", "face_presence_ratio",
        )
    ]
    sessions_df[event_type_columns] = sessions_df[event_type_columns].fillna(0)

    # face_presence_ratio can be NULL (Part 11: not estimable for a
    # session with no measurable duration) -- treat unknown as "fully
    # present" (1.0) rather than dropping the session from clustering.
    sessions_df["face_presence_ratio"] = sessions_df["face_presence_ratio"].fillna(1.0)

    return sessions_df


def cluster_sessions(k=3, n_pca_components=2, highlight_session_id=None):
    """
    Groups completed sessions into `k` behaviour clusters using
    K-Means over [integrity_score, face_presence_ratio, total_events,
    per-event-type counts], run through a
    StandardScaler -> PCA -> K-Means pipeline:

      1. StandardScaler -- standardizes every feature so no single
         one (e.g. a raw event count) dominates purely due to scale.
      2. PCA -- reduces the standardized features down to
         `n_pca_components` (2 by default) principal components. This
         both de-noises/de-correlates the feature set before
         clustering and gives a 2D projection that can be plotted
         directly, which the raw (often 10+ dimensional) feature space
         cannot.
      3. K-Means -- clusters sessions using the PCA-reduced
         components rather than the raw standardized features.

    Clusters are labelled by their mean integrity_score, ascending --
    the lowest-scoring cluster is always "High Concern" and the
    highest-scoring is always "Low Concern" (for k == 3; for any other
    k a generic "lower score = more concern" ranking is used instead),
    regardless of which numeric label K-Means happened to assign
    internally.

    Returns:
        {
            "k": int,
            "session_count": int,
            "clusters": [
                {"session_id", "candidate_id", "course",
                 "integrity_score", "cluster_label", "cluster_name",
                 "pc1", "pc2"},
                ...
            ],
            "cluster_summary": [
                {"cluster_name", "session_count", "mean_integrity_score"},
                ...
            ],
            "pca_explained_variance_ratio": [float, ...] | None,
            "image_base64": "<PNG bytes, base64-encoded>" | None,
            "message": str | None,  # set instead of clustering when
                                     # there isn't enough data yet
        }
    "pc1"/"pc2" and image_base64 are only populated when at least 2
    principal components were computed (the normal case); with a
    single usable feature they fall back to None / None respectively
    rather than raising.
    """
    connection = _get_connection()
    try:
        df = _build_cluster_feature_dataframe(connection)
    finally:
        connection.close()

    empty_result = {
        "k": k,
        "session_count": 0 if df.empty else int(len(df)),
        "clusters": [],
        "cluster_summary": [],
        "pca_explained_variance_ratio": None,
        "image_base64": None,
        "message": None,
    }

    if df.empty:
        empty_result["message"] = "No completed sessions yet -- nothing to cluster."
        return empty_result

    if len(df) < k:
        empty_result["message"] = (
            f"Only {len(df)} completed session(s) so far; need at least "
            f"{k} to form {k} clusters. Try again once more candidates "
            f"finish their exam."
        )
        return empty_result

    # Imported here (like report_agent.py lazily imports langchain)
    # so the rest of this module -- and the rest of the app -- still
    # works even in an environment where scikit-learn isn't installed
    # yet and clustering simply isn't being used.
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    feature_columns = [
        col for col in df.columns
        if col not in ("session_id", "candidate_id", "course", "risk_label")
    ]

    # ---------------- Step 1: StandardScaler ----------------
    scaled_features = StandardScaler().fit_transform(df[feature_columns])

    # ---------------- Step 2: PCA ----------------
    # Bounded by both the number of sessions and the number of
    # features -- PCA can't produce more components than
    # min(n_samples, n_features). With very little data/very few
    # feature columns this degrades gracefully to 1 component instead
    # of raising.
    effective_components = max(
        1, min(n_pca_components, len(feature_columns), len(df) - 1 or 1)
    )

    pca = PCA(n_components=effective_components, random_state=42)
    principal_components = pca.fit_transform(scaled_features)

    explained_variance_ratio = [
        round(float(ratio), 4) for ratio in pca.explained_variance_ratio_
    ]

    # ---------------- Step 3: K-Means (on the PCA output) ----------------
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    raw_labels = kmeans.fit_predict(principal_components)

    df = df.assign(raw_cluster_label=raw_labels)
    df = df.assign(pc1=principal_components[:, 0])
    df = df.assign(
        pc2=principal_components[:, 1] if effective_components >= 2 else None
    )

    # Rank raw cluster ids by mean integrity_score, ascending, so
    # naming is always consistent regardless of KMeans' arbitrary
    # internal numbering.
    ranked_cluster_ids = (
        df.groupby("raw_cluster_label")["integrity_score"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    if k == 3:
        names_by_rank = ["High Concern", "Moderate Concern", "Low Concern"]
    else:
        names_by_rank = [
            f"Cluster {rank + 1} of {k} (lower score = more concern)"
            for rank in range(k)
        ]

    name_by_raw_id = {
        raw_id: names_by_rank[rank]
        for rank, raw_id in enumerate(ranked_cluster_ids)
    }

    clusters = []
    for _, row in df.iterrows():
        clusters.append({
            "session_id": int(row["session_id"]),
            "candidate_id": int(row["candidate_id"]) if pd.notna(row["candidate_id"]) else None,
            "course": row["course"] if pd.notna(row["course"]) else None,
            "integrity_score": round(float(row["integrity_score"]), 2),
            "cluster_label": int(row["raw_cluster_label"]),
            "cluster_name": name_by_raw_id[row["raw_cluster_label"]],
            "pc1": round(float(row["pc1"]), 4),
            "pc2": round(float(row["pc2"]), 4) if pd.notna(row["pc2"]) else None,
        })

    cluster_summary = []
    for raw_id in ranked_cluster_ids:
        group = df[df["raw_cluster_label"] == raw_id]
        cluster_summary.append({
            "cluster_name": name_by_raw_id[raw_id],
            "session_count": int(len(group)),
            "mean_integrity_score": round(float(group["integrity_score"].mean()), 2),
        })

    # ---------------- render PCA 2D cluster scatter plot ----------------
    image_base64 = None
    if effective_components >= 2:
        colors_by_rank = {3: ["#D62728", "#FF7F0E", "#2CA02C"]}  # red/orange/green
        palette = colors_by_rank.get(k)

        fig, ax = plt.subplots(figsize=(7, 6))

        for rank, raw_id in enumerate(ranked_cluster_ids):
            group = df[df["raw_cluster_label"] == raw_id]
            color = palette[rank] if palette else None
            ax.scatter(
                group["pc1"], group["pc2"],
                label=name_by_raw_id[raw_id],
                color=color, alpha=0.75, edgecolor="black", linewidth=0.4,
            )

        centers_pca = kmeans.cluster_centers_
        ax.scatter(
            centers_pca[:, 0], centers_pca[:, 1],
            marker="X", s=180, color="black", label="Cluster centroid",
        )

        # Ring the caller's session of interest (e.g. the one a
        # candidate/invigilator currently has open on a per-session
        # dashboard) so its position among the cohort is visible at a
        # glance, without needing a second plot or a second K-Means
        # run. No-op (silently skipped) if the session_id isn't part
        # of this cohort run -- e.g. not scored yet.
        if highlight_session_id is not None:
            highlighted = df[df["session_id"] == highlight_session_id]
            if not highlighted.empty:
                ax.scatter(
                    highlighted["pc1"], highlighted["pc2"],
                    s=300, facecolors="none", edgecolors="black",
                    linewidth=2.2, marker="o", label="Selected session",
                    zorder=5,
                )

        ax.set_xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.1f}% variance)")
        ax.set_ylabel(
            f"PC2 ({explained_variance_ratio[1] * 100:.1f}% variance)"
            if len(explained_variance_ratio) > 1 else "PC2"
        )
        ax.set_title("Session behaviour clusters (PCA + K-Means)")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120)
        plt.close(fig)
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.read()).decode("ascii")

    return {
        "k": k,
        "session_count": int(len(df)),
        "clusters": clusters,
        "cluster_summary": cluster_summary,
        "pca_explained_variance_ratio": explained_variance_ratio,
        "image_base64": image_base64,
        "message": None,
    }


def save_session_clusters(k=3):
    """
    Runs cluster_sessions(k) and persists every session's cluster
    assignment into `session_clusters` (see update_database_part13.py),
    so a dashboard/export doesn't need to re-run K-Means on every
    view. Safe to call more than once -- session_id is UNIQUE in
    session_clusters, same upsert pattern as Part 12's
    save_session_report().

    Returns the same dict cluster_sessions(k) does. No-ops the DB
    write (but still returns the result) when there wasn't enough
    data to cluster.
    """
    result = cluster_sessions(k)

    if not result["clusters"]:
        return result

    connection = _get_connection()
    cursor = connection.cursor()

    for entry in result["clusters"]:
        cursor.execute("""
            INSERT INTO session_clusters
                (session_id, candidate_id, cluster_label, cluster_name)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                cluster_label=excluded.cluster_label,
                cluster_name=excluded.cluster_name,
                computed_at=CURRENT_TIMESTAMP
        """, (
            entry["session_id"],
            entry["candidate_id"],
            entry["cluster_label"],
            entry["cluster_name"],
        ))

    connection.commit()
    connection.close()

    return result


# =====================================
# 4. COHORT RISK PROFILE
# =====================================

def cohort_risk_profile():
    """
    Low/Medium/High risk breakdown across every completed session,
    overall and per exam course (per-course requires the `course`
    column added to `sessions` by update_database_part13.py --
    sessions created before that migration will show up under
    course = None).

    Returns:
        {
            "session_count": int,
            "overall": {
                "Low": {"count": int, "percentage": float},
                "Medium": {...}, "High": {...},
                "mean_integrity_score": float,
            },
            "by_course": {
                "<course>": {
                    "session_count": int,
                    "Low": {"count", "percentage"}, "Medium": {...}, "High": {...},
                    "mean_integrity_score": float,
                },
                ...
            },
        }
    """
    connection = _get_connection()
    try:
        df = _load_scored_sessions_dataframe(connection)
    finally:
        connection.close()

    risk_labels = ("Low", "Medium", "High")

    if df.empty:
        return {
            "session_count": 0,
            "overall": {
                label: {"count": 0, "percentage": 0.0} for label in risk_labels
            } | {"mean_integrity_score": None},
            "by_course": {},
        }

    def _risk_breakdown(sub_df):
        total = len(sub_df)
        counts = sub_df["risk_label"].value_counts()
        breakdown = {}
        for label in risk_labels:
            count = int(counts.get(label, 0))
            breakdown[label] = {
                "count": count,
                "percentage": round((count / total) * 100, 1) if total else 0.0,
            }
        breakdown["mean_integrity_score"] = round(float(sub_df["integrity_score"].mean()), 2)
        return breakdown

    overall = _risk_breakdown(df)

    by_course = {}
    for course_name, group in df.groupby(df["course"].fillna("Unknown")):
        by_course[str(course_name)] = {
            "session_count": int(len(group)),
            **_risk_breakdown(group),
        }

    return {
        "session_count": int(len(df)),
        "overall": overall,
        "by_course": by_course,
    }


# =====================================
# 5. PER-SESSION INTEGRITY DASHBOARD (Milestone 3 dashboard)
# =====================================
# Everything below answers a different question than functions 1-4
# above: not "how does the cohort look?" but "show me everything
# about THIS ONE session_id, right now." This is what a
# candidate/invigilator-facing "select a session" dashboard calls.
#
# It deliberately REUSES the existing, already-shipped modules
# instead of re-deriving any of their logic, so this can never drift
# from what actually happened during the exam:
#   - utils.scoring.get_integrity_score()        -- Part 11 score/risk
#   - utils.event_logger.get_event_counts() /
#     get_total_event_count()                    -- Part 2 raw counts
#   - utils.detection_rules.evaluate_session()    -- the SAME rule
#     engine app.py's /update_warning calls live during the exam, so
#     "marks deducted" here is guaranteed to match what was actually
#     applied -- not a second implementation that could drift from it.
#   - cluster_sessions() (this module) -- the cohort PCA + K-Means
#     run, asked to highlight this one session_id in its scatter plot.
#
# Every value and every chart below is computed fresh from
# sessions / event_logs / integrity_scores at call time. Nothing is
# cached, precomputed, saved to a static file, or hardcoded -- calling
# this twice in a row against a database that changed in between
# returns two different results.

EVENT_CATEGORY_TYPES = {
    "Tab Switches": ["tab_switch", "tab_switch_timeout"],
    "Fullscreen Exits": ["fullscreen_exit"],
    "Copy/Paste": ["copy", "paste", "copy_paste"],
    "Right-Click": ["right_click", "contextmenu"],
    "Face Absence": ["face_absent"],
    "Face Absence (Prolonged)": ["face_absent_prolonged"],
    "Multiple Persons": ["multiple_persons"],
    "Phone Detected": ["phone_detected"],
    "Laptop Detected": ["laptop_detected"],
    "Book Detected": ["book_detected"],
    "Head-Turn Warnings": ["head_turn"],
}


def _load_session_and_candidate(connection, session_id):
    """
    One row (or zero) with this session's own columns plus the
    owning candidate's display name/application_id, via a LEFT JOIN
    so a session somehow missing its candidate record still comes
    back (with candidate_name/application_id as None) instead of
    silently disappearing.
    """
    return pd.read_sql_query(
        """
        SELECT
            s.session_id, s.candidate_id, s.course, s.start_time,
            s.end_time, s.status,
            c.fullname AS candidate_name, c.application_id
        FROM sessions s
        LEFT JOIN candidates c ON c.candidate_id = s.candidate_id
        WHERE s.session_id = ?
        """,
        connection,
        params=(session_id,),
    )


def _render_score_gauge(score, risk_label):
    """
    This session's integrity score as a horizontal gauge against the
    SAME Low/Medium/High risk zone boundaries utils/scoring.py
    (RISK_THRESHOLDS) uses to assign risk_label, with a marker at the
    actual score. Rebuilt from `score`/`risk_label` on every call.
    """
    high_end = scoring.RISK_THRESHOLDS["Medium"]
    medium_end = scoring.RISK_THRESHOLDS["Low"]

    fig, ax = plt.subplots(figsize=(7, 1.6))

    ax.barh(0, high_end, color="#D62728", alpha=0.35)
    ax.barh(0, medium_end - high_end, left=high_end, color="#FF7F0E", alpha=0.35)
    ax.barh(0, 100 - medium_end, left=medium_end, color="#2CA02C", alpha=0.35)

    ax.axvline(x=score, color="black", linewidth=3)
    ax.scatter([score], [0], color="black", s=140, zorder=5, marker="v")

    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel("Integrity score")
    ax.set_title(f"Integrity score: {score:.1f} ({risk_label} risk)")

    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("ascii")


def _render_session_event_bar_chart(category_counts):
    """
    Bar chart of THIS session's own event counts, grouped into the
    same categories the dashboard displays. Distinct from
    event_frequency_heatmap() above, which is a corpus-wide,
    time-bucketed heatmap across every session, not one session's
    totals.
    """
    labels = list(category_counts.keys())
    counts = list(category_counts.values())

    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(labels)), 4.5))

    ax.bar(labels, counts, color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Event count")
    ax.set_title("Event frequency for this session")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("ascii")


def session_detail(session_id, cluster_k=3):
    """
    Full, live, single-session integrity dashboard payload for one
    candidate's one exam session.

    IMPORTANT -- submission-gated evaluation:
    The event-frequency chart and the PCA + K-Means cluster
    (including its scatter plot) are only computed once the exam has
    actually been SUBMITTED -- i.e. sessions.status is "completed" or
    "terminated" AND an integrity_scores row exists for it (written by
    utils.scoring.save_integrity_score(), called from app.py's
    /submit_exam and /update_warning routes). For a session that is
    still "in_progress", those two are intentionally left as None
    (not an empty/static placeholder image) -- there is nothing
    "submitted" yet for them to evaluate. Raw event counts are still
    returned either way since event_logs rows are real regardless of
    submission state, but the derived event/cluster CHARTS are not
    generated until submission.

    Returns:
        {
            "found": bool,  # False if session_id doesn't exist at all
                             # -- every other key is absent when False

            "session_id": int,
            "candidate_id": int | None,
            "candidate_name": str | None,
            "application_id": str | None,
            "course": str | None,
            "status": str | None,
            "start_time": str | None,
            "end_time": str | None,
            "is_submitted": bool,  # status in (completed, terminated)
                                    # AND an integrity score exists

            "integrity_score": float | None,   # None = not scored yet
            "risk_label": str | None,
            "face_presence_ratio": float | None,

            "total_events": int,
            "deducted_marks": int,
            "breached_rules": [event_type, ...],

            "event_category_counts": {category_label: count, ...},
            "raw_event_counts": {event_type: count, ...},

            "cluster_name": str | None,
            "cluster_label": int | None,
            "pca_explained_variance_ratio": [float, ...] | None,
            "cluster_message": str | None,   # explains why cluster_name
                                              # is None -- not submitted
                                              # yet, or cohort too small

            "score_gauge_image_base64": str | None,
            "event_bar_image_base64": str | None,
            "pca_cluster_image_base64": str | None,
        }
    """
    connection = _get_connection()
    try:
        session_df = _load_session_and_candidate(connection, session_id)
    finally:
        connection.close()

    if session_df.empty:
        return {"found": False, "session_id": session_id}

    session_row = session_df.iloc[0]

    # ---- integrity score + risk (Part 11 scoring, reused as-is) ----
    score_row = scoring.get_integrity_score(session_id)

    # A session only counts as "submitted" for evaluation purposes once
    # the rule engine/candidate has actually ended it (terminated by
    # detection_rules, or completed via /submit_exam) AND
    # save_integrity_score() has actually run for it. Checking BOTH
    # (not just status) means a session that was marked ended but
    # somehow never got scored still correctly shows "not evaluated
    # yet" instead of a chart built on missing data.
    is_submitted = (
        session_row["status"] in ("completed", "terminated")
        and score_row is not None
    )

    # ---- raw event counts (Part 2 event_logger, reused as-is) ----
    # These reflect real event_logs rows regardless of submission
    # state -- an in-progress session's events already happened and
    # are real data. Only the CHARTS built from them are gated below.
    raw_event_counts = event_logger.get_event_counts(session_id)
    total_events = event_logger.get_total_event_count(session_id)

    event_category_counts = {
        label: sum(raw_event_counts.get(event_type, 0) for event_type in event_types)
        for label, event_types in EVENT_CATEGORY_TYPES.items()
    }

    # Anything logged that ISN'T one of the named categories above
    # (e.g. the generic "browser_violation" fallback used when
    # exam.html sends no event_type) still counts toward
    # total_events/marks -- surfaced here as its own bucket so the
    # displayed categories always sum to total_events, with nothing
    # silently dropped.
    categorized_types = {
        event_type
        for event_types in EVENT_CATEGORY_TYPES.values()
        for event_type in event_types
    }
    other_count = sum(
        count for event_type, count in raw_event_counts.items()
        if event_type not in categorized_types
    )
    if other_count:
        event_category_counts["Other"] = other_count

    # ---- marks deducted (Part 2/5 rule engine, reused as-is -- the
    # SAME function app.py's /update_warning calls live during the
    # exam) ----
    rule_result = detection_rules.evaluate_session(session_id)

    # ---- charts + cluster (rebuilt fresh on every call -- nothing
    # cached -- and only computed at all once the exam is submitted) ----
    score_gauge_image = None
    event_bar_image = None
    pca_cluster_image = None
    cluster_name = None
    cluster_label = None
    pca_explained_variance_ratio = None

    if not is_submitted:
        cluster_message = (
            "This exam session hasn't been submitted yet -- the "
            "integrity score, event-frequency chart, and PCA cluster "
            "are computed from the SUBMITTED exam and will appear "
            "once this session is completed or terminated."
        )
    else:
        score_gauge_image = _render_score_gauge(
            score_row["integrity_score"], score_row["risk_label"]
        )

        if total_events > 0:
            event_bar_image = _render_session_event_bar_chart(event_category_counts)

        # Cohort PCA + K-Means run, this session highlighted in the
        # same run -- only worth running once this session actually
        # has a real integrity score to place on the plot.
        cluster_result = cluster_sessions(k=cluster_k, highlight_session_id=session_id)

        for entry in cluster_result["clusters"]:
            if entry["session_id"] == session_id:
                cluster_name = entry["cluster_name"]
                cluster_label = entry["cluster_label"]
                break

        pca_explained_variance_ratio = cluster_result["pca_explained_variance_ratio"]
        pca_cluster_image = cluster_result["image_base64"]

        if cluster_result["message"]:
            # Not enough scored sessions across the whole cohort yet
            # (e.g. fewer than cluster_k) -- a cohort-wide limitation,
            # not specific to this session.
            cluster_message = cluster_result["message"]
        elif cluster_name is None:
            # Should be rare given is_submitted already required a
            # score, but covers the edge case defensively rather than
            # silently showing a blank cluster name.
            cluster_message = (
                "This session wasn't included in the latest cohort "
                "clustering run."
            )
        else:
            cluster_message = None

    return {
        "found": True,
        "session_id": int(session_row["session_id"]),
        "candidate_id": (
            int(session_row["candidate_id"])
            if pd.notna(session_row["candidate_id"]) else None
        ),
        "candidate_name": session_row["candidate_name"],
        "application_id": session_row["application_id"],
        "course": session_row["course"],
        "status": session_row["status"],
        "start_time": session_row["start_time"],
        "end_time": session_row["end_time"],
        "is_submitted": is_submitted,

        "integrity_score": score_row["integrity_score"] if score_row else None,
        "risk_label": score_row["risk_label"] if score_row else None,
        "face_presence_ratio": score_row["face_presence_ratio"] if score_row else None,

        "total_events": total_events,
        "deducted_marks": rule_result["deducted_marks"],
        "breached_rules": rule_result["breached_rules"],

        "event_category_counts": event_category_counts,
        "raw_event_counts": raw_event_counts,

        "cluster_name": cluster_name,
        "cluster_label": cluster_label,
        "pca_explained_variance_ratio": pca_explained_variance_ratio,
        "cluster_message": cluster_message,

        "score_gauge_image_base64": score_gauge_image,
        "event_bar_image_base64": event_bar_image,
        "pca_cluster_image_base64": pca_cluster_image,
    }