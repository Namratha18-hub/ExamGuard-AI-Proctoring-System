"""
Milestone 5, Part 1 -- Streamlit Invigilator Dashboard.

A SEPARATE, additional entry point alongside the existing Flask app
(app.py) -- it does not import from, modify, or replace anything in
app.py, and it is run as its own process:

    streamlit run streamlit_dashboard.py

Everything shown here is read LIVE, on every interaction, straight
from the same SQLite database the Flask app already reads and writes
(config.DATABASE_PATH) -- through the SAME already-existing, already-
tested modules the rest of this project uses, not a second
implementation of anything:

    - analytics.py           integrity scores, event breakdown,
                              PCA + K-Means clustering, cohort risk
    - utils/report_agent.py  AI-generated integrity summaries
    - utils/scoring.py       RISK_THRESHOLDS, for consistent colors

This process NEVER writes to sessions, event_logs, integrity_scores,
or integrity_reports -- it only reads them. The one exception is the
optional "Recompute clusters" button, which calls the exact same
analytics.save_session_clusters() the rest of the app already relies
on, and that only ever touches the session_clusters table. Running
this dashboard alongside the live Flask app (even during a real exam)
is safe: nothing about a candidate's exam, scoring, or proctoring can
be affected by anything on this page.

Setup (one-time):
    pip install streamlit
Run:
    streamlit run streamlit_dashboard.py
"""

import base64
import sqlite3

import pandas as pd
import streamlit as st

import analytics
from config import DATABASE_PATH
from utils import report_agent


# =====================================
# Data access -- plain functions, no Streamlit calls in here at all,
# so this section is independently testable and reusable regardless
# of the UI layer below it.
# =====================================

def _get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def list_sessions():
    """
    Every exam session across every candidate, most recent first --
    this is the "session monitoring" data. Deliberately includes
    sessions that haven't been scored yet (status == "in_progress")
    so an invigilator can see what's happening RIGHT NOW, not just
    exams that have already finished. A LEFT JOIN to integrity_scores
    means a not-yet-scored session still shows up, just with
    integrity_score / risk_label as None.
    """
    connection = _get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT
                s.session_id, s.course, s.status,
                s.start_time, s.end_time,
                c.candidate_id, c.fullname, c.application_id,
                sc.integrity_score, sc.risk_label
            FROM sessions s
            LEFT JOIN candidates c ON c.candidate_id = s.candidate_id
            LEFT JOIN integrity_scores sc ON sc.session_id = s.session_id
            ORDER BY s.start_time DESC
        """)
        rows = cursor.fetchall()
    finally:
        connection.close()

    return [dict(row) for row in rows]


def risk_color(risk_label):
    return {
        "Low": "#2CA02C",
        "Medium": "#FF7F0E",
        "High": "#D62728",
    }.get(risk_label, "#64748B")


def image_bytes_from_base64(b64_string):
    """
    Decode a base64 PNG (exactly as returned by analytics.py's chart
    functions -- score_distribution/cluster_sessions/session_detail
    all return their charts this way) into raw bytes st.image() can
    render directly. Returns None unchanged for a missing chart, so
    callers can tell "no chart yet" apart from "decode failed".
    """
    if not b64_string:
        return None
    return base64.b64decode(b64_string)


# =====================================
# UI
# =====================================

st.set_page_config(
    page_title="ExamGuard AI -- Invigilator Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ ExamGuard AI -- Invigilator Dashboard")
st.caption(
    "Live view over the same database the exam platform writes to. "
    "Every number and chart below is computed fresh from "
    "sessions / event_logs / integrity_scores on each interaction -- "
    "nothing is hardcoded or cached beyond this page's own state."
)

all_sessions = list_sessions()

if not all_sessions:
    st.info("No exam sessions in the database yet.")
    st.stop()

sessions_df = pd.DataFrame(all_sessions)

# ---------------- Sidebar: cohort snapshot + controls ----------------
with st.sidebar:
    st.header("Cohort snapshot")

    profile = analytics.cohort_risk_profile()

    if profile["session_count"] == 0:
        st.write("No scored sessions yet.")
    else:
        st.metric("Scored sessions", profile["session_count"])
        st.metric("Mean integrity score", profile["overall"]["mean_integrity_score"])

        for label in ("Low", "Medium", "High"):
            bucket = profile["overall"].get(label, {"count": 0, "percentage": 0})
            st.write(f"**{label} risk:** {bucket['count']} session(s) ({bucket['percentage']}%)")

    st.divider()

    if st.button("🔄 Refresh now", use_container_width=True):
        st.rerun()

    st.divider()

    st.subheader("Cohort PCA + K-Means")
    st.caption(
        "Recomputes the cohort-wide clustering from every currently "
        "scored session and saves the assignments to the "
        "session_clusters table -- same function the rest of the app "
        "uses (analytics.save_session_clusters())."
    )
    cluster_k = st.number_input("Number of clusters (k)", min_value=2, max_value=8, value=3, step=1)

    if st.button("Recompute clusters", use_container_width=True):
        with st.spinner("Running StandardScaler -> PCA -> K-Means..."):
            saved = analytics.save_session_clusters(k=int(cluster_k))
        if saved["message"]:
            st.warning(saved["message"])
        else:
            st.success(f"Recomputed clusters for {saved['session_count']} session(s).")

# ---------------- Session monitoring table ----------------
st.subheader("📋 Session Monitoring")

status_options = sorted(sessions_df["status"].dropna().unique().tolist())
status_filter = st.multiselect(
    "Filter by status",
    options=status_options,
    default=status_options,
)

filtered_df = (
    sessions_df[sessions_df["status"].isin(status_filter)]
    if status_filter else sessions_df
)

if filtered_df.empty:
    st.info("No sessions match the current filter.")
    st.stop()

display_df = filtered_df.copy()
display_df["Candidate"] = (
    display_df["fullname"].fillna("Unknown")
    + " (" + display_df["application_id"].fillna("N/A") + ")"
)
display_df["Integrity"] = display_df.apply(
    lambda r: f"{r['integrity_score']:.1f} ({r['risk_label']})"
    if pd.notna(r["integrity_score"]) else "Not scored yet",
    axis=1,
)

st.dataframe(
    display_df[[
        "session_id", "Candidate", "course", "status",
        "start_time", "end_time", "Integrity",
    ]].rename(columns={
        "session_id": "Session",
        "course": "Course",
        "status": "Status",
        "start_time": "Start",
        "end_time": "End",
    }),
    use_container_width=True,
    hide_index=True,
)

# ---------------- Cohort-wide PCA + K-Means overview ----------------
with st.expander("🧩 Cohort PCA + K-Means overview (all scored sessions)", expanded=False):
    cohort_clusters = analytics.cluster_sessions(k=int(cluster_k))

    if cohort_clusters["message"]:
        st.write(cohort_clusters["message"])
    else:
        cohort_image = image_bytes_from_base64(cohort_clusters["image_base64"])
        if cohort_image:
            st.image(cohort_image, use_container_width=True)

        summary_df = pd.DataFrame(cohort_clusters["cluster_summary"])
        if not summary_df.empty:
            st.dataframe(
                summary_df.rename(columns={
                    "cluster_name": "Cluster",
                    "session_count": "Sessions",
                    "mean_integrity_score": "Mean Integrity Score",
                }),
                use_container_width=True,
                hide_index=True,
            )

# ---------------- Session detail drill-down ----------------
st.subheader("🔍 Session Detail")

session_options = {
    f"#{row['session_id']} -- {row['fullname'] or 'Unknown'} -- "
    f"{row['course'] or 'unknown course'} ({row['status']})": row["session_id"]
    for row in filtered_df.to_dict("records")
}

selected_label = st.selectbox("Select a session", options=list(session_options.keys()))
selected_session_id = session_options[selected_label]

detail = analytics.session_detail(selected_session_id)

if not detail["found"]:
    st.error("Session not found.")
    st.stop()

st.markdown(
    f"**Candidate:** {detail['candidate_name'] or 'Unknown'} "
    f"({detail['application_id'] or 'N/A'}) &nbsp;|&nbsp; "
    f"**Course:** {detail['course'] or 'Unknown'} &nbsp;|&nbsp; "
    f"**Status:** {detail['status'] or 'Unknown'}"
)

# ---- Integrity score / risk + quick metrics ----
col1, col2, col3, col4 = st.columns(4)

with col1:
    if detail["integrity_score"] is not None:
        st.metric("Integrity Score", detail["integrity_score"])
        st.markdown(
            f"<span style='background:{risk_color(detail['risk_label'])}22;"
            f"color:{risk_color(detail['risk_label'])};padding:4px 14px;"
            f"border-radius:20px;font-weight:700;font-size:0.85rem;'>"
            f"{detail['risk_label']} risk</span>",
            unsafe_allow_html=True,
        )
    else:
        st.metric("Integrity Score", "N/A")
        st.caption("Not submitted / not scored yet.")

with col2:
    st.metric("Total Events", detail["total_events"])

with col3:
    st.metric("Marks Deducted", detail["deducted_marks"])

with col4:
    if detail["face_presence_ratio"] is not None:
        st.metric("Face Presence", f"{detail['face_presence_ratio'] * 100:.1f}%")
    else:
        st.metric("Face Presence", "N/A")

# ---- Alerts / event breakdown ----
st.markdown("#### 🚨 Alerts / Event Breakdown")

if detail["event_category_counts"]:
    events_df = pd.DataFrame(
        list(detail["event_category_counts"].items()),
        columns=["Event Category", "Count"],
    )
    st.dataframe(events_df, use_container_width=True, hide_index=True)
else:
    st.write("No events logged for this session.")

if detail["breached_rules"]:
    st.warning(f"Breached thresholds: {', '.join(detail['breached_rules'])}")

# ---- AI-generated report ----
st.markdown("#### 🤖 AI-Generated Integrity Report")

report = report_agent.get_session_report(selected_session_id)

if report:
    st.info(report["summary_text"])
    method_label = "LLM (LangChain)" if report["generation_method"] == "llm" else "deterministic template"
    st.caption(f"Generated {report['generated_at']} via {method_label}.")
else:
    st.write(
        "No AI report has been generated for this session yet -- "
        "reports are generated automatically when an exam is "
        "submitted or terminated."
    )

# ---- Charts ----
st.markdown("#### 📈 Integrity Score Gauge")
gauge_bytes = image_bytes_from_base64(detail["score_gauge_image_base64"])
if gauge_bytes:
    st.image(gauge_bytes, use_container_width=True)
else:
    st.write("Not available -- this session hasn't been submitted/scored yet.")

st.markdown("#### 📊 Event Frequency (this session)")
event_bar_bytes = image_bytes_from_base64(detail["event_bar_image_base64"])
if event_bar_bytes:
    st.image(event_bar_bytes, use_container_width=True)
elif detail["is_submitted"]:
    st.write("No events were logged for this session.")
else:
    st.write("Available once this exam is submitted.")

st.markdown("#### 🧩 PCA + K-Means Cluster (this session highlighted)")
pca_bytes = image_bytes_from_base64(detail["pca_cluster_image_base64"])
if pca_bytes:
    st.image(pca_bytes, use_container_width=True)
    if detail["cluster_name"]:
        st.caption(f"Behaviour cluster: **{detail['cluster_name']}**")
elif detail["cluster_message"]:
    st.write(detail["cluster_message"])
    