"""
Part 12 - AI Integrity Report Agent (LangChain-based).

Reads a completed session's:
  - timestamp-level event_logs rows (utils/event_logger.py's table --
    read directly here, not through get_event_counts(), because a
    natural-language summary needs to describe WHEN events happened,
    not just how many there were), and
  - Part 11's computed integrity_score / risk_label / event_counts /
    face_presence_ratio (utils/scoring.py's calculate_integrity_score)

...and produces a concise, invigilator-friendly natural-language
summary, which is then persisted to the `integrity_reports` table
(see database.py / update_database_part12.py).

This module does not change any live proctoring behaviour (warning
counts, marks deduction, termination) or Part 11's scoring formula --
it is purely a downstream, post-hoc consumer of data those modules
already produce, in the same spirit utils/scoring.py was a post-hoc
consumer of utils/event_logger.py's data.

====================================================================
TWO GENERATION PATHS -- LLM (LangChain) vs. DETERMINISTIC FALLBACK
====================================================================
1. If config.OPENAI_API_KEY is set, the agent builds a structured,
   already-computed set of "session facts" (event counts + first/last
   occurrence minute per event type, face presence estimate, session
   duration, integrity score, risk label) and asks a LangChain-wrapped
   chat model to REWRITE those facts into 2-4 natural, invigilator-
   friendly sentences. The model is explicitly instructed not to
   invent any number, event, or conclusion beyond what it's given --
   it is a phrasing layer on top of numbers this codebase already
   computed deterministically, not a free-form analyst.

2. If no API key is configured, OR the LLM call fails for any reason
   (network error, invalid key, rate limit, malformed response), the
   SAME session facts are instead formatted into a summary directly by
   a deterministic template function -- no LLM involved. This is not a
   fake/mocked LLM call; it's a plain Python string-formatting
   function, used specifically so:
     (a) the app never breaks report generation for lack of a paid API
         key, and
     (b) Low / Medium / High risk scenarios can be exercised and
         tested without any network access or credentials.
   Every stored report records WHICH path produced it, in the
   `generation_method` column ("langchain_llm" or
   "deterministic_fallback"), so this is never silently ambiguous to
   an invigilator or a grader reading the table.

`langchain` / `langchain-openai` are imported lazily, only inside the
LLM code path (_generate_via_langchain), specifically so that this
whole module -- including the fallback path -- still works in an
environment where those packages aren't installed at all (e.g. no
LLM feature is being used yet). If OPENAI_API_KEY is set but the
packages aren't installed, that import failure is caught by the same
try/except as any other LLM failure and falls back automatically.
"""

import sqlite3
from datetime import datetime

import pandas as pd

from config import DATABASE_PATH, OPENAI_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE
from utils import scoring


def _get_connection():
    return sqlite3.connect(DATABASE_PATH)


# =====================================
# DATA LOADING (Pandas, timestamp-level)
# =====================================

def _load_timestamped_events(connection, session_id):
    """
    One row per logged event for this session, in chronological
    order, with its raw timestamp -- unlike
    utils.scoring._load_event_log_dataframe / event_logger's
    get_event_counts(), which both collapse events down to counts.
    A narrative summary needs the "when", not just the "how many".
    """
    return pd.read_sql_query(
        """
        SELECT event_type, timestamp
        FROM event_logs
        WHERE session_id = ?
        ORDER BY timestamp ASC
        """,
        connection,
        params=(session_id,),
    )


def _load_session_row(connection, session_id):
    return pd.read_sql_query(
        """
        SELECT start_time, end_time, status, candidate_id
        FROM sessions
        WHERE session_id = ?
        """,
        connection,
        params=(session_id,),
    )


# =====================================
# SESSION FACTS (shared by both generation paths)
# =====================================

def _build_session_facts(session_id):
    """
    Gathers everything a summary could need into one plain dict:
        - integrity_score, risk_label, total_events, event_counts,
          face_presence_ratio  (from Part 11's scoring module --
          UNCHANGED, reused as-is, not recomputed differently here)
        - session_duration_minutes (float or None)
        - event_timeline: list of (event_type, minutes_into_session)
          tuples, chronological, used to describe WHEN events happened
    """
    score_result = scoring.calculate_integrity_score(session_id)

    facts = {
        "session_id": session_id,
        "integrity_score": score_result["integrity_score"],
        "risk_label": score_result["risk_label"],
        "total_events": score_result["total_events"],
        "event_counts": score_result["event_counts"],
        "face_presence_ratio": score_result["face_presence_ratio"],
        "session_duration_minutes": None,
        "event_timeline": [],
    }

    if session_id is None:
        return facts

    connection = _get_connection()
    try:
        events_df = _load_timestamped_events(connection, session_id)
        session_df = _load_session_row(connection, session_id)
    finally:
        connection.close()

    if session_df.empty:
        return facts

    start_raw = session_df.at[0, "start_time"]
    end_raw = session_df.at[0, "end_time"]

    if not start_raw:
        return facts

    start_time = pd.to_datetime(start_raw, errors="coerce")
    # A still-in-progress session (end_time NULL) falls back to "now"
    # for duration purposes only -- same approach Part 11's
    # face-presence estimate already uses, kept consistent here.
    end_time = pd.to_datetime(end_raw, errors="coerce") if end_raw else pd.Timestamp(datetime.now())

    if pd.isna(start_time) or pd.isna(end_time):
        return facts

    duration_seconds = (end_time - start_time).total_seconds()
    if duration_seconds > 0:
        facts["session_duration_minutes"] = round(duration_seconds / 60, 1)

    if not events_df.empty:
        event_timestamps = pd.to_datetime(events_df["timestamp"], errors="coerce")
        minutes_into_session = ((event_timestamps - start_time).dt.total_seconds() / 60).round(1)
        facts["event_timeline"] = list(zip(events_df["event_type"], minutes_into_session))

    return facts


# =====================================
# DETERMINISTIC SUMMARY (fallback AND the factual grounding given to the LLM)
# =====================================

def _humanize_event_type(event_type):
    return str(event_type).replace("_", " ")


def _deterministic_summary(facts):
    """
    Builds a concise invigilator-friendly summary directly from
    `facts`, with no LLM involved. Used as-is when no LLM is
    configured / the LLM call fails, AND used as the factual text
    handed to the LLM in the langchain path below, so the LLM has
    nothing to hallucinate from -- it can only rephrase numbers this
    function already computed.
    """
    event_counts = facts["event_counts"]
    timeline = facts["event_timeline"]

    if not event_counts:
        behaviour_sentence = "No suspicious events were logged during this session."
    else:
        # Most frequent event type first, so the summary leads with
        # the candidate's most prominent behaviour.
        ordered_event_types = sorted(
            event_counts.items(), key=lambda item: item[1], reverse=True
        )

        phrases = []
        for event_type, count in ordered_event_types:
            occurrences = [minute for etype, minute in timeline if etype == event_type]
            label = _humanize_event_type(event_type)
            unit = "event" if count == 1 else "events"

            if occurrences:
                first_minute = min(occurrences)
                phrases.append(
                    f"{count} {label} {unit} (first at minute {first_minute:.1f})"
                )
            else:
                phrases.append(f"{count} {label} {unit}")

        behaviour_sentence = "Candidate triggered " + "; ".join(phrases) + "."

    face_presence_ratio = facts["face_presence_ratio"]
    if face_presence_ratio is not None:
        absent_percentage = round((1 - face_presence_ratio) * 100, 1)
        presence_sentence = (
            f" Face was estimated absent for approximately {absent_percentage}% "
            f"of the session (an estimate from logged absence events, not a "
            f"direct measurement)."
        )
    else:
        presence_sentence = " Face presence could not be estimated for this session."

    duration_sentence = ""
    if facts["session_duration_minutes"] is not None:
        duration_sentence = f" Session lasted approximately {facts['session_duration_minutes']} minutes."

    conclusion_sentence = (
        f" Overall integrity score: {facts['integrity_score']}/100. "
        f"Overall integrity risk: {facts['risk_label']}."
    )

    return (
        behaviour_sentence + presence_sentence + duration_sentence + conclusion_sentence
    ).strip()


# =====================================
# LLM SUMMARY (LangChain) -- only reached if OPENAI_API_KEY is set
# =====================================

def _generate_via_langchain(facts):
    """
    Rewrites the deterministic summary's facts into a more natural
    invigilator-friendly summary using a LangChain-wrapped chat model.

    Raises on any failure (missing packages, network error, bad key,
    empty response) -- the caller (generate_session_summary) is
    responsible for catching that and falling back. This function
    itself never falls back silently, so failures are never hidden
    from whoever is debugging a "why did it use the fallback" question.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    grounding_facts_text = _deterministic_summary(facts)

    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY,
    )

    system_prompt = (
        "You are an exam-integrity assistant writing a short, neutral "
        "summary of a candidate's exam session for a human invigilator "
        "to review. Use ONLY the facts provided -- do not invent, "
        "estimate, or infer any number, event, or timing that isn't "
        "already stated. Do not recommend or imply any disciplinary "
        "action; simply describe what was observed. Write 2 to 4 "
        "concise sentences."
    )

    human_prompt = (
        "Here are the already-computed facts for this exam session "
        "(do not change any number):\n\n"
        f"{grounding_facts_text}\n\n"
        "Rewrite this into a natural, concise, invigilator-friendly "
        "summary."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])

    summary_text = (response.content or "").strip()

    if not summary_text:
        raise ValueError("LLM returned an empty response")

    return summary_text


# =====================================
# PUBLIC API
# =====================================

def generate_session_summary(session_id):
    """
    Computes session facts and produces a summary via the LLM path if
    configured (falling back automatically on any failure), or the
    deterministic path otherwise.

    Pure -- does not write to the database. Returns:
        {
            "session_id": ...,
            "integrity_score": ...,
            "risk_label": "Low" | "Medium" | "High",
            "summary_text": "...",
            "generation_method": "langchain_llm" | "deterministic_fallback",
        }
    """
    facts = _build_session_facts(session_id)

    if OPENAI_API_KEY:
        try:
            summary_text = _generate_via_langchain(facts)
            generation_method = "langchain_llm"
        except Exception as error:
            # Deliberately broad: ANY failure in the LLM path (missing
            # packages, network error, invalid/expired key, malformed
            # response, rate limit, etc.) must degrade to the
            # deterministic summary rather than break report
            # generation at exam conclusion.
            print(
                f"WARNING: LangChain report generation failed for "
                f"session {session_id} ({error}); using deterministic "
                f"fallback summary instead."
            )
            summary_text = _deterministic_summary(facts)
            generation_method = "deterministic_fallback"
    else:
        summary_text = _deterministic_summary(facts)
        generation_method = "deterministic_fallback"

    return {
        "session_id": session_id,
        "integrity_score": facts["integrity_score"],
        "risk_label": facts["risk_label"],
        "summary_text": summary_text,
        "generation_method": generation_method,
    }


def save_session_report(session_id, candidate_id):
    """
    Generates (see generate_session_summary) and stores/updates the
    integrity_reports row for a session. Safe to call more than once
    for the same session_id -- e.g. if submit_exam runs after the rule
    engine already terminated the session -- since session_id is
    UNIQUE in integrity_reports, mirroring Part 11's
    save_integrity_score().

    No-ops (returns None) if session_id is None, so a call site that
    hasn't got a valid session yet doesn't need its own guard.
    """
    if session_id is None:
        return None

    result = generate_session_summary(session_id)

    connection = _get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO integrity_reports
            (session_id, candidate_id, integrity_score, risk_label,
             summary_text, generation_method)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            integrity_score=excluded.integrity_score,
            risk_label=excluded.risk_label,
            summary_text=excluded.summary_text,
            generation_method=excluded.generation_method,
            generated_at=CURRENT_TIMESTAMP
    """, (
        session_id,
        candidate_id,
        result["integrity_score"],
        result["risk_label"],
        result["summary_text"],
        result["generation_method"],
    ))

    connection.commit()
    connection.close()

    return result


def get_session_report(session_id):
    """
    Returns the stored integrity report row for a session as a dict,
    or None if no report has been generated yet.
    """
    connection = _get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT integrity_score, risk_label, summary_text,
               generation_method, generated_at
        FROM integrity_reports
        WHERE session_id=?
    """, (session_id,))

    row = cursor.fetchone()
    connection.close()

    if not row:
        return None

    return {
        "integrity_score": row[0],
        "risk_label": row[1],
        "summary_text": row[2],
        "generation_method": row[3],
        "generated_at": row[4],
    }