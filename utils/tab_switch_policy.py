"""
Part 5A - Real tab-switch detection policy.

Tab-switch behaviour:

1st tab switch:
    - The switch is recorded.
    - Candidate gets 60 seconds to return.
    - If they return within 60 seconds, exam continues.
    - If they remain away for more than 60 seconds, exam terminates.

2nd tab switch:
    - The switch is recorded.
    - Exam continues.

3rd tab switch:
    - The switch is recorded.
    - Exam terminates immediately.

Every tab switch is stored in the existing event_logs table.
"""

from utils.event_logger import get_event_counts


# =====================================
# CONFIGURATION
# =====================================

# Grace period for the FIRST tab switch only.
TAB_SWITCH_GRACE_SECONDS = 60

# Exam terminates when the candidate reaches the 3rd tab switch.
TAB_SWITCH_TERMINATION_LIMIT = 3


# =====================================
# TAB SWITCH COUNT
# =====================================

def get_tab_switch_count(session_id):
    """
    Return the number of tab_switch events recorded
    for this examination session.
    """

    event_counts = get_event_counts(session_id)

    return event_counts.get("tab_switch", 0)


# =====================================
# TERMINATION CHECK
# =====================================

def is_second_or_later_switch(session_id):
    """
    Despite the historical function name, this now checks
    whether the candidate has reached the 3rd tab switch.

    The current tab switch must already have been recorded
    before calling this function.

    Returns:
        True  -> terminate exam
        False -> continue exam
    """

    tab_switch_count = get_tab_switch_count(session_id)

    return tab_switch_count >= TAB_SWITCH_TERMINATION_LIMIT


# =====================================
# TIMEOUT VALIDATION
# =====================================

def is_valid_timeout_termination(session_id):
    """
    Validate the 60-second timeout termination.

    A timeout is valid only when at least one genuine
    tab_switch has already been recorded.
    """

    return get_tab_switch_count(session_id) >= 1