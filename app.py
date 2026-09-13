import random
import string
import sqlite3
import cv2
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    Response
)

from werkzeug.security import generate_password_hash, check_password_hash

from config import DATABASE_PATH, SECRET_KEY
from utils.face_detection import (
    generate_frames,
    capture_photo,
    stop_camera,
    reset_exam,
    start_camera_session,
    end_camera_session,
    get_face_detected
)
from utils import event_logger
from utils import detection_rules
from utils import scoring
from utils import report_agent
from utils import tab_switch_policy
import analytics
import os
from werkzeug.utils import secure_filename

# -----------------------------------
# FLASK APP
# -----------------------------------

app = Flask(__name__)
app.secret_key = "examguard_secret"
UPLOAD_FOLDER = "static/photos"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -----------------------------------
# SIDEBAR USER CONTEXT
# -----------------------------------
# Makes the logged-in candidate's name and profile picture available
# to every template that extends dashboard_base.html (dashboard,
# my-scores, profile, etc.) without having to pass it from each
# individual route. Reads only the existing `fullname` and
# `photo_path` columns -- no schema change, no other logic touched.

@app.context_processor
def inject_sidebar_user():

    if "candidate_id" not in session:
        return {"sidebar_user_name": None, "sidebar_user_photo": None}

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT fullname, photo_path FROM candidates WHERE candidate_id=?
    """, (session["candidate_id"],))

    row = cursor.fetchone()

    connection.close()

    if not row:
        return {"sidebar_user_name": None, "sidebar_user_photo": None}

    return {"sidebar_user_name": row[0], "sidebar_user_photo": row[1]}


# -----------------------------------
# NO-CACHE HEADERS FOR AUTHENTICATED PAGES
# -----------------------------------
# Prevents the browser (and its Back/Forward cache) from serving a
# stale copy of an authenticated page -- like /dashboard or /profile
# -- after logout. Without this, pressing Back after logging out can
# show the old page from cache without a fresh request ever reaching
# the server's session check. Static assets (CSS/JS/images) are left
# cacheable as normal since they contain no user-specific data.

@app.after_request
def set_no_cache_headers(response):

    if not request.path.startswith("/static/"):

        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response



# -----------------------------------
# APPLICATION ID GENERATOR
# -----------------------------------

def generate_application_id():
    characters = string.ascii_uppercase + string.digits
    random_part = ''.join(random.choices(characters, k=6))
    return "EG2026" + random_part


# -----------------------------------
# HOME
# -----------------------------------

@app.route("/")
def home():
    return render_template("home.html")


# -----------------------------------
# REGISTER
# -----------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":
 

        fullname = request.form["fullname"]
        email = request.form["email"]
        mobile = request.form["mobile"]
        college = request.form["college"]
        rollnumber = request.form["rollnumber"]

        department = request.form["department"]

        if department == "Others":
            department = request.form.get("otherBranch")

        password = request.form["password"]
        hashed_password = generate_password_hash(password)
    
        try:
            connection = sqlite3.connect(DATABASE_PATH)
            cursor = connection.cursor()

            cursor.execute("""
    INSERT INTO candidates
    (
        fullname,
        email,
        mobile,
        college,
        rollnumber,
        department,
        password
    )
    VALUES (?, ?, ?, ?, ?, ?, ?)
""", (
    fullname,
    email,
    mobile,
    college,
    rollnumber,
    department,
    hashed_password
    ))
     
            candidate_id = cursor.lastrowid

            application_id = f"EG2026{candidate_id:04d}"

            cursor.execute("""
                UPDATE candidates
                SET application_id=?
                WHERE candidate_id=?
            """, (
                application_id,
                candidate_id
            ))

            connection.commit()
            connection.close()

            flash(
                f"Registration Successful! Your Application ID is {application_id}. Please Login."
            )

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:

            flash("Email or Roll Number already exists!")

            return redirect(url_for("register"))

    return render_template("register.html")


# -----------------------------------
# LOGIN
# -----------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        application_id = request.form["application_id"]
        password = request.form["password"]

        connection = sqlite3.connect(DATABASE_PATH)
        cursor = connection.cursor()

        cursor.execute("""
            SELECT candidate_id, application_id, password
            FROM candidates
            WHERE application_id=?
        """, (application_id,))

        user = cursor.fetchone()

        connection.close()

        if user:

            if check_password_hash(user[2], password):

                session["candidate_id"] = user[0]
                session["application_id"] = user[1]

                return redirect(url_for("dashboard"))

        flash("Invalid Application ID or Password")

    return render_template("login.html")


# -----------------------------------
# ADMIN LOGIN
# -----------------------------------
# Separate route, separate credential check, separate destination --
# does not touch student authentication, the candidates table, or the
# session keys ("candidate_id"/"application_id") the student login
# flow relies on above. Admin credentials are intentionally NOT
# stored in the database (no new table, no schema change) -- they
# come from environment variables, with a fallback default ONLY for
# local development. Set ADMIN_USERNAME / ADMIN_PASSWORD (or
# ADMIN_PASSWORD_HASH for a hashed value, checked via the same
# check_password_hash already used for student login) before running
# in anything other than a local dev environment.
#
# The Streamlit invigilator dashboard (streamlit_dashboard.py) is a
# separate process/server with its own dependencies (see
# requirements-streamlit.txt) -- Flask cannot "open" it directly, so
# a successful admin login redirects the browser to wherever it's
# already running.

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH")
ADMIN_PASSWORD_FALLBACK = os.environ.get("ADMIN_PASSWORD", "admin123")  # dev-only default
STREAMLIT_DASHBOARD_URL = os.environ.get("STREAMLIT_DASHBOARD_URL", "http://localhost:8501")


def _is_streamlit_running(base_url, timeout_seconds=1.5):
    """
    Confirms the Streamlit dashboard is actually reachable before
    Flask redirects an admin to it -- otherwise a correct login still
    lands on the browser's own "This site can't be reached" page,
    which looks like a Flask/login bug rather than what it actually
    is (a second process that simply isn't running yet).

    Tries Streamlit's built-in health-check endpoint, which has lived
    at two different paths across Streamlit versions:
      - /_stcore/health  (current)
      - /healthz         (older releases)
    Uses only the standard library (urllib) -- no new dependency for
    this one check, and no dependency shared with the separate
    Streamlit venv.
    """
    import urllib.request
    import urllib.error

    base_url = base_url.rstrip("/")

    for health_path in ("/_stcore/health", "/healthz"):
        try:
            with urllib.request.urlopen(base_url + health_path, timeout=timeout_seconds) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            continue

    return False


@app.route("/admin-login", methods=["POST"])
def admin_login():
    username = request.form.get("username", "")
    password = request.form.get("password", "")


    valid = False

    if username == ADMIN_USERNAME:
        if ADMIN_PASSWORD_HASH:
            valid = check_password_hash(ADMIN_PASSWORD_HASH, password)
        else:
            valid = (password == ADMIN_PASSWORD_FALLBACK)

    if not valid:
        flash("Invalid Admin Username or Password")
        return redirect(url_for("login"))

    if _is_streamlit_running(STREAMLIT_DASHBOARD_URL):
        return redirect(STREAMLIT_DASHBOARD_URL)

    flash(
        "Admin login successful, but the Streamlit dashboard isn't "
        "running yet. Start it first: activate venv-dashboard, then "
        "run \"streamlit run streamlit_dashboard.py\" -- then log in "
        "again."
    )
    return redirect(url_for("login"))


# -----------------------------------
# LOGOUT
# -----------------------------------
# Minimal session-clearing route -- required for the dashboard
# sidebar's Logout link to function. Does not modify login/register
# logic in any way.

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))

# -----------------------------------
# DASHBOARD
# -----------------------------------

# Display names for course slugs used across instructions/exam routes.
# Only used for presentation on the dashboard -- does not affect
# question lookup, scoring, or exam_results storage.
COURSE_NAMES = {
    "java": "Java Programming",
    "python": "Python Programming",
    "dsa": "Data Structures",
    "dbms": "DBMS",
    "os": "Operating Systems",
    "cn": "Computer Networks",
    "ai": "Artificial Intelligence",
    "ml": "Machine Learning"
}

TOTAL_MARKS_PER_EXAM = 20


@app.route("/dashboard")
def dashboard():

    if "candidate_id" not in session:
        return redirect(url_for("login"))

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    # Candidate Details
    cursor.execute("""
        SELECT fullname, application_id
        FROM candidates
        WHERE candidate_id=?
    """, (session["candidate_id"],))

    candidate = cursor.fetchone()

    # All Exam Results (most recent first)
    cursor.execute("""
        SELECT course, marks
        FROM exam_results
        WHERE candidate_id=?
        ORDER BY submitted_at DESC
    """, (session["candidate_id"],))

    exam_results = cursor.fetchall()

    # -----------------------------------
    # FIX: Completed Exams must count DISTINCT courses, not every
    # attempt row. Counting every row (including retakes of the same
    # course) let "completed" exceed the hardcoded total_exams=8,
    # which drove "Exams Remaining" negative (observed: -83 after 91
    # total attempt rows across repeated testing).
    # -----------------------------------
    cursor.execute("""
        SELECT COUNT(DISTINCT course)
        FROM exam_results
        WHERE candidate_id=?
    """, (session["candidate_id"],))

    completed = cursor.fetchone()[0]

    total_exams = 8
    remaining = total_exams - completed

    connection.close()

    # -----------------------------------
    # MOST RECENT COMPLETED EXAM (real data)
    # -----------------------------------
    # exam_results is already ordered by submitted_at DESC, so the
    # first row (if any) is the candidate's most recently completed
    # exam. Nothing here is hardcoded -- it's built entirely from the
    # candidate's own exam_results rows.

    last_exam = None

    if exam_results:

        last_course_slug, last_marks = exam_results[0]

        last_exam = {
            "name": COURSE_NAMES.get(
    last_course_slug,
    (last_course_slug or "Unknown Course").replace("_", " ").title()
),
            "marks": last_marks,
            "total": TOTAL_MARKS_PER_EXAM
        }

    # -----------------------------------
    # AVERAGE SCORE (real data)
    # -----------------------------------
    # Computed from the candidate's own exam_results rows only.
    # None (not 0) when the candidate hasn't completed any exam yet,
    # so the template can show "No Data Yet" instead of a false 0%.

    average_score = None

    if exam_results:

        total_marks_scored = sum(marks for course, marks in exam_results)
        average_score = round(
            (total_marks_scored / len(exam_results)) / TOTAL_MARKS_PER_EXAM * 100
        )

    # -----------------------------------
    # EXAM INTEGRITY (Milestone 3/5 -- real, live data)
    # -----------------------------------
    # Reuses the SAME analytics.session_detail() the Streamlit
    # invigilator dashboard already calls (see streamlit_dashboard.py)
    # -- not a second implementation of scoring/charts/clustering.
    # Finds the candidate's most recently FINISHED (completed or
    # terminated) exam session via the existing `sessions` table, and
    # pulls its full integrity detail for display here. dashboard.html
    # already has the entire presentation for this built and waiting
    # -- it just never received these two variables before now.

    integrity = None
    integrity_setup_error = False

    try:
        integrity_connection = sqlite3.connect(DATABASE_PATH)
        integrity_cursor = integrity_connection.cursor()

        integrity_cursor.execute("""
            SELECT session_id
            FROM sessions
            WHERE candidate_id=?
              AND status IN ('completed', 'terminated')
            ORDER BY start_time DESC
            LIMIT 1
        """, (session["candidate_id"],))

        latest_session_row = integrity_cursor.fetchone()
        integrity_connection.close()

        if latest_session_row:
            integrity = analytics.session_detail(latest_session_row[0])

            if not integrity.get("found", True):
                integrity = None

    except sqlite3.OperationalError:
        # A Part 11/13 migration (face_presence_ratio column,
        # sessions.course column) hasn't been run against this
        # database yet -- dashboard.html already has a dedicated
        # "Setup incomplete" message for exactly this case.
        integrity_setup_error = True
        integrity = None

    return render_template(
        "dashboard.html",
        candidate=candidate,
        application_id=candidate[1],
        last_exam=last_exam,
        remaining=remaining,
        completed=completed,
        average_score=average_score,
        integrity=integrity,
        integrity_setup_error=integrity_setup_error
    )

# -----------------------------------
# MY SCORES
# -----------------------------------
# Full history of every exam the logged-in candidate has completed.
# Reuses the existing exam_results table only -- no schema change.

@app.route("/my-scores")
def my_scores():

    if "candidate_id" not in session:
        return redirect(url_for("login"))

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT course, marks, submitted_at
        FROM exam_results
        WHERE candidate_id=?
        ORDER BY submitted_at DESC
    """, (session["candidate_id"],))

    rows = cursor.fetchall()

    connection.close()

    # Build a clean, presentation-ready list from real DB rows only.
    # Nothing here is hardcoded -- name/marks/date all come straight
    # from the candidate's own exam_results rows.

    scores = []

    for course_slug, marks, submitted_at in rows:

        percentage = round((marks / TOTAL_MARKS_PER_EXAM) * 100)

        scores.append({
            "name": COURSE_NAMES.get(
    course_slug,
    (course_slug or "Unknown Course").replace("_", " ").title()
),
           
            "marks": marks,
            "total": TOTAL_MARKS_PER_EXAM,
            "percentage": percentage,
            "date": submitted_at
        })

    return render_template(
        "my_scores.html",
        scores=scores
    )


# -----------------------------------
# PROFILE
# -----------------------------------
# Read-only profile view of the logged-in candidate's own stored
# details, plus their profile picture path. No new columns, no
# schema change -- `photo_path` already existed on `candidates` from
# the original schema, it just was never populated by any route
# until the upload endpoint below.

ALLOWED_PROFILE_PICTURE_EXTENSIONS = {"jpg", "jpeg", "png"}


@app.route("/profile")
def profile():

    if "candidate_id" not in session:
        return redirect(url_for("login"))

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT fullname, email, mobile, college, rollnumber, department, application_id, photo_path
        FROM candidates
        WHERE candidate_id=?
    """, (session["candidate_id"],))

    row = cursor.fetchone()

    connection.close()

    candidate_profile = {
        "fullname": row[0],
        "email": row[1],
        "mobile": row[2],
        "college": row[3],
        "rollnumber": row[4],
        "department": row[5],
        "application_id": row[6],
        "photo_path": row[7]
    }

    return render_template(
        "profile.html",
        profile=candidate_profile
    )


# -----------------------------------
# PROFILE PICTURE UPLOAD
# -----------------------------------
# Saves the file as static/photos/profile_<candidate_id>.<ext> so
# each candidate has their own picture (unlike the shared
# candidate.jpg used during registration face capture), and stores
# the relative path in the existing photo_path column.

@app.route("/profile/upload-picture", methods=["POST"])
def upload_profile_picture():

    if "candidate_id" not in session:
        return redirect(url_for("login"))

    candidate_id = session["candidate_id"]

    uploaded_file = request.files.get("profile_picture")

    if not uploaded_file or uploaded_file.filename == "":
        flash("Please choose a JPG, JPEG, or PNG file to upload.", "error")
        return redirect(url_for("profile"))

    filename = secure_filename(uploaded_file.filename)
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in ALLOWED_PROFILE_PICTURE_EXTENSIONS:
        flash("Only JPG, JPEG, and PNG files are allowed for the profile picture.", "error")
        return redirect(url_for("profile"))

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    # Best-effort cleanup of a previous picture saved with a
    # different extension, so switching formats doesn't leave orphan
    # files behind.
    cursor.execute("""
        SELECT photo_path FROM candidates WHERE candidate_id=?
    """, (candidate_id,))

    existing_path = cursor.fetchone()[0]

    if existing_path:
        existing_full_path = os.path.join("static", existing_path.replace("static/", "", 1))
        if os.path.exists(existing_full_path):
            try:
                os.remove(existing_full_path)
            except OSError:
                pass

    new_filename = f"profile_{candidate_id}.{extension}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], new_filename)

    uploaded_file.save(save_path)

    relative_path = f"photos/{new_filename}"

    cursor.execute("""
        UPDATE candidates
        SET photo_path=?
        WHERE candidate_id=?
    """, (relative_path, candidate_id))

    connection.commit()
    connection.close()

    flash("Profile picture updated successfully.", "success")

    return redirect(url_for("profile"))


# -----------------------------------
# PROFILE PICTURE REMOVAL
# -----------------------------------
# Deletes the candidate's saved picture file (if any) and clears
# photo_path back to NULL, so the default avatar shows again. Reuses
# the same photo_path column -- no schema change.

@app.route("/profile/remove-picture", methods=["POST"])
def remove_profile_picture():

    if "candidate_id" not in session:
        return redirect(url_for("login"))

    candidate_id = session["candidate_id"]

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT photo_path FROM candidates WHERE candidate_id=?
    """, (candidate_id,))

    existing_path = cursor.fetchone()[0]

    if existing_path:

        existing_full_path = os.path.join("static", existing_path.replace("static/", "", 1))

        if os.path.exists(existing_full_path):
            try:
                os.remove(existing_full_path)
            except OSError:
                pass

        cursor.execute("""
            UPDATE candidates
            SET photo_path=NULL
            WHERE candidate_id=?
        """, (candidate_id,))

        connection.commit()

        flash("Profile picture removed.", "success")

    connection.close()

    return redirect(url_for("profile"))


# -----------------------------------
# PROFILE DETAILS UPDATE
# -----------------------------------
# Lets the candidate edit fullname/mobile/college/department only.
# Email and roll number are intentionally never read from the form,
# so they cannot be changed here even if someone tampered with the
# request. Reuses the existing candidates columns -- no schema change.

@app.route("/profile/update", methods=["POST"])
def update_profile():

    if "candidate_id" not in session:
        return redirect(url_for("login"))

    candidate_id = session["candidate_id"]

    fullname = request.form.get("fullname", "").strip()
    mobile = request.form.get("mobile", "").strip()
    college = request.form.get("college", "").strip()
    department = request.form.get("department", "").strip()

    if not fullname or not mobile or not college or not department:
        flash("All fields are required.", "error")
        return redirect(url_for("profile"))

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        UPDATE candidates
        SET fullname=?, mobile=?, college=?, department=?
        WHERE candidate_id=?
    """, (fullname, mobile, college, department, candidate_id))

    connection.commit()
    connection.close()

    flash("Profile updated successfully.", "success")

    return redirect(url_for("profile"))


# -----------------------------------
# SETTINGS
# -----------------------------------
# Account/security settings page. Reuses the existing candidates
# table and werkzeug password hashing already used by login/register
# -- no new tables, no duplicate profile-detail editing (that stays
# on the Profile page).

@app.route("/settings")
def settings():

    if "candidate_id" not in session:
        return redirect(url_for("login"))

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT application_id, email
        FROM candidates
        WHERE candidate_id=?
    """, (session["candidate_id"],))

    row = cursor.fetchone()

    connection.close()

    account = {
        "application_id": row[0],
        "email": row[1]
    }

    return render_template(
        "settings.html",
        account=account
    )


# -----------------------------------
# CHANGE PASSWORD
# -----------------------------------

@app.route("/settings/change-password", methods=["POST"])
def change_password():

    if "candidate_id" not in session:
        return redirect(url_for("login"))

    candidate_id = session["candidate_id"]

    current_password = request.form.get("current_password", "")
    new_password = request.form.get("new_password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not current_password or not new_password or not confirm_password:
        flash("All password fields are required.", "error")
        return redirect(url_for("settings"))

    if len(new_password) < 6:
        flash("New password must be at least 6 characters long.", "error")
        return redirect(url_for("settings"))

    if new_password != confirm_password:
        flash("New password and confirmation do not match.", "error")
        return redirect(url_for("settings"))

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT password FROM candidates WHERE candidate_id=?
    """, (candidate_id,))

    stored_hash = cursor.fetchone()[0]

    if not check_password_hash(stored_hash, current_password):
        connection.close()
        flash("Current password is incorrect.", "error")
        return redirect(url_for("settings"))

    new_hash = generate_password_hash(new_password)

    cursor.execute("""
        UPDATE candidates
        SET password=?
        WHERE candidate_id=?
    """, (new_hash, candidate_id))

    connection.commit()
    connection.close()

    flash("Password updated successfully.", "success")

    return redirect(url_for("settings"))

 # -----------------------------------
# INSTRUCTIONS
# -----------------------------------

@app.route("/instructions/<course>")
def instructions(course):

    if "candidate_id" not in session:
        return redirect(url_for("login"))

    return render_template(
        "instructions.html",
        course=course
    )


# -----------------------------------
# EXAM
# -----------------------------------

@app.route("/exam/<course>")
def exam(course):

    if "candidate_id" not in session:
        return redirect(url_for("login"))

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM questions
        WHERE course = ?
    """, (course,))

    questions = cursor.fetchall()
    print(course)
    print(len(questions))

    connection.close()

    # -----------------------------------
    # MILESTONE 2: START A NEW EXAM SESSION
    # -----------------------------------
    # Creates a row in `sessions` for this attempt and binds the
    # camera/proctoring loop to it, so every camera-triggered and
    # browser-triggered violation logs against the same session_id.

    exam_session_id = event_logger.create_session(session["candidate_id"])

    session["exam_session_id"] = exam_session_id
    session["exam_course"] = course

    start_camera_session(exam_session_id)

    return render_template(
        "exam.html",
        course=course,
        questions=questions
    )
# -----------------------------------
# CAMERA
# -----------------------------------

@app.route("/camera")
def camera():
    return render_template("camera.html")


# -----------------------------------
# VIDEO FEED
# -----------------------------------

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
# -----------------------------------
# CAPTURE FACE
# -----------------------------------

from flask import jsonify

@app.route("/capture", methods=["POST"])
def capture():

    success = capture_photo()

    if success:
        return jsonify({
            "success": True,
            "message": "Face Captured Successfully"
        })

    return jsonify({
        "success": False,
        "message": "Camera Capture Failed"
    })


# -----------------------------------
# FACE STATUS (REGISTRATION ONLY)
# -----------------------------------
# Lets the /register page show a clean "Face detected" / "No face
# detected" status next to the Capture Face button. Has nothing to do
# with exam sessions, warnings, or scoring.

@app.route("/face_status")
def face_status():

    return jsonify({
        "face_detected": get_face_detected()
    })

from flask import jsonify

@app.route("/update_warning", methods=["POST"])
def update_warning():

    if "application_id" not in session or "exam_session_id" not in session:
        return jsonify({"success": False})

    exam_session_id = session["exam_session_id"]

    # -----------------------------------
    # MILESTONE 2: TYPED BROWSER EVENT LOGGING
    # -----------------------------------
    # exam.html now sends {event_type: "tab_switch"} or
    # {event_type: "fullscreen_exit"} in the request body. Falls back
    # to a generic label if no body is sent, so this route stays
    # backward compatible with any old caller.

    data = request.get_json(silent=True) or {}
    event_type = data.get("event_type", "browser_violation")

    event_logger.log_event(exam_session_id, event_type)

    # -----------------------------------
    # UNIFIED WARNING COUNT (camera + browser)
    # -----------------------------------

    result = detection_rules.evaluate_session(exam_session_id)

    # -----------------------------------
    # PART 5A: TAB-SWITCH TERMINATION POLICY
    # -----------------------------------
    # Overrides the "terminated" flag on top of the general unified
    # warning threshold above, per the specific tab-switch rule:
    #   - 2nd (or later) tab_switch  -> terminate immediately.
    #   - tab_switch_timeout (client-side 1-minute grace period
    #     expired while still away, sent from exam.html) -> terminate,
    #     as long as a real switch was actually logged first.
    # Does not change how tab_switch/tab_switch_timeout events are
    # logged or counted above -- both still flow through the same
    # event_logs table and the same warnings/deducted-marks totals.
    if event_type == "tab_switch" and tab_switch_policy.is_second_or_later_switch(exam_session_id):
        result["terminated"] = True

    elif event_type == "tab_switch_timeout" and tab_switch_policy.is_valid_timeout_termination(exam_session_id):
        result["terminated"] = True

    # Keep the candidates table in sync (legacy columns still used
    # elsewhere, e.g. potential dashboard/report views).
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        UPDATE candidates
        SET warning_count=?,
            deducted_marks=?
        WHERE application_id=?
    """,
    (
        result["total_warnings"],
        result["deducted_marks"],
        session["application_id"]
    ))

    connection.commit()
    connection.close()

    if result["terminated"]:
        event_logger.end_session(exam_session_id, status="terminated")

        # -----------------------------------
        # MILESTONE 3: SAVE INTEGRITY SCORE
        # -----------------------------------
        # Session just ended (terminated by the rule engine), so this
        # is a good point to compute and persist the integrity score.
        # Does not affect warnings/marks/termination -- those are
        # already decided above by detection_rules.evaluate_session().
        scoring.save_integrity_score(exam_session_id, session["candidate_id"])

        # -----------------------------------
        # PART 12: SAVE AI INTEGRITY REPORT
        # -----------------------------------
        # Runs right after the integrity score above, using the same
        # data. Generates via LangChain if an LLM API key is
        # configured, otherwise a deterministic fallback -- either way
        # this does not affect warnings/marks/termination/scoring.
        report_agent.save_session_report(exam_session_id, session["candidate_id"])

    return jsonify({
        "warnings": result["total_warnings"],
        "deducted": result["deducted_marks"],
        "terminated": result["terminated"]
    })


# -----------------------------------
# SESSION STATUS (POLLING)
# -----------------------------------
# Lets exam.html periodically check the unified warning count, so
# camera-triggered terminations (which happen inside the video stream,
# not through a browser click) can still force a submit.

@app.route("/session_status")
def session_status():

    if "exam_session_id" not in session:
        return jsonify({"success": False})

    exam_session_id = session["exam_session_id"]

    result = detection_rules.evaluate_session(exam_session_id)

    return jsonify({
        "success": True,
        "warnings": result["total_warnings"],
        "deducted": result["deducted_marks"],
        "terminated": result["terminated"]
    })
@app.route("/submit_exam", methods=["POST"])
def submit_exam():

    if "candidate_id" not in session:
        return redirect(url_for("login"))

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    score = 0

    # Get all questions and correct answers
    cursor.execute("""
        SELECT question_id, answer
        FROM questions
    """)

    questions = cursor.fetchall()

    # Calculate score
    for question in questions:

        question_id = str(question[0])
        correct_answer = question[1]

        user_answer = request.form.get("q" + question_id)

        if user_answer == correct_answer:
            score += 1

    candidate_id = session["candidate_id"]

    # Update marks in candidates table
    cursor.execute("""
        UPDATE candidates
        SET marks = ?
        WHERE candidate_id = ?
    """, (score, candidate_id))

    # Get selected course
    course = request.form.get("course")

    # Save result
    cursor.execute("""
        INSERT INTO exam_results
        (candidate_id, course, marks)
        VALUES (?, ?, ?)
    """, (
        candidate_id,
        course,
        score
    ))

    connection.commit()
    connection.close()

    # -----------------------------------
    # MILESTONE 2: END THE EXAM SESSION
    # -----------------------------------
    # Only overwrite status to "completed" if the session wasn't
    # already marked "terminated" by the rule engine (camera or
    # browser side) -- termination should stay the recorded outcome.

    exam_session_id = session.get("exam_session_id")

    if exam_session_id:

        current_status = event_logger.get_session_status(exam_session_id)

        if current_status != "terminated":
            event_logger.end_session(exam_session_id, status="completed")

        # -----------------------------------
        # MILESTONE 3: SAVE INTEGRITY SCORE
        # -----------------------------------
        # Covers the normal submit path. If the session was already
        # terminated (and already scored in /update_warning), this is
        # a no-op re-save of the same numbers -- save_integrity_score()
        # is idempotent per session_id.
        scoring.save_integrity_score(exam_session_id, candidate_id)

        # -----------------------------------
        # PART 12: SAVE AI INTEGRITY REPORT
        # -----------------------------------
        # Covers the normal submit path, same idempotent-per-session_id
        # behaviour as save_integrity_score() above.
        report_agent.save_session_report(exam_session_id, candidate_id)

    end_camera_session()

    session.pop("exam_session_id", None)
    session.pop("exam_course", None)

    return render_template("success.html", score=score)
if __name__ == "__main__":
    app.run(debug=True)