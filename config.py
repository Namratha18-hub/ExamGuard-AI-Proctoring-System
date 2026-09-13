import os

# Base project directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Database location
DATABASE_PATH = os.path.join(BASE_DIR, "database", "examguard.db")

# Flask Secret Key
SECRET_KEY = "examguard_ai_secret_key_2026"

# ==================================================================
# >>> NEW IN PART 12 <
# ---------------- AI Integrity Report Agent (LangChain) ----------------
# Credentials are NEVER hardcoded here. OPENAI_API_KEY is read from the
# environment only -- set it as a real environment variable (or in a
# local, gitignored .env file loaded by whatever process starts the
# app) before running the server. If it's unset, utils/report_agent.py
# automatically uses its deterministic fallback summary instead of
# calling an LLM -- see that module's docstring for details. This is
# intentional: the app must keep working (report generation included)
# with zero LLM credentials configured.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Which chat model to request. Overridable via env var so the model
# can be changed without editing code; defaults to a small, cheap
# model since invigilator summaries are short and don't need a large
# model.
LLM_MODEL_NAME = os.environ.get("EXAMGUARD_LLM_MODEL", "gpt-4o-mini")

# Kept low and overridable -- an integrity report is meant to be a
# factual restatement of already-computed numbers (see
# utils/report_agent.py), not a creative one, so a low temperature is
# the sensible default.
LLM_TEMPERATURE = float(os.environ.get("EXAMGUARD_LLM_TEMPERATURE", "0.2"))
# <<< END NEW IN PART 12
# ==================================================================