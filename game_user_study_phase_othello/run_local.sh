#!/usr/bin/env bash
# Run the study LOCALLY for testing — reuses the same .env as deploy.sh, but
# starts the Flask dev server (http://localhost:5001) instead of deploying to
# Cloud Run. Every study setting falls back to the defaults baked into app.py,
# so the only thing you need in .env is OPENAI_API_KEY (for the chat assistant)
# and, if you want the prompt filter to actually score, TOGETHER_API_KEY.
#
# Override any setting inline for a quick test, e.g.:
#   OTH_TIME_LIMIT_SECONDS_2=30 ./run_local.sh     # 30s round-2 timer
#   PYTHON=.venv/bin/python ./run_local.sh          # use a specific interpreter
set -euo pipefail

# Load secrets / overrides from .env, exactly like deploy.sh does.
if [ -f .env ]; then set -a; source .env; set +a; fi

# Write participant data to a local folder (NOT the GCS mount used in prod).
export RECORDINGS_DIR="${RECORDINGS_DIR:-./recordings-local}"
# Boot even without real credentials. Chat needs a real OPENAI_API_KEY to reply;
# the prompt filter fails open (lets everyone through) without TOGETHER_API_KEY.
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local-dummy}"
# A fresh random key each run invalidates old session cookies, so restarting the
# server gives you a clean slate (back at the consent page) instead of resuming a
# half-finished session. Set FLASK_SECRET_KEY in .env to keep sessions across restarts.
export FLASK_SECRET_KEY="${FLASK_SECRET_KEY:-$(openssl rand -hex 16)}"
export POST_SURVEY_FORM_URL="${POST_SURVEY_FORM_URL:-https://docs.google.com/forms/d/e/1FAIpQLScnOMLT9MUPihhI0H63t-bKI3WHTNmmhQ71-tcAOk6JgZ7jKg/viewform?usp=pp_url&entry.1597274411=<PROLIFIC_ID>}"


# Default to the document_study_og venv (has the app's deps installed);
# override with PYTHON=/path/to/python ./run_local.sh
PYTHON="${PYTHON:-../document_study_og/.venv/bin/python}"
echo "Starting local study server on http://localhost:5001"
echo "  data dir : $RECORDINGS_DIR"
echo "  python   : $PYTHON   (override with PYTHON=/path/to/venv/bin/python)"
echo "  config   : all study settings default from app.py; override via .env or inline env vars"
exec "$PYTHON" app.py
