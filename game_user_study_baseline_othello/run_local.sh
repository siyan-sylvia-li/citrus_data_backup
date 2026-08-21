#!/usr/bin/env bash
# Run the NO-AI BASELINE arm LOCALLY for testing — reuses the same .env as
# deploy.sh, but starts the Flask dev server (http://localhost:5001) instead of
# deploying to Cloud Run. Every study setting falls back to the defaults baked
# into app.py. There is no in-game assistant here, so the only keys that matter
# are the prompt-filter judges': TOGETHER_API_KEY (Qwen + Llama) and
# OPENAI_API_KEY (GPT). Without them the filter fails open and lets everyone through.
#
# Override any setting inline for a quick test, e.g.:
#   OTH_TIME_LIMIT_TRANSFER=30 ./run_local.sh       # 30s transfer-puzzle timer
#   PYTHON=.venv/bin/python ./run_local.sh          # use a specific interpreter
set -euo pipefail

# Load secrets / overrides from .env, exactly like deploy.sh does.
if [ -f .env ]; then set -a; source .env; set +a; fi

# Write participant data to a local folder (NOT the GCS mount used in prod).
export RECORDINGS_DIR="${RECORDINGS_DIR:-./recordings-local}"
# Boot even without real credentials; the prompt filter fails open (lets
# everyone through) without TOGETHER_API_KEY / OPENAI_API_KEY.
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local-dummy}"
# A fresh random key each run invalidates old session cookies, so restarting the
# server gives you a clean slate (back at the consent page) instead of resuming a
# half-finished session. Set FLASK_SECRET_KEY in .env to keep sessions across restarts.
export FLASK_SECRET_KEY="${FLASK_SECRET_KEY:-$(openssl rand -hex 16)}"
# Unused in this arm (no round sets "ai_survey"), so the external form is never shown.
export POST_SURVEY_FORM_URL="${POST_SURVEY_FORM_URL:-}"


# Default to the document_study_og venv (has the app's deps installed);
# override with PYTHON=/path/to/python ./run_local.sh
PYTHON="${PYTHON:-../document_study_og/.venv/bin/python}"
echo "Starting local study server on http://localhost:5001"
echo "  data dir : $RECORDINGS_DIR"
echo "  python   : $PYTHON   (override with PYTHON=/path/to/venv/bin/python)"
echo "  config   : all study settings default from app.py; override via .env or inline env vars"
exec "$PYTHON" app.py
