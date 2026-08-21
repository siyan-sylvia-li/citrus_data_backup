#!/usr/bin/env bash
# Deploy the NO-AI BASELINE arm of the CITRUS Othello study (prompt task ->
# LLM-judge filter -> Othello, no in-game assistant) to Cloud Run, with
# participant data persisted to a GCS bucket mounted at /app/recordings
# (gcsfuse).
#
# Deploys under its own SERVICE name, so it gets its own bucket and never mixes
# data with the AI arm in game_user_study_phase_othello.
#
# Usage (reads OPENAI_API_KEY / PROLIFIC_CODE / SCREENOUT_*_CODE from ./.env if present):
#   PROJECT_ID=my-proj ./deploy.sh
#
# Re-run anytime to redeploy; the bucket (and all collected data) is left intact.
set -euo pipefail

# Pick up secrets from .env if present (OPENAI_API_KEY, PROLIFIC_CODE, ...).
if [ -f .env ]; then set -a; source .env; set +a; fi

# ---- configuration (override via environment) -------------------------------
PROJECT_ID="${PROJECT_ID:-citrus-user-study}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-game-study-baseline-othello}"
BUCKET="${BUCKET:-${PROJECT_ID}-${SERVICE}-data}"   # holds participant data
MOUNT_PATH="/app/recordings"

OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY (in .env or env)}"
# Together API key for the Qwen/Llama judge models in the JudgeSuite panel.
TOGETHER_API_KEY="${TOGETHER_API_KEY:?set TOGETHER_API_KEY (in .env or env)}"
PROLIFIC_CODE="${PROLIFIC_CODE:?set PROLIFIC_CODE (in .env or env)}"
# Three screen-out codes: #1 skilled Othello players, #2 low-quality prompts,
# #3 in-game gate (confident + correct unaided first move).
SCREENOUT_SKILL_CODE="${SCREENOUT_SKILL_CODE:?set SCREENOUT_SKILL_CODE (in .env or env)}"
SCREENOUT_PROMPT_CODE="${SCREENOUT_PROMPT_CODE:?set SCREENOUT_PROMPT_CODE (in .env or env)}"
SCREENOUT_INGAME_CODE="${SCREENOUT_INGAME_CODE:?set SCREENOUT_INGAME_CODE (in .env or env)}"
# A stable secret keeps participant sessions valid across restarts.
FLASK_SECRET_KEY="${FLASK_SECRET_KEY:-$(openssl rand -hex 16)}"

# ---- study config (override via environment) --------------------------------
# The round design (which puzzles, in which order) is defined in app.py as
# OTH_ROUNDS; it is structural, not an env var. In this arm every round has
# "ai": False, which is what makes it the baseline.
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.5}"            # unused here (no in-game assistant); kept so the env matches the AI arm
OTH_FIRST_N="${OTH_FIRST_N:-3}"                      # subscore window: how many opening decisions are also scored separately
# (puzzles are played to completion; there is no move quota)
OTH_TIME_LIMIT_SECONDS="${OTH_TIME_LIMIT_SECONDS:-300}"    # round 1 countdown (s); 300 = 5 min
# NOTE: app.py reads OTH_TIME_LIMIT_TRANSFER, so that is the name we must export.
OTH_TIME_LIMIT_TRANSFER="${OTH_TIME_LIMIT_TRANSFER:-90}"   # each transfer puzzle's countdown (s)
# UNUSED in this arm: no round sets "ai_survey", so the external AI-assessment
# form is never shown. Kept so the two deploy scripts stay diffable.
POST_SURVEY_FORM_URL="${POST_SURVEY_FORM_URL:-}"

# ---- Stage 1 prompt-filter config (override via environment) -----------------
# The judge panel models are fixed in prompt_filter.py (Qwen + Llama via Together,
# GPT via OpenAI); only the pass threshold is configurable here.
JUDGE_PASS_THRESHOLD="${JUDGE_PASS_THRESHOLD:-3.0}" # median score (1-4) needed to pass

# ---- one-time setup (safe to re-run) ----------------------------------------
gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com storage.googleapis.com

# Create the data bucket if it doesn't exist yet.
if ! gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  echo "Creating data bucket gs://${BUCKET} ..."
  gcloud storage buckets create "gs://${BUCKET}" --location="$REGION" --uniform-bucket-level-access
fi

# Cloud Run source deploys build as the Compute Engine default service account,
# which needs the builder role (idempotent — re-running is a no-op).
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
BUILD_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
echo "Ensuring ${BUILD_SA} can run builds ..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${BUILD_SA}" \
  --role="roles/cloudbuild.builds.builder" \
  --condition=None >/dev/null

# ---- deploy -----------------------------------------------------------------
# 1 instance keeps each participant's files in a single process. Extra CPU/RAM
# gives the parallel prompt-judging headroom alongside the chat streaming.
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --min-instances 1 --max-instances 1 \
  --cpu 2 --memory 1Gi \
  --timeout 3600 \
  --add-volume "name=recordings,type=cloud-storage,bucket=${BUCKET}" \
  --add-volume-mount "volume=recordings,mount-path=${MOUNT_PATH}" \
  --set-env-vars "RECORDINGS_DIR=${MOUNT_PATH},OPENAI_API_KEY=${OPENAI_API_KEY},TOGETHER_API_KEY=${TOGETHER_API_KEY},OPENAI_MODEL=${OPENAI_MODEL},FLASK_SECRET_KEY=${FLASK_SECRET_KEY},PROLIFIC_CODE=${PROLIFIC_CODE},SCREENOUT_SKILL_CODE=${SCREENOUT_SKILL_CODE},SCREENOUT_PROMPT_CODE=${SCREENOUT_PROMPT_CODE},SCREENOUT_INGAME_CODE=${SCREENOUT_INGAME_CODE},OTH_FIRST_N=${OTH_FIRST_N},OTH_TIME_LIMIT_SECONDS=${OTH_TIME_LIMIT_SECONDS},OTH_TIME_LIMIT_TRANSFER=${OTH_TIME_LIMIT_TRANSFER},POST_SURVEY_FORM_URL=${POST_SURVEY_FORM_URL},JUDGE_PASS_THRESHOLD=${JUDGE_PASS_THRESHOLD}"

echo
echo "Deployed. Participant data is written to gs://${BUCKET}/"
echo "Download it anytime with:  gcloud storage cp -r gs://${BUCKET}/ ./recordings-download/"
