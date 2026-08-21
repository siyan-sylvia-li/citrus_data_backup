#!/usr/bin/env bash
# Deploy the CITRUS pre-filter study (prompt task -> LLM-judge filter -> Connect
# Four + AI chat) to Cloud Run, with participant data persisted to a GCS bucket
# mounted at /app/recordings (gcsfuse).
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
SERVICE="${SERVICE:-game-study-baseline}"
BUCKET="${BUCKET:-${PROJECT_ID}-${SERVICE}-data}"   # holds participant data
MOUNT_PATH="/app/recordings"

OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY (in .env or env)}"
# Together API key for the Qwen/Llama judge models in the JudgeSuite panel.
TOGETHER_API_KEY="${TOGETHER_API_KEY:?set TOGETHER_API_KEY (in .env or env)}"
PROLIFIC_CODE="${PROLIFIC_CODE:?set PROLIFIC_CODE (in .env or env)}"
# Three screen-out codes: #1 skilled Connect 4 players, #2 low-quality prompts,
# #3 in-game gate (confident + correct unaided first move).
SCREENOUT_SKILL_CODE="${SCREENOUT_SKILL_CODE:?set SCREENOUT_SKILL_CODE (in .env or env)}"
SCREENOUT_PROMPT_CODE="${SCREENOUT_PROMPT_CODE:?set SCREENOUT_PROMPT_CODE (in .env or env)}"
SCREENOUT_INGAME_CODE="${SCREENOUT_INGAME_CODE:?set SCREENOUT_INGAME_CODE (in .env or env)}"
# A stable secret keeps participant sessions valid across restarts.
FLASK_SECRET_KEY="${FLASK_SECRET_KEY:-$(openssl rand -hex 16)}"

# ---- study config (override via environment) --------------------------------
# The two-round design (which puzzles, and which round has the AI) is defined in
# app.py as CF_ROUNDS; it is structural, not an env var.
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.5}"            # chat model the assistant uses
CF_NUM_MOVES="${CF_NUM_MOVES:-3}"                  # how many moves the participant makes per puzzle
CF_TIME_LIMIT_SECONDS="${CF_TIME_LIMIT_SECONDS:-300}"    # round 1 / puzzle 15 countdown (s); 360 = 6 min
CF_TIME_LIMIT_SECONDS_2="${CF_TIME_LIMIT_SECONDS_2:-180}"  # round 2 / puzzle 6 countdown (s); 180 = 3 min
# External survey (AI-assistant assessment) shown after the AI-enabled round's TLX.
POST_SURVEY_FORM_URL="${POST_SURVEY_FORM_URL:-https://docs.google.com/forms/d/e/1FAIpQLScnOMLT9MUPihhI0H63t-bKI3WHTNmmhQ71-tcAOk6JgZ7jKg/viewform?usp=pp_url&entry.1597274411=<PROLIFIC_ID>}"

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
  --set-env-vars "RECORDINGS_DIR=${MOUNT_PATH},OPENAI_API_KEY=${OPENAI_API_KEY},TOGETHER_API_KEY=${TOGETHER_API_KEY},OPENAI_MODEL=${OPENAI_MODEL},FLASK_SECRET_KEY=${FLASK_SECRET_KEY},PROLIFIC_CODE=${PROLIFIC_CODE},SCREENOUT_SKILL_CODE=${SCREENOUT_SKILL_CODE},SCREENOUT_PROMPT_CODE=${SCREENOUT_PROMPT_CODE},SCREENOUT_INGAME_CODE=${SCREENOUT_INGAME_CODE},CF_NUM_MOVES=${CF_NUM_MOVES},CF_TIME_LIMIT_SECONDS=${CF_TIME_LIMIT_SECONDS},CF_TIME_LIMIT_SECONDS_2=${CF_TIME_LIMIT_SECONDS_2},POST_SURVEY_FORM_URL=${POST_SURVEY_FORM_URL},JUDGE_PASS_THRESHOLD=${JUDGE_PASS_THRESHOLD}"

echo
echo "Deployed. Participant data is written to gs://${BUCKET}/"
echo "Download it anytime with:  gcloud storage cp -r gs://${BUCKET}/ ./recordings-download/"
