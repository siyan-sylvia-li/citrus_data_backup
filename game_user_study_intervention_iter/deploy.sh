#!/usr/bin/env bash
# Deploy the INTERVENTION-ITERATION study (prompt task -> LLM-judge filter ->
# intervention form -> Othello + AI chat) to Cloud Run, with participant data persisted
# to a GCS bucket mounted at /app/recordings (gcsfuse).
#
# This is a SEPARATE deployment from phase 2, with its own service and its own bucket.
# The service name is what keeps the two apart -- see the guard below.
#
# Usage (reads OPENAI_API_KEY / PROLIFIC_CODE / SCREENOUT_*_CODE from ./.env if present):
#   PROJECT_ID=my-proj ./deploy.sh
#
# Re-run anytime to redeploy; the bucket (and all collected data) is left intact.
set -euo pipefail

# Pick up secrets from .env if present (OPENAI_API_KEY, PROLIFIC_CODE, ...).
if [ -f .env ]; then set -a; source .env; set +a; fi

# ---- configuration (override via environment) -------------------------------
PROJECT_ID="${PROJECT_ID:-citrus-506513}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-game-study-intervention-iter}"
BUCKET="${BUCKET:-${PROJECT_ID}-${SERVICE}-data}"   # holds participant data
MOUNT_PATH="/app/recordings"

# This directory is a copy of game_user_study_phase_2, so the phase-2 service name is one
# stale .env away. Deploying it from here would overwrite the phase-2 service AND write
# iteration participants into the phase-2 bucket, where nothing distinguishes them.
if [ "$SERVICE" = "game-study-phase-2" ]; then
  echo "Refusing to deploy: SERVICE is 'game-study-phase-2', which is the phase-2" >&2
  echo "deployment. Unset SERVICE (or set it to something else) and re-run." >&2
  exit 2
fi

OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY (in .env or env)}"
# Together API key for the Qwen/Llama judge models in the JudgeSuite panel.
TOGETHER_API_KEY="${TOGETHER_API_KEY:?set TOGETHER_API_KEY (in .env or env)}"
PROLIFIC_CODE="${PROLIFIC_CODE:?set PROLIFIC_CODE (in .env or env)}"
# Three screen-out codes: #1 skilled Othello players, #2 low-quality prompts,
# #3 in-game gate (confident + correct unaided first move).
SCREENOUT_SKILL_CODE="${SCREENOUT_SKILL_CODE:?set SCREENOUT_SKILL_CODE (in .env or env)}"
SCREENOUT_PROMPT_CODE="${SCREENOUT_PROMPT_CODE:?set SCREENOUT_PROMPT_CODE (in .env or env)}"
SCREENOUT_INGAME_CODE="${SCREENOUT_INGAME_CODE:?set SCREENOUT_INGAME_CODE (in .env or env)}"
# Screen-out #4: no genuine engagement with the pre-task step (both arms). Falls back to
# the prompt code so an unset value does not break deploy; set it to tell them apart.
SCREENOUT_EFFORT_CODE="${SCREENOUT_EFFORT_CODE:-$SCREENOUT_PROMPT_CODE}"
EFFORT_PASS_THRESHOLD="${EFFORT_PASS_THRESHOLD:-2.0}"
JUDGE_PASS_RULE="${JUDGE_PASS_RULE:-mean}"
INTERVENTION_VERSION="${INTERVENTION_VERSION:-v9_reasons}"   # which TABLE, within contrasting_cases
# How many words of each assistant reply the screenshots show before "... [N more words]".
# The replies are stored in full, so this is display only. Kept short on purpose: the
# participant messages are the material, and the reply is there to show what each style
# pulled back -- not to be read.
INTERVENTION_REPLY_WORDS="${INTERVENTION_REPLY_WORDS:-5}"
# Which intervention FORM(s). INTERVENTION_FORM pins one and wins over LIVE_FORMS;
# LIVE_FORMS is the set to randomise over per participant. Leave both empty to randomise
# over every registered form except `skeleton`. Whatever is resolved is printed at
# container start as "[study] intervention forms in play: [...]".
# NOTE the single dash: "${VAR-default}" defaults only when VAR is UNSET, whereas
# "${VAR:-default}" also fires when VAR is set to the EMPTY string -- which meant
# `INTERVENTION_FORM= ./deploy.sh` silently re-pinned johnny and collapsed a
# three-arm run to two arms. Pass INTERVENTION_FORM= to randomise over LIVE_FORMS.
#
# Pinned to contrasting_cases because INTERVENTION_VERSION above only has any effect
# inside that form -- it selects which contrasting-case TABLE is shown. With any other
# form pinned, v9_reasons would be set and never rendered.
INTERVENTION_FORM="${INTERVENTION_FORM-contrasting_cases}"
LIVE_FORMS="${LIVE_FORMS:-}"
# Whether a no-intervention control runs alongside the form(s) above. MUST be forwarded to
# Cloud Run: app.py defaults it to 1, so a value set only in the deploying shell is read as
# "keep vanilla" inside the container -- which is how a v9-only run collected vanilla.
INCLUDE_VANILLA="${INCLUDE_VANILLA:-1}"
# gcloud's --set-env-vars argument is itself comma-delimited, so a comma inside a VALUE is
# read as the start of another variable -- and in practice the whole env block came back
# empty rather than erroring. app.py accepts ";" as a separator, so send it that way.
LIVE_FORMS="${LIVE_FORMS//,/;}"
# A stable secret keeps participant sessions valid across restarts.
FLASK_SECRET_KEY="${FLASK_SECRET_KEY:-$(openssl rand -hex 16)}"

# ---- study config (override via environment) --------------------------------
# Which puzzles run, and which of them has the AI, is defined in app.py as OTH_ROUNDS;
# that is structural, not an env var. What IS configurable is whether the two unassisted
# transfer puzzles run at all: this deployment drops them by default, so there is no
# transfer outcome and the readable endpoint is the manipulation check. Set to 0 for the
# full three-round protocol.
OTH_ASSISTED_ONLY="${OTH_ASSISTED_ONLY:-0}"
OTH_MIN_AI_TURNS="${OTH_MIN_AI_TURNS:-5}"          # messages required before the round can be submitted
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.5}"            # chat model the assistant uses
OTH_FIRST_N="${OTH_FIRST_N:-3}"                      # subscore window: how many opening decisions are also scored separately
# (puzzles are played to completion; there is no move quota)
OTH_TIME_LIMIT_SECONDS="${OTH_TIME_LIMIT_SECONDS:-480}"    # round 1 countdown (s); 480 = 8 min
# app.py reads OTH_TIME_LIMIT_TRANSFER, not OTH_TIME_LIMIT_SECONDS_2 -- the old name was
# passed to Cloud Run and silently ignored, so the transfer timer was always its default.
OTH_TIME_LIMIT_TRANSFER="${OTH_TIME_LIMIT_TRANSFER:-90}"   # each transfer puzzle (s), if they run
# External survey (AI-assistant assessment) shown after the AI-enabled round's TLX.
POST_SURVEY_FORM_URL="${POST_SURVEY_FORM_URL:-https://docs.google.com/forms/d/e/1FAIpQLSfXzbQo1w1e38pJobB9zbD5GsMQB-8elonTWfqCqJsw93Q2Qw/viewform?usp=pp_url&entry.1597274411=<PROLIFIC_ID>}"

# ---- Stage 1 prompt-filter config (override via environment) -----------------
# The judge panel models are fixed in llm_backends.panel_specs (Llama + Nemotron via
# Together, GPT via OpenAI); only the pass threshold is configurable here.
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
  --set-env-vars "RECORDINGS_DIR=${MOUNT_PATH},OPENAI_API_KEY=${OPENAI_API_KEY},TOGETHER_API_KEY=${TOGETHER_API_KEY},OPENAI_MODEL=${OPENAI_MODEL},FLASK_SECRET_KEY=${FLASK_SECRET_KEY},PROLIFIC_CODE=${PROLIFIC_CODE},SCREENOUT_SKILL_CODE=${SCREENOUT_SKILL_CODE},SCREENOUT_PROMPT_CODE=${SCREENOUT_PROMPT_CODE},SCREENOUT_INGAME_CODE=${SCREENOUT_INGAME_CODE},SCREENOUT_EFFORT_CODE=${SCREENOUT_EFFORT_CODE},EFFORT_PASS_THRESHOLD=${EFFORT_PASS_THRESHOLD},OTH_FIRST_N=${OTH_FIRST_N},OTH_TIME_LIMIT_SECONDS=${OTH_TIME_LIMIT_SECONDS},OTH_TIME_LIMIT_TRANSFER=${OTH_TIME_LIMIT_TRANSFER},POST_SURVEY_FORM_URL=${POST_SURVEY_FORM_URL},JUDGE_PASS_THRESHOLD=${JUDGE_PASS_THRESHOLD},JUDGE_PASS_RULE=${JUDGE_PASS_RULE},INTERVENTION_VERSION=${INTERVENTION_VERSION},INTERVENTION_REPLY_WORDS=${INTERVENTION_REPLY_WORDS},INTERVENTION_FORM=${INTERVENTION_FORM},LIVE_FORMS=${LIVE_FORMS},OTH_ASSISTED_ONLY=${OTH_ASSISTED_ONLY},INCLUDE_VANILLA=${INCLUDE_VANILLA},OTH_MIN_AI_TURNS=${OTH_MIN_AI_TURNS}"

echo
echo "Deployed [$SERVICE]. Participant data is written to gs://${BUCKET}/"
echo "  intervention form(s): ${INTERVENTION_FORM:-${LIVE_FORMS:-<all but skeleton>}}"
echo "  material / display  : ${INTERVENTION_VERSION}, assistant replies cut to ${INTERVENTION_REPLY_WORDS} words"
echo "  arms in play        : $([ "$INCLUDE_VANILLA" = "1" ] && echo "vanilla + ")${INTERVENTION_FORM:-${LIVE_FORMS:-<all but skeleton>}}"
echo "  transfer puzzles    : $([ "$OTH_ASSISTED_ONLY" = "1" ] && echo "OFF (assisted round only)" || echo "ON")"
echo "Download it anytime with:  PROJECT_ID=${PROJECT_ID} ./download_data.sh"
