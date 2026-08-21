#!/usr/bin/env bash
# Download collected participant data from the Cloud Run study's GCS bucket to a
# local directory. Safe to re-run — it syncs (only fetches new/changed files).
#
# Usage:
#   PROJECT_ID=my-proj ./download_data.sh
#   PROJECT_ID=my-proj DEST=./my-data ./download_data.sh
set -euo pipefail

# Same bucket-naming convention as deploy.sh.
PROJECT_ID="${PROJECT_ID:-citrus-user-study}"
SERVICE="${SERVICE:-game-study-baseline-othello}"
BUCKET="${BUCKET:-${PROJECT_ID}-${SERVICE}-data}"
DEST="${DEST:-./recordings-download}"

gcloud config set project "$PROJECT_ID" >/dev/null

if ! gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  echo "Bucket gs://${BUCKET} not found (check PROJECT_ID / SERVICE)." >&2
  exit 1
fi

mkdir -p "$DEST"
echo "Syncing gs://${BUCKET}/ -> ${DEST}/ ..."
gcloud storage rsync -r "gs://${BUCKET}" "$DEST"

echo
python score_from_logs.py
echo "Done. Participant data is in ${DEST}/"
