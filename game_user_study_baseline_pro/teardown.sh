#!/usr/bin/env bash
# Tear down the game study Cloud Run service.
#
# By default this deletes ONLY the Cloud Run service and LEAVES the data bucket
# (and all collected participant data) intact, so you can redeploy later.
#
# Usage:
#   PROJECT_ID=my-proj ./teardown.sh                 # delete service, keep data
#   PROJECT_ID=my-proj ./teardown.sh --delete-bucket # also delete the data bucket
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-citrus-user-study}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-game-study-baseline}"
BUCKET="${BUCKET:-${PROJECT_ID}-${SERVICE}-data}"

DELETE_BUCKET=0
for arg in "$@"; do
  case "$arg" in
    --delete-bucket) DELETE_BUCKET=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

gcloud config set project "$PROJECT_ID" >/dev/null

# ---- delete the Cloud Run service -------------------------------------------
if gcloud run services describe "$SERVICE" --region "$REGION" >/dev/null 2>&1; then
  echo "Deleting Cloud Run service [$SERVICE] in [$REGION] ..."
  gcloud run services delete "$SERVICE" --region "$REGION" --quiet
else
  echo "Service [$SERVICE] not found in [$REGION] — nothing to delete."
fi

# ---- optionally delete the data bucket --------------------------------------
if [[ "$DELETE_BUCKET" -eq 1 ]]; then
  if gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
    echo
    echo "!!  About to PERMANENTLY DELETE gs://${BUCKET} and ALL participant data in it."
    echo "!!  Download it first if unsure:  PROJECT_ID=${PROJECT_ID} ./download_data.sh"
    read -r -p "Type the bucket name to confirm deletion: " confirm
    if [[ "$confirm" == "$BUCKET" ]]; then
      gcloud storage rm -r "gs://${BUCKET}"
      echo "Deleted gs://${BUCKET}."
    else
      echo "Confirmation did not match — bucket left intact."
    fi
  else
    echo "Bucket gs://${BUCKET} not found — nothing to delete."
  fi
else
  echo
  echo "Data bucket gs://${BUCKET} left intact."
  echo "Re-run with --delete-bucket to remove it, or download data with:"
  echo "  PROJECT_ID=${PROJECT_ID} ./download_data.sh"
fi
