#!/usr/bin/env bash
# Download collected participant data from the Cloud Run study's GCS bucket to a
# local directory. Safe to re-run — it syncs (only fetches new/changed files).
#
# Usage:
#   PROJECT_ID=my-proj ./download_data.sh
#   PROJECT_ID=my-proj DEST=./my-data ./download_data.sh
#   ANNOTATE=0 ./download_data.sh          # sync + score only, no LLM calls
set -euo pipefail

# The annotators call OpenAI/Together/Anthropic. annotate_phase2.py finds this .env on its
# own, but phase2_assistant_fine.py lives in ../bad_user_sim and looks for a .env THERE,
# which does not exist -- so export the keys into the environment and both are covered
# wherever they load from.
if [ -f .env ]; then set -a; source .env; set +a; fi

# Same bucket-naming convention as deploy.sh.
PROJECT_ID="${PROJECT_ID:-citrus-506513}"
SERVICE="${SERVICE:-game-study-intervention-iter}"
BUCKET="${BUCKET:-${PROJECT_ID}-${SERVICE}-data}"
DEST="${DEST:-./recordings-v9-vanilla}"

gcloud config set project "$PROJECT_ID" >/dev/null

if ! gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  echo "Bucket gs://${BUCKET} not found (check PROJECT_ID / SERVICE)." >&2
  exit 1
fi

mkdir -p "$DEST"
echo "Syncing gs://${BUCKET}/ -> ${DEST}/ ..."
gcloud storage rsync -r "gs://${BUCKET}" "$DEST"

echo
# One interpreter for every step: `python` alone is often absent, and score_from_logs.py
# needs the same deps as the annotators.
PYTHON="${PYTHON:-/home/sylviali/miniconda3/envs/citrus/bin/python}"
# Both annotators resolve --root against the CWD, and the assistant-side one runs from a
# sibling directory, so pass an absolute path rather than "$DEST".
DEST_ABS="$(cd "$DEST" && pwd)"

# Every downstream step must be pointed at THIS download. score_from_logs.py defaults to
# --dir recordings-download and annotate_phase2.py to --root recordings-download, so
# running them bare scored and annotated a different directory than the one just synced.
#
# Scores every round found on disk. With OTH_ASSISTED_ONLY=1 there are no transfer
# rounds, so its transfer_* columns come out empty -- that is the design, not a failure.
"$PYTHON" score_from_logs.py --dir "$DEST_ABS"

if [ "${ANNOTATE:-1}" = "1" ]; then
  # Two annotation passes over the same files. Both skip turns that already carry their
  # field, so re-running is cheap and only new participants cost anything.
  #
  # Participant-side first: it writes annotation_user, and with no transfer outcome those
  # acts ARE the endpoint the manipulation check reads.
  echo "Annotating participant turns ..."
  "$PYTHON" annotate_phase2.py --root "$DEST_ABS"

  # Assistant-side second: writes annotation_assistant_fine into the same annotated file,
  # leaving annotation_user untouched. Same fine panel as the phase-1 Othello study, so
  # prevalence sits on one scale across studies.
  echo "Annotating assistant turns ..."
  "$PYTHON" ../bad_user_sim/phase2_assistant_fine.py --root "$DEST_ABS"
else
  echo "ANNOTATE=0 -- skipping both dialogue-act passes."
fi

echo
echo "Done. Participant data is in ${DEST}/"
echo "  scores            : scores.csv"
echo "  participant acts  : annotation_user in annotated_conversation_*.jsonl"
echo "  assistant acts    : annotation_assistant_fine in the same files"
