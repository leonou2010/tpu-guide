#!/bin/bash
# finish_experiment.sh — Post-experiment: show run report, then clean up GCS.
#
# Usage:
#   EXPERIMENTS="exp14:200" RESULTS_BASE=~/sf_bema/results \
#     bash ~/distributed_tpu_training/v2/finish_experiment.sh [--yes]
#
# Without --yes: shows report and prompts before cleanup.
# With --yes: runs cleanup automatically after report.

set -uo pipefail

RESULTS_BASE=${RESULTS_BASE:-$HOME/sf_bema/results}
GCLOUD=~/google-cloud-sdk/bin/gcloud
CTRL="gs://gcp-researchcredits-blocklab-europe-west4/coord_v2"

AUTO_YES="${1:-}"

if [ -z "${EXPERIMENTS:-}" ]; then
  echo "ERROR: EXPERIMENTS env var required (e.g. EXPERIMENTS=\"exp14:200\")"
  exit 1
fi

echo "================================================================"
echo "RUN REPORT — $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "================================================================"

# Parse EXPERIMENTS
declare -A EXP_TOTALS
EXP_NAMES=()
for spec in $EXPERIMENTS; do
  name="${spec%%:*}"
  total="${spec##*:}"
  EXP_NAMES+=("$name")
  EXP_TOTALS["$name"]="$total"
done

# Per-experiment validated count
all_done=true
for exp in "${EXP_NAMES[@]}"; do
  total=${EXP_TOTALS[$exp]:-0}
  validated_dir="${RESULTS_BASE}/${exp}/validated"
  val=$(ls "$validated_dir"/*.json 2>/dev/null | wc -l)
  if [ "$val" -ge "$total" ]; then
    status="✅ COMPLETE"
  else
    status="⏳ ${val}/${total}"
    all_done=false
  fi
  echo "  $exp: $status"
done

echo ""

# GCS queue counts
echo "GCS queue state:"
pending=$(gsutil ls "${CTRL}/pending/" 2>/dev/null | wc -l)
running=$(gsutil ls "${CTRL}/running/" 2>/dev/null | wc -l)
completed=$(gsutil ls "${CTRL}/completed/" 2>/dev/null | wc -l)
failed=$(gsutil ls "${CTRL}/failed/" 2>/dev/null | wc -l)
invalidated=$(gsutil ls "${CTRL}/invalidated/" 2>/dev/null | wc -l)
echo "  pending=$pending  running=$running  completed=$completed  failed=$failed  invalidated=$invalidated"

echo ""

# Bucket sizes before cleanup
echo "GCS storage before cleanup:"
for b in \
  gs://gcp-researchcredits-blocklab-europe-west4 \
  gs://gcp-researchcredits-blocklab-us-east1 \
  gs://gcp-researchcredits-blocklab-1-us-central2; do
  printf "  %-55s " "$b"
  gsutil du -sh "$b" 2>/dev/null | awk '{print $1}' || echo "?"
done

echo ""

if ! $all_done; then
  echo "WARNING: not all experiments complete. Cleanup will delete checkpoints."
  echo "Run again when validated counts reach target, or use --yes to force."
  echo ""
fi

if [ "$AUTO_YES" != "--yes" ]; then
  read -r -p "Proceed with GCS cleanup? [y/N] " answer
  [ "$answer" != "y" ] && [ "$answer" != "Y" ] && { echo "Aborted."; exit 0; }
fi

echo ""
echo "Running cleanup..."
bash "$(dirname "$0")/cleanup_gcs.sh"

# Clean up orphaned GCP health checks (GCP creates 5/VM, never auto-deletes them)
# These accumulate against HEALTH_CHECKS quota. Safe to delete all after experiment is done
# and all VMs are gone.
echo ""
echo "--- Cleaning up orphaned health checks ---"
GCLOUD=~/google-cloud-sdk/bin/gcloud
PROJECT=gcp-research-credits-489020
TOKEN=$($GCLOUD auth print-access-token 2>/dev/null)

hc_count=$($GCLOUD compute health-checks list \
  --project=$PROJECT --filter="name~^tpu-" --format="value(name)" 2>/dev/null | wc -l)
echo "Found $hc_count orphaned TPU health checks"

if [ "$hc_count" -gt 0 ] && [ -n "$TOKEN" ]; then
  # Step 1: delete health check services first (they reference health checks)
  # gcloud lacks 'health-check-services' subcommand — use REST API
  for region in europe-west4 us-east1 us-central2; do
    svc_count=$(curl -s -H "Authorization: Bearer $TOKEN" \
      "https://compute.googleapis.com/compute/v1/projects/${PROJECT}/regions/${region}/healthCheckServices" \
      2>/dev/null | python3 -c "
import sys,json; d=json.load(sys.stdin); items=d.get('items',[])
print(len(items))
for item in items: print(item['name'])
" 2>/dev/null | { read count; echo $count; while read svc; do
        curl -s -X DELETE \
          -H "Authorization: Bearer $TOKEN" \
          "https://compute.googleapis.com/compute/v1/projects/${PROJECT}/regions/${region}/healthCheckServices/${svc}" \
          >/dev/null 2>&1
        echo "  deleted service: $svc ($region)"
      done; })
  done
  sleep 5  # wait for service deletions to propagate

  # Step 2: delete health checks
  $GCLOUD compute health-checks list \
    --project=$PROJECT --filter="name~^tpu-" \
    --format="csv[no-heading](name,region)" 2>/dev/null | \
  while IFS=',' read -r hc_name region_url; do
    region="${region_url##*/}"
    $GCLOUD compute health-checks delete "$hc_name" \
      --region="$region" --project=$PROJECT --quiet 2>/dev/null && \
      echo "  deleted: $hc_name" || true
  done

  remaining=$($GCLOUD compute health-checks list \
    --project=$PROJECT --filter="name~^tpu-" --format="value(name)" 2>/dev/null | wc -l)
  echo "Health check cleanup done. Remaining: $remaining"
fi
