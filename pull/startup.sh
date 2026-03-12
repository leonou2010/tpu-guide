#!/bin/bash
# startup.sh — Boot-time launcher for pull-based TPU workers.
# Runs on every VM boot (including after preemption restart).
# Delegates entirely to deploy_babysitter.sh — single source of truth.
#
# Embedded as metadata startup-script in vm_requester.sh VM creation.
set -uo pipefail

LOG=/tmp/startup.log
exec >> "$LOG" 2>&1
echo "[startup] $(date -u +%H:%M:%S) Starting on $(hostname)"

# ── Read instance metadata ──
META_INST="http://metadata.google.internal/computeMetadata/v1/instance"
get_meta() {
    curl -sf -H "Metadata-Flavor: Google" "${META_INST}/attributes/$1" 2>/dev/null || echo "${2:-}"
}

ZONE=$(curl -sf -H "Metadata-Flavor: Google" "${META_INST}/zone" 2>/dev/null | awk -F/ '{print $NF}' || echo "")
TPU_NAME=$(get_meta tpu_name "$(hostname)")
WANDB_MODE=$(get_meta wandb_mode "disabled")

case "${ZONE:-}" in
    europe-west4*) BUCKET="gs://gcp-researchcredits-blocklab-europe-west4" ;;
    us-east1*)     BUCKET="gs://gcp-researchcredits-blocklab-us-east1" ;;
    us-central2*)  BUCKET="gs://gcp-researchcredits-blocklab-1-us-central2" ;;
    us-central1*)  BUCKET="gs://gcp-researchcredits-blocklab-europe-west4" ;;
    *)             BUCKET="gs://gcp-researchcredits-blocklab-europe-west4" ;;
esac

echo "[startup] TPU_NAME=$TPU_NAME ZONE=${ZONE:-unknown} BUCKET=$BUCKET"

# ── Download and run deploy_babysitter.sh ──
# deploy_babysitter.sh handles: kill old processes, install packages,
# download model/code/XLA cache, launch single flock'd babysitter.
echo "[startup] Downloading deploy_babysitter.sh from GCS..."
gsutil cp "${BUCKET}/pull_code/deploy_babysitter.sh" /tmp/deploy_babysitter.sh 2>/dev/null || \
    gcloud storage cp "${BUCKET}/pull_code/deploy_babysitter.sh" /tmp/deploy_babysitter.sh 2>/dev/null || {
    echo "[startup] ERROR: Could not download deploy_babysitter.sh from $BUCKET"
    exit 1
}
chmod +x /tmp/deploy_babysitter.sh

echo "[startup] Running deploy_babysitter.sh..."
export TPU_NAME ZONE WANDB_MODE BUCKET
bash /tmp/deploy_babysitter.sh
echo "[startup] deploy_babysitter.sh completed (exit $?)"
