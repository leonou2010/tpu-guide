#!/bin/bash
# GCS post-experiment cleanup
# Usage: bash cleanup_gcs.sh [--dry-run]
#
# KEEPS: XLA caches, model, data, results, wheels, pull_code, code bundles, pending/running task state
# DELETES: checkpoints (~459 GiB), coord_v2 history (heartbeats/telemetry/logs/completed/failed), old coord/
#
# Run after experiment completes (all tasks validated).
# Always do a dry-run first: bash cleanup_gcs.sh --dry-run

set -euo pipefail

DRY_RUN="${1:-}"
CTRL="gs://gcp-researchcredits-blocklab-europe-west4/coord_v2"
BUCKETS=(
  "gs://gcp-researchcredits-blocklab-europe-west4"
  "gs://gcp-researchcredits-blocklab-us-east1"
  "gs://gcp-researchcredits-blocklab-1-us-central2"
)

rm_path() {
  local path=$1
  if [ -n "$DRY_RUN" ]; then
    echo "  [dry-run] would delete: $path"
  else
    echo "  deleting: $path"
    gsutil -m rm -rf "$path" 2>/dev/null || true
  fi
}

echo "=== GCS Cleanup $(date -u '+%Y-%m-%d %H:%M UTC') ==="
echo "Mode: ${DRY_RUN:+DRY-RUN}${DRY_RUN:-LIVE}"
echo ""

echo "--- Sizes before cleanup ---"
for b in "${BUCKETS[@]}"; do
  printf "  %-55s " "$b"
  gsutil du -sh "$b" 2>/dev/null | awk '{print $1}' || echo "?"
done
echo ""

# ── 1. Checkpoints (biggest item: ~459 GiB across 3 buckets) ──────────────────
echo "--- Checkpoints (largest cost driver) ---"
for b in "${BUCKETS[@]}"; do
  rm_path "${b}/checkpoints/"
done

# ── 2. Coord_v2 historical state ───────────────────────────────────────────────
echo "--- coord_v2 history (heartbeats / telemetry / logs / completed / failed) ---"
rm_path "${CTRL}/heartbeats/"
rm_path "${CTRL}/telemetry/"
rm_path "${CTRL}/logs/"
rm_path "${CTRL}/completed/"
rm_path "${CTRL}/failed/"
# NOTE: pending/ and running/ are NOT deleted (needed if any experiment still active)

# ── 3. Old push-coordinator artifacts (replaced by pull system) ─────────────────
echo "--- Old push-coordinator state (coord/) ---"
for b in "${BUCKETS[@]}"; do
  rm_path "${b}/coord/"
done

echo ""
echo "--- Sizes after cleanup ---"
for b in "${BUCKETS[@]}"; do
  printf "  %-55s " "$b"
  gsutil du -sh "$b" 2>/dev/null | awk '{print $1}' || echo "?"
done

echo ""
echo "=== Done ==="
echo ""
echo "KEPT (do not delete):"
echo "  {bucket}/xla_cache_v6e_fresh/  — v6e XLA cache (version-locked, expensive to regenerate)"
echo "  {bucket}/xla_cache_v5e/        — v5e XLA cache"
echo "  {bucket}/xla_cache/            — v4 XLA cache"
echo "  {bucket}/models/               — SmolLM2-135M (needed for next experiment)"
echo "  {bucket}/data/                 — smoltalk training data"
echo "  {bucket}/results/              — validated experiment results"
echo "  {bucket}/wheels/, v4_wheels/   — pip wheels for VM setup"
echo "  {bucket}/pull_code/            — deployment scripts"
echo "  {bucket}/code/                 — training code bundles"
echo "  coord_v2/pending/, running/    — active task state"
