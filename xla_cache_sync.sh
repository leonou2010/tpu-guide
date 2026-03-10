#!/bin/bash
# xla_cache_sync.sh — Sync XLA compilation cache between VMs, GCS, and local server
#
# Usage:
#   bash ~/tpu_guide/xla_cache_sync.sh --pull-vm   TPU_NAME ZONE CHIP_FAMILY
#   bash ~/tpu_guide/xla_cache_sync.sh --push-vm   TPU_NAME ZONE CHIP_FAMILY
#   bash ~/tpu_guide/xla_cache_sync.sh --pull-gcs   CHIP_FAMILY
#   bash ~/tpu_guide/xla_cache_sync.sh --push-gcs   CHIP_FAMILY
#   bash ~/tpu_guide/xla_cache_sync.sh --status
#
# Local cache: ~/tpu_guide/xla_cache/{v4,v5e,v6e}/
# GCS cache:   gs://<bucket>/xla_cache/
# VM cache:    /tmp/xla_cache/

GCLOUD=~/google-cloud-sdk/bin/gcloud
GSUTIL=~/google-cloud-sdk/bin/gsutil
PROJECT=gcp-research-credits-489020
LOCAL_CACHE=~/tpu_guide/xla_cache

# Chip family → GCS buckets (all buckets that have this chip family)
declare -A CHIP_BUCKETS
CHIP_BUCKETS[v4]="gs://gcp-researchcredits-blocklab-1-us-central2"
CHIP_BUCKETS[v5e]="gs://gcp-researchcredits-blocklab-europe-west4 gs://gcp-researchcredits-blocklab-us-central1"
CHIP_BUCKETS[v6e]="gs://gcp-researchcredits-blocklab-europe-west4 gs://gcp-researchcredits-blocklab-us-east1"

MODE=${1:---status}

case $MODE in

  --pull-vm)
    # Pull XLA cache from a VM worker 0 to local server
    TPU_NAME=$2; ZONE=$3; CHIP=$4
    echo "=== Pulling XLA cache from $TPU_NAME (w0) → local $LOCAL_CACHE/$CHIP/ ==="
    mkdir -p /tmp/xla_cache_pull_$$
    $GCLOUD alpha compute tpus tpu-vm ssh "$TPU_NAME" \
      --zone="$ZONE" --project=$PROJECT --tunnel-through-iap \
      --worker=0 --command="tar czf - /tmp/xla_cache/ 2>/dev/null" > /tmp/xla_cache_pull_$$.tar.gz 2>/dev/null
    if [ -s /tmp/xla_cache_pull_$$.tar.gz ]; then
      tar xzf /tmp/xla_cache_pull_$$.tar.gz -C /tmp/xla_cache_pull_$$ --strip-components=2 2>/dev/null
      cp -n /tmp/xla_cache_pull_$$/* "$LOCAL_CACHE/$CHIP/" 2>/dev/null
      echo "Pulled $(ls /tmp/xla_cache_pull_$$/ 2>/dev/null | wc -l) files → $LOCAL_CACHE/$CHIP/"
    else
      echo "No cache found on $TPU_NAME"
    fi
    rm -rf /tmp/xla_cache_pull_$$ /tmp/xla_cache_pull_$$.tar.gz
    ;;

  --push-vm)
    # Push local cache to ALL workers of a VM
    TPU_NAME=$2; ZONE=$3; CHIP=$4
    count=$(ls "$LOCAL_CACHE/$CHIP/" 2>/dev/null | wc -l)
    echo "=== Pushing $count files from $LOCAL_CACHE/$CHIP/ → $TPU_NAME (all workers) ==="
    if [ "$count" -eq 0 ]; then
      echo "No local cache for $CHIP"
      exit 1
    fi
    # Upload to a temp GCS location, then pull on all workers
    TEMP_GCS="gs://gcp-researchcredits-blocklab-europe-west4/xla_cache_tmp"
    $GSUTIL -m cp "$LOCAL_CACHE/$CHIP/"* "$TEMP_GCS/" 2>/dev/null
    $GCLOUD alpha compute tpus tpu-vm ssh "$TPU_NAME" \
      --zone="$ZONE" --project=$PROJECT --tunnel-through-iap \
      --worker=all --command="mkdir -p /tmp/xla_cache && gcloud storage cp '$TEMP_GCS/*' /tmp/xla_cache/ 2>/dev/null; echo PUSH_OK_\$(ls /tmp/xla_cache/ | wc -l)" 2>/dev/null
    $GSUTIL -m rm "$TEMP_GCS/**" 2>/dev/null
    echo "Done."
    ;;

  --pull-gcs)
    # Pull from all GCS buckets for a chip family to local
    CHIP=$2
    echo "=== Pulling XLA cache from GCS → $LOCAL_CACHE/$CHIP/ ==="
    for bucket in ${CHIP_BUCKETS[$CHIP]}; do
      count=$($GSUTIL ls "$bucket/xla_cache/" 2>/dev/null | wc -l)
      if [ "$count" -gt 0 ]; then
        echo "  $bucket: $count files"
        $GSUTIL -m cp "$bucket/xla_cache/*" "$LOCAL_CACHE/$CHIP/" 2>/dev/null
      else
        echo "  $bucket: empty"
      fi
    done
    echo "Local: $(ls "$LOCAL_CACHE/$CHIP/" 2>/dev/null | wc -l) files"
    ;;

  --push-gcs)
    # Push local cache to ALL GCS buckets for a chip family
    CHIP=$2
    count=$(ls "$LOCAL_CACHE/$CHIP/" 2>/dev/null | wc -l)
    echo "=== Pushing $count files from $LOCAL_CACHE/$CHIP/ → all GCS buckets ==="
    if [ "$count" -eq 0 ]; then
      echo "No local cache for $CHIP"
      exit 1
    fi
    for bucket in ${CHIP_BUCKETS[$CHIP]}; do
      echo "  → $bucket/xla_cache/"
      $GSUTIL -m cp "$LOCAL_CACHE/$CHIP/"* "$bucket/xla_cache/" 2>/dev/null
    done
    echo "Done."
    ;;

  --status)
    echo "═══════════════════════════════════════════════════════"
    echo "  XLA CACHE STATUS — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    printf "%-8s %-10s %s\n" "CHIP" "LOCAL" "GCS"
    printf "%-8s %-10s %s\n" "----" "-----" "---"
    for chip in v4 v5e v6e; do
      local_count=$(ls "$LOCAL_CACHE/$chip/" 2>/dev/null | wc -l)
      gcs_info=""
      for bucket in ${CHIP_BUCKETS[$chip]}; do
        gcs_count=$($GSUTIL ls "$bucket/xla_cache/" 2>/dev/null | wc -l)
        region=$(echo "$bucket" | sed 's/.*blocklab-//' | sed 's/.*1-//')
        gcs_info="$gcs_info ${region}:${gcs_count}"
      done
      printf "%-8s %-10s %s\n" "$chip" "$local_count" "$gcs_info"
    done
    echo ""
    echo "Local cache: $LOCAL_CACHE/{v4,v5e,v6e}/"
    ;;

  *)
    echo "Usage: bash xla_cache_sync.sh [--pull-vm|--push-vm|--pull-gcs|--push-gcs|--status]"
    echo ""
    echo "  --pull-vm  TPU_NAME ZONE CHIP   Pull cache from VM worker 0 to local"
    echo "  --push-vm  TPU_NAME ZONE CHIP   Push local cache to all VM workers"
    echo "  --pull-gcs CHIP                 Pull from GCS buckets to local"
    echo "  --push-gcs CHIP                 Push local cache to all GCS buckets"
    echo "  --status                        Show cache counts"
    ;;
esac
