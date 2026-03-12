#!/bin/bash
# All-in-one TPU fleet monitor. Run: bash ~/distributed_tpu_training/vm_scan.sh
PROJECT=gcp-research-credits-489020

# Only scan zones with quota grants (skip zones with no capacity)
GRANTED_ZONES=(
  europe-west4-a    # 64 spot v6e
  us-east1-d        # 64 spot v6e
  us-central2-b     # 32 spot v4 + 32 on-demand v4
  us-central1-a     # 64 spot v5e
  europe-west4-b    # 64 spot v5e
)

# Also scan all other v6e zones for opportunistic VMs
OTHER_ZONES=(
  us-central1-a us-central1-b us-central1-c
  us-east4-a us-east5-a us-east5-b us-east5-c
  us-south1-a us-south1-c us-west1-c
  asia-east1-c asia-northeast1-b asia-south1-b asia-south1-c asia-southeast1-b
  southamerica-east1-c southamerica-west1-a
)

echo "================================================================"
echo "  TPU Fleet Monitor — $(date)"
echo "================================================================"
echo ""

total_chips=0
total_ready=0
total_creating=0
total_vms=0

# --- Granted Zones (priority) ---
echo "=== GRANTED ZONES (quota grants) ==="
echo ""
printf "  %-25s %-12s %-10s %-20s %s\n" "NAME" "TYPE" "STATUS" "ZONE" "CHIPS"
printf "  %-25s %-12s %-10s %-20s %s\n" "----" "----" "------" "----" "-----"

for zone in "${GRANTED_ZONES[@]}"; do
  result=$(~/google-cloud-sdk/bin/gcloud alpha compute tpus tpu-vm list \
    --zone=$zone --project=$PROJECT \
    --format='table[no-heading](name,acceleratorType,state)' 2>/dev/null)
  if [ -n "$result" ]; then
    while IFS= read -r line; do
      name=$(echo "$line" | awk '{print $1}')
      accel=$(echo "$line" | awk '{print $2}')
      state=$(echo "$line" | awk '{print $3}')
      chips=$(echo "$accel" | grep -oP '\d+$')
      printf "  %-25s %-12s %-10s %-20s %s\n" "$name" "$accel" "$state" "$zone" "$chips"
      if [ "$state" = "READY" ]; then
        total_ready=$((total_ready + chips))
      fi
      if [ "$state" = "CREATING" ]; then
        total_creating=$((total_creating + chips))
      fi
      total_chips=$((total_chips + chips))
      total_vms=$((total_vms + 1))
    done <<< "$result"
  fi
done

# --- Other Zones (opportunistic) ---
has_other=false
for zone in "${OTHER_ZONES[@]}"; do
  result=$(~/google-cloud-sdk/bin/gcloud alpha compute tpus tpu-vm list \
    --zone=$zone --project=$PROJECT \
    --format='table[no-heading](name,acceleratorType,state)' 2>/dev/null)
  if [ -n "$result" ]; then
    if [ "$has_other" = false ]; then
      echo ""
      echo "=== OTHER ZONES ==="
      echo ""
      printf "  %-25s %-12s %-10s %-20s %s\n" "NAME" "TYPE" "STATUS" "ZONE" "CHIPS"
      printf "  %-25s %-12s %-10s %-20s %s\n" "----" "----" "------" "----" "-----"
      has_other=true
    fi
    while IFS= read -r line; do
      name=$(echo "$line" | awk '{print $1}')
      accel=$(echo "$line" | awk '{print $2}')
      state=$(echo "$line" | awk '{print $3}')
      chips=$(echo "$accel" | grep -oP '\d+$')
      printf "  %-25s %-12s %-10s %-20s %s\n" "$name" "$accel" "$state" "$zone" "$chips"
      if [ "$state" = "READY" ]; then
        total_ready=$((total_ready + chips))
      fi
      if [ "$state" = "CREATING" ]; then
        total_creating=$((total_creating + chips))
      fi
      total_chips=$((total_chips + chips))
      total_vms=$((total_vms + 1))
    done <<< "$result"
  fi
done

echo ""
echo "================================================================"
echo "  SUMMARY: ${total_vms} VMs | ${total_ready} READY chips | ${total_creating} CREATING chips"
echo "================================================================"
echo ""

# --- Quota Reference ---
echo "=== QUOTA GRANTS (per-zone, as of 2026-03-09) ==="
echo ""
echo "  europe-west4-a:  64 spot v6e chips  (has internet, W&B works)"
echo "  us-east1-d:      64 spot v6e chips  (no internet, WANDB_MODE=disabled)"
echo "  us-central2-b:   32 spot v4 + 32 on-demand v4 chips  (no internet)"
echo "  us-central1-a:   64 spot v5e chips  (OOM fixed — bs halved automatically)"
echo "  europe-west4-b:  64 spot v5e chips  (OOM fixed — bs halved automatically)"
echo ""
echo "  Max usable: 64 + 64 + 64 + 64 + 64 = 320 chips"
echo ""

# --- Capacity Status (updated manually) ---
echo "=== ZONE CAPACITY STATUS ==="
echo ""
echo "  europe-west4-a:  Intermittent — v6e-8 sometimes works, v6e-16/32 often fail"
echo "  us-east1-d:      DEAD — zero v6e capacity (all sizes RESOURCE_EXHAUSTED)"
echo "  us-central2-b:   v4 spot quota stuck (ghost usage). On-demand may work."
echo ""
