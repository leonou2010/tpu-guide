# Unit Test Report (VM Setup + Smoke Training)

Date: 2026-03-15 (America/New_York)

Goal: For each VM type (`v4`, `v6e_us`, `v6e_eu`), achieve 3 independent smoke-training successes and confirm a monotonic `*_train_loss.jsonl` artifact in GCS per run.

## Setup Packages (Saved)

Latest tarballs built locally under:

- `/home/kwokchunau/distributed_tpu_training/v3/setup_packages/dist/`

And uploaded to per-zone buckets under:

- `gs://gcp-researchcredits-blocklab-us-east1/pull_code_v3/setup_packages/`
- `gs://gcp-researchcredits-blocklab-europe-west4/pull_code_v3/setup_packages/`
- `gs://gcp-researchcredits-blocklab-1-us-central2/pull_code_v3/setup_packages/`

Most recent package build/upload timestamp: `20260315_044050`

## Findings So Far (Why Fresh VMs Failed)

1. `setup_packages/*/run.sh` previously embedded a broken inline heredoc (literal `\\n` sequences). This caused bash syntax errors after training, preventing `LOSS_RESULT` emission.
2. `setup_packages/*/run.sh` previously `mkdir -p $EXP_DIR` before calling the deploy script. The deploy scripts skip downloading `sf_bema` code if the experiment directory exists, so fresh VMs ended up with an empty `EXP_DIR` and training failed with `No such file or directory` for `train_tpu.py`.
3. A separate heredoc bug existed in the `loss_result.json` writer: the delimiter line was `PY >"/tmp/...json"` (delimiter must be alone on a line). This caused the JSON writer to fail even when training succeeded.

All three issues were patched in the local setup package sources and re-uploaded (see “Setup Packages” above).

## Proof: Successful Loss JSONL Was Generated (v6e_us)

Even before the `LOSS_RESULT` writer was fixed, the underlying smoke training on `v6e_us` successfully produced and uploaded per-step loss JSONL:

- `gs://gcp-researchcredits-blocklab-us-east1/checkpoints/unittest_v6eus_20260315_040957_r1_unittest-v6eus-03150323_chip0/adamw_ema_pullback_v4_lr0.001_bs1_lp0.01_clamp_w50_uf5_s42_train_loss.jsonl`
  - Lines:
    - `{"step": 1, "loss": 2.4764111042022705}`
    - `{"step": 2, "loss": 1.089448094367981}`

This confirms the training loop can run and produce a monotonic-step loss artifact in GCS when the environment is otherwise correct.

## Status

- `v6e_us`: 0/3 “fully green” (package finishes + LOSS_RESULT + JSONL monotonic) after fixes; rerun required because the spot test VM was deleted before re-test.
- `v6e_eu`: pending
- `v4`: pending

Next step: create fresh spot VMs per type and run the latest setup tarballs (`*_20260315_044050.tar.gz`) 3 times per type, capturing `LOSS_RESULT` JSON and verifying the referenced `gcs_train_loss_jsonl` is present and strictly increasing in `step`.

