# setup_v4 (uc2b)

Target: TPU v4 in `us-central2-b` (e.g. `v4-uc2b-*`).

This package:
- ensures dataset + model exist locally under the exp13 experiment folder
- (optionally) runs the existing v3 `deploy_uc2b.sh` for environment setup
- runs a short single-chip smoke training
- uploads `*_train_loss.jsonl` to GCS and writes a `loss_result.json` record

