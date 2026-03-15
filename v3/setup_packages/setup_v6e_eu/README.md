# setup_v6e_eu (ew4a)

Target: TPU v6e in `europe-west4-a` (e.g. `v6e-ew4a-*`).

This package:
- ensures dataset + model exist locally under the exp13 experiment folder
- runs a short single-chip smoke training
- uploads `*_train_loss.jsonl` to GCS and writes a `loss_result.json` record

