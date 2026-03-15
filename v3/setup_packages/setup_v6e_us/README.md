# setup_v6e_us (ue1d)

Target: TPU v6e in `us-east1-d` (e.g. `v6e-ue1d-*`).

This package:
- ensures dataset + model exist locally under the exp13 experiment folder
- runs a short single-chip smoke training
- uploads `*_train_loss.jsonl` to GCS and writes a `loss_result.json` record

Note: ue1d commonly fails if the experiment `data` directory is a broken symlink to `/scratch/...`.
This package forcibly replaces that with a real directory before downloading data.

