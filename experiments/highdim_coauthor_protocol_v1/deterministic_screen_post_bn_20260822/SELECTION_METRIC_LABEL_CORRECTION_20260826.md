# Selection-metric label correction — 2026-08-26

`effective_config.json` and `metrics.json` for every row in this screen
record `selection_metric` / `primary_selection_metric` as
`equal_client_validation_mse`. That label is wrong for this campaign.

Checkpoint selection is driven by `primary_val_mse` in `mse_by_round.csv`
(see `_validate_round_curve`'s `required_numeric` in
[`run_manifest.py`](../../../scripts/run_manifest.py) and `evaluate()` in
[`fedavg_api.py`](../../../fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/fedavg_api.py)),
which is a **pooled** validation MSE. `equal_client_val_mse` is populated only
for eICU runs, whose `val_global` is the only one that carries a `client_id`;
FEMNIST/CIFAR validation data does not, so an equal-client statistic could not
have been computed even if the label said otherwise.

**What this does and does not affect:**

- The numerical checkpoint selection performed by all 108 attempted screen
  rows was always correct (pooled validation MSE). No training rerun is
  required.
- Only the metadata label is wrong. Readers of `effective_config.json` /
  `metrics.json` for this screen should treat `selection_metric` /
  `primary_selection_metric` as `pooled_validation_mse`, not
  `equal_client_validation_mse`.
- `run_manifest.py`'s default for `primary_selection_metric` is corrected to
  `pooled_validation_mse` for future manifests (closeout plan §4.3). eICU
  manifests are unaffected: their generators
  (`prepare_eicu_study_a_manifest.py`, `prepare_eicu_study_a_v2_manifest.py`)
  already set `primary_selection_metric` explicitly per row rather than
  relying on this default.
