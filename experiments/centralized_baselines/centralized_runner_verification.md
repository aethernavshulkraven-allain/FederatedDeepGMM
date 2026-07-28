# Centralized Runner Verification

This is a verification-only audit. No training was launched.

## 1. Centralized Method Mapping

| paper baseline | intended manifest method | verified launchable code method | status | note |
| --- | --- | --- | --- | --- |
| DeepGMM-GDA | `gda_d` | none | not implemented | The current launchable GDA-like method is `fedgda_d`, which runs through the federated FedAvgAPI path. |
| DeepGMM-SGDA / SGD(C) | `sgda_s` | none | not implemented | The current launchable stochastic GDA-like method is `fedgda_s`, also federated. |
| DeepGMM-OAdam | `oadam_s` | none | not implemented | `optimizers/oadam.py` exists, but OAdam is not wired into the active low-dimensional DeepGMM training dispatch. |
| deterministic OAdam extension | `oadam_d` | none | not implemented | This is an extension placeholder unless explicitly implemented and reported separately. |

Full mapping CSV: `experiments/centralized_baselines/centralized_method_mapping.csv`.

## 2. Entrypoint

There is no verified centralized DeepGMM entrypoint at present.

`scripts/run_manifest.py` is explicitly federated-only for launch selection. Its docstring says centralized rows are skipped until their runner is verified (`scripts/run_manifest.py:1-9`), and `select_rows` drops any row where `training_scope != "federated"` (`scripts/run_manifest.py:116-130`).

The usual `main.py --cf <config>` route is also not centralized. `main.py` constructs `FedMLRunner` and calls `run()` (`fedgmm/sp_decentralized_mnist_lr_example/main.py:11-26`). The runner dispatches simulation/SP execution to `SimulatorSingleProcess`, which then dispatches `FedAvg` to `FedAvgAPI` (`fedml/runner.py:36-81`, `fedml/simulation/simulator.py:27-67`).

The existing manifest centralized rows are placeholders. There are 48 centralized rows in `experiments/rerun_protocol_v1/manifest.csv`, covering `gda_d`, `sgda_s`, `oadam_d`, and `oadam_s` across `abs`, `step`, `linear`, and `sin` for seeds 0, 1, and 2. All 48 have:

```text
implementation_status = blocked_pending_true_centralized_runner_verification
run_status = blocked
```

The notes say not to launch until a true centralized DeepGMM/OAdam implementation is verified.

## 3. Config Fields

A future true centralized config should include:

| field | purpose |
| --- | --- |
| `training_scope=centralized` | Must select the centralized runner, not FedAvgAPI. |
| `method` / `variant` | One of `gda_d`, `sgda_s`, `oadam_s`; optionally `oadam_d` if implemented intentionally. |
| `dataset` | `abs`, `step`, `linear`, or `sin`. |
| `seed` / `random_seed` | Reproducible initialization and data split/partition recipe. |
| `output_dir` / `run_id` | Unique run artifact path with overwrite protection. |
| `iterations` or equivalent budget | Centralized optimization budget; should not be named as communication rounds unless explicitly mapped. |
| `epochs` / `batch_size` | Full-batch for deterministic GDA; minibatch for SGDA/OAdam if stochastic. |
| `g_learning_rate` / `f_learning_rate` | Direct centralized optimizer learning rates. |
| `optimizer` | `gda`, `sgda`, or `oadam`, not `federated_optimizer`. |
| `selection_metric_source=validation` | Best checkpoint must be validation-selected. |
| `test_mse_used_for_selection=false` | Test must not influence selection. |

Federated-only fields should be absent or ignored by the centralized runner:

```text
client_num_in_total
client_num_per_round
federated_optimizer
client_optimizer
server_learning_rate
partition_alpha for optimizer updates
client sampling settings
FedAvg aggregation settings
```

`partition_alpha` may still describe a data recipe if needed for fair comparison, but it must not drive federated client updates in a centralized baseline.

## 4. Training-Loop Verification

The active DeepGMM training loop is not true centralized.

Evidence:

- The data loader produces global splits and local client dictionaries for low-dimensional functions (`fedml/data/data_loader.py:369-421`).
- Model selection in `FedAvgAPI` does use global train/validation tensors (`fedavg_api.py:318-321`).
- After model selection, training creates `Client` objects from `train_data_local_dict` and `test_data_local_dict` (`fedavg_api.py:342-363`).
- Every communication round samples clients (`fedavg_api.py:399-413`).
- Each selected client trains on local data via `client.train(...)` and `client.train_reg(...)` (`fedavg_api.py:417-437`; `fedml/simulation/sp/fedavg/client.py:27-39`).
- Local weights are aggregated (`fedavg_api.py:440-444`).
- A server learning rate then updates global `g` and `f` from aggregated local deltas (`fedavg_api.py:446-493`).

This violates true centralized requirements because it uses clients, client sampling, local workers, FedAvg-style aggregation, and server learning-rate updates.

The generic `fedml/centralized/centralized_trainer.py` is also not sufficient. It trains a single classifier with `CrossEntropyLoss` and SGD/Adam (`centralized_trainer.py:9-46`, `centralized_trainer.py:48-64`). It is not wired from the low-dimensional DeepGMM `main.py` path and does not implement GDA/SGDA/OAdam for `g`/`f`.

Checkpoint selection in the current federated DeepGMM path is validation-only (`fedavg_api.py:534-547`), but that does not make the training loop centralized.

## 5. Artifact Verification

Required artifacts for each future centralized run should be:

```text
effective_config.json
metrics.json
mse_by_round.csv
predictions.npz
checkpoints/best_validation.pt
checkpoints/final.pt
test_mse_by_round.csv, if log_test_mse_by_round=true
```

The current `scripts/run_manifest.py` validator is shaped around the federated run directory and `variant` fields. It may be reusable in spirit, but a centralized validator should explicitly check:

- `training_scope == centralized`;
- no client sampling or aggregation metadata is required for correctness;
- histories are finite;
- `best_validation_round` matches the minimum validation MSE;
- `test_mse_used_for_selection == false`;
- `selection_metric_source == validation`;
- prediction and checkpoint artifacts exist.

No validator changes were implemented in this audit.

## 6. Smoke Plan

Smoke manifest created:

```text
experiments/centralized_baselines/centralized_smoke_manifest.csv
```

The smoke rows are blocked placeholders for:

- `abs`, seed 0, `gda_d`;
- `abs`, seed 0, `sgda_s`;
- `abs`, seed 0, `oadam_s`;
- `abs`, seed 0, `oadam_d` as an extension-only placeholder.

They are marked `blocked_not_implemented` and must not be launched until a true centralized runner exists.

## 7. Full Plan

Full manifest created:

```text
experiments/centralized_baselines/centralized_full_manifest.csv
```

It contains only a header because no true centralized method is verified as launchable. This intentionally avoids representing placeholder rows as runnable experiments.

## 8. Verdict

Config-only launch is not sufficient. A minimal fix is required: implement a true centralized DeepGMM runner or entrypoint that consumes pooled train/validation/test data, constructs one `g` model and one `f` model, performs direct GDA/SGDA/OAdam optimizer updates, selects checkpoints by validation MSE only, and writes the same artifact contract used by the federated runs.

NOT_IMPLEMENTED: no true centralized runner found
