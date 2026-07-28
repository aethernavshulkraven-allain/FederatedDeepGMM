# Centralized Runner Callgraph Audit

## Manifest Launcher Path

`scripts/run_manifest.py`

1. The launcher docstring says it queues federated runs and skips centralized rows until the runner is verified (`scripts/run_manifest.py:1-9`).
2. `SUPPORTED_FEDERATED_METHODS` contains only `fedgda_d`, `fedgda_s`, `fedogda_d`, and `fedogda_s` (`scripts/run_manifest.py:34`).
3. `select_rows` drops every row whose `training_scope` is not `federated` and drops methods outside the supported federated set (`scripts/run_manifest.py:116-130`).

Result: `scripts/run_manifest.py` cannot launch the centralized rows in `experiments/rerun_protocol_v1/manifest.csv` as currently written.

## Current DeepGMM Entrypoint Path

`fedgmm/sp_decentralized_mnist_lr_example/main.py`

1. `main.py` calls `fedml.init()`, loads data via `fedml.data.load(args)`, creates the model via `fedml.model.create(args, output_dim)`, constructs `FedMLRunner`, and calls `fedml_runner.run()` (`main.py:11-26`).
2. `FedMLRunner` sends `training_type == simulation` to `_init_simulation_runner` (`fedml/runner.py:36-81`).
3. For single-process simulation, `SimulatorSingleProcess` dispatches `federated_optimizer == FedAvg` to `FedAvgAPI` (`fedml/simulation/simulator.py:27-67`).

Result: the active low-dimensional DeepGMM path is simulation plus FedAvgAPI, not a centralized runner.

## Current Data Path

`fedgmm/sp_decentralized_mnist_lr_example/fedml/data/data_loader.py`

1. `load(args)` directly returns `load_synthetic_data(args)` (`data_loader.py:234-235`).
2. `load_synthetic_data` sets `centralized = False`, then for `linear`, `abs`, `sin`, and `step` calls `load_partition_data_mnist(...)` and constructs global plus local dictionaries (`data_loader.py:369-421`).

Result: global train/validation/test data exist, but the active training loop still uses client-local dictionaries for updates.

## Current Federated DeepGMM Training Path

`fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/fedavg_api.py`

1. Optimizer construction chooses `OGDA` only when `client_optimizer == "ogda"`; otherwise it chooses `CustomSGD` (`fedavg_api.py:219-235`, `fedavg_api.py:244-251`).
2. Model selection uses pooled/global train and validation tensors (`fedavg_api.py:318-321`).
3. Clients are created from local client dictionaries (`fedavg_api.py:342-363`).
4. Each round samples clients, updates each local client dataset, trains on local data, aggregates local weights, and applies a server learning-rate update (`fedavg_api.py:399-493`).
5. Evaluation, optional per-round Test MSE logging, and best checkpoint selection are validation-driven (`fedavg_api.py:500-547`).

Result: checkpoint selection is validation-only, but the training update is federated, uses client sampling/local workers/FedAvg aggregation/server LR, and is not a true centralized baseline.

## Generic Centralized Trainer Path

`fedgmm/sp_decentralized_mnist_lr_example/fedml/centralized/centralized_trainer.py`

1. A `CentralizedTrainer` class exists, but it trains a single classifier with `CrossEntropyLoss` and SGD/Adam (`centralized_trainer.py:9-46`).
2. Its train loop iterates over `train_global`, computes classification loss, and logs accuracy/loss to W&B (`centralized_trainer.py:48-64`, `centralized_trainer.py:80-164`).
3. No reachable call site was found from the low-dimensional DeepGMM `main.py` path.

Result: this is not the required DeepGMM centralized GDA/SGDA/OAdam runner and does not produce the required DeepGMM MSE/prediction/checkpoint artifacts.
