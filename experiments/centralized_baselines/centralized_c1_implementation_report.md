# Centralized C1 Implementation Report

## Method Mapping

- DeepGMM-GDA is implemented as `gda`.
- DeepGMM-SGDA is implemented as `sgda`.
- DeepGMM-OAdam is implemented as `oadam`.

No separate `oadam_d` or `oadam_s` method is exposed. OAdam is a single centralized optimizer baseline.

## Data Path

The centralized runner is `scripts/run_centralized_lowdim.py`.

It loads pooled low-dimensional splits directly from:

```text
fedgmm/sp_decentralized_mnist_lr_example/data/zoo/{dataset}.npz
```

using `AbstractScenario`. The runner converts the scenario to tensors and uses the global `train`, `dev`, and `test` splits as pooled centralized tensors. It does not call the FedML data loader, does not construct client dictionaries, and does not enter the client-local update path.

Supported datasets are:

```text
abs, step, linear, sin
```

`sine` is accepted as an alias for `sin`.

## Training Path

The runner constructs one structural model `g` and one critic/model `f`:

```text
g: MLPModel(input_dim=1, layer_widths=[20, 20])
f: MLPModel(input_dim=2, layer_widths=[20, 20])
```

It uses the existing DeepGMM moment objective:

```text
OptimalMomentObjective.calc_objective(g, f, x, z, y)
```

The active centralized path is:

```text
scripts/run_centralized_lowdim.py
  -> AbstractScenario pooled split loading
  -> MLPModel g/f construction
  -> OptimalMomentObjective
  -> direct optimizer steps on g and f
  -> validation-only checkpoint selection
  -> final/test reporting artifacts
```

This path does not use:

```text
FedMLRunner
FedAvgAPI
Client
client sampling
FedAvg aggregation
server-learning-rate aggregation
client_num_per_round
```

Every `effective_config.json` records:

```text
training_scope = centralized
uses_clients = false
uses_fedavg_aggregation = false
uses_client_sampling = false
uses_server_learning_rate_aggregation = false
selection_metric_source = validation
test_mse_used_for_selection = false
```

## Optimizer Details

For `gda`, the runner requires `--batch-size 0`, which means full-batch deterministic centralized updates. Both `g` and `f` are optimized with the existing `CustomSGD` optimizer.

For `sgda`, the runner requires a positive minibatch size. The default is `256`. Each iteration samples a fresh minibatch from the pooled centralized train split using the run seed. Both `g` and `f` use `CustomSGD`.

For `oadam`, the runner also requires a positive minibatch size, defaulting to `256`. It uses the existing `optimizers/oadam.py` implementation for both `g` and `f`. A tiny smoke run completed successfully. The existing OAdam code emits a PyTorch deprecation warning for an older `Tensor.add_` signature, but the run and validator both passed.

## Artifact Contract

Each centralized run writes:

```text
effective_config.json
metrics.json
mse_by_round.csv
predictions.npz
checkpoints/best_validation.pt
checkpoints/final.pt
```

`mse_by_round.csv` has exactly:

```text
round,train_mse,val_mse
```

If `--log-test-mse-by-round` is set, the runner writes a separate:

```text
test_mse_by_round.csv
```

Test MSE is not used for checkpoint selection. The best checkpoint is selected only by minimum validation MSE.

The validator is:

```text
scripts/validate_centralized_run.py
```

It checks required artifacts, centralized config flags, finite metrics/histories, validation-only best checkpoint selection, predictions, and checkpoints.

## Manifests

The manifest preparer is:

```text
scripts/prepare_centralized_lowdim_manifest.py
```

It created:

```text
experiments/centralized_baselines/centralized_lowdim_manifest.csv
experiments/centralized_baselines/centralized_smoke_manifest_runnable.csv
```

The full manifest has 36 planned rows:

```text
4 datasets x 3 methods x 3 seeds = 36
```

The runnable smoke manifest has 3 rows:

```text
abs seed 0 gda
abs seed 0 sgda
abs seed 0 oadam
```

## Smoke Commands Run

Compile checks:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python -m py_compile scripts/run_centralized_lowdim.py scripts/validate_centralized_run.py scripts/prepare_centralized_lowdim_manifest.py
```

Help checks:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_centralized_lowdim.py --help
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/validate_centralized_run.py --help
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/prepare_centralized_lowdim_manifest.py --help
```

Manifest generation:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/prepare_centralized_lowdim_manifest.py
```

Tiny 2-iteration smoke runs:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_centralized_lowdim.py --dataset abs --method gda --seed 0 --output-dir results/centralized_lowdim_v1_smoke_tiny/abs/gda/seed_0 --iterations 2 --batch-size 0 --g-lr 0.001 --f-lr 0.01 --no-cuda
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_centralized_lowdim.py --dataset abs --method sgda --seed 0 --output-dir results/centralized_lowdim_v1_smoke_tiny/abs/sgda/seed_0 --iterations 2 --batch-size 256 --g-lr 0.001 --f-lr 0.01 --no-cuda
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_centralized_lowdim.py --dataset abs --method oadam --seed 0 --output-dir results/centralized_lowdim_v1_smoke_tiny/abs/oadam/seed_0 --iterations 2 --batch-size 256 --g-lr 0.001 --f-lr 0.01 --no-cuda
```

Smoke validation:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/validate_centralized_run.py results/centralized_lowdim_v1_smoke_tiny/abs/gda/seed_0 --json
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/validate_centralized_run.py results/centralized_lowdim_v1_smoke_tiny/abs/sgda/seed_0 --json
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/validate_centralized_run.py results/centralized_lowdim_v1_smoke_tiny/abs/oadam/seed_0 --json
```

All three tiny smoke runs validated successfully.

## Full Launch Commands

Full runs were not launched.

Use the `command` column in:

```text
experiments/centralized_baselines/centralized_lowdim_manifest.csv
```

Example full command:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_centralized_lowdim.py --dataset abs --method gda --seed 0 --output-dir results/centralized_lowdim_v1/abs/gda/seed_0 --iterations 500 --batch-size 0 --g-lr 0.001 --f-lr 0.01 --weight-decay 0.0 --gradient-clip-norm 1.0
```

Before launching the full manifest, run the 3-row runnable smoke manifest or equivalent commands under `results/centralized_lowdim_v1_smoke/`.

READY_FOR_CENTRALIZED_SMOKE
