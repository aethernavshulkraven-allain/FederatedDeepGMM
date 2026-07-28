# Centralized C2 Smoke Report

## Scope

Ran only the three rows from:

```text
experiments/centralized_baselines/centralized_smoke_manifest_runnable.csv
```

Full 36-run centralized manifest was not launched.

The manifest-defined smoke rows are 2-iteration runs:

```text
abs seed 0 gda
abs seed 0 sgda
abs seed 0 oadam
```

## Commands Run

### GDA

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_centralized_lowdim.py --dataset abs --method gda --seed 0 --output-dir results/centralized_lowdim_v1_smoke/abs/gda/seed_0 --iterations 2 --batch-size 0 --g-lr 0.001 --f-lr 0.01 --weight-decay 0.0 --gradient-clip-norm 1.0
```

Output:

```text
results/centralized_lowdim_v1_smoke/abs/gda/seed_0
```

### SGDA

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_centralized_lowdim.py --dataset abs --method sgda --seed 0 --output-dir results/centralized_lowdim_v1_smoke/abs/sgda/seed_0 --iterations 2 --batch-size 256 --g-lr 0.001 --f-lr 0.01 --weight-decay 0.0 --gradient-clip-norm 1.0
```

Output:

```text
results/centralized_lowdim_v1_smoke/abs/sgda/seed_0
```

### OAdam

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_centralized_lowdim.py --dataset abs --method oadam --seed 0 --output-dir results/centralized_lowdim_v1_smoke/abs/oadam/seed_0 --iterations 2 --batch-size 256 --g-lr 0.001 --f-lr 0.01 --weight-decay 0.0 --gradient-clip-norm 1.0
```

Output:

```text
results/centralized_lowdim_v1_smoke/abs/oadam/seed_0
```

## Validation Commands

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/validate_centralized_run.py results/centralized_lowdim_v1_smoke/abs/gda/seed_0
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/validate_centralized_run.py results/centralized_lowdim_v1_smoke/abs/sgda/seed_0
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/validate_centralized_run.py results/centralized_lowdim_v1_smoke/abs/oadam/seed_0
```

All three returned `VALID`.

## Artifact Checks

Each smoke run produced:

```text
effective_config.json
metrics.json
mse_by_round.csv
predictions.npz
checkpoints/best_validation.pt
checkpoints/final.pt
```

Each `effective_config.json` has:

```text
training_scope = centralized
uses_clients = false
uses_fedavg_aggregation = false
uses_client_sampling = false
selection_metric_source = validation
test_mse_used_for_selection = false
```

Histories are finite for all three runs. For each run, `best_validation_round` matches the minimum validation MSE round in `mse_by_round.csv`.

## Results

| method | pass/fail | wall runtime sec | runner runtime sec | best validation round | best validation MSE | Test MSE at best validation | final Test MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gda | pass | 1.887534 | 0.481389 | 0 | 2.647537724561854 | 2.712138455226392 | 2.7220208425054957 |
| sgda | pass | 1.654245 | 0.245918 | 0 | 2.6475632212861178 | 2.712167644240736 | 2.7222112412424506 |
| oadam | pass | 1.544104 | 0.150303 | 1 | 2.801590254739793 | 2.8653814042744936 | 2.8653814042744936 |

Detailed CSV:

```text
experiments/centralized_baselines/centralized_c2_smoke_results.csv
```

## OAdam Warning

OAdam emitted the existing PyTorch deprecation warning from:

```text
fedgmm/sp_decentralized_mnist_lr_example/optimizers/oadam.py
```

Warning summary:

```text
Tensor.add_ overload is deprecated
```

This did not affect outputs: the OAdam smoke completed, wrote all required artifacts, had finite histories, and passed the validator.

## Full 36-Run Runtime Estimate

The full manifest has 36 rows:

```text
4 datasets x 3 methods x 3 seeds = 36
```

The smoke rows use 2 iterations; full rows use 500 iterations.

Conservative wall-clock linear estimate:

| method | rows in full launch | smoke wall sec | estimated sec per 500-iter run | estimated total |
| --- | ---: | ---: | ---: | ---: |
| gda | 12 | 1.887534 | 471.8835 | 94.38 min |
| sgda | 12 | 1.654245 | 413.5613 | 82.71 min |
| oadam | 12 | 1.544104 | 386.0260 | 77.21 min |

Conservative sequential total:

```text
approximately 4.24 hours
```

Runner-internal timing gives a lower estimate, approximately 45 minutes sequential, but the wall-clock estimate is safer for planning because it includes startup and artifact overhead from the actual commands.

READY_FOR_FULL_CENTRALIZED_LAUNCH
