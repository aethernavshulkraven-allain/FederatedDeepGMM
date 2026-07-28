# Centralized C5a GDA/SGDA LR Tuning Report

## Scope

Ran a validation-only LR screen for centralized GDA and SGDA. No training code was modified.

Manifest:

```text
experiments/centralized_baselines/centralized_c5a_tuning_manifest.csv
```

Output root:

```text
results/centralized_lowdim_v1_tuning/c5_gda_sgda_lr_screen/
```

Grid:

```text
datasets: abs, step, linear, sin
methods: gda, sgda
seed: 0
iterations: 500
batch size: gda=0, sgda=256
g_lr: 0.001, 0.002, 0.005
f_lr: 0.01, 0.03
excluded existing C3 combo: g_lr=0.001, f_lr=0.01
```

## Completion

- Expected C5a runs: 40
- Completed/skipped-valid runs: 40
- Failed/incomplete runs: 0
- Validation pass: 40
- Validation fail: 0
- Wall-clock runtime: 4953.82 sec (82.56 min)

Status counts:

```text
{'completed_valid': 40}
```

Validation counts:

```text
{'pass': 40}
```

Runtime by method:

| method | runs | mean wall sec | total wall sec |
| --- | --- | --- | --- |
| gda | 20 | 146.1 | 2922.6 |
| sgda | 20 | 101.3 | 2026.9 |

## Selection Rule

Hyperparameters were selected using validation metrics only:

```text
primary: minimum best_validation_mse on seed 0
candidate pool: C3 baseline combo plus C5a new LR combos
```

Test MSE columns are post-selection readouts only and were not used for selection.

## C5a Best New Run vs C3 Baseline

| dataset | method | C3 best val | best C5a val | C3 test@best | C5a test@best readout | C5a val improvement % | better than C3? | best C5a g/f lr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abs | gda | 0.459771 | 0.0530258 | 0.475888 | 0.0543149 | 88.47 | true | 0.005/0.03 |
| abs | sgda | 0.45714 | 0.0503808 | 0.473025 | 0.0514712 | 88.98 | true | 0.005/0.03 |
| step | gda | 0.0998472 | 0.0380992 | 0.100854 | 0.0383629 | 61.84 | true | 0.005/0.03 |
| step | sgda | 0.0988982 | 0.038295 | 0.0999384 | 0.0385908 | 61.28 | true | 0.005/0.03 |
| linear | gda | 0.184883 | 0.0153011 | 0.189976 | 0.0155969 | 91.72 | true | 0.005/0.03 |
| linear | sgda | 0.184814 | 0.0108895 | 0.18998 | 0.0111464 | 94.11 | true | 0.005/0.03 |
| sin | gda | 0.127183 | 0.0937426 | 0.12672 | 0.0948203 | 26.29 | true | 0.005/0.03 |
| sin | sgda | 0.126643 | 0.094401 | 0.1262 | 0.0956829 | 25.46 | true | 0.005/0.03 |

## Selected Recipe Per Dataset/Method

| dataset | method | selected source | g_lr | f_lr | best val | test@best readout | val improvement vs C3 % | needs seed1/2 confirmation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abs | gda | c5a_new | 0.005 | 0.03 | 0.0530258 | 0.0543149 | 88.47 | true |
| abs | sgda | c5a_new | 0.005 | 0.03 | 0.0503808 | 0.0514712 | 88.98 | true |
| step | gda | c5a_new | 0.005 | 0.03 | 0.0380992 | 0.0383629 | 61.84 | true |
| step | sgda | c5a_new | 0.005 | 0.03 | 0.038295 | 0.0385908 | 61.28 | true |
| linear | gda | c5a_new | 0.005 | 0.03 | 0.0153011 | 0.0155969 | 91.72 | true |
| linear | sgda | c5a_new | 0.005 | 0.03 | 0.0108895 | 0.0111464 | 94.11 | true |
| sin | gda | c5a_new | 0.005 | 0.03 | 0.0937426 | 0.0948203 | 26.29 | true |
| sin | sgda | c5a_new | 0.005 | 0.03 | 0.094401 | 0.0956829 | 25.46 | true |

Machine-readable selected recipes:

```text
experiments/centralized_baselines/centralized_c5a_best_by_validation.csv
```

Full per-run results:

```text
experiments/centralized_baselines/centralized_c5a_tuning_results.csv
```

## Readout

C5a found better validation-selected recipes than C3 for `8/8` dataset/method pairs. Only those selected C5a-new recipes need fresh seed-1/seed-2 confirmation; any pair where the C3 baseline remains selected already has seeds 1 and 2 from C3.

Estimated C5b confirmation size:

```text
16 new runs = 8 selected C5a-new recipes x 2 remaining seeds
```

If confirming all 8 dataset/method recipes regardless of C3 reuse, the conservative manifest size would be 16 rows.
