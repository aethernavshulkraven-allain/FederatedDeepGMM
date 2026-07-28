# Centralized C5 Tuning Plan

## Goal

Tune centralized GDA and SGDA enough that the centralized baselines are not obviously under-trained. Do not tune OAdam in C5 unless a later paper-target comparison requires it.

Selection rule: validation-only. Test MSE is reported only after choosing by validation metrics.

## Why Tuning Is Needed

C4 found that GDA/SGDA are valid but not well calibrated. For each method, 7/12 runs select their best validation checkpoint at the final iteration (`499/499`), while the remaining runs peak earlier and then worsen by the final checkpoint. Their mean last-50 validation trend is still decreasing, but the per-run behavior is mixed. This indicates the current learning rates and/or `500`-iteration budget should be screened before treating GDA/SGDA as final paper-quality centralized baselines.

## Current Baseline

Current C3 baseline:

```text
methods: gda, sgda
datasets: abs, step, linear, sin
seeds: 0, 1, 2
iterations: 500
g_lr: 0.001
f_lr: 0.01
batch_size: gda=0, sgda=256
```

## C5a: Small LR Screen

Use seed `0` only and reuse the current C3 run as the `(g_lr=0.001, f_lr=0.01)` baseline instead of rerunning it.

New screening matrix:

```text
datasets: abs, step, linear, sin
methods: gda, sgda
seed: 0
iterations: 500
g_lr: 0.001, 0.002, 0.005
f_lr: 0.01, 0.03
exclude already-completed current C3 combo: g_lr=0.001, f_lr=0.01
batch_size: gda=0, sgda=256
```

New runs:

```text
4 datasets x 2 methods x (3 x 2 - 1 existing combo) x 1 seed = 40 runs
```

Ranking metric:

```text
primary: best_validation_mse
secondary: last50_validation_mse_mean
tertiary: last50_validation_mse_slope closer to 0 from below
reject: divergence or non-finite history
```

## C5b: Confirmation

For each dataset and method, select the best C5a recipe by validation only. Confirm on seeds `1` and `2`; seed `0` is already available from C5a.

New confirmation runs:

```text
4 datasets x 2 methods x 2 remaining seeds = 16 runs
```

## Optional C5c: Iteration Extension

Only if the selected C5b recipe still has best validation at the final iteration and a clearly negative last-50 validation slope, extend that selected recipe to:

```text
iterations: 1000
seeds: 0, 1, 2
```

Run this only for affected dataset/method pairs.

Worst-case optional runs:

```text
4 datasets x 2 methods x 3 seeds = 24 runs
```

## Runtime Estimate

Observed C3 mean wall runtime:

```text
gda: 33.3 sec/run
sgda: 23.9 sec/run
```

C5a estimate:

```text
20 GDA runs x 33.3 sec + 20 SGDA runs x 23.9 sec ~= 19.1 minutes sequential
```

C5b estimate:

```text
8 GDA runs x 33.3 sec + 8 SGDA runs x 23.9 sec ~= 7.6 minutes sequential
```

C5a + C5b estimate:

```text
approximately 27 minutes sequential
```

Worst-case optional C5c estimate:

```text
approximately 23 minutes more if all dataset/method pairs need 1000-iteration confirmation
```

## Recommended Output Root

Use a separate path to avoid overwriting C3:

```text
results/centralized_lowdim_v1_tuning/c5_gda_sgda_lr_screen/
```

## Exact Recommended C5 Matrix

C5a should launch only these new hyperparameter pairs for each dataset/method at seed 0:

```text
(g_lr=0.001, f_lr=0.03)
(g_lr=0.002, f_lr=0.01)
(g_lr=0.002, f_lr=0.03)
(g_lr=0.005, f_lr=0.01)
(g_lr=0.005, f_lr=0.03)
```

plus the existing C3 baseline:

```text
(g_lr=0.001, f_lr=0.01)
```

used analytically, not rerun.

## Stop Conditions

Stop before C5b if:

```text
more than 20% of C5a runs diverge
validators fail
any script accidentally uses Test MSE for selection
new outputs would overwrite C3
```
