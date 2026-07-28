# Step/Sine ASAP Tuning Plan

This note records the local commit/artifact audit for improving low-dimensional curve fitting on `step` and `sin`. It is intended to drive the next launch without using Test MSE for selection.

## Executive Readout

- Step should be tuned first.
- The interrupted current Step mini grid is searching the wrong neighborhood: it uses full participation (`client_num_per_round=1000`), alpha `0.1`, `T=200`, and weight decay `0.1`.
- Geetika's successful Step commit used partial-client recipes (`client_num_per_round=10`), alpha `0.5`, `T=1500`, local epochs `R=7`, weight decay `0.02`, critic multiplier `15`, and server LR around `1.5`.
- Sine already has a validation-locked positive deterministic FedOGDA-D result, but the best-validation round was the final round for every seed, so the cheapest improvement is to extend the locked recipe to `T=1000` before widening the grid.

All selection below must use validation metrics only. Test MSE and curve metrics are post-selection readouts.

## Old Step Evidence From Local Commit

Local commit:

```text
8959997 Experiments for Step function (tuned).
```

Important config from that commit:

```text
dataset = step
partition_alpha = 0.5
client_num_in_total = 1000
client_num_per_round = 10
comm_round = 1500
epochs = 7
critic_multiplier = 15
server_learning_rate ~= 1.5
weight_decay = 0.02
```

Old tuned recipes encoded in `fedml_config.yaml`:

| old label | current interpretation | optimizer | batch size | g_lr | f_lr | weight decay |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| BEST RUN OGDA | FedOGDA-S style partial-client minibatch | `ogda` | 256 | 0.01 | 0.15 | 0.02 |
| BEST WORKING SGD | FedGDA-S style partial-client minibatch | `sgd` | 256 | 0.03 | 0.45 | 0.02 |
| BEST RUN OGDA-FB | partial-client full-batch OGDA, not full participation | `ogda` | 0 | 0.01 | 0.15 | 0.02 |
| BEST WORKING GDA-FB | partial-client full-batch GDA, not full participation | `sgd` | 0 | 0.03 | 0.45 | 0.02 |

Important caveat: the old "FB" rows use `batch_size=0` but still use `client_num_per_round=10`. They should not be merged with our standard deterministic full-participation FedGDA-D/FedOGDA-D protocol unless clearly labeled.

Old tracked Step CSV evidence:

| file | rows | best MSE | best row index | final MSE |
| --- | ---: | ---: | ---: | ---: |
| `ogda_step.csv` | 16308 | 0.014724974 | 15929 | 0.099148890 |
| `ogda_stepnew.csv` | 11999 | 0.004589016 | 11494 | 0.068124682 |
| `ogda_stepnew_fullbatch.csv` | 1500 | 0.004498046 | 1379 | 0.039760692 |
| `ogda_stepnewbestworking.csv` | 9000 | 0.004419461 | 8787 | 0.074223983 |
| `sgd_stepnew.csv` | 13515 | 0.007109170 | 12339 | 0.148588821 |
| `sgd_stepnew_fullbatch.csv` | 1500 | 0.007167773 | 512 | 0.055652284 |

These old CSVs show a much better Step neighborhood than the current mini grid. They also show why final-last-iterate reporting is risky: final MSE is often much worse than the best/selected point.

## Current Step Gap

Current interrupted mini grid:

```text
experiments/curve_fitting_tuning/step_fedogda_d_mini_v1/manifest.csv
results/curve_fitting_tuning/step_fedogda_d_mini_v1/
```

Completed rows so far:

| run | best validation MSE | test@best validation | best round |
| --- | ---: | ---: | ---: |
| alpha 0.1, T=200, R=3, cm=15, g_lr=0.002, wd=0.1 | 0.029550326 | 0.030100881 | 199 |
| alpha 0.1, T=200, R=3, cm=15, g_lr=0.005, wd=0.1 | 0.028433039 | 0.028932612 | 199 |

This is only marginally better than the original Step sweep and far worse than Geetika's old tuned Step records. The next Step action should not continue this full-participation mini grid until the old recipe family has been reproduced.

## Current Sine Status

Validation-locked deterministic Sine FedOGDA-D result:

```text
experiments/sine_fedogda_tuning/a2_lite_final_report.md
experiments/sine_fedogda_tuning/a2_lite_locked_recipe_summary.json
```

Locked recipe:

```text
dataset = sin
alpha = 1.0
client_num_in_total = 1000
client_num_per_round = 1000
batch_size = 0
T = 500
R = 3
g_lr = 0.002
critic_multiplier = 15
f_lr = 0.03
server_lr = 1.5
weight_decay = 0.1
```

Result:

```text
FedOGDA-D mean test@best validation = 0.080011535
FedGDA-D mean test@best validation = 0.086106863
FedOGDA-D wins 3/3 seeds
```

Sine is already positive. The improvement opportunity is that all three FedOGDA-D best-validation rounds are `499`, meaning validation was still best at the end of `T=500`.

## ASAP Launch Plan

### Stage Step-1: Reproduce Old Step Recipe Family

Create a small seed-0 manifest under:

```text
experiments/curve_fitting_tuning/step_geetika_repro_v1/
results/curve_fitting_tuning/step_geetika_repro_v1/
```

Run exactly these four Step rows first:

| planned row | method field | optimizer | alpha | clients per round | T | R | batch | g_lr | wd | cm | server_lr |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old Step OGDA minibatch | `fedogda_s` | `ogda` | 0.5 | 10 | 1500 | 7 | 256 | 0.01 | 0.02 | 15 | 1.5 |
| old Step SGD minibatch | `fedgda_s` | `sgd` | 0.5 | 10 | 1500 | 7 | 256 | 0.03 | 0.02 | 15 | 1.5 |
| old Step OGDA partial-FB | `fedogda_d` with clear notes | `ogda` | 0.5 | 10 | 1500 | 7 | 0 | 0.01 | 0.02 | 15 | 1.5 |
| old Step GDA partial-FB | `fedgda_d` with clear notes | `sgd` | 0.5 | 10 | 1500 | 7 | 0 | 0.03 | 0.02 | 15 | 1.5 |

Validation-only ranking:

1. no divergence and finite histories;
2. lowest `best_validation_mse`;
3. lowest `last50_validation_mse_mean`;
4. lower last-50 validation CV/range;
5. Test MSE and curve metrics only after selection.

Do not compare partial-client full-batch rows to standard full-participation deterministic baselines unless we also run matched FedGDA rows under the same client policy.

### Stage Step-2: Confirm The Best Step Recipe

If Stage Step-1 recovers a strong validation-selected Step curve, run the selected FedOGDA recipe on:

```text
seeds = 0, 1, 2
same alpha/T/R/client policy/batch/g_lr/wd/cm/server_lr
```

Also run the paired FedGDA baseline under exactly the same:

```text
seed
alpha
client_num_per_round
batch_size
T
R
data path
evaluation policy
```

This is the comparison that can become paper-ready.

### Stage Step-3: If Old Recipe Does Not Reproduce

Use a tiny local grid around the old recipe, not the previous full-participation mini grid:

```text
dataset = step
alpha = 0.5
seed = 0
client_num_in_total = 1000
client_num_per_round = 10
T = 1500
R = [5, 7]
mode = minibatch first, batch_size=256
g_lr = [0.005, 0.01, 0.02]
weight_decay = [0.02, 0.05]
critic_multiplier = [15]
server_lr = [1.5]
```

This is 12 FedOGDA rows. Only add partial-FB rows if the old partial-FB reproduction is clearly promising by validation.

### Stage Sine-1: Extend Locked Sine Recipe

Run the already locked Sine FedOGDA-D recipe at `T=1000`, seed 0 first:

```text
dataset = sin
alpha = 1.0
client_num_in_total = 1000
client_num_per_round = 1000
batch_size = 0
T = 1000
R = 3
g_lr = 0.002
weight_decay = 0.1
critic_multiplier = 15
server_lr = 1.5
```

Rationale: the `T=500` run was still improving at the last round. Extending horizon is the smallest intervention.

If validation improves at `T=1000`, confirm seeds 1 and 2. If the final Sine claim requires exact fairness at `T=1000`, run matched FedGDA-D at `T=1000` only after the FedOGDA-D extension is validation-promising.

## Immediate Recommendation

1. Stop treating the current Step full-participation mini grid as the main route.
2. Prepare and launch Stage Step-1, the four-row old Step recipe reproduction.
3. In parallel only if GPU quota allows, prepare Sine `T=1000` seed-0 extension, but Step is the urgent unresolved function.
4. After Stage Step-1, generate Step curve plots in the same coauthor-style layout and compare against the old commit plots.

## Reporting Guardrails

- Old commit plots/CSVs can guide the search but cannot be used as current paper evidence.
- Any selected Step/Sine recipe must be selected by validation metrics only.
- Test MSE and curve-fit MSE/MAE/max error are reported only after the recipe is locked.
- The old Step `new` prediction arrays referenced by `curve_plot.py` are not fully tracked, so the modern run must regenerate plots from current artifacts.
- Label partial-client full-batch Step separately from standard full-participation deterministic Step.
