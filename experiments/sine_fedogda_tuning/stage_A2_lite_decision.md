# Stage A2-lite Decision

## Why Full A2 Was Stopped

The original Stage A2 plan had `9` medium deterministic Sine FedOGDA-D runs. The live estimate for full A2 was approximately `35-36` GPU-hours, while remaining quota was approximately `34` GPU-hours before the first A2 run completed. That was too close to the quota limit to continue blindly.

The A2 launcher was paused while the first run was already mid-run. After that run completed, the paused launcher was terminated before it could start row 2. No additional A2 candidate was launched.

Current quota after the completed run was approximately `32.1` GPU-hours. Continuing all remaining `8` A2 rows would still consume nearly all remaining quota, so A2 is narrowed to a validation-only A2-lite continuation.

## Completed Current Run

Completed run:

```text
stage_A2_from_A1_mini_sin_fedogda_d_seed0_alpha1p0_R3_cm15_slr1.5_glr0p002
```

Recipe:

```text
dataset = sin
method = fedogda_d
alpha = 1.0
seed = 0
T = 500
R = 3
g_lr = 0.002
critic_multiplier = 15
server_lr = 1.5
weight_decay = 0.1
client_num_in_total = 1000
client_num_per_round = 1000
batch_size = 0
```

Validation-only metrics:

```text
best_validation_mse = 0.08107361413428762
best_validation_round = 499
final_validation_mse = 0.08107361413428762
last50_validation_mse_mean = 0.08174772477114323
last50_validation_mse_std = 0.0003887488089372199
last50_validation_mse_range = 0.0013258761153596837
last50_validation_mse_cv = 0.0047554694644473755
diverged = false
finite_history = true
```

Post-selection Test metrics, not used for selection:

```text
test_mse_at_best_validation = 0.08321021932625397
final_test_mse = 0.08321021932625397
last50_test_mse_mean = 0.08390051439266659
last50_test_mse_std = 0.00039845899025553407
```

Selection metadata:

```text
test_mse_used_for_selection = false
selection_metric_source = validation
```

## Comparison With A1-mini

A1-mini ranked this same recipe first using validation metrics only:

```text
rank = 1
R = 3
g_lr = 0.002
critic_multiplier = 15
server_lr = 1.5
last50_validation_mse_mean = 0.087979011
best_validation_mse = 0.087528339
last50_validation_mse_cv = 0.0030342985
```

The completed A2 seed-0 confirmation improved the validation metrics at the longer `T=500` budget:

```text
A2 last50_validation_mse_mean = 0.08174772477114323
A2 best_validation_mse = 0.08107361413428762
A2 last50_validation_mse_cv = 0.0047554694644473755
```

The A2 result therefore confirms, rather than changes, the A1-mini top candidate under validation-only criteria.

## Selected A2-lite Candidate

Selected candidate:

```text
alpha = 1.0
T = 500
R = 3
g_lr = 0.002
critic_multiplier = 15
server_lr = 1.5
weight_decay = 0.1
```

Reason:

```text
The recipe was the A1-mini validation-ranked top candidate, remains strong after the seed-0 T=500 validation-only confirmation, has finite/stable validation history, and uses R=3.
```

Test MSE was not used to choose this candidate.

## FedGDA-D Baseline Match

This selected recipe has a matching existing deterministic FedGDA-D baseline for:

```text
dataset = sin
alpha = 1.0
T = 500
R = 3
seeds = 0, 1, 2
```

Existing FedGDA-D baseline rows are present in:

```text
experiments/sine_fedogda_tuning/current_sine_runs.csv
```

## Remaining A2-lite Runs

Run only the selected FedOGDA-D candidate for remaining seeds:

```text
seed = 1
seed = 2
```

Prepared A2-lite selected manifest:

```text
experiments/sine_fedogda_tuning/stage_A2_lite_selected_manifest.csv
```

This manifest has `2` rows and excludes all other A2 candidates.

## Remaining Command

Do not run the original 9-row A2 manifest. To continue A2-lite only, use:

```bash
WANDB_MODE=disabled gpurun -g 1 /home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_manifest.py \
  --manifest experiments/sine_fedogda_tuning/stage_A2_lite_selected_manifest.csv \
  --config-dir experiments/sine_fedogda_tuning/stage_A2_lite_generated_configs \
  --output-root results/sine_fedogda_tuning/stage_A2_from_A1_mini \
  --gpu-ids 0 \
  --max-parallel 1 \
  --resume-skip-completed \
  --results-json experiments/sine_fedogda_tuning/stage_A2_lite_run_results.json \
  --keep-going 2>&1 | tee experiments/sine_fedogda_tuning/stage_A2_lite_launcher_$(date +%Y%m%d_%H%M%S).log
```

Do not launch additional A2 runs until this decision is reviewed.
