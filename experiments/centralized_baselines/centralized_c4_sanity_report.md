# Centralized C4 Sanity Report

## Scope

This is an analysis-only pass over completed C3 centralized low-dimensional runs. No new training was launched and no training code was modified.

Inputs:

```text
experiments/centralized_baselines/centralized_c3_full_run_results.csv
experiments/centralized_baselines/centralized_lowdim_summary_by_function_method.csv
results/centralized_lowdim_v1/
experiments/lowdim_fedogda_d_vs_fedgda_d_summary.csv
experiments/sine_fedogda_tuning/a2_lite_pairwise_fedogda_vs_fedgda.csv
```

## Integrity Verdict

- Runs checked: 36
- Integrity errors: 0
- Prediction files found: 36/36
- Prediction arrays sane: `true`

Checks performed:

```text
all key metrics finite
best_validation_round == min validation MSE round
test_mse_used_for_selection == false
selection_metric_source == validation
predictions.npz has x, true_g, best_validation_prediction, final_prediction
prediction arrays are finite and shape-compatible
```

C3 is valid/completed as a centralized run set.

## Curve Plots

Seed-0 centralized curve plots compare true `g` against validation-selected GDA, SGDA, and OAdam predictions.

- `experiments/centralized_baselines/plots/centralized_curve_abs_seed0.png`
- `experiments/centralized_baselines/plots/centralized_curve_step_seed0.png`
- `experiments/centralized_baselines/plots/centralized_curve_linear_seed0.png`
- `experiments/centralized_baselines/plots/centralized_curve_sin_seed0.png`

## Centralized vs Federated Low-Dimensional Comparison

Metric: validation-selected Test MSE. Lower is better. Federated original sweep values aggregate over alpha `0.1/0.5/1.0` and seeds `0/1/2`; they are useful for context but are not exactly paired to centralized pooled-data runs. Tuned Sine is labeled separately and is not merged into the original sweep.

| dataset | Central GDA | Central SGDA | Central OAdam | FedGDA-D original | FedOGDA-D original/tuned |
| --- | --- | --- | --- | --- | --- |
| abs | 0.305203 | 0.303931 | 0.00177951 | 0.0180081 | 0.0168918 |
| step | 0.0582591 | 0.0581996 | 0.017214 | 0.0297296 | 0.0291688 |
| linear | 0.0843963 | 0.0857472 | 0.000759664 | 0.00422609 | 0.0028616 |
| sin | 0.101136 | 0.101199 | 0.0376761 | 0.0861523 | 0.0862134 |
| sin tuned A2-lite |  |  |  | 0.0861069 | 0.0800115 |

Full comparison table:

```text
experiments/centralized_baselines/centralized_c4_comparison_table.csv
```

## GDA/SGDA Tuning Diagnosis

| method | runs | best at final | mean final-vs-best val gap % | mean last50 val slope | mean last50 drift % |
| --- | --- | --- | --- | --- | --- |
| gda | 12 | 7/12 | 763.6 | -0.0002771 | -3.385 |
| sgda | 12 | 7/12 | 802.8 | -0.0002843 | -3.211 |
| oadam | 12 | 2/12 | 1831 | -8.686e-05 | -12.66 |

Readout:

- GDA and SGDA are valid preliminary baselines, but they do not look well calibrated enough for final reporting.
- For GDA/SGDA, the best validation checkpoint is at the final iteration in 7/12 runs for each method. The remaining runs peak earlier and then worsen by the final checkpoint, so validation selection is important.
- The mean last-50 validation slope/drift is still negative for GDA/SGDA, but the per-run picture is mixed: some runs are still improving at the 500-iteration cutoff, while others have already passed their validation optimum.
- GDA and SGDA are very close to each other, suggesting the current SGDA minibatch setting is not materially changing the trajectory relative to GDA.
- OAdam is strong in current C3: it wins every function by validation-selected Test MSE and does not need immediate tuning for this sanity pass.

GDA vs SGDA similarity by dataset:

| dataset | GDA mean test@best | SGDA mean test@best | mean SGDA-GDA relative % | max abs relative % |
| --- | --- | --- | --- | --- |
| abs | 0.305203 | 0.303931 | -0.2493 | 0.6317 |
| step | 0.0582591 | 0.0581996 | 0.245 | 1.514 |
| linear | 0.0843963 | 0.0857472 | -0.6606 | 8.64 |
| sin | 0.101136 | 0.101199 | 0.1167 | 1.026 |

## Scientific Use As-Is

C3 is scientifically usable as a completed, validated centralized baseline artifact set. However, current C3 GDA/SGDA should be treated as preliminary baselines rather than final paper-quality tuned baselines, because their validation curves are not settled: several are still improving at the end, while several others peak earlier and then drift upward. OAdam is already strong and comparison-ready as a first centralized result, subject to any paper-specific hyperparameter protocol constraints.

## Caveats

- Centralized vs federated values are not fully paired because centralized runs use pooled data and no alpha partition, while federated original runs are aggregated over partition alphas.
- OAdam emits the known PyTorch `Tensor.add_` deprecation warning; it does not affect validation or artifacts.
- This pass does not compare against paper target values directly; it checks internal sanity and tuning need.
