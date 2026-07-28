# Deterministic Sine A2-lite Final Report

## 1. Objective

Evaluate the validation-locked deterministic FedOGDA-D Sine recipe against fully paired FedGDA-D runs, using validation-selected Test MSE as the primary post-selection metric.

## 2. Locked FedOGDA-D Recipe

- Dataset/mode: `sin` / `deterministic`
- Alpha: `1.0`
- Rounds/local epochs: `500` / `3`
- g LR / f LR: `0.002` / `0.03`
- Critic multiplier / server LR: `15.0` / `1.5`
- Weight decay: `0.1`
- Clients total/per round: `1000` / `1000`
- Batch size: `0`
- Data: `data`, `hetero`, alpha `1.0`

Machine-readable recipe: `experiments/sine_fedogda_tuning/a2_lite_locked_recipe_summary.json`.

## 3. Validation-only Selection Proof

Selection audit passed: `true`. Recipe ranking and checkpoint selection used validation metrics only. Test MSE was inspected only after lock.

Evidence: `experiments/sine_fedogda_tuning/a2_lite_selection_audit.md`.

## 4. FedOGDA-D Seed Metrics

| seed | best val MSE | best round | selected Test MSE | final Test MSE | last-50 Test mean | last-50 Test std |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.08107361 | 499 | 0.08321022 | 0.08321022 | 0.08390051 | 0.00039846 |
| 1 | 0.07270592 | 499 | 0.07464631 | 0.07464631 | 0.07579932 | 0.00067757 |
| 2 | 0.08005923 | 499 | 0.08217807 | 0.08217807 | 0.08253779 | 0.00020783 |

## 5. FedOGDA-D Aggregate Test MSE

- Primary selected Test MSE: mean `0.0800115346`, population std `0.0038171146`.
- Final Test MSE: mean `0.0800115346`, population std `0.0038171146`.
- Last-50 Test mean: `0.0807458741` across seeds.

## 6. Paired FedGDA-D Baseline Availability

Fully matched baseline available for all seeds: `true`.

The audit matched dataset, alpha, seed, T, R, client counts, full participation, batch size, evaluation policy, and exact saved `x`/`true_g` arrays.

Audit: `experiments/sine_fedogda_tuning/a2_lite_fedgda_baseline_match_audit.csv`.

## 7. Pairwise FedOGDA-D vs FedGDA-D

| seed | FedGDA-D selected Test MSE | FedOGDA-D selected Test MSE | gap | relative gap | winner |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.09006007 | 0.08321022 | -0.00684985 | -7.606% | FedOGDA-D |
| 1 | 0.08103073 | 0.07464631 | -0.00638442 | -7.879% | FedOGDA-D |
| 2 | 0.08722979 | 0.08217807 | -0.00505172 | -5.791% | FedOGDA-D |

FedOGDA-D mean `0.0800115346` vs FedGDA-D mean `0.0861068629`. Absolute improvement `0.0060953283` (7.079%). FedOGDA-D won `3/3` seeds.

## 8. Last-50 Behavior

All FedOGDA-D histories are finite and non-divergent. FedOGDA-D last-50 Test MSE is available because A2-lite enabled per-round Test logging. The legacy FedGDA-D baselines lack per-round Test MSE, so last-50 Test MSE is secondary and is not used for the paired claim.

| seed | FedGDA-D validation CV | FedOGDA-D validation CV | FedOGDA-D Test CV | final vs last-50 Test gap |
| --- | --- | --- | --- | --- |
| 0 | 0.000623 | 0.004755 | 0.004749 | 0.823% |
| 1 | 0.005714 | 0.008834 | 0.008939 | 1.521% |
| 2 | 0.000455 | 0.002215 | 0.002518 | 0.436% |

FedOGDA-D last-50 validation and Test CV are below 1% for every seed, and final Test MSE is within 1.6% of the corresponding last-50 mean. Its final Test MSE is therefore numerically stable enough to report for these A2-lite runs; it also equals the validation-selected Test MSE because all three best-validation rounds are 499.

FedOGDA-D does not improve validation CV relative to FedGDA-D in these pairs: FedGDA-D has the lower last-50 validation CV on all three seeds. The stability-improvement subclaim is therefore not supported, even though both methods are stable.

## 9. Curve-fitting Summary

Metrics use saved test points and `best_validation_prediction`; they are not dense-grid checkpoint evaluations.

- Mean curve MSE: FedGDA-D `0.0861068629`, FedOGDA-D `0.0800115346`.
- Mean curve MAE: FedGDA-D `0.2451058630`, FedOGDA-D `0.2294069237`.
- Mean maximum absolute error: FedGDA-D `0.9475485709`, FedOGDA-D `0.9172711531`.

Plots:

- `experiments/sine_fedogda_tuning/plots/a2_lite_sine_curve_seed0.png`
- `experiments/sine_fedogda_tuning/plots/a2_lite_sine_curve_all_seeds.png`
- `experiments/sine_fedogda_tuning/plots/a2_lite_validation_curves.png`
- `experiments/sine_fedogda_tuning/plots/a2_lite_test_curves.png`
- `experiments/sine_fedogda_tuning/plots/a2_lite_last50_zoom.png`

## 10. Verdict

**SUPPORTED: FedOGDA-D achieves lower validation-selected Test MSE than paired FedGDA-D on Sine.**

This verdict is specific to deterministic Sine, alpha 1.0, the locked recipe, and the three paired seeds. It does not establish universal FedOGDA superiority or a stochastic Sine result.
