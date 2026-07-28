# Test MSE Stability Report

Scope: completed existing outputs only. No training was launched and no training logic was changed.

## Direct Answer For Geetika

All inspected completed runs are numerically stable: no non-finite history rows and no `diverged=true` metrics were found.

However, the existing `mse_by_round.csv` files do **not** store per-round Test MSE. They store only:

`round, train_mse, val_mse, gmm_train_objective, gmm_val_objective, gmm_eval, finite, diverged`

Therefore, from the current artifacts we **cannot** compute `last50_mean_test_mse`, `last50_std_test_mse`, Test-MSE CV, Test-MSE drift, or decide whether final Test MSE is stabilized by directly inspecting a Test-MSE curve.

The safest wording is:

> The runs are numerically stable, but per-round Test MSE was not logged, so we cannot verify Test-MSE stabilization or report a last-50 average Test MSE from existing outputs. We can report `final_test_mse` / `test_mse_at_best_validation` as scalar held-out Test metrics, and use the last-50 validation curve as a secondary stability diagnostic. If Geetika wants last-50 average Test MSE, we need to add per-round Test MSE logging and rerun or at least re-evaluate checkpoints per round.

## Availability Summary

- Main federated synthetic runs inspected: `144`
- FedOGDA-S tuning pilot runs inspected: `144`
- Total inspected runs: `288`
- Numerically stable runs: `288/288`
- Runs with per-round Test MSE: `0/288`
- Main matrix per-round Test MSE availability: `0/144`
- Tuning pilot per-round Test MSE availability: `0/144`

## Main 144-Run Matrix: Validation-Curve Stability Secondary Signal

Because Test MSE is not logged per round, the following stability counts use `val_mse` only as a secondary signal.

### By Method

| method | runs | numerically_stable | has_per_round_test_mse | val_stable_5pct | val_stable_10pct | val_stable_20pct | val_large_drift_gt20pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fedgda_d | 36 | 36 | 0 | 18 | 18 | 24 | 15 |
| fedgda_s | 36 | 36 | 0 | 10 | 14 | 17 | 11 |
| fedogda_d | 36 | 36 | 0 | 18 | 18 | 25 | 17 |
| fedogda_s | 36 | 36 | 0 | 16 | 18 | 21 | 14 |

### By Dataset

| dataset | runs | numerically_stable | has_per_round_test_mse | val_stable_5pct | val_stable_10pct | val_stable_20pct | val_large_drift_gt20pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| abs | 36 | 36 | 0 | 0 | 0 | 14 | 28 |
| linear | 36 | 36 | 0 | 0 | 0 | 3 | 28 |
| sin | 36 | 36 | 0 | 33 | 36 | 36 | 0 |
| step | 36 | 36 | 0 | 29 | 32 | 34 | 1 |

### By Alpha

| alpha | runs | numerically_stable | has_per_round_test_mse | val_stable_5pct | val_stable_10pct | val_stable_20pct | val_large_drift_gt20pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 48 | 48 | 0 | 18 | 22 | 29 | 16 |
| 0.5 | 48 | 48 | 0 | 22 | 23 | 30 | 21 |
| 1.0 | 48 | 48 | 0 | 22 | 23 | 28 | 20 |

### By Deterministic/Stochastic Mode

| mode | runs | numerically_stable | has_per_round_test_mse | val_stable_5pct | val_stable_10pct | val_stable_20pct | val_large_drift_gt20pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deterministic | 72 | 72 | 0 | 36 | 36 | 49 | 32 |
| stochastic | 72 | 72 | 0 | 26 | 32 | 38 | 25 |

## FedGDA vs FedOGDA Paired Comparisons

Test last-50 comparisons are not evaluable because neither side logs per-round Test MSE. Final Test MSE comparisons are evaluable from `metrics.json`; validation oscillation comparisons are included as secondary evidence.

| mode | pairs | fedogda_lower_final_test_mse | fedogda_lower_last50_mean_test_mse | fedogda_lower_last50_std_test_mse | fedogda_lower_last50_std_val_mse_secondary | fedogda_lower_last50_cv_val_mse_secondary |
| --- | --- | --- | --- | --- | --- | --- |
| deterministic | 36 | 26 | not_evaluable | not_evaluable | 23 | 16 |
| stochastic | 36 | 6 | not_evaluable | not_evaluable | 29 | 34 |

Interpretation:

- Deterministic FedOGDA has lower scalar final Test MSE than deterministic FedGDA in many pairs, but Test-curve last-50 behavior is not available.
- Stochastic FedOGDA does not generally beat stochastic FedGDA on scalar final Test MSE in the existing main matrix.
- FedOGDA often looks less oscillatory by validation-curve standard deviation/CV, but that is a validation-curve statement, not a Test-MSE curve statement.

## FedOGDA-S Tuning Pilot

The tuning pilot has the same logging limitation: no per-round Test MSE. It is numerically stable, and its validation-curve diagnostics are available.

| method | runs | numerically_stable | has_per_round_test_mse | val_stable_5pct | val_stable_10pct | val_stable_20pct | val_large_drift_gt20pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fedogda_s | 144 | 144 | 0 | 56 | 64 | 84 | 64 |

Existing tuning-pilot conclusion remains: tuned FedOGDA-S improves over current FedOGDA-S and reduces validation oscillation, but it still does not beat FedGDA-S on mean scalar Test MSE for `abs`, `linear`, or `step` at alpha `0.5`.

## Recommendation

For the current repo outputs:

- Report scalar `test_mse_at_best_validation` or `final_test_mse` only with the caveat that per-round Test-MSE stability cannot be verified.
- If Geetika specifically wants stabilized Test MSE or last-50 average Test MSE, add per-round Test MSE logging to `mse_by_round.csv` at the same evaluation frequency and rerun or re-evaluate saved per-round checkpoints.
- Do not substitute last-50 validation MSE and call it Test MSE. Validation last-50 metrics are useful only as secondary stability diagnostics.

## Output Files

- Per-run diagnostics: `experiments/rerun_protocol_v1/test_mse_stability_diagnostics.csv`
- Paired comparisons: `experiments/rerun_protocol_v1/test_mse_stability_pairs.csv`
- Markdown report: `experiments/rerun_protocol_v1/test_mse_stability_report.md`
