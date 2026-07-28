# Sine FedOGDA Tuning Metric Policy

This policy is predeclared before any new Sine tuning launch. Hyperparameter choices must be validation-only.

## Primary Tuning Metric

Use `last50_validation_mse_mean` when per-round validation MSE is logged, as it is for the existing Sine runs.

Fallback only if last-50 validation averaging is not meaningful: `best_validation_mse`.

Candidate ranking:

1. lower `last50_validation_mse_mean`;
2. lower `best_validation_mse`;
3. lower `last50_validation_mse_cv`;
4. no divergence and finite history;
5. lower `last50_validation_mse_range`.

## Stability Metrics

Compute from validation MSE over the last 50 rounds:

- `last50_validation_mse_mean`
- `last50_validation_mse_std`
- `last50_validation_mse_min`
- `last50_validation_mse_max`
- `last50_validation_mse_range`
- `last50_validation_mse_cv = std / max(abs(mean), 1e-12)`

Stable validation behavior is defined as either:

- `last50_validation_mse_cv <= 0.05`; or
- `last50_validation_mse_range <= 1e-4` when means are very small.

Record both criteria; do not hide instability.

## Test Reporting Metrics

After a recipe is locked by validation, report:

- `test_mse_at_best_validation`
- `final_test_mse`
- `last50_test_mse_mean`, only if per-round Test MSE exists
- `last50_test_mse_std`, only if per-round Test MSE exists

Do not choose the primary reported Test metric after seeing which one favors FedOGDA.

Predeclared reporting rule:

- If validation curves are stable, final Test MSE may be reported.
- If validation curves are oscillatory, report last-50 average Test MSE only when available, plus validation-selected Test MSE.
- Always include `test_mse_at_best_validation` for paired comparison.
