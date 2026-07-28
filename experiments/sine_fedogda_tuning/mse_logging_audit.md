# Sine MSE Logging Audit

This audit inspects existing completed Sine artifacts only. No training logic was changed.

## Answers

1. Is per-round Test MSE available for existing runs?

No. Per-round Test MSE is available in `0/36` existing Sine runs.

Observed `mse_by_round.csv` column sets:

- `round|train_mse|val_mse|gmm_train_objective|gmm_val_objective|gmm_eval|finite|diverged`

2. If yes, can we compute average Test MSE over last 50 rounds?

No. Because per-round Test MSE is absent, the last-50 average Test MSE cannot be computed from the existing runs.

3. If no, what is available?

- Per-round `train_mse` and `val_mse` in `mse_by_round.csv`.
- Scalar `test_mse_at_best_validation` and `final_test_mse` in `metrics.json`.
- Test-point predictions for `best_validation_prediction` and `final_prediction` in `predictions.npz`.
- Checkpoints for `best_validation.pt`, `final.pt`, and sparse periodic checkpoints such as `round_0.pt`, `round_200.pt`, and `round_400.pt`.

4. Can last-50 validation MSE be used for tuning?

Yes. Last-50 validation MSE is available for every completed Sine run and is validation-only, so it can be used for tuning and stability ranking without touching Test MSE.

5. Can future runs safely log per-round Test MSE without using it for tuning?

Yes, if the workflow treats it as a reporting-only diagnostic after the recipe is locked by validation. The selection code/report must not read or rank candidates by per-round Test MSE.

6. Are there enough round checkpoints in the last 50 rounds to reconstruct last-50 Test MSE?

No. Across existing Sine runs, the upper-bound count of available last-50 checkpoint states ranges from `1` to `2`; runs with all last-50 model states available: `0/36`.

## Required Policy Because Test Curve Is Missing

- Tuning metric: `last50_validation_mse_mean`, with `best_validation_mse` as fallback/tie-break.
- Stability diagnostics: last-50 validation mean/std/range/CV.
- Final reporting from current artifacts: `test_mse_at_best_validation` and `final_test_mse` only.
- Optional future reporting: `last50_test_mse_mean` only for future runs that log per-round Test MSE, and only after validation-only recipe lock.

## Stop Condition

The requested stop condition is met: per-round Test MSE is unavailable and last-50 Test MSE cannot be computed from existing Sine artifacts. Therefore no Sine tuning runs were launched by this audit.
