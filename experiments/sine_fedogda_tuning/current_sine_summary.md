# Current Low-Dimensional Sine FedGDA/FedOGDA Audit

Scope: completed federated Sine rows from `experiments/rerun_protocol_v1/manifest.csv` with methods `fedgda_d`, `fedgda_s`, `fedogda_d`, and `fedogda_s`.

This is a pre-tuning audit only. No hyperparameter selection was made using Test MSE, and no new training was launched.

## Artifact Count

- Completed Sine runs found: `36`.
- Diverged runs: `0`.
- Runs missing per-round Test MSE: `36`.
- Paired FedOGDA-vs-FedGDA comparisons: `18`.

## Existing Test-MSE Baseline

The table below uses scalar `test_mse_at_best_validation` from `metrics.json`. This is valid for reporting after validation-only checkpoint selection, but it is not a last-50 Test-MSE curve.

| mode | method | partition_alpha | runs | mean_best_validation_mse | mean_test_mse_at_best_validation | std_test_mse_at_best_validation | mean_final_test_mse | mean_last50_validation_mse_std | stable_cv_le_0p05_runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deterministic | fedgda_d | 0.1 | 3 | 0.083678913 | 0.086153943 | 0.003738607 | 0.088147039 | 0.00020026641 | 3 |
| deterministic | fedgda_d | 0.5 | 3 | 0.083707024 | 0.086196144 | 0.0037883913 | 0.087943086 | 0.0001810623 | 3 |
| deterministic | fedgda_d | 1 | 3 | 0.083629071 | 0.086106863 | 0.0037707639 | 0.08789507 | 0.00019224067 | 3 |
| deterministic | fedogda_d | 0.1 | 3 | 0.083667574 | 0.086211588 | 0.0036697093 | 0.087972439 | 8.9662258e-05 | 3 |
| deterministic | fedogda_d | 0.5 | 3 | 0.083725496 | 0.086272828 | 0.0037704504 | 0.087986882 | 0.00011243019 | 3 |
| deterministic | fedogda_d | 1 | 3 | 0.083617031 | 0.086155917 | 0.003744335 | 0.087934385 | 0.00012365326 | 3 |
| stochastic | fedgda_s | 0.1 | 3 | 0.076994668 | 0.078790298 | 0.0046679737 | 0.080481547 | 0.0031181189 | 2 |
| stochastic | fedgda_s | 0.5 | 3 | 0.076304426 | 0.078034135 | 0.0050974931 | 0.081633253 | 0.0024313611 | 2 |
| stochastic | fedgda_s | 1 | 3 | 0.076794439 | 0.078642541 | 0.002152011 | 0.081627515 | 0.002262188 | 2 |
| stochastic | fedogda_s | 0.1 | 3 | 0.083273687 | 0.085609375 | 0.0041426987 | 0.090966622 | 0.0010873379 | 3 |
| stochastic | fedogda_s | 0.5 | 3 | 0.083601371 | 0.086140245 | 0.0041675161 | 0.088860462 | 0.00036491939 | 3 |
| stochastic | fedogda_s | 1 | 3 | 0.083395516 | 0.085795117 | 0.0040959607 | 0.088003737 | 0.00012371282 | 3 |

## Paired Summary By Mode

| mode | pairs | fedogda_lower_test_mse_at_best_validation | fedogda_lower_final_test_mse | fedogda_lower_best_validation_mse | fedogda_lower_last50_validation_std | fedogda_lower_last50_validation_cv | mean_test_mse_gap_fedogda_minus_fedgda | mean_relative_gap_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deterministic | 9 | 2 | 5 | 5 | 4 | 4 | 6.112776e-05 | 0.072986452 |
| stochastic | 9 | 0 | 0 | 0 | 9 | 9 | 0.0073592544 | 9.4437246 |

## Paired Summary By Alpha

| alpha | pairs | fedogda_lower_test_mse_at_best_validation | fedogda_lower_final_test_mse | fedogda_lower_last50_validation_std | mean_test_mse_gap_fedogda_minus_fedgda |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 6 | 1 | 2 | 5 | 0.0034383608 |
| 0.5 | 6 | 0 | 1 | 4 | 0.0040913971 |
| 1 | 6 | 1 | 2 | 4 | 0.0036008153 |

## Paired Summary By Mode And Alpha

| mode | alpha | pairs | fedogda_lower_test_mse_at_best_validation | fedogda_lower_final_test_mse | fedogda_lower_last50_validation_std | fedogda_lower_last50_validation_cv | mean_test_mse_gap_fedogda_minus_fedgda |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deterministic | 0.1 | 3 | 1 | 2 | 2 | 2 | 5.7644609e-05 |
| deterministic | 0.5 | 3 | 0 | 1 | 1 | 1 | 7.6684094e-05 |
| deterministic | 1 | 3 | 1 | 2 | 1 | 1 | 4.9054578e-05 |
| stochastic | 0.1 | 3 | 0 | 0 | 3 | 3 | 0.0068190771 |
| stochastic | 0.5 | 3 | 0 | 0 | 3 | 3 | 0.0081061102 |
| stochastic | 1 | 3 | 0 | 0 | 3 | 3 | 0.007152576 |

## Interpretation

- Existing deterministic Sine FedOGDA-D is close to FedGDA-D but does not clearly win by `test_mse_at_best_validation`.
- Existing stochastic Sine FedOGDA-S is more stable by last-50 validation oscillation, but loses to FedGDA-S on scalar `test_mse_at_best_validation` in the current runs.
- Per-round Test MSE is not present in existing `mse_by_round.csv`, so the current artifacts cannot answer whether last-50 average Test MSE favors FedOGDA.
- Because the requested stop condition is met, this audit does not launch Sine tuning runs.

## Output Files

- Per-run audit: `experiments/sine_fedogda_tuning/current_sine_runs.csv`
- Pairwise audit: `experiments/sine_fedogda_tuning/current_sine_pairwise.csv`
- MSE logging audit: `experiments/sine_fedogda_tuning/mse_logging_audit.md`
- Metric policy: `experiments/sine_fedogda_tuning/metric_policy.md`
