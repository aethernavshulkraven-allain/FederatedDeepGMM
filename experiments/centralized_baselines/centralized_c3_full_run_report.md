# Centralized C3 Full Run Report

## Scope

Launched the full centralized low-dimensional manifest:

```text
experiments/centralized_baselines/centralized_lowdim_manifest.csv
```

Expected matrix:

```text
4 datasets/functions x 3 methods x 3 seeds = 36 runs
```

Datasets/functions: `abs`, `step`, `linear`, `sin`.

Methods: `gda`, `sgda`, `oadam`.

Full outputs are under:

```text
results/centralized_lowdim_v1/
```

Smoke outputs were not used for reporting.

## Completion Summary

- Total expected runs: 36
- Completed runs: 36
- Failed runs: 0
- Skipped completed runs: 0
- Validation pass count: 36
- Validation fail count: 0
- Wall-clock launch runtime: 975.72 seconds (16.26 minutes)

Status counts:

```text
{'completed_valid': 36}
```

Validation counts:

```text
{'pass': 36}
```

## Centralized-Semantics Check

All validation-passing runs were checked for:

```text
training_scope == centralized
uses_clients == false
uses_fedavg_aggregation == false
uses_client_sampling == false
test_mse_used_for_selection == false
selection_metric_source == validation
```

All centralized config checks passed: `true`.

Best checkpoint selection remained validation-only. The validator confirmed `best_validation_round` matches the minimum validation MSE round in `mse_by_round.csv` for every run.

## Runtime Summary By Method

| method | runs | mean runner sec | mean wall sec | total wall sec |
| --- | ---: | ---: | ---: | ---: |
| gda | 12 | 31.807 | 33.308 | 399.699 |
| oadam | 12 | 22.304 | 23.787 | 285.438 |
| sgda | 12 | 22.432 | 23.928 | 287.140 |

## Runtime Summary By Function

| dataset | runs | mean runner sec | mean wall sec | total wall sec |
| --- | ---: | ---: | ---: | ---: |
| abs | 9 | 25.661 | 27.161 | 244.450 |
| linear | 9 | 25.630 | 27.144 | 244.294 |
| sin | 9 | 25.235 | 26.698 | 240.285 |
| step | 9 | 25.530 | 27.028 | 243.249 |

## Aggregate Results By Function And Method

Primary reporting metric here is `test_mse_at_best_validation`, computed after validation-only checkpoint selection. Lower is better.

| dataset | method | seeds | mean test@best_val | std test@best_val | mean best val | mean final test | mean runtime sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs | gda | 3 | 0.305203325 | 0.158315058 | 0.296594829 | 0.305203325 | 32.449 |
| abs | oadam | 3 | 0.00177951451 | 0.00082711982 | 0.00173833869 | 0.0273019727 | 21.873 |
| abs | sgda | 3 | 0.303930673 | 0.156576327 | 0.295411991 | 0.303930673 | 22.660 |
| linear | gda | 3 | 0.0843962836 | 0.0964093988 | 0.0821176197 | 0.114892954 | 32.312 |
| linear | oadam | 3 | 0.000759663735 | 0.000619417804 | 0.000743962058 | 0.0216644726 | 22.121 |
| linear | sgda | 3 | 0.0857471792 | 0.0960029483 | 0.0834273432 | 0.115113811 | 22.459 |
| sin | gda | 3 | 0.10113595 | 0.022348427 | 0.100242923 | 0.118828698 | 31.183 |
| sin | oadam | 3 | 0.0376761192 | 0.00343317351 | 0.0367957397 | 0.0414579451 | 21.940 |
| sin | sgda | 3 | 0.101199136 | 0.0219331854 | 0.100323343 | 0.118783719 | 22.584 |
| step | gda | 3 | 0.0582590652 | 0.0380661319 | 0.0576630927 | 0.0721480197 | 31.283 |
| step | oadam | 3 | 0.0172140142 | 0.00057276779 | 0.0169725664 | 0.0223010232 | 23.282 |
| step | sgda | 3 | 0.0581996134 | 0.0374331025 | 0.0576081499 | 0.0725897692 | 22.024 |

## Comparison-Ready Table: GDA vs SGDA vs OAdam

| dataset | GDA mean test@best_val | SGDA mean test@best_val | OAdam mean test@best_val | best lower-is-better |
| --- | ---: | ---: | ---: | --- |
| abs | 0.305203325 | 0.303930673 | 0.00177951451 | oadam |
| linear | 0.0843962836 | 0.0857471792 | 0.000759663735 | oadam |
| sin | 0.10113595 | 0.101199136 | 0.0376761192 | oadam |
| step | 0.0582590652 | 0.0581996134 | 0.0172140142 | oadam |

## Warnings

OAdam emitted the existing PyTorch `Tensor.add_` deprecation warning in 12 run logs. This warning did not block completion or validation; all OAdam runs wrote required artifacts, had finite histories, and passed validation.

No runner failures or validation failures occurred.

## Output Files

```text
experiments/centralized_baselines/centralized_c3_full_run_report.md
experiments/centralized_baselines/centralized_c3_full_run_results.csv
experiments/centralized_baselines/centralized_c3_validation_summary.csv
experiments/centralized_baselines/centralized_lowdim_summary_by_function_method.csv
experiments/centralized_baselines/centralized_c3_raw_run_log.json
experiments/centralized_baselines/c3_logs/
```

## Final C3 Verdict

All 36 full centralized low-dimensional runs completed and passed validation.
