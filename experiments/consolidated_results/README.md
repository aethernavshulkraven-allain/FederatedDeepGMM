# Consolidated result index

Generated from local artifacts at `2026-07-11T05:09:20.190501+00:00`.

This index does not claim paper reproduction. Current synthetic data is reproducible but is not verified as paper-aligned. Hyperparameter candidates are never selected using Test MSE; `test_mse_at_best_validation` is reported only for the checkpoint chosen by validation.

## Inventory

| artifact status | runs |
| --- | --- |
| archived_failure | 4 |
| golden_reference | 1 |
| primary | 192 |
| smoke | 14 |
| tuning | 215 |

| result family | runs |
| --- | --- |
| _failed | 4 |
| _golden | 1 |
| _smoke | 4 |
| abs | 13 |
| centralized_lowdim_v1 | 36 |
| centralized_lowdim_v1_smoke | 3 |
| centralized_lowdim_v1_smoke_tiny | 3 |
| centralized_lowdim_v1_tuning | 56 |
| rerun_protocol_v1 | 144 |
| rerun_protocol_v1_smoke | 1 |
| rerun_protocol_v1_tuning | 144 |
| sine_fedogda_tuning | 15 |
| step | 2 |

## Files

- `all_runs.csv`: every discovered `metrics.json`, joined to its `effective_config.json` when present.
- `aggregates_by_exact_config.csv`: seed aggregates with all scientifically relevant configuration fields in the grouping key.
- `primary_aggregates.csv`: the exact-config aggregate restricted to primary artifacts.
- `supplementary_result_files.csv`: existing derived reports/summaries and legacy result files outside the run directories.
- `inventory.json`: machine-readable counts and generation metadata.

Tuning, smoke, archived failures, and golden references remain in the ledger but are excluded from the primary table. Existing tuning-selection reports remain authoritative because their choices are validation-driven.
The supplementary catalog contains 62 files. It is an index, not a second metric source: many entries are derived from the runs in `all_runs.csv`.

## Primary result aggregates

| family | dataset | method | alpha | runs | test MSE @ best val | best val MSE | diverged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| abs | abs | fedgda_s | 0.5 | 5 | 0.00225705 ± 0.0030789 | 0.00219313 ± 0.00300467 | 0 |
| abs | abs | fedogda_s | 0.5 | 5 | 0.00161265 ± 0.00195759 | 0.00156943 ± 0.00192188 | 0 |
| centralized_lowdim_v1 | abs | gda | — | 3 | 0.305203 ± 0.158315 | 0.296595 ± 0.151775 | 0 |
| centralized_lowdim_v1 | abs | oadam | — | 3 | 0.00177951 ± 0.00082712 | 0.00173834 ± 0.000782066 | 0 |
| centralized_lowdim_v1 | abs | sgda | — | 3 | 0.303931 ± 0.156576 | 0.295412 ± 0.15014 | 0 |
| centralized_lowdim_v1 | linear | gda | — | 3 | 0.0843963 ± 0.0964094 | 0.0821176 ± 0.0938242 | 0 |
| centralized_lowdim_v1 | linear | oadam | — | 3 | 0.000759664 ± 0.000619418 | 0.000743962 ± 0.000601607 | 0 |
| centralized_lowdim_v1 | linear | sgda | — | 3 | 0.0857472 ± 0.0960029 | 0.0834273 ± 0.0933817 | 0 |
| centralized_lowdim_v1 | sin | gda | — | 3 | 0.101136 ± 0.0223484 | 0.100243 ± 0.0236837 | 0 |
| centralized_lowdim_v1 | sin | oadam | — | 3 | 0.0376761 ± 0.00343317 | 0.0367957 ± 0.0031499 | 0 |
| centralized_lowdim_v1 | sin | sgda | — | 3 | 0.101199 ± 0.0219332 | 0.100323 ± 0.0232607 | 0 |
| centralized_lowdim_v1 | step | gda | — | 3 | 0.0582591 ± 0.0380661 | 0.0576631 ± 0.0377369 | 0 |
| centralized_lowdim_v1 | step | oadam | — | 3 | 0.017214 ± 0.000572768 | 0.0169726 ± 0.00049668 | 0 |
| centralized_lowdim_v1 | step | sgda | — | 3 | 0.0581996 ± 0.0374331 | 0.0576081 ± 0.0370755 | 0 |
| rerun_protocol_v1 | abs | fedgda_d | 0.1 | 3 | 0.0185482 ± 0.0164181 | 0.0180862 ± 0.015988 | 0 |
| rerun_protocol_v1 | abs | fedgda_d | 0.5 | 3 | 0.0177515 ± 0.0160089 | 0.0173095 ± 0.015594 | 0 |
| rerun_protocol_v1 | abs | fedgda_d | 1 | 3 | 0.0177247 ± 0.0163049 | 0.0172902 ± 0.0158821 | 0 |
| rerun_protocol_v1 | abs | fedgda_s | 0.1 | 3 | 0.00283293 ± 0.00310859 | 0.00275588 ± 0.00302447 | 0 |
| rerun_protocol_v1 | abs | fedgda_s | 0.5 | 3 | 0.00270991 ± 0.00334563 | 0.00264952 ± 0.00327342 | 0 |
| rerun_protocol_v1 | abs | fedgda_s | 1 | 3 | 0.00286926 ± 0.00350378 | 0.00280334 ± 0.00343801 | 0 |
| rerun_protocol_v1 | abs | fedogda_d | 0.1 | 3 | 0.017629 ± 0.0154356 | 0.0171935 ± 0.0150336 | 0 |
| rerun_protocol_v1 | abs | fedogda_d | 0.5 | 3 | 0.0165894 ± 0.0149232 | 0.0161789 ± 0.0145402 | 0 |
| rerun_protocol_v1 | abs | fedogda_d | 1 | 3 | 0.0164572 ± 0.0151039 | 0.0160588 ± 0.0147178 | 0 |
| rerun_protocol_v1 | abs | fedogda_s | 0.1 | 3 | 0.0120175 ± 0.01159 | 0.0117828 ± 0.0113329 | 0 |
| rerun_protocol_v1 | abs | fedogda_s | 0.5 | 3 | 0.0153626 ± 0.0147699 | 0.0150473 ± 0.014493 | 0 |
| rerun_protocol_v1 | abs | fedogda_s | 1 | 3 | 0.0161022 ± 0.0158602 | 0.0157127 ± 0.0154021 | 0 |
| rerun_protocol_v1 | linear | fedgda_d | 0.1 | 3 | 0.00428733 ± 0.0058206 | 0.00424503 ± 0.00575195 | 0 |
| rerun_protocol_v1 | linear | fedgda_d | 0.5 | 3 | 0.00423304 ± 0.00581815 | 0.00412228 ± 0.00562948 | 0 |
| rerun_protocol_v1 | linear | fedgda_d | 1 | 3 | 0.0041579 ± 0.00577115 | 0.00404959 ± 0.00558586 | 0 |
| rerun_protocol_v1 | linear | fedgda_s | 0.1 | 3 | 0.000384863 ± 0.000357279 | 0.000382283 ± 0.000356009 | 0 |
| rerun_protocol_v1 | linear | fedgda_s | 0.5 | 3 | 0.000356482 ± 0.000173576 | 0.000352984 ± 0.000172032 | 0 |
| rerun_protocol_v1 | linear | fedgda_s | 1 | 3 | 0.00041891 ± 0.000291057 | 0.000412306 ± 0.000283306 | 0 |
| rerun_protocol_v1 | linear | fedogda_d | 0.1 | 3 | 0.00292841 ± 0.00403187 | 0.00287798 ± 0.00394829 | 0 |
| rerun_protocol_v1 | linear | fedogda_d | 0.5 | 3 | 0.00284033 ± 0.00387466 | 0.00279637 ± 0.00380219 | 0 |
| rerun_protocol_v1 | linear | fedogda_d | 1 | 3 | 0.00281605 ± 0.00383223 | 0.00277367 ± 0.0037625 | 0 |
| rerun_protocol_v1 | linear | fedogda_s | 0.1 | 3 | 0.00279142 ± 0.00298998 | 0.00274531 ± 0.00290974 | 0 |
| rerun_protocol_v1 | linear | fedogda_s | 0.5 | 3 | 0.00293434 ± 0.00373839 | 0.00287395 ± 0.0036388 | 0 |
| rerun_protocol_v1 | linear | fedogda_s | 1 | 3 | 0.00257821 ± 0.00348634 | 0.00254773 ± 0.00344042 | 0 |
| rerun_protocol_v1 | sin | fedgda_d | 0.1 | 3 | 0.0861539 ± 0.00457884 | 0.0836789 ± 0.00410022 | 0 |
| rerun_protocol_v1 | sin | fedgda_d | 0.5 | 3 | 0.0861961 ± 0.00463981 | 0.083707 ± 0.0041716 | 0 |
| rerun_protocol_v1 | sin | fedgda_d | 1 | 3 | 0.0861069 ± 0.00461822 | 0.0836291 ± 0.00412314 | 0 |
| rerun_protocol_v1 | sin | fedgda_s | 0.1 | 3 | 0.0787903 ± 0.00571708 | 0.0769947 ± 0.00565687 | 0 |
| rerun_protocol_v1 | sin | fedgda_s | 0.5 | 3 | 0.0780341 ± 0.00624313 | 0.0763044 ± 0.00580005 | 0 |
| rerun_protocol_v1 | sin | fedgda_s | 1 | 3 | 0.0786425 ± 0.00263566 | 0.0767944 ± 0.002663 | 0 |
| rerun_protocol_v1 | sin | fedogda_d | 0.1 | 3 | 0.0862116 ± 0.00449446 | 0.0836676 ± 0.00410445 | 0 |
| rerun_protocol_v1 | sin | fedogda_d | 0.5 | 3 | 0.0862728 ± 0.00461784 | 0.0837255 ± 0.00422995 | 0 |
| rerun_protocol_v1 | sin | fedogda_d | 1 | 3 | 0.0861559 ± 0.00458586 | 0.083617 ± 0.00416584 | 0 |
| rerun_protocol_v1 | sin | fedogda_s | 0.1 | 3 | 0.0856094 ± 0.00507375 | 0.0832737 ± 0.00435101 | 0 |
| rerun_protocol_v1 | sin | fedogda_s | 0.5 | 3 | 0.0861402 ± 0.00510414 | 0.0836014 ± 0.00457686 | 0 |
| rerun_protocol_v1 | sin | fedogda_s | 1 | 3 | 0.0857951 ± 0.00501651 | 0.0833955 ± 0.00437789 | 0 |
| rerun_protocol_v1 | step | fedgda_d | 0.1 | 3 | 0.029788 ± 0.00318905 | 0.0292478 ± 0.00303053 | 0 |
| rerun_protocol_v1 | step | fedgda_d | 0.5 | 3 | 0.0297255 ± 0.0031464 | 0.0291797 ± 0.00299704 | 0 |
| rerun_protocol_v1 | step | fedgda_d | 1 | 3 | 0.0296754 ± 0.003155 | 0.0291297 ± 0.00300404 | 0 |
| rerun_protocol_v1 | step | fedgda_s | 0.1 | 3 | 0.027293 ± 0.00180015 | 0.0269202 ± 0.00178863 | 0 |
| rerun_protocol_v1 | step | fedgda_s | 0.5 | 3 | 0.0262674 ± 0.00232125 | 0.025923 ± 0.00224726 | 0 |
| rerun_protocol_v1 | step | fedgda_s | 1 | 3 | 0.0272549 ± 0.00201374 | 0.0268796 ± 0.00197805 | 0 |
| rerun_protocol_v1 | step | fedogda_d | 0.1 | 3 | 0.0291882 ± 0.00336389 | 0.028655 ± 0.00319305 | 0 |
| rerun_protocol_v1 | step | fedogda_d | 0.5 | 3 | 0.0291784 ± 0.00335879 | 0.0286475 ± 0.00318314 | 0 |
| rerun_protocol_v1 | step | fedogda_d | 1 | 3 | 0.0291398 ± 0.00334716 | 0.0286043 ± 0.00317468 | 0 |
| rerun_protocol_v1 | step | fedogda_s | 0.1 | 3 | 0.0291816 ± 0.00355441 | 0.0286646 ± 0.00337932 | 0 |
| rerun_protocol_v1 | step | fedogda_s | 0.5 | 3 | 0.0297861 ± 0.00332951 | 0.0292368 ± 0.00324424 | 0 |
| rerun_protocol_v1 | step | fedogda_s | 1 | 3 | 0.0293306 ± 0.00360404 | 0.0287884 ± 0.00343101 | 0 |
| step | step | fedgda_s | 0.5 | 1 | 0.0148086 ± 0 | 0.014504 ± 0 | 0 |
| step | step | fedogda_s | 0.5 | 1 | 0.0185379 ± 0 | 0.018185 ± 0 | 0 |
