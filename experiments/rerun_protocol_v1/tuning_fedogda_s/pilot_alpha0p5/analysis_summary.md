# FedOGDA-S Alpha 0.5 Tuning Pilot Analysis

Scope: `fedogda_s`, datasets `abs`, `step`, `linear`, alpha `0.5`, seeds `0,1,2`, 16 critic/weight-decay configs.

Selection rule: choose by validation only. Primary key is lowest mean `best_validation_mse` across seeds. Tie-breakers are lower mean `last_50_val_mse_std`, then lower mean `final_vs_best_validation_gap`. `diverged_count` must be zero. Test MSE is reported only after selection.

## Selected Tuned Configs

| dataset | alpha | critic_multiplier | weight_decay | mean_best_validation_mse | mean_test_mse_at_best_validation | std_test_mse_at_best_validation | mean_last_50_val_mse_std | diverged_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abs | 0.5 | 20 | 0.001 | 0.00593008088235 | 0.00605828534058 | 0.00334809097414 | 0.00126847227272 | 0 |
| linear | 0.5 | 20 | 0.001 | 0.000954419036 | 0.000967615052491 | 0.00073151855795 | 0.000255950708821 | 0 |
| step | 0.5 | 20 | 0.001 | 0.0281032442743 | 0.0286008227067 | 0.00257409564726 | 0.000398907161677 | 0 |

## FedGDA-S Baseline Table

| dataset | alpha | selected_critic_multiplier | selected_weight_decay | mean_validation_mse | mean_test_mse | test_std | oscillation_score | win_loss_vs_fedgda_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abs | 0.5 | 10 | 0.03 | 0.00264951785392 | 0.002709909146 | 0.00273169540481 | 0.00173260387593 | reference |
| linear | 0.5 | 10 | 0.03 | 0.000352983759796 | 0.000356481786872 | 0.00014172438579 | 0.00400625707727 | reference |
| step | 0.5 | 10 | 0.03 | 0.0259229728331 | 0.0262673881653 | 0.00189529169581 | 0.00298818003223 | reference |

## Current FedOGDA-S Baseline Table

| dataset | alpha | selected_critic_multiplier | selected_weight_decay | mean_validation_mse | mean_test_mse | test_std | oscillation_score | win_loss_vs_fedgda_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abs | 0.5 | 10 | 0.1 | 0.0150473407663 | 0.0153625747236 | 0.0120595365231 | 0.00266440943451 | loss |
| linear | 0.5 | 10 | 0.1 | 0.00287394996331 | 0.00293433955738 | 0.0030523854999 | 0.00112352519573 | loss |
| step | 0.5 | 10 | 0.1 | 0.0292367914921 | 0.0297860953175 | 0.00271853078107 | 0.000375021407205 | loss |

## Tuned FedOGDA-S Table

| dataset | alpha | selected_critic_multiplier | selected_weight_decay | mean_validation_mse | mean_test_mse | test_std | oscillation_score | win_loss_vs_fedgda_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abs | 0.5 | 20 | 0.001 | 0.00593008088235 | 0.00605828534058 | 0.00334809097414 | 0.00126847227272 | loss |
| linear | 0.5 | 20 | 0.001 | 0.000954419036 | 0.000967615052491 | 0.00073151855795 | 0.000255950708821 | loss |
| step | 0.5 | 20 | 0.001 | 0.0281032442743 | 0.0286008227067 | 0.00257409564726 | 0.000398907161677 | loss |

## Success Criteria Check

| dataset | selected_critic_multiplier | selected_weight_decay | tuned_mean_best_validation_mse | tuned_mean_test_mse_at_best_validation | tuned_test_ratio_vs_fedgda | tuned_oscillation_score | tuned_improves_over_current_fedogda_test | tuned_reduces_oscillation_vs_fedgda | tuned_better_test_than_fedgda | no_tuned_divergence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abs | 20 | 0.001 | 0.00593008088235 | 0.00605828534058 | 2.23560459564 | 0.00126847227272 | True | True | False | True |
| linear | 20 | 0.001 | 0.000954419036 | 0.000967615052491 | 2.71434639335 | 0.000255950708821 | True | True | False | True |
| step | 20 | 0.001 | 0.0281032442743 | 0.0286008227067 | 1.0888339003 | 0.000398907161677 | True | True | False | True |

Notes:

- `oscillation_score` is mean `last_50_val_mse_std` across seeds.
- `win_loss_vs_fedgda_s` uses post-selection mean `test_mse_at_best_validation`.
- `competitive` is not thresholded here; use `tuned_test_ratio_vs_fedgda` from `baseline_comparison.csv` to choose any tolerance.
