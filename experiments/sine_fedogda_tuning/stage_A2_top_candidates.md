# Stage A2 Current Results

Full A2 was stopped after the first completed run because of quota risk.

Candidate ranking used validation metrics only. Test MSE columns are reported for transparency but were not used for selection.

Completed candidates analyzed: `1`.

| rank_validation_only | run_id | alpha | seed | R | g_lr | critic_multiplier | server_lr | last50_validation_mse_mean | best_validation_mse | last50_validation_mse_cv | test_mse_at_best_validation | last50_test_mse_mean | validated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | stage_A2_from_A1_mini_sin_fedogda_d_seed0_alpha1p0_R3_cm15_slr1.5_glr0p002 | 1.0 | 0 | 3 | 0.002 | 15.0 | 1.5 | 0.081747725 | 0.081073614 | 0.0047554695 | 0.083210219 | 0.083900514 | True |

See `stage_A2_lite_decision.md` for the selected continuation plan.