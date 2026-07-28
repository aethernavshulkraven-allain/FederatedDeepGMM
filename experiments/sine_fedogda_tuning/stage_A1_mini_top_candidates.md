# Stage A1-mini Deterministic Sine Top Candidates

Candidate ranking used validation metrics only.

Test MSE columns were reported for transparency but were not used for selection.

Completed candidates analyzed: `12`.

| rank_validation_only | run_id | alpha | R | g_lr | critic_multiplier | server_lr | last50_validation_mse_mean | best_validation_mse | last50_validation_mse_cv | test_mse_at_best_validation | last50_test_mse_mean | last50_test_mse_status | diverged | finite_history | validated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | stage_A1_mini_sin_fedogda_d_seed0_alpha1p0_R3_cm15_slr1p5_glr0p002 | 1 | 3 | 0.002 | 15 | 1.5 | 0.087979011 | 0.087528339 | 0.0030342985 | 0.090252692 | 0.090798159 | available | False | True | True |
| 2 | stage_A1_mini_sin_fedogda_d_seed0_alpha1p0_R3_cm10_slr1p5_glr0p002 | 1 | 3 | 0.002 | 10 | 1.5 | 0.088515202 | 0.088122002 | 0.0033094834 | 0.09096078 | 0.091448912 | available | False | True | True |
| 3 | stage_A1_mini_sin_fedogda_d_seed0_alpha1p0_R2_cm15_slr1p5_glr0p002 | 1 | 2 | 0.002 | 15 | 1.5 | 0.089279605 | 0.088640668 | 0.004327653 | 0.091612174 | 0.092353031 | available | False | True | True |

Ranking key:

1. no divergence;
2. finite history;
3. lower `last50_validation_mse_mean`;
4. lower `best_validation_mse`;
5. lower `last50_validation_mse_cv`;
6. lower `last50_validation_mse_range`.

## FedGDA-D Baseline Fairness Notes

- Rank 1 candidate `R=3`, alpha `1.0`: matched FedGDA-D baseline exists.
- Rank 2 candidate `R=3`, alpha `1.0`: matched FedGDA-D baseline exists.
- Rank 3 candidate `R=2`, alpha `1.0`: matched FedGDA-D baseline must be run later.
