# fedogda_s_focused_v3 screen

Selection is validation-only: finite/non-diverged; lowest `best_validation_mse`; tie lower `last_50_val_mse_std`; tie lower `final_vs_best_validation_gap`.

## Completion

- Completed candidate rows: 114.
- Missing/incomplete rows: 4.

## Outputs

- `experiments/curve_fitting_plots/png/fedogda_s_focused_v3/fedogda_s_focused_v3_screen_curves.png`
- `experiments/curve_fitting_plots/pdf/fedogda_s_focused_v3/fedogda_s_focused_v3_screen_curves.pdf`
- `experiments/curve_fitting_plots/csv/fedogda_s_focused_v3_screen_candidates.csv`
- `experiments/curve_fitting_plots/csv/fedogda_s_focused_v3_screen_selected.csv`

## Selected

- sin: lr `0.01`, cm `8`, lambda `0.01`, slr `1.5`, T `500`, R `3`, seed `0`, val `0.029642769`, test@best `0.030104425`, corr `0.887129`.
- step: lr `0.0075`, cm `20`, lambda `0.05`, slr `1`, T `600`, R `7`, seed `0`, val `0.015320748`, test@best `0.015592838`, corr `0.920359`.

## Invalid Or Missing Rows

- `v3_screen_step_proxy_step_fedogda_s_seed0_alpha0p5_T600_R7_batch256_glr0p0075_cm20_lam0p05_slr2`
- `v3_screen_step_proxy_step_fedogda_s_seed0_alpha0p5_T600_R7_batch256_glr0p0075_cm20_lam0p1_slr1`
- `v3_screen_step_proxy_step_fedogda_s_seed0_alpha0p5_T600_R7_batch256_glr0p0075_cm20_lam0p1_slr1p5`
- `v3_screen_step_proxy_step_fedogda_s_seed0_alpha0p5_T600_R7_batch256_glr0p0075_cm20_lam0p1_slr2`
