# fedogda_s_sine_fast_v4

Selection is validation-only. Test MSE and curve diagnostics are reported only after validation ranking.

## Execution

- GPU launch: `gpurun -g 2` with `--gpu-ids 0,1 --max-parallel 2`.
- Thread caps: `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB=4`.
- Elapsed wall-clock: about `51m 13s` from `2026-07-18 23:21:30` to `2026-07-19 00:12:43` local time.
- Pipeline log: `logs/fedogda_s_sine_fast_v4/pipeline_20260718_232130.log`.

## Completion

- Stage A completed: `24`; missing/invalid: `0`.
- Stage B completed: `6`; missing/invalid: `0`.
- Stage C completed: `6`; missing/invalid: `0`.
- Stage C rows materialized: `6`.

## Selected

- lr `0.01`, cm `7`, lambda `0.005`, server_lr `1.75`, T `1000`, R `3`.
- confirmation seeds `0|1|2`.
- mean validation MSE `0.013317147` +/- `0.000677422`.
- mean test@best `0.013406540` +/- `0.000721780`.
- mean curve MAE `0.096267272`, corr `0.940952680`, amp ratio `1.112528086`.
- plotted representative seed `0`: `v4_stage_c_confirm_sin_fedogda_s_seed0_alpha1_T1000_R3_batch256_glr0p01_cm7_lam0p005_slr1p75`.

## Challenger Gate

- Best non-current challenger val `0.024144017`, curve MAE `0.109633883`, amp ratio `0.772269486`.
- Clears gate: `true` (requires val < `0.029642769`, MAE <= `0.134679880`, amp ratio >= `0.619`).

## Outputs

- `experiments/curve_fitting_plots/png/fedogda_s_sine_fast_v4/fedogda_s_sine_fast_v4_best_optimistic_only.png`
- `experiments/curve_fitting_plots/pdf/fedogda_s_sine_fast_v4/fedogda_s_sine_fast_v4_best_optimistic_only.pdf`
- `experiments/curve_fitting_plots/png/fedogda_s_sine_fast_v4/fedogda_s_sine_fast_v4_validation_heatmap.png`
- `experiments/curve_fitting_plots/pdf/fedogda_s_sine_fast_v4/fedogda_s_sine_fast_v4_validation_heatmap.pdf`
- `experiments/curve_fitting_plots/csv/fedogda_s_sine_fast_v4_all_candidates.csv`
- `experiments/curve_fitting_plots/csv/fedogda_s_sine_fast_v4_stage_a_selected.csv`
- `experiments/curve_fitting_plots/csv/fedogda_s_sine_fast_v4_stage_b_selected.csv`
- `experiments/curve_fitting_plots/csv/fedogda_s_sine_fast_v4_confirmation_aggregate.csv`
- `experiments/curve_fitting_plots/csv/fedogda_s_sine_fast_v4_final_selected.csv`

## Baseline

- Current v3 baseline val `0.029642769`, test@best `0.030104425`.
