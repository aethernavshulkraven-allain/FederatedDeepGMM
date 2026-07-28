# fedogda_s_step_fast_v5

Selection is validation-only. Test MSE and curve diagnostics are reported only after validation ranking.

## Execution

- GPU launch: `gpurun -g 2` with `--gpu-ids 0,1 --max-parallel 2`.
- Thread caps: `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB=4`.
- Elapsed wall-clock: about `1h 29m 47s`.

## Completion

- Stage A completed: `13`; missing/invalid: `0`.
- Stage B completed: `4`; missing/invalid: `0`.
- Stage B rows materialized: `4`.

## Selected

- lr `0.0075`, cm `20`, lambda `0.05`, server_lr `1`, T `1500`, R `7`.
- confirmation seeds `0|1|2`.
- mean validation MSE `0.004777577` +/- `0.000541914`.
- mean test@best `0.004943105` +/- `0.000522217`.
- mean curve MAE `0.041889645`, corr `0.976461746`, amp ratio `1.185841067`.
- mean last-50 val std `0.014739026`, mean final-vs-best val gap `0.024312996`.
- plotted representative seed `0`: `v5_stage_a_full_budget_probe_step_fedogda_s_seed0_alpha0p5_T1500_R7_batch256_glr0p0075_cm20_lam0p05_slr1`.

## Promotion

- `incumbent_control` promoted=`true` reason=`always_confirm_seeds_1_2` val=`0.010456711` lr=`0.005` cm=`15` lambda=`0.1` server_lr=`1.5`.
- `best_non_incumbent_challenger` promoted=`true` reason=`beats_control_validation` val=`0.004615987` lr=`0.0075` cm=`20` lambda=`0.05` server_lr=`1`.

## FedGDA-S Reference

- FedGDA-S reference val `0.006003955`, test@best `0.006126142`.
- Previous FedOGDA-S incumbent val `0.010456711`, test@best `0.010636226`.
- FedOGDA-S beats FedGDA-S post-selection test reference: `true`.

## Outputs

- `experiments/curve_fitting_plots/png/fedogda_s_step_fast_v5/fedogda_s_step_fast_v5_best_step_optimistic_only.png`
- `experiments/curve_fitting_plots/pdf/fedogda_s_step_fast_v5/fedogda_s_step_fast_v5_best_step_optimistic_only.pdf`
- `experiments/curve_fitting_plots/csv/fedogda_s_step_fast_v5_all_candidates.csv`
- `experiments/curve_fitting_plots/csv/fedogda_s_step_fast_v5_stage_a_ranked.csv`
- `experiments/curve_fitting_plots/csv/fedogda_s_step_fast_v5_promoted_configs.csv`
- `experiments/curve_fitting_plots/csv/fedogda_s_step_fast_v5_confirmation_aggregate.csv`
- `experiments/curve_fitting_plots/csv/fedogda_s_step_fast_v5_final_selected.csv`
- `experiments/curve_fitting_plots/csv/fedogda_s_step_fast_v5_invalid_missing.csv`
