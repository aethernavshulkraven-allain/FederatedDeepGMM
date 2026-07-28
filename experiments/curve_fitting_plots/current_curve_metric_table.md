# Current Curve-Fit Numeric Metrics Table

## Scope

This is Step 1 of the curve-fitting improvement plan: generate numeric curve MSE / MAE / max-error tables for the current generated plot artifacts. No training was launched, and no hyperparameter decision is made here.

Primary source files:

- `experiments/curve_fitting_plots/csv/curve_fit_metrics_all_runs.csv`
- `experiments/curve_fitting_plots/csv/curve_fit_pairwise_fedogda_vs_fedgda.csv`
- `experiments/curve_fitting_plots/csv/curve_fit_plot_index.csv`

Generated Step-1 outputs:

- `experiments/curve_fitting_plots/csv/current_curve_metric_table.csv`
- `experiments/curve_fitting_plots/csv/current_curve_pairwise_summary_by_function_alpha.csv`
- `experiments/curve_fitting_plots/csv/current_curve_selected_lowdim_summary.csv`
- `experiments/curve_fitting_plots/csv/current_curve_plot_coverage_counts.csv`
- `experiments/curve_fitting_plots/csv/current_curve_method_summary_all_sources.csv`

## Coverage

Metric rows by source family:

| source_family | metric_rows | included_rows | datasets |
| --- | --- | --- | --- |
| base_sweep | 144 | 144 | abs, linear, sin, step |
| centralized_c3_oadam | 12 | 12 | abs, linear, sin, step |
| centralized_c5_tuned_gda_sgda | 24 | 24 | abs, linear, sin, step |
| fedogda_s_tuning_pilot_all_configs | 135 | 135 | abs, linear, step |
| fedogda_s_tuning_pilot_selected | 9 | 9 | abs, linear, step |
| tuned_sine_a2_lite | 3 | 3 | sin |

Plot rows by family/status:

| plot_family | status | count |
| --- | --- | --- |
| all_methods_original | created | 36 |
| all_methods_original_aggregate | created | 12 |
| centralized | created | 4 |
| centralized_aggregate | created | 4 |
| coauthor_summary | created | 4 |
| coauthor_summary_2x2 | created | 1 |
| fedogda_s_tuning_pilot | created | 9 |
| fedogda_s_tuning_pilot_aggregate | created | 3 |
| main_pairwise | created | 36 |
| main_pairwise_aggregate | created | 12 |
| tuned_sine_a2_lite | created | 3 |
| tuned_sine_a2_lite_aggregate | created | 1 |

## Selected Low-Dimensional Summary

These are the currently selected low-dimensional plots from the existing curve-fitting report. Lower MSE/MAE/max-error is better. Negative gap means FedOGDA-D is lower than FedGDA-D. Tuned Sine is kept separate from the original sweep.

| function | source_family | alpha | pairs | fedogda_curve_mse_wins | mean_fedgda_curve_mse | mean_fedogda_curve_mse | mean_curve_mse_gap_fedogda_minus_fedgda | mean_fedgda_curve_mae | mean_fedogda_curve_mae | mean_fedgda_curve_max_abs | mean_fedogda_curve_max_abs | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Absolute | base_sweep | 1 | 3 | 2 | 0.01772472 | 0.0164571807 | -0.00126753924 | 0.102531716 | 0.0987821462 | 0.500816676 | 0.481766078 | selected original-sweep alpha=1.0 |
| Step | base_sweep | 0.1 | 3 | 3 | 0.0297879823 | 0.02918822 | -0.000599762314 | 0.142805218 | 0.141370411 | 0.629348087 | 0.612536285 | selected original-sweep alpha=0.1 |
| Linear | base_sweep | 0.1 | 3 | 2 | 0.00428732767 | 0.0029284127 | -0.00135891497 | 0.0442661011 | 0.0371540325 | 0.27765568 | 0.197817038 | selected original-sweep alpha=0.1 |
| Sine | tuned_sine_a2_lite | 1 | 3 | 3 | 0.0861068629 | 0.0800115346 | -0.00609532833 | 0.245105863 | 0.229406924 | 0.947548571 | 0.917271153 | validation-locked tuned Sine A2-lite |

## All Function/Alpha Pairwise Summary

Full pairwise summary is saved to CSV. Preview:

| source_family | dataset | function | alpha | pairs | fedogda_curve_mse_wins | fedogda_test_mse_wins | mean_fedgda_curve_mse | mean_fedogda_curve_mse | mean_curve_mse_gap | mean_fedgda_curve_mae | mean_fedogda_curve_mae | mean_curve_mae_gap | mean_fedgda_curve_max_abs | mean_fedogda_curve_max_abs | mean_curve_max_abs_gap | mean_fedgda_test_mse | mean_fedogda_test_mse | mean_test_mse_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_sweep | abs | Absolute | 0.1 | 3 | 2 | 2 | 0.0185482147 | 0.0176289575 | -0.000919257178 | 0.105482156 | 0.10300021 | -0.00248194561 | 0.508550742 | 0.495934252 | -0.0126164903 | 0.0185482147 | 0.0176289575 | -0.000919257178 |
| base_sweep | abs | Absolute | 0.5 | 3 | 2 | 2 | 0.0177515092 | 0.016589363 | -0.00116214621 | 0.102930153 | 0.0995040464 | -0.00342610676 | 0.499092181 | 0.482916134 | -0.016176047 | 0.0177515092 | 0.016589363 | -0.00116214621 |
| base_sweep | abs | Absolute | 1 | 3 | 2 | 2 | 0.01772472 | 0.0164571807 | -0.00126753924 | 0.102531716 | 0.0987821462 | -0.00374956963 | 0.500816676 | 0.481766078 | -0.0190505974 | 0.01772472 | 0.0164571807 | -0.00126753924 |
| base_sweep | linear | Linear | 0.1 | 3 | 2 | 2 | 0.00428732767 | 0.0029284127 | -0.00135891497 | 0.0442661011 | 0.0371540325 | -0.00711206864 | 0.27765568 | 0.197817038 | -0.0798386413 | 0.00428732767 | 0.0029284127 | -0.00135891497 |
| base_sweep | linear | Linear | 0.5 | 3 | 2 | 2 | 0.00423304302 | 0.00284033303 | -0.00139270999 | 0.0437202249 | 0.0365938427 | -0.00712638226 | 0.255855598 | 0.202482656 | -0.0533729419 | 0.00423304302 | 0.00284033303 | -0.00139270999 |
| base_sweep | linear | Linear | 1 | 3 | 2 | 2 | 0.00415789785 | 0.0028160473 | -0.00134185055 | 0.0427008631 | 0.0365757619 | -0.00612510124 | 0.254484874 | 0.201596019 | -0.0528888545 | 0.00415789785 | 0.0028160473 | -0.00134185055 |
| base_sweep | sin | Sine | 0.1 | 3 | 1 | 1 | 0.0861539434 | 0.0862115881 | 5.76446093e-05 | 0.245263825 | 0.244303507 | -0.000960318133 | 0.946691692 | 0.965035683 | 0.0183439906 | 0.0861539434 | 0.0862115881 | 5.76446093e-05 |
| base_sweep | sin | Sine | 0.5 | 3 | 0 | 0 | 0.0861961437 | 0.0862728278 | 7.6684094e-05 | 0.245064656 | 0.244107411 | -0.000957245669 | 0.9500185 | 0.96603068 | 0.0160121802 | 0.0861961437 | 0.0862728278 | 7.6684094e-05 |
| base_sweep | sin | Sine | 1 | 3 | 1 | 1 | 0.0861068629 | 0.0861559175 | 4.90545775e-05 | 0.245105863 | 0.24421174 | -0.000894122763 | 0.947548571 | 0.964521661 | 0.0169730898 | 0.0861068629 | 0.0861559175 | 4.90545775e-05 |
| base_sweep | step | Step | 0.1 | 3 | 3 | 3 | 0.0297879823 | 0.02918822 | -0.000599762314 | 0.142805218 | 0.141370411 | -0.00143480682 | 0.629348087 | 0.612536285 | -0.0168118016 | 0.0297879823 | 0.02918822 | -0.000599762314 |
| base_sweep | step | Step | 0.5 | 3 | 3 | 3 | 0.0297254879 | 0.0291784126 | -0.000547075214 | 0.142742093 | 0.141397124 | -0.00134496915 | 0.63230822 | 0.610361623 | -0.0219465971 | 0.0297254879 | 0.0291784126 | -0.000547075214 |
| base_sweep | step | Step | 1 | 3 | 3 | 3 | 0.0296753667 | 0.0291397604 | -0.000535606339 | 0.142651793 | 0.141318492 | -0.00133330039 | 0.63164329 | 0.613102819 | -0.0185404708 | 0.0296753667 | 0.0291397604 | -0.000535606339 |
| tuned_sine_a2_lite | sin | Sine | 1 | 3 | 3 | 3 | 0.0861068629 | 0.0800115346 | -0.00609532833 | 0.245105863 | 0.229406924 | -0.0156989392 | 0.947548571 | 0.917271153 | -0.0302774178 | 0.0861068629 | 0.0800115346 | -0.00609532833 |

## Notes

- These curve metrics are diagnostic/reporting metrics from saved prediction artifacts.
- They must not be used directly to select hyperparameters if the curve grid is test-derived.
- The next gated step is to identify the weakest function using these diagnostics plus visual shape, then separately design a validation-only tuning screen.
