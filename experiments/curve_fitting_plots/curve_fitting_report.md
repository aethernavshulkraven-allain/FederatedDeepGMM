# Curve-Fitting Plot Report

Generated: 2026-07-11T11:25:23

## 1. Executive Summary

- Prediction artifacts considered: 327
- Prediction artifacts used in metrics: 327
- Prediction artifacts skipped: 0
- Plot files created: 250 (125 PNG + 125 PDF)
- Plot requests skipped: 0.
- Main deterministic FedOGDA-D vs FedGDA-D plots were generated for the original low-dimensional sweep. Tuned Sine A2-lite plots are separate.
- Centralized plots use reportable centralized outputs: C5 tuned GDA/SGDA plus C3 OAdam. Tiny smoke is not included.
- High-dimensional FEMNIST/CIFAR is not applicable for curve-fitting here because final validated low-dimensional-style `x`/`true_g` prediction artifacts are not complete.

## 2. Metric Direction and Selection Policy

Lower curve MSE, MAE, and max absolute error are better. The prediction used is validation-selected when a validation-selected key is available, with preference order `best_validation_prediction`, `best_prediction`, then `pred_best`. Test MSE and visual appearance were not used for selection. Tuned Sine A2-lite is kept separate from the original Sine sweep.

## 3. Artifact Inventory

Full inventory CSV: `experiments/curve_fitting_plots/csv/curve_fit_artifact_inventory.csv`

Compact inventory by family:

| source_family | included | count |
| --- | --- | --- |
| base_sweep | True | 144 |
| centralized_c3_oadam | True | 12 |
| centralized_c5_tuned_gda_sgda | True | 24 |
| fedogda_s_tuning_pilot_all_configs | True | 135 |
| fedogda_s_tuning_pilot_selected | True | 9 |
| tuned_sine_a2_lite | True | 3 |

Skipped artifact reasons:

_No rows._

## 4. Curve-Fit Metrics

All-run metric CSV: `experiments/curve_fitting_plots/csv/curve_fit_metrics_all_runs.csv`

Aggregate metric sample:

| source_family | dataset | method_label | alpha | runs | mean_curve_mse | mean_curve_mae | mean_curve_max_abs_error | mean_test_mse_at_best_validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_sweep | abs | FedGDA-D | 0.1 | 3 | 0.0185482 | 0.105482 | 0.508551 | 0.0185482 |
| base_sweep | abs | FedGDA-D | 0.5 | 3 | 0.0177515 | 0.10293 | 0.499092 | 0.0177515 |
| base_sweep | abs | FedGDA-D | 1.0 | 3 | 0.0177247 | 0.102532 | 0.500817 | 0.0177247 |
| base_sweep | abs | FedGDA-S | 0.1 | 3 | 0.00283293 | 0.0371398 | 0.189223 | 0.00283293 |
| base_sweep | abs | FedGDA-S | 0.5 | 3 | 0.00270991 | 0.0364786 | 0.16492 | 0.00270991 |
| base_sweep | abs | FedGDA-S | 1.0 | 3 | 0.00286926 | 0.0373555 | 0.165741 | 0.00286926 |
| base_sweep | abs | FedOGDA-D | 0.1 | 3 | 0.017629 | 0.103 | 0.495934 | 0.017629 |
| base_sweep | abs | FedOGDA-D | 0.5 | 3 | 0.0165894 | 0.099504 | 0.482916 | 0.0165894 |
| base_sweep | abs | FedOGDA-D | 1.0 | 3 | 0.0164572 | 0.0987821 | 0.481766 | 0.0164572 |
| base_sweep | abs | FedOGDA-S | 0.1 | 3 | 0.0120175 | 0.0827995 | 0.382939 | 0.0120175 |
| base_sweep | abs | FedOGDA-S | 0.5 | 3 | 0.0153626 | 0.0951764 | 0.411144 | 0.0153626 |
| base_sweep | abs | FedOGDA-S | 1.0 | 3 | 0.0161022 | 0.0961364 | 0.47736 | 0.0161022 |
| base_sweep | linear | FedGDA-D | 0.1 | 3 | 0.00428733 | 0.0442661 | 0.277656 | 0.00428733 |
| base_sweep | linear | FedGDA-D | 0.5 | 3 | 0.00423304 | 0.0437202 | 0.255856 | 0.00423304 |
| base_sweep | linear | FedGDA-D | 1.0 | 3 | 0.0041579 | 0.0427009 | 0.254485 | 0.0041579 |
| base_sweep | linear | FedGDA-S | 0.1 | 3 | 0.000384863 | 0.0155706 | 0.0485958 | 0.000384863 |
| base_sweep | linear | FedGDA-S | 0.5 | 3 | 0.000356482 | 0.015778 | 0.0829357 | 0.000356482 |
| base_sweep | linear | FedGDA-S | 1.0 | 3 | 0.00041891 | 0.0155119 | 0.0743635 | 0.00041891 |
| base_sweep | linear | FedOGDA-D | 0.1 | 3 | 0.00292841 | 0.037154 | 0.197817 | 0.00292841 |
| base_sweep | linear | FedOGDA-D | 0.5 | 3 | 0.00284033 | 0.0365938 | 0.202483 | 0.00284033 |
| base_sweep | linear | FedOGDA-D | 1.0 | 3 | 0.00281605 | 0.0365758 | 0.201596 | 0.00281605 |
| base_sweep | linear | FedOGDA-S | 0.1 | 3 | 0.00279142 | 0.0377036 | 0.235579 | 0.00279142 |
| base_sweep | linear | FedOGDA-S | 0.5 | 3 | 0.00293434 | 0.0370368 | 0.199347 | 0.00293434 |
| base_sweep | linear | FedOGDA-S | 1.0 | 3 | 0.00257821 | 0.0353392 | 0.192178 | 0.00257821 |
| base_sweep | sin | FedGDA-D | 0.1 | 3 | 0.0861539 | 0.245264 | 0.946692 | 0.0861539 |
| base_sweep | sin | FedGDA-D | 0.5 | 3 | 0.0861961 | 0.245065 | 0.950018 | 0.0861961 |
| base_sweep | sin | FedGDA-D | 1.0 | 3 | 0.0861069 | 0.245106 | 0.947549 | 0.0861069 |
| base_sweep | sin | FedGDA-S | 0.1 | 3 | 0.0787903 | 0.230029 | 0.914113 | 0.0787903 |
| base_sweep | sin | FedGDA-S | 0.5 | 3 | 0.0780341 | 0.236019 | 0.854478 | 0.0780341 |
| base_sweep | sin | FedGDA-S | 1.0 | 3 | 0.0786425 | 0.230656 | 0.896778 | 0.0786425 |

FedOGDA-D vs FedGDA-D pairwise summary:

| source_family | function | pairs | fedogda_curve_mse_wins | fedogda_test_mse_wins | mean_curve_mse_gap | mean_test_mse_gap |
| --- | --- | --- | --- | --- | --- | --- |
| base_sweep | Absolute | 9 | 6 | 6 | -0.00111631 | -0.00111631 |
| base_sweep | Linear | 9 | 6 | 6 | -0.00136449 | -0.00136449 |
| base_sweep | Sine | 9 | 2 | 2 | 6.11278e-05 | 6.11278e-05 |
| base_sweep | Step | 9 | 9 | 9 | -0.000560815 | -0.000560815 |
| tuned_sine_a2_lite | Sine | 3 | 3 | 3 | -0.00609533 | -0.00609533 |

Pairwise CSV: `experiments/curve_fitting_plots/csv/curve_fit_pairwise_fedogda_vs_fedgda.csv`

## 5. Plot Index

Plot index CSV: `experiments/curve_fitting_plots/csv/curve_fit_plot_index.csv`

Created plot counts:

| plot_family | created_plots |
| --- | --- |
| all_methods_original | 36 |
| all_methods_original_aggregate | 12 |
| centralized | 4 |
| centralized_aggregate | 4 |
| coauthor_summary | 4 |
| coauthor_summary_2x2 | 1 |
| fedogda_s_tuning_pilot | 9 |
| fedogda_s_tuning_pilot_aggregate | 3 |
| main_pairwise | 36 |
| main_pairwise_aggregate | 12 |
| tuned_sine_a2_lite | 3 |
| tuned_sine_a2_lite_aggregate | 1 |

Skipped plot counts:

_No rows._

## 6. Recommended Plots to Send Geetika

| plot_family | dataset | alpha | seed | png_path | pdf_path | methods |
| --- | --- | --- | --- | --- | --- | --- |
| coauthor_summary_2x2 | lowdim | selected_by_validation | aggregate_or_seed0 | experiments/curve_fitting_plots/png/coauthor_summary/lowdim_deterministic_summary_2x2.png | experiments/curve_fitting_plots/pdf/coauthor_summary/lowdim_deterministic_summary_2x2.pdf | FedGDA-D\|FedOGDA-D/Tuned FedOGDA-D |
| tuned_sine_a2_lite_aggregate | sin | 1.0 | aggregate | experiments/curve_fitting_plots/png/tuned_sine_a2_lite/sine_a2_lite_all_seeds_mean.png | experiments/curve_fitting_plots/pdf/tuned_sine_a2_lite/sine_a2_lite_all_seeds_mean.pdf | FedGDA-D\|Tuned FedOGDA-D |
| main_pairwise_aggregate | abs | 1.0 | aggregate | experiments/curve_fitting_plots/png/main_pairwise_aggregate/abs_alpha1p0_fedgda_d_vs_fedogda_d_mean.png | experiments/curve_fitting_plots/pdf/main_pairwise_aggregate/abs_alpha1p0_fedgda_d_vs_fedogda_d_mean.pdf | FedGDA-D\|FedOGDA-D |
| main_pairwise_aggregate | step | 0.1 | aggregate | experiments/curve_fitting_plots/png/main_pairwise_aggregate/step_alpha0p1_fedgda_d_vs_fedogda_d_mean.png | experiments/curve_fitting_plots/pdf/main_pairwise_aggregate/step_alpha0p1_fedgda_d_vs_fedogda_d_mean.pdf | FedGDA-D\|FedOGDA-D |
| main_pairwise_aggregate | linear | 0.1 | aggregate | experiments/curve_fitting_plots/png/main_pairwise_aggregate/linear_alpha0p1_fedgda_d_vs_fedogda_d_mean.png | experiments/curve_fitting_plots/pdf/main_pairwise_aggregate/linear_alpha0p1_fedgda_d_vs_fedogda_d_mean.pdf | FedGDA-D\|FedOGDA-D |
| centralized_aggregate | abs | na | aggregate | experiments/curve_fitting_plots/png/centralized_aggregate/abs_centralized_gda_sgda_oadam_mean.png | experiments/curve_fitting_plots/pdf/centralized_aggregate/abs_centralized_gda_sgda_oadam_mean.pdf | DeepGMM-GDA\|DeepGMM-SGDA\|DeepGMM-OAdam |
| centralized_aggregate | linear | na | aggregate | experiments/curve_fitting_plots/png/centralized_aggregate/linear_centralized_gda_sgda_oadam_mean.png | experiments/curve_fitting_plots/pdf/centralized_aggregate/linear_centralized_gda_sgda_oadam_mean.pdf | DeepGMM-GDA\|DeepGMM-SGDA\|DeepGMM-OAdam |
| centralized_aggregate | sin | na | aggregate | experiments/curve_fitting_plots/png/centralized_aggregate/sin_centralized_gda_sgda_oadam_mean.png | experiments/curve_fitting_plots/pdf/centralized_aggregate/sin_centralized_gda_sgda_oadam_mean.pdf | DeepGMM-GDA\|DeepGMM-SGDA\|DeepGMM-OAdam |

Why these are useful:

- `lowdim_deterministic_summary_2x2` is the cleanest four-function visual summary.
- Tuned Sine A2-lite all-seed mean directly supports the scoped Sine claim.
- Absolute/Step/Linear aggregate pairwise plots show the original deterministic FedOGDA-D vs FedGDA-D behavior without mixing in tuning extensions.
- Centralized aggregate plots show the reportable DeepGMM centralized baselines after C5 tuning for GDA/SGDA and C3 OAdam.

Summary-plot selection rule:

| function | alpha | source | selection_rule |
| --- | --- | --- | --- |
| Absolute | 1.0 | base_sweep | largest mean best-validation-MSE improvement across seeds |
| Step | 0.1 | base_sweep | largest mean best-validation-MSE improvement across seeds |
| Linear | 0.1 | base_sweep | largest mean best-validation-MSE improvement across seeds |
| Sine | 1.0 | tuned_sine_a2_lite | pre-locked tuned Sine A2-lite validation-only recipe |

## 7. Caveats

- Legacy base sweep lacks per-round Test MSE; curve plots use saved prediction artifacts and scalar metrics, not reconstructed last-50 Test MSE.
- Curve plots use saved sorted test-point predictions from `predictions.npz`; they are not dense-grid checkpoint re-evaluations unless the saved artifact itself is dense.
- Do not overclaim visual superiority if numeric metrics disagree. Use the CSV metrics as the authoritative numeric readout.
- Tiny centralized smoke runs are excluded from paper-facing plots.
- High-dimensional FEMNIST/CIFAR should be summarized with MSE bars/tables after validated runs exist, not with low-dimensional curve-fitting plots.
