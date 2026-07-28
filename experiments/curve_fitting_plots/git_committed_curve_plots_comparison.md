# Git-Committed Coauthor Curve Plot Comparison

## Correction

The earlier comparison against `experiments/sine_fedogda_tuning/plots/a2_lite_*.png` was not the plot family you meant. I re-checked local git history and found the older committed, non-our-work curve plots under `fedgmm/sp_decentralized_mnist_lr_example/`.

The main committed plot family is:

- `fedgmm/sp_decentralized_mnist_lr_example/All_3.png` from Geetika commit `895bc3c` on 2026-06-09.
- Historical versions `All_3latest.png`, `All_3best.png`, and `All_3best2.png` from Geetika commits `7a8e756` and `8959997`.
- Earlier `plot_abs_comparison.png`, `plot_linear_comparison.png`, and `linear_gda_vs_ogda_comparison.png` from `ggauranshi-03` commit `eef5a22`.

## Inventory From Git History

| path | commit | author | date | current_tree | content |
| --- | --- | --- | --- | --- | --- |
| fedgmm/sp_decentralized_mnist_lr_example/All_3.png | 895bc3c | Geetika <geetikai@iiitd.ac.in> | 2026-06-09 | True | Absolute/Step/Linear combined curve plot; GDA/OGDA variants |
| fedgmm/sp_decentralized_mnist_lr_example/All_3latest.png | 7a8e756 | Geetika <geetikai@iiitd.ac.in> | 2026-05-26 | True | Absolute/Step/Linear combined curve plot; includes minibatch/full-batch OGDA variants in Absolute/Step panels |
| fedgmm/sp_decentralized_mnist_lr_example/All_3best.png | 8959997 | Geetika <geetikai@iiitd.ac.in> | 2026-05-25 | False | Historical Absolute/Step/Linear combined curve plot; deleted later |
| fedgmm/sp_decentralized_mnist_lr_example/All_3best2.png | 8959997 | Geetika <geetikai@iiitd.ac.in> | 2026-05-25 | False | Historical Absolute/Step/Linear combined curve plot; deleted later |
| fedgmm/sp_decentralized_mnist_lr_example/All_3.png | 69bb595 | ggauranshi-03 <gauranshigupta2000@gmail.com> | 2026-05-06 | overwritten_later | Original combined plot before Geetika updates |
| fedgmm/sp_decentralized_mnist_lr_example/linear_gda_vs_ogda_comparison.png | eef5a22 | ggauranshi-03 <gauranshigupta2000@gmail.com> | 2026-05-03 | True | Single Linear GDA vs OGDA curve plot |
| fedgmm/sp_decentralized_mnist_lr_example/plot_abs_comparison.png | eef5a22 | ggauranshi-03 <gauranshigupta2000@gmail.com> | 2026-05-03 | True | Single Absolute GDA curve plot |
| fedgmm/sp_decentralized_mnist_lr_example/plot_linear_comparison.png | eef5a22 | ggauranshi-03 <gauranshigupta2000@gmail.com> | 2026-05-03 | True | Single Linear GDA vs OGDA curve plot |
| fedgmm/sp_decentralized_mnist_lr_example/abs_points_with_ogda.png | 7936f8a | Geetika <geetikai@iiitd.ac.in> | 2026-05-07 | True | Absolute plot with centralized/federated methods; partly hand-entered points plus OGDA CSV |

## What These Old Plots Show

The committed coauthor plot family covers:

- Absolute
- Step
- Linear

I did not find a committed Sine curve-fitting plot in this older `All_3*` / `curve_plot.py` family. Sine-related committed files do exist elsewhere, including configs/logs/results and `data/zoo/sin.npz`, but not as this older tuned curve plot.

The current `curve_plot.py` builds `All_3.png` from local `.npy` files such as `results_abs_sgd_x.npy`, `results_abs_ogda_y_pred.npy`, `results_step_ogda_y_pred.npy`, etc. Some lines shown in the later committed images reference files like `results_abs_ogda_y_prednew.npy` and `results_step_ogda_y_prednew_fullbatch.npy` that are not currently tracked, so not every plotted trace is reconstructable from tracked source arrays.

`plot_abs_points_with_ogda.py` is even more manual: most curves are hard-coded point lists, with only FedDeepGMM-OGDA loaded from `results_abs_ogda_xy.csv`.

## Metrics From The Tracked Old `.npy` Arrays

These are computed against each old array's own saved `y_true`. This is useful for understanding the old plot source, but it is not a matched comparison to our validated rerun protocol because seed/alpha/T/R/checkpoint-selection metadata are absent.

| dataset | method_in_file | plotted_label_approx | n_points | curve_mse_vs_saved_y_true | curve_mae_vs_saved_y_true | curve_max_abs_vs_saved_y_true | metadata_available |
| --- | --- | --- | --- | --- | --- | --- | --- |
| abs | sgd | FedDeepGMM-GDA | 20000 | 0.0360315372 | 0.153002419 | 0.807288793 | False |
| abs | ogda | FedDeepGMM-OGDA | 20000 | 0.000782758091 | 0.0202629554 | 0.128891937 | False |
| linear | sgd | FedDeepGMM-GDA | 20000 | 0.000194422514 | 0.0117661987 | 0.0397717388 | False |
| linear | ogda | FedDeepGMM-OGDA | 20000 | 8.35776141e-05 | 0.00800351804 | 0.0230782223 | False |
| step | sgd | FedDeepGMM-GDA | 20000 | 0.0436084035 | 0.161429934 | 0.924248311 | False |
| step | ogda | FedDeepGMM-OGDA | 20000 | 0.0149287044 | 0.0912774223 | 0.783083513 | False |

Old array-level GDA vs OGDA MSE summary:

| dataset | old_ogda_curve_mse | old_gda_curve_mse | ogda_minus_gda |
| --- | --- | --- | --- |
| abs | 0.000782758091 | 0.0360315372 | -0.0352487791 |
| linear | 8.35776141e-05 | 0.000194422514 | -0.0001108449 |
| step | 0.0149287044 | 0.0436084035 | -0.028679699 |

## Comparison To Our Current Validated Plots

Our current plots under `experiments/curve_fitting_plots/` are generated from validated result artifacts with `predictions.npz`, `metrics.json`, and validation-selected predictions. They include explicit source labels and keep tuned Sine separate from the original sweep.

Current validated original-sweep deterministic FedGDA-D vs FedOGDA-D summary by dataset/alpha:

| dataset | alpha | pairs | fedogda_wins | mean_fedgda_curve_mse | mean_fedogda_curve_mse | mean_gap_fedogda_minus_fedgda |
| --- | --- | --- | --- | --- | --- | --- |
| abs | 0.1 | 3 | 2 | 0.0185482147 | 0.0176289575 | -0.000919257178 |
| abs | 0.5 | 3 | 2 | 0.0177515092 | 0.016589363 | -0.00116214621 |
| abs | 1 | 3 | 2 | 0.01772472 | 0.0164571807 | -0.00126753924 |
| linear | 0.1 | 3 | 2 | 0.00428732767 | 0.0029284127 | -0.00135891497 |
| linear | 0.5 | 3 | 2 | 0.00423304302 | 0.00284033303 | -0.00139270999 |
| linear | 1 | 3 | 2 | 0.00415789785 | 0.0028160473 | -0.00134185055 |
| sin | 0.1 | 3 | 1 | 0.0861539434 | 0.0862115881 | 5.76446093e-05 |
| sin | 0.5 | 3 | 0 | 0.0861961437 | 0.0862728278 | 7.6684094e-05 |
| sin | 1 | 3 | 1 | 0.0861068629 | 0.0861559175 | 4.90545775e-05 |
| step | 0.1 | 3 | 3 | 0.0297879823 | 0.02918822 | -0.000599762314 |
| step | 0.5 | 3 | 3 | 0.0297254879 | 0.0291784126 | -0.000547075214 |
| step | 1 | 3 | 3 | 0.0296753667 | 0.0291397604 | -0.000535606339 |

Current tuned Sine A2-lite summary:

| dataset | alpha | pairs | fedogda_wins | mean_fedgda_curve_mse | mean_fedogda_curve_mse | mean_gap_fedogda_minus_fedgda |
| --- | --- | --- | --- | --- | --- | --- |
| sin | 1 | 3 | 3 | 0.0861068629 | 0.0800115346 | -0.00609532833 |

## Key Differences

1. The coauthor committed plots are older and definitely not our generated plots. Their commit authors are Geetika and `ggauranshi-03`.
2. The coauthor plots look more like selected/tuned illustrative curve overlays. They do not carry enough metadata to verify fairness against our manifest-level protocol.
3. The old plots do not include Sine in the same `All_3*` family. Our tuned Sine result is therefore not duplicated by those old committed plots.
4. For Absolute/Step/Linear, the tracked old arrays show OGDA much closer than GDA under their own saved curves. Our validated rerun-protocol plots also show FedOGDA-D generally better, but the margin is smaller because the protocol/data/config/checkpoint selection are different and fully audited.
5. The old plot images should not be merged into the current result tables unless the exact configs and selection protocol can be reconstructed or rerun through the current validators.

## Practical Recommendation

Use the old committed plots as historical/coauthor visual evidence only. For final paper/report figures, prefer the current generated plots because they are tied to validated artifacts and metrics:

- `experiments/curve_fitting_plots/png/coauthor_summary/lowdim_deterministic_summary_2x2.png`
- `experiments/curve_fitting_plots/png/main_pairwise_aggregate/abs_alpha1p0_fedgda_d_vs_fedogda_d_mean.png`
- `experiments/curve_fitting_plots/png/main_pairwise_aggregate/step_alpha0p1_fedgda_d_vs_fedogda_d_mean.png`
- `experiments/curve_fitting_plots/png/main_pairwise_aggregate/linear_alpha0p1_fedgda_d_vs_fedogda_d_mean.png`
- `experiments/curve_fitting_plots/png/tuned_sine_a2_lite/sine_a2_lite_all_seeds_mean.png`

If Geetika wants the old curves reflected exactly, the next clean step is to reconstruct the old configs behind `All_3*.png` and rerun/report them under the current manifest + validator scheme.

## Output CSVs

- `experiments/curve_fitting_plots/csv/git_committed_curve_plot_inventory.csv`
- `experiments/curve_fitting_plots/csv/git_committed_npy_curve_metrics.csv`
- `experiments/curve_fitting_plots/csv/current_validated_lowdim_curve_summary_for_git_plot_compare.csv`
