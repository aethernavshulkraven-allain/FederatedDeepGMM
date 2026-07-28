# lowdim_common_alpha0p5_tuned_sine_paired_v1

Common-alpha paper candidate generated from validation-selected artifacts.
All learned curves use saved `best_validation_prediction` arrays.

## Outputs

- `experiments/curve_fitting_plots/png/coauthor_summary/lowdim_common_alpha0p5_tuned_sine_paired_v1.png`
- `experiments/curve_fitting_plots/pdf/coauthor_summary/lowdim_common_alpha0p5_tuned_sine_paired_v1.pdf`
- `experiments/curve_fitting_plots/csv/lowdim_common_alpha0p5_tuned_sine_paired_v1_curve_metrics.csv`

## Protocol

- Common federated Dirichlet concentration: `alpha=0.5`.
- Every plotted method curve is the pointwise mean over seeds `0,1,2`.
- FedDeepGMM-SGDA uses the completed alpha=0.5 base sweep for Absolute, Step, and Linear; Sine uses the paired confirmation.
- FedDeepGMM-OGDA-S is validation-tuned for Absolute and Linear (alpha=0.5 pilot) and Step (v5 final).
- Sine FedDeepGMM-SGDA and FedDeepGMM-OGDA-S use matched validation-driven screening/refinement budgets and independent 1000-round confirmations over seeds 0,1,2.
- Centralized methods do not use the federated partition alpha.

## Publication status

**Paper candidate with scope caveats.** The common-alpha presentation and Sine tuning are internally consistent. Comparative claims across all four scenarios should still disclose that the non-Sine federated methods do not all have the same matched tuning budget, and the synthetic DGP remains reproducible but not independently certified as paper-aligned.

## Panel metrics

### (a) Absolute ($\alpha=0.5$)
- FedDeepGMM-SGDA: validation MSE `0.002649518`, test@best `0.002709909`, curve MAE `0.028210090`; mean base_sweep alpha=0.5, seeds 0-2.
- FedDeepGMM-OGDA-S: validation MSE `0.005930081`, test@best `0.006058285`, curve MAE `0.037328539`; fedogda_s pilot alpha=0.5, validation-selected config, mean seeds 0-2.
- DeepGMM-GDA: validation MSE `0.054519128`, test@best `0.056168981`, curve MAE `0.171943268`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-SGDA: validation MSE `0.054661552`, test@best `0.056258241`, curve MAE `0.169322652`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-OAdam: validation MSE `0.001738339`, test@best `0.001779515`, curve MAE `0.024154826`; centralized C3 OAdam, mean seeds 0-2.

### (b) Step ($\alpha=0.5$)
- FedDeepGMM-SGDA: validation MSE `0.025922973`, test@best `0.026267388`, curve MAE `0.135090776`; mean base_sweep alpha=0.5, seeds 0-2.
- FedDeepGMM-OGDA-S: validation MSE `0.004777577`, test@best `0.004943105`, curve MAE `0.035419593`; fedogda_s_step_fast_v5 final config, mean seeds 0-2.
- DeepGMM-GDA: validation MSE `0.031832872`, test@best `0.032191864`, curve MAE `0.143521621`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-SGDA: validation MSE `0.031582548`, test@best `0.031968220`, curve MAE `0.142713636`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-OAdam: validation MSE `0.016972566`, test@best `0.017214014`, curve MAE `0.106203666`; centralized C3 OAdam, mean seeds 0-2.

### (c) Linear ($\alpha=0.5$)
- FedDeepGMM-SGDA: validation MSE `0.000352984`, test@best `0.000356482`, curve MAE `0.014313348`; mean base_sweep alpha=0.5, seeds 0-2.
- FedDeepGMM-OGDA-S: validation MSE `0.000954419`, test@best `0.000967615`, curve MAE `0.018738286`; fedogda_s pilot alpha=0.5, validation-selected config, mean seeds 0-2.
- DeepGMM-GDA: validation MSE `0.009346492`, test@best `0.009543763`, curve MAE `0.057260864`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-SGDA: validation MSE `0.007804446`, test@best `0.007997298`, curve MAE `0.052694131`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-OAdam: validation MSE `0.000743962`, test@best `0.000759664`, curve MAE `0.018789999`; centralized C3 OAdam, mean seeds 0-2.

### (d) Sine ($\alpha=0.5$)
- FedDeepGMM-SGDA: validation MSE `0.011788976`, test@best `0.011822252`, curve MAE `0.085340308`; Sine alpha=0.5 paired v1, validation-selected configuration, mean confirmed seeds 0-2.
- FedDeepGMM-OGDA-S: validation MSE `0.013372759`, test@best `0.013451321`, curve MAE `0.090198619`; Sine alpha=0.5 paired v1, validation-selected configuration, mean confirmed seeds 0-2.
- DeepGMM-GDA: validation MSE `0.085547715`, test@best `0.087117365`, curve MAE `0.246178518`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-SGDA: validation MSE `0.086021981`, test@best `0.087679950`, curve MAE `0.247029818`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-OAdam: validation MSE `0.036795740`, test@best `0.037676119`, curve MAE `0.150366469`; centralized C3 OAdam, mean seeds 0-2.

## Draft caption

`Validation-checkpoint estimates of the structural function $g_0$ in four low-dimensional scenarios. Federated experiments use a common Dirichlet concentration $\alpha=0.5$; curves are pointwise means over seeds 0--2.`
