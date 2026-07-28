# lowdim_common_alpha0p5_lower_cost_preview

Lower-cost common-alpha preview generated from existing artifacts only.
All learned curves use saved `best_validation_prediction` arrays.

## Outputs

- `experiments/curve_fitting_plots/png/coauthor_summary/lowdim_common_alpha0p5_lower_cost_preview.png`
- `experiments/curve_fitting_plots/pdf/coauthor_summary/lowdim_common_alpha0p5_lower_cost_preview.pdf`
- `experiments/curve_fitting_plots/csv/lowdim_common_alpha0p5_lower_cost_preview_curve_metrics.csv`

## Protocol

- Common federated Dirichlet concentration: `alpha=0.5`.
- Every plotted method curve is the pointwise mean over seeds `0,1,2`.
- FedDeepGMM-SGDA uses the completed alpha=0.5 base sweep in every panel.
- FedDeepGMM-OGDA-S is validation-tuned for Absolute and Linear (alpha=0.5 pilot) and Step (v5 final).
- Sine FedDeepGMM-OGDA-S uses the existing alpha=0.5 base preset; no tuned alpha=0.5 Sine confirmation exists.
- Centralized methods do not use the federated partition alpha.

## Publication status

**Preview, not final.** The common-alpha presentation is internally consistent, but Sine still requires validation-driven alpha=0.5 tuning/confirmation and competing-method tuning budgets should be matched before using the figure for a comparative paper claim.

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
- FedDeepGMM-SGDA: validation MSE `0.076304426`, test@best `0.078034135`, curve MAE `0.234563701`; mean base_sweep alpha=0.5, seeds 0-2.
- FedDeepGMM-OGDA-S: validation MSE `0.083601371`, test@best `0.086140245`, curve MAE `0.243505848`; base_sweep alpha=0.5 preset, mean seeds 0-2; provisional (not tuned).
- DeepGMM-GDA: validation MSE `0.085547715`, test@best `0.087117365`, curve MAE `0.246178518`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-SGDA: validation MSE `0.086021981`, test@best `0.087679950`, curve MAE `0.247029818`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-OAdam: validation MSE `0.036795740`, test@best `0.037676119`, curve MAE `0.150366469`; centralized C3 OAdam, mean seeds 0-2.

## Draft caption

`Validation-checkpoint estimates of the structural function $g_0$ in four low-dimensional scenarios. Federated experiments use a common Dirichlet concentration $\alpha=0.5$; curves are pointwise means over seeds 0--2.`
