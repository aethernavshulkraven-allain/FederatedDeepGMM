# lowdim_tuned_with_centralized_summary_2x2_paper_v1

Unified 2x2 low-dimensional curve summary generated from saved `best_validation_prediction` arrays.
Federated tuned/final selections are validation-selected; test MSE values below are post-selection readouts.
Paper v1 changes presentation only; underlying run directories, predictions, and scientific metrics are unchanged.

## Outputs

- `experiments/curve_fitting_plots/png/coauthor_summary/lowdim_tuned_with_centralized_summary_2x2_paper_v1.png`
- `experiments/curve_fitting_plots/pdf/coauthor_summary/lowdim_tuned_with_centralized_summary_2x2_paper_v1.pdf`
- `experiments/curve_fitting_plots/csv/lowdim_tuned_with_centralized_summary_2x2_paper_v1_curve_metrics.csv`

## Panel Sources

### (a) Absolute ($\alpha=1.0$)
- FedDeepGMM-GDA: test@best `0.017724720`, curve MAE `0.082038460`; mean base_sweep alpha=1.0, seeds 0-2.
- FedDeepGMM-OGDA-D: test@best `0.016457181`, curve MAE `0.078189461`; mean base_sweep alpha=1.0, seeds 0-2.
- DeepGMM-GDA: test@best `0.056168981`, curve MAE `0.171943268`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-SGDA: test@best `0.056258241`, curve MAE `0.169322652`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-OAdam: test@best `0.001779515`, curve MAE `0.024154826`; centralized C3 OAdam, mean seeds 0-2.

### (b) Step ($\alpha=0.5$)
- FedDeepGMM-SGDA: test@best `0.006126142`, curve MAE `0.059170107`; step_geetika_repro_v1 FedGDA-S reference, seed 0.
- FedDeepGMM-OGDA-S: test@best `0.004943105`, curve MAE `0.035419593`; fedogda_s_step_fast_v5 final config, mean seeds 0-2.
- DeepGMM-GDA: test@best `0.032191864`, curve MAE `0.143521621`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-SGDA: test@best `0.031968220`, curve MAE `0.142713636`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-OAdam: test@best `0.017214014`, curve MAE `0.106203666`; centralized C3 OAdam, mean seeds 0-2.

### (c) Linear ($\alpha=0.1$)
- FedDeepGMM-GDA: test@best `0.004287328`, curve MAE `0.037147017`; mean base_sweep alpha=0.1, seeds 0-2.
- FedDeepGMM-OGDA-D: test@best `0.002928413`, curve MAE `0.031374618`; mean base_sweep alpha=0.1, seeds 0-2.
- DeepGMM-GDA: test@best `0.009543763`, curve MAE `0.057260864`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-SGDA: test@best `0.007997298`, curve MAE `0.052694131`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-OAdam: test@best `0.000759664`, curve MAE `0.018789999`; centralized C3 OAdam, mean seeds 0-2.

### (d) Sine ($\alpha=1.0$)
- FedDeepGMM-SGDA: test@best `0.078642541`, curve MAE `0.230445683`; mean base_sweep alpha=1.0, seeds 0-2.
- FedDeepGMM-OGDA-S: test@best `0.013406540`, curve MAE `0.094333582`; fedogda_s_sine_fast_v4 final config, mean seeds 0-2.
- DeepGMM-GDA: test@best `0.087117365`, curve MAE `0.246178518`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-SGDA: test@best `0.087679950`, curve MAE `0.247029818`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- DeepGMM-OAdam: test@best `0.037676119`, curve MAE `0.150366469`; centralized C3 OAdam, mean seeds 0-2.
