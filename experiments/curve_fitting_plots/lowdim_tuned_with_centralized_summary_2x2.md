# lowdim_tuned_with_centralized_summary_2x2

Unified 2x2 low-dimensional curve summary generated from saved `best_validation_prediction` arrays.
Federated tuned/final selections are validation-selected; test MSE values below are post-selection readouts.

## Outputs

- `experiments/curve_fitting_plots/png/coauthor_summary/lowdim_tuned_with_centralized_summary_2x2.png`
- `experiments/curve_fitting_plots/pdf/coauthor_summary/lowdim_tuned_with_centralized_summary_2x2.pdf`
- `experiments/curve_fitting_plots/csv/lowdim_tuned_with_centralized_summary_2x2_curve_metrics.csv`

## Panel Sources

### Absolute alpha=1.0
- FedGDA-D: test@best `0.017724720`, curve MAE `0.082038460`; mean base_sweep alpha=1.0, seeds 0-2.
- FedOGDA-D: test@best `0.016457181`, curve MAE `0.078189461`; mean base_sweep alpha=1.0, seeds 0-2.
- Central GDA: test@best `0.056168981`, curve MAE `0.171943268`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- Central SGDA: test@best `0.056258241`, curve MAE `0.169322652`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- Central OAdam: test@best `0.001779515`, curve MAE `0.024154826`; centralized C3 OAdam, mean seeds 0-2.

### Step alpha=0.5 tuned v5
- FedGDA-S ref: test@best `0.006126142`, curve MAE `0.059170107`; step_geetika_repro_v1 FedGDA-S reference, seed 0.
- Tuned FedOGDA-S v5: test@best `0.004943105`, curve MAE `0.035419593`; fedogda_s_step_fast_v5 final config, mean seeds 0-2.
- Central GDA: test@best `0.032191864`, curve MAE `0.143521621`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- Central SGDA: test@best `0.031968220`, curve MAE `0.142713636`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- Central OAdam: test@best `0.017214014`, curve MAE `0.106203666`; centralized C3 OAdam, mean seeds 0-2.

### Linear alpha=0.1
- FedGDA-D: test@best `0.004287328`, curve MAE `0.037147017`; mean base_sweep alpha=0.1, seeds 0-2.
- FedOGDA-D: test@best `0.002928413`, curve MAE `0.031374618`; mean base_sweep alpha=0.1, seeds 0-2.
- Central GDA: test@best `0.009543763`, curve MAE `0.057260864`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- Central SGDA: test@best `0.007997298`, curve MAE `0.052694131`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- Central OAdam: test@best `0.000759664`, curve MAE `0.018789999`; centralized C3 OAdam, mean seeds 0-2.

### Sine alpha=1.0 tuned v4
- FedGDA-S: test@best `0.078642541`, curve MAE `0.230445683`; mean base_sweep alpha=1.0, seeds 0-2.
- Tuned FedOGDA-S v4: test@best `0.013406540`, curve MAE `0.094333582`; fedogda_s_sine_fast_v4 final config, mean seeds 0-2.
- Central GDA: test@best `0.087117365`, curve MAE `0.246178518`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- Central SGDA: test@best `0.087679950`, curve MAE `0.247029818`; centralized C5 validation-tuned GDA/SGDA, mean seeds 0-2.
- Central OAdam: test@best `0.037676119`, curve MAE `0.150366469`; centralized C3 OAdam, mean seeds 0-2.
