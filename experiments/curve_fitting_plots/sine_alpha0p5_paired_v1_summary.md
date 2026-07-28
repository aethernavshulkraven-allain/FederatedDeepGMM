# sine_alpha0p5_paired_v1

All configuration and checkpoint selection is validation-only.
Test and curve metrics are read only after the confirmed validation winner is fixed.

## Completion

- Stage A: `32` valid; `0` missing/invalid.
- Stage B: `12` valid; `0` missing/invalid.
- Stage C: `12` valid; `0` missing/invalid.

## Selected configurations

### FedGDA-S

- lr `0.01`, cm `10`, lambda `0.01`, server lr `1.5`.
- Mean validation MSE `0.011788976` (reduction vs existing preset `84.6%`).
- Post-selection mean test@best `0.011822252` +/- `0.003742712` (reduction vs existing preset `84.8%`).
- Mean curve MAE `0.090969728`, correlation `0.947904628`, amplitude ratio `1.190009897`.

### FedOGDA-S

- lr `0.01`, cm `5`, lambda `0.005`, server lr `2`.
- Mean validation MSE `0.013372759` (reduction vs existing preset `84.0%`).
- Post-selection mean test@best `0.013451321` +/- `0.001953458` (reduction vs existing preset `84.4%`).
- Mean curve MAE `0.095780511`, correlation `0.940528567`, amplitude ratio `1.158083829`.

## Outputs

- `experiments/curve_fitting_plots/png/sine_alpha0p5_paired_v1/sine_alpha0p5_paired_v1_selected_curves.png`
- `experiments/curve_fitting_plots/pdf/sine_alpha0p5_paired_v1/sine_alpha0p5_paired_v1_selected_curves.pdf`
- `experiments/curve_fitting_plots/png/sine_alpha0p5_paired_v1/sine_alpha0p5_paired_v1_selected_validation_dynamics.png`
- `experiments/curve_fitting_plots/pdf/sine_alpha0p5_paired_v1/sine_alpha0p5_paired_v1_selected_validation_dynamics.pdf`
- `experiments/curve_fitting_plots/csv/sine_alpha0p5_paired_v1_all_runs.csv`
- `experiments/curve_fitting_plots/csv/sine_alpha0p5_paired_v1_confirmation_aggregates.csv`
- `experiments/curve_fitting_plots/csv/sine_alpha0p5_paired_v1_final_selected.csv`
- `experiments/curve_fitting_plots/csv/sine_alpha0p5_paired_v1_selected_seed_runs.csv`
