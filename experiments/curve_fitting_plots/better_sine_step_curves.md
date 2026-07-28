# Better Sine/Step Curves

Generated from saved `best_validation_prediction` arrays. Test MSE values are post-selection readouts.

## Outputs

- `experiments/curve_fitting_plots/png/better_sine_step/better_sine_step_curves.png`
- `experiments/curve_fitting_plots/pdf/better_sine_step/better_sine_step_curves.pdf`
- `experiments/curve_fitting_plots/csv/better_sine_step_curve_metrics.csv`

## Completion Status

- Sine A2-lite: seed 0 plus continuation seeds 1 and 2 are available; continuation manifest is 2/2 passed.
- Step Geetika-recipe reproduction: 4/4 rows passed.

## Sine Mean Readout

- FedDeepGMM-GDA: mean test@best `0.086106863`, curve MSE `0.084916527`.
- FedDeepGMM-OGDA-D tuned: mean test@best `0.080011535`, curve MSE `0.079916005`.

## Step Seed-0 Ranking By Validation

- FedDeepGMM-SGDA: best val `0.006003955`, test@best `0.006126142`, best round `266`.
- FedDeepGMM-GDA partial-FB: best val `0.006352679`, test@best `0.006585997`, best round `423`.
- FedDeepGMM-OGDA-S: best val `0.013276625`, test@best `0.013467858`, best round `1384`.
- FedDeepGMM-OGDA-D partial-FB: best val `0.013380280`, test@best `0.013571059`, best round `1311`.

Caveat: the Step partial-FB rows use partial clients, so they are labeled separately from standard full-participation deterministic rows.
