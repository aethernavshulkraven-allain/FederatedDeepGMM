# Optimistic Curve Screen v1

Selection rule: finite/non-diverged candidates only; lowest `best_validation_mse`; tie-break lower `last_50_val_mse_std`; tie-break lower `final_vs_best_validation_gap`.

Test MSE and curve diagnostics below are post-selection readouts.

## Outputs

- `experiments/curve_fitting_plots/png/optimistic_curve_screen_v1/optimistic_curve_screen_v1_curves.png`
- `experiments/curve_fitting_plots/pdf/optimistic_curve_screen_v1/optimistic_curve_screen_v1_curves.pdf`
- `experiments/curve_fitting_plots/csv/optimistic_curve_screen_v1_candidates.csv`
- `experiments/curve_fitting_plots/csv/optimistic_curve_screen_v1_selected.csv`

## Completion

- Sine screen rows completed: 3/12.
- Step screen rows completed: 0/8, plus the existing Step OGDA reference candidate.
- Missing/incomplete manifest rows: 17.

## Validation-Selected Readout

- sin: lr `0.002`, wd `0.001`, R `3`, best val `0.070932657`, test@best `0.072987757`, curve corr `0.572678`.
- step: lr `0.01`, wd `0.02`, R `7`, best val `0.013276625`, test@best `0.013467858`, curve corr `0.930780`.

## Notes

- The Sine panel includes the old tuned deterministic OGDA line for visual context; it is not part of this screen's stochastic OGDA-S selection pool.
- The Step selection pool includes the existing reproduced FedOGDA-S Geetika-reference row (`lr=0.01`, `wd=0.02`) because it was intentionally not relaunched.
