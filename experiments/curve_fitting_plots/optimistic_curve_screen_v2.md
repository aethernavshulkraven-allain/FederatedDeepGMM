# optimistic_curve_screen_v2

Selection rule: finite/non-diverged candidates only; lowest `best_validation_mse`; tie-break lower `last_50_val_mse_std`; tie-break lower `final_vs_best_validation_gap`.

Test MSE and curve diagnostics below are post-selection readouts.

## Outputs

- `experiments/curve_fitting_plots/png/optimistic_curve_screen_v2/optimistic_curve_screen_v2_curves.png`
- `experiments/curve_fitting_plots/pdf/optimistic_curve_screen_v2/optimistic_curve_screen_v2_curves.pdf`
- `experiments/curve_fitting_plots/csv/optimistic_curve_screen_v2_candidates.csv`
- `experiments/curve_fitting_plots/csv/optimistic_curve_screen_v2_selected.csv`

## Completion

- Sine screen rows completed: 12/12.
- Step screen rows completed: 5/6, plus the existing Step OGDA reference candidate.
- Missing/incomplete manifest rows: 1.

## Validation-Selected Readout

- sin: lr `0.005`, wd `0`, lambda `0.03`, R `3`, best val `0.041984859`, test@best `0.042874493`, curve corr `0.802101`.
- step: lr `0.005`, wd `0`, lambda `0.1`, R `7`, best val `0.010456711`, test@best `0.010636226`, curve corr `0.947906`.

## Notes

- The Sine panel includes the old tuned deterministic OGDA line for visual context; it is not part of this screen's stochastic OGDA-S selection pool.
- The Step selection pool includes the existing reproduced FedOGDA-S Geetika-reference row (`lr=0.01`, `wd=0.02`) because it was intentionally not relaunched.
