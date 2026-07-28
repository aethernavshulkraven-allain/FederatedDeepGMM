# Low-Dimensional FedOGDA-D vs FedGDA-D Summary

Primary metric: `test_mse_at_best_validation`, reported after validation-only checkpoint selection. Lower is better. The tuned Sine rows are labeled separately and are not merged into the original sweep.

## Per-Function Summary

| source | function | pairs | FedOGDA-D wins | FedGDA-D wins | mean FedGDA-D Test MSE | mean FedOGDA-D Test MSE | mean relative gap | readout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_sweep | Absolute | 9 | 6 | 3 | 0.018008148 | 0.016891834 | -5.324% | positive in original sweep |
| baseline_sweep | Step | 9 | 9 | 0 | 0.029729612 | 0.029168798 | -1.937% | positive in original sweep |
| baseline_sweep | Linear | 9 | 6 | 3 | 0.0042260895 | 0.0028615977 | -23.653% | positive in original sweep |
| baseline_sweep | Sine | 9 | 2 | 7 | 0.086152317 | 0.086213444 | 0.073% | not positive enough; tune if this function matters |
| tuned_sine_a2_lite | Sine | 3 | 3 | 0 | 0.086106863 | 0.080011535 | -7.092% | positive locked deterministic Sine result |

## Main Takeaways

- Original Absolute sweep: FedOGDA-D wins 6/9 pairs; mean Test MSE 0.018008148 -> 0.016891834 (-5.324% FedOGDA-minus-FedGDA relative gap).
- Original Step sweep: FedOGDA-D wins 9/9 pairs; mean Test MSE 0.029729612 -> 0.029168798 (-1.937% FedOGDA-minus-FedGDA relative gap).
- Original Linear sweep: FedOGDA-D wins 6/9 pairs; mean Test MSE 0.0042260895 -> 0.0028615977 (-23.653% FedOGDA-minus-FedGDA relative gap).
- Original Sine sweep: FedOGDA-D wins 2/9 pairs; mean Test MSE 0.086152317 -> 0.086213444 (0.073% FedOGDA-minus-FedGDA relative gap).
- Tuned Sine A2-lite: FedOGDA-D wins 3/3 seeds; mean Test MSE 0.086106863 -> 0.080011535 (-7.092% relative gap).

## Files

- Pair-level CSV: `experiments/lowdim_fedogda_d_vs_fedgda_d_summary.csv`
- Sine tuned report: `experiments/sine_fedogda_tuning/a2_lite_final_report.md`
