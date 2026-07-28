# lowdim_common_alpha0p5_tuned_sine_reference_style_v2

Paper-reference rendering of the tuned common-alpha low-dimensional curves.
Presentation changed only; validation-selected predictions and metrics are unchanged.

## Protocol

- All federated curves use `alpha=0.5` and are means over seeds `0,1,2`.
- Sine FedDeepGMM-SGDA and FedDeepGMM-OGDA-S use the paired v1 confirmed winners.
- The other curve sources and selection provenance are recorded in the metrics CSV.

## Outputs

- `experiments/curve_fitting_plots/png/coauthor_summary/lowdim_common_alpha0p5_tuned_sine_reference_style_v2.png`
- `experiments/curve_fitting_plots/pdf/coauthor_summary/lowdim_common_alpha0p5_tuned_sine_reference_style_v2.pdf`
- `experiments/curve_fitting_plots/csv/lowdim_common_alpha0p5_tuned_sine_reference_style_v2_curve_metrics.csv`

## Suggested caption

`Validation-selected estimates of the causal response function compared with the true effect in four low-dimensional scenarios. All federated curves use a common Dirichlet concentration $\alpha=0.5$ and show pointwise means over seeds 0--2.`
