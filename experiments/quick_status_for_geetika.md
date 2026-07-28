# Quick Status For Geetika

Scope: existing local artifacts only. No training was launched, no jobs were queued, and no training code was modified for this wrap-up.

## Executive Summary

- Low-dimensional federated base sweep: `144/144` validated runs complete for Absolute, Step, Linear, and Sine across FedGDA-D, FedOGDA-D, FedGDA-S, and FedOGDA-S.
- The old 144-run base sweep has `metrics.json`, `mse_by_round.csv`, and `predictions.npz` for every row, but it does not have `test_mse_by_round.csv`; per-round Test MSE starts with the newer Sine tuning runs.
- FedOGDA-S tuning pilot: `144/144` validated runs complete for Absolute/Step/Linear at alpha 0.5; this is a stochastic tuning extension, not part of the base 144-run sweep.
- Deterministic Sine tuning: A1-mini `12/12` validated runs complete, and the locked A2-lite confirmation has `3/3` validated seed metrics. The locked A2-lite Sine recipe is positive: FedOGDA-D beats paired FedGDA-D on 3/3 seeds at alpha 1.0.
- Centralized GDA/SGDA/OAdam baselines: not completed/validated. The rerun manifest marks these rows as blocked pending true centralized runner verification.
- High-dimensional FEMNIST/CIFAR-10 x/z/xz: manifest rows exist for the real-image abs protocol, but no validated result artifacts were found; mark pending/not completed.

## Low-Dimensional Federated Completion Matrix

| function | method | expected | completed | seeds | alphas | status | notes |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| Absolute | `fedgda_d` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |
| Absolute | `fedogda_d` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |
| Absolute | `fedgda_s` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |
| Absolute | `fedogda_s` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |
| Step | `fedgda_d` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |
| Step | `fedogda_d` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |
| Step | `fedgda_s` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |
| Step | `fedogda_s` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |
| Linear | `fedgda_d` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |
| Linear | `fedogda_d` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |
| Linear | `fedgda_s` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |
| Linear | `fedogda_s` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |
| Sine | `fedgda_d` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |
| Sine | `fedogda_d` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |
| Sine | `fedgda_s` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |
| Sine | `fedogda_s` | 9 | 9 | 0|1|2 | 0.1|0.5|1.0 | validated_complete | legacy sweep lacks per-round Test MSE |

## Centralized Status

| algorithm | expected rows | completed validated | status | notes |
| --- | ---: | ---: | --- | --- |
| GDA | 12 | 0 | blocked_pending_runner_verification | manifest rows are blocked_pending_true_centralized_runner_verification; no validated result artifacts found |
| SGDA/SGD | 12 | 0 | blocked_pending_runner_verification | manifest rows are blocked_pending_true_centralized_runner_verification; no validated result artifacts found |
| OAdam-D | 12 | 0 | blocked_pending_runner_verification | manifest rows are blocked_pending_true_centralized_runner_verification; no validated result artifacts found |
| OAdam-S | 12 | 0 | blocked_pending_runner_verification | manifest rows are blocked_pending_true_centralized_runner_verification; no validated result artifacts found |

## High-Dimensional FEMNIST/CIFAR-10 Status

The real-image manifest exists at `experiments/rerun_protocol_v1_real_images_abs_alpha0p5/manifest.csv`, but no validated outputs were found under `results/rerun_protocol_v1_real_images_abs_alpha0p5`.

| scenario | methods covered in manifest | expected rows | completed validated | status |
| --- | --- | ---: | ---: | --- |
| femnist_x | `fedgda_d`, `fedogda_d`, `fedgda_s`, `fedogda_s` | 12 | 0 | pending/not completed |
| femnist_z | `fedgda_d`, `fedogda_d`, `fedgda_s`, `fedogda_s` | 12 | 0 | pending/not completed |
| femnist_xz | `fedgda_d`, `fedogda_d`, `fedgda_s`, `fedogda_s` | 12 | 0 | pending/not completed |
| cifar10_x | `fedgda_d`, `fedogda_d`, `fedgda_s`, `fedogda_s` | 12 | 0 | pending/not completed |
| cifar10_z | `fedgda_d`, `fedogda_d`, `fedgda_s`, `fedogda_s` | 12 | 0 | pending/not completed |
| cifar10_xz | `fedgda_d`, `fedogda_d`, `fedgda_s`, `fedogda_s` | 12 | 0 | pending/not completed |

## FedOGDA-D vs FedGDA-D: Low-Dimensional Deterministic Evidence

Original sweep and tuned Sine are separated below. Metric is `test_mse_at_best_validation`; lower is better.

| source | function | pairs | FedOGDA-D wins | mean FedGDA-D | mean FedOGDA-D | mean relative gap | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_sweep | Absolute | 9 | 6 | 0.018008148 | 0.016891834 | -5.324% | already positive in original sweep |
| baseline_sweep | Step | 9 | 9 | 0.029729612 | 0.029168798 | -1.937% | already positive in original sweep |
| baseline_sweep | Linear | 9 | 6 | 0.0042260895 | 0.0028615977 | -23.653% | already positive in original sweep |
| baseline_sweep | Sine | 9 | 2 | 0.086152317 | 0.086213444 | 0.073% | not positive; needs tuning if targeted |
| tuned_sine_a2_lite | Sine | 3 | 3 | 0.086106863 | 0.080011535 | -7.092% | positive locked result |

## Coauthor-Ready Message

```text
Hi Geetika, quick status update from the existing artifacts: the low-dimensional federated runs for Sine, Linear, Absolute, and Step are completed/validated for FedGDA-D, FedGDA-S, FedOGDA-D, and FedOGDA-S across alpha 0.1/0.5/1.0 and seeds 0/1/2. The older sweep has scalar Test MSE and predictions, but not per-round Test MSE.

The deterministic Sine tuning result is positive. With the validation-locked recipe at sin, alpha=1.0, T=500, R=3, g_lr=0.002, f_lr=0.03, critic_multiplier=15, server_lr=1.5, FedOGDA-D beats fully paired FedGDA-D on 3/3 seeds. Mean validation-selected Test MSE improves from 0.08611 to 0.08001, about a 7.1% relative improvement, and the saved curve-fit metrics also improve.

Centralized GDA/SGDA/OAdam baselines are not yet completed/verified; the current manifest marks them blocked pending true centralized runner verification. High-dimensional FEMNIST/CIFAR-10 x/z/xz scenarios are also not yet completed/verified; we have a manifest for the real-image abs protocol, but no validated result artifacts.

I also checked whether the existing deterministic low-dimensional sweep already shows FedOGDA-D wins beyond tuned Sine. In the original sweep, Absolute, Step, and Linear already have lower mean validation-selected Test MSE for FedOGDA-D than FedGDA-D. Linear is the strongest margin; Step is consistent but modest; Absolute is positive on average but mixed by seed. Original Sine was not positive, but the tuned Sine recipe now is. The next best action is to tune whichever additional low-dimensional deterministic case we want to strengthen, likely Absolute for mixed seeds or Step for another non-smooth example, while keeping selection validation-only.
```

## Output Files

- Completion matrix CSV: `experiments/quick_status_matrix.csv`
- FedOGDA-D pair CSV: `experiments/lowdim_fedogda_d_vs_fedgda_d_summary.csv`
- FedOGDA-D pair summary: `experiments/lowdim_fedogda_d_vs_fedgda_d_summary.md`
