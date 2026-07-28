# Consolidated Results For Geetika

Generated from existing local artifacts in `/home/arnav22103/FederatedDeepGMM` on 2026-07-11 10:59:02. No training was launched. Lower MSE is better. All Test MSE values below are post-selection readouts after validation-selected checkpoints/configs; Test MSE was not used for hyperparameter selection.

## 1. Executive Summary

- Low-dimensional federated base sweep: `144/144` runs validated across `abs`, `step`, `linear`, `sin` and FedGDA/FedOGDA deterministic/stochastic variants; completed `144/144`, `diverged=0`.
- Base legacy runs did not log per-round Test MSE, so old last-50 Test MSE cannot be reconstructed; scalar `test_mse_at_best_validation` and `final_test_mse` are available.
- Original deterministic Absolute: FedOGDA-D wins `6/9` pairs; mean Test MSE `0.0180081` for FedGDA-D vs `0.0168918` for FedOGDA-D (6.20% improvement).
- Original deterministic Linear: FedOGDA-D wins `6/9` pairs; mean Test MSE `0.00422609` for FedGDA-D vs `0.0028616` for FedOGDA-D (32.29% improvement).
- Original deterministic Sine: FedOGDA-D wins `2/9` pairs; mean Test MSE `0.0861523` for FedGDA-D vs `0.0862134` for FedOGDA-D (-0.07% improvement).
- Original deterministic Step: FedOGDA-D wins `9/9` pairs; mean Test MSE `0.0297296` for FedGDA-D vs `0.0291688` for FedOGDA-D (1.89% improvement).
- Tuned deterministic Sine A2-lite: FedOGDA-D wins `3/3` paired seeds at alpha=1.0; mean Test MSE `0.0861069` -> `0.0800115` (7.08% improvement).
- FedOGDA-S tuning pilot: 144/144 runs completed for abs/linear/step at alpha=0.5; tuned FedOGDA-S improved over current FedOGDA-S and often reduced validation oscillation, but did not beat FedGDA-S on Test MSE.
- Centralized low-dimensional baselines: true centralized runner is implemented; C3 full OAdam/GDA/SGDA completed 36/36; C5 tuned GDA/SGDA completed 24/24. OAdam is strongest overall among centralized methods in current results.
- High-dimensional FEMNIST/CIFAR x/z/xz: federated manifest/preflight/smokes exist, but validated full result artifacts are `0/72`; high-dimensional centralized artifacts/manifests found: `0` reportable runs. Status is pending/blocked, not complete.
- Recommended next action: finalize one paper-facing low-dimensional table using original FedOGDA-D wins plus tuned Sine, then decide whether to run high-dimensional jobs once GPU permissions/queue are clear; do not claim FedOGDA-S lower Test MSE yet.

## 2. Experiment Completion Matrix

Supporting CSV: `experiments/consolidated_results/completion_matrix.csv`.

### 2.1 Low-Dimensional Federated Base Sweep

| dataset_or_scope | method | expected_runs | completed_runs | validated_runs | seeds_present | alphas_present | metrics_json_present_count | mse_by_round_csv_present_count | test_mse_by_round_csv_present_count | predictions_npz_present_count | best_validation_pt_present_count | final_pt_present_count | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abs | fedgda_d | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |
| abs | fedogda_d | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |
| abs | fedgda_s | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |
| abs | fedogda_s | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |
| step | fedgda_d | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |
| step | fedogda_d | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |
| step | fedgda_s | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |
| step | fedogda_s | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |
| linear | fedgda_d | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |
| linear | fedogda_d | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |
| linear | fedgda_s | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |
| linear | fedogda_s | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |
| sin | fedgda_d | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |
| sin | fedogda_d | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |
| sin | fedgda_s | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |
| sin | fedogda_s | 9 | 9 | 9 | 0 1 2 | 0.1 0.5 1.0 | 9 | 9 | 0 | 9 | 9 | 9 | legacy base sweep; no per-round Test MSE logging |

### 2.2 Tuned Sine FedOGDA-D

| dataset_or_scope | method | expected_runs | completed_runs | validated_runs | metrics_json_present_count | mse_by_round_csv_present_count | test_mse_by_round_csv_present_count | predictions_npz_present_count | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sin alpha=1.0 | fedogda_d | 3 | 3 | 3 | 3 | 3 | 3 | 3 | locked by validation-only A2-lite; paired FedGDA-D baseline exists; do not merge into original sweep |

A1-mini completed and selected the A2-lite continuation using validation metrics only. A2-lite locked recipe completed seeds `0,1,2`, selection audit reports validation-only selection, and per-round Test MSE exists for the tuned FedOGDA-D runs. The paired FedGDA-D baseline comes from the original sweep and is legacy, so its per-round Test MSE is unavailable.

### 2.3 FedOGDA-S Tuning Pilot

| dataset_or_scope | method | expected_runs | completed_runs | validated_runs | seeds_present | alphas_present | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| abs linear step | fedogda_s | 144 | 144 | 144 | 0 1 2 | 0.5 | separate FedOGDA-S hyperparameter pilot; validation-only selection; improves current FedOGDA-S but does not beat FedGDA-S Test MSE |

| dataset | alpha | selected_critic_multiplier | selected_weight_decay | tuned_mean_best_validation_mse | tuned_mean_test_mse_at_best_validation | fedgda_mean_test_mse_at_best_validation | tuned_reduces_oscillation_vs_fedgda | tuned_better_test_than_fedgda | no_tuned_divergence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abs | 0.5 | 20 | 0.001 | 0.00593008 | 0.00605829 | 0.00270991 | 1 | 0 | 1 |
| linear | 0.5 | 20 | 0.001 | 0.000954419 | 0.000967615 | 0.000356482 | 1 | 0 | 1 |
| step | 0.5 | 20 | 0.001 | 0.0281032 | 0.0286008 | 0.0262674 | 1 | 0 | 1 |

### 2.4 Centralized Baselines

| experiment_family | dataset_or_scope | method | expected_runs | completed_runs | validated_runs | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| centralized_lowdim_c3_full | abs step linear sin | gda sgda oadam | 36 | 36 | 36 | complete | true centralized C3 completed; OAdam strong; C3 GDA/SGDA under-tuned but valid |
| centralized_lowdim_c5_tuned_gda_sgda | abs step linear sin | gda sgda | 24 | 24 | 24 | complete | C5a/C5b selected g_lr=0.005 f_lr=0.03 by validation only; improved all GDA/SGDA datasets vs C3 |

Tiny smoke artifacts, where present, are implementation checks only and are not used as scientific results.

### 2.5 High-Dimensional FEMNIST/CIFAR

- Federated real-image manifest exists, but no validated full result artifacts were found for FEMNIST/CIFAR x/z/xz. The status report says full tuning is blocked by host GPU permissions; only certification/preflight/smoke evidence exists.
- High-dimensional centralized rows have no manifest or validated artifacts in this repo snapshot.

| scenario | method_label | training_scope | manifest_exists | expected_runs | completed_full_result_runs | validated_full_result_runs | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cifar10_x | FedGDA-D | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| cifar10_x | FedGDA-S | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| cifar10_x | FedOGDA-D | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| cifar10_x | FedOGDA-S | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| cifar10_xz | FedGDA-D | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| cifar10_xz | FedGDA-S | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| cifar10_xz | FedOGDA-D | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| cifar10_xz | FedOGDA-S | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| cifar10_z | FedGDA-D | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| cifar10_z | FedGDA-S | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| cifar10_z | FedOGDA-D | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| cifar10_z | FedOGDA-S | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| femnist_x | FedGDA-D | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| femnist_x | FedGDA-S | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| femnist_x | FedOGDA-D | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| femnist_x | FedOGDA-S | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| femnist_xz | FedGDA-D | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| femnist_xz | FedGDA-S | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| femnist_xz | FedOGDA-D | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| femnist_xz | FedOGDA-S | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| femnist_z | FedGDA-D | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| femnist_z | FedGDA-S | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| femnist_z | FedOGDA-D | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| femnist_z | FedOGDA-S | federated | 1 | 3 | 0 | 0 | pending_blocked_gpu_permissions |
| femnist_x | DeepGMM-GDA | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| femnist_x | DeepGMM-SGDA | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| femnist_x | DeepGMM-OAdam | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| femnist_z | DeepGMM-GDA | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| femnist_z | DeepGMM-SGDA | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| femnist_z | DeepGMM-OAdam | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| femnist_xz | DeepGMM-GDA | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| femnist_xz | DeepGMM-SGDA | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| femnist_xz | DeepGMM-OAdam | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| cifar10_x | DeepGMM-GDA | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| cifar10_x | DeepGMM-SGDA | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| cifar10_x | DeepGMM-OAdam | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| cifar10_z | DeepGMM-GDA | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| cifar10_z | DeepGMM-SGDA | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| cifar10_z | DeepGMM-OAdam | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| cifar10_xz | DeepGMM-GDA | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| cifar10_xz | DeepGMM-SGDA | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |
| cifar10_xz | DeepGMM-OAdam | centralized | 0 | 0 | 0 | 0 | not_launched_no_validated_artifacts_found |

## 3. Low-Dimensional Federated Results

Full per-run supporting CSV: `experiments/consolidated_results/lowdim_federated_run_metrics.csv`. Aggregated supporting CSV: `experiments/consolidated_results/lowdim_federated_summary.csv`.

Each low-dimensional federated run was checked for required artifacts (`metrics.json`, `mse_by_round.csv`, `predictions.npz`, best/final checkpoints), finite train/validation histories, finite saved predictions, and consistency between `metrics.json` best validation and the minimum validation row in `mse_by_round.csv`.

| function | method_label | validated_runs | mean_best_validation_mse | std_best_validation_mse | mean_test_mse_at_best_validation | std_test_mse_at_best_validation | mean_final_test_mse | mean_last50_val_mse_cv | mean_curve_mse_best | diverged_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Absolute | FedGDA-D | 9 | 0.017562 | 0.0129241 | 0.0180081 | 0.0132694 | 0.0186363 | 0.11147 | 0.0180081 | 0 |
| Absolute | FedGDA-S | 9 | 0.00273625 | 0.00265419 | 0.00280403 | 0.00271432 | 0.00547096 | 0.420038 | 0.00280403 | 0 |
| Absolute | FedOGDA-D | 9 | 0.0164771 | 0.0120665 | 0.0168918 | 0.0123857 | 0.0176441 | 0.119827 | 0.0168918 | 0 |
| Absolute | FedOGDA-S | 9 | 0.0141809 | 0.0114405 | 0.0144941 | 0.0117211 | 0.0162327 | 0.161361 | 0.0144941 | 0 |
| Linear | FedGDA-D | 9 | 0.00413897 | 0.00461897 | 0.00422609 | 0.00473871 | 0.00505035 | 0.35443 | 0.00422609 | 0 |
| Linear | FedGDA-S | 9 | 0.000382524 | 0.000230574 | 0.000386752 | 0.000233534 | 0.00875078 | 0.550913 | 0.000386752 | 0 |
| Linear | FedOGDA-D | 9 | 0.002816 | 0.00313443 | 0.0028616 | 0.00319602 | 0.00310883 | 0.182335 | 0.0028616 | 0 |
| Linear | FedOGDA-S | 9 | 0.00272233 | 0.00273354 | 0.00276799 | 0.00279549 | 0.00326299 | 0.279948 | 0.00276799 | 0 |
| Sine | FedGDA-D | 9 | 0.0836717 | 0.00337372 | 0.0861523 | 0.00376615 | 0.0879951 | 0.00225107 | 0.0861523 | 0 |
| Sine | FedGDA-S | 9 | 0.0766978 | 0.00403073 | 0.078489 | 0.00419233 | 0.0812474 | 0.0328835 | 0.078489 | 0 |
| Sine | FedOGDA-D | 9 | 0.08367 | 0.00340268 | 0.0862134 | 0.00372871 | 0.0879646 | 0.00127988 | 0.0862134 | 0 |
| Sine | FedOGDA-S | 9 | 0.0834235 | 0.00362483 | 0.0858482 | 0.00414134 | 0.0892769 | 0.00598569 | 0.0858482 | 0 |
| Step | FedGDA-D | 9 | 0.0291857 | 0.0024586 | 0.0297296 | 0.00258343 | 0.0306026 | 0.00828062 | 0.0297296 | 0 |
| Step | FedGDA-S | 9 | 0.0265743 | 0.00170735 | 0.0269384 | 0.00174472 | 0.0298803 | 0.0887055 | 0.0269384 | 0 |
| Step | FedOGDA-D | 9 | 0.0286356 | 0.00259952 | 0.0291688 | 0.00274075 | 0.0301273 | 0.00325898 | 0.0291688 | 0 |
| Step | FedOGDA-S | 9 | 0.0288966 | 0.00274828 | 0.0294328 | 0.00286768 | 0.0313423 | 0.0150303 | 0.0294328 | 0 |

Legacy Test-MSE note: all 144 original base-sweep runs have no `test_mse_by_round.csv`, so last-50 Test MSE is unavailable for those runs. Last-50 train/validation statistics and scalar Test MSE are available.

## 4. FedOGDA-D vs FedGDA-D Deterministic Pairwise Comparison

Supporting CSV: `experiments/consolidated_results/lowdim_fedogda_vs_fedgda_pairwise.csv`. Pairing requires same function, alpha, seed, T/R/client settings, batch size, and data/evaluation policy. Tuned Sine A2-lite is labeled separately and is not merged into the original sweep.

| source | function | pairs | FedOGDA-D wins | FedGDA-D wins | mean FedGDA-D Test MSE | mean FedOGDA-D Test MSE | relative improvement pct | best/weakest alpha | seed consistency | readout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_sweep | Absolute | 9 | 6 | 3 | 0.0180081 | 0.0168918 | 6.19894 | best alpha 1; weakest alpha 0.1 | 6/9 pairs won by FedOGDA-D | positive on average but mixed |
| baseline_sweep | Linear | 9 | 6 | 3 | 0.00422609 | 0.0028616 | 32.2873 | best alpha 0.1; weakest alpha 1 | 6/9 pairs won by FedOGDA-D | positive on average but mixed |
| baseline_sweep | Sine | 9 | 2 | 7 | 0.0861523 | 0.0862134 | -0.0709531 | best alpha 1; weakest alpha 0.5 | 2/9 pairs won by FedOGDA-D | not positive in original sweep |
| baseline_sweep | Step | 9 | 9 | 0 | 0.0297296 | 0.0291688 | 1.88638 | best alpha 0.1; weakest alpha 1 | 9/9 pairs won by FedOGDA-D | positive and consistent |
| tuned_sine_a2_lite | Sine | 3 | 3 | 0 | 0.0861069 | 0.0800115 | 7.0788 | best alpha 1; weakest alpha 1 | 3/3 pairs won by FedOGDA-D | positive after tuning |

Interpretation: original Absolute, Step, and Linear support a positive deterministic FedOGDA-D readout. Original Sine did not; tuned Sine A2-lite provides the scoped positive Sine result.

## 5. Tuned Sine A2-lite Detailed Result

Locked recipe: `dataset=sin`, `alpha=1.0`, `T=500`, `R=3`, `g_lr=0.002`, `f_lr=0.03`, `critic_multiplier=15`, `server_lr=1.5`, `weight_decay=0.1`, `client_num_in_total=1000`, `client_num_per_round=1000`, `batch_size=0`, deterministic/full-participation. Selection policy: validation-only; Test MSE was post-selection only.

| seed | fedogda_best_validation_mse | fedogda_best_validation_round | fedgda_test_mse_at_best_validation | fedogda_test_mse_at_best_validation | fedogda_final_test_mse | fedogda_last50_test_mse_mean | fedogda_last50_test_mse_std | fedogda_last50_test_mse_cv | final_vs_last50_gap_pct | relative_gap_pct | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0810736 | 499 | 0.0900601 | 0.0832102 | 0.0832102 | 0.0839005 | 0.000398459 | 0.00474918 | 0.822754 | -7.60587 | FedOGDA-D |
| 1 | 0.0727059 | 499 | 0.0810307 | 0.0746463 | 0.0746463 | 0.0757993 | 0.000677569 | 0.00893899 | 1.52113 | -7.87901 | FedOGDA-D |
| 2 | 0.0800592 | 499 | 0.0872298 | 0.0821781 | 0.0821781 | 0.0825378 | 0.00020783 | 0.002518 | 0.435822 | -5.79127 | FedOGDA-D |

Pairwise FedGDA-D vs FedOGDA-D:
| seed | alpha | T | R | fedgda_test_mse_at_best_validation | fedogda_test_mse_at_best_validation | absolute_gap | relative_gap_pct | winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 500 | 3 | 0.0900601 | 0.0832102 | -0.00684985 | -7.60587 | FedOGDA-D |
| 1 | 1 | 500 | 3 | 0.0810307 | 0.0746463 | -0.00638442 | -7.87901 | FedOGDA-D |
| 2 | 1 | 500 | 3 | 0.0872298 | 0.0821781 | -0.00505172 | -5.79127 | FedOGDA-D |

Curve-fit metrics on saved sorted test points:
| method | rows | curve_mse | curve_mae | curve_max_abs_error |
| --- | --- | --- | --- | --- |
| FedGDA-D | 4 | 0.0861069 | 0.245106 | 0.947549 |
| FedOGDA-D | 4 | 0.0800115 | 0.229407 | 0.917271 |

Plots: `experiments/sine_fedogda_tuning/plots/a2_lite_sine_curve_seed0.png`, `a2_lite_sine_curve_all_seeds.png`, `a2_lite_validation_curves.png`, `a2_lite_test_curves.png`, `a2_lite_last50_zoom.png`.

Stability caveat: tuned FedOGDA-D histories are finite and its last-50 Test CV is below 1%, so final Test MSE is stable enough for this tuned subset. However, do not claim FedOGDA-D improves validation oscillation versus FedGDA-D here; the A2-lite report notes FedGDA-D had lower last-50 validation CV on all three seeds.

## 6. Stochastic Results / FedOGDA-S

Supporting CSVs: `experiments/consolidated_results/lowdim_fedogda_s_vs_fedgda_s_pairwise.csv` and the FedOGDA-S pilot files under `experiments/rerun_protocol_v1/tuning_fedogda_s/pilot_alpha0p5/`.

Original stochastic base sweep:
| function | pairs | FedOGDA_S_wins | mean_FedGDA_S_Test_MSE | mean_FedOGDA_S_Test_MSE | mean gap FedOGDA-S minus FedGDA-S | FedOGDA_lower_val_CV_pairs |
| --- | --- | --- | --- | --- | --- | --- |
| Absolute | 9 | 1 | 0.00280403 | 0.0144941 | 0.0116901 | 9 |
| Linear | 9 | 0 | 0.000386752 | 0.00276799 | 0.00238124 | 8 |
| Sine | 9 | 0 | 0.078489 | 0.0858482 | 0.00735925 | 9 |
| Step | 9 | 1 | 0.0269384 | 0.0294328 | 0.00249433 | 8 |

FedOGDA-S tuning pilot conclusion: validation-selected tuning improved over current FedOGDA-S and reduced validation oscillation versus FedGDA-S for the pilot functions, but tuned FedOGDA-S still had higher mean Test MSE than FedGDA-S on abs, linear, and step. It is not currently suitable for a lower-Test-MSE paper claim.

## 7. Centralized Baselines Status / Results

Supporting CSV: `experiments/consolidated_results/centralized_status_or_results.csv`. C3 full centralized results and C5 tuned GDA/SGDA results are validated. OAdam remains from C3 unless separately tuned.

Comparison-ready centralized table, using C5 tuned GDA/SGDA and C3 OAdam:
| dataset | method | source | mean Test@best val | std | mean best val | mean final test |
| --- | --- | --- | --- | --- | --- | --- |
| abs | DeepGMM-GDA | C5 tuned | 0.056169 | 0.052764 | 0.0545191 | 0.056169 |
| abs | DeepGMM-SGDA | C5 tuned | 0.0562582 | 0.0551927 | 0.0546616 | 0.0562582 |
| abs | DeepGMM-OAdam | C3 full | 0.00177951 | 0.00082712 | 0.00173834 | 0.027302 |
| step | DeepGMM-GDA | C5 tuned | 0.0321919 | 0.00542391 | 0.0318329 | 0.0348947 |
| step | DeepGMM-SGDA | C5 tuned | 0.0319682 | 0.00573632 | 0.0315825 | 0.0368767 |
| step | DeepGMM-OAdam | C3 full | 0.017214 | 0.000572768 | 0.0169726 | 0.022301 |
| linear | DeepGMM-GDA | C5 tuned | 0.00954376 | 0.00761666 | 0.00934649 | 0.0513145 |
| linear | DeepGMM-SGDA | C5 tuned | 0.0079973 | 0.00593145 | 0.00780445 | 0.0579672 |
| linear | DeepGMM-OAdam | C3 full | 0.000759664 | 0.000619418 | 0.000743962 | 0.0216645 |
| sin | DeepGMM-GDA | C5 tuned | 0.0871174 | 0.00698819 | 0.0855477 | 0.0894263 |
| sin | DeepGMM-SGDA | C5 tuned | 0.08768 | 0.00720353 | 0.086022 | 0.0926086 |
| sin | DeepGMM-OAdam | C3 full | 0.0376761 | 0.00343317 | 0.0367957 | 0.0414579 |

C5 tuning materially improved GDA/SGDA over C3:
| dataset | method | g_lr | f_lr | mean_test_mse_at_best_validation | std_test_mse_at_best_validation | c3_mean_test_mse_at_best_validation | c5_relative_test_improvement_pct | improved_over_c3_test | best_at_final_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abs | gda | 0.005 | 0.03 | 0.056169 | 0.052764 | 0.305203 | 81.5962 | 1 | 3 |
| abs | sgda | 0.005 | 0.03 | 0.0562582 | 0.0551927 | 0.303931 | 81.4898 | 1 | 3 |
| step | gda | 0.005 | 0.03 | 0.0321919 | 0.00542391 | 0.0582591 | 44.7436 | 1 | 1 |
| step | sgda | 0.005 | 0.03 | 0.0319682 | 0.00573632 | 0.0581996 | 45.0714 | 1 | 1 |
| linear | gda | 0.005 | 0.03 | 0.00954376 | 0.00761666 | 0.0843963 | 88.6917 | 1 | 0 |
| linear | sgda | 0.005 | 0.03 | 0.0079973 | 0.00593145 | 0.0857472 | 90.6734 | 1 | 0 |
| sin | gda | 0.005 | 0.03 | 0.0871174 | 0.00698819 | 0.101136 | 13.8611 | 1 | 0 |
| sin | sgda | 0.005 | 0.03 | 0.08768 | 0.00720353 | 0.101199 | 13.359 | 1 | 0 |

Centralized vs federated low-dimensional comparison table is saved at `experiments/consolidated_results/centralized_vs_federated_lowdim_comparison.csv`. Current readout: OAdam is strongest in centralized C3; tuned GDA/SGDA are much more reasonable than original C3 but not as strong as OAdam on most functions.

## 8. High-Dimensional Status

Supporting CSV: `experiments/consolidated_results/highdim_status.csv`. For FEMNIST/CIFAR x/z/xz, full reportable results are not complete. The real-image manifest exists, data/preflight/smokes are ready, but `execution_status.md` records that full tuning was blocked by GPU permissions. Therefore high-dimensional results should be reported as pending/blocked, not completed.

## 9. Artifact Inventory

Supporting CSV: `experiments/consolidated_results/artifact_inventory.csv`. Key paths:
| path | exists | modified_time | purpose | paper_facing_or_internal | caveat |
| --- | --- | --- | --- | --- | --- |
| experiments/consolidated_results_for_geetika.md | 1 | 2026-07-11T10:55:22 | consolidated coauthor-ready markdown report | paper-facing summary | generated from existing artifacts; no training launched |
| experiments/consolidated_results/completion_matrix.csv | 1 | 2026-07-11T10:52:05 | completion/status matrix | paper-facing/supporting |  |
| experiments/consolidated_results/lowdim_federated_summary.csv | 1 | 2026-07-11T10:52:05 | low-dimensional federated aggregate metrics | paper-facing/supporting | legacy base runs have no per-round Test MSE |
| experiments/consolidated_results/lowdim_federated_run_metrics.csv | 1 | 2026-07-11T10:52:05 | low-dimensional federated per-run diagnostics | internal/supporting | includes finite-history and saved-prediction checks |
| experiments/consolidated_results/lowdim_fedogda_vs_fedgda_pairwise.csv | 1 | 2026-07-11T10:52:05 | deterministic FedOGDA-D vs FedGDA-D pairwise comparison | paper-facing/supporting | tuned Sine rows labeled separately |
| experiments/consolidated_results/lowdim_fedogda_s_vs_fedgda_s_pairwise.csv | 1 | 2026-07-11T10:50:52 | stochastic FedOGDA-S vs FedGDA-S pairwise comparison | internal/supporting | FedOGDA-S lower-Test-MSE claim not supported |
| experiments/consolidated_results/sine_tuned_summary.csv | 1 | 2026-07-11T10:52:05 | A2-lite tuned Sine seed-level summary | paper-facing/supporting | validation-only locked recipe |
| experiments/consolidated_results/centralized_status_or_results.csv | 1 | 2026-07-11T10:52:05 | centralized C3/C5 results status | paper-facing/supporting | GDA/SGDA tuned in C5; OAdam from C3 |
| experiments/consolidated_results/centralized_vs_federated_lowdim_comparison.csv | 1 | 2026-07-11T10:50:52 | centralized vs federated low-dimensional comparison | paper-facing/supporting | sources labeled; tuned Sine separate |
| experiments/consolidated_results/highdim_status.csv | 1 | 2026-07-11T10:56:55 | FEMNIST/CIFAR status matrix | status/supporting | pending/blocked; no full validated high-dimensional results |
| experiments/consolidated_results/artifact_inventory.csv | 1 | 2026-07-11T10:55:03 | artifact inventory | internal/supporting | this file |
| experiments/rerun_protocol_v1/manifest.csv | 1 | 2026-06-25T19:15:33 | base low-dimensional federated manifest | internal/source | run_status columns are stale; artifacts checked directly |
| results/rerun_protocol_v1 | 1 | 2026-06-27T20:38:57 | base low-dimensional federated result root | artifact root | 144 validated low-dimensional federated runs |
| experiments/rerun_protocol_v1/test_mse_stability_report.md | 1 | 2026-07-06T18:47:01 | legacy per-round Test-MSE audit | analysis/source | per-round Test MSE absent for old runs |
| experiments/sine_fedogda_tuning/a2_lite_final_report.md | 1 | 2026-07-09T14:37:43 | tuned Sine A2-lite final report | paper-facing/source | do not merge silently into original sweep |
| experiments/sine_fedogda_tuning/plots/a2_lite_sine_curve_all_seeds.png | 1 | 2026-07-09T14:37:42 | tuned Sine all-seed curve plot | paper-facing/plot | saved sorted test points |
| experiments/rerun_protocol_v1/tuning_fedogda_s/pilot_alpha0p5/analysis_summary.md | 1 | 2026-07-05T22:00:29 | FedOGDA-S pilot analysis report | analysis/source | pilot does not beat FedGDA-S Test MSE |
| experiments/centralized_baselines/centralized_c3_full_run_report.md | 1 | 2026-07-10T20:46:43 | centralized C3 full report | paper-facing/source | C3 GDA/SGDA under-tuned; OAdam strong |
| experiments/centralized_baselines/centralized_c5b_confirm_report.md | 1 | 2026-07-11T04:49:10 | centralized C5b confirmation report | paper-facing/source | C5 tuned GDA/SGDA complete |
| experiments/centralized_baselines/centralized_c5_final_gda_sgda_tuned_summary.csv | 1 | 2026-07-11T04:49:10 | centralized tuned GDA/SGDA summary | paper-facing/source | OAdam not retuned |
| experiments/centralized_baselines/plots/centralized_curve_abs_seed0.png | 1 | 2026-07-11T02:21:44 | centralized abs curve plot | paper-facing/plot | seed 0 |
| experiments/centralized_baselines/plots/centralized_curve_step_seed0.png | 1 | 2026-07-11T02:21:44 | centralized step curve plot | paper-facing/plot | seed 0 |
| experiments/centralized_baselines/plots/centralized_curve_linear_seed0.png | 1 | 2026-07-11T02:21:44 | centralized linear curve plot | paper-facing/plot | seed 0 |
| experiments/centralized_baselines/plots/centralized_curve_sin_seed0.png | 1 | 2026-07-11T02:21:44 | centralized sine curve plot | paper-facing/plot | seed 0 |
| experiments/rerun_protocol_v1_real_images_abs_alpha0p5/execution_status.md | 1 | 2026-07-11T04:30:32 | high-dimensional execution status | status/source | blocked by GPU permissions; no final full result |
| experiments/rerun_protocol_v1_real_images_abs_alpha0p5/manifest.csv | 1 | 2026-07-05T15:37:47 | high-dimensional federated manifest | internal/source | manifest only for full results |
| results/_smoke/real_image_abs | 1 | 2026-07-11T04:25:28 | real-image smoke root | internal/smoke | not final reportable results |

## 10. Claims We Can Safely Make

Supported:

- Low-dimensional federated base sweep completed and validated for four synthetic functions and four federated variants: 144/144 valid, 0 divergence.
- In the original deterministic sweep, FedOGDA-D improves average validation-selected Test MSE versus FedGDA-D for Absolute, Step, and Linear.
- A validation-tuned deterministic Sine A2-lite recipe supports the scoped claim that well-tuned FedOGDA-D can outperform paired FedGDA-D on Sine at alpha=1.0, with 3/3 seed wins and lower saved-point curve error.
- Centralized low-dimensional baselines are implemented and validated. OAdam is strong; tuned GDA/SGDA are available from C5.

Unsupported / do not claim:

- Do not claim universal FedOGDA superiority across all methods/functions/settings.
- Do not claim FedOGDA-S beats FedGDA-S on Test MSE; current data do not support that.
- Do not claim old base-sweep Test MSE stability over the last 50 rounds; those runs did not log per-round Test MSE.
- Do not claim high-dimensional FEMNIST/CIFAR completion; full validated artifacts are absent.
- Do not say Test MSE was used for selection; all tuning selections documented here are validation-only.

## 11. Recommended Next Steps

1. Freeze the low-dimensional reporting story: original deterministic FedOGDA-D wins on abs/step/linear plus tuned Sine A2-lite as the Sine result.
2. Build the final paper table with columns for FedGDA-D, FedOGDA-D original, tuned Sine FedOGDA-D, centralized tuned GDA/SGDA, and centralized OAdam, keeping sources labeled.
3. For stochastic results, report stability/oscillation improvements cautiously; do not position FedOGDA-S as a lower-Test-MSE win yet.
4. If Geetika wants an even cleaner deterministic story, consider one small validation-only tuning pass on the weakest original deterministic function after Sine, but Step/Absolute/Linear already look positive in the original sweep.
5. Resume high-dimensional FEMNIST/CIFAR only after GPU permissions/queue are solved; first goal should be a small validated representative subset before any full matrix.

## 12. Short Message Draft to Geetika

Hi Geetika, quick consolidated status: the low-dimensional federated base sweep is complete and validated for Sine, Linear, Absolute, and Step across FedGDA-D/S and FedOGDA-D/S: 144/144 runs, no divergence. In the original deterministic sweep, FedOGDA-D is positive on Absolute (6/9 wins, mean Test MSE 0.01801 -> 0.01689), Step (9/9 wins, 0.02973 -> 0.02917), and Linear (6/9 wins, 0.00423 -> 0.00286). Original Sine was not positive, but after validation-only tuning the deterministic Sine A2-lite recipe wins 3/3 paired seeds at alpha=1.0, mean Test MSE 0.08611 -> 0.08001, with better saved-point curve fitting. FedOGDA-S tuning improved over current FedOGDA-S and reduced oscillation, but it still does not beat FedGDA-S on Test MSE, so I would not claim a stochastic lower-Test-MSE win yet. Centralized low-dimensional baselines are now implemented and validated: C3 has 36/36 complete, and C5 tuned GDA/SGDA is also complete; OAdam is strong. High-dimensional FEMNIST/CIFAR full results are still pending/blocked by GPU permissions, so I will not mark those as complete. My suggested next step is to finalize the low-dimensional table with sources clearly labeled, then decide whether to run a small high-dimensional representative subset once GPU access is sorted.

