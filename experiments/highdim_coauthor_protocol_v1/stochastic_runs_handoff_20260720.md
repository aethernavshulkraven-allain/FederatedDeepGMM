# High-Dimensional Stochastic Runs Handoff

Generated: `2026-07-20`.

This note documents the high-dimensional stochastic federated runs only. It is
intended for review and verification of the experiment design, launch history,
result locations, runtime/code changes, and final artifact state.

## Final State

The stochastic final matrix is complete.

| Item | Value |
|---|---:|
| Planned stochastic final runs | 180 |
| Valid completed final runs | 180 |
| Partial final runs | 0 |
| Pending final runs | 0 |
| Invalid completed final runs | 0 |
| Aggregate summary rows | 36 |

The final audit script is:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/audit_highdim_stochastic_finals.py
```

Latest audit output:

```json
{
  "valid_final_runs": 180,
  "aggregate_rows": 36,
  "index_csv": "experiments/highdim_coauthor_protocol_v1/stochastic_final_artifact_index.csv",
  "summary_csv": "experiments/highdim_coauthor_protocol_v1/stochastic_final_aggregate_summary.csv"
}
```

The audit checks each run for:

- `effective_config.json`
- `metrics.json`
- `mse_by_round.csv`
- `predictions.npz`
- `checkpoints/best_validation.pt`
- `checkpoints/final.pt`
- exactly 1500 rows in `mse_by_round.csv`
- matching dataset, method, seed in `effective_config.json`
- `diverged != true`
- finite validation/test metrics
- `test_mse_used_for_selection != true`

## Design Of Experiment

Protocol: `highdim_coauthor_protocol_v1`.

Design sources:

- `high_dim_exp.md`
- `high_dim_doe_2.md`
- `experiments/highdim_coauthor_protocol_v1/protocol_summary.json`
- `scripts/prepare_highdim_coauthor_protocol.py`
- `scripts/materialize_highdim_stochastic_finals.py`

Scientific objective: evaluate stochastic federated DeepGMM methods when `x`,
`z`, or both are represented by real images. The structural response is fixed:

```text
g(x) = abs(x)
```

Scenarios:

| Scenario | x representation | z representation |
|---|---|---|
| `femnist_x` | FEMNIST image | scalar |
| `femnist_z` | scalar | FEMNIST image |
| `femnist_xz` | FEMNIST image | FEMNIST image |
| `cifar10_x` | CIFAR-10 image | scalar |
| `cifar10_z` | scalar | CIFAR-10 image |
| `cifar10_xz` | CIFAR-10 image | CIFAR-10 image |

Stochastic federated methods:

| Report label | Repository method | Client optimizer |
|---|---|---|
| FedGDA-S / FedSGDA | `fedgda_s` | `sgd` |
| FedOGDA-S | `fedogda_s` | `ogda` |

Model mapping:

- image side uses CNN: CIFAR-10 uses `CIFAR10CNN`, FEMNIST uses `DefaultCNN`;
- scalar side uses `MLPModel`;
- therefore `*_x` is CNN `g` plus MLP `f`, `*_z` is MLP `g` plus CNN `f`,
  and `*_xz` is CNN `g` plus CNN `f`.

Fixed stochastic run shape:

| Parameter | Value |
|---|---:|
| Dirichlet alphas | `0.1`, `0.5`, `1.0` |
| Scenarios | 6 |
| Methods | 2 |
| Seeds | `0, 1, 2, 3, 4` |
| Total clients | 1000 |
| Clients per round | 10 |
| Batch size | 256 |
| Local steps / epochs | 3 |
| Communication rounds | 1500 |

This gives:

```text
3 alphas x 6 scenarios x 2 stochastic methods x 5 seeds = 180 runs
```

## Tuning And Selection

Tuning used seed `0`, validation metrics only, 150 communication rounds, and
two learning-rate candidates per alpha/scenario/method:

```text
learning_rate in {0.003, 0.01}
weight_decay = 0.05
critic_multiplier = 10
server_learning_rate = 1.5
gradient_clip_norm = 1.0
```

Tuning manifests:

- `experiments/highdim_coauthor_protocol_v1/alpha0p1/tuning_manifest_stochastic.csv`
- `experiments/highdim_coauthor_protocol_v1/alpha0p5/tuning_manifest_stochastic.csv`
- `experiments/highdim_coauthor_protocol_v1/alpha1/tuning_manifest_stochastic.csv`

Tuning result roots:

- `results/rerun_protocol_v1_real_images_abs_alpha0p1_tuning`
- `results/rerun_protocol_v1_real_images_abs_alpha0p5_tuning`
- `results/rerun_protocol_v1_real_images_abs_alpha1_tuning`

Tuning revalidation status:

| Alpha dir | Rows | Status |
|---|---:|---|
| `alpha0p1` | 24/24 | `skipped_completed` |
| `alpha0p5` | 24/24 | `skipped_completed` |
| `alpha1` | 24/24 | `skipped_completed` |

Important note: `alpha1/tuning_stochastic_launcher_results.json` contains
wrapper `returncode=120` / `failed_process` noise, but
`alpha1/tuning_stochastic_revalidation_results.json` confirms all 24 stochastic
tuning artifacts were complete and reusable. The selected config files below
are derived from validated artifacts.

Selection files:

- `experiments/highdim_coauthor_protocol_v1/alpha0p1/selected_configs_stochastic.csv`
- `experiments/highdim_coauthor_protocol_v1/alpha0p5/selected_configs_stochastic.csv`
- `experiments/highdim_coauthor_protocol_v1/alpha1/selected_configs_stochastic.csv`

Final stochastic manifests:

- `experiments/highdim_coauthor_protocol_v1/alpha0p1/final_manifest_stochastic.csv`
- `experiments/highdim_coauthor_protocol_v1/alpha0p5/final_manifest_stochastic.csv`
- `experiments/highdim_coauthor_protocol_v1/alpha1/final_manifest_stochastic.csv`

Selection rule:

1. Exclude numerical divergence only.
2. Rank by lowest `best_validation_mse`.
3. Tie-break by lower last-50-round validation-MSE standard deviation.
4. Tie-break by smaller final-minus-best validation gap.
5. Tie-break by lower learning rate.

Test MSE was not used for tuning or selection. The final reporting metric is
`test_mse_at_best_validation`, read only after the validation-selected run is
fixed.

## Execution History

The final stochastic artifacts are intentionally spread across preserved roots
and fresh continuation roots. Do not assume one alpha root contains the full
completed matrix. Use `stochastic_final_artifact_index.csv` as the source of
truth for final result paths.

| Phase | Manifest / index | Output root | Result |
|---|---|---|---:|
| Existing preserved finals | `stochastic_speedup_migration_20260719_123539/preserved_old_completed_runs.csv` | original alpha roots | 19 complete |
| Safe speedup continuation v1 | `stochastic_speedup_migration_20260719_123539/remaining_pending_manifest_stochastic_safe_speedup_v1.csv` | `results/rerun_protocol_v1_real_images_abs_remaining_safe_speedup_v1_20260719_123539` | 84 complete, 1 partial preserved |
| Safe speedup continuation v2 | `stochastic_speedup_migration_20260719_233531/remaining_pending_manifest_stochastic_safe_speedup_v2.csv` | `results/rerun_protocol_v1_real_images_abs_remaining_safe_speedup_v2_20260719_233531` | 77 complete |

Total:

```text
19 old complete + 84 v1 complete + 77 v2 complete = 180 final runs
```

The v1 partial was not resumed from its checkpoint. It was preserved and rerun
from scratch in the v2 root because the available resume path did not restore
the full best-validation state/history/auxiliary-regression state safely.

Correction from independent review (see "Independent Review And Verification"
below): one run filed under `old_original`,
`highdim_abs_cifar10_xz_fedgda_s_seed0_alpha1`, actually carries the post-fix
runtime flags (`append_round_csv=True`, `auxiliary_regression_epochs=3`,
`periodic_checkpoint_interval=200`) and a ~21-minute runtime consistent with
the safe-speedup code path, not the original path. The true pre-fix/post-fix
split is **18 pre-fix / 162 post-fix**, not 19/161. This does not change the
180/19/84/77 provenance-source counts used for indexing, only which code
version actually produced one of the `old_original`-labeled runs.

Launch records:

- v1 migration summary:
  `experiments/highdim_coauthor_protocol_v1/stochastic_speedup_migration_20260719_123539/migration_summary.json`
- v1 launch status:
  `experiments/highdim_coauthor_protocol_v1/stochastic_speedup_migration_20260719_123539/launch_status.json`
- v2 migration summary:
  `experiments/highdim_coauthor_protocol_v1/stochastic_speedup_migration_20260719_233531/migration_summary.json`
- v2 launch status:
  `experiments/highdim_coauthor_protocol_v1/stochastic_speedup_migration_20260719_233531/launch_status.json`
- v2 launcher results:
  `experiments/highdim_coauthor_protocol_v1/stochastic_speedup_migration_20260719_233531/launcher_results.json`

Actual v1 launch command recorded by the migration:

```bash
tmux new-session -d -s hd_stoch_safe_speedup_v1_20260719_123539 -c /home/arnav22103/FederatedDeepGMM gpurun -g 2 /home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_manifest.py --manifest experiments/highdim_coauthor_protocol_v1/stochastic_speedup_migration_20260719_123539/remaining_pending_manifest_stochastic_safe_speedup_v1.csv --config-dir experiments/highdim_coauthor_protocol_v1/stochastic_speedup_migration_20260719_123539/generated_configs --output-root results/rerun_protocol_v1_real_images_abs_remaining_safe_speedup_v1_20260719_123539 --gpu-ids 0,1 --max-parallel 2 --resume-skip-completed --keep-going --results-json experiments/highdim_coauthor_protocol_v1/stochastic_speedup_migration_20260719_123539/launcher_results.json
```

Actual v2 launch command:

```bash
tmux new-session -d -s hd_stoch_safe_speedup_v2_20260719_233531 -c /home/arnav22103/FederatedDeepGMM gpurun -g 1 /home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_manifest.py --manifest experiments/highdim_coauthor_protocol_v1/stochastic_speedup_migration_20260719_233531/remaining_pending_manifest_stochastic_safe_speedup_v2.csv --config-dir experiments/highdim_coauthor_protocol_v1/stochastic_speedup_migration_20260719_233531/generated_configs --output-root results/rerun_protocol_v1_real_images_abs_remaining_safe_speedup_v2_20260719_233531 --gpu-ids 0 --max-parallel 1 --resume-skip-completed --keep-going --results-json experiments/highdim_coauthor_protocol_v1/stochastic_speedup_migration_20260719_233531/launcher_results.json
```

v2 launcher result status:

```text
77 passed / 77 rows
```

## Result Files For Review

Primary review files generated by the final audit:

- `experiments/highdim_coauthor_protocol_v1/stochastic_final_artifact_index.csv`
- `experiments/highdim_coauthor_protocol_v1/stochastic_final_aggregate_summary.csv`
- `scripts/audit_highdim_stochastic_finals.py`

The 180-row index has one row per final run and includes:

- alpha, dataset, method, seed, run ID
- exact result directory
- provenance source: `old_original`, `safe_speedup_v1`, or `safe_speedup_v2`
- learning rate and fixed protocol constants
- best validation round and MSE
- `test_mse_at_best_validation`
- final-round Test MSE
- runtime
- runtime-control flags recorded in `metrics.json`

The 36-row aggregate summary has one row per
`alpha x scenario x stochastic method` and reports mean/std over five seeds.

## Aggregate Final Results

Metric: mean `test_mse_at_best_validation` over five seeds, with sample standard
deviation.

| alpha | scenario | method | n | Test MSE at best val | runtime median min |
|---:|---|---|---:|---:|---:|
| 0.1 | `cifar10_x` | FedGDA-S | 5 | 0.1588 +/- 0.0125 | 13.4 |
| 0.1 | `cifar10_x` | FedOGDA-S | 5 | 0.1730 +/- 0.0268 | 13.9 |
| 0.1 | `cifar10_xz` | FedGDA-S | 5 | 0.1679 +/- 0.0235 | 22.6 |
| 0.1 | `cifar10_xz` | FedOGDA-S | 5 | 0.1621 +/- 0.0104 | 22.7 |
| 0.1 | `cifar10_z` | FedGDA-S | 5 | 0.0449 +/- 0.0147 | 12.5 |
| 0.1 | `cifar10_z` | FedOGDA-S | 5 | 0.0791 +/- 0.0240 | 12.9 |
| 0.1 | `femnist_x` | FedGDA-S | 5 | 0.1590 +/- 0.0186 | 10.3 |
| 0.1 | `femnist_x` | FedOGDA-S | 5 | 0.1541 +/- 0.0115 | 11.1 |
| 0.1 | `femnist_xz` | FedGDA-S | 5 | 0.1554 +/- 0.0165 | 16.4 |
| 0.1 | `femnist_xz` | FedOGDA-S | 5 | 0.1519 +/- 0.0133 | 16.2 |
| 0.1 | `femnist_z` | FedGDA-S | 5 | 0.0118 +/- 0.0021 | 9.4 |
| 0.1 | `femnist_z` | FedOGDA-S | 5 | 0.0103 +/- 0.0011 | 9.9 |
| 0.5 | `cifar10_x` | FedGDA-S | 5 | 0.1890 +/- 0.0561 | 31.8 |
| 0.5 | `cifar10_x` | FedOGDA-S | 5 | 0.1648 +/- 0.0178 | 32.1 |
| 0.5 | `cifar10_xz` | FedGDA-S | 5 | 0.1641 +/- 0.0150 | 22.3 |
| 0.5 | `cifar10_xz` | FedOGDA-S | 5 | 0.1644 +/- 0.0095 | 22.4 |
| 0.5 | `cifar10_z` | FedGDA-S | 5 | 0.0553 +/- 0.0080 | 12.4 |
| 0.5 | `cifar10_z` | FedOGDA-S | 5 | 0.0646 +/- 0.0054 | 12.9 |
| 0.5 | `femnist_x` | FedGDA-S | 5 | 0.1695 +/- 0.0252 | 10.5 |
| 0.5 | `femnist_x` | FedOGDA-S | 5 | 0.1365 +/- 0.0076 | 10.8 |
| 0.5 | `femnist_xz` | FedGDA-S | 5 | 0.1729 +/- 0.0281 | 16.3 |
| 0.5 | `femnist_xz` | FedOGDA-S | 5 | 0.1433 +/- 0.0124 | 16.4 |
| 0.5 | `femnist_z` | FedGDA-S | 5 | 0.0130 +/- 0.0023 | 9.6 |
| 0.5 | `femnist_z` | FedOGDA-S | 5 | 0.0175 +/- 0.0033 | 10.4 |
| 1 | `cifar10_x` | FedGDA-S | 5 | 0.1579 +/- 0.0121 | 31.8 |
| 1 | `cifar10_x` | FedOGDA-S | 5 | 0.1728 +/- 0.0304 | 30.6 |
| 1 | `cifar10_xz` | FedGDA-S | 5 | 0.1656 +/- 0.0149 | 21.9 |
| 1 | `cifar10_xz` | FedOGDA-S | 5 | 0.1657 +/- 0.0089 | 22.0 |
| 1 | `cifar10_z` | FedGDA-S | 5 | 0.0513 +/- 0.0092 | 12.1 |
| 1 | `cifar10_z` | FedOGDA-S | 5 | 0.0876 +/- 0.0251 | 12.7 |
| 1 | `femnist_x` | FedGDA-S | 5 | 0.1586 +/- 0.0126 | 10.1 |
| 1 | `femnist_x` | FedOGDA-S | 5 | 0.1375 +/- 0.0133 | 10.8 |
| 1 | `femnist_xz` | FedGDA-S | 5 | 0.1444 +/- 0.0092 | 16.3 |
| 1 | `femnist_xz` | FedOGDA-S | 5 | 0.1489 +/- 0.0108 | 16.7 |
| 1 | `femnist_z` | FedGDA-S | 5 | 0.0147 +/- 0.0034 | 9.4 |
| 1 | `femnist_z` | FedOGDA-S | 5 | 0.0174 +/- 0.0036 | 10.3 |

## Runtime Investigation And Code Changes

Runtime investigation report:

- `experiments/highdim_coauthor_protocol_v1/stochastic_gpu_util_investigation.md`
- `experiments/highdim_coauthor_protocol_v1/stochastic_gpu_util_profile_summary.csv`
- `results/_profiling/highdim_stochastic_gpu_util`

The profiling found low average H100 utilization and large overhead from
serial client orchestration, model-selection setup, auxiliary regression,
full validation every round, and repeated state/CSV work. The production final
continuations used only the protocol-preserving safe path:

- no `--skip-model-selection`
- no `--skip-gmm-eval`
- no `--disable-auxiliary-regression`
- append round CSV rows instead of rewriting the whole CSV each round
- periodic checkpoints every 200 rounds
- exact 3 auxiliary-regression local passes instead of accidental nested
  `3 x 3 = 9` passes

Relevant code paths:

- `scripts/run_manifest.py`
- `scripts/run_highdim_stochastic_gpu_profile.py`
- `scripts/analyze_highdim_stochastic_gpu_util.py`
- `fedgmm/sp_decentralized_mnist_lr_example/experiment_utils.py`
- `fedgmm/sp_decentralized_mnist_lr_example/fedml/simulation/sp/fedavg/fedavg_api.py`
- `fedgmm/sp_decentralized_mnist_lr_example/fedml/ml/trainer/my_model_trainer_classification.py`
- `fedgmm/sp_decentralized_mnist_lr_example/fedml/data/cifar10/efficient_loader.py`

Important reviewer caveat: the final matrix combines 18 older pre-fix runs
with 162 runs produced after the safe runtime changes (see the correction in
"Execution History" above; the `old_original` provenance label covers 19 rows
but one of those 19 actually ran post-fix code). The main scientific
configuration stayed the same, but the auxiliary-regression epoch bug fix is a
real implementation change and should be called out in review. Each result row
records runtime flags in `metrics.json`, though only for the 162 post-fix rows
— the 18 genuinely pre-fix rows have empty runtime-flag columns in the index.
Three aggregate cells mix pre-fix and post-fix seeds: (alpha=0.5, `cifar10_x`,
FedGDA-S) and both methods at (alpha=1.0, `cifar10_x`) are entirely pre-fix;
(alpha=0.5, `cifar10_x`, FedOGDA-S) mixes 3 pre-fix seeds with 2 post-fix
seeds. Runtime medians are not comparable across alpha for `cifar10_x`: the
~31-32 min medians at alpha=0.5/1.0 versus ~13-14 min at alpha=0.1 reflect the
code-version split (9 vs 3 auxiliary-regression passes), not an alpha effect.

## Files That Are Noisy Or Stale

Do not use these as the sole status source:

- `experiments/highdim_coauthor_protocol_v1/final_stochastic_queue_summary.json`
  is stale/noisy from an earlier wrapper attempt.
- `alpha*/completion_status.json` mixes deterministic and stochastic tuning
  state and is not the final stochastic-only status.
- `alpha1/tuning_stochastic_launcher_results.json` contains wrapper
  `returncode=120` noise, but revalidation proves the artifacts are complete.
- v1 has no final `launcher_results.json`; use v1 preserved indexes plus the
  final audit instead.

Use these instead:

- `scripts/audit_highdim_stochastic_finals.py`
- `experiments/highdim_coauthor_protocol_v1/stochastic_final_artifact_index.csv`
- `experiments/highdim_coauthor_protocol_v1/stochastic_final_aggregate_summary.csv`
- `experiments/highdim_coauthor_protocol_v1/alpha*/tuning_stochastic_revalidation_results.json`
- `experiments/highdim_coauthor_protocol_v1/stochastic_speedup_migration_20260719_*/migration_summary.json`

## Independent Review And Verification (2026-07-20)

An independent pass re-derived every claim in this document from source
artifacts rather than trusting the audit script's own output. Summary
verdict: the audit script and this handoff are **mechanically accurate and
reproducible**, but the aggregate results **do not confirm the expected
heterogeneity trend or the expected FedOGDA-S stability advantage**, and the
headline `test_mse_at_best_validation` numbers come from checkpoints selected
very early in a 1500-round budget, with severe last-iterate blowup afterward.

### Mechanical checks (all passed)

- **Audit reproducibility**: rerunning
  `scripts/audit_highdim_stochastic_finals.py` regenerated
  `stochastic_final_artifact_index.csv` and
  `stochastic_final_aggregate_summary.csv` byte-identical to the committed
  versions.
- **Provenance accounting**: manifest row counts sum to exactly
  19 + 84 + 77 = 180; both `migration_summary.json` files are internally
  consistent with the preserved/pending counts they report; the v2 launcher
  results show 77/77 `passed`; v1 has no `launcher_results.json`, as this
  document already states.
- **Matrix coverage**: the 180 index keys are exactly the cartesian product
  of 3 alphas x 6 scenarios x 2 methods x 5 seeds, with no duplicate keys and
  no duplicate `result_dir` values. Every `alpha x scenario x method` group
  has `n=5`.
- **Protocol constants**: all 180 rows have `comm_round=1500`,
  `client_num_in_total=1000`, `client_num_per_round=10`, `epochs=3`,
  `batch_size=256`, `weight_decay=0.05`, `critic_multiplier=10`,
  `server_learning_rate=1.5`, `gradient_clip_norm=1.0`, `learning_rate` in
  `{0.003, 0.01}`, and `test_mse_used_for_selection=False`.
- **metrics.json vs raw curves**: for all 180 runs, `best_validation_mse`,
  `best_validation_round`, and `final_validation_mse` in `metrics.json` were
  recomputed independently from `mse_by_round.csv` and matched exactly; no
  round-level `diverged`/non-finite flags were found anywhere.
- **Selection rule**: for all 36 alpha/scenario/method cells, the learning
  rate recorded in `selected_configs_stochastic.csv` matches the documented
  five-step selection rule applied to `candidate_validation_metrics_stochastic.csv`
  (lowest `best_validation_mse`, then last-50-round std, then final-minus-best
  gap, then lower learning rate); all tuning candidates are seed 0 with
  `best_validation_round <= 149`. Final-run learning rates match the
  selections in all 180 rows.
- **Tuning revalidation**: `alpha0p1`, `alpha0p5`, `alpha1` each show 24/24
  `skipped_completed`, as claimed.
- **Aggregate table**: all 36 rows of the "Aggregate Final Results" table
  were independently recomputed from the index (mean, sample std, runtime
  median) and match to within floating-point tolerance.

### Discrepancies found (see corrections inlined above)

1. One `old_original`-labeled run
   (`highdim_abs_cifar10_xz_fedgda_s_seed0_alpha1`) actually carries post-fix
   runtime flags and runtime — true pre-fix/post-fix split is 18/162, not
   19/161.
2. Three aggregate cells mix pre-fix and post-fix provenance across seeds
   (listed above); values do not show an obvious systematic shift within
   those cells at n=5, but the mixing should be disclosed.
3. `cifar10_x` runtime medians are not comparable across alpha values due to
   the code-version split, not an alpha effect.
4. "Each result row records runtime flags in `metrics.json`" is only true
   for the 162 post-fix rows; the 18 pre-fix rows have empty flag columns.

### Audit script robustness gaps

`scripts/audit_highdim_stochastic_finals.py` is fail-fast and correctly
raises on missing artifacts, wrong row counts, divergence, and non-finite
metrics, but it does not independently enforce several things this review
checked by hand. Currently all pass, but the script would not catch a
regression in these should one occur later:

- exact equality of the 180 keys to the expected alpha/scenario/method/seed
  cartesian product (a different but still-180-and-non-duplicated set would
  pass silently);
- per-group `n == 5`;
- protocol-constant values (`comm_round`, `client_num_in_total`, etc.) beyond
  what is implicitly recorded;
- `learning_rate` consistency against `selected_configs_stochastic.csv`;
- recomputation of `best_validation_mse`/`best_validation_round` from
  `mse_by_round.csv` rather than trusting `metrics.json` alone.

### Scientific assessment against expected stochastic behavior

**Confirmed:**

- Best validation occurs before the final round in all 180 runs (max best
  round 1497 of 1499), and final-round test MSE never equals
  `test_mse_at_best_validation` — consistent with expectation, but see the
  magnitude caveat below.
- OGDA-vs-GDA superiority is scenario-dependent, not uniform: FedOGDA-S has
  the lower mean `test_mse_at_best_validation` in 8/18 alpha/scenario cells,
  FedGDA-S in 10/18. FedGDA-S is better in every `cifar10_z` cell; FedOGDA-S
  is better in most `femnist_x`/`femnist_xz` cells.
- FedOGDA-S has lower seed-to-seed variance than FedGDA-S (mean within-cell
  std 0.0131 vs 0.0158 across the 18 cells) — a genuine stability advantage.
- FEMNIST is easier than CIFAR-10 on the `z` scenarios (0.010-0.018 vs
  0.045-0.088 mean test MSE); `x`/`xz` scenarios are roughly comparable
  between datasets.

**Not confirmed / contradicted:**

1. **No heterogeneity trend.** Pooled mean `test_mse_at_best_validation` is
   0.1190 / 0.1212 / 0.1185 for alpha = 0.1 / 0.5 / 1.0 — essentially flat.
   Zero of the 12 scenario/method combinations show the expected monotone
   alpha=0.1 > alpha=0.5 > alpha=1.0 ordering; `femnist_z` FedGDA-S trends in
   the opposite direction (0.0118 to 0.0147 as alpha increases). The paper's
   own claim is only "marginally higher" MSE under higher heterogeneity, and
   its reference setup uses alpha=0.3 rather than this protocol's
   {0.1, 0.5, 1.0}, which may partly explain the mismatch; per-alpha learning
   rate retuning and best-validation checkpointing over 1500 rounds with
   10-of-1000 client sampling per round are also plausible dampers.
2. **Last-iterate behavior is worse than "noisy" — it is a large blowup for
   image-`x` scenarios.** For `*_x`/`*_xz` scenarios, median
   `best_validation_round` is ~13-25 (of 1500) for FedGDA-S and ~20-80 for
   FedOGDA-S. Median final/best test-MSE ratio is ~7x for both methods;
   83/90 FedGDA-S runs and 81/90 FedOGDA-S runs finish more than 2x above
   their best-validation test MSE; 40/90 runs per method finish with final
   test MSE > 1.0 versus ~0.15 at the selected checkpoint. The reported
   headline numbers are effectively early-stopped checkpoints from the first
   1-2% of the round budget for these scenarios, not a converged last
   iterate.

   This blowup is a genuine property of the training dynamics, not a bug or
   an artifact of the final runs, on three grounds. (a) It reproduces across
   three independent execution contexts — the 150-round tuning runs, the
   pre-fix `old_original` finals, and the post-fix v1/v2 finals — with the
   same shape. The tuning metrics already show it: in the `*_x` scenarios,
   *both* learning-rate candidates ended round 150 at 2-15x their best
   validation MSE (e.g., alpha=0.1 `cifar10_x` FedGDA-S: lr=0.003 ended at
   3.6x best, lr=0.01 at 15.2x). (b) Independent measurement paths agree:
   the per-round eval curve and the separately-computed checkpoint test MSE
   match at both the best round and the final round. (c) The same code path
   produces stable late-improving curves for the `*_z` scenarios and blowup
   for `*_x`/`*_xz`, tracking the model-role structure (CNN vs MLP on the
   structural side), which a code defect would not respect.

   However, the magnitude is partly protocol-inflicted, in two specific
   ways. First, the searched grid contained no stable configuration for the
   `*_x` scenarios: both lr candidates blow up within 150 rounds, so tuning
   could only choose between two unstable options (fixed
   `critic_multiplier=10`, `weight_decay=0.05`, `server_learning_rate=1.5`
   were never varied). Second, the selection rule ranks by
   `best_validation_mse` alone, with the final-vs-best gap used only as a
   late tie-break — so it systematically preferred a deeper early dip over
   tail stability. Concrete example: alpha=0.1 `cifar10_x` FedGDA-S chose
   lr=0.01 over lr=0.003 on a 0.157-vs-0.158 best-val difference, thereby
   accepting a 15.2x round-150 gap instead of a 3.6x one. The data
   therefore shows that no stable config existed *in the searched grid*,
   not that stochastic federated GDA cannot be stabilized here; a targeted
   stability probe (one blowup cell rerun with lower lr and/or lower critic
   multiplier and/or lr decay) would be needed to distinguish those.
3. **FedOGDA-S does not show a smaller final-vs-best gap.** Median final/best
   ratio is 7.6 (p90 24.5) for FedOGDA-S versus 7.0 (p90 17.2) for FedGDA-S —
   the opposite of the expected optimism-damping benefit, at least as
   measured by post-hoc test MSE ratio. Where FedOGDA-S does differ: on the
   `z` scenarios it keeps improving much later into the budget (median best
   round 1481 vs 600 for FedGDA-S on `cifar10_z`) rather than peaking early,
   yet still ends with higher mean test MSE there.
4. **Scenario difficulty is inverted relative to the stated prior.** The
   `z`-image scenarios are the easiest (femnist_z ~ 0.01, cifar10_z ~ 0.05)
   and `x`-image scenarios are the hardest (~0.15-0.19); `cifar10_z` under
   high heterogeneity is among the better cells, not the unstable worst case
   suggested by the older appendix-style numbers. This is mechanistically
   plausible — MSE measures the structural function `g`, which in `*_z` is a
   scalar-to-scalar MLP learning `abs(x)` while the CNN burden falls on the
   critic side — but it should be reconciled explicitly against whatever
   prior table motivated the "z/xz harder, cifar10_z especially unstable"
   expectation.
5. **FedOGDA-S is the only method that produces literal NaN excursions
   mid-training, contradicting its expected stability advantage.** A
   round-level scan of every `mse_by_round.csv` (all 180 runs, both
   `val_mse` and `train_mse` columns) found 29/90 FedOGDA-S runs (32%) with
   at least one round where these values are NaN, totaling 167 bad rows
   across the matrix; **zero of the 90 FedGDA-S runs have any NaN round.**
   The NaN rounds are concentrated in `cifar10_xz` (14/30 FedOGDA-S runs
   there, 146 of the 167 bad rows); the remainder are scattered singletons
   or short clusters in `cifar10_x`, `femnist_x`, and `femnist_xz`. All
   affected values are NaN, not +/-inf, and in every case checked the
   selected `best_validation_round` lands well clear of the NaN round(s), so
   the headline `test_mse_at_best_validation` numbers are not directly
   corrupted by this (Python-style argmin over a list containing NaN safely
   skips NaN entries rather than adopting them, since any comparison against
   NaN is `False`, and no bad row is `-inf`, which is the only value that
   could hijack an argmin this way). Separately, **the per-round `finite`
   column in `mse_by_round.csv` is unreliable**: all 167 NaN rows found in
   this scan are mislabeled `finite=True`, so that column cannot be used as
   a stability signal without a fix to how it is computed. This finding
   should be treated as evidence against, not for, FedOGDA-S's claimed
   numerical-stability benefit in this stochastic high-dimensional setting,
   at least for the `cifar10_xz` scenario.

   Mechanism (established by code trace, not speculation): the OGDA
   implementation itself is correct — standard update
   `x <- x - (2*lr*g_t - lr*g_{t-1})`, per-round optimizer-state reset, and
   per-gradient clipping before each step. The NaNs arise because gradient
   clipping bounds each gradient to `gradient_clip_norm` *before* the
   optimistic combination, so when consecutive stochastic gradients disagree
   (batch noise plus 10-of-1000 non-i.i.d. client sampling), the combined
   OGDA step can reach up to 3x the SGD bound at the same learning rate.
   This compounds on the critic side, which trains at
   `critic_multiplier * lr = 10 * lr` with an f-gradient that scales with
   the squared structural residual (`OptimalMomentObjective` f_reg term),
   and is largest in `cifar10_xz` where the critic is a full `CIFAR10CNN`.
   The weights never actually become non-finite — the round-level
   `finite`/`diverged` flags check weight tensors via `state_is_finite`,
   and those checks pass — but the finitely-large weights overflow the
   forward pass during evaluation, producing NaN train/val MSE for that
   round before server aggregation pulls the model back. This is why
   `diverged` never trips (it is a weight-level check, and weights stay
   finite) and why the `finite` column mislabels these rounds (it measures
   weight finiteness, not metric finiteness — the logging gap noted above).
   Reported headline metrics are unaffected: the best-checkpoint update
   uses `val_mse < best`, and any comparison against NaN is False, so a
   NaN round can never be selected as the best-validation checkpoint.

### Recommendation

Treat the 180/36 completeness and provenance claims in this document as
verified. Do not yet treat the aggregate table as confirming the paper's
heterogeneity or optimism-correction claims in this high-dimensional,
stochastic, real-image setting — the last-iterate divergence on `*_x`/`*_xz`
scenarios means `test_mse_at_best_validation` is summarizing a narrow,
early-training checkpoint rather than stabilized behavior, and any writeup
should show the best-vs-final gap and best-round distribution alongside the
mean test MSE, not the mean alone. Do not describe FedOGDA-S as the more
numerically stable method without qualification: it has lower seed-to-seed
variance in the reported metric, but it is also the only method exhibiting
outright NaN excursions mid-training (item 5 above), concentrated in
`cifar10_xz`. Fix or remove the per-round `finite` column in the training
loop before relying on it for any future audit, since it does not currently
detect NaN values.

Method comparison, stated plainly: FedGDA-S is not uniformly better than
FedOGDA-S. Wins split 10/18 vs 8/18 by cell count, and roughly cancel out
in magnitude — FedGDA-S's advantage is concentrated in `cifar10_z` (70-76%
relative MSE reduction at alpha 0.1 and 1.0), FedOGDA-S's in `femnist_x` and
`femnist_xz` (up to ~19% relative reduction at alpha 0.5). FedOGDA-S's only
consistent, unambiguous advantage in this data is lower seed-to-seed
variance; it does not win on mean MSE overall, on final-vs-best gap, or on
freedom from numerical blowup.

## Reproduction Commands

Regenerate the base protocol manifests:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/prepare_highdim_coauthor_protocol.py
```

Revalidate/materialize stochastic final manifests after tuning artifacts exist:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/materialize_highdim_stochastic_finals.py
```

Audit the completed final matrix and regenerate final index/summary:

```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/audit_highdim_stochastic_finals.py
```

Canonical one-GPU stochastic tuning wrapper:

```bash
gpurun -g 1 /home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_highdim_coauthor_tuning_queue.py --regime stochastic
```

Canonical final-run launcher wrapper, if starting from per-alpha manifests:

```bash
gpurun -g 1 /home/arnav22103/miniconda3/envs/fedgmm/bin/python scripts/run_highdim_coauthor_final_queue.py --regime stochastic
```

For actual continuation work, prefer fresh output roots and disjoint manifests
as done in v1/v2, so no existing scientific artifacts are overwritten.
