# Invalidation note: mislabeled FedOGDA-D boundary-expansion-2 runs — 2026-08-19

## What happened

`screen_expand2_manifest.csv` (generated 2026-08-18, launched as job 531)
contained 17 rows: 7 `method=fedgda_d` and 10 `method=fedogda_d`. The
generator, `scripts/prepare_highdim_deterministic_screen_expand2_20260818.py`,
picked each row's hyperparameter-defaults template via `reference_row()`,
which matched on **dataset only**. The source manifest
(`alpha0p5/tuning_manifest_deterministic.csv`) lists its `fedgda_d` row
before its `fedogda_d` row for every dataset, so `reference_row()` always
returned the `fedgda_d` template — including for rows whose `method` column
correctly said `fedogda_d`. The template's `client_optimizer="sgd"` and
`method_label="FedGDA-D"` were never overridden for those 10 rows.

Consequence: all 10 `fedogda_d`-labeled rows were configured with
`client_optimizer="sgd"`, which is FedGDA-D, not FedOGDA-D. All 10 launched,
ran the full 150-round screen protocol to completion, and produced
complete, structurally valid-looking artifacts (`metrics.json`,
`mse_by_round.csv`, `predictions.npz`, checkpoints) — genuinely trained
under SGD, not OGDA. The training logs confirm `CustomSGD` (not `OGDA`) was
instantiated in every case.

Because these artifacts' true `variant` (computed internally by
`experiment_utils.get_effective_config()` from `client_optimizer`, not from
the manifest's `method` column) is `fedgda_d`, they were written to
`results/highdim_deterministic_screen_20260813/<dataset>/fedgda_d/seed_0/`
— under `fedogda_d`-named run IDs, inside the `fedgda_d` directory tree.

`run_manifest.py`'s own post-run artifact check (`validate_artifacts`,
which does compare `effective_config.json`'s `variant` against the
manifest's `method`) never got the chance to catch this: it looked for
artifacts at the path implied by the manifest's `method` column
(`.../fedogda_d/...`), found nothing there, and correctly reported
`VALIDATION FAIL: missing artifacts` for all 10 rows — so none of these
runs were ever recorded as `"passed"` in
`screen_expand2_launcher_results.json`, and no downstream script (e.g.
`score_highdim_screen_by_psi.py`) that trusts that results file would have
picked them up as valid FedOGDA-D screen data. The actual risk was narrower
but still real: the completed, mislabeled artifacts sat on disk inside the
normal `fedgda_d` results tree, reachable by anything that scans the
results directory directly rather than the launcher's own bookkeeping.

The other 7 rows (`method=fedgda_d`) were correct throughout — their
template match happened to be right by construction, since `fedgda_d` was
always the row `reference_row()` returned regardless of what was asked for.
4 of the 7 passed; 3 failed for an unrelated, genuine reason (`lr=0.333333`,
the new boundary rung, triggered `RuntimeError: No valid model-selection
candidate was selected` during pretraining — a real divergence result, not
a labeling bug).

## Affected run IDs (all 10, method=fedogda_d, actually trained as fedgda_d/sgd)

| Run ID | Dataset | Intended | Actually ran as |
|---|---|---|---|
| `det_screen_expand2_cifar10_xz_fedogda_d_seed0_alpha0p5_lr0p009_cm5` | cifar10_xz | FedOGDA-D | FedGDA-D (sgd) |
| `det_screen_expand2_cifar10_xz_fedogda_d_seed0_alpha0p5_lr0p003_cm10` | cifar10_xz | FedOGDA-D | FedGDA-D (sgd) |
| `det_screen_expand2_cifar10_xz_fedogda_d_seed0_alpha0p5_lr0p009_cm10` | cifar10_xz | FedOGDA-D | FedGDA-D (sgd) |
| `det_screen_expand2_cifar10_x_fedogda_d_seed0_alpha0p5_lr0p01_cm40` | cifar10_x | FedOGDA-D | FedGDA-D (sgd) |
| `det_screen_expand2_cifar10_z_fedogda_d_seed0_alpha0p5_lr0p009_cm1` | cifar10_z | FedOGDA-D | FedGDA-D (sgd) |
| `det_screen_expand2_cifar10_z_fedogda_d_seed0_alpha0p5_lr0p003_cm2` | cifar10_z | FedOGDA-D | FedGDA-D (sgd) |
| `det_screen_expand2_cifar10_z_fedogda_d_seed0_alpha0p5_lr0p009_cm2` | cifar10_z | FedOGDA-D | FedGDA-D (sgd) |
| `det_screen_expand2_femnist_xz_fedogda_d_seed0_alpha0p5_lr0p001_cm10` | femnist_xz | FedOGDA-D | FedGDA-D (sgd) |
| `det_screen_expand2_femnist_x_fedogda_d_seed0_alpha0p5_lr0p09_cm10` | femnist_x | FedOGDA-D | FedGDA-D (sgd) |
| `det_screen_expand2_femnist_z_fedogda_d_seed0_alpha0p5_lr0p001_cm10` | femnist_z | FedOGDA-D | FedGDA-D (sgd) |

GPU cost actually spent on these 10 mislabeled runs: **2.48 GPU-h** (measured,
summed `runtime_seconds` from the quarantined `metrics.json` files) —
real quota consumed, not recoverable, but the resulting artifacts are
unusable as FedOGDA-D candidates.

## What was checked and found clean

Every other manifest in `experiments/highdim_coauthor_protocol_v1/` was
audited for the same pattern (`client_optimizer` vs.
`METHOD_TO_OPTIMIZER[method]` mismatch), including the 108-run finals
manifest, the 72-row original screen, and the 19-row first expansion. **All
clean.** This bug is confined to the two scripts written 2026-08-18:
`prepare_highdim_deterministic_screen_expand2_20260818.py` (this incident)
and `prepare_highdim_psi_adjudication_20260818.py`, which had the identical
`reference_row()`-by-dataset-only bug and produced 33 of its 63 planned
rows (12/21 in `adjudication_x_manifest.csv`, 21/42 in
`adjudication_signal_manifest.csv`) with the same wrong `client_optimizer`
— those manifests were never launched, so no GPU time was lost there, but
they must be regenerated from the fixed script before use.

## Disposition

- Mislabeled artifacts, logs, and traces moved to
  `QUARANTINE_20260819_mislabeled_fedogda_expand2/` (this directory),
  intact, not deleted — available for audit but excluded from every
  scoring/selection path.
- `reference_row()` fixed in both generator scripts to match on
  `(dataset, method)`; both scripts now also set `client_optimizer` and
  `method_label` explicitly from a `METHOD_TO_OPTIMIZER`/`METHOD_LABEL`
  map rather than trusting the copied template.
- `scripts/run_manifest.py` now asserts, before generating any config
  (dry-run and real launch both), that `client_optimizer` matches
  `METHOD_TO_OPTIMIZER[method]` and that `method_label` matches
  `METHOD_LABEL[method]` — confirmed this now rejects the original
  (uncorrected, still on disk for the audit trail)
  `screen_expand2_manifest.csv`'s 10 bad rows immediately at dry-run,
  before any GPU time is spent, while still accepting its 7 correct rows
  and a held-out `fedgda_d` row from the original screen manifest as a
  regression check.
- Ten corrected rows generated under new run IDs
  (`det_screen_expand2corr_...`, `protocol_version =
  highdim_deterministic_screen_expand2_corrected_v1`) in
  `screen_expand2_corrected_v1_manifest.csv` — does not reuse or overwrite
  any old identifier or output path. Relaunched 2026-08-19 (job 532).
