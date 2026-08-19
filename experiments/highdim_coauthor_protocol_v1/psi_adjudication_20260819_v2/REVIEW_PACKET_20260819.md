# Review packet — corrected re-score, rebuilt adjudication manifests — 2026-08-19

Prepared per the go/no-go gate: no adjudication launch until this packet
passes review. Nothing in this packet was launched — every step below is
non-GPU (re-scoring existing artifacts, generating manifests, dry-run
validation).

## Scientific finding this packet is built on

At the one-rung boundary extensions tested at alpha 0.5, seed 0,
**FedOGDA-D produced non-finite events in 7/10 candidates** (6 diverged, 1
non-finite metrics), **whereas the accidentally-run SGD counterparts at the
same configurations and seed all remained finite.** This is local evidence
only — scoped to these specific boundary configurations, alpha 0.5, seed 0
— and is not generalized to other alphas, seeds, or configurations.

## 1. Before/after candidate diff (all 12 cells)

File: `../deterministic_screen_20260813/review_packet_1_before_after_diff.json`

"Before" = `psi_rescore.json` (2026-08-18, screen + expansion-1 only).
"After" = `psi_rescore_corrected_v2.json` (2026-08-19, full corrected pool:
screen + expansion-1 + 4 valid fedgda_d expansion-2 + 3 valid corrected
fedogda_d expansion-2).

- **3 of 12 cells changed rank-1 and/or rank-2**: `cifar10_xz/fedgda_d`,
  `cifar10_z/fedgda_d`, `femnist_z/fedogda_d` — all newly promoted by
  expansion-2 data that didn't exist when the 2026-08-18 psi_rescore was
  computed (the first two by valid `fedgda_d` expansion-2 candidates, the
  third by the corrected `fedogda_d` expansion-2 candidate).
- **Of the 3 surviving corrected FedOGDA-D expansion-2 candidates, only 1
  entered a cell's new top-2**: `femnist_z/fedogda_d` (lr=0.001, cm=10) is
  the new rank-1. `cifar10_x/fedogda_d` (lr=0.01, cm=40) and
  `cifar10_z/fedogda_d` (lr=0.003, cm=2) both trained successfully but did
  not outrank the existing screen+expansion-1 candidates for their cells.
- **Existing MSE winner sits in the new Ψ top-2 for 3 of 12 cells**
  (`cifar10_x/fedgda_d`, `cifar10_z/fedogda_d`, `femnist_z/fedgda_d`); for
  the other 9 it does not — but per generator logic (verified in §3 below)
  it is **unconditionally included as a labeled candidate in the
  adjudication manifest for every cell regardless of Ψ rank.**

## 2. Complete eligibility/exclusion ledger

Files: `../deterministic_screen_20260813/candidate_audit_ledger_20260819.{json,csv}`

118 total candidates tracked across all 5 sources (original screen 72,
expansion-1 19, expansion-2 fedgda_d 7, expansion-2 mislabeled fedogda_d
10, expansion-2 corrected fedogda_d 10). **85 eligible, 33 excluded**, every
exclusion with an explicit category and reason:

| Category | n | Reason |
|---|---|---|
| `original_screen_or_expansion1_exclusion` | 13 | diverged / NaN metrics, pre-existing (9 screen + 4 expansion-1) |
| `mislabeled_optimizer_bug` | 10 | client_optimizer='sgd' instead of 'ogda'; quarantined, retained for audit, never eligible |
| `fedgda_d_pretraining_failure` | 3 | genuine divergence at lr=0.333333 boundary rung during model selection |
| `corrected_fedogda_d_nonfinite_or_diverged` | 7 | genuine OGDA instability at boundary rungs (the scientific finding above) |

The 10 quarantined mislabeled runs retain their achieved `gmm_eval`/
`val_mse` in the ledger (labeled `actually_ran_as: fedgda_d (sgd)`) for the
audit trail, but are excluded from every ranking.

## 3. Reuse ledger

File: `psi_adjudication_20260819_v2/review_packet_3_reuse_ledger.json`

For every candidate the v2 manifests mark `reused_from_finals` (one per
cell, 12 candidates × 3 seeds = 36 trajectory checks), verified against
`deterministic_finals_20260813/finals_manifest.csv` that the matched row
agrees on method, optimizer, **alpha (0.5, not 0.1 or 1.0)**, seed,
learning rate, critic multiplier, comm_round=500, and that its artifacts
exist and are not diverged.

**A note on how this check was built:** the first version of this ledger
keyed reused trajectories by `(dataset, method, lr, cm, seed)` only,
omitting alpha. Since `finals_manifest.csv` covers alpha ∈ {0.1, 0.5, 1.0},
that omission caused all 36 checks to silently verify against the **wrong
alpha (1.0)** finals rows — a bug in this checker, not in the actual
adjudication pipeline (which never reads finals data directly; it only
skips generating a new run when a candidate's (lr, cm) matches the MSE
winner). Fixed by adding alpha=0.5 to the lookup key; disclosing this
because it's exactly the kind of self-caught verification bug worth being
transparent about.

**Result after the fix: 35/36 verified clean, 1 flagged.**
`cifar10_xz/fedogda_d` seed=2's existing finals trajectory
(`det_final_cifar10_xz_fedogda_d_seed2_alpha0p5_lr0p003_cm1`) has
`diverged: true` in its own `metrics.json` (best_gmm_eval=0.012,
best_validation_mse=0.227). Since this campaign's federation is fully
deterministic (`client_num_in_total == client_num_per_round == 10`,
`batch_size=0`, no client sampling), this is not a data-availability gap
that a rerun could resolve — the same configuration and seed will
reproduce the same divergence every time. Per the non-finite-seed rule
(§0 below), **this candidate is already known ineligible for promotion**:
a candidate is promotable only if all 3 seeds are complete, finite, and
non-diverged, and no two-seed median or reproduction-attempt substitutes
for a missing third. `cifar10_xz/fedogda_d`'s existing MSE winner remains
in the three-way comparison ledger for that cell (never silently dropped)
but enters the scorer already excluded by this rule — resolved between
whichever of Ψ rank-1/rank-2 remain eligible, per the amendment's §0
decision tree, not by any special-casing for this cell.

New-run candidates (not reused, will be generated fresh): 21.

## 4. Semantic manifest validation + rendered-YAML checks

Both `adjudication_x_manifest.csv` (21 rows) and `adjudication_signal_manifest.csv`
(42 rows):
- 0/63 rows have `client_optimizer`/`method_label` mismatched against
  `method` (checked directly against `run_manifest.METHOD_TO_OPTIMIZER`/
  `METHOD_LABEL`).
- `run_manifest.py --dry-run`: **63/63 launchable, 0 skipped_unlaunchable**
  (this exercises the mandatory prelaunch invariant added 2026-08-19).
- Rendered one YAML per method and read `client_optimizer` directly off
  disk: `fedgda_d` row → `client_optimizer: "sgd"`; `fedogda_d` row →
  `client_optimizer: "ogda"`. Both correct.
- `test_mse_used_for_selection` is `"False"` in all 63 rows;
  `log_test_mse_by_round` is `"False"` in all 63 rows. No scoring/ranking
  script in this pipeline (`score_highdim_screen_corrected_v2_20260819.py`,
  `build_review_packet_20260819.py`, `prepare_highdim_psi_adjudication_20260818.py`,
  `build_reuse_ledger_20260819.py`) reads any `test_mse` field.

## 5. Recalculated cost + immutable hashes

File: `review_packet_4_recalculated_cost.json`

Recalculated from **scenario-specific measured 500-round runtimes** (mean
per (dataset, method) cell, from the 108 completed
`deterministic_finals_20260813` runs) applied to the actual v2 manifest row
counts per cell -- not the flat 47.9 GPU-h estimate quoted 2026-08-18.

**Total: 39.89 GPU-h** (21 `_x` rows + 42 `_signal` rows = 63 new runs).
This is materially different from the earlier estimate and matters for
planning: it now fits inside a single 48 GPU-h/week quota reset in
principle, though not alongside anything else that week.

File: `../deterministic_screen_20260813/review_packet_5_immutable_hashes.json`
— SHA-256 of every frozen input to this packet (both amendment docs, the
invalidation note, the corrected re-score and audit ledger, both v2
manifests, the reuse ledger, the RELABEL doc, frozen_winners.json, and
every preparer/scorer script version used to build this packet).

## 6. Adjudication manifest changes vs. the provisional (2026-08-18) version

File: `psi_adjudication_20260819_v2/review_packet_provisional_vs_v2_diff.json`

- `adjudication_x_manifest.csv`: **0 rows changed** (21/21 identical
  candidates) -- none of the corrected FedOGDA-D expansion-2 survivors
  changed the `_x` cells' top-2, and the mislabeled-optimizer bug never
  touched the `_x` provisional manifest's candidate set in the first place
  (that manifest's fedogda_d rows were always built from the pre-bug
  screen+expansion-1-only psi_rescore.json).
- `adjudication_signal_manifest.csv`: **9 of 42 rows changed** (3 candidates
  × 3 seeds, across `cifar10_xz/fedgda_d`, `cifar10_z/fedgda_d`,
  `femnist_z/fedogda_d` -- exactly the 3 cells flagged as rank-changed in
  §1). Cross-validated two independent ways (candidate-set diff here, and
  the rank-1/rank-2 diff in `review_packet_1`) -- both agree on the same 3
  cells.

## 7. Adjudication scorer, written and tested before launch

File: `../../../scripts/score_highdim_adjudication_20260819.py`, tests in
`../../../tests/test_adjudication_scorer_20260819.py`.

Implements `PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md` §0's full decision
tree mechanically: eligibility (all 3 seeds complete/finite/non-diverged,
no two-seed median, no imputation), then 0/1/≥2-eligible branching, then
median-Ψ ranking, top-vs-others pairwise tie test, tie-set construction,
and MSE tiebreak within the tie set. Critic collapse has no representation
in this module at all — it cannot gate promotion because there is no input
for it to act on.

**6/6 synthetic tests pass** (`python3 -m unittest
tests.test_adjudication_scorer_20260819`): three separated candidates (top
wins outright), a pairwise tie (2 of 3, resolved by MSE), a three-way tie
(resolved by MSE), one candidate excluded by a single diverged seed (its
two good seeds are confirmed NOT averaged in), a single eligible candidate
(promoted without ranking), and zero eligible candidates (retune_required,
no winner).

## Go/no-go

All five requested pieces are complete, plus the four supplementary checks
(surviving-candidate top-2 membership, provisional-vs-v2 row change count,
MSE-winner retention, no-test-metric confirmation). One caveat surfaced
(§3, `cifar10_xz/fedogda_d` seed=2 pre-existing divergence) — carried
forward as a scoring-time caveat, not a blocker.

**Recommendation: packet is ready for review.** Pending your review, next
step is committing this snapshot (amendments, corrected scoring inputs,
preparers, manifests, hashes) as one immutable prelaunch commit, then the
orchestrator waits for Monday's quota reset before launching anything.
