# Protocol decision addendum — 2026-08-26

This addendum freezes the decisions required by
[`HIGH_DIM_DETERMINISTIC_CLOSEOUT_PLAN_20260826.md`](HIGH_DIM_DETERMINISTIC_CLOSEOUT_PLAN_20260826.md)
§3.2, before any new results are observed. It supersedes the "unresolved" Ψ
framing in
[`POST_BN_SCREEN_REVIEW_20260826.md`](POST_BN_SCREEN_REVIEW_20260826.md) §7.3.

## 1. Ψ implementation

The campaign evaluates the **historical legacy FederatedDeepGMM/CausalML Ψ
implementation**, as it exists in this repository today. This resolves the
decision tree in the post-BN review §7.3 down its first branch: preserve the
104 completed corrected-screen trajectories, fix only the last-50 scorer, and
rerun only the diagnostic and the four failures. A complete 108-configuration
rerun is not required.

Two known legacy-implementation issues are explicitly **not** changed by this
campaign:

- the critic-history pooling statement
  `f_of_z_dev_list.extend(f_of_z_dev_list)` in
  [`model_selection_class.py`](../../../fedgmm/sp_decentralized_mnist_lr_example/model_selection_class.py)
  (numerically inert here because this campaign uses exactly one internal
  model/critic/learning-setup combination per run — see §3 below); and
- the tilde-residual returned by
  [`max_approx_psi_eval`](../../../fedgmm/sp_decentralized_mnist_lr_example/game_objectives/approximate_psi_objective.py),
  which returns the last-evaluated candidate's residual rather than the
  residual belonging to the maximizing candidate.

Both are real bugs, but fixing either would change the legacy Ψ arithmetic
mid-campaign. They are deferred to a separately versioned implementation, per
the closeout plan §3.3.

## 2. Screen and V4 scoring statistics

- Screen scoring uses **mean Ψ over communication rounds 100–149** (last 50 of
  150), per
  [`PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md`](../deterministic_screen_20260813/PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md).
- V4 candidate scoring uses **mean Ψ over communication rounds 450–499** (last
  50 of 500) per seed, followed by the **median across seeds {0,1,2}**.
- Best-round Ψ (`best_gmm_eval`) is retained in all scorer output as
  diagnostic metadata only. It is never used for ranking or selection.

## 3. Model-selection scope for this campaign

Every screen, V4, stability, and final row in this campaign uses exactly one
structural model, one critic model, and one learning setup — i.e.
`len(g_model_list) == len(f_model_list) == len(learning_args_list) == 1`, so
`g_f_args_list` in `do_model_selection` has exactly one entry. This is now a
fail-closed assertion in code (closeout plan §3.3), not just an observation
about the existing screen.

## 4. Terminal pretraining failures

A model-selection failure before federated round 0 is a terminal scientific
outcome (`terminal_pretraining_ineligible`) **only** when backed by a
validated `pretraining_failure.json` artifact written by the training process
itself at the moment of failure (closeout plan §4.2). A bare nonzero return
code with no such artifact remains an unexplained process failure and must not
be silently reclassified as terminal.

## 5. Final campaign target

The reported deterministic evidence covers all three protocol alpha values
(`{0.1, 0.5, 1.0}`) and seeds `0..4` — 6 datasets × 2 methods × 3 alphas × 5
seeds = 180 trajectories, per
[`protocol_summary.json`](../protocol_summary.json) and
[`doe_review_and_revised_grid.md`](../doe_review_and_revised_grid.md).
Completing V4 candidate adjudication at `alpha=0.5` closes only the candidate-
selection stage; it does not by itself close the campaign.

## 6. Reuse of V4 and stability trajectories in the final table

**Decision: exact reuse.** The 500-round `alpha=0.5`, seeds {0,1,2}
trajectories produced by V4 for each cell's frozen winner, and the 500-round
`alpha=0.1`, seed 0 stability trajectory for each cell (when it passes without
per-cell retuning), are reused verbatim as final-table rows rather than
rerun. Seed, alpha, configuration, and the 500-round horizon are identical
between those stages and the final table, and staged reuse is already part of
the adopted DOE (`doe_review_and_revised_grid.md`, adopted minimal plan:
"Finals" = 144 runs, "Stability @ alpha=0.1" = 12 runs — the 144 already
assumes the 36 V4-winner trajectories are not separately rerun at
`alpha=0.5`).

Under exact reuse, the accounting is:

| Final evidence | Runs |
|---|---:|
| V4 winners at `alpha=0.5`, seeds 0–2 (reused) | 36 |
| `alpha=0.1` stability runs, seed 0 (reused) | 12 |
| New `alpha=0.1` runs, seeds 1–4 | 48 |
| New `alpha=0.5` runs, seeds 3–4 | 24 |
| New `alpha=1.0` runs, seeds 0–4 | 60 |
| **Total final evidence** | **180** |
| **New runs required after V4 + stability** | **132** |

If a cell requires `alpha=0.1` retuning under the escape hatch in
`doe_review_and_revised_grid.md`, that cell's original failed stability run is
**not** reused as a final-table row for `alpha=0.1`; the newly selected
`alpha=0.1` winner for that cell is run across all required final seeds
instead, per closeout plan §9.3.

## 7. This addendum's own source edits stale the screen's prelaunch hash freeze

Implementing SS3.3 and Phase 1 (SS4.1-SS4.7) required editing several
CORE_SOURCES files that were already executed by the 104 completed screen
runs (`model_selection_class.py`, `experiment_utils.py`,
`fedml/simulation/sp/fedavg/fedavg_api.py`, `run_manifest.py`). As a direct,
expected consequence, `deterministic_screen_post_bn_20260822/generated_artifact_hashes.json`
--frozen when the screen was originally launched, before these edits -- no
longer verifies against the live tree
(`scripts/verify_protocol_hashes.py --hashes .../generated_artifact_hashes.json`
now reports a hash mismatch for `experiment_utils.py`).

This is a **provenance-record staleness**, not evidence that the 104
completed runs trained under different code than they actually did. None of
these edits changed any arithmetic already executed by those runs (SS3.3's
assertion, SS4.2's failure-evidence capture, SS4.4's compact-artifact
writer, and the new `val_target_variance`/hash-closure fields are all
observational additions or apply only to runs launched after this addendum).
The existing screen's hash record is **not regenerated** to match the new
source -- doing so would misrepresent what code the already-completed runs
actually ran under, which is exactly the discipline SS1 above already
applies to the retired v3 diagnostic. This is the same retrospective-
provenance caveat POST_BN_SCREEN_REVIEW_20260826.md SS10.3 already flagged
as a formal weakness, not a correctness defect, and it is unaffected by this
addendum.

Any *new* protocol hash freeze created after this addendum (the fresh
120-round diagnostic, the corrected-screen rescoring, V4, stability, or
finals) is generated from the post-edit source tree, so this specific
staleness does not propagate forward.

## 8. Provenance caveat

This addendum itself, and every script it depends on, is included in the
expanded prelaunch hash closure (closeout plan §4.5) before any further GPU
work is launched under this protocol.
