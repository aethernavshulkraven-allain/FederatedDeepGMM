# High-dimensional deterministic closeout plan — 2026-08-26

## Executive decision

The 104 successful corrected-screen runs should be preserved. Because the
campaign will use the historical legacy FederatedDeepGMM/CausalML Ψ
implementation, a complete 108-run screen rerun is not required.

The immediate work is to:

1. repair the scoring and provenance pipeline;
2. certify the four failures that occurred before federated round 0;
3. repeat the inexpensive BatchNorm diagnostic under a prelaunch hash freeze;
4. rescore and freeze the corrected screen;
5. perform the three-seed V4 adjudication; and
6. finish the α-stability and five-seed final stages.

V4 by itself does **not** close the complete deterministic experiment. V4
selects candidates at `alpha=0.5` over seeds `{0,1,2}`. The adopted DOE also
calls for an `alpha=0.1` stability check and a final table over three alpha
values and five seeds.

## 1. Current campaign state

| Component | Current status |
|---|---|
| BatchNorm server-update fix | Implemented; corrected policy appears healthy |
| Corrected screen | 108 attempted: 104 complete, 4 failed before round 0 |
| Audit of 104 runs | Passed; no BatchNorm-driven rerun needed |
| Ψ implementation choice | Decided: retain historical legacy implementation |
| Screen scorer | Still wrong: uses best-round Ψ instead of last-50 mean |
| Four failed rows | Plausible boundary failures, but not properly certified |
| `screen_results.json` | Not generated |
| Boundary decisions | Not frozen |
| V4 packet | Preparation scripts exist, packet not generated |
| V4 signal/X | Not launched |
| `alpha=0.1` stability stage | No post-BN packet/launcher yet |
| Full post-BN finals | No packet/launcher or runs yet |

The detailed audit and rationale are in
[`POST_BN_SCREEN_REVIEW_20260826.md`](POST_BN_SCREEN_REVIEW_20260826.md).

## 2. What “complete” means

The deterministic campaign has three remaining scientific levels:

```text
Corrected screen
    ↓
V4 candidate adjudication at alpha=0.5, seeds 0–2
    ↓
alpha=0.1 stability check
    ↓
Final evidence: alpha ∈ {0.1, 0.5, 1.0}, seeds 0–4
```

The V4 preparer currently generates 500-round runs for seeds `0`, `1`, and
`2` only. That is appropriate for candidate promotion: the frozen rule selects
using the median last-50 Ψ across those three seeds.

The adopted DOE says that the reported deterministic table covers all three
alpha values and five seeds. The repository-wide protocol summary also lists
seeds `0..4`.

Therefore:

- completing V4 means **post-BN candidate adjudication is complete**;
- completing V4 does not mean the full deterministic campaign is complete;
- full closure requires the stability and five-seed final stages as well.

Relevant protocol records:

- [`doe_review_and_revised_grid.md`](../doe_review_and_revised_grid.md)
- [`protocol_summary.json`](../protocol_summary.json)
- [`PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md`](../deterministic_screen_20260813/PSI_SUMMARY_STATISTIC_AMENDMENT_20260819.md)
- [`BOUNDARY_RULE_AMENDMENT_20260818.md`](../deterministic_screen_20260813/BOUNDARY_RULE_AMENDMENT_20260818.md)

## 3. Phase 0 — Freeze the protocol and integration state

Complete this phase before any additional GPU work.

### 3.1 Assign one integration owner

Stop concurrent edits to the campaign files and designate one session as the
integration owner. Preserve the existing corrected screen, its attempt ledger,
and all result directories unchanged.

Review the dirty worktree carefully. Do not commit all modified and untracked
files indiscriminately, because the tree contains overlapping work from
multiple sessions and older campaign artifacts.

### 3.2 Write a dated protocol decision addendum

Record the following decisions before seeing any new results:

1. The campaign evaluates the historical legacy FederatedDeepGMM/CausalML Ψ
   implementation.
2. Do not change the legacy tilde-residual or Ψ arithmetic during this
   campaign.
3. Screen scoring uses mean Ψ over rounds `100..149`.
4. V4 candidate scoring uses mean Ψ over rounds `450..499`, followed by the
   median across seeds `{0,1,2}`.
5. Best-round Ψ is diagnostic only.
6. A model-selection failure before round 0 can be a terminal scientific
   outcome only when supported by the frozen structured-failure contract.
7. The final target is three alpha values and five seeds.
8. State explicitly whether exact V4 and stability trajectories are reused in
   the final table.

The post-BN review currently describes the Ψ choice as unresolved. Add a dated
decision note stating that the legacy branch has been selected.

### 3.3 Preserve the legacy Ψ implementation

The active model-selection code contains:

```python
f_of_z_dev_list.extend(f_of_z_dev_list)
```

This is a real general-purpose pooling bug, but it was numerically inert in the
corrected screen because each run had exactly one structural model, one critic,
and one learning setup. Duplicating identical critic entries does not change
the minimum used by Ψ.

For this campaign:

- leave the legacy Ψ and tilde-residual arithmetic unchanged;
- add a fail-closed assertion that the campaign has exactly one internal
  model/critic/learning combination; and
- fix general multi-candidate critic pooling later under a separately versioned
  implementation.

## 4. Phase 1 — Finish the remaining code-level fixes

### 4.1 Correct the screen scorer

The current scorer reads `metrics["best_gmm_eval"]`. That is the best Ψ seen
anywhere in the run, whereas the frozen protocol requires the last-50-round
mean.

Change [`scripts/score_highdim_screen_post_bn_20260822.py`](../../../scripts/score_highdim_screen_post_bn_20260822.py)
to:

1. load `mse_by_round.csv`;
2. require exactly 150 ordered rows with indices `0..149`;
3. compute `psi_last50_mean` from rounds `100..149`;
4. compute tail validation MSE over the same window for fallback/tie handling;
5. retain `best_gmm_eval` only as diagnostic metadata;
6. reject missing, duplicated, unordered, blank, or nonfinite values; and
7. report exact ties without using manifest order as a tie breaker.

Add tests in which best-round and last-50 rankings deliberately disagree. The
wrong statistic currently changes top-two membership in 8 of 12 cells and the
rank-one candidate in 4 of 12 cells.

This change does not require a training rerun because all 104 successful runs
already store their complete 150-round Ψ curves.

### 4.2 Add structured pretraining-failure evidence

For a failure before federated round 0, write an atomic artifact such as
`pretraining_failure.json` containing:

- schema version;
- run ID and effective-config checksum;
- `failure_phase: model_selection`;
- number of federated rounds started, expected to be zero;
- model-selection epochs attempted;
- best model-selection score;
- per-epoch finite/nonfinite status;
- first nonfinite epoch and component, when applicable;
- terminal reason;
- traceback;
- stdout/stderr hashes; and
- source/hash bundle identifier.

The manifest runner must preserve separate stdout and stderr logs for every
job.

Update the launcher, stage checker, validator, and screen scorer so that only a
specific validated failure artifact qualifies as
`terminal_pretraining_ineligible`. A generic return code 1 must remain an
unexplained process failure.

### 4.3 Correct misleading selection metadata

For FEMNIST/CIFAR runs, the actual checkpoint selector is pooled validation
MSE, not equal-client validation MSE. Correct the future default in
[`scripts/run_manifest.py`](../../../scripts/run_manifest.py) and add a dated
correction note for the existing screen.

The existing numerical checkpoint selection is correct, so this does not
require a training rerun.

### 4.4 Introduce compact prediction artifacts

Future V4 and final image runs should not save the complete test image tensor
in every `predictions.npz`. Use a versioned compact schema containing:

- sample IDs or a compact evaluation coordinate;
- ground truth;
- best-validation prediction;
- final prediction; and
- dataset/run metadata.

Compact copies may be derived for the existing 104 runs. Verify numerical
equality before considering removal of the approximately 10 GiB of original
prediction artifacts, and do not delete them without explicit approval.

### 4.5 Expand the protocol hash closure

The prelaunch hash closure must include:

- every execution-critical source on the active call path;
- server, client, trainer, and execution-mode code;
- model-selection and Ψ modules;
- optimizer implementations;
- active data loaders and scenarios;
- the six source dataset NPZ files;
- manifests and generated YAMLs;
- scorers, validators, preparers, and launchers;
- protocol amendments;
- Python, PyTorch, CUDA, and GPU environment metadata; and
- the Git revision plus a checksum of any intentional dirty diff.

### 4.6 Rewire V4 to the fresh diagnostic

The current V4 preparer and launchers reference the older retrospectively
certified V3 diagnostic. Change them to require the fresh post-hash diagnostic,
its certification artifact, and its exact hash bundle.

### 4.7 Build the missing post-V4 stages

Add and test:

- an `alpha=0.1` stability manifest/preparer;
- a stability validator and frozen escape-hatch implementation;
- a final three-alpha/five-seed manifest/preparer;
- resumable, fail-closed final-stage launchers;
- immutable attempt accounting; and
- final aggregation/reporting scripts.

These stages are not currently provided by the post-BN V4 machinery.

## 5. Phase 2 — Verify and freeze the implementation

Complete all of the following before launching any GPU work:

1. Run `git diff --check`.
2. Run the required `compileall` checks.
3. Run focused campaign, scorer, manifest, buffer-policy, diagnostic, and hash
   tests.
4. Run the complete test suite.
5. Run `bash -n` over every launcher.
6. Dry-run every preparer and launcher.
7. Verify unique run IDs and nonoverlapping result directories.
8. Verify every deterministic row has:

   - `batch_size=0`;
   - all 10 clients participating;
   - `sgd` for FedGDA-D or `ogda` for FedOGDA-D;
   - `server_buffer_policy=direct_client_aggregate`;
   - exact seeds, alpha values, and communication rounds;
   - validation-only selection; and
   - no reuse of v2, v3, or old-finals scientific artifacts.

9. Regenerate and independently verify all hashes.
10. Create a focused, reviewable Git checkpoint without sweeping unrelated
    dirty-worktree content into it.

Any scientific source change after this freeze invalidates the launch hashes
and requires repeating the freeze and the fresh diagnostic.

## 6. Phase 3 — Run the inexpensive certifications

### 6.1 Repeat the exact BatchNorm diagnostic

Use a new result namespace with this exact configuration:

| Field | Value |
|---|---|
| Dataset | `femnist_z` |
| Method | `fedogda_d` |
| Seed | `1` |
| Structural learning rate | `0.001` |
| Critic multiplier | `10` |
| Communication rounds | `120` |

Require:

- rounds exactly `0..119`;
- completed/nondivergent status;
- finite model state, critic output, and metrics;
- nonnegative BatchNorm running variance;
- corrected direct-buffer aggregation policy; and
- exact frozen source, config, and dataset hashes.

### 6.2 Reproduce the four failed screen rows once

Rerun these exact configurations in a dedicated certification namespace:

| Dataset | Method | Learning rate | Critic multiplier |
|---|---|---:|---:|
| FEMNIST-Z | FedGDA-D | 0.1 | 5 |
| CIFAR10-X | FedGDA-D | 0.333333 | 10 |
| FEMNIST-X | FedGDA-D | 0.333333 | 10 |
| FEMNIST-X | FedGDA-D | 0.333333 | 20 |

Do not change their hyperparameters and do not overwrite their original screen
directories.

Apply this outcome rule:

- same model-selection failure with complete evidence: classify it as
  terminal-pretraining-ineligible;
- unexpected success: stop and investigate determinism, environment, and
  launcher identity;
- different failure: keep the row unresolved and investigate;
- do not retry repeatedly; and
- do not insert a replacement learning rate.

Once resolved, the corrected screen should account for all 108 rows as:

```text
104 eligible completed runs
  4 certified terminal-pretraining-ineligible runs
```

## 7. Phase 4 — Rescore and freeze the corrected screen

### 7.1 Generate frozen screen results

Run the corrected last-50 scorer and produce `screen_results.json` containing:

- all 108 planned rows;
- the 104 eligible trajectories;
- the four certified terminal outcomes;
- exactly 12 dataset/method cells;
- Ψ rank 1 and rank 2 for every cell;
- the validation-MSE winner;
- diagnostic best-round values;
- boundary flags; and
- source, manifest, and result hashes.

### 7.2 Apply the frozen boundary rule

For every flagged axis:

- if the winning axis has never received its one permitted expansion rung,
  run exactly that additional rung;
- if the axis was already expanded once, do not recursively expand it; and
- record the resolution in a boundary-review artifact tied to the exact
  `screen_results.json` checksum.

Once its permitted expansion is complete, send the candidate to the
500-round, three-seed adjudication stage. Do not invent a new improvement
threshold after observing the corrected scores.

## 8. Phase 5 — Generate and run V4 adjudication

Generate the V4 packet only after the screen and boundary artifacts are
immutable.

For each of the 12 dataset/method cells, include the deduplicated union of:

- Ψ rank 1;
- Ψ rank 2; and
- validation-MSE winner.

Each candidate runs with:

- `alpha=0.5`;
- seeds `{0,1,2}`;
- 500 communication rounds;
- full-batch deterministic training; and
- fresh initialization.

Depending on candidate overlap, the packet will contain:

| Stage | Cells | Possible runs |
|---|---:|---:|
| Signal: Z and XZ | 8 | 48–72 |
| X | 4 | 24–36 |
| Total | 12 | 72–108 |

The exact count must come from the newly frozen packet. Do not assume the old
V3 split of 66 signal and 33 X runs will remain unchanged after correct
last-50 scoring.

### 8.1 Run signal first

Run the FEMNIST/CIFAR Z and XZ cells.

For every candidate, require all three seeds to have:

- exactly 500 ordered rounds;
- complete, config-matched artifacts;
- a finite full curve;
- no negative or nonfinite BatchNorm variance; and
- hashes matching the frozen packet.

Promotion follows the frozen decision tree:

1. A candidate is eligible only when all three seeds are complete and valid.
2. If zero candidates are eligible, the cell requires retuning.
3. If exactly one candidate is eligible, promote it directly.
4. Otherwise rank by median last-50 Ψ across seeds.
5. Apply the frozen practical-tie test.
6. Resolve a practical Ψ tie using median last-50 validation MSE.
7. Leave an exact fallback MSE tie unresolved rather than using manifest
   order.

Do not launch X if any signal cell is incomplete, requires retuning, or has an
unresolved exact tie.

### 8.2 Run X second

Only after all eight signal cells have frozen promotions, launch the four X
cells under the same acceptance and promotion rules.

At V4 completion, freeze exactly one winner for each of the 12 dataset/method
cells.

## 9. Phase 6 — Complete the full deterministic final matrix

### 9.1 Run the `alpha=0.1` stability check

Run each of the 12 frozen V4 winners at `alpha=0.1`, normally using seed 0,
for 500 rounds.

Apply the already-declared stability and constant-predictor rules:

- pass: share the `alpha=0.5`-selected configuration across alpha values;
- fail or diverge: invoke per-cell `alpha=0.1` retuning; and
- do not retune unaffected cells.

The fallback branch must be implemented before launching this stage. It must
not be designed after observing which cells fail.

### 9.2 Assemble the five-seed final evidence

The complete deterministic final evidence matrix is:

```text
6 datasets × 2 methods × 3 alpha values × 5 seeds = 180 trajectories
```

Recommended exact reuse accounting, provided it is frozen in the protocol
addendum before launch:

| Final evidence | Runs |
|---|---:|
| V4 winners at `alpha=0.5`, seeds 0–2 | 36 |
| `alpha=0.1` stability runs, seed 0 | 12 |
| New `alpha=0.1` runs, seeds 1–4 | 48 |
| New `alpha=0.5` runs, seeds 3–4 | 24 |
| New `alpha=1.0` runs, seeds 0–4 | 60 |
| **Total final evidence** | **180** |

Under this reuse rule, 132 additional final runs remain after V4 and the
stability stage, assuming no per-cell `alpha=0.1` retuning.

### 9.3 Resolve the existing run-count ambiguity

The adopted DOE lists 144 final runs in addition to 12 stability runs. The
180-run final matrix minus the 36 reusable V4 winner trajectories is 144, but
the separate 12 stability runs overlap with that set if they use the exact
final configuration.

Resolve this in the dated addendum before generating manifests:

- reuse exact stability trajectories: 132 more runs after stability; or
- deliberately rerun them: 144 more runs after stability.

The recommended policy is exact reuse because seed, alpha, configuration, and
500-round horizon are identical, and staged reuse is already part of the
adopted design.

If a cell requires `alpha=0.1` retuning, its original failed stability run does
not count as a final winner trajectory. Add the frozen per-cell tuning and
confirmation work, then run the newly selected `alpha=0.1` winner across all
required final seeds.

### 9.4 Treat deterministic final failures as results

During the final stage, do not repeatedly rerun an identical deterministic
seed. A complete terminal-divergent trajectory is a reportable stability
result, not an excuse to change the selected configuration after seeing final
or test behavior.

The only configuration-change branch is the predeclared `alpha=0.1` stability
escape hatch.

## 10. Phase 7 — Audit, report, and close

After every planned final trajectory has an auditable resolution:

1. Validate configuration identity, round counts, finiteness, BatchNorm state,
   checkpoints, predictions, and artifact checksums.
2. Unlock test metrics only after every winner is frozen.
3. Report per dataset/method/alpha/seed:

   - best-validation test MSE;
   - final test MSE;
   - last-50 Ψ;
   - last-50 validation MSE;
   - full-curve stability;
   - minimum BatchNorm running variance;
   - terminal failures or divergence; and
   - runtime/resource usage.

4. Report all five individual seeds and a cross-seed summary frozen before
   opening the final test report.
5. Generate compact tables and plots using the compact result contract.
6. Keep every pre-fix screen, final, v2, and v3 scientific trajectory marked
   legacy/ineligible.
7. Freeze final manifests, source/data hashes, result checksums, and environment
   metadata.
8. Write a closeout report that distinguishes:

   - artifact/numerical completion;
   - scientific stability;
   - terminal boundary outcomes;
   - the retained legacy Ψ definition; and
   - the transparent retrospective-provenance caveat for the 104 preserved
     screen trajectories.

## 11. Hard stop conditions

Stop the relevant stage when any of the following occurs:

- scientific source or data hashes change after the prelaunch freeze;
- the fresh 120-round BatchNorm diagnostic fails;
- any of the four pretraining reproductions produces an unexplained or
  inconsistent outcome;
- `screen_results.json` does not cover all 108 planned rows;
- a boundary expansion required by the frozen rule is missing;
- a V4 candidate lacks any one of its three confirmation seeds;
- a signal cell has no valid frozen promotion;
- a practical-tie fallback ends in an exact MSE tie;
- an `alpha=0.1` stability failure has not gone through the frozen per-cell
  retuning branch; or
- a stage contains a config/hash/artifact mismatch.

Do not repair any of these conditions by silently dropping a seed, imputing a
score, changing a learning rate, selecting with test MSE, or resuming a
pre-BatchNorm trajectory.

## 12. Definition of done

The high-dimensional deterministic campaign is fully closed only when:

- the corrected screen has 108 auditable outcomes;
- the last-50 scorer is correct and tested;
- all four pretraining failures are reproduced and classified;
- the fresh post-hash diagnostic passes;
- boundary review is mechanically resolved;
- all 12 cells have frozen V4 winners based on three valid seeds;
- the `alpha=0.1` stability branch is resolved;
- the three-alpha, five-seed final matrix has an auditable outcome for every
  planned trajectory;
- test MSE was never used for candidate or checkpoint selection;
- no pre-fix BatchNorm trajectory was mixed into corrected scientific evidence;
  and
- final reports, compact artifacts, manifests, hashes, and environment records
  are frozen.

## 13. Immediate next action

Do **not** launch V4 yet.

The next action is the integration/code-freeze batch:

1. record the legacy Ψ and five-seed protocol decisions;
2. fix and test the last-50 scorer;
3. implement structured pretraining-failure evidence and log capture;
4. correct metadata and compact future prediction outputs;
5. expand and freeze source/data hashes;
6. rewire V4 to the fresh diagnostic; and
7. implement the missing stability/final-stage machinery.

After verification and hash freeze, launch only:

1. the fresh 120-round diagnostic; and
2. the four exact failure reproductions.

Only after those five inexpensive jobs resolve should the corrected screen be
rescored and the larger V4 packet be generated.