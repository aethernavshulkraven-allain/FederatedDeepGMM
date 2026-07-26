# Study A coordinator reconciliation — 2026-07-27

## Integration update

The source-integration step described as pending in the original audit is now
complete:

- Claude's handoff-listed Study A source and tests were committed on
  `experimentsrerun` as `60f708f`.
- The protocol, full-release preflight, campaign validator, and this
  reconciliation were merged as `634f7ee`.
- The six implementation gates were independently reviewed, tested, and
  committed as `beef03c` (`Close Study A implementation gates`).
- The pre-merge base remains recoverable at
  `backup/experimentsrerun-pre-study-a-merge-20260727`.

Generated demo cohorts, scenarios, scratch runs, result directories, audit
outputs, and debug plots were preserved in place but were not added to the
source commits. Commit `beef03c` closes the implementation mismatches for seed
separation, equal-client selection/reporting, the required 105-row matrix,
paper-aligned centralized baselines, ATE/ITE metric semantics, and the
canonical campaign schema. Full-release and protocol-freeze gates documented
below remain open; source integration does not by itself make the campaign
launch-ready.

## Status

Study A is **implementation-ready but not data/protocol-ready for full-eICU
tuning or confirmatory analysis**. The three independent worktrees and
`beef03c` now implement one prelaunch-validating 105-row campaign contract.
The remaining blockers are the absent full credentialed release, the
full-cohort audit, and unresolved decisions D01-D08, D10-D15, and D18-D19.

The authoritative design is `protocol_v1.md` plus `protocol_v1.json`. The
105-row `confirmatory_matrix.csv` remains descriptive. The manifest generator
can now produce a canonical 105-row launch manifest, but a generated manifest
must not be treated as final until full-release scenarios and legitimate
validation-selected or explicitly preregistered fixed configurations exist.

No result produced from the eICU demo release is a Study A scientific result.

## Integrated clean branches

The following commits were cherry-picked, without content conflicts, onto
`codex/eicu-study-a-integration`:

| Component | Source commit | Integration commit | Disposition |
|---|---|---|---|
| Study A protocol | `eb7c0518` | `88a4e37` | Authoritative protocol specification |
| Full-release preflight | `294aa34d` | `c31e845` | Data-readiness check, not scientific certification |
| Campaign validator | `16400fe1` | `3f78b80` | Validator engine accepted; default contract requires reconciliation below |

Combined verification on the integration branch:

- 27 unit tests passed (10 preflight and 17 campaign-validator tests).
- `validate_protocol.py` passed with 105 unique required rows:
  30 primary federated, 45 centralized baseline, and 30 aggregation-ablation
  rows.
- `git diff --check` passed.

These commits were subsequently merged into the dirty shared
`experimentsrerun` worktree through merge commit `634f7ee`, without adding or
deleting the unrelated local work.

## Gate-closing implementation

The implementation below was reviewed and committed as `beef03c`:

- paper-aligned federated moment objective with a frozen previous-global
  structural iterate and lambda `1/4`;
- explicit uniform-client versus sample-size model aggregation;
- separate best-validation, best-moment-violation, and final checkpoints;
- data-driven eICU model dimensions;
- natural hospital partitions;
- scenario counterfactual arrays, coefficients, and byte checksums;
- real-instrument variation filtering and simulated first-stage diagnostics;
- a standalone per-client checkpoint evaluator; and
- resumable manifest/centralized orchestration.

Independent verification after reconciliation produced:

- 282 passing targeted eICU/gate/validator tests;
- 362 passing tests in the full suite, with one pre-existing unrelated error
  caused by missing local `results/_golden` smoke artifacts;
- two finite real-data demo federated smokes (FedGDA-S and FedOGDA-S);
- three finite real-data demo centralized smokes (GDA-D, SGDA-S, OAdam-S);
- a successful post-hoc ATE/ITE evaluation of a real checkpoint; and
- a generated 105-row manifest that passed prelaunch validation with zero
  blocking errors.

These checks establish implementation and contract behavior. They do not turn
demo outputs into scientific results or close full-release decisions.

## Original implementation mismatches and current disposition

The original P0 audit is retained below with its reconciled status.

### 1. Separate randomness domains

**Closed in `beef03c`.**

Protocol v1 requires `scenario_seed`, `optimizer_seed`, and `seed_pair_id`.
The current scenario generator, federated manifest, launcher, centralized
runner, metrics, and analysis use one `seed`. Reusing one scalar for both
randomness domains is explicitly noncompliant.

Required pairs are:

- tuning: `(11, 1011)`, `(22, 1022)`, `(33, 1033)`;
- confirmatory: `(101, 1101)` through `(105, 1105)`.

Every compared method and aggregation arm must use the same scenario artifact
for a seed pair, while initialization/minibatch/optimizer randomness comes only
from `optimizer_seed`.

### 2. Equal-client selection and reporting

**Closed in `beef03c`.**

The protocol primary metric is equal-client structural validation MSE for
checkpoint selection and equal-client structural Test MSE at that already
selected checkpoint.

The current federated and centralized loops concatenate hospital rows and
select on pooled/sample-weighted `val_mse`. Their
`test_mse_at_best_validation` fields therefore do not yet have the required
Study A semantics. The per-client evaluator cannot repair a checkpoint that
was selected with the wrong metric.

Training/evaluation must retain client IDs for every split, record both
equal-client and sample-weighted curves, select on equal-client validation
MSE with earlier-round tie-breaking, and write all fields required by
`metric_policy.md`.

### 3. Required 105-row matrix

**Closed in `beef03c`.**

The current orchestration generates:

- 30 federated confirmatory rows;
- 30 centralized rows (GDA and OAdam only); and
- 10 linear-only sample-size ablation rows.

Protocol v1 requires:

- 30 federated confirmatory rows;
- 45 centralized rows (`gda_d`, `sgda_s`, `oadam_s`); and
- 30 sample-size ablation rows across all three structural functions.

The current FedAvg guard rejects every eICU `sample_size` run. It needs an
explicit, auditable non-primary Study A ablation authorization while continuing
to reject accidental sample-size weighting in primary runs.

### 4. Centralized baselines

**Implementation closed in `beef03c`; centralized tuning fairness and final
budgets remain open under D15.**

The centralized runner is dimension-aware, but it still instantiates the
legacy `OptimalMomentObjective`, selects on pooled validation MSE, and omits
SGDA. All three centralized methods must use the frozen paper-aligned
objective/cadence, reconstruct hospital-aware validation/test metrics, and
receive a predeclared validation-only tuning budget.

### 5. Tuning policy

**Selection semantics and three-pair completeness checks are implemented;
full-data candidate spaces, budgets, and actual validation-only tuning remain
open under D12-D15.**

The current federated tuner uses one scenario/optimizer seed and six
learning-rate/server-rate candidates. It also uses:

- the best moment violation from any round instead of moment violation at the
  validation-MSE-selected checkpoint; and
- final validation MSE instead of the required final-minus-best validation
  gap.

Protocol v1 requires three disjoint tuning pairs, eligibility only when every
required pair completed finitely, selection by mean equal-client validation
MSE, then the declared tie-breaks, and an immutable selection record.
Centralized baselines also require comparable validation-only tuning.
Candidate factors, shortlist/racing policy, and budgets remain unresolved
pending the full-release runtime preflight.

### 6. Scenario provenance and eligibility

**Partially closed in `beef03c`.** Separate eligibility randomness,
non-Test first-stage certification, eligible-client IDs, scenario scope,
dimensions, provenance, and checksums are implemented. Full-release thresholds,
client flow, calibration, and mismatch-refusal policy remain open under
D01-D08 and D19.

Scenario metadata currently does not satisfy the validator contract and the
launcher does not refuse checksum mismatches. Canonical metadata must record
the full-eICU/demo scope, release/cohort/split provenance, eligible client IDs
and per-split counts, label mapping for `frozen_random_mlp`, dimensions,
scenario/g0 seeds, and checksums.

Eligibility must be frozen before scenario generation. The current real-Z
filter is rerun with the scenario seed and records only counts, so the included
hospital set can vary silently across seeds. Per-client first-stage gates and
failure behavior must be frozen from non-Test diagnostics before launch.

### 7. ATE and individual-effect metrics

**Closed in `beef03c`.**

The current post-hoc field named `ate_error_abs` is the mean absolute
individual-effect error. That is individual-effect MAE, not absolute ATE
error. Both must be computed and stored separately according to
`metric_policy.md`, together with true/predicted client ATEs.

### 8. Full-data runtime assumptions

**Open.** Demo timing is not a substitute for a full-release runtime audit.

The current manifest sets stochastic `batch_size` from the number of clients
(three on the demo) and marks all jobs CPU-only based on demo runtime. Batch
size is a row/minibatch quantity and must be chosen from the frozen
full-release tuning/runtime policy. Demo timing cannot determine full-eICU
hardware or round budgets.

### 9. Campaign contract and artifact layout

**Closed in `beef03c`.**

The campaign validator's engine is useful, but its shipped default contract
uses one `seed`, `mlp`, `gda`/`sgda`/`oadam`, and root-level checkpoint names.
Protocol v1 uses separate seeds, `frozen_random_mlp`,
`gda_d`/`sgda_s`/`oadam_s`, and the implementation writes checkpoints under
`checkpoints/`.

The manifest generator, effective configs, scenario metadata, metrics, result
paths, artifact filenames, protocol contract, and validator must adopt one
canonical schema. A prelaunch validation must pass before any required run is
started.

### 10. Reproducible source state

The gate-closing Study A implementation is committed as `beef03c`. Future
effective configurations must record the actual later launch commit, including
the full-data freeze records, rather than treating either `60f708f` or
`beef03c` as a scientific launch snapshot. Generated scenarios and runtime
artifacts remain outside the source commit and must be identified by their own
checksums.

## Demo campaign audit

Claude's scratch directory currently contains 96 finite `metrics.json` files:

- 36 tuning runs;
- 30 federated confirmatory runs; and
- 30 centralized runs.

There are zero recorded divergences. These are useful execution smokes only.
They use the demo release, three retained hospitals with extremely small
splits, a single conflated seed field, pooled checkpoint metrics, two rather
than three centralized methods, and the older linear-only ablation design.
The 10 generated ablation rows were not run.

The scratch confirmatory manifest fails the current campaign validator
prelaunch check. The scratch artifacts must not be presented as a completed
Study A v1 campaign or used for scientific comparisons.

## Full-release gate

`preflight_eicu_release.py` checks table presence, headers, readability, and
optional checksums. Its `likely_full` classification is a readiness heuristic:
a conventionally named `eicu-crd` path can be accepted without a streamed row
count. Passing it does not close protocol decisions D01-D08 or certify the
cohort, instrument, or scenario.

For the full launch gate, preserve the preflight report and additionally freeze
the exact release/version, inventory and checksums, cohort flow, cohort
checksum, client eligibility report and IDs, split definition/checksum,
instrument specification, DGP calibration, and scenario checksums.

## Required execution order

1. ~~Commit/isolate Claude's implementation and merge it with the clean
   integration branch.~~ Completed by `60f708f` and `634f7ee`.
2. ~~Implement the P0 corrections above and add protocol-level contract
   tests.~~ Completed by `beef03c`.
3. Run the full-release preflight with row counting and checksums.
4. Build the full cohort and conduct a blinded flow/client/instrument audit.
5. Freeze eligibility, DGP, structural-function parameters, scenario metadata,
   architectures, participation, budgets, and artifact policy.
6. Generate and checksum three tuning scenario pairs per structural function.
7. Run demo/full-data execution smokes that do not inspect confirmatory Test
   outcomes.
8. Run validation-only tuning for each `(g0, method)` and freeze immutable
   selections.
9. Generate the exact 105-row manifest and require a clean prelaunch campaign
   validation.
10. Run the 30 primary federated, 45 centralized, and 30 aggregation-ablation
    jobs without replacing unfavorable seeds.
11. Materialize per-client evaluations, curves, checkpoints, effective
    configs, checksums, failures, and resource summaries.
12. Require a clean postrun validation before opening the confirmatory summary.
13. Produce the descriptive and paired analyses specified in protocol v1.

Optional deterministic sensitivity adds 30 runs only after the required
campaign; it cannot replace any of the 105 required rows.
