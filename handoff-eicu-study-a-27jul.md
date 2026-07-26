# Handoff: eICU Federated-IV Extension (Study A) — 27 Jul 2026

**Status: infrastructure complete, tested, and run end-to-end on real data.**
272 new/updated tests, all green. 60/60 real training runs completed (30 federated
confirmatory + 30 centralized baselines), zero divergence. **The one thing not done
is Study B (real outcomes)** — blocked on full eICU access, not on code. Everything
below is a precise map of what exists, why, and how to pick it up.

---

## 1. The one-sentence version

FedDeepGMM's paper only tests synthetic Dirichlet-partitioned data. This extension
adds a real-world federated-IV benchmark on eICU (hospitals as clients, cross-fitted
ward-preference as the instrument), split into **Study A** (semi-synthetic: real
partitions/covariates, simulated causal layer, so structural recovery is checkable
against a known ground truth) and **Study B** (real outcomes — not yet buildable,
see §7). Study A's full pipeline — cohort → instrument → scenario → paper-aligned
federated training → checkpointing → tuning → confirmatory aggregation — is built,
tested, and has been run for real on the eICU demo dataset.

**The demo dataset is too small for a reportable result.** After the correct
client-relevance filter, only **3 of 89 demo hospitals** have genuine ward-preference
variation (`experiments/eicu_v1_demo/construction_decision.json`:
`"construction": "insufficient_data"`, `n_ward_eligible_hospitals: 0` for a real
causal estimate; 3 for the semi-synthetic study). Every number this pipeline has
produced so far is a **pipeline-correctness result**, not a methods result. Full
eICU-CRD v2.0 (credentialed PhysioNet access, ~200k stays vs. the demo's 2,520) is
the actual prerequisite for a reportable number, and nothing else blocks it.

---

## 2. Everything that exists, by file

### 2.1 New scripts (`scripts/`) — 3,689 lines total

| File | Lines | Role |
|---|---|---|
| `eicu_common.py` | 340 | Shared IO (case-insensitive table resolution across eICU mirrors), vasopressor name normalization (`Norepinephrine (mcg/min)` → `norepinephrine`, order-sensitive so `epinephrine` doesn't false-match inside `norepinephrine`), age top-coding (`'> 89'` → 90.0), mortality parsing |
| `prepare_eicu_cohort.py` | 551 | Cohort ETL from raw eICU tables → `cohort.csv` + `cohort_flow.json`. Sepsis-at-admission (not vasopressor-defined shock), treatment = vasopressor infusion in `[0,360]` min of ICU admission, outcome = hospital mortality |
| `eicu_instrument.py` | 329 | Cross-fitted, Beta-Binomial-shrunk ward-preference instrument. This is the fix for the identification bug in the original proposal (see §4) |
| `eicu_iv_diagnostics.py` | 215 | Hand-rolled OLS/2SLS/partial-F/SMD balance (no `statsmodels`/`linearmodels` in this env) |
| `audit_eicu_clients.py` | 413 | Stage-1 feasibility **gate**: pre-registered eligibility thresholds, decides `ward` vs `grouped` vs `insufficient_data` construction *before* any effect is computed |
| `analyze_eicu_iv_diagnostics.py` | 262 | Relevance/overlap/balance report on a built cohort |
| `prepare_eicu_semisynth.py` | 531 | **Study A scenario generator.** Real X/partitions/instrument, simulated confounded D/Y from a known g0 (linear/interaction/frozen-MLP). Filters clients by real Z variation, certifies simulated first-stage per client, stores equal-client/sample-weighted true ATE, per-client true effects, full simulator coefficients, scenario checksum |
| `analyze_eicu_study_a_checkpoint.py` | 258 | **Standalone post-hoc per-client evaluator.** Loads a checkpoint + scenario `.npz` directly (bypasses the FedML data pipeline, which drops `client_id` before eval) — reports per-client MSE/ATE-error/moment-violation, equal-client vs. sample-weighted aggregates |
| `prepare_eicu_study_a_manifest.py` | 284 | Generates the tuning (36 rows) / confirmatory (30 rows) / ablation (10 rows) manifests, in the schema `run_manifest.py` consumes |
| `select_eicu_study_a_tuning.py` | 147 | Applies the frozen tuning-selection rule: no-divergence → lowest val MSE → val moment-violation tiebreak → final-iterate val MSE tiebreak |
| `analyze_eicu_study_a_confirmatory.py` | 225 | Aggregates confirmatory results: paired FedOGDA−FedGDA differences, best-vs-final degradation, per-g0 summaries |
| `run_eicu_centralized_baselines.py` | 134 | Resumable loop over the centralized GDA/OAdam grid (skips any run with an existing `metrics.json`) |
| `audit_aggregation_weighting.py` | ~190 | Labels every existing `results/` run as `legacy_sample_weighted` / `sample_size_explicit` / `uniform_clients` — read-only, reruns nothing |

### 2.2 Modified core repo files

| File | What changed |
|---|---|
| `fedml/simulation/sp/fedavg/fedavg_api.py` | `aggregation_weighting`/`objective_mode` config + validation + eICU enforcement; `_aggregate`/`_aggregate_reg` now go through shared, tested `compute_client_weights`/`weighted_average_state_dicts`; second checkpoint track (`best_moment_violation.pt`); `theta~` snapshot wired at eval time |
| `game_objectives/simple_moment_objective.py` | New `PaperAlignedMomentObjective` class (legacy `OptimalMomentObjective` untouched) |
| `fedml/ml/trainer/my_model_trainer_classification.py` | `set_theta_tilde(g)` called once per round, per client, before local training starts |
| `fedml/data/MNIST/data_loader.py` | `load_data_natural()` — partitions by real `hospitalid`, all three splits keyed to the same client (unlike Dirichlet, which draws them independently) |
| `fedml/data/data_loader.py` | Registers `eicu*` datasets in the zoo-style dispatch |
| `fedml/model/model_hub.py` | Data-driven `input_dim_g`/`input_dim_f` for `eicu*` (was hardcoded 1/2), `[64,64]` widths |
| `scenarios/abstract_scenario.py` | Optional `client_id` on `Dataset`, backwards compatible (legacy `.npz` files load with `client_id=None`) |
| `experiment_utils.py` | `compute_client_weights`, `weighted_average_state_dicts`, `moment_violation`, `AGGREGATION_WEIGHTING_CHOICES`, `OBJECTIVE_MODE_CHOICES`, `input_dim_g/f` added to `EFFECTIVE_CONFIG_FIELDS`, new `aggregation_weights_by_round.csv` writer |
| `scripts/run_manifest.py` | 5 new optional CSV columns (`scenario_name`, `objective_mode`, `aggregation_weighting`, `input_dim_g`, `input_dim_f`) — pure additions, defaults preserve every existing manifest's behavior unchanged; `using_gpu` now row-overridable (was hardcoded `True`) |
| `scripts/run_centralized_lowdim.py` | `eicu_semisynth` dataset support, data-driven dims/widths (was hardcoded `input_dim=1`/`2`, `[20,20]`) |
| `main.py` | CPU thread cap (`OMP_NUM_THREADS=4` etc. — a shared-machine fix, not eICU-specific) |

### 2.3 Tests (`tests/`) — 202 eICU-specific, all passing

| File | Tests | Protects |
|---|---|---|
| `test_eicu_cohort.py` | 32 | Cohort ETL against a hand-built mini eICU release |
| `test_eicu_instrument.py` | 20 | Cross-fitting has no leakage; naive leave-one-out is provably broken (own-D recoverable) |
| `test_eicu_iv_diagnostics.py` | 19 | OLS/2SLS checked against designs with a known planted coefficient |
| `test_eicu_client_audit.py` | 15 | Gate picks correct construction; **explicitly asserts the demo cohort is refused** |
| `test_eicu_semisynth.py` | 43 | Splits never cross admissions, confounding is real, IV beats OLS, client filtering, ATE certification |
| `test_aggregation_weighting.py` | 20 | `sample_size` is bit-exact vs. the original inline formula; unequal-size known-answer test |
| `test_paper_aligned_objective.py` | 12 | **Closed-form gradient check** against hand-derived math + empirical descent/ascent verification |
| `test_analyze_eicu_study_a_checkpoint.py` | 12 | Per-client eval script, end-to-end with a hand-built checkpoint |
| `test_run_centralized_lowdim_eicu.py` | 7 | Data-driven dims, zoo backward-compat |
| `test_eicu_study_a_orchestration.py` | 22 | Manifest generation, tuning selection, confirmatory aggregation — **including a regression test for a real key-naming bug found while integrating** |

Run everything: `python -m unittest <module list above>` (stdlib `unittest`, not pytest — not
installed in the `fedgmm` env). Run `tests.test_certify_synthetic_data` in its own process;
it has a pre-existing (not mine) `os.chdir`-at-import leak that breaks a later test if
combined in one run.

### 2.4 Data/artifact locations

- `experiments/eicu_v1_demo/` — cohort, audit, IV diagnostics for the demo release, plus
  `EICU_EXTENSION_VERIFICATION.md` (an earlier, narrower verification doc — this handoff supersedes it for anything that conflicts)
- `fedgmm/sp_decentralized_mnist_lr_example/data/eicu_semisynth/*.npz` + `*_metadata.json` —
  15 generated scenarios (3 g0 variants × 5 seeds)
- `experiments/aggregation_weighting_audit/` — labels all 953 pre-existing `results/` runs
  `legacy_sample_weighted` (this fix predates them)

**Everything above is currently untracked in git** (worktree convention per `AGENTS.md`:
local uncommitted experiment work). Nothing has been committed.

---

## 3. The pipeline, exact commands, in order

```bash
P=/home/arnav22103/miniconda3/envs/fedgmm/bin/python

# 1. Cohort ETL (real eICU data only)
$P scripts/prepare_eicu_cohort.py --eicu-root physionet.org/files/eicu-crd-demo/2.0.1 \
    --out experiments/eicu_v1_demo

# 2. Feasibility gate (real data; freezes ward vs. grouped vs. insufficient_data)
$P scripts/audit_eicu_clients.py --cohort experiments/eicu_v1_demo/cohort.csv

# 3. Semi-synthetic scenario (Study A) -- repeat per g0 x seed
$P scripts/prepare_eicu_semisynth.py --cohort experiments/eicu_v1_demo/cohort.csv \
    --g0 linear --seed 0            # g0 in {linear, interaction, mlp}

# 4. Tuning manifest -> launch -> select (frozen hyperparameters)
$P scripts/prepare_eicu_study_a_manifest.py --stage tuning \
    --out experiments/eicu_study_a/tuning_manifest.csv --output-root results/eicu_study_a
$P scripts/run_manifest.py --manifest experiments/eicu_study_a/tuning_manifest.csv \
    --config-dir experiments/eicu_study_a/tuning_configs \
    --output-root results/eicu_study_a --gpu-ids 0 --max-parallel 1 \
    --skip-model-selection --resume-skip-completed --keep-going --overwrite-incomplete
$P scripts/select_eicu_study_a_tuning.py \
    --manifest experiments/eicu_study_a/tuning_manifest.json \
    --out experiments/eicu_study_a/selected_hyperparameters.json

# 5. Confirmatory manifest -> launch -> aggregate
$P scripts/prepare_eicu_study_a_manifest.py --stage confirmatory \
    --out experiments/eicu_study_a/confirmatory_manifest.csv --output-root results/eicu_study_a \
    --selected-hyperparameters experiments/eicu_study_a/selected_hyperparameters.json
$P scripts/run_manifest.py --manifest experiments/eicu_study_a/confirmatory_manifest.csv \
    --config-dir experiments/eicu_study_a/confirmatory_configs \
    --output-root results/eicu_study_a --gpu-ids 0 --max-parallel 1 \
    --skip-model-selection --resume-skip-completed --keep-going --overwrite-incomplete
$P scripts/analyze_eicu_study_a_confirmatory.py \
    --manifest experiments/eicu_study_a/confirmatory_manifest.json \
    --out experiments/eicu_study_a/confirmatory_report

# 6. Centralized baselines (independent code path, own resumable runner)
$P scripts/run_eicu_centralized_baselines.py --output-root results/eicu_study_a/centralized

# 7. Ablation (aggregation weighting; linear g0 only, sample_size arm)
$P scripts/prepare_eicu_study_a_manifest.py --stage ablation \
    --out experiments/eicu_study_a/ablation_manifest.csv --output-root results/eicu_study_a \
    --selected-hyperparameters experiments/eicu_study_a/selected_hyperparameters.json
$P scripts/run_manifest.py --manifest experiments/eicu_study_a/ablation_manifest.csv \
    --config-dir experiments/eicu_study_a/ablation_configs \
    --output-root results/eicu_study_a --gpu-ids 0 --max-parallel 1 \
    --skip-model-selection --resume-skip-completed --keep-going --overwrite-incomplete

# 8. Post-hoc per-client evaluation of any specific checkpoint
$P scripts/analyze_eicu_study_a_checkpoint.py \
    --checkpoint <run_dir>/checkpoints/best_validation.pt \
    --scenario fedgmm/sp_decentralized_mnist_lr_example/data/eicu_semisynth/linear_seed0.npz \
    --split test
```

**Two flags that matter and are easy to forget** (both bit me during integration —
see §6): `--keep-going` (default is stop-on-first-failure, which aborts *every*
remaining job in the manifest, not just the failed one) and `--overwrite-incomplete`
(a run killed mid-training leaves partial artifacts with no `metrics.json`; without
this flag `run_manifest.py` refuses to retry it).

**`--cf`/`--output-dir` paths into `main.py`/`run_centralized_lowdim.py` must be
absolute** — both scripts `chdir` into their own directory at import.

---

## 4. Why the design looks the way it does

### 4.1 The instrument had to be replaced, not tuned

The originally proposed instrument was leave-one-out hospital preference,
`Z_hi = (S_h - D_hi)/(n_h-1)`. With client = hospital, this is an affine function
of the patient's own treatment (`D_hi = S_h - (n_h-1)Z_hi`) — it has ≤2 distinct
values per hospital and zero real within-client variation. `eicu_instrument.py`
replaces it with cross-fitted (5-fold), Beta-Binomial-shrunk **ward**-preference:
a patient's ward's early-treatment rate, estimated from *other patients in other
folds*, never from the patient's own outcome. Verified directly:
`test_own_treatment_never_enters_own_instrument` (flipping a patient's D doesn't
move their own Z) and `test_leave_one_out_is_recoverable_from_own_treatment`
(the naive construction *is* mechanically the patient's own D, proving the bug).

A subtlety found mid-build: cross-fitting itself induces patient-level Z spread
that is pure fold noise (different folds hold out different patients), not real
practice variation. The audit measures **structural** (between-ward) variation,
not raw within-client spread — this cut "clients with variation" on the demo from
19 (naive) to 3 (correct); see `FoldNoiseTest` in `test_eicu_instrument.py`.

### 4.2 The federated objective — three separate, load-bearing fixes

Your co-author's review of the audit surfaced that the *pre-existing* repo code
(not something this extension introduced, but something Study A's premise required
fixing) did not implement the paper's actual objective:

**(a) Aggregation was sample-size-weighted, not equal-per-client.**
`fedavg_api.py`'s `_aggregate` used `w = local_sample_number/training_num` — ordinary
FedAvg — contradicting the paper's `U = (1/N) Σ U^i`. Fixed via `aggregation_weighting:
uniform_clients | sample_size`, defaulting to `sample_size` (so all 953 pre-existing
results stay reproducible — verified bit-exact against the original formula, not just
"close"). eICU datasets *require* `uniform_clients` (hard `ValueError` otherwise).

**(b) θ̃ (the regularizer's reference point) was never frozen.** The paper's client
objective uses `g_θ̃` — the **previous global iterate**, fixed for the whole round —
in the quadratic regularizer, while `g_θ` (live, being optimized) appears in the
raw moment term. The existing code reused the live model for both. `PaperAlignedMomentObjective.set_theta_tilde()` snapshots g once per round (called from
`train_gmm`, right when the client receives `g_global`, before any local steps).

**(c) λ was 0.1 (tunable), not the paper's fixed 1/4.**

All three are additive: `objective_mode: legacy` (default) reproduces the exact
pre-existing behavior; `objective_mode: paper_aligned` implements the corrected
formula. The sign convention (`epsilon = Y - g(X)`, not `g(X) - Y`) was re-derived
by hand and checked against closed-form gradients on a toy linear model
(`test_paper_aligned_objective.py::ClosedFormGradientTest`) — this is the one place
in the whole extension where "looks like it trains" would not have been sufficient
evidence of correctness.

### 4.3 Client relevance is checked twice, not once

Before simulating anything: does the *real* ward-preference Z vary structurally
within this hospital (`filter_clients_by_real_z_variation`, using the cohort's real
treatment column)? After simulating D: does Z predict the *simulated* D within each
hospital (`certify_simulated_first_stage`)? A real-data F-statistic computed before
D exists is not sufficient on its own — both checks are load-bearing and both are
in `prepare_eicu_semisynth.py::generate()`, applied automatically, not optional.

### 4.4 Two checkpoints because there is no ground truth on real data

`best_validation.pt` (lowest structural MSE against known g0 — only possible
because Study A simulates g0) and `best_moment_violation.pt` (lowest
`‖E_val[f(Z,X)(Y-g(D,X))]‖²` — works even without ground truth, so it's the one
metric Study B could also use) are tracked independently. On the real confirmatory
runs they picked **different rounds** in every single run — proof they're not
redundant, not just a defensive design choice.

### 4.5 Per-client evaluation deliberately bypasses the training pipeline

`FedAvgAPI`'s global train/val/test objects are built via `_wrap_global`/
`combine_batches` from a 5-tuple `(g,w,x,y,z)` — `client_id` is dropped before it
ever reaches evaluation. Rather than thread `client_id` through that shared,
heavily-used pipeline (risk to every other experiment family in the repo),
`analyze_eicu_study_a_checkpoint.py` loads a checkpoint + the scenario `.npz`
directly and evaluates outside the training loop entirely.

---

## 5. Real bugs found by actually running things, not just unit tests

Every one of these was caught by running the full pipeline end-to-end, not by unit
tests alone — worth internalizing as the reason the "run it for real" step mattered:

1. **θ̃ crash at eval time.** `FedAvgAPI.model_trainer` is a `copy.deepcopy` per client
   (`_setup_clients`); the eval-only instance's objective never got `set_theta_tilde`
   called on it. Fixed at the `eval_global_model` call site directly.
2. **Key-naming mismatch**: `select_eicu_study_a_tuning.py` groups by the manifest's
   full `method` column (`fedgda_s`), but `prepare_eicu_study_a_manifest.py` looped
   over bare names (`fedgda`) when building lookup keys for the *next* stage. Silent
   `KeyError` on every real confirmatory run. Now a shared `selection_key()` helper,
   plus `CrossScriptKeyConventionTest` so it can't silently reappear.
3. **Demo dimensionality isn't fixed across seeds.** `n_features_x` varies 42↔43
   across seeds (which categorical levels appear depends on which few rows land in
   a 3-9-patient training split). The manifest generator reads dims per-`(g0,seed)`
   scenario metadata, never assumes one global width.
4. **`run_manifest.py`'s stop-on-first-failure default.** A stale, killed-mid-training
   run (no `metrics.json`) caused `run_manifest.py` to mark *all 27 remaining jobs*
   failed without attempting them — not just the one bad row. `--keep-going` +
   `--overwrite-incomplete` fixes this; see §3.
5. **Unbounded CPU threading on a shared machine.** Not eICU-specific, but hit during
   this work: PyTorch defaults to one BLAS thread per core; a tiny-MLP training
   process was measured grabbing ~29 cores for no speedup. `main.py` and
   `run_centralized_lowdim.py` now cap `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/
   `OPENBLAS_NUM_THREADS` to 4 before `numpy`/`torch` import.

---

## 6. Results obtained so far (demo data — pipeline verification, not a finding)

**60/60 real training runs completed, zero divergence**: 30 confirmatory (3 g0 ×
5 seeds × {FedGDA, FedOGDA}) + 30 centralized (3 g0 × 5 seeds × {GDA, OAdam}).
Tuning: 36/36 candidates completed cleanly, 6 frozen hyperparameter sets selected.

Aggregated report: `experiments/eicu_study_a/confirmatory_report/README.md` (path
depends on where you point `--out`; this session's run is under the session
scratchpad and was not persisted into the repo — regenerate via §3 step 5).

Qualitative shape worth re-checking on real data, **not a claim**: `final_vs_best_degradation`
was negative for FedOGDA in all three g0 variants (final iterate better than its own
best-validation checkpoint) and positive for FedGDA in 2 of 3 — the direction the
paper's optimistic-correction argument predicts. Test MSEs (100–780, std larger than
the mean) are noise, consistent with ~9 total patients across 3 usable hospitals.

---

## 7. What's not done

- **Study B** (real outcomes) has no scenario builder, no ground-truth-free
  checkpoint-selection wiring beyond what already exists (`best_moment_violation`
  *is* ground-truth-free, so less new work is needed here than originally scoped —
  but real-Y cohort assembly, the `min_deaths`-style Study-B eligibility gate, and
  a Study-B-specific per-client report are all unbuilt).
- **Full eICU-CRD v2.0 integration** — untested at scale; `eicu_common.py`'s
  case-insensitive table resolution and `iter_table_chunks` streaming were written
  to *not* assume demo-scale data, but no full-release run has happened.
- **Statistical significance** on the FedGDA-vs-FedOGDA comparison — deliberately
  not attempted (5 seeds; the report emphasizes effect size and per-seed consistency
  by design, per the frozen protocol).
- **Aggregation ablation stage** — manifest generator exists and is tested, but the
  actual 10 ablation runs (linear g0, sample_size arm) were not launched this
  session (deprioritized behind confirmatory + centralized, which were explicitly
  requested).
- Stray debug artifacts from early ad-hoc smoke runs (`csv/*eicu_semisynthnewtrial.csv`,
  `plots/aaaa_eicu_semisynth_smoke_*.png*` under `fedgmm/sp_decentralized_mnist_lr_example/`)
  are harmless but uncommitted clutter — fine to delete.

## 8. Immediate next steps, in priority order

1. Decide on full eICU-CRD access — this is the actual blocker, not more code.
2. If proceeding on the demo anyway: run the ablation stage (step 7 in §3) and
   fold aggregation-weighting comparison into the confirmatory report.
3. If Study B becomes reachable: build its cohort/scenario path reusing
   `prepare_eicu_cohort.py` (already real-outcome-shaped) and the already-built
   `best_moment_violation` checkpoint as the primary selection criterion.
