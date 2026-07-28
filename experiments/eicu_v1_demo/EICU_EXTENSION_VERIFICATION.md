# eICU Federated-IV Extension — Verification Document

**Purpose of this document:** a complete, start-to-finish account of the eICU
extension, written so you can check every claim against your paper
(`paper_ctxt.md`) and against the code, and decide whether to sign off on running
training. Every number below was regenerated on 2026-07-26 from the artifacts
currently on disk — nothing here is remembered or assumed.

---

## 1. What your paper actually says (for reference)

From `paper_ctxt.md`, Section 2, the paper's own notation:

> Consider a set of N clients. The treatment X and outcome Y are related by
> `Y^i = g_0^i(X^i) + eps^i`. The treatment X^i is endogenous
> (`E[eps^i | X^i] != 0`), and is influenced by an instrument Z^i satisfying
> relevance and exogeneity: `E[eps^i | Z^i] = 0`.

Client-local objective (Section 2, "zero-sum game for centralized DEEPGMM"):

```
U^i(theta, tau) = (1/n_i) sum_k f(Z_k^i, tau) * (Y_k^i - g(X_k^i; theta))
                  - (1/4n_i) sum_k f(Z_k^i, tau)^2 * (Y_k^i - g(X_k^i; theta~))^2
```

Federated objective (Section 3): **equal-client average**, `U(theta,tau) = (1/N) sum_i U^i(theta,tau)`.

Experiments (Section 5): low-dim synthetic (`abs`, `step`, `linear`) and high-dim
(FEMNIST/CIFAR pixels wrapped in a *synthetic* IV design), partitioned with a
Dirichlet distribution — i.e. **artificial** non-i.i.d.-ness, not real
distributional heterogeneity. FEDGDA/FEDOGDA compared on convergence stability.

**The gap this extension addresses:** nothing in the paper uses a instrument or a
client partition that comes from real, uncurated data. `U^i` here is exactly
`OptimalMomentObjective.calc_objective` in the code
(`game_objectives/simple_moment_objective.py:97-111`) — confirmed line-for-line
below.

---

## 2. The original proposal's flaw, and why it matters for your paper's assumptions

Your paper's identification requires, per client `i`: **relevance**
(`Z^i` predicts `X^i`) and **exogeneity** (`E[eps^i | Z^i] = 0`), both holding
*locally*, because the federated objective averages client-local moments.

The originally proposed instrument was leave-one-out hospital preference, with
client = hospital:

```
Z_hi = (S_h - D_hi) / (n_h - 1),   S_h = sum_j D_hj
```

This fails relevance **by construction**: since `D_hi = S_h - (n_h-1) Z_hi`, the
instrument is an affine function of the patient's own treatment. It has at most
two distinct values inside one hospital and zero genuine within-client variation.
This is not a data problem — it would be false on the full eICU release too.

**Verified in code** — `tests/test_eicu_instrument.py`:
- `test_leave_one_out_is_recoverable_from_own_treatment` — reconstructs each
  patient's `D` exactly from their own naive `Z`. Passes.
- `test_leave_one_out_hospital_preference_has_two_values_per_hospital` — confirms
  ≤2 distinct values per hospital. Passes.

**The fix:** keep client = hospital, but instrument on **cross-fitted ward
preference** — a patient's ward's early-treatment rate, estimated from *other
patients in other cross-fitting folds*, never from the patient's own outcome.
This has real within-client (within-hospital) variation because different wards
in the same hospital practice differently.

**Verified in code**:
- `test_own_treatment_never_enters_own_instrument` — flipping one patient's
  treatment does not change that patient's own Z. Passes.
- `test_flipping_a_patient_moves_other_folds` — sanity check that the instrument
  is not simply constant (it *should* respond to other patients' treatment).
  Passes.

This directly restores the relevance condition your paper's `U^i` needs to be
meaningful at the client level.

---

## 3. Data status: demo vs. full eICU (read this before anything else)

The eICU data currently on disk is the **demo release v2.0.1**
(`physionet.org/files/eicu-crd-demo/2.0.1/`, 131 MB, all 32 files
SHA256-verified). It is **not** a representative 20-hospital sample — it is a
thin stratified sample across 186 hospitals.

| Quantity | Value |
|---|---|
| Total ICU stays in demo | 2,520 |
| Distinct hospitals | 186 |
| Final cohort after all inclusion/exclusion (Section 4) | **201** |
| Hospitals holding the cohort | 89 |
| Patients per hospital (median / max) | 2 / 6 |
| Hospitals meeting the pre-registered eligibility gate (Section 5) | **0** |

**This means: no real causal estimate can be produced from this data, on
either instrument construction (ward or grouped-hospital fallback).** This is
verified programmatically, not eyeballed — see Section 5.

Full eICU-CRD v2.0 (credentialed PhysioNet access, ~200k stays, 208 hospitals)
is a hard prerequisite for a real result. All code below runs unchanged on it
via `--eicu-root`; nothing is demo-specific except the numbers.

---

## 4. Step-by-step pipeline (what runs, in order, with real output)

### Step 1 — Cohort extraction from raw eICU tables

```bash
python scripts/prepare_eicu_cohort.py \
    --eicu-root physionet.org/files/eicu-crd-demo/2.0.1 \
    --out experiments/eicu_v1_demo
```

Cohort definition (all choices documented with rationale in the script's
docstring):
- Adults (age ≥ 18; eICU's `'> 89'` top-code mapped to 90, not dropped)
- First ICU stay of the hospital admission (`unitvisitnumber == 1`, de-duplicated
  by `patienthealthsystemstayid`)
- **Sepsis at/near ICU admission** (ICD-9 codes + text match on `diagnosisstring`
  within a `[-360, +360]` minute window around admission, OR sepsis admission
  diagnosis) — deliberately **not** vasopressor-defined septic shock, which would
  condition cohort membership on the treatment itself
- Known hospital discharge status (unknown-mortality rows excluded, not
  imputed as survival)
- Hospital has a working infusion interface (verified against the hospital's
  *entire* patient population, not just the cohort, so a hospital isn't dropped
  merely because no cohort member happened to be on an infusion)
- No vasopressor before ICU admission (excluded from the primary cohort; kept as
  a documented sensitivity option `--keep-pre-icu-vasopressors`)

**Treatment (D):** qualifying vasopressor infusion (norepinephrine, epinephrine,
vasopressin, phenylephrine; dopamine excluded by default, sensitivity-only)
starting within **0–360 minutes of ICU admission** — relative to admission, not
to `diagnosisOffset`, because `diagnosisOffset` is a charting timestamp, not a
biological onset time.

**Outcome (Y):** `hospitaldischargestatus == 'Expired'` — hospital mortality,
not ICU mortality, since patients can leave the ICU and later die in the same
admission.

**Actual cohort flow** (`experiments/eicu_v1_demo/cohort_flow.json`, regenerated
2026-07-26):

```
all ICU stays                                        2520
adults (age >= 18)                                    2512   (-8)
first unit visit (unitvisitnumber == 1)               2111   (-401)
first ICU stay per hospital admission                 2111   (-0)
known hospital discharge status                       2086   (-25)
known hospital and ward id                            2086   (-0)
sepsis at / near ICU admission                          304   (-1782)
hospital has a working infusion interface               205   (-99)
no vasopressor before ICU admission                     201   (-4)
final cohort                                            201
```

**What to check yourself:** open `experiments/eicu_v1_demo/cohort_flow.json` and
`cohort.csv` (201 rows) and confirm the counts reconcile. Every row here is
100% real eICU data — nothing simulated at this stage.

### Step 2 — Client feasibility audit (the gate)

```bash
python scripts/audit_eicu_clients.py --cohort experiments/eicu_v1_demo/cohort.csv
```

Applies **pre-registered** eligibility thresholds — frozen before any effect is
looked at, so the construction choice cannot be reverse-engineered from a
result: `n_k >= 200` patients, `>=20` treated, `>=20` untreated, `>=20` deaths,
`>=2` wards with `>=50` patients each.

Two candidate constructions are probed:
1. **Ward** (primary): client = hospital, instrument = cross-fitted ward preference
2. **Grouped** (fallback): client = hospitals grouped by
   region × teaching status × bed category, instrument = cross-fitted hospital
   preference

A methodological correction found *while building this*: naive within-client
instrument spread is contaminated by **cross-fitting fold noise** — even a
hospital with a single ward shows nonzero patient-level Z variation, purely
because different folds hold out different patients. The audit measures
**structural** (between-ward, or between-hospital-in-group) variation instead,
which correctly reports zero for a single-ward hospital. This cut the
"ward clients with variation" count on the demo from 19 (naive) to 3 (correct) —
see `tests/test_eicu_instrument.py::FoldNoiseTest`.

**Actual result** (`experiments/eicu_v1_demo/construction_decision.json`,
regenerated 2026-07-26):

```json
{
  "construction": "insufficient_data",
  "n_ward_eligible_hospitals": 0,
  "n_eligible_groups": 0,
  "n_ward_clients_with_variation": 3,
  "n_group_clients_with_variation": 14,
  "reasons": [
    "ward-eligible hospitals: 0 (need >= 5); eligible hospital groups: 0 (need >= 5)",
    "no construction has enough clients with within-client instrument variation; this release cannot support the federated IV analysis"
  ]
}
```

**This is the load-bearing fact for signoff**: on demo data, **no real causal
estimate is possible**, under either construction. This is a data-size fact
about the demo release, verified by code, not a limitation of the method.

**What to check yourself**: `experiments/eicu_v1_demo/client_audit.csv` (per-
hospital detail) and `client_audit.md` (human-readable version of the same
decision).

### Step 3 — Instrument construction (used inside Step 4)

`scripts/eicu_instrument.py` implements the cross-fitted, Beta-Binomial-shrunk
preference estimator described in Section 2. Key guarantees, each backed by a
test in `tests/test_eicu_instrument.py` (17 tests):

| Guarantee | Test |
|---|---|
| Own treatment never enters own instrument | `test_own_treatment_never_enters_own_instrument` |
| Validation/test rows scored from training split only, never their own outcome | `test_validation_rows_are_scored_from_training_only` |
| Small wards shrunk toward hospital rate (not reporting exactly 0 or 1) | `test_tiny_ward_is_pulled_toward_hospital_rate` |
| Naive leave-one-out construction is recoverable from own D (the bug this replaces) | `test_leave_one_out_is_recoverable_from_own_treatment` |

### Step 4 — Semi-synthetic scenario generation ("Study A" — what is runnable today)

```bash
python scripts/prepare_eicu_semisynth.py \
    --cohort experiments/eicu_v1_demo/cohort.csv --g0 linear --seed 0
```

Because Step 2 shows no real causal estimate is possible on this data, and
because *no real eICU release has a known ground truth*, testing your paper's
core claim ("FedDeepGMM recovers `g_0` under real client heterogeneity")
requires a **known** `g_0`. This is the same logic your paper itself uses:
Section 5's low/high-dim experiments only report structural MSE because the
synthetic data-generating process is known.

**Kept real**: hospital partitions, covariate distributions (age, comorbidities,
first-hour labs/vitals), missingness pattern, ward structure, client size
imbalance, the Step-3 instrument.

**Simulated** (everything else): an unobserved confounder `U ~ N(0,1)`, treatment
`D ~ Bernoulli(sigmoid(a*Z + b'X + c*U + hospital_offset))`, outcome
`Y = g_0(D,X) + rho*U + noise`, for a known `g_0` (three variants: `linear`,
`interaction`, frozen-random-MLP). `U` enters both `D` and `Y`, so — as in your
paper's endogeneity assumption — naive regression of `Y` on `D` is biased and an
instrument is genuinely required to recover `g_0`.

**Splits** are within-client (a hospital's train/dev/test rows are that
hospital's own patients, keyed on hospital admission, not different hospitals
per split) and standardization statistics come from training rows only.

**Actual output** (`fedgmm/sp_decentralized_mnist_lr_example/data/eicu_semisynth/linear_seed0_metadata.json`,
regenerated 2026-07-26):

```
n_total: 201            n_clients: 89
n_covariates: 67        input dims (g and f networks): 68
treatment_rate: 0.433
true_ate: 1.0            <- ground truth, exactly matches g0's beta_d=1.0
split_sizes:       {train: 118, dev: 32, test: 51}
clients_per_split: {train: 89,  dev: 32, test: 51}
```

**What to check yourself**: `true_ate` in the metadata should equal `beta_d` in
the same file's `g0` block — an internal consistency check that the simulator's
ground truth is self-consistent. Confirmed programmatically in
`tests/test_eicu_semisynth.py::GenerateTest::test_true_ate_matches_linear_coefficient`.

Also verified (so the benchmark isn't accidentally trivial):
- `test_confounding_biases_naive_regression` — plain OLS on this data is
  measurably biased away from the true ATE.
- `test_instrument_recovers_the_effect_where_ols_fails` — 2SLS using the Step-3
  instrument gets closer to the true ATE than OLS does.

### Step 5 — Federated partitioning by real hospital, not Dirichlet

This is the direct fix to the gap noted in Section 1: the paper's existing
loader (`fedml/data/MNIST/data_loader.py`) partitions clients via
`np.random.dirichlet([alpha]*N)` over a **random permutation of an i.i.d. pool**
— every client sees the same underlying distribution, just different sample
counts. It cannot produce genuine distributional heterogeneity, only size
imbalance.

A new `load_data_natural()` path partitions by the real `hospitalid` carried in
the `.npz` from Step 4, and — critically — keys **all three splits (train/dev/
test) to the same client**, unlike the Dirichlet path which draws them
independently (harmless for i.i.d. data, wrong for hospitals: it would validate
one hospital's model against a different hospital's patients).

**Caveat found during testing, and why it's a demo-only artifact**: a client
must have at least one row in every split to be evaluable (the trainer indexes
each client's first eval batch). At this cohort's size (201 patients, 89
hospitals, median 2/hospital), this drops 57 of 89 hospitals, leaving **32
usable federated clients** for training. On full eICU (200k+ stays), this
constraint would drop close to zero hospitals — it is a consequence of the
demo's size, not of the method.

### Step 6 — Federated training launch

```bash
python fedgmm/sp_decentralized_mnist_lr_example/main.py \
    --cf $PWD/experiments/eicu_v1_demo/smoke_config.yaml
```

(`--cf` must be an **absolute** path — `main.py` `chdir`s into its own directory
at import, so a repo-relative path silently resolves to nothing.)

**No changes to the objective, optimizer, or FedAvg loop.** The paper's
`OptimalMomentObjective.calc_objective(g, f, x, z, y)`
(`game_objectives/simple_moment_objective.py:97-111`) is called completely
unmodified — verified by direct code inspection, quoted here:

```python
epsilon = g_of_x - y
moment  = f_of_z.mul(epsilon).mean()
f_reg   = self._lambda_1 * (f_of_z ** 2).mul(epsilon ** 2).mean()
return moment + g_reg, -moment + f_reg
```

The conditional moment restriction your paper needs,
`E[Y - g(D,X) | Z,X] = 0`, is obtained purely by **what tensors are packed
into the existing `x` and `z` slots**:

```
x = [D, X]   ->   g(x)  is exactly  g_theta(D, X)
z = [Z, X]   ->   f(z)  is exactly  f_tau(Z, X)
```

Three small, necessary changes were made to the surrounding plumbing (not the
game itself):
1. `scenarios/abstract_scenario.py` — optional `client_id` field on `Dataset`,
   used only for partitioning, never fed to `g` or `f`. Backwards compatible:
   pre-existing `.npz` files (`abs.npz`, `step.npz`, etc.) still load, with
   `client_id = None`, unchanged.
2. `fedml/data/MNIST/data_loader.py` — `load_data_natural()`, described in
   Step 5.
3. `fedml/model/model_hub.py` — `input_dim_g`/`input_dim_f` become data-driven
   (from the run config) instead of the hardcoded `1`/`2` used for the paper's
   1-D synthetic functions; hidden widths `[64,64]` for eICU vs. the paper's
   `[20,20]` for its 1-D functions, since eICU has ~68 input dimensions instead
   of 1.

**No change was needed** to model-selection / checkpointing: because Study A's
`g_0` is known, the existing validation-MSE selection
(`fedavg_api.py:705-725`, the same mechanism the paper's own experiments use)
works completely unchanged. A ground-truth-free selection metric (moment
violation) is only needed for a real-data Study B, which the demo cannot
support (Step 2) and has therefore not been built yet.

**Actual smoke-run result** (20 rounds, FedGDA-deterministic, regenerated
2026-07-26, `results/_smoke_eicu/eicu_semisynth/fedgda_s/seed_0/smoke/metrics.json`):

```json
{
  "best_validation_mse": 2.4436905059626,
  "best_validation_round": 19,
  "final_validation_mse": 2.4436905059626,
  "test_mse_at_best_validation": 6.9369488274292,
  "diverged": false,
  "selection_metric_source": "validation",
  "runtime_seconds": 42.5
}
```
32 real hospital clients, structural MSE against the known `g_0` decreasing
across rounds, no divergence. This is a **pipeline verification run**, not a
tuned result — 20 rounds, one seed, default learning rates.

---

## 5. Full verification trail — tests

All new code is covered by stdlib `unittest` (the repo does not use pytest).
**111 new tests, 0 failures**, plus the pre-existing suite unaffected (64
tests, run separately due to one pre-existing, unrelated `os.chdir`-at-import
leak in `certify_synthetic_data.py` that is not part of this work).

| File | Tests | What it protects |
|---|---|---|
| `test_eicu_cohort.py` | 32 | Cohort logic against a hand-built mini eICU release: correct inclusion/exclusion at every step, drug-name normalization, age top-coding, mortality parsing |
| `test_eicu_instrument.py` | 17 | Cross-fitting has no leakage, shrinkage behaves correctly, naive construction is provably broken (the Section 2 bug) |
| `test_eicu_iv_diagnostics.py` | 19 | Hand-rolled OLS/2SLS/partial-F/SMD balance, checked against designs with a known planted coefficient (no `statsmodels`/`linearmodels` in this environment) |
| `test_eicu_client_audit.py` | 15 | The eligibility gate picks the right construction under controlled synthetic scenarios, and **explicitly asserts the real demo cohort is refused** |
| `test_eicu_semisynth.py` | 25 | Splits never cross a hospital admission, standardization uses training rows only, no identifier leaks into covariates, confounding is real, IV beats OLS, natural partitioning behaves correctly including edge cases (empty splits, mismatched client counts) |

Run yourself:
```bash
/home/arnav22103/miniconda3/envs/fedgmm/bin/python -m unittest \
    tests.test_eicu_cohort tests.test_eicu_instrument \
    tests.test_eicu_iv_diagnostics tests.test_eicu_client_audit \
    tests.test_eicu_semisynth
```

---

## 6. What you can and cannot claim (directly from the plan's Section 17, restated against real numbers)

**You may claim**, once a full multi-seed Study A sweep is run:
- FedDeepGMM trains and recovers a known structural function over **real**
  hospital partitions with real covariate/missingness heterogeneity — something
  the paper's Dirichlet-only experiments cannot demonstrate.
- FedOGDA vs. FedGDA optimization stability, measured on this real-structure
  benchmark.
- Held-out moment violation and structural MSE comparisons against baselines,
  under stated simulation assumptions.

**You may not claim, from this data**:
- Anything about vasopressors and mortality — the demo fails the feasibility
  gate (Section 4, Step 2). This requires full eICU access.
- That ward preference is exogenous — it remains a stated assumption with a
  documented threat (ward practice may correlate with unit specialization /
  quality of care), not an established fact.
- That this is a secure or private federated deployment — it is a single-
  machine simulation, same as the paper's existing experiments.

---

## 7. Decision point

Everything above is built, tested, and verified end-to-end on CPU using the
demo. **Nothing has been launched at scale.** The concrete next action, pending
your signoff, is:

> Launch a multi-seed Study A sweep (N seeds × 3 `g_0` variants: linear /
> interaction / frozen-MLP) on the demo data, CPU-only, to produce the first
> real structural-MSE-vs-heterogeneity numbers for the paper. This is
> unambiguously a **methods/robustness result** ("recovery under real hospital
> heterogeneity"), not a clinical claim.

Full eICU access remains a separate, independent prerequisite for any Study B
(real vasopressor-mortality) result and does not block the above.
