# eICU Study A v2: off-hours semi-synthetic benchmark

Status: frozen setup specification for Study A v2. The materialized 27 July
campaign uses the eICU demo and is therefore an engineering campaign, not a
scientific full-eICU result.

This version supersedes the Study A v1 cohort and instrument for new runs. It
does not alter or invalidate the archived v1 pipeline-validation artifacts.
The historical implementation and failure analysis are in
`handoff-eicu-study-a-b-27jul-2026.md`.

**Amendment 1 (2026-07-27)** revises the tuning/final training horizon, the
learning-rate tuning grid, and adds two pre-finals gates. It does not change
the cohort, instrument, DGP, primary endpoint, selection metric, test-blindness
discipline, or decision rule. See "Amendment 1" near the end of this document
for the full record of what changed, when, and why; the original "Tuning and
final matrix" section below is left as written and is marked with a pointer
to the amendment rather than edited in place.

## Aim and claim boundary

Study A v2 evaluates recovery of a known structural response under natural
hospital partitions, covariate/missingness heterogeneity, and client-size
imbalance. It does not estimate the effect of vasopressors, off-hours
admission, or any real treatment; validate off-hours admission as a clinical
instrument; or support treatment recommendations.

## Cohort and clients

- Adult patients.
- First ICU stay per hospital admission.
- Valid `hospitalid`, hospital admission clock, stay ID, and patient ID.
- No sepsis, mortality, vasopressor, infusion-interface, multi-ward, or
  ward-preference requirement.
- A patient seen at multiple hospitals is excluded so it cannot belong to two
  federated clients.
- One hospital is one client. `wardid` is retained only for audit and is never
  a client or model input.
- Hospitals require at least seven rows, three patient groups, non-empty
  Train/Dev/Test, and real off-hours variation overall and in Train.
- The retained hospital list is frozen before simulation and cannot change
  with `scenario_seed`.

Splits are 70/15/15 within hospital, with whole patients and hospital
admissions kept together. The fixed split seed is `20260727`. Whole-patient
grouping may cause small per-hospital rounding deviations; the campaign-wide
materialized split is 1,420/306/305 rows.

## Instrument and covariates

The primary instrument is

\[
Z_i=1\{\texttt{hospitaladmittime24}\notin[07{:}00,19{:}00)\}.
\]

The eICU patient table does not provide a reliable admission date or weekday,
so the optional weekend instrument is not used.

Covariates are age, gender, admission weight, hospital admission source, first
hour heart rate, mean arterial pressure, respiratory rate, oxygen saturation,
temperature, and selected labs. Continuous variables use Train-only median
imputation and standardization plus explicit missingness indicators.
Categorical levels and reference categories are fitted on Train only.
Hospital, ward, patient, and stay IDs are forbidden model inputs.

## Semi-synthetic DGP

For scenario seed \(s\),

\[
U_i\sim N(0,1),
\]

\[
X_i=2Z_i+\beta^\top W_i+\eta_{h(i)}+U_i+\epsilon_i^X,
\quad \epsilon_i^X\sim N(0,0.5^2),
\quad \eta_h\sim N(0,0.5^2),
\]

\[
Y_i=g_0(X_i,W_i)+U_i+\epsilon_i^Y,
\quad \epsilon_i^Y\sim N(0,0.5^2).
\]

`linear` uses \(g_0=aX+b^\top W\) with \(a=1\).
`interaction` additionally uses \(0.5XW_1\). `mlp` uses a separately frozen,
scenario-seeded, width-32 ReLU MLP. All coefficients, MLP weights, hospital
effects, preprocessing fits, truths, counterfactual contrasts, and checksums
are stored.

The reported effect contrast for the continuous treatment is explicitly
\(g_0(1,W)-g_0(0,W)\).

## Certification

Acceptance uses Train+Validation only. Test is never inspected when accepting
or regenerating a scenario. A weak candidate regenerates the entire seed from
the next deterministic attempt; individual hospitals are never removed using
a simulated first stage. At most 25 attempts are allowed.

Required checks are:

- the frozen client list is identical across scenarios and all clients occur
  in Train/Dev/Test;
- real \(Z\) varies within every retained hospital;
- global conditional first-stage F is at least 10;
- \(|\operatorname{corr}(U,Z)|\le 0.10\);
- the absolute \(U\) coefficient in both generated treatment and structural
  outcome residual is at least 0.20;
- maximum absolute empirical moment at the true \(g_0\) is at most 0.20;
- patient and hospital-stay split leakage counts are zero;
- in the linear scenario, naive structural-coefficient bias is at least 0.05
  and 2SLS absolute error is smaller than naive-regression error.

## Models and objective

Both response and critic networks are

```text
Input -> Linear(32) -> ReLU -> Linear(32) -> ReLU -> Linear(1)
```

There is no BatchNorm or dropout. Federated and centralized implementations
use the same architecture. `[64,64]` is reserved for a separately labeled
sensitivity analysis.

All primary runs use `objective_mode=paper_aligned`, frozen previous-global
\(\tilde\theta\), critic regularization coefficient \(1/4\), full client
participation, and `aggregation_weighting=uniform_clients`. Sample-size
weighting is allowed only in the aggregation-ablation role. The federated
minibatch size is 4; every retained client's Train split has at least five
rows, so the stochastic method labels correspond to at least two local
minibatches rather than a disguised full-client batch.

## Tuning and final matrix

> **Superseded by Amendment 1 (2026-07-27) — see below.** This section is
> preserved exactly as originally written, as the historical record of what
> was pre-registered before the amendment. The `tuning_rounds=150` /
> `final_rounds=500` horizon and the learning-rate / server-learning-rate grid
> described here were replaced by a shared 4000-round horizon and a widened
> grid before any further runs were launched; do not use the numbers in this
> section to configure new runs. See "Amendment 1" for the current values,
> the evidence, and the two gates that must pass before finals launch.

Tuning uses scenario/optimizer pair `(11, 1011)`, 150 rounds, and the grid:

- local LR multiplier: 0.5, 1.0, 2.0 around base LR 0.001;
- server LR: 1.0, 1.5.

Selection is separate for every `(g0, federated method)`: lowest equal-client
validation structural MSE, tie-broken by equal-client validation moment
violation at that checkpoint. Test is not read. The selected JSON must exist
before the final manifest can be materialized.

Final seed pairs are `(101,1101)` through `(105,1105)`. The frozen matrix is:

- 30 primary federated runs: 3 functions × 5 pairs × FedGDA/FedOGDA;
- 45 centralized runs: 3 functions × 5 pairs × GDA/SGDA/OAdam;
- 30 sample-size aggregation ablations;
- 105 total.

The primary result is equal-client Test structural MSE at the
validation-selected checkpoint. Secondary reporting includes sample-weighted
MSE, held-out moments, effect errors, individual-effect MAE, per-hospital
distribution summaries, final-versus-best degradation, oscillation,
divergence, and runtime. FedOGDA–FedGDA differences are paired by scenario and
optimizer seed.

## Amendment 1 (2026-07-27): shared training horizon, widened tuning grid, pre-finals gates

**Status.** This amendment is dated 2026-07-27 and is in effect for all runs
launched after this date. It supersedes the `tuning_rounds`, `final_rounds`,
and tuning-grid values stated in "Tuning and final matrix" above, which are
left unedited above as the historical pre-registration record. It changes
nothing else: the cohort and client list, the instrument, the semi-synthetic
DGP, the model architecture, the primary endpoint (equal-client Test
structural MSE at the validation-selected checkpoint), the selection metric
(`equal_client_validation_mse`), the test-blindness discipline, and the
pre-registered decision rule (a method is "favored" only at ≥4/5 wins per g0
and ≥12/15 pooled with the mean paired difference agreeing in sign; otherwise
"inconclusive") all stand exactly as written elsewhere in this document. The
cohort is unaffected: 2,031 admissions, 179 hospital clients, 1,420/306/305
splits all stand, and the checksummed artifacts `cohort_metadata.json`,
`frozen_client_list.json`, and `cohort.csv` are untouched by this amendment.

**Trigger.** The first Study A v2 final campaign was launched under the
original "Tuning and final matrix" values, then stopped after 47 of 105 rows
had run and quarantined at
`results/_failed/20260727-study-a-v2-superseded-finals-unconverged-horizon/`
(see that directory's `README.md` for the full incident record). Three
defects, described and corrected below, invalidated it as a basis for a
FedGDA-vs-FedOGDA claim.

### 1. Training horizon: 150/500 rounds → a single shared 4000 rounds

Previously: `tuning_rounds=150`, `final_rounds=500` (mismatched).
Now: `tuning_rounds=4000`, `final_rounds=4000` (identical, shared by tuning
and finals).

The tuning/final mismatch was itself a defect, independent of either horizon
being individually too short: hyperparameters selected under a 150-round
budget were carried forward unverified into a 500-round campaign. Tuning and
the reported campaign must now share one horizon so that the selected
configuration is verified optimal at the horizon it is actually reported at.
See "Evidence" below for why 150 and 500 rounds were both insufficient and
why 4000 rounds is adequate.

### 2. Tuning grid: widen upward

- Learning rate: `{0.0005, 0.001, 0.002}` → `{0.001, 0.002, 0.004, 0.008}`
  (base LR stays `0.001`; the local LR multiplier grid becomes `1.0, 2.0,
  4.0, 8.0` around it, replacing `0.5, 1.0, 2.0`).
- Server learning rate: `{1.0, 1.5}` → `{1.0, 1.5, 2.0}`.

`lr=0.0005` is dropped from the grid; see "Why drop lr=0.0005" below.

### 3. Two gates that must pass before finals launch

Both gates are evaluated on the re-tuned, 4000-round tuning sweep, before the
final manifest is materialized or any final run is launched.

- **Convergence gate.** If `best_validation_round` falls within the last 5%
  of `comm_round` for more than 20% of runs in any `(g0, method)` group, the
  horizon is insufficient — extend it and re-tune. Do not proceed to finals.
- **Grid-edge gate.** The `at_grid_edge` flag (already computed by
  `scripts/select_eicu_study_a_v2_tuning.py`) must be `False` for all six
  `(g0, method)` groups. An edge selection means the search did not find an
  interior optimum and the grid must be widened again.

If either gate fails, the corrected sequence is: extend the horizon and/or
widen the grid further, re-tune, and re-check both gates. Finals may not be
launched while either gate is failing.

### Evidence

All figures below are measured, on the `equal_client_val_mse` metric, seed
1011, at `learning_rate=0.002`, `server_learning_rate=1.5`. **All probe runs
were validation-only diagnostics — they were never used for hyperparameter
selection, and the test split was never read for any of them.**

**Grid-edge evidence.** All six `(g0, method)` tuning groups selected the
ceiling of both tuned axes: `learning_rate=0.002` (max of the grid) and
`server_learning_rate=1.5` (max of the grid). Validation MSE decreased
monotonically across all 36 candidates with no interior optimum.

**Non-convergence evidence.** At 150 tuning rounds, 29 of 36 runs peaked at
round 149 (the final round); 30 of 36 peaked within the final 10 rounds. The
six 500-round horizon-confirmation runs then peaked at round 499 in 6 of 6
cases.

**Horizon probe at 2000 rounds.** FedOGDA leads early, FedGDA overtakes, and
the crossover round tracks how fast each structural function converges:

| g0 | FedOGDA lead peaks | crossover round | round 499 (FedGDA / FedOGDA) | round 1999 (FedGDA / FedOGDA) |
|---|---|---|---|---|
| mlp | round 3 (−0.0022) | 45 | 0.1514 / 0.2137 | 0.1208 / 0.1976 |
| linear | round 487 (−0.1178) | 680 | 1.3480 / 1.2308 | 0.3396 / 0.4613 |
| interaction | round 547 (−0.1178) | 734 | 3.0934 / 2.9793 | 0.7203 / 0.8184 |

The consequence must be stated plainly: the 500-round horizon captured
essentially 100% of the maximum advantage FedOGDA ever holds on `linear`
(peak at round 487), and the sign of the FedGDA-vs-FedOGDA difference
reverses by round 700. At 500 rounds the campaign would have reported FedOGDA
ahead on 2 of 3 structural functions; at 2000 rounds FedGDA is ahead on 3 of
3.

**Why 4000 and not 2000.** `interaction` had not converged at 2000 rounds —
still improving 6.7% per 250 rounds at round 1750, with
`best_validation_round = 1999`. `mlp` converges fastest and begins overfitting
after round 1184. The protocol already selects the best-validation
checkpoint, so a longer horizon is safe for fast-converging families — this
is the justification for one shared horizon of 4000 rounds rather than
per-g0 horizons.

**Why drop lr=0.0005.** At 2000 rounds it reached only 1.348 (linear/FedGDA)
versus 0.340 at `lr=0.002` — four times worse, and still descending 40.5%
over the final 500 rounds. It is not a distinct optimum, only a slower path
to the same place, so it is dropped rather than kept as a fourth grid point.

## Data scope

The demo release can validate the pipeline and tuning mechanics but cannot
support a full-eICU scientific claim. A credentialed full release requires a
new read-only release preflight and fresh, separately checksummed cohort and
scenario artifacts under the same v2 protocol.
