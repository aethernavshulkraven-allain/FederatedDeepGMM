# eICU Study A protocol v1

Status: protocol specification; no experiment launch is authorized by this
document.

## 1. Status vocabulary

Every design choice in this protocol has one of three statuses.

- **Frozen**: required for any result labeled “Study A v1.” Changing it requires
  a versioned protocol amendment made before inspecting confirmatory test
  outcomes.
- **Proposed pending full-eICU runtime preflight**: the recommended starting
  choice, but it may be revised using runtime, memory, launchability, or
  validation-only evidence before confirmatory runs begin.
- **Unresolved before launch**: must be resolved, recorded, and frozen before
  tuning or confirmatory execution. These items are collected in
  `decision_register.md`.

`protocol_v1.json` is the authoritative machine-readable companion. If prose
and JSON disagree, execution must stop until the discrepancy is resolved by a
versioned amendment.

## 2. Purpose, questions, and claims

**Frozen.** Study A is a semi-synthetic benchmark of structural-function
recovery under natural eICU hospital heterogeneity. It keeps the real hospital
partition, covariate distributions, observed missingness patterns, ward
structure, and client-size imbalance, while simulating the confounding,
treatment, outcome, and a known common structural function \(g_0\).

The primary scientific question is whether FedGDA-S and FedOGDA-S recover the
known \(g_0\) under real hospital structure when the paper-aligned federated
objective and uniform client aggregation are used. Secondary questions compare
optimization stability, centralized baselines, and the effect of replacing
uniform-client aggregation with sample-size aggregation.

Study A can support a methods/robustness claim about a semi-synthetic benchmark.
It cannot estimate or identify a real clinical treatment effect, establish
instrument validity in clinical practice, reproduce a published numerical
eICU target, or demonstrate a private/secure multi-institution deployment.

## 3. Population and client construction

### 3.1 Data release and reporting population

**Frozen.**

- Final reported results require the full eICU release.
- The eICU demo release may be used only for smoke tests.
- A hospital is one federated client. Hospitals must never be pooled into a
  pseudo-client for the primary Study A analysis.
- Every included client must be represented in train, validation, and test.
- Splits are within hospital and keyed by hospital admission so a single
  admission cannot cross splits.
- Preprocessing statistics, imputation values, category vocabularies,
  cross-fitting fits, and standardization parameters are learned on training
  rows only.
- Hospital and ward identifiers may define clients or the instrument, but may
  not be input features to \(g_\theta\) or \(f_\tau\) unless a versioned
  amendment explicitly changes that rule.

**Proposed pending the full-eICU cohort audit.** Use the existing eICU cohort
construction as the starting point: adults, first ICU stay within a hospital
admission, known hospital discharge status, known hospital and ward, sepsis at
or near admission, a working infusion interface, and no qualifying
vasopressor before ICU admission. Use a within-hospital 60/20/20
train/validation/test split by admission.

This proposal does not assert the full-release cohort size, number of eligible
hospitals, treatment prevalence, mortality, ward coverage, missingness, or
instrument strength. Those values must come from a versioned full-eICU audit
and must be frozen before scenario generation.

### 3.2 Real inputs retained by the benchmark

**Frozen.** Retain measured baseline covariates, first-hour labs/vitals,
comorbidity features, categorical structure, explicit missingness indicators,
hospital membership, ward membership, and the empirical client-size
distribution. Continuous values are train-median imputed and train-standardized;
missingness indicators remain explicit. Categorical levels are learned from
training rows.

**Proposed.** Construct \(Z\) as a training-only, cross-fitted,
Beta-Binomial-shrunk ward-preference feature with within-hospital variation.
Validation and test rows are scored only from training fits. Eligibility based
on real structural \(Z\) variation and the simulated within-client first stage
must be certified before launch. Exact gates remain unresolved.

## 4. Semi-synthetic data-generating process

Let hospital \(i\) contain patient/admission rows \(j\), observed covariates
\(W_{ij}\), ward-based instrument \(Z_{ij}\), and hospital offset \(\xi_i\).
For each scenario seed, Study A uses

\[
U_{ij}\sim\mathcal N(0,1),
\]

\[
D_{ij}\sim\operatorname{Bernoulli}\left[
\operatorname{logit}^{-1}\{
aZ_{ij}+b^\top W_{ij}+cU_{ij}+\xi_i
\}\right],
\]

\[
Y_{ij}=g_0(D_{ij},W_{ij})+\rho U_{ij}+\varepsilon_{ij},
\qquad
\varepsilon_{ij}\sim\mathcal N(0,\sigma^2).
\]

**Frozen.**

- \(U\), \(D\), \(Y\), \(\xi_i\), and all stochastic causal-layer quantities
  are simulated.
- The same scenario artifact is shared across all methods, centralized
  baselines, and aggregation arms for a given `g0` and `scenario_seed`.
- \(U\) enters both treatment and outcome, creating endogeneity by design.
- \(Z\) affects simulated treatment and has no direct term in the simulated
  outcome equation.
- Tensor packing is \(x=[D,W]\) for \(g_\theta\) and \(z=[Z,W]\) for
  \(f_\tau\).
- Scenario generation records counterfactual truths \(g_0(1,W)\),
  \(g_0(0,W)\), the individual structural effect, per-client true ATE, and
  equal-client and sample-weighted true ATE.
- Every scenario artifact and its metadata receive a cryptographic checksum.
  Regeneration after tuning begins is prohibited unless the protocol version
  changes.

The numerical values of \(a,c,\rho,\sigma\), the distribution/scale of
\(\xi_i\), and the coefficient-generation rules are **proposed pending
full-eICU certification**. They must be chosen without Test MSE and then frozen
in scenario metadata.

### 4.1 Structural-function variants

All three variants are **frozen as required scenarios**:

1. `linear`: a treatment main effect plus a linear covariate component.
2. `interaction`: a treatment main effect plus covariate-dependent treatment
   heterogeneity and a covariate component.
3. `frozen_random_mlp`: a randomly initialized MLP evaluated as a fixed
   function of \([D,W]\).

The random MLP weights must be generated once from a dedicated `g0_seed`,
stored or fully reconstructible, checksummed, and held identical across all
scenario and optimizer seeds. Its architecture, scale normalization, and
dedicated seed are unresolved before launch. The displayed label
`frozen_random_mlp` maps to implementation label `mlp` only if metadata records
that mapping.

## 5. Estimands

### 5.1 Primary estimand

**Frozen.** The primary estimand is the mean, over eligible test hospitals, of
the hospital-specific structural MSE:

\[
\operatorname{MSE}_{EC}(\hat g)
=\frac{1}{N_T}\sum_{i\in\mathcal C_T}
\frac{1}{n_{iT}}\sum_{j\in T_i}
\{\hat g(D_{ij},W_{ij})-g_0(D_{ij},W_{ij})\}^2.
\]

The primary reported metric is this equal-client test MSE evaluated at the
checkpoint selected by the lowest equal-client validation structural MSE. Its
required compatibility field is `test_mse_at_best_validation`.

### 5.2 Secondary estimands

**Frozen.** Secondary estimands are:

- sample-weighted structural MSE;
- equal-client and sample-weighted ATE error;
- per-client MSE, ATE error, and held-out moment-violation distributions;
- individual-effect MAE, reported separately from ATE error;
- held-out moment violation;
- final-versus-best-validation gaps and final-iterate test metrics;
- oscillation summaries; and
- runtime and resource summaries.

Definitions and required field names are in `metric_policy.md`.

## 6. Methods, objective, and participation

### 6.1 Primary federated methods

**Frozen.**

- `fedgda_s` — FedGDA-S.
- `fedogda_s` — FedOGDA-S, including the intended optimistic corrections at
  both client and server.

No direction of superiority is assumed. FedOGDA-S need not beat FedGDA-S on
every \(g_0\), seed, metric, or client.

### 6.2 Centralized baselines

**Frozen.**

- `gda_d` — centralized full-gradient GDA.
- `sgda_s` — centralized stochastic GDA.
- `oadam_s` — centralized optimistic Adam.

Centralized training may pool training rows, but validation selection and test
reporting must reconstruct hospital membership and use the same equal-client
primary metric. Centralized methods use `aggregation_mode =
not_applicable_centralized`.

### 6.3 Paper-aligned objective and aggregation

**Frozen.**

- The local variational objective uses a frozen \(\tilde\theta\) in the
  quadratic critic regularizer and coefficient \(\lambda=1/4\).
- Primary federated runs use uniform aggregation over participating clients:
  each participating client receives weight \(1/|\mathcal P_t|\) in round
  \(t\), regardless of local sample count.
- Full participation is the recommended primary policy. If infeasible, the
  participation schedule must be fixed before tuning, shared across paired
  methods, and uniform over participating clients.
- `objective_mode` is `paper_aligned`; `aggregation_mode` is
  `uniform_clients`.

The exact refresh cadence for frozen \(\tilde\theta\) and the final
participation policy must be resolved and recorded in effective configuration
before launch.

### 6.4 Optional deterministic sensitivity

`fedgda_d` and `fedogda_d` are optional sensitivity methods: 3 \(g_0\) variants
× 5 confirmatory seed pairs × 2 methods = 30 additional runs. They are outside
the required 105-row matrix and cannot replace any required run.

## 7. Randomness and pairing policy

### 7.1 Separate seed semantics

**Frozen.**

- `scenario_seed` controls the admission split and all stochastic scenario/DGP
  draws. It never controls model initialization, minibatch order, or optimizer
  randomness.
- `optimizer_seed` controls model initialization, data-loader/minibatch order,
  stochastic optimizer state, and any randomized participation schedule. It
  never regenerates the scenario.
- A `seed_pair_id` binds one scenario seed to one optimizer seed. All compared
  methods and aggregation arms use the identical pair.
- Reusing one scalar program argument for both roles is noncompliant even when
  the two numerical values happen to match.

### 7.2 Tuning seeds

**Proposed pending full-eICU runtime preflight.** Use three disjoint tuning
pairs:

| seed_pair_id | scenario_seed | optimizer_seed |
|---|---:|---:|
| tuning_01 | 11 | 1011 |
| tuning_02 | 22 | 1022 |
| tuning_03 | 33 | 1033 |

Run a cheap screening pass on `tuning_01`, then evaluate every shortlisted
candidate on all three pairs. A candidate is not eligible for final selection
unless it has results for all three tuning pairs. Tuning scenarios are never
used in the confirmatory summary.

### 7.3 Confirmatory seeds

**Frozen for Study A v1.**

| seed_pair_id | scenario_seed | optimizer_seed |
|---|---:|---:|
| confirmatory_01 | 101 | 1101 |
| confirmatory_02 | 102 | 1102 |
| confirmatory_03 | 103 | 1103 |
| confirmatory_04 | 104 | 1104 |
| confirmatory_05 | 105 | 1105 |

No failed or unfavorable pair may be replaced. A versioned amendment made
without consulting test outcomes is required to change this list.

## 8. Tuning policy

### 8.1 Frozen rules

- Tuning is separate from confirmatory evaluation.
- Tune independently for each (`g0`, `method`) combination.
- Use identical scenario artifacts, tuning seed pairs, training budget, and
  comparable search effort for paired methods.
- Test arrays, Test MSE, test ATE, and all test summaries are inaccessible to
  tuning and selection code.
- Within each run, select the checkpoint by minimum equal-client validation
  structural MSE; tie-break by the earlier round.
- A candidate is eligible only if all required tuning pairs completed without
  non-finite parameters or metrics.
- Select the candidate lexicographically by:
  1. lowest mean best equal-client validation structural MSE across tuning
     pairs;
  2. lowest mean equal-client held-out validation moment violation at the
     selected checkpoints;
  3. lowest mean final-minus-best equal-client validation MSE gap;
  4. stable deterministic candidate ID order.
- After tuning, write an immutable selection record containing every
  candidate, validation fields, exclusion reason, selected candidate,
  scenario/config checksums, and selection-code version.

If no candidate is eligible, the method/scenario is blocked. The grid may be
amended using validation and numerical diagnostics only; no confirmatory Test
MSE may be inspected.

### 8.2 Proposed search space

**Proposed pending full-eICU runtime preflight.** Use staged successive
screening rather than an assumed full Cartesian sweep. Candidate factor levels
are:

- learner learning rate: `0.0005, 0.001, 0.002`;
- critic multiplier: `5, 10, 20`;
- weight decay: `0.001, 0.01, 0.03`;
- federated server learning rate: `0.5, 1.0, 1.5`;
- gradient clipping norm: `1.0, 5.0`;
- stochastic batch size: `128, 256` where client sizes permit.

The centralized search must tune comparable learner/critic learning rates,
weight decay, clipping, batch size, and training budget, omitting federated-only
server parameters. The exact fractional design, shortlist size, round budget,
and wall-clock cap are unresolved until a full-eICU runtime preflight. Search
budget must not be increased for a method after viewing its confirmatory
performance.

## 9. Confirmatory policy and required matrix

**Frozen.** After all (`g0`, `method`) hyperparameters and checkpoint rules are
frozen:

- Primary federated: 3 \(g_0\) × 5 seed pairs × 2 methods = 30 runs.
- Centralized baselines: 3 \(g_0\) × 5 seed pairs × 3 methods = 45 runs.
- Aggregation ablation: 3 \(g_0\) × 5 seed pairs × 2 federated methods using
  `sample_size` aggregation = 30 additional runs.
- Required total = 105 runs.

`confirmatory_matrix.csv` is descriptive, not a launch manifest. The 30
uniform-client primary rows serve as the comparator arms for the 30
sample-size ablation rows; they are not duplicated.

Confirmatory runs use frozen tuning selections without adaptation. Test
metrics may be computed only after the run’s best-validation checkpoint is
fixed. Missing, failed, and diverged runs remain visible with reasons; they are
not silently rerun under different settings or replaced with new seeds.

## 10. Checkpoint policy

**Frozen.**

1. At each scheduled validation point, compute per-client validation structural
   MSE.
2. Average those client values uniformly.
3. Save a candidate checkpoint containing both \(g\) and \(f\), optimizer
   state if required for audit, round, `effective_config`, and artifact
   checksums.
4. Choose the smallest equal-client validation structural MSE; exact ties go
   to the earlier round.
5. Preserve both `best_validation` and `final` checkpoints.
6. Evaluate the test set at the already chosen best-validation checkpoint and
   report `test_mse_at_best_validation`.

Test MSE, test ATE, test moment violation, or any client’s test metric must
never choose a checkpoint. The final checkpoint is a stability diagnostic, not
the primary estimator.

## 11. Aggregation ablation

**Frozen.** For every primary federated row, add one otherwise identical
`sample_size` run. Match scenario artifact, seed pair, method, objective,
architecture, selected hyperparameters, training budget, participation
schedule, and checkpoint policy. Only aggregation weighting changes:

\[
w_i^{EC}=1/|\mathcal P_t|,
\qquad
w_i^{SW}=n_{it}/\sum_{k\in\mathcal P_t}n_{kt}.
\]

The ablation estimates the paired effect of weighting, not a new primary
method. It remains secondary even if it yields lower Test MSE.

## 12. Divergence, failures, and stability

**Frozen.**

- `diverged: true` means at least one model parameter or required metric became
  NaN or infinite. A finite but poor MSE is not divergence.
- OOM, missing artifact, scheduler failure, invalid configuration, timeout, or
  user interruption is `run_status: failed` with `failure_reason`; it is not
  automatically divergence.
- Report the count and identities of diverged, failed, and incomplete runs for
  every method/scenario.
- Do not discard finite outliers. Report robust summaries and per-seed values
  alongside means.
- Final-versus-best validation gap is final equal-client validation MSE minus
  best equal-client validation MSE.
- Oscillation is summarized over the last `min(50, number_of_recorded_rounds)`
  validation points by the standard deviation and range of equal-client
  validation structural MSE, plus the final-versus-best gap.

## 13. Analysis plan

**Frozen.**

1. Verify matrix completeness, checksums, seed pairing, objective mode, and
   aggregation mode before reading test outcomes.
2. For each (`g0`, `method`, `role`), report all five seed-pair values, mean,
   population standard deviation, median, minimum, and maximum of the primary
   metric; also report failure/divergence counts.
3. Within each `g0`, compute paired FedOGDA-S minus FedGDA-S differences at
   matched seed pairs for primary MSE, final-iterate MSE, final-versus-best gap,
   oscillation, and runtime. Negative MSE difference favors FedOGDA-S.
4. Compare each federated method with each centralized baseline descriptively
   at matched scenario seeds. Do not call numerical similarity “matching the
   paper”; Study A has no published numerical target.
5. Compare `sample_size` with its matched `uniform_clients` run using paired
   differences. The equal-client estimand remains primary for both arms.
6. Report equal-client and sample-weighted metrics side by side and the full
   per-client distribution. Do not substitute the sample-weighted result for
   the primary metric.
7. With only five confirmatory pairs, emphasize effect sizes, paired
   directions, and uncertainty descriptively. Any formal interval or test must
   be labeled exploratory and must not support a binary superiority claim.
8. Report deterministic sensitivities separately if run.

No method ordering or superiority threshold is preregistered. Conclusions must
reflect observed consistency, magnitude, and stability rather than a
requirement that FedOGDA win.

## 14. Claim boundaries and demo/full distinction

### Permitted after a compliant full-eICU campaign

- Recovery of a known simulated structural function under real eICU hospital,
  covariate, missingness, ward, and client-size structure.
- Descriptive FedGDA-S/FedOGDA-S stability and error comparisons on that
  benchmark.
- Comparisons with centralized training and with sample-size aggregation under
  the frozen design.

### Not permitted

- A causal estimate of vasopressor effects on mortality or another real
  outcome.
- A claim that ward preference is clinically exogenous.
- A claim that the full-eICU cohort has any unmeasured size, balance,
  representativeness, or first-stage property.
- A paper-reproduction or paper-match claim for Study A.
- A privacy, security, deployment, or multi-site systems claim.
- A conclusion selected by Test MSE.

### Demo data

The demo release is smoke-only. Demo runs may check parsing, scenario
generation, dimensions, finite gradients, checkpoint serialization, metric
plumbing, and deterministic reruns. Demo Test MSE is not a final result, must
not tune the full-eICU campaign, and must not enter confirmatory tables. The
labels `smoke` and `data_scope: eicu_demo` are mandatory for demo artifacts.

## 15. Audit and reproducibility requirements

Every required run must preserve the fields listed in
`protocol_v1.json.required_effective_config_fields` and
`protocol_v1.json.required_metrics_fields`. At minimum, retain:

- immutable cohort, split, scenario, and configuration checksums;
- complete seed roles and seed-pair ID;
- actual client list and per-split counts;
- requested and effective objective/aggregation settings;
- frozen \(\tilde\theta\) policy and \(\lambda\);
- model/optimizer hyperparameters and training budget;
- best-validation and final checkpoint identities;
- equal-client, sample-weighted, and per-client validation/test metrics;
- divergence/failure diagnostics and runtime.

Any mismatch between requested and effective settings invalidates the run until
explained and classified. Existing scientific results, including
`results/_golden`, must never be overwritten.
