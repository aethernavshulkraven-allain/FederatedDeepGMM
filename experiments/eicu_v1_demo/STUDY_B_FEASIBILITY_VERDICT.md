# Study B on the eICU demo release: final feasibility verdict

**Date:** 2026-07-28
**Data:** `physionet.org/files/eicu-crd-demo/2.0.1` (2,520 unit stays, 186 hospitals)
**Question asked:** is there *any* cohort / treatment / instrument / client
configuration under which the real-outcome IV study (Study B) becomes estimable
on the demo release?

**Answer: no.** Not because of a definitional mistake that can be corrected, but
because of a sampling property of the demo release that no analysis choice can
alter. This document records the evidence so the decision does not have to be
revisited.

---

## 1. Both of your cohort variants reproduce exactly

| Cohort | n | treated | untreated | deaths | hospitals | wards |
|---|---:|---:|---:|---:|---:|---:|
| `diagnosisoffset <= 0` (pre-correction) | 6–16 | 0 | all | 1–2 | 6–14 | 6–14 |
| `admissionDx` sepsis (corrected) | **256** | **28** | **228** | **42** | **125** | **136** |

The corrected cohort reproduces to the row. The correction was real and it fixed
what it was meant to fix: the strict `diagnosisoffset <= 0` rule was selecting on
charting latency rather than on clinical state, and it removed every treated
patient. Using the admission diagnosis is the right call.

It did not, and could not, fix the second problem.

Note one difference from the repo's v1 builder: the 256-row cohort drops the
infusion-interface gate (`prepare_eicu_cohort.py:185`). Without that gate, a
hospital that never connected its infusion pumps enters the study as a hospital
where nobody is ever treated — measurement error in D, not a real zero. This
does not change any conclusion below, but the gate should stay on for anything
reported.

---

## 2. The binding constraint: the demo is wide and shallow, by construction

This is the fact that decides everything, and it is a property of the release,
not of any cohort definition:

| | demo |
|---|---:|
| unit stays | 2,520 |
| distinct hospitals | **186** (of ~208 in full eICU) |
| stays per hospital, median / max | **12 / 40** |
| hospitals with ≥ 50 stays | **0** |
| hospitals with ≥ 10 in-hospital deaths | **0** (max in any hospital: 7) |
| hospitals with ≥ 2 wards of ≥ 15 patients each | **0** |

The demo is not a shrunken eICU. It is a near-complete *census of hospitals* with
about a dozen patients each — precisely the opposite shape from what a federated
within-hospital design needs. In your 256-patient cohort the largest single
hospital holds **6 patients**, 54 of 125 hospitals contribute **exactly one**
patient, and only **14** hospitals have both a treated and an untreated patient.

> Correction to the working notes: the demo has **186 hospitals averaging 13.5
> stays**, not "20 hospitals × ~125 stays". This inverts the diagnosis. The
> problem is not too few clusters for cluster-robust inference; it is too few
> patients inside each cluster for any client-local moment condition to exist.

---

## 3. Why no redefinition can rescue it (a monotonicity argument)

Every candidate Study B cohort is a **subset** of those 2,520 stays. Patients per
hospital, deaths per hospital, wards per hospital and per-hospital treated /
untreated support are all monotone non-increasing under subsetting. So the
numbers in §2, computed with **no restriction of any kind**, are hard ceilings.
Any sepsis definition, time-zero rule, treatment window or drug list can only
move you further from them.

The one thing that is *not* monotone is the treatment definition, so that was
swept separately: vasopressor 0–6 h, 0–24 h, any-time, day-1 ventilation and
day-1 dialysis, crossed with six populations from `diagnosisoffset <= 0` through
"no sepsis filter at all" (30 configurations, `sweep.csv`).

**Across all 30, the number of hospitals with ≥ 20 treated and ≥ 20 untreated
patients is 0.** The best cell anywhere in the space — every adult ICU stay,
ventilation as treatment, i.e. no longer Study B's clinical question at all —
reaches 15 hospitals with ≥ 5 treated and ≥ 5 untreated. The pre-registered gate
in `audit_eicu_clients.py:47` asks for ≥ 200 patients, ≥ 20 treated, ≥ 20
untreated, ≥ 20 deaths, ≥ 2 wards of ≥ 50. Nothing in the release comes within an
order of magnitude of it.

---

## 4. The ward-preference instrument carries no real signal (placebo test)

This is the most important new evidence, and it is the reason the first-stage F
should not be read as encouraging.

Depending on the control set, the pooled first-stage F on the corrected cohort
lands anywhere from 2.99 (repo's 17-covariate block, ~12 rows/parameter) to
16.64 (age + sex). Your 3.564 is consistent with the fuller control set. But the
F itself is the wrong thing to look at, because it does not distinguish practice
variation from small-ward accident.

Placebo test — randomise which ward each patient was in, **within** their
hospital, preserving hospital, ward sizes and every treatment decision. Real
practice-style variation must not survive this. 40 permutations:

| Configuration | observed F | placebo mean | p |
|---|---:|---:|---:|
| Study B corrected, pooled | 16.64 | 15.85 | 0.050 |
| Study B corrected, within-hospital | 771 | 833 | 0.950 |
| All adult ICU + vaso 0–6 h, pooled | 27.53 | 32.82 | 0.750 |
| All adult ICU + vent, pooled | 67.23 | 74.83 | 0.825 |
| All adult ICU + vent, within-hospital | 1079 | 1112 | 0.550 |

The observed first stage is **indistinguishable from randomly assigned wards**,
and in the larger configurations it is *below* the placebo mean. The spectacular
within-hospital F values (771–1214) are pure artifact: inside multi-ward
hospitals the median ward holds **4 patients**, 22 wards hold exactly one, and at
19 % treatment prevalence a 2-patient ward is homogeneous **69 %** of the time by
chance alone. Cross-fitting then hands patient *i* a shrunken copy of the one
other patient's treatment status. That is a coin flip, not a practice style.

By contrast, shuffling patients **across hospitals** does destroy the first stage
(observed 16.6–67.2 vs. placebo 1.5–2.5, p ≤ 0.025). So there is genuine
signal in the data — but it is entirely **between** hospitals:

| | Var(Z) | between-hospital | within-hospital | % usable federated |
|---|---:|---:|---:|---:|
| Study B corrected | 0.0532 | 0.0382 | 0.0151 | 28 % |
| All adult ICU + vaso | 0.0078 | 0.0065 | 0.0012 | 16 % |
| All adult ICU + vent | 0.0303 | 0.0261 | 0.0042 | 14 % |

and the within-hospital remainder is the artifact just described.

What this does *not* mean is that the estimator cannot see between-hospital
variation at all. Under equal-client aggregation the summed moment
$\frac{1}{N}\sum_i \frac{1}{n_i}\sum_k f(Z_{ik},W_{ik})\varepsilon_{ik}$ is a
reweighted **pooled** moment, and the critic $f$ is a global model, so
between-client $Z$ variation does enter the objective.

What collapses is the **justification for the ward construction**. Ward
preference was adopted over hospital preference precisely because a
hospital-level instrument supplies no within-client variation
(`eicu_instrument.py:3-13`). On this release the ward level supplies none
either, beyond what randomly assigned wards produce. So the design silently
degenerates into a hospital-preference IV — one whose real variation is only
38–52 % excess over binomial noise (below), whose exclusion restriction is at
its weakest (hospital quality and case mix are exactly the omitted pathways),
and whose precision is the interval in §5. The instrument is not identifying
*within* clients, which is the property the extension's design claims.

Even that between-hospital signal is roughly half noise: with ~13 patients per
hospital, the observed spread of hospital treatment rates (SD 0.082–0.165) is
close to what pure binomial sampling would produce with *identical* true rates
(SD 0.065–0.120). Only 38–52 % of the observed between-hospital variance is
excess over chance.

### The grouped-client fallback fails the same test

`audit_eicu_clients.py` offers client = region × teaching × beds with hospital
preference as Z. It gives 30 clients of median 48 patients — but 0 groups with
≥ 20 treated and ≥ 20 untreated, **0 groups with ≥ 20 deaths**, and a
within-client first stage that is again indistinguishable from a placebo that
randomises hospital membership (F = 4.44 vs. placebo 8.51, p = 0.57 for
vasopressors; F = 14.44 vs. 7.64, p = 0.10 for ventilation). It also abandons
"one hospital = one client", which is the claim the extension exists to make.

---

## 5. Even ignoring all of the above, the precision is not there

Pooled 2SLS on the corrected cohort, treating it as a single centre:

| Instrument | ATE (risk difference) | 95 % CI |
|---|---:|---|
| cross-fitted ward preference | −0.058 | **[−0.64, +0.52]** |
| cross-fitted hospital preference | −0.089 | [−0.68, +0.50] |
| leave-one-out hospital rate | −0.106 | [−0.75, +0.54] |

The interval spans "kills two thirds of patients" to "saves half of them". This
is not a borderline result that a better estimator sharpens; it is the absence of
information.

The ceiling is easy to state. With `Var(β₂ₛₗₛ) ≈ σ²/(n · R²_partial · Var(D))`,
using the most favourable numbers the release can offer (n = 2,086 — every adult
ICU stay; ventilation, the best-supported treatment; 8.4 % mortality):

| assumed first-stage partial R² | implied F | 95 % CI width |
|---:|---:|---:|
| 0.005 | 10.5 | 0.86 |
| 0.010 | 21.0 | 0.61 |
| 0.020 | 42.5 | 0.43 |
| 0.050 | 109.6 | 0.27 |
| 0.100 | 231.4 | 0.19 |

A facility-preference instrument in the comparative-effectiveness literature
typically achieves partial R² of 0.01–0.05. At the top of that range the demo
still cannot distinguish a 13-point mortality increase from a 13-point decrease —
**with a hypothetically valid instrument and the entire release used as one
pooled sample**. Restricting to sepsis (n = 256, 42 deaths) makes it far worse.

---

## 5b. The noise floor, and exactly what cluster depth would fix it

A cross-fitted preference estimated from `m` peers is a binomial proportion: it
carries the true between-unit practice signal `tau^2` plus sampling noise
`p(1-p)/m`. The classical reliability ratio

```
lambda(m) = tau^2 / (tau^2 + p(1-p)/m)
```

is the attenuation on the first stage — `lambda -> 1` is a clean instrument,
`lambda -> 0` is a coin flip. Estimating `tau` from the demo as the excess
between-hospital SD over binomial noise (§4) gives `tau = 0.051` for
vasopressors (p = 0.046) and `tau = 0.113` for ventilation (p = 0.188):

| patients per hospital | lambda (vasopressor) | lambda (ventilation) |
|---:|---:|---:|
| **13.5 — eICU demo** | **0.44** | **0.53** |
| 50 | 0.75 | 0.81 |
| 100 | 0.85 | 0.89 |
| 250 | 0.94 | 0.95 |
| **966 — full eICU-CRD** | **0.98** | **0.99** |

Roughly **110–155 patients per hospital** are needed for `lambda >= 0.90`. The
demo has 13.5. This converts the qualitative "not enough data" into a
requirement with a number attached, and it is the natural analytical backbone
for a feasibility study.

**Important caveat for planning.** The full release fixes the *instrument* noise
floor decisively (lambda 0.98–0.99 at 966 stays per hospital), but it does not
automatically clear the pre-registered per-hospital gate for a *sepsis-restricted*
Study B. Projecting the demo's rates onto ~966 stays per hospital:

| gate | projected (sepsis cohort) | required | |
|---|---:|---:|---|
| patients per hospital | ~97 | 200 | short |
| treated per hospital | ~4 | 20 | short |
| deaths per hospital | ~8 | 20 | short |

At ~10 % sepsis prevalence, a sepsis cohort keeps only ~97 stays per hospital and
the instrument reliability falls back to 0.85–0.89. So full-eICU Study B would
still need either a broader clinical population (all adult ICU, or a wider
infection cohort), a revised gate with a documented rationale, or acceptance that
a minority of large hospitals carry the analysis. **Credentialing is necessary
but, on the current cohort definition, may not be sufficient** — worth knowing
before the effort is spent, not after.

## 6. On the proposed ventilation + leave-one-out hospital-rate design

Assessed on the same evidence, this does not change the outcome:

1. **The leave-one-out hospital rate cannot be the instrument for a federated
   design with client = hospital.** Since `D_hi = S_h − (n_h − 1)·Z_hi`, it is an
   affine function of the patient's own treatment and takes at most two values
   inside a hospital — zero within-client variation. This is already proven in
   `tests/test_eicu_instrument.py` and is the exact bug the ward construction was
   written to replace. It is a property of the estimator, not of the demo, so it
   would be equally wrong on the full release.
2. **Ventilation does help on the margins** — 393 treated vs. 95 for
   vasopressors, and it is the single best cell in the entire sweep. But it still
   yields 0 hospitals meeting the eligibility gate, and it answers a different
   clinical question.
3. The synthetic dry-run's `F = 90` reflects the generator's planted
   `0.6 · hosp_practice` coefficient, not the data. On the real demo the same
   construction gives F = 81.7 pooled — which the placebo test in §4 shows is
   between-hospital variation, half of it binomial noise.
4. `apacheApsVar` is **APACHE-day worst values**, so using its vitals and labs as
   "confounders" for a first-day treatment puts post-treatment measurements into
   the adjustment set. `prepare_eicu_cohort.py:14` restricts baseline covariates
   to the first 60 minutes for exactly this reason and gates APS behind
   `--include-apache` as a sensitivity arm only.

---

## 7. What this means

**Study B is not includable from the demo release, and the reason is not fixable
by design.** Estimating it would require reporting a confidence interval wider
than the outcome's own range, from an instrument that a placebo test cannot
distinguish from randomly assigned wards.

Three things that *are* defensible:

1. **Report Study B as a quantified feasibility finding, not a failed attempt.**
   The demo's wide-shallow sampling is a genuine and non-obvious obstacle for
   federated within-client IV designs, and §2–§5 quantify it precisely. The
   placebo test is a reusable diagnostic: a first-stage F alone cannot tell
   practice variation from small-cluster accident, and on cluster-sparse data it
   routinely will not.
2. **Study A stands on its own.** It never needed real treatment effects — it
   measures structural recovery against a known `g₀` under real hospital
   partitions, and the demo supports that (with the documented caveats about
   per-client depth).
3. **The full credentialed eICU-CRD is the fix, and only it.** ~200,000 stays
   across 208 hospitals is roughly **966 stays per hospital versus 13.5** — a
   ~70× increase in exactly the dimension that is binding. The per-hospital gate
   would still need re-checking on a sepsis cohort there, and the exclusion
   restriction remains an argument rather than a test, but the identification
   problem documented here disappears.

## Reproducing this

Scripts are archived in [`study_b_feasibility/`](study_b_feasibility/) and run
against the `fedgmm` environment from the repository root:

```bash
P=/home/arnav22103/miniconda3/envs/fedgmm/bin/python
$P experiments/eicu_v1_demo/study_b_feasibility/census.py        # section 2
$P experiments/eicu_v1_demo/study_b_feasibility/studyb_sweep.py  # sections 1, 3
$P experiments/eicu_v1_demo/study_b_feasibility/power.py         # sections 5, 6
$P experiments/eicu_v1_demo/study_b_feasibility/decompose.py     # section 4
$P experiments/eicu_v1_demo/study_b_feasibility/placebo.py       # section 4 (~5 min)
$P experiments/eicu_v1_demo/study_b_feasibility/final_checks.py  # sections 4, 5
$P experiments/eicu_v1_demo/study_b_feasibility/reliability.py   # section 5b
```

Each reads only from `physionet.org/files/eicu-crd-demo/2.0.1` and writes nothing
to the release. They reuse `scripts/eicu_instrument.py`,
`scripts/eicu_iv_diagnostics.py` and `scripts/eicu_common.py` unchanged, so the
numbers above are produced by the repo's own estimators, not by a parallel
implementation. `placebo.py` and `final_checks.py` use fixed seeds
(`20260728`), so the permutation p-values are reproducible.
