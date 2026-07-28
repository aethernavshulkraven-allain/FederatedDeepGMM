"""First-stage strength of every candidate instrument, and the precision ceiling.

Two questions:
  (1) Does ANY instrument on this data have a usable first stage?
  (2) Even granting a hypothetically strong instrument, what is the tightest
      confidence interval the demo could ever produce for a risk difference?
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/arnav22103/FederatedDeepGMM/scripts")
from eicu_common import (  # noqa: E402
    read_table, parse_age, expired_flag, normalize_vasopressor_series,
    PRIMARY_VASOPRESSORS, SEPSIS_TEXT_PATTERN, matches_sepsis_icd9,
)
from eicu_iv_diagnostics import (  # noqa: E402
    first_stage_diagnostics, two_stage_least_squares, ols,
)
from eicu_instrument import build_instrument  # noqa: E402

ROOT = "/home/arnav22103/FederatedDeepGMM/physionet.org/files/eicu-crd-demo/2.0.1"

patient = read_table(ROOT, "patient")
patient["age_years"] = parse_age(patient["age"])
base = patient[patient["age_years"] >= 18]
base = base[base["unitvisitnumber"] == 1]
base = (base.sort_values(["patienthealthsystemstayid", "hospitaladmitoffset"],
                         ascending=[True, False])
        .drop_duplicates("patienthealthsystemstayid", keep="first").copy())
base["died"] = expired_flag(base["hospitaldischargestatus"])
base = base[base["died"].notna()]
base = base[base["hospitalid"].notna() & base["wardid"].notna()].copy()

# treatments
infusion = read_table(ROOT, "infusiondrug",
                      usecols=["patientunitstayid", "infusionoffset", "drugname"])
infusion["agent"] = normalize_vasopressor_series(infusion["drugname"])
vaso = infusion[infusion["agent"].isin(PRIMARY_VASOPRESSORS)]
pre_icu = set(vaso.loc[vaso["infusionoffset"] < 0, "patientunitstayid"])
aps = read_table(ROOT, "apacheApsVar")

admit = read_table(ROOT, "admissionDx", usecols=["patientunitstayid", "admitdxpath"])
admit_sepsis = set(admit.loc[admit["admitdxpath"].str.contains(
    SEPSIS_TEXT_PATTERN, case=False, na=False, regex=True), "patientunitstayid"])

CONFIGS = {
    "Study B as corrected (admitDx sepsis, vaso 0-6h)": dict(
        pop=admit_sepsis, trt=set(vaso.loc[vaso["infusionoffset"].between(0, 360),
                                           "patientunitstayid"]), drop_pre=True),
    "Sepsis + vaso 0-24h (widest sepsis treatment)": dict(
        pop=admit_sepsis, trt=set(vaso.loc[vaso["infusionoffset"].between(0, 1440),
                                           "patientunitstayid"]), drop_pre=True),
    "ALL adult ICU + vaso 0-6h": dict(
        pop=None, trt=set(vaso.loc[vaso["infusionoffset"].between(0, 360),
                                   "patientunitstayid"]), drop_pre=True),
    "ALL adult ICU + vent day1 (max-power config)": dict(
        pop=None, trt=set(aps.loc[aps["vent"] == 1, "patientunitstayid"]),
        drop_pre=False),
}

# baseline covariates available without post-treatment leakage
base["male"] = (base["gender"] == "Male").astype(float)
COVS = ["age_years", "male"]


def build_frame(cfg):
    c = base if cfg["pop"] is None else base[base["patientunitstayid"].isin(cfg["pop"])]
    c = c.copy()
    if cfg["drop_pre"]:
        c = c[~c["patientunitstayid"].isin(pre_icu)].copy()
    c["treatment"] = c["patientunitstayid"].isin(cfg["trt"]).astype(float)
    c["Y"] = c["died"].astype(float)
    for col in COVS:
        c[col] = pd.to_numeric(c[col], errors="coerce")
        c[col] = c[col].fillna(c[col].median())
    return c


def loo_hospital_rate(frame):
    g = frame.groupby("hospitalid")["treatment"]
    s, n = g.transform("sum"), g.transform("count")
    z = (s - frame["treatment"]) / (n - 1).clip(lower=1)
    return z.fillna(frame["treatment"].mean())


def offhours(frame):
    t = frame["hospitaladmittime24"].astype(str).str.slice(0, 2)
    hh = pd.to_numeric(t, errors="coerce")
    return ((hh < 7) | (hh >= 19)).astype(float)


print("=" * 100)
print("3. FIRST-STAGE STRENGTH OF EVERY CANDIDATE INSTRUMENT (pooled, best case)")
print("=" * 100)
print(f"{'configuration':<48}{'instrument':<26}{'n':>6}{'partial F':>11}{'partial R2':>12}")
print("-" * 100)

results = []
for cname, cfg in CONFIGS.items():
    c = build_frame(cfg)
    X = c[COVS].to_numpy(float)
    D = c["treatment"].to_numpy(float)
    Y = c["Y"].to_numpy(float)

    instruments = {}
    # cross-fitted ward preference (the repo's primary construction)
    try:
        z_ward, _, _ = build_instrument(c, construction="ward", client_col="hospitalid",
                                        seed=0)
        instruments["cross-fitted ward pref"] = z_ward.to_numpy(float)
    except Exception as exc:                                    # noqa: BLE001
        print(f"  (ward instrument unavailable for {cname}: {exc})")
    # cross-fitted hospital preference
    try:
        z_hosp, _, _ = build_instrument(c, construction="hospital",
                                        client_col="hospitalid", seed=0)
        instruments["cross-fitted hosp pref"] = z_hosp.to_numpy(float)
    except Exception:                                           # noqa: BLE001
        pass
    instruments["leave-one-out hosp rate"] = loo_hospital_rate(c).to_numpy(float)
    instruments["off-hours admission"] = offhours(c).to_numpy(float)

    for iname, z in instruments.items():
        if np.nanstd(z) == 0 or np.isnan(z).any():
            print(f"{cname:<48}{iname:<26}{len(c):>6}{'degenerate':>11}")
            continue
        fs = first_stage_diagnostics(z, D, covariates=X)
        results.append(dict(config=cname, instrument=iname, n=len(c),
                            F=fs["partial_f"], R2=fs["partial_r2"], z=z, X=X, D=D, Y=Y))
        print(f"{cname:<48}{iname:<26}{len(c):>6}{fs['partial_f']:>11.3f}"
              f"{fs['partial_r2']:>12.5f}")

print()
print("=" * 100)
print("4. WHAT THOSE FIRST STAGES BUY YOU: 2SLS estimate and 95% CI")
print("=" * 100)
print(f"{'configuration':<44}{'instrument':<24}{'ATE (risk diff)':>16}{'95% CI':>26}")
print("-" * 100)
for r in results:
    if r["F"] < 0.5:
        continue
    iv = two_stage_least_squares(r["z"], r["D"], covariates=r["X"], outcome=r["Y"])
    lo = iv["effect"] - 1.96 * iv["effect_stderr"]
    hi = iv["effect"] + 1.96 * iv["effect_stderr"]
    print(f"{r['config']:<44}{r['instrument']:<24}{iv['effect']:>+16.3f}"
          f"{f'[{lo:+.2f}, {hi:+.2f}]':>26}")

print()
print("=" * 100)
print("5. PRECISION CEILING: the best CI the demo could EVER produce")
print("=" * 100)
print("Var(beta_2SLS) ~= sigma_e^2 / (n * R2_partial * Var(D)).  Take the most")
print("favourable numbers the demo can offer: n = 2086 (every adult ICU stay),")
print("mortality 8.4%, and ventilation as treatment (the highest-prevalence,")
print("best-supported treatment in the release).")
print()
c = build_frame(CONFIGS["ALL adult ICU + vent day1 (max-power config)"])
n = len(c)
p_y = c["Y"].mean()
var_d = c["treatment"].var(ddof=0)
sigma_e = np.sqrt(p_y * (1 - p_y))
print(f"  n = {n},  P(death) = {p_y:.4f},  P(vent) = {c['treatment'].mean():.4f},  "
      f"Var(D) = {var_d:.4f}")
print()
print(f"{'assumed first-stage partial R^2':>32} {'implied F':>10} {'SE(ATE)':>10} "
      f"{'95% CI width':>14} {'verdict':>34}")
print("-" * 100)
for r2 in (0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25):
    f_stat = r2 / (1 - r2) * (n - 3)
    se = sigma_e / np.sqrt(n * r2 * var_d)
    width = 2 * 1.96 * se
    if width > 1.0:
        verdict = "CI wider than the whole 0-1 range"
    elif width > 0.4:
        verdict = "cannot distinguish +20% from -20%"
    elif width > 0.2:
        verdict = "cannot rule out large harm or benefit"
    else:
        verdict = "would be informative"
    print(f"{r2:>32.3f} {f_stat:>10.1f} {se:>10.3f} {width:>14.3f} {verdict:>34}")

print()
print("For reference, the strongest first stage measured on this data above is")
best = max((r for r in results), key=lambda r: r["R2"])
print(f"  partial R^2 = {best['R2']:.5f}  (F = {best['F']:.2f}, {best['instrument']},")
print(f"   {best['config']})")
se_best = sigma_e / np.sqrt(n * max(best["R2"], 1e-9) * var_d)
print(f"  -> implied 95% CI half-width on a risk difference: +/- {1.96*se_best:.2f}")
