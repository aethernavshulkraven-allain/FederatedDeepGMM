"""Where does the instrument's variation actually live, and what survives
covariate adjustment / hospital fixed effects?

FedDeepGMM aggregates CLIENT-LOCAL moment conditions. With client = hospital,
only WITHIN-hospital instrument variation can identify anything. This script
splits Var(Z) into between- and within-hospital parts and re-runs the first
stage with hospital fixed effects -- the federated-equivalent first stage.
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/arnav22103/FederatedDeepGMM/scripts")
from eicu_common import (  # noqa: E402
    read_table, parse_age, expired_flag, normalize_vasopressor_series,
    PRIMARY_VASOPRESSORS, SEPSIS_TEXT_PATTERN, continuous_covariate_columns,
)
from eicu_iv_diagnostics import first_stage_diagnostics  # noqa: E402
from eicu_instrument import build_instrument, structural_instrument_variation  # noqa

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
base["male"] = (base["gender"] == "Male").astype(float)

infusion = read_table(ROOT, "infusiondrug",
                      usecols=["patientunitstayid", "infusionoffset", "drugname"])
infusion["agent"] = normalize_vasopressor_series(infusion["drugname"])
vaso = infusion[infusion["agent"].isin(PRIMARY_VASOPRESSORS)]
pre_icu = set(vaso.loc[vaso["infusionoffset"] < 0, "patientunitstayid"])
vaso6 = set(vaso.loc[vaso["infusionoffset"].between(0, 360), "patientunitstayid"])
aps = read_table(ROOT, "apacheApsVar")
vent = set(aps.loc[aps["vent"] == 1, "patientunitstayid"])

admit = read_table(ROOT, "admissionDx", usecols=["patientunitstayid", "admitdxpath"])
admit_sepsis = set(admit.loc[admit["admitdxpath"].str.contains(
    SEPSIS_TEXT_PATTERN, case=False, na=False, regex=True), "patientunitstayid"])

CONFIGS = {
    "Study B corrected (sepsis, vaso 0-6h)": (admit_sepsis, vaso6, True),
    "ALL adult ICU + vaso 0-6h": (None, vaso6, True),
    "ALL adult ICU + vent day1": (None, vent, False),
}


def frame_for(pop, trt, drop_pre):
    c = base if pop is None else base[base["patientunitstayid"].isin(pop)]
    c = c.copy()
    if drop_pre:
        c = c[~c["patientunitstayid"].isin(pre_icu)].copy()
    c["treatment"] = c["patientunitstayid"].isin(trt).astype(float)
    c["Y"] = c["died"].astype(float)
    return c


def demean_by(values, groups):
    s = pd.Series(np.asarray(values, float))
    return (s - s.groupby(np.asarray(groups)).transform("mean")).to_numpy()


print("=" * 100)
print("6. VARIANCE DECOMPOSITION OF THE WARD-PREFERENCE INSTRUMENT")
print("=" * 100)
print("The federated estimator can only use the WITHIN-hospital part.")
print()
print(f"{'configuration':<40}{'Var(Z)':>10}{'between-hosp':>14}{'within-hosp':>13}"
      f"{'% usable':>10}{'hosp w/ real ward var':>24}")
print("-" * 100)

frames = {}
for name, (pop, trt, dp) in CONFIGS.items():
    c = frame_for(pop, trt, dp)
    z, _, _ = build_instrument(c, construction="ward", client_col="hospitalid", seed=0)
    c = c.assign(z=z.values)
    frames[name] = c

    total = c["z"].var(ddof=0)
    within = demean_by(c["z"], c["hospitalid"]).var(ddof=0)
    between = total - within
    # hospitals whose ward-level (fold-noise-free) preference genuinely differs
    sv = structural_instrument_variation(c, "hospitalid", "wardid", "z")
    n_real = int((sv > 0.01).sum())
    print(f"{name:<40}{total:>10.5f}{between:>14.5f}{within:>13.5f}"
          f"{100*within/total:>9.1f}%{f'{n_real} / {c.hospitalid.nunique()}':>24}")

print()
print("=" * 100)
print("7. THE FEDERATED-EQUIVALENT FIRST STAGE (hospital fixed effects)")
print("=" * 100)
print("Pooled F uses between-hospital variation, which a client-local moment")
print("condition cannot see. Adding hospital fixed effects leaves exactly the")
print("variation FedDeepGMM would actually have.")
print()
print(f"{'configuration':<40}{'pooled F':>10}{'F | hospital FE':>17}{'n':>7}"
      f"{'effective n (hosp w/ >1 ward)':>31}")
print("-" * 100)
for name, c in frames.items():
    X = c[["age_years", "male"]].to_numpy(float)
    D = c["treatment"].to_numpy(float)
    Z = c["z"].to_numpy(float)
    pooled = first_stage_diagnostics(Z, D, covariates=X)["partial_f"]

    zw = demean_by(Z, c["hospitalid"])
    dw = demean_by(D, c["hospitalid"])
    xw = np.column_stack([demean_by(X[:, j], c["hospitalid"]) for j in range(X.shape[1])])
    fe = first_stage_diagnostics(zw, dw, covariates=xw)["partial_f"] if zw.std() > 0 \
        else float("nan")
    multi = c.groupby("hospitalid")["wardid"].nunique()
    eff = int(c["hospitalid"].isin(multi[multi > 1].index).sum())
    print(f"{name:<40}{pooled:>10.2f}{fe:>17.3f}{len(c):>7}{eff:>31}")

print()
print("=" * 100)
print("8. RECONCILING F = 3.564: effect of the full covariate set")
print("=" * 100)
print("A partial F depends on what you condition on. With 256 rows the repo's")
print("full lab/vital covariate block is badly overparameterized.")
print()
c = frames["Study B corrected (sepsis, vaso 0-6h)"]
D = c["treatment"].to_numpy(float)
Z = c["z"].to_numpy(float)
print(f"{'covariate set':<44}{'p (controls)':>14}{'rows/param':>12}{'partial F':>12}")
print("-" * 100)
sets = {
    "none": [],
    "age + sex": ["age_years", "male"],
    "age + sex + weight": ["age_years", "male", "admissionweight"],
}
for label, cols in sets.items():
    if cols:
        Xm = c[cols].apply(pd.to_numeric, errors="coerce")
        Xm = Xm.fillna(Xm.median()).to_numpy(float)
    else:
        Xm = None
    f = first_stage_diagnostics(Z, D, covariates=Xm)["partial_f"]
    p = 0 if Xm is None else Xm.shape[1]
    print(f"{label:<44}{p:>14}{len(c)/max(p,1):>12.1f}{f:>12.3f}")

# repo's own continuous covariate block, pulled from the v1 cohort file
v1 = pd.read_csv("/home/arnav22103/FederatedDeepGMM/experiments/eicu_v1_demo/cohort.csv")
cov_cols = continuous_covariate_columns(v1)
print(f"\nrepo's continuous covariate block on the v1 cohort: {len(cov_cols)} columns")
print("(with missingness indicators the design matrix roughly doubles)")
v1z, _, _ = build_instrument(v1, construction="ward", client_col="hospitalid", seed=0)
Xv = v1[cov_cols].apply(pd.to_numeric, errors="coerce")
Xv = Xv.fillna(Xv.median()).to_numpy(float)
fv = first_stage_diagnostics(v1z.to_numpy(float), v1["treatment"].to_numpy(float),
                             covariates=Xv)["partial_f"]
print(f"partial F on the v1 201-row cohort with that block: {fv:.3f}  "
      f"(rows/param = {len(v1)/max(len(cov_cols),1):.1f})")
