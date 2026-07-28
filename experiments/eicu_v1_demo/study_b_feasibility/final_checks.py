"""Last two candidate rescues, and the noise audit of the between-hospital signal.

(a) Grouped-client fallback: client = region x teaching x beds, Z = hospital
    preference. This is the only construction in the repo not yet ruled out.
(b) Is the real between-hospital treatment-rate variation practice style, or
    just binomial noise from 13-patient hospitals?
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/arnav22103/FederatedDeepGMM/scripts")
from eicu_common import (  # noqa: E402
    read_table, parse_age, expired_flag, normalize_vasopressor_series,
    PRIMARY_VASOPRESSORS, SEPSIS_TEXT_PATTERN,
)
from eicu_iv_diagnostics import first_stage_diagnostics  # noqa: E402
from eicu_instrument import build_instrument, structural_instrument_variation  # noqa

ROOT = "/home/arnav22103/FederatedDeepGMM/physionet.org/files/eicu-crd-demo/2.0.1"
RNG = np.random.default_rng(20260728)

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
hosp = read_table(ROOT, "hospital")
base = base.merge(hosp, on="hospitalid", how="left")

infusion = read_table(ROOT, "infusiondrug",
                      usecols=["patientunitstayid", "infusionoffset", "drugname"])
infusion["agent"] = normalize_vasopressor_series(infusion["drugname"])
vaso = infusion[infusion["agent"].isin(PRIMARY_VASOPRESSORS)]
pre_icu = set(vaso.loc[vaso["infusionoffset"] < 0, "patientunitstayid"])
vaso6 = set(vaso.loc[vaso["infusionoffset"].between(0, 360), "patientunitstayid"])
aps = read_table(ROOT, "apacheApsVar")
vent = set(aps.loc[aps["vent"] == 1, "patientunitstayid"])


def frame_for(trt, drop_pre):
    c = base.copy()
    if drop_pre:
        c = c[~c["patientunitstayid"].isin(pre_icu)].copy()
    c["treatment"] = c["patientunitstayid"].isin(trt).astype(float)
    c["Y"] = c["died"].astype(float)
    cols = [x for x in ("region", "teachingstatus", "numbedscategory") if x in c.columns]
    c["client_group"] = c[cols].astype("string").fillna("NA").agg(" | ".join, axis=1)
    return c.reset_index(drop=True)


def demean_by(v, g):
    s = pd.Series(np.asarray(v, float))
    return (s - s.groupby(np.asarray(g)).transform("mean")).to_numpy()


print("=" * 100)
print("11. GROUPED-CLIENT FALLBACK  (client = region x teaching x beds)")
print("=" * 100)
for label, (trt, dp) in {"vaso 0-6h": (vaso6, True), "vent day1": (vent, False)}.items():
    c = frame_for(trt, dp)
    z, _, _ = build_instrument(c, construction="hospital", client_col="client_group",
                              seed=0)
    c = c.assign(z=z.values)
    ng = c["client_group"].nunique()
    sizes = c.groupby("client_group").size()
    tr = c.groupby("client_group")["treatment"].agg(["sum", "size"])
    tr["untreated"] = tr["size"] - tr["sum"]
    dth = c.groupby("client_group")["Y"].sum()

    Z, D = c["z"].to_numpy(float), c["treatment"].to_numpy(float)
    X = c[["age_years", "male"]].to_numpy(float)
    pooled = first_stage_diagnostics(Z, D, covariates=X)["partial_f"]
    zw, dw = demean_by(Z, c["client_group"]), demean_by(D, c["client_group"])
    xw = np.column_stack([demean_by(X[:, j], c["client_group"]) for j in range(2)])
    fe = first_stage_diagnostics(zw, dw, covariates=xw)["partial_f"]

    # placebo: randomise which hospital each patient belongs to, within group
    ph = []
    for k in range(30):
        cp = c.copy()
        for g, rows in c.groupby("client_group"):
            idx = rows.index.to_numpy()
            cp.loc[idx, "hospitalid"] = RNG.permutation(rows["hospitalid"].to_numpy())
        zp, _, _ = build_instrument(cp, construction="hospital",
                                    client_col="client_group", seed=k)
        zpw = demean_by(zp.to_numpy(float), cp["client_group"])
        ph.append(first_stage_diagnostics(zpw, dw, covariates=xw)["partial_f"])
    ph = np.array(ph)

    print(f"\n--- treatment = {label} ---")
    print(f"  clients (groups)                      : {ng}")
    print(f"  patients per group  median/min/max    : "
          f"{sizes.median():.0f} / {sizes.min()} / {sizes.max()}")
    print(f"  groups with >=20 treated & >=20 untr. : "
          f"{int(((tr['sum']>=20)&(tr['untreated']>=20)).sum())}")
    print(f"  groups with >=20 deaths               : {int((dth>=20).sum())}")
    print(f"  pooled first-stage F                  : {pooled:.2f}")
    print(f"  WITHIN-client first-stage F           : {fe:.2f}")
    print(f"  placebo within-client F (mean / 95th) : "
          f"{ph.mean():.2f} / {np.percentile(ph,95):.2f}   "
          f"p = {(ph >= fe).mean():.3f}")
    print("  verdict                               : "
          + ("REAL within-client signal" if (ph >= fe).mean() < 0.05
             else "indistinguishable from noise"))

print()
print("=" * 100)
print("12. IS THE BETWEEN-HOSPITAL VARIATION PRACTICE STYLE, OR BINOMIAL NOISE?")
print("=" * 100)
print("If every hospital had an identical true treatment rate, the observed")
print("spread of hospital rates would still be sqrt(p(1-p)/n_h) by chance alone.")
print()
print(f"{'treatment':<14}{'obs SD of hosp rates':>22}{'SD expected if NO':>20}"
      f"{'real practice SD':>18}{'share real':>12}")
print(f"{'':<14}{'':>22}{'practice variation':>20}{'(excess)':>18}{'':>12}")
print("-" * 100)
for label, (trt, dp) in {"vaso 0-6h": (vaso6, True), "vaso any": (
        set(vaso["patientunitstayid"]), True), "vent day1": (vent, False)}.items():
    c = frame_for(trt, dp)
    g = c.groupby("hospitalid")["treatment"].agg(["mean", "size"])
    p = c["treatment"].mean()
    obs_var = g["mean"].var(ddof=0)
    exp_var = float(np.mean(p * (1 - p) / g["size"]))
    real_var = max(obs_var - exp_var, 0.0)
    share = real_var / obs_var if obs_var > 0 else 0.0
    print(f"{label:<14}{np.sqrt(obs_var):>22.4f}{np.sqrt(exp_var):>20.4f}"
          f"{np.sqrt(real_var):>18.4f}{share:>11.1%}")

print()
print("=" * 100)
print("13. FACT CHECK: how many hospitals are actually in the demo?")
print("=" * 100)
raw = read_table(ROOT, "patient", usecols=["patientunitstayid", "hospitalid"])
print(f"  distinct hospitalid in patient.csv : {raw['hospitalid'].nunique()}")
print(f"  total unit stays                   : {len(raw)}")
print(f"  mean stays per hospital            : {len(raw)/raw['hospitalid'].nunique():.1f}")
print(f"  rows in hospital.csv               : {len(hosp)}")
print("  -> the demo is a WIDE, SHALLOW sample: nearly every hospital in eICU,")
print("     about a dozen patients each. Not 20 hospitals x 125 patients.")
