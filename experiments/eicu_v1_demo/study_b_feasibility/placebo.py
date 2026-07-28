"""Placebo test: is the ward-preference first stage real, or a small-ward artifact?

If ward preference measures genuine practice style, then destroying the
patient-to-ward mapping (while preserving ward sizes and every treatment
decision) must destroy the first stage.

If instead the first stage is manufactured by tiny wards -- a 2-patient ward is
homogeneous ~80% of the time by pure chance, and cross-fitting then hands
patient i a near-copy of patient j's treatment -- the permuted data will show
the SAME first stage. That would mean the instrument is measuring nothing.
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
from eicu_instrument import build_instrument  # noqa: E402

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


def frame_for(pop, trt, drop_pre):
    c = base if pop is None else base[base["patientunitstayid"].isin(pop)]
    c = c.copy()
    if drop_pre:
        c = c[~c["patientunitstayid"].isin(pre_icu)].copy()
    c["treatment"] = c["patientunitstayid"].isin(trt).astype(float)
    c["Y"] = c["died"].astype(float)
    return c.reset_index(drop=True)


def demean_by(v, g):
    s = pd.Series(np.asarray(v, float))
    return (s - s.groupby(np.asarray(g)).transform("mean")).to_numpy()


def first_stages(c, seed=0):
    """Pooled F and hospital-fixed-effects F for the cross-fitted ward instrument."""
    z, _, _ = build_instrument(c, construction="ward", client_col="hospitalid",
                              seed=seed)
    Z = z.to_numpy(float)
    D = c["treatment"].to_numpy(float)
    X = c[["age_years", "male"]].to_numpy(float)
    pooled = first_stage_diagnostics(Z, D, covariates=X)["partial_f"]
    zw, dw = demean_by(Z, c["hospitalid"]), demean_by(D, c["hospitalid"])
    xw = np.column_stack([demean_by(X[:, j], c["hospitalid"]) for j in range(2)])
    fe = first_stage_diagnostics(zw, dw, covariates=xw)["partial_f"] if zw.std() > 0 \
        else np.nan
    return pooled, fe


def shuffle_wards_within_hospital(c, rng):
    """Preserve hospital, ward sizes and every treatment decision; break the
    real patient->ward assignment."""
    out = c.copy()
    for h, rows in c.groupby("hospitalid"):
        idx = rows.index.to_numpy()
        out.loc[idx, "wardid"] = rng.permutation(rows["wardid"].to_numpy())
    return out


def shuffle_hospitals(c, rng):
    """Preserve hospital/ward sizes; break the real patient->hospital assignment."""
    out = c.copy()
    perm = rng.permutation(len(c))
    out[["hospitalid", "wardid"]] = c[["hospitalid", "wardid"]].to_numpy()[perm]
    return out


CONFIGS = {
    "Study B corrected (sepsis, vaso 0-6h)": (admit_sepsis, vaso6, True),
    "ALL adult ICU + vaso 0-6h": (None, vaso6, True),
    "ALL adult ICU + vent day1": (None, vent, False),
}

N_PERM = 40

print("=" * 100)
print("9. PLACEBO TEST — IS THE WARD-PREFERENCE FIRST STAGE REAL?")
print("=" * 100)
print(f"{N_PERM} permutations. 'ward shuffle' keeps hospital, ward sizes and all")
print("treatments but randomises WHICH ward each patient was in. If the observed F")
print("sits inside the placebo distribution, the instrument carries no real signal.")
print()

for name, (pop, trt, dp) in CONFIGS.items():
    c = frame_for(pop, trt, dp)
    obs_pooled, obs_fe = first_stages(c, seed=0)

    pw, fw, ph, fh = [], [], [], []
    for k in range(N_PERM):
        cw = shuffle_wards_within_hospital(c, RNG)
        a, b = first_stages(cw, seed=k)
        pw.append(a); fw.append(b)
        ch = shuffle_hospitals(c, RNG)
        a2, b2 = first_stages(ch, seed=k)
        ph.append(a2); fh.append(b2)
    pw, fw, ph, fh = map(np.array, (pw, fw, ph, fh))

    print(f"--- {name}  (n={len(c)}) ---")
    print(f"{'statistic':<26}{'observed':>10}{'placebo mean':>14}{'placebo 95th':>14}"
          f"{'p-value':>10}   verdict")
    p_pooled_w = (pw >= obs_pooled).mean()
    p_fe_w = (fw >= obs_fe).mean()
    p_pooled_h = (ph >= obs_pooled).mean()

    def verdict(p):
        return "REAL signal" if p < 0.05 else "indistinguishable from noise"

    print(f"{'pooled F (ward shuffle)':<26}{obs_pooled:>10.2f}{pw.mean():>14.2f}"
          f"{np.percentile(pw,95):>14.2f}{p_pooled_w:>10.3f}   {verdict(p_pooled_w)}")
    print(f"{'within-hosp F (ward shuf)':<26}{obs_fe:>10.2f}{fw.mean():>14.2f}"
          f"{np.percentile(fw,95):>14.2f}{p_fe_w:>10.3f}   {verdict(p_fe_w)}")
    print(f"{'pooled F (hosp shuffle)':<26}{obs_pooled:>10.2f}{ph.mean():>14.2f}"
          f"{np.percentile(ph,95):>14.2f}{p_pooled_h:>10.3f}   {verdict(p_pooled_h)}")
    print()

print("=" * 100)
print("10. WHY: ward sizes in the multi-ward hospitals")
print("=" * 100)
c = frame_for(None, vent, False)
wsz = c.groupby(["hospitalid", "wardid"]).size()
multi = c.groupby("hospitalid")["wardid"].nunique()
multi_h = multi[multi > 1].index
wsz_multi = wsz[wsz.index.get_level_values(0).isin(multi_h)]
print(f"hospitals with >1 ward: {len(multi_h)}")
print(f"ward sizes inside them: median {wsz_multi.median():.0f}, "
      f"mean {wsz_multi.mean():.1f}, max {wsz_multi.max()}")
print(f"  wards with 1 patient : {(wsz_multi == 1).sum()} / {len(wsz_multi)}")
print(f"  wards with <=3       : {(wsz_multi <= 3).sum()} / {len(wsz_multi)}")
print(f"  wards with >=10      : {(wsz_multi >= 10).sum()} / {len(wsz_multi)}")
p = c["treatment"].mean()
print(f"\nP(a 2-patient ward is homogeneous by chance alone) = "
      f"{p**2 + (1-p)**2:.3f}  (treatment prevalence {p:.3f})")
print("A ward preference estimated from one or two other patients is a coin flip,")
print("not a practice style.")
