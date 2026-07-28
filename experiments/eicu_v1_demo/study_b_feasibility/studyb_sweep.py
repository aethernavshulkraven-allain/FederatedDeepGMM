"""Reconstruct the Study B cohort variants and sweep the whole design space.

Question being answered: is there ANY (population x treatment x instrument)
configuration on the eICU demo that makes a real-outcome IV study estimable,
either federated (client = hospital) or pooled?
"""

import itertools
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/arnav22103/FederatedDeepGMM/scripts")
from eicu_common import (  # noqa: E402
    read_table,
    parse_age,
    expired_flag,
    normalize_vasopressor_series,
    PRIMARY_VASOPRESSORS,
    SEPSIS_TEXT_PATTERN,
    matches_sepsis_icd9,
)
from eicu_iv_diagnostics import first_stage_diagnostics, two_stage_least_squares  # noqa
from eicu_instrument import build_instrument, structural_instrument_variation  # noqa

ROOT = "/home/arnav22103/FederatedDeepGMM/physionet.org/files/eicu-crd-demo/2.0.1"

# ---------------------------------------------------------------------------
# Base population (identical to scripts/prepare_eicu_cohort.load_base_population)
# ---------------------------------------------------------------------------
patient = read_table(ROOT, "patient")
patient["age_years"] = parse_age(patient["age"])
base = patient[patient["age_years"] >= 18]
base = base[base["unitvisitnumber"] == 1]
base = (
    base.sort_values(
        ["patienthealthsystemstayid", "hospitaladmitoffset"], ascending=[True, False]
    )
    .drop_duplicates("patienthealthsystemstayid", keep="first")
    .copy()
)
base["died"] = expired_flag(base["hospitaldischargestatus"])
base = base[base["died"].notna()]
base = base[base["hospitalid"].notna() & base["wardid"].notna()].copy()
print(f"base population (adult, first stay, known outcome/ids): {len(base)} stays, "
      f"{base['hospitalid'].nunique()} hospitals, {base['wardid'].nunique()} wards")

# ---------------------------------------------------------------------------
# Sepsis definitions
# ---------------------------------------------------------------------------
diagnosis = read_table(
    ROOT, "diagnosis",
    usecols=["patientunitstayid", "diagnosisoffset", "diagnosisstring", "icd9code"],
)
dx_text = diagnosis["diagnosisstring"].str.contains(
    SEPSIS_TEXT_PATTERN, case=False, na=False, regex=True
)
dx_code = matches_sepsis_icd9(diagnosis["icd9code"])
dx_hit = dx_text | dx_code

admit = read_table(ROOT, "admissionDx", usecols=["patientunitstayid", "admitdxpath"])
admit_sepsis = set(
    admit.loc[
        admit["admitdxpath"].str.contains(SEPSIS_TEXT_PATTERN, case=False, na=False,
                                          regex=True),
        "patientunitstayid",
    ]
)

apache_sepsis = set(
    base.loc[
        base["apacheadmissiondx"].str.contains(SEPSIS_TEXT_PATTERN, case=False,
                                               na=False, regex=True),
        "patientunitstayid",
    ]
)

SEPSIS_DEFS = {
    "A. dx offset <= 0 (old Study B)": set(
        diagnosis.loc[dx_hit & (diagnosis["diagnosisoffset"] <= 0), "patientunitstayid"]),
    "B. admissionDx only (corrected)": admit_sepsis,
    "C. admissionDx or apacheAdmitDx": admit_sepsis | apache_sepsis,
    "D. dx +/-360min or admitDx (v1)": set(
        diagnosis.loc[dx_hit & diagnosis["diagnosisoffset"].between(-360, 360),
                      "patientunitstayid"]) | admit_sepsis | apache_sepsis,
    "E. sepsis ANY time (maximal)": set(
        diagnosis.loc[dx_hit, "patientunitstayid"]) | admit_sepsis | apache_sepsis,
    "F. no sepsis filter (all adult ICU)": set(base["patientunitstayid"]),
}

# ---------------------------------------------------------------------------
# Treatment definitions
# ---------------------------------------------------------------------------
infusion = read_table(
    ROOT, "infusiondrug", usecols=["patientunitstayid", "infusionoffset", "drugname"]
)
infusion["agent"] = normalize_vasopressor_series(infusion["drugname"])
vaso = infusion[infusion["agent"].isin(PRIMARY_VASOPRESSORS)]
pre_icu_vaso = set(vaso.loc[vaso["infusionoffset"] < 0, "patientunitstayid"])

aps = read_table(ROOT, "apacheApsVar", usecols=["patientunitstayid", "vent", "dialysis"])

TREATMENTS = {
    "vaso 0-6h": set(vaso.loc[vaso["infusionoffset"].between(0, 360), "patientunitstayid"]),
    "vaso 0-24h": set(vaso.loc[vaso["infusionoffset"].between(0, 1440), "patientunitstayid"]),
    "vaso any": set(vaso["patientunitstayid"]),
    "vent day1": set(aps.loc[aps["vent"] == 1, "patientunitstayid"]),
    "dialysis day1": set(aps.loc[aps["dialysis"] == 1, "patientunitstayid"]),
}

# infusion-interface hospitals (as in the repo)
inf_all = read_table(ROOT, "infusiondrug", usecols=["patientunitstayid"])
pat_ids = read_table(ROOT, "patient", usecols=["patientunitstayid", "hospitalid"])
iface_hosp = set(
    pat_ids[pat_ids["patientunitstayid"].isin(set(inf_all["patientunitstayid"]))]
    ["hospitalid"].dropna().unique()
)

# ---------------------------------------------------------------------------
print()
print("=" * 96)
print("1. REPRODUCING THE STUDY B COHORT VARIANTS")
print("=" * 96)
hdr = f"{'sepsis definition':<36}{'n':>6}{'treated':>8}{'deaths':>7}{'hosp':>6}{'wards':>6}"
print(hdr)
print("-" * 96)
cohorts = {}
for name, ids in SEPSIS_DEFS.items():
    c = base[base["patientunitstayid"].isin(ids)]
    c = c[c["hospitalid"].isin(iface_hosp)]                    # infusion interface
    c = c[~c["patientunitstayid"].isin(pre_icu_vaso)].copy()   # no pre-ICU vasopressor
    c["D"] = c["patientunitstayid"].isin(TREATMENTS["vaso 0-6h"]).astype(float)
    c["Y"] = c["died"].astype(float)
    cohorts[name] = c
    print(f"{name:<36}{len(c):>6}{int(c['D'].sum()):>8}{int(c['Y'].sum()):>7}"
          f"{c['hospitalid'].nunique():>6}{c['wardid'].nunique():>6}")

# ---------------------------------------------------------------------------
print()
print("=" * 96)
print("2. EXHAUSTIVE SWEEP: population x treatment, federated support")
print("=" * 96)
print("'ward-usable' = hospital has >=2 wards, both treated & untreated, and")
print("ward treatment rates that actually differ (the minimum for a within-client")
print("ward-preference instrument to carry ANY identifying variation).")
print()
print(f"{'population':<36}{'treatment':<14}{'n':>6}{'trt':>5}{'dth':>5}"
      f"{'hosp':>6}{'ward-usable':>12}{'>=5/5 hosp':>11}{'>=20/20':>9}")
print("-" * 96)

rows = []
for pname, ids in SEPSIS_DEFS.items():
    pop = base[base["patientunitstayid"].isin(ids)]
    for tname, tids in TREATMENTS.items():
        c = pop.copy()
        if tname.startswith("vaso"):
            c = c[c["hospitalid"].isin(iface_hosp)]
            c = c[~c["patientunitstayid"].isin(pre_icu_vaso)]
        c = c.copy()
        c["D"] = c["patientunitstayid"].isin(tids).astype(float)
        c["Y"] = c["died"].astype(float)
        if len(c) == 0:
            continue
        usable, big5, big20 = 0, 0, 0
        for h, r in c.groupby("hospitalid"):
            t, u = r["D"].sum(), (1 - r["D"]).sum()
            if t >= 5 and u >= 5:
                big5 += 1
            if t >= 20 and u >= 20:
                big20 += 1
            if r["wardid"].nunique() >= 2 and t >= 1 and u >= 1:
                if r.groupby("wardid")["D"].mean().std(ddof=0) > 0:
                    usable += 1
        rows.append(dict(pop=pname, trt=tname, n=len(c), treated=int(c["D"].sum()),
                         deaths=int(c["Y"].sum()), hosp=c["hospitalid"].nunique(),
                         usable=usable, big5=big5, big20=big20))
        print(f"{pname:<36}{tname:<14}{len(c):>6}{int(c['D'].sum()):>5}"
              f"{int(c['Y'].sum()):>5}{c['hospitalid'].nunique():>6}{usable:>12}"
              f"{big5:>11}{big20:>9}")

sweep = pd.DataFrame(rows)
sweep.to_csv("/tmp/claude-1015/-home-arnav22103-FederatedDeepGMM/"
             "37c30c82-dc2e-4251-8d42-7265e67fc812/scratchpad/sweep.csv", index=False)

print()
print(f"MAX over the whole sweep -> ward-usable hospitals: {sweep['usable'].max()}, "
      f"hospitals with >=5 treated & >=5 untreated: {sweep['big5'].max()}, "
      f"with >=20/20: {sweep['big20'].max()}")
