"""Definition-independent structural census of the eICU demo release.

The point: any Study B cohort is a SUBSET of the full demo patient table.
Row counts, patients-per-hospital, patients-per-ward, and wards-per-hospital
are all monotone non-increasing under subsetting. So whatever the *maximum*
of these quantities is over the unrestricted population is a hard ceiling
that no sepsis definition, treatment window, or drug list can exceed.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/arnav22103/FederatedDeepGMM/scripts")
from eicu_common import read_table, parse_age, expired_flag  # noqa: E402

ROOT = "/home/arnav22103/FederatedDeepGMM/physionet.org/files/eicu-crd-demo/2.0.1"

pd.set_option("display.width", 160)

patient = read_table(ROOT, "patient")
print("=" * 78)
print("A. RAW DEMO CENSUS (no cohort restriction of any kind)")
print("=" * 78)
print(f"total unit stays          : {len(patient)}")
print(f"distinct hospitalid       : {patient['hospitalid'].nunique()}")
print(f"distinct wardid           : {patient['wardid'].nunique()}")
print(f"distinct health-system ids: {patient['patienthealthsystemstayid'].nunique()}")
print(f"distinct uniquepid        : {patient['uniquepid'].nunique()}")

per_hosp = patient.groupby("hospitalid").size()
print("\nstays per hospital (ALL 2520 stays):")
print(per_hosp.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
print(f"  hospitals with >= 200 stays : {(per_hosp >= 200).sum()}")
print(f"  hospitals with >= 100 stays : {(per_hosp >= 100).sum()}")
print(f"  hospitals with >=  50 stays : {(per_hosp >= 50).sum()}")
print(f"  hospitals with >=  25 stays : {(per_hosp >= 25).sum()}")
print(f"  hospitals with >=  10 stays : {(per_hosp >= 10).sum()}")
print(f"  largest hospital            : {per_hosp.max()} stays")
print(f"  top-5 hospital sizes        : {sorted(per_hosp.values)[-5:]}")

# wards per hospital, and ward sizes
wh = patient.groupby(["hospitalid", "wardid"]).size().rename("n").reset_index()
wards_per_hosp = wh.groupby("hospitalid").size()
print("\nwards per hospital (ALL stays):")
print(wards_per_hosp.describe(percentiles=[0.5, 0.9]).to_string())

print("\nBEST-CASE ward-preference support, over the WHOLE demo:")
print("(a hospital is usable for a within-hospital ward IV only if it has >= 2")
print(" wards that each carry at least K patients)")
print(f"{'K':>5} | {'hospitals with >=2 wards of size>=K':>36}")
for K in (100, 50, 25, 20, 15, 10, 5, 3, 2, 1):
    ok = wh[wh["n"] >= K].groupby("hospitalid").size()
    print(f"{K:>5} | {(ok >= 2).sum():>36}")

print("\nnumber of stays living in those best-case hospitals:")
for K in (50, 25, 10, 5):
    ok = wh[wh["n"] >= K].groupby("hospitalid").size()
    hosps = set(ok[ok >= 2].index)
    n = int(per_hosp[per_hosp.index.isin(hosps)].sum())
    print(f"  K={K:>3}: {len(hosps):>3} hospitals, {n:>5} total stays")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("B. MORTALITY / OUTCOME CEILING")
print("=" * 78)
patient["died"] = expired_flag(patient["hospitaldischargestatus"])
known = patient[patient["died"].notna()]
print(f"stays with known discharge status : {len(known)}")
print(f"total in-hospital deaths in demo  : {int(known['died'].sum())}")
print(f"overall mortality                 : {known['died'].mean():.4f}")
deaths_per_hosp = known.groupby("hospitalid")["died"].sum()
print(f"hospitals with >= 20 deaths       : {(deaths_per_hosp >= 20).sum()}")
print(f"hospitals with >= 10 deaths       : {(deaths_per_hosp >= 10).sum()}")
print(f"hospitals with >=  5 deaths       : {(deaths_per_hosp >= 5).sum()}")
print(f"max deaths in any one hospital    : {int(deaths_per_hosp.max())}")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("C. TREATMENT-SIDE CEILING: every plausible binary ICU treatment")
print("=" * 78)

adult_first = patient.copy()
adult_first["age_years"] = parse_age(adult_first["age"])

infusion = read_table(
    ROOT, "infusiondrug", usecols=["patientunitstayid", "infusionoffset", "drugname"]
)
sys.path.insert(0, "/home/arnav22103/FederatedDeepGMM/scripts")
from eicu_common import normalize_vasopressor_series, SENSITIVITY_VASOPRESSORS  # noqa

infusion["agent"] = normalize_vasopressor_series(infusion["drugname"])
vaso_any = set(infusion.loc[infusion["agent"].notna(), "patientunitstayid"])
vaso_6h = set(
    infusion.loc[
        infusion["agent"].notna() & infusion["infusionoffset"].between(0, 360),
        "patientunitstayid",
    ]
)
vaso_24h = set(
    infusion.loc[
        infusion["agent"].notna() & infusion["infusionoffset"].between(0, 1440),
        "patientunitstayid",
    ]
)

aps = read_table(ROOT, "apacheApsVar", usecols=["patientunitstayid", "vent", "dialysis"])
vent_ids = set(aps.loc[aps["vent"] == 1, "patientunitstayid"])
dial_ids = set(aps.loc[aps["dialysis"] == 1, "patientunitstayid"])

treatments = {
    "vasopressor 0-6h": vaso_6h,
    "vasopressor 0-24h": vaso_24h,
    "vasopressor any time": vaso_any,
    "apache vent (day 1)": vent_ids,
    "apache dialysis (day 1)": dial_ids,
}

print(f"{'treatment':<26} {'n treated (all stays)':>22} {'rate':>8}")
for name, ids in treatments.items():
    n = patient["patientunitstayid"].isin(ids).sum()
    print(f"{name:<26} {n:>22} {n/len(patient):>8.3f}")

print("\nPer-hospital treated/untreated support (ALL 2520 stays, best case):")
print(f"{'treatment':<26} {'hosp >=20/20':>13} {'hosp >=10/10':>13} {'hosp >=5/5':>11} {'hosp >=1/1':>11}")
for name, ids in treatments.items():
    d = patient.assign(D=patient["patientunitstayid"].isin(ids).astype(int))
    g = d.groupby("hospitalid")["D"].agg(["sum", "size"])
    g["untreated"] = g["size"] - g["sum"]
    row = []
    for k in (20, 10, 5, 1):
        row.append(int(((g["sum"] >= k) & (g["untreated"] >= k)).sum()))
    print(f"{name:<26} {row[0]:>13} {row[1]:>13} {row[2]:>11} {row[3]:>11}")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("D. THE JOINT REQUIREMENT (what a federated ward-preference IV needs)")
print("=" * 78)
print("A hospital must simultaneously: have >=2 wards with real patients, have")
print("both treated and untreated patients, and have the treatment rate DIFFER")
print("across its wards (otherwise ward preference is constant within client).")
print()
print(f"{'treatment':<26} {'>=2 wards & treated&untreated':>31} {'+ ward-rate sd>0':>18}")
for name, ids in treatments.items():
    d = patient.assign(D=patient["patientunitstayid"].isin(ids).astype(int))
    ok_h = []
    ok_h_var = []
    for h, rows in d.groupby("hospitalid"):
        nw = rows["wardid"].nunique()
        t, u = rows["D"].sum(), (1 - rows["D"]).sum()
        if nw >= 2 and t >= 1 and u >= 1:
            ok_h.append(h)
            wr = rows.groupby("wardid")["D"].mean()
            if wr.std(ddof=0) > 0:
                ok_h_var.append(h)
    print(f"{name:<26} {len(ok_h):>31} {len(ok_h_var):>18}")
