"""Noise floor of a cross-fitted preference instrument, and what cluster depth buys.

A preference estimated from m peers is a binomial proportion: it carries the true
between-unit practice signal tau^2 plus sampling noise p(1-p)/m. The classical
reliability ratio

    lambda(m) = tau^2 / (tau^2 + p(1-p)/m)

is the attenuation factor on the first stage. lambda -> 1 means the instrument is
essentially noise-free; lambda -> 0 means it is a coin flip. This is the
analytical backbone of a cluster-depth feasibility study, and it turns
"is the full eICU worth credentialing?" into an arithmetic question.
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/arnav22103/FederatedDeepGMM/scripts")
from eicu_common import (  # noqa: E402
    read_table, parse_age, expired_flag, normalize_vasopressor_series,
    PRIMARY_VASOPRESSORS,
)

ROOT = "/home/arnav22103/FederatedDeepGMM/physionet.org/files/eicu-crd-demo/2.0.1"

patient = read_table(ROOT, "patient")
patient["age_years"] = parse_age(patient["age"])
base = patient[(patient["age_years"] >= 18) & (patient["unitvisitnumber"] == 1)]
base = (base.sort_values(["patienthealthsystemstayid", "hospitaladmitoffset"],
                         ascending=[True, False])
        .drop_duplicates("patienthealthsystemstayid", keep="first").copy())
base["died"] = expired_flag(base["hospitaldischargestatus"])
base = base[base["died"].notna()]
base = base[base["hospitalid"].notna() & base["wardid"].notna()].copy()

infusion = read_table(ROOT, "infusiondrug",
                      usecols=["patientunitstayid", "infusionoffset", "drugname"])
infusion["agent"] = normalize_vasopressor_series(infusion["drugname"])
vaso = infusion[infusion["agent"].isin(PRIMARY_VASOPRESSORS)]
pre_icu = set(vaso.loc[vaso["infusionoffset"] < 0, "patientunitstayid"])
aps = read_table(ROOT, "apacheApsVar")

TRT = {
    "vasopressor 0-6h": (set(vaso.loc[vaso["infusionoffset"].between(0, 360),
                                      "patientunitstayid"]), True),
    "ventilation day 1": (set(aps.loc[aps["vent"] == 1, "patientunitstayid"]), False),
}


def estimate_tau(trt, drop_pre):
    """Excess between-hospital SD over binomial noise = real practice variation."""
    c = base.copy()
    if drop_pre:
        c = c[~c["patientunitstayid"].isin(pre_icu)]
    c = c.copy()
    c["D"] = c["patientunitstayid"].isin(trt).astype(float)
    g = c.groupby("hospitalid")["D"].agg(["mean", "size"])
    p = c["D"].mean()
    obs = g["mean"].var(ddof=0)
    noise = float(np.mean(p * (1 - p) / g["size"]))
    return p, np.sqrt(max(obs - noise, 0.0)), float(g["size"].mean())


print("=" * 92)
print("14. RELIABILITY OF A PREFERENCE INSTRUMENT vs. CLUSTER DEPTH")
print("=" * 92)
print("lambda(m) = tau^2 / (tau^2 + p(1-p)/m).  tau estimated from the demo as the")
print("excess between-hospital SD over binomial noise (section 12).")
print()

DEPTHS = [(13.5, "eICU DEMO (measured)"), (25, ""), (50, ""), (100, ""),
          (250, ""), (500, ""), (966, "full eICU-CRD (projected)"), (2000, "")]

for label, (trt, dp) in TRT.items():
    p, tau, m_demo = estimate_tau(trt, dp)
    print(f"--- {label}:  p = {p:.3f},  real practice SD tau = {tau:.4f} ---")
    print(f"{'patients per hospital (m)':>28}{'reliability lambda':>20}"
          f"{'usable signal':>16}   note")
    for m, note in DEPTHS:
        lam = tau ** 2 / (tau ** 2 + p * (1 - p) / m)
        bar = "#" * int(round(lam * 20))
        print(f"{m:>28.0f}{lam:>20.3f}   {bar:<20} {note}")
    m_needed = p * (1 - p) / (tau ** 2) * (0.9 / 0.1)
    print(f"  -> patients per hospital needed for lambda >= 0.90: {m_needed:.0f}")
    print()

print("=" * 92)
print("15. DOES CREDENTIALING FOR THE FULL eICU-CRD ACTUALLY FIX IT?")
print("=" * 92)
full_stays, full_hosp = 200_859, 208
m_full = full_stays / full_hosp
print(f"full eICU-CRD: ~{full_stays:,} stays / {full_hosp} hospitals "
      f"= {m_full:.0f} stays per hospital ({m_full/13.5:.0f}x the demo)")
print()
for label, (trt, dp) in TRT.items():
    p, tau, _ = estimate_tau(trt, dp)
    lam_demo = tau ** 2 / (tau ** 2 + p * (1 - p) / 13.5)
    lam_full = tau ** 2 / (tau ** 2 + p * (1 - p) / m_full)
    # sepsis cohort is ~10% of stays; vasopressor Study B lives there
    m_sep = m_full * 0.10
    lam_sep = tau ** 2 / (tau ** 2 + p * (1 - p) / m_sep)
    print(f"{label}")
    print(f"   demo   (m = 13.5) : lambda = {lam_demo:.3f}")
    print(f"   full   (m = {m_full:.0f}) : lambda = {lam_full:.3f}")
    print(f"   full, sepsis-only (m = {m_sep:.0f}) : lambda = {lam_sep:.3f}")
    print()

print("Eligibility gate (>=200 patients, >=20 treated, >=20 untreated, >=20 deaths)")
print("projected onto the full release, using demo rates:")
p_v, _, _ = estimate_tau(*TRT["vasopressor 0-6h"])
mort = base["died"].mean()
sep_rate = 0.10
m_sep = m_full * sep_rate
print(f"  sepsis stays per hospital      : {m_sep:.0f}   (need >= 200) "
      f"{'PASS' if m_sep >= 200 else 'FAIL -- still short'}")
print(f"  treated per hospital           : {m_sep*p_v:.0f}   (need >= 20)  "
      f"{'PASS' if m_sep*p_v >= 20 else 'FAIL'}")
print(f"  deaths per hospital            : {m_sep*mort:.0f}   (need >= 20)  "
      f"{'PASS' if m_sep*mort >= 20 else 'FAIL'}")
print()
print("=> the full release fixes the INSTRUMENT noise floor decisively, but a")
print("   sepsis-restricted Study B is still near the edge of the per-hospital")
print("   gate. Widening the clinical population, or relaxing min_patients with")
print("   a documented rationale, would likely be needed even there.")
