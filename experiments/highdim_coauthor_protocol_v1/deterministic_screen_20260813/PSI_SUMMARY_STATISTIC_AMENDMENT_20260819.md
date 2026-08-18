# Protocol amendment: which Ψ summary feeds adjudication/confirmation — 2026-08-19

## Gap this closes

"Promote by median validation Ψ across seeds" (stated 2026-08-18) is
underspecified: the `_x` diagnostic
(`PSI_X_SCENARIO_DIAGNOSTIC_20260818.md`, Finding 1) already showed that
best-round Ψ, final-round Ψ, and last-50-round mean Ψ can each select a
*different* candidate for the same cell. Without predeclaring which one of
these gets promoted per-seed and then medianed across seeds, the 500-round
adjudication/confirmation stage could finish and still leave a
discretionary choice about which Ψ summary to trust — exactly the kind of
after-the-fact judgment call this campaign's protocol amendments exist to
rule out in advance.

## Rule (effective 2026-08-19, applies to the pending `_x` adjudication and
## `_z`/`_xz` confirmation stages, and any future Ψ-based selection in this
## campaign)

1. **Configuration score (per run):** last-50-round mean Ψ. Not best-round
   Ψ — best-round Ψ is a max over a trajectory that can be non-monotonic
   (confirmed for `_x` cells; may also apply elsewhere), so it rewards a
   single lucky checkpoint rather than a stabilized state.
2. **Promotion (per candidate, across seeds):** median of the per-seed
   configuration score (step 1) across the 3 seeds.
3. **Best-round Ψ is still reported**, alongside the configuration score,
   for every run — but only as a diagnostic, never as the value promotion
   is computed from.
4. **Practical-tie fallback:** if two candidates' per-seed configuration
   scores cross rank order between seeds (candidate A leads in one seed,
   candidate B leads in another), or their median gap is comparable to the
   spread of either candidate's own scores across its 3 seeds, classify the
   pair as a practical Ψ tie for that cell. Resolve via the already-declared
   secondary criterion: median validation MSE across the same 3 seeds — not
   best-round MSE, not seed-0 MSE.

## How this changes what each pending stage is called

- **`_x` cells (`femnist_x`, `cifar10_x`):** the pending 500-round, 3-seed
  run is a **metric-reliability adjudication**, not automatic Ψ-based
  promotion. Its job is to establish whether Ψ (scored per this rule) is
  even usable here at all — per the 2026-08-18 diagnostic, these cells
  already show rank instability and, in some candidates, critic collapse.
  If, under this scoring rule, `_x` candidates still fail to separate
  (fall into the practical-tie fallback) or a candidate's last-50-round Ψ
  shows the collapse signature (near-constant critic output) documented in
  the diagnostic, that is itself the adjudication's answer — Ψ does not
  select for that cell, MSE does, and this must be recorded as such rather
  than silently defaulting to whichever candidate happens to score highest.
- **`_z`/`_xz` cells:** this is the **missing Rank/Confirm stage** the
  adopted plan specified from the start. Ψ (scored per this rule) is
  expected to be usable here — the 2026-08-18 screen-stage re-score showed
  wide seed-0 gaps for these cells — and this stage's job is to confirm
  that gap survives 500 rounds and 3 seeds, not to establish reliability
  from scratch.

## Provenance

This amendment was written before the `_x` adjudication or `_z`/`_xz`
confirmation runs were launched, using only the existing 150-round
screen-stage trajectories already on disk (no new GPU runs) — consistent
with the pre-registration discipline established in
`BOUNDARY_RULE_AMENDMENT_20260818.md`.
