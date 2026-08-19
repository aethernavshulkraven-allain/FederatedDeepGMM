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
4. **Practical-tie fallback, made mechanical (2026-08-19, second pass):**
   "comparable to the spread" above was qualitative. Exact definitions,
   fixed before any adjudication/confirmation results exist:

   - Per-run Ψ score: `S_{c,s} = mean(Ψ_{451:500})` (last-50-round mean of
     candidate `c`, seed `s`, over the 500-round run).
   - Candidate score: `M_c = median_{s in {0,1,2}}(S_{c,s})`.
   - For two candidates `a`, `b`, define paired per-seed differences
     `d_s = S_{a,s} - S_{b,s}` for `s in {0,1,2}`.
   - Declare a **practical Ψ tie** between `a` and `b` if either:
     - the sign of `d_s` is not the same across all 3 seeds (rank order
       flips between at least two seeds), **or**
     - `|M_a - M_b| <= max(range(S_a), range(S_b))`, where `range(S_c) =
       max_s(S_{c,s}) - min_s(S_{c,s})` over that candidate's own 3 seeds.
   - **MSE fallback, defined symmetrically:** `E_{c,s} = mean(validation
     MSE_{451:500})` for candidate `c`, seed `s`; fallback score =
     `median_s(E_{c,s})`. Lower fallback score wins. This is the same
     last-50-round-mean-then-cross-seed-median construction as the Ψ score,
     applied to validation MSE instead — removing any residual choice
     between final-round, best-round, or tail-average MSE once results are
     in hand.

## 0. Eligibility and the full promotion decision tree (2026-08-19, third pass)

Added after the reuse ledger (built while preparing the review packet)
found that `cifar10_xz/fedogda_d`'s existing MSE-winner has a `diverged:
true` seed-2 finals trajectory. This campaign's federation is fully
deterministic (`client_num_in_total == client_num_per_round == 10`,
`batch_size=0`, no client sampling) — a seed either reproducibly converges
or reproducibly diverges for a given configuration. There is no
stochastic retry that could resolve a bad seed, so the rule below treats a
failed seed as a permanent property of the candidate, not a data gap to
patch around.

**Eligibility rule:** a candidate is promotable only if **all 3**
confirmation seeds produce complete artifacts, finite required metrics,
and `diverged=False`. If any one seed fails this, the candidate fails
confirmation entirely. Do **not** compute a two-seed median, do **not**
impute or substitute a score for the missing seed, and do **not** rerun
the identical deterministic seed expecting a different outcome.

**Full decision tree, per cell** (typically 3 candidates: Ψ rank-1, Ψ
rank-2, existing MSE winner — fewer if candidates coincide):

1. Apply the eligibility rule above to every candidate in the cell.
2. **Zero candidates eligible** → the cell requires retuning. Do not
   promote anything; do not fall back to a disqualified candidate.
3. **Exactly one candidate eligible** → it is promoted directly. No Ψ
   comparison, no tie check — a single survivor needs no ranking.
4. **Two or more candidates eligible** → rank eligible candidates by
   `M_c` (median last-50-round mean Ψ, as defined above), descending.
   Let `top` be the highest-ranked.
   a. Compare `top` against every other eligible candidate using the
      practical-tie formula above (signs of paired seed differences, or
      gap ≤ max own-range).
   b. Form the **tie set**: `top` plus every other eligible candidate that
      ties with `top` by that comparison. (Comparisons are always made
      against `top`, not all pairs — with the cell sizes in this campaign,
      2–3 eligible candidates, this fully determines the set: any
      candidate not directly comparable to `top` is not in it.)
   c. If the tie set contains only `top` → promote `top` directly (no
      practical tie).
   d. If the tie set contains more than one candidate → promote whichever
      member of the tie set has the lowest median last-50-round mean
      validation MSE (the symmetric MSE fallback score defined above).

An existing MSE winner that fails the eligibility rule (as
`cifar10_xz/fedogda_d`'s does) is never removed from the comparison ledger
— it stays recorded as a candidate for that cell — but enters step 1
already excluded, and the remaining eligible candidates (if any) are
ranked and resolved among themselves per steps 2–4 without it.

**Critic collapse is diagnostic-only.** The `_x` diagnostic
(`PSI_X_SCENARIO_DIAGNOSTIC_20260818.md`) reports near-constant critic
output as a signal worth investigating, but no numerical collapse
threshold is frozen, and none is being added now. Critic-collapse
reporting does **not** independently change promotion — promotion is
controlled only by the eligibility rule and the mechanical Ψ tie
procedure above. If collapse is observed alongside a practical Ψ tie, the
tie already resolves to MSE per step 4d; collapse by itself does not force
a fallback, skip a candidate, or trigger retuning. A numerical collapse
threshold would need its own dated amendment before it could gate
anything.

## How this changes what each pending stage is called

- **`_x` cells (`femnist_x`, `cifar10_x`):** the pending 500-round, 3-seed
  run is a **metric-reliability adjudication**, not automatic Ψ-based
  promotion. Its job is to establish whether Ψ (scored per this rule) is
  even usable here at all — per the 2026-08-18 diagnostic, these cells
  already show rank instability and, in some candidates, critic collapse.
  If, under this scoring rule, `_x` candidates fail to separate (fall into
  the practical-tie fallback, per §0), that is itself part of the
  adjudication's answer for that cell — resolved via the tie procedure in
  §0, not by silently defaulting to whichever candidate scores highest.
  Critic collapse observed alongside this (per the diagnostic) is recorded
  for context but does not independently change the outcome — see §0's
  "critic collapse is diagnostic-only" rule.
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
