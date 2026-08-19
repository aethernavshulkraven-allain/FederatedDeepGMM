# Protocol amendment: boundary-expansion stopping rule — 2026-08-18

## Provenance of the rule being replaced

The ">2% relative Ψ improvement, second-highest→highest tested value"
threshold was **not** sourced from any repository document, manifest, or
preparation script. It does not appear in
`doe_review_and_revised_grid.md`, `deterministic_10client_proposal.md`, or
any `prepare_highdim_*` script. It was declared ad hoc, in-conversation,
immediately before the 2026-08-18 Ψ re-scoring pass, as a same-session
carry-over of the *val-MSE* diminishing-returns heuristic used informally
during the original 2026-08-13 screen boundary check (also not a written
protocol document at the time — an in-conversation judgment call, applied
after seeing results, which is itself part of what this amendment corrects).

## Why it failed

Applied mechanically against real Ψ values, the rule degenerated: Ψ crosses
zero and goes negative for several cells (e.g. `cifar10_z/fedogda_d`:
-0.0078 → 0.0279), so a percentage computed against a near-zero or negative
base produces meaningless magnitudes (`+1135%`, `+3123%`). It flagged every
boundary-touching axis for expansion — not because expansion was always
warranted, but because the arithmetic broke.

A single global *absolute* threshold would fail differently: Ψ's scale
differs by orders of magnitude across scenario architecture (`_x` cells:
~1e-5; `_xz` cells: ~1e-2; `_z` cells: ~1e-1), so one absolute cutoff would
either never fire for `_x` or fire on noise for `_z`.

## Replacement rule (effective 2026-08-18, applies to all remaining screen
## boundary decisions in this campaign)

1. If the Ψ rank-1 candidate for a cell has its winning learning rate
   and/or critic multiplier at the tested grid's maximum on that axis, add
   **exactly one** grid rung on that axis (same step convention as before:
   ~3.3x for learning rate, doubling for critic multiplier).
2. **Do not recursively expand** a second time on the same axis for the
   same cell, regardless of what the one added rung shows.
3. Resolve the candidate through the **500-round, 3-seed Rank/Confirm**
   stage, not through a second percentage or magnitude calculation. No
   further screen-stage (150-round, seed-0) arithmetic decides the outcome
   once one rung has been added.
4. If, at Rank/Confirm, candidates are practically indistinguishable under
   Ψ across seeds (flips sign of ranking between seeds, or the spread
   across seeds is comparable to the gap between candidates), invoke the
   already-declared secondary criterion: validation MSE.

This is scale- and sign-safe (no division by a value that can be zero or
negative), mechanically reproducible (one rung, one time, per axis, no
judgment calls), and has a bounded cost (at most one extra screen candidate
per flagged axis, ever). If applying it mechanically expands every
boundary-touching axis, that is an accepted consequence of the rule, not a
sign the rule needs another patch.

## Application to this campaign's flagged cells

Rule step 1 applied to every cell flagged in the 2026-08-18 Ψ re-score
(`psi_rescore.json`) — see `psi_boundary_expand_manifest.csv` for the
generated candidates (one rung per flagged axis, cells with both axes
flagged get both new rungs plus the corner).
