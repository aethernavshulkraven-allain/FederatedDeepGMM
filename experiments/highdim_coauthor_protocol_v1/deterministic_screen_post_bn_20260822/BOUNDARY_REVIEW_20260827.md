# Boundary review — 2026-08-27

Applies `BOUNDARY_RULE_AMENDMENT_20260818.md`'s replacement rule to the
corrected, now-fully-resolved 108-row screen (104 eligible + 4 certified
terminal). This is a draft review, not yet a frozen decision — see
"Open item" below.

## Scorer correction

`score_highdim_screen_post_bn_20260822.py` was flagging a cell for boundary
review whenever **either** the Ψ rank-1 candidate **or** the validation-MSE
winner sat at the tested grid's edge. The frozen rule
(`BOUNDARY_RULE_AMENDMENT_20260818.md`, "Replacement rule" step 1) is scoped
to **the Ψ rank-1 candidate only** — the reference implementation in
`score_highdim_screen_by_psi.py` likewise only ever checks the Ψ-ranked top
candidate. Fixed to only let the Ψ rank-1 candidate's boundary status drive
`boundary_review_required`; the MSE winner's boundary status is still
recorded (`boundary_detail.mse_winner`) as diagnostic metadata, matching
what `boundary_flags()` was always capable of reporting, but no longer
triggers review by itself.

This removes two cells that were flagged only via their MSE winner:
`cifar10_x|fedgda_d` and `cifar10_x|fedogda_d`. Six cells remain genuinely
flagged via their Ψ rank-1 candidate.

## Per-cell resolution

For each flagged cell, checked whether the exact winning `(lr, cm)` pair was
already tested as a genuine "extend beyond the then-current max" expansion
rung (not merely present somewhere in an expansion manifest — the *exact*
combination), by tracing every source manifest
(`screen_manifest.csv` → `screen_expand_manifest.csv` →
`screen_expand2_manifest.csv` → `screen_expand2_corrected_v1_manifest.csv`).

| Cell | Axis | Winner (lr, cm) | Resolution |
|---|---|---|---|
| `cifar10_xz\|fedgda_d` | critic multiplier | (0.003, 10) | Exact match to `screen_expand2_manifest.csv`'s added rung (5→10). Already expanded once — **no further expansion**. |
| `cifar10_z\|fedgda_d` | critic multiplier | (0.01, 20) | Exact match to `screen_expand2_manifest.csv`'s rung (10→20); a first rung (5→10) via `screen_expand_manifest.csv` preceded it. Already expanded (twice, in fact) — **no further expansion**. |
| `femnist_x\|fedgda_d` | critic multiplier | (0.1, 20) | Exact match to `screen_expand2_manifest.csv`'s added rung (10→20). Already expanded once — **no further expansion**. |
| `femnist_x\|fedogda_d` | critic multiplier | (0.01, 20) | Exact match to `screen_expand_manifest.csv`'s added rung (10→20). Already expanded once — **no further expansion**. |
| `femnist_z\|fedogda_d` | critic multiplier | (0.001, 10) | Exact match to `screen_expand2_corrected_v1_manifest.csv`'s added rung (5→10). Already expanded once — **no further expansion**. |
| `cifar10_z\|fedogda_d` | critic multiplier | (0.003, 5) | **cm=5 was already the original, unexpanded screen's max** for this cell (present in `screen_manifest.csv` from the start, not added by any expansion). `screen_expand2_manifest.csv` / `..._corrected_v1` only added an *interior* point (cm=2, between the existing 1 and 5) and a new learning rate (0.009) — neither tests any critic-multiplier value above 5. No rung has ever been added beyond this cell's max. **Still requires its one permitted expansion rung: (lr=0.003, cm=10)**, per the doubling convention. |

Five of the six cells check out exactly against the frozen rule: the winning
`(lr, cm)` pair is precisely the point a genuine expansion rung added, and
the rule says not to expand a second time regardless of outcome. The sixth,
`cifar10_z|fedogda_d`, has not actually had its flagged axis pushed past the
original grid edge — the historical "expand2" work for that cell explored a
different part of the grid (an interior critic-multiplier infill and a new
learning rate), not the upward critic-multiplier direction that triggered
the flag.

## Open item

One new screen-protocol run is needed before this cell's boundary review can
close: `cifar10_z`, `fedogda_d`, seed 0, 150 rounds, `lr=0.003`, `cm=10` —
the single permitted expansion rung for its Ψ rank-1 winner. Not yet
launched. `screen_results.json` has not been frozen and this document is not
yet a final decision, pending that run (or an explicit decision not to run
it).
