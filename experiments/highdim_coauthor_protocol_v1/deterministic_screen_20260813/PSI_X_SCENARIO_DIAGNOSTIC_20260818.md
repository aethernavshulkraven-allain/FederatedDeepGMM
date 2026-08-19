# Ψ diagnostic for `_x` scenarios (`femnist_x`, `cifar10_x`) — 2026-08-18

Existing artifacts only, no new GPU runs. Raw per-candidate numbers in
`psi_x_diagnostic_raw.txt` alongside this file (best Ψ, final-round Ψ,
full-trajectory min/max, last-50-round Ψ mean/std, moment violation
best + last-50 mean/std, validation MSE, critic output f(z) mean/std/norm
on the dev set at the best-validation checkpoint — pulled from
`checkpoint['metrics']['f_of_z_dev']`, which is saved per run and needed no
model reload or data re-fetch).

## Finding 1: rank-1 is not stable across reasonable evaluation criteria

| Cell | Rank-1 by best-round Ψ | by final-round Ψ | by last-50-round mean Ψ | Stable? |
|---|---|---|---|---|
| `cifar10_x / fedgda_d` | lr=0.1, cm=10 | lr=0.1, cm=20 | lr=0.1, cm=20 | **No** |
| `cifar10_x / fedogda_d` | lr=0.01, cm=20 | lr=0.01, cm=10 | lr=0.01, cm=10 | **No** |
| `femnist_x / fedgda_d` | lr=0.1, cm=10 | lr=0.003, cm=5 | lr=0.1, cm=10 | **No** |
| `femnist_x / fedogda_d` | lr=0.03, cm=10 | lr=0.03, cm=10 | lr=0.03, cm=10 | Yes |

3 of 4 cells flip winner depending on which single checkpoint you read Ψ
from. "Best Ψ" is a max over a 150-round trajectory that is itself highly
non-monotonic (see next finding) — it is picking out whichever round
happened to spike closest to zero, not a converged state.

## Finding 2: the Ψ trajectory is not converging, it's oscillating

Example, `femnist_x/fedgda_d, lr=0.03, cm=5` (the eventual MSE-selected
winner for this cell): Ψ ranges from -7.68 to -0.006 over the 150-round
run; the *last 50 rounds* have mean -1.26 with std 1.31 — a standard
deviation **larger than the mean's magnitude**. That is not a stabilizing
metric, that is noise dominating signal at the end of training. This
pattern repeats across most `femnist_x` candidates and, less severely, the
`cifar10_x` ones.

Compare `val_moment_violation`: several candidates have a tiny
*best*-round value (1e-7–1e-5, looks great) but a last-50-round mean two to
four orders of magnitude larger (1e-2–1e-1) — the same "one lucky round"
pattern, not genuine convergence of the moment condition.

## Finding 3: several candidates show critic collapse

`f(z)` on the dev set, at the best-validation checkpoint:

- `femnist_x/fedgda_d, lr=0.1, cm=5`: mean=-1.32, **std=0.036**, norm=132
- `femnist_x/fedgda_d, lr=0.1, cm=10`: mean=-1.36, **std=0.022**, norm=136

Standard deviation two orders of magnitude smaller than the mean — the
critic is outputting essentially the same large negative number for every
validation point, regardless of z. It has stopped discriminating anything;
it has just pushed its output to an extreme, unconstrained value. This is
consistent with the "critic-starved" architecture concern already on record
for `_x` scenarios (small-MLP critic against a big-CNN structural model) —
at high learning rates the critic appears to escape rather than converge.
`cifar10_x`'s critics are large-magnitude too but not as flatly collapsed
(std comparable to mean in most cases) — a real difference between the two
`_x` datasets worth keeping separate rather than treating "`_x` scenarios"
as one uniform diagnosis.

## Reading against the pre-declared decision rule

Per the user's stated interpretation guide:

- Rankings **do** flip by evaluation criterion (Finding 1) and differences
  are small relative to trajectory/seed-scale noise (Finding 2) →
  **this points toward "practical Ψ tie," which the protocol resolves with
  the pre-specified secondary criterion (validation MSE)** for the
  candidates where that applies.
- The `femnist_x` lr=0.1 candidates specifically show collapse (Finding 3)
  strongly enough to warrant the "investigate critic/evaluator capacity"
  branch before trusting Ψ near those points at all -- not yet strong
  enough on its own to conclude the critic *architecture* itself must
  change; that determination needs the 500-round/3-seed adjudication data,
  not more 150-round/seed-0 diagnostics.

No architecture change is proposed here. This diagnostic only establishes
that Ψ, as currently evaluated (150 rounds, seed 0, single checkpoint), is
not yet a trustworthy selector for these 4 cells — which is exactly the
question the pending small adjudication matrix is designed to resolve.
