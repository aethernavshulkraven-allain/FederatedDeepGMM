# Deterministic 10-Client Runtime Profile — Findings

Generated: 2026-08-07. Diagnostic campaign, not a tuning or production queue.
Fills the GPU-h TBD placeholders left in `doe_review_and_revised_grid.md` and
`deterministic_10client_proposal.md` §5.

Configuration: `client_num_in_total = client_num_per_round = 10`, full
participation, `batch_size=0`, 3 local steps, `fedgda_d`, `lr=0.03`,
`critic_multiplier=3`, alpha 0.5, seed 0, 6 rounds, legacy objective,
sample-size weighting, `auxiliary_regression=false` (only arm — already
decided, unlike the N=1000 probe which compared aux on/off).

Reproduce:

```bash
python scripts/prepare_highdim_deterministic_10client_runtime_profile.py
gpurun -g 2 bash scripts/launch_highdim_deterministic_10client_runtime_profile.sh
```

## 1. Measured per-round cost

| Scenario | setup | s/round | basis |
|---|---:|---:|---|
| `femnist_z` | 53.7 s | 2.478 | measured |
| `femnist_x` | 54.4 s | 2.453 | measured |
| `femnist_xz` | 54.4 s | 5.723 | interpolated (mean of femnist_x, cifar10_xz) |
| `cifar10_z` | 164.9 s | 5.723 | interpolated (mean of femnist_x, cifar10_xz) |
| `cifar10_x` | 164.9 s | 5.723 | interpolated (mean of femnist_x, cifar10_xz) |
| `cifar10_xz` | 164.9 s | 8.992 | measured |

Interpolation follows the same "which scenario carries two CNNs vs one"
rule established by the N=1000 profile
(`deterministic_runtime_profile_20260805/runtime_profile_findings.md` §2) —
`femnist_xz` and `cifar10_z`/`cifar10_x` were not measured directly here
either, and should be measured before committing to a schedule.

**Speedup vs. the 1000-client measurement, same per-round comparison:**

| Scenario | N=1000 s/round | N=10 s/round | speedup |
|---|---:|---:|---:|
| `femnist_z` | 19.09 | 2.478 | 7.7x |
| `femnist_x` | 18.99 | 2.453 | 7.7x |
| `cifar10_xz` | 29.74 | 8.992 | 3.3x |

This is real but well short of the ~9.5–22x range the original proposal
guessed at (`deterministic_10client_proposal.md` §5, pre-benchmark estimate).
Setup time is essentially unchanged (54–165s either way — it does not scale
with client count), and the per-round win shrinks as the model gets more
GPU-bound (CIFAR's two CNNs pipeline more efficiently even at 1000 clients,
leaving less headroom to recover).

## 2. Real GPU-h for the two designs under consideration

Both use the measured/interpolated per-scenario cost above, `fedgda_d` timing
applied to `fedogda_d` as well (OGDA's optimistic combination adds no extra
forward/backward pass, so its round cost is assumed equal — not separately
measured).

### Design 1 — staged pipeline (adopted minimal plan)

324 runs: Screen(150rd)=72, Rank(500rd)=24, Confirm(500rd)=72,
alpha-0.1 stability check(500rd)=12, Finals(500rd)=144.

| Stage | Rounds | Runs | GPU-h |
|---|---:|---:|---:|
| Screen | 150 | 72 | 17.7 |
| Rank | 500 | 24 | 18.0 |
| Confirm | 500 | 72 | 54.0 |
| Stability check | 500 | 12 | 9.0 |
| Finals | 500 | 144 | 107.9 |
| **Total** | | **324** | **206.6 GPU-h (4.30 quota weeks)** |

### Design 2 — direct launch (skip Gate/Screen/Rank/Confirm)

180 runs: Finals only (36 cells x 5 seeds), one pre-picked config per cell,
500 rounds, no tuning stage at all.

**Total: 134.9 GPU-h (2.81 quota weeks).**

### Comparison

Staged costs **71.7 GPU-h more** than direct launch (≈1.5 quota weeks) — a
1.53x premium, not the ~2.1x the pre-benchmark placeholder numbers implied.

## 3. What is still not measured

`femnist_xz`, `cifar10_z`, `cifar10_x` are interpolated, not measured, same
caveat as the N=1000 profile. `fedogda_d` per-round cost is assumed equal to
`fedgda_d`, not measured. Both GPU-h totals above should be treated as good
planning numbers, not launch-grade figures, until those gaps are closed —
the same standard applied to the 665.1 GPU-h N=1000 figure before it was used
to size a campaign.
