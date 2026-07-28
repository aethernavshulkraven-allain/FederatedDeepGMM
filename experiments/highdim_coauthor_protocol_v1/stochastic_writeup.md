# High-Dimensional Stochastic Runs Handoff

## Design Of Experiment

objective: evaluate stochastic federated DeepGMM methods when `x`,
`z`, or both are represented by real images. The structural response is fixed:

```text
g(x) = abs(x)
```

Scenarios:

| Scenario | x representation | z representation |
|---|---|---|
| `femnist_x` | FEMNIST image | scalar |
| `femnist_z` | scalar | FEMNIST image |
| `femnist_xz` | FEMNIST image | FEMNIST image |
| `cifar10_x` | CIFAR-10 image | scalar |
| `cifar10_z` | scalar | CIFAR-10 image |
| `cifar10_xz` | CIFAR-10 image | CIFAR-10 image |

Stochastic federated methods:

| Report label | Repository method | Client optimizer |
|---|---|---|
| FedGDA-S / FedSGDA | `fedgda_s` | `sgd` |
| FedOGDA-S | `fedogda_s` | `ogda` |

Model mapping:

- image side uses CNN: CIFAR-10 uses `CIFAR10CNN`, FEMNIST uses `DefaultCNN`;
- scalar side uses `MLPModel`;
- therefore `*_x` is CNN `g` plus MLP `f`, `*_z` is MLP `g` plus CNN `f`,
  and `*_xz` is CNN `g` plus CNN `f`.

Fixed stochastic run shape:

| Parameter | Value |
|---|---:|
| Dirichlet alphas | `0.1`, `0.5`, `1.0` |
| Scenarios | 6 |
| Methods | 2 |
| Seeds | `0, 1, 2, 3, 4` |
| Total clients | 1000 |
| Clients per round | 10 |
| Batch size | 256 |
| Local steps / epochs | 3 |
| Communication rounds | 1500 |

This gives:

```text
3 alphas x 6 scenarios x 2 stochastic methods x 5 seeds = 180 runs
```

## Tuning And Selection

Tuning used seed `0`, validation metrics only, 150 communication rounds, and
two learning-rate candidates per alpha/scenario/method:

```text
learning_rate in {0.003, 0.01}
weight_decay = 0.05
critic_multiplier = 10
server_learning_rate = 1.5
gradient_clip_norm = 1.0
```

Selection rule:

1. Exclude numerical divergence only.
2. Rank by lowest `best_validation_mse`.
3. Tie-break by lower last-50-round validation-MSE standard deviation.
4. Tie-break by smaller final-minus-best validation gap.
5. Tie-break by lower learning rate.

## Aggregate Final Results

Metric: mean `test_mse_at_best_validation` over five seeds, with sample standard
deviation.

| alpha | scenario | method | n | Test MSE at best val | runtime median min |
|---:|---|---|---:|---:|---:|
| 0.1 | `cifar10_x` | FedGDA-S | 5 | 0.1588 +/- 0.0125 | 13.4 |
| 0.1 | `cifar10_x` | FedOGDA-S | 5 | 0.1730 +/- 0.0268 | 13.9 |
| 0.1 | `cifar10_xz` | FedGDA-S | 5 | 0.1679 +/- 0.0235 | 22.6 |
| 0.1 | `cifar10_xz` | FedOGDA-S | 5 | 0.1621 +/- 0.0104 | 22.7 |
| 0.1 | `cifar10_z` | FedGDA-S | 5 | 0.0449 +/- 0.0147 | 12.5 |
| 0.1 | `cifar10_z` | FedOGDA-S | 5 | 0.0791 +/- 0.0240 | 12.9 |
| 0.1 | `femnist_x` | FedGDA-S | 5 | 0.1590 +/- 0.0186 | 10.3 |
| 0.1 | `femnist_x` | FedOGDA-S | 5 | 0.1541 +/- 0.0115 | 11.1 |
| 0.1 | `femnist_xz` | FedGDA-S | 5 | 0.1554 +/- 0.0165 | 16.4 |
| 0.1 | `femnist_xz` | FedOGDA-S | 5 | 0.1519 +/- 0.0133 | 16.2 |
| 0.1 | `femnist_z` | FedGDA-S | 5 | 0.0118 +/- 0.0021 | 9.4 |
| 0.1 | `femnist_z` | FedOGDA-S | 5 | 0.0103 +/- 0.0011 | 9.9 |
| 0.5 | `cifar10_x` | FedGDA-S | 5 | 0.1890 +/- 0.0561 | 31.8 |
| 0.5 | `cifar10_x` | FedOGDA-S | 5 | 0.1648 +/- 0.0178 | 32.1 |
| 0.5 | `cifar10_xz` | FedGDA-S | 5 | 0.1641 +/- 0.0150 | 22.3 |
| 0.5 | `cifar10_xz` | FedOGDA-S | 5 | 0.1644 +/- 0.0095 | 22.4 |
| 0.5 | `cifar10_z` | FedGDA-S | 5 | 0.0553 +/- 0.0080 | 12.4 |
| 0.5 | `cifar10_z` | FedOGDA-S | 5 | 0.0646 +/- 0.0054 | 12.9 |
| 0.5 | `femnist_x` | FedGDA-S | 5 | 0.1695 +/- 0.0252 | 10.5 |
| 0.5 | `femnist_x` | FedOGDA-S | 5 | 0.1365 +/- 0.0076 | 10.8 |
| 0.5 | `femnist_xz` | FedGDA-S | 5 | 0.1729 +/- 0.0281 | 16.3 |
| 0.5 | `femnist_xz` | FedOGDA-S | 5 | 0.1433 +/- 0.0124 | 16.4 |
| 0.5 | `femnist_z` | FedGDA-S | 5 | 0.0130 +/- 0.0023 | 9.6 |
| 0.5 | `femnist_z` | FedOGDA-S | 5 | 0.0175 +/- 0.0033 | 10.4 |
| 1 | `cifar10_x` | FedGDA-S | 5 | 0.1579 +/- 0.0121 | 31.8 |
| 1 | `cifar10_x` | FedOGDA-S | 5 | 0.1728 +/- 0.0304 | 30.6 |
| 1 | `cifar10_xz` | FedGDA-S | 5 | 0.1656 +/- 0.0149 | 21.9 |
| 1 | `cifar10_xz` | FedOGDA-S | 5 | 0.1657 +/- 0.0089 | 22.0 |
| 1 | `cifar10_z` | FedGDA-S | 5 | 0.0513 +/- 0.0092 | 12.1 |
| 1 | `cifar10_z` | FedOGDA-S | 5 | 0.0876 +/- 0.0251 | 12.7 |
| 1 | `femnist_x` | FedGDA-S | 5 | 0.1586 +/- 0.0126 | 10.1 |
| 1 | `femnist_x` | FedOGDA-S | 5 | 0.1375 +/- 0.0133 | 10.8 |
| 1 | `femnist_xz` | FedGDA-S | 5 | 0.1444 +/- 0.0092 | 16.3 |
| 1 | `femnist_xz` | FedOGDA-S | 5 | 0.1489 +/- 0.0108 | 16.7 |
| 1 | `femnist_z` | FedGDA-S | 5 | 0.0147 +/- 0.0034 | 9.4 |
| 1 | `femnist_z` | FedOGDA-S | 5 | 0.0174 +/- 0.0036 | 10.3 |

## Review And Verification

The aggregate results **do not confirm the expected
heterogeneity trend or the expected FedOGDA-S stability advantage**, and the
headline `test_mse_at_best_validation` numbers come from checkpoints selected
very early in a 1500-round budget, with severe last-iterate blowup afterward.

### Scientific assessment against expected stochastic behavior

**Confirmed:**

- Best validation occurs before the final round in all 180 runs (max best
  round 1497 of 1499), and final-round test MSE never equals
  `test_mse_at_best_validation` — consistent with expectation, but see the
  magnitude caveat below.
- OGDA-vs-GDA superiority is scenario-dependent, not uniform: FedOGDA-S has
  the lower mean `test_mse_at_best_validation` in 8/18 alpha/scenario cells,
  FedGDA-S in 10/18. FedGDA-S is better in every `cifar10_z` cell; FedOGDA-S
  is better in most `femnist_x`/`femnist_xz` cells.
- FedOGDA-S has lower seed-to-seed variance than FedGDA-S (mean within-cell
  std 0.0131 vs 0.0158 across the 18 cells) — a genuine stability advantage.
- FEMNIST is easier than CIFAR-10 on the `z` scenarios (0.010-0.018 vs
  0.045-0.088 mean test MSE); `x`/`xz` scenarios are roughly comparable
  between datasets.

**Not confirmed / contradicted:**

1. **No heterogeneity trend.** Pooled mean `test_mse_at_best_validation` is
   0.1190 / 0.1212 / 0.1185 for alpha = 0.1 / 0.5 / 1.0 — essentially flat.
   Zero of the 12 scenario/method combinations show the expected monotone
   alpha=0.1 > alpha=0.5 > alpha=1.0 ordering; `femnist_z` FedGDA-S trends in
   the opposite direction (0.0118 to 0.0147 as alpha increases). The paper's
   own claim is only "marginally higher" MSE under higher heterogeneity, and
   its reference setup uses alpha=0.3 rather than this protocol's
   {0.1, 0.5, 1.0}, which may partly explain the mismatch; per-alpha learning
   rate retuning and best-validation checkpointing over 1500 rounds with
   10-of-1000 client sampling per round are also plausible dampers.
2. **Last-iterate behavior is worse than "noisy" — it is a large blowup for
   image-`x` scenarios.** For `*_x`/`*_xz` scenarios, median
   `best_validation_round` is ~13-25 (of 1500) for FedGDA-S and ~20-80 for
   FedOGDA-S. Median final/best test-MSE ratio is ~7x for both methods;
   83/90 FedGDA-S runs and 81/90 FedOGDA-S runs finish more than 2x above
   their best-validation test MSE; 40/90 runs per method finish with final
   test MSE > 1.0 versus ~0.15 at the selected checkpoint. The reported
   headline numbers are effectively early-stopped checkpoints from the first
   1-2% of the round budget for these scenarios, not a converged last
   iterate.
3. **FedOGDA-S does not show a smaller final-vs-best gap.** Median final/best
   ratio is 7.6 (p90 24.5) for FedOGDA-S versus 7.0 (p90 17.2) for FedGDA-S —
   the opposite of the expected optimism-damping benefit, at least as
   measured by post-hoc test MSE ratio. Where FedOGDA-S does differ: on the
   `z` scenarios it keeps improving much later into the budget (median best
   round 1481 vs 600 for FedGDA-S on `cifar10_z`) rather than peaking early,
   yet still ends with higher mean test MSE there.
4. **Scenario difficulty is inverted relative to the stated prior.** The
   `z`-image scenarios are the easiest (femnist_z ~ 0.01, cifar10_z ~ 0.05)
   and `x`-image scenarios are the hardest (~0.15-0.19); `cifar10_z` under
   high heterogeneity is among the better cells, not the unstable worst case
   suggested by the older appendix-style numbers. This is mechanistically
   plausible — MSE measures the structural function `g`, which in `*_z` is a
   scalar-to-scalar MLP learning `abs(x)` while the CNN burden falls on the
   critic side — but it should be reconciled explicitly against whatever
   prior table motivated the "z/xz harder, cifar10_z especially unstable"
   expectation.