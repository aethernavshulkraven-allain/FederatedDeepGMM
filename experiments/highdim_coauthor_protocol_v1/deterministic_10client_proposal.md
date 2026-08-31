# Deterministic Arm at 10 Clients — Design Decision

Prepared 2026-08-05. Updated 2026-08-07: decision recorded. This document
previously asked for sign-off between three options; that question is now
resolved below.

Context: the protocol-v2 deterministic campaign as originally specified
(1000/1000 clients) costs a measured **665 GPU-h, ~14 quota weeks**
(`deterministic_runtime_profile_20260805/runtime_profile_findings.md`), which
is not affordable.

## Decision: Option B — shrink the client pool to 10, keep 100% participation

**`client_num_in_total = client_num_per_round = 10`.** Every one of the 10
clients trains every round. There is no client sampling anywhere in this arm.

Option A (sample 10 of 1000 clients per round) and Option C (keep all 1000,
100% participation) are both rejected:

* **Option A rejected** — sampling 10 of 1000 makes the global update a
  function of *which* random clients were drawn each round. That is
  stochasticity, not determinism, no matter what the run is labeled. It would
  have made `fedgda_d`/`fedgda_s` literally the same code path (§1 below),
  collapsing the deterministic arm into a second stochastic configuration and
  leaving nothing for the paper's deterministic-vs-stochastic contrast to
  compare against.
* **Option C rejected** — correct in principle (full participation, no
  sampling) but unaffordable at 665 GPU-h / ~14 quota weeks.
* **Option B chosen** — full participation (so the update stays a
  deterministic function of the global iterate, which is what `FedGDA-D` is
  supposed to mean) at a client count small enough to be affordable. This is
  a genuine cost reduction, not a redefinition: the same ~20,000 training
  points still get processed every round; they are just split into 10 large
  batches per round instead of 1000 tiny ones. The bottleneck identified in
  the runtime profile was **3,000 tiny sequential per-client steps per
  round**, not raw arithmetic — cutting the client count removes that
  overhead directly.

## 1. Why the method label alone doesn't guarantee determinism

`fedgda_d` and `fedgda_s` are the same code path — the method labels are pure
configuration, not different algorithms:

| Manifest | method | `client_optimizer` | `batch_size` | `client_num_per_round` |
|---|---|---|---|---|
| deterministic (this design) | `fedgda_d` | `sgd` | 0 | **10 = client_num_in_total** |
| stochastic (completed arm) | `fedgda_s` | `sgd` | 256 | 10 (sampled of 1000) |

The server update in `fedavg_api.py` branches only on `client_optimizer`, so
`fedgda_d`/`fedgda_s` execute identical logic. **What actually makes an arm
deterministic is `client_num_per_round == client_num_in_total`** — full
participation, no sampling. That is the one condition this design satisfies
and the rejected Option A did not.

## 2. Batch size is now a real, meaningful setting (unlike under the rejected option)

Under 10-of-1000 sampling, every sampled client held so few points (~20 on
average) that `batch_size=0` (full local batch) and `batch_size=256` were
numerically identical for 99%+ of clients — the axis was inert.

At `client_num_in_total = 10`, each client holds roughly **2,000 samples on
average** (20,000 points / 10 clients, before Dirichlet skew). Full-batch and
minibatch are now genuinely different regimes. This design uses
**`batch_size = 0` (full local batch)** — one full-data gradient step per
local epoch, with no minibatch sampling noise either. This is a second,
independent reason (beyond full participation) that this configuration
matches what "deterministic" is supposed to mean.

## 3. Consequence: `alpha` no longer means what it means in the stochastic table

Dirichlet partitioning is computed at load time from `client_num_in_total` and
`partition_alpha` (`fedml/data/MNIST/data_loader.py:154-161`). Partitioning
20,000 points across 10 clients at a given `alpha` produces a materially
different heterogeneity profile than partitioning it across 1000 clients at
the same nominal `alpha`. **The deterministic arm's `alpha ∈ {0.1, 0.5, 1.0}`
is not directly comparable to the stochastic arm's `alpha ∈ {0.1, 0.5, 1.0}`**
— same label, different partition. State this explicitly wherever the two
arms are compared side by side; it is a scientific-protocol fact, not a
footnote.

## 4. Complete configuration

### Federated / participation

```text
client_num_in_total       = 10
client_num_per_round      = 10        # = total; full participation, no sampling
partition_method          = hetero
partition_alpha           = 0.1, 0.5, 1.0    # meaning changes vs. stochastic arm, see §3
batch_size                = 0          # full local batch, now a real setting, see §2
epochs                    = 3          # local steps
comm_round                = 500        # matches protocol_summary.json's existing
                                        # deterministic_final_rounds target; the 1500-round
                                        # figure used for the stochastic arm is a
                                        # coupon-collector argument for *sampled* clients
                                        # and does not apply here — every client is seen
                                        # every round by construction
```

### Optimisation

```text
methods                   = fedgda_d (sgd), fedogda_d (ogda)
server_learning_rate      = 1.5        # unresolved — measure {1.0, 1.5} in the gate
gradient_clip_norm        = 1.0
aggregation_weighting     = sample_size
objective_mode            = legacy
objective lambda_1        = 0.1
```

### Tuning grid

**Do not reuse the old `femnist_z, alpha=0.5` learning-gate candidates
(`lr=0.01 cm=3`, etc.) unchanged.** That evidence was collected at
`client_num_in_total = 1000` under full participation — average local batch
size ~20 samples. This design runs full participation at
`client_num_in_total = 10` — average local batch size ~2,000 samples, a ~100x
change in gradient batch size and therefore in gradient noise. That is at
least as large a distribution shift as the rejected sampling option was, and
the old candidates should not be assumed to transfer.

Use the DeepGMM-literature-grounded grid from `doe_review_and_revised_grid.md`
Part VI instead — it is derived from the paper's own scenario-specific critic
ratios rather than from evidence collected at a different client count:

```text
FedGDA   η_g ∈ {0.003, 0.01, 0.03}
FedOGDA  η_g ∈ {0.001, 0.003, 0.01}      # shifted 3x down, OGDA effective-step concern

critic multiplier c, by architecture group:
  Group Z   (*_z)  : g=MLP,  f=CNN       -> c ∈ {1, 5}
  Group XZ  (*_xz) : g=CNN,  f=CNN       -> c ∈ {1, 5}
  Group X   (*_x)  : g=CNN,  f=small MLP -> c ∈ {5, 50}
```

Re-run Gate and Screen at `client_num_in_total = 10` before trusting any
candidate — do not skip straight to Rank/Confirm using the old gate's
winners.

### Runtime / correctness

```text
auxiliary_regression          = false   # must be set explicitly, see §6
auxiliary_regression_epochs   = 0
append_round_csv              = true
periodic_checkpoint_interval  = 0
log_test_mse_by_round         = false
test_mse_used_for_selection   = false
selection_metric_source       = validation
enable_legacy_outputs         = false
precision                     = torch.float64
```

`enable_legacy_outputs=false` is required: with it on, every round appends to
a **shared** path `csv/{optimizer}_{dataset}newtrial.csv` (`fedavg_api.py:793`)
that is not run-specific, so concurrent runs of the same dataset+optimizer
interleave into one file.

### Scale

```text
scenarios  = 6   (femnist_x/z/xz, cifar10_x/z/xz)
alphas     = 3   (0.1, 0.5, 1.0 — meaning changes vs. stochastic, see §3)
seeds      = 5   (0,1,2,3,4) for finals; seed 0 (plus confirm seeds) for tuning
```

## 5. Cost — measured 2026-08-07

The 665 GPU-h figure was measured at `client_num_in_total=1000`. The
~30–70 GPU-h pre-benchmark guess for 10-client full participation was **too
optimistic**. Real measurement
(`deterministic_10client_runtime_profile_20260807/runtime_profile_findings.md`,
same methodology as the 1000-client profile):

| Scenario | setup | s/round | speedup vs. N=1000 |
|---|---:|---:|---:|
| `femnist_z` | 53.7 s | 2.478 | 7.7x |
| `femnist_x` | 54.4 s | 2.453 | 7.7x |
| `cifar10_xz` | 164.9 s | 8.992 | 3.3x |

Setup time barely changes with client count (~54–165s either way), and the
per-round win is smaller for CIFAR than for FEMNIST. Applied to the adopted
staged pipeline (`doe_review_and_revised_grid.md` Part VI/VII, 324 runs,
minimal plan): **206.6 GPU-h (4.30 quota weeks)**. Applied to a direct-launch
design (180 finals-only runs, no tuning stage): **134.9 GPU-h (2.81 quota
weeks)**.

## 6. Launch precondition (unchanged)

`alpha0p5/tuning_manifest_deterministic.csv` predates these columns:

```text
auxiliary_regression   auxiliary_regression_epochs   append_round_csv
periodic_checkpoint_interval   log_test_mse_by_round
test_mse_used_for_selection    selection_metric_source
objective_mode                 aggregation_weighting
```

Writing new rows against that header with `csv.DictWriter` **silently drops
them** and the run falls back to defaults — auxiliary regression **on**,
periodic checkpoints every 200 rounds. This was hit during the runtime
profile: the first pass declared `auxiliary_regression=False` and ran aux-on.

Any prepare script must append them explicitly (as
`prepare_highdim_deterministic_learning_gate.py` does via `EXTRA_FIELDS`), and
the generated YAML must be verified before launch:

```bash
grep -H "auxiliary_regression:" <config-dir>/*/*/*/*.yaml
```

## 7. Open items before a manifest is generated

1. ~~Benchmark real per-round cost at `client_num_in_total=10`~~ — done §5
   (2026-08-07): `femnist_xz`, `cifar10_z`, `cifar10_x` are still
   interpolated, not measured; `fedogda_d` cost is assumed equal to
   `fedgda_d`, not measured separately.
2. Re-run Gate/Screen at the new client count rather than reusing
   `client_num_in_total=1000` gate candidates (§4, tuning grid).
3. Decide the Dirichlet-partition-meaning caveat's exact wording wherever
   deterministic-vs-stochastic tables are reported side by side (§3).
4. Confirm 500 rounds for finals (matches the pre-existing
   `protocol_summary.json` target); revisit only if the benchmark or gate
   shows convergence is not complete by then.
